###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import time
import math
import array
import statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlperf_loadgen as lg

from dataset import Dataset
import habana_generation_utils as hgu
import modeling_gptj as hpu_modeling_gptj
import quantization.quantize as quantize


gen_kwargs = {
    "max_new_tokens": 128,
    "min_new_tokens": 30,
}


def setup_pt_profiler(schedule):
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.extend([torch.profiler.ProfilerActivity.HPU])

    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('.', use_gzip=True),
        record_shapes=True,
        with_stack=True)
    return profiler


def setup_hltv_profiler(schedule):
    import sys
    import os
    sys.path.append(os.environ['PYTORCH_MODULES_ROOT_PATH'])
    from topologies.tools import SynapseProfilerApi, TraceType
    api = SynapseProfilerApi()

    class SynapseProfiler:
        def check(self):
            if schedule(self.cur_step) == torch.profiler.ProfilerAction.RECORD_AND_SAVE:
                api.profiler_start(TraceType.TraceAll, 0)

        def start(self):
            self.cur_step = 0
            self.check()

        def step(self):
            self.cur_step = self.cur_step + 1
            self.check()

        def stop(self):
            api.profiler_stop(TraceType.TraceAll, 0)
            api.profiler_get_trace_json(TraceType.TraceAll, 0)

    return SynapseProfiler()


def setup_profiler(step, profile_type):
    active = 1
    warmup = 1 if step > 0 else 0
    wait = max(step - warmup, 0)

    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)

    if profile_type == 'tb':
        return setup_pt_profiler(schedule)
    else:
        return setup_hltv_profiler(schedule)


class SUT_base():
    def __init__(self, args, options):
        print("Loading PyTorch model...")
        self.dataset_path = args.dataset_path
        self.model_path = args.model_path
        self.batch_size = args.batch_size
        self.max_input_length = 1919
        self.profile = args.profile
        self.profile_type = args.profile_type
        self.inference_times = []

        gen_kwargs["num_beams"] = options["num_beams"]
        gen_kwargs["early_stopping"] = options["early_stopping"]

        if args.device == "cuda":
            assert torch.cuda.is_available(), "CUDA device is not available!"
        elif args.device == "hpu":
            import habana_frameworks.torch.core
            assert torch.hpu.is_available(), "HPU device is not available!"
        self.device = torch.device(args.device)

        if self.device.type == "hpu":
            self.model = hpu_modeling_gptj.GPTJForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16
            )
        else:
            is_gpu = self.device.type == "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if not is_gpu else None,
                low_cpu_mem_usage=True if not is_gpu else False,
                torch_dtype=torch.bfloat16
            )

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.to(torch.bfloat16)
        self.model.to(self.device)

        if self.device.type == "hpu":
            import habana_frameworks.torch.hpu.graphs as htgraphs

            self.model = htgraphs.wrap_in_hpu_graph(self.model)
            if args.quantization_file:
                self.model = quantize.setup_quantization(self.model, args.quantization_file)
            self.hgu_pipeline = hgu.create_pipeline(
                self.model, tokenizer=None, mode=hgu.GenerationMode.OPTIMIZED, calc_stats=False)

            self.hgu_opts = hgu.GenerationOptions(
                max_length=self.max_input_length+gen_kwargs['max_new_tokens']+1,
                min_length=self.max_input_length+gen_kwargs['min_new_tokens'],
                max_input_length=self.max_input_length+gen_kwargs['max_new_tokens']+1,
                **options,
            )
            if self.profile:
                self.hgu_opts.max_iterations = args.profile_tokens
            if args.dtype == "float8":
                self.hgu_opts.kv_cache_fp8 = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=self.max_input_length,
            padding_side="left",
            use_fast=True,)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_object = Dataset(
            self.model_path, self.dataset_path, total_count_override=args.max_examples)
        self.qsl = lg.ConstructQSL(self.data_object.count, self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)

    def _issue_dummy_query(self, batch_size=1):
        # dummy input tensors
        input_ids_tensor = torch.ones([batch_size, self.max_input_length], dtype=torch.int64)
        input_masks_tensor = input_ids_tensor.detach().clone()

        input_ids_tensor = input_ids_tensor.to(self.device)
        input_masks_tensor = input_masks_tensor.to(self.device)

        t_start = time.time()
        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()
        t_end = time.time()
        print("Warmup took {:.2f} ms".format((t_end-t_start)*1000))

    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        print("Number of Samples in query_samples : ", num_samples)

        batches = math.ceil(num_samples / self.batch_size)
        if self.profile:
            profiler = setup_profiler(batches - 1, self.profile_type)
        else:
            profiler = None

        if profiler:
            profiler.start()
        for i in range(batches):
            batch_size = min(num_samples - i * self.batch_size, self.batch_size)

            indices = [
                query_samples[i * self.batch_size + j].index for j in range(batch_size)
            ]
            while len(indices) < self.batch_size:
                indices.append(indices[0])

            input_ids_tensor = torch.cat(
                [self.data_object.source_encoded_input_ids[index] for index in indices]
            )
            input_masks_tensor = torch.cat(
                [self.data_object.source_encoded_attn_masks[index] for index in indices]
            )
            input_ids_tensor = input_ids_tensor.to(self.device)
            input_masks_tensor = input_masks_tensor.to(self.device)

            t_start = time.time()
            pred_output_batch = self.inference_call(
                input_ids_tensor, input_masks_tensor).cpu().numpy()
            t_end = time.time()
            if profiler:
                profiler.step()
            print("Batch {} : {:.2f} ms".format(i, (t_end-t_start)*1000))
            self.inference_times.append(t_end - t_start)

            responses_array = [
                array.array("B", pred_output_batch[i].tobytes()) for i in range(batch_size)
            ]
            bi = [
                response_array.buffer_info() for response_array in responses_array
            ]
            responses = [
                lg.QuerySampleResponse(
                    query_samples[i * self.batch_size + j].id, bi[j][0], bi[j][1]
                ) for j in range(batch_size)
            ]
            lg.QuerySamplesComplete(responses)
        if profiler:
            profiler.stop()

    def inference_call(self, input_ids_tensor, input_masks_tensor):
        ''' Common for all scenarios '''

        with torch.inference_mode():
            input_batch = dict()
            input_batch['input_ids'] = input_ids_tensor
            input_batch['attention_mask'] = input_masks_tensor
            input_batch_lengths = [x.shape[0]
                                   for x in input_batch["input_ids"]]

            if self.device.type == "hpu":
                output_batch = self.hgu_pipeline(input_batch, self.hgu_opts)

                # TODO: Remove for submission - output's only printed for convenience
                # print(self.tokenizer.batch_decode(output_batch, skip_special_tokens=True))
            else:
                output_batch = self.model.generate(
                    **input_batch, **gen_kwargs, pad_token_id=self.tokenizer.eos_token_id)

            output_batch_truncated = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_truncated.append(data[source_len:])

            output_batch_truncated = torch.stack(output_batch_truncated)
            # print(self.tokenizer.batch_decode(output_batch_truncated, skip_special_tokens=True))

        return output_batch_truncated

    def flush_queries(self):
        pass

    def __del__(self):
        if self.inference_times:
            mean = statistics.fmean(self.inference_times)
            print(f"Average performance: {self.batch_size / mean:.3f} samples/s")

        if self.device.type == "hpu":
            from habana_frameworks.torch.hpu.memory import memory_stats
            GB = 1024**3
            memory_stats_dict = memory_stats(self.device)
            max_in_use = memory_stats_dict['MaxInUse'] / GB
            limit = memory_stats_dict['Limit'] / GB
            print(
                "HPU memory usage: {:.1f} GB / {:.1f} GB ({:.0f}%)".format(
                    max_in_use, limit, max_in_use / limit * 100.0
                )
            )
        print("Finished destroying SUT.")


class SUT_Offline(SUT_base):
    def __init__(self, args, options):
        SUT_base.__init__(self, args, options)
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("Warming up...")
        self._issue_dummy_query(args.batch_size)
    '''IssueQuery and inference methods implemented in Base class'''


class SUT_Server(SUT_base):
    def __init__(self, args, options):
        SUT_base.__init__(self, args, options)
        self.total_samples_done = 0
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        print("SUT Server")
        print("Warming up...")
        self._issue_dummy_query()

    def issue_queries(self, query_samples):

        index = query_samples[0].index
        input_ids_tensor = self.data_object.source_encoded_input_ids[index]
        input_masks_tensor = self.data_object.source_encoded_attn_masks[index]

        input_ids_tensor = input_ids_tensor.to(self.device)
        input_masks_tensor = input_masks_tensor.to(self.device)

        t_start = time.time()
        pred_output_batch = self.inference_call(
            input_ids_tensor, input_masks_tensor).cpu().numpy()
        t_end = time.time()
        print("Sample time : {:.2f} ms".format((t_end-t_start)*1000))

        response_array = array.array("B", pred_output_batch.tobytes())
        bi = response_array.buffer_info()
        responses = [lg.QuerySampleResponse(query_samples[0].id, bi[0], bi[1])]
        lg.QuerySamplesComplete(responses)
        self.total_samples_done += 1
        if self.total_samples_done % 5 == 0:
            print("Completed : ", self.total_samples_done)
