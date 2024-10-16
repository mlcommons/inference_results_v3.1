import array
import copy
import json
import mlperf_loadgen as lg
import numpy as np
import os
import threading
import time
import torch
import torch.multiprocessing as mp
# import multiprocessing as mp

from contextlib import nullcontext
from dataset import Dataset
from evaluation import ndarray2prompt
from metadata import InputItem, OutputItem
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_batch_size, get_memory_usage, profile_handler, logger


CORES_PER_SOCKET = 56
NUM_SOCKETS = 2


class OfflineSUT(object):
    def __init__(self, model_path, dataset_path, dtype="float32", device_type="cpu",
            num_workers=1, num_beams=4, scenario="Offline", args=None, **kwargs):
        if device_type == "cuda":
            mp.set_start_method("spawn")
        self.batch_size = args.batch_size
        self.dynamic_batching = args.dynamic_batching
        self.scenario = scenario
        self.enable_sort = args.sort and (self.scenario == "Offline")

        self.dataset = Dataset(
            model_path, dataset_path,
            total_count_override=args.max_examples,
            pick_index=args.pick_index,
            repeat=args.repeat,
            padding_side=args.padding_side
        )

        self.warmup_dataset = Dataset(
            model_path, args.warmup_path,
            padding_side=args.padding_side
        ) if args.warmup else None

        logger.info("Creating tokenizer")
        self.padding_side = args.padding_side
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=1919,
            padding_side=self.padding_side,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.is_int4 = False
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "int4":
            self.dtype = torch.float16
            self.is_int4 = True
        else:
            self.dtype = torch.float32
        self.device_type = device_type
        if self.is_int4:  # lazy load model in sub-process
            self.model = None
        else:
            logger.info("Loading model to Host")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if self.device_type == "cpu" else None,
                low_cpu_mem_usage=True,
                torch_dtype=self.dtype
            ).eval()
            get_memory_usage("Host", "cpu")

        if args.verbose:
            with open(args.ref_log_path, "r") as f:
                ref_prompts = json.load(f)
        self.num_insts = min(num_workers, args.world_size - args.start_rank)
        logger.info(f"Creating {self.num_insts} instances on {self.device_type}:{args.start_rank}-{args.start_rank+self.num_insts-1}")
        self.insts = []
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()
        self.alive_counter = mp.Value("i", 0)
        self.cond_var = mp.Condition(lock=mp.Lock())
        cores_per_inst = CORES_PER_SOCKET * NUM_SOCKETS // self.num_insts
        ranks = list(range(args.start_rank, args.start_rank+self.num_insts, 1))
        core_starts = [rank * cores_per_inst for rank in ranks]
        for rank, core_start in zip(ranks, core_starts):
            inst = Instance(
                model_path=model_path, model=self.model, dataset=self.dataset, warmup_dataset=self.warmup_dataset,
                tokenizer=self.tokenizer, input_queue=self.input_queue, output_queue=self.output_queue,
                dtype=self.dtype, device_type=self.device_type, num_beams=num_beams, batch_size=self.batch_size,
                rank=rank, core_list=list(range(core_start, core_start+cores_per_inst)),
                alive_counter=self.alive_counter, cond_var=self.cond_var,
                enable_warmup=args.warmup, enable_profile=args.profile, dynamic_batching=self.dynamic_batching,
                optimize_transformers=args.optimize_transformers, is_int4=self.is_int4,
                enable_verbose=args.verbose,
                ref_prompts=ref_prompts if args.verbose else None,
                scenario=self.scenario
            )
            self.insts.append(inst)
            inst.start()

        with self.cond_var:
            self.cond_var.wait_for(lambda : self.alive_counter.value==self.num_insts)

        self.qsl = lg.ConstructQSL(
            self.dataset.count,
            self.dataset.perf_count,
            self.dataset.LoadSamplesToRam,
            self.dataset.UnloadSamplesFromRam
        )

        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)

        logger.info("Starting Loadgen response thread")
        self.response_thread = threading.Thread(target=self.response_loadgen)
        self.response_thread.daemon = True
        self.response_thread.start()

    def issue_queries(self, samples):
        # sorting
        if self.enable_sort:
            samples.sort(key=lambda s: self.dataset[s.index][-1], reverse=True)
        if self.batch_size != 1 or self.dynamic_batching:  # batching
            i = 0
            while True:
                if i >= len(samples):
                    break
                cur_max_len = self.dataset[samples[i].index][2]
                cur_batch_size = get_batch_size(cur_max_len, self.is_int4) if self.dynamic_batching else self.batch_size
                query_samples = samples[i : min(i+cur_batch_size, len(samples))]
                query_id = [sample.id for sample in query_samples]
                query_idx = [sample.index for sample in query_samples]
                input_ids, attn_masks, input_lens, actual_lens = self.dataset.collect(query_idx, cur_batch_size)
                self.input_queue.put(InputItem(query_id, query_idx, input_ids, attn_masks, input_lens, actual_lens, query_samples))
                i += cur_batch_size
        else:  # single-batch
            for i in range(len(samples)):
                sample = samples[i]
                input_ids = self.dataset[sample.index][0]
                attn_masks = self.dataset[sample.index][1]
                input_lens = [self.dataset[sample.index][2]]
                actual_lens = copy.deepcopy(input_lens)
                self.input_queue.put(InputItem([sample.id], [sample.index], input_ids, attn_masks, input_lens, actual_lens, [sample]))

    def response_loadgen(self):
        num_response = 0
        while True:
            next_result = self.output_queue.get()
            if next_result is None:
                logger.info("Exiting response thread")
                for inst in self.insts:
                    inst.join()
                break

            query_id = next_result.id
            result = next_result.result

            for id, out in zip(query_id, result):
                response_array = array.array('B', out.tobytes())
                bi = response_array.buffer_info()
                responses = [lg.QuerySampleResponse(id, bi[0], bi[1]*response_array.itemsize)]
                lg.QuerySamplesComplete(responses)
                num_response += 1
                if num_response % 100 == 0:
                    logger.debug(f"finish {num_response} samples")

    def flush_queries(self):
        self.input_queue.put(None)

    def __del__():
        lg.DestroyQSL(self.qsl)
        lg.DestroySUT(self.sut)
        logger.info("Finished destorying QSL & SUT.")


class ServerSUT(OfflineSUT):
    def __init__(self, model_path, dataset_path, dtype="float32", device_type="cpu",
            num_workers=1, num_beams=4, scenario="Server", args=None, **kwargs):
        super().__init__(model_path, dataset_path, dtype, device_type, num_workers, num_beams, scenario, args, **kwargs)

    def issue_queries(self, samples):
        # print(f'issue_queries:{len(samples)},{time.time()}')
        for i in range(len(samples)):
            sample = samples[i]
            input_ids = self.dataset[sample.index][0]
            attn_masks = self.dataset[sample.index][1]
            input_lens = [self.dataset[sample.index][2]]
            actual_lens = copy.deepcopy(input_lens)
            self.input_queue.put(InputItem([sample.id], [sample.index], input_ids, attn_masks, input_lens, actual_lens, [sample]))

class Instance(mp.Process):
    def __init__(self, model_path=None, model=None, dataset=None, warmup_dataset=None, tokenizer=None,
            input_queue=None, output_queue=None, dtype=torch.float32, device_type="cpu", dynamic_batching=False,
            num_beams=4, batch_size=1, rank=0, core_list=[], alive_counter=1, cond_var=None, enable_warmup=False, enable_profile=False,
            optimize_transformers=True, is_int4=False, enable_verbose=False, ref_prompts=None, scenario="Offline"):
        mp.Process.__init__(self)
        self.gen_kwargs = {
            "early_stopping": True,
            "max_new_tokens": 128,
            "min_new_tokens": 30,
            "num_beams": num_beams,
            # only beam_size 4 is allowed for official submission
        }
        self.scenario = scenario
        self.batch_size = batch_size
        self.dynamic_batching = dynamic_batching
        self.model_path = model_path
        self.model = model
        self.dataset = dataset
        self.warmup_dataset = warmup_dataset
        self.tokenizer = tokenizer
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.dtype = dtype
        self.enable_amp = (self.dtype == torch.bfloat16 or self.dtype == torch.float16)
        self.device_type = device_type
        self.rank = rank
        if self.device_type == "xpu":
            self.device = torch.device("xpu:0")
        else:
            self.device = torch.device(f"{self.device_type}:{self.rank}")
        self.alive_counter = alive_counter
        self.cond_var = cond_var
        self.enable_warmup = enable_warmup
        self.enable_profile = enable_profile
        self.profile_iter = 0
        self.profile_ctx = nullcontext()
        self.is_int4 = is_int4
        self.optimize_transformers = optimize_transformers
        self.enable_verbose = enable_verbose
        if self.enable_verbose:
            torch.set_printoptions(threshold=100000)
        self.ref_prompts = ref_prompts
        self.reach_end = False
        os.sched_setaffinity(os.getpid(), core_list)
        os.environ["OMP_NUM_THREADS"] = f"{len(core_list)}"
        os.environ["KMP_AFFINITY"] = f"explicit,proclist={core_list}"
        logger.info(f"rank,{self.rank},OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
        logger.info(f"rank,{self.rank},KMP_AFFINITY={os.environ['KMP_AFFINITY']}")

    def init_model(self):
        if self.device_type != "cpu":
            if self.device_type == "xpu":
                os.environ["ZE_AFFINITY_MASK"] = f"{self.rank//2}.{self.rank%2}"
                logger.info(f"rank,{self.rank},ZE_AFFINITY_MASK={os.environ['ZE_AFFINITY_MASK']}")
                import intel_extension_for_pytorch as ipex
                if self.is_int4:
                    logger.info(f"Instance {self.rank} loading model to Host")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto" if self.device_type == "cpu" else None,
                        low_cpu_mem_usage=True,
                        torch_dtype=self.dtype
                    ).eval()
                    get_memory_usage("Host", "cpu")
                logger.info(f"Casting model to {self.device_type}:{self.rank}")
            else:
                logger.info(f"Casting model to {self.device}")
            self.model.to(self.device)
            get_memory_usage(f"{self.device}", self.device_type, self.device)

        self.model = self.model.to(memory_format=torch.channels_last)
        if self.optimize_transformers and self.device_type == "xpu":
            logger.info(f"Optimizing model on {self.device}")
            try:
                ipex._C.disable_jit_linear_repack()
            except Exception:
                pass
            if self.is_int4:
                self.model = ipex.optimize_transformers(self.model, dtype=self.dtype, is_int4=self.is_int4)
            else:
                self.model = ipex.optimize_transformers(self.model, dtype=self.dtype)
            get_memory_usage(f"{self.device}", self.device_type, self.device)

    def init_profiler(self):
        logger.info(f"Creating profiler on rank {self.rank}")
        os.makedirs("profile", exist_ok=True)
        self.profile_prefix = os.environ.get("PROFILE_PREFIX", f"gpt-j-6B_rank{self.rank}")
        self.profile_iter = os.environ.get("PROFILE_ITER", 1)
        if self.device_type == "xpu":
            self.profile_ctx = torch.autograd.profiler_legacy.profile(use_xpu=True, record_shapes=False)
            # self.profile_ctx = torch.autograd.profiler.profile(use_xpu=True, record_shapes=False)
        elif self.device_type == "cuda":
            self.profile_ctx = torch.autograd.profiler_legacy.profile(use_cuda=True, record_shapes=False)
            # self.profile_ctx = torch.autograd.profiler.profile(use_cuda=True, record_shapes=False)

    def do_warmup(self):
        logger.info(f"Running warmup on rank {self.rank}")
        warmup_dataset_idxes = [*range(len(self.warmup_dataset))]
        warmup_dataset_idxes.sort(key=lambda s: self.warmup_dataset[s][-1], reverse=True)
        # [print(f"idx,{idx},in_len,{self.warmup_dataset[idx][-1]}") for idx in warmup_dataset_idxes]
        warmup_idx = 2348  # input_len=512
        # warmup_batch_size = get_batch_size(self.warmup_dataset[warmup_idx][-1], self.is_int4)
        warmup_batch_size = 95
        query_idx = [warmup_idx] * warmup_batch_size
        logger.debug(f"rank,{self.rank},warmup,bs,{warmup_batch_size},in_len,{self.warmup_dataset[warmup_idx][-1]}")
        input_ids, attn_masks, input_lens, actual_lens = self.warmup_dataset.collect(query_idx)
        start = time.time()
        if self.device_type != "cpu":
            input_ids = input_ids.to(self.device)
            attn_masks = attn_masks.to(self.device)

        outputs = self.inference(input_ids, attn_masks)
        results = self.truncate(input_lens, outputs)

        latency = time.time() - start
        logger.debug(f"rank,{self.rank},warmup,bs,{len(query_idx)},idx,{query_idx},in_len,{input_lens},actual_in_len,{actual_lens},out_len,{outputs.shape[1]},gen_len,{results[0].shape[0]},cost,{latency}")

    def inference(self, input_ids, attn_masks=None):
        with torch.inference_mode(), torch.autocast(device_type=self.device_type, enabled=self.enable_amp, dtype=self.dtype if self.enable_amp else None):
            outputs = self.model.generate(
                input_ids, attention_mask=attn_masks,
                **self.gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return outputs

    def truncate(self, input_lens, outputs):
        truncated_outputs = []
        for i in range(outputs.shape[0]):
            truncated_outputs.append(outputs[i, input_lens[i]:].reshape(-1).cpu().numpy())
        return truncated_outputs

    def fetch_task(self):
        tasks = []
        while len(tasks) == 0:
            for _ in range(self.batch_size):
                try:
                    tasks.append(self.input_queue.get_nowait())
                except Exception as ex:
                    break
        return tasks

    def handle_task(self):
        if self.scenario == "Offline":
            next_task = self.input_queue.get()
            if next_task is None:
                return False
            query_id = next_task.id
            query_idx = next_task.idx
            input_ids = next_task.input_ids
            attn_masks = next_task.attn_masks
            input_lens = next_task.input_lens
            actual_lens = next_task.actual_lens
        else:
            if self.reach_end:
                return False
            next_task = self.fetch_task()
            query_id = []
            query_idx = []
            for i, task in enumerate(next_task):
                if task is None:
                    self.reach_end = True
                    break
                else:
                    query_id.append(task.id[0])
                    query_idx.append(task.idx[0])
            if len(query_idx) == 0:
                return False
            input_ids, attn_masks, input_lens, actual_lens = self.dataset.collect(query_idx, len(query_id))

        # logger.debug(f"before infer,rank,{self.rank},bs,{len(query_idx)},idx,{query_idx},in_len,{input_lens},actual_in_len,{actual_lens}")
        start = time.time()
        if self.device_type != "cpu":
            input_ids = input_ids.to(self.device)
            attn_masks = attn_masks.to(self.device)

        infer_start = time.time()
        outputs = self.inference(input_ids, attn_masks)
        infer_latency = time.time() - infer_start
        results = self.truncate(input_lens, outputs)

        self.output_queue.put(OutputItem(query_id, results))
        self.input_queue.task_done()
        latency = time.time() - start
        logger.debug(f"rank,{self.rank},bs,{len(query_idx)},idx,{query_idx},in_len,{input_lens},actual_in_len,{actual_lens},gen_len,{results[0].shape[0]},end2end_cost,{latency},infer_cost,{infer_latency}")
        if self.enable_verbose:
            # TODO: batch decode
            for i, idx in enumerate(query_idx):
                pred_prompt = ndarray2prompt(results[i], self.dataset.tokenizer)
                print(f"rank,{self.rank},idx,{self.dataset.pick_index[idx]},id,{query_id[i]},in_len,{input_lens[i]},actual_in_len,{actual_lens[i]},actual_out_len,{len(self.dataset.tokenizer.encode(pred_prompt))}"+'\n'+'-'*20)
                print(repr(f"rank,{self.rank},idx,{self.dataset.pick_index[idx]},pred prompt,{pred_prompt}")+'\n'+'-'*20)
                ref_prompt = self.ref_prompts[self.dataset.pick_index[idx]]['output']
                print(repr(f"rank,{self.rank},idx,{self.dataset.pick_index[idx]},ref prompt,{ref_prompt}")+'\n'+'='*20)
        return True

    def run(self):
        self.init_model()

        if self.enable_warmup:
            self.do_warmup()

        if self.enable_profile:
            self.init_profiler()

        with self.cond_var:
            self.alive_counter.value += 1
            self.cond_var.notify()

        try:
            if self.enable_profile:
                iter = 1
                with self.profile_ctx:
                    while self.handle_task() and iter < self.profile_iter:
                        iter += 1
                profile_handler(self.profile_ctx, self.device_type, self.profile_prefix)
            else:
                while self.handle_task():
                    pass
        except Exception as ex:
            logger.error(f"{self.rank}: {ex}")
            import traceback
            trace = traceback.format_exc()
            logger.error(trace)

        with self.cond_var:
            self.alive_counter.value -= 1
            self.input_queue.put(None)
            if self.alive_counter.value == 0:
                self.output_queue.put(None)
            self.cond_var.notify()
            logger.info(f"Exiting inference thread on rank {self.rank}")
