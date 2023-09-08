#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


__doc__ = """Scripts that tests the accuracy of GPTJ-6B model, using either engines generated
from TRT-LLM or Pytorch reference implementation
"""

import argparse
import ctypes
import json
import evaluate
import nltk
import os
import time
import subprocess
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pycuda.driver as cuda
import pycuda.autoprimaryctx

import tensorrt as trt

from code.common import logging
from code.common.constants import TRT_LOGGER, Scenario

# Intel reference implementation, uses 2048-128 as the maximum input seqlen
G_GPTJ6B_MAX_INPUT_SEQLEN = 1919
G_GPTJ6B_MAX_OUTPUT_SEQLEN = 128
G_GPTJ6B_MAX_SEQLEN = 2047
G_GPTJ6B_NUM_LAYERS = 28
G_GPTJ6B_VOCAB_SIZE = 50401
G_CNNDAILYMAIL_CALSET_PATH = None
G_CNNDAILYMAIL_CALMAP_PATH = None
G_CNNDAILYMAIL_VALSET_PATH = "/home/mlperf_inference_data/data/cnn-daily-mail/cnn_eval.json"
G_CNNDAILYMAIL_VALMAP_PATH = None
G_CNNDAILYMAIL_CALIBRATION_CACHE_PATH = None

# Prompt for GPTJ model input
G_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)

nltk.download("punkt", quiet=False)

# Debug flags
G_DEBUG_TRTLLM_PLUGIN = False


def prepare_tokenizer(checkpoint_path, padding_side="left"):
    """
    Prepare the tokenizer for the cnn dailymail
    """
    logging.info(f"Initializing tokenizer from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=G_GPTJ6B_MAX_SEQLEN,
        padding_side=padding_side,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def preprocess_cnndailymail():
    # Load from CNN dailymail
    with open(G_CNNDAILYMAIL_VALSET_PATH, 'r') as fh:
        list_data_dict = json.load(fh)

    sources = [G_PROMPT_INPUT.format_map(
        example) for example in list_data_dict]
    targets = [f"{example['output']}" for example in list_data_dict]

    logging.info(
        f"Loaded {len(sources)} samples from {G_CNNDAILYMAIL_VALSET_PATH}")
    return sources, targets


def postprocess_text(preds, targets):
    # Post-process output texts for ROUGE evaluation
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def calculate_rouge_score(preds, targets):
    logging.info("Calculating ROUGE scores...")
    metric = evaluate.load("rouge")
    preds, targets = postprocess_text(preds, targets[0:len(preds)])
    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.sum(prediction_lens)
    result["gen_num"] = len(preds)

    return result


class HostDeviceMem(object):
    """
    A class to encapsulate a input/output tensor
    """

    def __init__(
        self,
        host: Optional[pycuda.driver.DeviceAllocation],
        device: Optional[pycuda.driver.DeviceAllocation],
        tensor_shape: Tuple,
        tensor_dtype: trt.DataType,
        tensor_location: trt.TensorLocation,
        is_allocated: bool
    ):
        self.host = host
        self.device = device
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.tensor_location = tensor_location
        self.is_allocated = is_allocated


class GPTEngineRunner:
    """
    TRT engine runner class for GPT, specifically for Tekit-generated engines.
    Encapsulate the stream handling, execution context management, and buffer allocation.
    """

    def __init__(
        self,
        engine_file: str,
        batch_size: int,
        tokenizer,
        gen_kwargs: dict,
        decoding_step: bool = False,
        verbose: bool = False,
    ):
        self.engine_file = engine_file
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.gen_kwargs = gen_kwargs
        self.decoding_step = decoding_step
        self.verbose = verbose
        self.num_beams = self.gen_kwargs['num_beams']
        self.device = torch.device(f'cuda:0')  # Hard code to use device 0

        # Load TRT-LLM dynamic decoder and set up for decoding
        self.dynamic_decoder = torch.classes.FasterTransformer.DynamicDecodeOp(
            G_GPTJ6B_VOCAB_SIZE,  # vocab_size
            G_GPTJ6B_VOCAB_SIZE,  # padded vocab_size (for TP)
            1,  # TP
            1,  # PP
            torch.float16,  # Decoder T for logits, hard coded to FP16 now.
        )
        self.top_k = torch.full([batch_size], 1, dtype=torch.int32)
        self.top_p = torch.full([batch_size], 1.0, dtype=torch.float32)
        self.temperature = torch.full([batch_size], 1.0, dtype=torch.float32)
        self.repetition_penalty = None
        self.min_length = torch.full([batch_size], self.gen_kwargs["min_new_tokens"], dtype=torch.int32)
        self.length_penalty = torch.FloatTensor([1])
        self.presence_penalty = None
        self.random_seed = torch.zeros([batch_size], dtype=torch.int64)
        self.beam_search_diversity_rate = self.top_p_decay = self.top_p_min =\
            self.top_p_reset_ids = self.embedding_bias_opt = self.stop_words_list = \
            self.bad_words_list = None
        self.dynamic_decoder.setup(
            self.batch_size, self.num_beams, self.top_k, self.top_p,
            self.temperature, self.repetition_penalty, self.presence_penalty,
            self.min_length, self.length_penalty,
            self.beam_search_diversity_rate, self.random_seed, self.top_p_decay,
            self.top_p_min, self.top_p_reset_ids)

        # Load gather_tree ops for beam search
        self.gather_tree = torch.ops.tensorrt_llm.gather_tree

        # Load engine from the engine file
        logging.info(f"Loading the engine from: {self.engine_file}")
        TRT_LOGGER.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            buf = f.read()
            self.engine = runtime.deserialize_cuda_engine(buf)

        self._validate_engine_tensor()

        # Pre-allocate the buffer for the batch size specified.
        self.input_tensor_map = {}
        self.output_tensor_map = {}
        self.stream = cuda.Stream()
        print(f"stream handle: {self.stream.handle} torch stream: {torch.cuda.current_stream().cuda_stream}")
        self._total_device_mem_byte = 0
        self._pre_allocate_buffers()

        # Create the execution context for both phases
        self.ctx_context = self.engine.create_execution_context()
        self.gen_context = self.engine.create_execution_context()

    def _validate_engine_tensor(self):
        """
        Verify that the Engine contains same tensor names as expected.
        """
        logging.info(f"Validating the tensor in the GPTJ-6B engine...")
        found_tensor_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]

        expected_tensor_names = ['input_ids'] \
            + [f'past_key_value_{i}' for i in range(G_GPTJ6B_NUM_LAYERS)] \
            + ['logits'] \
            + [f'present_key_value_{i}' for i in range(G_GPTJ6B_NUM_LAYERS)] \
            + ['input_lengths'] \
            + ['position_ids'] \
            + ['max_input_length'] \
            + ['last_token_ids'] \
            + ['cache_indirection'] \
            + ['sequence_length', 'past_key_value_length', 'masked_tokens']

        if set(expected_tensor_names) != set(found_tensor_names):
            logging.error(
                f"These expected tensors not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"These tensors are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            raise RuntimeError("Tensor names in engine are not the same as expected.")

        logging.info(f"Validation complete! Total {self.engine.num_io_tensors} tensors")

    def _pre_allocate_buffers(self):
        """
        Read the engine for IO tensor information.
        Allocate buffers whose shape don't change across inference for GPTJ-6B
        For others, we allocate the maximum size, and adjust the shape dynamically during runtime.
        """
        # Allocate buffers for inference
        num_io_tensors = self.engine.num_io_tensors

        # Get the max batch size, beam size, and max_seq_len from the "cache_indirection" input
        tensor_name = "cache_indirection"
        opt_max_shape = self.engine.get_tensor_profile_shape(name=tensor_name, profile_index=0)[2]
        self.engine_max_batch_size = opt_max_shape[0]
        self.engine_beam_width = opt_max_shape[1]
        self.engine_max_sum_seqlen = opt_max_shape[2]

        # Get the max input seqlen from input id
        tensor_name = "input_ids"
        opt_max_shape = self.engine.get_tensor_profile_shape(name=tensor_name, profile_index=0)[2]
        self.engine_max_input_seqlen = opt_max_shape[1]
        self.engine_max_output_seqlen = self.engine_max_sum_seqlen - self.engine_max_input_seqlen

        tensor_name = "logits"
        tensor_shape = self.engine.get_tensor_shape(tensor_name)
        self.engine_vocab_size = tensor_shape[1]

        # Size checks
        assert self.batch_size <= self.engine_max_batch_size, \
            f"Actual batch_size {self.batch_size} cannot be larger than the maximum engine batch size {self.engine_max_batch_size}"
        assert self.num_beams == self.engine_beam_width, \
            f"Specified num_beams={self.num_beams}, but engines suggest num_beams={self.engine_beam_width}"
        assert self.engine_vocab_size == G_GPTJ6B_VOCAB_SIZE, \
            f"Engine vocab size is {self.engine_vocab_size}, while expecting {G_GPTJ6B_VOCAB_SIZE}"
        logging.info(f"The engine has {num_io_tensors} IO tensors, with max BS {self.engine_max_batch_size}, num_beams {self.engine_beam_width}, " +
                     f"max input seqlen {self.engine_max_input_seqlen}, max combined seqlen {self.engine_max_sum_seqlen}")

        for tensor_idx in range(num_io_tensors):
            tensor_name = self.engine.get_tensor_name(tensor_idx)
            tensor_io_type = self.engine.get_tensor_mode(tensor_name)
            tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
            tensor_location = self.engine.get_tensor_location(tensor_name)
            # Assume the context profile index is 0
            tensor_format = self.engine.get_tensor_format(
                tensor_name, profile_index=0)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            logging.debug(
                f"Tensor idx: {tensor_idx}, Tensor name: {tensor_name}, iotype: {tensor_io_type}," +
                f" dtype: {tensor_dtype}, location: {tensor_location}, format: {tensor_format}, shape: {tensor_shape}")

            # Fill the batch_size and seq_len into the tensor shape for memory allocation
            # Note:
            # 1. We only allocate the actual batch size instead of max bs to save memory.
            # 2. For shapes that change between context and generation, we allocate to the maximum and adjust dynamically
            if tensor_name == "input_ids":
                # Shape (-1, -1) -> (BS, runtime_input_seqlen) in context
                # Shape (-1, -1) -> (BS * beam_width, 1) in generation
                tensor_shape[0] = self.batch_size * self.num_beams
                tensor_shape[1] = self.engine_max_input_seqlen
            elif tensor_name == "sequence_length":
                # The padded sequence length of the inputs
                # Shape (-1) -> (BS) for context
                # Shape (-1) -> (BS * beam_width) for generation
                tensor_shape[0] = self.batch_size * self.num_beams
            elif tensor_name == "past_key_value_length":
                # The KV length tensor records the step of the inference, which is used for attention
                # (-1) -> (2)
                tensor_shape[0] = 2
                # TODO: Make sure the tensor is on host, not device
                tensor_location = trt.TensorLocation.HOST
            elif tensor_name == "logits":
                # Shape (-1, vocab size) -> (bs, vocab_size) for context
                # Shape (-1, vocab size) -> (bs * beam_width, vocab_size) for generation
                tensor_shape[0] = self.batch_size * self.num_beams
            elif tensor_name == "input_lengths":
                # the real length of each input sequences
                # (-1) -> (BS) for context
                # (-1) -> (BS * beam_width) for generation
                tensor_shape[0] = self.batch_size * self.num_beams
            elif tensor_name == "max_input_length":
                # Record the largest input length as the shape
                # (-1) -> (runtime_input_seqlen) for context
                tensor_shape[0] = self.engine_max_input_seqlen
            elif tensor_name == "position_ids":
                # This is BS copies of arange(runtime_input_seqlen). For positional embedding
                # (-1, -1) -> (BS, runtime_input_seqlen) for context
                # (-1, -1) -> (BS * beam_width, 1) for generation (<-- TODO: Check correctness)
                tensor_shape[0] = self.batch_size * self.num_beams
                tensor_shape[1] = self.engine_max_input_seqlen
            elif tensor_name == "last_token_ids":
                # This has the same shape as input_lengths
                # (-1) -> (BS) for context
                # (-1) -> (BS * beam_width) for generation
                tensor_shape[0] = self.batch_size * self.num_beams
            elif tensor_name == "masked_tokens":
                # Input mask for BS>1, which signals the real sequence lengths
                # (-1, -1) -> (BS, max_seqlen) for context
                # (-1, -1) -> (BS * beam_width, max_seqlen) for gen
                tensor_shape[0] = self.batch_size * self.num_beams
                tensor_shape[1] = self.engine_max_sum_seqlen
            elif tensor_name == "cache_indirection":
                # For beam search
                # (-1, beam_width, -1) -> (BS, beam_width, max_seqlen)
                tensor_shape[0] = self.batch_size
                tensor_shape[1] = self.num_beams
                tensor_shape[2] = self.engine_max_sum_seqlen
            elif any([s in tensor_name for s in ['present_key_value', 'past_key_value']]):
                # (-1, 2, nhead, -1, dhead) -> (bs, 2, nhead, seqlen, dhead) for context
                # (-1, 2, nhead, -1, dhead) -> (bs * beam_width, 2, nhead, seqlen, dhead) for generation
                tensor_shape[0] = self.batch_size * self.num_beams
                tensor_shape[-2] = self.engine_max_sum_seqlen
            else:
                raise RuntimeError(
                    f"Unknown IO tensor: {tensor_name} at idx {tensor_idx}")

            # Allocate device/pinned memory and save to the tensor address map
            # TODO: handle dtype that might need padding, e.g. CHW4, CDHW32 etc
            tensor_volume = trt.volume(tensor_shape)
            tensor_size_byte = tensor_volume * tensor_dtype.itemsize
            logging.debug(
                f"name: {tensor_name}, shape: {tensor_shape}, byte: {tensor_size_byte}")
            assert tensor_size_byte > 0, f"{tensor_name} has dynamic shape that is not resolved yet: {tensor_shape}"

            if 'present' in tensor_name:
                # As of TRTLLM-9 - fp16, present KV Cache can reuse past KV Cache buffer for in-place cache update
                # Note that output KV Cache is never copied back to host at the moment
                past_tensor_name = tensor_name.replace('present', 'past')
                assert past_tensor_name in self.input_tensor_map, f"{past_tensor_name} needs to be allocated before {tensor_name} is allocated."
                # This will be a shallow copy of the past kv cache, so the memory will not be double freed
                self.output_tensor_map[tensor_name] = self.input_tensor_map[past_tensor_name]
            else:
                # TODO: No need to allocate tensor for tensorlocation not on device
                device_mem = cuda.mem_alloc(tensor_size_byte)
                self._total_device_mem_byte += tensor_size_byte
                if tensor_io_type == trt.TensorIOMode.OUTPUT:
                    # Allocate pinned memory for output to copy back
                    host_mem = cuda.pagelocked_empty(
                        tensor_volume, trt.nptype(tensor_dtype))
                    self.output_tensor_map[tensor_name] = HostDeviceMem(
                        host_mem, device_mem, tensor_shape, tensor_dtype, tensor_location, True)
                    logging.info(
                        f"Allocating host pinned memory for {tensor_name}, shape: {tensor_shape}, dtype: {trt.nptype(tensor_dtype)}")
                elif tensor_io_type == trt.TensorIOMode.INPUT:
                    # We need ping_pong buffer for cache indirection, thus allocating two copies
                    if tensor_name == "cache_indirection":
                        device_mem_ping = device_mem
                        device_mem_pong = cuda.mem_alloc(tensor_size_byte)
                        self._total_device_mem_byte += tensor_size_byte
                        self.input_tensor_map[tensor_name + "_0"] = HostDeviceMem(
                            None, device_mem_ping, tensor_shape, tensor_dtype, tensor_location, True)
                        self.input_tensor_map[tensor_name + "_1"] = HostDeviceMem(
                            None, device_mem_pong, tensor_shape, tensor_dtype, tensor_location, True)
                    else:
                        self.input_tensor_map[tensor_name] = HostDeviceMem(
                            None, device_mem, tensor_shape, tensor_dtype, tensor_location, True)
                else:
                    raise RuntimeError(
                        f"Unknown tensor type: {tensor_io_type} for {tensor_name}")

        logging.info(
            f"Allocated a total of {self._total_device_mem_byte} byte during pre-allocation.")

    def _runtime_adjust_shape(self):
        """
        Set the shape and allocate device memory for runtime variant buffers
        """
        runtime_is_context = self.step == 0

        for tensor_name, host_device_mem in self.input_tensor_map.items():
            assert host_device_mem.is_allocated, f"{tensor_name} should have been allocated in context."
            if tensor_name == "input_ids":
                # Shape (-1, -1) -> (BS, runtime_input_seqlen) in context
                # Shape (-1, -1) -> (BS * beam_width, 1) in generation
                if runtime_is_context:
                    host_device_mem.tensor_shape[0] = self.batch_size
                    host_device_mem.tensor_shape[1] = self.input_seqlen
                elif self.step == 1:
                    host_device_mem.tensor_shape[0] = self.batch_size * self.num_beams
                    host_device_mem.tensor_shape[1] = 1
            elif any([s in tensor_name for s in ['sequence_length', 'input_lengths', 'last_token_ids',
                                                 'logits', 'masked_tokens']]):
                # Shape (-1, ...) -> (BS, ...) for context
                # Shape (-1, ...) -> (BS * beam_width, ...) for generation
                if runtime_is_context:
                    host_device_mem.tensor_shape[0] = self.batch_size
                elif self.step == 1:
                    host_device_mem.tensor_shape[0] = self.batch_size * self.num_beams
            elif tensor_name == "max_input_length":
                # (-1) -> (runtime_input_seqlen) for context
                if runtime_is_context:
                    host_device_mem.tensor_shape[0] = self.input_seqlen
            elif tensor_name == "position_ids":
                # (-1, -1) -> (BS, runtime_input_seqlen) for context
                # (-1, -1) -> (BS * beam_width, 1) for generation
                if runtime_is_context:
                    host_device_mem.tensor_shape[0] = self.batch_size
                    host_device_mem.tensor_shape[1] = self.input_seqlen
                elif self.step == 1:
                    host_device_mem.tensor_shape[0] = self.batch_size * self.num_beams
                    host_device_mem.tensor_shape[1] = 1
            elif any([s in tensor_name for s in ['past_key_value', 'present_key_value']]) and \
                tensor_name != "past_key_value_length":
                # (-1, 2, nhead, -1, dhead) -> (bs, 2, nhead, seqlen, dhead) for context
                # (-1, 2, nhead, -1, dhead) -> (bs * beam_width, 2, nhead, seqlen, dhead) for generation
                if runtime_is_context:
                    host_device_mem.tensor_shape[0] = self.batch_size
                elif self.step == 1:
                    host_device_mem.tensor_shape[0] = self.batch_size * self.num_beams

    def _runtime_deallocate_buffers(self):
        """
        Deallocate buffers that are allocated during the runtime:
        """
        # TODO: Right now all buffers are pre-allocated to the maximum and never deallocated
        # We can re-start deallocating if we hit memory capacity issue.
        return

    def _setup_execution_context(self):
        """
        This function will:
        1. set execution context tensor addresses
        2. set execution context tensor runtime shape
        """
        runtime_is_context = self.step == 0
        context = self.ctx_context if runtime_is_context else self.gen_context

        for tensor_name, host_device_mem in self.input_tensor_map.items():
            assert host_device_mem.is_allocated, f"{tensor_name} is not allocated yet!"
            # Handle cache_indirection based on the step, note it's different from TRT-LLM because we count context as step 0.
            # step odd: use 0; step even: use 1 (including context)
            if "cache_indirection" in tensor_name:
                cache_to_use = 1 - self.step % 2
                if tensor_name == f"cache_indirection_{cache_to_use}":
                    tensor_name = "cache_indirection"
                else:
                    continue
            if host_device_mem.tensor_location == trt.TensorLocation.DEVICE:
                context.set_tensor_address(tensor_name, int(host_device_mem.device))
            elif host_device_mem.tensor_location == trt.TensorLocation.HOST:
                context.set_tensor_address(tensor_name, host_device_mem.host)
            context.set_input_shape(tensor_name, host_device_mem.tensor_shape)
            # logging.debug(f"Setting input tensor {tensor_name} to shape {host_device_mem.tensor_shape}, address {int(host_device_mem.device):x}")

        for tensor_name, host_device_mem in self.output_tensor_map.items():
            assert host_device_mem.is_allocated, f"{tensor_name} is not allocated yet!"
            context.set_tensor_address(
                tensor_name, int(host_device_mem.device))
            # logging.debug(f"Setting output tensor {tensor_name} to shape {host_device_mem.tensor_shape}, address {int(host_device_mem.device):x}")

        return context

    def __call__(self, inputs):
        """
        Entry point of the GPT inference, which handle the loops.
        """
        # Generation step counts
        self.step = 0

        # As of 7/6/2023, seems like both context uses profile 0
        self.ctx_context.set_optimization_profile_async(0, self.stream.handle)
        self.gen_context.set_optimization_profile_async(0, self.stream.handle)

        # The actual output ids to be returned.
        self.output_ids = None

        buffer_dict = {}
        # 0: input_ids (BS, max_seqlen) int32
        # 1: masked_tokens (BS, max_seqlen) the tokens which are masked are 1, otherwise 0. (0,0,...,1,1,..,0,...0)
        # 2: input_seqlens (BS) int32 actual input seqlen for each sequence
        buffer_dict['input_ids'] = inputs[0]
        buffer_dict['masked_tokens'] = inputs[1]
        buffer_dict['input_lengths'] = inputs[2]
        self.actual_batch_size = buffer_dict['input_ids'].shape[0]
        self.input_seqlen = buffer_dict['input_ids'].shape[1]
        self.batch_max_sum_seqlen = self.input_seqlen + self.engine_max_output_seqlen

        # Pad the input to the max batch size if actual batch size is smaller
        if self.actual_batch_size < self.batch_size:
            repeat = [1] * (self.actual_batch_size - 1) + \
                [self.batch_size - self.actual_batch_size + 1]
            for name in ['input_ids', 'masked_tokens', 'input_lengths']:
                buffer_dict[name] = np.repeat(buffer_dict[name], repeat, axis=0)
                assert buffer_dict[name].shape[0] == self.batch_size

        # TRT-LLM decoding Op variables:
        self.end_ids = torch.full(
            (self.batch_size * self.num_beams, ),
            50256, dtype=torch.int32, device=self.device
        )
        self.decoder_output_ids = torch.zeros(
            (G_GPTJ6B_MAX_SEQLEN, self.batch_size, self.num_beams),
            dtype=torch.int32, device=self.device
        )
        self.parent_ids = torch.zeros(
            (G_GPTJ6B_MAX_SEQLEN, self.batch_size, self.num_beams),
            dtype=torch.int32, device=self.device
        )
        if self.num_beams > 1:
            self.cum_log_probs = torch.full(
                (self.batch_size, self.num_beams),
                -1e20, dtype=torch.float32, device=self.device
            )
            self.cum_log_probs[:, 0] = 0.0
        else:
            self.cum_log_probs = None
        self.log_probs = None
        self.finished = torch.zeros(
            (self.batch_size, self.num_beams),
            dtype=torch.bool,
            device=self.device
        )
        self.sequence_limit_lengths = torch.full(
            (self.batch_size, 1), G_GPTJ6B_MAX_SEQLEN,
            dtype=torch.int32, device=self.device
        )
        self.sequence_lengths = torch.full(
            (self.batch_size * self.num_beams, 1), self.input_seqlen,
            dtype=torch.int32, device=self.device
        )

        # EOS-based early stopping
        unfinished_sequence = np.ones(self.batch_size, dtype=np.int32)
        # Sentences after actual batch size are automatically finished
        unfinished_sequence[self.actual_batch_size:] = 0
        eos_token_id_np = np.array([self.tokenizer.eos_token_id])
        for i in range(self.engine_max_output_seqlen):
            # Create buffers based on stages
            if self.step == 0:
                sequence_length_buffer = np.ones(self.batch_size, dtype=np.int32)
                buffer_dict["position_ids"] = torch.IntTensor(range(self.input_seqlen)).reshape(
                    [1, -1]).expand([self.batch_size, -1]).numpy()
                buffer_dict["last_token_ids"] = np.copy(buffer_dict['input_lengths'])
                # Zero-pad and flip the attention mask according to Tekit
                # 6/6/2023: Tekit fmha use 0 for real sequence, 1 for padding, 0 for dummy output part again.
                buffer_dict["masked_tokens"] = np.pad(buffer_dict["masked_tokens"],
                                                      ((0, 0), (0, self.engine_max_sum_seqlen - self.input_seqlen)), 'constant', constant_values=1)
                buffer_dict["masked_tokens"] = 1 - buffer_dict['masked_tokens']
                buffer_dict["sequence_length"] = sequence_length_buffer * (self.input_seqlen + self.step)
                # past_key_value_length needs to be on host, instead of device. Using torch
                past_key_value_length = torch.tensor([0, 1], dtype=torch.int32)
                self.input_tensor_map["past_key_value_length"].host = past_key_value_length.data_ptr()
                buffer_dict["max_input_length"] = np.zeros((self.input_seqlen), dtype=np.int32)
                buffer_dict["cache_indirection_0"] = np.zeros((self.batch_size, self.num_beams, self.engine_max_sum_seqlen), dtype=np.int32)
                buffer_dict["cache_indirection_1"] = np.zeros((self.batch_size, self.num_beams, self.engine_max_sum_seqlen), dtype=np.int32)
            else:
                # In TRT-LLM, the buffer is set after the runtime, so we need to minus one from the step.
                buffer_dict["position_ids"] = (buffer_dict['input_lengths'] + self.step - 1).reshape([-1, 1])
                buffer_dict["last_token_ids"] = np.ones_like(buffer_dict['input_lengths'])
                buffer_dict["sequence_length"] = sequence_length_buffer * (self.input_seqlen + self.step - 1)
                # past_key_value_length needs to be on host, instead of device. Using torch
                past_key_value_length = torch.tensor([self.input_seqlen + self.step - 1, 0], dtype=torch.int32)
                self.input_tensor_map["past_key_value_length"].host = past_key_value_length.data_ptr()
                if self.step == 1:
                    buffer_dict.pop("cache_indirection_0")
                    buffer_dict.pop("cache_indirection_1")
                if self.step == 2:
                    buffer_dict.pop("masked_tokens")

            def assert_same_size(arr: np.ndarray, mem_obj: HostDeviceMem, name: str):
                arr_size = arr.nbytes
                buffer_shape = mem_obj.tensor_shape
                buffer_dtype = mem_obj.tensor_dtype
                buffer_size = trt.volume(buffer_shape) * buffer_dtype.itemsize
                assert arr_size == buffer_size, \
                    f"Size mismatch: {name} numpy array is {arr.dtype} {arr.shape}, but buffer has {buffer_dtype} {buffer_shape}"

            # Adjust the shape of the buffer to use the right amount of buffers.
            self._runtime_adjust_shape()
            context = self._setup_execution_context()

            # Handle H2D for input tensors
            # Make sure the np array has the same byte as the buffer
            for name in buffer_dict.keys():
                assert_same_size(buffer_dict[name], self.input_tensor_map[name], name)
                cuda.memcpy_htod_async(self.input_tensor_map[name].device, np.ascontiguousarray(
                    buffer_dict[name]), self.stream)
                # logging.debug(f"Step {self.step}: Tensor {name} copied from host to device, dtype: {buffer_dict[name].dtype} shape: {buffer_dict[name].shape} content: {buffer_dict[name]}")

            context.execute_async_v3(self.stream.handle)

            # Handle D2H for output logtis
            output_mem = self.output_tensor_map["logits"]
            cuda.memcpy_dtoh_async(
                output_mem.host, output_mem.device, self.stream)
            self.stream.synchronize()
            logits = np.copy(output_mem.host)
            # logging.debug(f"Host output has shape: {output_mem.host.shape}, dtype: {output_mem.host.dtype}")

            if self.num_beams == 1:
                # Prepare inputs to decoderop
                next_token_logits = torch.from_numpy(logits).to(torch.float16).\
                    reshape(self.batch_size, self.num_beams, -1).cuda()
                decode_step = self.input_seqlen + self.step
                input_lengths = torch.from_numpy(buffer_dict['input_lengths']).cuda()
                # For greedy, cache_indirection is not needed, so all zeros
                this_src_cache_indirection = torch.zeros(
                    (self.batch_size, self.num_beams, self.engine_max_sum_seqlen), dtype=torch.int32).cuda()
                this_tgt_cache_indirection = torch.zeros(
                    (self.batch_size, self.num_beams, self.engine_max_sum_seqlen), dtype=torch.int32).cuda()

                should_stop = self.dynamic_decoder.forward(
                    next_token_logits, decode_step, self.input_seqlen, 0,
                    self.batch_size, self.end_ids, self.top_k, self.top_p,
                    self.temperature, self.repetition_penalty,
                    self.presence_penalty, self.min_length, self.length_penalty,
                    self.beam_search_diversity_rate, self.top_p_decay,
                    self.top_p_min, self.top_p_reset_ids,
                    self.embedding_bias_opt,
                    input_lengths.reshape((self.batch_size, self.num_beams)), self.sequence_limit_lengths,
                    self.stop_words_list, self.bad_words_list,
                    this_src_cache_indirection, self.decoder_output_ids, self.finished,
                    self.sequence_lengths, self.cum_log_probs, self.log_probs,
                    self.parent_ids, this_tgt_cache_indirection)

                # Decoder runs on a different stream, so synchronize again
                torch.cuda.current_stream().synchronize()
                if self.decoding_step:
                    logging.info(f"step {self.step} output_ids: {self.decoder_output_ids[self.step + self.input_seqlen]} finished: {self.finished}")

                # Prepare input ids for next round
                buffer_dict['input_ids'] = self.decoder_output_ids[self.step + self.input_seqlen].cpu().numpy()

                # Implement should_stop
                if should_stop.item():
                    break
            elif self.num_beams > 1:
                # For beam search, tile the following input/output in step 0
                # input_lengths, sequence_length, masked_tokens, all kvcache, logits
                if self.step == 0:
                    def tile_beam_width(tensor, num_beams: int):
                        new_shape = np.array(tensor.shape)
                        new_shape[0] = new_shape[0] * num_beams

                        tile_size = np.ones(new_shape.shape, dtype=np.int32)
                        tile_size = np.insert(tile_size, 1, num_beams)

                        if type(tensor) == torch.Tensor:
                            new_tensor = torch.unsqueeze(tensor, 1)
                            new_tensor = new_tensor.tile(tile_size.tolist())
                            new_tensor = new_tensor.reshape(new_shape.tolist())
                        elif type(tensor) == np.ndarray:
                            new_tensor = np.expand_dims(tensor, 1)
                            new_tensor = np.tile(new_tensor, tile_size.tolist())
                            new_tensor = np.reshape(new_tensor, new_shape.tolist())
                        return new_tensor

                    buffer_dict['input_lengths'] = tile_beam_width(buffer_dict['input_lengths'], self.num_beams)
                    sequence_length_buffer = tile_beam_width(sequence_length_buffer, self.num_beams)
                    buffer_dict['masked_tokens'] = tile_beam_width(buffer_dict['masked_tokens'], self.num_beams)

                    # For logits and kvcache, we need to d2h, tile them, h2d because we don't have kernels
                    # The logits shape is (BS * beam_size * vocab_size). But for context, only (BS * vocab_size) is meaningful
                    logits = logits[:self.batch_size * self.engine_vocab_size].reshape(self.batch_size, self.engine_vocab_size)
                    logits = tile_beam_width(logits, self.num_beams)

                    kvcache_names = [f'present_key_value_{n}' for n in range(G_GPTJ6B_NUM_LAYERS)]
                    for name in kvcache_names:
                        host_device_mem = self.output_tensor_map[name]
                        host_cache = np.zeros(host_device_mem.tensor_shape,
                                              dtype=trt.nptype(host_device_mem.tensor_dtype))
                        cuda.memcpy_dtoh_async(host_cache, host_device_mem.device, self.stream)
                        self.stream.synchronize()
                        host_cache = tile_beam_width(host_cache, self.num_beams)
                        cuda.memcpy_htod_async(host_device_mem.device, np.ascontiguousarray(
                            host_cache), self.stream)

                # Prepare decoding operations
                next_token_logits = torch.from_numpy(logits).to(torch.float16).\
                    reshape(self.batch_size, self.num_beams, -1).cuda()
                decode_step = self.input_seqlen + self.step
                input_lengths = torch.from_numpy(buffer_dict['input_lengths']).cuda()

                # Copy cache indirection depending on the step (d2d)
                tgt_cache = self.step % 2
                src_cache = 1 - self.step % 2
                tensor_volume = trt.volume(self.input_tensor_map[f"cache_indirection_{src_cache}"].tensor_shape)
                tensor_dtype = self.input_tensor_map[f"cache_indirection_{src_cache}"].tensor_dtype
                cache_indirection_size = tensor_volume * tensor_dtype.itemsize
                this_src_cache_indirection = torch.zeros(
                    (self.batch_size, self.num_beams, self.engine_max_sum_seqlen), dtype=torch.int32).cuda()
                this_tgt_cache_indirection = torch.zeros(
                    (self.batch_size, self.num_beams, self.engine_max_sum_seqlen), dtype=torch.int32).cuda()
                cuda.memcpy_dtod_async(
                    this_src_cache_indirection.data_ptr(),
                    int(self.input_tensor_map[f"cache_indirection_{src_cache}"].device),
                    cache_indirection_size,
                    self.stream
                )
                # Synchronize because decoder runs on a different stream
                self.stream.synchronize()

                should_stop = self.dynamic_decoder.forward(
                    next_token_logits, decode_step, self.input_seqlen, 0,
                    self.batch_size, self.end_ids, self.top_k, self.top_p,
                    self.temperature, self.repetition_penalty,
                    self.presence_penalty, self.min_length, self.length_penalty,
                    self.beam_search_diversity_rate, self.top_p_decay,
                    self.top_p_min, self.top_p_reset_ids,
                    self.embedding_bias_opt,
                    input_lengths.reshape((self.batch_size, self.num_beams)), self.sequence_limit_lengths,
                    self.stop_words_list, self.bad_words_list,
                    this_src_cache_indirection, self.decoder_output_ids, self.finished,
                    self.sequence_lengths, self.cum_log_probs, self.log_probs,
                    self.parent_ids, this_tgt_cache_indirection)

                # Decoder runs on a different stream, so synchronize again
                torch.cuda.current_stream().synchronize()
                if self.decoding_step:
                    logging.info(f"step {self.step} output_ids: {self.decoder_output_ids[self.step + self.input_seqlen]} cum probs: {self.cum_log_probs}")
                    # logging.info(f"this_src_cache_indirection: {this_src_cache_indirection[:, :, :20]} this_tgt_cache_indirection: {this_src_cache_indirection[:, :, :20]}")
                    logging.info(f"step: {self.step} finished: {self.finished}")
                # Copy cache indirection back
                cuda.memcpy_dtod_async(
                    int(self.input_tensor_map[f"cache_indirection_{tgt_cache}"].device),
                    this_tgt_cache_indirection.data_ptr(),
                    cache_indirection_size,
                    self.stream
                )

                # Prepare input ids for next round
                buffer_dict['input_ids'] = self.decoder_output_ids[self.step + self.input_seqlen].cpu().numpy()

                # Implement should_stop
                if should_stop.item():
                    break

            # Increment step
            self.step += 1

        self._runtime_deallocate_buffers()
        if self.num_beams == 1:
            # return self.output_ids[:self.actual_batch_size]
            self.decoder_output_ids = self.decoder_output_ids.reshape(
                [self.decoder_output_ids.shape[0], self.batch_size, self.num_beams])
            self.decoder_output_ids = self.decoder_output_ids.transpose(0, 1)
            return self.decoder_output_ids[:self.actual_batch_size, self.input_seqlen:self.input_seqlen + self.step, :].squeeze(-1)
        else:
            final_output_ids = self.gather_tree(
                self.sequence_lengths, self.decoder_output_ids, self.parent_ids,
                self.end_ids, input_lengths, self.batch_size, self.num_beams,
                self.input_seqlen, G_GPTJ6B_MAX_SEQLEN
            )
            final_output_ids = final_output_ids.permute([2, 0, 1])
            final_output_ids = final_output_ids.transpose(0, 1)
            # For beam search, the output is closely next to the input. So create new tensors and copy over
            real_input_lengths = inputs[2]
            output_copy = torch.ones(
                self.batch_size, G_GPTJ6B_MAX_OUTPUT_SEQLEN, self.num_beams, dtype=torch.int32
            ).cuda() * self.tokenizer.eos_token_id
            for i in range(self.actual_batch_size):
                start_idx = real_input_lengths[i]
                out_len = final_output_ids.shape[1] - start_idx
                out_len = min(out_len, G_GPTJ6B_MAX_OUTPUT_SEQLEN)
                output_copy[i, :out_len, :] = final_output_ids[i, start_idx:start_idx+out_len, :]
            # Pick the highest cum_log_prob
            top_idx = torch.argmax(self.cum_log_probs, dim=1)
            output_copy =  output_copy[range(self.batch_size), :, top_idx]
            return output_copy[:self.actual_batch_size, :].cpu().numpy()

    def __del__(self):
        """ Deallocate all the device memory as needed. """
        for tensor_name, host_device_mem in self.input_tensor_map.items():
            if host_device_mem.is_allocated:
                if isinstance(host_device_mem.device, pycuda.driver.DeviceAllocation):
                    host_device_mem.device.free()
                host_device_mem.is_allocated = False
        for tensor_name, host_device_mem in self.output_tensor_map.items():
            # The present KV Cache should have been deallocated, so no double free
            if host_device_mem.is_allocated:
                if isinstance(host_device_mem.device, pycuda.driver.DeviceAllocation):
                    host_device_mem.device.free()
                host_device_mem.is_allocated = False
        del self.stream


class TRTTester:
    """
    Wrapper class to encapsulate the TRT tester util functions.
    """

    def __init__(self,
                 engine_file: str,
                 batch_size: int,
                 precision: str,
                 pyt_ckpt_path: str,
                 num_beams: Optional[int] = 1,
                 use_dla: Optional[bool] = False,
                 skip_engine_build: Optional[bool] = False,
                 engine_build_only: Optional[bool] = False,
                 decoding_step: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                 ):
        """
        Test GPT model through the TRT path.
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.engine_file = engine_file
        self.cache_file = G_CNNDAILYMAIL_CALIBRATION_CACHE_PATH
        self.precision = precision
        self.pyt_ckpt_path = pyt_ckpt_path
        self.decoding_step = decoding_step

        self.gen_kwargs = {
            "early_stopping": True,
            "max_new_tokens": G_GPTJ6B_MAX_OUTPUT_SEQLEN,
            "min_new_tokens": 30,
            "num_beams": num_beams,
        }

        # TensorRT engine related fields
        if use_dla:
            self.dla_core = 0
        else:
            self.dla_core = None

        # Initiate the plugin and logger
        # Use the global singleton, which is required by TRT.
        self.logger = TRT_LOGGER
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        trt.init_libnvinfer_plugins(self.logger, "")

        logging.info(f"Loading plugins from the plugin .so")
        if G_DEBUG_TRTLLM_PLUGIN:
            # Load tekit plugins by reading the .so directly
            # TODO: Remove later
            tekit_plugin_path = "./build/plugins/libnvinfer_plugin_tensorrt_llm.so"
            handle = ctypes.CDLL(tekit_plugin_path)
            handle.initLibNvInferPlugins.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p]
            handle.initLibNvInferPlugins.restype = ctypes.c_bool
            success = handle.initLibNvInferPlugins(None, "tensorrt_llm".encode('utf-8'))
            assert success, "Failed to load tekit plugins."
        else:
            try:
                import tensorrt_llm
            except:
                logging.error("TRT-LLM is not installed, please run make clone_trt_llm && make build_trt_llm")
                raise
            tensorrt_llm.plugin._load_plugin_lib()

        # Initiate tokenizer, use right padding for TRT-LLM
        self.tokenizer = prepare_tokenizer(self.pyt_ckpt_path, "right")

        if not skip_engine_build:
            # TODO: Build the engine
            engine_dir = Path(os.path.dirname(self.engine_file))
            engine_dir.mkdir(parents=True, exist_ok=True)

            if not os.path.exists("build/TRTLLM/examples/gptj/build.py"):
                raise RuntimeError(f"TRTLLM not found under build/TRTLLM, please run make clone_trt_llm")
            build_cmd = [
                "python", "build/TRTLLM/examples/gptj/build.py", "--dtype=float16",
                "--log_level=verbose", "--enable_context_fmha",
                "--use_gpt_attention_plugin=float16", "--use_gemm_plugin=float16",
                "--use_layernorm_plugin=float16", "--max_batch_size=32",
                f"--max_input_len={G_GPTJ6B_MAX_INPUT_SEQLEN}",
                f"--max_output_len={G_GPTJ6B_MAX_OUTPUT_SEQLEN}", f"--vocab_size={G_GPTJ6B_VOCAB_SIZE}",
                f"--max_beam_width={self.gen_kwargs['num_beams']}",
                f"--output_dir={engine_dir}",
                f"--model_dir={self.pyt_ckpt_path}",
            ]
            logging.info(f"Building engine in {engine_dir}, command: {' '.join(build_cmd)}")
            tik = time.time()
            ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if ret.returncode != 0:
                raise RuntimeError(f"Engine build fails! stderr: {ret.stderr}")
            tok = time.time()

            logging.info(f"Engine built complete and took {tok-tik}s.")

            if engine_build_only:
                logging.info(f"--engine_build_only specified, exiting...")
                exit(0)
        else:
            if not os.path.exists(engine_file):
                raise RuntimeError(
                    f"Cannot find engine file {engine_file}. Please supply the onnx file or engine file.")

    def apply_flag(self, flag):
        """Apply a TRT builder flag."""
        self.builder_config.flags = (
            self.builder_config.flags) | (1 << int(flag))

    def clear_flag(self, flag):
        """Clear a TRT builder flag."""
        self.builder_config.flags = (
            self.builder_config.flags) & ~(1 << int(flag))

    # Helper function to build a TRT engine
    # TODO: This function will call Tekit API to build the engine.
    def create_trt_engine(self):
        raise NotImplementedError(f"Create TRT engine is not implemented yet.")
        return

    def run_inference(self, num_samples):
        """
        Perform the actual inference and calculate ROUGE accuracy
        """
        # Create runner wrapper from the engine file
        self.runner = GPTEngineRunner(
            self.engine_file,
            self.batch_size,
            self.tokenizer,
            self.gen_kwargs,
            self.decoding_step,
            self.verbose
        )

        sources, targets = preprocess_cnndailymail()

        # Start batch inferencing
        batch_idx = 0
        preds = []
        total_time = 0.0
        for start_idx in range(0, num_samples, self.batch_size):
            # Print Progress
            if batch_idx % 20 == 0:
                logging.info(
                    f"Processing batch: {batch_idx} image: {start_idx}/{num_samples}")

            start_time = time.time()
            # Tokenize a batch and record the seqlen info
            end_idx = min(start_idx + self.batch_size, num_samples)
            input_batch = self.tokenizer.batch_encode_plus(sources[start_idx:end_idx], return_tensors="pt",
                                                           padding=True, truncation=True,
                                                           max_length=G_GPTJ6B_MAX_INPUT_SEQLEN)

            input_batch_lengths = [x.shape[0] for x in input_batch.input_ids]

            input_ids = input_batch.input_ids.numpy().astype(np.int32)
            attention_mask = input_batch.attention_mask.numpy().astype(np.int32)
            input_batch_lengths = np.stack(
                input_batch_lengths).astype(np.int32)
            input_real_seqlen = np.sum(attention_mask, axis=1).astype(np.int32)
            logging.debug(
                f"input_batch shape: {input_ids.shape}, mask shape: {attention_mask.shape} input_real_seqlen: {input_real_seqlen}")

            # Input shape:
            # input batch: (BS, max_seq_len),
            # attention_masks: (BS, max_seq_len),
            # input_real_seqlen (BS)
            output_ids = self.runner(
                [input_ids, attention_mask, input_real_seqlen])

            duration = time.time() - start_time
            logging.info(
                f"Batch {batch_idx} >>> inference time: {duration:.2f}s")
            total_time += duration

            # Decode the output
            top_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            logging.debug(f"output_texts:\n{top_texts}")
            preds.extend(output_ids)
            batch_idx += 1

        logging.info(
            f"Total inference time for {num_samples} samples: {total_time:.2f}s")

        # De-tokenize the ids into text
        logging.info(f"Decoding tokenized ids into text...")
        if self.decoding_step:
            print(preds)
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.decoding_step:
            print(preds)
        results = calculate_rouge_score(preds, targets)

        return results


class PytorchTester:
    """
    Pytorch reference tester
    """

    def __init__(
        self,
        pyt_ckpt_path: str,
        batch_size: Optional[int] = 1,
        precision: Optional[str] = "bf16",
        max_input_seq_len: Optional[int] = 1600,
        num_beams: Optional[int] = 1,
        decoding_step: Optional[bool] = False,
    ):
        self.device = torch.device("cuda:0")
        self.batch_size = batch_size
        self.pyt_ckpt_path = pyt_ckpt_path
        self.max_input_seq_len = max_input_seq_len
        self.decoding_step = decoding_step

        # Set torch dtype and mixed precision flag.
        self.amp_enabled = True
        if precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif precision == "fp16":
            self.amp_dtype = torch.float16
        elif precision == "fp32":
            self.amp_enabled = False
            self.amp_dtype = torch.float32
        else:
            raise NotImplementedError(f"Unknown dtype {precision}")

        logging.info(
            f"Loading GPTJ-6B tokenizer and checkpoint from {self.pyt_ckpt_path}")
        self.tokenizer = prepare_tokenizer(self.pyt_ckpt_path)
        self.model_kwargs = {
            "torch_dtype": self.amp_dtype
        }
        self.gen_kwargs = {
            "early_stopping": True,
            "max_new_tokens": G_GPTJ6B_MAX_OUTPUT_SEQLEN,
            "min_new_tokens": 30,
            "num_beams": num_beams,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.pyt_ckpt_path, **self.model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

    def run_inference(self, num_samples):
        """
        Perform the inference steps in pytorch. Note the auto-regressive part is implicit for pytorch
        AutoModelForCausalLM.

        Pytorch maximum batch size is 16 on A100, costing 74GB device memory
        """
        # Read from cnn daily mail and pre-process the data
        with open(G_CNNDAILYMAIL_VALSET_PATH, 'r') as fh:
            list_data_dict = json.load(fh)

        sources, targets = preprocess_cnndailymail()

        # Loop through all the CNN dailymail inputs
        time_stat_dict = {'encoding': 0.0,
                          'inference': 0.0, 'decoding': 0.0, 'total': 0.0}
        preds = []
        batch_idx = 0
        for start_idx in range(0, num_samples, self.batch_size):
            # Print Progress
            if batch_idx % 20 == 0:
                logging.info(
                    f"Processing batch: {batch_idx} image: {start_idx}/{num_samples}")

            end_idx = min(start_idx + self.batch_size, num_samples)

            start_time = time.time()

            # Padding behavior:
            #   1) pad short seq to the maximum within the batch
            #   2) Truncate long seq to 1919
            input_batch = self.tokenizer.batch_encode_plus(
                sources[start_idx:end_idx],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=G_GPTJ6B_MAX_INPUT_SEQLEN
            )

            encoding_time = time.time()

            for t in input_batch:
                if torch.is_tensor(input_batch[t]):
                    input_batch[t] = input_batch[t].to(self.device)

            # Record the input padded seqlen so we know where the output starts from.
            input_batch_lengths = [x.shape[0] for x in input_batch.input_ids]

            with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=self.amp_enabled, dtype=self.amp_dtype if self.amp_enabled else None):
                output_batch = self.model.generate(
                    **input_batch,
                    **self.gen_kwargs,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            inference_time = time.time()

            # Truncate the input portion of the outputs
            output_batch_response_only = []
            for data, source_len in zip(output_batch, input_batch_lengths):
                output_batch_response_only.append(data[source_len:])

            # Decode the output into text, and append to the output list
            # print(f"output_batch_response_only: {output_batch_response_only}")
            output_text = self.tokenizer.batch_decode(
                output_batch_response_only, skip_special_tokens=True)

            decoding_time = time.time()

            # Collect time for parts
            encoding_duration = encoding_time - start_time
            inference_duration = inference_time - encoding_time
            decoding_duration = decoding_time - inference_time
            total_duration = decoding_time - start_time
            time_stat_dict['encoding'] += encoding_duration
            time_stat_dict['inference'] += inference_duration
            time_stat_dict['decoding'] += decoding_duration
            time_stat_dict['total'] += total_duration
            logging.info(f"Batch {batch_idx} >>> encoding: {encoding_duration:.2f}s, infer: {inference_duration:.2f}s, " +
                         f"decoding: {decoding_duration:.2f}s, total: {total_duration:.2f}s")

            # Break down the generate() cycle into inference loops for debugging purpose.
            if self.decoding_step:
                if self.gen_kwargs['num_beams'] > 1:
                    raise NotImplementedError(f"Num_beams > 1 is not supported for per-step decoding. Exiting...")
                # Make a copy of the gen_kargs so they are not contaminated
                copy_model_kwargs = deepcopy(self.model_kwargs)
                input_ids = input_batch.input_ids
                copy_model_kwargs["attention_mask"] = input_batch.attention_mask
                # Track seqs that are still running
                unfinished_sequence = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
                eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)
                pred_text = ["" for i in range(input_ids.shape[0])]
                for i in range(self.gen_kwargs["max_new_tokens"]):
                    model_inputs = self.model.prepare_inputs_for_generation(
                        input_ids=input_ids,
                        **copy_model_kwargs
                    )
                    # outputs is a CausalLMOutputWithPast type, with following fields:
                    # loss, logits, past_key_values, hidden_states, attentions.
                    # logits and past_key_values are useful in inference.
                    outputs = self.model(**model_inputs)
                    next_token_logits = outputs.logits[:, -1, :]

                    # If the min token length is not met, force generate new tokens by
                    # changing the logits of EOS to -inf.
                    # See https://github.com/huggingface/transformers/blob/f49a3453caa6fe606bb31c571423f72264152fce/src/transformers/generation/logits_process.py#L161
                    if i < self.gen_kwargs["min_new_tokens"]:
                        next_token_logits[:, self.tokenizer.eos_token_id] = -float('inf')

                    # Debug top 5 scores
                    top_5_scores, top_5_ids = torch.topk(input=next_token_logits, k=5, dim=-1)
                    logging.debug(f"step {i}: top_id {top_5_ids} top_score: {top_5_scores}")

                    # use greedy for debugging
                    # Pad tokens for sequences that are already finished
                    top_scores, top_ids = torch.topk(input=next_token_logits, k=1, dim=-1)
                    next_tokens = top_ids.squeeze(-1) * unfinished_sequence + self.tokenizer.pad_token_id * (1 - unfinished_sequence)
                    unfinished_sequence = unfinished_sequence.mul(
                        next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )

                    input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                    copy_model_kwargs = self.model._update_model_kwargs_for_generation(
                        outputs, copy_model_kwargs,
                    )
                    top_texts = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=True)
                    # logging.debug(f"step {i}: top_ids: {top_ids}, top_scores: {top_scores}, unfinished: {unfinished_sequence}, top_texts: {top_texts}")

                    for idx, seq in enumerate(pred_text):
                        if unfinished_sequence[idx]:
                            pred_text[idx] += top_texts[idx]
                    # Break if all sequences reach the end
                    if unfinished_sequence.max() == 0:
                        break

                logging.info(f"Per-step generation is equal to gen(): {pred_text == output_text}")

            logging.debug(f"output_text of batch {batch_idx}: {output_text}")
            preds.extend(output_text)
            batch_idx += 1

        logging.info(f"time_stat_dict: {time_stat_dict}")

        results = calculate_rouge_score(preds, targets)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine_file",
                        help="Specify where the GPTJ6B engine file is",
                        default="build/TRTLLM/examples/gptj/gptj-engine/gptj_float16_tp1_rank0.engine",
                        required=False)
    parser.add_argument("--pyt_ckpt_path",
                        help="Specify where the PyTorch checkpoint path is",
                        default="build/models/GPTJ-6B/checkpoint-final")
    parser.add_argument("--batch_size",
                        help="batch size. 80GB can run a maximum of BS=8 for FP32 greedy",
                        type=int,
                        default=1)
    parser.add_argument("--max_input_seq_len",
                        help="Maximum number of input sequence length",
                        type=int,
                        default=1919)
    parser.add_argument("--num_beams",
                        help="The maximum beam width of the decoding op.",
                        type=int,
                        default=1)
    parser.add_argument("--num_samples",
                        help="Number of samples to run. We have 13368 in total for cnn-dailymail validation set",
                        type=int,
                        default=13368)
    parser.add_argument("--torch_precision",
                        help="Run Pytorch in the specified precision",
                        choices=("fp32", "fp16", "bf16"),
                        default="bf16")
    parser.add_argument("--trt_precision",
                        help="Run TensorRT in the specified precision",
                        choices=("fp32", "fp16", "int8", "fp8"),
                        default="fp32")
    parser.add_argument("--use_dla",
                        help="Use DLA instead of gpu",
                        action="store_true")
    parser.add_argument("--skip_engine_build",
                        help="Skip the TRT engine build phase if possible.",
                        action="store_true")
    parser.add_argument("--engine_build_only",
                        help="Build the engine and skip the testing part",
                        action="store_true")
    parser.add_argument("--pytorch",
                        help="whether to run pytorch inference",
                        action="store_true")
    parser.add_argument("--decoding_step",
                        help="Enable to step into the generation cycle of the model.",
                        action="store_true")
    parser.add_argument("--verbose",
                        help="verbose output",
                        action="store_true")
    args = parser.parse_args()

    # Pytorch Tester
    if args.pytorch:
        logging.info(
            f"Running Accuracy test for Pytorch reference implementation.")
        if not os.path.exists(args.pyt_ckpt_path):
            raise RuntimeError(
                f"Cannot access {args.pyt_ckpt_path}. Please download the model or mount the scratch path.")
        pt_tester = PytorchTester(
            args.pyt_ckpt_path,
            args.batch_size,
            args.torch_precision,
            args.max_input_seq_len,
            args.num_beams,
            args.decoding_step,
        )
        rouge = pt_tester.run_inference(args.num_samples)
        logging.info(f"Pytorch ROUGE Score: {rouge}")
    else:
        # TRT Tester
        logging.info(
            f"Running accuracy test for GPTJ6B using {args.engine_file} ...")
        tester = TRTTester(
            args.engine_file,
            args.batch_size,
            args.trt_precision,
            args.pyt_ckpt_path,
            args.num_beams,
            args.use_dla,
            args.skip_engine_build,
            args.engine_build_only,
            args.decoding_step,
            args.verbose)
        rouge = tester.run_inference(args.num_samples)
        logging.info(f"TRT ROUGE Score: {rouge}")

    # TRT
    # To run the TRT tester:
    # python3 -m code.gptj6b.tensorrt.infer --engine_file /home/scratch.zhihanj_sw/gitlab_root/tekit/examples/gptj/gptj_engine_from_hf_h100/gptj_float16_tp1_rank0.engine --num_samples=2 --batch_size=2 --skip_engine_build
    # Torch
    # To run the pytorch tester:
    # python3 -m code.gptj6b.tensorrt.infer --pytorch --batch_size=2 --num_samples=8
    # To check per-step torch output
    # VERBOSE=1 python3 -m code.gptj6b.tensorrt.infer --pytorch --batch_size=2 --num_samples=2 --decoding_step


if __name__ == "__main__":
    main()
