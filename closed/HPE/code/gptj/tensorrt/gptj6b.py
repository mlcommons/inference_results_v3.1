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

import os
import time
import subprocess

import numpy as np
import pycuda.driver as cuda
import pycuda.autoprimaryctx
import tensorrt as trt

from code.common import logging, dict_get
from code.common.builder import TensorRTEngineBuilder
from code.common.constants import Benchmark
from code.common.systems.system_list import SystemClassifications, DETECTED_SYSTEM
from importlib import import_module


class GPTJ6BBuilder(TensorRTEngineBuilder):
    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=(60 << 30))
        super().__init__(args, Benchmark.GPTJ, workspace_size=workspace_size)

        self.use_fp8 = dict_get(self.args, "use_fp8", default=False)
        self.gpu_batch_size = dict_get(self.args, "gpu_batch_size", default=16)

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/GPTJ-6B/checkpoint-final")
        ngc_path = "/opt/GPTJ-07142023.pth"
        if os.path.exists(ngc_path):
            self.fp8_quant_model_path = ngc_path
        else:
            self.fp8_quant_model_path = "build/models/GPTJ-6B/fp8-quantized-ammo/GPTJ-07142023.pth"

        self.trt_llm_path = "build/TRTLLM/"

        if self.precision == "fp16":
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
        elif self.precision == "fp8":
            raise NotImplementedError(f"Don't use precision flag for FP8, please enable use_fp8 in the config.")
        else:
            raise NotImplementedError(f"Precision {self.precision} is not supported yet.")

    def initialize(self):
        self.initialized = True

    def calibrate(self):
        """
        Calibration will be done through AMMO toolkit
        """
        raise NotImplementedError(f"Calibration for GPTJ6B is not implemented yet")

    def build_engines(self):
        """
        Override the build_engines function to call TRT-LLM.
        """
        try:
            import tensorrt_llm
            build_script_path = os.path.join(self.trt_llm_path, "examples/gptj/build.py")
            assert os.path.exists(build_script_path)
        except:
            logging.error(f"Cannot import tensorrt_llm module, please run make build_trt_llm")
            raise

        # Create engine dir
        real_engine_path = self._get_engine_fpath(None, None)
        os.makedirs(self.engine_dir, exist_ok=True)

        build_cmd = [
            "python", build_script_path, "--dtype=float16",
            "--use_gpt_attention_plugin=float16",
            "--use_gemm_plugin=float16",
            f"--max_batch_size={self.gpu_batch_size}",
            f"--max_input_len=1919",
            "--max_output_len=128", f"--vocab_size=50401",
            f"--max_beam_width=4",
            f"--output_dir={self.engine_dir}",
            f"--model_dir={self.model_path}",
        ]
        build_cmd += ["--enable_context_fmha"]
        build_cmd += ["--enable_two_optimization_profiles"]

        if self.use_fp8:
            assert os.path.exists(self.fp8_quant_model_path), f"Cannot find FP8 quantized model at {self.fp8_quant_model_path}"
            build_cmd += [
                "--enable_fp8", f"--quantized_fp8_model_path={self.fp8_quant_model_path}"
            ]
            build_cmd += ["--fp8_kv_cache"]
        logging.info(f"Building GPTJ engine in {self.engine_dir}, use_fp8: {self.use_fp8} command: {' '.join(build_cmd)}")
        tik = time.time()
        ret = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Store logs to engine dir
        stdout_fn = real_engine_path.replace("plan", "stdout")
        stderr_fn = real_engine_path.replace("plan", "stderr")
        with open(stdout_fn, 'w') as fh:
            fh.write(ret.stdout)
        with open(stderr_fn, 'w') as fh:
            fh.write(ret.stderr)

        if ret.returncode != 0:
            raise RuntimeError(f"Engine build fails! stderr: {ret.stderr}. See engine log: {stdout_fn} and {stderr_fn}")
        tok = time.time()

        # Change the engine name according the engine fpath
        cur_engine_path = os.path.join(self.engine_dir, "gptj_float16_tp1_rank0.engine")
        os.rename(cur_engine_path, real_engine_path)

        logging.info(f"Engine built complete and took {tok-tik}s. Stored at {real_engine_path}")
