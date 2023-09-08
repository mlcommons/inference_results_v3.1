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
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *
from configs.rnnt import GPUBaseConfig


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    audio_batch_size = 1
    audio_fp16_input = True
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    use_graphs = True
    disable_encoder_plugin = True
    nobatch_sorting = True
    num_warmups = 32


class GH200_96GB_aarch64x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class L4x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class L40x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class H100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class H100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class H100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class A100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class A100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class A100_PCIe_aarch64x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class A100_PCIe_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 78000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    single_stream_expected_latency_ns = 80000000


class A100_SXM_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    start_from_device = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 38400000
    nouse_copy_kernel = True


class A100_SXM4_40GB_MIG_1x1g5gb(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    workspace_size = 1073741824


class A100_SXM4_40GBx1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = True
    start_from_device = True


class A2x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 105000000
    nouse_copy_kernel = True


class A30_MIG_1x1g6gb(SingleStreamGPUBaseConfig):
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    single_stream_expected_latency_ns = 76133687
    nouse_copy_kernel = False
    workspace_size = 1610612736


class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 78812921


class A30x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 1
    single_stream_expected_latency_ns = 10000000
    nouse_copy_kernel = False


class T4x1(SingleStreamGPUBaseConfig):
    audio_buffer_num_lines = 4
    single_stream_expected_latency_ns = 25000000
    nouse_copy_kernel = True


class Orin(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 95000000
    audio_fp16_input = None
    dali_batches_issue_ahead = None
    dali_pipeline_depth = None
    nobatch_sorting = None
    num_warmups = None
    disable_encoder_plugin = True
    # https://nvbugs/3863492
    # TRT from 8.5.0.5 has perf regression with pipelined execution, could be myelin backend issue
    # Disable pipelined execution for better perf
    nopipelined_execution = True


class Orin_MaxQ(Orin):
    soc_cpu_freq = 652800
    soc_gpu_freq = 612000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    single_stream_expected_latency_ns = 100000000


class Orin_NX(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 210000000
    audio_fp16_input = None
    dali_batches_issue_ahead = None
    dali_pipeline_depth = None
    nobatch_sorting = None
    num_warmups = None
    disable_encoder_plugin = True
    # https://nvbugs/3863492
    # TRT from 8.5.0.5 has perf regression with pipelined execution, could be myelin backend issue
    # Disable pipelined execution for better perf
    nopipelined_execution = True


class Orin_NX_MaxQ(Orin_NX):
    soc_cpu_freq = 806400
    soc_gpu_freq = 714000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 0
    orin_num_cores = 4
    single_stream_expected_latency_ns = 210000000
