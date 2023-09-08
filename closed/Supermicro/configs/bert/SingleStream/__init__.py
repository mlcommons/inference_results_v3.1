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
from configs.bert import GPUBaseConfig, CPUBaseConfig


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream

    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = True
    bert_opt_seqlen = 270
    use_small_tile_gemm_plugin = False


class GH200_96GB_aarch64x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    precision = "fp16"
    use_fp8 = True


class L4x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class L4x1_HighAccuracy(L4x1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class L40x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class L40x1_HighAccuracy(L40x1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class H100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class H100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class H100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class A100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True


class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GBx1_HighAccuracy_TritonUnified(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_PCIe_80GB_aarch64x1_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_PCIe_aarch64x1_HighAccuracy_Triton(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 5500000
    workspace_size = 2147483648


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb):
    precision = "fp16"
    single_stream_expected_latency_ns = 11000000


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    single_stream_expected_latency_ns = 5800000


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    single_stream_expected_latency_ns = 12000000


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_PCIe_80GB_MIG_1x1g10gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 12000000


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 12000000


class A100_SXM_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 5342000
    workspace_size = 2147483648


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    precision = "fp16"
    single_stream_expected_latency_ns = 11000000


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx1):
    precision = "fp16"


class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_HighAccuracy_Triton(A100_SXM_80GBx1_Triton):
    precision = "fp16"


class A100_SXM_80GBx1_HighAccuracy_TritonUnified(A100_SXM_80GBx1_Triton):
    precision = "fp16"


class A100_SXM_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1552000


class A100_SXM_80GB_aarch64x1_HighAccuracy(A100_SXM_80GB_aarch64x1):
    precision = "fp16"
    single_stream_expected_latency_ns = 2442000


class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True
    single_stream_expected_latency_ns = 1636500


class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    single_stream_expected_latency_ns = 1636500


class A100_SXM_80GB_aarch64x1_HighAccuracy_Triton(A100_SXM_80GB_aarch64x1_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 2516000


class A100_SXM_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x1_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 2516000


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 1700000
    workspace_size = 2147483648


class A100_SXM_80GB_MIG_aarch64_1x1g10gb_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_aarch64_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A100_SXM4_40GB_MIG_1x1g5gb(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 1700000
    workspace_size = 2147483648


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(A100_SXM4_40GB_MIG_1x1g5gb):
    precision = "fp16"


class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(A100_SXM4_40GB_MIG_1x1g5gb_Triton):
    precision = "fp16"


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb_Triton):
    precision = "fp16"


class A100_SXM4_40GBx1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class A100_SXM4_40GBx1_HighAccuracy(A100_SXM4_40GBx1):
    precision = "fp16"


class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_HighAccuracy_Triton(A100_SXM4_40GBx1_Triton):
    precision = "fp16"


class A100_SXM4_40GBx1_HighAccuracy_TritonUnified(A100_SXM4_40GBx1_Triton):
    precision = "fp16"


class A2x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 9000000


class A2x1_HighAccuracy(A2x1):
    precision = "fp16"


class A2x1_Triton(A2x1):
    use_triton = True


class A2x1_TritonUnified(A2x1):
    use_triton = True


class A2x1_HighAccuracy_Triton(A2x1_HighAccuracy):
    use_triton = True


class A2x1_HighAccuracy_TritonUnified(A2x1_HighAccuracy):
    use_triton = True


class A30_MIG_1x1g6gb(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 5999404
    workspace_size = 1610612736


class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    precision = "fp16"
    single_stream_expected_latency_ns = 11000950


class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 6055419


class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_HighAccuracy):
    single_stream_expected_latency_ns = 11558755


class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 5999404
    use_triton = True


class A30_MIG_1x1g6gb_TritonUnified(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 5999404
    use_triton = True


class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 7452826


class A30_MIG_1x1g6gb_HighAccuracy_TritonUnified(A30_MIG_1x1g6gb_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 7452826


class A30x1(SingleStreamGPUBaseConfig):
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000


class A30x1_HighAccuracy(A30x1):
    precision = "fp16"


class A30x1_Triton(A30x1):
    single_stream_expected_latency_ns = 3400000
    use_triton = True


class A30x1_TritonUnified(A30x1):
    single_stream_expected_latency_ns = 3400000
    use_triton = True


class A30x1_HighAccuracy_Triton(A30x1_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class A30x1_HighAccuracy_TritonUnified(A30x1_Triton):
    precision = "fp16"
    single_stream_expected_latency_ns = 1700000


class Orin(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 6200000
    use_graphs = True


class Orin_Triton(Orin):
    use_triton = True


class Orin_TritonUnified(Orin):
    use_triton = True


class Orin_MaxQ(Orin):
    soc_cpu_freq = 576000
    soc_gpu_freq = 816000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    single_stream_expected_latency_ns = 11914844


class Orin_NX(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 15000000
    use_graphs = True


class Orin_NX_MaxQ(Orin_NX):
    soc_cpu_freq = 576000
    soc_gpu_freq = 714000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 0
    orin_num_cores = 4
    single_stream_expected_latency_ns = 24000000


class T4x1(SingleStreamGPUBaseConfig):
    single_stream_expected_latency_ns = 6400000


class T4x1_HighAccuracy(T4x1):
    precision = "fp16"


class T4x1_Triton(T4x1):
    use_triton = True


class T4x1_TritonUnified(T4x1):
    use_triton = True


class T4x1_HighAccuracy_Triton(T4x1_Triton):
    precision = "fp16"


class T4x1_HighAccuracy_TritonUnified(T4x1_Triton):
    precision = "fp16"


class Triton_CPU_2S_8380x1_Triton(CPUBaseConfig):
    scenario = Scenario.SingleStream
    batch_size = 0
    single_stream_expected_latency_ns = 60000000
    num_instances = 1
    ov_parameters = {'CPU_THREADS_NUM': '80', 'CPU_THROUGHPUT_STREAMS': '1', 'ENABLE_BATCH_PADDING': 'NO', 'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'}
