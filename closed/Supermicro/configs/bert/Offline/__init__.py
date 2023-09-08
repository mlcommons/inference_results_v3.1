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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline

    gpu_copy_streams = 2
    gpu_inference_streams = 2
    enable_interleaved = False


class OfflineCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Offline

    max_queue_delay_usec = 100


class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 10000
    workspace_size = 7516192768


class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024
    offline_expected_qps = 8600


class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 5700
    workspace_size = 7516192768


class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 5000
    use_graphs = False
    gpu_batch_size = 1024


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


class H100_PCIe_80GBx1_HighAccuracy_Triton(H100_PCIe_80GBx1_HighAccuracy):
    offline_expected_qps = 1800
    use_triton = True


class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    offline_expected_qps = 46000


class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx1_HighAccuracy):
    offline_expected_qps = 5000 * 8


class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx8):
    use_triton = True


class H100_PCIe_80GBx8_HighAccuracy_Triton(H100_PCIe_80GBx8_HighAccuracy):
    use_triton = True


class H100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 4000
    workspace_size = 7516192768


class H100_PCIe_80GB_aarch64x1_HighAccuracy(H100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1800
    use_graphs = False
    gpu_batch_size = 1024


class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    offline_expected_qps = 16000


class H100_PCIe_80GB_aarch64x4_HighAccuracy(H100_PCIe_80GB_aarch64x1_HighAccuracy):
    offline_expected_qps = 8000


class H100_SXM_80GB_02x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 8900
    workspace_size = 7516192768


class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 9400
    workspace_size = 7516192768


class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024
    offline_expected_qps = 8200


class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


class H100_SXM_80GBx1_HighAccuracy_Triton(H100_SXM_80GBx1_HighAccuracy):
    use_triton = True


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    offline_expected_qps = 9400 * 8


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    offline_expected_qps = 54000
    power_limit = 400


class H100_SXM_80GBx8_HighAccuracy(H100_SXM_80GBx1_HighAccuracy):
    offline_expected_qps = 8200 * 8


class H100_SXM_80GBx8_HighAccuracy_MaxQ(H100_SXM_80GBx8_HighAccuracy):
    power_limit = 450
    offline_expected_qps = 51000


class H100_SXM_80GBx8_Triton(H100_SXM_80GBx8):
    use_triton = True


class H100_SXM_80GBx8_HighAccuracy_Triton(H100_SXM_80GBx8_HighAccuracy):
    use_triton = True


class L4x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 16
    energy_aware_kernels = True
    offline_expected_qps = 1000
    workspace_size = 7516192768


class L4x1_HighAccuracy(L4x1):
    precision = "fp16"
    use_fp8 = True
    gpu_batch_size = 16
    offline_expected_qps = 640
    gpu_inference_streams = 1
    energy_aware_kernels = False
    gpu_copy_streams = 1


class L40x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


class L40x1_HighAccuracy(L40x1):
    precision = "fp16"
    offline_expected_qps = 1750


class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True
    offline_expected_qps = 3000


class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True
    offline_expected_qps = 3000


class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1550


class A100_PCIe_80GBx1_HighAccuracy_TritonUnified(A100_PCIe_80GBx1_HighAccuracy):
    use_triton = True
    offline_expected_qps = 1550


class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    offline_expected_qps = 27200


class A100_PCIe_80GBx8_HighAccuracy(A100_PCIe_80GBx8):
    precision = "fp16"
    offline_expected_qps = 12800


class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    use_triton = True
    offline_expected_qps = 27000


class A100_PCIe_80GBx8_TritonUnified(A100_PCIe_80GBx8):
    use_triton = True
    offline_expected_qps = 27000


class A100_PCIe_80GBx8_HighAccuracy_Triton(A100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 12800


class A100_PCIe_80GBx8_HighAccuracy_TritonUnified(A100_PCIe_80GBx8_HighAccuracy):
    use_triton = True
    offline_expected_qps = 12800


class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    offline_expected_qps = 27200
    power_limit = 240


class A100_PCIe_80GBx8_HighAccuracy_MaxQ(A100_PCIe_80GBx8_MaxQ):
    precision = "fp16"
    offline_expected_qps = 11000


class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


class A100_PCIe_80GBx8_TritonUnified_MaxQ(A100_PCIe_80GBx8_MaxQ):
    use_triton = True
    offline_expected_qps = 27200


class A100_PCIe_80GBx8_HighAccuracy_Triton_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


class A100_PCIe_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_PCIe_80GBx8_HighAccuracy_MaxQ):
    use_triton = True
    offline_expected_qps = 11168


class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


class A100_PCIe_80GB_aarch64x1_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x2(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 6500
    workspace_size = 7516192768


class A100_PCIe_80GB_aarch64x2_Triton(A100_PCIe_80GB_aarch64x2):
    use_triton = True


class A100_PCIe_80GB_aarch64x2_TritonUnified(A100_PCIe_80GB_aarch64x2):
    use_triton = True


class A100_PCIe_80GB_aarch64x2_HighAccuracy(A100_PCIe_80GB_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


class A100_PCIe_80GB_aarch64x2_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x2_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x4(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 13600
    workspace_size = 7516192768


class A100_PCIe_80GB_aarch64x4_Triton(A100_PCIe_80GB_aarch64x4):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_TritonUnified(A100_PCIe_80GB_aarch64x4):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_HighAccuracy(A100_PCIe_80GB_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


class A100_PCIe_80GB_aarch64x4_HighAccuracy_Triton(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_80GB_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 10000
    power_limit = 225


class A100_PCIe_80GB_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_80GB_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 5000


class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400
    workspace_size = 7516192768


class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    precision = "fp16"
    offline_expected_qps = 1950


class A100_PCIe_aarch64x1_HighAccuracy_Triton(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x1_HighAccuracy_TritonUnified(A100_PCIe_aarch64x1_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x2(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 6500
    workspace_size = 7516192768


class A100_PCIe_aarch64x2_Triton(A100_PCIe_aarch64x2):
    use_triton = True


class A100_PCIe_aarch64x2_TritonUnified(A100_PCIe_aarch64x2):
    use_triton = True


class A100_PCIe_aarch64x2_HighAccuracy(A100_PCIe_aarch64x2):
    precision = "fp16"
    offline_expected_qps = 3900


class A100_PCIe_aarch64x2_HighAccuracy_Triton(A100_PCIe_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x2_HighAccuracy_TritonUnified(A100_PCIe_aarch64x2_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x4(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 13600
    workspace_size = 7516192768


class A100_PCIe_aarch64x4_Triton(A100_PCIe_aarch64x4):
    use_triton = True


class A100_PCIe_aarch64x4_TritonUnified(A100_PCIe_aarch64x4):
    use_triton = True


class A100_PCIe_aarch64x4_HighAccuracy(A100_PCIe_aarch64x4):
    precision = "fp16"
    offline_expected_qps = 8200


class A100_PCIe_aarch64x4_HighAccuracy_Triton(A100_PCIe_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x4_HighAccuracy_TritonUnified(A100_PCIe_aarch64x4_HighAccuracy):
    use_triton = True


class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 9000
    power_limit = 225


class A100_PCIe_aarch64x4_HighAccuracy_MaxQ(A100_PCIe_aarch64x4_MaxQ):
    precision = "fp16"
    offline_expected_qps = 4500


class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_PCIe_MIG_1x1g5gb_HighAccuracy(A100_PCIe_MIG_1x1g5gb):
    precision = "fp16"
    offline_expected_qps = 225


class A100_PCIe_MIG_1x1g5gb_Triton(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


class A100_PCIe_MIG_1x1g5gb_TritonUnified(A100_PCIe_MIG_1x1g5gb):
    use_triton = True


class A100_PCIe_MIG_1x1g5gb_HighAccuracy_Triton(A100_PCIe_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_PCIe_MIG_1x1g5gb_HighAccuracy_TritonUnified(A100_PCIe_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_PCIe_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    offline_expected_qps = 470


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 225


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    offline_expected_qps = 210


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 225


class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 3500


class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx1):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750


class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_HighAccuracy_Triton(A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


class A100_SXM_80GBx1_HighAccuracy_TritonUnified(A100_SXM_80GBx1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


class A100_SXM_80GBx8(A100_SXM_80GBx1):
    offline_expected_qps = 30000
    workspace_size = 7516192768


class A100_SXM_80GBx8_HighAccuracy(A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15000


class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GBx8_TritonUnified(A100_SXM_80GBx8):
    use_triton = True
    offline_expected_qps = 29000
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GBx8_HighAccuracy_Triton(A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000


class A100_SXM_80GBx8_HighAccuracy_TritonUnified(A100_SXM_80GBx8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 15000


class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    power_limit = 275


class A100_SXM_80GBx8_HighAccuracy_MaxQ(A100_SXM_80GBx8_MaxQ):
    power_limit = 275
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 11000


class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_MaxQ):
    use_triton = True


class A100_SXM_80GBx8_TritonUnified_MaxQ(A100_SXM_80GBx8_MaxQ):
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_Triton_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    use_triton = True


class A100_SXM_80GBx8_HighAccuracy_TritonUnified_MaxQ(A100_SXM_80GBx8_HighAccuracy_MaxQ):
    use_triton = True


class A100_SXM_80GB_aarch64x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 2500


class A100_SXM_80GB_aarch64x1_HighAccuracy(A100_SXM_80GB_aarch64x1):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 1750


class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True
    offline_expected_qps = 2200


class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    offline_expected_qps = 2200


class A100_SXM_80GB_aarch64x1_HighAccuracy_Triton(A100_SXM_80GB_aarch64x1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


class A100_SXM_80GB_aarch64x1_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x1_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    offline_expected_qps = 1750


class A100_SXM_80GB_aarch64x8(A100_SXM_80GB_aarch64x1):
    offline_expected_qps = 27500
    workspace_size = 7516192768


class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    offline_expected_qps = 22000
    power_limit = 250           # Set to 250 initially, increase to 300 w/ optimal fan setting


class A100_SXM_80GB_aarch64x8_HighAccuracy(A100_SXM_80GB_aarch64x8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 14000


class A100_SXM_80GB_aarch64x8_HighAccuracy_MaxQ(A100_SXM_80GB_aarch64x8_HighAccuracy):
    offline_expected_qps = 12000
    power_limit = 250           # Set to 250 initially, increase to 300 w/ optimal fan setting


class A100_SXM_80GB_aarch64x8_Triton(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 27500
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GB_aarch64x8_TritonUnified(A100_SXM_80GB_aarch64x8):
    use_triton = True
    offline_expected_qps = 27500
    workspace_size = 7516192768
    batch_triton_requests = False


class A100_SXM_80GB_aarch64x8_HighAccuracy_Triton(A100_SXM_80GB_aarch64x8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 14000


class A100_SXM_80GB_aarch64x8_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64x8_HighAccuracy):
    use_triton = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1
    offline_expected_qps = 14000


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 350
    workspace_size = 2147483648


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    precision = "fp16"
    offline_expected_qps = 250


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 250


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 250


class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64
    offline_expected_qps = 500
    workspace_size = 2147483648


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(A100_SXM4_40GB_MIG_1x1g5gb):
    precision = "fp16"
    gpu_batch_size = 64
    offline_expected_qps = 225


class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 32
    offline_expected_qps = 225


class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 3400


class A100_SXM4_40GBx1_HighAccuracy(A100_SXM4_40GBx1):
    precision = "fp16"
    offline_expected_qps = 1750


class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_HighAccuracy_Triton(A100_SXM4_40GBx1_HighAccuracy):
    use_triton = True


class A100_SXM4_40GBx1_HighAccuracy_TritonUnified(A100_SXM4_40GBx1_HighAccuracy):
    use_triton = True


class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    offline_expected_qps = 30000
    workspace_size = 7516192768


class A100_SXM4_40GBx8_HighAccuracy(A100_SXM4_40GBx8):
    precision = "fp16"
    gpu_batch_size = 1024
    offline_expected_qps = 15000


class A100_SXM4_40GBx8_Triton(A100_SXM4_40GBx8):
    use_triton = True


class A100_SXM4_40GBx8_TritonUnified(A100_SXM4_40GBx8):
    use_triton = True


class A100_SXM4_40GBx8_HighAccuracy_Triton(A100_SXM4_40GBx8_HighAccuracy):
    use_triton = True


class A100_SXM4_40GBx8_HighAccuracy_TritonUnified(A100_SXM4_40GBx8_HighAccuracy):
    use_triton = True


class A2x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 250


class A2x1_HighAccuracy(A2x1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 120


class A2x1_Triton(A2x1):
    use_triton = True


class A2x1_TritonUnified(A2x1):
    use_triton = True


class A2x1_HighAccuracy_Triton(A2x1_HighAccuracy):
    use_triton = True


class A2x1_HighAccuracy_TritonUnified(A2x1_HighAccuracy):
    use_triton = True


class A2x2(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    offline_expected_qps = 500


class A2x2_HighAccuracy(A2x2):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 240


class A2x2_Triton(A2x2):
    use_triton = True


class A2x2_TritonUnified(A2x2):
    use_triton = True


class A2x2_HighAccuracy_Triton(A2x2_HighAccuracy):
    use_triton = True


class A2x2_HighAccuracy_TritonUnified(A2x2_HighAccuracy):
    use_triton = True


class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 96
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 505
    workspace_size = 805306368


class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    precision = "fp16"
    offline_expected_qps = 246.3


class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 457.658


class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    precision = "fp16"
    offline_expected_qps = 219.18


class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True


class A30_MIG_1x1g6gb_TritonUnified(A30_MIG_1x1g6gb):
    use_triton = True


class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 64
    offline_expected_qps = 240


class A30_MIG_1x1g6gb_HighAccuracy_TritonUnified(A30_MIG_1x1g6gb_HighAccuracy):
    use_triton = True
    gpu_batch_size = 64
    offline_expected_qps = 240


class A30x1(OfflineGPUBaseConfig):
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    offline_expected_qps = 1971.9999999999998
    workspace_size = 7516192768


class A30x1_HighAccuracy(A30x1):
    precision = "fp16"
    offline_expected_qps = 1014.9999999999999


class A30x1_Triton(A30x1):
    use_triton = True
    offline_expected_qps = 1739.9999999999998


class A30x1_TritonUnified(A30x1):
    use_triton = True
    offline_expected_qps = 1739.9999999999998


class A30x1_HighAccuracy_Triton(A30x1_HighAccuracy):
    use_triton = True


class A30x1_HighAccuracy_TritonUnified(A30x1_HighAccuracy):
    use_triton = True


class A30x8(A30x1):
    offline_expected_qps = 13000


class A30x8_HighAccuracy(A30x8):
    precision = "fp16"
    offline_expected_qps = 8119.999999999999


class A30x8_Triton(A30x8):
    use_triton = True


class A30x8_TritonUnified(A30x8):
    use_triton = True


class A30x8_HighAccuracy_Triton(A30x8_HighAccuracy):
    use_triton = True


class A30x8_HighAccuracy_TritonUnified(A30x8_HighAccuracy):
    use_triton = True


class Orin(OfflineGPUBaseConfig):
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 550


class Orin_Triton(Orin):
    use_triton = True
    batch_triton_requests = True


class Orin_TritonUnified(Orin):
    use_triton = True
    batch_triton_requests = True


class Orin_MaxQ(Orin):
    # NOTE: Orin AGX 3.1 Shmoo
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 384
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    soc_cpu_freq = 576000
    soc_gpu_freq = 714000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    offline_expected_qps = 300


class Orin_NX(OfflineGPUBaseConfig):
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 256
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    offline_expected_qps = 190


class Orin_NX_MaxQ(Orin_NX):
    # NOTE: Orin NX 3.1 Shmoo
    enable_interleaved = False
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 384
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    soc_cpu_freq = 499200
    soc_gpu_freq = 714000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 0
    orin_num_cores = 4
    offline_expected_qps = 140


class T4x1(OfflineGPUBaseConfig):
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 430


class T4x1_HighAccuracy(T4x1):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 210


class T4x1_Triton(T4x1):
    use_triton = True


class T4x1_TritonUnified(T4x1):
    use_triton = True


class T4x1_HighAccuracy_Triton(T4x1_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 189


class T4x1_HighAccuracy_TritonUnified(T4x1_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 189


class T4x20(T4x1):
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 8800


class T4x20_HighAccuracy(T4x20):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 4400


class T4x20_Triton(T4x20):
    use_triton = True
    offline_expected_qps = 7920


class T4x20_TritonUnified(T4x20):
    use_triton = True
    offline_expected_qps = 7920


class T4x20_HighAccuracy_Triton(T4x20_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 3960


class T4x20_HighAccuracy_TritonUnified(T4x20_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 3960


class T4x8(T4x1):
    enable_interleaved = True
    use_small_tile_gemm_plugin = False
    gpu_batch_size = 256
    offline_expected_qps = 3500


class T4x8_HighAccuracy(T4x8):
    precision = "fp16"
    gpu_inference_streams = 1
    offline_expected_qps = 1680


class T4x8_Triton(T4x8):
    use_triton = True
    offline_expected_qps = 3150


class T4x8_TritonUnified(T4x8):
    use_triton = True
    offline_expected_qps = 3150


class T4x8_HighAccuracy_Triton(T4x8_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 1512


class T4x8_HighAccuracy_TritonUnified(T4x8_HighAccuracy):
    use_triton = True
    gpu_inference_streams = 2
    offline_expected_qps = 1512


class Triton_CPU_2S_8380x1(OfflineCPUBaseConfig):
    batch_size = 0
    offline_expected_qps = 70
    num_instances = 16
    ov_parameters = {
        'CPU_THREADS_NUM': '80',
        'CPU_THROUGHPUT_STREAMS': '16',
        'ENABLE_BATCH_PADDING': 'NO',
        'SKIP_OV_DYNAMIC_BATCHSIZE': 'YES'
    }


class Triton_Inferentia_INF1_2XLARGEx1(BenchmarkConfiguration):
    offline_expected_qps = 70
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 4
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 1
    instance_group_count = 4


class Triton_Inferentia_HighAccuracy_INF1_2XLARGEx1(BenchmarkConfiguration):
    offline_expected_qps = 70
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 4
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 1
    instance_group_count = 4


class Triton_Inferentia_INF1_6XLARGEx1(BenchmarkConfiguration):
    offline_expected_qps = 275
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 4
    instance_group_count = 16


class Triton_Inferentia_HighAccuracy_INF1_6XLARGEx1(BenchmarkConfiguration):
    offline_expected_qps = 275
    benchmark = Benchmark.BERT
    tensor_path = "/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_ids.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/input_mask.npy,/home/ubuntu/mlperf_scratch/preprocessed_data/squad_tokenized/segment_ids.npy"
    precision = "fp32"
    input_dtype = "int32"
    bert_opt_seqlen = 384
    coalesced_tensor = True
    use_triton = True
    scenario = Scenario.Offline
    inferentia_neuron_core_count = 16
    inferentia_threads_per_core = 1
    inferentia_compiled_model_framework = "pytorch"
    inferentia_compiled_model_batch_size = 1
    batch_triton_requests = False
    inferentia_request_batch_size = 4
    instance_group_count = 16


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    offline_expected_qps = 39500
    power_limit = 290


class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_HighAccuracy):
    offline_expected_qps = 33000
    power_limit = 300
