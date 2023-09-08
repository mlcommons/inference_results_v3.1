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

from importlib import import_module
from code.common.constants import Benchmark, Scenario
from code.common.systems.system_list import KnownSystem
from configs.configuration import *

ParentConfig = import_module("configs.3d-unet")
GPUBaseConfig = ParentConfig.GPUBaseConfig
CPUBaseConfig = ParentConfig.CPUBaseConfig


class SingleStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.SingleStream
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    gpu_batch_size = 4
    slice_overlap_patch_kernel_cg_impl = False


class SingleStreamCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.SingleStream


class GH200_96GB_aarch64x1(SingleStreamGPUBaseConfig):
    slice_overlap_patch_kernel_cg_impl = True
    gpu_batch_size = 8
    single_stream_expected_latency_ns = 572434000


class GH200_96GB_aarch64x1_HighAccuracy(GH200_96GB_aarch64x1):
    pass


class L4x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 572434000
    slice_overlap_patch_kernel_cg_impl = True


class L4x1_HighAccuracy(L4x1):
    pass


class L40x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 572434000
    slice_overlap_patch_kernel_cg_impl = True


class L40x1_HighAccuracy(L40x1):
    pass


class H100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    single_stream_expected_latency_ns = 572434000


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    pass


class H100_PCIe_80GBx1_HighAccuracy_Triton(H100_PCIe_80GBx1_Triton):
    pass


class H100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 572434000


class H100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    single_stream_expected_latency_ns = 572434000


class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    pass


class H100_SXM_80GBx1_HighAccuracy_Triton(H100_SXM_80GBx1_Triton):
    pass


class A100_PCIe_80GBx1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    single_stream_expected_latency_ns = 542444704


class A100_PCIe_80GBx1_HighAccuracy(A100_PCIe_80GBx1):
    pass


class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


class A100_PCIe_80GBx1_HighAccuracy_Triton(A100_PCIe_80GBx1_Triton):
    pass


class A100_PCIe_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 90000000000
    workspace_size = 1073741824


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_PCIe_80GB_MIG_1x1g10gb_Triton):
    pass


class A100_PCIe_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    single_stream_expected_latency_ns = 780000000


class A100_PCIe_80GB_aarch64x1_HighAccuracy(A100_PCIe_80GB_aarch64x1):
    pass


class A100_PCIe_aarch64x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    single_stream_expected_latency_ns = 780000000


class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_HighAccuracy(A100_PCIe_aarch64x1):
    pass


class A100_PCIe_aarch64x1_HighAccuracy_Triton(A100_PCIe_aarch64x1_Triton):
    pass


class A100_SXM_80GB_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 5256000000
    start_from_device = True
    end_on_device = True
    workspace_size = 1073741824


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_MIG_1x1g10gb_Triton):
    pass


class A100_SXM_80GBx1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    start_from_device = True
    end_on_device = True
    single_stream_expected_latency_ns = 515000000


class A100_SXM_80GBx1_HighAccuracy(A100_SXM_80GBx1):
    pass


class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_HighAccuracy_Triton(A100_SXM_80GBx1_Triton):
    pass


class A100_SXM_80GB_aarch64x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    single_stream_expected_latency_ns = 520000000


class A100_SXM_80GB_aarch64x1_HighAccuracy(A100_SXM_80GB_aarch64x1):
    pass


class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True


class A100_SXM_80GB_aarch64x1_HighAccuracy_Triton(A100_SXM_80GB_aarch64x1_Triton):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 5256000000
    workspace_size = 1073741824


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero_HighAccuracy(A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_HighAccuracy_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton):
    pass


class A100_SXM4_40GB_MIG_1x1g5gb(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 5256000000
    start_from_device = True
    end_on_device = True
    workspace_size = 1073741824


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy(A100_SXM4_40GB_MIG_1x1g5gb):
    pass


class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_HighAccuracy_Triton(A100_SXM4_40GB_MIG_1x1g5gb_Triton):
    pass


class A100_SXM4_40GBx1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 8
    start_from_device = True
    end_on_device = True
    single_stream_expected_latency_ns = 657000000


class A100_SXM4_40GBx1_HighAccuracy(A100_SXM4_40GBx1):
    pass


class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_HighAccuracy_Triton(A100_SXM4_40GBx1_Triton):
    pass


class A2x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 5378800000


class A2x1_Triton(A2x1):
    use_triton = True


class A2x1_HighAccuracy(A2x1):
    pass


class A2x1_HighAccuracy_Triton(A2x1_Triton):
    pass


class A2x2(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 5378800000


class A2x2_HighAccuracy(A2x2):
    pass


class A30_MIG_1x1g6gb(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 5256000000
    workspace_size = 805306368


class A30_MIG_1x1g6gb_HighAccuracy(A30_MIG_1x1g6gb):
    pass


class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    single_stream_expected_latency_ns = 5256000000


class A30_MIG_1x1g6gb_Hetero_HighAccuracy(A30_MIG_1x1g6gb_Hetero):
    pass


class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_triton = True


class A30_MIG_1x1g6gb_HighAccuracy_Triton(A30_MIG_1x1g6gb_Triton):
    pass


class A30x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 2
    single_stream_expected_latency_ns = 1226400000


class A30x1_HighAccuracy(A30x1):
    pass


class A30x1_Triton(A30x1):
    use_triton = True


class A30x1_HighAccuracy_Triton(A30x1_Triton):
    pass


class Orin(SingleStreamGPUBaseConfig):
    gpu_batch_size = 2
    single_stream_expected_latency_ns = 2222222222
    use_direct_host_access = True


class Orin_MaxQ(Orin):
    soc_cpu_freq = 576000
    soc_gpu_freq = 816000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    min_query_count = 430
    min_duration = 1200000
    single_stream_expected_latency_ns = 5991338811


class Orin_HighAccuracy(Orin):
    pass


class Orin_MaxQ_HighAccuracy(Orin_MaxQ):
    pass


class Orin_NX(SingleStreamGPUBaseConfig):
    gpu_batch_size = 2
    single_stream_expected_latency_ns = 9826000000
    use_direct_host_access = True


class Orin_NX_MaxQ(Orin_NX):
    soc_cpu_freq = 1113600
    soc_gpu_freq = 918000000
    soc_dla_freq = 0
    soc_emc_freq = 3199000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    min_query_count = 430
    min_duration = 2400000
    single_stream_expected_latency_ns = 11741058560


class Orin_NX_HighAccuracy(Orin_NX):
    pass


class Orin_NX_HighAccuracy(Orin_NX_MaxQ):
    pass


class T4x1(SingleStreamGPUBaseConfig):
    gpu_batch_size = 1
    single_stream_expected_latency_ns = 2500000000


class T4x1_HighAccuracy(T4x1):
    pass


class T4x1_Triton(T4x1):
    use_triton = True


class T4x1_HighAccuracy_Triton(T4x1_Triton):
    pass
