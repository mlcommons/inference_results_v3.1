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
from configs.resnet50 import GPUBaseConfig


class MultiStreamGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.MultiStream
    gpu_batch_size = 8
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    multi_stream_samples_per_query = 8
    multi_stream_target_latency_percentile = 99
    use_graphs = True


class GH200_96GB_aarch64x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class L4x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class L40x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class H100_SXM_80GBx1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    use_triton = True


class H100_PCIe_80GBx1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    use_triton = True


class H100_PCIe_80GB_aarch64x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class A100_PCIe_80GBx1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class A100_PCIe_80GBx1_Triton(A100_PCIe_80GBx1):
    use_triton = True


class A100_PCIe_80GBx1_TritonUnified(A100_PCIe_80GBx1):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_80GB_aarch64x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class A100_PCIe_80GB_aarch64x1_Triton(A100_PCIe_80GB_aarch64x1):
    use_triton = True


class A100_PCIe_80GB_aarch64x1_TritonUnified(A100_PCIe_80GB_aarch64x1):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_aarch64x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 830000


class A100_PCIe_aarch64x1_Triton(A100_PCIe_aarch64x1):
    use_triton = True


class A100_PCIe_aarch64x1_TritonUnified(A100_PCIe_aarch64x1):
    use_triton = True
    batch_triton_requests = True


class A100_PCIe_80GB_MIG_1x1g10gb(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 2160000


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


class A100_PCIe_80GB_MIG_1x1g10gb_Triton(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_PCIe_80GB_MIG_1x1g10gb_TritonUnified(A100_PCIe_80GB_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM_80GB_MIG_1x1g10gb(MultiStreamGPUBaseConfig):
    start_from_device = True
    multi_stream_expected_latency_ns = 2100000


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM_80GBx1(MultiStreamGPUBaseConfig):
    start_from_device = True
    multi_stream_expected_latency_ns = 693000


class A100_SXM_80GBx1_Triton(A100_SXM_80GBx1):
    use_triton = True


class A100_SXM_80GBx1_TritonUnified(A100_SXM_80GBx1):
    use_triton = True
    batch_triton_requests = True


class A100_SXM_80GB_aarch64x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 6430000


class A100_SXM_80GB_aarch64x1_Triton(A100_SXM_80GB_aarch64x1):
    use_triton = True
    multi_stream_expected_latency_ns = 1000000


class A100_SXM_80GB_aarch64x1_TritonUnified(A100_SXM_80GB_aarch64x1):
    use_triton = True
    multi_stream_expected_latency_ns = 1000000
    batch_triton_requests = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 2100000


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    pass


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Triton(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_TritonUnified(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM4_40GB_MIG_1x1g5gb(MultiStreamGPUBaseConfig):
    start_from_device = True
    multi_stream_expected_latency_ns = 670000


class A100_SXM4_40GB_MIG_1x1g5gb_Triton(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True


class A100_SXM4_40GB_MIG_1x1g5gb_TritonUnified(A100_SXM4_40GB_MIG_1x1g5gb):
    use_triton = True
    batch_triton_requests = True


class A100_SXM4_40GBx1(MultiStreamGPUBaseConfig):
    start_from_device = True
    multi_stream_expected_latency_ns = 670000


class A100_SXM4_40GBx1_Triton(A100_SXM4_40GBx1):
    use_triton = True


class A100_SXM4_40GBx1_TritonUnified(A100_SXM4_40GBx1):
    use_triton = True
    batch_triton_requests = True


class A2x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 5840000


class A2x1_Triton(A2x1):
    use_triton = True


class A2x1_TritonUnified(A2x1):
    use_triton = True
    batch_triton_requests = True


class A30_MIG_1x1g6gb(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 5726152


class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    multi_stream_expected_latency_ns = 5968320


class A30_MIG_1x1g6gb_Triton(A30_MIG_1x1g6gb):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 6889088


class A30_MIG_1x1g6gb_TritonUnified(A30_MIG_1x1g6gb):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 6889088
    batch_triton_requests = True


class A30x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 960000


class A30x1_Triton(A30x1):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 1330000


class A30x1_TritonUnified(A30x1):
    use_graphs = False
    use_triton = True
    multi_stream_expected_latency_ns = 1330000
    batch_triton_requests = True


class T4x1(MultiStreamGPUBaseConfig):
    multi_stream_expected_latency_ns = 7973184


class T4x1_Triton(T4x1):
    use_triton = True


class T4x1_TritonUnified(T4x1):
    use_triton = True
    batch_triton_requests = True


class Orin(MultiStreamGPUBaseConfig):
    gpu_copy_streams = 2
    use_direct_host_access = True
    multi_stream_expected_latency_ns = 2330000
    disable_beta1_smallk = True  # TODO: Test perf impact in MS


class Orin_Triton(Orin):
    use_triton = True


class Orin_TritonUnified(Orin):
    use_triton = True
    batch_triton_requests = True


class Orin_MaxQ(Orin):
    soc_cpu_freq = 576000
    soc_gpu_freq = 624750000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    gpu_copy_streams = 1
    multi_stream_expected_latency_ns = 2870000


class Orin_NX(MultiStreamGPUBaseConfig):
    gpu_copy_streams = 2
    use_direct_host_access = True
    multi_stream_expected_latency_ns = 5800000
    disable_beta1_smallk = True  # TODO: Test perf impact in MS


class Orin_NX_MaxQ(Orin_NX):
    soc_cpu_freq = 576000
    soc_gpu_freq = 624750000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 4
    gpu_copy_streams = 1
    multi_stream_expected_latency_ns = 5800000
