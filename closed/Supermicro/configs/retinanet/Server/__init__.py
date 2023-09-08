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


ParentConfig = import_module("configs.retinanet")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    active_sms = 100
    use_graphs = False
    use_cuda_thread_per_device = True


class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 2
    server_target_qps = 1730
    workspace_size = 60000000000


class L4x1(ServerGPUBaseConfig):
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 2
    gpu_inference_streams = 2
    server_target_qps = 200
    workspace_size = 20000000000


class L40x1(ServerGPUBaseConfig):
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 250
    workspace_size = 70000000000


class H100_SXM_80GBx1(ServerGPUBaseConfig):
    start_from_device = True
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    # gpu_batch_size = 8
    gpu_batch_size = 16
    gpu_inference_streams = 2
    server_target_qps = 1620
    workspace_size = 60000000000


class H100_SXM_80GBx1_Triton(H100_SXM_80GBx1):
    server_target_qps = 680
    instance_group_count = 4
    use_triton = True


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    gpu_copy_streams = 4
    gpu_batch_size = 8
    # server_target_qps = 1560 * 8
    server_target_qps = 1610 * 8


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    power_limit = 350
    server_target_qps = 1100 * 8


class H100_SXM_80GBx8_Triton(H100_SXM_80GBx1_Triton):
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 840 * 8


class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 1050
    workspace_size = 60000000000


class H100_PCIe_80GBx1_Triton(H100_PCIe_80GBx1):
    server_target_qps = 600
    instance_group_count = 4
    use_triton = True


class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 1050 * 8


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    server_target_qps = 6300
    power_limit = 225


class H100_PCIe_80GBx8_Triton(H100_PCIe_80GBx1_Triton):
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 700 * 8


class H100_PCIe_80GB_aarch64x1(ServerGPUBaseConfig):
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 700
    workspace_size = 60000000000


class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    gpu_copy_streams = 4
    gpu_batch_size = 8
    server_target_qps = 700 * 4


class A100_PCIe_80GBx8(ServerGPUBaseConfig):
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 580 * 8
    workspace_size = 70000000000


class A100_PCIe_80GBx8_Triton(A100_PCIe_80GBx8):
    server_target_qps = 3300
    instance_group_count = 4
    use_triton = True


class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    server_target_qps = 450 * 8
    power_limit = 200


class A100_PCIe_80GBx8_Triton_MaxQ(A100_PCIe_80GBx8_Triton):
    server_target_qps = 2100
    power_limit = 200


class A100_SXM_80GBx8(ServerGPUBaseConfig):
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 16
    gpu_inference_streams = 4
    server_target_qps = 700 * 8
    start_from_device = True
    workspace_size = 70000000000


class A100_SXM_80GBx8_Triton(A100_SXM_80GBx8):
    start_from_device = None
    server_target_qps = 3735
    instance_group_count = 4
    use_triton = True


class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    server_target_qps = 560 * 8
    power_limit = 250


class A100_SXM_80GBx8_Triton_MaxQ(A100_SXM_80GBx8_Triton):
    server_target_qps = 2400
    power_limit = 250


class A100_SXM_80GB_MIG_1x1g10gb(ServerGPUBaseConfig):
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 20000
    gpu_batch_size = 1
    gpu_inference_streams = 2
    server_target_qps = 45
    start_from_device = True

    # Early stopping. For some reason this breaks --fast...
    min_query_count = 1
    min_duration = 30 * 60 * 1000


class A100_SXM_80GB_MIG_1x1g10gb_Triton(A100_SXM_80GB_MIG_1x1g10gb):
    server_target_qps = 40
    use_triton = True


class A2x2(ServerGPUBaseConfig):
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 20000
    gpu_batch_size = 2
    gpu_inference_streams = 1
    server_target_qps = 55


class A2x2_Triton(A2x2):
    use_triton = True
    server_target_qps = 47
