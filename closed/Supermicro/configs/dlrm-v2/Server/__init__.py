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

ParentConfig = import_module("configs.dlrm-v2")
GPUBaseConfig = ParentConfig.GPUBaseConfig


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server


class GH200_96GB_aarch64x1(ServerGPUBaseConfig):
    gpu_batch_size = 204800
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 48500


class GH200_96GB_aarch64x1_High_Accuracy(GH200_96GB_aarch64x1):
    pass


class H100_SXM_80GBx1(ServerGPUBaseConfig):
    gpu_batch_size = 51200 * 2
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 41500


class H100_SXM_80GBx1_HighAccuracy(H100_SXM_80GBx1):
    pass


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    gpu_batch_size = 51200
    server_target_qps = 315000
    server_num_issue_query_threads = 8
    numa_config = "0-3:0-55,112-167&4-7:56-111,168-223"


class H100_SXM_80GBx8_HighAccuracy(H100_SXM_80GBx8):
    pass


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    server_target_qps = 30500 * 8
    power_limit = 450


class H100_SXM_80GBx8_HighAccuracy_MaxQ(H100_SXM_80GBx8_MaxQ):
    pass


class H100_PCIe_80GBx1(ServerGPUBaseConfig):
    gpu_batch_size = 51200
    embedding_weights_on_gpu_part: float = 1.0
    server_target_qps = 24500


class H100_PCIe_80GBx1_HighAccuracy(H100_PCIe_80GBx1):
    pass


class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    server_target_qps = 170000
    server_num_issue_query_threads = 8
    numa_config = "3:0-15,128-143&2:16-31,144-159&1:32-47,160-175&0:48-63,176-191&7:64-79,192-207&6:80-95,208-223&5:96-111,224-239&4:112-127,240-255"
    qsl_numa_override = "0-3&4-7"


class H100_PCIe_80GBx8_HighAccuracy(H100_PCIe_80GBx8):
    pass


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    server_target_qps = 16500 * 8
    power_limit = 275


class H100_PCIe_80GBx8_HighAccuracy_MaxQ(H100_PCIe_80GBx8_MaxQ):
    pass


class L4x1(ServerGPUBaseConfig):
    embedding_weights_on_gpu_part = 0.3
    gpu_batch_size = 1400
    server_target_qps = 3300
    max_pairs_per_staging_thread = 262100


class L4x1_HighAccuracy(L4x1):
    pass
