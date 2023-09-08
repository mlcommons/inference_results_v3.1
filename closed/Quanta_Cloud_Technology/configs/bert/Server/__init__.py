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


class ServerGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Server
    enable_interleaved = False
    use_graphs = True
    gpu_copy_streams = 1
    gpu_inference_streams = 2
    use_small_tile_gemm_plugin = True


class ServerCPUBaseConfig(CPUBaseConfig):
    scenario = Scenario.Server

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54U_3U_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = 256
    server_target_qps = 18000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    soft_drop = 1.0
    #soft_drop = 0.99

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54U_3U_H100_PCIe_80GBx4_HighAccuracy(D54U_3U_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 16000
    soft_drop = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54Q_2U_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.D54Q_2U_H100_PCIe_80GBx2
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = 256
    server_target_qps = 9000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    # soft_drop = 1.0
    soft_drop = 0.99

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54Q_2U_H100_PCIe_80GBx2_HighAccuracy(D54Q_2U_H100_PCIe_80GBx2):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 8000
    soft_drop = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class D54Q_2U_L4_PCIe_24GBx4(ServerGPUBaseConfig):
    system = KnownSystem.D54Q_2U_L4_PCIe_24GBx4
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 3600
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class D54Q_2U_L4_PCIe_24GBx4_HighAccuracy(D54Q_2U_L4_PCIe_24GBx4):
    precision = "fp16"
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 2000
    soft_drop = 1.0
    use_fp8 = True
    use_graphs = False
    use_small_tile_gemm_plugin = False
