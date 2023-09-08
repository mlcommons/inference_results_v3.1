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


class OfflineGPUBaseConfig(GPUBaseConfig):
    scenario = Scenario.Offline
    use_graphs = True
    num_warmups = 512
    audio_batch_size = 512
    audio_buffer_num_lines = 4096
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_inference_streams = 1
    gpu_copy_streams = 1


class GH200_96GB_aarch64x1(OfflineGPUBaseConfig):
    start_from_device = True
    gpu_batch_size = 2048
    offline_expected_qps = 26000
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False


class L4x1(OfflineGPUBaseConfig):
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024


class L40x1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 13300


class A100_PCIe_80GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 13300


class H100_PCIe_80GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    use_graphs = True  # MLPINF-1773
    offline_expected_qps = 17000
    disable_encoder_plugin = False


class H100_PCIe_80GBx8(H100_PCIe_80GBx1):
    gpu_batch_size = 2048
    offline_expected_qps = 15000 * 8


class H100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    use_graphs = False  # MLPINF-1773
    offline_expected_qps = 16000
    disable_encoder_plugin = False


class H100_PCIe_80GB_aarch64x4(H100_PCIe_80GB_aarch64x1):
    gpu_batch_size = 2048
    offline_expected_qps = 16000 * 4


class H100_SXM_80GBx1(OfflineGPUBaseConfig):
    start_from_device = True
    gpu_batch_size = 2048
    offline_expected_qps = 23000
    # audio_batch_size = 512
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False


class H100_SXM_80GBx8(H100_SXM_80GBx1):
    start_from_device = False
    offline_expected_qps = 21000 * 8
    gpu_batch_size = 2048
    # max_seq_length = 180


class H100_SXM_80GBx8_MaxQ(H100_SXM_80GBx8):
    power_limit = 300
    offline_expected_qps = 144000


class H100_SXM_80GB_02x1(H100_SXM_80GBx1):
    pass


class A100_PCIe_80GBx8(A100_PCIe_80GBx1):
    offline_expected_qps = 96000.0
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    num_warmups = 40480


class A100_PCIe_80GBx8_MaxQ(A100_PCIe_80GBx8):
    offline_expected_qps = 80000
    power_limit = 180
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    num_warmups = 40480


class A100_PCIe_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 1024
    offline_expected_qps = 1550
    num_warmups = 64
    workspace_size = 3221225472
    max_seq_length = 64


class A100_PCIe_80GB_MIG_1x1g10gb_Hetero(A100_PCIe_80GB_MIG_1x1g10gb):
    pass


class A100_PCIe_80GB_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 12000


class A100_PCIe_80GB_aarch64x2(A100_PCIe_80GB_aarch64x1):
    offline_expected_qps = 24000.0


class A100_PCIe_80GB_aarch64x4(A100_PCIe_80GB_aarch64x1):
    offline_expected_qps = 48000.0


class A100_PCIe_80GB_aarch64x4_MaxQ(A100_PCIe_80GB_aarch64x4):
    offline_expected_qps = 40000
    power_limit = 200


class A100_PCIe_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 12000


class A100_PCIe_aarch64x2(A100_PCIe_aarch64x1):
    offline_expected_qps = 24000.0


class A100_PCIe_aarch64x4(A100_PCIe_aarch64x1):
    offline_expected_qps = 48000.0


class A100_PCIe_aarch64x4_MaxQ(A100_PCIe_aarch64x4):
    offline_expected_qps = 40000
    power_limit = 200


class A100_PCIe_MIG_1x1g5gb(OfflineGPUBaseConfig):
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 256
    offline_expected_qps = 1350
    num_warmups = 64
    workspace_size = 3221225472


class A100_SXM_80GB_MIG_1x1g10gb(OfflineGPUBaseConfig):
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 1024
    offline_expected_qps = 1760
    num_warmups = 64
    workspace_size = 3221225472
    max_seq_length = 64


class A100_SXM_80GB_MIG_1x1g10gb_Hetero(A100_SXM_80GB_MIG_1x1g10gb):
    offline_expected_qps = 1500


class A100_SXM_80GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 13800
    start_from_device = True


class A100_SXM_80GBx8(A100_SXM_80GBx1):
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    offline_expected_qps = 104000
    num_warmups = 40480
    nobatch_sorting = True


class A100_SXM_80GBx8_MaxQ(A100_SXM_80GBx8):
    power_limit = 225


class A100_SXM_80GB_aarch64_MIG_1x1g10gb(OfflineGPUBaseConfig):
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 1024
    offline_expected_qps = 1550
    num_warmups = 64
    workspace_size = 3221225472
    max_seq_length = 64


class A100_SXM_80GB_aarch64_MIG_1x1g10gb_Hetero(A100_SXM_80GB_aarch64_MIG_1x1g10gb):
    offline_expected_qps = 1500


class A100_SXM_80GB_aarch64x1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 13800


class A100_SXM_80GB_aarch64x8(A100_SXM_80GB_aarch64x1):
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    offline_expected_qps = 105000
    num_warmups = 40480
    nobatch_sorting = True


class A100_SXM_80GB_aarch64x8_MaxQ(A100_SXM_80GB_aarch64x8):
    offline_expected_qps = 93000
    power_limit = 250


class A100_SXM4_40GB_MIG_1x1g5gb(OfflineGPUBaseConfig):
    audio_batch_size = 64
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 256
    offline_expected_qps = 1350
    num_warmups = 64
    workspace_size = 3221225472


class A100_SXM4_40GBx1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 11800
    start_from_device = True


class A100_SXM4_40GBx8(A100_SXM4_40GBx1):
    offline_expected_qps = 100000
    num_warmups = 40480
    audio_buffer_num_lines = None


class A2x1(OfflineGPUBaseConfig):
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 1150


class A2x2(OfflineGPUBaseConfig):
    dali_batches_issue_ahead = 2
    dali_pipeline_depth = 2
    gpu_batch_size = 512
    audio_batch_size = 128
    offline_expected_qps = 2260


class A30_MIG_1x1g6gb(OfflineGPUBaseConfig):
    audio_batch_size = 32
    audio_buffer_num_lines = 512
    dali_batches_issue_ahead = 1
    dali_pipeline_depth = 1
    gpu_batch_size = 1024
    offline_expected_qps = 1450
    max_seq_length = 64
    num_warmups = 32
    workspace_size = 1610612736


class A30_MIG_1x1g6gb_Hetero(A30_MIG_1x1g6gb):
    offline_expected_qps = 1378


class A30x1(OfflineGPUBaseConfig):
    gpu_batch_size = 2048
    offline_expected_qps = 6959.999999999999


class A30x8(A30x1):
    offline_expected_qps = 55679.99999999999


class A30x8_MaxQ(A30x8):
    offline_expected_qps = 46400.0
    power_limit = 250


class T4x1(OfflineGPUBaseConfig):
    audio_batch_size = 128
    disable_encoder_plugin = True
    gpu_batch_size = 2048
    gpu_copy_streams = 4
    offline_expected_qps = 1400
    num_warmups = 2048


class T4x20(T4x1):
    offline_expected_qps = 30000
    num_warmups = 40960


class T4x8(T4x1):
    offline_expected_qps = 11400
    num_warmups = 20480


class Orin(OfflineGPUBaseConfig):
    audio_batch_size = 64
    audio_buffer_num_lines = 1024
    disable_encoder_plugin = True
    gpu_batch_size = 512
    gpu_copy_streams = 4
    offline_expected_qps = 1110
    num_warmups = 2048


class Orin_MaxQ(Orin):
    #NOTE: Orin AGX 3.1 Shmoo
    soc_cpu_freq = 652800
    soc_gpu_freq = 510000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 115000000
    orin_num_cores = 2
    offline_expected_qps = 580


class Orin_NX(OfflineGPUBaseConfig):
    audio_batch_size = 64
    audio_buffer_num_lines = 1024
    disable_encoder_plugin = True
    gpu_batch_size = 512
    gpu_copy_streams = 4
    offline_expected_qps = 410
    num_warmups = 2048


class Orin_NX_MaxQ(Orin_NX):
    #NOTE: Orin NX 3.1 Shmoo
    soc_cpu_freq = 652800
    soc_gpu_freq = 714000000
    soc_dla_freq = 0
    soc_emc_freq = 2133000000
    soc_pva_freq = 0
    orin_num_cores = 2
    offline_expected_qps = 300


class H100_PCIe_80GBx8_MaxQ(H100_PCIe_80GBx8):
    offline_expected_qps = 99000
    power_limit = 200
