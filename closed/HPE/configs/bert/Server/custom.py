# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = 256
    server_target_qps = 14000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    soft_drop = 0.99

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 3500*4
    soft_drop = 1.0
    gpu_batch_size = 94

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL320_Gen11_L4x4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL320_Gen11_L4x4
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 900*4
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL320_Gen11_L4x4_HighAccuracy(HPE_ProLiant_DL320_Gen11_L4x4):
    precision = "fp16"
    gpu_batch_size = 16
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 540*4
    soft_drop = 1.0
    use_fp8 = True
    use_graphs = False
    use_small_tile_gemm_plugin = False

class HPE_ProLiant_XL675d_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_XL675d_A100_SXM_80GBx8
    active_sms = 60
    gpu_batch_size = 48
    graphs_max_seqlen = 240
    server_num_issue_query_threads = 0
    server_target_qps = 25400
    soft_drop = 1.0
    gpu_copy_streams = 4
    gpu_inference_streams = 2

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8_HighAccuracy(HPE_ProLiant_XL675d_A100_SXM_80GBx8):
    gpu_batch_size = 24
    precision = "fp16"
    server_target_qps = 12820
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    soft_drop = 1.0
