# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8

    gpu_batch_size = 16               
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 18560        #18520         
    soft_drop = 1.0        
    use_small_tile_gemm_plugin = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_L40X8_HighAccuracy(R5300G6_L40X8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 32
    server_target_qps = 10200              

