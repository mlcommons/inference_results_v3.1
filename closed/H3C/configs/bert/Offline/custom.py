# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 128     
    offline_expected_qps = 32000       
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_L40X8_HighAccuracy(R5300G6_L40X8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 64     
    offline_expected_qps = 25000       

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 64      
    offline_expected_qps = 3400     
    workspace_size = 7516192768


