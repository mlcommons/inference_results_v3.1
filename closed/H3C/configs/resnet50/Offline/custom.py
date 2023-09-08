# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8
    gpu_inference_streams = 2    
    gpu_copy_streams = 4
    gpu_batch_size = 64          
    offline_expected_qps = 320000            

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x1
    gpu_batch_size = 512
    offline_expected_qps = 40000




