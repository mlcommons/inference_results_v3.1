# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8

    gpu_inference_streams = 1
    gpu_copy_streams = 3
    gpu_batch_size = 1024     
    server_target_qps = 81000                 


