# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND_H100_V5(ServerGPUBaseConfig):
    system = KnownSystem.ND_H100_v5
    gpu_batch_size = 32
    use_fp8 = True
    server_target_qps = 84

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND_H100_V5_HighAccuracy(ND_H100_V5):
    pass
