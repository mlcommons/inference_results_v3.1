# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class ND_H100_V5(OfflineGPUBaseConfig):
    system = KnownSystem.ND_H100_v5
    gpu_batch_size = 32
    offline_expected_qps = 106
    enable_sort = True
    use_fp8 = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class ND_H100_V5_HighAccuracy(ND_H100_V5):
    pass
