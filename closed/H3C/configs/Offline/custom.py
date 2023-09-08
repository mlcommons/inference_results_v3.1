# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5350G6_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5350G6_L40x8
    gpu_batch_size = 11         
    use_fp8 = True
    offline_expected_qps = 26.2         
	

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5350G6_L40x8_HighAccuracy(R5350G6_L40X8):
    precision = "fp16"

