# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1(OfflineGPUBaseConfig):
    system = KnownSystem.PRIMERGY_CDI_V1
    offline_expected_qps = 150000.0
    gpu_batch_size = 2048
    gpu_copy_streams = 4



