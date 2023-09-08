# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1(ServerGPUBaseConfig):
    system = KnownSystem.PRIMERGY_CDI_V1
    server_target_qps = 48200
    gpu_batch_size = 2048


