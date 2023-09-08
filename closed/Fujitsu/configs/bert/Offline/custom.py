# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1(OfflineGPUBaseConfig):
    system = KnownSystem.PRIMERGY_CDI_V1
    offline_expected_qps = 12800
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1024
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class PRIMERGY_CDI_V1_HighAccuracy(PRIMERGY_CDI_V1):
    precision = "fp16"
    offline_expected_qps = PRIMERGY_CDI_V1.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1_Triton(PRIMERGY_CDI_V1):
    use_triton = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class PRIMERGY_CDI_V1_HighAccuracy_Triton(PRIMERGY_CDI_V1_HighAccuracy):
    use_triton = True


