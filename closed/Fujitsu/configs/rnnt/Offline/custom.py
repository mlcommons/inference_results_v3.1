# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1(OfflineGPUBaseConfig):
    system = KnownSystem.PRIMERGY_CDI_V1
    offline_expected_qps = 51800.0
    audio_batch_size = 1024
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 1
    num_warmups = 40480
    gpu_batch_size = 2048





