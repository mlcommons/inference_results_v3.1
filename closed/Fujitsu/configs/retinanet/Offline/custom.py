# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1(OfflineGPUBaseConfig):
    system = KnownSystem.PRIMERGY_CDI_V1
    gpu_batch_size = 16
    gpu_copy_streams = 4
    gpu_inference_streams = 4
    offline_expected_qps = 2970
    run_infer_on_copy_streams = False
    workspace_size = 60000000000




