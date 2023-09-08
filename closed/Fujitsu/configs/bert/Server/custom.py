# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1(ServerGPUBaseConfig):
    system = KnownSystem.PRIMERGY_CDI_V1
    active_sms = 60
    gpu_batch_size = 64
    graphs_max_seqlen = 200
    server_num_issue_query_threads = 1
    server_target_qps = 11845.0
    soft_drop = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class PRIMERGY_CDI_V1_HighAccuracy(PRIMERGY_CDI_V1):
    precision = "fp16"
    server_target_qps = 5783



