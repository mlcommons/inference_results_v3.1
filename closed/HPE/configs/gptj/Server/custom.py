# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4
    gpu_batch_size = 24
    use_fp8 = True
    server_target_qps = 7.5
#    server_target_qps = 2.5*4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4):
    server_target_qps = 7.5
#    server_target_qps = 2.5*4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL320_Gen11_L4x4(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL320_Gen11_L4x4
    gpu_batch_size = 512
    server_target_qps = 3800*4
    audio_batch_size = 64
    audio_buffer_num_lines = 1024

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8(ServerGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_XL675d_A100_SXM_80GBx8
    gpu_batch_size = 24
    server_target_qps = 16
    use_fp8 = False

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8_HighAccuracy(HPE_ProLiant_XL675d_A100_SXM_80GBx8):
    precision = "fp16"
