# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 5700*4
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 5000*4
    use_graphs = False
    gpu_batch_size = 1024

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL320_Gen11_L4x4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL320_Gen11_L4x4
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 16
    offline_expected_qps = 980*4
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL320_Gen11_L4x4_HighAccuracy(HPE_ProLiant_DL320_Gen11_L4x4):
    precision = "fp16"
    use_fp8 = True
    gpu_batch_size = 16
    offline_expected_qps = 630*4
    gpu_inference_streams = 1
    gpu_copy_streams = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_XL675d_A100_SXM_80GBx8
    offline_expected_qps = 30000
    workspace_size = 7516192768
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 1280
    gpu_inference_streams = 1

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8_HighAccuracy(HPE_ProLiant_XL675d_A100_SXM_80GBx8):
    precision = "fp16"
    gpu_batch_size = 512
    offline_expected_qps = 15000
