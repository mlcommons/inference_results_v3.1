# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 57000*4
    #start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL320_Gen11_L4x4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL320_Gen11_L4x4
    gpu_batch_size = 32
    gpu_inference_streams = 1
    gpu_copy_streams = 2
    offline_expected_qps = 13000*4
    use_graphs = True

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_XL675d_A100_SXM_80GBx8
    run_infer_on_copy_streams = False
    gpu_inference_streams = 2
    gpu_copy_streams = 3
    offline_expected_qps = 340000
    gpu_batch_size = 2048
