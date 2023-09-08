# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4
    gpu_batch_size = 8
    slice_overlap_patch_kernel_cg_impl = True
    offline_expected_qps = 4.8*4

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4_HighAccuracy(HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_DL320_Gen11_L4x4(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_DL320_Gen11_L4x4
    gpu_batch_size = 1
    offline_expected_qps = 1.3*4
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8(OfflineGPUBaseConfig):
    system = KnownSystem.HPE_ProLiant_XL675d_A100_SXM_80GBx8
    offline_expected_qps = 30
    use_cuda_thread_per_device = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class HPE_ProLiant_XL675d_A100_SXM_80GBx8_HighAccuracy(HPE_ProLiant_XL675d_A100_SXM_80GBx8):
    pass

