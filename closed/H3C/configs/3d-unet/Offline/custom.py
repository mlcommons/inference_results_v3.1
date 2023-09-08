# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8

    gpu_batch_size = 1  
    offline_expected_qps = 27 
    slice_overlap_patch_kernel_cg_impl = True
    numa_config = "0-7:0-43"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_L40X8_HighAccuracy(R5300G6_L40X8):
    offline_expected_qps = 24

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x1

    gpu_batch_size = 1
    offline_expected_qps = 1.3
    slice_overlap_patch_kernel_cg_impl = True

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_L40X1_HighAccuracy(R5300G6_L40X1):
    pass





