# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x1

    gpu_batch_size = 1
    single_stream_expected_latency_ns = 520000000     #520000000       
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R5300G6_L40X1_HighAccuracy(R5300G6_L40X1):
    single_stream_expected_latency_ns = 500000000       



