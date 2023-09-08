# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(SingleStreamGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    use_fp8 = True
    single_stream_expected_latency_ns = 1876267940

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_L40X4_HighAccuracy(R760XA_L40X4):
    pass

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(SingleStreamGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_fp8 = True
    single_stream_expected_latency_ns = 1876267940

