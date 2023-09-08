# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(MultiStreamGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    multi_stream_expected_latency_ns = 12000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(MultiStreamGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    multi_stream_expected_latency_ns = 1800000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4_Triton(R760XA_L40X4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    multi_stream_expected_latency_ns = 40000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(MultiStreamGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    multi_stream_expected_latency_ns = 40000000
