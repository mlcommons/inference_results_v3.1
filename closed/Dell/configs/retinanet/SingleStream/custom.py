# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(SingleStreamGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    single_stream_expected_latency_ns = 2900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(SingleStreamGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    single_stream_expected_latency_ns = 1891750


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4_Triton(R760XA_L40X4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_L4X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_L4x1
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = False
    disable_beta1_smallk = True
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    single_stream_expected_latency_ns = 5900000
