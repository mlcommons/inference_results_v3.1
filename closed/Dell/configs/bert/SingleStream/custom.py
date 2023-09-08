# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(SingleStreamGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000

@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4_Triton(R750XA_A100_PCIE_80GBX4):
    use_triton = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(SingleStreamGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_L40X4_HighAccuracy(R760XA_L40X4):
    precision = "fp16"
    single_stream_expected_latency_ns = R760XA_L40X4.single_stream_expected_latency_ns * 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_L4X1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR4520c_L4x1
    gpu_batch_size = 1
    gpu_copy_streams = 1
    gpu_inference_streams = 1
    use_graphs = True
    bert_opt_seqlen = 270
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    single_stream_expected_latency_ns = 1700000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_L4X1_HighAccuracy(XR4520C_L4X1):
    precision = "fp16"
    single_stream_expected_latency_ns = XR4520C_L4X1.single_stream_expected_latency_ns * 2


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    single_stream_expected_latency_ns = 5900000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(SingleStreamGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    enable_interleaved = False
    single_stream_expected_latency_ns = 1601000 
