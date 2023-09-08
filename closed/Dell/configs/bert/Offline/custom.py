# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 1024
    gemm_plugin_fairshare_cache_size: int = 120
    offline_expected_qps: int = 15000
    use_small_tile_gemm_plugin: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4_HighAccuracy(R750XA_A100_PCIE_80GBX4):
    precision = "fp16"
    offline_expected_qps = R750XA_A100_PCIE_80GBX4.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 1280
    enable_interleaved: bool = False
    offline_expected_qps: float = 25000
    use_small_tile_gemm_plugin: bool = False
    workspace_size: int = 7516192768
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 20900
    use_graphs = False
    gpu_batch_size = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    offline_expected_qps = 13000
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2_HighAccuracy(R760xa_H100_PCIe_80GBx2):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 10000
    use_graphs = False
    gpu_batch_size = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 1280
    enable_interleaved: bool = False
    offline_expected_qps: float = 25000
    use_small_tile_gemm_plugin: bool = False
    workspace_size: int = 7516192768
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4_HighAccuracy(R760xa_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    offline_expected_qps = 21000
    use_graphs = False
    gpu_batch_size = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    use_small_tile_gemm_plugin = True
    gpu_copy_streams = 2
    gpu_inference_streams = 4
    gpu_batch_size = 134
    offline_expected_qps = 4*3500
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_L40X4_HighAccuracy(R760XA_L40X4):
    precision = "fp16"
    gpu_inference_streams = 3
    gpu_copy_streams = 3
    gpu_batch_size = 128
    offline_expected_qps = 3400 *2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1356
    gpu_copy_streams = 10
    gpu_inference_streams = 2
    offline_expected_qps = 40420
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024
    offline_expected_qps = 8200*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 74500
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1024
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    offline_expected_qps = 64600
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_batch_size = 1280
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 39000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4_HighAccuracy(XE9640_H100_SXM_80GBX4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 1380
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    offline_expected_qps = 33800
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_L4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_L4x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 16
    offline_expected_qps = 1000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_L4X1_HighAccuracy(XR4520C_L4X1):
    precision = "fp16"
    offline_expected_qps = XR4520C_L4X1.offline_expected_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 16
    energy_aware_kernels = True
    offline_expected_qps = 1000
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ_HighAccuracy(XR5610_L4x1_MaxQ):
    precision = "fp16"
    use_fp8 = True
    gpu_batch_size = 16
    offline_expected_qps = 640
    gpu_inference_streams = 1
    energy_aware_kernels = False
    gpu_copy_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    use_small_tile_gemm_plugin = True
    gpu_batch_size = 13
    offline_expected_qps = 1060
    workspace_size = 7516192768

