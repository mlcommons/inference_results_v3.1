# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size: int = 1
    offline_expected_qps: int = 15


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4_HighAccuracy(R750XA_A100_PCIE_80GBX4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 8
    offline_expected_qps: float = 19.2
    slice_overlap_patch_kernel_cg_impl: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 8
    offline_expected_qps: float = 38.2
    slice_overlap_patch_kernel_cg_impl: bool = True
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4_HighAccuracy(R760xa_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    gpu_batch_size = 1
    offline_expected_qps = 14
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_L40X4_HighAccuracy(R760XA_L40X4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    gpu_batch_size = 8
    slice_overlap_patch_kernel_cg_impl = True
    offline_expected_qps = 10.2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2_HighAccuracy(R760xa_H100_PCIe_80GBx2):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 8
    offline_expected_qps: float = 38.2
    slice_overlap_patch_kernel_cg_impl: bool = True
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4_HighAccuracy(R760xa_H100_PCIe_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 8
    offline_expected_qps = 6.8*4


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 54
    gpu_batch_size = 8
    start_from_device = True
    slice_overlap_patch_kernel_cg_impl = False
    use_cuda_thread_per_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_L4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_L4x1
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    gpu_batch_size = 1
    offline_expected_qps = 1.5
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR4520C_L4X1_HighAccuracy(XR4520C_L4X1):
    gpu_inference_streams = 1
    gpu_copy_streams = 1
    gpu_batch_size = 1
    offline_expected_qps = 1.3
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    gpu_batch_size = 1
    offline_expected_qps = 1.3
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxQ)
class XR5610_L4x1_HighAccuracy(XR5610_L4x1_MaxQ):
    pass


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 1
    offline_expected_qps = 1.3
    slice_overlap_patch_kernel_cg_impl = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XR7620_L4x1_HighAccuracy(XR7620_L4x1):
    pass


