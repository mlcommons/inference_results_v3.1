# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size = 13
    gpu_copy_streams = 4
    gpu_inference_streams = 1
    offline_expected_qps = 2872
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 1
    gpu_copy_streams: int = 2
    gpu_inference_streams: int = 2
    offline_expected_qps: float = 5300
    run_infer_on_copy_streams: bool = False
    workspace_size: int = 60000000000
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 2200
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 1
    gpu_copy_streams: int = 2
    gpu_inference_streams: int = 2
    offline_expected_qps: float = 5300
    run_infer_on_copy_streams: bool = False
    workspace_size: int = 60000000000
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    gpu_batch_size = 16
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1900
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4_Triton(R760XA_L40X4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 36
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 1700*4.5
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    gpu_batch_size = 18
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 6950
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 14100
    gpu_batch_size = 48
    gpu_copy_streams = 2
    gpu_inference_streams = 6
    run_infer_on_copy_streams = False
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_L4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_L4x1
    gpu_batch_size = 2
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 250
    run_infer_on_copy_streams = False
    workspace_size = 20000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    gpu_batch_size = 2
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 220
    run_infer_on_copy_streams = False
    workspace_size = 20000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 2
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 250 #170
    run_infer_on_copy_streams = False
    workspace_size = 20000000000
