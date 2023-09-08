# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 36911
    gpu_batch_size = 13
    gpu_inference_streams = 1
    server_target_qps = 2700
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 13
    deque_timeout_usec: int = 49953
    gpu_copy_streams: int = 2
    gpu_inference_streams: int = 2
    server_target_qps: int = 4276
    use_deque_limit: bool = True
    workspace_size: int = 60000000000
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    gpu_batch_size: int = 9
    deque_timeout_usec: int = 32672
    gpu_copy_streams: int = 7
    gpu_inference_streams: int = 2
    server_target_qps: int = 2250
    use_deque_limit: bool = True
    workspace_size: int = 60000000000
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 9
    deque_timeout_usec: int = 32672
    gpu_copy_streams: int = 7
    gpu_inference_streams: int = 2
    server_target_qps: int = 4350
    use_deque_limit: bool = True
    workspace_size: int = 60000000000
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    gpu_copy_streams = 4
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 8
    gpu_inference_streams = 2
    server_target_qps = 1900
    workspace_size = 70000000000


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4_Triton(R760XA_L40X4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    start_from_device = True
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 35166
    gpu_batch_size = 15
    gpu_inference_streams = 2
    server_target_qps = 6753
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    start_from_device = True
    gpu_copy_streams = 2
    use_deque_limit = True
    deque_timeout_usec = 35166
    gpu_batch_size = 15
    gpu_inference_streams = 2
    server_target_qps = 6653
    workspace_size = 60000000000


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    start_from_device = True
    gpu_copy_streams = 3
    use_deque_limit = True
    deque_timeout_usec = 31592
    gpu_batch_size = 13
    gpu_inference_streams = 8
   workspace_size = 60000000000
    server_target_qps = 12480

