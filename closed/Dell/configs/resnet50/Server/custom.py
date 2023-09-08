# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 276
    gpu_copy_streams = 5
    gpu_inference_streams = 1
    server_target_qps = 147233
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 105
    deque_timeout_usec: int = 2505
    gpu_copy_streams: int = 5
    gpu_inference_streams: int = 2
    server_target_qps: int = 196538
    use_batcher_thread_per_device: bool = True
    use_cuda_thread_per_device: bool = True
    use_deque_limit: bool = True
    use_graphs: bool = True
    start_from_device: int = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    gpu_batch_size: int = 124
    deque_timeout_usec: int = 2954
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    server_target_qps: int = 103000
    use_batcher_thread_per_device: bool = True
    use_cuda_thread_per_device: bool = True
    use_deque_limit: bool = True
    use_graphs: bool = True
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 124
    deque_timeout_usec: int = 2954
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    server_target_qps: int = 207781
    use_batcher_thread_per_device: bool = True
    use_cuda_thread_per_device: bool = True
    use_deque_limit: bool = True
    use_graphs: bool = True
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 87
    gpu_copy_streams = 12
    gpu_inference_streams = 4
    server_target_qps = 130000
    use_cuda_thread_per_device = True
    use_graphs = True


@ConfigRegistry.register(HarnessType.Triton, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4_Triton(R760XA_L40X4):
    use_triton = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_deque_limit = True
    deque_timeout_usec = 3548
    gpu_batch_size = 273
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 310274
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    gpu_batch_size = 273 
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    deque_timeout_usec = 3548 
    server_target_qps = 305020


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    use_deque_limit = True
    deque_timeout_usec = 4182
    use_cuda_thread_per_device = True
    use_graphs = True
    start_from_device = True
    gpu_batch_size = 391
    gpu_copy_streams = 5
    gpu_inference_streams = 1
    server_target_qps = 620658

