# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8

    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 4       
    gpu_inference_streams = 1          
    server_target_qps = 282000             
    use_cuda_thread_per_device = True
    use_graphs = True





