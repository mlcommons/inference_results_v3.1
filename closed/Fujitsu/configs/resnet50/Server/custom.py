# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class PRIMERGY_CDI_V1(ServerGPUBaseConfig):
    system = KnownSystem.PRIMERGY_CDI_V1
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 64
    gpu_copy_streams = 9
    gpu_inference_streams = 2
    server_target_qps = 130470
    use_cuda_thread_per_device = True
    use_graphs = True





