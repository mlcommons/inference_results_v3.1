# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(ServerGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8
    gpu_copy_streams = 1     
    use_deque_limit = True
    deque_timeout_usec = 30000
    gpu_batch_size = 4           
    gpu_inference_streams = 2    
    server_target_qps = 4900                 
    workspace_size = 70000000000


