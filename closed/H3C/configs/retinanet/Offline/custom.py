# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X8(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x8
    gpu_batch_size = 2              
    gpu_copy_streams = 2           
    gpu_inference_streams = 1               
    offline_expected_qps = 5200         
    run_infer_on_copy_streams = False
    workspace_size = 60000000000

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X1(OfflineGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x1
    gpu_batch_size = 2 
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    offline_expected_qps = 600   
    run_infer_on_copy_streams = False
    workspace_size = 60000000000
