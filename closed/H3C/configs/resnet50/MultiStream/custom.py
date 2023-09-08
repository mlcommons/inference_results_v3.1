# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class R5300G6_L40X1(MultiStreamGPUBaseConfig):
    system = KnownSystem.R5300G6_L40x1
    gpu_copy_streams = 2  
    gpu_inference_streams = 2   
    multi_stream_expected_latency_ns = 830000    
