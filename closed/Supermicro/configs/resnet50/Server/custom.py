from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    numa_config = "0:42-55,154-167&1:28-41,140-153&2:0-13,112-125&3:14-27,126-139&4:98-111,210-223&5:84-97,196-209&6:56-69,168-181&7:70-83,182-195"
    server_target_qps = 73736 * 8

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    numa_config = "0,3:0-23,96-119&1,2:24-47,120-143&4,7:48-71,144-167&5,6:72-95,168-191"
    server_target_qps = 74173 * 8

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_521GE_TNRT_H100_PCIe_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.H100x8
    use_deque_limit = True
    deque_timeout_usec = 2000
    gpu_batch_size = 128
    gpu_copy_streams = 4
    gpu_inference_streams = 2
    server_target_qps = 46000 * 8
    use_cuda_thread_per_device = True
    use_batcher_thread_per_device = True
    use_graphs = True
    numa_config = "0,1,2,3:0-31,64-95&4,5,6,7:32-63,96-127"
