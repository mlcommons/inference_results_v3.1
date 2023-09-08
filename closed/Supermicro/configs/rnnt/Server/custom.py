from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    numa_config = "0:42-55,154-167&1:28-41,140-153&2:0-13,112-125&3:14-27,126-139&4:98-111,210-223&5:84-97,196-209&6:56-69,168-181&7:70-83,182-195"
    server_target_qps = 21500 * 8 

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    numa_config = "0,1,2,3:0-47,96-143&4,5,6,7:48-95,144-191"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_521GE_TNRT_H100_PCIe_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.H100x8
    gpu_batch_size = 2048
    server_target_qps = 12500 * 8
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773

