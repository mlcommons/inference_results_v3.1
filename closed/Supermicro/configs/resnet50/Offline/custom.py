from . import *


@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    numa_config = "0:42-55,154-167&1:28-41,140-153&2:0-13,112-125&3:14-27,126-139&4:98-111,210-223&5:84-97,196-209&6:56-69,168-181&7:70-83,182-195"

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8
    numa_config = "0,3:0-23,96-119&1,2:24-47,120-143&4,7:48-71,144-167&5,6:72-95,168-191"

@ConfigRegistry.register(HarnessType.LWIS, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_521GE_TNRT_H100_PCIe_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.H100x8
    gpu_batch_size = 2048
    offline_expected_qps = 450000
    numa_config = "0,1,2,3:0-31,64-95&4,5,6,7:32-63,96-127"
