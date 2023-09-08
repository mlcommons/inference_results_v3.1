from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class AS_8125GS_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.AS_8125GS_TNHR_H100_SXM_80GBx8
    numa_config = "0:42-55,154-167&1:28-41,140-153&2:0-13,112-125&3:14-27,126-139&4:98-111,210-223&5:84-97,196-209&6:56-69,168-181&7:70-83,182-195"

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_821GE_TNHR_H100_SXM_80GBX8(H100_SXM_80GBx8):
    system = KnownSystem.SYS_821GE_TNHR_H100_SXM_80GBx8

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class SYS_521GE_TNRT_H100_PCIe_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.H100x8
    gpu_batch_size = 2048
    use_graphs = True  # MLPINF-1773
    offline_expected_qps = 15000 * 8
    disable_encoder_plugin = False
