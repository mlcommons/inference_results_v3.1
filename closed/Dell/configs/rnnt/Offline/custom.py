# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size = 2048
    offline_expected_qps = 55000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 2048
    disable_encoder_plugin: bool = False
    offline_expected_qps: float = 68000
    use_graphs: bool = True # MLPINF-1773
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    gpu_batch_size = 2048
    use_graphs = True  # MLPINF-1773
    offline_expected_qps = 35000
    disable_encoder_plugin = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 2048
    disable_encoder_plugin: bool = False
    offline_expected_qps: float = 88264
    use_graphs: bool = True # MLPINF-1773
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(OfflineGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    gpu_batch_size = 2048
    offline_expected_qps = 41700


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(OfflineGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    start_from_device = True
    gpu_batch_size = 2284
    offline_expected_qps = 23000*5
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(OfflineGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    gpu_batch_size = 2048
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False
    offline_expected_qps = 94000
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(OfflineGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    offline_expected_qps = 192150
    gpu_batch_size = 2383
    start_from_device = False
    audio_batch_size = 1024
    audio_buffer_num_lines = 8192
    use_graphs = True
    disable_encoder_plugin = False
    workspace_size = 80000000000
    num_warmups = 512
    dali_batches_issue_ahead = 4
    dali_pipeline_depth = 4
    gpu_inference_streams = 2
    gpu_copy_streams = 5
    nobatch_sorting = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR4520C_L4X1(OfflineGPUBaseConfig):
    system = KnownSystem.XR4520c_L4x1
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxQ)
class XR5610_L4x1_MaxQ(OfflineGPUBaseConfig):
    system = KnownSystem.XR5610_L4x1
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XR7620_L4x1(OfflineGPUBaseConfig):
    system = KnownSystem.XR7620_L4x1
    gpu_batch_size = 512
    offline_expected_qps = 3900
    audio_batch_size = 64
    audio_buffer_num_lines = 1024
