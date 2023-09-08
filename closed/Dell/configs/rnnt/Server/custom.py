# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIe_80GBx4
    gpu_batch_size = 2040
    server_target_qps = 49599
    gpu_copy_streams = 1
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 2021
    audio_batch_size: int = 512
    audio_buffer_num_lines: int = 8192
    server_target_qps: int = 66015
    use_graphs: bool = True # MLPINF-1773
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    gpu_batch_size = 2048
    server_target_qps = 34042
    gpu_copy_streams = 8
    gpu_inference_streams=1
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 2250
    gpu_copy_streams: int = 7
    gpu_inference_streams: int = 2
    audio_batch_size: int = 512
    audio_buffer_num_lines: int = 8192
    server_target_qps: int = 68000
    use_graphs: bool = True # MLPINF-1773
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    gpu_batch_size = 2041
    server_target_qps = 36500
    gpu_copy_streams = 1
    gpu_inference_streams = 1


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    gpu_batch_size = 2117
    gpu_inference_streams = 2
    gpu_copy_streams = 5
    server_target_qps = 96204
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    gpu_batch_size = 1994
    audio_buffer_num_lines = 8192
    audio_batch_size = 512
    use_graphs = True  # MLPINF-1773
    server_target_qps = 90000
    gpu_copy_streams = 6
    gpu_inference_streams = 2
    dali_pipeline_depth = 4
    start_from_device = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_batch_size = 2829
    server_target_qps = 178000
    audio_batch_size = 512
    audio_buffer_num_lines = 8192
    use_graphs = True  # MLPINF-1773
    gpu_inference_streams = 2
    gpu_copy_streams = 2
    num_warmups = 20480
    nobatch_sorting = True
    audio_fp16_input = True
    dali_batches_issue_ahead = 0
    dali_pipeline_depth = 2
    workspace_size = 60000000000


