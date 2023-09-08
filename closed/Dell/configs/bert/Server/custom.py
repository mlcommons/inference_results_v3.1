# Generated file by scripts/custom_systems/add_custom_system.py
# Contains configs for all custom systems in code/common/systems/custom_list.py

from . import *


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_A100_PCIE_80GBx4
    gpu_batch_size: int = 64
    active_sms: int = 60
    gpu_copy_streams: int = 4
    gpu_inference_streams: int = 2
    graphs_max_seqlen: int = 200
    server_num_issue_query_threads: int = 1
    server_target_qps: int = 11700
    soft_drop: float = 1.0


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750XA_A100_PCIE_80GBX4_HighAccuracy(R750XA_A100_PCIE_80GBX4):
    precision = "fp16"
    server_target_qps = R750XA_A100_PCIE_80GBX4.server_target_qps / 2


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R750xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 221
    enable_interleaved: bool = False
    graphs_max_seqlen: int = 200
    server_target_qps: int = 15100
    server_num_issue_query_threads: int = 1
    soft_drop: float = 1.0
    use_graphs: bool = False
    use_small_tile_gemm_plugin: bool = False
    workspace_size: int = 7516192768
    start_from_device: bool = True


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R750xa_H100_PCIe_80GBx4_HighAccuracy(R750xa_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 14960
    soft_drop = 1.0

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx2
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    graphs_max_seqlen = 200
    gpu_batch_size = 152
    server_target_qps = 9144
    server_num_issue_query_threads = 1
    workspace_size = 7516192768
    soft_drop = 0.99


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx2_HighAccuracy(R760xa_H100_PCIe_80GBx2):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 8250
    soft_drop = 1.0
    gpu_batch_size= 132


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_H100_PCIe_80GBx4
    gpu_batch_size: int = 276
    enable_interleaved: bool = False
    graphs_max_seqlen: int = 200
    server_target_qps: int = 17882
    server_num_issue_query_threads: int = 1
    soft_drop: float = 0.99
    use_graphs: bool = False
    use_small_tile_gemm_plugin: bool = False
    workspace_size: int = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760xa_H100_PCIe_80GBx4_HighAccuracy(R760xa_H100_PCIe_80GBx4):
    precision = "fp16"
    use_fp8 = True
    server_target_qps = 16300
    soft_drop = 1.0
    gpu_batch_size: int = 254


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class R760XA_L40X4(ServerGPUBaseConfig):
    system = KnownSystem.R760xa_L40x4
    gpu_batch_size = 95
    gpu_copy_streams = 2
    gpu_inference_streams = 1
    server_target_qps = 8617
    soft_drop = 1.0
    use_small_tile_gemm_plugin = True
    workspace_size = 7000000000000

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class R760XA_L40X4_HighAccuracy(R760XA_L40X4):
    precision = "fp16"
    server_target_qps = R760XA_L40X4.server_target_qps / 2
    workspace_size = 7000000000000


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4(ServerGPUBaseConfig):
    system = KnownSystem.XE8640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    gpu_batch_size = 201
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 29080
    server_num_issue_query_threads = 1
    workspace_size = 7516192768

@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE8640_H100_SXM_80GBx4_HighAccuracy(XE8640_H100_SXM_80GBx4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 178
    server_target_qps = 25500


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8(ServerGPUBaseConfig):
    system = KnownSystem.XE9680_H100_SXM_80GBx8
    gpu_inference_streams = 2
    gpu_copy_streams = 4
    server_target_qps = 57310
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    gpu_batch_size = 171
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9680_H100_SXM_80GBX8_HighAccuracy(XE9680_H100_SXM_80GBX8):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 292
    server_target_qps = 51200
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    gpu_copy_streams = 6
    gpu_inference_streams = 1
    server_num_issue_query_threads = 1
   workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4(ServerGPUBaseConfig):
    system = KnownSystem.XE9640_H100_SXM_80GBx4
    use_small_tile_gemm_plugin = False
    enable_interleaved = False
    use_graphs = False
    gpu_batch_size = 128
    gpu_copy_streams = 2
    gpu_inference_streams = 2
    server_target_qps = 28000
    server_num_issue_query_threads = 1
    workspace_size = 7516192768


@ConfigRegistry.register(HarnessType.Custom, AccuracyTarget.k_99_9, PowerSetting.MaxP)
class XE9640_H100_SXM_80GBX4_HighAccuracy(XE9640_H100_SXM_80GBX4):
    precision = "fp16"
    use_fp8 = True
    use_graphs = False
    gpu_batch_size = 136
    server_target_qps = 24800

