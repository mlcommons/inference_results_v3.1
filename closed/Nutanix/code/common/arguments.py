# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

from code.common.constants import Scenario


#: Global args for all scripts
arguments_dict = {
    # Common arguments
    "gpu_batch_size": {
        "help": "GPU batch size to use for the engine.",
        "type": int,
    },
    "gpu_res2res3_loop_count": {
        "help": "Res3 subnetwork loop count. Only take effect when gpu_res2res3_loop_count > 1. Default: 1.",
        "type": int,
    },
    "dla_batch_size": {
        "help": "DLA batch size to use for the engine.",
        "type": int,
    },
    "dla_loop_count": {
        "help": "How many times to loop over the DLA subnetwork. Orin ResNet Offline argument.",
        "type": int,
        "default": 1
    },

    "batch_size": {
        "help": "Batch size to use for the engine.",
        "type": int,
    },
    "verbose": {
        "help": "Use verbose output",
        "action": "store_true",
    },
    "verbose_nvtx": {
        "help": "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.",
        "action": "store_true",
    },
    "no_child_process": {
        "help": "Do not generate engines on child process. Do it on current process instead.",
        "action": "store_true"
    },
    "workspace_size": {
        "help": "The maximum size of temporary workspace that any layer in the network can use in TRT",
        "type": int,
        "default": None
    },

    # Power measurements
    "power": {
        "help": "Select if you would like to measure power",
        "action": "store_true"
    },
    "power_limit": {
        "help": "Set power upper limit to the specified value.",
        "type": int,
        "default": None
    },
    # Dataset location
    "data_dir": {
        "help": "Directory containing unprocessed datasets",
        "default": os.environ.get("DATA_DIR", "build/data"),
    },
    "preprocessed_data_dir": {
        "help": "Directory containing preprocessed datasets",
        "default": os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
    },

    # Arguments related to precision
    "precision": {
        "help": "Precision. Default: int8",
        "choices": ["fp32", "fp16", "int8", None],
        # None needs to be set as default since passthrough arguments will
        # contain a default value and override configs. Specifying None as the
        # default will cause it to not be inserted into passthrough / override
        # arguments.
        "default": None,
    },
    "input_dtype": {
        "help": "Input datatype. Choices: fp32, int8.",
        "choices": ["fp32", "fp16", "int8", None],
        "default": None
    },
    "input_format": {
        "help": "Input format (layout). Choices: linear, chw4, dhwc8, cdhw32",
        "choices": ["linear", "chw4", "dhwc8", "cdhw32", None],
        "default": None
    },
    "audio_fp16_input": {
        "help": "Is input format for raw audio file in fp16?. Choices: true, false",
        "action": "store_true",
    },

    # Arguments related to quantization calibration
    "force_calibration": {
        "help": "Run quantization calibration even if the cache exists. (Only used for quantized models)",
        "action": "store_true",
    },
    "calib_batch_size": {
        "help": "Batch size for calibration.",
        "type": int
    },
    "calib_max_batches": {
        "help": "Number of batches to run for calibration.",
        "type": int
    },
    "cache_file": {
        "help": "Path to calibration cache.",
        "default": None,
    },
    "calib_data_map": {
        "help": "Path to the data map of the calibration set.",
        "default": None,
    },

    # Benchmark configuration arguments
    "scenario": {
        "help": "Name for the scenario. Used to generate engine name.",
    },
    "dla_core": {
        "help": "DLA core to use. Do not set if not using DLA",
        "default": None,
    },
    "model_path": {
        "help": "Path to the model (weights) file.",
    },
    "active_sms": {
        "help": "Control the percentage of active SMs while generating engines.",
        "type": int
    },

    # Profiler selection
    "profile": {
        "help": "[INTERNAL ONLY] Select if you would like to profile -- select among nsys, nvprof and ncu",
        "type": str
    },

    # Harness configuration arguments
    "log_dir": {
        "help": "Directory for all output logs.",
        "default": os.environ.get("LOG_DIR", "build/logs/default"),
    },
    "use_graphs": {
        "help": "Enable CUDA graphs",
        "action": "store_true",
    },

    "use_fp8": {
        "help": "Use fp8 for BERT",
        "action": "store_true",
    },

    # 3D-UNet-KiTS19 Harness
    "unet3d_sw_gaussian_patch_path": {
        "help": "Path to the numpy file holding preconditioned Gaussian patches used in 3D-UNet-KiTS19.",
        "default": os.environ.get("PREPROCESSED_DATA_DIR", "build/preprocessed_data") + "/KiTS19/etc/gaussian_patches.npy",
    },

    "slice_overlap_patch_kernel_cg_impl": {
        "help": "Use 3D-UNet patch kernel implemented using cooperative-group; if false, kernel impl using CPU implicit sync is used",
        "type": bool,
    },

    "nopipelined_execution": {
        "help": "Disable pipelined execution",
        "action": "store_true",
    },

    "nobatch_sorting": {
        "help": "Disable batch sorting by sequence length",
        "action": "store_true",
    },

    "noenable_audio_processing": {
        "help": "Disable DALI preprocessing and fallback to preprocessed npy files",
        "action": "store_true",
    },

    "nouse_copy_kernel": {
        "help": "Disable using DALI's scatter gather kernel instead of using cudamemcpyAsync",
        "action": "store_true",
    },

    "num_warmups": {
        "help": "Number of samples to warmup on. A value of -1 runs two full batches for each stream (2*batch_size*streams_per_gpu*NUM_GPUS), 0 turns off warmups.",
        "type": int
    },

    "max_seq_length": {
        "help": "Max sequence length for audio",
        "type": int
    },

    "audio_batch_size": {
        "help": "Batch size for DALI's processing",
        "type": int
    },

    "audio_buffer_num_lines": {
        "help": "Number of audio samples in flight for DALI's processing",
        "type": int
    },

    "dali_batches_issue_ahead": {
        "help": "Number of batches for which cudamemcpy is issued ahead of DALI compute",
        "type": int
    },

    "dali_pipeline_depth": {
        "help": "Depth of sub-batch processing in DALI pipeline",
        "type": int
    },

    "disable_encoder_plugin": {
        "help": "Disable the INT8 Encoder TRT plugin and use the fallback TRT API for Encoder",
        "action": "store_true",
    },

    # LWIS settings
    "devices": {
        "help": "Comma-separated list of numbered devices",
    },
    "map_path": {
        "help": "Path to map file for samples",
    },
    "tensor_path": {
        "help": "Path to preprocessed samples in .npy format",
    },
    "performance_sample_count": {
        "help": "Number of samples to load in performance set.  0=use default",
        "type": int,
    },
    "performance_sample_count_override": {
        "help": "If set, this overrides performance_sample_count.  0=don't override",
        "type": int,
    },
    "gpu_copy_streams": {
        "help": "Number of copy streams to use for GPU",
        "type": int,
    },
    "gpu_inference_streams": {
        "help": "Number of inference streams to use for GPU",
        "type": int,
    },
    "dla_copy_streams": {
        "help": "Number of copy streams to use for DLA",
        "type": int,
    },
    "dla_inference_streams": {
        "help": "Number of inference streams to use for DLA",
        "type": int,
    },
    "run_infer_on_copy_streams": {
        "help": "Run inference on copy streams.",
    },
    "warmup_duration": {
        "help": "Minimum duration to perform warmup for",
        "type": float,
    },
    "use_direct_host_access": {
        "help": "Use direct access to host memory for all devices",
    },
    "use_deque_limit": {
        "help": "Use a max number of elements dequed from work queue",
    },
    "deque_timeout_usec": {
        "help": "Timeout in us for deque from work queue.",
        "type": int,
    },
    "use_batcher_thread_per_device": {
        "help": "Enable a separate batcher thread per device",
    },
    "use_cuda_thread_per_device": {
        "help": "Enable a separate cuda thread per device",
    },
    "start_from_device": {
        "help": "Assuming that inputs start from device memory in QSL"
    },
    "end_on_device": {
        "help": "Copy outputs to host in untimed loadgen callback"
    },
    "max_dlas": {
        "help": "Max number of DLAs to use per device",
        "type": int,
    },
    "coalesced_tensor": {
        "help": "Turn on if all the samples are coalesced into one single npy file"
    },
    "assume_contiguous": {
        "help": "Assume that the data in a query is already contiguous"
    },
    "complete_threads": {
        "help": "Number of threads per device for sending responses",
        "type": int,
    },
    "use_same_context": {
        "help": "Use the same TRT context for all copy streams (shape must be static and gpu_inference_streams must be 1).",
    },

    # Shared settings
    "mlperf_conf_path": {
        "help": "Path to mlperf.conf",
    },
    "user_conf_path": {
        "help": "Path to user.conf",
    },

    # Loadgen settings
    "test_mode": {
        "help": "Testing mode for Loadgen",
        "choices": ["SubmissionRun", "AccuracyOnly", "PerformanceOnly", "FindPeakPerformance"],
    },
    "min_duration": {
        "help": "Minimum test duration",
        "type": int,
    },
    "max_duration": {
        "help": "Maximum test duration",
        "type": int,
    },
    "min_query_count": {
        "help": "Minimum number of queries in test",
        "type": int,
    },
    "max_query_count": {
        "help": "Maximum number of queries in test",
        "type": int,
    },
    "qsl_rng_seed": {
        "help": "Seed for RNG that specifies which QSL samples are chosen for performance set and the order in which samples are processed in AccuracyOnly mode",
        "type": int,
    },
    "sample_index_rng_seed": {
        "help": "Seed for RNG that specifies order in which samples from performance set are included in queries",
        "type": int,
    },
    "fast": {
        "help": "If set, will set min_duration to 1 minute (60000ms). For Offline and Server, min_query_count is set to 1.",
        "action": "store_true",
    },

    # Loadgen logging settings
    "logfile_suffix": {
        "help": "Specify the filename suffix for the LoadGen log files",
    },
    "logfile_prefix_with_datetime": {
        "help": "Prefix filenames for LoadGen log files",
        "action": "store_true",
    },
    "log_copy_detail_to_stdout": {
        "help": "Copy LoadGen detailed logging to stdout",
        "action": "store_true",
    },
    "disable_log_copy_summary_to_stdout": {
        "help": "Disable copy LoadGen summary logging to stdout",
        "action": "store_true",
    },
    "log_mode": {
        "help": "Logging mode for Loadgen",
        "choices": ["AsyncPoll", "EndOfTestOnly", "Synchronous"],
    },
    "log_mode_async_poll_interval_ms": {
        "help": "Specify the poll interval for asynchrounous logging",
        "type": int,
    },
    "log_enable_trace": {
        "help": "Enable trace logging",
    },

    # Triton args
    "use_triton": {
        "help": "Use Triton harness",
        "action": "store_true",
    },
    "preferred_batch_size": {
        "help": "Preferred batch sizes"
    },
    "max_queue_delay_usec": {
        "help": "Set max queuing delay in usec.",
        "type": int
    },
    "instance_group_count": {
        "help": "Set number of instance groups on each GPU.",
        "type": int
    },
    "request_timeout_usec": {
        "help": "Set the timeout for every request in usec.",
        "type": int
    },
    "buffer_manager_thread_count": {
        "help": "The number of threads used to accelerate copies and other operations required to manage input and output tensor contents.",
        "type": int,
        "default": 0
    },
    "gather_kernel_buffer_threshold": {
        "help": "Set the threshold number of buffers for triton to use gather kernel to gather input data. 0 disables the gather kernel",
        "type": int,
        "default": 0
    },
    "batch_triton_requests": {
        "help": "Send a batch of query samples to triton instead of single query at a time",
        "action": "store_true"
    },
    "gather_kernel_buffer_threshold": {
        "help": "Buffer size for gather kernel",
        "type": int,
    },
    "output_pinned_memory": {
        "help": "Use pinned memory when data transfer for output is between device mem and non-pinned sys mem",
        "action": "store_true"
    },
    "num_concurrent_batchers": {
        "help": "Number of threads that will service issue query requests to form batches",
        "type": int,
    },
    "num_concurrent_issuers": {
        "help": "Number of threads that will issue requests to triton",
        "type": int,
    },
    "dla_num_batchers": {
        "help": "Number of threads that will service issue query requests to form dla batches",
        "type": int,
    },
    "dla_num_issuers": {
        "help": "Number of threads that will issue requests to the dla model in triton",
        "type": int,
    },
    "inferentia_compiled_model_batch_size": {
        "help": "Batch size of the compiled inferentia model. (min 1- max 6)",
        "type": int,
    },
    "inferentia_compiled_model_framework": {
        "help": "The framework of the compiled inferentia model.",
        "choices": ["pytorch", "tf"],
    },
    "inferentia_threads_per_core": {
        "help": "Number of threads issuing request for a single inferentia core.",
        "type": int,
    },
    "inferentia_neuron_core_count": {
        "help": "Number of neuron cores in the device (Determined by the instance type)",
        "type": int,
    },
    "inferentia_request_batch_size": {
        "help": "Batch size of the request generated by the harness.",
        "type": int,
    },
    # Server harness arguments
    "server_target_qps": {
        "help": "Target QPS for server scenario.",
        "type": int,
    },
    "server_target_qps_adj_factor": {
        "help": "Adjustment Factor for Target QPS for server scenario.",
        "type": float,
        "default": 1.0,
    },
    "server_target_latency_ns": {
        "help": "Desired latency constraint for server scenario",
        "type": int,
    },
    "server_target_latency_percentile": {
        "help": "Desired latency percentile constraint for server scenario",
        "type": float,
    },
    "server_coalesce_queries": {
        "help": "Enable coalescing outstanding queries in the server scenario",
        "action": "store_true",
    },
    "server_num_issue_query_threads": {
        "help": "Number of IssueQuery threads to use for Loadgen in Server scenario",
        "type": int,
    },
    "schedule_rng_seed": {
        "help": "Seed for RNG that affects the poisson arrival process in server scenario",
        "type": int,
    },
    "accuracy_log_rng_seed": {
        "help": "Affects which samples have their query returns logged to the accuracy log in performance mode.",
        "type": int,
    },

    # Single stream harness arguments
    "single_stream_expected_latency_ns": {
        "help": "Inverse of desired target QPS",
        "type": int,
    },
    "single_stream_target_latency_percentile": {
        "help": "Desired latency percentile for single stream scenario",
        "type": float,
    },

    # Multi stream harness arguments
    "multi_stream_expected_latency_ns": {
        "help": "Expected latency to process a query with multiple Samples, in nanoseconds",
        "type": int,
    },
    "multi_stream_target_latency_percentile": {
        "help": "Desired latency percentile to report as a performance metric, for multi stream scenario",
        "type": float,
    },
    "multi_stream_samples_per_query": {
        "help": "Number of samples bundled together as a single query",
        "type": int,
    },

    # Offline harness arguments
    "offline_expected_qps": {
        "help": "Target samples per second rate for the SUT",
        "type": float,
    },

    # Args used by code.main
    "action": {
        "help": "generate_engines / run_harness / calibrate / generate_conf_files",
        "choices": ["generate_engines", "run_harness", "calibrate", "generate_conf_files", "run_audit_harness", "run_cpu_audit_harness", "run_audit_verification", "run_cpu_audit_verification"],
    },
    "benchmarks": {
        "help": "Specify the benchmark(s) with a comma-separated list. " +
        "Default: run all benchmarks.",
        "default": None,
    },
    "configs": {
        "help": "Specify the config files with a comma-separated list. " +
        "Wild card (*) is also allowed. If \"\", detect platform and attempt to load configs. " +
        "Default: \"\"",
        "default": "",
    },
    "config_ver": {
        "help": "Config version to run. Uses 'default' if not set.",
        "default": "default",
    },
    "openvino_version": {
        "help": "OpenVINO version to run. Uses 'f2f281e6' if not set.",
        "default": "f2f281e6",
    },
    "scenarios": {
        "help": "Specify the scenarios with a comma-separated list. " +
        "Choices:[\"Server\", \"Offline\", \"SingleStream\", \"MultiStream\"] " +
        "Default: \"*\"",
        "default": None,
    },
    "no_gpu": {
        "help": "Do not perform action with GPU parameters (run on DLA only).",
        "action": "store_true",
    },
    "gpu_only": {
        "help": "Only perform action with GPU parameters (do not run DLA).",
        "action": "store_true",
    },
    "audit_test": {
        "help": "Defines audit test to run.",
        "choices": ["TEST01", "TEST04-A", "TEST04-B", "TEST05"],
    },
    "system_name": {
        "help": "Override the system name to run under",
        "type": str
    },

    # Args used for engine runners
    "engine_file": {
        "help": "File to load engine from",
    },
    "num_samples": {
        "help": "Number of samples to use for accuracy runner",
        "type": int,
    },

    # DLRM harness
    "sample_partition_path": {
        "help": "Path to sample partition file in npy format.",
    },
    "num_staging_threads": {
        "help": "Number of staging threads in DLRM BatchMaker",
        "type": int,
    },
    "num_staging_batches": {
        "help": "Number of staging batches in DLRM BatchMaker",
        "type": int,
    },
    "max_pairs_per_staging_thread": {
        "help": "Maximum pairs to copy in one BatchMaker staging thread",
        "type": int,
    },
    "gpu_num_bundles": {
        "help": "Number of event+buffer bundles per GPU (default: 2)",
        "type": int,
        "default": 2,
    },
    "check_contiguity": {
        "help": "Check if inputs are already contiguous in QSL to avoid copying",
        "action": "store_true",
    },
    "use_jemalloc": {
        "help": "Use libjemalloc.so.2 as the malloc(3) implementation",
        "action": "store_true",
    },
    "compress_categorical_inputs": {
        "help": "Compress categorical features into 16 * sizeof(int32) bytes (uncompressed is 26 * sizeof(int32) bytes)",
        "action": "store_true",
    },


    "use_spin_wait": {
        "help": "Use spin waiting for LWIS. Recommended for single stream",
        "action": "store_true",
    },
    # This `numa_config` parameter has a convention of `[Node 0 config]&[Node 1 config]&[Node 2 config]&...`.
    # And each `[Node n config]` can be configured as `[GPU ID(s)]:[CPU ID(s)]`. Each `ID(s)` can be single digit,
    # comma-separated digits, or digits with dash.
    # EX) "numa_config": "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"
    # For example `3:0-15,64-79` means GPU 3 and CPU 0,1,2,...15,64,65,66,...79 are in the same node,
    # and since this key is the very first of elements connected with &, they are in node 0.
    "numa_config": {
        "help": "NUMA settings: GPU and CPU cores for each NUMA node. For example: 0,2:0-63&1,3:64-127 means " +
        "NUMA node 0 has GPU 0 and GPU 2 and CPU 0-63, and NUMA node 1 has GPU 1 and GPU3 and CPU 64-127",
    },
}

# ================== Argument groups ================== #

# Engine generation
PRECISION_ARGS = [
    "audio_fp16_input",
    "input_dtype",
    "input_format",
    "precision",
]
CALIBRATION_ARGS = [
    "cache_file",
    "calib_batch_size",
    "calib_data_map",
    "calib_max_batches",
    "force_calibration",
    "model_path",
    "verbose",
]
GENERATE_ENGINE_ARGS = [
    "active_sms",
    "dla_batch_size",
    "dla_loop_count",
    "dla_core",
    "gpu_batch_size",
    "gpu_res2res3_loop_count",
    "gpu_copy_streams",
    "gpu_inference_streams",
    "max_seq_length",
    "power_limit",
    "verbose_nvtx",
    "workspace_size",
    "disable_encoder_plugin",
    "compress_categorical_inputs",
] + PRECISION_ARGS + CALIBRATION_ARGS

# Harness framework arguments
LOADGEN_ARGS = [
    "accuracy_log_rng_seed",
    "disable_log_copy_summary_to_stdout",
    "log_copy_detail_to_stdout",
    "log_enable_trace",
    "log_mode",
    "log_mode_async_poll_interval_ms",
    "logfile_prefix_with_datetime",
    "logfile_suffix",
    "max_duration",
    "max_query_count",
    "min_duration",
    "min_query_count",
    "qsl_rng_seed",
    "sample_index_rng_seed",
    "schedule_rng_seed",
    "server_target_latency_ns",
    "server_target_latency_percentile",
    "server_target_qps",
    "server_target_qps_adj_factor",
    "server_num_issue_query_threads",
    "server_coalesce_queries",
    "single_stream_target_latency_percentile",
    "multi_stream_target_latency_percentile",
    "multi_stream_samples_per_query",
    "test_mode",
    "fast",
]
LWIS_ARGS = [
    "assume_contiguous",
    "coalesced_tensor",
    "complete_threads",
    "deque_timeout_usec",
    "devices",
    "dla_batch_size",
    "dla_loop_count",
    "dla_copy_streams",
    "dla_inference_streams",
    "gpu_copy_streams",
    "gpu_inference_streams",
    "max_dlas",
    "run_infer_on_copy_streams",
    "start_from_device",
    "end_on_device",
    "use_batcher_thread_per_device",
    "use_cuda_thread_per_device",
    "use_deque_limit",
    "use_direct_host_access",
    "use_spin_wait",
    "warmup_duration",
]
TRITON_ARGS = [
    "instance_group_count",
    "max_queue_delay_usec",
    "preferred_batch_size",
    "request_timeout_usec",
    "buffer_manager_thread_count",
    "batch_triton_requests",
    "gather_kernel_buffer_threshold",
    "output_pinned_memory",
    "num_concurrent_batchers",
    "num_concurrent_issuers",
    "dla_num_batchers",
    "dla_num_issuers",
    "inferentia_compiled_model_batch_size",
    "inferentia_compiled_model_framework",
    "inferentia_threads_per_core",
    "inferentia_neuron_core_count",
    "inferentia_request_batch_size",
    "openvino_version"
]
SHARED_ARGS = [
    "gpu_batch_size",
    "gpu_res2res3_loop_count",
    "map_path",
    "mlperf_conf_path",
    "performance_sample_count",
    "performance_sample_count_override",
    "tensor_path",
    "use_graphs",
    "user_conf_path",
    "numa_config",
    "use_fp8",
]
OTHER_HARNESS_ARGS = [
    "audio_batch_size",
    "audio_buffer_num_lines",
    "check_contiguity",
    "dali_batches_issue_ahead",
    "dali_pipeline_depth",
    "gpu_num_bundles",
    "log_dir",
    "max_pairs_per_staging_thread",
    "max_seq_length",
    "nobatch_sorting",
    "noenable_audio_processing",
    "nopipelined_execution",
    "nouse_copy_kernel",
    "disable_encoder_plugin",
    "num_staging_batches",
    "num_staging_threads",
    "num_warmups",
    "power_limit",
    "sample_partition_path",
    "use_jemalloc",
    "use_triton",
    "compress_categorical_inputs",
]
HARNESS_ARGS = ["verbose", "scenario"] + PRECISION_ARGS + LOADGEN_ARGS + LWIS_ARGS + TRITON_ARGS + SHARED_ARGS + OTHER_HARNESS_ARGS

# TODO: To reduce risk to v1.0 submission, the following lists will not be added to arguments_dict. There is a planned
# refactor of arguments.py in v1.1 where this will be fixed.
# CPU_HARNESS_ARGS and OPENVINO_ARGS define a list of valid parameters for Triton CPU system configurations
# SMALL_GEMM_KERNEL_ARGS defines a list of valid parameters to do with the SMALL_GEMM_KERNEL
# DLRM_ARGS defines a list of valid parameters for DLRM benchmark
# BERT_ARGS defines a list of valid parameters for BERT benchmark
# RNNT_ARGS defines a list of valid parameters for RNNT benchmark
CPU_HARNESS_ARGS = ["model_name", "ov_parameters", "num_instances", "batch_size", "copy_streams", "infer_streams"]
OPENVINO_ARGS = ["CPU_THROUGHPUT_STREAMS", "CPU_THREADS_NUM", "ENABLE_BATCH_PADDING", "SKIP_OV_DYNAMIC_BATCHSIZE"]
SMALL_GEMM_KERNEL_ARGS = ["use_small_tile_gemm_plugin", "enable_interleaved", "gemm_plugin_fairshare_cache_size"]
DLRM_ARGS = ["embedding_weights_on_gpu_part", "enable_interleaved_top_mlp", "output_padding_granularity"]
BERT_ARGS = ["soft_drop", "graph_specs", "graphs_max_seqlen", "bert_opt_seqlen"]
RNNT_ARGS = ["pipelined_execution"]
RETINANET_ARGS = ["use_nmopt"]

# Scenario dependent arguments. These are prefixed with device: "gpu_", "dla_", "concurrent_"
SCENARIO_METRIC_PREFIXES = ["gpu_", "dla_", "concurrent_"]
OFFLINE_PARAMS = ["offline_expected_qps"]
SINGLE_STREAM_PARAMS = ["single_stream_expected_latency_ns"]
MULTI_STREAM_PARAMS = ["multi_stream_expected_latency_ns"]
SERVER_PARAMS = []

#: Args for code.main
MAIN_ARGS = [
    "action",
    "audit_test",
    "benchmarks",
    "config_ver",
    "configs",
    "gpu_only",
    "no_child_process",
    "no_gpu",
    "power",
    "power_limit",
    "profile",
    "scenarios",
    "system_name",
]

# For accuracy runners
ACCURACY_ARGS = [
    "batch_size",
    "engine_file",
    "num_samples",
    "verbose",
]


def parse_args(whitelist):
    """Parse whitelist args in user input and return parsed args."""

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for flag in whitelist:

        # Check with global arg list
        if flag not in arguments_dict:
            raise IndexError("Unknown flag '{:}'".format(flag))

        parser.add_argument("--{:}".format(flag), **arguments_dict[flag])
    return vars(parser.parse_known_args()[0])


def check_args():
    """Create arg parser with global args and check if it works."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for flag in arguments_dict:
        parser.add_argument("--{:}".format(flag), **arguments_dict[flag])
    parser.parse_args()


def apply_overrides(config, keys):
    """Apply overrides from user input on config file data."""
    # Make a copy so we don't modify original dict
    config = dict(config)
    override_args = parse_args(keys)
    for key in override_args:
        # Unset values (None) and unset store_true values (False) are both false-y
        if override_args[key]:
            config[key] = override_args[key]
    return config

##
# @brief Create an argument list based on scenario and benchmark name


def getScenarioMetricArgs(scenario, prefixes=("",)):
    """
    Returns a list of metric arguments specific to a scenario, prepended by all individual prefixes specified.

    i.e. for scenario key "foo" and prefixes ["a_", "b_"], this method would return ["a_foo", "b_foo"].

    By default, prefixes is SCENARIO_METRIC_PREFIXES
    """
    arglist = None
    if Scenario.SingleStream == scenario:
        arglist = SINGLE_STREAM_PARAMS
    elif Scenario.MultiStream == scenario:
        arglist = MULTI_STREAM_PARAMS
    elif Scenario.Offline == scenario:
        arglist = OFFLINE_PARAMS
    elif Scenario.Server == scenario:
        arglist = SERVER_PARAMS
    else:
        raise RuntimeError("Unknown Scenario \"{}\"".format(scenario))

    # Apply prefixes
    return [
        prefix + arg
        for prefix in prefixes
        for arg in arglist
    ]


def getScenarioBasedHarnessArgs(scenario, prefixes=("",)):
    """Return arguments for harness for a given scenario."""
    return HARNESS_ARGS + getScenarioMetricArgs(scenario, prefixes=prefixes)
