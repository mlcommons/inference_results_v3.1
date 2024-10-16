/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NvInferPlugin.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <dlfcn.h>

#include "loadgen.h"
#include "logger.h"
#include "test_settings.h"

#include "gpt_server.hpp"
#include "gpt_utils.hpp"
#include "qsl.hpp"

#include "cuda_profiler_api.h"

DEFINE_string(gpu_engines, "", "Engine");
DEFINE_string(devices, "all", "Enable comma separated numbered devices");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, Server, SingleStream)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "gptj", "Model name");
DEFINE_bool(use_fp8, false, "Use FP8 GPT harness");
DEFINE_bool(enable_sort, false, "Sort QSL samples before inference");
DEFINE_uint32(tensor_parallelism, 1, "Tensor parallelism count used for the GPT harness");
DEFINE_uint32(gpu_batch_size, 64, "Max Batch size to use for all devices and engines");
DEFINE_uint32(gpu_inference_streams, 1, "Number of streams (GPTCores) for inference");
DEFINE_uint32(gpu_copy_streams, 1, "Number of copy streams");
// TODO future CUDA graph optimization
DEFINE_bool(use_graphs, false, "Enable CUDA Graphs for TensorRT engines");
DEFINE_uint32(graphs_max_seqlen, 2048, "Max seqlen is used to control how many CUDA Graphs will be generated");
DEFINE_string(graph_specs, "",
    "Specify a comma separeated list of (maxSeqLen, min totSeqLen, max totSeqLen, step size) for CUDA graphs to be "
    "captured");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(verbose_nvtx, false, "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.");
DEFINE_bool(load_plugins, true, "Load TRT NvInfer plugins");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");
// TODO future optimization
// DEFINE_double(soft_drop, 1.0, "The threshold to soft drop requests when total length in a batch is too long");
// DEFINE_double(eviction_last, 0.0, "Set percentage of persistance cache limit");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");
DEFINE_uint64(server_num_issue_query_threads, 0, "Number of IssueQuery threads used in Server scenario");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// QSL arguments
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated list if there are more than "
    "one input.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {{"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream}, {"Server", mlperf::TestScenario::Server}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

std::string parseEngineFromRank(std::string const& enginePaths, int32_t rank, bool isGPTJ)
{
    if (isGPTJ)
    {
        return enginePaths;
    }
    std::vector<std::string> enginePathVec = splitString(enginePaths, ",");
    for (auto enginePath : enginePathVec)
    {
        if (enginePath.find("rank" + std::to_string(rank)) != std::string::npos)
        {
            return enginePath;
        }
    }
    LOG(ERROR) << "GPT175 did not find engine file for rank " << rank << "!";
    return "";
}

std::vector<int32_t> parseDevice(bool isGPTJ, int32_t mpiRank, int32_t numRanks)
{
    std::vector<int32_t> gpus;
    int32_t numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (isGPTJ)
    {
        if (FLAGS_devices == "all")
        {
            LOG(INFO) << "Found " << numDevices << " GPUs";
            for (int32_t i = 0; i < numDevices; i++)
            {
                gpus.emplace_back(i);
            }
        }
        else
        {
            LOG(INFO) << "Use GPUs: " << FLAGS_devices;
            auto deviceNames = splitString(FLAGS_devices, ",");
            for (auto& n : deviceNames)
            {
                gpus.emplace_back(std::stoi(n));
            }
        }
    }
    else
    {
        // set GPU device according to ranks
        CHECK(numRanks == numDevices) << "MPI world size does not match GPU device count!";
        CHECK(FLAGS_devices == "all") << "GPT175 has to set devices to \"all\"";
        gpus.emplace_back(mpiRank);
    }
    return gpus;
}

int main(int argc, char** argv)
{
    // MPI initialize
    int32_t mpiRank;
    int32_t numRanks;

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));
    bool const isMain{mpiRank == 0};
    bool const isGPTJ{numRanks <= 1};

    // initialize gLogger
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "GPT_HARNESS";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    gLogger.reportTestStart(sampleTest); // TODO maybe need to bypass for worker processes

    // global plugin init
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    // get plugin paths
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (auto& s : plugin_files)
    {
        gLogInfo << "Loading plugin: " << s << std::endl;
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh)
        {
            gLogError << "Error loading plugin library " << s << std::endl;
            return 1;
        }

        // TENSORRTAPI bool initLibNvInferPlugins(void* logger, char const* libNamespace);
        using RawFuncType = bool(void*, char const*);
        // TRT LLM plugin require non global init
        std::function<bool(void*, char const*)> initTRTLLMPlugin{
            reinterpret_cast<RawFuncType*>(dlsym(dlh, "initLibNvInferPlugins"))};
        initTRTLLMPlugin(&gLogger.getTRTLogger(), "tensorrt_llm");
    }

    // Scope to force all smart objects destruction before CUDA context resets
    {
        CHECK(numRanks == FLAGS_tensor_parallelism) << "MPI world size does not match tensor parallelism!";
        // Configure the test settings
        mlperf::TestSettings testSettings;
        testSettings.scenario = scenarioMap[FLAGS_scenario];
        testSettings.mode = testModeMap[FLAGS_test_mode];
        testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.server_coalesce_queries = true;
        testSettings.server_num_issue_query_threads = FLAGS_server_num_issue_query_threads;

        // Configure the logging settings
        mlperf::LogSettings logSettings;
        logSettings.log_output.outdir = FLAGS_logfile_outdir;
        logSettings.log_output.prefix = FLAGS_logfile_prefix;
        logSettings.log_output.suffix = FLAGS_logfile_suffix;
        logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
        logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
        logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
        logSettings.log_mode = logModeMap[FLAGS_log_mode];
        logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
        logSettings.enable_trace = FLAGS_log_enable_trace;

        std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
        std::vector<bool> start_from_device(tensor_paths.size(), false);
        std::vector<int> gpus = parseDevice(isGPTJ, mpiRank, numRanks);

        std::string sutName = isMain ? "GPT SERVER MAIN" : std::string{"GPT SERVER WORKER"} + std::to_string(mpiRank);
        std::shared_ptr<qsl::SampleLibrary> qsl = isMain
            ? std::make_shared<qsl::SampleLibrary>("GPT QSL", FLAGS_map_path, splitString(FLAGS_tensor_path, ","),
                FLAGS_performance_sample_count, 0, true, start_from_device)
            : nullptr;

        // TODO remove GPTJ hard code testing
        auto sut = std::make_shared<GPTServer>(sutName, qsl, parseEngineFromRank(FLAGS_gpu_engines, mpiRank, isGPTJ),
            gpus, FLAGS_gpu_copy_streams, FLAGS_gpu_inference_streams, FLAGS_gpu_batch_size, FLAGS_tensor_parallelism,
            /*beamWidth = */ 4, testSettings.server_target_latency_percentile,
            testSettings.server_num_issue_query_threads, mpiRank, numRanks, FLAGS_verbose_nvtx,
            FLAGS_use_fp8, isMain, isGPTJ, FLAGS_enable_sort);

        if (isMain)
        {
            LOG(INFO) << "Starting running actual test.";
            cudaProfilerStart();
            StartTest(sut.get(), qsl.get(), testSettings, logSettings);
            cudaProfilerStop();
            LOG(INFO) << "Finished running actual test.";
        }
    }

    MPICHECK(MPI_Finalize());
    return 0;
}
