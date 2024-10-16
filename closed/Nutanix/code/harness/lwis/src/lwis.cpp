/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "lwis.hpp"
#include "loadgen.h"
#include "query_sample_library.h"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "logger.h"
#include <glog/logging.h>

namespace lwis
{
using namespace std::chrono_literals;

constexpr size_t const kFIRST_ENGINE = 0;
constexpr size_t const kFIRST_LOOP = 0;

inline int32_t getPersistentCacheSizeLimit()
{
    size_t persistentL2CacheSizeLimit = 0;
#if CUDART_VERSION >= 11030
    CHECK(cudaDeviceGetLimit(&persistentL2CacheSizeLimit, cudaLimitPersistingL2CacheSize) == 0);
#else
    persistentL2CacheSizeLimit = 0;
#endif
    return persistentL2CacheSizeLimit;
}

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

void enqueueShim(std::vector<nvinfer1::IExecutionContext*> const& contextVec, int batchSizeTotal,
    std::vector<size_t> const& batchLoopCounts, std::vector<std::vector<EngineBindings>> bindingsVec,
    cudaStream_t inferStream, cudaEvent_t* inputConsumed)
{
    // For each engine, do loop count times inference with batchSizeTotal / loop count times batches
    // Assume the first dim is batch dim. Each profile has numBindings bindings.
    for (size_t engineIdx = 0; engineIdx < contextVec.size(); ++engineIdx)
    {
        nvinfer1::IExecutionContext* context = contextVec[engineIdx];
        auto& engine = context->getEngine();
        if (engine.hasImplicitBatchDimension())
        {
            CHECK(batchLoopCounts.size() == 1 && batchLoopCounts.front() == 1)
                << "Engine with implicit batch size does not support batch looping.";
        }
        size_t const batchSize = batchSizeTotal / batchLoopCounts[engineIdx];
        for (size_t loop = 0; loop < batchLoopCounts[engineIdx]; ++loop)
        {
            if (engine.hasImplicitBatchDimension())
            {
                CHECK_EQ(
                    context->enqueue(batchSize, &(bindingsVec[engineIdx][loop][0]), inferStream, inputConsumed), true);
            }
            else
            {
                int32_t profileNum = context->getOptimizationProfile();
                CHECK_EQ(profileNum >= 0 && profileNum < engine.getNbOptimizationProfiles(), true);
                int32_t numBindings = engine.getNbBindings() / engine.getNbOptimizationProfiles();
                for (int i = 0; i < numBindings; i++)
                {
                    if (engine.bindingIsInput(i))
                    {
                        int bindingIdx = numBindings * profileNum + i;
                        auto inputDims = context->getBindingDimensions(bindingIdx);
                        // Only set binding dimension if batch size changes.
                        if (inputDims.d[0] != batchSize)
                        {
                            inputDims.d[0] = batchSize;
                            CHECK_EQ(context->setBindingDimensions(bindingIdx, inputDims), true);
                        }
                    }
                }
                CHECK_EQ(context->allInputDimensionsSpecified(), true);
                CHECK_EQ(context->enqueueV2(&(bindingsVec[engineIdx][loop][0]), inferStream, inputConsumed), true);
            }
        }
        // Total batch size for the next engine to loop with
        batchSizeTotal = batchSize * batchLoopCounts[engineIdx];
    }
}

//----------------
// Device
//----------------
void Device::AddEngine(EnginePtr_t engine, size_t batchSize, size_t batchLoopCount)
{
    size_t engineBatchSize{0};
    if (engine->GetCudaEngine()->hasImplicitBatchDimension())
    {
        engineBatchSize = engine->GetCudaEngine()->getMaxBatchSize();
    }
    else
    {
        // Assuming the first dimension of the first input is batch dim.
        engineBatchSize = engine->GetCudaEngine()->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
    }
    batchSize = std::min(batchSize, engineBatchSize);
    // Append batch size info in the same order of inference engines
    m_NumEngines++;
    m_BatchSizes.emplace_back(batchSize);
    m_BatchLoopCounts.emplace_back(batchLoopCount);
    m_Engines.emplace_back(engine);
}

void Device::BuildGraphs(bool directHostAccess)
{
    CHECK(m_BatchLoopCounts.size() == 1 && m_BatchLoopCounts.front() == 1)
        << "Batch looping does not support CUDA Graph";
    Issue();

    // build the graph by performing a single execution.  the engines are stored in ascending
    // order of maxBatchSize.  build graphs up to and including this size
    size_t batchSize = 1;
    auto maxBatchSize = m_BatchSizes.front();
    while (batchSize <= maxBatchSize)
    {
        for (auto& streamState : m_StreamState)
        {
            auto& stream = streamState.first;
            auto& state = streamState.second;
            auto& bufferManager = std::get<0>(state);
            std::vector<std::vector<EngineBindings>> enqueueBuffers
                = directHostAccess ? bufferManager->getHostBindings() : bufferManager->getDeviceBindings();
            auto& contextVec = std::get<4>(state);

            // need to issue enqueue to TRT to setup ressources properly _before_ starting graph construction
            enqueueShim(contextVec, batchSize, m_BatchLoopCounts, enqueueBuffers, m_InferStreams[0], nullptr);

            cudaGraph_t graph;
#if (CUDA_VERSION >= 10010)
            CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0], cudaStreamCaptureModeThreadLocal), CUDA_SUCCESS);
#else
            CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0]), CUDA_SUCCESS);
#endif
            enqueueShim(contextVec, batchSize, m_BatchLoopCounts, enqueueBuffers, m_InferStreams[0], nullptr);
            CHECK_EQ(cudaStreamEndCapture(m_InferStreams[0], &graph), CUDA_SUCCESS);

            cudaGraphExec_t graphExec;
            CHECK_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), CUDA_SUCCESS);

            t_GraphKey key = std::make_pair(stream, batchSize);
            m_CudaGraphExecs[key] = graphExec;

            CHECK_EQ(cudaGraphDestroy(graph), CUDA_SUCCESS);
        }
        batchSize++;
    }
    gLogInfo << "Capture " << m_CudaGraphExecs.size() << " CUDA graphs" << std::endl;
}

void Device::Setup()
{
    cudaSetDevice(m_Id);
    if (m_EnableDeviceScheduleSpin)
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    unsigned int cudaEventFlags
        = (m_EnableSpinWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming;

    for (auto& inferStream : m_InferStreams)
    {
        CHECK_EQ(cudaStreamCreate(&inferStream), CUDA_SUCCESS);
    }

    // Setup execution context for each engine
    std::vector<std::vector<std::shared_ptr<nvinfer1::ICudaEngine>>> copyStreamEngines(m_CopyStreams.size());
    std::vector<std::vector<nvinfer1::IExecutionContext*>> copyStreamContexts(m_CopyStreams.size());
    std::vector<std::vector<int32_t>> copyStreamProfiles(m_CopyStreams.size());
    for (auto enginePtr : m_Engines)
    {
        int32_t profileIdx{0};
        auto engine = enginePtr->GetCudaEngine();
        nvinfer1::IExecutionContext* context{nullptr};

        // Use the same TRT execution contexts for all copy streams
        // shape must be static and gpu_inference_streams must be 1
        if (m_UseSameContext)
        {
            context = engine->createExecutionContext();
            if (m_VerboseNVTX)
            {
                context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
            }
            else
            {
                context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
            }
            CHECK_EQ(context->getOptimizationProfile() == 0, true);
            CHECK_EQ(m_InferStreams.size() == 1, true);
            CHECK_EQ(context->allInputDimensionsSpecified(), true);
        }

        for (size_t i = 0; i < m_CopyStreams.size(); ++i)
        {
            // Create execution context for each profile
            // The number of engine profile should be the same as the number of copy streams
            if (!m_UseSameContext)
            {
                context = engine->createExecutionContext();
                if (m_VerboseNVTX)
                {
                    context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
                }
                else
                {
                    context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
                }
                // Set optimization profile if necessary.
                CHECK_EQ(profileIdx < engine->getNbOptimizationProfiles(), true);
                if (context->getOptimizationProfile() < 0)
                {
                    CHECK_EQ(context->setOptimizationProfile(profileIdx), true);
                }
            }
            CHECK_EQ(context->getOptimizationProfile() == profileIdx, true);
            copyStreamContexts[i].emplace_back(context);

            // eviction last setup
            if (m_elRatio>0.0) {
                int32_t persistentCacheLimitCUDAValue = getPersistentCacheSizeLimit();
                context->setPersistentCacheLimit(m_elRatio*persistentCacheLimitCUDAValue);
            }


            std::shared_ptr<nvinfer1::ICudaEngine> emptyPtr{};
            std::shared_ptr<nvinfer1::ICudaEngine> aliasPtr(emptyPtr, engine);
            copyStreamEngines[i].emplace_back(aliasPtr);
            copyStreamProfiles[i].emplace_back(profileIdx);

            // Engine with implicit batch only has one profile. DLA engine also only has one profile.
            // DLA engines use static batch size thus can create multiple contexts with one profile.
            if (!engine->hasImplicitBatchDimension() && engine->getNbOptimizationProfiles() > 1 && !m_UseSameContext)
            {
                ++profileIdx;
            }
        }
    }

    // Setup buffer manager and state for each copy stream
    for (size_t i = 0; i < m_CopyStreams.size(); ++i)
    {
        auto& copyStream = m_CopyStreams[i];
        CHECK_EQ(cudaStreamCreate(&copyStream), CUDA_SUCCESS);

        auto state = std::make_tuple(std::make_shared<BufferManager>(
                                         copyStreamEngines[i], m_BatchSizes, m_BatchLoopCounts, copyStreamProfiles[i]),
            cudaEvent_t(), cudaEvent_t(), cudaEvent_t(), copyStreamContexts[i]);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<1>(state), cudaEventFlags), CUDA_SUCCESS);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<2>(state), cudaEventFlags), CUDA_SUCCESS);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<3>(state), cudaEventFlags), CUDA_SUCCESS);
        m_StreamState.insert(std::make_pair(copyStream, state));
        m_StreamQueue.emplace_back(copyStream);
    }
}

void Device::Issue()
{
    CHECK_EQ(cudaSetDevice(m_Id), cudaSuccess);
}

void Device::Done()
{
    // join before destroying all members
    for (auto& thread : m_Threads)
    {
        thread.join();
    }

    // destroy member objects
    cudaSetDevice(m_Id);

    for (auto& inferStream : m_InferStreams)
    {
        cudaStreamDestroy(inferStream);
    }
    for (auto& copyStream : m_CopyStreams)
    {
        auto& state = m_StreamState[copyStream];

        cudaStreamDestroy(copyStream);
        cudaEventDestroy(std::get<1>(state));
        cudaEventDestroy(std::get<2>(state));
        cudaEventDestroy(std::get<3>(state));
        if (!m_UseSameContext)
        {
            std::vector<nvinfer1::IExecutionContext*>& engineContexts = std::get<4>(state);
            for (auto context : engineContexts)
            {
                context->destroy();
            }
        }
    }
    if (m_UseSameContext)
    {
        std::vector<nvinfer1::IExecutionContext*>& engineContexts = std::get<4>(m_StreamState[m_CopyStreams[0]]);
        for (auto context : engineContexts)
        {
            context->destroy();
        }
    }
    for (auto& kv : m_CudaGraphExecs)
    {
        CHECK_EQ(cudaGraphExecDestroy(kv.second), CUDA_SUCCESS);
    }
}

void Device::Completion()
{
    // Testing for completion needs to be based on the main thread finishing submission and
    // providing events for the completion thread to wait on.  The resources exist as part of the
    // Device class.
    //
    // Samples and responses are assumed to be contiguous and have the same sizes across samples.

    // Flow:
    // Main thread
    // - Find Device (check may be based on data buffer availability)
    // - Enqueue work
    // - Enqueue CompletionQueue batch
    // ...
    // - Enqueue CompletionQueue null batch

    // Completion thread(s)
    // - Wait for entry
    // - Wait for queue head to have data ready (wait for event)
    // - Dequeue CompletionQueue

    while (true)
    {
        // TODO: with multiple CudaStream inference it may be beneficial to handle these out of order
        auto batch = m_CompletionQueue.front_then_pop();

        if (batch.Responses.empty())
            break;

        // wait on event completion
        CHECK_EQ(cudaEventSynchronize(batch.Event), cudaSuccess);

        // callback if it exists
        if (m_ResponseCallback)
        {
            CHECK(batch.SampleIds.size() == batch.Responses.size()) << "missing sample IDs";
            m_ResponseCallback(&batch.Responses[0], batch.SampleIds, batch.Responses.size());
        }

        // assume this function is reentrant for multiple devices
        TIMER_START(QuerySamplesComplete);
        mlperf::QuerySamplesComplete(
            &batch.Responses[0], batch.Responses.size(), batch.ResponseCb.value_or(mlperf::ResponseCallback{}));
        TIMER_END(QuerySamplesComplete);

        m_StreamQueue.emplace_back(batch.Stream);
    }
}

//! validateBatchSettings
//!
//! Validate device m_BatchSizes, m_BatchLoopCounts
//! If DLA batch loop is used, the batch size of each engine must comply with the loop times
bool Device::validateBatchSettings()
{
    if (m_BatchSizes.size() > 1)
    {
        for (size_t i = 0; i < m_NumEngines - 1; ++i)
        {
            // Inference engine[i+1] should consume (inference engine[i] output * loop time[i]) / loop time[i+1] batches
            if (m_BatchSizes[i] * m_BatchLoopCounts[i] != m_BatchSizes[i + 1] * m_BatchLoopCounts[i + 1])
            {
                gLogError << "batch looping counts doesn't match engine batch sizes" << std::endl;
                return false;
            }
        }
    }
    return true;
}

//! validateEngineIOFormat
//!
//! If DLA batch loop is used, cuda graph should not be used and engine number should be 2
//! Otherwise, only 1 engine should be used for each device
bool Device::validateEngine(bool EnableCudaGraphs)
{
    if (m_BatchSizes.size() > 1)
    {
        if (EnableCudaGraphs)
        {
            gLogError << "Batch looping does not support CUDA graphs" << std::endl;
            return false;
        }
    }
    else
    {
        if (m_NumEngines != 1)
        {
            gLogError << "Only a single engine should be used for each device without DLA batch looping" << std::endl;
            return false;
        }
        if (m_BatchLoopCounts.back() != 1)
        {
            gLogError << "Single engine does not support batch looping" << std::endl;
        }
    }
    return true;
}

//----------------
// Server
//----------------

//! Setup
//!
//! Perform all necessary (untimed) setup in order to perform inference including: building
//! graphs and allocating device memory.
void Server::Setup(ServerSettings& settings, ServerParams& params)
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    m_ServerSettings = settings;

    // enumerate devices
    std::vector<size_t> devices;
    if (params.DeviceNames == "all")
    {
        int numDevices = 0;
        cudaGetDeviceCount(&numDevices);
        for (int i = 0; i < numDevices; i++)
        {
            devices.emplace_back(i);
        }
    }
    else
    {
        auto deviceNames = split(params.DeviceNames, ',');
        for (auto& n : deviceNames)
            devices.emplace_back(std::stoi(n));
    }

    // check if an engine was specified
    if (!params.EngineNames.size())
    {
        gLogError << "Engine file(s) not specified" << std::endl;
    }

    auto runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());

    for (auto const& deviceNum : devices)
    {
        cudaSetDevice(deviceNum);

        size_t type = 0;
        for (auto const& deviceTypes : params.EngineNames)
        {
            bool const isDLADevice = type == 1;
            size_t const numCopyStreams
                = isDLADevice ? m_ServerSettings.DLACopyStreams : m_ServerSettings.GPUCopyStreams;
            size_t const numInferStreams
                = isDLADevice ? m_ServerSettings.DLAInferStreams : m_ServerSettings.GPUInferStreams;
            // For each DLA device, create a lwis::Device instance
            for (int32_t deviceInstance = 0; deviceInstance < (isDLADevice ? runtime->getNbDLACores() : 1);
                 deviceInstance++)
            {
                if (deviceTypes.size() == 0)
                {
                    continue;
                }
                if (isDLADevice && m_ServerSettings.MaxDLAs != -1 && deviceInstance >= m_ServerSettings.MaxDLAs)
                {
                    continue;
                }
                // Check engine name, batch size and batch loop count parsing
                if (isDLADevice)
                {
                    CHECK(deviceTypes.size() == m_ServerSettings.DLABatchSizes.size())
                        << "Number of DLA engines and DLA batch sizes mismatch";
                    CHECK(m_ServerSettings.DLABatchSizes.size() == m_ServerSettings.DLALoopCounts.size())
                        << "Number of DLA batch sizes and DLA loop times mismatch";
                }
                else
                {
                    CHECK(deviceTypes.size() == m_ServerSettings.GPUBatchSizes.size())
                        << "Number of GPU engines and GPU batch sizes mismatch";
                    CHECK(m_ServerSettings.GPUBatchSizes.size() == m_ServerSettings.GPULoopCounts.size())
                        << "Number of GPU batch sizes and GPU loop times mismatch";
                }
                if (UseNuma())
                {
                    bindNumaMemPolicy(GetNumaIdx(deviceNum), GetNbNumas());
                }
                auto device = std::make_shared<lwis::Device>(deviceNum, numCopyStreams, numInferStreams,
                    m_ServerSettings.CompleteThreads, m_ServerSettings.EnableSpinWait,
                    m_ServerSettings.EnableDeviceScheduleSpin, isDLADevice, m_ServerSettings.UseSameContext,
                    m_ServerSettings.elRatio,
                    m_ServerSettings.VerboseNVTX);

                m_Devices.emplace_back(device);
                if (UseNuma())
                {
                    resetNumaMemPolicy();
                }
                // For each engine running on a device instance
                for (size_t engineIdx = 0; engineIdx < deviceTypes.size(); ++engineIdx)
                {
                    auto const& engineName = deviceTypes[engineIdx];
                    size_t const batchSize
                        = isDLADevice ? m_ServerSettings.DLABatchSizes[engineIdx] : m_ServerSettings.GPUBatchSizes[engineIdx];
                    size_t const batchLoopCount
                        = isDLADevice ? m_ServerSettings.DLALoopCounts[engineIdx] : m_ServerSettings.GPULoopCounts[engineIdx];
                    std::vector<char> trtModelStream;
                    auto size = GetModelStream(trtModelStream, engineName);
                    if (isDLADevice)
                    {
                        runtime->setDLACore(deviceInstance);
                    }
                    auto engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
                    // Add engines for each device instance and parse batch size
                    device->AddEngine(std::make_shared<lwis::Engine>(engine), batchSize, batchLoopCount);
                    std::ostringstream deviceName;
                    deviceName << "Device:" << deviceNum;
                    if (isDLADevice)
                    {
                        deviceName << ".DLA-" << deviceInstance;
                    }
                    else
                    {
                        deviceName << ".GPU";
                    }
                    device->m_Name = deviceName.str();
                    gLogInfo << device->m_Name << ": [" << engineIdx << "] " << engineName
                             << " has been successfully loaded." << std::endl;
                }
                CHECK(device->validateEngine(m_ServerSettings.EnableCudaGraphs)) << "Device engine settings invalid";
                CHECK(device->validateBatchSettings()) << "Device batch size settings invalid";
            }
            type++;
        }
    }

    runtime->destroy();

    CHECK(m_Devices.size()) << "No devices or engines available";

    for (auto& device : m_Devices)
    {
        device->Setup();
    }

    if (m_ServerSettings.EnableCudaGraphs)
    {
        gLogInfo << "Start creating CUDA graphs" << std::endl;
        std::vector<std::thread> tmpGraphsThreads;
        for (auto& device : m_Devices)
        {
            if (!device->m_DLA)
            {
                tmpGraphsThreads.emplace_back(
                    &Device::BuildGraphs, device.get(), m_ServerSettings.EnableDirectHostAccess);
            }
        }
        for (auto& thread : tmpGraphsThreads)
        {
            thread.join();
        }
        gLogInfo << "Finish creating CUDA graphs" << std::endl;
    }

    Reset();

    // create batchers
    for (size_t deviceNum = 0; deviceNum < (m_ServerSettings.EnableBatcherThreadPerDevice ? m_Devices.size() : 1);
         deviceNum++)
    {
        gLogInfo << "Creating batcher thread: " << deviceNum << " EnableBatcherThreadPerDevice: "
                 << (m_ServerSettings.EnableBatcherThreadPerDevice ? "true" : "false") << std::endl;
        m_Threads.emplace_back(std::thread(&Server::ProcessSamples, this));
        if (UseNuma() && m_ServerSettings.EnableBatcherThreadPerDevice)
        {
            bindThreadToCpus(m_Threads.back(), GetClosestCpus(m_Devices[deviceNum]->GetId()));
        }
    }

    // create issue threads
    if (m_ServerSettings.EnableCudaThreadPerDevice)
    {
        for (size_t deviceNum = 0; deviceNum < m_Devices.size(); deviceNum++)
        {
            gLogInfo << "Creating cuda thread: " << deviceNum << std::endl;
            m_IssueThreads.emplace_back(std::thread(&Server::ProcessBatches, this));
            if (UseNuma())
            {
                bindThreadToCpus(m_IssueThreads.back(), GetClosestCpus(m_Devices[deviceNum]->GetId()));
            }
        }
    }

    // If NUMA is used, make sure that the NUMA config makes sense.
    if (UseNuma())
    {
        CHECK(m_Devices.size() == m_ServerSettings.m_GpuToNumaMap.size())
            << "NUMA config does not match number of GPUs";
        CHECK(m_SampleLibraries.size() == m_ServerSettings.m_NumaConfig.size())
            << "Number of QSLs does not match NUMA config";
    }
}

void Server::Done()
{
    // send dummy batch to signal completion
    for (auto& device : m_Devices)
    {
        for (size_t i = 0; i < m_ServerSettings.CompleteThreads; i++)
        {
            device->m_CompletionQueue.push_back(Batch{});
        }
    }
    for (auto& device : m_Devices)
        device->Done();

    // send end sample to signal completion
    while (!m_WorkQueue.empty())
    {
    }

    while (m_DeviceNum)
    {
        size_t currentDeviceId = m_DeviceNum;
        m_WorkQueue.emplace_back(mlperf::QuerySample{0, 0});
        while (currentDeviceId == m_DeviceNum)
        {
        }
    }

    if (m_ServerSettings.EnableCudaThreadPerDevice)
    {
        for (auto& device : m_Devices)
        {
            std::deque<mlperf::QuerySample> batch;
            auto pair = std::make_pair(std::move(batch), nullptr);
            device->m_IssueQueue.emplace_back(pair);
        }
        for (auto& thread : m_IssueThreads)
            thread.join();
    }

    // join after we insert the dummy sample
    for (auto& thread : m_Threads)
    {
        thread.join();
    }
}

void Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    TIMER_START(IssueQuery);
    m_WorkQueue.insert(samples);
    TIMER_END(IssueQuery);
}

DevicePtr_t Server::GetNextAvailableDevice(size_t deviceId)
{
    DevicePtr_t device;
    if (!m_ServerSettings.EnableBatcherThreadPerDevice)
    {
        do
        {
            device = m_Devices[m_DeviceIndex];
            m_DeviceIndex = (m_DeviceIndex + 1) % m_Devices.size();
        } while (device->m_StreamQueue.empty());
    }
    else
    {
        device = m_Devices[deviceId];
        while (device->m_StreamQueue.empty())
        {
        }
    }

    return device;
}

void Server::IssueBatch(DevicePtr_t device, size_t batchSizeTotal, std::deque<mlperf::QuerySample>::iterator begin,
    std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream)
{
    // DLA engine uses static batch size
    auto enqueueBatchSizeTotal = device->m_DLA
        ? device->m_BatchSizes[kFIRST_ENGINE] * device->m_BatchLoopCounts[kFIRST_ENGINE]
        : batchSizeTotal;
    auto inferStream
        = (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : device->m_InferStreams[device->m_InferStreamIdx];

    auto& state = device->m_StreamState[copyStream];
    auto& bufferManager = std::get<0>(state);
    auto& htod = std::get<1>(state);
    auto& inf = std::get<2>(state);
    auto& dtoh = std::get<3>(state);
    auto& contextVec = std::get<4>(state);

    // setup Device
    device->Issue();

    // perform copy to device
#ifndef LWIS_DEBUG_DISABLE_INFERENCE
    std::vector<std::vector<EngineBindings>> enqueueBuffers = bufferManager->getDeviceBindings();
    TIMER_START(CopySamples);
    if (m_ServerSettings.EnableDma)
    {
        enqueueBuffers = CopySamples(device, batchSizeTotal, begin, end, copyStream,
            ((device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess) || m_ServerSettings.EnableDirectHostAccess),
            m_ServerSettings.EnableDmaStaging);
    }

    TIMER_END(CopySamples);
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);
    }
#ifndef LWIS_DEBUG_DISABLE_COMPUTE
    // perform inference
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
    }
    Device::t_GraphKey key = std::make_pair(copyStream, device->m_BatchSizes[kFIRST_ENGINE]);
    auto g_it = device->m_CudaGraphExecs.lower_bound(key);
    if (g_it != device->m_CudaGraphExecs.end())
    {
        CHECK_EQ(cudaGraphLaunch(g_it->second, inferStream), CUDA_SUCCESS);
    }
    else
    {
        TIMER_START(enqueueShim);
        enqueueShim(contextVec, enqueueBatchSizeTotal, device->m_BatchLoopCounts, enqueueBuffers, inferStream, nullptr);
        TIMER_END(enqueueShim);
    }
#endif
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);
    }

    // perform copy back to host
    size_t const lastEngine = device->m_NumEngines - 1;
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
    }
    if (m_ServerSettings.EnableDma && (!device->m_DLA || !m_ServerSettings.EnableDLADirectHostAccess)
        && !m_ServerSettings.EnableDirectHostAccess)
    {
        auto engine = device->m_Engines[lastEngine]->GetCudaEngine();

        if (!m_ServerSettings.EndOnDevice)
        {
            for (auto i = 0; i < engine->getNbBindings() / engine->getNbOptimizationProfiles(); i++)
            {
                if (!engine->bindingIsInput(i))
                {
                    bufferManager->copyOutputToHostAsync(lastEngine, i, copyStream);
                }
            }
        }
    }
    else
    {
        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
    }
#endif
    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);

    // optional synchronization
    if (m_ServerSettings.EnableSyncOnEvent)
    {
        cudaEventSynchronize(dtoh);
    }

    // generate asynchronous response
    TIMER_START(asynchronous_response);
    if (m_ServerSettings.EnableResponse)
    {
        // Assuming output binding is the last binding of the current profile.
        auto engine = device->m_Engines[lastEngine]->GetCudaEngine();
        int32_t bufferIdx = engine->getNbBindings() / engine->getNbOptimizationProfiles() - 1;
        int32_t bindingIdx = engine->getNbBindings() / engine->getNbOptimizationProfiles() - 1;
        // For device multiple engines, the last engine's host buffer only stores output buffers, thus the output buffer
        // index should be 0
        if (device->m_NumEngines > 1)
        {
            bufferIdx = 0;
        }
        auto buffer = static_cast<int8_t*>(bufferManager->getHostBuffer(lastEngine, bufferIdx));

        size_t sampleSize = volume(engine->getBindingDimensions(bindingIdx), engine->hasImplicitBatchDimension())
            * getElementSize(engine->getBindingDataType(bindingIdx));

        Batch batch;
        for (auto it = begin; it != end; ++it)
        {
            batch.Responses.emplace_back(mlperf::QuerySampleResponse{it->id, (uintptr_t) buffer, sampleSize});
            if (device->m_ResponseCallback)
            {
                batch.SampleIds.emplace_back(it->index);
            }
            buffer += sampleSize;
        }

        batch.Event = dtoh;
        batch.Stream = copyStream;
        if (m_ServerSettings.EndOnDevice)
        {
            auto baseDevice = reinterpret_cast<uintptr_t>(bufferManager->getDeviceBuffer(lastEngine, bufferIdx));
            auto baseHost = reinterpret_cast<uintptr_t>(bufferManager->getHostBuffer(lastEngine, bufferIdx));
            batch.ResponseCb = [=](mlperf::QuerySampleResponse* response) {
                auto dbuf = (response->data - baseHost) + baseDevice;
                CHECK_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(response->data), reinterpret_cast<void*>(dbuf),
                             response->size, cudaMemcpyDeviceToHost, copyStream),
                    cudaSuccess);
                // in accuracy mode, response data is copied to accuracy log immediately after callback,
                // so sync stream before returning
                CHECK_EQ(cudaStreamSynchronize(copyStream), cudaSuccess);
            };
        }

        device->m_CompletionQueue.emplace_back(batch);
    }
    TIMER_END(asynchronous_response);

    // Simple round-robin across inference streams.  These don't need to be managed like copy
    // streams since they are not tied with a resource that is re-used and not managed by hardware.
    device->m_InferStreamIdx = (device->m_InferStreamIdx + 1) % device->m_InferStreams.size();

    if (device->m_Stats.m_BatchSizeHistogram.find(batchSizeTotal) == device->m_Stats.m_BatchSizeHistogram.end())
    {
        device->m_Stats.m_BatchSizeHistogram[batchSizeTotal] = 1;
    }
    else
    {
        device->m_Stats.m_BatchSizeHistogram[batchSizeTotal]++;
    }
}

std::vector<std::vector<EngineBindings>> Server::CopySamples(DevicePtr_t device, size_t batchSizeTotal, std::deque<mlperf::QuerySample>::iterator begin,
    std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream, bool directHostAccess, bool staging)
{
    // Cover the following conditions:
    // 1) No sample library.  This is a debug mode and will copy whatever data is in the host buffer.
    // 2) Unified memory.  If contiguous supply the pointer directly to the engine (no copy).  Else
    // copy into the device buffer.

    // Note: We only copy input from loadgen to the first engine's input buffer.
    //       engines following the first engine will reuse previous engines' outputs
    auto& bufferManager = std::get<0>(device->m_StreamState[copyStream]);
    auto deviceId = device->GetId();

    // Setup default device buffers based on modes
    std::vector<std::vector<EngineBindings>> inputBuffers;
    if (directHostAccess)
    {
        inputBuffers = bufferManager->getHostBindings();
    }
    else
    {
        inputBuffers = bufferManager->getDeviceBindings();
    }
    size_t const inputBatchLoopCount{device->m_BatchLoopCounts[0]};
    size_t const batchSizePerLoop{batchSizeTotal / device->m_BatchLoopCounts[0]};

    // Currently assumes that input bindings always come before output bindings.
    auto engine = device->m_Engines[kFIRST_ENGINE]->GetCudaEngine();
    size_t numInputs{0};
    int numBindingsPerProfile{engine->getNbBindings() / engine->getNbOptimizationProfiles()};

    for (int i = 0; i < numBindingsPerProfile; i++)
    {
        if (engine->bindingIsInput(i))
        {
            ++numInputs;
        }
    }

    auto sampleLibrary = m_SampleLibraries[GetNumaIdx(deviceId)];

    if (sampleLibrary)
    {
        // test sample size vs buffer size derived from engine
        for (size_t i = 0; i < numInputs; i++)
        {
            size_t sampleSize = volume(engine->getBindingDimensions(i), engine->getBindingFormat(i),
                                    engine->hasImplicitBatchDimension())
                * getElementSize(engine->getBindingDataType(i));
            CHECK(sampleLibrary->GetSampleSize(i) == sampleSize)
                << "Sample size (" << sampleLibrary->GetSampleSize(i) << ") does not match engine input size ("
                << sampleSize << ")";
        }

        // detect contiguous samples
        TIMER_START(contiguity_detection);
        bool contiguous = true;
        if (!m_ServerSettings.ForceContiguous)
        {
            for (size_t i = 0; i < numInputs && contiguous; i++)
            {
                auto prev = static_cast<int8_t*>(sampleLibrary->GetSampleAddress(begin->index, i, deviceId));
                for (auto it = begin + 1; it != end; ++it)
                {
                    auto next = static_cast<int8_t*>(sampleLibrary->GetSampleAddress(it->index, i, deviceId));
                    if (next != prev + sampleLibrary->GetSampleSize(i))
                    {
                        contiguous = false;
                        break;
                    }
                    prev = next;
                }
            }
        }
        TIMER_END(contiguity_detection);

        TIMER_START(host_to_device_copy);
        for (size_t i = 0; i < numInputs; i++)
        {
            size_t const sampleSize{sampleLibrary->GetSampleSize(i)};
            if (!contiguous)
            {
                size_t offset = 0;
                for (auto it = begin; it != end; ++it)
                {
                    if (m_ServerSettings.EnableStartFromDeviceMem)
                    {
                        // copy from device buffer to staging device buffer
                        CHECK_EQ(cudaMemcpyAsync(static_cast<int8_t*>(bufferManager->getDeviceBuffer(kFIRST_ENGINE, i))
                                         + offset++ * sampleSize,
                                     sampleLibrary->GetSampleAddress(it->index, i, deviceId), sampleSize,
                                     cudaMemcpyDeviceToDevice, copyStream),
                            cudaSuccess);
                        device->m_Stats.m_PerSampleCudaMemcpyCalls++;
                    }
                    else if (directHostAccess)
                    {
                        // copy to the host staging buffer which is used as device buffer
                        memcpy(static_cast<int8_t*>(bufferManager->getHostBuffer(kFIRST_ENGINE, i))
                                + offset++ * sampleSize,
                            sampleLibrary->GetSampleAddress(it->index, i), sampleSize);
                        device->m_Stats.m_MemcpyCalls++;
                    }
                    else if (staging)
                    {
                        // copy to the host staging buffer and then to device buffer
                        memcpy(static_cast<int8_t*>(bufferManager->getHostBuffer(kFIRST_ENGINE, i))
                                + offset++ * sampleSize,
                            sampleLibrary->GetSampleAddress(it->index, i), sampleSize);
                        bufferManager->copyInputToDeviceAsync(kFIRST_ENGINE, i, copyStream);
                        device->m_Stats.m_MemcpyCalls++;
                    }
                    else
                    {
                        // copy direct to device buffer
                        bufferManager->copyInputToDeviceAsync(kFIRST_ENGINE, i, copyStream,
                            sampleLibrary->GetSampleAddress(it->index, i), sampleSize, offset++);
                        device->m_Stats.m_PerSampleCudaMemcpyCalls++;
                    }
                }
            }
            else
            {
                if (m_ServerSettings.EnableStartFromDeviceMem)
                {
                    if (!m_ServerSettings.EnableCudaGraphs)
                    {
                        // access samples directly when they are contiguous
                        for (int profileIdx = 0; profileIdx < engine->getNbOptimizationProfiles(); ++profileIdx)
                        {
                            uint8_t* sampleAddressBegin
                                = static_cast<uint8_t*>(sampleLibrary->GetSampleAddress(begin->index, i, deviceId));
                            for (size_t loop = 0; loop < inputBatchLoopCount; ++loop)
                            {
                                uint8_t* packedBinding = sampleAddressBegin + sampleSize * batchSizePerLoop * loop;
                                inputBuffers[kFIRST_ENGINE][loop][i + profileIdx * numBindingsPerProfile]
                                    = static_cast<void*>(packedBinding);
                            }
                        }
                    }
                    else
                    {
                        // with cuda graphs, we have to do D2D copies because we cannot change bindings for graphs
                        CHECK_EQ(cudaMemcpyAsync(static_cast<int8_t*>(bufferManager->getDeviceBuffer(kFIRST_ENGINE, i)),
                                     sampleLibrary->GetSampleAddress(begin->index, i, deviceId),
                                     sampleSize * batchSizeTotal, cudaMemcpyDeviceToDevice, copyStream),
                            cudaSuccess);
                        device->m_Stats.m_BatchedCudaMemcpyCalls++;
                    }
                }
                else if (directHostAccess)
                {
                    if (!m_ServerSettings.EnableCudaGraphs)
                    {
                        // access samples directly when they are contiguous
                        uint8_t* sampleAddressBegin
                            = static_cast<uint8_t*>(sampleLibrary->GetSampleAddress(begin->index, i));
                        for (int profileIdx = 0; profileIdx < engine->getNbOptimizationProfiles(); ++profileIdx)
                        {
                            for (size_t loop = 0; loop < inputBatchLoopCount; ++loop)
                            {
                                uint8_t* packedBinding = sampleAddressBegin + sampleSize * batchSizePerLoop * loop;
                                inputBuffers[kFIRST_ENGINE][loop][i + profileIdx * numBindingsPerProfile]
                                    = static_cast<void*>(packedBinding);
                            }
                        }
                    }
                    else
                    {
                        // with cuda graphs, we have to do H2H copies because we cannot change bindings for graphs
                        memcpy(bufferManager->getHostBuffer(kFIRST_ENGINE, i),
                            sampleLibrary->GetSampleAddress(begin->index, i), sampleSize * batchSizeTotal);
                    }
                }
                else
                {
                    // copy direct to device buffer with single DMA
                    bufferManager->copyInputToDeviceAsync(kFIRST_ENGINE, i, copyStream,
                        sampleLibrary->GetSampleAddress(begin->index, i), sampleSize * batchSizeTotal);
                    device->m_Stats.m_BatchedCudaMemcpyCalls++;
                }
            }
        }
        TIMER_END(host_to_device_copy);
    }
    else
    {
        // no sample library.  copy to device memory if necessary
        if (!directHostAccess)
        {
            for (size_t i = 0; i < numInputs; i++)
            {
                bufferManager->copyInputToDeviceAsync(kFIRST_ENGINE, i, copyStream);
                device->m_Stats.m_BatchedCudaMemcpyCalls++;
            }
        }
    }

    return inputBuffers;
}

void Server::Reset()
{
    m_DeviceIndex = 0;

    for (auto& device : m_Devices)
    {
        device->m_InferStreamIdx = 0;
        device->m_Stats.reset();
    }
}

void Server::ProcessSamples()
{
    // until Setup is called we may not have valid devices
    size_t deviceId = m_DeviceNum++;

    // initial device available
    auto device = GetNextAvailableDevice(deviceId);

    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        TIMER_START(m_WorkQueue_acquire_total);
        do
        {
            TIMER_START(m_WorkQueue_acquire);
            m_WorkQueue.acquire(samples, m_ServerSettings.Timeout,
                device->m_BatchSizes[kFIRST_ENGINE] * device->m_BatchLoopCounts[kFIRST_ENGINE],
                m_ServerSettings.EnableDequeLimit || m_ServerSettings.EnableBatcherThreadPerDevice);
            TIMER_END(m_WorkQueue_acquire);
        } while (samples.empty());
        TIMER_END(m_WorkQueue_acquire_total);

        auto begin = samples.begin();
        auto end = samples.end();

        // Use a null (0) id to represent the end of samples
        if (!begin->id)
        {
            m_DeviceNum--;
            break;
        }

        auto batchBegin = begin;

        // build batches up to maximum supported batchSize
        while (batchBegin != end)
        {
            // Input batch size depends on the first engine
            auto batchSizeTotal
                = std::min(device->m_BatchSizes[kFIRST_ENGINE] * device->m_BatchLoopCounts[kFIRST_ENGINE],
                    static_cast<size_t>(std::distance(batchBegin, end)));
            auto batchEnd = batchBegin + batchSizeTotal;

            // Acquire resources
            TIMER_START(m_StreamQueue_pop_front);
            auto copyStream = device->m_StreamQueue.front();
            device->m_StreamQueue.pop_front();
            TIMER_END(m_StreamQueue_pop_front);

            // Issue this batch
            if (!m_ServerSettings.EnableCudaThreadPerDevice)
            {
                // issue on this thread
                TIMER_START(IssueBatch);
                IssueBatch(device, batchSizeTotal, batchBegin, batchEnd, copyStream);
                TIMER_END(IssueBatch);
            }
            else
            {
                // issue on device specific thread
                std::deque<mlperf::QuerySample> batch(batchBegin, batchEnd);
                auto pair = std::make_pair(std::move(batch), copyStream);
                device->m_IssueQueue.emplace_back(pair);
            }

            // Advance to next batch
            batchBegin = batchEnd;

            // Get available device for next batch
            TIMER_START(GetNextAvailableDevice);
            device = GetNextAvailableDevice(deviceId);
            TIMER_END(GetNextAvailableDevice);
        }
    }
}

void Server::ProcessBatches()
{
    // until Setup is called we may not have valid devices
    size_t deviceId = m_IssueNum++;
    auto& device = m_Devices[deviceId];
    auto& issueQueue = device->m_IssueQueue;

    while (true)
    {
        auto pair = issueQueue.front();
        issueQueue.pop_front();

        auto& batch = pair.first;
        auto& stream = pair.second;

        if (batch.empty())
        {
            m_IssueNum--;
            break;
        }

        IssueBatch(device, batch.size(), batch.begin(), batch.end(), stream);
    }
}

void Server::Warmup(double duration)
{
    double elapsed = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();

    do
    {
        for (size_t deviceIndex = 0; deviceIndex < m_Devices.size(); ++deviceIndex)
        {
            // get next device to send batch to
            auto device = m_Devices[deviceIndex];

            for (auto copyStream : device->m_CopyStreams)
            {
                for (auto inferStream : device->m_InferStreams)
                {
                    auto& state = device->m_StreamState[copyStream];
                    auto& bufferManager = std::get<0>(state);
                    auto& htod = std::get<1>(state);
                    auto& inf = std::get<2>(state);
                    auto& dtoh = std::get<3>(state);
                    auto& contextVec = std::get<4>(state);

                    device->Issue();
                    auto engine = device->m_Engines[kFIRST_ENGINE]->GetCudaEngine();

                    if (m_ServerSettings.EnableDma && !(device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess)
                        && !m_ServerSettings.EnableDirectHostAccess && !m_ServerSettings.EnableDmaStaging)
                    {
                        for (auto i = 0; i < engine->getNbBindings() / engine->getNbOptimizationProfiles(); i++)
                        {
                            if (engine->bindingIsInput(i))
                            {
                                bufferManager->copyInputToDeviceAsync(kFIRST_ENGINE, i, copyStream);
                            }
                        }
                    }

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                    {
                        CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);
                        CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
                    }

                    Device::t_GraphKey key = std::make_pair(copyStream, device->m_BatchSizes[kFIRST_ENGINE]);
                    auto g_it = device->m_CudaGraphExecs.lower_bound(key);
                    if (g_it != device->m_CudaGraphExecs.end())
                    {
                        CHECK_EQ(cudaGraphLaunch(
                                     g_it->second, (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream),
                            CUDA_SUCCESS);
                    }
                    else
                    {
                        std::vector<std::vector<EngineBindings>> enqueueBuffers = bufferManager->getDeviceBindings();
                        if ((m_ServerSettings.EnableDLADirectHostAccess && device->m_DLA)
                            || m_ServerSettings.EnableDirectHostAccess)
                        {
                            enqueueBuffers = bufferManager->getHostBindings();
                        }
                        enqueueShim(contextVec,
                            device->m_BatchSizes[kFIRST_ENGINE] * device->m_BatchLoopCounts[kFIRST_ENGINE],
                            device->m_BatchLoopCounts, enqueueBuffers,
                            (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream, nullptr);
                    }

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                    {
                        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);
                    }

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                    {
                        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
                    }

                    size_t const lastEngine = device->m_NumEngines - 1;
                    if (m_ServerSettings.EnableDma && !(device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess)
                        && !m_ServerSettings.EnableDirectHostAccess && !m_ServerSettings.EnableDmaStaging)
                    {
                        for (auto i = 0; i < engine->getNbBindings() / engine->getNbOptimizationProfiles(); i++)
                        {
                            if (!engine->bindingIsInput(i))
                            {
                                bufferManager->copyOutputToHostAsync(lastEngine, i, copyStream);
                            }
                        }
                    }
                    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);
                }
            }
        }
        elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    } while (elapsed < duration);

    for (auto& device : m_Devices)
    {
        device->Issue();
        cudaDeviceSynchronize();
    }

    // reset server state
    Reset();
}

void Server::FlushQueries()
{
    // This function is called at the end of a series of IssueQuery calls (typically the end of a
    // region of queries that define a performance or accuracy test).  Its purpose is to allow a
    // SUT to force all remaining queued samples out to avoid implementing timeouts.

    // Currently, there is no use case for it in this IS.
}

void Server::SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
        std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
        callback)
{
    std::for_each(
        m_Devices.begin(), m_Devices.end(), [callback](DevicePtr_t device) { device->SetResponseCallback(callback); });
}

}; // namespace lwis
