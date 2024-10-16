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

#include "dlrm_server.h"

#include "glog/logging.h"
#include "loadgen.h"

#include <chrono>
#include <fstream>
#include <set>

#include "dlrm_kernels.h"
#include "dlrm_qsl.hpp"

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

bool operator==(const nvinfer1::Dims& d1, const nvinfer1 ::Dims& d2)
{
    if (d1.nbDims != d2.nbDims)
        return false;
    for (int it = 0; it < d1.nbDims; it++)
    {
        if (d1.d[it] != d2.d[it])
            return false;
    }
    return true;
}

size_t bindingVolume(std::shared_ptr<nvinfer1::ICudaEngine> engine, int idx)
{
    return lwis::volume(
        engine->getBindingDimensions(idx), engine->getBindingFormat(idx), engine->hasImplicitBatchDimension());
}

size_t GetEffectiveBatchSize(const std::vector<DLRMTask>& tasks)
{
    return std::accumulate(tasks.begin(), tasks.end(), 0ULL,
        [](const size_t curr, const DLRMTask& t) { return curr + t.numIndividualPairs; });
}

DLRMCore::DLRMCore(std::shared_ptr<nvinfer1::ICudaEngine> engine, int maxBatchSize, int numBundles,
    int numCompleteThreads, int profileIdx, double elRatio, bool verboseNVTX)
    : mEngine(engine)
    , mMaxBatchSize(maxBatchSize)
    , mResHandlerPool(numCompleteThreads)
    , mBundleCounter(0)
{
    mContext = InferObject(mEngine->createExecutionContext());
    if (verboseNVTX)
    {
        mContext->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    }
    else
    {
        mContext->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
    }
    LOG(INFO) << "Setting profile = " << profileIdx;
    mContext->setOptimizationProfile(profileIdx);
    SetBatchSize(maxBatchSize);
    LOG(INFO) << "Context creation complete";

    // eviction last setup
    if (elRatio > 0.0) {
        int32_t persistentCacheLimitCUDAValue = getPersistentCacheSizeLimit();
        mContext->setPersistentCacheLimit(elRatio*persistentCacheLimitCUDAValue);
    }

    CHECK_EQ(cudaStreamCreate(&mH2DStream), cudaSuccess);
    CHECK_EQ(cudaStreamCreate(&mComputeStream), cudaSuccess);
    CHECK_EQ(cudaStreamCreate(&mD2HStream), cudaSuccess);
    LOG(INFO) << "Created streams";

    size_t numBindings = mEngine->getNbBindings();
    CHECK_EQ(numBindings / mEngine->getNbOptimizationProfiles(), 3) << "Harness expects 3 bindings per engine profile";
    size_t firstBinding = profileIdx * numBindings / mEngine->getNbOptimizationProfiles();

    mNumInVol = bindingVolume(mEngine, firstBinding + 0);
    LOG(INFO) << "Profile - Numeric Input Volume: " << mNumInVol;
    mCatInVol = bindingVolume(mEngine, firstBinding + 1);
    LOG(INFO) << "Profile - Categorical Input Volume: " << mCatInVol;
    mOutVol = bindingVolume(mEngine, firstBinding + 2);
    LOG(INFO) << "Profile - Output Volume: " << mOutVol;

    mBindings.resize(numBundles);
    for (int i = 0; i < numBundles; ++i)
    {
        auto bundle = std::make_shared<DLRMEventBufferBundle>(i, mNumInVol, mCatInVol, mOutVol, mMaxBatchSize);
        mEventBufferBundle.push_back(bundle);

        // set this profile's bindings to DevicePtrs, set rest to nullptr
        mBindings[i].assign(numBindings, nullptr);
        mBindings[i][firstBinding + 0] = bundle->numericInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 1] = bundle->categoricalInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 2] = bundle->outputBuf.GetDevicePtr();
    }

    LOG(INFO) << "Created copy streams and buffers";
    LOG(INFO) << "Setup complete";
}

void DLRMCore::SetBatchSize(int batchSize)
{
    int profileNum = mContext->getOptimizationProfile();
    CHECK_EQ(profileNum >= 0 && profileNum < mEngine->getNbOptimizationProfiles(), true);
    int numBindings = mEngine->getNbBindings() / mEngine->getNbOptimizationProfiles();
    for (int i = 0; i < numBindings; i++)
    {
        if (mEngine->bindingIsInput(i))
        {
            int bindingIdx = numBindings * profileNum + i;
            auto inputDims = mContext->getBindingDimensions(bindingIdx);
            if (inputDims.d[0] != batchSize)
            {
                inputDims.d[0] = batchSize;
                CHECK_EQ(mContext->setBindingDimensions(bindingIdx, inputDims), true);
            }
        }
    }
    CHECK_EQ(mContext->allInputDimensionsSpecified(), true);
}

void DLRMCore::infer(std::shared_ptr<DLRMEventBufferBundle> ebBundle,
    size_t batchSize, // We assume that this batchsize is the effective batch size (padded to even value)
    std::vector<DLRMTask>& tasks, Batch* batch, void (*h2dCallBack)(void*), DLRMInferCallback resultCallback,
    DLRMNumericInputType* numericInputPtr, DLRMCategoricalInputType* categoricalInputPtr)
{        
    DLOG(INFO) << "infer() batch = " << batch;
    NVTX_RANGE_PUSH(("DLRMCore::infer: batchSize=" + std::to_string(batchSize)).c_str());
    CHECK_EQ((batchSize % 2 == 0) && (batchSize <= mMaxBatchSize), true);
    SetBatchSize(batchSize);

    bool oddBatch = batch ? batch->isOddBatch() : false; // batch is nullptr in WarmUp run
    // Copy buffers
    ebBundle->numericInputBuf.H2DAsync(numericInputPtr, batchSize, mH2DStream, oddBatch);
    ebBundle->categoricalInputBuf.H2DAsync(categoricalInputPtr, batchSize, mH2DStream, oddBatch);
    ebBundle->recordH2D(mH2DStream);

    // callback upon H2D completion
    CHECK_EQ(cudaLaunchHostFunc(mH2DStream, h2dCallBack, batch), cudaSuccess);

    void** bindings = mBindings[ebBundle->idx].data();

    // Run inference
    ebBundle->makeAwaitH2D(mComputeStream);
    CHECK_EQ(mContext->enqueueV2(bindings, mComputeStream, nullptr), true);
    ebBundle->recordInfer(mComputeStream);

    // Get output
    ebBundle->makeAwaitInfer(mD2HStream);
    ebBundle->outputBuf.D2HAsync(ebBundle->outputBuf.GetHostPtr(), batchSize, mD2HStream);
    ebBundle->recordD2H(mD2HStream);

    DLRMDeferredResult* deferredResult = new DLRMDeferredResult{batchSize, ebBundle->outputBuf.GetHostPtr(),
        std::move(tasks), resultCallback, [=](const DLRMResult& r) { mResHandlerPool.Enqueue(r); }};
    CHECK_EQ(cudaLaunchHostFunc(
                mD2HStream,
                [](void* deferredResult) -> void {
                    NVTX_RANGE_PUSH("deferredResult processing");
                    DLRMDeferredResult* res = reinterpret_cast<DLRMDeferredResult*>(deferredResult);
                    DLRMResult r
                        = {std::make_shared<std::vector<DLRMOutputType>>(res->outputs, res->outputs + res->batchSize),
                           std::move(res->tasks), res->callback};
                    res->resultCallback(r);
                    delete res;
                    NVTX_RANGE_POP();
                },
                deferredResult),
        cudaSuccess);

    NVTX_RANGE_POP();
}

void DLRMCore::inferFromDevice(std::shared_ptr<DLRMEventBufferBundle> ebBundle, size_t batchSize,
    std::vector<DLRMTask>& tasks, DLRMInferCallback resultCallback)
{
    NVTX_RANGE_PUSH(("DLRMCore::inferFromDevice: batchSize=" + std::to_string(batchSize)).c_str());

    CHECK_EQ((batchSize % 2 == 0) && (batchSize <= mMaxBatchSize), true);
    SetBatchSize(batchSize);

    bool contiguousData = true;
    for (size_t i = 1; (i < tasks.size()) && contiguousData; ++i)
    {
        contiguousData = contiguousData
            && (ebBundle->numericInputPtrBuf.GetHostPtr()[i]
                == ebBundle->numericInputPtrBuf.GetHostPtr()[i - 1]
                    + ebBundle->sampleSizesBuf.GetHostPtr()[i - 1] * mNumInVol);
        contiguousData = contiguousData
            && (ebBundle->categoricalInputPtrBuf.GetHostPtr()[i]
                == ebBundle->categoricalInputPtrBuf.GetHostPtr()[i - 1]
                    + ebBundle->sampleSizesBuf.GetHostPtr()[i - 1] * mCatInVol);
    }

    if (!contiguousData)
    {
        ebBundle->numericInputPtrBuf.H2DAsync(ebBundle->numericInputPtrBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->categoricalInputPtrBuf.H2DAsync(
            ebBundle->categoricalInputPtrBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->sampleSizesBuf.H2DAsync(ebBundle->sampleSizesBuf.GetHostPtr(), tasks.size(), mH2DStream);
        ebBundle->sampleOffsetsBuf.H2DAsync(ebBundle->sampleOffsetsBuf.GetHostPtr(), tasks.size(), mH2DStream);
    }
    ebBundle->recordH2D(mH2DStream);

    // Run inference
    ebBundle->makeAwaitH2D(mComputeStream);

    // Run gather kernel to prepare input data
    if (!contiguousData)
    {
        runGatherKernel((const int8_t**) (ebBundle->numericInputPtrBuf.GetDevicePtr()),
            (const int32_t**) (ebBundle->categoricalInputPtrBuf.GetDevicePtr()),
            (const size_t*) (ebBundle->sampleSizesBuf.GetDevicePtr()),
            (const size_t*) (ebBundle->sampleOffsetsBuf.GetDevicePtr()), ebBundle->numericInputBuf.GetDevicePtr(),
            ebBundle->categoricalInputBuf.GetDevicePtr(), static_cast<int>(tasks.size()), static_cast<int>(mNumInVol),
            static_cast<int>(mCatInVol), mComputeStream);
    }

    void** bindings = mBindings[ebBundle->idx].data();

    std::vector<void*> actualBindings;
    if (contiguousData)
    {
        actualBindings.push_back(ebBundle->numericInputPtrBuf.GetHostPtr()[0]);
        actualBindings.push_back(ebBundle->categoricalInputPtrBuf.GetHostPtr()[0]);
        actualBindings.push_back(bindings[2]);
        bindings = actualBindings.data();
    }

    CHECK_EQ(mContext->enqueueV2(bindings, mComputeStream, nullptr), true);
    ebBundle->recordInfer(mComputeStream);

    // Get output
    ebBundle->makeAwaitInfer(mD2HStream);
    ebBundle->outputBuf.D2HAsync(ebBundle->outputBuf.GetHostPtr(), batchSize, mD2HStream);
    ebBundle->recordD2H(mD2HStream);

    DLRMDeferredResult* deferredResult = new DLRMDeferredResult{batchSize, ebBundle->outputBuf.GetHostPtr(),
        std::move(tasks), resultCallback, [=](const DLRMResult& r) { mResHandlerPool.Enqueue(r); }};
    CHECK_EQ(cudaLaunchHostFunc(
                 mD2HStream,
                 [](void* deferredResult) -> void {
                     NVTX_RANGE_PUSH("deferredResult processing");
                     DLRMDeferredResult* res = reinterpret_cast<DLRMDeferredResult*>(deferredResult);
                     DLRMResult r
                         = {std::make_shared<std::vector<DLRMOutputType>>(res->outputs, res->outputs + res->batchSize),
                             std::move(res->tasks), res->callback};
                     res->resultCallback(r);
                     delete res;
                     NVTX_RANGE_POP();
                 },
                 deferredResult),
        cudaSuccess);

    NVTX_RANGE_POP();
}

void DLRMCore::WarmUp(double duration)
{
    double elapsed = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();

    std::vector<DLRMTask> dummyTasks(mMaxBatchSize, {{0, 0}, 1});
    std::vector<DLRMNumericInputType> dummyNumIn(mMaxBatchSize * mNumInVol);
    std::vector<DLRMCategoricalInputType> dummyCatIn(mMaxBatchSize * mCatInVol);

    LOG(INFO) << "Running warmup for " << duration << "s.";
    do
    {
        auto bundle = NextForegroundBundle();
        bundle->syncD2H();
        infer(
            bundle, mMaxBatchSize, dummyTasks, nullptr, [](void* batch) -> void { return; },
            [](std::vector<mlperf::QuerySampleResponse>&) { return; }, dummyNumIn.data(), dummyCatIn.data());
        elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    } while (elapsed < duration);
    for (size_t i = 0; i < mEventBufferBundle.size(); ++i)
        NextForegroundBundle()->syncD2H();
    LOG(INFO) << "Warmup complete, ran for " << elapsed << "s.";
}

std::shared_ptr<DLRMEventBufferBundle> DLRMCore::NextForegroundBundle()
{
    size_t idx = mBundleCounter;
    mBundleCounter = (mBundleCounter + 1) % mEventBufferBundle.size();
    return mEventBufferBundle[idx];
}

DLRMCore::~DLRMCore()
{
    CHECK_EQ(cudaStreamDestroy(mH2DStream), cudaSuccess);
    CHECK_EQ(cudaStreamDestroy(mComputeStream), cudaSuccess);
    CHECK_EQ(cudaStreamDestroy(mD2HStream), cudaSuccess);
}

DLRMServer::DLRMServer(const std::string name, const std::string enginePath, std::vector<DLRMSampleLibraryPtr_t> qsls,
    const std::vector<int>& gpus, int maxBatchSize, int numBundles, int numCompleteThreads, int numDLRMCores,
    double warmupDuration, int numStagingThreads, int numStagingBatches, int maxPairsPerThread, bool checkContiguity, 
    bool startFromDevice, NumaConfig numaConfig, uint64_t serverNumIssueQueryThreads, double elRatio, bool verbostNVTX)
    : mName{name}
    , mQsls{qsls}
    , mStartFromDevice(startFromDevice)
    , mStopWork{false}
    , mDLRMCores{gpus.size() * numDLRMCores}
    , mNumInVol(0)
    , mCatInVol(0)
    , mNumaConfig(numaConfig)
    , mGpuToNumaMap(getGpuToNumaMap(mNumaConfig))
    , mServerNumIssueQueryThreads(serverNumIssueQueryThreads)
    , mElRatio(elRatio)
    , mVerboseNVTX(verbostNVTX)
{
    NVTX_NAME_THIS_THREAD("DLRMServer");
    LOG(INFO) << "Using " << numDLRMCores << " DLRM Core(s) per Device";

    if (UseNuma())
    {
        LOG(INFO) << "Using NUMA nodes";
        CHECK(mNumaConfig.size() == qsls.size()) << "Number of QSLs should match number of NUMA nodes!";
    }

    mMaxBatchSize = maxBatchSize;
    // Enforce that max batch size is even due to Top MLP plugin
    if (mMaxBatchSize % 2 == 1)
    {
        mMaxBatchSize = mMaxBatchSize - 1;
    }

    std::vector<std::thread> setupThreads;
    for (const auto& deviceId : gpus)
    {
        setupThreads.emplace_back(&DLRMServer::SetupDevice, this, enginePath, numBundles, numCompleteThreads,
            numDLRMCores, warmupDuration, deviceId);
    }

    for (auto& t : setupThreads)
    {
        t.join();
    }

    if (mServerNumIssueQueryThreads > 0)
    {
        // FIXME: only use multiple issue query thread when NUMA is used
        //        number of issue query threads == number of GPUs
        CHECK(UseNuma());
        CHECK_EQ(mServerNumIssueQueryThreads, gpus.size());

        // when start_from_device is used, use only one issue query thread
        CHECK(!startFromDevice);

        LOG(INFO) << "Use number of server IssueQuery threads = " << mServerNumIssueQueryThreads;

        for (int deviceId = 0; deviceId < mServerNumIssueQueryThreads; ++deviceId)
        {
            bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
            mIssueQueryThreads.emplace_back(&DLRMServer::StartIssueThread, this, deviceId);
            bindThreadToCpus(mIssueQueryThreads.back(), mNumaConfig[mGpuToNumaMap[deviceId]].second);
            mThreadMap[mIssueQueryThreads.back().get_id()] = deviceId;
            resetNumaMemPolicy();
        }
    }

    if (!startFromDevice)
    {
        int numBatchMakers = UseNuma() ? mNumaConfig.size() : 1;
        for (int i = 0; i < numBatchMakers; ++i)
        {
            // Construct BatchMaker, in the order of device it works with
            auto devId = UseNuma() ? i : 0;
            auto numaNum = UseNuma() ? mNumaConfig.size() : 0;
            auto numaIdx = UseNuma() ? mGpuToNumaMap[devId] : -1;
            auto cpus = UseNuma() ? mNumaConfig[numaIdx].second : std::vector<int>();
            auto qslIdx = UseNuma() ? devId : 0;
            mBatchMakers.emplace_back(std::make_shared<BatchMaker>(
                /* numStagingThreads = */ numStagingThreads,
                /* numBatches = */ numStagingBatches,
                /* maxBatchSize = */ maxBatchSize,
                /* maxPairsPerThread = */ maxPairsPerThread,
                /* numericVolume = */ mNumInVol,
                /* categoricalVolume = */ mCatInVol,
                /* checkContiguity = */ checkContiguity,
                /* qsl = */ qsls[qslIdx],
                /* numaIdx = */ numaIdx,
                /* numaNum = */ numaNum,
                /* cpus = */ cpus));
        }
    }

    for (int deviceId = 0; deviceId < gpus.size(); ++deviceId)
    {
        // Bind the worker to the NUMA memory if needed
        if (UseNuma())
        {
            bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
        }
        mWorkerThreads.reserve(numDLRMCores);
        for (size_t profileIdx = 0; profileIdx < numDLRMCores; ++profileIdx)
        {
            auto dlrmCore = mDLRMCores[deviceId * numDLRMCores + profileIdx];
            mWorkerThreads.emplace_back(
                startFromDevice ? &DLRMServer::ProcessTasksFromDevice : &DLRMServer::ProcessTasks, this, dlrmCore,
                deviceId, profileIdx);
            // Limit the worker thread to the closest CPUs.
            if (UseNuma())
            {
                bindThreadToCpus(mWorkerThreads.back(), mNumaConfig[mGpuToNumaMap[deviceId]].second);
            }
        }
        // Reset memory allocation setting
        if (UseNuma())
        {
            resetNumaMemPolicy();
        }
    }
}

void DLRMServer::SetupDevice(const std::string enginePath, int numBundles, int numCompleteThreads, int numDLRMCores,
    int warmupDuration, int deviceId)
{
    if (UseNuma())
    {
        bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
    }

    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
    auto engine = DeserializeEngine(enginePath);
    CHECK_LE(numDLRMCores, engine->getNbOptimizationProfiles());

    mNumInVol = bindingVolume(engine, 0);
    mCatInVol = bindingVolume(engine, 1);

    for (size_t profileIdx = 0; profileIdx < numDLRMCores; ++profileIdx)
    {
        auto dlrmCore = std::make_shared<DLRMCore>(
            engine, mMaxBatchSize, numBundles, numCompleteThreads, profileIdx, mElRatio, mVerboseNVTX);
        mDLRMCores[deviceId * numDLRMCores + profileIdx] = dlrmCore;
        CHECK_LE(mMaxBatchSize, dlrmCore->GetMaxBatchSize());
        dlrmCore->WarmUp(warmupDuration);
    }

    // Reset memory allocation setting
    if (UseNuma())
    {
        resetNumaMemPolicy();
    }
};

DLRMServer::~DLRMServer()
{
    DLOG(INFO) << "~DLRMServer";
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mCondVar.notify_all();
    }
    for (auto batchMaker : mBatchMakers)
    {
        if (batchMaker)
        {
            batchMaker->StopWork();
        }
    }

    for (auto& workerThread : mWorkerThreads)
    {
        workerThread.join();
    }

    for (auto& issueQueryThread : mIssueQueryThreads)
    {
        issueQueryThread.join();
    }
}

const std::string& DLRMServer::Name()
{
    return mName;
}

void DLRMServer::StartIssueThread(int tId)
{
    LOG(INFO) << "Registering IssueQueryThread of " << std::this_thread::get_id()
              << " for Device[" << tId << "]";
    mlperf::RegisterIssueQueryThread();
}


void DLRMServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    NVTX_RANGE_PUSH(("DLRMServer::IssueQuery for " + std::to_string(samples.size()) + " samples").c_str());
    if (mStartFromDevice)
    {
        std::vector<size_t> numPairs(samples.size());
        std::transform(samples.begin(), samples.end(), numPairs.begin(),
            [&](const mlperf::QuerySample& x) { return mQsls[0]->GetNumUserItemPairs(x.index); });

        std::unique_lock<std::mutex> lck(mMtx);

        for (size_t i = 0; i < samples.size(); ++i)
            mTasks.push_back({samples[i], numPairs[i]});

        mCondVar.notify_one();
    }
    else
    {
        if (mServerNumIssueQueryThreads > 0)
        {
            // tId is device Id
            int tId = mThreadMap[std::this_thread::get_id()];
            mBatchMakers[tId]->IssueQuery(samples, 0, samples.size());
        }
        else if (samples.size() < SPLIT_THRESHOLD)
        {
            int nextBatchMakerIdx = (mPrevBatchMakerIdx + 1) % mBatchMakers.size();
            mBatchMakers[nextBatchMakerIdx]->IssueQuery(samples, 0, samples.size());
            mPrevBatchMakerIdx = nextBatchMakerIdx;
        }
        else
        {
            // Offline case where LoadGen won't use multiple issue threads
            // One time sample distribution - NUMA thread binding won't help (syscall outweighs the possible upside)
            const int64_t numSamplesTotal = samples.size();
            const int64_t numBatchMakers = mBatchMakers.size();
            // round robin to BatchMakers
            const int64_t numSamplesPerBM = numSamplesTotal / numBatchMakers;
            const int64_t remainder = numSamplesTotal % numBatchMakers;

            // Use a thread per batchMaker to issue queries in parallel. Each batchMaker has its own lock, so running
            // with multiple threads should not cause lock contention.
            auto issueQueryOneBatchMaker
                = [](std::shared_ptr<BatchMaker> batchMaker, const std::vector<mlperf::QuerySample>& samples,
                    const int64_t offset, const int64_t size) { batchMaker->IssueQuery(samples, offset, size); };
            std::vector<std::thread> issueThreads;
            issueThreads.reserve(numBatchMakers);

            int64_t offset{0};
            for (int64_t myIdx = 0; myIdx < numBatchMakers; ++myIdx)
            {
                int64_t size = numSamplesPerBM + (myIdx < remainder ? 1 : 0);
                if (size == 0)
                {
                    break;
                }

                issueThreads.emplace_back(
                    issueQueryOneBatchMaker, mBatchMakers[myIdx], std::ref(samples), offset, size);
                offset += size;
            }
            CHECK_EQ(offset, numSamplesTotal);

            for (auto& th : issueThreads)
            {
                th.join();
            }
        }
    }
    NVTX_RANGE_POP();
}

void DLRMServer::FlushQueries()
{
    NVTX_RANGE_PUSH("DLRMServer::FlushQueries");
    if (!mStartFromDevice)
    {
        for (auto batchMaker : mBatchMakers)
        {
            batchMaker->FlushQueries();
        }
    }
    NVTX_RANGE_POP();
}

void DLRMServer::ProcessTasks(std::shared_ptr<DLRMCore> dlrmCore, int deviceId, int profileIdx)
{
    NVTX_NAME_THIS_THREAD(("ProcessTasks" + std::to_string(profileIdx)).c_str());
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    if (UseNuma())
    {
        bindNumaMemPolicy(mGpuToNumaMap[deviceId], mNumaConfig.size());
    }

    // Process samples in batches
    while (true)
    {
        auto ebBundle = dlrmCore->NextForegroundBundle();

        // Only grab tasks when the copy stream is ready. This allows `tasks` to fill up with more batches while the
        // stream is still working. The GPU should still be busy with inference on other bundle(s)
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasks syncing for the foreground bundle to complete");
        ebBundle->syncD2H();
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH(("DLRMServer::ProcessTasks iteration, profile" + std::to_string(profileIdx)).c_str());

        int batchMakerIdx = UseNuma() ? mGpuToNumaMap[deviceId] : 0;
        Batch* batch = mBatchMakers[batchMakerIdx]->GetBatch();

        if (!batch)
        {
            NVTX_RANGE_POP();
            break;
        }

        size_t actualBatchSize = batch->getCommittedCopies();
        auto tasks = batch->getTasks();
        auto numericHostPtr = batch->getNumericHostPtr();
        auto categoricalHostPtr = batch->getCategoricalHostPtr();
        bool oddBatch = batch->isOddBatch();
        DLOG(INFO) << "Batch Size : " << actualBatchSize;

        dlrmCore->infer(
            ebBundle, actualBatchSize, tasks, batch,
            [](void* batch) -> void {
                reinterpret_cast<Batch*>(batch)->mBatchMaker->NotifyH2D(reinterpret_cast<Batch*>(batch));
            },
            [=](std::vector<mlperf::QuerySampleResponse>& responses) {
                if (oddBatch)
                {
                    responses.pop_back();
                }
                mlperf::QuerySamplesComplete(responses.data(), responses.size());
            },
            numericHostPtr, categoricalHostPtr);
        
        NVTX_RANGE_POP();
    }

    // Reset memory allocation setting
    if (UseNuma())
    {
        resetNumaMemPolicy();
    }
}

void DLRMServer::ProcessTasksFromDevice(std::shared_ptr<DLRMCore> dlrmCore, int deviceId, int profileIdx)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    // Process samples in batches
    while (true)
    {
        auto ebBundle = dlrmCore->NextForegroundBundle();

        // Only grab tasks when the copy stream is ready. This allows `tasks` to fill up with more batches while the
        // stream is still working. The GPU should still be busy with inference on other bundle(s)
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice syncing for the foreground bundle to complete");
        ebBundle->syncD2H();
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH(("DLRMServer::ProcessTasksFromDevice iteration, profile" + std::to_string(profileIdx)).c_str());

        auto tasks = GetBatch();
        if (tasks.empty())
        {
            NVTX_RANGE_POP();
            break;
        }

        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing batch host buffers");
        size_t originalBatchSize = std::accumulate(tasks.begin(), tasks.end(), (size_t) 0,
            [](size_t x, const DLRMTask& y) { return x + y.numIndividualPairs; });
        // Pas the tasks so that the batch size is even
        bool isBatchPadded = false;
        if ((originalBatchSize % 2) != 0)
        {
            tasks.push_back({tasks.back().querySample, 1});
            isBatchPadded = true;
        }

        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing numericInputPtrBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->numericInputPtrBuf.GetHostPtr(), [&](const DLRMTask& x) {
            return reinterpret_cast<DLRMNumericInputType*>(
                mQsls[0]->GetSampleAddress(x.querySample.index, 0, 0, deviceId));
        });
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing categoricalInputPtrBuf");
        std::transform(
            tasks.begin(), tasks.end(), ebBundle->categoricalInputPtrBuf.GetHostPtr(), [&](const DLRMTask& x) {
                return reinterpret_cast<DLRMCategoricalInputType*>(
                    mQsls[0]->GetSampleAddress(x.querySample.index, 1, 0, deviceId));
            });
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing sampleSizesBuf");
        std::transform(tasks.begin(), tasks.end(), ebBundle->sampleSizesBuf.GetHostPtr(),
            [&](const DLRMTask& x) { return x.numIndividualPairs; });
        NVTX_RANGE_POP();
        NVTX_RANGE_PUSH("DLRMServer::ProcessTasksFromDevice preparing sampleOffsetsBuf");
        ebBundle->sampleOffsetsBuf.GetHostPtr()[0] = 0;
        std::partial_sum(ebBundle->sampleSizesBuf.GetHostPtr(),
            ebBundle->sampleSizesBuf.GetHostPtr() + tasks.size() - 1, ebBundle->sampleOffsetsBuf.GetHostPtr() + 1);
        NVTX_RANGE_POP();

        size_t batchSize = std::accumulate(
            ebBundle->sampleSizesBuf.GetHostPtr(), ebBundle->sampleSizesBuf.GetHostPtr() + tasks.size(), (size_t) 0);
        NVTX_RANGE_POP();

        dlrmCore->inferFromDevice(ebBundle, batchSize, tasks, [=](std::vector<mlperf::QuerySampleResponse>& responses) {
            if (isBatchPadded)
            {
                responses.pop_back();
            }
            mlperf::QuerySamplesComplete(responses.data(), responses.size());
        });

        NVTX_RANGE_POP();
    }
}

std::shared_ptr<nvinfer1::ICudaEngine> DLRMServer::DeserializeEngine(std::string enginePath)
{
    int whichDevice;
    CHECK_EQ(cudaGetDevice(&whichDevice), cudaSuccess);
    LOG(INFO) << "Deserializing Engine on GPU#" << whichDevice;

    auto runtime = InferObject(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
    std::vector<char> trtModelStream;
    auto size = lwis::GetModelStream(trtModelStream, enginePath);
    auto engine = InferObject(runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr));

    LOG(INFO) << "Engine - Device Memory requirements: " << engine->getDeviceMemorySize();
    LOG(INFO) << "Engine - Number of Optimization Profiles: " << engine->getNbOptimizationProfiles();
    return engine;
}

std::vector<DLRMTask> DLRMServer::GetBatch()
{
    NVTX_RANGE_PUSH("DLRMServer::GetBatch");
    std::vector<DLRMTask> res;
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck(mMtx);
    mCondVar.wait(lck, [&] { return (!mTasks.empty()) || mStopWork; });

    NVTX_RANGE_PUSH(("Extracting tasks from queue with length " + std::to_string(mTasks.size())).c_str());
    // Consume up to mMaxBatchSize pairs
    int currentBatchSize = 0;

    res.reserve(mTasks.size());
    while (!mTasks.empty())
    {
        const auto& topTask = mTasks.front();
        currentBatchSize += topTask.numIndividualPairs;
        if (currentBatchSize > mMaxBatchSize)
            break;
        res.push_back(topTask);
        mTasks.pop_front();
    }

    // Let some other thread to consume more tasks
    if (!mTasks.empty())
        mCondVar.notify_one();

    NVTX_RANGE_POP();
    NVTX_RANGE_POP();

    return res;
}
