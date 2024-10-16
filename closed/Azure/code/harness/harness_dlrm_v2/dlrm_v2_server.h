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

#pragma once

#include "half.h"
#include "qsl.hpp"
#include "system_under_test.h"
#include "utils.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "dlrm_v2_qsl.h"
#include "lwis_buffers.h"

#include "batch_maker.h"

constexpr size_t SPLIT_THRESHOLD = 1000;

struct DLRMv2Result
{
    std::shared_ptr<std::vector<DLRMv2OutputType>> outputs;
    std::vector<DLRMv2Task> tasks;
    DLRMv2InferCallback callback;
};

using DLRMv2ResultProcessingCallback = std::function<void(const DLRMv2Result& r)>;

using DLRMv2NumericPtrBuffer = DLRMv2DataBuffer<DLRMv2NumericInputType*>;
using DLRMv2CategoricalPtrBuffer = DLRMv2DataBuffer<DLRMv2CategoricalInputType*>;
using DLRMv2SampleSizeBuffer = DLRMv2DataBuffer<size_t>;

struct DLRMv2DeferredResult
{
    size_t batchSize;
    const DLRMv2OutputType* outputs;
    std::vector<DLRMv2Task> tasks;
    DLRMv2InferCallback callback;
    DLRMv2ResultProcessingCallback resultCallback;
};

class DLRMv2ResultHandlerPool
{
public:
    DLRMv2ResultHandlerPool(size_t numThreads)
        : mStopWork(false)
    {
        for (int i = 0; i < numThreads; ++i)
        {
            mThreads.emplace_back(&DLRMv2ResultHandlerPool::HandleResult, this);
        }
    }

    ~DLRMv2ResultHandlerPool()
    {
        {
            std::unique_lock<std::mutex> lock(mMtx);
            mStopWork = true;
            mCondVar.notify_all();
        }
        for (auto& t : mThreads)
        {
            t.join();
        }
    }

    void Enqueue(const DLRMv2Result& r)
    {
        NVTX_RANGE_PUSH("DLRMv2ResultHandlerPool::Enqueue");
        std::unique_lock<std::mutex> lock(mMtx);
        mResultQ.emplace_back(r);
        mCondVar.notify_one();
        NVTX_RANGE_POP();
    }

    void HandleResult()
    {
        DLRMv2Result res;
        std::vector<mlperf::QuerySampleResponse> responses;
        while (true)
        {
            NVTX_RANGE_PUSH("DLRMv2ResultHandlerPool::HandleResult iteration");
            {
                NVTX_RANGE_PUSH("Extract result from queue");
                std::unique_lock<std::mutex> lock(mMtx);
                mCondVar.wait(lock, [&]() { return (!mResultQ.empty()) || mStopWork; });

                if (mStopWork)
                {
                    NVTX_RANGE_POP();
                    NVTX_RANGE_POP();
                    break;
                }

                res = mResultQ.front();
                mResultQ.pop_front();
                mCondVar.notify_one();
                NVTX_RANGE_POP();
            }

            NVTX_RANGE_PUSH("Handle result");

            int offset = 0;
            responses.clear();
            for (const auto& task : res.tasks)
            {
                mlperf::QuerySampleResponse response{task.querySample.id,
                    reinterpret_cast<uintptr_t>(&res.outputs->at(offset)),
                    sizeof(DLRMv2OutputType) * task.numIndividualPairs};
                responses.emplace_back(response);

                offset = offset + task.numIndividualPairs;
            }
            CHECK_EQ(responses.size(), res.tasks.size());
            res.callback(responses);
            DLOG(INFO) << "Handled " << offset << " pairs";
            NVTX_RANGE_POP();
            NVTX_RANGE_POP();
        }
    }

private:
    std::vector<std::thread> mThreads;
    std::deque<DLRMv2Result> mResultQ;
    std::mutex mMtx;
    std::condition_variable mCondVar;
    bool mStopWork;
};

// Convenience bundle for {cuda events, input/output device buffers, output host buffers}
// We have 2 sets of such bundles per GPU, owned by DLRMv2Core, to correspond to a foreground
// task being readied on the host and a background task running inference on GPU
class DLRMv2EventBufferBundle
{
public:
    DLRMv2EventBufferBundle(size_t bundleIdx, size_t numInVol, size_t catInVol, size_t outVol, size_t maxBatchSize)
        : idx(bundleIdx)
        , numericInputBuf(numInVol, maxBatchSize, false)
        , categoricalInputBuf(catInVol, maxBatchSize, false)
        , outputBuf(outVol, maxBatchSize, true)
        , numericInputPtrBuf(1, maxBatchSize, true)
        , categoricalInputPtrBuf(1, maxBatchSize, true)
        , sampleSizesBuf(1, maxBatchSize, true)
        , sampleOffsetsBuf(1, maxBatchSize, true)
    {
        unsigned int flags = cudaEventDefault | cudaEventDisableTiming;
        CHECK_EQ(cudaEventCreateWithFlags(&h2dEvent, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&inferEvent, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&d2hEvent, flags), cudaSuccess);
    }
    ~DLRMv2EventBufferBundle()
    {
        CHECK_EQ(cudaEventDestroy(h2dEvent), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(inferEvent), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(d2hEvent), cudaSuccess);
    }

    void recordH2D(cudaStream_t& h2dStream)
    {
        CHECK_EQ(cudaEventRecord(h2dEvent, h2dStream), cudaSuccess);
    }
    void makeAwaitH2D(cudaStream_t& inferStream)
    {
        CHECK_EQ(cudaStreamWaitEvent(inferStream, h2dEvent, 0), cudaSuccess);
    }

    void recordInfer(cudaStream_t& inferStream)
    {
        CHECK_EQ(cudaEventRecord(inferEvent, inferStream), cudaSuccess);
    }
    void makeAwaitInfer(cudaStream_t& d2hStream)
    {
        CHECK_EQ(cudaStreamWaitEvent(d2hStream, inferEvent, 0), cudaSuccess);
    }

    void recordD2H(cudaStream_t& d2hStream)
    {
        CHECK_EQ(cudaEventRecord(d2hEvent, d2hStream), cudaSuccess);
    }
    void syncD2H()
    {
        CHECK_EQ(cudaEventSynchronize(d2hEvent), cudaSuccess);
    }

    size_t idx;
    cudaEvent_t h2dEvent;
    cudaEvent_t d2hEvent;
    cudaEvent_t inferEvent;
    DLRMv2NumericBuffer numericInputBuf;
    DLRMv2CategoricalBuffer categoricalInputBuf;
    DLRMv2OutputBuffer outputBuf;

    // The next 4 buffers are used only in start_from_device mode
    DLRMv2NumericPtrBuffer numericInputPtrBuf;
    DLRMv2CategoricalPtrBuffer categoricalInputPtrBuf;
    DLRMv2SampleSizeBuffer sampleSizesBuf;
    DLRMv2SampleSizeBuffer sampleOffsetsBuf;
};

class DLRMv2Core
{
public:
    DLRMv2Core(std::shared_ptr<nvinfer1::ICudaEngine> engine, int maxBatchSize, int numBundles, int numCompleteThreads,
        int profileIdx, double elRatio, bool verboseNVTX);
    ~DLRMv2Core();
    void infer(std::shared_ptr<DLRMv2EventBufferBundle> ebBundle, size_t batchSize, std::vector<DLRMv2Task>& tasks,
        Batch* batch, void (*h2dCallBack)(void*), DLRMv2InferCallback resultCallback,
        DLRMv2NumericInputType* numericInputPtr, DLRMv2CategoricalInputType* categoricalInputPtr);
    void inferFromDevice(std::shared_ptr<DLRMv2EventBufferBundle> ebBundle, size_t batchSize,
        std::vector<DLRMv2Task>& tasks, DLRMv2InferCallback resultCallback);
    void WarmUp(double duration);
    size_t GetMaxBatchSize()
    {
        return mMaxBatchSize;
    };

    size_t mNumInVol;
    size_t mCatInVol;
    size_t mOutVol;

    std::shared_ptr<DLRMv2EventBufferBundle> NextForegroundBundle();

private:
    void SetBatchSize(int batchSize);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::shared_ptr<nvinfer1::IExecutionContext> mContext;
    size_t mMaxBatchSize;

    DLRMv2ResultHandlerPool mResHandlerPool;

    cudaStream_t mH2DStream;
    cudaStream_t mComputeStream;
    cudaStream_t mD2HStream;

    std::vector<std::shared_ptr<DLRMv2EventBufferBundle>> mEventBufferBundle;
    size_t mBundleCounter;

    std::vector<std::vector<void*>> mBindings;
};

class DLRMv2Server : public mlperf::SystemUnderTest
{
public:
    DLRMv2Server(const std::string name, const std::string enginePath, std::vector<DLRMv2SampleLibraryPtr_t> qsls,
        const std::vector<int>& gpus, int maxBatchSize, int numBundles, int numCompleteThreads, int numDLRMv2Cores,
        double warmupDuration, int numStagingThreads, int numStagingBatches, int maxPairsPerThread,
        bool checkContiguity, bool startFromDevice, NumaConfig numaConfig, uint64_t serverNumIssueQueryThreads,
        bool useBatcherPerGpu, double elRatio, NumaToQSLMap numaToQslMap, bool verboseNVTX);

    virtual ~DLRMv2Server();

    const std::string& Name() override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;

    void StartIssueThread(int threadIdx);

    bool UseNuma()
    {
        return !mNumaConfig.empty();
    };

private:
    void ProcessTasks(std::shared_ptr<DLRMv2Core>, int deviceId, int profileIdx);
    void ProcessTasksFromDevice(std::shared_ptr<DLRMv2Core>, int deviceId, int profileIdx);
    void SetupDevice(const std::string enginePath, int numBundles, int numCompleteThreads, int numDLRMv2Cores,
        int warmupDuration, int deviceId);
    std::shared_ptr<nvinfer1::ICudaEngine> DeserializeEngine(const std::string enginePath);
    std::vector<DLRMv2Task> GetBatch();

    // Batch Makers for the start from host (start_from_device = false) case. One per NUMA node.
    std::vector<std::shared_ptr<BatchMaker>> mBatchMakers;

    const std::string mName;
    int mMaxBatchSize;
    bool mUseSingleQSL;
    std::vector<DLRMv2SampleLibraryPtr_t> mQsls;
    bool mStartFromDevice;

    uint64_t mServerNumIssueQueryThreads;
    bool mUseBatcherPerGpu;

    // Queue to be used for the start_from_device case, each sample is accompanied by the pair count
    std::deque<DLRMv2Task> mTasks;

    // mutex to serialize access to mTasks member variable
    std::mutex mMtx;

    // The object to allow threads to avoid spinning on mMtx and mTasks for the new work to arrive
    std::condition_variable mCondVar;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopWork;

    // TensorRT Runtime NVTX verbosity
    bool const mVerboseNVTX;

    std::map<std::thread::id, int> mThreadMap;

    std::vector<std::thread> mIssueQueryThreads;
    std::vector<std::thread> mWorkerThreads;
    std::vector<std::shared_ptr<DLRMv2Core>> mDLRMv2Cores;
    size_t mNumInVol;
    size_t mCatInVol;

    // NUMA configs of the machine: list of CPUs for each NUMA node, assuming each GPU corresponds to one NUMA node.
    NumaConfig mNumaConfig;
    GpuToNumaMap mGpuToNumaMap;

    // May use QSL that covers multiple NUMA nodes
    NumaToQSLMap mNumaToQslMap;

    // eviction last controls
    double mElRatio;

    // When NUMA is used, we issue queries around NUMA nodes in round-robin.
    int64_t mPrevBatchMakerIdx{-1};
};
