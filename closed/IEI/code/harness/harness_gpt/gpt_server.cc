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

#include "gpt_server.hpp"
#include "glog/logging.h"
#include "gpt_core.hpp"
#include "gpt_utils.hpp"
#include "loadgen.h"

#include <fstream>
#include <limits>
#include <set>

std::string const& GPTServer::Name()
{
    return mName;
};

// Sort samples in the descending order of input length
std::vector<std::pair<int32_t, int32_t>> GPTServer::sortSamples(std::vector<mlperf::QuerySample> const& samples) const
{
    CHECK(mIsMain) << "Only main process should communicate with QSL";
    std::vector<std::pair<int32_t, int32_t>> sequenceSamplePosAndLength(samples.size());
    for (size_t samplePos = 0; samplePos < samples.size(); ++samplePos)
    {
        sequenceSamplePosAndLength[samplePos] = std::make_pair(
            samplePos, *static_cast<int32_t*>(mQsl->GetSampleAddress(samples[samplePos].index, kINPUT_LENGTH_IDX)));
    }
    if (mEnableSort)
    {
        VLOG(1) << "Sorting enabled!";
        std::sort(sequenceSamplePosAndLength.begin(), sequenceSamplePosAndLength.end(),
            [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) -> bool {
                return a.second > b.second;
            });
        VLOG(1) << "Sorted " << samples.size() << " sample(s).";
    }
    else
    {
        VLOG(1) << "Sorting disabled!";
    }
    return sequenceSamplePosAndLength;
}

// SUT override IssueQuery entrance
void GPTServer::IssueQuery(std::vector<mlperf::QuerySample> const& samples)
{
    if (mIsMain)
    {
        // push samples from LoadGen to mTasksVec (task vector)
        NVTX_MARK("IssueQuery:" + std::to_string(samples.size()) + " tasks", nvtx::COLOR_BLUE_2);
        auto queryArrivedTime = std::chrono::high_resolution_clock::now();

        // sort samples in the descending order of input length
        std::vector<std::pair<int32_t, int32_t>> sequenceSamplePosAndLength = sortSamples(samples);

        // find which task vector to enqueue tasks
        int32_t qThreadIdx = mIssueQueryThreadMap[std::this_thread::get_id()];
        // put sorted samples into task queue as a batch
        // one batch signals one worker thread
        for (size_t beginSamplePos = 0; beginSamplePos < sequenceSamplePosAndLength.size(); beginSamplePos += mMaxBatchSize)
        {
            int32_t actualBatchSize
                = std::min(mMaxBatchSize, static_cast<int32_t>(sequenceSamplePosAndLength.size() - beginSamplePos));
            mTotalSampleComsumed += actualBatchSize;
            {
                std::unique_lock<std::mutex> lck((*mTasksVecMtxs)[qThreadIdx]);
                for (int32_t i = 0; i < actualBatchSize; ++i)
                {
                    int32_t samplePos = sequenceSamplePosAndLength[beginSamplePos + i].first;
                    mTasksVec[qThreadIdx].push_back({samples[samplePos], queryArrivedTime});
                }
                // notify getTasks() to consume tasks
                (*mTasksVecCondVars)[qThreadIdx].notify_one();
            }
        }
    }
}

// SUT override FlushQueries entrance
void GPTServer::FlushQueries() {}

// SUT register issue thread for server scenario
void GPTServer::startIssueThread(int32_t threadIdx)
{
    if (mIsMain)
    {
        // issue query thread
        {
            CHECK_EQ(!mTasksVecMtxs->empty(), true);
            std::lock_guard<std::mutex> lock((*mTasksVecMtxs)[0]);
            mIssueQueryThreadMap[std::this_thread::get_id()] = threadIdx;
        }
        mlperf::RegisterIssueQueryThread();
    }
}

GPTBatch GPTServer::getTasks(int32_t qThreadIdx)
{
    CHECK(mIsMain) << "Only main process should get tasks";

    // construct batch from task queue
    GPTBatch batch;
    batch.reserve(mMaxBatchSize);
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck((*mTasksVecMtxs)[qThreadIdx]);
    (*mTasksVecCondVars)[qThreadIdx].wait(lck, [&] { return (!mTasksVec[qThreadIdx].empty()) || mStopGetTasks; });

    // Consume up to mMaxBatchSize samples
    for (int i = 0; !mTasksVec[qThreadIdx].empty() && i < mMaxBatchSize; ++i)
    {
        batch.push_back(mTasksVec[qThreadIdx].front());
        mTasksVec[qThreadIdx].pop_front();
    }
    VLOG(1) << "Formed batch with " << batch.size() << " sample(s)";
    // Let some other thread consume remaining tasks
    if (!mTasksVec[qThreadIdx].empty())
    {
        (*mTasksVecCondVars)[qThreadIdx].notify_one();
    }

    return batch;
}

void GPTServer::processTasks(std::shared_ptr<GPTCore> gptCore, int32_t deviceId, int32_t qThreadIdx)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
    // main process handles batch construction
    if (mIsMain)
    {
        uint64_t totalCountInThread = 0;
        // Process samples in batches
        NVTX_THREAD_START("getTasks", nvtx::COLOR_BLUE_1);
        auto batch = getTasks(qThreadIdx);
        NVTX_THREAD_END();

        while (!batch.empty())
        {
            totalCountInThread += batch.size();
            NVTX_THREAD_START("MPI infer:" + std::to_string(batch.size()) + " samples", nvtx::COLOR_BLUE_0);
            mStartInfer = true;
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            MPICHECK(MPI_Bcast(&mStartInfer, sizeof(mStartInfer), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));
            VLOG(1) << "Start inference on batch with " << batch.size() << " sample(s)";
            gptCore->infer(batch, mQsl);
            // sync batch complete
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            NVTX_THREAD_END();

            NVTX_THREAD_START("getTasks", nvtx::COLOR_BLUE_1);
            batch = getTasks(qThreadIdx);
            NVTX_THREAD_END();
        }
    }
    else // worker busy waiting for mStartInfer
    {
        while (true)
        {
            // wait for main signal to start next batch or abort
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            MPICHECK(MPI_Bcast(&mStartInfer, sizeof(mStartInfer), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));
            if (!mStartInfer)
            {
                break;
            }
            gptCore->infer({}, mQsl);
            // sync batch complete
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
    }

    using CLK = std::chrono::high_resolution_clock;
    VLOG(1) << "End of processTasks - process " << mRank;
}

static void createModelStreams(std::string const& enginePath, std::vector<std::vector<char>>& trtModelStreams)
{
    // we get a comma-separated list of engine paths
    std::vector<std::string> paths;
    int32_t from = 0;
    int32_t to;
    while ((to = enginePath.find(',', from)) != std::string::npos)
    {
        paths.emplace_back(enginePath.substr(from, to - from));
        from = to + 1;
    }

    if (from < enginePath.size())
    {
        paths.emplace_back(enginePath.substr(from, enginePath.size() - from));
    }

    LOG(INFO) << "Loading " << paths.size() << " engine(s)";
    for (auto& p : paths)
    {
        LOG(INFO) << "Engine Path: " << p;
    }

    trtModelStreams.resize(paths.size());
    for (size_t i = 0; i < trtModelStreams.size(); ++i)
    {
        lwis::GetModelStream(trtModelStreams[i], paths[i]);
    }
}

void GPTServer::createEnginesPerGPU(
    int32_t deviceId, std::shared_ptr<std::mutex> pMtx, std::vector<std::vector<char>> const& trtModelStreams)
{
    CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);

    auto runtime = InferObject(nvinfer1::createInferRuntime(gLogger.getTRTLogger()));
    // load all the engines
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> engines(trtModelStreams.size());
    std::transform(trtModelStreams.begin(), trtModelStreams.end(), engines.begin(),
        [&](const std::vector<char>& trtModelStream) {
            return InferObject(runtime->deserializeCudaEngine(trtModelStream.data(), trtModelStream.size(), nullptr));
        });

    {
        std::unique_lock<std::mutex> lck(*pMtx.get());
        mEnginesPerGPU[deviceId] = std::move(engines);
    }
    // prevent runtime from being destroyed throwing API usage error
    mRuntimes.push_back(std::move(runtime));
}

GPTServer::GPTServer(std::string const& name, std::shared_ptr<qsl::SampleLibrary> qsl, std::string const enginePath,
    std::vector<int32_t> const& gpuDeviceIds, int32_t numCopyStreams, int32_t numGPTCores, int32_t maxBatchSize,
    int32_t numTensorParallelism, int32_t beamWidth,
    // /* bool useGraphs */, /*int32_t graphsMaxSeqLen*/, /*std::string const& graphSpecs*/, /* double softDrop */, /*
    // double elRatio ,*/
    double targetLatencyPercentile, int32_t serverNumIssueQueryThreads, int32_t mpiRank, int32_t numRanks,
    bool verboseNVTX, bool useFp8, bool isMain, bool isGPTJ, bool enableSort)
    : mName(name)
    , mQsl(qsl)
    , mStopGetTasks(false)
    , mTotalSampleComsumed(0)
    , mMaxBatchSize(maxBatchSize)
    // , mGraphMaxSeqLen{graphMaxSeqLen}
    // , mSoftDrop{softDrop}
    // , mSoftDropCount{0}
    // , mTotalLengthSet{mSoftDrop}
    , mTargetLatencyPercentile(targetLatencyPercentile)
    , mTotalTasksCount(0)
    , mNumWP(numTensorParallelism - 1)
    , mRank(mpiRank)
    , mIsMain(isMain)
    , mIsGPTJ(isGPTJ)
    , mEnableSort(enableSort)
{
    {
        // only create one model streams
        std::vector<std::vector<char>> trtModelStreams;
        createModelStreams(enginePath, trtModelStreams);

        // create TRT engines in parallel
        std::shared_ptr<std::mutex> pMtx = std::make_shared<std::mutex>();
        std::vector<std::thread> engineCreationThreads;
        for (auto deviceId : gpuDeviceIds)
        {
            engineCreationThreads.emplace_back(&GPTServer::createEnginesPerGPU, this, deviceId, pMtx, trtModelStreams);
        }
        for (auto& thread : engineCreationThreads)
        {
            thread.join();
        }
        LOG(INFO) << "Engines Deserialization Completed";
    }

    // create 1 GPTCore for each infer stream on each gpu and store them in temporary vector
    // each GPTCore uses 1 infer stream and 1 engine profile
    // TODO capture CUDA graphs in parallel
    std::vector<std::vector<std::shared_ptr<GPTCore>>> gptCoreVec(gpuDeviceIds.size());
    for (size_t idx = 0; idx < gpuDeviceIds.size(); ++idx)
    {
        for (int32_t gptCoreIdx = 0; gptCoreIdx < numGPTCores; ++gptCoreIdx)
        {
            auto deviceId = gpuDeviceIds[idx];
            CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
            gptCoreVec[idx].push_back(std::make_shared<GPTCore>(mMaxBatchSize, numTensorParallelism, beamWidth,
                mEnginesPerGPU.at(deviceId), numCopyStreams, numGPTCores, gptCoreIdx, /*useGraphs,*/ mRank, numRanks,
                deviceId, verboseNVTX, useFp8, mIsMain, mIsGPTJ));
        }
    }

    // create issue query thread for server scenario
    if (serverNumIssueQueryThreads > 0)
    {
        CHECK_EQ((gpuDeviceIds.size() * numGPTCores) % serverNumIssueQueryThreads == 0, true);
        LOG(INFO) << "Use number of server IssueQuery threads = " << serverNumIssueQueryThreads;
        mTasksVec.resize(serverNumIssueQueryThreads);
        mTasksVecMtxs = std::make_unique<std::vector<std::mutex>>(serverNumIssueQueryThreads);
        mTasksVecCondVars = std::make_unique<std::vector<std::condition_variable>>(serverNumIssueQueryThreads);
        for (int32_t i = 0; i < serverNumIssueQueryThreads; ++i)
        {
            mIssueQueryThreads.emplace_back(&GPTServer::startIssueThread, this, i);
        }
    }
    else // create a single set of task queue and locks
    {
        mTasksVec.resize(1);
        mTasksVecMtxs = std::make_unique<std::vector<std::mutex>>(1);
        mTasksVecCondVars = std::make_unique<std::vector<std::condition_variable>>(1);
    }

    // warm up GPTCore and launch threads for processing tasks
    int32_t GPTCoresPerQThread = (serverNumIssueQueryThreads == 0)
        ? std::numeric_limits<int>::max()
        : (gpuDeviceIds.size() * numGPTCores) / serverNumIssueQueryThreads;
    int32_t counter = 0;
    int32_t qThreadIdx = 0;

    mWorkerThreads.reserve(gpuDeviceIds.size());
    for (size_t idx = 0; idx < gpuDeviceIds.size(); ++idx)
    {
        auto deviceId = gpuDeviceIds[idx];
        CHECK_EQ(cudaSetDevice(deviceId), cudaSuccess);
        for (int32_t gptCoreIdx = 0; gptCoreIdx < numGPTCores; ++gptCoreIdx)
        {
            auto gptCore = gptCoreVec[idx][gptCoreIdx];
            gptCore->warmUp();
            CHECK(mMaxBatchSize <= gptCore->getMaxBatchSize());
            mWorkerThreads.emplace_back(&GPTServer::processTasks, this, gptCore, deviceId, qThreadIdx);
            ++counter;
            if (counter == GPTCoresPerQThread)
            {
                ++qThreadIdx;
                counter = 0;
            }
        }
    }
}

GPTServer::~GPTServer()
{
    // MPI stop all process
    if (mIsMain)
    {
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        mStartInfer = false;
        MPICHECK(MPI_Bcast(&mStartInfer, sizeof(mStartInfer), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));
    }
    {
        std::vector<std::unique_lock<std::mutex>> lcks;
        for (size_t i = 0; i < mTasksVecMtxs->size(); ++i)
        {
            lcks.emplace_back((*mTasksVecMtxs)[i]);
        }
        mStopGetTasks = true;
        for (size_t i = 0; i < mTasksVecCondVars->size(); ++i)
        {
            (*mTasksVecCondVars)[i].notify_all();
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
