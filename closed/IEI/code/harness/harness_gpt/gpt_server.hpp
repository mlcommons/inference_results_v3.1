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

#ifndef GPT_SERVER_HPP
#define GPT_SERVER_HPP

#include "gpt_core.hpp"
#include "half.h"
#include "lwis_buffers.h"
#include "qsl.hpp"
#include "system_under_test.h"
#include "utils.hpp"

constexpr int32_t kMPI_MAIN_RANK{0};
constexpr int32_t kMPI_INFER_TAG{0};
constexpr int32_t kINPUT_TOKEN_IDX{0};
constexpr int32_t kINPUT_MASK_IDX{1};
constexpr int32_t kINPUT_LENGTH_IDX{2};

//! This class is the overrided mlperf SUT class. GPTServer initializes TRT LLM engines, manages data transfer from the
//! QSL and sends query samples to GPTCore instances.
class GPTServer : public mlperf::SystemUnderTest
{
public:
    GPTServer(std::string const& name, std::shared_ptr<qsl::SampleLibrary> qsl, std::string const enginePath,
        std::vector<int32_t> const& gpuDeviceIds, int32_t numCopyStreams, int32_t numGPTCores, int32_t maxBatchSize,
        int32_t numTensorParallelism, int32_t beamWidth,
        // /* bool useGraphs */, /*int32_t graphsMaxSeqLen*/, /*std::string const& graphSpecs*/, /* double softDrop */,
        // /* double elRatio ,*/
        double targetLatencyPercentile, int32_t serverNumIssueQueryThreads, int32_t mpiRank, int32_t numRanks,
        bool verboseNVTX, bool useFp8, bool isMain, bool isGPTJ, bool enableSort);

    virtual ~GPTServer();

    std::string const& Name() override;
    void IssueQuery(std::vector<mlperf::QuerySample> const& samples) override;
    void FlushQueries() override;

private:
    std::vector<std::pair<int32_t, int32_t>> sortSamples(std::vector<mlperf::QuerySample> const& samples) const;
    void startIssueThread(int32_t threadIdx);

    // If the function returns empty vector then there are no tasks remained and the caller should
    // exit
    GPTBatch getTasks(int32_t qThreadIdx);
    void processTasks(std::shared_ptr<GPTCore> gptCore, int32_t deviceId, int32_t qThreadIdx);

    void createEnginesPerGPU(int32_t deviceId, std::shared_ptr<std::mutex> pMtx, const std::vector<std::vector<char>>& trtModelStreams);

    std::string const mName;
    std::string const mEnginePath;

    // MPI attributes
    // MPI is used for host data collective commnunication
    bool mStartInfer;
    int32_t const mRank;
    int32_t const mNumWP;
    bool const mIsMain;
    bool const mIsGPTJ;

    // For each GPU device id, create a vector of ptrs to ICudaEngine
    std::vector<std::shared_ptr<nvinfer1::IRuntime>> mRuntimes;
    std::unordered_map<int32_t, std::vector<std::shared_ptr<nvinfer1::ICudaEngine>>> mEnginesPerGPU;

    // SUT attributes
    int32_t mMaxBatchSize;
    int32_t mTotalSampleComsumed;
    std::shared_ptr<qsl::SampleLibrary> mQsl;
    double mTargetLatencyPercentile;
    uint64_t mTotalTasksCount;
    bool mEnableSort;

    // Each query sample is accompanied by the time query arrived
    std::vector<std::deque<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>>> mTasksVec;

    // mutex to serialize access to mTasks member variable
    std::unique_ptr<std::vector<std::mutex>> mTasksVecMtxs;

    // The object to allow threads to avoid spinning on mMtx and mTasks for the new work to arrive
    std::unique_ptr<std::vector<std::condition_variable>> mTasksVecCondVars;

    // Indicates that there will no new tasks and the worker threads should stop processing samples
    bool mStopGetTasks;

    // mutex for both mTotalTasksCount and mSoftDropCount
    // std::mutex mSoftDropMtx;

    std::map<std::thread::id, int32_t> mIssueQueryThreadMap;
    std::vector<std::thread> mIssueQueryThreads; // one issue thread for each taskqueue
    std::vector<std::thread> mWorkerThreads;     // one worker thread for each gpt core
};

#endif // GPT_SERVER_HPP
