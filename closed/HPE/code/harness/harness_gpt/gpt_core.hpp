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

#ifndef GPT_CORE_HPP
#define GPT_CORE_HPP

#include "gpt_utils.hpp"
#include "half.h"
#include "nccl.h"
#include "qsl.hpp"
#include "system_under_test.h"
#include <mpi.h>

#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tr = tensorrt_llm::runtime;

using GPTBatch = std::vector<std::pair<mlperf::QuerySample, std::chrono::high_resolution_clock::time_point>>;
using GPTMask = std::array<int32_t, kGPTJ_MAX_INPUT_LENGTH>;

struct GPTResponse
{
    std::vector<mlperf::QuerySampleResponse> QSRs;
    // cuda event for host buffer ready
    cudaEvent_t resultReady;
    // cuda stream handles the device host copy
    int32_t copyStreamIdx;
    // all sequences in the batch are finsihed and the copy stream is ready to be reused by another batch
    bool copyStreamFinished{false};
};

//! This class manages the execution context, data buffers, cuda streams, and response queues for TRT LLM engines
//! Each GPTCore runs on a single inference stream but can have multiple instances of copy streams and corresponding
//! data buffers
class GPTCore
{
public:
    GPTCore(int32_t batchSize, int32_t numTensorParallelism, int32_t beamWidth,
        std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> const& engines, int32_t numCopyStreams, int32_t numGPTCores,
        int32_t gptCoreIdx, int32_t mpiRank, int32_t numRanks, int32_t deviceId, bool verboseNVTX, bool isFp8,
        bool isMain, bool isGPTJ);
    ~GPTCore();
    void warmUp();
    void infer(GPTBatch const& tasks, std::shared_ptr<qsl::SampleLibrary> qsl);
    // void BuildGraphs(int graphMaxSeqLen, const std::string& graphSpecs, int numBERTCores);
    // static int CountTotalLength(const BERTTask_t& tasks, std::shared_ptr<qsl::SampleLibrary> qsl);

    size_t getMaxBatchSize() const
    {
        return mMaxBatchSize;
    };

    void waitUntilQueueEmpty()
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mResponseQueueCondVar.wait(lck, [&]() { return mResponseQueue.empty(); });
    }

    int32_t const mDeviceId;

private:
    int64_t getTotalGPUMemoryInMiB() const;
    void setupGPTModel(std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> const& engines, int32_t numTensorParallelism,
        int32_t beamWidth);
    void setupDecoder(int32_t beamWidth, int32_t minLength);
    void stageHostBuffers(GPTBatch const& tasks, std::shared_ptr<qsl::SampleLibrary> qsl, int32_t& actualBatchSize,
        int32_t& maxInputLength, int32_t step, GPTManagedBuffer& gptBuffers);
    void stageDeviceBuffers(int32_t actualBatchSize, int32_t maxInputLength, int32_t step, cudaStream_t copyStream,
        GPTManagedBuffer& gptBuffers);
    void broadcastDeviceBuffers(
        int32_t actualBatchSize, int32_t maxInputLength, int32_t step, GPTManagedBuffer& gptBuffers);
    void tileDeviceBuffers(
        int32_t const actualBatchSize, int32_t const maxInputLength, int32_t const step, GPTManagedBuffer& gptBuffers);
    void stageDecoderBuffers(
        int32_t const actualBatchSize, int32_t const maxInputLength, int32_t step, GPTManagedBuffer& gptBuffers);
    void finalizeDecoderOutput(int32_t const actualBatchSize, int32_t const maxInputLength, int32_t const step,
        GPTManagedBuffer& gptBuffers, CopyStream& stream);
    std::shared_ptr<nvinfer1::IExecutionContext> getContext(int32_t inputLength, bool isContextPhase = true);
    // MPI attributes
    // MPI is used for host data collective commnunication
    int32_t const mRank;
    bool const mIsMain;
    bool const mIsGPTJ;

    // NCCL attributes
    // NCCL is used for device data collective commnunication
    ncclComm_t mNCCLComm;

    // GPT Model attributes
    bool const mIsFp8;
    std::unique_ptr<GPTModel> mGPTModel;

    // TensorRT engine context attributes
    int32_t mMaxBatchSize;
    std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> mContextVec;
    // TODO for length based context + profile mapping
    std::map<int32_t, std::shared_ptr<nvinfer1::IExecutionContext>> mInputLenToContext;

    // TRTLLM decoder
    tr::SamplingConfig mSamplingConfig;
    std::unique_ptr<tr::IGptDecoder> mDecoder;
    std::shared_ptr<tr::CudaStream> mDecoderStream;
    std::unique_ptr<tr::BufferManager> mDecoderBufferManager;
    std::unique_ptr<tr::DecodingInput> mDecoderInput;
    std::unique_ptr<tr::DecodingOutput> mDecoderOutput;

    // buffer attributes
    // we create a context per profile but share the device memory between them, assuming that we use only one context
    // at a time
    std::shared_ptr<lwis::DeviceBuffer> mContextBuf;
    // we need one buffer per copy stream
    std::vector<GPTManagedBuffer> mGPTBufferVec;

    // GPU stream attributes
    cudaStream_t mInferStream;
    std::vector<CopyStream> mCopyStreamVec;
    SyncQueue<int32_t> mCopyStreamIdxQueue;

    // threading attributes
    // GPTCore manages a pool of threads to process the responses asynchronously wrt. the inference
    std::vector<std::thread> mResponseThreadVec;
    std::deque<GPTResponse> mResponseQueue;
    std::mutex mMtx;
    std::condition_variable mResponseQueueCondVar;
    bool mStopWork;

    void enqueueResponse(const GPTResponse& r);
    // Logic of a response thread. Calls to QSL to write out the response
    static void processResponse(GPTCore* GPTCore, int32_t const deviceId);
};

#endif // GPT_CORE_HPP
