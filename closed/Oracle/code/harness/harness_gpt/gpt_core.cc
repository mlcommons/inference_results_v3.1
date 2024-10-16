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

#include "gpt_core.hpp"
#include "glog/logging.h"
#include "gpt_server.hpp"
#include "gpt_utils.hpp"
#include "loadgen.h"
#include "nccl.h"

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <cuda_fp16.h>
#include <fstream>
#include <mpi.h>
#include <numeric>
#include <set>
#include <unordered_set>

#undef CUDA_GRAPH_STATS

namespace tr = tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

int64_t GPTCore::getTotalGPUMemoryInMiB() const
{
    // return total memory in MiB
    cudaDeviceProp properties;
    CHECK_EQ(cudaGetDeviceProperties(&properties, mDeviceId), CUDA_SUCCESS);
    // totalGlobalMem is in Bytes
    return properties.totalGlobalMem / (1024 * 1024);
}

GPTCore::GPTCore(int32_t batchSize, int32_t numTensorParallelism, int32_t beamWidth,
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> const& engines, int32_t numCopyStreams, int32_t numGPTCores,
    int32_t gptCoreIdx, int32_t mpiRank, int32_t numRanks, int32_t deviceId, bool verboseNVTX, bool isFp8, bool isMain,
    bool isGPTJ)
    : mCopyStreamVec(isMain ? numCopyStreams : 0U) // created copy stream on main process
    , mRank(mpiRank)
    , mDeviceId(deviceId)
    , mIsMain(isMain)
    , mIsGPTJ(isGPTJ)
    , mStopWork(false)
    , mIsFp8(isFp8)
    , mMaxBatchSize(batchSize)
{
    LOG(INFO) << "GPTCore " << gptCoreIdx << ": MPI Rank - " << mRank << " at Device Id - " << mDeviceId;
    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);

    if (!mIsGPTJ)
    {
        // NCCL initialize
        ncclUniqueId id;
        if (mIsMain)
        {
            ncclGetUniqueId(&id);
        }
        MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));
        NCCLCHECK(ncclCommInitRank(&mNCCLComm, numRanks, id, mRank));
    }

    // create infer stream
    CHECK_EQ(cudaStreamCreate(&mInferStream), cudaSuccess);

    // setup mGPTModel
    setupGPTModel(engines, numTensorParallelism, beamWidth);

    // get the maximum device memory requirement
    auto maxDevMemSize = (*std::max_element(engines.begin(), engines.end(), [](auto ePtr1, auto ePtr2) {
        return (ePtr1->getDeviceMemorySize() < ePtr2->getDeviceMemorySize());
    }))->getDeviceMemorySize();
    mContextBuf = std::make_shared<lwis::DeviceBuffer>(maxDevMemSize);
    LOG(INFO) << "Engine - Device Memory requirements: " << maxDevMemSize;

    // create contexts from profiles of each engine
    for (auto engine : engines)
    {
        int32_t numProfilesTotal = engine->getNbOptimizationProfiles();
        int32_t numProfilesPerCore = engine->getNbOptimizationProfiles() / numGPTCores;
        int32_t profileStartIdx = gptCoreIdx * numProfilesPerCore;
        CHECK(numProfilesTotal % numGPTCores == 0) << "Total profile count and GPT core count mismatch!";
        LOG(INFO) << "Engine - Total Number of Optimization Profiles: " << numProfilesTotal;
        LOG(INFO) << "Engine - Number of Optimization Profiles Per Core: " << numProfilesPerCore;
        LOG(INFO) << "Engine - Start Index of Optimization Profiles: " << profileStartIdx;

        // create context and set optimization profile
        for(int32_t idx = profileStartIdx; idx < profileStartIdx + numProfilesPerCore; ++idx)
        {
            auto tmpContext = InferObject(engine->createExecutionContextWithoutDeviceMemory());
            if (verboseNVTX)
            {
                tmpContext->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
            }
            else
            {
                tmpContext->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
            }
            LOG(INFO) << "Setting Opt.Prof. to " << idx;
            tmpContext->setOptimizationProfile(idx);

            tmpContext->setDeviceMemory(mContextBuf->data());
            mContextVec.emplace_back(tmpContext);
            CHECK_EQ(tmpContext->getOptimizationProfile(), idx);
        }
    }

    // TODO zhihanj: opt1 dynamic allocation
    // allocate buffers and reset free copy streams queue
    for (size_t idx = 0; idx < numCopyStreams; idx++)
    {
        mGPTBufferVec.emplace_back(mMaxBatchSize, mGPTModel, mIsFp8, mIsMain);
        mGPTBufferVec.back().memsetDeviceAll(0);
        if (mIsMain) // Worker process should not use copy streams
        {
            mCopyStreamIdxQueue.push_back(idx);
        }
    }

    // set up TRTLLM gpt decoder
    setupDecoder(beamWidth, /*minLength = */ 30);

    // create async response thread on main process
    for (int32_t idx = 0; mIsMain && idx < kNUM_RESPONSE_THREADS; idx++)
    {
        mResponseThreadVec.emplace_back(GPTCore::processResponse, this, mDeviceId);
    }
    LOG(INFO) << "Setup complete";
}

void GPTCore::setupGPTModel(
    std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> const& engines, int32_t numTensorParallelism, int32_t beamWidth)
{
    // check engine integrity
    CHECK(engines.size() == 1) << "Multiple engines not supported yet";
    bool const isCorrectTensorIONum{
        engines[0]->getNbIOTensors() == (mIsGPTJ ? kNUM_GPTJ_ENGINE_IO_TENSOR : kNUM_GPT175_ENGINE_IO_TENSOR)};
    CHECK(isCorrectTensorIONum) << "Number of engines' IO tensors is not correct. Expecting "
                                << (mIsGPTJ ? kNUM_GPTJ_ENGINE_IO_TENSOR : kNUM_GPT175_ENGINE_IO_TENSOR)
                                << " IO tensors. But got " << engines[0]->getNbIOTensors() << " IO tensors.";

    // load engine tensor names and check model attributes
    std::string inputTokenName;
    std::string inputLengthName;
    std::string sequenceLengthName;
    std::string positionTokenName;
    std::string lastTokenName;
    std::string maskTokenName;
    std::vector<std::string> inputKVCacheNames;
    std::string KVCacheLengthTensorName;
    std::vector<std::string> cacheIndirectionNames;
    std::string maxInputLengthName;
    std::string outputLogitName;
    std::string outputTokenName{kGPT_OUTPUT_TOKEN_NAME};
    std::vector<std::string> outputKVCacheNames;
    // tensor name is hardcoded by TRT LLM
    for (int32_t i = 0; i < engines[0]->getNbIOTensors(); ++i)
    {
        std::string tensorName = engines[0]->getIOTensorName(i);
        if (tensorName.find("input_ids") != std::string::npos)
        {
            inputTokenName = tensorName;
        }
        else if (tensorName.find("input_lengths") != std::string::npos)
        {
            inputLengthName = tensorName; // the real length of each input sequences
        }
        else if (tensorName.find("sequence_length") != std::string::npos)
        {
            sequenceLengthName = tensorName; // The padded sequence length of the inputs
        }
        else if (tensorName.find("past_key_value_length") != std::string::npos)
        {
            KVCacheLengthTensorName = tensorName;
        }
        else if (tensorName.find("position_ids") != std::string::npos)
        {
            positionTokenName = tensorName;
        }
        else if (tensorName.find("last_token_ids") != std::string::npos)
        {
            lastTokenName = tensorName;
        }
        else if (tensorName.find("masked_tokens") != std::string::npos)
        {
            maskTokenName = tensorName;
        }
        else if (tensorName.find("past_key_value") != std::string::npos)
        {
            inputKVCacheNames.emplace_back(tensorName);
        }
        else if (tensorName.find("present_key_value") != std::string::npos)
        {
            outputKVCacheNames.emplace_back(tensorName);
        }
        else if (tensorName.find("logits") != std::string::npos)
        {
            outputLogitName = tensorName;
        }
        else if (tensorName.find("max_input_length") != std::string::npos)
        {
            maxInputLengthName = tensorName;
        }
        else if (tensorName.find("cache_indirection") != std::string::npos)
        {
            cacheIndirectionNames.push_back(tensorName + "_0");
            cacheIndirectionNames.push_back(tensorName + "_1");
        }
        else
        {
            LOG(ERROR) << "Engine tensor not recognized!";
        }
    }
    // read the model parameters from engine
    int32_t const numKVCache = inputKVCacheNames.size();
    int32_t const numHeads = (engines[0]->getTensorShape(inputKVCacheNames[0].c_str())).d[2]; // (-1, 2, nhead, -1, dhead)
    int32_t const dimHeads = (engines[0]->getTensorShape(inputKVCacheNames[0].c_str())).d[4]; // (-1, 2, nhead, -1, dhead)
    int32_t const dimVocab = (engines[0]->getTensorShape(outputLogitName.c_str())).d[1];      // (-1, vocab size)
    int32_t const maxInputLength
        = (engines[0]->getProfileShape(inputTokenName.c_str(), 0, nvinfer1::OptProfileSelector::kMAX))
              .d[1]; // (max bs, max input_seqlen)
    nvinfer1::Dims maxCacheIndirectionDim
        = engines[0]->getProfileShape("cache_indirection", 0, nvinfer1::OptProfileSelector::kMAX);
    // (max bs, beam size, max sum_seqlen)
    int32_t const maxBatchSize = maxCacheIndirectionDim.d[0];
    int32_t const maxBeamWidth = maxCacheIndirectionDim.d[1];
    int32_t const maxSumLength = maxCacheIndirectionDim.d[2];
    int32_t const kGPTMaxSeqLength = mIsGPTJ ? kGPTJ_MAX_SEQ_LENGTH : kGPT175_MAX_SEQ_LENGTH;

    // validate the engine profile
    CHECK(maxBeamWidth == beamWidth) << "Profile beam size: " << maxBeamWidth
                                     << " unmatches GPT beam size: " << beamWidth;
    CHECK(maxSumLength == kGPTMaxSeqLength)
        << "Profile max sum length: " << maxSumLength << " unmatches GPT max seq length: " << kGPTMaxSeqLength;
    CHECK((numKVCache == inputKVCacheNames.size() && numKVCache == outputKVCacheNames.size()))
        << "KV Cache layer number mismatch!";
    CHECK((kGPT_MAX_OUTPUT_LENGTH + maxInputLength == maxSumLength))
        << "Profile max sum length - max input length != max ouptut length!";

    // set the model parameters
    mGPTModel.reset(new GPTModel{inputTokenName, inputLengthName, sequenceLengthName, KVCacheLengthTensorName,
        positionTokenName, lastTokenName, maskTokenName, inputKVCacheNames, outputKVCacheNames, cacheIndirectionNames,
        maxInputLengthName, outputLogitName, outputTokenName, mMaxBatchSize, numKVCache, numHeads, dimHeads,
        numTensorParallelism, maxInputLength, kGPT_MAX_OUTPUT_LENGTH, maxSumLength, dimVocab, beamWidth});

    LOG(INFO) << "Engine - Vocab size: " << mGPTModel->mDimVocab << " Padded vocab size: " << mGPTModel->mDimVocabPadded
              << " Beam width: " << mGPTModel->mBeamWidth;
}

void GPTCore::setupDecoder(int32_t beamWidth, int32_t minLength)
{
    int32_t const maxBatchSize = mGPTModel->mMaxBatchSize;
    mSamplingConfig.beamWidth = beamWidth;
    mSamplingConfig.temperature = std::vector(maxBatchSize, 1.0f);
    mSamplingConfig.minLength = std::vector(maxBatchSize, minLength);
    mSamplingConfig.randomSeed = std::vector(maxBatchSize, 0ull);
    mSamplingConfig.topK = std::vector(maxBatchSize, 1);
    mSamplingConfig.topP = std::vector(maxBatchSize, 1.0f);
    mSamplingConfig.lengthPenalty = 1.0f;
    CHECK(mSamplingConfig.beamWidth == mGPTModel->mBeamWidth) << "Sampling config and model beam width mismatch!";

    VLOG(2) << "Sampling config beam width: " << beamWidth;
    VLOG(2) << "Sampling config batch size: " << maxBatchSize;

    // setup buffer manager for decoder
    mDecoderStream = std::make_shared<tr::CudaStream>(mInferStream, mDeviceId, /*ownsStream =*/false);
    mDecoderBufferManager = std::make_unique<tr::BufferManager>(mDecoderStream);

    // create TRTLLM decoder
    mDecoder = tr::IGptDecoder::create(
        nvinfer1::DataType::kHALF, mGPTModel->mDimVocab, mGPTModel->mDimVocabPadded, mDecoderStream);
    mDecoder->setup(mSamplingConfig, mGPTModel->mMaxBatchSize);
}

void GPTCore::stageHostBuffers(GPTBatch const& tasks, std::shared_ptr<qsl::SampleLibrary> qsl, int32_t& actualBatchSize,
    int32_t& maxInputLength, int32_t step, GPTManagedBuffer& gptBuffers)
{
    CHECK(mIsMain) << "Only MP should interact with QSL and host buffers";
    int32_t const beamWidth = mGPTModel->mBeamWidth;
    if (step == 0) // context phase
    {
        // iterate through the batch and fetch input length
        actualBatchSize = tasks.size();
        std::vector<int32_t> inputLengthVec(actualBatchSize);
        for (int32_t i = 0; i < actualBatchSize; ++i)
        {
            inputLengthVec[i] = *static_cast<int32_t*>(qsl->GetSampleAddress(tasks[i].first.index, kINPUT_LENGTH_IDX));
        }
        maxInputLength = *std::max_element(inputLengthVec.begin(), inputLengthVec.end());
        VLOG(1) << "Longest input length in batch: " << maxInputLength;

        // initialize non-preprocessed values
        std::vector<int32_t> positionTokenVec(maxInputLength);
        std::iota(positionTokenVec.begin(), positionTokenVec.end(), 0);
        std::vector<int32_t> sequenceLengthVec(actualBatchSize, maxInputLength + step);
        std::vector<int32_t> KVCacheLengthTensorVec = {0, 1};

        for (int32_t i = 0; i < actualBatchSize; ++i)
        {
            // input_ids
            gptBuffers.H2H(mGPTModel->mInputTokenName, qsl->GetSampleAddress(tasks[i].first.index, kINPUT_TOKEN_IDX),
                maxInputLength,
                /*offset =*/i * maxInputLength);
            // position_ids
            gptBuffers.H2H(
                mGPTModel->mPositionTokenName, positionTokenVec.data(), maxInputLength, /*offset =*/i * maxInputLength);
            // masked_tokens
            auto maskPtr = static_cast<int32_t*>(qsl->GetSampleAddress(tasks[i].first.index, kINPUT_MASK_IDX));
            std::vector<int32_t> maskToken(maskPtr, maskPtr + maxInputLength);
            std::vector<int32_t> padMaskToken(mGPTModel->mMaxSumLength - maxInputLength, 0);
            maskToken.insert(maskToken.end(), padMaskToken.begin(), padMaskToken.end());
            gptBuffers.H2H(mGPTModel->mMaskTokenName, maskToken.data(), mGPTModel->mMaxSumLength,
                /*offset =*/i * mGPTModel->mMaxSumLength);
        }
        // input_lengths
        gptBuffers.H2H(mGPTModel->mInputLengthName, inputLengthVec.data(), inputLengthVec.size(), /*offset =*/0U);
        // last_token_ids
        gptBuffers.H2H(mGPTModel->mLastTokenName, inputLengthVec.data(), inputLengthVec.size(), /*offset =*/0U);
        // sequence_length
        gptBuffers.H2H(
            mGPTModel->mSequenceLengthName, sequenceLengthVec.data(), sequenceLengthVec.size(), /*offset =*/0U);
        // past_key_value_length
        gptBuffers.H2H(
            mGPTModel->mKVCacheLengthTensorName, KVCacheLengthTensorVec.data(), kGPT_SHAPE_TENSOR_DIM0, /*offset =*/0U);
        // cache_indirection
        for (auto const& cacheIndirectionName : mGPTModel->mCacheIndirectionNameVec)
        {
            gptBuffers.memsetHost(cacheIndirectionName, 0);
        }
        // max_input_length
        gptBuffers.memsetHost(mGPTModel->mMaxInputLengthName, 0);
    }
    else // generation phase
    {
        // initialize non-preprocessed values
        auto lengthPtr = static_cast<int32_t*>(gptBuffers.getHostBuffer(mGPTModel->mInputLengthName));
        std::vector<int32_t> positionTokenVec(
            lengthPtr, lengthPtr + actualBatchSize * beamWidth); // shape (BS * beam_width, 1)
        std::for_each(positionTokenVec.begin(), positionTokenVec.end(), [step](int32_t& v) { v += (step - 1); });
        std::vector<int32_t> sequenceLengthVec(
            actualBatchSize * beamWidth, maxInputLength + step - 1); // shape (BS * beam_width)
        std::vector<int32_t> KVCacheLengthTensorVec = {maxInputLength + step - 1, 0};

        // position_ids
        gptBuffers.H2H(mGPTModel->mPositionTokenName, positionTokenVec.data(), positionTokenVec.size(), /*offset =*/0U);
        // sequence_length
        gptBuffers.H2H(
            mGPTModel->mSequenceLengthName, sequenceLengthVec.data(), sequenceLengthVec.size(), /*offset =*/0U);
        // past_key_value_length
        gptBuffers.H2H(
            mGPTModel->mKVCacheLengthTensorName, KVCacheLengthTensorVec.data(), kGPT_SHAPE_TENSOR_DIM0, /*offset =*/0U);

        if (step < 2) // last_token_ids stays the same after 1 context + 1 generation step
        {
            // last_token_ids
            std::vector<int32_t> lastTokenVec(actualBatchSize * beamWidth, 1); // shape (BS * beam_width)
            gptBuffers.H2H(mGPTModel->mLastTokenName, lastTokenVec.data(), lastTokenVec.size(), /*offset =*/0U);
        }
    }
}

void GPTCore::stageDeviceBuffers(int32_t actualBatchSize, int32_t maxInputLength, int32_t step, cudaStream_t copyStream,
    GPTManagedBuffer& gptBuffers)
{
    CHECK(mIsMain) << "Only MP should stage device buffers";
    int32_t const beamWidth = mGPTModel->mBeamWidth;
    if (step == 0) // context phase
    {
        gptBuffers.memsetDeviceAll(0);
        // input_ids
        gptBuffers.H2DAsync(
            mGPTModel->mInputTokenName, /* shape (BS, seq_len) */ actualBatchSize * maxInputLength, copyStream);
        // input_lengths
        gptBuffers.H2DAsync(mGPTModel->mInputLengthName, /* shape (BS)*/ actualBatchSize, copyStream);
        // position_ids
        gptBuffers.H2DAsync(
            mGPTModel->mPositionTokenName, /* shape (BS, seq_len) */ actualBatchSize * maxInputLength, copyStream);
        // masked_tokens
        gptBuffers.H2DAsync(
            mGPTModel->mMaskTokenName, /* shape (BS, seq_len) */ actualBatchSize * mGPTModel->mMaxSumLength, copyStream);
        // cache_indirection
        for (auto const& cacheIndirectionName : mGPTModel->mCacheIndirectionNameVec)
        {
            gptBuffers.H2DAsync(cacheIndirectionName,
                /* shape [BS, beam_width, max_seqlen] */ actualBatchSize * beamWidth * mGPTModel->mMaxSumLength,
                copyStream);
        }
        // max_input_length
        gptBuffers.H2DAsync(mGPTModel->mMaxInputLengthName, maxInputLength, copyStream);
    }
    else // generation phase
    {
        // copy token from the output to the input
        // input_ids
        // offset = [max_input_lenghth + step - 1 : max_input_lenghth + step, bs, beam_width]
        size_t srcOffset = tc::flat_index3(maxInputLength + step - 1, 0, 0, actualBatchSize, beamWidth);
        gptBuffers.D2DAsync(mGPTModel->mOutputTokenName, mGPTModel->mInputTokenName, actualBatchSize * beamWidth,
            srcOffset, copyStream);
        // position_ids
        gptBuffers.H2DAsync(
            mGPTModel->mPositionTokenName, /* shape (BS * beam_width, 1) */ actualBatchSize * beamWidth, copyStream);
        // reuse input_length, masked_tokens device buffer in generation phase
    }
    if (step < 2) // last_token_ids stays the same after 1 context + 1 generation step
    {
        // last_token_ids
        gptBuffers.H2DAsync(
            mGPTModel->mLastTokenName, /* shape (BS * beam_width) */ actualBatchSize * beamWidth, copyStream);
    }
    // sequence_length
    gptBuffers.H2DAsync(
        mGPTModel->mSequenceLengthName, /* shape (BS * beam_width) */ actualBatchSize * beamWidth, copyStream);
}

//! TODO broadcast device buffer for GPT175
//! development postponed to 4.0
void GPTCore::broadcastDeviceBuffers(
    int32_t actualBatchSize, int32_t maxInputLength, int32_t step, GPTManagedBuffer& gptBuffers)
{
#ifdef MLPINF_40
    if (step == 0) // context phase
    {
        NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mInputTokenName),
            gptBuffers.getDeviceBuffer(mGPTModel->mInputTokenName), actualBatchSize * maxInputLength, ncclInt32, mRank,
            mNCCLComm, mInferStream));
        NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mInputLengthName),
            gptBuffers.getDeviceBuffer(mGPTModel->mInputLengthName), actualBatchSize, ncclInt32, mRank, mNCCLComm,
            mInferStream));
        NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mPositionTokenName),
            gptBuffers.getDeviceBuffer(mGPTModel->mInputLengthName), actualBatchSize * maxInputLength, ncclInt32, mRank,
            mNCCLComm, mInferStream));
        NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mMaskTokenName),
            gptBuffers.getDeviceBuffer(mGPTModel->mMaskTokenName), actualBatchSize * maxInputLength, ncclInt32, mRank,
            mNCCLComm, mInferStream));
    }
    else // generation phase
    {
        NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mInputTokenName),
            gptBuffers.getDeviceBuffer(mGPTModel->mInputTokenName), actualBatchSize * kGEN_LENGTH, ncclInt32, mRank,
            mNCCLComm, mInferStream));
        NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mPositionTokenName),
            gptBuffers.getDeviceBuffer(mGPTModel->mPositionTokenName), actualBatchSize, ncclInt32, mRank, mNCCLComm,
            mInferStream));
        // reuse input length, mask token device buffer in generation phase
    }
    if (step < 2) // last_token_ids stays the same after 1 context + 1 generation step
    {
        NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mLastTokenName),
            gptBuffers.getDeviceBuffer(mGPTModel->mLastTokenName), actualBatchSize, ncclInt32, mRank, mNCCLComm,
            mInferStream));
    }
    // sequence_length
    NCCLCHECK(ncclBroadcast(gptBuffers.getDeviceBuffer(mGPTModel->mSequenceLengthName),
        gptBuffers.getDeviceBuffer(mGPTModel->mSequenceLengthName), actualBatchSize, ncclInt32, mRank, mNCCLComm,
        mInferStream));
#endif
}

void GPTCore::tileDeviceBuffers(
    int32_t const actualBatchSize, int32_t const maxInputLength, int32_t const step, GPTManagedBuffer& gptBuffers)
{
    int32_t const beamWidth = mGPTModel->mBeamWidth;
    CHECK(step == 0 && beamWidth > 1)
        << "Buffer tiling should only be called once with beam width > 1 at the first generation phase.";
    VLOG(2) << "Tiling engine tensors";

    auto tileBuffer = [&beamWidth, &gptBuffers, this](std::string const& tensorName,
                          nvinfer1::Dims const& originalShape, nvinfer1::Dims const& tiledShape) {
        auto const dataType = gptBuffers.getDataType(tensorName);
        // wrap device buffer into target TRTLLM tensors
        auto tiledTensor
            = std::shared_ptr(tr::ITensor::wrap(gptBuffers.getDeviceBuffer(tensorName), dataType, tiledShape));
        // copy device buffer into original TRTLLM tensors
        std::shared_ptr<tr::ITensor> originalTensor;
        switch (dataType)
        {
        case nvinfer1::DataType::kINT32:
            originalTensor = std::shared_ptr(mDecoderBufferManager->copyFrom(
                static_cast<int32_t*>(gptBuffers.getDeviceBuffer(tensorName)), originalShape, tr::MemoryType::kGPU));
            tr::kernels::invokeTileTensor<int32_t>(*tiledTensor, *originalTensor, beamWidth, *mDecoderStream);
            break;
        case nvinfer1::DataType::kHALF:
            originalTensor = std::shared_ptr(mDecoderBufferManager->copyFrom(
                static_cast<half*>(gptBuffers.getDeviceBuffer(tensorName)), originalShape, tr::MemoryType::kGPU));
            tr::kernels::invokeTileTensor<half>(*tiledTensor, *originalTensor, beamWidth, *mDecoderStream);
            break;
        case nvinfer1::DataType::kINT8:
            originalTensor = std::shared_ptr(mDecoderBufferManager->copyFrom(
                static_cast<int8_t*>(gptBuffers.getDeviceBuffer(tensorName)), originalShape, tr::MemoryType::kGPU));
            tr::kernels::invokeTileTensor<int8_t>(*tiledTensor, *originalTensor, beamWidth, *mDecoderStream);
            break;
        default: CHECK(false) << "data type not supported";
        }
        mDecoderStream->synchronize(); // check if we need this
    };

    // tile
    tileBuffer(mGPTModel->mInputLengthName, tr::ITensor::makeShape({actualBatchSize}),
        tr::ITensor::makeShape({actualBatchSize * beamWidth}));
    tileBuffer(mGPTModel->mMaskTokenName, tr::ITensor::makeShape({actualBatchSize, mGPTModel->mMaxSumLength}),
        tr::ITensor::makeShape({actualBatchSize * beamWidth, mGPTModel->mMaxSumLength}));
    tileBuffer(mGPTModel->mOutputLogitName, tr::ITensor::makeShape({actualBatchSize, mGPTModel->mDimVocabPadded}),
        tr::ITensor::makeShape({actualBatchSize * beamWidth, mGPTModel->mDimVocabPadded}));

    for (auto const& inputKVCacheName : mGPTModel->mInputKVCacheNameVec)
    {
        tileBuffer(inputKVCacheName,
            tr::ITensor::makeShape(
                {actualBatchSize, 2, mGPTModel->mNumHeads, mGPTModel->mMaxSumLength, mGPTModel->mDimHeads}),
            tr::ITensor::makeShape(
                {actualBatchSize * beamWidth, 2, mGPTModel->mNumHeads, mGPTModel->mMaxSumLength, mGPTModel->mDimHeads}));
    }
    // no need to tile sequenceLengths and lastTokenIds because it is directly assigned in stageHost(Device)Buffers and
    VLOG(2) << "Finished tiling engine tensors";
}

void GPTCore::stageDecoderBuffers(
    int32_t const actualBatchSize, int32_t const maxInputLength, int32_t step, GPTManagedBuffer& gptBuffers)
{
    CHECK(mIsMain) << "Only MP should stage decoder buffers";
    int32_t const beamWidth{mSamplingConfig.beamWidth};
    if (step == 0)
    {
        VLOG(2) << "Reset host output sequence";
        gptBuffers.decoderOutputTokenResetToPad();

        // decoder input required
        auto logits = std::shared_ptr(tr::ITensor::wrap(gptBuffers.getDeviceBuffer(mGPTModel->mOutputLogitName),
            gptBuffers.getDataType(mGPTModel->mOutputLogitName),
            tr::ITensor::makeShape({actualBatchSize, beamWidth, mGPTModel->mDimVocabPadded})));
        std::vector<int32_t> const endIdsVec(actualBatchSize * beamWidth, kGPTJ_END_ID);
        auto endIds = std::shared_ptr(mDecoderBufferManager->copyFrom(
            endIdsVec.data(), tr::ITensor::makeShape({actualBatchSize, beamWidth}), tr::MemoryType::kGPU));

        mDecoderInput.reset(
            new tr::DecodingInput{maxInputLength + step, maxInputLength, actualBatchSize, logits, endIds});

        std::vector<int32_t> squenceLimitLengthsVec(actualBatchSize, mGPTModel->mMaxSumLength);
        mDecoderInput->sequenceLimitLength = mDecoderBufferManager->copyFrom(
            squenceLimitLengthsVec.data(), tr::ITensor::makeShape({actualBatchSize}), tr::MemoryType::kGPU);

        mDecoderInput->lengths = std::shared_ptr(tr::ITensor::wrap(
            gptBuffers.getDeviceBuffer(mGPTModel->mInputLengthName),
            gptBuffers.getDataType(mGPTModel->mInputLengthName), tr::ITensor::makeShape({actualBatchSize, beamWidth})));

        // decoder output required
        auto outputTokens = std::shared_ptr(tr::ITensor::wrap(gptBuffers.getDeviceBuffer(mGPTModel->mOutputTokenName),
            gptBuffers.getDataType(mGPTModel->mOutputTokenName),
            tr::ITensor::makeShape({mGPTModel->mMaxSumLength, actualBatchSize, beamWidth})));
        mDecoderOutput.reset(new tr::DecodingOutput{outputTokens});

        std::vector<int32_t> sequenceLengthsVec(actualBatchSize * beamWidth, maxInputLength);
        mDecoderOutput->lengths = mDecoderBufferManager->copyFrom(
            sequenceLengthsVec.data(), tr::ITensor::makeShape({actualBatchSize * beamWidth}), tr::MemoryType::kGPU);

        // decoder output beam search required
        auto finished = std::shared_ptr(mDecoderBufferManager->gpu(
            tr::ITensor::makeShape({actualBatchSize * beamWidth}), nvinfer1::DataType::kBOOL));
        mDecoderBufferManager->setZero(*finished);
        mDecoderOutput->finished = finished;

        if (beamWidth > 1)
        {
            std::vector<float> cumLogProbsVec(actualBatchSize * beamWidth, -1e20f);
            for (int32_t b = 0; b < actualBatchSize; ++b)
            {
                cumLogProbsVec[b * beamWidth] = 0;
            }
            mDecoderOutput->cumLogProbs = mDecoderBufferManager->copyFrom(
                cumLogProbsVec.data(), tr::ITensor::makeShape({actualBatchSize * beamWidth}), tr::MemoryType::kGPU);

            auto parentIds = std::shared_ptr(mDecoderBufferManager->gpu(
                tr::ITensor::makeShape({mGPTModel->mMaxSumLength, actualBatchSize, beamWidth}),
                nvinfer1::DataType::kINT32));
            mDecoderBufferManager->setZero(*parentIds);
            mDecoderOutput->parentIds = parentIds;
        }

        // HF beamhyps
        mBeamHyps.outputIdsTgt = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kINT32);
        mBeamHyps.sequenceLengthsTgt = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kINT32);
        mBeamHyps.cumLogProbs = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
        mBeamHyps.normedScores = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
        mBeamHyps.logProbs = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
        mBeamHyps.minNormedScores = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
        mBeamHyps.numBeams = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kINT32);
        mBeamHyps.isDone = mDecoderBufferManager->emptyTensor(tr::MemoryType::kGPU, nvinfer1::DataType::kBOOL);

        mBeamHyps.outputIdsTgt->reshape(tr::ITensor::makeShape({actualBatchSize, 2 * beamWidth, kGPTJ_MAX_SEQ_LENGTH}));
        mBeamHyps.sequenceLengthsTgt->reshape(tr::ITensor::makeShape({actualBatchSize, 2 * beamWidth}));
        mBeamHyps.cumLogProbs->reshape(tr::ITensor::makeShape({actualBatchSize, 2 * beamWidth}));
        mBeamHyps.normedScores->reshape(tr::ITensor::makeShape({actualBatchSize, 2 * beamWidth}));
        mBeamHyps.logProbs->reshape(tr::ITensor::makeShape({actualBatchSize, 2 * beamWidth, kGPTJ_MAX_SEQ_LENGTH}));
        mBeamHyps.minNormedScores->reshape(tr::ITensor::makeShape({actualBatchSize}));
        mBeamHyps.numBeams->reshape(tr::ITensor::makeShape({actualBatchSize}));
        mBeamHyps.isDone->reshape(tr::ITensor::makeShape({actualBatchSize}));

        tr::kernels::invokeFill(*(mBeamHyps.outputIdsTgt), kGPTJ_END_ID, *mDecoderStream);
        mDecoderBufferManager->setZero(*(mBeamHyps.sequenceLengthsTgt));
        mDecoderBufferManager->setZero(*(mBeamHyps.cumLogProbs));
        mDecoderBufferManager->setZero(*(mBeamHyps.normedScores));
        mDecoderBufferManager->setZero(*(mBeamHyps.logProbs));
        mDecoderBufferManager->setZero(*(mBeamHyps.minNormedScores));
        mDecoderBufferManager->setZero(*(mBeamHyps.numBeams));
        mDecoderBufferManager->setZero(*(mBeamHyps.isDone));
        mDecoderOutput->beamHyps = mBeamHyps;
    }
    else
    {
        mDecoderInput->step = maxInputLength + step;
    }
    // ping pong cache indirection
    // TODO maybe try std::swap/std::move in the future? Not yet allowed by TRTLLM July release
    auto srcCacheIndirection = std::shared_ptr(
        tr::ITensor::wrap(gptBuffers.getDeviceBuffer(mGPTModel->mCacheIndirectionNameVec[1 - step % 2]),
            gptBuffers.getDataType(mGPTModel->mCacheIndirectionNameVec[1 - step % 2]),
            tr::ITensor::makeShape({actualBatchSize, beamWidth, mGPTModel->mMaxSumLength})));
    mDecoderInput->cacheIndirection = srcCacheIndirection;

    auto tgtCacheIndirection
        = std::shared_ptr(tr::ITensor::wrap(gptBuffers.getDeviceBuffer(mGPTModel->mCacheIndirectionNameVec[step % 2]),
            gptBuffers.getDataType(mGPTModel->mCacheIndirectionNameVec[step % 2]),
            tr::ITensor::makeShape({actualBatchSize, beamWidth, mGPTModel->mMaxSumLength})));
    mDecoderOutput->cacheIndirection = tgtCacheIndirection;
}

// Run gather tree for beam search and copy GPU output token from GptDecoder to output vector on host
void GPTCore::finalizeDecoderOutput(int32_t const actualBatchSize, int32_t const maxInputLength, int32_t const step,
    GPTManagedBuffer& gptBuffers, CopyStream& copyStream)
{
    CHECK(mIsMain) << "finalizeDecoderOutput() should only be called by the main process";
    int32_t const beamWidth{mGPTModel->mBeamWidth};
    bool const isBeamSearch{beamWidth > 1};
    int32_t const finalSeqLength{kGPTJ_MAX_SEQ_LENGTH};
    void* finalOutputTokenPtr;
    // additional output buffer for beam search
    auto gatheredOutputIds = mDecoderBufferManager->gpu(
        tr::ITensor::makeShape({actualBatchSize, beamWidth, finalSeqLength}), nvinfer1::DataType::kINT32);
    mDecoderBufferManager->setZero(*gatheredOutputIds);
    // run gather tree for beam search and store in gatheredOutputIds
    if (isBeamSearch)
    {
        VLOG(2) << "Invoking gather tree op";
        tensorrt_llm::kernels::invokeInitializeOutPut(tr::bufferCast<int32_t>(*gatheredOutputIds), tr::bufferCast<int32_t>(*(mDecoderInput->endIds)),
            actualBatchSize * beamWidth, finalSeqLength, mDecoderStream->get());
        mDecoderStream->synchronize(); // check if we need this

        tensorrt_llm::kernels::BeamHypotheses beamHyps;
        beamHyps.sequence_lengths_src = tr::bufferCast<int32_t>(*(mDecoderOutput->lengths));
        beamHyps.parent_ids_src = tr::bufferCast<int32_t>(*(mDecoderOutput->parentIds));
        beamHyps.output_ids_src = tr::bufferCast<int32_t>(*(mDecoderOutput->ids));
        beamHyps.log_probs_src = nullptr;
        beamHyps.max_seq_len = finalSeqLength;
        beamHyps.length_penalty = 1.0f;

        beamHyps.output_ids_tgt = tr::bufferCast<int32_t>(*(mBeamHyps.outputIdsTgt));
        beamHyps.sequence_lengths_tgt = tr::bufferCast<int32_t>(*(mBeamHyps.sequenceLengthsTgt));
        beamHyps.cum_log_probs = tr::bufferCast<float>(*(mBeamHyps.cumLogProbs));
        beamHyps.normed_scores = tr::bufferCast<float>(*(mBeamHyps.normedScores));
        beamHyps.log_probs = tr::bufferCast<float>(*(mBeamHyps.logProbs));
        beamHyps.min_normed_scores = tr::bufferCast<float>(*(mBeamHyps.normedScores));
        beamHyps.num_beams = tr::bufferCast<int32_t>(*(mBeamHyps.numBeams));
        beamHyps.is_done = tr::bufferCast<bool>(*(mBeamHyps.isDone));
        beamHyps.input_lengths = tr::bufferCast<int32_t>(*(mDecoderInput->lengths));

        tensorrt_llm::kernels::invokeInsertUnfinishedPath(beamHyps, tr::bufferCast<bool>(*(mDecoderOutput->finished)),
            tr::bufferCast<float>(*(mDecoderOutput->cumLogProbs)), actualBatchSize, beamWidth, mDecoderStream->get());
        mDecoderStream->synchronize(); // check if we need this

        tensorrt_llm::kernels::invokeFinalize(
            tr::bufferCast<int32_t>(*gatheredOutputIds),
            tr::bufferCast<int32_t>(*(mDecoderOutput->lengths)),
            tr::bufferCast<float>(*(mDecoderOutput->cumLogProbs)),
            nullptr, // output_logs
            beamHyps.output_ids_tgt, beamHyps.sequence_lengths_tgt, beamHyps.normed_scores, beamHyps.cum_log_probs,
            beamHyps.log_probs, beamHyps.num_beams, beamHyps.input_lengths, beamWidth, finalSeqLength, actualBatchSize,
            maxInputLength, mDecoderStream->get());
        mDecoderStream->synchronize(); // check if we need this

        VLOG(2) << "Finished gather tree op";
        finalOutputTokenPtr = gatheredOutputIds->data();
    }
    else
    {
        finalOutputTokenPtr = gptBuffers.getDeviceBuffer(mGPTModel->mOutputTokenName);
    }

    copyStream.recordInferDone(mInferStream);
    copyStream.awaitInfer();

    for (int32_t i = 0; i < actualBatchSize; ++i)
    {
        int32_t const realInputLength = static_cast<int32_t*>(gptBuffers.getHostBuffer(mGPTModel->mInputLengthName))[i];
        for (int32_t id = 0; id < step && id < kGPT_MAX_OUTPUT_LENGTH; ++id)
        {
            int32_t elementSize = lwis::getElementSize(GPTOutputTokenDataType);
            size_t const dstOffset = id;
            // [batch_max_length + id, i, beamWidth] for greedy
            // [i, 0, batch_max_length + id] for beam search
            size_t const srcOffset = isBeamSearch
                ? tc::flat_index3(i, 0, realInputLength + id, beamWidth, finalSeqLength)
                : tc::flat_index3(maxInputLength + id, i, 0, actualBatchSize, mGPTModel->mBeamWidth);
            ;
            CHECK_EQ(
                cudaMemcpyAsync(static_cast<char*>(gptBuffers.getHostOutputTokenBuffer(i)) + dstOffset * elementSize,
                    static_cast<char*>(finalOutputTokenPtr) + srcOffset * elementSize, kGEN_LENGTH * elementSize,
                    cudaMemcpyDeviceToHost, copyStream.get()),
                cudaSuccess);
        }
    }
}

//! GPTCore::getContext is a place holder function for future input length based optimization profile
std::shared_ptr<nvinfer1::IExecutionContext> GPTCore::getContext(int32_t inputLength, bool isContextPhase)
{
    // only 1 or 2 opt profiles and contexts supported so far
    if (mContextVec.size() == 1)
    {
        return mContextVec[0];
    }
    else
    {
        int32_t optProfileIdx = isContextPhase ? 0 : 1;
        return mContextVec[optProfileIdx];
    }
}

void GPTCore::infer(GPTBatch const& tasks, std::shared_ptr<qsl::SampleLibrary> qsl)
{
    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);
    bool shouldStop{false};
    NVTX_THREAD_START("Context", nvtx::COLOR_PINK_0);
    // ====================== CONTEXT PHASE ======================
    int32_t actualBatchSize;
    int32_t maxInputLength;
    int32_t copyStreamIdx;
    int32_t step{0};

    // TODO v4.0 please test MP locks!!!
    // MP context phase get copy stream
    if (mIsMain)
    {
        // get free copy stream idx from sync queue
        copyStreamIdx = mCopyStreamIdxQueue.front_then_pop();
        CHECK(copyStreamIdx < mCopyStreamVec.size() && copyStreamIdx < mGPTBufferVec.size())
            << "Copy stream index out of range";
    }

    // get buffer to use
    MPICHECK(MPI_Bcast(&copyStreamIdx, sizeof(copyStreamIdx), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));
    auto& gptBuffers = mGPTBufferVec[copyStreamIdx];

    // MP prepare batch
    if (mIsMain)
    {
        VLOG(2) << "Starting context step: " << step;
        auto& copyStream = mCopyStreamVec[copyStreamIdx];
        // stage host buffers from qsl
        stageHostBuffers(tasks, qsl, actualBatchSize, maxInputLength, step, gptBuffers);
        // get device buffer data prepared by copy streams
        stageDeviceBuffers(actualBatchSize, maxInputLength, step, copyStream.get(), gptBuffers);
        copyStream.recordH2D();
        copyStream.makeAwaitH2D(mInferStream);

#ifdef OUTPUT_DEBUG_DETAILED
        // CHECK_EQ(cudaStreamSynchronize(copyStream.get()), cudaSuccess);
        gptBuffers.dumpInputHost(step, mGPTModel->mInputLengthName, actualBatchSize, 1);
        gptBuffers.dumpInputHost(step, mGPTModel->mInputTokenName, actualBatchSize, maxInputLength);
        gptBuffers.dumpInputHost(step, mGPTModel->mSequenceLengthName, actualBatchSize, 1);
        gptBuffers.dumpInputHost(step, mGPTModel->mPositionTokenName, actualBatchSize, maxInputLength);
        gptBuffers.dumpInputHost(step, mGPTModel->mLastTokenName, actualBatchSize, 1);
        gptBuffers.dumpInputHost(step, mGPTModel->mMaskTokenName, actualBatchSize, mGPTModel->mMaxSumLength);
#endif
    }

    // sync batch info
    MPICHECK(MPI_Bcast(&actualBatchSize, sizeof(actualBatchSize), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(&maxInputLength, sizeof(maxInputLength), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));

    // prepare TRT engine tensors
    // TODO now only support fixed sequence length
    auto context = getContext(maxInputLength, /*isContextPhase = */ true);
    mGPTModel->setInputShapes(context, actualBatchSize, maxInputLength, step, /*isContext = */ true);
    mGPTModel->setTensorAddresses(context, gptBuffers, step);

    // NCCL sync device buffer for GPT 175
    if (!mIsGPTJ)
    {
        broadcastDeviceBuffers(actualBatchSize, maxInputLength, step, gptBuffers);
    }

    // run transformers
    CHECK_EQ(context->enqueueV3(mInferStream), true);

    if (mIsMain)
    {
        auto& copyStream = mCopyStreamVec[copyStreamIdx];
        if (mGPTModel->mBeamWidth > 1)
        {
            // tile buffer after the context phase for beam search
            tileDeviceBuffers(actualBatchSize, maxInputLength, step, gptBuffers);
        }
        // run decoder
        stageDecoderBuffers(actualBatchSize, maxInputLength, step, gptBuffers);
        shouldStop = mDecoder->forward(*mDecoderOutput, *mDecoderInput);
        CHECK_EQ(shouldStop, false);
        copyStream.recordInferDone(mInferStream);
        copyStream.awaitInfer();

#ifdef OUTPUT_DEBUG_STEP
        gptBuffers.dumpOutputTokens(step, actualBatchSize, mGPTModel->mBeamWidth, maxInputLength);
        if (mGPTModel->mBeamWidth > 1)
        {
            // debug cumProb
            std::string cumProbString = "[DEBUG] Output cumProb at step " + std::to_string(step) + ": [";
            auto cumProbHost = mDecoderBufferManager->copyFrom(*(mDecoderOutput->cumLogProbs), tr::MemoryType::kCPU);
            auto cumProbVec = tr::bufferCast<float>(*cumProbHost);
            mDecoderBufferManager->getStream().synchronize();
            for (int32_t b = 0; b < actualBatchSize; ++b)
            {
                cumProbString += "[";
                for (int32_t beam = 0; beam < mGPTModel->mBeamWidth; ++beam)
                {
                    int32_t offset = tc::flat_index2(b, beam, mGPTModel->mBeamWidth);
                    cumProbString += std::to_string(cumProbVec[offset]) + ", ";
                }
                cumProbString += "]";
                if (b != actualBatchSize - 1)
                {
                    cumProbString += "\n";
                }
            }
            cumProbString += "]\n";
            LOG(INFO) << cumProbString;
            // debug finished
            std::string finishedString = "[DEBUG] Output finished at step " + std::to_string(step) + ": [";
            auto finishedHost = mDecoderBufferManager->copyFrom(*(mDecoderOutput->finished), tr::MemoryType::kCPU);
            auto finishedVec = tr::bufferCast<bool>(*finishedHost);
            mDecoderBufferManager->getStream().synchronize();
            for (int32_t b = 0; b < actualBatchSize; ++b)
            {
                finishedString += "[";
                for (int32_t beam = 0; beam < mGPTModel->mBeamWidth; ++beam)
                {
                    int32_t offset = tc::flat_index2(b, beam, mGPTModel->mBeamWidth);
                    finishedString += std::to_string(finishedVec[offset]) + ", ";
                }
                finishedString += "]";
                if (b != actualBatchSize - 1)
                {
                    finishedString += "\n";
                }
            }
            finishedString += "]\n";
            LOG(INFO) << finishedString;
        }
#endif
    }

    NVTX_THREAD_END();

    // ====================== GENERATION PHASE ======================
    context = getContext(maxInputLength, /*isContextPhase = */ false);
    while (!shouldStop && ++step < mGPTModel->mMaxOutputLength)
    {
        NVTX_THREAD_START("Generation", nvtx::COLOR_YELLOW_7);
        // MP prepare batch
        if (mIsMain)
        {
            VLOG(2) << "Starting generation step: " << step;
            auto& copyStream = mCopyStreamVec[copyStreamIdx];
            // Block for previous inference done before modifying the host buffers
            copyStream.syncInferEvent();
            // stage host buffers
            stageHostBuffers(tasks, qsl, actualBatchSize, maxInputLength, step, gptBuffers);
            // get device buffer data prepared by copy streams
            stageDeviceBuffers(actualBatchSize, maxInputLength, step, copyStream.get(), gptBuffers);
            copyStream.recordH2D();
            copyStream.makeAwaitH2D(mInferStream);
#ifdef OUTPUT_DEBUG_DETAILED
            CHECK_EQ(cudaStreamSynchronize(copyStream.get()), cudaSuccess);
            gptBuffers.dumpInputHost(step, mGPTModel->mInputLengthName, actualBatchSize * mGPTModel->mBeamWidth, 1);
            gptBuffers.dumpInputHost(step, mGPTModel->mSequenceLengthName, actualBatchSize * mGPTModel->mBeamWidth, 1);
            gptBuffers.dumpInputHost(step, mGPTModel->mPositionTokenName, actualBatchSize * mGPTModel->mBeamWidth, 1);
            gptBuffers.dumpInputHost(step, mGPTModel->mLastTokenName, actualBatchSize * mGPTModel->mBeamWidth, 1);
            gptBuffers.dumpInputHost(step, mGPTModel->mMaskTokenName, actualBatchSize, mGPTModel->mMaxSumLength);
#endif
        }

        // prepare TRT engine tensors
        mGPTModel->setInputShapes(context, actualBatchSize, maxInputLength, step, /*isContext = */ false);
        mGPTModel->setTensorAddresses(context, gptBuffers, step);

        // NCCL sync device buffer
        if (!mIsGPTJ)
        {
            broadcastDeviceBuffers(actualBatchSize, maxInputLength, step, gptBuffers);
        }

        // run transformers
        CHECK_EQ(context->enqueueV3(mInferStream), true);

        // run decoder
        if (mIsMain)
        {
            auto& copyStream = mCopyStreamVec[copyStreamIdx];
            stageDecoderBuffers(actualBatchSize, maxInputLength, step, gptBuffers);
            shouldStop = mDecoder->forward(*mDecoderOutput, *mDecoderInput);
#ifdef OUTPUT_DEBUG_STEP
            gptBuffers.dumpOutputTokens(step, actualBatchSize, mGPTModel->mBeamWidth, maxInputLength);
            if (mGPTModel->mBeamWidth > 1)
            {
                // debug cumProb
                std::string cumProbString = "[DEBUG] Output cumProb at step " + std::to_string(step) + ": [";
                auto cumProbHost
                    = mDecoderBufferManager->copyFrom(*(mDecoderOutput->cumLogProbs), tr::MemoryType::kCPU);
                auto cumProbVec = tr::bufferCast<float>(*cumProbHost);
                mDecoderBufferManager->getStream().synchronize();
                for (int32_t b = 0; b < actualBatchSize; ++b)
                {
                    cumProbString += "[";
                    for (int32_t beam = 0; beam < mGPTModel->mBeamWidth; ++beam)
                    {
                        int32_t offset = tc::flat_index2(b, beam, mGPTModel->mBeamWidth);
                        cumProbString += std::to_string(cumProbVec[offset]) + ", ";
                    }
                    cumProbString += "]";
                    if (b != actualBatchSize - 1)
                    {
                        cumProbString += "\n";
                    }
                }
                cumProbString += "]\n";
                LOG(INFO) << cumProbString;
                // debug finished
                std::string finishedString = "[DEBUG] Output finished at step " + std::to_string(step) + ": [";
                auto finishedHost = mDecoderBufferManager->copyFrom(*(mDecoderOutput->finished), tr::MemoryType::kCPU);
                auto finishedVec = tr::bufferCast<bool>(*finishedHost);
                mDecoderBufferManager->getStream().synchronize();
                for (int32_t b = 0; b < actualBatchSize; ++b)
                {
                    finishedString += "[";
                    for (int32_t beam = 0; beam < mGPTModel->mBeamWidth; ++beam)
                    {
                        int32_t offset = tc::flat_index2(b, beam, mGPTModel->mBeamWidth);
                        finishedString += std::to_string(finishedVec[offset]) + ", ";
                    }
                    finishedString += "]";
                    if (b != actualBatchSize - 1)
                    {
                        finishedString += "\n";
                    }
                }
                finishedString += "]\n";
                LOG(INFO) << finishedString;
            }
            else
            {
                int32_t outputTokenoffset
                    = tc::flat_index3(maxInputLength + step, 0, 0, actualBatchSize, mGPTModel->mBeamWidth);
                int32_t tokenId
                    = *(static_cast<int32_t*>(gptBuffers.getHostBuffer(kGPT_OUTPUT_TOKEN_NAME)) + outputTokenoffset);
                gptBuffers.dumpOutputLogits(step, mGPTModel->mOutputLogitName, actualBatchSize, mGPTModel->mBeamWidth,
                    mGPTModel->mDimVocab, tokenId);
            }
            // gptBuffers.dumpOutputLogits(step, mGPTModel->mOutputLogitName, actualBatchSize, mGPTModel->mBeamWidth,
            // mGPTModel->mDimVocab, kGPTJ_END_ID, " After decoder->forward()");
#endif
            VLOG(2) << "shouldStop: " << shouldStop << "\n";
            copyStream.recordInferDone(mInferStream);
            copyStream.awaitInfer();
        }

        MPICHECK(MPI_Bcast(&shouldStop, sizeof(shouldStop), MPI_BYTE, kMPI_MAIN_RANK, MPI_COMM_WORLD));
        NVTX_THREAD_END();
    }
    // ====================== MP QSR REPORT ======================
    if (mIsMain)
    {
        auto& copyStream = mCopyStreamVec[copyStreamIdx];
        // transpose and copy to host
        finalizeDecoderOutput(actualBatchSize, maxInputLength, step, gptBuffers, copyStream);
        copyStream.recordD2H();
#ifdef OUTPUT_DEBUG
        CHECK_EQ(cudaStreamSynchronize(copyStream.get()), cudaSuccess);
        gptBuffers.dumpOutputSequenceVec(kNUM_DEBUG_TOKEN);
#endif
        // prepare the response
        GPTResponse resp;
        resp.resultReady = copyStream.getD2HEvent();
        resp.QSRs.reserve(actualBatchSize);

        // read decoder output
        for (int32_t i = 0; i < actualBatchSize; ++i)
        {
            const size_t generatedSequenceInBytes
                = mGPTModel->mMaxOutputLength * kGEN_LENGTH * lwis::getElementSize(GPTOutputTokenDataType);
            mlperf::QuerySampleResponse response{tasks[i].first.id,
                reinterpret_cast<uintptr_t>(gptBuffers.getHostOutputTokenBuffer(i)), generatedSequenceInBytes};
            resp.QSRs.emplace_back(response);
        }
        resp.copyStreamIdx = copyStreamIdx;
        VLOG(1) << "Device " << mDeviceId << ": Reported batch with " << actualBatchSize << " sample(s)";
        enqueueResponse(resp);
    }
}

void GPTCore::warmUp()
{
    LOG(INFO) << "Device " << mDeviceId << ": Warm up bypassed.";
    return;
}

void GPTCore::enqueueResponse(GPTResponse const& r)
{
    std::unique_lock<std::mutex> lck(mMtx);
    mResponseQueue.emplace_back(r);
    mResponseQueueCondVar.notify_one();
}

void GPTCore::processResponse(GPTCore* GPTCore, int32_t const deviceId)
{
    CHECK(GPTCore->mIsMain) << "Worker processes should not process the responses!";
    size_t totSamples = 0;
    while (true)
    {
        // wait for signal from infer()->enqueueResponse()
        std::unique_lock<std::mutex> lck(GPTCore->mMtx);
        GPTCore->mResponseQueueCondVar.wait(
            lck, [&]() { return !GPTCore->mResponseQueue.empty() || GPTCore->mStopWork; });
        if (GPTCore->mStopWork)
            break;
        auto& resp = GPTCore->mResponseQueue.front();

        CHECK_EQ(cudaEventSynchronize(resp.resultReady), cudaSuccess);

        // TODO if we want to report individual sequence inside the batch to Loadgen
        // we need to hold the copy stream instead of put it back to mCopyStreamIdxQueue until
        // all sequences in the batch are completed
        for (auto& qsr : resp.QSRs)
        {
            mlperf::QuerySamplesComplete(&qsr, 1);
        }
        totSamples += resp.QSRs.size();
        VLOG(1) << "Device " << deviceId << ": Finished " << totSamples << " samples.";

        // return the copy stream and pop response queue
        GPTCore->mCopyStreamIdxQueue.push_back(resp.copyStreamIdx);
        GPTCore->mResponseQueue.pop_front();
        // continue to consume the mResponseQueue
        GPTCore->mResponseQueueCondVar.notify_one();
    }

    VLOG(1) << "Device " << deviceId << ": QuerySamplesCompelete " << totSamples << " samples.";
}

GPTCore::~GPTCore()
{
    CHECK_EQ(cudaSetDevice(mDeviceId), cudaSuccess);
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mResponseQueueCondVar.notify_all();
    }

    for (auto& rt : mResponseThreadVec)
    {
        rt.join();
    }

    // As of 0707 Tekit ToT, mInferStream is destoryed by the decoder even ownsStream is set to false
    // CHECK_EQ(cudaStreamDestroy(mInferStream), cudaSuccess);

    if (!mIsGPTJ)
    {
        // finalizing NCCL
        NCCLCHECK(ncclCommDestroy(mNCCLComm));
    }
}
