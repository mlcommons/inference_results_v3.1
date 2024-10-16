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

#ifndef GPT_UTILS_HPP
#define GPT_UTILS_HPP

#include "half.h"
#include "lwis_buffers.h"
#include "nvtxUtils.hpp"
#include "utils.hpp"

#include "tensorrt_llm/common/memoryUtils.h"

#include <string>
#include <unordered_map>

namespace tc = tensorrt_llm::common;

// ================================
//     Debug support: nvtx ranges
// ================================
#define NVTX_ON

#ifndef NVTX_ON
#define NVTX_GLOBAL_START(A, B, C)
#define NVTX_GLOBAL_END(A)
#define NVTX_THREAD_START(B, C)
#define NVTX_THREAD_END()
#define NVTX_MARK(B, C)
#endif

// #define OUTPUT_DEBUG_DETAILED
// #define OUTPUT_DEBUG_STEP
// #define OUTPUT_DEBUG
#ifdef OUTPUT_DEBUG
constexpr int32_t kNUM_DEBUG_TOKEN{128};
#endif

#define MPICHECK(cmd)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        int32_t e = cmd;                                                                                               \
        if (e != MPI_SUCCESS)                                                                                          \
        {                                                                                                              \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define NCCLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess)                                                                                          \
        {                                                                                                              \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

constexpr int32_t kGPTJ_MAX_SEQ_LENGTH{2047};
constexpr int32_t kGPT175_MAX_SEQ_LENGTH{2176};
constexpr int32_t kGPTJ_MAX_INPUT_LENGTH{1919};
constexpr int32_t kGPT_MAX_OUTPUT_LENGTH{128};
constexpr int32_t kGPT_SHAPE_TENSOR_DIM0{2};
constexpr int32_t kGEN_LENGTH{1};
constexpr int32_t kNUM_RESPONSE_THREADS{1};
constexpr int32_t kNUM_GPTJ_ENGINE_IO_TENSOR{66};
constexpr int32_t kNUM_GPT175_ENGINE_IO_TENSOR{394};

constexpr int32_t kGPTJ_END_ID{50256};
constexpr int32_t kGPTJ_PAD_ID{50256};

std::string const kGPT_OUTPUT_TOKEN_NAME{"output_ids"};

// maybe replace with C++ native data types
constexpr nvinfer1::DataType GPTInputTokenDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTFp16KVCacheDataType{nvinfer1::DataType::kHALF};
constexpr nvinfer1::DataType GPTFp8KVCacheDataType{nvinfer1::DataType::kINT8};
constexpr nvinfer1::DataType GPTSequenceLengthDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTShapeTensorDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTInputLengthDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTPositionTokenDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTLastTokenDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTMaskTokenDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTCacheIndrectionDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTMaxInputLengthDataType{nvinfer1::DataType::kINT32};
constexpr nvinfer1::DataType GPTOutputLogitDataType{nvinfer1::DataType::kHALF};
constexpr nvinfer1::DataType GPTOutputTokenDataType{nvinfer1::DataType::kINT32};

inline int32_t padVocabSize(int32_t vocabSize, int32_t tpSize)
{
    return ((vocabSize + tpSize - 1) / tpSize) * tpSize;
}

class CopyStream
{
public:
    CopyStream()
    {
        uint32_t flags = cudaEventDefault | cudaEventDisableTiming;
        CHECK_EQ(cudaStreamCreate(&mStream), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&h2d, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&d2h, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&d2d, flags), cudaSuccess);
        CHECK_EQ(cudaEventCreateWithFlags(&infer, flags), cudaSuccess);
    }

    ~CopyStream()
    {
        CHECK_EQ(cudaStreamDestroy(mStream), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(h2d), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(d2h), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(d2d), cudaSuccess);
        CHECK_EQ(cudaEventDestroy(infer), cudaSuccess);
    }

    // Context Phase H2D
    void recordH2D()
    {
        CHECK_EQ(cudaEventRecord(h2d, mStream), cudaSuccess);
    }

    void makeAwaitH2D(cudaStream_t inferStream)
    {
        CHECK_EQ(cudaStreamWaitEvent(inferStream, h2d, 0), cudaSuccess);
    }

    // Generation Phase D2D
    void makeAwaitD2D(cudaStream_t inferStream)
    {
        CHECK_EQ(cudaStreamWaitEvent(inferStream, d2d, 0), cudaSuccess);
    }

    // Wait for Inference
    void awaitInfer()
    {
        CHECK_EQ(cudaStreamWaitEvent(mStream, infer, 0), cudaSuccess);
    }

    void recordInferDone(cudaStream_t inferStream)
    {
        CHECK_EQ(cudaEventRecord(infer, inferStream), cudaSuccess);
    }

    void recordD2H()
    {
        CHECK_EQ(cudaEventRecord(d2h, mStream), cudaSuccess);
    }

    void recordD2D()
    {
        CHECK_EQ(cudaEventRecord(d2d, mStream), cudaSuccess);
    }

    void syncD2H()
    {
        CHECK_EQ(cudaEventSynchronize(d2h), cudaSuccess);
    }

    // Sync for infer done
    void syncInferEvent()
    {
        CHECK_EQ(cudaEventSynchronize(infer), cudaSuccess);
    }

    cudaStream_t get() const
    {
        return mStream;
    }

    cudaEvent_t getD2HEvent() const
    {
        return d2h;
    }

private:
    cudaStream_t mStream;
    cudaEvent_t h2d;
    cudaEvent_t d2h;
    cudaEvent_t d2d;
    cudaEvent_t infer;
};

class GPTModel;

//! This class manages all data types, memory sizes, allocated host and device buffers for tensors in the TRT LLM engine
//! Tensor names should be used to index the  data types, memory sizes, host and device buffers
class GPTManagedBuffer
{
public:
    GPTManagedBuffer(int32_t batchSize, std::unique_ptr<GPTModel> const& gptModel,
        bool isFp8 /* placeholder for fp8 variance */, bool isMain);

    nvinfer1::DataType getDataType(std::string const& tensorName)
    {
        CHECK(mBufferDataTypes.find(tensorName) != mBufferDataTypes.end());
        return mBufferDataTypes[tensorName];
    }

    void* getHostBuffer(std::string const& tensorName)
    {
        CHECK(mIsMain) << "Host buffer should only be used by the main process";
        CHECK(mHostBuffers.find(tensorName) != mHostBuffers.end());
        return mHostBuffers[tensorName].data();
    }

    void* getDeviceBuffer(std::string const& tensorName)
    {
        CHECK(mDeviceBuffers.find(tensorName) != mDeviceBuffers.end());
        return mDeviceBuffers[tensorName].data();
    }

    void* getHostOutputTokenBuffer(int32_t index)
    {
        CHECK(mIsMain) << "Host buffer should only be used by the main process";
        return mHostOutputTokenBufferVec[index].data();
    }

    // Set value of target buffer on host
    void memsetHost(std::string const& tensorName, int32_t const value)
    {
        CHECK(mIsMain) << "Host memset should only be called by the main process";
        CHECK(mHostBuffers.find(tensorName) != mHostBuffers.end());
        memset(mHostBuffers[tensorName].data(), value, mBufferSizeInBytes[tensorName]);
    }

    // memset all device buffer
    void memsetDeviceAll(int32_t const value)
    {
        for (auto& deviceBuffer : mDeviceBuffers)
        {
            CHECK_EQ(
                cudaMemset(deviceBuffer.second.data(), value, mBufferSizeInBytes[deviceBuffer.first]), cudaSuccess);
        }
    }

    // memset named device buffer
    void memsetDeviceAsync(std::string const& tensorName, int32_t const value, cudaStream_t stream)
    {
        CHECK(mDeviceBuffers.find(tensorName) != mDeviceBuffers.end());
        CHECK_EQ(cudaMemsetAsync(mDeviceBuffers[tensorName].data(), value, mBufferSizeInBytes[tensorName], stream),
            cudaSuccess);
    }

    // host to device async copy
    void H2DAsync(std::string const& tensorName, size_t const num, cudaStream_t stream)
    {
        CHECK(mIsMain) << "Host - Device copy should only be called by the main process";
        CHECK(mHostBuffers.find(tensorName) != mHostBuffers.end());
        CHECK(mDeviceBuffers.find(tensorName) != mDeviceBuffers.end());
        CHECK_EQ(cudaMemcpyAsync(mDeviceBuffers[tensorName].data(), mHostBuffers[tensorName].data(),
                     num * lwis::getElementSize(mBufferDataTypes[tensorName]), cudaMemcpyHostToDevice, stream),
            cudaSuccess);
    }

    // device to device async copy
    void D2DAsync(std::string const& srcTensorName, std::string const& dstTensorName, size_t const num,
        size_t const srcOffset, cudaStream_t stream)
    {
        int32_t elementSize = lwis::getElementSize(mBufferDataTypes[dstTensorName]);
        CHECK_EQ(cudaMemcpyAsync(mDeviceBuffers[dstTensorName].data(),
                     static_cast<char*>(mDeviceBuffers[srcTensorName].data()) + srcOffset * elementSize,
                     num * elementSize, cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);
    }

    // host to host memcopy
    void H2H(std::string const& tensorName, void* src, size_t const num, size_t const offset)
    {
        CHECK(mIsMain) << "Host - Host copy should only be called by the main process";
        memcpy(static_cast<char*>(mHostBuffers[tensorName].data())
                + offset * lwis::getElementSize(mBufferDataTypes[tensorName]),
            src, num * lwis::getElementSize(mBufferDataTypes[tensorName]));
    }

    // Copy output token from device to the end of output sequence on host
    void outputD2HAsync(int32_t const step, int32_t const actualBatchSize, cudaStream_t stream)
    {
        CHECK(mIsMain) << "Device - Host copy should only be called by the main process";
        CHECK(mBufferDataTypes[kGPT_OUTPUT_TOKEN_NAME] == GPTOutputTokenDataType);
        for (int32_t i = 0; i < actualBatchSize; ++i)
        {
            size_t const tokenSize = kGEN_LENGTH * lwis::getElementSize(mBufferDataTypes[kGPT_OUTPUT_TOKEN_NAME]);
            size_t const hostOffset = step * tokenSize;
            size_t const deviceOffset = i * tokenSize;
            CHECK_EQ(cudaMemcpyAsync(static_cast<char*>(mHostOutputTokenBufferVec[i].data()) + hostOffset,
                         static_cast<char*>(mDeviceBuffers[kGPT_OUTPUT_TOKEN_NAME].data()) + deviceOffset, tokenSize,
                         cudaMemcpyDeviceToHost, stream),
                cudaSuccess);
        }
    }

    // memset all host output token to pad id
    void decoderOutputTokenResetToPad()
    {
        // memset not working for unknown reason, switch to memcpy as a war
        std::vector<int32_t> defaultPadVec(kGPT_MAX_OUTPUT_LENGTH, kGPTJ_PAD_ID);
        for (auto& outputBuffer : mHostOutputTokenBufferVec)
        {
            memcpy(outputBuffer.data(), defaultPadVec.data(), outputBuffer.size());
            // memset(outputBuffer.data(), kGPTJ_PAD_ID, mBufferSizeInBytes[kGPT_OUTPUT_TOKEN_NAME]);
        }
    }

#ifdef OUTPUT_DEBUG
    // debug function for inspecting input host buffers
    void dumpInputHost(
        int32_t const step, std::string const& tensorName, int32_t const batchSize, int32_t const sizeDimOne)
    {
        CHECK(mHostBuffers.find(tensorName) != mHostBuffers.end());
        CHECK(mIsMain) << "Input " << tensorName << " host debug should only be called by the main process";
        std::string terminalOutput = "[DEBUG] " + tensorName + " at step " + std::to_string(step) + " :\n[";
        for (int32_t i = 0; i < batchSize; ++i)
        {
            terminalOutput += "[";
            for (int32_t j = 0; j < sizeDimOne; ++j)
            {
                terminalOutput
                    += (std::to_string(*(static_cast<int32_t*>(mHostBuffers[tensorName].data()) + j + i * sizeDimOne))
                        + ", ");
            }
            terminalOutput += "]\n";
        }
        terminalOutput.insert(terminalOutput.end() - 1, ']');
        LOG(INFO) << terminalOutput;
    }

    // debug function for inspecting output logits
    void dumpOutputLogits(int32_t const step, std::string const& tensorName, int32_t const batchSize,
        int32_t const beamWidth, int32_t const dimVocab, int32_t const vocabId, std::string customDebugMsg = "")
    {
        CHECK(mHostBuffers.find(tensorName) != mHostBuffers.end());
        CHECK(mDeviceBuffers.find(tensorName) != mDeviceBuffers.end());
        CHECK(vocabId <= dimVocab);
        CHECK(mIsMain) << "Logits debug should only be called by the main process";
        std::string terminalOutput = "[DEBUG]" + customDebugMsg + " Logit [" + std::to_string(vocabId) + "] at step "
            + std::to_string(step) + " :\n[";
        cudaMemcpy(mHostBuffers[tensorName].data(), mDeviceBuffers[tensorName].data(), mBufferSizeInBytes[tensorName],
            cudaMemcpyDeviceToHost);
        for (int32_t b = 0; b < batchSize; ++b)
        {
            terminalOutput += "[";
            int32_t offset = tc::flat_index3(b, 0, vocabId, beamWidth, dimVocab);
            // half has 2 bytes, using uint16_t as the pointer type
            terminalOutput += (std::to_string(half_float::detail::half2float<float>(
                                   *(static_cast<uint16_t*>(mHostBuffers[tensorName].data()) + offset)))
                + ", ");
            terminalOutput += "]\n";
        }
        terminalOutput.insert(terminalOutput.end() - 1, ']');
        LOG(INFO) << terminalOutput;
    }

    // debug function for inspecting output token
    void dumpOutputTokens(int32_t const step, int32_t const batchSize,
        int32_t const beamWidth, int32_t const maxInputLength
    )
    {
        CHECK(mHostBuffers.find(kGPT_OUTPUT_TOKEN_NAME) != mHostBuffers.end());
        CHECK(mDeviceBuffers.find(kGPT_OUTPUT_TOKEN_NAME) != mDeviceBuffers.end());
        CHECK(mIsMain) << "Output tokens debug should only be called by the main process";
        std::string terminalOutput = "[DEBUG] Output token at step " + std::to_string(step) + ": [";
        cudaMemcpy(mHostBuffers[kGPT_OUTPUT_TOKEN_NAME].data(), mDeviceBuffers[kGPT_OUTPUT_TOKEN_NAME].data(),
            mBufferSizeInBytes[kGPT_OUTPUT_TOKEN_NAME], cudaMemcpyDeviceToHost);
        for (int32_t b = 0; b < batchSize; ++b)
        {
            terminalOutput += "[";
            for (int32_t beam = 0; beam < beamWidth; ++beam)
            {
                int32_t offset = tc::flat_index3(maxInputLength + step, b, beam, batchSize, beamWidth);
                terminalOutput
                    += std::to_string(*(static_cast<int32_t*>(mHostBuffers[kGPT_OUTPUT_TOKEN_NAME].data()) + offset))
                    + ", ";
            }
            terminalOutput += "]";
            if (b != batchSize - 1)
            {
                terminalOutput += "\n";
            }
        }
        terminalOutput += "]\n";
        LOG(INFO) << terminalOutput;
    }

    // debug function for inspecting output sequence
    void dumpOutputSequenceVec(int32_t const step)
    {
        CHECK(mIsMain) << "Output sequences debug should only be called by the main process";
        std::string terminalOutput = "[DEBUG] Full output at step: " + std::to_string(step) + " :\n";
        for (auto const& outputBuffer : mHostOutputTokenBufferVec)
        {
            terminalOutput += "[";
            for (int32_t i = 0; i < step; ++i)
            {
                terminalOutput += (std::to_string(*(static_cast<int32_t const*>(outputBuffer.data()) + i)) + ", ");
            }
            terminalOutput += "]\n";
        }
        LOG(INFO) << terminalOutput;
    }
#endif

private:
    // only main process will create host buffers
    bool const mIsMain;
    int32_t const mBatchSize;
    std::unordered_map<std::string, nvinfer1::DataType> mBufferDataTypes;
    std::unordered_map<std::string, int32_t> mBufferSizeInBytes;
    // Host and device buffers
    // TODO zhihanj: opt1 change to dynamic buffer
    // TODO zhihanj: opt2 support context + generation buffer for phased scheduling
    std::unordered_map<std::string, lwis::DeviceBuffer> mDeviceBuffers;
    std::unordered_map<std::string, lwis::HostBuffer> mHostBuffers;
    std::vector<lwis::HostBuffer> mHostOutputTokenBufferVec;
};

//! GPT Model sets all tensor shapes and addresses for given TensorRT execution context
class GPTModel
{
public:
    GPTModel(std::string const& inputTokenName, std::string const& inputLengthName,
        std::string const& sequenceLengthName, std::string const& KVCacheLengthTensorName,
        std::string const& positionTokenName, std::string const& lastTokenName, std::string const& maskTokenName,
        std::vector<std::string> const& inputKVCacheNames, std::vector<std::string> const& outputKVCacheNames,
        std::vector<std::string> const& cacheIndirectionNames, std::string const& maxInputLengthName,
        std::string const& outputLogitName, std::string const& outputTokenName, int32_t maxBatchSize,
        int32_t numKVCache, int32_t numHeads, int32_t dimHeads, int32_t numTensorParallelism, int32_t maxInputLength,
        int32_t maxOutputLength, int32_t maxSumLength, int32_t dimVocab, int32_t beamWidth)
        : mInputTokenName(inputTokenName)
        , mInputLengthName(inputLengthName)
        , mSequenceLengthName(sequenceLengthName)
        , mKVCacheLengthTensorName(KVCacheLengthTensorName)
        , mPositionTokenName(positionTokenName)
        , mLastTokenName(lastTokenName)
        , mMaskTokenName(maskTokenName)
        , mInputKVCacheNameVec(inputKVCacheNames)
        , mOutputKVCacheNameVec(outputKVCacheNames)
        , mCacheIndirectionNameVec(cacheIndirectionNames)
        , mMaxInputLengthName(maxInputLengthName)
        , mOutputLogitName(outputLogitName)
        , mOutputTokenName(outputTokenName)
        , mMaxBatchSize(maxBatchSize)
        , mNumKVCache(numKVCache)
        , mNumHeads(numHeads)
        , mDimHeads(dimHeads)
        , mNumTensorParallelism(numTensorParallelism)
        , mMaxInputLength(maxInputLength)
        , mMaxOutputLength(maxOutputLength)
        , mMaxSumLength(maxSumLength)
        , mDimVocab(dimVocab)
        , mDimVocabPadded(padVocabSize(mDimVocab, mNumTensorParallelism))
        , mBeamWidth(beamWidth)
    {
        CHECK(mMaxBatchSize > 0);
        CHECK(mNumKVCache > 0);
        CHECK(mNumHeads > 0);
        CHECK(mDimHeads > 0);
        CHECK(mNumTensorParallelism > 0);
        CHECK(mMaxInputLength > 0);
        CHECK(mMaxOutputLength > 0);
        CHECK(mMaxSumLength > 0);
        CHECK(mDimVocab > 0);
        CHECK(mBeamWidth > 0);
    }

    ~GPTModel() = default;

    int32_t getMaxKVCacheSize() const
    {
        return mNumHeads * 2 * mMaxSumLength * mDimHeads;
    }

    void setInputShapes(std::shared_ptr<nvinfer1::IExecutionContext> context, int32_t actualBatchSize,
        int32_t inputSeqLength, int32_t step, bool isContext) const;
    void setTensorAddresses(
        std::shared_ptr<nvinfer1::IExecutionContext> context, GPTManagedBuffer& GPTBuffers, int32_t step) const;

    // GPT TensorRT engine I/O tensor names
    // TODO we should use name -> shape map in the future to reduce the efforts of setting the shape manually in
    // setInputShapes()
    std::string const mInputTokenName;
    std::string const mInputLengthName;
    std::string const mSequenceLengthName;
    std::string const mKVCacheLengthTensorName;
    std::string const mPositionTokenName;
    std::string const mLastTokenName;
    std::string const mMaskTokenName;
    std::vector<std::string> const mInputKVCacheNameVec;
    std::vector<std::string> const mOutputKVCacheNameVec;
    std::vector<std::string> const mCacheIndirectionNameVec;
    std::string const mMaxInputLengthName;
    std::string const mOutputLogitName;
    std::string const mOutputTokenName;

    // GPT Model Config
    int32_t const mMaxBatchSize;
    int32_t const mNumKVCache;
    int32_t const mNumHeads;
    int32_t const mDimHeads;
    int32_t const mNumTensorParallelism; // TODO remove possible redundant model info
    int32_t const mMaxInputLength;
    int32_t const mMaxOutputLength;
    int32_t const mMaxSumLength;
    int32_t const mDimVocab;
    int32_t const mDimVocabPadded;
    int32_t const mBeamWidth;
};

#endif // GPT_UTILS_HPP
