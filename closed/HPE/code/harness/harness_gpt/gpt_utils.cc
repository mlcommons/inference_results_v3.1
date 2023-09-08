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

#include "gpt_utils.hpp"
#include "glog/logging.h"
#include "lwis_buffers.h"

GPTManagedBuffer::GPTManagedBuffer(
    int32_t batchSize, std::unique_ptr<GPTModel> const& gptModel, bool isFp8 /*placeholder for fp8*/, bool isMain)
    : mIsMain(isMain)
    , mBatchSize(batchSize)
{
    // store data type
    mBufferDataTypes.emplace(gptModel->mInputTokenName, GPTInputTokenDataType);
    for (auto const& KVCacheName : gptModel->mInputKVCacheNameVec)
    {
        mBufferDataTypes.emplace(KVCacheName, isFp8 ? GPTFp8KVCacheDataType : GPTFp16KVCacheDataType);
    }
    mBufferDataTypes.emplace(gptModel->mSequenceLengthName, GPTSequenceLengthDataType);
    mBufferDataTypes.emplace(gptModel->mKVCacheLengthTensorName, GPTShapeTensorDataType);
    mBufferDataTypes.emplace(gptModel->mInputLengthName, GPTInputLengthDataType);
    mBufferDataTypes.emplace(gptModel->mPositionTokenName, GPTPositionTokenDataType);
    mBufferDataTypes.emplace(gptModel->mLastTokenName, GPTLastTokenDataType);
    mBufferDataTypes.emplace(gptModel->mMaskTokenName, GPTMaskTokenDataType);
    for (auto const& cacheIndirectionName : gptModel->mCacheIndirectionNameVec)
    {
        mBufferDataTypes.emplace(cacheIndirectionName, GPTCacheIndrectionDataType);
    }
    mBufferDataTypes.emplace(gptModel->mMaxInputLengthName, GPTMaxInputLengthDataType);
    mBufferDataTypes.emplace(gptModel->mOutputLogitName, GPTOutputLogitDataType);
    mBufferDataTypes.emplace(gptModel->mOutputTokenName, GPTOutputTokenDataType);

    // calculate max buffer size necessary
    mBufferSizeInBytes.emplace(gptModel->mInputTokenName,
        mBatchSize * gptModel->mBeamWidth * gptModel->mMaxSumLength
            * lwis::getElementSize(mBufferDataTypes[gptModel->mInputTokenName]));
    mBufferSizeInBytes.emplace(gptModel->mSequenceLengthName,
        mBatchSize * gptModel->mBeamWidth * lwis::getElementSize(mBufferDataTypes[gptModel->mSequenceLengthName]));
    mBufferSizeInBytes.emplace(gptModel->mKVCacheLengthTensorName,
        kGPT_SHAPE_TENSOR_DIM0 * lwis::getElementSize(mBufferDataTypes[gptModel->mKVCacheLengthTensorName]));
    mBufferSizeInBytes.emplace(gptModel->mInputLengthName,
        mBatchSize * gptModel->mBeamWidth * lwis::getElementSize(mBufferDataTypes[gptModel->mInputLengthName]));
    mBufferSizeInBytes.emplace(gptModel->mPositionTokenName,
        mBatchSize * gptModel->mBeamWidth * gptModel->mMaxInputLength
            * lwis::getElementSize(mBufferDataTypes[gptModel->mPositionTokenName]));
    mBufferSizeInBytes.emplace(gptModel->mLastTokenName,
        mBatchSize * gptModel->mBeamWidth * lwis::getElementSize(mBufferDataTypes[gptModel->mLastTokenName]));
    mBufferSizeInBytes.emplace(gptModel->mMaskTokenName,
        mBatchSize * gptModel->mBeamWidth * gptModel->mMaxSumLength
            * lwis::getElementSize(mBufferDataTypes[gptModel->mMaskTokenName]));
    for (auto const& KVCacheName : gptModel->mInputKVCacheNameVec)
    {
        mBufferSizeInBytes.emplace(KVCacheName,
            mBatchSize * gptModel->mBeamWidth * gptModel->getMaxKVCacheSize()
                * lwis::getElementSize(mBufferDataTypes[KVCacheName]));
    }
    for (auto const& cacheIndirectionName : gptModel->mCacheIndirectionNameVec)
    {
        mBufferSizeInBytes.emplace(cacheIndirectionName,
            mBatchSize * gptModel->mBeamWidth * gptModel->mMaxSumLength
                * lwis::getElementSize(mBufferDataTypes[cacheIndirectionName]));
    }
    mBufferSizeInBytes.emplace(gptModel->mMaxInputLengthName,
        gptModel->mMaxInputLength * lwis::getElementSize(mBufferDataTypes[gptModel->mMaxInputLengthName]));
    mBufferSizeInBytes.emplace(gptModel->mOutputLogitName,
        mBatchSize * gptModel->mBeamWidth * gptModel->mDimVocabPadded
            * lwis::getElementSize(mBufferDataTypes[gptModel->mOutputLogitName]));
    mBufferSizeInBytes.emplace(gptModel->mOutputTokenName,
        gptModel->mMaxSumLength * mBatchSize * gptModel->mBeamWidth
            * lwis::getElementSize(mBufferDataTypes[gptModel->mOutputTokenName])); // device output token buffer stores
                                                                                   // input token + generated output
                                                                                   // token following trtllm design

    // allocate device buffers
    mDeviceBuffers.emplace(gptModel->mInputTokenName, mBufferSizeInBytes[gptModel->mInputTokenName]);
    for (auto const& KVCacheName : gptModel->mInputKVCacheNameVec)
    {
        mDeviceBuffers.emplace(KVCacheName, mBufferSizeInBytes[KVCacheName]);
    }
    mDeviceBuffers.emplace(gptModel->mSequenceLengthName, mBufferSizeInBytes[gptModel->mSequenceLengthName]);
    mDeviceBuffers.emplace(gptModel->mInputLengthName, mBufferSizeInBytes[gptModel->mInputLengthName]);
    mDeviceBuffers.emplace(gptModel->mPositionTokenName, mBufferSizeInBytes[gptModel->mPositionTokenName]);
    mDeviceBuffers.emplace(gptModel->mLastTokenName, mBufferSizeInBytes[gptModel->mLastTokenName]);
    mDeviceBuffers.emplace(gptModel->mMaskTokenName, mBufferSizeInBytes[gptModel->mMaskTokenName]);
    for (auto const& cacheIndirectionName : gptModel->mCacheIndirectionNameVec)
    {
        mDeviceBuffers.emplace(cacheIndirectionName, mBufferSizeInBytes[cacheIndirectionName]);
    }
    mDeviceBuffers.emplace(gptModel->mMaxInputLengthName, mBufferSizeInBytes[gptModel->mMaxInputLengthName]);
    mDeviceBuffers.emplace(gptModel->mOutputLogitName, mBufferSizeInBytes[gptModel->mOutputLogitName]);
    mDeviceBuffers.emplace(gptModel->mOutputTokenName, mBufferSizeInBytes[gptModel->mOutputTokenName]);
    // mDeviceBuffers.emplace(gptModel->mKVCacheLengthTensorName,
    // mBufferSizeInBytes[gptModel->mKVCacheLengthTensorName]);
    mHostBuffers.emplace(gptModel->mKVCacheLengthTensorName,
        mBufferSizeInBytes[gptModel->mKVCacheLengthTensorName]); // MHA shape tensor uses host memory!

    // allocate host buffers for MP
    if (mIsMain)
    {
        mHostBuffers.emplace(gptModel->mInputTokenName, mBufferSizeInBytes[gptModel->mInputTokenName]);
        mHostBuffers.emplace(gptModel->mSequenceLengthName, mBufferSizeInBytes[gptModel->mSequenceLengthName]);
        mHostBuffers.emplace(gptModel->mInputLengthName, mBufferSizeInBytes[gptModel->mInputLengthName]);
        mHostBuffers.emplace(gptModel->mPositionTokenName, mBufferSizeInBytes[gptModel->mPositionTokenName]);
        mHostBuffers.emplace(gptModel->mLastTokenName, mBufferSizeInBytes[gptModel->mLastTokenName]);
        mHostBuffers.emplace(gptModel->mMaskTokenName, mBufferSizeInBytes[gptModel->mMaskTokenName]);
        for (auto const& cacheIndirectionName : gptModel->mCacheIndirectionNameVec)
        {
            mHostBuffers.emplace(cacheIndirectionName, mBufferSizeInBytes[cacheIndirectionName]);
        }
        mHostBuffers.emplace(gptModel->mMaxInputLengthName, mBufferSizeInBytes[gptModel->mMaxInputLengthName]);
#ifdef OUTPUT_DEBUG
        mHostBuffers.emplace(gptModel->mOutputLogitName, mBufferSizeInBytes[gptModel->mOutputLogitName]);
        mHostBuffers.emplace(gptModel->mOutputTokenName, mBufferSizeInBytes[gptModel->mOutputTokenName]);
#endif
        // each sequence i in batch managees its own generated output at mHostOutputTokenBufferVec[i]
        for (int32_t i = 0; i < mBatchSize; ++i)
        {
            mHostOutputTokenBufferVec.emplace_back(kGEN_LENGTH * kGPT_MAX_OUTPUT_LENGTH
                * lwis::getElementSize(mBufferDataTypes[gptModel->mOutputTokenName]));
        }
    }

    // sanity checks
    CHECK(mBatchSize > 0);
    for (auto const& deviceBuffer : mDeviceBuffers)
    {
        CHECK(deviceBuffer.second.data() != nullptr);
    }
    for (auto const& hostBuffer : mHostBuffers)
    {
        CHECK(hostBuffer.second.data() != nullptr);
    }
    for (auto const& outputTokenBuffer : mHostOutputTokenBufferVec)
    {
        CHECK(mHostOutputTokenBufferVec.data() != nullptr);
    }
}

void GPTModel::setInputShapes(std::shared_ptr<nvinfer1::IExecutionContext> context, int32_t actualBatchSize,
    int32_t inputSeqLength, int32_t step, bool isContext) const
{
    // context profile sanity check
    auto& engine = context->getEngine();
    int32_t profileNum = context->getOptimizationProfile();
    CHECK_EQ(profileNum >= 0 && profileNum < engine.getNbOptimizationProfiles(), true);

    auto createDims = [](std::vector<int32_t> const& v) {
        nvinfer1::Dims d;
        d.nbDims = v.size();
        for (int32_t i = 0; i < v.size(); i++)
        {
            d.d[i] = v[i];
        }
        return d;
    };

    // set all input tensor shapes for the execution context
    // Context: [BS, runtime_input_seqlen]
    // Genetation: [BS * beam_width, 1]
    CHECK(context->setInputShape(mInputTokenName.c_str(),
        isContext ? createDims({actualBatchSize, inputSeqLength})
                  : createDims({actualBatchSize * mBeamWidth, kGEN_LENGTH})));
    // Context: [BS]
    // Genetation: [BS * beam_width]
    CHECK(context->setInputShape(mSequenceLengthName.c_str(),
        isContext ? createDims({actualBatchSize}) : createDims({actualBatchSize * mBeamWidth})));
    // MHA dummy tensor shape is always [2]
    CHECK(context->setInputShape(mKVCacheLengthTensorName.c_str(), createDims({kGPT_SHAPE_TENSOR_DIM0})));
    // real input lengths
    // Context: [BS]
    // Genetation: [BS * beam_width]
    CHECK(context->setInputShape(mInputLengthName.c_str(),
        isContext ? createDims({actualBatchSize}) : createDims({actualBatchSize * mBeamWidth})));
    // Context: [BS, runtime_input_seqlen]
    // Genetation: [BS * beam_width, 1]
    CHECK(context->setInputShape(mPositionTokenName.c_str(),
        isContext ? createDims({actualBatchSize, inputSeqLength}) : createDims({actualBatchSize * mBeamWidth, 1})));
    // Context: [BS]
    // Genetation: [BS * beam_width]
    CHECK(context->setInputShape(mLastTokenName.c_str(),
        isContext ? createDims({actualBatchSize}) : createDims({actualBatchSize * mBeamWidth})));
    // Context: [BS, max_seqlen]
    // Genetation: [BS * beam_width, max_seqlen]
    CHECK(context->setInputShape(mMaskTokenName.c_str(),
        isContext ? createDims({actualBatchSize, mMaxSumLength})
                  : createDims({actualBatchSize * mBeamWidth, mMaxSumLength})));
    // [BS, beam_width, max_seqlen]
    CHECK(context->setInputShape("cache_indirection", createDims({actualBatchSize, mBeamWidth, mMaxSumLength})));
    // [runtime_input_seqlen]
    CHECK(context->setInputShape(mMaxInputLengthName.c_str(), createDims({inputSeqLength})));
    // Context: [BS, 2, nhead, seqlen, dhead]
    // Genetation: [BS * beam_width, 2, nhead, seqlen, dhead]
    for (auto const& inputKVCacheName : mInputKVCacheNameVec)
    {
        CHECK(context->setInputShape(inputKVCacheName.c_str(),
            isContext ? createDims({actualBatchSize, 2, mNumHeads, mMaxSumLength, mDimHeads})
                      : createDims({actualBatchSize * mBeamWidth, 2, mNumHeads, mMaxSumLength, mDimHeads})));
    }
    CHECK(context->inferShapes(0, nullptr) == 0);
}

void GPTModel::setTensorAddresses(
    std::shared_ptr<nvinfer1::IExecutionContext> context, GPTManagedBuffer& gptBuffers, int32_t step) const
{
    if (step < 2) // one time effort for context + generation step
    {
        // context profile sanity check
        auto& engine = context->getEngine();
        int32_t profileIdx = context->getOptimizationProfile();
        CHECK_EQ(profileIdx >= 0 && profileIdx < engine.getNbOptimizationProfiles(), true);

        // set context all I/O tensor memory addresses
        CHECK(context->setTensorAddress(mKVCacheLengthTensorName.c_str(),
            gptBuffers.getHostBuffer(mKVCacheLengthTensorName))); // MHA shape tensor uses host memory!
        CHECK(context->setTensorAddress(mInputTokenName.c_str(), gptBuffers.getDeviceBuffer(mInputTokenName)));
        CHECK(context->setTensorAddress(mSequenceLengthName.c_str(), gptBuffers.getDeviceBuffer(mSequenceLengthName)));
        CHECK(context->setTensorAddress(mInputLengthName.c_str(), gptBuffers.getDeviceBuffer(mInputLengthName)));
        CHECK(context->setTensorAddress(mPositionTokenName.c_str(), gptBuffers.getDeviceBuffer(mPositionTokenName)));
        CHECK(context->setTensorAddress(mLastTokenName.c_str(), gptBuffers.getDeviceBuffer(mLastTokenName)));
        CHECK(context->setTensorAddress(mMaskTokenName.c_str(), gptBuffers.getDeviceBuffer(mMaskTokenName)));
        CHECK(context->setTensorAddress(mMaxInputLengthName.c_str(), gptBuffers.getDeviceBuffer(mMaxInputLengthName)));
        CHECK(mInputKVCacheNameVec.size() == mOutputKVCacheNameVec.size());
        for (size_t i = 0; i < mInputKVCacheNameVec.size(); ++i)
        {
            CHECK(context->setTensorAddress(
                mInputKVCacheNameVec[i].c_str(), gptBuffers.getDeviceBuffer(mInputKVCacheNameVec[i])));
            CHECK(context->setTensorAddress(
                mOutputKVCacheNameVec[i].c_str(), gptBuffers.getDeviceBuffer(mInputKVCacheNameVec[i])));
        }
        CHECK(context->setTensorAddress(mOutputLogitName.c_str(), gptBuffers.getDeviceBuffer(mOutputLogitName)));
    }
    if (mBeamWidth > 1)
    {
        // ping pong cache_indirection
        CHECK(context->setTensorAddress(
            "cache_indirection", gptBuffers.getDeviceBuffer(mCacheIndirectionNameVec[1 - step % 2])));
    }
    else
    {
        CHECK(context->setTensorAddress(
            "cache_indirection", gptBuffers.getDeviceBuffer(mCacheIndirectionNameVec.front())));
    }
}
