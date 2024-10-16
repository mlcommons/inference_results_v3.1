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

#include "NvInferPlugin.h"
#include <cassert>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class DLRMInteractionsPlugin : public IPluginV2DynamicExt
{
public:
    // Constructor, Destructor
    DLRMInteractionsPlugin(int embeddingSize, int embeddingRows, int reducedPrecisionIO,
        float embeddingWeightsOnGpuPart, int interactionsOutputInterleaved, int outputPaddingGranularity,
        const std::vector<int>& tableOffsets, const std::string& embeddingWeightsFilepath,
        const std::string& rowFrequenciesFilepath, bool compressCategoricalInputs);
    DLRMInteractionsPlugin(const void* data, size_t length);
    ~DLRMInteractionsPlugin() override = default;

    // IPluginV2Ext fields
    int getNbOutputs() const noexcept override;
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }
    IPluginV2DynamicExt* clone() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;

    // IPluginV2DynamicExt new fields
    DimsExprs getOutputDimensions(
        int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs,
        int nbOutputs) const noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    struct HostData
    {
        HostData() : mHostEmbeddings(nullptr), mHostEmbeddingsDevicePtr(nullptr), mEmbeddingRowsOnDevice(0), mCounter(0) {}
        void * mHostEmbeddings;
        void * mHostEmbeddingsDevicePtr;
        int mEmbeddingRowsOnDevice;
        std::vector<float> mEmbeddingsScales;
        int mCounter;
    };

    //! Embedding and related data stored on a single GPU.
    struct DeviceData
    {
        DeviceData() : mEmbeddings(nullptr), mIndexRemapping(nullptr), mTableOffsets(nullptr), mCounter(0) {}
        void * mEmbeddings; //! All rows of embedding values.
        void * mIndexRemapping; //! Map from row index to frequency-sorted row index.
        void * mTableOffsets; //! Offsets of each table's rows in global row list.
        int mCounter;
    };

private:
    std::string mNamespace;

    bool mInitialized;
    int mDeviceId;
    static std::mutex mSharedDataMutex;
    static HostData mHostData;
    static std::map<int, DeviceData *> mDeviceData; //! Map of GPU device to its local embedding data.
    DeviceData * mLocalDeviceData;
    void * mHelperData;

    int mTotalInteractionFeatures;
    float mInScale;
    float mOutScale;

    int mEmbeddingSize; //! Number of embedding values in one row
    int mEmbeddingRows; //! Number of embedding rows
    int mReducedPrecisionIO;
    float mEmbeddingWeightsOnGpuPart; //! Fraction of embedding rows to keep on GPU.
    int mInteractionsOutputInterleaved;
    int mOutputPaddingGranularity; //! Padding multiple size for the output of the plugin
    std::vector<int> mTableOffsets; //! Index offsets into global rows for each table.
    std::string mEmbeddingWeightsFilepath;
    std::string mRowFrequenciesFilepath;

    bool mCompressCategoricalInputs{false};
};

class DLRMInteractionsPluginCreator : public IPluginCreator
{
public:
    DLRMInteractionsPluginCreator();

    ~DLRMInteractionsPluginCreator() noexcept override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1
