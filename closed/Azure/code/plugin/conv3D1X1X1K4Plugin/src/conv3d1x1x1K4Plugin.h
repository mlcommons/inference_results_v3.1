/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#ifndef TRT_CONV_3D_1X1X1_K4_PLUGIN_H
#define TRT_CONV_3D_1X1X1_K4_PLUGIN_H
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <cudnn.h>

#include "serialize.hpp"
#include "plugin.h"
#include "conv3d_1x1x1_k4.h"

typedef uint16_t half_type;

namespace nvinfer1
{
namespace plugin
{
class conv3D1X1X1K4Plugin final : public nvinfer1::IPluginV2DynamicExt
{

public:
    conv3D1X1X1K4Plugin(int inputChannels, const std::vector<float>& weights, const std::vector<float>& bias);
    conv3D1X1X1K4Plugin(void const* serialData, size_t serialLength);

    conv3D1X1X1K4Plugin() = delete;

    ~conv3D1X1X1K4Plugin() override;

    int getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    using nvinfer1::IPluginV2::getOutputDimensions;
    DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    using nvinfer1::IPluginV2::getWorkspaceSize;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    using nvinfer1::IPluginV2::enqueue;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

private:
    int mInputChannels;
    void* mDeviceWeights;
    void* mDeviceBias;

    std::vector<float> mWeights;
    std::vector<float> mBias;
    float mWeightScale;
    float mInActivationScale, mOutActivationScale;

    const char* mPluginNamespace;
    std::string mNamespace;
    bool mInitialized{false};

    // NCDHW32 implementation
    Conv3d1x1x1k4Params mParams;
    Conv3d1x1x1k4Context mContext;

    // LINEAR implementation
    cudnnHandle_t mCudnnHandle;
    cudnnTensorDescriptor_t mOutDesc, mImageDesc, mBiasDesc;
    cudnnFilterDescriptor_t mFltDesc;
    cudnnConvolutionDescriptor_t mConvDesc;
    cudnnActivationDescriptor_t mIdentityDesc;

    size_t setCudnnDescriptors(const nvinfer1::PluginTensorDesc* inputs) const;
};

class conv3D1X1X1K4PluginCreator : public BaseCreator
{
public:
    conv3D1X1X1K4PluginCreator();

    ~conv3D1X1X1K4PluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} //namespace plugin
} //namespace nvinfer1

#endif // TRT_CONV_3D_1X1X1_K4_PLUGIN_H
