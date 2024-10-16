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

#ifndef __DLRMv2_QSL_H__
#define __DLRMv2_QSL_H__

#include "qsl.hpp"
#include "utils.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <iterator>


class DLRMv2SampleLibrary : public qsl::SampleLibrary
  {
  public:
    DLRMv2SampleLibrary(std::string name, std::string mapPath, std::vector<std::string> tensorPaths,
      std::vector<int> samplePartition, size_t perfSampleCount, size_t perfPairCount, size_t padding = 0,
      bool coalesced = false, bool startFromDevice = false, std::vector<int32_t> numaIndices = {}, 
      std::map<int32_t, int32_t> deviceOffsetMap = std::map<int32_t, int32_t>(), int32_t numNuma = 0)
    : mSampleStartIdxs(samplePartition)
      , mPerfSampleCount(perfSampleCount)
      , mTotalSampleCount(samplePartition.size() - 1)
      , qsl::SampleLibrary(
        name, mapPath, tensorPaths, perfPairCount, padding, coalesced, {startFromDevice, startFromDevice})
      , mNumIndividualPairs(std::min(perfPairCount, qsl::SampleLibrary::TotalSampleCount()))
      , mNumaIndices(numaIndices)
      , mQSLMappingNumNodes(numaIndices.size())
      , mDeviceOffsetMap(deviceOffsetMap)
      , mNumNuma(numNuma)
    {
      CHECK_EQ(mSampleStartIdxs.back(), mNumIndividualPairs);
      LOG(INFO) << "PerformanceSampleCount: " << mPerfSampleCount;
      LOG(INFO) << "TotalSampleCount: " << mTotalSampleCount << " (" << mNumIndividualPairs << " pairs).";

      if (startFromDevice)
        CHECK_LE(numNuma, 1);

      std::ostringstream indices_csv;
      std::copy(mNumaIndices.begin(), mNumaIndices.end(), std::ostream_iterator<int32_t>(indices_csv, ","));
      mNumaIndicesStr = indices_csv.str();
      mNumaIndicesStr.pop_back();
    }

    size_t TotalSampleCount() override
    {
      return mTotalSampleCount;
    }
    size_t PerformanceSampleCount() override
    {
      return mPerfSampleCount;
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
      LOG(INFO) << "Calling LoadSamplesToRam() for QSL[" << (UseNuma() ? mNumaIndicesStr : std::to_string(0)) << "] of " << samples.size()
        << " samples...";

      // Set memory allocation setting
      if (UseNuma())
      {
        bindNumaMemPolicy(mNumaIndices, mNumNuma);
      }

      auto num_dev = UseNuma() ? 1 : m_NumDevices;
      sampleAddresses.resize(num_dev * mTotalSampleCount * m_NumInputs);

      auto remappedSamples = mapToIndividualIndices(samples);

      // copy the samples into pinned memory
      for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
      {
        std::vector<char> data;
        m_NpyFiles[input_idx]->streamSamples(data, remappedSamples);

        auto sampleSize = m_SampleSizes[input_idx];

        for (int device_idx = 0; device_idx < m_NumDevices; device_idx++)
        {
          auto sampleAddress = m_SampleMemory[input_idx][device_idx];

          if (!m_StartFromDevice[input_idx])
          {
            memcpy((char*) sampleAddress, data.data(), remappedSamples.size() * sampleSize);
          }
          else
          {
            CHECK_EQ(cudaSetDevice(device_idx), cudaSuccess);
            CHECK_EQ(cudaMemcpy(
              sampleAddress, data.data(), remappedSamples.size() * sampleSize, cudaMemcpyHostToDevice),
              cudaSuccess);
          }

          std::map<mlperf::QuerySampleIndex, char*> remappedSampleAddresses;
          for (size_t sampleIndex = 0; sampleIndex < remappedSamples.size(); sampleIndex++)
          {
            auto sampleId = remappedSamples[sampleIndex];
            remappedSampleAddresses[sampleId]
              = static_cast<char*>(sampleAddress) + sampleIndex * sampleSize;
          }

          //for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
          for (auto& sampleId: samples)
          {
            //auto sampleId = samples[sampleIndex];
            auto remappedSampleId = mSampleStartIdxs[sampleId];
            sampleAddresses[(device_idx * mTotalSampleCount + sampleId) * m_NumInputs + input_idx] = \
              static_cast<void*>(remappedSampleAddresses[remappedSampleId]);
          }
        }
      }

      if (std::any_of(m_StartFromDevice.begin(), m_StartFromDevice.end(), [](bool i) { return i == true; }))
      {
        for (int device_idx = 0; device_idx < num_dev; device_idx++)
        {
          CHECK_EQ(cudaSetDevice(device_idx), cudaSuccess);
          CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }
      }

      // Reset memory allocation setting
      if (UseNuma())
      {
        resetNumaMemPolicy();
      }

      LOG(INFO) << "Completed LoadSamplesToRam() for QSL[" << (UseNuma() ? mNumaIndicesStr : std::to_string(0)) << "]";
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
      LOG(INFO) << "Calling UnloadSamplesFromRam() for QSL[" << (UseNuma() ? mNumaIndicesStr : std::to_string(0)) << "] of "
        << samples.size() << " samples...";

      sampleAddresses.clear();

      LOG(INFO) << "Completed UnloadSamplesFromRam() for QSL[" << (UseNuma() ? mNumaIndicesStr : std::to_string(0)) << "]";
    }

    void* GetSampleAddress(
      mlperf::QuerySampleIndex sampleIdx, size_t inputIdx, size_t pairIdx = 0, size_t deviceIdx = 0)
    {
      auto d_idx = UseNuma() ? 0 : deviceIdx;
      auto addr
        = static_cast<char*>(sampleAddresses[(d_idx * mTotalSampleCount + sampleIdx) * m_NumInputs + inputIdx])
          + pairIdx * qsl::SampleLibrary::GetSampleSize(inputIdx);
      return addr;
    }

    size_t GetNumUserItemPairs(size_t index)
    {
      size_t start = mSampleStartIdxs[index];
      size_t end = mSampleStartIdxs[index + 1];
      return end - start;
    }

    inline bool UseNuma()
    {
      return mNumNuma > 0;
    };


  private:
    std::vector<mlperf::QuerySampleIndex> mapToIndividualIndices(const std::vector<mlperf::QuerySampleIndex>& samples)
    {
      std::vector<mlperf::QuerySampleIndex> remappedIndices;

      for (auto index : samples)
      {
        CHECK_EQ((index >= 0) && (index < mTotalSampleCount), true);
        int start = mSampleStartIdxs[index];
        int end = mSampleStartIdxs[index + 1];
        for (int i = start; i < end; ++i)
        {
          remappedIndices.push_back(i);
        }
      }
      return remappedIndices;
    }

    int mNumIndividualPairs;
    size_t mPerfSampleCount;
    size_t mTotalSampleCount;
    std::vector<int> mSampleStartIdxs;
    std::vector<void*> sampleAddresses;
    int32_t mNumNuma;
    std::vector<int32_t> mNumaIndices;
    int32_t mQSLMappingNumNodes;
    std::string mNumaIndicesStr;
    std::map<int32_t, int32_t> mDeviceOffsetMap;
  };

typedef std::shared_ptr<DLRMv2SampleLibrary> DLRMv2SampleLibraryPtr_t;

class DLRMv2SampleLibraryEnsemble : public mlperf::QuerySampleLibrary
  {
  public:
    DLRMv2SampleLibraryEnsemble(const std::vector<DLRMv2SampleLibraryPtr_t> qsls)
    : m_qsls(qsls){};
    const std::string& Name() override
    {
      return m_qsls[0]->Name();
    }
    size_t TotalSampleCount() override
    {
      return m_qsls[0]->TotalSampleCount();
    }
    size_t PerformanceSampleCount() override
    {
      return m_qsls[0]->PerformanceSampleCount();
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
      LOG(INFO) << "Calling LoadSamplesToRam() for QSL ensemble...";
      for (auto qsl : m_qsls)
      {
        qsl->LoadSamplesToRam(samples);
      }
      LOG(INFO) << "Completed LoadSamplesToRam() for QSL ensemble.";
    }
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
      LOG(INFO) << "Calling UnloadSamplesFromRam() for QSL ensemble...";
      for (auto qsl : m_qsls)
      {
        qsl->UnloadSamplesFromRam(samples);
      }
      LOG(INFO) << "Completed UnloadSamplesFromRam() for QSL ensemble.";
    }

  private:
    std::vector<DLRMv2SampleLibraryPtr_t> m_qsls;
  };

#endif // __DLRMv2_QSL_H__
