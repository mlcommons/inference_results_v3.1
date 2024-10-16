/*
 * Copyright © 2022 Shanghai Biren Technology Co., Ltd. All rights reserved.
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

/*
 * Copyright © 2023 Moffett System Inc. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

#include "loadgen.h"
#include "samples_loader.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "utils.hpp"
#include "timer.h"
#include "syncqueue.h"

#include "device.h"

namespace moffett {
namespace spu_backend {

using namespace std::chrono_literals;
using namespace moffett;

class Server;

using ServerPtr_t = std::shared_ptr<Server>;

struct ServerParams {
  std::string device_ids;
  std::string scenario;
  std::vector<std::vector<std::vector < std::string>>> EngineNames;
};

// Create buffers and other execution resources.
// Perform queuing, batching, and manage execution resources.
class Server : public mlperf::SystemUnderTest {
 public:
  // Query management
  using BatchQueue = QUEUE_TYPE<std::vector<mlperf::QuerySample *>*>;

  Server(std::string name) : name_(name) {}
  ~Server() {
    for (auto &thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void AddSampleLibrary(std::size_t device_idx, samplesLoader::SampleLoaderPtr sl)
  noexcept {
    if (device_idx >= sample_libraries_.size()) {
      sample_libraries_.resize(device_idx + 1);
    }
    sample_libraries_[device_idx] = sl;
  }

  void Setup(const BackendSettings &settings, const ServerParams &params,
             ResultProcessFun result_process, InputDataPrepareFun data_prepare)
  noexcept;
  void Done()
  noexcept;
  const std::vector<DevicePtr_t> &GetDevices() const
  noexcept { return devices_; }

  // SUT virtual interface
  const std::string &Name() override { return name_; }
  void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override;
  void FlushQueries() override;

 private:

  template<typename T>
  void MergeQueryDispatcher(QUEUE_TYPE<T *> *queue)
  noexcept;

  inline void BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                         std::vector<mlperf::QuerySample *>::iterator begin,
                         std::vector<mlperf::QuerySample *>::iterator end, std::size_t samples_cnt,
                         Batch &batch)
  noexcept;
  inline void IssueBatch(DevicePtr_t device, std::size_t batch_size, std::vector<mlperf::QuerySample *>::iterator begin,
                         std::vector<mlperf::QuerySample *>::iterator end)
  noexcept;

  DevicePtr_t GetNextDispatchDevice()
  noexcept;
  void Reset()
  noexcept;

 private:
  const std::string name_;
  std::size_t device_index_;
  std::vector<std::size_t> device_idxs_;
  std::vector<DevicePtr_t> devices_;
  std::size_t valid_device_num_{0};
  std::size_t input_num_;
  std::vector<samplesLoader::SampleLoaderPtr> sample_libraries_;

  BackendSettings server_settings_;
  std::string scenario_;

  BatchQueue request_queue_;

  // Not use shared_ptr for performance considerations.
  std::vector<std::thread> threads_;
};

}
}
