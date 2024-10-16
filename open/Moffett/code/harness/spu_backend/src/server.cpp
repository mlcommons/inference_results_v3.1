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

#include <algorithm>
#include <fstream>
#include <glog/logging.h>
#include <math.h>
#include <memory>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <unistd.h>
#include <hwloc.h>
#include "common.h"
#include "server.h"
#include "loadgen.h"
#include "query_sample_library.h"
#include "timer.h"

#include "device_gptj.h"

uint64_t g_samples_size = 0;
using namespace std::chrono_literals;

namespace moffett {
namespace spu_backend {

void Server::Setup(const BackendSettings& settings, const ServerParams& params,
                   ResultProcessFun result_process, InputDataPrepareFun data_prepare) noexcept {
  server_settings_ = settings;
  scenario_ = params.scenario;
  device_idxs_ = server_settings_.devices;

  Reset();

  // GPTJ only need to operate one device
  if (name_.find("GPTJ") != std::string::npos) {
    for (auto device_idx : device_idxs_) {
      auto dev = std::make_shared<DeviceGptJ>(name_ + "_" + std::to_string(device_idx), device_idx,
                                        server_settings_, result_process, data_prepare);
      dev->Init();
      devices_.emplace_back(dev);
    }
  } else {
    // We assume there will be no device failures during running.
    std::vector<std::thread> device_creators;
    valid_device_num_ = device_idxs_.size();

    size_t max_size_per_one_card = 0;

    device_creators.reserve(valid_device_num_ + 1);
    auto creator = [&](size_t device_idx) {
      auto dev = std::make_shared<Device>(name_ + "_" + std::to_string(device_idx), device_idx,
                                          server_settings_, result_process, data_prepare);
      dev->Init();
      devices_.emplace_back(dev);
    };

    for (auto device_idx : device_idxs_) {
      device_creators.emplace_back(std::thread(creator, device_idx));
    }
    for (auto& t : device_creators) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  // create batchers/dispatcher
  {
    threads_.emplace_back(
      std::thread(&Server::MergeQueryDispatcher<std::vector<mlperf::QuerySample*>>, this, &request_queue_));
  }
}

void Server::Done() noexcept {
  for (auto& device : devices_) {
    if (device) {
      device->done();
    }
  }

  std::size_t thread_idx = 0;
  for (auto& thread : threads_) {
    std::vector<mlperf::QuerySample*>* empty_samples = new std::vector<mlperf::QuerySample*>();
    empty_samples->push_back(std::make_unique<mlperf::QuerySample>().release());
    PUSH_QUEUE(request_queue_, empty_samples);
    ++thread_idx;
  }

  // join after we insert the dummy sample
  for (auto& thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

void Server::Reset() noexcept {
  device_index_ = 0;

  for (auto& device : devices_) {
    device->reset();
  }
}

void Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
  {
    TIME("IssueQuery", 1);
    std::vector<mlperf::QuerySample*>* tmp_samples = new std::vector<mlperf::QuerySample*>();
    tmp_samples->reserve(samples.size());

    for (auto iter = samples.begin(); iter != samples.end(); ++iter) {
      tmp_samples->push_back((mlperf::QuerySample*)&(*iter));
    }
    if (server_settings_.mode == 0 )
    {  // Offline
      auto vir_devices = GetDevices();
      printf("[Offline] Issuequery samples size = %d, vir_device size = %d\n", samples.size(), vir_devices.size());
      size_t begin = 0;
      size_t batch_size = samples.size() / vir_devices.size();
      size_t residue = samples.size() % vir_devices.size();
      size_t expand_batch = batch_size + 1;
      for (size_t i = 0; i < vir_devices.size(); ++i) {
        if (residue > 0 && i < residue) {
          IssueBatch(vir_devices[i], expand_batch, tmp_samples->begin() + begin, tmp_samples->begin() + begin + expand_batch);
          begin += expand_batch;
        } else {
          IssueBatch(vir_devices[i], batch_size, tmp_samples->begin() + begin, tmp_samples->begin() + begin + batch_size);
          begin += batch_size;
        }
      }
      delete tmp_samples;
    }
    else
      PUSH_QUEUE(request_queue_, tmp_samples);
  }
}

DevicePtr_t Server::GetNextDispatchDevice() noexcept {
  auto vir_devices = GetDevices();
  auto dev_index = device_index_;
  device_index_ = (device_index_ + 1) % vir_devices.size();
  return vir_devices[dev_index];
}

template<typename T>
void Server::MergeQueryDispatcher(QUEUE_TYPE<T*>* queue) noexcept {
  pthread_setname_np(pthread_self(), std::string("dispatcher").c_str());

  std::deque<T*> samples;
  auto device = GetNextDispatchDevice();
  std::vector<mlperf::QuerySample*> tmp_samples;
  tmp_samples.reserve(device->getBatchSize());
  bool quit{false};
  auto first_query_arrive_time = std::chrono::steady_clock::now();
  auto timeout = server_settings_.target_latency_ms - std::chrono::microseconds{8000us};

  for (; !quit;) {
    samples.clear();
    {
      ACQUIRE_QUEUE_TIMEOUT(request_queue_, samples, timeout);
    }

    if (samples.empty() && tmp_samples.empty()) {
      continue;
    }

    // merge samples
    {
      TIME("process samples", 99);
      auto now_time = std::chrono::steady_clock::now();
      // Use a null (0) id to represent the end of samples
      if (tmp_samples.empty()) {
        first_query_arrive_time = now_time;
      }

      auto batch_size = device->getBatchSize();

      for (auto iter = samples.begin(); iter != samples.end(); ++iter) {
        std::vector<mlperf::QuerySample*>* p_sample_vec = *iter;
        for (auto p_sample : *p_sample_vec) {
          // Use a null (0) id to represent the end of samples
          if (!p_sample->id) {
            quit = true;
            break;
          }
          tmp_samples.push_back(p_sample);
          g_samples_size++;
          if (tmp_samples.size() >= batch_size) {
            {
              TIME("issue batch", 99);
              IssueBatch(device, batch_size, tmp_samples.begin(), tmp_samples.end());
            }
            device = GetNextDispatchDevice();
            batch_size = device->getBatchSize();
            tmp_samples.clear();
            // refresh timer
            first_query_arrive_time = now_time;
          }
        }
        delete p_sample_vec;
      }

      // dispatch when full or timedout
      if ((now_time - first_query_arrive_time >= timeout) || (tmp_samples.size() >= device->getBatchSize())) {
        auto batch_size = std::min(device->getBatchSize(), tmp_samples.size());
        printf("infer the remain sample: %d\n", batch_size);
        if (batch_size > 0) {
          TIME("issue batch", 99);
          IssueBatch(device, batch_size, tmp_samples.begin(), tmp_samples.end());
        }
        device = GetNextDispatchDevice();
        tmp_samples.clear();
      }
    }
  }
}

void Server::IssueBatch(DevicePtr_t device, std::size_t batch_size, std::vector<mlperf::QuerySample*>::iterator begin,
                        std::vector<mlperf::QuerySample*>::iterator end) noexcept {
  assert(!sample_libraries_.empty());
  std::size_t device_idx = device->getDeviceIdx();
  auto batch = std::make_unique<Batch>();

  // This can be member viriable which is set in setup.
  //std::size_t input_num = device->getInputNum();
  std::size_t input_num = 1;
  if (input_num == 0) {
    return;
  }

  batch->datas.resize(input_num);
  batch->responses->reserve(batch_size);
  batch->sampleIds->reserve(batch_size);

  for (auto itr = begin; itr < end; ++itr) {
    batch->responses->push_back({(*itr)->id, 0, 0});
    batch->sampleIds->emplace_back((*itr)->index);
  }

  for (std::size_t input_idx = 0; input_idx < input_num; ++input_idx) {
    BuildBatch(sample_libraries_[device_idx], input_idx, device_idx, begin, end, batch_size, *batch);
  }
  device->infer(std::move(batch));
}

void Server::BuildBatch(samplesLoader::SampleLoaderPtr sl, std::size_t input_idx, std::size_t device_idx,
                        std::vector<mlperf::QuerySample*>::iterator begin,
                        std::vector<mlperf::QuerySample*>::iterator end, std::size_t samples_cnt,
                        Batch& batch) noexcept {
  auto& data_ref = batch.datas[input_idx];
  data_ref.resize(samples_cnt);
  std::size_t sample_idx = 0;
  for (auto itr = begin; itr < end; ++itr, ++sample_idx) {
    auto data_adders = sl->GetSampleAddress((*itr)->index, input_idx, server_settings_.numa_map[device_idx], device_idx);
    data_ref[sample_idx] = data_adders;
  }
}

void Server::FlushQueries() {
  // Not used.
}

} // namespace spu_backend
}
