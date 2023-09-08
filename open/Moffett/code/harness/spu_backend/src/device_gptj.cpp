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

#include "device_gptj.h"

#include "syncqueue.h"

namespace moffett {
namespace spu_backend {

void DeviceGptJ::Init() {
  engine_ = CreateLlmEngine("gptj", device_id_);
  engine_->Init(module_path_);

  datacombine_thread_ = std::make_unique<std::thread>(&DeviceGptJ::_dataPrepare, this);

  mfrtSetDevice(device_id_);
  mfrtSetDeviceMode(MF_MODE_FOUR_CORE_BROADCAST);
  mfrtStreamCreate(&stream_);
}

void DeviceGptJ::OfflineInfer(BatchUniquePtr batch) noexcept {
  if (batch->sampleIds->empty()) {
    return;
  }

  batch->input_bindings = &(batch->datas);

  size_t model_batch_size = batch_size_;
  size_t total_count = batch->datas[0].size();
  size_t infer_count = (total_count + (model_batch_size - 1)) / model_batch_size;
  printf("Device %d, total count %d, infer_count %d, model_batch %d \n", device_id_, total_count, infer_count, model_batch_size);
  std::vector<std::vector<uint32_t*>> inputs = std::vector<std::vector<uint32_t*>>(infer_count, std::vector<uint32_t*>(model_batch_size));
  for (int i=0;i<inputs.size();++i) {
    for (int j=0;j<inputs[0].size();++j) {
      if (i * model_batch_size + j >= total_count) {
        inputs[i][j] = reinterpret_cast<uint32_t*>((*(batch->input_bindings))[0][0]);
      } else {
        inputs[i][j] = reinterpret_cast<uint32_t*>((*(batch->input_bindings))[0][i * model_batch_size + j]);
      }
    }
  }

  std::vector<std::vector<std::vector<int64_t>>> total_output;

  for (int batch_idx = 0; batch_idx < infer_count; batch_idx++) {
    std::vector<std::vector<int64_t>> predict_token_id;

    engine_->Infer(inputs[batch_idx], predict_token_id, stream_);

    total_output.emplace_back(predict_token_id);
    if (batch_idx % 10 == 0) {
      printf("[Infer] device %d finish %d\n", device_id_, batch_idx);
    }
  }

  {
    TIME("_completion result process:", device_idx_);

    std::vector<std::vector<uint8_t*>> output_bindings(1, std::vector<uint8_t*>(total_count));
    for (int i=0;i<infer_count;++i) {
      for (int j=0;j<model_batch_size;++j) {
	if (i * model_batch_size + j >= total_count) break;
        output_bindings[0][i * model_batch_size + j] = reinterpret_cast<uint8_t*>(total_output[i][j].data());
      }
    }

    if (result_processor_) {
      result_processor_(batch.get(), &output_bindings, real_output_size_, batch_size_);
    }
  }
}

void DeviceGptJ::_dataPrepare() {
  pthread_setname_np(pthread_self(), _generateThreadNameWithCardId("dPrep").c_str());

  auto batch = std::make_unique<Batch>();
  for (;;) {
    {
      TIME("_dataPrepare get request: ", device_id_);
      batch = POP_QUEUE(request_queue_);
      if (batch->sampleIds->empty()) {
        break;
      }
    }
    OfflineInfer(std::move(batch));
    // if (server_settings_.mode == 0) {

    //   continue;
    // }
    // {
    //   TIME("_dataPrepare data prepare: ", device_id_);
    //   data_preparer_(nullptr, real_input_size_, batch.get(), batch_size_, device_id_,
    //                  server_settings_, sample_size_);
    // }

    // {
    //   TIME("_combineMempush batch back:", device_idx_);
    //   PUSH_QUEUE(copied_data_queue_, std::move(batch));
    // }
  }
  engine_->End();
}

}
}
