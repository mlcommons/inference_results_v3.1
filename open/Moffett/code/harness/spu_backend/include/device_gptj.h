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

#include "device.h"
#include "engine_llm.h"

namespace moffett {
namespace spu_backend {

class DeviceGptJ : public Device {
 public:
  DeviceGptJ(const std::string &name, int device_id, const BackendSettings &settings,
         const ResultProcessFun &result_process, const InputDataPrepareFun &data_process)
      : Device(name, device_id, settings, result_process, data_process)
      {}

  void Init() override;
  void OfflineInfer(BatchUniquePtr batch)
  noexcept override;

 protected:
  void _dataPrepare() override;

 private:
  MFStream stream_;
  std::unique_ptr<LlmEngine> engine_;
};

}
}
