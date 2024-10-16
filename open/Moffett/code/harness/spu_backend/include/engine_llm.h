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

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <sola_runtime.h>
#include <sola_internal.h>
#include "json.hpp"

namespace moffett {
namespace spu_backend {

#define FULL_TOKEN 0
#define DEBUG_PRINT 0

// check the accuracy of each model
#define ACCURACY_VERIFICATION_MODE 0
#define VERIFY_ONE_MODEL 0

#define JSON_CHECK(expr) do {                                             \
  if (!(expr)) {                                                          \
    printf("JSON check format failed in line %d: %s\n", __LINE__, #expr); \
    return false;                                                         \
  } } while (0)

#define TENSOR_SUFFIX  ".raw"

#define ALIGN64(x) (((x) + 63) & ~63)

// ------------------------------------------------------------------------------------------------
// datastruct definition
// ------------------------------------------------------------------------------------------------
using json = nlohmann::json;
using moffett::internal::MFModelConfig;

enum ProgramMode {
    kQuestion = 0,  // 交互问答
    kAuto = 1,      // 自动问答
    kInf = 2,       // 无限问答
    kPPL = 3,       // ppl计算
    kAccuracy = 4,  // 精度验证
};

enum ModelNodeType {
    kNormal = 0, kConcat = 1, kSlice = 2
};

struct InputDesc {
    std::string name;
    size_t size;
    size_t offset;

    InputDesc(const std::string &name) : name(name), size(0), offset(0) {
    }
};

std::ostream &operator<<(std::ostream &os, const MFModelConfig &desc);

struct ModelOpDesc {
    ModelNodeType model_type = kNormal;
    MFMode model_mode = MF_MODE_INVALID;
    int device_id = 0;
    int model_id = 0;
    int part_id = 0; // not used
    uint32_t stationary_input_size{};
    uint32_t stationary_output_size{}; // not used
    uint32_t input_size{};
    uint32_t output_size{};
    std::vector<uint32_t> each_output_size;
    uint32_t real_input_count = 0; // input from outside at launch kernel
    uint32_t slice_offset = 0;
    uint32_t slice_size = 0;
    uint32_t trigger_count = 0;
    std::string model_path;
    MFModelConfig dma_config;
    std::vector<InputDesc> inputs;

    friend std::ostream &operator<<(std::ostream &os, const ModelOpDesc &desc) {
        auto origin_w = os.width(16);
        os << " model_type: " << desc.model_type << std::endl;
        os << " model_mode: " << desc.model_mode << std::endl;
        os << " device_id: " << desc.device_id << std::endl;
        os << " model_id: " << desc.model_id << std::endl;
        os << " part_id: " << desc.part_id << std::endl;
        os << " stationary_input_size: " << desc.stationary_input_size << std::endl;
        os << " stationary_output_size: " << desc.stationary_output_size << std::endl;
        os << " input_size: " << desc.input_size << std::endl;
        os << " output_size: " << desc.output_size << std::endl;
        os << " real_input_count: " << desc.real_input_count << std::endl;
        os << " slice_offset: " << desc.slice_offset << std::endl;
        os << " slice_size: " << desc.slice_size << std::endl;
        os << " trigger_count: " << desc.trigger_count << std::endl;
        os << " model_path: " << desc.model_path << std::endl;
        os << " dma_config: " << desc.dma_config << std::endl;

        os << " inputs: " << std::endl;
        for (auto &&input : desc.inputs) {
            os << " ";
            os << " ";
            os << input.name << " ";
            os << input.size << " ";
            os << input.offset << " ";
            os << std::endl;
        }

        os.width(origin_w);

        return os;
    }
};

struct DeviceBarMemory {
    void *base_addr = nullptr;
    uint32_t offset = 0;
};

class IndexArrayGetter {
 private:
  int max_seq_len;
  int cgb;
  int curr_seq_len;
  std::vector<int> scatter_w_0_full; // curr_len --> insert into --> max_seq_len
  std::vector<int> scatter_w_1_full; //curr_len -- insert into --> cgb
  // std::vector<int> scttter_gcb_full;
  std::vector<int> range;
  std::vector<std::vector<int>> casual_mask_full;

 public:
  IndexArrayGetter(int max_seq_len, int cgb, int curr_seq_len)
      : max_seq_len(max_seq_len), cgb(cgb), curr_seq_len(curr_seq_len) {
    assert(max_seq_len % curr_seq_len == 0);
    assert(cgb % curr_seq_len == 0);
    range.resize(max_seq_len);
    scatter_w_0_full.resize(max_seq_len);
    scatter_w_1_full.resize(max_seq_len);
    // scttter_gcb_full.resize(max_seq_len / cgb);
    casual_mask_full.resize(max_seq_len, std::vector<int>(max_seq_len, 0));

    for (int i = 0; i < max_seq_len; i++) {
      scatter_w_0_full[i] = i;
      range[i] = i;
      scatter_w_1_full[i] = (i % cgb);
      for (int j = 0; j <= i; j++) {
        casual_mask_full[i][j] = 1;
      }
    }

    // for (int i = 0; i < max_seq_len / cgb; i++) {
    // scttter_gcb_full[i] = i;
    // }
  }
  //"0_scatter_Gcb_index",
  std::vector<int> get_scatter_w_index_0(int pos_id) {
    return {scatter_w_0_full.begin() + pos_id / curr_seq_len, scatter_w_0_full.begin() + pos_id / curr_seq_len + 1};
  }

  //0_scatter_W_index_1
  std::vector<int> get_scatter_w_index_1(int pos_id) {
    auto begin = scatter_w_1_full[pos_id] / curr_seq_len;
    auto end = scatter_w_1_full[pos_id] / curr_seq_len + 1;
    return {range.begin() + begin, range.begin() + end};
  }
  //0_scatter_Gcb_index
  std::vector<int> get_scatter_gcb_index(int pos_id) {
    int start = pos_id / cgb;
    int end = (pos_id + curr_seq_len + cgb - 1) / cgb;
    return {range.begin() + start, range.begin() + end};
  }

  void print_casual_mask_full() {
    for (int i = 0; i < max_seq_len; i++) {
      for (int j = 0; j < max_seq_len; j++) {
        std::cout << casual_mask_full[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::vector<std::vector<int>> get_casual_mask(int pos_id, int curr_seq_len) {
    return {casual_mask_full.begin() + pos_id, casual_mask_full.begin() + pos_id + curr_seq_len};
  }
};

struct LlMEngineConfig {};
struct LlMEngineConfigMaxTokenLen : public LlMEngineConfig {
  uint32_t max_token_len = 0;
};
struct LlMEngineConfigVocSize : public LlMEngineConfig {
  int32_t voc_size = 0;
};
struct LlMEnginePcieUseBandwidth : public LlMEngineConfig {
  uint32_t pcie_use_bandwidth = 0;
};
struct LlMEngineGptJCommonConfig : public LlMEngineConfig {
  bool enable_profile = false;
  uint32_t model_batch_size = 0;
  uint32_t model_token_num = 0;
};

class LlmEngine {
 public:
  enum class ConfigType {
    kMaxTokenLen,
    kVocSize,
    kPcieUseBandwidth,
    kGptJCommonConfig,
  };
public:
  virtual ~LlmEngine() {}
  virtual bool Init(const std::string& deploy_path) = 0;
  virtual bool Infer(const std::vector<uint32_t*>& prompt_token_ids,
             std::vector<std::vector<int64_t>>& predict_token_ids,
             MFStream stream) = 0;
  virtual bool End() = 0;
  virtual bool GetEngineConfig(const ConfigType& type, LlMEngineConfig* config) = 0;
};


struct LoadModelToDeviceParam {
  LoadModelToDeviceParam(int device_id, LlmEngine* engine)
  : device_id(device_id), bloom_engine(engine) {}

  int device_id = 0;
  LlmEngine *bloom_engine;
};
std::unique_ptr<LlmEngine> CreateLlmEngine(const std::string& model_name, int device_id);


}
}
