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

#include "engine_llm.h"
#include "llm_utils.h"

#include <mutex>
#include <condition_variable>
#include <pthread.h>
#include <vector>

namespace moffett {
namespace spu_backend {

class GptJEngine : public LlmEngine {
 public:
  GptJEngine();
  GptJEngine(int device_id);
  bool Init(const std::string &deploy_path) override;
  bool Infer(const std::vector<uint32_t *> &prompt_token_ids,
             std::vector<std::vector<int64_t>> &predict_token_ids,
             MFStream stream) override;
  bool End() override;
  bool GetEngineConfig(const ConfigType &type, LlMEngineConfig *config) override;
  void layer_norm(float *data, size_t length);
  float dot_product(const float *a, const float *b);
 protected:
  static void stream_callback(MFStream stream, MFResult status, void *data);
  static void *load_model_to_device(void *param);

  bool is_real_input(const std::string &input);
  bool is_stationary_input(const std::string &input);
  bool prepare_device_bar_memory(const json &module_infos);
  bool prepare_stationary_input(const json &module_infos, const char *model_root_path);
  MFResult prepare_global_memory(const char *model_bin, int device_count, uint64_t value = 0);
  int get_real_model_num(const json &module_infos);
  bool parse_module_infos(const json &module_infos, const char *model_root_path);
  void config_output_dma_op();
  void load_model_to_device();
  bool compare_memory(const void *a, const void *b, size_t size);
  void modify_path(const std::string &path);

  void init_sampling(const PostInitParams &params);
  void sampling(const std::vector<std::vector<uint8_t *>> &embedding,
                const std::vector<std::vector<uint8_t *>> &topk_list,
                std::vector<std::vector<int64_t>> &result,
                int token_num);
  void end_sampling();

 protected:
  // llm
  gpt_vocab vocab;

  std::string model_bin_path = "/sub_1/model.bin";
  std::map<std::string, ModelOpDesc> model_map;
  std::map<int, ModelOpDesc> model_idx_map;
  std::vector<std::string> model_load_list;
  std::unordered_map<int, uint32_t> device_model_count;
  std::unordered_map<int, std::vector<std::string>> device_model_list;
  std::unordered_map<int, DeviceBarMemory> device_bar_memory;
  int launch_device = 0;
  MFModule launch_model = nullptr;
  std::vector<uint8_t *> compare_result;
  std::vector<size_t> compare_size;
  std::vector<uint16_t *> bloom_host_input;
  std::vector<uint8_t *> bloom_output;
  std::vector<void *> bloom_device_output;
  std::vector<uint32_t> bloom_output_size;
  std::vector<size_t> last_output_device;
  std::vector<std::string> real_input_model;

  uint8_t *stationary_host_input = nullptr;
  uint32_t stationary_size = 0;
  std::unordered_map<std::string, uint32_t> stationary_offset;
  std::vector<uint32_t> preprocess_stationary_offset;
  uint32_t pcie_use_bandwidth = 0;

  uint32_t model_batch_size = 0;
  uint32_t model_token_num = 0;
  uint32_t embd_result_size = 0;
  uint32_t topk_result_size = 0;
  uint32_t logits_result_size = 0;

  int actual_model_num = 0;
  int lm_head_part_0 = 999999;
  int lm_head_part_16 = 999999;
  size_t lm_offset = 0;

  int embedding_model_num = 0;
  int topk_model_num = 0;
  int model_topk = 0;
  int output_model_num = 0;
  int lm_head_num = 0;
  int embd_dim = 0;
  int max_seq_len = 0;
  int max_token_len = 0;
  int past_kv_c = 0;

  bool output_token = false;
  bool enable_profile = false;

  int debug_model_id = -1;
  int debug_model_device = -1;
  void *debug_model_input = nullptr;
  uint32_t debug_model_input_size = 0;
  size_t debug_model_offset = 0;

  // extra for split
  int voc_size = 0;
  size_t one_token_size = 0;
  std::vector<uint8_t> attention_masks;
  std::vector<uint8_t> token_id_to_embd_map;
  std::vector<uint8_t> scatter_index_0_map;
  std::vector<uint8_t> scatter_index_1_map;
  std::vector<uint8_t> scatter_index_2_map;
  std::vector<uint8_t> causal_mask_map;
  std::vector<uint8_t> cos_map;
  std::vector<uint8_t> sin_map;
  uint16_t *attention_mask = nullptr;
  uint16_t *causal_mask = nullptr;
  uint16_t *cos = nullptr;
  uint16_t *sin = nullptr;
  uint32_t *scatter_index_0 = nullptr;
  uint32_t *scatter_index_1 = nullptr;
  uint32_t *scatter_index_2 = nullptr;
  uint16_t *bloom_input_embd = nullptr;

  std::vector<std::vector<uint8_t *>> embd_result;
  std::vector<std::vector<uint8_t *>> topk_result;
  std::vector<std::vector<uint8_t *>> logits_result;

 public:
  // sampling
  int N = 0;
  int max_seq_length = 0;
  bool enable_nncore_sort = true;
  int total_topk_num = 0;
  int topk_model_count = 0;
  int model_topk_num = 0;
  bool model_bf16 = false;
  int batch_size = 1;
  const int NUM_TOKEN = 1;
  const int TOKEN_LENGTH = 1;
  const int NUM_THREAD = 16;
  int NUM_ITER = 0;
  int run_token_num = 0;
  size_t token_size = 0;
  int8_t *emb_i;
  float *emb_f;
  float *weights;
  float *bias;
  int64_t *topk;
  float *layer_norm_scale;
  float *layer_norm_bias;

  std::mutex batch_mtx;
  std::condition_variable batch_cv;
  int batch_task[64];
  std::mutex batch_finish_mtx;
  std::condition_variable batch_finish_cv;
  std::vector<std::vector<int64_t>> dot_result;
  std::vector<std::vector<uint8_t *>> batch_embd;
  std::vector<std::vector<float *>> batch_embd_float;
  std::vector<std::vector<int32_t *>> batch_topk;
  std::vector<std::vector<gpt_vocab::id>> batch_repeat_history;
  bool batch_finish = false;
  bool batch_need_exit = false;

  uint8_t *embed;
  pthread_t batch_threads[64];
};

}
}
