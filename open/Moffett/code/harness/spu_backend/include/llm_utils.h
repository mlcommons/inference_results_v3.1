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

#ifndef SAMPLES_GPT_UTILS_H_
#define SAMPLES_GPT_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <random>
#include <thread>
// #include "common.h"

namespace moffett {
namespace spu_backend {

struct gpt_vocab {
  using id = int32_t;
  using token = std::string;

  std::map<token, id> token_to_id;
  std::map<id, token> token_id_to_string;
};

struct TokenItem {
  std::vector<int32_t> input_ids;
  std::vector<int8_t> attention_mask;
};

struct PostInitParams {
  std::string root_path;
  int input_length;
  int max_seq_len;
  int topk_model_num;
  int model_topk;
  int voc_size;
  int batch_size = 1;
  int token_num = 1;
  bool enable_nncore_sort;
  bool is_model_bf16;
};

std::vector<uint8_t> read_binary(const char *fname);
gpt_vocab ParseVocab(const std::string &file_name);

}
}

#endif
