/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
 *******************************************************************************/

#ifndef RUNTIME_RUNTIME_HPP
#define RUNTIME_RUNTIME_HPP
#include <stddef.h>
#include <stdint.h>
#include <util/def.hpp>

extern "C" {

SC_API void print_float(float f);
SC_API void print_index(uint64_t f);
SC_API void print_int(int f);
SC_API void print_str(char *f);
SC_API uint64_t boundary_check(
        const char *name, uint64_t idx, uint64_t acc_len, uint64_t tsr_len);
SC_API void *sc_global_aligned_alloc(size_t sz, size_t align);
SC_API void sc_global_aligned_free(void *ptr, size_t align);
SC_API void sc_make_trace(int id, int in_or_out);
SC_API void sc_dump_tensor(void *tsr, const char *name, const char *shape,
        size_t size, size_t limit, const char *path, bool binary_fmt,
        uint64_t idtype);
SC_API void sc_value_check(void *tsr, const char *name, size_t size);
};

namespace sc {
namespace runtime {
struct engine_t;
}
SC_API void release_runtime_memory(runtime::engine_t *engine);
} // namespace sc

#endif
