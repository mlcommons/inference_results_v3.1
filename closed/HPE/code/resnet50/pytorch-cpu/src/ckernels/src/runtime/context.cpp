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

#include "context.hpp"
#include <assert.h>
#include "memorypool.hpp"
#include "parallel.hpp"
#include "runtime.hpp"

namespace sc {

namespace runtime {
static void *global_alloc(runtime::engine_t *eng, size_t sz) {
    return sc_global_aligned_alloc(sz, 64);
}

static void global_free(runtime::engine_t *eng, void *p) {
    return sc_global_aligned_free(p, 64);
}

static engine_vtable_t vtable {global_alloc, global_free,
        sc::memory_pool::alloc_by_mmap, sc::memory_pool::dealloc_by_mmap};

static engine_t default_engine {&vtable};
stream_t default_stream {{sc_parallel_call_cpu_with_env_impl}, &default_engine};

static_assert(sizeof(stream_vtable_t) == sizeof(void *),
        "stream_vtable_t has more than 1 field. Need to change "
        "stream_t::vtable_ to pointer now");

} // namespace runtime
} // namespace sc
