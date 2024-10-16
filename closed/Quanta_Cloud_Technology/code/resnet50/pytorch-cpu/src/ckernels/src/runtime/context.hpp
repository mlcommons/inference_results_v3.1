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

#ifndef RUNTIME_CONTEXT_HPP
#define RUNTIME_CONTEXT_HPP

#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <util/def.hpp>
namespace sc {
union generic_val;

namespace runtime {

struct engine_t;

struct engine_vtable_t {
    using alloc_t = void *(*)(engine_t *, size_t);
    using dealloc_t = void (*)(engine_t *, void *);
    alloc_t persistent_alloc;
    dealloc_t persistent_dealloc;
    alloc_t temp_alloc;
    dealloc_t temp_dealloc;
};

struct stream_vtable_t {
    using parallel_call_cpu_t
            = void (*)(void (*)(void *, void *, int64_t, generic_val *), void *,
                    void *, int64_t, int64_t, int64_t, generic_val *);
    parallel_call_cpu_t parallel_call;
    constexpr stream_vtable_t(parallel_call_cpu_t pcall)
        : parallel_call(pcall) {}
};

struct engine_t {
    engine_vtable_t *vtable_;
    engine_t(engine_vtable_t *vtable) : vtable_(vtable) {}
};

struct stream_t {
    // we are using stream_vtable_t instead of stream_vtable_t* because
    // currently stream_vtable_t has only one field. Using the value type
    // instead of the pointer saves a memory access. Need to change back to
    // pointer if the vtable has move than 1 field
    stream_vtable_t vtable_;
    engine_t *engine_;

    constexpr stream_t(stream_vtable_t vtable, engine_t *engine)
        : vtable_ {vtable}, engine_ {engine} {}
    const stream_vtable_t *vtable() const { return &vtable_; }
};

SC_API extern stream_t default_stream;

inline stream_t *get_default_stream() {
    return &default_stream;
}

} // namespace runtime

} // namespace sc

#endif
