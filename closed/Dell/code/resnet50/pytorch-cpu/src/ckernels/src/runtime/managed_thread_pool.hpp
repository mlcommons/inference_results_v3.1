/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef RUNTIME_MANAGED_THREAD_POOL_HPP
#define RUNTIME_MANAGED_THREAD_POOL_HPP
#include <atomic>
#include <runtime/context.hpp>

namespace sc {

namespace runtime {
struct thread_manager {
    struct thread_pool_state {
        struct task_type {
            void (*pfunc)(void *, void *, int64_t, sc::generic_val *);
            void *stream;
            void *module_env;
            int64_t begin;
            int64_t end;
            int64_t step;
            sc::generic_val *args;
        } task;
        std::atomic<int> trigger;
        int num_threads;
        std::atomic<int> remaining;

        void wait_all();
        void reset_scoreboard();
    } state;
    thread_manager();
    using main_func_t = void (*)(runtime::stream_t *, void *, generic_val *);
    void run_main_function(main_func_t f, runtime::stream_t *stream,
            void *mod_data, generic_val *args);
    static thread_local thread_manager cur_mgr;
};
} // namespace runtime
} // namespace sc

extern "C" SC_API void sc_parallel_call_managed(
        void (*pfunc)(void *, void *, int64_t, sc::generic_val *),
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, sc::generic_val *args);

#endif
