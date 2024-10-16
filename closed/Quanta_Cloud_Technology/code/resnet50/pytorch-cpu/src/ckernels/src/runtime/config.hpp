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

#ifndef RUNTIME_CONFIG_HPP
#define RUNTIME_CONFIG_HPP
#include <stdint.h>
#include <string>
#include <runtime/generic_val.hpp>
#include <util/def.hpp>

namespace sc {

struct thread_pool_table {
    // submits job in thread pool
    void (*parallel_call)(void (*pfunc)(void *, void *, int64_t, generic_val *),
            void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
            int64_t step, generic_val *args);
    // submits job in GC-managed thread pool
    void (*parallel_call_managed)(
            void (*pfunc)(void *, void *, int64_t, generic_val *),
            void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
            int64_t step, generic_val *args);
    // gets the max number of threads in pool
    int (*get_num_threads)();
    // sets the max number of threads in pool
    void (*set_num_threads)(int v);
    // get the current thread id in pool. Should be 0~N
    int (*get_thread_id)();
    // returns non-zero if is in parallel section
    int (*is_in_parallel)();
};

struct SC_INTERNAL_API runtime_config_t {
#ifndef SC_PRODUCTION
    // the total num threads for the thread pool
    // Should >= get_thread_num()
    int total_num_threads_;
#endif
    thread_pool_table *thread_pool_table_;
    // if in muti-instance simulation, the number of threads per instance.
    int get_num_threads() { return thread_pool_table_->get_num_threads(); }
    void set_num_threads(int num) { thread_pool_table_->set_num_threads(num); }
    std::string trace_out_path_;
    int trace_initial_cap_;
    bool execution_verbose_;
#ifndef SC_PRODUCTION
    bool managed_thread_pool_;
#endif
    int verbose_level_;
    static runtime_config_t &get();

private:
    runtime_config_t();
};

} // namespace sc
#endif
