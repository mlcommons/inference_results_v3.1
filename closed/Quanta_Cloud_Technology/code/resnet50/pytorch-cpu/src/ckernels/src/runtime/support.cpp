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

#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "config.hpp"
#include <runtime/data_type.hpp>
#include <runtime/env_var.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/logging.hpp>
#ifndef SC_PRODUCTION
#include <runtime/managed_thread_pool.hpp>
#endif
#include <runtime/os.hpp>
#include <runtime/parallel.hpp>
#include <runtime/runtime.hpp>
#include <util/os.hpp>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

SC_MODULE(runtime.support)

using namespace sc;
extern "C" void print_float(float f) {
    printf("%f\n", f);
}

extern "C" void print_index(uint64_t f) {
    printf("%llu\n", static_cast<unsigned long long>(f)); // NOLINT
}

extern "C" void print_int(int f) {
    printf("%d\n", f);
}

extern "C" void print_str(char *f) {
    fputs(f, stdout);
}

extern "C" uint64_t boundary_check(
        const char *name, uint64_t idx, uint64_t acc_len, uint64_t tsr_len) {
    if (idx >= tsr_len || idx + acc_len > tsr_len) {
        fprintf(stderr,
                "Boundary check for tensor %s failed. idx=%llu acc_len=%llu "
                "tsr_len=%llu\n",
                name, static_cast<unsigned long long>(idx), // NOLINT
                static_cast<unsigned long long>(acc_len), // NOLINT
                static_cast<unsigned long long>(tsr_len)); // NOLINT
        abort();
    }
    return idx;
}

extern "C" void *sc_global_aligned_alloc(size_t sz, size_t align) {
    return aligned_alloc(align, (sz / align + 1) * align);
}

extern "C" void sc_global_aligned_free(void *ptr, size_t align) {
    aligned_free(ptr);
}

namespace sc {

namespace runtime {
size_t get_os_page_size() {
#ifdef _WIN32
    // fix-me: (win32) impl
    return 4096;
#else
    static size_t v = getpagesize();
    return v;
#endif
}
} // namespace runtime

runtime_config_t &runtime_config_t::get() {
    static runtime_config_t cfg {};
    return cfg;
}

using namespace env_key;
runtime_config_t::runtime_config_t() {
    thread_pool_table_ = &sc_pool_table;
#ifndef SC_PRODUCTION
    managed_thread_pool_
            = (utils::getenv_int(env_names[SC_MANAGED_THREAD_POOL], 1) != 0);
    if (managed_thread_pool_) {
        thread_pool_table_->parallel_call_managed = &sc_parallel_call_managed;
    }
    int ompmaxthreads = get_num_threads();
    int threads_per_instance = utils::getenv_int(env_names[SC_RUN_THREADS], 0);
    if (threads_per_instance > 0) {
        set_num_threads(threads_per_instance);
        ompmaxthreads = threads_per_instance;
    } else {
        if (threads_per_instance != 0) {
            // don't warn if SC_RUN_THREADS is not set
            SC_WARN << "thread_pool_num_threads_per_instance < 0";
        }
        threads_per_instance = ompmaxthreads;
    }
    total_num_threads_
            = utils::getenv_int(env_names[SC_TUNE_THREADS], ompmaxthreads);
    if (total_num_threads_ <= 0) {
        SC_WARN << "thread_pool_total_num_threads <= 0";
        total_num_threads_ = ompmaxthreads;
    }
    if (total_num_threads_ < threads_per_instance) {
        SC_WARN << "thread_pool_total_num_threads < "
                   "thread_pool_num_threads_per_instance";
        threads_per_instance = total_num_threads_;
    }
    trace_initial_cap_ = sc::utils::getenv_int(env_names[SC_TRACE_INIT_CAP],
            total_num_threads_ == threads_per_instance
                    ? 2048 /*single-instance*/
                    : 2048 * 1024 /*multi-instance*/);
#else
    trace_initial_cap_ = 2048 * 1024;
#endif
    trace_out_path_ = utils::getenv_string(env_names[SC_TRACE]);
    execution_verbose_
            = (utils::getenv_int(env_names[SC_EXECUTION_VERBOSE], 0) == 1);

#ifdef SC_PRODUCTION
    constexpr int default_verbose = 0;
#else
    constexpr int default_verbose = 1;
#endif
    int tmp_get_verbose_level
            = utils::getenv_int(env_names[SC_VERBOSE], default_verbose);
    if (tmp_get_verbose_level < 0 || tmp_get_verbose_level > 2) {
        tmp_get_verbose_level = 0;
    }
    verbose_level_ = tmp_get_verbose_level;
}
} // namespace sc

extern "C" void sc_value_check(void *tsr, const char *name, size_t size) {
    // temporarily assume dtype is float32
    float *buf = reinterpret_cast<float *>(tsr);
    for (size_t i = 0; i < size / sizeof(float); i++) {
        float val = static_cast<float>(buf[i]);
        if (std::isnan(val) || std::isinf(val)) {
            SC_MODULE_WARN << "Invalid value (nan or inf) found in tensor "
                           << name << " idx=" << i;
        }
    }
}
