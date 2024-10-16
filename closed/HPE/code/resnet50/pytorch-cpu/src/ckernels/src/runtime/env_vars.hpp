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

#ifndef RUNTIME_ENV_VARS_HPP
#define RUNTIME_ENV_VARS_HPP

namespace sc {

namespace env_key {
enum key {
    SC_CPU_JIT,
    SC_TRACE,
    SC_DUMP_GRAPH,
    SC_GRAPH_DUMP_TENSORS,
    SC_VALUE_CHECK,
    SC_OPT_LEVEL,
    SC_BUFFER_SCHEDULE,
    SC_KERNEL,
    SC_MICRO_KERNEL_OPTIM,
    SC_DEAD_WRITE_ELIMINATION,
    SC_INDEX2VAR,
    SC_PRINT_IR,
    SC_BOUNDARY_CHECK,
    SC_PRINT_GENCODE,
    SC_KEEP_GENCODE,
    SC_JIT_CC_OPTIONS_GROUP,
    SC_CPU_JIT_FLAGS,
    SC_TEMP_DIR,
    SC_VERBOSE,
    SC_RUN_THREADS,
    SC_TRACE_INIT_CAP,
    SC_EXECUTION_VERBOSE,
    SC_LOGGING_FILTER,
    SC_HOME_,
    SC_SSA_PASSES,
    SC_PRINT_PASS_TIME,
    SC_PRINT_PASS_RESULT,
    SC_JIT_PROFILE,
    SC_DUMP_GRAPH_JSON,
    SC_TUNING_IMPORT,
#ifndef SC_PRODUCTION
    SC_MANAGED_THREAD_POOL,
    SC_TUNING_EXPORT,
    SC_TUNING_TIMEOUT,
    SC_TUNING_LOG,
    SC_GPU_SIM,
    SC_TUNE_THREADS,
    SC_XBYAK_JIT_SAVE_OBJ,
    SC_XBYAK_JIT_ASM_LISTING,
    SC_XBYAK_JIT_LOG_STACK_FRAME_MODEL,
    SC_XBYAK_JIT_PAUSE_AFTER_CODEGEN,
#endif
    NUM_KEYS
};
} // namespace env_key

extern const char *env_names[env_key::NUM_KEYS];
} // namespace sc
#endif
