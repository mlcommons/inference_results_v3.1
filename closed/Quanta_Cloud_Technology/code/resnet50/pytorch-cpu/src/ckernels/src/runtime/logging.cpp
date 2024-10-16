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
#include <iostream>
#include <set>
#include <string>
#include <runtime/config.hpp>
#include <runtime/env_var.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/logging.hpp>
#include <util/string_utils.hpp>
namespace sc {
namespace runtime {

// by default, prints all if filter is empty
static bool is_filter_exclude = true;

static std::set<std::string> init_name_list() {
    auto filter = utils::getenv_string(env_names[env_key::SC_LOGGING_FILTER]);
    auto splitted = utils::string_split(filter, ":");
    std::set<std::string> ret;
    if (!splitted.empty()) {
        if (splitted[0] == "+") {
            is_filter_exclude = false;
        } else if (splitted[0] == "-") {
            is_filter_exclude = true;
        } else {
            // unknown filter, return
            return ret;
        }
        for (unsigned i = 1; i < splitted.size(); i++) {
            ret.insert(splitted[i]);
        }
    }
    return ret;
}

static bool should_pass_filter(const char *module_name) {
    // lazy initialization here, will be initialized at the first time it is
    // used
    static std::set<std::string> filter = init_name_list();
    std::string pack = module_name;
    // find if module_name starts with any name in the filter
    // if a name in the filter is prefix of pack, it will be the first element
    // <= pack.
    // find in the filter: the first element > pack then check if
    // *(itr-1) is the prefix of pack
    auto itr = filter.upper_bound(pack);
    bool in_filter = (itr != filter.end() && itr != filter.begin()
            && utils::string_startswith(pack, *--itr));
    // if itr == filter.end(), the element to find may be the largest element,
    // double check
    if (itr == filter.end() && !filter.empty()) {
        in_filter = utils::string_startswith(pack, *filter.rbegin());
    }
    if (is_filter_exclude) {
        return !in_filter;
    } else {
        return in_filter;
    }
}

static std::ostream *stream_target = &std::cerr;

void set_logging_stream(std::ostream *s) {
    stream_target = s;
}

static logging_stream_t get_stream(verbose_level level, const char *module_name,
        const char *appender, const char *prefix) {
    if (runtime_config_t::get().verbose_level_ < level) {
        return logging_stream_t(nullptr, nullptr);
    }
    if (!module_name || should_pass_filter(module_name)) {
        *stream_target << prefix;
        if (module_name) { *stream_target << '[' << module_name << ']' << ' '; }
        return logging_stream_t(stream_target, appender);
    }
    return logging_stream_t(nullptr, nullptr);
}

logging_stream_t get_info_logging_stream(const char *module_name) {
    return get_stream(INFO, module_name, "\n", "[INFO] ");
}

logging_stream_t get_warning_logging_stream(const char *module_name) {
    return get_stream(WARNING, module_name, "\033[0m\n", "\033[33m[WARN] ");
}

logging_stream_t get_fatal_logging_stream(const char *module_name) {
    return get_stream(FATAL, module_name, "\033[0m\n", "\033[31m[FATAL] ");
}

} // namespace runtime
} // namespace sc
