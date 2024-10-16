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
#ifndef UTIL_MATH_UTILS_HPP
#define UTIL_MATH_UTILS_HPP
#include <algorithm>
#include <vector>
#include "parallel.hpp"
#include <runtime/config.hpp>
namespace sc {
namespace math_utils {

template <class T>
std::vector<T> vector_mul(
        const std::vector<T> &inputs1, const std::vector<T> &inputs2) {
    assert(inputs1.size() == inputs2.size() || inputs1.size() == 1UL
            || inputs2.size() == 1UL);
    size_t outsize = std::max(inputs1.size(), inputs2.size());
    std::vector<T> outputs(outsize);
    auto func = [&](uint64_t iter, uint64_t end) {
        T input1, input2;
        if (inputs1.size() == 1UL) {
            input1 = inputs1[0];
        } else {
            input1 = inputs1[iter];
        }
        if (inputs2.size() == 1UL) {
            input2 = inputs2[0];
        } else {
            input2 = inputs2[iter];
        }
        outputs[iter] = input1 * input2;
    };
    utils::parallel(func, 0, outsize);
    return outputs;
}

template <class T>
std::vector<T> vector_mul(const std::vector<T> &inputs1, const T &input2) {
    std::vector<T> outputs(inputs1.size());
    auto func = [&](uint64_t iter, uint64_t end) {
        outputs[iter] = inputs1[iter] * input2;
    };
    utils::parallel(func, 0, inputs1.size());
    return outputs;
}

template <class T>
T get_dims_product(const std::vector<T> &dims) {
    T ret = 1;
    for (unsigned i = 0; i < dims.size(); ++i) {
        ret *= dims[i];
    }
    assert(ret > 0 && "Overflow or non-constant shape detected");
    return ret;
}

template <typename T,
        typename dummy
        = typename std::enable_if<std::is_same<float, std::decay<T>>::value
                || std::is_same<double, std::decay<T>>::value>>
std::vector<T> vector_rcp(const std::vector<T> &inputs) {
    std::vector<T> outputs(inputs.size());
    auto func = [&](uint64_t iter, uint64_t end) {
        outputs[iter] = 1.0 / inputs[iter];
    };
    utils::parallel(func, 0, inputs.size());
    return outputs;
}
} // namespace math_utils
} // namespace sc
#endif
