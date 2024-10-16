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
#ifndef RUNTIME_KERNEL_INCLUDE_X86SIMD_VECTOR_MASKLOADSTORE_HPP
#define RUNTIME_KERNEL_INCLUDE_X86SIMD_VECTOR_MASKLOADSTORE_HPP
#include "common.hpp"

INLINE vec_f32x8 mask_load(const float *p, vec_s32x8 mask) {
    return _mm256_maskload_ps(p, mask.v);
}
INLINE vec_f32x4 mask_load(const float *p, vec_s32x4 mask) {
    return _mm_maskload_ps(p, mask.v);
}
INLINE void mask_store(float *p, vec_s32x8 mask, vec_f32x8 &a) {
    _mm256_maskstore_ps(p, mask.v, a.v);
}
INLINE void mask_store(float *p, vec_s32x4 mask, vec_f32x4 &a) {
    _mm_maskstore_ps(p, mask.v, a.v);
}
#endif
