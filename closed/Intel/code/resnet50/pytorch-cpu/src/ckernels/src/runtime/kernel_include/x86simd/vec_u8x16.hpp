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
#ifndef RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U8X16_HPP
#define RUNTIME_KERNEL_INCLUDE_X86SIMD_VEC_U8X16_HPP
#include <immintrin.h>
#include <stdint.h>
#include "common.hpp"
class vec_f32x16;
class vec_s32x16;
class vec_s8x16;
class vec_u8x16 {
public:
    union {
        __m128i v;
        uint8_t raw[16];
    } __attribute__((aligned(16)));

    INLINE vec_u8x16() = default;
    INLINE vec_u8x16(uint8_t f) { v = _mm_set1_epi8(f); }
    INLINE vec_u8x16(uint8_t i0, uint8_t i1, uint8_t i2, uint8_t i3, uint8_t i4,
            uint8_t i5, uint8_t i6, uint8_t i7, uint8_t i8, uint8_t i9,
            uint8_t i10, uint8_t i11, uint8_t i12, uint8_t i13, uint8_t i14,
            uint8_t i15) {
        v = _mm_setr_epi8(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12,
                i13, i14, i15);
    }
    INLINE vec_u8x16(__m128i const &x) { v = x; }
    INLINE operator vec_f32x16() const;
    INLINE operator vec_s32x16() const;
    INLINE operator vec_s8x16() const;
    static INLINE vec_u8x16 load(const uint8_t *p) {
        return _mm_loadu_si128((const __m128i *)p);
    }
    static INLINE vec_u8x16 load_aligned(const int8_t *p) {
        return _mm_load_si128((const __m128i *)p);
    }
    static INLINE void store(vec_u8x16 v, uint8_t *p) {
        _mm_storeu_si128((__m128i *)p, v.v);
    }
    static INLINE void store_aligned(vec_u8x16 v, int8_t *p) {
        _mm_store_si128((__m128i *)p, v.v);
    }
};

INLINE vec_u8x16 operator+(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_adds_epu8(a.v, b.v);
}

INLINE vec_u8x16 operator-(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_subs_epu8(a.v, b.v);
}

INLINE vec_u8x16 operator-(vec_u8x16 const &a) {
    return _mm_subs_epu8(_mm_setzero_si128(), a.v);
}

// INLINE vec_u8x16 operator*(vec_u8x16 const &a, vec_u8x16 const &b) {
//     return _mm_mullo_epu8(a.v, b.v);
// }

// INLINE vec_u9x16 operator/(vec_u8x16 const &a, vec_u8x16 const &b) {
//     return _mm_div_epu8(a.v, b.v);
// }

INLINE vec_u8x16 operator~(vec_u8x16 const &a) {
    return _mm_xor_si128(a.v, _mm_set1_epi8(-1));
}
INLINE vec_u8x16 operator&(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_and_si128(a.v, b.v);
}
INLINE vec_u8x16 operator|(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_or_si128(a.v, b.v);
}
INLINE vec_u8x16 operator^(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_xor_si128(a.v, b.v);
}

#ifdef __AVX512F__
INLINE __mmask16 operator!(vec_u8x16 const &a) {
    return _mm_cmp_epu8_mask(a.v, _mm_setzero_si128(), _MM_CMPINT_EQ);
}
INLINE __mmask16 operator==(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
INLINE __mmask16 operator!=(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
INLINE __mmask16 operator>(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_GT);
}
INLINE __mmask16 operator<(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
INLINE __mmask16 operator>=(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_GE);
}
INLINE __mmask16 operator<=(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
INLINE vec_u8x16 sc_select(
        __mmask16 mask, vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_mask_blend_epi8(mask, b.v, a.v);
}
#endif

INLINE vec_u8x16 sc_max(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_max_epu8(a.v, b.v);
}
INLINE vec_u8x16 sc_min(vec_u8x16 const &a, vec_u8x16 const &b) {
    return _mm_min_epu8(a.v, b.v);
}
#endif
