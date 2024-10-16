#pragma once
#include <x86intrin.h>

template <int V> struct _mm_traits {
  typedef struct {float m_[V];} vector_type;
  typedef struct {int m_[V];} vector_itype;
};


#ifdef __AVX512F__
template <> struct _mm_traits<16> {
  typedef __m512 vector_type;
  typedef __m512i vector_itype;
};
#endif

#ifdef __AVX2__
template <> struct _mm_traits<8> {
  typedef __m256 vector_type;
  typedef __m256i vector_itype;
};
#endif

template <> struct _mm_traits<4> {
  typedef __m128 vector_type;
  typedef __m128i vector_itype;
};

template <int V> using __m = typename _mm_traits<V>::vector_type;
template <int V> using __i = typename _mm_traits<V>::vector_itype;

template <int V> struct _mm {
  static inline __m<V> load_ps(void const *adrs) noexcept {
    __m<V> v;
    auto a = reinterpret_cast<const float *>(adrs);
    for (int i = 0; i < V; i ++)
      v.m_[i] = a[i];

    return v;
  }
  static inline void store_ps(void const *adrs, __m<V> m) noexcept {
    auto a = reinterpret_cast<const float *>(adrs);
    for (int i = 0; i < V; i ++)
      a[i] = m.m_[i];
  }
  static inline __m<V> setzero_ps () noexcept {
    __m<V> v;
    for (int i = 0; i < V; i ++)
      v.m_[i] = 0;
    return v;
  }
};

#ifdef __AVX512F__
#if 1
template <> struct _mm<16> {
  static constexpr int V = 16;
  static inline __m<V> load_ps(void const *adrs) noexcept {
    return _mm512_load_ps(adrs);
  }
  static inline void store_ps(void *adrs, __m<V> m) noexcept {
    _mm512_store_ps(adrs, m);
  }
  static inline void stream_ps(float *adrs, __m<V> m) noexcept {
    _mm512_stream_ps(adrs, m);
  }
  static inline void i32scatter_ps(void *adrs, __i<V> vidx,
      __m<V> m, int scale) noexcept {
    switch (scale) {
      case 1:
        _mm512_i32scatter_ps(adrs, vidx, m, 1);
        break;
      case 2:
        _mm512_i32scatter_ps(adrs, vidx, m, 2);
        break;
      case 4:
        _mm512_i32scatter_ps(adrs, vidx, m, 4);
        break;
      case 8:
        _mm512_i32scatter_ps(adrs, vidx, m, 8);
        break;
    }
  }
  static inline __m<V> i32gather_ps(__i<V> vidx, void *adrs, int scale)
  noexcept {
    switch (scale) {
      case 1:
        return _mm512_i32gather_ps(vidx, adrs, 1);
      case 2:
        return _mm512_i32gather_ps(vidx, adrs, 2);
      case 4:
        return _mm512_i32gather_ps(vidx, adrs, 4);
      case 8:
        return _mm512_i32gather_ps(vidx, adrs, 8);
    }

    return _mm512_i32gather_ps(vidx, adrs, 1);
  }
  static inline __m<V> setzero_ps(void) noexcept {
    return _mm512_setzero_ps();
  }
  static inline __i<V> set_epi32(int e15, int e14, int e13, int e12,
      int e11, int e10, int e9, int e8,
      int e7, int e6, int e5, int e4,
      int e3, int e2, int e1, int e0) noexcept {
    return _mm512_set_epi32(e15, e14, e13, e12, e11, e10, e9, e8,
        e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __m<V> set1_ps(float e) noexcept {
    return _mm512_set1_ps(e);
  }
  static inline __m<V> set_ps(float e15, float e14, float e13, float e12,
      float e11, float e10, float e9, float e8, float e7, float e6, float e5,
      float e4, float e3, float e2, float e1, float e0) noexcept {
    return _mm512_set_ps(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4,
        e3, e2, e1, e0);
  }
  static inline __m<V> add_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_add_ps(op1, op2);
  }
  static inline __m<V> sub_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_sub_ps(op1, op2);
  }
  static inline __m<V> mul_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_mul_ps(op1, op2);
  }
  static inline __m<V> fmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fmsub_ps(op1, op2, op3);
  }
  static inline __m<V> fnmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fnmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fnmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fnmsub_ps(op1, op2, op3);
  }
  static inline __m<V> max_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_max_ps(op1, op2);
  }
  static inline __m<V> xor_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_xor_ps(op1, op2);
  }
  static inline __m<V> broadcastss_ps(__m128 b) noexcept {
    return _mm512_broadcastss_ps(b);
  }
};
#else
/* ICC Bug! */
template <> struct _mm<16> {
  static constexpr auto load_ps = _mm512_load_ps;
  static constexpr auto store_ps = _mm512_store_ps;
  static constexpr auto setzero_ps = _mm512_setzero_ps;
  static constexpr auto set1_ps = _mm512_set1_ps;
  static constexpr auto add_ps = _mm512_add_ps;
  static constexpr auto sub_ps = _mm512_sub_ps;
  static constexpr auto mul_ps = _mm512_mul_ps;
  static constexpr auto fmadd_ps = _mm512_fmadd_ps;
  static constexpr auto fmsub_ps = _mm512_fmsub_ps;
  static constexpr auto fnmadd_ps = _mm512_fnmadd_ps;
  static constexpr auto fnmsub_ps = _mm512_fnmsub_ps;
  static constexpr auto max_ps = _mm512_max_ps;
  static constexpr auto xor_ps = _mm512_xor_ps;
  static constexpr auto broadcastss_ps = _mm512_broadcastss_ps;
};
#endif
#endif

#ifdef __AVX2__
#if 1
template <> struct _mm<8> {
  static constexpr int V = 8;
  static inline __m<V> load_ps(float const *adrs) noexcept {
    return _mm256_load_ps(adrs);
  }
  static inline void store_ps(float *adrs, __m<V> m) noexcept {
    _mm256_store_ps(adrs, m);
  }
  static inline void stream_ps(float *adrs, __m<V> m) noexcept {
    _mm256_stream_ps(adrs, m);
  }
  static inline void i32scatter_ps(void *adrs, __i<V> vidx,
      __m<V> m, int scale) noexcept {
    switch(scale) {
      case 1:
        _mm256_i32scatter_ps(adrs, vidx, m, 1);
      case 2:
        _mm256_i32scatter_ps(adrs, vidx, m, 2);
      case 4:
        _mm256_i32scatter_ps(adrs, vidx, m, 4);
      case 8:
        _mm256_i32scatter_ps(adrs, vidx, m, 8);
    }
  }
  static inline __m<V> i32gather_ps(__i<V> vidx, const float *adrs, int scale)
  noexcept {
    switch (scale) {
      case 1:
        return _mm256_i32gather_ps(adrs, vidx, 1);
      case 2:
        return _mm256_i32gather_ps(adrs, vidx, 2);
      case 4:
        return _mm256_i32gather_ps(adrs, vidx, 4);
      case 8:
        return _mm256_i32gather_ps(adrs, vidx, 8);
    }

    return _mm256_i32gather_ps(adrs, vidx, 1);
  }
  static inline __m<V> setzero_ps(void) noexcept {
    return _mm256_setzero_ps();
  }
  static inline __m<V> set1_ps(float e) noexcept {
    return _mm256_set1_ps(e);
  }
  static inline __i<V> set_epi32(int, int, int, int, int, int, int, int,
      int e7, int e6, int e5, int e4,
      int e3, int e2, int e1, int e0) noexcept {
    return _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __i<V> set_epi32(int e7, int e6, int e5, int e4,
      int e3, int e2, int e1, int e0) noexcept {
    return _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __m<V> set_ps(float e7, float e6, float e5,
      float e4, float e3, float e2, float e1, float e0) noexcept {
    return _mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __m<V> add_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_add_ps(op1, op2);
  }
  static inline __m<V> sub_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_sub_ps(op1, op2);
  }
  static inline __m<V> mul_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_mul_ps(op1, op2);
  }
  static inline __m<V> fmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fmsub_ps(op1, op2, op3);
  }
  static inline __m<V> fnmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fnmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fnmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fnmsub_ps(op1, op2, op3);
  }
  static inline __m<V> max_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_max_ps(op1, op2);
  }
  static inline __m<V> xor_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_xor_ps(op1, op2);
  }
  static inline __m<V> broadcastss_ps(__m128 b) noexcept {
    return _mm256_broadcastss_ps(b);
  }
};
#else
/* ICC Bug! */
template <> struct _mm<8> {
  static constexpr auto load_ps = _mm256_load_ps;
  static constexpr auto store_ps = _mm256_store_ps;
  static constexpr auto setzero_ps = _mm256_setzero_ps;
  static constexpr auto set1_ps = _mm256_set1_ps;
  static constexpr auto add_ps = _mm256_add_ps;
  static constexpr auto sub_ps = _mm256_sub_ps;
  static constexpr auto mul_ps = _mm256_mul_ps;
  static constexpr auto fmadd_ps = _mm256_fmadd_ps;
  static constexpr auto fmsub_ps = _mm256_fmsub_ps;
  static constexpr auto fnmadd_ps = _mm256_fnmadd_ps;
  static constexpr auto fnmsub_ps = _mm256_fnmsub_ps;
  static constexpr auto max_ps = _mm256_max_ps;
  static constexpr auto xor_ps = _mm256_xor_ps;
  static constexpr auto broadcastss_ps = _mm256_broadcastss_ps;
};
#endif
#endif

template <typename scale_type, int lane_witdh> class simd_io;

template <int lane_witdh> class simd_io<float, lane_witdh> :
  public _mm<lane_witdh> {
public:
  static constexpr auto simd_l = lane_witdh;
  // Hack
  static constexpr auto reg_num = 2 * simd_l;
  typedef float elem_type;
  typedef __m<simd_l> vec_type;
};

#if defined(__AVX2__)
typedef simd_io<float, 8> avx2_io;
#endif

#if defined(__AVX512F__)
typedef simd_io<float, 16> avx3_io;
#endif
