// Shared definitions for the oxideon CUDA kernels. Compiled with NVRTC, so it includes no toolkit headers: every type,
// conversion and math wrapper the kernels need is defined here.
//
// A module is built for one element type, selected by the define DTYPE (see the table below); the kernels are written
// against the aliases `T` (storage type) and `C` (compute type), so one source serves all twelve dtypes.
#pragma once

typedef signed char i8;
typedef short i16;
typedef int i32;
typedef long long i64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define DT_F32 0
#define DT_F64 1
#define DT_F16 2
#define DT_BF16 3
#define DT_I8 4
#define DT_I16 5
#define DT_I32 6
#define DT_I64 7
#define DT_U8 8
#define DT_U16 9
#define DT_U32 10
#define DT_U64 11
#define DT_BOOL 12

// ---- 16-bit floats are stored as raw bits and computed in f32 ------------------------------------------------------
__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ u16 f2h(float f) {
    u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
__device__ __forceinline__ float b2f(u16 b) { return __uint_as_float(((u32)b) << 16); }
__device__ __forceinline__ u16 f2b(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)((u >> 16) | 0x0040u);  // keep a quiet NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                        // round to nearest even
    return (u16)(u >> 16);
}

#if DTYPE == DT_F32
typedef float T;
typedef float C;
#define IS_FLOAT 1
__device__ __forceinline__ C ld(T x) { return x; }
__device__ __forceinline__ T st(C x) { return x; }
#elif DTYPE == DT_F64
typedef double T;
typedef double C;
#define IS_FLOAT 1
__device__ __forceinline__ C ld(T x) { return x; }
__device__ __forceinline__ T st(C x) { return x; }
#elif DTYPE == DT_F16
typedef u16 T;
typedef float C;
#define IS_FLOAT 1
__device__ __forceinline__ C ld(T x) { return h2f(x); }
__device__ __forceinline__ T st(C x) { return f2h(x); }
#elif DTYPE == DT_BF16
typedef u16 T;
typedef float C;
#define IS_FLOAT 1
__device__ __forceinline__ C ld(T x) { return b2f(x); }
__device__ __forceinline__ T st(C x) { return f2b(x); }
#elif DTYPE == DT_I8 || DTYPE == DT_I16 || DTYPE == DT_I32 || DTYPE == DT_I64
#if DTYPE == DT_I8
typedef i8 T;
#elif DTYPE == DT_I16
typedef i16 T;
#elif DTYPE == DT_I32
typedef i32 T;
#else
typedef i64 T;
#endif
typedef i64 C;
#define IS_FLOAT 0
#define IS_SIGNED 1
__device__ __forceinline__ C ld(T x) { return (C)x; }
__device__ __forceinline__ T st(C x) { return (T)x; }
#elif DTYPE == DT_U8 || DTYPE == DT_U16 || DTYPE == DT_U32 || DTYPE == DT_U64 || DTYPE == DT_BOOL
#if DTYPE == DT_U8 || DTYPE == DT_BOOL
typedef u8 T;
#elif DTYPE == DT_U16
typedef u16 T;
#elif DTYPE == DT_U32
typedef u32 T;
#else
typedef u64 T;
#endif
typedef u64 C;
#define IS_FLOAT 0
#define IS_SIGNED 0
__device__ __forceinline__ C ld(T x) { return (C)x; }
__device__ __forceinline__ T st(C x) { return (T)x; }
#endif

// ---- math overloads (float / double) ---------------------------------------------------------------------------------
#define MATH1(NAME, FF, FD)                                           \
    __device__ __forceinline__ float NAME(float x) { return FF(x); } \
    __device__ __forceinline__ double NAME(double x) { return FD(x); }
MATH1(m_exp, expf, exp)
MATH1(m_log, logf, log)
MATH1(m_sqrt, sqrtf, sqrt)
MATH1(m_sin, sinf, sin)
MATH1(m_cos, cosf, cos)
MATH1(m_tanh, tanhf, tanh)
MATH1(m_erf, erff, erf)
MATH1(m_erfc, erfcf, erfc)
MATH1(m_floor, floorf, floor)
MATH1(m_ceil, ceilf, ceil)
MATH1(m_rint, rintf, rint)  // round half to even, like torch.round
MATH1(m_log1p, log1pf, log1p)
MATH1(m_abs, fabsf, fabs)
__device__ __forceinline__ float m_pow(float x, float y) { return powf(x, y); }
__device__ __forceinline__ double m_pow(double x, double y) { return pow(x, y); }
__device__ __forceinline__ float m_fmod(float x, float y) { return fmodf(x, y); }
__device__ __forceinline__ double m_fmod(double x, double y) { return fmod(x, y); }
__device__ __forceinline__ float m_copysign(float x, float y) { return copysignf(x, y); }
__device__ __forceinline__ double m_copysign(double x, double y) { return copysign(x, y); }
__device__ __forceinline__ bool m_isnan(float x) { return x != x; }
__device__ __forceinline__ bool m_isnan(double x) { return x != x; }

// ---- strided element addressing ----------------------------------------------------------------------------------------
// A logical shape walked in row-major order, with up to three operands, each with its own element strides (0 = broadcast).
// `nd == 0` means "every operand is contiguous with this element count": index i is used as is.
#define MAXD 8
struct Strided {
    int nd;
    i64 shape[MAXD];
    i64 stride[3][MAXD];
};

template <int K>
__device__ __forceinline__ i64 offset_of(const Strided& s, u64 i) {
    if (s.nd == 0) return (i64)i;
    i64 off = 0;
    u64 rem = i;
#pragma unroll
    for (int d = MAXD - 1; d >= 0; --d) {
        if (d < s.nd) {
            const u64 sh = (u64)s.shape[d];
            const u64 c = rem % sh;
            rem /= sh;
            off += (i64)c * s.stride[K][d];
        }
    }
    return off;
}

#define GRID_STRIDE_LOOP(i, n) for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (u64)gridDim.x * blockDim.x)

// ---- NaN-propagating extrema, like torch.maximum / minimum ------------------------------------------------------------
template <typename X>
__device__ __forceinline__ X nan_max(X a, X b) {
    return a >= b ? a : (b > a ? b : a + b);  // a NaN falls through to `+`, which stays NaN
}
template <typename X>
__device__ __forceinline__ X nan_min(X a, X b) {
    return a <= b ? a : (b < a ? b : a + b);
}

// ---- warp / block reductions --------------------------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Block-wide sum; `scratch` holds 32 floats. Every thread receives the total.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = lane < nw ? scratch[lane] : 0.0f;
    v = warp_sum(v);
    return v;
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) scratch[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    v = lane < nw ? scratch[lane] : -__int_as_float(0x7f800000);
    v = warp_max(v);
    return v;
}
