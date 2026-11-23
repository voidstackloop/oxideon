// Elementwise kernels: unary, binary, comparison, select, logical not, fill. One module per element type (see common.cuh).
// Semantics follow the CPU backend exactly: NaN-propagating maximum / minimum, torch's remainder and floor-division,
// round half to even, relu that passes NaN through, wrapping integer arithmetic.
#include "common.cuh"

// ---- float math, shared by the f32 / f64 / f16 / bf16 modules (and by the integer modules, which compute in f64) --------
template <typename X>
__device__ __forceinline__ X unary_math(X x, int op, X p0, X p1) {
    switch (op) {
        case 0: return -x;                                                        // Neg
        case 1: return m_abs(x);                                                  // Abs
        case 2: return x > (X)0 ? (X)1 : (x < (X)0 ? (X)-1 : (X)0);              // Sign (NaN -> 0)
        case 3: return m_exp(x);                                                  // Exp
        case 4: return m_log(x);                                                  // Ln
        case 5: return m_sqrt(x);                                                 // Sqrt
        case 6: return (X)1 / m_sqrt(x);                                          // Rsqrt
        case 7: return m_sin(x);                                                  // Sin
        case 8: return m_cos(x);                                                  // Cos
        case 9: return m_tanh(x);                                                 // Tanh
        case 10: {                                                                // Sigmoid
            if (x >= (X)0) return (X)1 / ((X)1 + m_exp(-x));
            X e = m_exp(x);
            return e / ((X)1 + e);
        }
        case 11: return x < (X)0 ? (X)0 : x;                                      // Relu (NaN passes)
        case 12: return x > (X)0 ? x : p0 * x;                                    // LeakyRelu
        case 13: return (X)0.5 * x * m_erfc(-x * (X)0.70710678118654752440);     // Gelu (exact, via erfc)
        case 14: {                                                                // Silu
            X s = x >= (X)0 ? (X)1 / ((X)1 + m_exp(-x)) : m_exp(x) / ((X)1 + m_exp(x));
            return x * s;
        }
        case 15: return x > (X)0 ? x : p0 * (m_exp(x) - (X)1);                    // Elu
        case 16: return x > (X)20 ? x : m_log1p(m_exp(x));                        // Softplus
        case 17: return x * x;                                                    // Square
        case 18: return (X)1 / x;                                                 // Recip
        case 19: return m_floor(x);                                               // Floor
        case 20: return m_ceil(x);                                                // Ceil
        case 21: return m_rint(x);                                                // Round (half to even)
        case 22: return m_erf(x);                                                 // Erf
        case 23: return m_pow(x, p0);                                             // Powf
        case 24: return x + p0;                                                   // AddScalar
        case 25: return x * p0;                                                   // MulScalar
        case 26: return x < p0 ? p0 : (x > p1 ? p1 : x);                          // Clamp (NaN passes)
        default: return x;
    }
}

// Rust's `f64 as i64`: NaN -> 0, out-of-range values saturate, everything else truncates toward zero. The CPU backend
// evaluates integer unary operations in f64 and converts back this way (then wraps to the element width).
__device__ __forceinline__ i64 d2i64(double v) {
    if (v != v) return 0;
    if (v >= 9223372036854775807.0) return (i64)0x7fffffffffffffffll;
    if (v <= -9223372036854775808.0) return (i64)0x8000000000000000ll;
    return (i64)v;
}

#if IS_FLOAT
extern "C" __global__ void unary(const T* x, T* y, u64 n, int op, double p0, double p1, Strided s) {
    GRID_STRIDE_LOOP(i, n) { y[i] = st(unary_math<C>(ld(x[offset_of<0>(s, i)]), op, (C)p0, (C)p1)); }
}
#elif DTYPE != DT_BOOL
// integer tensors: the CPU backend evaluates in f64 and converts back (truncation, then wrap to the element width)
extern "C" __global__ void unary(const T* x, T* y, u64 n, int op, double p0, double p1, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        const double v = (double)ld(x[offset_of<0>(s, i)]);
        y[i] = (T)d2i64(unary_math<double>(v, op, p0, p1));
    }
}
#endif

extern "C" __global__ void logical_not(const T* x, u8* y, u64 n, Strided s) {
    GRID_STRIDE_LOOP(i, n) { y[i] = (u8)(ld(x[offset_of<0>(s, i)]) == (C)0); }
}

// ---- binary -------------------------------------------------------------------------------------------------------------
#if IS_FLOAT
template <typename X>
__device__ __forceinline__ X remainder_k(X x, X y) {
    X m = m_fmod(x, y);
    return (m != (X)0 && ((m < (X)0) != (y < (X)0))) ? m + y : m;
}
template <typename X>
__device__ __forceinline__ X floor_div_k(X x, X y) {
    if (y == (X)0) return x / y;
    X m = m_fmod(x, y);
    X div = (x - m) / y;
    if (m != (X)0 && ((y < (X)0) != (m < (X)0))) div -= (X)1;
    if (div != (X)0) {
        X fl = m_floor(div);
        return (div - fl > (X)0.5) ? fl + (X)1 : fl;
    }
    return m_copysign((X)0, x / y);
}
extern "C" __global__ void binary(const T* a, const T* b, T* out, u64 n, int op, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        const C x = ld(a[offset_of<0>(s, i)]), y = ld(b[offset_of<1>(s, i)]);
        C r;
        switch (op) {
            case 0: r = x + y; break;
            case 1: r = x - y; break;
            case 2: r = x * y; break;
            case 3: r = x / y; break;
            case 4: r = nan_max(x, y); break;
            case 5: r = nan_min(x, y); break;
            case 6: r = m_pow(x, y); break;
            case 7: r = m_fmod(x, y); break;
            case 8: r = remainder_k(x, y); break;
            default: r = floor_div_k(x, y); break;
        }
        out[i] = st(r);
    }
}
#elif DTYPE != DT_BOOL
// integer tensors: add / sub / mul wrap, maximum / minimum compare. Division, remainder and pow stay on the CPU path (they
// must report division by zero and overflow as errors, which a kernel cannot).
extern "C" __global__ void binary(const T* a, const T* b, T* out, u64 n, int op, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        const C x = ld(a[offset_of<0>(s, i)]), y = ld(b[offset_of<1>(s, i)]);
        u64 ux = (u64)x, uy = (u64)y, r;
        switch (op) {
            case 0: r = ux + uy; break;
            case 1: r = ux - uy; break;
            case 2: r = ux * uy; break;
            case 4: r = (u64)(x >= y ? x : y); break;
            default: r = (u64)(x <= y ? x : y); break;
        }
        out[i] = (T)r;
    }
}
#endif

// ---- comparison -> bool ---------------------------------------------------------------------------------------------------
extern "C" __global__ void cmp(const T* a, const T* b, u8* out, u64 n, int op, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        const C x = ld(a[offset_of<0>(s, i)]), y = ld(b[offset_of<1>(s, i)]);
        bool r;
        switch (op) {
            case 0: r = x == y; break;
            case 1: r = x != y; break;
            case 2: r = x < y; break;
            case 3: r = x <= y; break;
            case 4: r = x > y; break;
            default: r = x >= y; break;
        }
        out[i] = (u8)r;
    }
}

// logical and / or / xor with `!= 0` truthiness on any element type -> bool
extern "C" __global__ void logic(const T* a, const T* b, u8* out, u64 n, int op, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        const bool x = ld(a[offset_of<0>(s, i)]) != (C)0, y = ld(b[offset_of<1>(s, i)]) != (C)0;
        out[i] = (u8)(op == 0 ? (x && y) : op == 1 ? (x || y) : (x != y));
    }
}

// ---- select ------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void where_sel(const u8* cond, const T* a, const T* b, T* out, u64 n, Strided s) {
    GRID_STRIDE_LOOP(i, n) { out[i] = cond[offset_of<0>(s, i)] ? a[offset_of<1>(s, i)] : b[offset_of<2>(s, i)]; }
}

// ---- fill ----------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void fill(T* y, u64 n, double fv, i64 iv) {
    GRID_STRIDE_LOOP(i, n) {
#if IS_FLOAT
        y[i] = st((C)fv);
#else
        y[i] = (T)iv;
#endif
    }
}
