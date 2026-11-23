// Row-wise neural-network kernels over a contiguous [rows, cols] tensor: softmax / log-softmax (forward and backward),
// layer norm and RMS norm (forward; the backward pass is composed from tensor operations on the saved statistics, which keeps
// higher-order gradients exact). One block per row; one module per float type (16-bit types compute in f32).
#include "common.cuh"

#if IS_FLOAT

// ---- block reductions over C ---------------------------------------------------------------------------------------------------
__device__ __forceinline__ C wsum(C v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ C wmax_nan(C v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = nan_max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// `sc` holds 32 values of C. Every thread receives the block total.
__device__ __forceinline__ C bsum(C v, C* sc) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) sc[wid] = v;
    __syncthreads();
    v = lane < ((blockDim.x + 31) >> 5) ? sc[lane] : (C)0;
    return wsum(v);
}
__device__ __forceinline__ C bmax_nan(C v, C* sc) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = wmax_nan(v);
    __syncthreads();
    if (lane == 0) sc[wid] = v;
    __syncthreads();
    v = lane < ((blockDim.x + 31) >> 5) ? sc[lane] : (C)(-__int_as_float(0x7f800000));
    return wmax_nan(v);
}

// ---- softmax ---------------------------------------------------------------------------------------------------------------------
// log == 0: y = softmax(x); log == 1: y = log_softmax(x). A NaN anywhere in a row gives a NaN row (the row maximum propagates
// it). A row whose maximum is +inf becomes the one-hot (uniform) distribution over its +inf entries, like the CPU backend.
// When `flag` is non-null, bit 0 is set for every row whose maximum is not finite: callers that need gradients take the
// composed (zero-gradient) path for those rows.
extern "C" __global__ void softmax_fwd(const T* x, T* y, u64 rows, u64 cols, int log, int* flag) {
    __shared__ C sc[32];
    const u64 row = blockIdx.x;
    if (row >= rows) return;
    const T* px = x + row * cols;
    T* py = y + row * cols;
    const C inf = (C)__int_as_float(0x7f800000);
    C m = -inf;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) m = nan_max(m, ld(px[i]));
    m = bmax_nan(m, sc);
    if (threadIdx.x == 0 && flag && (m - m != (C)0)) atomicOr(flag, 1);
    if (m == inf) {
        C cnt = (C)0;
        for (u64 i = threadIdx.x; i < cols; i += blockDim.x) cnt += ld(px[i]) == inf ? (C)1 : (C)0;
        cnt = bsum(cnt, sc);
        for (u64 i = threadIdx.x; i < cols; i += blockDim.x) {
            const bool hit = ld(px[i]) == inf;
            py[i] = st(log ? (hit ? -m_log(cnt) : -inf) : (hit ? (C)1 / cnt : (C)0));
        }
        return;
    }
    C s = (C)0;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) s += m_exp(ld(px[i]) - m);
    s = bsum(s, sc);
    if (log) {
        const C ls = m_log(s);
        for (u64 i = threadIdx.x; i < cols; i += blockDim.x) py[i] = st((ld(px[i]) - m) - ls);
    } else {
        const C inv = (C)1 / s;
        for (u64 i = threadIdx.x; i < cols; i += blockDim.x) py[i] = st(m_exp(ld(px[i]) - m) * inv);
    }
}

// softmax:      gx = y * (gy - sum(gy * y))
// log_softmax:  gx = gy - exp(y) * sum(gy)
extern "C" __global__ void softmax_bwd(const T* y, const T* gy, T* gx, u64 rows, u64 cols, int log) {
    __shared__ C sc[32];
    const u64 row = blockIdx.x;
    if (row >= rows) return;
    const T* py = y + row * cols;
    const T* pg = gy + row * cols;
    T* po = gx + row * cols;
    C s = (C)0;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) s += log ? ld(pg[i]) : ld(pg[i]) * ld(py[i]);
    s = bsum(s, sc);
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) {
        const C g = ld(pg[i]), yv = ld(py[i]);
        po[i] = st(log ? g - m_exp(yv) * s : yv * (g - s));
    }
}

// ---- layer norm / RMS norm ------------------------------------------------------------------------------------------------------
// y = (x - mean) * rstd * w + b over each row. `w` / `b` may be null. When `xhat` is non-null it receives the normalized values
// (before the affine transform) and `rstd` the per-row reciprocal standard deviation, both needed by the backward pass.
extern "C" __global__ void layernorm_fwd(const T* x, const T* w, const T* b, T* y, T* xhat, C* rstd_out, u64 rows, u64 cols, double eps) {
    __shared__ C sc[32];
    const u64 row = blockIdx.x;
    if (row >= rows) return;
    const T* px = x + row * cols;
    C s = (C)0;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) s += ld(px[i]);
    const C mean = bsum(s, sc) / (C)cols;
    C v = (C)0;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) {
        const C d = ld(px[i]) - mean;
        v += d * d;
    }
    const C var = bsum(v, sc) / (C)cols;
    const C rstd = (C)1 / m_sqrt(var + (C)eps);
    if (threadIdx.x == 0 && rstd_out) rstd_out[row] = rstd;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) {
        const C h = (ld(px[i]) - mean) * rstd;
        if (xhat) xhat[row * cols + i] = st(h);
        C o = h;
        if (w) o *= ld(w[i]);
        if (b) o += ld(b[i]);
        y[row * cols + i] = st(o);
    }
}

// y = x * rstd * w with rstd = 1 / sqrt(mean(x^2) + eps)
extern "C" __global__ void rmsnorm_fwd(const T* x, const T* w, T* y, C* rstd_out, u64 rows, u64 cols, double eps) {
    __shared__ C sc[32];
    const u64 row = blockIdx.x;
    if (row >= rows) return;
    const T* px = x + row * cols;
    C s = (C)0;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) {
        const C v = ld(px[i]);
        s += v * v;
    }
    const C rstd = (C)1 / m_sqrt(bsum(s, sc) / (C)cols + (C)eps);
    if (threadIdx.x == 0 && rstd_out) rstd_out[row] = rstd;
    for (u64 i = threadIdx.x; i < cols; i += blockDim.x) {
        C o = ld(px[i]) * rstd;
        if (w) o *= ld(w[i]);
        y[row * cols + i] = st(o);
    }
}

#endif  // IS_FLOAT
