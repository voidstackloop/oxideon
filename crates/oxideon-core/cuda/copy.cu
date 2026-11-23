// Data movement kernels. `copy` is dtype-agnostic (one module per element size, built with DTYPE = the unsigned type of that
// size); `cast` converts between any two dtypes with runtime codes; the index kernels (index_select, gather, scatter,
// index_add, triangular) are built per element type.
#include "common.cuh"

// Strided -> strided copy. operand 0 = source strides, operand 1 = destination strides over the same logical shape, so one
// kernel serves to_contiguous (contiguous destination), writing into a slice of a larger tensor (cat, pad, view assignment)
// and plain transposes / broadcasts / flips (negative strides are fine: offsets are signed).
extern "C" __global__ void copy(const T* src, T* dst, u64 n, Strided s) {
    GRID_STRIDE_LOOP(i, n) { dst[offset_of<1>(s, i)] = src[offset_of<0>(s, i)]; }
}

// ---- cast -----------------------------------------------------------------------------------------------------------------------
// Element loaders / stores by dtype code (the DT_* values of common.cuh); the source operand uses stride set 0.
struct Wide {
    int kind;  // 0 float, 1 signed, 2 unsigned
    double f;
    i64 s;
    u64 u;
};

__device__ __forceinline__ Wide load_wide(const unsigned char* base, i64 idx, int code) {
    Wide w;
    w.f = 0; w.s = 0; w.u = 0;
    switch (code) {
        case 0: w.kind = 0; w.f = ((const float*)base)[idx]; break;
        case 1: w.kind = 0; w.f = ((const double*)base)[idx]; break;
        case 2: w.kind = 0; w.f = (double)h2f(((const u16*)base)[idx]); break;
        case 3: w.kind = 0; w.f = (double)b2f(((const u16*)base)[idx]); break;
        case 4: w.kind = 1; w.s = ((const i8*)base)[idx]; break;
        case 5: w.kind = 1; w.s = ((const i16*)base)[idx]; break;
        case 6: w.kind = 1; w.s = ((const i32*)base)[idx]; break;
        case 7: w.kind = 1; w.s = ((const i64*)base)[idx]; break;
        case 8: w.kind = 2; w.u = ((const u8*)base)[idx]; break;
        case 9: w.kind = 2; w.u = ((const u16*)base)[idx]; break;
        case 10: w.kind = 2; w.u = ((const u32*)base)[idx]; break;
        case 11: w.kind = 2; w.u = ((const u64*)base)[idx]; break;
        default: w.kind = 2; w.u = ((const u8*)base)[idx] != 0; break;  // bool
    }
    return w;
}

__device__ __forceinline__ i64 d2i64_torch(double v) {
    return (v == v && v > -9223372036854775808.0 && v < 9223372036854775808.0) ? (i64)v : (i64)0x8000000000000000ll;
}
__device__ __forceinline__ double wide_f(const Wide& w) { return w.kind == 0 ? w.f : (w.kind == 1 ? (double)w.s : (double)w.u); }
// integer view of a value: floats go through torch's conversion; unsigned 64-bit floats above 2^63 keep their magnitude
__device__ __forceinline__ u64 wide_bits(const Wide& w, bool dst_u64) {
    if (w.kind == 1) return (u64)w.s;
    if (w.kind == 2) return w.u;
    if (dst_u64 && w.f >= 9223372036854775808.0) return w.f >= 18446744073709551616.0 ? 0xffffffffffffffffull : (u64)w.f;  // Rust's saturating `as u64`
    return (u64)d2i64_torch(w.f);
}

__device__ __forceinline__ void store_wide(unsigned char* base, i64 idx, int code, const Wide& w) {
    switch (code) {
        case 0: ((float*)base)[idx] = (float)wide_f(w); break;
        case 1: ((double*)base)[idx] = wide_f(w); break;
        case 2: ((u16*)base)[idx] = f2h((float)wide_f(w)); break;
        case 3: ((u16*)base)[idx] = f2b((float)wide_f(w)); break;
        case 4: ((i8*)base)[idx] = (i8)wide_bits(w, false); break;
        case 5: ((i16*)base)[idx] = (i16)wide_bits(w, false); break;
        case 6: {  // i32 conversion: invalid or out-of-range values become INT_MIN rather than wrapping (as on the CPU)
            const double v = wide_f(w);
            ((i32*)base)[idx] = (v == v && v > -2147483648.0 && v < 2147483647.0) ? (i32)v : (i32)0x80000000;
            break;
        }
        case 7: ((i64*)base)[idx] = (i64)wide_bits(w, false); break;
        case 8: ((u8*)base)[idx] = (u8)wide_bits(w, false); break;
        case 9: ((u16*)base)[idx] = (u16)wide_bits(w, false); break;
        case 10: ((u32*)base)[idx] = (u32)wide_bits(w, false); break;
        case 11: ((u64*)base)[idx] = wide_bits(w, true); break;
        default: {  // bool: any non-zero value (a float NaN counts as non-zero, like the CPU path)
            bool nz = w.kind == 0 ? (w.f != 0.0) : (w.kind == 1 ? w.s != 0 : w.u != 0);
            ((u8*)base)[idx] = (u8)nz;
        }
    }
}

extern "C" __global__ void cast(const unsigned char* src, unsigned char* dst, u64 n, int src_code, int dst_code, Strided s) {
    GRID_STRIDE_LOOP(i, n) { store_wide(dst, (i64)i, dst_code, load_wide(src, offset_of<0>(s, i), src_code)); }
}

// ---- index validation ----------------------------------------------------------------------------------------------------------
// Index tensors may be produced on the device, so the kernels validate them themselves. `err` (16 bytes, zeroed by the host,
// may be null when the host already validated) receives a flag in its first int and the offending raw index in its second
// 8 bytes; invalid entries are skipped, never dereferenced. Negative indices wrap, like the CPU backend.
__device__ __forceinline__ bool norm_index(i64 raw, i64 dim, int* err, i64& out) {
    const i64 v = raw < 0 ? raw + dim : raw;
    if (v < 0 || v >= dim) {
        if (err) {
            atomicOr(err, 1);
            *(volatile i64*)(err + 2) = raw;
        }
        return false;
    }
    out = v;
    return true;
}

// ---- index_select / embedding ---------------------------------------------------------------------------------------------------
// out[o, k, j] = src[o, idx[k], j] for a contiguous source viewed as [outer, dim, inner].
extern "C" __global__ void index_select(const T* src, const i64* idx, T* out, u64 outer, u64 dim, u64 inner, u64 nidx, int* err) {
    const u64 n = outer * nidx * inner;
    GRID_STRIDE_LOOP(i, n) {
        const u64 j = i % inner;
        const u64 k = (i / inner) % nidx;
        const u64 o = i / (inner * nidx);
        i64 v;
        if (!norm_index(idx[k], (i64)dim, err, v)) continue;
        out[i] = src[(o * dim + (u64)v) * inner + j];
    }
}

// ---- gather / scatter along one axis, general layouts ----------------------------------------------------------------------------
// Walks the logical shape of the index tensor. operand 0 strides address the data tensor (the entry for `ax` is ignored:
// the index value supplies that coordinate, scaled by `ax_stride`), operand 1 strides address the index tensor, operand 2
// strides address the value tensor of scatter (the source) or the contiguous output of gather.
__device__ __forceinline__ bool gather_offsets(const Strided& s, u64 i, int ax, i64 ax_stride, i64 dim_ax, const i64* idx, int* err, i64& data_off, i64& val_off) {
    data_off = 0;
    i64 idx_off = 0;
    val_off = 0;
    u64 rem = i;
#pragma unroll
    for (int d = MAXD - 1; d >= 0; --d) {
        if (d < s.nd) {
            const u64 sh = (u64)s.shape[d];
            const i64 c = (i64)(rem % sh);
            rem /= sh;
            if (d != ax) data_off += c * s.stride[0][d];
            idx_off += c * s.stride[1][d];
            val_off += c * s.stride[2][d];
        }
    }
    i64 v;
    if (!norm_index(idx[idx_off], dim_ax, err, v)) return false;
    data_off += v * ax_stride;
    return true;
}

extern "C" __global__ void gather(const T* data, const i64* idx, T* out, u64 n, int ax, i64 ax_stride, i64 dim_ax, int* err, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        i64 d, v;
        if (gather_offsets(s, i, ax, ax_stride, dim_ax, idx, err, d, v)) out[v] = data[d];
    }
}

// scatter: data[.. idx ..] = src[..] (the last write wins when indices repeat; which one is unspecified, like torch on CUDA)
extern "C" __global__ void scatter(T* data, const i64* idx, const T* src, u64 n, int ax, i64 ax_stride, i64 dim_ax, int* err, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        i64 d, v;
        if (gather_offsets(s, i, ax, ax_stride, dim_ax, idx, err, d, v)) data[d] = src[v];
    }
}

#if DTYPE == DT_F32 || DTYPE == DT_F64 || DTYPE == DT_I32 || DTYPE == DT_I64 || DTYPE == DT_U32 || DTYPE == DT_U64
#define HAS_ATOMIC_ADD 1
__device__ __forceinline__ void atomic_add_t(T* p, T v) {
#if DTYPE == DT_F32 || DTYPE == DT_F64
    atomicAdd(p, v);
#elif DTYPE == DT_I32
    atomicAdd((int*)p, (int)v);
#elif DTYPE == DT_U32
    atomicAdd((unsigned int*)p, (unsigned int)v);
#else
    atomicAdd((unsigned long long*)p, (unsigned long long)v);  // 64-bit integers wrap, so the unsigned add is exact
#endif
}

extern "C" __global__ void scatter_add(T* data, const i64* idx, const T* src, u64 n, int ax, i64 ax_stride, i64 dim_ax, int* err, Strided s) {
    GRID_STRIDE_LOOP(i, n) {
        i64 d, v;
        if (gather_offsets(s, i, ax, ax_stride, dim_ax, idx, err, d, v)) atomic_add_t(data + d, src[v]);
    }
}

// dst[o, idx[k], j] += src[o, k, j] for contiguous tensors viewed as [outer, dim, inner] / [outer, nidx, inner]
// (embedding backward, index_add).
extern "C" __global__ void index_add(T* dst, const i64* idx, const T* src, u64 outer, u64 dim, u64 inner, u64 nidx, int* err) {
    const u64 n = outer * nidx * inner;
    GRID_STRIDE_LOOP(i, n) {
        const u64 j = i % inner;
        const u64 k = (i / inner) % nidx;
        const u64 o = i / (inner * nidx);
        i64 v;
        if (!norm_index(idx[k], (i64)dim, err, v)) continue;
        atomic_add_t(dst + (o * dim + (u64)v) * inner + j, src[i]);
    }
}
#endif

// ---- tril / triu ---------------------------------------------------------------------------------------------------------------
// y = x where the (row, col) of each [rows, cols] matrix satisfies col - row <= diag (lower) or >= diag (upper), else 0.
extern "C" __global__ void triangular(const T* x, T* y, u64 n, u64 rows, u64 cols, i64 diag, int upper) {
    GRID_STRIDE_LOOP(i, n) {
        const i64 c = (i64)(i % cols), r = (i64)((i / cols) % rows);
        const bool keep = upper ? (c - r >= diag) : (c - r <= diag);
        y[i] = keep ? x[i] : (T)0;
    }
}
