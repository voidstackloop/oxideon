// Reductions over one block of axes of a contiguous tensor viewed as [outer, red, inner]. One module per element type.
//
// op codes: 0 sum, 1 mean (sum scaled by `scale`), 2 max, 3 min, 4 prod, 5 argmax, 6 argmin.
// Max / min propagate NaN (like torch.max); argmax / argmin pick the first extremal index and treat NaN as the extreme
// (the first NaN wins), exactly as the CPU backend does. Sums add in a tree, so the last bits can differ from the CPU's
// summation order. A long reduction is split over `splits` blocks per row: each writes a partial result and a second launch
// reduces the partials (the host drives that; see cuda_nn.rs).
#include "common.cuh"

#if IS_FLOAT
#define HAS_ARITH 1
#define LOWEST ((C)(-__int_as_float(0x7f800000)))
#define HIGHEST ((C)__int_as_float(0x7f800000))
#elif DTYPE == DT_I64
#define HAS_ARITH 1
#define LOWEST ((C)(-9223372036854775807ll - 1))
#define HIGHEST ((C)9223372036854775807ll)
#elif IS_SIGNED
#define LOWEST ((C)(-9223372036854775807ll - 1))
#define HIGHEST ((C)9223372036854775807ll)
#else
#define LOWEST ((C)0)
#define HIGHEST ((C)0xffffffffffffffffull)
#endif

__device__ __forceinline__ C identity_of(int op) {
    switch (op) {
        case 0: case 1: return (C)0;
        case 4: return (C)1;
        case 2: return LOWEST;
        default: return HIGHEST;
    }
}

__device__ __forceinline__ C combine(C a, C b, int op) {
    switch (op) {
        case 0: case 1: return a + b;
        case 4: return a * b;
        case 2: return nan_max(a, b);
        default: return nan_min(a, b);
    }
}

// ---- reduction along a contiguous trailing run (inner == 1): block per (row, split) ------------------------------------
// x: [outer, red]; out: [outer, splits]. Dynamic shared memory: blockDim.x * sizeof(C).
#if defined(HAS_ARITH) || (DTYPE != DT_BOOL)
extern "C" __global__ void reduce_rows(const T* x, T* out, u64 outer, u64 red, u64 chunk, int op, double scale) {
    extern __shared__ unsigned char smem_raw[];
    C* sm = (C*)smem_raw;
    const u64 row = blockIdx.x, sp = blockIdx.y, splits = gridDim.y;
    if (row >= outer) return;
    const u64 begin = sp * chunk;
    u64 end = begin + chunk;
    if (end > red) end = red;
    const T* p = x + row * red;
    C acc = identity_of(op);
    for (u64 i = begin + threadIdx.x; i < end; i += blockDim.x) acc = combine(acc, ld(p[i]), op);
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = combine(sm[threadIdx.x], sm[threadIdx.x + s], op);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        C r = sm[0];
        if (op == 1) r = r * (C)scale;
        out[row * splits + sp] = st(r);
    }
}

// ---- reduction over a middle / leading axis (inner > 1): 32 x 8 thread tiles ----------------------------------------------
// x: [outer, red, inner]; out: [outer, splits, inner]. grid = (ceil(inner / 32), outer, splits).
extern "C" __global__ void reduce_cols(const T* x, T* out, u64 outer, u64 red, u64 inner, u64 chunk, int op, double scale) {
    __shared__ C sm[8][33];
    const u64 o = blockIdx.y, sp = blockIdx.z, splits = gridDim.z;
    const u64 j = (u64)blockIdx.x * 32 + threadIdx.x;
    const u64 begin = sp * chunk;
    u64 end = begin + chunk;
    if (end > red) end = red;
    C acc = identity_of(op);
    if (j < inner) {
        const T* p = x + (o * red) * inner + j;
        for (u64 r = begin + threadIdx.y; r < end; r += 8) acc = combine(acc, ld(p[r * inner]), op);
    }
    sm[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0 && j < inner) {
        C r = sm[0][threadIdx.x];
#pragma unroll
        for (int k = 1; k < 8; ++k) r = combine(r, sm[k][threadIdx.x], op);
        if (op == 1) r = r * (C)scale;
        out[(o * splits + sp) * inner + j] = st(r);
    }
}
#endif

// ---- argmax / argmin ------------------------------------------------------------------------------------------------------------
struct VI {
    C v;
    u64 i;
};

// a wins over b when it is strictly more extreme, a NaN beats a number, and ties keep the smaller index
__device__ __forceinline__ VI arg_combine(VI a, VI b, bool is_max) {
    const bool an = a.v != a.v, bn = b.v != b.v;
    bool a_better = is_max ? (a.v > b.v) : (a.v < b.v);
    bool b_better = is_max ? (b.v > a.v) : (b.v < a.v);
    a_better = a_better || (an && !bn);
    b_better = b_better || (bn && !an);
    if (a_better) return a;
    if (b_better) return b;
    return a.i <= b.i ? a : b;
}

extern "C" __global__ void argreduce_rows(const T* x, i64* out, u64 outer, u64 red, int is_max) {
    extern __shared__ unsigned char smem_raw[];
    VI* sm = (VI*)smem_raw;
    const u64 row = blockIdx.x;
    if (row >= outer) return;
    const T* p = x + row * red;
    VI best;
    best.v = (C)0;
    best.i = 0xffffffffffffffffull;
    bool have = false;
    for (u64 i = threadIdx.x; i < red; i += blockDim.x) {
        VI cur;
        cur.v = ld(p[i]);
        cur.i = i;
        best = have ? arg_combine(best, cur, is_max != 0) : cur;
        have = true;
    }
    // threads that saw nothing carry index = max so they lose every tie, and a value that never beats anything
    if (!have) best.v = is_max ? LOWEST : HIGHEST;
    sm[threadIdx.x] = best;
    __syncthreads();
    for (unsigned s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = arg_combine(sm[threadIdx.x], sm[threadIdx.x + s], is_max != 0);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = (i64)sm[0].i;
}

extern "C" __global__ void argreduce_cols(const T* x, i64* out, u64 outer, u64 red, u64 inner, int is_max) {
    const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 o = blockIdx.y;
    if (j >= inner) return;
    const T* p = x + o * red * inner + j;
    VI best;
    best.v = ld(p[0]);
    best.i = 0;
    for (u64 r = 1; r < red; ++r) {
        VI cur;
        cur.v = ld(p[r * inner]);
        cur.i = r;
        best = arg_combine(best, cur, is_max != 0);
    }
    out[o * inner + j] = (i64)best.i;
}

// ---- cumulative sum / product ---------------------------------------------------------------------------------------------------
// Both accumulate in double for float types (the CPU backend widens to f64 as well) and in i64 for i64 tensors.
#if IS_FLOAT
typedef double ACC;
#else
typedef i64 ACC;
#endif

__device__ __forceinline__ ACC scan_step(ACC a, ACC b, int prod) { return prod ? a * b : a + b; }

// inner == 1: block per row. Each thread scans a contiguous chunk, the chunk totals are scanned in shared memory, then every
// thread rewrites its chunk with the exclusive prefix. Dynamic shared memory: blockDim.x * sizeof(ACC).
extern "C" __global__ void scan_rows(const T* x, T* y, u64 outer, u64 n, int prod) {
    extern __shared__ unsigned char smem_raw[];
    ACC* sm = (ACC*)smem_raw;
    const u64 row = blockIdx.x;
    if (row >= outer) return;
    const T* px = x + row * n;
    T* py = y + row * n;
    const u64 per = (n + blockDim.x - 1) / blockDim.x;
    const u64 begin = (u64)threadIdx.x * per;
    u64 end = begin + per;
    if (end > n) end = n;
    ACC total = prod ? (ACC)1 : (ACC)0;
    for (u64 i = begin; i < end; ++i) total = scan_step(total, (ACC)ld(px[i]), prod);
    sm[threadIdx.x] = total;
    __syncthreads();
    if (threadIdx.x == 0) {
        ACC run = prod ? (ACC)1 : (ACC)0;
        for (unsigned t = 0; t < blockDim.x; ++t) {
            const ACC v = sm[t];
            sm[t] = run;  // exclusive prefix of the chunk totals
            run = scan_step(run, v, prod);
        }
    }
    __syncthreads();
    ACC acc = sm[threadIdx.x];
    for (u64 i = begin; i < end; ++i) {
        acc = scan_step(acc, (ACC)ld(px[i]), prod);
        py[i] = st((C)acc);
    }
}

// inner > 1: thread per (outer, inner) position, walking the axis (neighbouring threads touch neighbouring elements)
extern "C" __global__ void scan_cols(const T* x, T* y, u64 outer, u64 n, u64 inner, int prod) {
    const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 o = blockIdx.y;
    if (j >= inner) return;
    ACC acc = prod ? (ACC)1 : (ACC)0;
    for (u64 r = 0; r < n; ++r) {
        const u64 at = (o * n + r) * inner + j;
        acc = scan_step(acc, (ACC)ld(x[at]), prod);
        y[at] = st((C)acc);
    }
}

// cumprod backward, zero-safe like the CPU path: gx[k] = sum_{m>=k} go[m] * prod_{j<=m, j!=k} x[j]  (out = inclusive cumprod of x)
extern "C" __global__ void cumprod_bwd(const T* x, const T* out, const T* go, T* gx, u64 outer, u64 n, u64 inner) {
    const u64 total = outer * n * inner;
    GRID_STRIDE_LOOP(t, total) {
        const u64 j = t % inner, k = (t / inner) % n, o = t / (inner * n);
        const u64 base = o * n * inner + j;
        ACC prefix = k == 0 ? (ACC)1 : (ACC)ld(out[base + (k - 1) * inner]);
        ACC suffix = (ACC)1, sum = (ACC)0;
        for (u64 m = k; m < n; ++m) {
            if (m > k) suffix *= (ACC)ld(x[base + m * inner]);
            sum += (ACC)ld(go[base + m * inner]) * prefix * suffix;
        }
        gx[base + k * inner] = st((C)sum);
    }
}

// ---- top-k / sort ------------------------------------------------------------------------------------------------------------------
// Sorts the indices of each row by value with the CPU backend's order: a total order on the values (NaN largest, -0 before +0)
// and ties broken by the original index. One block sorts a chunk of up to SORT_N candidates with a bitonic network and keeps its
// best `k`; the host repeats the stage over the surviving candidates until one chunk is left, which writes the final i64
// indices. Only indices are sorted (keys are re-read from the row), so the stages need no value buffers.
#define SORT_N 4096

__device__ __forceinline__ u64 sort_key(C v) {
#if IS_FLOAT
    const u64 b = (u64)__double_as_longlong((double)v);
    return (b >> 63) ? ~b : (b | 0x8000000000000000ull);
#elif defined(IS_SIGNED) && IS_SIGNED
    return (u64)v ^ 0x8000000000000000ull;
#else
    return (u64)v;
#endif
}

__device__ __forceinline__ bool comes_first(u64 ka, u32 ia, u64 kb, u32 ib, bool largest) {
    if (ia == 0xffffffffu) return false;  // padding sorts last
    if (ib == 0xffffffffu) return true;
    if (ka != kb) return largest ? ka > kb : ka < kb;
    return ia < ib;
}

extern "C" __global__ void __launch_bounds__(1024) topk_stage(const T* x, const u32* cand_in, u32* cand_out, i64* final_out, u64 n, u64 count_in, u64 in_stride, u64 k, u64 out_stride, int largest) {
    extern __shared__ unsigned char smem_raw[];
    u64* keys = (u64*)smem_raw;
    u32* ids = (u32*)(keys + SORT_N);
    const u64 row = blockIdx.y, chunk = blockIdx.x;
    const u64 base = chunk * SORT_N;
    for (int t = 0; t < SORT_N / 1024; ++t) {
        const u32 p = threadIdx.x + t * 1024;
        const u64 pos = base + p;
        if (pos < count_in) {
            const u32 id = cand_in ? cand_in[row * in_stride + pos] : (u32)pos;
            keys[p] = sort_key(ld(x[row * n + id]));
            ids[p] = id;
        } else {
            keys[p] = 0;
            ids[p] = 0xffffffffu;
        }
    }
    __syncthreads();
    for (u32 size = 2; size <= SORT_N; size <<= 1) {
        for (u32 stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = 0; t < SORT_N / 2048; ++t) {
                const u32 i = threadIdx.x + t * 1024;
                const u32 pos = ((i & ~(stride - 1)) << 1) | (i & (stride - 1));
                const u32 other = pos | stride;
                const bool up = (pos & size) == 0;
                const bool other_first = comes_first(keys[other], ids[other], keys[pos], ids[pos], largest != 0);
                const bool pos_first = comes_first(keys[pos], ids[pos], keys[other], ids[other], largest != 0);
                if (up ? other_first : pos_first) {
                    const u64 tk = keys[pos]; keys[pos] = keys[other]; keys[other] = tk;
                    const u32 ti = ids[pos]; ids[pos] = ids[other]; ids[other] = ti;
                }
            }
            __syncthreads();
        }
    }
    u64 valid = count_in - base;
    if (valid > SORT_N) valid = SORT_N;
    const u64 keep = k < valid ? k : valid;
    for (u64 r = threadIdx.x; r < keep; r += blockDim.x) {
        if (final_out) final_out[row * out_stride + r] = (i64)ids[r];
        else cand_out[row * out_stride + chunk * (k < SORT_N ? k : SORT_N) + r] = ids[r];
    }
}

// ---- global bitonic sort (large k on long rows) ------------------------------------------------------------------------------------
// When more than SORT_N candidates must survive (k > SORT_N / 2) the staged block sort cannot shrink the candidate set, so the
// whole row of indices is sorted by a bitonic network over global memory instead: one launch per (size, stride) step, same
// comparison order as above. npad is the row length rounded up to a power of two; padding entries carry id 0xffffffff.
extern "C" __global__ void gsort_init(u32* ids, u64 n, u64 npad, u64 total) {
    GRID_STRIDE_LOOP(i, total) {
        const u64 p = i % npad;
        ids[i] = p < n ? (u32)p : 0xffffffffu;
    }
}

extern "C" __global__ void gsort_step(const T* x, u32* ids, u64 n, u64 npad, u32 size, u32 stride, int largest) {
    const u64 row = blockIdx.y;
    u32* r = ids + row * npad;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < (npad >> 1); i += (u64)gridDim.x * blockDim.x) {
        const u64 pos = ((i & ~(u64)(stride - 1)) << 1) | (i & (stride - 1));
        const u64 other = pos | stride;
        const u32 ip = r[pos], io = r[other];
        const u64 kp = ip == 0xffffffffu ? 0 : sort_key(ld(x[row * n + ip]));
        const u64 ko = io == 0xffffffffu ? 0 : sort_key(ld(x[row * n + io]));
        const bool up = (pos & size) == 0;
        const bool other_first = comes_first(ko, io, kp, ip, largest != 0);
        const bool pos_first = comes_first(kp, ip, ko, io, largest != 0);
        if (up ? other_first : pos_first) {
            r[pos] = io;
            r[other] = ip;
        }
    }
}

extern "C" __global__ void gsort_take(const u32* ids, i64* out, u64 npad, u64 k, u64 total) {
    GRID_STRIDE_LOOP(i, total) out[i] = (i64)ids[(i / k) * npad + (i % k)];
}
