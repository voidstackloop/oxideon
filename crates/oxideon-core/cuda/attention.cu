// Fused attention forward: out = softmax(q k^T * scale + mask) v, streamed over key tiles with an online softmax, so the
// [L, S] score matrix is never written to memory (flash-attention style, one warp per query row). Float types, head
// dimension up to 128; q is [BH, L, D], k is [BH, S, D], v is [BH, S, Dv], out is [BH, L, Dv].
//
// The mask is either absent, causal (key j visible to query i when j <= i, top-left aligned like the CPU path), an additive
// float mask given by element strides over (batch, head, L, S) - stride 0 broadcasts - or both. Rows whose scores are all -inf produce
// NaN, as the composed softmax does.
#include "common.cuh"

#if IS_FLOAT

#define ATT_WARPS 4
#define ATT_KEYS 32
#define ATT_MAXD 128

extern "C" __global__ void __launch_bounds__(128) attn_fwd(const T* q, const T* k, const T* v, T* out, const T* mask, u64 L, u64 S, int D, int Dv, double scale_, int causal, u64 H, i64 m_b, i64 m_h, i64 m_l, i64 m_s, C* lse) {
    // shared tiles: K and V for ATT_KEYS keys, padded by one element to avoid bank conflicts; one q row per warp
    __shared__ C ks[ATT_KEYS][ATT_MAXD + 1];
    __shared__ C vs[ATT_KEYS][ATT_MAXD + 1];
    __shared__ C qs[ATT_WARPS][ATT_MAXD];
    const C scale = (C)scale_;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u64 bh = blockIdx.y;
    const u64 row = (u64)blockIdx.x * ATT_WARPS + warp;
    const bool active = row < L;

    const T* qrow = q + (bh * L + (active ? row : 0)) * D;
    for (int d = lane; d < D; d += 32) qs[warp][d] = ld(qrow[d]);
    const T* kb = k + bh * S * D;
    const T* vb = v + bh * S * Dv;
    // the mask is addressed by (batch, head, query row, key) strides; a stride of 0 broadcasts that dimension
    const T* mrow = mask ? mask + (i64)(bh / H) * m_b + (i64)(bh % H) * m_h + (i64)(active ? row : 0) * m_l : nullptr;

    const C neg_inf = (C)(-__int_as_float(0x7f800000));
    C running_max = neg_inf, running_sum = (C)0;
    C acc[ATT_MAXD / 32];
#pragma unroll
    for (int t = 0; t < ATT_MAXD / 32; ++t) acc[t] = (C)0;

    const u64 last_key = causal ? (active ? (row + 1 < S ? row + 1 : S) : 0) : S;  // keys this row may see
    const u64 block_last = causal ? (((u64)blockIdx.x * ATT_WARPS + ATT_WARPS) < S ? ((u64)blockIdx.x * ATT_WARPS + ATT_WARPS) : S) : S;

    for (u64 j0 = 0; j0 < block_last; j0 += ATT_KEYS) {
        __syncthreads();
        const int nk = (int)(block_last - j0 < ATT_KEYS ? block_last - j0 : ATT_KEYS);
        for (int e = threadIdx.x; e < nk * D; e += blockDim.x) ks[e / D][e % D] = ld(kb[(j0 + e / D) * D + e % D]);
        for (int e = threadIdx.x; e < nk * Dv; e += blockDim.x) vs[e / Dv][e % Dv] = ld(vb[(j0 + e / Dv) * Dv + e % Dv]);
        __syncthreads();
        if (!active) continue;
        // lane j scores key j0 + j
        C s = neg_inf;
        const u64 key = j0 + lane;
        if (lane < nk && key < last_key) {
            C dot = (C)0;
            for (int d = 0; d < D; ++d) dot += qs[warp][d] * ks[lane][d];
            s = dot * scale;
            if (mrow) s += ld(mrow[(i64)key * m_s]);
        }
        C tile_max = s;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const C other = __shfl_xor_sync(0xffffffffu, tile_max, o);
            tile_max = (other > tile_max || other != other) ? other : tile_max;
        }
        if (tile_max == neg_inf && running_max == neg_inf) continue;  // nothing visible yet: keep the state at "empty"
        const C new_max = tile_max > running_max ? tile_max : running_max;
        const C correction = m_exp(running_max - new_max);
        C p = (s == neg_inf) ? (C)0 : m_exp(s - new_max);
        C psum = p;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
        running_sum = running_sum * correction + psum;
        running_max = new_max;
#pragma unroll
        for (int t = 0; t < ATT_MAXD / 32; ++t) acc[t] *= correction;
        for (int jj = 0; jj < nk; ++jj) {
            const C pj = __shfl_sync(0xffffffffu, p, jj);
            if (pj == (C)0) continue;
#pragma unroll
            for (int t = 0; t < ATT_MAXD / 32; ++t) {
                const int d = lane + 32 * t;
                if (d < Dv) acc[t] += pj * vs[jj][d];
            }
        }
    }
    if (!active) return;
    T* orow = out + (bh * L + row) * Dv;
    const C inv = running_sum > (C)0 ? (C)1 / running_sum : (C)(__int_as_float(0x7fc00000));  // all keys masked -> NaN
#pragma unroll
    for (int t = 0; t < ATT_MAXD / 32; ++t) {
        const int d = lane + 32 * t;
        if (d < Dv) orow[d] = st(acc[t] * inv);
    }
    // training: the log-sum-exp of the row, from which the backward kernels recompute every probability (+inf: no visible key)
    if (lse && lane == 0) lse[bh * L + row] = running_sum > (C)0 ? running_max + m_log(running_sum) : (C)__int_as_float(0x7f800000);
}

// ---- backward ------------------------------------------------------------------------------------------------------------------
// With P = softmax(S), S = q k^T * scale + mask, and dO the output gradient: Drow_i = sum_d dO_i,d * O_i,d,
// dS_ij = P_ij (dO_i . v_j - Drow_i), dq_i = scale * sum_j dS_ij k_j, dk_j = scale * sum_i dS_ij q_i, dv_j = sum_i P_ij dO_i.
// P_ij = exp(S_ij - lse_i) is recomputed tile by tile, so no [L, S] matrix is stored. Two kernels, no atomics: one warp per
// query row (dq, and Drow for the second), then one warp per key row (dk, dv). The mask (additive, strided) gets no gradient.

__device__ __forceinline__ C att_dot(const C* a, const C* b, int D) {
    C dot = (C)0;
    for (int d = 0; d < D; ++d) dot += a[d] * b[d];
    return dot;
}

extern "C" __global__ void __launch_bounds__(128) attn_bwd_dq(const T* q, const T* k, const T* v, const T* o, const T* g, const C* lse, const T* mask, T* dq, C* drow, u64 L, u64 S, int D, int Dv, double scale_, int causal, u64 H, i64 m_b, i64 m_h, i64 m_l, i64 m_s) {
    __shared__ C ks[ATT_KEYS][ATT_MAXD + 1];
    __shared__ C vs[ATT_KEYS][ATT_MAXD + 1];
    __shared__ C qs[ATT_WARPS][ATT_MAXD];
    __shared__ C gs[ATT_WARPS][ATT_MAXD];
    const C scale = (C)scale_;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u64 bh = blockIdx.y;
    const u64 row = (u64)blockIdx.x * ATT_WARPS + warp;
    const bool active = row < L;
    const u64 r = active ? row : 0;
    const T* qrow = q + (bh * L + r) * D;
    const T* grow = g + (bh * L + r) * Dv;
    const T* orow = o + (bh * L + r) * Dv;
    for (int d = lane; d < D; d += 32) qs[warp][d] = ld(qrow[d]);
    C dr = (C)0;
    for (int d = lane; d < Dv; d += 32) {
        const C gd = ld(grow[d]);
        gs[warp][d] = gd;
        dr += gd * ld(orow[d]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dr += __shfl_xor_sync(0xffffffffu, dr, off);
    if (active && lane == 0) drow[bh * L + row] = dr;
    const C li = lse[bh * L + r];
    const T* kb = k + bh * S * D;
    const T* vb = v + bh * S * Dv;
    const T* mrow = mask ? mask + (i64)(bh / H) * m_b + (i64)(bh % H) * m_h + (i64)r * m_l : nullptr;
    const C pos_inf = (C)__int_as_float(0x7f800000);
    C acc[ATT_MAXD / 32];
#pragma unroll
    for (int t = 0; t < ATT_MAXD / 32; ++t) acc[t] = (C)0;
    const u64 last_key = causal ? (r + 1 < S ? r + 1 : S) : S;
    const u64 block_last = causal ? (((u64)blockIdx.x * ATT_WARPS + ATT_WARPS) < S ? ((u64)blockIdx.x * ATT_WARPS + ATT_WARPS) : S) : S;
    for (u64 j0 = 0; j0 < block_last; j0 += ATT_KEYS) {
        __syncthreads();
        const int nk = (int)(block_last - j0 < ATT_KEYS ? block_last - j0 : ATT_KEYS);
        for (int e = threadIdx.x; e < nk * D; e += blockDim.x) ks[e / D][e % D] = ld(kb[(j0 + e / D) * D + e % D]);
        for (int e = threadIdx.x; e < nk * Dv; e += blockDim.x) vs[e / Dv][e % Dv] = ld(vb[(j0 + e / Dv) * Dv + e % Dv]);
        __syncthreads();
        if (!active || li == pos_inf) continue;
        C ds = (C)0;
        const u64 key = j0 + lane;
        if (lane < nk && key < last_key) {
            C sc = att_dot(qs[warp], ks[lane], D) * scale;
            if (mrow) sc += ld(mrow[(i64)key * m_s]);
            const C p = m_exp(sc - li);  // a -inf score gives 0
            ds = p * (att_dot(gs[warp], vs[lane], Dv) - dr);
        }
        for (int jj = 0; jj < nk; ++jj) {
            const C dj = __shfl_sync(0xffffffffu, ds, jj);
            if (dj == (C)0) continue;
#pragma unroll
            for (int t = 0; t < ATT_MAXD / 32; ++t) {
                const int d = lane + 32 * t;
                if (d < D) acc[t] += dj * ks[jj][d];
            }
        }
    }
    if (!active) return;
    T* out = dq + (bh * L + row) * D;
#pragma unroll
    for (int t = 0; t < ATT_MAXD / 32; ++t) {
        const int d = lane + 32 * t;
        if (d < D) out[d] = st(acc[t] * scale);
    }
}

extern "C" __global__ void __launch_bounds__(128) attn_bwd_dkv(const T* q, const T* k, const T* v, const T* g, const C* lse, const C* drow, const T* mask, T* dk, T* dv, u64 L, u64 S, int D, int Dv, double scale_, int causal, u64 H, i64 m_b, i64 m_h, i64 m_l, i64 m_s) {
    // tiles of ATT_KEYS query rows (q and dO) shared by the warps of the block; one key row (k, v) per warp
    __shared__ C qt[ATT_KEYS][ATT_MAXD + 1];
    __shared__ C gt[ATT_KEYS][ATT_MAXD + 1];
    __shared__ C lt[ATT_KEYS];
    __shared__ C dt[ATT_KEYS];
    __shared__ C kr[ATT_WARPS][ATT_MAXD];
    __shared__ C vr[ATT_WARPS][ATT_MAXD];
    const C scale = (C)scale_;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const u64 bh = blockIdx.y;
    const u64 col = (u64)blockIdx.x * ATT_WARPS + warp;  // the key row of this warp
    const bool active = col < S;
    const u64 c = active ? col : 0;
    for (int d = lane; d < D; d += 32) kr[warp][d] = ld(k[(bh * S + c) * D + d]);
    for (int d = lane; d < Dv; d += 32) vr[warp][d] = ld(v[(bh * S + c) * Dv + d]);
    const T* qb = q + bh * L * D;
    const T* gb = g + bh * L * Dv;
    const C* lb = lse + bh * L;
    const C* db = drow + bh * L;
    const T* mb = mask ? mask + (i64)(bh / H) * m_b + (i64)(bh % H) * m_h + (i64)c * m_s : nullptr;
    const C pos_inf = (C)__int_as_float(0x7f800000);
    C ak[ATT_MAXD / 32], av[ATT_MAXD / 32];
#pragma unroll
    for (int t = 0; t < ATT_MAXD / 32; ++t) ak[t] = av[t] = (C)0;
    // causal: query i sees key j only when j <= i, so the keys of this block need queries from its first key on
    const u64 first_q = causal ? (u64)blockIdx.x * ATT_WARPS : 0;
    for (u64 i0 = first_q; i0 < L; i0 += ATT_KEYS) {
        __syncthreads();
        const int nq = (int)(L - i0 < ATT_KEYS ? L - i0 : ATT_KEYS);
        for (int e = threadIdx.x; e < nq * D; e += blockDim.x) qt[e / D][e % D] = ld(qb[(i0 + e / D) * D + e % D]);
        for (int e = threadIdx.x; e < nq * Dv; e += blockDim.x) gt[e / Dv][e % Dv] = ld(gb[(i0 + e / Dv) * Dv + e % Dv]);
        for (int e = threadIdx.x; e < nq; e += blockDim.x) {
            lt[e] = lb[i0 + e];
            dt[e] = db[i0 + e];
        }
        __syncthreads();
        if (!active) continue;
        C p = (C)0, ds = (C)0;
        const u64 qi = i0 + lane;
        if (lane < nq && lt[lane] != pos_inf && (!causal || c <= qi)) {
            C sc = att_dot(qt[lane], kr[warp], D) * scale;
            if (mb) sc += ld(mb[(i64)qi * m_l]);
            p = m_exp(sc - lt[lane]);
            ds = p * (att_dot(gt[lane], vr[warp], Dv) - dt[lane]);
        }
        for (int ii = 0; ii < nq; ++ii) {
            const C pi = __shfl_sync(0xffffffffu, p, ii);
            const C di = __shfl_sync(0xffffffffu, ds, ii);
            if (pi == (C)0 && di == (C)0) continue;
#pragma unroll
            for (int t = 0; t < ATT_MAXD / 32; ++t) {
                const int d = lane + 32 * t;
                if (d < D) ak[t] += di * qt[ii][d];
                if (d < Dv) av[t] += pi * gt[ii][d];
            }
        }
    }
    if (!active) return;
#pragma unroll
    for (int t = 0; t < ATT_MAXD / 32; ++t) {
        const int d = lane + 32 * t;
        if (d < D) dk[(bh * S + col) * D + d] = st(ak[t] * scale);
        if (d < Dv) dv[(bh * S + col) * Dv + d] = st(av[t]);
    }
}

#endif  // IS_FLOAT
