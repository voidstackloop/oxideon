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

extern "C" __global__ void __launch_bounds__(128) attn_fwd(const T* q, const T* k, const T* v, T* out, const T* mask, u64 L, u64 S, int D, int Dv, double scale_, int causal, u64 H, i64 m_b, i64 m_h, i64 m_l, i64 m_s) {
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
}

#endif  // IS_FLOAT
