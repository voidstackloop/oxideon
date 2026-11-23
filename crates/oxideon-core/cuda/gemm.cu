// Batched matrix multiply C[b] = A[b] @ B[b] for f32 / f64 / f16 / bf16 (one module per type; 16-bit types accumulate in f32).
//
//   * arbitrary element strides for A and B, so transposed and sliced views are multiplied in place;
//   * broadcast batch dimensions through a Strided descriptor (stride set 0 = A, 1 = B), with strides in elements;
//   * a 128x128 block tile with an 8x8 register tile per thread (256 threads) for large products, and a 32x32 tile with
//     2x2 per thread for small and skinny ones. Columns of the register tile are interleaved across threads so that both the
//     shared-memory reads of B and the global writes of C are coalesced.
#include "common.cuh"

// i32 results outside the i32 range become INT_MIN, like the CPU backend (it accumulates in f64 and casts); everything else
// is the plain conversion
__device__ __forceinline__ T gemm_store(C v) {
#if DTYPE == DT_I32
    return (v < (C)(-2147483647 - 1) || v > (C)2147483647) ? (T)(-2147483647 - 1) : (T)v;
#else
    return st(v);
#endif
}

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void gemm_tile(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ Cm, int M, int N, int K, i64 rsa, i64 csa, i64 rsb, i64 csb) {
    constexpr int TX = BN / TN;  // threads along n
    constexpr int TY = BM / TM;  // threads along m
    constexpr int NT = TX * TY;
    __shared__ C As[BK][BM + 4];
    __shared__ C Bs[BK][BN + 4];

    const int tid = threadIdx.x;
    const int tx = tid % TX, ty = tid / TX;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    C acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (C)0;

    const bool a_row_major = csa == 1;  // consecutive k are adjacent in memory
    const bool b_row_major = csb == 1;  // consecutive n are adjacent in memory

    for (int k0 = 0; k0 < K; k0 += BK) {
        // A tile: BM x BK, stored k-major
#pragma unroll
        for (int e = tid; e < BM * BK; e += NT) {
            const int m = a_row_major ? e / BK : e % BM;
            const int k = a_row_major ? e % BK : e / BM;
            const int gm = m0 + m, gk = k0 + k;
            As[k][m] = (gm < M && gk < K) ? ld(A[(i64)gm * rsa + (i64)gk * csa]) : (C)0;
        }
        // B tile: BK x BN
#pragma unroll
        for (int e = tid; e < BK * BN; e += NT) {
            const int k = b_row_major ? e / BN : e % BK;
            const int n = b_row_major ? e % BN : e / BK;
            const int gk = k0 + k, gn = n0 + n;
            Bs[k][n] = (gk < K && gn < N) ? ld(B[(i64)gk * rsb + (i64)gn * csb]) : (C)0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            C a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty + i * TY];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx + j * TX];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = m0 + ty + i * TY;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gn = n0 + tx + j * TX;
            if (gn < N) Cm[(i64)gm * N + gn] = gemm_store(acc[i][j]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) gemm_large(const T* A, const T* B, T* Cm, int M, int N, int K, i64 rsa, i64 csa, i64 rsb, i64 csb, Strided bs) {
    const u64 z = blockIdx.z;
    gemm_tile<128, 128, 8, 8, 8>(A + offset_of<0>(bs, z), B + offset_of<1>(bs, z), Cm + (i64)z * M * N, M, N, K, rsa, csa, rsb, csb);
}

extern "C" __global__ void __launch_bounds__(256) gemm_small(const T* A, const T* B, T* Cm, int M, int N, int K, i64 rsa, i64 csa, i64 rsb, i64 csb, Strided bs) {
    const u64 z = blockIdx.z;
    gemm_tile<32, 32, 16, 2, 2>(A + offset_of<0>(bs, z), B + offset_of<1>(bs, z), Cm + (i64)z * M * N, M, N, K, rsa, csa, rsb, csb);
}
