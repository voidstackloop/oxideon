// Tensor-core matrix multiply (sm_80 and newer): C[b] = A[b] @ B[b] with `mma.sync.aligned` issued through inline PTX, so no
// toolkit headers are needed.
//
//   f16 / bf16 modules: m16n8k16 with f32 accumulation (products of 16-bit values are exact in f32, so only the summation
//                       order differs from the SIMT kernel).
//   f32 module:         m16n8k8 on TF32 operands (10-bit mantissa) - faster, but less precise than f32; the host enables it
//                       only on request (`set_allow_tf32` / OXIDEON_CUDA_TF32=1).
//
// Same interface as gemm.cu: arbitrary element strides for A and B, broadcast batches through a Strided descriptor, output
// row-major [batch, M, N]. A 64x64 block tile, four warps each owning a 32x32 piece.
#include "common.cuh"

#if DTYPE == DT_F16 || DTYPE == DT_BF16

#define BM 64
#define BN 64
#define BK 32
#define PAD 8

__device__ __forceinline__ void mma16816(float& d0, float& d1, float& d2, float& d3, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
#if DTYPE == DT_F16
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

extern "C" __global__ void __launch_bounds__(128) gemm_tc(const T* A, const T* B, T* Cm, int M, int N, int K, i64 rsa, i64 csa, i64 rsb, i64 csb, Strided bs) {
    __shared__ u16 As[BM][BK + PAD];  // A tile, row-major (m, k)
    __shared__ u16 Bs[BN][BK + PAD];  // B tile, stored (n, k) so a column of B is contiguous in k
    const u64 z = blockIdx.z;
    A += offset_of<0>(bs, z);
    B += offset_of<1>(bs, z);
    Cm += (i64)z * M * N;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const bool a_row_major = csa == 1, b_row_major = csb == 1;

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int e = tid; e < BM * BK; e += 128) {
            const int m = a_row_major ? e / BK : e % BM;
            const int k = a_row_major ? e % BK : e / BM;
            const int gm = m0 + m, gk = k0 + k;
            As[m][k] = (gm < M && gk < K) ? A[(i64)gm * rsa + (i64)gk * csa] : (u16)0;
        }
        for (int e = tid; e < BK * BN; e += 128) {
            const int k = b_row_major ? e / BN : e % BK;
            const int n = b_row_major ? e % BN : e / BK;
            const int gk = k0 + k, gn = n0 + n;
            Bs[n][k] = (gk < K && gn < N) ? B[(i64)gk * rsb + (i64)gn * csb] : (u16)0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            u32 a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wm * 32 + mi * 16 + gid;
                a[mi][0] = *(const u32*)&As[r][kk + tig * 2];
                a[mi][1] = *(const u32*)&As[r + 8][kk + tig * 2];
                a[mi][2] = *(const u32*)&As[r][kk + tig * 2 + 8];
                a[mi][3] = *(const u32*)&As[r + 8][kk + tig * 2 + 8];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int n = wn * 32 + ni * 8 + gid;
                const u32 b0 = *(const u32*)&Bs[n][kk + tig * 2];
                const u32 b1 = *(const u32*)&Bs[n][kk + tig * 2 + 8];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) mma16816(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int r = m0 + wm * 32 + mi * 16 + gid;
            const int c = n0 + wn * 32 + ni * 8 + tig * 2;
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const int rr = r + (h >> 1) * 8, cc = c + (h & 1);
                if (rr < M && cc < N) Cm[(i64)rr * N + cc] = st(acc[mi][ni][h]);
            }
        }
    }
}

#endif  // f16 / bf16

#if DTYPE == DT_F32

#define BM 64
#define BN 64
#define BK 16
#define PADF 4

__device__ __forceinline__ u32 to_tf32(float x) {
    u32 r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma1688(float& d0, float& d1, float& d2, float& d3, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void __launch_bounds__(128) gemm_tf32(const T* A, const T* B, T* Cm, int M, int N, int K, i64 rsa, i64 csa, i64 rsb, i64 csb, Strided bs) {
    __shared__ u32 As[BM][BK + PADF];  // TF32 bit patterns, (m, k)
    __shared__ u32 Bs[BN][BK + PADF];  // (n, k)
    const u64 z = blockIdx.z;
    A += offset_of<0>(bs, z);
    B += offset_of<1>(bs, z);
    Cm += (i64)z * M * N;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, tig = lane & 3;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const bool a_row_major = csa == 1, b_row_major = csb == 1;

    float acc[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int e = tid; e < BM * BK; e += 128) {
            const int m = a_row_major ? e / BK : e % BM;
            const int k = a_row_major ? e % BK : e / BM;
            const int gm = m0 + m, gk = k0 + k;
            As[m][k] = (gm < M && gk < K) ? to_tf32(A[(i64)gm * rsa + (i64)gk * csa]) : 0u;
        }
        for (int e = tid; e < BK * BN; e += 128) {
            const int k = b_row_major ? e / BN : e % BK;
            const int n = b_row_major ? e % BN : e / BK;
            const int gk = k0 + k, gn = n0 + n;
            Bs[n][k] = (gk < K && gn < N) ? to_tf32(B[(i64)gk * rsb + (i64)gn * csb]) : 0u;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            u32 a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = wm * 32 + mi * 16 + gid;
                a[mi][0] = As[r][kk + tig];
                a[mi][1] = As[r + 8][kk + tig];
                a[mi][2] = As[r][kk + tig + 4];
                a[mi][3] = As[r + 8][kk + tig + 4];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int n = wn * 32 + ni * 8 + gid;
                const u32 b0 = Bs[n][kk + tig], b1 = Bs[n][kk + tig + 4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) mma1688(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int r = m0 + wm * 32 + mi * 16 + gid;
            const int c = n0 + wn * 32 + ni * 8 + tig * 2;
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const int rr = r + (h >> 1) * 8, cc = c + (h & 1);
                if (rr < M && cc < N) Cm[(i64)rr * N + cc] = acc[mi][ni][h];
            }
        }
    }
}

#endif  // f32 (TF32)
