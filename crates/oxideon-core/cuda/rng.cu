// Counter-based random numbers (Philox 4x32-10, Salmon et al., SC 2011): element i of a call draws from the block (i, 0, 0, 0)
// under the call's 64-bit key, so the result does not depend on the launch shape or the number of threads. The host derives
// the key from the library's global generator, which keeps `manual_seed` deterministic (the stream differs from the CPU
// generator's, as in torch). One module per float type.
#include "common.cuh"

#if IS_FLOAT

struct Philox {
    u32 x0, x1, x2, x3;
};

__device__ __forceinline__ Philox philox10(u64 counter, u64 key) {
    u32 c0 = (u32)counter, c1 = (u32)(counter >> 32), c2 = 0, c3 = 0;
    u32 k0 = (u32)key, k1 = (u32)(key >> 32);
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const u64 p0 = (u64)0xD2511F53u * c0, p1 = (u64)0xCD9E8D57u * c2;
        const u32 n0 = (u32)(p1 >> 32) ^ c1 ^ k0, n2 = (u32)(p0 >> 32) ^ c3 ^ k1;
        c1 = (u32)p1;
        c3 = (u32)p0;
        c0 = n0;
        c2 = n2;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return {c0, c1, c2, c3};
}

// 24 random bits -> [0, 1)
__device__ __forceinline__ float to_unit(u32 x) { return (float)(x >> 8) * (1.0f / 16777216.0f); }

extern "C" __global__ void rand_uniform(T* y, u64 n, u64 key, double low, double high) {
    GRID_STRIDE_LOOP(i, n) {
        const Philox r = philox10(i, key);
        y[i] = st((C)(low + (high - low) * (double)to_unit(r.x0)));
    }
}

// standard normal by the Box-Muller transform (cosine branch); u1 is shifted into (0, 1] so the logarithm is finite
extern "C" __global__ void rand_normal(T* y, u64 n, u64 key) {
    GRID_STRIDE_LOOP(i, n) {
        const Philox r = philox10(i, key);
        const float u1 = (float)((r.x0 >> 8) + 1u) * (1.0f / 16777216.0f);
        const float u2 = to_unit(r.x1);
        y[i] = st((C)(sqrtf(-2.0f * logf(u1)) * cosf(6.2831853071795864f * u2)));
    }
}

// dropout mask: keep with probability `keep`, scaled by 1 / keep so the expectation is unchanged
extern "C" __global__ void bernoulli_mask(T* y, u64 n, u64 key, double keep, double scale) {
    GRID_STRIDE_LOOP(i, n) {
        const Philox r = philox10(i, key);
        y[i] = st(((double)to_unit(r.x0) < keep) ? (C)scale : (C)0);
    }
}

#endif  // IS_FLOAT
