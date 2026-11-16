# Oxideon Unified Multi-Backend Hardware Kernel Benchmark Report

Generated: 2026-11-07 13:48:20  
Oxideon Version: `0.1.0`  
Hardware Target: Cross-Platform Unified Runtime (Metal MSL, NVIDIA PTX, WebGPU WGSL, Vulkan SPIR-V)

---

## 1. Hardware Kernel Emission Matrix

| Operator | Apple Metal (MSL) | NVIDIA PTX | CUDA C | WebGPU (WGSL) | Vulkan (SPIR-V) |
|:---------|:------------------|:-----------|:-------|:--------------|:----------------|
| `flash_attention` | 75 lines | 54 lines | 54 lines | 68 lines | 54 lines |
| `rmsnorm` | 46 lines | 57 lines | 29 lines | 53 lines | 54 lines |
| `swiglu` | 16 lines | 49 lines | 29 lines | 16 lines | 51 lines |
| `rope` | 30 lines | 57 lines | 54 lines | 22 lines | 54 lines |
| `q4_gemv` | 44 lines | 48 lines | 29 lines | 34 lines | 54 lines |
| `matmul` | 32 lines | 74 lines | 54 lines | 56 lines | 54 lines |

---

## 2. FlashAttention-2 Tiled vs Naive Attention

| Sequence Length ($S$) | Naive Attention Latency | Fused FlashAttn Latency | Speedup |
|:----------------------|:------------------------|:------------------------|:--------|
| $S = 64$ | 4.805 ms | 0.687 ms | **6.99x** |
| $S = 128$ | 7.064 ms | 2.437 ms | **2.90x** |
| $S = 256$ | 12.897 ms | 9.098 ms | **1.42x** |
| $S = 512$ | 30.480 ms | 35.427 ms | **0.86x** |
| $S = 1024$ | 125.890 ms | 140.269 ms | **0.90x** |

---

## 3. Paged KV Cache Dynamic Serving Attention

| Concurrency (Batch Size) | Per-Token Decode Latency | Generation Throughput |
|:-------------------------|:-------------------------|:----------------------|
| $B = 1$ | 0.011 ms | **87416.4 tokens/sec** |
| $B = 2$ | 0.020 ms | **100854.7 tokens/sec** |
| $B = 4$ | 0.043 ms | **92517.6 tokens/sec** |
| $B = 8$ | 0.096 ms | **83537.0 tokens/sec** |
| $B = 16$ | 0.207 ms | **77281.1 tokens/sec** |

---

## 4. MLX 4-bit Affine Quantized GEMM vs Float32

| Weight Matrix $[M \times K]$ | Dense F32 Latency | MLX Q4 GEMM Latency | Weight Compression | Relative Speed |
|:------------------------------|:------------------|:--------------------|:-------------------|:---------------|
| `[128x64]` | 0.023 ms | 0.043 ms | **1.78x** | **0.53x** |
| `[256x128]` | 0.133 ms | 0.198 ms | **1.78x** | **0.68x** |
| `[512x256]` | 0.627 ms | 0.918 ms | **1.78x** | **0.68x** |
| `[1024x512]` | 0.541 ms | 2.083 ms | **1.78x** | **0.26x** |

---

## 5. Fused Memory-Bound Operators

- **Fused Add + RMSNorm**: `0.813 ms` vs `0.076 ms` (**0.09x speedup**)
- **Fused SwiGLU Gating**: `2.025 ms` vs `3.797 ms` (**1.88x speedup**)
