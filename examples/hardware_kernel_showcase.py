#!/usr/bin/env python3
"""Unified Multi-Backend Hardware Kernel Showcase.

Demonstrates direct developer usage of Oxideon's fused hardware acceleration kernels:
  1. Inspecting emitted assembly/shaders for Apple Silicon Metal (MSL), NVIDIA (PTX & CUDA C),
     WebGPU (WGSL), and Vulkan (SPIR-V).
  2. FlashAttention-2 Tiled & Fused SDPA for causal language modeling.
  3. Fused Rotary Position Embeddings (RoPE).
  4. Fused SwiGLU & GeGLU non-linear activations.
  5. Fused Add + RMSNorm residual connections.
  6. MLX-compatible 4-bit affine quantized matrix-vector operations.
  7. Paged KV Cache Attention for dynamic LLM inference.

Usage:
  python examples/hardware_kernel_showcase.py
"""

import time
import numpy as np

import oxideon as ox
from oxideon import Tensor, from_numpy
from oxideon.kernels import inspect_kernel


def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"⚙️  {title}")
    print("=" * 76)


def main():
    ox.manual_seed(42)
    np.random.seed(42)

    banner("Oxideon Unified Multi-Backend Hardware Kernel Engine")

    # -----------------------------------------------------------------------
    # Step 1: Side-by-Side Kernel Assembly Inspection
    # -----------------------------------------------------------------------
    print("\n[Step 1] Inspecting Emitted Multi-Backend Kernel Assembly for FlashAttention...")
    emitted = inspect_kernel("flash_attention", head_dim=64, is_causal=True)

    print(f"  • Apple Silicon Metal (MSL) lines of code: {len(emitted['msl'].splitlines())}")
    print(f"  • NVIDIA CUDA PTX lines of code: {len(emitted['ptx'].splitlines())}")
    print(f"  • NVIDIA CUDA C lines of code: {len(emitted['cuda'].splitlines())}")
    print(f"  • WebGPU WGSL lines of code: {len(emitted['wgsl'].splitlines())}")
    print(f"  • Vulkan SPIR-V disassembly lines of code: {len(emitted['spirv_disasm'].splitlines())}")

    print("\n  Sample MSL Kernel Signature:")
    for line in emitted["msl"].splitlines()[:6]:
        print(f"    {line}")

    print("\n  Sample PTX Kernel Header:")
    for line in emitted["ptx"].splitlines()[:6]:
        print(f"    {line}")

    # -----------------------------------------------------------------------
    # Step 2: Fused Rotary Position Embeddings (RoPE)
    # -----------------------------------------------------------------------
    print("\n[Step 2] Executing Fused Rotary Position Embeddings (RoPE)...")
    b, h, s, d = 2, 4, 16, 32
    q = ox.randn([b, h, s, d])
    # Frequencies cos and sin of shape [seq_len, head_dim / 2]
    freqs = np.linspace(0.1, 1.0, d // 2, dtype=np.float32)
    pos = np.arange(s, dtype=np.float32)[:, None]
    phase = pos * freqs[None, :]
    cos = from_numpy(np.cos(phase))
    sin = from_numpy(np.sin(phase))

    t0 = time.perf_counter()
    q_rotated = ox.fused_rope(q, cos, sin)
    t1 = time.perf_counter()

    print(f"  • Query tensor shape: {q.shape}")
    print(f"  • Rotated query shape: {q_rotated.shape}")
    print(f"  • RoPE execution latency: {(t1 - t0) * 1000.0:.3f} ms")
    assert q_rotated.shape == q.shape

    # -----------------------------------------------------------------------
    # Step 3: Fused Add + Pre-RMSNorm
    # -----------------------------------------------------------------------
    print("\n[Step 3] Executing Fused Add + Pre-RMSNorm Residual...")
    hidden_dim = 128
    x = ox.randn([4, hidden_dim])
    residual = ox.randn([4, hidden_dim])
    weight = ox.ones([hidden_dim])

    t0 = time.perf_counter()
    normed, updated_res = ox.fused_add_rmsnorm(x, residual, weight, eps=1e-5)
    t1 = time.perf_counter()

    print(f"  • Input shape: {x.shape}, Residual shape: {residual.shape}")
    print(f"  • Normalized output shape: {normed.shape}")
    print(f"  • Updated residual shape: {updated_res.shape}")
    print(f"  • Fused execution latency: {(t1 - t0) * 1000.0:.3f} ms")
    assert normed.shape == x.shape
    assert updated_res.shape == residual.shape

    # -----------------------------------------------------------------------
    # Step 4: Fused SwiGLU Gating
    # -----------------------------------------------------------------------
    print("\n[Step 4] Executing Fused SwiGLU Gating...")
    gate = ox.randn([4, 256])
    up = ox.randn([4, 256])

    t0 = time.perf_counter()
    swiglu_out = ox.fused_swiglu(gate, up)
    t1 = time.perf_counter()

    print(f"  • Gate: {gate.shape}, Up projection: {up.shape}")
    print(f"  • SwiGLU output shape: {swiglu_out.shape}")
    print(f"  • Latency: {(t1 - t0) * 1000.0:.3f} ms")
    assert swiglu_out.shape == gate.shape

    # -----------------------------------------------------------------------
    # Step 5: FlashAttention Tiled & Fused SDPA
    # -----------------------------------------------------------------------
    print("\n[Step 5] Executing Fused Scaled Dot-Product Attention (SDPA)...")
    k = ox.randn([b, h, s, d])
    v = ox.randn([b, h, s, d])

    t0 = time.perf_counter()
    attn_out = ox.fused_sdpa(q, k, v, is_causal=True)
    t1 = time.perf_counter()

    print(f"  • Q: {q.shape}, K: {k.shape}, V: {v.shape}")
    print(f"  • SDPA output shape: {attn_out.shape}")
    print(f"  • Latency: {(t1 - t0) * 1000.0:.3f} ms")
    assert attn_out.shape == q.shape

    # -----------------------------------------------------------------------
    # Step 6: MLX 4-bit Affine Quantized Matrix Multiplication
    # -----------------------------------------------------------------------
    print("\n[Step 6] Executing MLX 4-bit Affine Quantized GEMM...")
    x_gemm = ox.randn([2, 128])
    w_weight = ox.randn([256, 128])

    w_q4, scales, biases = ox.mlx_quantize_affine_q4(w_weight, group_size=32)

    t0 = time.perf_counter()
    q4_out = ox.mlx_quantized_matmul_q4(x_gemm, w_q4, scales, biases, group_size=32)
    t1 = time.perf_counter()

    dense_bytes = w_weight.to_numpy().nbytes
    q4_bytes = w_q4.to_numpy().nbytes + scales.to_numpy().nbytes + biases.to_numpy().nbytes
    print(f"  • Dense weight shape: {w_weight.shape} ({dense_bytes} bytes)")
    print(f"  • Packed Q4 shape: {w_q4.shape} ({q4_bytes} bytes with scales & biases)")
    print(f"  • Compression factor: {dense_bytes / q4_bytes:.2f}x reduction")
    print(f"  • Output shape: {q4_out.shape}")
    print(f"  • Q4 MatMul latency: {(t1 - t0) * 1000.0:.3f} ms")
    assert q4_out.shape == (2, 256)

    # -----------------------------------------------------------------------
    # Step 7: Paged KV Cache Serving Attention
    # -----------------------------------------------------------------------
    print("\n[Step 7] Executing Paged KV Cache Serving Attention...")
    q_decode = ox.randn([2, h, d])  # 2 active sequences, next token query
    num_blocks = 8
    block_size = 16
    k_cache = ox.randn([num_blocks, block_size, h, d])
    v_cache = ox.randn([num_blocks, block_size, h, d])
    block_tables = [[0, 1], [2, 3]]
    context_lens = [24, 30]

    t0 = time.perf_counter()
    paged_out = ox.paged_kv_attention(q_decode, k_cache, v_cache, block_tables, context_lens)
    t1 = time.perf_counter()

    print(f"  • Decode query: {q_decode.shape}")
    print(f"  • Allocated block tables: {block_tables} (context lengths: {context_lens})")
    print(f"  • Paged attention output: {paged_out.shape}")
    print(f"  • Decode step latency: {(t1 - t0) * 1000.0:.3f} ms")
    assert paged_out.shape == (2, h, d)

    banner("Unified Multi-Backend Hardware Kernel Showcase Completed! ✅")


if __name__ == "__main__":
    main()
