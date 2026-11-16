#!/usr/bin/env python3
"""Unified Multi-Backend Hardware Kernel Matrix & Performance Benchmark.

Benchmarks and inspects Oxideon's unified hardware kernel engine across:
  1. Multi-Backend Kernel Assembly & Shader Inspection:
     - Apple Silicon Metal MSL (`MslKernelEmitter`)
     - NVIDIA CUDA PTX & CUDA C (`PtxKernelEmitter`)
     - WebGPU WGSL (`WgslKernelBuilder`)
     - Vulkan SPIR-V Binary & Disassembly (`VulkanSpirvKernel`)
  2. Native Fused Hardware Kernel Performance:
     - Fused FlashAttention-2 Tiled vs Standard Naive Attention
     - Paged KV Cache Attention (vLLM style dynamic memory pages)
     - MLX 4-bit Affine Quantized MatMul vs Dense float32 GEMM
     - Fused SwiGLU Gating vs Separate Elementwise Operations
     - Fused Add + RMSNorm vs Separate Residual and Normalization
  3. Formatted Performance Matrices (CLI tables & Markdown output)

Usage:
  python tools/hardware_kernel_matrix_bench.py [--markdown] [--warmup W] [--reps R]
"""

import argparse
import os
import sys
import time
import numpy as np

import oxideon as ox
from oxideon import Tensor, from_numpy
from oxideon.kernels import inspect_kernel


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"🚀  {title}")
    print("=" * 78)


def measure_ms(fn, warmup=3, reps=10) -> float:
    """Measures median execution latency in milliseconds."""
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    return float(np.median(latencies))


def benchmark_kernel_emitters() -> list:
    """Inspects emitted code across all 5 backends for each core operator."""
    ops = ["flash_attention", "rmsnorm", "swiglu", "rope", "q4_gemv", "matmul"]
    results = []

    for op in ops:
        emitted = inspect_kernel(op, head_dim=64, d_model=128, group_size=32)
        row = {
            "operator": op,
            "msl_lines": len(emitted["msl"].splitlines()),
            "ptx_lines": len(emitted["ptx"].splitlines()),
            "cuda_lines": len(emitted["cuda"].splitlines()),
            "wgsl_lines": len(emitted["wgsl"].splitlines()),
            "spirv_lines": len(emitted["spirv_disasm"].splitlines()),
        }
        results.append(row)
    return results


def benchmark_flash_attention() -> list:
    """Benchmarks Fused FlashAttention tiled vs standard attention across sequence lengths."""
    seq_lengths = [64, 128, 256, 512, 1024]
    b, h, d = 2, 4, 32
    results = []

    for s in seq_lengths:
        q = ox.randn([b, h, s, d])
        k = ox.randn([b, h, s, d])
        v = ox.randn([b, h, s, d])

        # Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
        scale = 1.0 / np.sqrt(d)
        def standard_attn():
            scores = q.matmul(k.transpose(-2, -1)) * scale
            probs = scores.softmax(-1)
            return probs.matmul(v)

        # Fused FlashAttention tiled
        def fused_attn():
            return ox.flash_attention_tiled(q, k, v, is_causal=True)

        std_ms = measure_ms(standard_attn, warmup=2, reps=5)
        fused_ms = measure_ms(fused_attn, warmup=2, reps=5)
        speedup = std_ms / max(fused_ms, 1e-6)

        results.append({
            "seq_len": s,
            "std_ms": std_ms,
            "fused_ms": fused_ms,
            "speedup": speedup,
        })
    return results


def benchmark_paged_attention() -> list:
    """Benchmarks Paged KV Cache Attention across batch sizes."""
    batches = [1, 2, 4, 8, 16]
    h, d = 4, 32
    block_size = 16
    results = []

    for b in batches:
        num_blocks = b * 4
        q = ox.randn([b, h, d])
        k_table = ox.randn([num_blocks, block_size, h, d])
        v_table = ox.randn([num_blocks, block_size, h, d])

        block_tables = [[i * 2, i * 2 + 1] for i in range(b)]
        context_lens = [28] * b

        def run_paged():
            return ox.paged_kv_attention(q, k_table, v_table, block_tables, context_lens)

        paged_ms = measure_ms(run_paged, warmup=3, reps=10)
        tok_per_sec = (b / (paged_ms / 1000.0))

        results.append({
            "batch_size": b,
            "paged_ms": paged_ms,
            "tokens_sec": tok_per_sec,
        })
    return results


def benchmark_mlx_q4_gemm() -> list:
    """Benchmarks 4-bit affine quantized GEMM vs dense float32 GEMM."""
    dims = [(128, 64), (256, 128), (512, 256), (1024, 512)]
    batch_size = 8
    results = []

    for m_out, k_in in dims:
        x = ox.randn([batch_size, k_in])
        w_dense = ox.randn([m_out, k_in])
        w_q4, scales, biases = ox.mlx_quantize_affine_q4(w_dense, group_size=32)

        def dense_gemm():
            return x.matmul(w_dense.transpose(0, 1))

        def q4_gemm():
            return ox.mlx_quantized_matmul_q4(x, w_q4, scales, biases, group_size=32)

        dense_ms = measure_ms(dense_gemm, warmup=3, reps=10)
        q4_ms = measure_ms(q4_gemm, warmup=3, reps=10)

        mem_dense = w_dense.to_numpy().nbytes
        mem_q4 = w_q4.to_numpy().nbytes + scales.to_numpy().nbytes + biases.to_numpy().nbytes
        comp_ratio = mem_dense / max(mem_q4, 1)

        results.append({
            "dims": f"[{m_out}x{k_in}]",
            "dense_ms": dense_ms,
            "q4_ms": q4_ms,
            "comp_ratio": comp_ratio,
            "speedup": dense_ms / max(q4_ms, 1e-6),
        })
    return results


def benchmark_fused_activations() -> dict:
    """Benchmarks fused activations and normalization vs separate operations."""
    b, d = 32, 256
    x = ox.randn([b, d])
    res = ox.randn([b, d])
    w = ox.ones([d])

    # RMSNorm
    def sep_rmsnorm():
        added = x + res
        rms = (added * added).mean(dim=-1, keepdim=True)
        return (added / (rms + 1e-5).sqrt()) * w

    def fused_rmsnorm():
        return ox.fused_add_rmsnorm(x, res, w)

    # SwiGLU
    gate = ox.randn([b, d * 2])
    up = ox.randn([b, d * 2])

    def sep_swiglu():
        return gate.silu() * up

    def fused_swiglu():
        return ox.fused_swiglu(gate, up)

    return {
        "rmsnorm_sep_ms": measure_ms(sep_rmsnorm, warmup=5, reps=20),
        "rmsnorm_fused_ms": measure_ms(fused_rmsnorm, warmup=5, reps=20),
        "swiglu_sep_ms": measure_ms(sep_swiglu, warmup=5, reps=20),
        "swiglu_fused_ms": measure_ms(fused_swiglu, warmup=5, reps=20),
    }


def main():
    parser = argparse.ArgumentParser(description="Oxideon Unified Hardware Kernel Matrix Benchmark")
    parser.add_argument("--markdown", action="store_true", help="Generate Markdown formatted tables")
    parser.add_argument("--out", type=str, default="docs/HARDWARE_BENCHMARK_REPORT.md", help="Path to write report")
    args = parser.parse_args()

    ox.manual_seed(42)
    np.random.seed(42)

    banner("Oxideon Unified Multi-Backend Hardware Kernel Matrix Benchmark")

    print("\n🔍 Phase 1: Inspecting Hardware Kernel Emitters across 5 Execution Targets...")
    emit_res = benchmark_kernel_emitters()

    print("\n  Operator        | MSL (Metal) | PTX (NVIDIA) | CUDA C | WGSL (WebGPU) | SPIR-V (Vulkan)")
    print("  ----------------|-------------|--------------|--------|---------------|----------------")
    for r in emit_res:
        print(f"  {r['operator']:15s} | {r['msl_lines']:7d} loc | {r['ptx_lines']:8d} loc | {r['cuda_lines']:6d} | {r['wgsl_lines']:9d} loc | {r['spirv_lines']:11d} loc")

    print("\n⚡ Phase 2: Benchmarking FlashAttention-2 Tiled vs Naive Self-Attention...")
    fa_res = benchmark_flash_attention()
    print("\n  Sequence Length | Naive Attention (ms) | Fused FlashAttn (ms) | Speedup Factor")
    print("  ----------------|----------------------|----------------------|---------------")
    for r in fa_res:
        print(f"    S = {r['seq_len']:5d}     | {r['std_ms']:20.3f} | {r['fused_ms']:20.3f} | {r['speedup']:13.2f}x")

    print("\n📦 Phase 3: Benchmarking Paged KV Cache Serving Attention...")
    paged_res = benchmark_paged_attention()
    print("\n  Batch Size | Paged Step Latency (ms) | Effective Throughput (tok/sec)")
    print("  -----------|-------------------------|--------------------------------")
    for r in paged_res:
        print(f"    B = {r['batch_size']:4d}   | {r['paged_ms']:23.3f} | {r['tokens_sec']:28.1f}")

    print("\n🍏 Phase 4: Benchmarking MLX 4-bit Affine Quantized MatMul vs F32 GEMM...")
    q4_res = benchmark_mlx_q4_gemm()
    print("\n  Weight Matrix | Dense F32 (ms) | MLX Q4 GEMM (ms) | Compression | Relative Speed")
    print("  --------------|----------------|------------------|-------------|---------------")
    for r in q4_res:
        print(f"  {r['dims']:13s} | {r['dense_ms']:14.3f} | {r['q4_ms']:16.3f} | {r['comp_ratio']:9.2f}x | {r['speedup']:13.2f}x")

    print("\n🔥 Phase 5: Benchmarking Fused Normalization & Gating Operators...")
    fused_res = benchmark_fused_activations()
    rms_speedup = fused_res["rmsnorm_sep_ms"] / max(fused_res["rmsnorm_fused_ms"], 1e-6)
    swiglu_speedup = fused_res["swiglu_sep_ms"] / max(fused_res["swiglu_fused_ms"], 1e-6)

    print(f"  • Fused Add + RMSNorm : {fused_res['rmsnorm_fused_ms']:.3f} ms vs {fused_res['rmsnorm_sep_ms']:.3f} ms (Speedup: {rms_speedup:.2f}x)")
    print(f"  • Fused SwiGLU Gating : {fused_res['swiglu_fused_ms']:.3f} ms vs {fused_res['swiglu_sep_ms']:.3f} ms (Speedup: {swiglu_speedup:.2f}x)")

    banner("Hardware Kernel Matrix Benchmark Completed Successfully! ✅")

    # Generate Markdown Report if requested
    if args.markdown:
        md = f"""# Oxideon Unified Multi-Backend Hardware Kernel Benchmark Report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  
Oxideon Version: `0.1.0`  
Hardware Target: Cross-Platform Unified Runtime (Metal MSL, NVIDIA PTX, WebGPU WGSL, Vulkan SPIR-V)

---

## 1. Hardware Kernel Emission Matrix

| Operator | Apple Metal (MSL) | NVIDIA PTX | CUDA C | WebGPU (WGSL) | Vulkan (SPIR-V) |
|:---------|:------------------|:-----------|:-------|:--------------|:----------------|
"""
        for r in emit_res:
            md += f"| `{r['operator']}` | {r['msl_lines']} lines | {r['ptx_lines']} lines | {r['cuda_lines']} lines | {r['wgsl_lines']} lines | {r['spirv_lines']} lines |\n"

        md += """
---

## 2. FlashAttention-2 Tiled vs Naive Attention

| Sequence Length ($S$) | Naive Attention Latency | Fused FlashAttn Latency | Speedup |
|:----------------------|:------------------------|:------------------------|:--------|
"""
        for r in fa_res:
            md += f"| $S = {r['seq_len']}$ | {r['std_ms']:.3f} ms | {r['fused_ms']:.3f} ms | **{r['speedup']:.2f}x** |\n"

        md += """
---

## 3. Paged KV Cache Dynamic Serving Attention

| Concurrency (Batch Size) | Per-Token Decode Latency | Generation Throughput |
|:-------------------------|:-------------------------|:----------------------|
"""
        for r in paged_res:
            md += f"| $B = {r['batch_size']}$ | {r['paged_ms']:.3f} ms | **{r['tokens_sec']:.1f} tokens/sec** |\n"

        md += """
---

## 4. MLX 4-bit Affine Quantized GEMM vs Float32

| Weight Matrix $[M \\times K]$ | Dense F32 Latency | MLX Q4 GEMM Latency | Weight Compression | Relative Speed |
|:------------------------------|:------------------|:--------------------|:-------------------|:---------------|
"""
        for r in q4_res:
            md += f"| `{r['dims']}` | {r['dense_ms']:.3f} ms | {r['q4_ms']:.3f} ms | **{r['comp_ratio']:.2f}x** | **{r['speedup']:.2f}x** |\n"

        md += f"""
---

## 5. Fused Memory-Bound Operators

- **Fused Add + RMSNorm**: `{fused_res['rmsnorm_fused_ms']:.3f} ms` vs `{fused_res['rmsnorm_sep_ms']:.3f} ms` (**{rms_speedup:.2f}x speedup**)
- **Fused SwiGLU Gating**: `{fused_res['swiglu_fused_ms']:.3f} ms` vs `{fused_res['swiglu_sep_ms']:.3f} ms` (**{swiglu_speedup:.2f}x speedup**)
"""
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(md)
        print(f"\n📄 Markdown benchmark report written to '{args.out}'.")


if __name__ == "__main__":
    main()
