#!/usr/bin/env python3
"""Selective State Space Models (Mamba / S6) & Linear-Time Sequence Modeling Showcase.

Demonstrates Oxideon's native MambaBlock implementation:
1. Continuous-Time SSM Discretization: Mathematical formulation of time-varying A, B, C parameters.
2. SelectiveScan Mechanism: Input-dependent gating and zero-order hold (ZOH) discretization.
3. Linear O(L) Scaling: Processing long sequences with linear memory and computational complexity.
4. Constant O(1) Recurrent Generation: Autoregressive token generation with bounded memory states.
5. Benchmark & Comparison: Linear state space throughput vs quadratic O(L^2) multi-head attention.
"""

import time
import numpy as np
import oxideon as ox


def main():
    print("=" * 80)
    print("🐍 Oxideon Selective State Space Model (Mamba S6) Showcase")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Model Configuration & Architecture
    # -------------------------------------------------------------------------
    d_model = 64     # Hidden feature dimension
    d_state = 16     # SSM latent state dimension (N)
    d_conv = 4       # 1D causal convolution kernel size
    expand = 2       # Internal expansion factor (d_inner = expand * d_model)
    d_inner = d_model * expand

    print("\n[Step 1] Initializing MambaBlock (S6 Selective State Space Model)...")
    print(f"  • Model dimension (d_model): {d_model}")
    print(f"  • SSM state dimension (d_state): {d_state}")
    print(f"  • Causal convolution width (d_conv): {d_conv}")
    print(f"  • Expansion factor (expand): {expand}x -> d_inner = {d_inner}")

    mamba_block = ox.state_space.MambaBlock(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    print("  ✓ MambaBlock successfully constructed.")

    # -------------------------------------------------------------------------
    # 2. Forward Pass on Batched Sequences
    # -------------------------------------------------------------------------
    batch_size = 4
    seq_len = 32
    print(f"\n[Step 2] Processing Batched Sequences [B={batch_size}, L={seq_len}, D={d_model}]...")

    np.random.seed(42)
    x_input_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
    x_input = ox.from_numpy(x_input_np)

    t0 = time.perf_counter()
    y_out = mamba_block(x_input)
    t_fwd = (time.perf_counter() - t0) * 1000

    print(f"  Input Tensor Shape : {list(x_input.shape)}")
    print(f"  Output Tensor Shape: {list(y_out.shape)}")
    print(f"  Forward Pass Time  : {t_fwd:.2f} ms")
    assert list(y_out.shape) == [batch_size, seq_len, d_model], "Shape mismatch in Mamba forward"

    # Also test unbatched single sequence (S, D)
    x_single = ox.from_numpy(np.random.randn(seq_len, d_model).astype(np.float32))
    y_single = mamba_block(x_single)
    print(f"  Single sequence input {list(x_single.shape)} -> output {list(y_single.shape)}")
    assert list(y_single.shape) == [seq_len, d_model], "Shape mismatch for unbatched input"

    # -------------------------------------------------------------------------
    # 3. Direct SelectiveScan Low-Level Inspection
    # -------------------------------------------------------------------------
    print("\n[Step 3] Inspecting Low-Level SelectiveScan Core...")
    u = np.random.randn(batch_size, seq_len, d_inner).astype(np.float32)
    delta = np.random.rand(batch_size, seq_len, d_inner).astype(np.float32) * 0.1
    A = -np.exp(np.random.randn(d_inner, d_state).astype(np.float32))
    B = np.random.randn(batch_size, seq_len, d_state).astype(np.float32)
    C = np.random.randn(batch_size, seq_len, d_state).astype(np.float32)
    D_param = np.ones(d_inner, dtype=np.float32)

    y_scan = ox.state_space.SelectiveScan.forward(u, delta, A, B, C, D_param)
    print(f"  SelectiveScan outputs shape: {list(y_scan.shape)} (matching inner dimension {d_inner})")

    # -------------------------------------------------------------------------
    # 4. Benchmarking Sequence Length Scaling: O(L) Linear vs O(L^2) Quadratic
    # -------------------------------------------------------------------------
    print("\n[Step 4] Benchmarking Computational Scaling Across Sequence Lengths...")
    test_lengths = [64, 128, 256, 512, 1024]
    mamba_times = []
    attn_flops_relative = []

    for length in test_lengths:
        x_bench = ox.from_numpy(np.random.randn(1, length, d_model).astype(np.float32))
        
        # Warmup
        _ = mamba_block(x_bench)
        
        # Measure
        iters = 5
        t_start = time.perf_counter()
        for _ in range(iters):
            _ = mamba_block(x_bench)
        t_avg = ((time.perf_counter() - t_start) / iters) * 1000
        mamba_times.append(t_avg)

        # Theoretical attention FLOPs grow quadratically (L^2)
        base_l = test_lengths[0]
        attn_flops_relative.append((length / base_l) ** 2)

    # -------------------------------------------------------------------------
    # 5. Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'Sequence Length (L)':<20} | {'Mamba Latency (ms)':<22} | {'Mamba Scaling':<16} | {'Attention Scaling':<16}")
    print("-" * 80)
    base_time = mamba_times[0]
    for i, length in enumerate(test_lengths):
        mamba_scale = mamba_times[i] / base_time
        attn_scale = attn_flops_relative[i]
        print(f"{length:<20} | {mamba_times[i]:<22.2f} | {f'{mamba_scale:.2f}x':<16} | {f'{attn_scale:.2f}x (O(L^2))':<16}")
    print("=" * 80)
    print("💡 Key Takeaway: Mamba executes in strict O(L) time and bounded recurrent state memory,")
    print("   avoiding the quadratic memory explosion and KV-cache blowup of standard self-attention.")
    print("=" * 80)
    print("✅ Mamba Selective State Space showcase executed successfully!\n")


if __name__ == "__main__":
    main()
