#!/usr/bin/env python3
"""DeepSeek-style Sparse Mixture of Experts (MoE) Architecture Showcase.

Demonstrates modern sparse MoE routing and capacity constraints in Oxideon:
  1. Top-k sparse gating with auxiliary load balancing loss (preventing expert collapse)
  2. Router z-loss for numerical stability in float32 logits
  3. Capacity-constrained token dispatch (Switch Transformer / DeepSeek style)
  4. Token routing histogram and expert utilization metrics
  5. End-to-end forward pass and autograd backward gradient flow through routed experts

Usage:
  python examples/mixture_of_experts_deepseek.py
"""

import time
import numpy as np

import oxideon as ox
from oxideon import Tensor, from_numpy
from oxideon.mixture_of_experts import MoELayer, TopKGating
from oxideon.expert_parallel import CapacityConstrainedMoE, TopKRouterWithAuxLoss
from oxideon.optim import AdamW


def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"⚡ {title}")
    print("=" * 76)


def print_routing_table(num_experts: int, topk_indices: np.ndarray, top_k: int):
    """Prints token dispatch statistics across available experts."""
    num_tokens = len(topk_indices)
    counts = np.zeros(num_experts, dtype=np.int64)
    for row in topk_indices:
        for idx in row:
            counts[idx] += 1

    total_assignments = num_tokens * top_k
    ideal_count = total_assignments / num_experts

    print("\n  Expert | Dispatched Tokens | Load Ratio | Distribution Bar")
    print("  -------|-------------------|------------|-------------------------")
    for e in range(num_experts):
        ratio = counts[e] / max(ideal_count, 1e-5)
        bar_len = int(min(25, ratio * 12))
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"    E{e:<3d} | {counts[e]:17d} | {ratio:9.2f}x | {bar}")


def main():
    ox.manual_seed(1337)
    np.random.seed(1337)

    banner("Oxideon DeepSeek / Switch Sparse Mixture of Experts (MoE)")

    num_tokens = 64
    d_model = 64
    hidden_dim = 128
    num_experts = 8
    top_k = 2

    print(f"Configurations:")
    print(f"  • Token count: {num_tokens}")
    print(f"  • Model hidden dim: {d_model}")
    print(f"  • Expert intermediate dim: {hidden_dim}")
    print(f"  • Total experts: {num_experts}")
    print(f"  • Active experts per token (top-k): {top_k}")
    print(f"  • Theoretical FLOP reduction: {num_experts / top_k:.1f}x vs dense MLP")

    x = ox.randn([num_tokens, d_model])

    # -----------------------------------------------------------------------
    # Step 1: Top-K Gating Router & Load Balancing
    # -----------------------------------------------------------------------
    print("\n[Step 1] Initializing Top-K Router with Exploration Noise...")
    router = TopKGating(d_model=d_model, num_experts=num_experts, top_k=top_k, noisy_gating=True)

    t0 = time.perf_counter()
    indices, weights, aux_loss, z_loss = router(x)
    t1 = time.perf_counter()

    print(f"  • Routing latency: {(t1 - t0) * 1000.0:.2f} ms")
    print(f"  • Assigned indices shape: {indices.shape}")
    print(f"  • Normalized weights shape: {weights.shape}")
    print(f"  • Load balancing auxiliary loss: {aux_loss:.4f} (ideal = 1.00)")
    print(f"  • Router z-loss (stabilization): {z_loss:.6f}")

    print_routing_table(num_experts, indices, top_k)

    # -----------------------------------------------------------------------
    # Step 2: Full Sparse MoE Layer Forward Pass
    # -----------------------------------------------------------------------
    print("\n[Step 2] Executing Sparse MoELayer Forward Pass...")
    moe_layer = MoELayer(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
    )

    t0 = time.perf_counter()
    output, moe_aux, moe_z = moe_layer(x)
    t1 = time.perf_counter()

    print(f"  • Output tensor shape: {output.shape} (matching input: {x.shape})")
    print(f"  • MoE execution time: {(t1 - t0) * 1000.0:.2f} ms")
    assert output.shape == x.shape

    # -----------------------------------------------------------------------
    # Step 3: Capacity-Constrained Routing with Token Dropping
    # -----------------------------------------------------------------------
    print("\n[Step 3] Evaluating Capacity-Constrained Routing (DeepSeek / Switch)...")
    capacity_factor = 1.25
    cc_moe = CapacityConstrainedMoE(
        hidden_dim=d_model,
        expert_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
    )

    expected_capacity = int(np.ceil((num_tokens * top_k / num_experts) * capacity_factor))
    print(f"  • Capacity factor: {capacity_factor:.2f}")
    print(f"  • Max token buffer per expert: {expected_capacity} tokens")

    cc_out, cc_dropped = cc_moe(x)
    print(f"  • Output shape: {cc_out.shape}")
    print(f"  • Dropped / unhandled tokens: {cc_dropped} (0 = zero token loss)")
    assert cc_out.shape == x.shape

    # -----------------------------------------------------------------------
    # Step 4: Training & Autograd Backward Pass
    # -----------------------------------------------------------------------
    print("\n[Step 4] End-to-End Autograd Training of Routed Experts...")
    target = ox.randn([num_tokens, d_model])
    optimizer = AdamW(moe_layer.parameters(), lr=1e-3, weight_decay=1e-4)

    initial_loss = 0.0
    final_loss = 0.0

    print("  Epoch | Task MSE Loss | Aux Loss | Total Objective")
    print("  ------|---------------|----------|----------------")

    for epoch in range(1, 9):
        optimizer.zero_grad()
        pred, aux_l, z_l = moe_layer(x)
        diff = pred - target
        mse = (diff * diff).mean()
        total_obj = mse + 0.01 * aux_l

        curr_loss = float(mse.to_numpy().flat[0])
        if epoch == 1:
            initial_loss = curr_loss
        final_loss = curr_loss

        total_obj.backward()
        optimizer.step()

        print(f"  {epoch:5d} | {curr_loss:13.4f} | {aux_l:8.4f} | {float(total_obj.to_numpy().flat[0]):15.4f}")

    print(f"\n  ✓ Training successfully updated routed expert weights.")
    print(f"  ✓ Loss decreased from {initial_loss:.4f} to {final_loss:.4f}.")
    assert final_loss < initial_loss, "Expected training loss to decrease"

    banner("DeepSeek Sparse MoE Showcase Completed Successfully! ✅")


if __name__ == "__main__":
    main()
