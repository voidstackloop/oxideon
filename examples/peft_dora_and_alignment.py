#!/usr/bin/env python3
"""Parameter-Efficient Fine-Tuning (DoRA/LoRA+) and Preference Alignment Showcase.

Demonstrates:
1. Weight-Decomposed Low-Rank Adaptation (DoRA): Directional + Magnitude decomposition
2. Asymmetric LoRA+ learning rate dynamics (eta_B / eta_A = 8.0)
3. Direct fusion (.merge()) of low-rank adapters into base model weights
4. Preference Optimization with ORPOLoss (monolithic SFT + Odds Ratio)
5. Kahneman-Tversky Optimization with KTOLoss on unpaired binary human feedback
"""

import numpy as np
import oxideon as ox
import oxideon.nn as nn
import oxideon.optim as optim


def main():
    print("=" * 70)
    print("🎯 Oxideon PEFT (DoRA / LoRA+) & Preference Alignment (ORPO / KTO)")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. DoRA: Weight-Decomposed Low-Rank Adaptation
    # ---------------------------------------------------------
    print("\n[Step 1] Initializing DoRALinear Adapter Layer...")
    in_dim = 128
    out_dim = 128
    rank = 8
    alpha = 16.0

    dora = ox.DoRALinear(in_features=in_dim, out_features=out_dim, r=rank, lora_alpha=alpha)
    print(f"  Base weights: [{out_dim}, {in_dim}]")
    print(f"  DoRA adapter rank: {rank}, scaling alpha: {alpha}")
    print(f"  Magnitude vector m: shape {list(dora.magnitude.shape)}")

    # Forward pass through DoRA
    batch_size = 16
    x = ox.Tensor.randn([batch_size, in_dim])
    out_dora = dora(x)
    print(f"  DoRA forward output shape: {out_dora.shape}")

    # Parameter count comparison
    base_params = in_dim * out_dim
    adapter_params = (in_dim * rank) + (rank * out_dim) + out_dim  # A + B + magnitude
    print(f"  Base parameter count:    {base_params:,}")
    print(f"  Trainable DoRA params:   {adapter_params:,} ({(adapter_params / base_params) * 100:.2f}% of base)")

    # Fuse DoRA into base weights for zero-overhead inference
    dora.merge()
    print("  Merged DoRA directional & magnitude adjustments directly into base weights (.merge())")

    # ---------------------------------------------------------
    # 2. LoRA+: Asymmetric Low-Rank Learning Rate Dynamics
    # ---------------------------------------------------------
    print("\n[Step 2] LoRAPlusLinear with Asymmetric Matrix Scaling...")
    lora_plus = ox.LoRAPlusLinear(in_features=in_dim, out_features=out_dim, r=rank, b_ratio=8.0)
    out_lora = lora_plus(x)
    print(f"  LoRA+ forward output shape: {out_lora.shape}")
    print("  Asymmetric ratio lambda = eta_B / eta_A = 8.0 enabled for balanced feature learning.")

    # ---------------------------------------------------------
    # 3. Preference Alignment with ORPOLoss
    # ---------------------------------------------------------
    print("\n[Step 3] Odds Ratio Preference Optimization (ORPOLoss)...")
    orpo = ox.ORPOLoss(lambda_coeff=0.1)

    # Simulated batch of chosen vs rejected completion log-probabilities
    chosen_logps = ox.Tensor.from_slice([-0.25, -0.15, -0.30, -0.10], [4])
    rejected_logps = ox.Tensor.from_slice([-1.80, -2.10, -1.45, -2.50], [4])
    sft_loss = ox.Tensor.from_slice([0.45], [1])

    loss_orpo, metrics_orpo = orpo(chosen_logps, rejected_logps, sft_loss=sft_loss)
    print(f"  ORPO Combined Loss:     {loss_orpo.item():.4f}")
    print(f"  Odds Ratio Loss term:   {metrics_orpo['odds_ratio_loss']:.4f}")
    print(f"  Log-Odds Ratio (w vs l): {metrics_orpo['log_odds_ratio']:.4f}")

    # ---------------------------------------------------------
    # 4. Kahneman-Tversky Optimization (KTOLoss)
    # ---------------------------------------------------------
    print("\n[Step 4] Kahneman-Tversky Optimization (KTOLoss) on Unpaired Feedback...")
    kto = ox.KTOLoss(beta=0.1, desirable_weight=1.0, undesirable_weight=1.33)

    policy_logps = ox.Tensor.from_slice([-0.20, -1.50, -0.35, -2.10], [4])
    ref_logps = ox.Tensor.from_slice([-0.50, -0.80, -0.60, -1.00], [4])
    labels = ox.Tensor.from_slice([1.0, 0.0, 1.0, 0.0], [4])  # 1 = desirable (upvoted), 0 = undesirable (downvoted)

    loss_kto, metrics_kto = kto(policy_logps, ref_logps, labels)
    print(f"  KTO Prospect Loss:      {loss_kto.item():.4f}")
    print(f"  Desirable Risk Penalty: {metrics_kto.get('desirable_loss', 0.0):.4f}")
    print(f"  Undesirable Loss:       {metrics_kto.get('undesirable_loss', 0.0):.4f}")

    print("\n✅ PEFT & Alignment Workflow Completed Successfully!")


if __name__ == "__main__":
    main()
