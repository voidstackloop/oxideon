#!/usr/bin/env python3
"""Scientific AI: Causal Discovery, Uncertainty Decomposition, and Survival Analysis.

Demonstrates:
1. Causal DAG Discovery with NOTEARS (matrix exponential acyclicity constraint)
2. Epistemic and Aleatoric Uncertainty Decomposition
3. Survival Analysis with Cox Proportional Hazards and Harrell's Concordance Index (C-Index)
4. Fast O(N^2) Decoupled Geometric Random Walk Graph Kernel
"""

import numpy as np
import oxideon as ox


def main():
    print("=" * 70)
    print("🔬 Oxideon Scientific AI: Causal Discovery & Uncertainty Modeling")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Continuous Causal DAG Discovery (NOTEARS)
    # ---------------------------------------------------------
    print("\n[Step 1] Causal Structure Discovery via NOTEARS...")
    np.random.seed(42)
    n_samples = 200
    d_vars = 6

    # Synthesize ground truth DAG: X0 -> X1 -> X2, X3 -> X4 -> X5
    W_true = np.zeros((d_vars, d_vars), dtype=np.float32)
    W_true[0, 1] = 1.2
    W_true[1, 2] = 0.8
    W_true[3, 4] = -1.1
    W_true[4, 5] = 1.5

    # Generate synthetic linear structural equation model (SEM)
    X = np.random.randn(n_samples, d_vars).astype(np.float32)
    X[:, 1] += X[:, 0] * W_true[0, 1]
    X[:, 2] += X[:, 1] * W_true[1, 2]
    X[:, 4] += X[:, 3] * W_true[3, 4]
    X[:, 5] += X[:, 4] * W_true[4, 5]

    X_tensor = ox.Tensor.from_numpy(X)
    print(f"  Observational data: {n_samples} samples across {d_vars} variables")

    # Run NOTEARS continuous optimization
    W_est = ox.notears_linear(X_tensor, lambda1=0.01, max_iter=60, w_threshold=0.1)
    W_est_np = W_est.to_numpy() if hasattr(W_est, "to_numpy") else np.asarray(W_est)

    num_discovered_edges = int(np.sum(np.abs(W_est_np) > 0.05))
    print(f"  Estimated weighted adjacency matrix shape: {W_est_np.shape}")
    print(f"  Discovered active causal edges: {num_discovered_edges}")
    print(f"  Recovered edge X0 -> X1 weight: {W_est_np[0, 1]:.3f} (ground truth: 1.20)")
    print(f"  Recovered edge X3 -> X4 weight: {W_est_np[3, 4]:.3f} (ground truth: -1.10)")

    # ---------------------------------------------------------
    # 2. Epistemic and Aleatoric Uncertainty Decomposition
    # ---------------------------------------------------------
    print("\n[Step 2] Deep Uncertainty Decomposition (Entropy / Aleatoric / Epistemic)...")
    n_ensemble = 5
    n_batch = 8
    n_classes = 4

    # Simulate predictive probability distributions across ensemble members
    raw_logits = np.random.randn(n_ensemble, n_batch, n_classes).astype(np.float32)
    exp_logits = np.exp(raw_logits - np.max(raw_logits, axis=-1, keepdims=True))
    ensemble_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    probs_tensor = ox.Tensor.from_numpy(ensemble_probs)
    total_entropy, aleatoric, epistemic = ox.decompose_uncertainty(probs_tensor)

    print(f"  Ensemble members: {n_ensemble}, Batch size: {n_batch}, Classes: {n_classes}")
    print(f"  Total predictive entropy: {total_entropy.to_numpy().mean():.4f}")
    print(f"  Aleatoric data uncertainty: {aleatoric.to_numpy().mean():.4f}")
    print(f"  Epistemic model uncertainty: {epistemic.to_numpy().mean():.4f}")

    # ---------------------------------------------------------
    # 3. Survival Analysis & Harrell's Concordance Index
    # ---------------------------------------------------------
    print("\n[Step 3] Survival Analysis & Concordance Index (C-Index)...")
    n_patients = 100
    # True underlying risk determines survival duration
    true_risk = np.random.uniform(0.1, 2.0, size=n_patients).astype(np.float32)
    durations = (100.0 / true_risk) + np.random.normal(0, 5, size=n_patients).astype(np.float32)
    durations = np.maximum(durations, 1.0)
    # Right censoring: 80% observed death events, 20% censored
    events = (np.random.rand(n_patients) > 0.2).astype(np.float32)

    risk_tensor = ox.Tensor.from_numpy(true_risk)
    durations_tensor = ox.Tensor.from_numpy(durations)
    events_tensor = ox.Tensor.from_numpy(events)

    # Negative of duration correlates with higher risk
    c_index = ox.concordance_index(risk_tensor, durations_tensor, events_tensor)
    print(f"  Evaluated cohort of {n_patients} patients (right-censoring rate: 20%)")
    print(f"  Harrell's Concordance Index: {c_index:.4f} (1.0 = perfect ranking)")

    # ---------------------------------------------------------
    # 4. Decoupled Geometric Random Walk Graph Kernel
    # ---------------------------------------------------------
    print("\n[Step 4] Geometric Random Walk Graph Kernel (O(N^2) Kronecker Decoupled)...")
    n_nodes = 12
    adj_a = (np.random.rand(n_nodes, n_nodes) > 0.6).astype(np.float32)
    adj_a = np.triu(adj_a, 1) + np.triu(adj_a, 1).T

    adj_b = (np.random.rand(n_nodes, n_nodes) > 0.5).astype(np.float32)
    adj_b = np.triu(adj_b, 1) + np.triu(adj_b, 1).T

    tensor_a = ox.Tensor.from_numpy(adj_a)
    tensor_b = ox.Tensor.from_numpy(adj_b)

    rw_kernel_val = ox.random_walk_kernel(tensor_a, tensor_b, num_steps=4, decay=0.05)
    print(f"  Computed topological similarity between two {n_nodes}-node networks")
    print(f"  Decoupled Random Walk Kernel value: {rw_kernel_val:.4f}")

    print("\n✅ Scientific AI & Causal Discovery Showcase Completed Successfully!")


if __name__ == "__main__":
    main()
