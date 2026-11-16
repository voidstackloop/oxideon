#!/usr/bin/env python3
"""Geometric Deep Learning, Equivariant Molecular GNNs, and Graph Matching.

Demonstrates modern structural biology and geometric graph architectures in Oxideon:
  1. Amino acid sequence one-hot encoding & 3D C-alpha residue contact map computation
  2. E(3)-Equivariant Graph Neural Network (EGNN) coordinate and feature updates
  3. Kabsch RMSD optimal structural superposition under SE(3) spatial transformations
  4. Gromov-Wasserstein Optimal Transport for topological graph matching
  5. GraphSAGE neighborhood aggregation with autograd node classification

Usage:
  python examples/graph_neural_and_molecular.py
"""

import time
import numpy as np

import oxideon as ox
from oxideon import Tensor, from_numpy
from oxideon.protein import (
    amino_acid_to_onehot,
    compute_contact_map,
    kabsch_rmsd,
    EGNNCoordinateUpdate,
)
from oxideon.graph_matching import gromov_wasserstein_distance
from oxideon.gnn_ext import GraphSAGE, GINConv
from oxideon.optim import AdamW


def banner(title: str) -> None:
    print("\n" + "=" * 76)
    print(f"🧬 {title}")
    print("=" * 76)


def main():
    ox.manual_seed(42)
    np.random.seed(42)

    banner("Oxideon Geometric Deep Learning, Equivariant GNNs & Graph Matching")

    # -----------------------------------------------------------------------
    # Step 1: Structural Biology & Residue Contact Map
    # -----------------------------------------------------------------------
    print("\n[Step 1] Protein Sequence Encoding & C-Alpha Contact Map...")
    seq = "MKWVTFISLLLLFSSAYSRG"
    encoded_seq = amino_acid_to_onehot(seq)
    print(f"  • Amino acid sequence: '{seq}' (length = {len(seq)})")
    print(f"  • One-hot residue feature tensor: {encoded_seq.shape} (20 standard AAs)")

    # Simulated 3D C-alpha backbone coordinates in Angstroms
    num_residues = len(seq)
    t = np.linspace(0, 4.0 * np.pi, num_residues)
    coords_np = np.stack([np.cos(t) * 5.0, np.sin(t) * 5.0, np.linspace(0, 15.0, num_residues)], axis=1).astype(np.float32)
    ca_coords = from_numpy(coords_np)

    contact_threshold = 8.0  # Angstroms
    contact_map = compute_contact_map(ca_coords, threshold_angstrom=contact_threshold)
    contact_density = float(np.mean(contact_map.to_numpy())) * 100.0

    print(f"  • 3D backbone coordinates: {ca_coords.shape}")
    print(f"  • Contact map matrix shape: {contact_map.shape} (threshold = {contact_threshold} Å)")
    print(f"  • Residue contact density: {contact_density:.1f}%")

    # -----------------------------------------------------------------------
    # Step 2: SE(3)-Invariant Kabsch RMSD Superposition
    # -----------------------------------------------------------------------
    print("\n[Step 2] SE(3) Rigid Superposition & Kabsch RMSD Alignment...")
    # Rotate and translate target coordinates
    theta = np.pi / 4.0
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ], dtype=np.float32)
    shift = np.array([12.0, -5.0, 7.5], dtype=np.float32)

    rotated_np = (coords_np @ rot_matrix.T) + shift + np.random.randn(*coords_np.shape).astype(np.float32) * 0.05
    rotated_coords = from_numpy(rotated_np)

    unaligned_rmsd = float(np.sqrt(np.mean(np.sum((coords_np - rotated_np) ** 2, axis=-1))))
    aligned_rmsd = kabsch_rmsd(ca_coords, rotated_coords)

    print(f"  • Unaligned Euclidean RMSD: {unaligned_rmsd:.3f} Å (due to rotation & translation)")
    print(f"  • Kabsch optimally aligned RMSD: {aligned_rmsd:.4f} Å")
    assert aligned_rmsd < 0.2, "Expected near-zero RMSD after Kabsch superposition"

    # -----------------------------------------------------------------------
    # Step 3: E(n)-Equivariant Graph Neural Network (EGNN)
    # -----------------------------------------------------------------------
    print("\n[Step 3] E(3)-Equivariant Coordinate & Feature Updates (EGNN)...")
    node_dim = 16
    hidden_dim = 32
    egnn = EGNNCoordinateUpdate(node_dim=node_dim, hidden_dim=hidden_dim)

    h_features = ox.randn([num_residues, node_dim])
    t0 = time.perf_counter()
    h_next, coords_next = egnn(h_features, ca_coords)
    t1 = time.perf_counter()

    coord_delta = float(np.mean(np.abs(coords_next.to_numpy() - ca_coords.to_numpy())))
    print(f"  • Input features: {h_features.shape}, Coordinates: {ca_coords.shape}")
    print(f"  • Updated features: {h_next.shape}, Updated Coordinates: {coords_next.shape}")
    print(f"  • Mean coordinate displacement: {coord_delta:.4f} Å")
    print(f"  • EGNN layer latency: {(t1 - t0) * 1000.0:.2f} ms")
    assert h_next.shape == h_features.shape
    assert coords_next.shape == ca_coords.shape

    # -----------------------------------------------------------------------
    # Step 4: Gromov-Wasserstein Graph Matching
    # -----------------------------------------------------------------------
    print("\n[Step 4] Entropic Gromov-Wasserstein Topological Graph Matching...")
    # Generate two graph distance cost matrices
    cost1 = from_numpy(np.abs(coords_np[:8, None] - coords_np[None, :8]).sum(axis=-1))
    cost2 = from_numpy(np.abs(rotated_np[:8, None] - rotated_np[None, :8]).sum(axis=-1))

    t0 = time.perf_counter()
    gw_dist, plan = gromov_wasserstein_distance(cost1, cost2, num_iters=20, epsilon=0.05)
    t1 = time.perf_counter()

    gw_val = float(gw_dist.to_numpy().flat[0])
    plan_shape = plan.shape
    print(f"  • Graph 1 cost shape: {cost1.shape}, Graph 2 cost shape: {cost2.shape}")
    print(f"  • Gromov-Wasserstein distance: {gw_val:.5f}")
    print(f"  • Optimal coupling plan shape: {plan_shape}")
    print(f"  • Matching latency: {(t1 - t0) * 1000.0:.2f} ms")

    # -----------------------------------------------------------------------
    # Step 5: Inductive Graph Representation Learning (GraphSAGE)
    # -----------------------------------------------------------------------
    print("\n[Step 5] GraphSAGE Inductive Representation Training...")
    # Use contact map as adjacency matrix
    adj = contact_map
    sage = GraphSAGE(in_channels=20, out_channels=4, aggr="mean")
    optimizer = AdamW(sage.parameters(), lr=1e-2)

    # Simulated targets for 4-class functional residue annotation
    targets_np = (np.arange(num_residues) % 4).astype(np.int64)
    targets = from_numpy(targets_np)

    init_loss = 0.0
    final_loss = 0.0

    print("  Epoch | GraphSAGE Cross-Entropy | Loss Reduction")
    print("  ------|-------------------------|---------------")

    for epoch in range(1, 9):
        optimizer.zero_grad()
        logits = sage(encoded_seq, adj)
        loss = ox.functional.cross_entropy(logits, targets)

        loss_val = float(loss.to_numpy().flat[0])
        if epoch == 1:
            init_loss = loss_val
        final_loss = loss_val

        loss.backward()
        optimizer.step()

        reduction = (1.0 - loss_val / init_loss) * 100.0 if init_loss > 0 else 0.0
        print(f"  {epoch:5d} | {loss_val:23.4f} | {reduction:12.1f}%")

    print(f"\n  ✓ GraphSAGE node classification loss converged from {init_loss:.4f} down to {final_loss:.4f}.")
    assert final_loss < init_loss, "Expected classification loss to decrease"

    banner("Geometric Deep Learning & Molecular GNN Showcase Completed Successfully! ✅")


if __name__ == "__main__":
    main()
