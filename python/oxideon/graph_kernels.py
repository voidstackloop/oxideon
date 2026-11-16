"""Graph kernels for structured graph classification and topological similarity.

Implements:
  - weisfeiler_lehman_kernel: 1-WL Subtree Isomorphism Kernel (Shervashidze et al., 2011)
  - random_walk_kernel: Geometric Random Walk Kernel on the direct Kronecker product graph
  - shortest_path_kernel: All-pairs Shortest-Path Graph Kernel (Borgwardt & Kriegel, 2005)
"""

from typing import List, Optional, Tuple, Dict
import hashlib
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _normalize_gram(gram: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.maximum(np.diag(gram), 1e-12))
    denom = np.outer(diag, diag)
    return gram / denom


def weisfeiler_lehman_kernel(
    adj_matrices: List[_Tensor],
    node_labels: Optional[List[List[int]]] = None,
    num_iterations: int = 3,
    normalize: bool = False,
) -> _Tensor:
    """Computes the Weisfeiler-Lehman subtree kernel Gram matrix over a list of graphs.

    Args:
        adj_matrices: List of adjacency matrix Tensors [N_i, N_i]
        node_labels: Optional list of initial integer node labels for each graph
        num_iterations: Number of color refinement iterations h
        normalize: If True, cosine-normalizes the Gram matrix so diagonal is 1.0

    Returns:
        Gram matrix Tensor of shape [num_graphs, num_graphs]
    """
    num_graphs = len(adj_matrices)
    if num_graphs == 0:
        return _from_numpy(np.zeros((0, 0), dtype=np.float32))

    adjs = [a.to_numpy() for a in adj_matrices]

    # Initialize node labels if not provided
    if node_labels is None:
        curr_labels = [[f"deg_{int(np.sum(np.abs(adj[i]) > 0))}" for i in range(adj.shape[0])] for adj in adjs]
    else:
        curr_labels = [[str(l) for l in g_lbls] for g_lbls in node_labels]

    # Global multiset feature tracker: graph_idx -> dict of label -> count
    histograms: List[Dict[str, int]] = [{} for _ in range(num_graphs)]

    # Initial labels
    for g_idx in range(num_graphs):
        for lbl in curr_labels[g_idx]:
            histograms[g_idx][lbl] = histograms[g_idx].get(lbl, 0) + 1

    # WL color refinement iterations
    for _ in range(num_iterations):
        new_labels: List[List[str]] = []
        for g_idx in range(num_graphs):
            adj = adjs[g_idx]
            n_nodes = adj.shape[0]
            g_new_lbls = []

            for v in range(n_nodes):
                neighbors = np.where(np.abs(adj[v]) > 0)[0]
                neigh_labels = sorted([curr_labels[g_idx][u] for u in neighbors])
                combined = f"{curr_labels[g_idx][v]}_" + "_".join(neigh_labels)
                label_hash = hashlib.md5(combined.encode()).hexdigest()[:8]
                g_new_lbls.append(label_hash)

                histograms[g_idx][label_hash] = histograms[g_idx].get(label_hash, 0) + 1

            new_labels.append(g_new_lbls)
        curr_labels = new_labels

    # Build unique vocabulary of all subtree features
    all_features = sorted(list({k for h in histograms for k in h.keys()}))
    feat_map = {feat: idx for idx, feat in enumerate(all_features)}

    # Construct feature vectors [num_graphs, num_features]
    phi = np.zeros((num_graphs, len(all_features)), dtype=np.float32)
    for g_idx, h in enumerate(histograms):
        for feat, count in h.items():
            phi[g_idx, feat_map[feat]] = count

    # Gram matrix: K = Phi @ Phi^T
    gram = phi @ phi.T
    if normalize and num_graphs > 0:
        gram = _normalize_gram(gram)
    return _from_numpy(np.ascontiguousarray(gram.astype(np.float32)))


def shortest_path_kernel(
    adj_matrices: List[_Tensor],
    normalize: bool = False,
) -> _Tensor:
    """Computes the all-pairs Shortest-Path Graph Kernel (Borgwardt & Kriegel, 2005).

    Uses Floyd-Warshall all-pairs shortest path distances and builds a histogram
    of finite path lengths for each graph.
    """
    num_graphs = len(adj_matrices)
    if num_graphs == 0:
        return _from_numpy(np.zeros((0, 0), dtype=np.float32))

    histograms: List[Dict[int, int]] = [{} for _ in range(num_graphs)]
    for g_idx, adj_t in enumerate(adj_matrices):
        adj = adj_t.to_numpy()
        n = adj.shape[0]
        dist = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(dist, 0.0)
        edges = np.abs(adj) > 0
        dist[edges] = 1.0
        for k in range(n):
            dist = np.minimum(dist, dist[:, k : k + 1] + dist[k : k + 1, :])
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                if np.isfinite(d):
                    d_int = int(round(d))
                    histograms[g_idx][d_int] = histograms[g_idx].get(d_int, 0) + 1

    all_lengths = sorted(list({k for h in histograms for k in h.keys()}))
    if not all_lengths:
        return _from_numpy(np.zeros((num_graphs, num_graphs), dtype=np.float32))
    feat_map = {length: idx for idx, length in enumerate(all_lengths)}
    phi = np.zeros((num_graphs, len(all_lengths)), dtype=np.float32)
    for g_idx, h in enumerate(histograms):
        for length, count in h.items():
            phi[g_idx, feat_map[length]] = count

    gram = phi @ phi.T
    if normalize and num_graphs > 0:
        gram = _normalize_gram(gram)
    return _from_numpy(np.ascontiguousarray(gram.astype(np.float32)))


def random_walk_kernel(
    adj1: _Tensor,
    adj2: _Tensor,
    num_steps: int = 4,
    decay: float = 0.1,
) -> float:
    """Computes the direct product graph Random Walk kernel between two graphs.

    $$K(G_1, G_2) = \\sum_{k=0}^{S} \\lambda^k \\mathbf{1}^T (A_1 \\otimes A_2)^k \\mathbf{1}$$

    Optimized via algebraic Kronecker decoupling:
    $$\\mathbf{1}^T (A_1 \\otimes A_2)^k \\mathbf{1} = (\\mathbf{1}^T A_1^k \\mathbf{1}) \\cdot (\\mathbf{1}^T A_2^k \\mathbf{1})$$
    reducing computational complexity from $O((N_1 N_2)^3)$ to $O(S(N_1^2 + N_2^2))$.
    """
    if num_steps < 0:
        raise ValueError(f"num_steps must be >= 0, got {num_steps}")
    a1 = adj1.to_numpy().astype(np.float64)
    a2 = adj2.to_numpy().astype(np.float64)
    if a1.ndim != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError(f"adj1 must be a square adjacency matrix, got shape {a1.shape}")
    if a2.ndim != 2 or a2.shape[0] != a2.shape[1]:
        raise ValueError(f"adj2 must be a square adjacency matrix, got shape {a2.shape}")

    n1 = a1.shape[0]
    n2 = a2.shape[0]

    # k = 0 term: sum(1_{n1 * n2}) = n1 * n2
    k_val = float(n1 * n2)

    # Iterative vector propagation: v^{(k)} = A v^{(k-1)} starting from 1
    v1 = np.ones((n1, 1), dtype=np.float64)
    v2 = np.ones((n2, 1), dtype=np.float64)
    weight = decay

    for _ in range(1, num_steps + 1):
        v1 = a1 @ v1
        v2 = a2 @ v2
        step_val = float(np.sum(v1) * np.sum(v2))
        k_val += weight * step_val
        weight *= decay

    return float(k_val)


__all__ = [
    "weisfeiler_lehman_kernel",
    "shortest_path_kernel",
    "random_walk_kernel",
]
