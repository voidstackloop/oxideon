"""
Graph generation algorithms and synthetic network topology generation for Oxideon.
Supports Erdos-Renyi, Barabasi-Albert (preferential attachment), Watts-Strogatz (small-world),
Stochastic Block Models (SBM), and Random Geometric Graphs (RGG).
"""
from typing import Optional, List, Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy
from .gnn import GraphData


def erdos_renyi_graph(num_nodes: int, p: float, directed: bool = False, seed: Optional[int] = None) -> Tuple[Tensor, GraphData]:
    """Generates a random Erdos-Renyi G(n, p) graph."""
    rng = np.random.default_rng(seed)
    if directed:
        adj = (rng.uniform(0.0, 1.0, size=(num_nodes, num_nodes)) < p).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
    else:
        tri = rng.uniform(0.0, 1.0, size=(num_nodes, num_nodes)) < p
        tri = np.triu(tri, k=1).astype(np.float32)
        adj = tri + tri.T

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    x = np.eye(num_nodes, dtype=np.float32)

    return _from_numpy(adj), GraphData(x=_from_numpy(x), edge_index=_from_numpy(edge_index))


def barabasi_albert_graph(num_nodes: int, m: int, seed: Optional[int] = None) -> Tuple[Tensor, GraphData]:
    """Generates a scale-free network using the Barabasi-Albert preferential attachment model."""
    if m < 1 or m >= num_nodes:
        raise ValueError(f"m must be between 1 and num_nodes - 1 (got m={m}, num_nodes={num_nodes})")

    rng = np.random.default_rng(seed)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    # Start with a small clique of m + 1 nodes
    init_nodes = m + 1
    for i in range(init_nodes):
        for j in range(i + 1, init_nodes):
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    degrees = np.sum(adj, axis=1)

    for new_node in range(init_nodes, num_nodes):
        existing_nodes = np.arange(new_node)
        total_deg = np.sum(degrees[:new_node])
        probs = degrees[:new_node] / (total_deg if total_deg > 0 else 1.0)
        chosen = rng.choice(existing_nodes, size=m, replace=False, p=probs)
        for target in chosen:
            adj[new_node, target] = 1.0
            adj[target, new_node] = 1.0
            degrees[new_node] += 1.0
            degrees[target] += 1.0

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    x = np.eye(num_nodes, dtype=np.float32)

    return _from_numpy(adj), GraphData(x=_from_numpy(x), edge_index=_from_numpy(edge_index))


def watts_strogatz_graph(num_nodes: int, k: int, p: float, seed: Optional[int] = None) -> Tuple[Tensor, GraphData]:
    """Generates a Watts-Strogatz small-world graph with mean degree k and rewiring probability p."""
    if num_nodes < 3:
        raise ValueError(f"num_nodes must be >= 3, got {num_nodes}")
    if k < 2 or k % 2 != 0:
        raise ValueError(f"k must be a positive even integer, got {k}")
    if k >= num_nodes:
        raise ValueError(f"k must be less than num_nodes (k={k}, num_nodes={num_nodes})")
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    rng = np.random.default_rng(seed)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 1. Regular ring lattice
    for i in range(num_nodes):
        for j in range(1, k // 2 + 1):
            target = (i + j) % num_nodes
            adj[i, target] = 1.0
            adj[target, i] = 1.0

    # 2. Random rewiring preserving exact edge count
    for j in range(1, k // 2 + 1):
        for i in range(num_nodes):
            if rng.uniform(0.0, 1.0) < p:
                target = (i + j) % num_nodes
                if adj[i, target] == 0.0:
                    continue
                candidates = [c for c in range(num_nodes) if c != i and adj[i, c] == 0.0]
                if candidates:
                    new_target = int(rng.choice(candidates))
                    adj[i, target] = 0.0
                    adj[target, i] = 0.0
                    adj[i, new_target] = 1.0
                    adj[new_target, i] = 1.0

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    x = np.eye(num_nodes, dtype=np.float32)

    return _from_numpy(adj), GraphData(x=_from_numpy(x), edge_index=_from_numpy(edge_index))


def stochastic_block_model(block_sizes: List[int], prob_matrix: np.ndarray, seed: Optional[int] = None) -> Tuple[Tensor, GraphData, Tensor]:
    """Generates a Stochastic Block Model (SBM) graph with community partition structure."""
    rng = np.random.default_rng(seed)
    num_nodes = sum(block_sizes)

    labels = []
    for b_idx, size in enumerate(block_sizes):
        labels.extend([b_idx] * size)
    labels_arr = np.array(labels, dtype=np.int64)

    prob_m = np.asarray(prob_matrix, dtype=np.float32)
    P = prob_m[labels_arr[:, None], labels_arr[None, :]]
    rand_mat = rng.uniform(0.0, 1.0, size=(num_nodes, num_nodes))
    tri = np.triu(rand_mat < P, k=1).astype(np.float32)
    adj = tri + tri.T

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    x = np.eye(num_nodes, dtype=np.float32)

    return _from_numpy(adj), GraphData(x=_from_numpy(x), edge_index=_from_numpy(edge_index)), _from_numpy(labels_arr)


def random_geometric_graph(
    num_nodes: int,
    radius: float = 0.5,
    dim: int = 2,
    seed: Optional[int] = None,
) -> Tuple[Tensor, GraphData, Tensor]:
    """Generates a Random Geometric Graph (RGG) by placing nodes uniformly in [0, 1]^dim
    and connecting pairs whose Euclidean distance is at most `radius`.
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    if radius < 0.0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1.0, size=(num_nodes, dim)).astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    adj = ((dist <= radius) & (np.eye(num_nodes) == 0)).astype(np.float32)

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    coords_t = _from_numpy(coords)
    return _from_numpy(adj), GraphData(x=coords_t, edge_index=_from_numpy(edge_index)), coords_t


__all__ = [
    "erdos_renyi_graph",
    "barabasi_albert_graph",
    "watts_strogatz_graph",
    "stochastic_block_model",
    "random_geometric_graph",
]
