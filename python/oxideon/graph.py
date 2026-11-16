"""Spectral Graph Theory, Centrality, and Graph Algorithms for Oxideon.

Enables structural graph analysis, spectral embedding, and node importance scoring:
  - laplacian_matrix: Unnormalized and symmetrically normalized Graph Laplacian
  - spectral_clustering: Graph partitioning via Laplacian spectral decomposition
  - shortest_path_dijkstra: Dijkstra's single-source shortest path algorithm
  - pagerank: Power-iteration PageRank centrality vector
"""

from typing import Tuple, Optional, Any, List, Dict
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_numpy_2d(adj: Any) -> np.ndarray:
    if isinstance(adj, _Tensor):
        return adj.to_numpy()
    return np.asarray(adj, dtype=np.float32)


def laplacian_matrix(adjacency: Any, normalized: bool = True) -> _Tensor:
    """Computes the Graph Laplacian matrix L."""
    a = _to_numpy_2d(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be a square adjacency matrix, got shape {a.shape}")

    if isinstance(adjacency, _Tensor) and adjacency.requires_grad:
        n = a.shape[0]
        eye = _from_numpy(np.eye(n, dtype=np.float32))
        deg = adjacency.sum(dim=1, keepdim=True)
        if not normalized:
            return (deg * eye) - adjacency
        deg_safe = (deg - 1e-12).relu() + 1e-12
        deg_inv_sqrt = deg_safe.rsqrt()
        scale = deg_inv_sqrt * deg_inv_sqrt.transpose(0, 1)
        return eye - (adjacency * scale)

    deg = np.sum(a, axis=1)

    if not normalized:
        l = np.diag(deg) - a
    else:
        deg_inv_sqrt = np.power(np.maximum(deg, 1e-12), -0.5)
        d_mat = np.diag(deg_inv_sqrt)
        l = np.eye(a.shape[0], dtype=np.float32) - d_mat @ a @ d_mat

    return _from_numpy(np.ascontiguousarray(l, dtype=np.float32))


def spectral_clustering(adjacency: Any, k: int = 2, seed: Optional[int] = None) -> np.ndarray:
    """Partitions graph nodes into k clusters using eigenvectors of normalized Laplacian."""
    a = _to_numpy_2d(adjacency)
    n = a.shape[0]
    if k < 1 or k > n:
        raise ValueError(f"k must be between 1 and num_nodes ({n}), got {k}")
    l = laplacian_matrix(a, normalized=True).to_numpy()
    eigvals, eigvecs = np.linalg.eigh(l)

    # First k eigenvectors corresponding to smallest eigenvalues
    u = eigvecs[:, :k]
    # Normalize rows
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)

    rng = np.random.default_rng(seed)
    centers = [u[rng.choice(len(u))]]
    for _ in range(1, k):
        dists = np.min(np.stack([np.sum((u - c) ** 2, axis=1) for c in centers], axis=1), axis=1)
        probs = dists / np.maximum(np.sum(dists), 1e-12)
        centers.append(u[rng.choice(len(u), p=probs)])
    centers = np.array(centers)

    for _ in range(50):
        dists = np.stack([np.sum((u - c) ** 2, axis=1) for c in centers], axis=1)
        labels = np.argmin(dists, axis=1)
        for c in range(k):
            if np.any(labels == c):
                centers[c] = np.mean(u[labels == c], axis=0)

    return labels


def shortest_path_dijkstra(adjacency: Any, source: int) -> np.ndarray:
    """Computes shortest path distances from source to all reachable nodes."""
    a = _to_numpy_2d(adjacency)
    n = a.shape[0]
    if not (0 <= source < n):
        raise ValueError(f"source must be in range [0, {n-1}], got {source}")

    dist = np.full(n, np.inf, dtype=np.float32)
    dist[source] = 0.0
    visited = np.zeros(n, dtype=bool)

    for _ in range(n):
        unvisited_dists = np.where(visited, np.inf, dist)
        u = int(np.argmin(unvisited_dists))
        if np.isinf(dist[u]):
            break
        visited[u] = True

        for v in range(n):
            if a[u, v] > 0 and not visited[v]:
                alt = dist[u] + a[u, v]
                if alt < dist[v]:
                    dist[v] = alt

    return dist


def pagerank(
    adjacency: Any, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
) -> _Tensor:
    """Computes PageRank centrality vector for directed or undirected graphs."""
    a = _to_numpy_2d(adjacency)
    n = a.shape[0]

    if isinstance(adjacency, _Tensor) and adjacency.requires_grad:
        out_deg = adjacency.sum(dim=1, keepdim=True)
        has_out = (out_deg.to_numpy().reshape(-1) > 0)
        mask_pos = _from_numpy(has_out.astype(np.float32).reshape(n, 1))
        mask_zero = _from_numpy((~has_out).astype(np.float32).reshape(n, 1))
        denom = out_deg * mask_pos + mask_zero
        p_mat = (adjacency / denom) * mask_pos + mask_zero * (1.0 / float(n))
        m_t = p_mat.transpose(0, 1)

        rank = _from_numpy(np.full((n, 1), 1.0 / float(n), dtype=np.float32))
        teleport = (1.0 - damping) / float(n)
        for _ in range(max_iter):
            new_rank = m_t.matmul(rank) * damping + teleport
            if np.linalg.norm((new_rank - rank).to_numpy(), ord=1) < tol:
                rank = new_rank
                break
            rank = new_rank
        return rank.reshape([n])

    out_deg = np.sum(a, axis=1)
    # Transition matrix M[j, i] = prob i -> j
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if out_deg[i] > 0:
            m[:, i] = a[i, :] / out_deg[i]
        else:
            m[:, i] = 1.0 / n

    rank = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        new_rank = damping * (m @ rank) + teleport
        if np.linalg.norm(new_rank - rank, ord=1) < tol:
            break
        rank = new_rank

    return _from_numpy(np.ascontiguousarray(rank, dtype=np.float32))


__all__ = [
    "laplacian_matrix",
    "spectral_clustering",
    "shortest_path_dijkstra",
    "pagerank",
]
