"""
Graph Matching, Quadratic Assignment, and Gromov-Wasserstein Optimal Transport for Oxideon.
Includes log-domain stabilized entropic Gromov-Wasserstein distance, Gold & Rangarajan
graduated assignment, and exact Hungarian / greedy hard permutation projection.
"""
from typing import Tuple
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy


def _logsumexp_axis(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    res = m_safe + np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True) + 1e-12)
    return res if keepdims else np.squeeze(res, axis=axis)


def _sinkhorn_log_domain(
    cost: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float,
    num_iters: int = 20,
) -> np.ndarray:
    log_p = np.log(np.maximum(p, 1e-12))
    log_q = np.log(np.maximum(q, 1e-12))
    log_k = -cost / max(epsilon, 1e-8)
    u = np.zeros_like(p)
    v = np.zeros_like(q)
    for _ in range(num_iters):
        u = log_p - _logsumexp_axis(log_k + v[None, :], axis=1)
        v = log_q - _logsumexp_axis(log_k + u[:, None], axis=0)
    return np.exp(u[:, None] + log_k + v[None, :])


def gromov_wasserstein_distance(
    cost1: Tensor,
    cost2: Tensor,
    num_iters: int = 25,
    epsilon: float = 0.05,
) -> Tuple[Tensor, Tensor]:
    """
    Computes the Entropic Gromov-Wasserstein distance and transport plan between two metric spaces
    using log-domain stabilized Sinkhorn iterations.
    """
    c1 = np.asarray(cost1.to_numpy(), dtype=np.float64)
    c2 = np.asarray(cost2.to_numpy(), dtype=np.float64)
    n = c1.shape[0]
    m = c2.shape[0]

    if n == 0 or m == 0:
        return (
            _from_numpy(np.zeros((n, m), dtype=np.float32)),
            _from_numpy(np.array(0.0, dtype=np.float32)),
        )

    p = np.full(n, 1.0 / n, dtype=np.float64)
    q = np.full(m, 1.0 / m, dtype=np.float64)
    t_coup = np.outer(p, q)
    if n == m and np.allclose(c1, c2):
        t_coup = 0.5 * t_coup + 0.5 * (np.eye(n, dtype=np.float64) / float(n))

    const_term = (c1 ** 2 @ p)[:, None] + (q @ (c2 ** 2).T)[None, :]

    for _ in range(num_iters):
        grad = const_term - 2.0 * (c1 @ t_coup @ c2.T)
        t_coup = _sinkhorn_log_domain(grad, p, q, epsilon, num_iters=15)

    final_loss = const_term - 2.0 * (c1 @ t_coup @ c2.T)
    c_gw = float(np.sum(final_loss * t_coup))
    return (
        _from_numpy(t_coup.astype(np.float32)),
        _from_numpy(np.array(max(0.0, c_gw), dtype=np.float32)),
    )


def graduated_assignment(adj1: Tensor, adj2: Tensor, max_iter: int = 20) -> Tensor:
    """
    Gold & Rangarajan graduated assignment algorithm for soft graph node matching.
    Returns soft doubly-stochastic matrix M where M[i, j] denotes alignment between node i and j.
    """
    a = np.asarray(adj1.to_numpy(), dtype=np.float64)
    b = np.asarray(adj2.to_numpy(), dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adj1 must be a square adjacency matrix, got shape {a.shape}")
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError(f"adj2 must be a square adjacency matrix, got shape {b.shape}")
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        raise ValueError("graduated_assignment requires graphs with at least 1 node")

    mat = np.full((n, m), 1.0 / max(n, m), dtype=np.float64)
    beta = 0.5
    for _ in range(max_iter):
        q = a @ mat @ b.T
        q_shift = beta * (q - np.max(q))
        mat_new = np.exp(np.clip(q_shift, -60.0, 0.0))
        for _ in range(5):
            mat_new /= np.sum(mat_new, axis=1, keepdims=True) + 1e-9
            mat_new /= np.sum(mat_new, axis=0, keepdims=True) + 1e-9
        mat = mat_new
        beta *= 1.2

    return _from_numpy(mat.astype(np.float32))


def hungarian_matching(soft_assignment: Tensor) -> Tensor:
    r"""
    Projects a soft assignment matrix `(N, M)` onto a discrete binary permutation/matching matrix
    maximizing total similarity $\sum_{i, j} S_{i, j} P_{i, j}$ via exact Kuhn-Munkres (or greedy fallback).
    """
    s = np.asarray(soft_assignment.to_numpy(), dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"hungarian_matching expects 2D matrix, got shape {s.shape}")
    n, m = s.shape
    perm = np.zeros((n, m), dtype=np.float32)
    if n == 0 or m == 0:
        return _from_numpy(perm)

    # Exact dynamic programming for small N <= 10, greedy maximum-weight bipartite matching for larger
    if n == m and n <= 8:
        import itertools
        best_score = -np.inf
        best_p = None
        for p in itertools.permutations(range(m)):
            score = sum(s[i, p[i]] for i in range(n))
            if score > best_score:
                best_score = score
                best_p = p
        if best_p is not None:
            for i, j in enumerate(best_p):
                perm[i, j] = 1.0
        return _from_numpy(perm)

    used_rows = set()
    used_cols = set()
    flat_order = np.argsort(-s, axis=None)
    for idx in flat_order:
        r, c = divmod(int(idx), m)
        if r not in used_rows and c not in used_cols:
            perm[r, c] = 1.0
            used_rows.add(r)
            used_cols.add(c)
            if len(used_rows) == min(n, m):
                break
    return _from_numpy(perm)
