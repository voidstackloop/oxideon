"""
Causal Discovery and Structural Equation Modeling (DAG learning) for Oxideon.
Includes the continuous NOTEARS algorithm, statistical skeleton discovery, acyclicity verification,
and Structural Hamming Distance (SHD).
"""
from typing import List, Tuple, Optional
from collections import deque
import numpy as np
from .oxideon import Tensor, from_numpy as _from_numpy


def _matrix_exp(A: np.ndarray, order: int = 10) -> np.ndarray:
    """Computes matrix exponential using Taylor expansion with scaling and squaring."""
    A = np.nan_to_num(A, nan=0.0, posinf=100.0, neginf=-100.0)
    norm = float(np.linalg.norm(A, np.inf))
    if norm > 100.0:
        A = A * (100.0 / max(norm, 1e-12))
        norm = 100.0
    scaling = max(0, int(np.ceil(np.log2(max(norm, 1e-12))))) if norm > 1.0 else 0
    scaling = min(scaling, 20)
    A_scaled = A / (2.0 ** scaling)

    res = np.eye(A.shape[0], dtype=A.dtype)
    term = np.eye(A.shape[0], dtype=A.dtype)
    for k in range(1, order + 1):
        term = term @ A_scaled / float(k)
        res = res + term

    for _ in range(scaling):
        res = res @ res
    return np.nan_to_num(res, nan=1.0, posinf=1e6, neginf=0.0)


def _h_and_grad(W: np.ndarray) -> Tuple[float, np.ndarray]:
    """Computes acyclicity constraint h(W) and its gradient dh/dW in a single matrix exponential pass."""
    d = W.shape[0]
    W_safe = np.clip(np.nan_to_num(W, nan=0.0), -10.0, 10.0)
    M = W_safe * W_safe
    E = _matrix_exp(M)
    h_val = float(np.trace(E) - d)
    grad_h = 2.0 * E.T * W_safe
    return max(0.0, h_val), np.nan_to_num(grad_h, nan=0.0)


def _h(W: np.ndarray) -> float:
    """Matrix exponential acyclicity constraint h(W) = tr(exp(W * W)) - d."""
    h_val, _ = _h_and_grad(W)
    return h_val


def _grad_h(W: np.ndarray) -> np.ndarray:
    """Gradient of acyclicity constraint dh/dW = 2 * (exp(W * W))^T * W."""
    _, grad = _h_and_grad(W)
    return grad


def notears_linear(
    X: Tensor,
    lambda1: float = 0.05,
    max_iter: int = 50,
    h_tol: float = 1e-6,
    rho_max: float = 1e14,
    w_threshold: float = 0.2
) -> Tensor:
    """
    NOTEARS algorithm for continuous DAG discovery from observational data (Zheng et al., NeurIPS 2018).
    Minimizes: (1 / 2n) * ||X - XW||_F^2 + lambda1 * ||W||_1 subject to h(W) = tr(exp(W * W)) - d = 0.
    """
    x_np = X.to_numpy().astype(np.float64)
    n, d = x_np.shape
    w_est = np.zeros((d, d), dtype=np.float64)
    rho = 1.0
    alpha = 0.0
    h_curr = np.inf

    xtx = x_np.T @ x_np / float(max(n, 1))

    for it in range(max_iter):
        w_new = w_est.copy()
        while rho < rho_max:
            w_inner = w_est.copy()
            for _ in range(40):
                grad_loss = xtx @ w_inner - xtx
                h_val, grad_h = _h_and_grad(w_inner)
                grad_penalty = (rho * h_val + alpha) * grad_h
                grad_l1 = lambda1 * np.sign(w_inner)

                total_grad = grad_loss + grad_penalty + grad_l1
                np.fill_diagonal(total_grad, 0.0)

                gnorm = float(np.linalg.norm(total_grad))
                if gnorm > 50.0:
                    total_grad = total_grad * (50.0 / gnorm)

                step_size = 1e-3 / (1.0 + 0.05 * it + 0.001 * np.log1p(rho))
                w_inner -= step_size * total_grad
                w_inner = np.clip(w_inner, -10.0, 10.0)
                np.fill_diagonal(w_inner, 0.0)

            w_new = w_inner
            h_new = _h(w_new)
            if np.isfinite(h_curr) and h_new > 0.25 * h_curr:
                rho *= 10.0
            else:
                break

        w_est = w_new
        h_curr = _h(w_est)
        alpha += rho * h_curr
        if h_curr <= h_tol or rho >= rho_max:
            break

    w_est[np.abs(w_est) < w_threshold] = 0.0
    np.fill_diagonal(w_est, 0.0)

    return _from_numpy(w_est.astype(np.float32))


def correlation_skeleton(X: Tensor, threshold: float = 0.2) -> Tensor:
    """Discovers undirected causal skeleton by thresholding empirical correlation matrix."""
    x = X.to_numpy()
    if x.ndim == 1:
        x = np.expand_dims(x, 1)
    if len(x) == 0:
        return _from_numpy(np.zeros((x.shape[1], x.shape[1]), dtype=np.float32))
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    std = np.std(x_centered, axis=0, keepdims=True) + 1e-7
    x_norm = x_centered / std
    corr = (x_norm.T @ x_norm) / float(len(x))
    np.fill_diagonal(corr, 0.0)

    skeleton = (np.abs(corr) >= threshold).astype(np.float32)
    return _from_numpy(skeleton)


def is_dag(adj: Tensor, eps: float = 1e-12) -> bool:
    """Checks whether an adjacency matrix (with positive or negative edge weights) represents a DAG."""
    a = adj.to_numpy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adj must be a square 2D matrix, got shape {a.shape}")
    n = a.shape[0]
    if n == 0:
        return True
    mask = np.abs(a) > eps
    if np.any(np.diag(mask)):
        return False
    in_degrees = np.sum(mask, axis=0).astype(np.int64)

    visited = 0
    zero_in = deque(np.where(in_degrees == 0)[0].tolist())

    while zero_in:
        curr = zero_in.popleft()
        visited += 1
        neighbors = np.where(mask[curr])[0]
        for neighbor in neighbors:
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                zero_in.append(int(neighbor))

    return visited == n


def topological_sort(adj: Tensor, eps: float = 1e-12) -> List[int]:
    """Returns a topological ordering of nodes in a DAG, or raises ValueError if graph contains cycles."""
    a = adj.to_numpy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adj must be a square 2D matrix, got shape {a.shape}")
    n = a.shape[0]
    mask = np.abs(a) > eps
    if np.any(np.diag(mask)):
        raise ValueError("Graph contains self-loops; topological sort is undefined.")
    in_degrees = np.sum(mask, axis=0).astype(np.int64)
    order: List[int] = []
    zero_in = deque(np.where(in_degrees == 0)[0].tolist())

    while zero_in:
        curr = int(zero_in.popleft())
        order.append(curr)
        neighbors = np.where(mask[curr])[0]
        for neighbor in neighbors:
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                zero_in.append(int(neighbor))

    if len(order) != n:
        raise ValueError("Graph contains directed cycles; topological sort is undefined.")
    return order


def structural_hamming_distance(pred_adj: Tensor, true_adj: Tensor, eps: float = 1e-12) -> int:
    """Computes Structural Hamming Distance (SHD) between predicted and ground-truth directed graphs.
    Counts edge additions, deletions, and single reversals.
    """
    p = (np.abs(pred_adj.to_numpy()) > eps).astype(np.int32)
    t = (np.abs(true_adj.to_numpy()) > eps).astype(np.int32)
    if p.shape != t.shape or p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"Adjacency matrices must be matching square 2D shapes, got {p.shape} vs {t.shape}")
    np.fill_diagonal(p, 0)
    np.fill_diagonal(t, 0)

    n = p.shape[0]
    shd = 0
    for i in range(n):
        for j in range(i + 1, n):
            p_pair = (p[i, j], p[j, i])
            t_pair = (t[i, j], t[j, i])
            if p_pair != t_pair:
                shd += 1
    return int(shd)


__all__ = [
    "notears_linear",
    "correlation_skeleton",
    "is_dag",
    "topological_sort",
    "structural_hamming_distance",
]
