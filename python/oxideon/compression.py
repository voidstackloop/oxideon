"""Tensor Decompositions and Low-Rank Layer Compression for Oxideon.

Enables compressing parameters and accelerating inference via spectral tensor factorization:
  - svd_linear_lowrank: Factorizes Linear(M, N) into Sequential(Linear(M, R), Linear(R, N))
  - tucker_decomposition: Higher-Order SVD (HOSVD) Tucker decomposition of 3D/4D tensors
  - tucker_reconstruct: Reconstructs dense tensor from Tucker core and factor matrices
  - cp_decomposition: Canonical Polyadic (CP) tensor factorization via alternating least squares
  - cp_reconstruct: Reconstructs dense tensor from CP factor matrices
"""

from typing import Tuple, List, Optional, Any
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .nn import Module, Linear, Sequential, Parameter


def svd_linear_lowrank(linear: Any, rank: int) -> Sequential:
    """Factorizes a dense Linear layer into two low-rank Linear layers (M -> R -> N)."""
    if rank < 1:
        raise ValueError("rank must be >= 1")
    w = linear.weight.to_numpy()
    out_f, in_f = w.shape
    r = min(rank, out_f, in_f)

    # SVD: W = U @ S @ Vt
    u, s, vt = np.linalg.svd(w, full_matrices=False)

    # First layer weights: sqrt(S) * Vt -> [R, in_f]
    w1 = np.diag(np.sqrt(s[:r])) @ vt[:r, :]
    # Second layer weights: U * sqrt(S) -> [out_f, R]
    w2 = u[:, :r] @ np.diag(np.sqrt(s[:r]))

    l1 = Linear(in_f, r, bias=False)
    l1.weight = Parameter(_from_numpy(np.ascontiguousarray(w1, dtype=np.float32)))

    has_bias = getattr(linear, "bias", None) is not None
    l2 = Linear(r, out_f, bias=has_bias)
    l2.weight = Parameter(_from_numpy(np.ascontiguousarray(w2, dtype=np.float32)))
    if has_bias and linear.bias is not None:
        l2.bias = Parameter(_from_numpy(np.ascontiguousarray(linear.bias.to_numpy(), dtype=np.float32)))

    return Sequential(l1, l2)


def tucker_decomposition(
    tensor: Any, ranks: List[int]
) -> Tuple[_Tensor, List[_Tensor]]:
    """Higher-Order SVD (HOSVD) Tucker decomposition: X ~= Core x_1 U_1 x_2 U_2 ..."""
    arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.asarray(tensor, dtype=np.float32)
    ndim = arr.ndim
    if len(ranks) != ndim:
        raise ValueError(f"ranks length {len(ranks)} must match tensor dimensions {ndim}")
    for n, r in enumerate(ranks):
        if r < 1 or r > arr.shape[n]:
            raise ValueError(
                f"Mode-{n} rank must be in [1, {arr.shape[n]}], got {r}"
            )
    factors = []

    # Compute mode-n unfoldings and SVD factor matrices
    for n in range(ndim):
        r = ranks[n]
        unfolded = np.rollaxis(arr, n, 0).reshape(arr.shape[n], -1)
        u, _, _ = np.linalg.svd(unfolded, full_matrices=False)
        factor = u[:, :r]
        factors.append(_from_numpy(np.ascontiguousarray(factor, dtype=np.float32)))

    # Compute core tensor: Core = X x_1 U_1^T x_2 U_2^T ...
    core = arr.copy()
    for n in range(ndim):
        u_mat = factors[n].to_numpy()
        core = np.tensordot(core, u_mat, axes=(0, 0))

    core_tensor = _from_numpy(np.ascontiguousarray(core, dtype=np.float32))
    return core_tensor, factors


def tucker_reconstruct(core: Any, factors: List[Any]) -> _Tensor:
    """Reconstructs a full tensor from its Tucker core and mode factor matrices."""
    has_grad = (isinstance(core, _Tensor) and core.requires_grad) or any(
        isinstance(f, _Tensor) and f.requires_grad for f in factors
    )
    if has_grad:
        rec_t = core if isinstance(core, _Tensor) else _from_numpy(np.ascontiguousarray(core, dtype=np.float32))
        if len(factors) != rec_t.ndim:
            raise ValueError(f"factors length {len(factors)} must match core ndim {rec_t.ndim}")
        for f in factors:
            u_t = f if isinstance(f, _Tensor) else _from_numpy(np.ascontiguousarray(f, dtype=np.float32))
            shape = list(rec_t.shape)
            r0 = shape[0]
            rest = shape[1:]
            flat = rec_t.reshape([r0, -1])
            contracted = u_t.matmul(flat)  # [I_n, prod(rest)]
            # Move axis 0 to the end so the next factor contracts against the new axis 0
            perm_out = contracted.transpose(0, 1).reshape(rest + [u_t.shape[0]])
            rec_t = perm_out
        return rec_t

    rec = core.to_numpy() if isinstance(core, _Tensor) else np.asarray(core, dtype=np.float32)
    if len(factors) != rec.ndim:
        raise ValueError(f"factors length {len(factors)} must match core ndim {rec.ndim}")
    for f in factors:
        u_mat = f.to_numpy() if isinstance(f, _Tensor) else np.asarray(f, dtype=np.float32)
        rec = np.tensordot(rec, u_mat.T, axes=(0, 0))
    return _from_numpy(np.ascontiguousarray(rec, dtype=np.float32))


def cp_decomposition(
    tensor: Any,
    rank: int = 5,
    max_iter: int = 20,
    random_state: Optional[int] = None,
) -> List[_Tensor]:
    """Canonical Polyadic (CP) decomposition via Alternating Least Squares (ALS)."""
    arr = tensor.to_numpy() if isinstance(tensor, _Tensor) else np.asarray(tensor, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"tensor must have dimension >= 2, got {arr.ndim}")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    rng = np.random.RandomState(random_state)
    ndim = arr.ndim
    factors = [rng.randn(s, rank).astype(np.float32) for s in arr.shape]

    for _ in range(max_iter):
        for n in range(ndim):
            other_factors = [factors[i] for i in range(ndim) if i != n]
            kr = other_factors[0]
            for f in other_factors[1:]:
                kr = np.einsum("ir,jr->ijr", kr, f).reshape(-1, rank)

            unfolded = np.rollaxis(arr, n, 0).reshape(arr.shape[n], -1)
            reg = 1e-4 * np.eye(rank)
            v = np.linalg.solve(kr.T @ kr + reg, kr.T @ unfolded.T).T
            factors[n] = v

    return [_from_numpy(np.ascontiguousarray(f, dtype=np.float32)) for f in factors]


def cp_reconstruct(factors: List[Any]) -> _Tensor:
    """Reconstructs a dense tensor from its CP factor matrices."""
    if len(factors) < 2:
        raise ValueError("cp_reconstruct requires at least 2 factor matrices")
    has_grad = any(isinstance(f, _Tensor) and f.requires_grad for f in factors)
    if has_grad:
        t_factors = [f if isinstance(f, _Tensor) else _from_numpy(np.ascontiguousarray(f, dtype=np.float32)) for f in factors]
        rank = t_factors[0].shape[1]
        shape = [f.shape[0] for f in t_factors]
        kr = t_factors[0]  # [I_0, R]
        for f in t_factors[1:]:
            if f.shape[1] != rank:
                raise ValueError(f"Factor rank mismatch: expected {rank}, got {f.shape[1]}")
            # Khatri-Rao product via broadcasting: [N, 1, R] * [1, M, R] -> [N * M, R]
            n_curr = kr.shape[0]
            m_curr = f.shape[0]
            kr = (kr.reshape([n_curr, 1, rank]) * f.reshape([1, m_curr, rank])).reshape([n_curr * m_curr, rank])
        return kr.sum(dim=-1).reshape(shape)

    np_factors = [f.to_numpy() if isinstance(f, _Tensor) else np.asarray(f, dtype=np.float32) for f in factors]
    rank = np_factors[0].shape[1]
    shape = [f.shape[0] for f in np_factors]
    kr = np_factors[0]
    for f in np_factors[1:]:
        if f.shape[1] != rank:
            raise ValueError(f"Factor rank mismatch: expected {rank}, got {f.shape[1]}")
        kr = np.einsum("ir,jr->ijr", kr, f).reshape(-1, rank)
    rec = np.sum(kr, axis=-1).reshape(shape)
    return _from_numpy(np.ascontiguousarray(rec, dtype=np.float32))


__all__ = [
    "svd_linear_lowrank",
    "tucker_decomposition",
    "tucker_reconstruct",
    "cp_decomposition",
    "cp_reconstruct",
]
