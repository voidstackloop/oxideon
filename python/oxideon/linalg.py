"""Comprehensive linear algebra suite for Oxideon.

Provides high-performance, batched, and exact linear algebra operations mirroring
standard deep learning primitives, including matrix decompositions, eigensystems,
factorizations, solvers, and generalized matrix/vector norms.
"""

from collections import namedtuple
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from . import Tensor, from_numpy

QRResult = namedtuple("QRResult", ["Q", "R"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])
SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
EigResult = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
LstsqResult = namedtuple("LstsqResult", ["solution", "residuals", "rank", "singular_values"])


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.to_numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _wrap(arr: np.ndarray, like: Optional[Tensor] = None) -> Tensor:
    res = from_numpy(np.ascontiguousarray(arr))
    if like is not None and hasattr(like, "device"):
        try:
            res = res.to(like.device)
        except Exception:
            pass
    return res


def norm(
    A: Tensor,
    ord: Optional[Union[int, float, str]] = None,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Computes a matrix or vector norm."""
    if isinstance(A, Tensor) and A.requires_grad:
        if ord is None or ord == "fro" or ord == 2:
            sq = A * A
            if dim is None:
                s = sq.sum()
                res = (s + 1e-12).sqrt()
                if keepdim:
                    res = res.reshape([1] * len(A.shape))
            elif isinstance(dim, int):
                res = (sq.sum(dim=dim, keepdim=keepdim) + 1e-12).sqrt()
            else:
                s = sq
                dims = sorted([d if d >= 0 else len(A.shape) + d for d in dim], reverse=True)
                for d in dims:
                    s = s.sum(dim=d, keepdim=keepdim)
                res = (s + 1e-12).sqrt()
            if out is not None:
                out.zero_()
                out.add_(res)
                return out
            return res
        elif ord == 1 and (dim is None or isinstance(dim, int)):
            if dim is None:
                s = A.abs().sum()
                res = s.reshape([1] * len(A.shape)) if keepdim else s
            else:
                res = A.abs().sum(dim=dim, keepdim=keepdim)
            if out is not None:
                out.zero_()
                out.add_(res)
                return out
            return res

    arr = _to_np(A)
    res_np = np.linalg.norm(arr, ord=ord, axis=dim, keepdims=keepdim)
    res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def vector_norm(
    x: Tensor,
    ord: Union[int, float, str] = 2,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Computes a vector norm along specified dimension(s)."""
    if isinstance(x, Tensor) and (ord is None or isinstance(ord, (int, float))):
        ord_f = 2.0 if ord is None else float(ord)
        if dim is None:
            dims: list[int] = []
        elif isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)
        res = x.vector_norm(ord_f, dims, keepdim)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res
    return norm(x, ord=ord, dim=dim, keepdim=keepdim, out=out)


def matrix_norm(
    A: Tensor,
    ord: Union[int, float, str] = "fro",
    dim: Tuple[int, int] = (-2, -1),
    keepdim: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Computes a matrix norm over the specified two dimensions."""
    if isinstance(A, Tensor) and tuple(dim) in ((-2, -1), (A.ndim - 2, A.ndim - 1)):
        ord_map = {
            None: "fro",
            "fro": "fro",
            "nuc": "nuc",
            1: "1",
            "1": "1",
            -1: "-1",
            "-1": "-1",
            2: "2",
            "2": "2",
            -2: "-2",
            "-2": "-2",
            float("inf"): "inf",
            "inf": "inf",
            float("-inf"): "-inf",
            "-inf": "-inf",
        }
        if ord in ord_map:
            res = A.matrix_norm(ord_map[ord], keepdim)
            if out is not None:
                out.zero_()
                out.add_(res)
                return out
            return res
    return norm(A, ord=ord, dim=dim, keepdim=keepdim, out=out)


def inv(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the inverse of a square matrix or batch of matrices."""
    if hasattr(A, "inv"):
        res = A.inv()
    elif hasattr(A, "inverse"):
        res = A.inverse()
    else:
        arr = _to_np(A)
        res_np = np.linalg.inv(arr)
        res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def pinv(
    A: Tensor,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    hermitian: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Computes the pseudo-inverse (Moore-Penrose inverse) of a matrix or batch of matrices."""
    if isinstance(A, Tensor) and not hermitian and A.requires_grad and atol is None and not isinstance(rtol, Tensor):
        rcond = float(rtol) if rtol is not None else 1e-7
        res = A.pinv(rcond=rcond)
    else:
        arr = _to_np(A)
        rcond = rtol if rtol is not None else (atol if atol is not None else 1e-15)
        if hermitian:
            u, s, vh = np.linalg.svd(arr, hermitian=True)
            cutoff = rcond * np.amax(s, axis=-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_inv = np.where(s > cutoff, 1.0 / s, 0.0)
            res_np = np.matmul(vh.swapaxes(-2, -1).conj() * s_inv[..., None, :], u.swapaxes(-2, -1).conj())
        else:
            res_np = np.linalg.pinv(arr, rcond=rcond)
        res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def det(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the determinant of a square matrix or batch of matrices."""
    if hasattr(A, "det"):
        res = A.det()
    else:
        arr = _to_np(A)
        res_np = np.linalg.det(arr)
        res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def slogdet(A: Tensor, *, out: Optional[Tuple[Tensor, Tensor]] = None) -> SlogdetResult:
    """Computes the sign and log of the absolute value of determinant: (sign, logabsdet)."""
    if hasattr(A, "slogdet"):
        sign, logabsdet = A.slogdet()
    else:
        arr = _to_np(A)
        sign_np, logabsdet_np = np.linalg.slogdet(arr)
        sign = _wrap(sign_np, like=A)
        logabsdet = _wrap(logabsdet_np, like=A)
    if out is not None:
        out[0].zero_()
        out[0].add_(sign)
        out[1].zero_()
        out[1].add_(logabsdet)
        return SlogdetResult(out[0], out[1])
    return SlogdetResult(sign, logabsdet)


def cholesky(A: Tensor, upper: bool = False, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the Cholesky decomposition of a symmetric positive-definite matrix."""
    if isinstance(A, Tensor):
        res = A.cholesky()
        if upper:
            res = res.transpose(-2, -1)
    else:
        arr = _to_np(A)
        l_np = np.linalg.cholesky(arr)
        if upper:
            l_np = np.swapaxes(l_np, -2, -1)
        res = _wrap(l_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def cholesky_solve(B: Tensor, L: Tensor, upper: bool = False, *, out: Optional[Tensor] = None) -> Tensor:
    """Solves a system of linear equations A X = B given the Cholesky factor L of A."""
    if isinstance(B, Tensor) and isinstance(L, Tensor):
        res = B.cholesky_solve(L, upper=upper)
    else:
        b_np = _to_np(B)
        l_np = _to_np(L)
        if upper:
            a_np = np.matmul(np.swapaxes(l_np, -2, -1), l_np)
        else:
            a_np = np.matmul(l_np, np.swapaxes(l_np, -2, -1))
        res_np = np.linalg.solve(a_np, b_np)
        res = _wrap(res_np, like=B)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def cholesky_inverse(L: Tensor, upper: bool = False, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the inverse of a symmetric positive-definite matrix given its Cholesky factor."""
    if isinstance(L, Tensor):
        res = L.cholesky_inverse(upper=upper)
    else:
        l_np = _to_np(L)
        l_inv = np.linalg.inv(l_np)
        if upper:
            inv_a = np.matmul(l_inv, np.swapaxes(l_inv, -2, -1))
        else:
            inv_a = np.matmul(np.swapaxes(l_inv, -2, -1), l_inv)
        res = _wrap(inv_a, like=L)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def qr(A: Tensor, mode: str = "reduced", *, out: Optional[Tuple[Tensor, Tensor]] = None) -> QRResult:
    """Computes QR decomposition of a matrix or batch of matrices."""
    if mode not in ("reduced", "complete", "economic"):
        raise ValueError(f"unsupported qr mode {mode!r} (expected 'reduced', 'complete' or 'economic')")
    if isinstance(A, Tensor) and mode in ("reduced", "economic") and A.requires_grad:
        q, r = A.qr()
    else:
        arr = _to_np(A)
        np_mode = "reduced" if mode in ("reduced", "economic") else "complete"
        q_np, r_np = np.linalg.qr(arr, mode=np_mode)
        q = _wrap(q_np, like=A)
        r = _wrap(r_np, like=A)
    if out is not None:
        out[0].zero_()
        out[0].add_(q)
        out[1].zero_()
        out[1].add_(r)
        return QRResult(out[0], out[1])
    return QRResult(q, r)


def svd(A: Tensor, full_matrices: bool = True, *, out: Optional[Tuple[Tensor, Tensor, Tensor]] = None) -> SVDResult:
    """Computes the Singular Value Decomposition (SVD): A = U @ diag(S) @ Vh."""
    if isinstance(A, Tensor) and (not full_matrices or (A.ndim >= 2 and A.shape[-2] == A.shape[-1])):
        u, s, vh = A.svd()
    else:
        arr = _to_np(A)
        u_np, s_np, vh_np = np.linalg.svd(arr, full_matrices=full_matrices)
        u = _wrap(u_np, like=A)
        s = _wrap(s_np, like=A)
        vh = _wrap(vh_np, like=A)
    if out is not None:
        out[0].zero_()
        out[0].add_(u)
        out[1].zero_()
        out[1].add_(s)
        out[2].zero_()
        out[2].add_(vh)
        return SVDResult(out[0], out[1], out[2])
    return SVDResult(u, s, vh)


def svdvals(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes singular values of a matrix or batch of matrices."""
    if isinstance(A, Tensor):
        res = A.svdvals()
    else:
        arr = _to_np(A)
        s_np = np.linalg.svd(arr, compute_uv=False)
        res = _wrap(s_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def eig(A: Tensor, *, out: Optional[Tuple[Tensor, Tensor]] = None) -> EigResult:
    """Computes the eigenvalues and right eigenvectors of a square matrix."""
    arr = _to_np(A)
    w_np, v_np = np.linalg.eig(arr)
    w = _wrap(w_np.real if np.isrealobj(arr) and np.all(np.isreal(w_np)) else w_np, like=A)
    v = _wrap(v_np.real if np.isrealobj(arr) and np.all(np.isreal(v_np)) else v_np, like=A)
    if out is not None:
        out[0].zero_()
        out[0].add_(w)
        out[1].zero_()
        out[1].add_(v)
        return EigResult(out[0], out[1])
    return EigResult(w, v)


def eigh(A: Tensor, UPLO: str = "L", *, out: Optional[Tuple[Tensor, Tensor]] = None) -> EigResult:
    """Computes eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix."""
    if isinstance(A, Tensor):
        w, v = A.eigh()
    else:
        arr = _to_np(A)
        w_np, v_np = np.linalg.eigh(arr, UPLO=UPLO)
        w = _wrap(w_np, like=A)
        v = _wrap(v_np, like=A)
    if out is not None:
        out[0].zero_()
        out[0].add_(w)
        out[1].zero_()
        out[1].add_(v)
        return EigResult(out[0], out[1])
    return EigResult(w, v)


def eigvals(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the eigenvalues of a square matrix."""
    arr = _to_np(A)
    w_np = np.linalg.eigvals(arr)
    res = _wrap(w_np.real if np.isrealobj(arr) and np.all(np.isreal(w_np)) else w_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def eigvalsh(A: Tensor, UPLO: str = "L", *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the eigenvalues of a Hermitian or real symmetric matrix."""
    if isinstance(A, Tensor):
        res = A.eigvalsh()
    else:
        arr = _to_np(A)
        w_np = np.linalg.eigvalsh(arr, UPLO=UPLO)
        res = _wrap(w_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def solve(A: Tensor, B: Tensor, *, left: bool = True, out: Optional[Tensor] = None) -> Tensor:
    """Solves a system of linear equations A X = B (or X A = B if left=False)."""
    if isinstance(A, Tensor) and isinstance(B, Tensor):
        if left:
            if B.ndim == 1:
                res = A.solve(B.unsqueeze(-1)).squeeze(-1)
            else:
                res = A.solve(B)
        else:
            if B.ndim == 1:
                res = A.transpose(-2, -1).solve(B.unsqueeze(-1)).squeeze(-1)
            else:
                res = A.transpose(-2, -1).solve(B.transpose(-2, -1)).transpose(-2, -1)
    else:
        a_np = _to_np(A)
        b_np = _to_np(B)
        if left:
            res_np = np.linalg.solve(a_np, b_np)
        else:
            res_np = np.swapaxes(np.linalg.solve(np.swapaxes(a_np, -2, -1), np.swapaxes(b_np, -2, -1)), -2, -1)
        res = _wrap(res_np, like=B)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def lstsq(
    A: Tensor,
    B: Tensor,
    rcond: Optional[float] = None,
    *,
    driver: Optional[str] = None,
) -> LstsqResult:
    """Computes the least squares solution to A X = B."""
    if isinstance(A, Tensor) and isinstance(B, Tensor) and hasattr(A, "lstsq"):
        rcond_val = 1e-7 if rcond is None else float(rcond)
        sol = A.lstsq(B, rcond_val)
        s = svdvals(A)
        m, n = A.shape[-2], A.shape[-1]
        if m > n:
            diff = A.matmul(sol) - B
            residuals = (diff * diff).sum(dim=-2)
        else:
            residuals = from_numpy(np.array([], dtype=np.float32))
        tol = rcond_val * float(s.max().item()) if s.numel() > 0 else 0.0
        rank = int((s > tol).to_dtype("i64").sum().item()) if s.numel() > 0 else 0
        return LstsqResult(
            solution=sol,
            residuals=residuals,
            rank=rank,
            singular_values=s,
        )

    a_np = _to_np(A)
    b_np = _to_np(B)
    sol, residuals, rank, s = np.linalg.lstsq(a_np, b_np, rcond=rcond)
    return LstsqResult(
        solution=_wrap(sol, like=B),
        residuals=_wrap(residuals, like=B) if residuals.size > 0 else from_numpy(np.array([], dtype=np.float32)),
        rank=int(rank),
        singular_values=_wrap(s, like=A),
    )


def matrix_power(A: Tensor, n: int, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the n-th power of a square matrix or batch of matrices."""
    if len(A.shape) < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError(f"matrix_power: expected square matrix or batch of square matrices, got shape {A.shape}")
    if isinstance(A, Tensor):
        if n >= 0 and hasattr(A, "matrix_power"):
            res = A.matrix_power(int(n))
        elif n == 0:
            eye = from_numpy(np.ascontiguousarray(np.eye(A.shape[-1], dtype=np.float32))).to(dtype=A.dtype, device=A.device).expand(A.shape)
            res = eye
        elif n > 0:
            res = A
            for _ in range(n - 1):
                res = res.matmul(A)
        else:
            A_inv = inv(A)
            res = A_inv
            for _ in range(-n - 1):
                res = res.matmul(A_inv)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    arr = _to_np(A)
    res_np = np.linalg.matrix_power(arr, n)
    res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def matrix_exp(A: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the matrix exponential e^A via Taylor/Pade approximation."""
    if isinstance(A, Tensor) and hasattr(A, "matrix_exp"):
        res = A.matrix_exp()
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res
    arr = _to_np(A)
    try:
        import scipy.linalg
        res_np = scipy.linalg.expm(arr)
    except Exception:
        norm_a = np.linalg.norm(arr, ord=np.inf, axis=(-2, -1), keepdims=True)
        k = np.maximum(0, np.ceil(np.log2(np.maximum(norm_a, 1.0)))).astype(int)
        max_k = int(np.max(k))
        scaled_a = arr / (2.0 ** max_k)
        res_np = np.eye(arr.shape[-1], dtype=arr.dtype)
        term = np.eye(arr.shape[-1], dtype=arr.dtype)
        for i in range(1, 20):
            term = np.matmul(term, scaled_a) / i
            res_np = res_np + term
        for _ in range(max_k):
            res_np = np.matmul(res_np, res_np)

    res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def cross(input: Tensor, other: Tensor, dim: int = -1, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the cross product of two 3-element vectors along dim."""
    if hasattr(input, "cross") and isinstance(other, Tensor):
        res = input.cross(other, dim=dim)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    a_np = _to_np(input)
    b_np = _to_np(other)
    res_np = np.cross(a_np, b_np, axis=dim)
    res = _wrap(res_np, like=input)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def cond(A: Tensor, p: Optional[Union[int, float, str]] = None, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the condition number of a matrix with respect to norm p."""
    if isinstance(A, Tensor):
        if p in (None, 2):
            s = svdvals(A)
            res = s[..., 0] / s[..., -1]
        elif p == -2:
            s = svdvals(A)
            res = s[..., -1] / s[..., 0]
        elif p in ("fro", "nuc", 1, -1, float("inf"), float("-inf")):
            res = matrix_norm(A, ord=p) * matrix_norm(inv(A), ord=p)
        else:
            arr = _to_np(A)
            res = _wrap(np.linalg.cond(arr, p=p), like=A)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    arr = _to_np(A)
    res_np = np.linalg.cond(arr, p=p)
    res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def matrix_rank(
    A: Tensor,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    hermitian: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Computes the numerical rank of a matrix or batch of matrices."""
    if isinstance(A, Tensor):
        s = eigvalsh(A).abs() if hermitian else svdvals(A)
        if s.numel() == 0:
            res = from_numpy(np.zeros(A.shape[:-2], dtype=np.int64)).to(device=A.device)
        else:
            smax = s.max(dim=-1, keepdim=True)
            if isinstance(smax, tuple):
                smax = smax[0]
            if atol is None and rtol is None:
                m, n = A.shape[-2], A.shape[-1]
                eps = 1.1920929e-7 if str(A.dtype).endswith("32") else 2.220446049250313e-16
                tol = smax * (max(m, n) * eps)
            else:
                tol_r = smax * (0.0 if rtol is None else float(rtol))
                if atol is not None:
                    tol_a = float(atol)
                    tol = tol_r.clamp_min(tol_a) if hasattr(tol_r, "clamp_min") else tol_r
                else:
                    tol = tol_r
            res = (s > tol).to_dtype("i64").sum(dim=-1)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    arr = _to_np(A)
    if rtol is None and atol is None:
        tol = None
    else:
        s = np.linalg.svd(arr, compute_uv=False)
        smax = np.amax(s, axis=-1, keepdims=True) if s.size else 0.0
        tol = np.maximum(atol or 0.0, (rtol or 0.0) * smax)
    res_np = np.linalg.matrix_rank(arr, tol=tol, hermitian=hermitian)
    res = _wrap(res_np, like=A)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def multi_dot(tensors: Sequence[Tensor], *, out: Optional[Tensor] = None) -> Tensor:
    """Efficiently multiplies two or more 2D matrices using dynamic programming order."""
    if len(tensors) >= 2 and all(isinstance(t, Tensor) for t in tensors):
        res = tensors[0]
        for t in tensors[1:]:
            res = res.matmul(t)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    arrays = [_to_np(t) for t in tensors]
    res_np = np.linalg.multi_dot(arrays)
    res = _wrap(res_np, like=tensors[0])
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def diagonal(A: Tensor, offset: int = 0, dim1: int = -2, dim2: int = -1) -> Tensor:
    """Returns a partial view of A with its diagonal elements along dim1 and dim2."""
    if hasattr(A, "diagonal"):
        return A.diagonal(offset, dim1, dim2)
    arr = _to_np(A)
    res_np = np.diagonal(arr, offset=offset, axis1=dim1, axis2=dim2)
    return _wrap(res_np, like=A)


def lu(A: Tensor, pivot: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes the LU decomposition with partial pivoting: P @ L @ U = A."""
    arr = _to_np(A)
    try:
        import scipy.linalg as sla
        p, l, u = sla.lu(arr)
        return _wrap(p, like=A), _wrap(l, like=A), _wrap(u, like=A)
    except Exception:
        n = arr.shape[-1]
        m = arr.shape[-2]
        k = min(m, n)
        u = arr.copy().astype(np.float64)
        l = np.eye(m, k, dtype=np.float64)
        p = np.eye(m, dtype=np.float64)
        for i in range(min(m, n)):
            if pivot:
                max_row = i + np.argmax(np.abs(u[i:, i]))
                if max_row != i:
                    u[[i, max_row]] = u[[max_row, i]]
                    p[[i, max_row]] = p[[max_row, i]]
                    if i > 0:
                        l[[i, max_row], :i] = l[[max_row, i], :i]
            for j in range(i + 1, m):
                if abs(u[i, i]) > 1e-15:
                    factor = u[j, i] / u[i, i]
                    l[j, i] = factor
                    u[j, i:] -= factor * u[i, i:]
        return _wrap(p, like=A), _wrap(l, like=A), _wrap(u[:k, :], like=A)


def lu_factor(A: Tensor, pivot: bool = True) -> Tuple[Tensor, Tensor]:
    """Computes the compact LU factorization and pivot indices of a matrix."""
    arr = _to_np(A)
    try:
        import scipy.linalg as sla
        lu_mat, piv = sla.lu_factor(arr)
        return _wrap(lu_mat, like=A), _wrap(piv, like=A)
    except Exception:
        # No-scipy convention: pivots are a plain row-permutation array
        # (row i of P @ A is row piv[i] of A), unlike LAPACK's swap list.
        p, l, u = lu(A, pivot=pivot)
        p_arr, l_arr, u_arr = _to_np(p), _to_np(l), _to_np(u)
        lu_mat = u_arr + l_arr - np.eye(arr.shape[-1])
        piv = np.argmax(p_arr, axis=1).astype(np.int32)
        return _wrap(lu_mat, like=A), _wrap(piv, like=A)


def lu_solve(LU: Tensor, pivots: Tensor, B: Tensor, *, left: bool = True, out: Optional[Tensor] = None) -> Tensor:
    """Solves the system of linear equations A @ X = B given the LU factorization."""
    if isinstance(LU, Tensor) and isinstance(B, Tensor) and hasattr(LU, "triangular_solve"):
        n = LU.shape[-1]
        piv_np = np.asarray(_to_np(pivots)).astype(np.int64).reshape(-1)
        try:
            import scipy.linalg as _sla  # noqa: F401
            perm = list(range(n))
            for i, p in enumerate(piv_np):
                pi = int(p)
                perm[i], perm[pi] = perm[pi], perm[i]
        except Exception:
            perm = [int(v) for v in piv_np]
        p_mat_np = np.zeros((n, n), dtype=np.float64)
        for i, j in enumerate(perm):
            p_mat_np[i, j] = 1.0
        eye_n = from_numpy(np.eye(n, dtype=np.float64)).to_dtype(LU.dtype).to(device=LU.device)
        L = LU.tril(-1) + eye_n
        U = LU.triu(0)
        b_eff = B.to_dtype(LU.dtype) if B.dtype != LU.dtype else B
        p_mat = from_numpy(p_mat_np).to_dtype(LU.dtype).to(device=B.device)
        if left:
            pb = p_mat.matmul(b_eff) if b_eff.ndim >= 2 else p_mat.matmul(b_eff.unsqueeze(-1)).squeeze(-1)
            y = solve_triangular(L, pb, upper=False, unitriangular=True)
            res = solve_triangular(U, y, upper=True)
        else:
            bt = b_eff.transpose(-2, -1)
            y = U.triangular_solve(bt, False, True)
            z = L.triangular_solve(y, True, True)
            res = p_mat.transpose(-2, -1).matmul(z).transpose(-2, -1)
        if res.dtype != B.dtype:
            res = res.to_dtype(B.dtype)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    lu_arr = _to_np(LU)
    piv_arr = _to_np(pivots)
    b_arr = _to_np(B)
    try:
        import scipy.linalg as sla
        res_np = sla.lu_solve((lu_arr, piv_arr.astype(np.int32)), b_arr)
    except Exception:
        # Pairs with the no-scipy lu_factor above: pivots are a row
        # permutation with P @ A = L @ U and (P @ B)[i] = B[piv[i]].
        n = lu_arr.shape[-1]
        l = np.tril(lu_arr, -1) + np.eye(n)
        u = np.triu(lu_arr)
        piv = np.asarray(piv_arr).astype(np.int64).reshape(-1)
        if left:
            pb = b_arr[piv] if b_arr.ndim == 1 else b_arr[piv, :]
            res_np = np.linalg.solve(l @ u, pb)
        else:
            inv = np.empty_like(piv)
            inv[piv] = np.arange(n)
            a_full = (l @ u)[inv]
            res_np = np.swapaxes(np.linalg.solve(np.swapaxes(a_full, -2, -1), np.swapaxes(b_arr, -2, -1)), -2, -1)
    res = _wrap(res_np, like=B)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def solve_triangular(
    A: Tensor,
    B: Tensor,
    upper: bool,
    left: bool = True,
    unitriangular: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Solves a system of linear equations with a triangular coefficient matrix A and multiple right-hand sides B."""
    if isinstance(A, Tensor) and isinstance(B, Tensor) and hasattr(A, "triangular_solve"):
        a_eff = A
        if unitriangular:
            n = A.shape[-1]
            eye_n = from_numpy(np.eye(n, dtype=np.float64)).to(dtype=A.dtype, device=A.device)
            a_eff = (A.triu(1) if upper else A.tril(-1)) + eye_n
        b_1d = B.ndim == 1
        b_eff = B.unsqueeze(-1) if b_1d else B
        if left:
            res = a_eff.triangular_solve(b_eff, not upper, False)
        else:
            res = a_eff.triangular_solve(b_eff.transpose(-2, -1), not upper, True).transpose(-2, -1)
        if b_1d:
            res = res.squeeze(-1)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    a_arr = _to_np(A)
    b_arr = _to_np(B)
    try:
        import scipy.linalg as sla
        res_np = sla.solve_triangular(a_arr, b_arr, lower=not upper, unit_diagonal=unitriangular)
    except Exception:
        res_np = np.linalg.solve(a_arr, b_arr)
    res = _wrap(res_np, like=B)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def vecdot(x: Tensor, y: Tensor, dim: int = -1, *, out: Optional[Tensor] = None) -> Tensor:
    """Computes the dot product of two batches of vectors along dim."""
    if isinstance(x, Tensor) and isinstance(y, Tensor) and hasattr(x, "vecdot"):
        res = x.vecdot(y, dim=dim)
        if out is not None:
            out.zero_()
            out.add_(res)
            return out
        return res

    arr_x = _to_np(x)
    arr_y = _to_np(y)
    res_np = np.sum(arr_x * arr_y, axis=dim)
    res = _wrap(res_np, like=x)
    if out is not None:
        out.zero_()
        out.add_(res)
        return out
    return res


def vander(x: Tensor, N: Optional[int] = None, increasing: bool = False) -> Tensor:
    """Generates a Vandermonde matrix."""
    if isinstance(x, Tensor):
        n_cols = x.shape[-1] if N is None else int(N)
        if increasing:
            pows = np.arange(n_cols, dtype=np.float64)
        else:
            pows = np.arange(n_cols - 1, -1, -1, dtype=np.float64)
        p_t = from_numpy(pows).to(dtype=x.dtype, device=x.device)
        return x.unsqueeze(-1) ** p_t

    arr = _to_np(x)
    res_np = np.vander(arr, N=N, increasing=increasing)
    return _wrap(res_np, like=x)


def tensorinv(a: Tensor, ind: int = 2) -> Tensor:
    """Computes the multiplicative inverse of an N-dimensional tensor."""
    if isinstance(a, Tensor) and hasattr(a, "tensorinv"):
        return a.tensorinv(ind)
    arr = _to_np(a)
    res_np = np.linalg.tensorinv(arr, ind=ind)
    return _wrap(res_np, like=a)


def tensorsolve(a: Tensor, b: Tensor, dims: Optional[Sequence[int]] = None) -> Tensor:
    """Solves the tensor equation a x = b for x."""
    if isinstance(a, Tensor) and isinstance(b, Tensor) and dims is None and hasattr(a, "tensorsolve"):
        return a.tensorsolve(b)
    arr_a = _to_np(a)
    arr_b = _to_np(b)
    res_np = np.linalg.tensorsolve(arr_a, arr_b, axes=dims)
    return _wrap(res_np, like=b)


def householder_product(A: Tensor, tau: Tensor) -> Tensor:
    """Computes the first n columns of a product of Householder matrices."""
    return A.householder_product(tau)


__all__ = [
    "norm",
    "vector_norm",
    "matrix_norm",
    "inv",
    "pinv",
    "det",
    "slogdet",
    "cholesky",
    "cholesky_solve",
    "cholesky_inverse",
    "householder_product",
    "qr",
    "svd",
    "svdvals",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "solve",
    "solve_triangular",
    "lu",
    "lu_factor",
    "lu_solve",
    "lstsq",
    "matrix_power",
    "matrix_exp",
    "cross",
    "cond",
    "matrix_rank",
    "multi_dot",
    "diagonal",
    "vecdot",
    "vander",
    "tensorinv",
    "tensorsolve",
    "QRResult",
    "SVDResult",
    "SlogdetResult",
    "EigResult",
    "LstsqResult",
]
