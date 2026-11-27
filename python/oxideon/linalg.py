"""Comprehensive linear algebra suite for Oxideon.

Provides high-performance, batched, and exact linear algebra operations mirroring
standard deep learning primitives, including matrix decompositions, eigensystems,
factorizations, solvers, and generalized matrix/vector norms.
"""

from collections import namedtuple
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from ._contig import contig as _contig

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


def _as_t(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else from_numpy(_contig(np.asarray(x)))


def _wrap(arr: np.ndarray, like: Optional[Tensor] = None) -> Tensor:
    res = from_numpy(_contig(arr))
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

    A = _as_t(A)
    nd = len(A.shape)
    if dim is None and ord is None:
        res = vector_norm(A, 2, None, keepdim)
    elif dim is None:
        if nd == 1:
            res = vector_norm(A, ord, None, keepdim)
        elif nd == 2:
            res = matrix_norm(A, ord, (-2, -1), keepdim)
        else:
            raise RuntimeError(f"linalg.norm: with ord={ord!r} and dim=None the input must be 1-D or 2-D, got {nd}-D")
    elif isinstance(dim, int) or len(dim) == 1:
        res = vector_norm(A, 2 if ord is None else ord, dim if isinstance(dim, int) else dim[0], keepdim)
    elif len(dim) == 2:
        res = matrix_norm(A, "fro" if ord is None else ord, tuple(dim), keepdim)
    else:
        raise RuntimeError(f"linalg.norm: dim must be an int or a pair, got {dim}")
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
        if ord_f == 0.0 and x.requires_grad and not res.requires_grad:
            res = res + (x * 0.0).sum(dim=dims if dims else None, keepdim=keepdim)
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
    """The Moore-Penrose pseudo-inverse, from the SVD (or, with `hermitian`, the eigendecomposition), dropping singular
    values at or below max(atol, rtol * largest); like PyTorch, rtol defaults to eps * max(m, n) when atol is not given
    (and to 0 when it is). Computed in Rust."""
    A = _as_t(A)
    m, n = int(A.shape[-2]), int(A.shape[-1])
    eps = float(np.finfo(np.float64 if A.dtype == "f64" else np.float32).eps)
    if rtol is None:
        rtol = eps * max(m, n) if atol is None else 0.0
    atol = 0.0 if atol is None else float(atol)
    from .primitives import where
    if hermitian:
        w, v = A.eigh()
        mag = w.abs()
        cut = (mag.max(len(mag.shape) - 1, keepdim=True).values * float(rtol)).clamp_min(atol)
        inv = where(mag > cut, 1.0 / w, w * 0.0)
        res = (v * inv.unsqueeze(-2)).matmul(v.transpose(-2, -1))
    else:
        u, s, vh = A.svd()
        cut = (s.max(len(s.shape) - 1, keepdim=True).values * float(rtol)).clamp_min(atol)
        inv = where(s > cut, 1.0 / s, s * 0.0)
        res = (vh.transpose(-2, -1) * inv.unsqueeze(-2)).matmul(u.transpose(-2, -1))
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
    """QR decomposition of a matrix or batch of matrices (`mode` "reduced" or "complete"). Computed in Rust: by the
    differentiable Householder QR when `A` requires grad (reduced, m >= n), otherwise by LAPACK-convention Householder
    reflections (any shape; signs agree with NumPy and PyTorch)."""
    if mode not in ("reduced", "complete", "economic"):
        raise ValueError(f"unsupported qr mode {mode!r} (expected 'reduced', 'complete' or 'economic')")
    A = _as_t(A)
    if mode in ("reduced", "economic") and A.requires_grad and A.shape[-2] >= A.shape[-1]:
        q, r = A.qr()
    else:
        from .oxideon import _qr
        q, r = _qr(A, mode == "complete")
    if out is not None:
        out[0].zero_()
        out[0].add_(q)
        out[1].zero_()
        out[1].add_(r)
        return QRResult(out[0], out[1])
    return QRResult(q, r)


def svd(A: Tensor, full_matrices: bool = True, *, out: Optional[Tuple[Tensor, Tensor, Tensor]] = None) -> SVDResult:
    """Singular value decomposition A = U @ diag(S) @ Vh (computed in Rust). With `full_matrices` on a non-square
    matrix, U (or Vh) is completed to a square orthogonal matrix by a complete QR of its columns."""
    A = _as_t(A)
    u, s, vh = A.svd()
    m, n = int(A.shape[-2]), int(A.shape[-1])
    if full_matrices and m != n:
        from .oxideon import _qr, cat
        if m > n:
            q, _ = _qr(u.detach(), True)
            u = cat([u, q.narrow(len(q.shape) - 1, n, m - n)], len(u.shape) - 1)
        else:
            v = vh.transpose(-2, -1)
            q, _ = _qr(v.detach(), True)
            v = cat([v, q.narrow(len(q.shape) - 1, m, n - m)], len(v.shape) - 1)
            vh = v.transpose(-2, -1).contiguous()
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


def _complex_or_real(re: Tensor, im: Tensor) -> Tensor:
    """re + i im, as a real tensor when every imaginary part is zero (as the previous NumPy-based version did)."""
    if not bool((im != 0).any().item()):
        return re
    return _wrap(re.to_numpy() + 1j * im.to_numpy())


def eig(A: Tensor, *, out: Optional[Tuple[Tensor, Tensor]] = None) -> EigResult:
    """Eigenvalues and right (unit-norm) eigenvectors of a general square matrix, computed in Rust; complex results
    come back complex, real ones real."""
    wr, wi, vr, vi = _as_t(A).eig()
    w = _complex_or_real(wr, wi)
    v = _complex_or_real(vr, vi)
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
    """Eigenvalues of a general square matrix, computed in Rust (complex when any is complex)."""
    wr, wi = _as_t(A).eigvals()
    res = _complex_or_real(wr, wi)
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
        vec = B.ndim == 1 or (B.ndim == A.ndim - 1 and list(B.shape) == list(A.shape[:-1]))
        if left:
            if vec:
                res = A.solve(B.unsqueeze(-1)).squeeze(-1)
            else:
                res = A.solve(B)
        else:
            if vec:
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
            eye = from_numpy(_contig(np.eye(A.shape[-1], dtype=np.float32))).to(dtype=A.dtype, device=A.device).expand(A.shape)
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
            smax = s.max(dim=-1, keepdim=True).values
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


def _lu_packed(A: Tensor) -> Tensor:
    """`L - I + U` of a square matrix (batched) as a differentiable function of A (first order): the value is the Rust LU, and
    the gradient follows `dL = L tril(X, -1)`, `dU = triu(X) U` with `X = L^-1 P^T dA U^-1`, written with `dA = A - A.detach()`."""
    P, L, U = A.detach().lu()
    eye = from_numpy(np.eye(A.shape[-1], dtype=np.float32)).to(dtype=A.dtype)
    Linv = solve_triangular(L, eye, upper=False, unitriangular=True)
    Uinv = solve_triangular(U, eye, upper=True)
    X = Linv @ P.transpose(-2, -1) @ (A - A.detach()) @ Uinv
    return (L.tril(-1) + U) + L @ X.tril(-1) + X.triu() @ U


def _lu_is_differentiable(A: Tensor) -> bool:
    return A.requires_grad and A.ndim >= 2 and A.shape[-1] == A.shape[-2]


def lu(A: Tensor, pivot: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """The LU decomposition with partial pivoting, A = P @ L @ U (computed in Rust; differentiable for square matrices)."""
    if not pivot:
        raise NotImplementedError("lu: pivot=False is not supported")
    A = _as_t(A)
    if _lu_is_differentiable(A):
        P, _, _ = A.detach().lu()
        packed = _lu_packed(A)
        eye = from_numpy(np.eye(A.shape[-1], dtype=np.float32)).to(dtype=A.dtype)
        return P, packed.tril(-1) + eye, packed.triu()
    return A.lu()


def lu_factor(A: Tensor, pivot: bool = True) -> Tuple[Tensor, Tensor]:
    """The compact LU factorization with partial pivoting, as `torch.linalg.lu_factor`: `LU` holds L (unit diagonal,
    below) and U (on and above the diagonal), `pivots` the 1-based row interchanges (int32). Batched input is
    supported. Computed in Rust (LAPACK getrf)."""
    if not pivot:
        raise NotImplementedError("lu_factor: pivot=False is not supported")
    from .oxideon import _lu_factor_ex
    A = _as_t(A)
    lu, piv, _ = _lu_factor_ex(A)
    if _lu_is_differentiable(A):
        lu = _lu_packed(A)
    return lu, piv


def lu_solve(LU: Tensor, pivots: Tensor, B: Tensor, *, left: bool = True, out: Optional[Tensor] = None) -> Tensor:
    """Solves A X = B (or X A = B with left=False) from `lu_factor`'s result (1-based pivots, as in PyTorch).
    Computed in Rust (LAPACK getrs); batched factors are supported."""
    from .oxideon import _lu_solve
    res = _lu_solve(_as_t(LU), _as_t(pivots), _as_t(B), left)
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
        batch = np.broadcast_shapes(tuple(a_eff.shape[:-2]), tuple(b_eff.shape[:-2]))  # torch broadcasts the batch dims
        if tuple(a_eff.shape[:-2]) != batch:
            a_eff = a_eff.expand(list(batch) + list(a_eff.shape[-2:]))
        if tuple(b_eff.shape[:-2]) != batch:
            b_eff = b_eff.expand(list(batch) + list(b_eff.shape[-2:]))
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


def vander(x: Tensor, N: Optional[int] = None, increasing: bool = True) -> Tensor:
    """Generates a Vandermonde matrix with columns x**0, x**1, ... as `torch.linalg.vander` (`increasing=False` reverses them)."""
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


# ---------------------------------------------------------------------------------------------------------------------
# *_ex variants (error codes instead of exceptions), LDL^T (Bunch-Kaufman), matrix_sqrth, polar, matmul
# ---------------------------------------------------------------------------------------------------------------------

CholeskyExResult = namedtuple("CholeskyExResult", ["L", "info"])
InvExResult = namedtuple("InvExResult", ["inverse", "info"])
SolveExResult = namedtuple("SolveExResult", ["result", "info"])
LUFactorExResult = namedtuple("LUFactorExResult", ["LU", "pivots", "info"])
LDLFactorResult = namedtuple("LDLFactorResult", ["LD", "pivots"])
LDLFactorExResult = namedtuple("LDLFactorExResult", ["LD", "pivots", "info"])
PolarResult = namedtuple("PolarResult", ["U", "P"])


def _raise_if(info: Tensor, check: bool, message: str) -> None:
    if check and bool((info != 0).any().item()):
        raise RuntimeError(message)


def cholesky_ex(A: Any, *, upper: bool = False, check_errors: bool = False) -> CholeskyExResult:
    """Cholesky factorization that reports failure in `info` (0, or the order of the first leading minor that is not
    positive definite) instead of raising, unless `check_errors=True`. Computed in Rust (LAPACK potrf)."""
    from .oxideon import _cholesky_ex
    L, info = _cholesky_ex(_as_t(A), upper)
    _raise_if(info, check_errors, "linalg.cholesky_ex: the input is not positive-definite")
    return CholeskyExResult(L, info)


def inv_ex(A: Any, *, check_errors: bool = False) -> InvExResult:
    """Matrix inverse that reports a singular matrix in `info` (1 + the index of the zero pivot) instead of raising.
    Computed in Rust (LU)."""
    from .oxideon import _inv_ex
    inv, info = _inv_ex(_as_t(A))
    _raise_if(info, check_errors, "linalg.inv_ex: the matrix is singular")
    return InvExResult(inv, info)


def solve_ex(A: Any, B: Any, *, left: bool = True, check_errors: bool = False) -> SolveExResult:
    """Solves A X = B (or X A = B with left=False), reporting a singular A in `info` instead of raising (its solution
    is then NaN). Computed in Rust (LU)."""
    from .oxideon import _solve_ex
    x, info = _solve_ex(_as_t(A), _as_t(B), left)
    _raise_if(info, check_errors, "linalg.solve_ex: the matrix A is singular")
    return SolveExResult(x, info)


def lu_factor_ex(A: Any, *, pivot: bool = True, check_errors: bool = False) -> LUFactorExResult:
    """LU factorization with partial pivoting plus LAPACK's `info` (0, or 1 + the index of an exactly-zero pivot).
    Computed in Rust (LAPACK getrf)."""
    if not pivot:
        raise NotImplementedError("lu_factor_ex: pivot=False is not supported")
    from .oxideon import _lu_factor_ex
    lu, piv, info = _lu_factor_ex(_as_t(A))
    _raise_if(info, check_errors, "linalg.lu_factor_ex: U is singular")
    return LUFactorExResult(lu, piv, info)


def ldl_factor_ex(A: Any, *, hermitian: bool = False, check_errors: bool = False) -> LDLFactorExResult:
    """The Bunch-Kaufman factorization A = L D L^T of a symmetric (possibly indefinite) matrix, in LAPACK's compact
    form: `LD` holds L below the diagonal and the 1x1 / 2x2 blocks of D, `pivots` the (1-based) interchanges, negative
    for a 2x2 block. `info` is 0, or k when D[k-1, k-1] is exactly zero (D singular). Only the lower triangle of A is
    read. Computed in Rust (LAPACK sytrf)."""
    from .oxideon import _ldl_factor_ex
    ld, piv, info = _ldl_factor_ex(_as_t(A))
    _raise_if(info, check_errors, "linalg.ldl_factor_ex: the factor D is singular")
    return LDLFactorExResult(ld, piv, info)


def ldl_factor(A: Any, *, hermitian: bool = False) -> LDLFactorResult:
    """`ldl_factor_ex` without the error code (the factorization of a matrix with a singular D is still returned)."""
    LD, piv, _ = ldl_factor_ex(A, hermitian=hermitian)
    return LDLFactorResult(LD, piv)


def ldl_solve(LD: Any, pivots: Any, B: Any, *, hermitian: bool = False) -> Tensor:
    """Solves A X = B given the factorization of A from `ldl_factor`. Computed in Rust (LAPACK sytrs)."""
    from .oxideon import _ldl_solve
    return _ldl_solve(_as_t(LD), _as_t(pivots), _as_t(B))


def _per_matrix(A: Tensor, fn):
    """Applies fn to every matrix of a [..., n, n] tensor (fn runs Rust tensor operations) and restacks the results."""
    if len(A.shape) == 2:
        return fn(A)
    from .oxideon import stack
    lead = list(A.shape[:-2])
    flat = A.reshape([-1] + list(A.shape[-2:]))
    outs = [fn(flat[i]) for i in range(flat.shape[0])]
    if isinstance(outs[0], tuple):
        return tuple(stack([o[j] for o in outs], 0).reshape(lead + list(outs[0][j].shape)) for j in range(len(outs[0])))
    return stack(outs, 0).reshape(lead + list(outs[0].shape))


def matrix_sqrth(A: Any) -> Tensor:
    """The square root of a symmetric positive semi-definite matrix, V sqrt(max(w, 0)) V^T from its eigendecomposition
    (computed in Rust; negative eigenvalues are treated as zero)."""
    def one(m):
        w, v = m.eigh()
        return (v * w.clamp_min(0.0).sqrt().reshape([1, -1])).matmul(v.transpose(0, 1))
    return _per_matrix(_as_t(A), one)


def polar(A: Any) -> PolarResult:
    """The polar decomposition A = U P of a square matrix: U orthogonal, P symmetric positive semi-definite
    (computed in Rust)."""
    u, p = _per_matrix(_as_t(A), lambda m: m.polar())
    return PolarResult(u, p)


def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Matrix product (an alias of `Tensor.matmul`, as `torch.linalg.matmul`)."""
    return input.matmul(other)


__all__ += [
    "cholesky_ex", "inv_ex", "solve_ex", "lu_factor_ex", "ldl_factor", "ldl_factor_ex", "ldl_solve", "matrix_sqrth",
    "polar", "matmul", "CholeskyExResult", "InvExResult", "SolveExResult", "LUFactorExResult", "LDLFactorResult",
    "LDLFactorExResult", "PolarResult",
]
