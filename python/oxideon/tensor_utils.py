"""Tensor utilities with PyTorch-compatible names: shape helpers, elementwise math, reductions, logical and bitwise
operators, and top-level forms of linear-algebra and special functions.

Everything here works on `oxideon.Tensor` (and on anything `numpy.asarray` accepts). Functions built from tensor
operations stay differentiable; the ones that go through NumPy (integer and bit tricks, `histogramdd`, `scatter_reduce`,
...) return fresh tensors without a graph, and say so in their docstring.
"""

import builtins
import functools
import itertools
import math
from collections import namedtuple
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from ._contig import contig as _contig

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy

__all__ = [
    # shape and layout
    "ravel", "moveaxis", "movedim", "swapaxes", "swapdims", "tensor_split", "hsplit", "vsplit", "dsplit",
    "broadcast_tensors", "broadcast_to", "broadcast_shapes", "combinations", "msort",
    # elementwise math
    "lerp", "float_power", "signbit", "gcd", "lcm", "ldexp", "frexp", "nextafter", "sgn", "positive", "negative",
    "fix", "trunc", "frac", "square", "addcmul", "addcdiv", "lgamma", "mvlgamma", "logspace", "vander",
    # predicates and metadata
    "isnan", "isinf", "isfinite", "isneginf", "isposinf", "isreal", "is_tensor", "numel", "result_type",
    # reductions
    "logsumexp", "cumulative_logsumexp", "std_mean", "var_mean", "aminmax", "count_nonzero", "renorm",
    "scatter_reduce", "histogramdd",
    # linear algebra (top-level forms)
    "chain_matmul", "matrix_power", "matrix_exp", "pinverse", "cholesky_solve", "lu", "det", "inverse", "slogdet", "ger",
    # logical and bitwise
    "logical_and", "logical_or", "logical_xor", "logical_not", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
]


def _np(x: Any) -> np.ndarray:
    return x.detach().to_numpy() if isinstance(x, _Tensor) else np.asarray(x)


def _t(a: Any) -> _Tensor:
    return _from_numpy(_contig(a))


def _like(a: np.ndarray, ref: Any) -> _Tensor:
    """`a` as a tensor of the dtype of `ref` (a tensor), when `ref` is floating point."""
    return _t(a.astype(_np(ref).dtype, copy=False)) if isinstance(ref, _Tensor) else _t(a)


def _dim(d: int, ndim: int) -> int:
    if not -max(ndim, 1) <= d < max(ndim, 1):
        raise IndexError(f"dimension {d} out of range for a tensor with {ndim} dimensions")
    return d % ndim if ndim else 0


def _bcast2(a: Any, b: Any) -> Tuple[_Tensor, _Tensor]:
    """Both operands as tensors of their common shape (for the Rust kernels, which take one shape)."""
    ta = a if isinstance(a, _Tensor) else _t(np.asarray(a))
    tb = b if isinstance(b, _Tensor) else _t(np.asarray(b))
    shape = broadcast_shapes(tuple(ta.shape), tuple(tb.shape))
    if tuple(ta.shape) != shape:
        ta = ta.expand(list(shape)).contiguous()
    if tuple(tb.shape) != shape:
        tb = tb.expand(list(shape)).contiguous()
    return ta, tb


# ---------------------------------------------------------------------------------------------------------------- shape


def ravel(x: _Tensor) -> _Tensor:
    """Flattens `x` into one dimension (a view when `x` is contiguous)."""
    return x.reshape([-1])


def movedim(x: _Tensor, source: Union[int, Sequence[int]], destination: Union[int, Sequence[int]]) -> _Tensor:
    """Moves the dimensions in `source` to the positions in `destination` (the others keep their order)."""
    nd = len(x.shape)
    src = [_dim(s, nd) for s in ([source] if isinstance(source, int) else source)]
    dst = [_dim(d, nd) for d in ([destination] if isinstance(destination, int) else destination)]
    if len(src) != len(dst) or len(set(src)) != len(src) or len(set(dst)) != len(dst):
        raise ValueError("movedim needs matching, repeat-free source and destination dimensions")
    order = [i for i in range(nd) if i not in src]
    for d, s in sorted(zip(dst, src)):
        order.insert(d, s)
    return x.permute(order)


moveaxis = movedim


def swapaxes(x: _Tensor, axis0: int, axis1: int) -> _Tensor:
    """Swaps two dimensions."""
    return x.transpose(axis0, axis1)


swapdims = swapaxes


def tensor_split(x: _Tensor, indices_or_sections: Union[int, Sequence[int]], dim: int = 0) -> List[_Tensor]:
    """Splits `x` into pieces along `dim`: into N nearly equal ones (the first `n % N` are one longer) for an integer, or at
    the given indices for a sequence. Unlike `split`, the pieces need not have equal sizes."""
    n = x.shape[_dim(dim, len(x.shape))]
    if isinstance(indices_or_sections, int):
        if indices_or_sections <= 0:
            raise ValueError("number of sections must be larger than 0")
        q, r = divmod(n, indices_or_sections)
        sizes = [q + 1] * r + [q] * (indices_or_sections - r)
    else:
        # like NumPy: pieces x[0:i0], x[i0:i1], ..., x[ik:]; an index that goes backwards gives an empty piece
        cuts = [min(max(int(i) if i >= 0 else int(i) + n, 0), n) for i in indices_or_sections]
        starts, ends = [0] + cuts, cuts + [n]
        return [x.narrow(dim, a, max(e - a, 0)) for a, e in zip(starts, ends)]
    return list(x.split(sizes, dim))


def hsplit(x: _Tensor, indices_or_sections: Union[int, Sequence[int]]) -> List[_Tensor]:
    """`tensor_split` along dimension 1 (dimension 0 for a 1-D tensor)."""
    return tensor_split(x, indices_or_sections, 1 if len(x.shape) > 1 else 0)


def vsplit(x: _Tensor, indices_or_sections: Union[int, Sequence[int]]) -> List[_Tensor]:
    """`tensor_split` along dimension 0 (needs at least two dimensions)."""
    if len(x.shape) < 2:
        raise ValueError("vsplit needs a tensor with at least 2 dimensions")
    return tensor_split(x, indices_or_sections, 0)


def dsplit(x: _Tensor, indices_or_sections: Union[int, Sequence[int]]) -> List[_Tensor]:
    """`tensor_split` along dimension 2 (needs at least three dimensions)."""
    if len(x.shape) < 3:
        raise ValueError("dsplit needs a tensor with at least 3 dimensions")
    return tensor_split(x, indices_or_sections, 2)


def broadcast_shapes(*shapes: Sequence[int]) -> Tuple[int, ...]:
    """The shape that tensors of the given shapes broadcast to."""
    return tuple(int(d) for d in np.broadcast_shapes(*[tuple(s) if not isinstance(s, int) else (s,) for s in shapes]))


def broadcast_to(x: _Tensor, shape: Sequence[int]) -> _Tensor:
    """Expands `x` to `shape` following the broadcasting rules."""
    return x.expand([int(d) for d in shape])


def broadcast_tensors(*tensors: _Tensor) -> List[_Tensor]:
    """Broadcasts the tensors against each other and returns them all at the common shape."""
    shape = broadcast_shapes(*[tuple(t.shape) for t in tensors])
    return [broadcast_to(t, shape) for t in tensors]


def combinations(x: _Tensor, r: int = 2, with_replacement: bool = False) -> _Tensor:
    """All length-`r` combinations of the elements of the 1-D tensor `x`, as an (n_combinations, r) tensor."""
    if len(x.shape) != 1:
        raise ValueError("combinations expects a 1-D tensor")
    if r < 0:
        raise ValueError("r must be non-negative")
    pick = itertools.combinations_with_replacement if with_replacement else itertools.combinations
    idx = np.array(list(pick(range(x.shape[0]), r)), dtype=np.int64).reshape(-1, r)
    if idx.shape[0] == 0 or r == 0:
        return _t(np.empty((0, r) if r else (0,), dtype=_np(x).dtype))
    from .oxideon import stack
    return stack([x.index_select(0, _t(idx[:, j].copy())) for j in range(r)], 1)


def msort(x: _Tensor) -> _Tensor:
    """Sorts `x` along its first dimension."""
    return x.sort(0)[0]


# ---------------------------------------------------------------------------------------------------------- elementwise


def lerp(start: _Tensor, end: _Tensor, weight: Union[float, _Tensor]) -> _Tensor:
    """`start + weight * (end - start)`, evaluated from the nearer endpoint (so lerp(a, b, 1) is exactly b)."""
    diff = end - start
    if isinstance(weight, _Tensor):
        from .oxideon import where
        return where(weight.abs() < 0.5, start + weight * diff, end - diff * (1 - weight))
    return start + weight * diff if abs(weight) < 0.5 else end - diff * (1 - weight)


def float_power(x: Any, exponent: Any) -> _Tensor:
    """`x ** exponent` computed in float64, whatever the input dtypes."""
    base = x.double() if isinstance(x, _Tensor) else _t(np.asarray(x, dtype=np.float64))
    return base.pow(exponent.double() if isinstance(exponent, _Tensor) else float(exponent))


def signbit(x: _Tensor) -> _Tensor:
    """True where the sign bit is set (negative numbers, -0.0, -inf). Computed in Rust; no graph."""
    from .oxideon import _float_class
    return _float_class(x, "signbit")


def gcd(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise greatest common divisor of integer tensors (never negative). Computed in Rust; no graph."""
    from .oxideon import _gcd_lcm
    return _gcd_lcm(*_bcast2(a, b), False)


def lcm(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise least common multiple of integer tensors. Computed in Rust; no graph."""
    from .oxideon import _gcd_lcm
    return _gcd_lcm(*_bcast2(a, b), True)


def ldexp(x: _Tensor, other: Union[_Tensor, int]) -> _Tensor:
    """`x * 2 ** other`, exact for floating-point `x`. Computed in Rust; no graph."""
    from .oxideon import _ldexp
    xt, et = _bcast2(x, other)
    return _ldexp(xt, et.double() if et.dtype != "f64" else et)


def frexp(x: _Tensor) -> Tuple[_Tensor, _Tensor]:
    """Splits `x` into a mantissa in [0.5, 1) (with the sign of `x`) and an int32 exponent, so that x = mantissa * 2**exponent.
    Computed in Rust."""
    from .oxideon import _frexp
    return tuple(_frexp(x))


def nextafter(x: _Tensor, other: _Tensor) -> _Tensor:
    """The next representable floating-point value after `x` towards `other`. Computed in Rust; no graph."""
    from .oxideon import _nextafter
    xt, ot = _bcast2(x, other)
    return _nextafter(xt, ot)


def sgn(x: _Tensor) -> _Tensor:
    """The sign of `x` (for real tensors the same as `sign`)."""
    return x.sign()


def positive(x: _Tensor) -> _Tensor:
    """`x` unchanged (the unary plus)."""
    return x


def negative(x: _Tensor) -> _Tensor:
    """`-x`."""
    return x.neg()


def trunc(x: _Tensor) -> _Tensor:
    """Rounds towards zero."""
    return x.trunc()


fix = trunc


def frac(x: _Tensor) -> _Tensor:
    """The fractional part `x - trunc(x)`."""
    return x.frac()


def square(x: _Tensor) -> _Tensor:
    """`x * x`."""
    return x.square()


def addcmul(input: _Tensor, tensor1: _Tensor, tensor2: _Tensor, *, value: float = 1.0) -> _Tensor:
    """`input + value * tensor1 * tensor2`."""
    return input + (tensor1 * tensor2) * value


def addcdiv(input: _Tensor, tensor1: _Tensor, tensor2: _Tensor, *, value: float = 1.0) -> _Tensor:
    """`input + value * tensor1 / tensor2`."""
    return input + (tensor1 / tensor2) * value


def lgamma(x: _Tensor) -> _Tensor:
    """The natural log of the absolute value of the gamma function."""
    return x.lgamma()


def mvlgamma(x: _Tensor, p: int) -> _Tensor:
    """The multivariate log-gamma function of dimension `p`: p(p-1)/4 log(pi) + sum_{i<p} lgamma(x - i/2)."""
    if p < 1:
        raise ValueError("p has to be greater than or equal to 1")
    if bool((x <= (p - 1) / 2).any().item()):
        raise ValueError(f"all elements must be greater than (p-1)/2 = {(p - 1) / 2}")
    out = x.lgamma()
    for i in range(1, p):
        out = out + (x - i / 2).lgamma()
    return out + p * (p - 1) / 4 * math.log(math.pi)


def logspace(start: float, end: float, steps: int, base: float = 10.0, dtype: str = "float32") -> _Tensor:
    """`steps` numbers spaced evenly on a log scale from base**start to base**end."""
    from .primitives import linspace
    exps = linspace(float(start), float(end), int(steps)).double()  # computed in float64, then cast
    out = (exps * math.log(base)).exp()
    return out if dtype in ("float64", "f64") else out.float()


def vander(x: _Tensor, N: Optional[int] = None, increasing: bool = False) -> _Tensor:
    """The Vandermonde matrix of the 1-D tensor `x`: columns x**(N-1), ..., x**0 (reversed when `increasing`)."""
    if len(x.shape) != 1:
        raise ValueError("vander expects a 1-D tensor")
    n = x.shape[0] if N is None else N
    from .oxideon import stack
    powers = [x ** p for p in (range(n) if increasing else range(n - 1, -1, -1))]
    return stack(powers, 1) if powers else _t(np.empty((x.shape[0], 0), dtype=_np(x).dtype))


# ------------------------------------------------------------------------------------------------------------ predicates


def isnan(x: _Tensor) -> _Tensor:
    """True where `x` is NaN."""
    return x.isnan()


def isinf(x: _Tensor) -> _Tensor:
    """True where `x` is +inf or -inf."""
    return x.isinf()


def isfinite(x: _Tensor) -> _Tensor:
    """True where `x` is neither NaN nor infinite."""
    return x.isfinite()


def isposinf(x: _Tensor) -> _Tensor:
    """True where `x` is +inf."""
    from .oxideon import _float_class
    return _float_class(x, "posinf")


def isneginf(x: _Tensor) -> _Tensor:
    """True where `x` is -inf."""
    from .oxideon import _float_class
    return _float_class(x, "neginf")


def isreal(x: _Tensor) -> _Tensor:
    """True where the element is real (every element of a real tensor)."""
    return _t(np.ones(tuple(x.shape), dtype=bool))


def is_tensor(obj: Any) -> bool:
    """Whether `obj` is an `oxideon.Tensor`."""
    return isinstance(obj, _Tensor)


def numel(x: _Tensor) -> int:
    """The number of elements."""
    return int(x.numel())


def result_type(a: Any, b: Any) -> str:
    """The dtype that `a + b` has. Tensors promote like PyTorch (within a category the wider dtype, across categories the higher one); a Python scalar only changes the result when it is of a
    higher category (bool < int < float) than the tensor, then to int64 / float32."""
    def info(v):
        if isinstance(v, _Tensor):
            return _np(v).dtype, True
        if isinstance(v, bool):
            return np.dtype(bool), False
        if isinstance(v, int):
            return np.dtype(np.int64), False
        if isinstance(v, float):
            return np.dtype(np.float32), False
        arr = np.asarray(v)
        return arr.dtype, True

    (da, ta), (db, tb) = info(a), info(b)
    rank = {"b": 0, "u": 1, "i": 1, "f": 2, "c": 3}
    if ta and tb:
        # like PyTorch (not NumPy): int64 + float32 is float32, the dtype of the higher category wins
        out = np.promote_types(da, db) if rank[da.kind] == rank[db.kind] else (da if rank[da.kind] > rank[db.kind] else db)
    elif ta or tb:
        (dt, ds) = (da, db) if ta else (db, da)
        out = dt if rank[dt.kind] >= rank[ds.kind] else ds
    else:
        out = np.promote_types(da, db)
    return _from_numpy(np.zeros(1, dtype=out)).dtype


# ------------------------------------------------------------------------------------------------------------ reductions


def _dims(dim: Union[None, int, Sequence[int]], ndim: int) -> List[int]:
    if dim is None:
        return list(range(ndim))
    return sorted({_dim(d, ndim) for d in ([dim] if isinstance(dim, int) else dim)})


def logsumexp(x: _Tensor, dim: Union[int, Sequence[int], None] = None, keepdim: bool = False) -> _Tensor:
    """`log(sum(exp(x)))` over `dim` (all dimensions by default), without overflow."""
    nd = len(x.shape)
    out = x
    for d in reversed(_dims(dim, nd)):
        m = out.max(d, keepdim=True).values.detach()
        from .oxideon import where
        m = where(m.isfinite(), m, _t(np.zeros(tuple(m.shape), dtype=_np(m).dtype)))  # an infinite max would give inf - inf
        out = ((out - m).exp().sum(d, keepdim=True)).log() + m
    return out if keepdim else _squeeze_dims(out, _dims(dim, nd))


def _squeeze_dims(x: _Tensor, dims: List[int]) -> _Tensor:
    shape = [s for i, s in enumerate(x.shape) if i not in set(dims)]
    return x.reshape(shape)


def cumulative_logsumexp(x: _Tensor, dim: int) -> _Tensor:
    """The running `logsumexp` along `dim` (an alias of `logcumsumexp`)."""
    return x.logcumsumexp(dim)


def _var_args(dim, correction, unbiased):
    if unbiased is None:
        unbiased = True if correction is None else bool(correction)
    return unbiased


def var_mean(x: _Tensor, dim: Union[int, Sequence[int], None] = None, unbiased: Optional[bool] = None,
             keepdim: bool = False, correction: Optional[int] = None) -> Tuple[_Tensor, _Tensor]:
    """The variance and the mean of `x` over `dim` (all elements by default). `correction=0` is the biased estimate."""
    unb = _var_args(dim, correction, unbiased)
    if dim is None:
        v = x.var(unbiased=unb)
        m = x.mean()
        if keepdim:
            v, m = v.reshape([1] * len(x.shape)), m.reshape([1] * len(x.shape))
        return v, m
    out_v, out_m = x, x
    dims = _dims(dim, len(x.shape))
    if len(dims) == 1:
        return x.var(dims[0], unbiased=unb, keepdim=keepdim), x.mean(dims[0], keepdim=keepdim)
    # several dimensions: move them to the end, merge, and reduce once
    keep = [i for i in range(len(x.shape)) if i not in dims]
    xt = x.permute(keep + dims)
    lead = [x.shape[i] for i in keep]
    flat = xt.reshape(lead + [-1])
    last = len(lead)
    v, m = flat.var(last, unbiased=unb), flat.mean(last)
    if keepdim:
        shape = [1 if i in dims else x.shape[i] for i in range(len(x.shape))]
        v, m = v.reshape(shape), m.reshape(shape)
    return v, m


def std_mean(x: _Tensor, dim: Union[int, Sequence[int], None] = None, unbiased: Optional[bool] = None,
             keepdim: bool = False, correction: Optional[int] = None) -> Tuple[_Tensor, _Tensor]:
    """The standard deviation and the mean of `x` over `dim` (all elements by default)."""
    v, m = var_mean(x, dim, unbiased, keepdim, correction)
    return v.sqrt(), m


AminmaxResult = namedtuple("AminmaxResult", ["min", "max"])


def aminmax(x: _Tensor, *, dim: Optional[int] = None, keepdim: bool = False) -> AminmaxResult:
    """The minimum and the maximum of `x` (over `dim`, or of all elements)."""
    if dim is None:
        lo, hi = x.min(), x.max()
        if keepdim:
            lo, hi = lo.reshape([1] * len(x.shape)), hi.reshape([1] * len(x.shape))
        return AminmaxResult(lo, hi)
    return AminmaxResult(x.min(dim, keepdim=keepdim).values, x.max(dim, keepdim=keepdim).values)


def count_nonzero(x: _Tensor, dim: Union[int, Sequence[int], None] = None) -> _Tensor:
    """The number of non-zero elements (as int64), over `dim` or in total."""
    nz = (x != 0).long()
    if dim is None:
        return nz.sum()
    for d in sorted({_dim(v, len(x.shape)) for v in ([dim] if isinstance(dim, int) else dim)}, reverse=True):
        nz = nz.sum(d)
    return nz


def renorm(x: _Tensor, p: float, dim: int, maxnorm: float) -> _Tensor:
    """Rescales every sub-tensor `x.select(dim, i)` whose p-norm exceeds `maxnorm` to have norm `maxnorm`."""
    d = _dim(dim, len(x.shape))
    other = [i for i in range(len(x.shape)) if i != d]
    norms = x.abs().pow(float(p))
    for i in reversed(other):
        norms = norms.sum(i, keepdim=True)
    norms = norms.pow(1.0 / p)
    from .oxideon import where
    scale = where(norms > maxnorm, maxnorm / (norms + 1e-7), norms * 0 + 1)
    return x * scale


def scatter_reduce(x: _Tensor, dim: int, index: _Tensor, src: _Tensor, reduce: str, *, include_self: bool = True) -> _Tensor:
    """Reduces the values of `src` into a copy of `x` at the positions in `index` along `dim`, with `reduce` one of
    "sum", "prod", "mean", "amax", "amin". With `include_self=False` the original values of `x` at the touched positions
    do not take part. Computed in Rust; "sum" and "mean" are differentiable. `src` may be larger than `index` (only the
    part `index` covers is used, as in PyTorch)."""
    if reduce not in ("sum", "prod", "mean", "amax", "amin"):
        raise ValueError(f"unknown reduce {reduce!r}")
    from .oxideon import _scatter_reduce
    idx = index if isinstance(index, _Tensor) else _t(np.asarray(index, dtype=np.int64))
    s = src if isinstance(src, _Tensor) else _t(np.asarray(src))
    if tuple(s.shape) != tuple(idx.shape):
        if len(s.shape) != len(idx.shape) or any(i > j for i, j in zip(idx.shape, s.shape)):
            raise ValueError(f"scatter_reduce: index {tuple(idx.shape)} must not be larger than src {tuple(s.shape)} in any dimension")
        for d, n in enumerate(idx.shape):
            s = s.narrow(d, 0, n)
        s = s.contiguous()
    return _scatter_reduce(x, dim, idx.long() if idx.dtype != "i64" else idx, s, reduce, include_self)


def histogramdd(sample: _Tensor, bins: Union[int, Sequence[int]] = 10, *, range: Optional[Sequence[float]] = None,
                weight: Optional[_Tensor] = None, density: bool = False) -> Tuple[_Tensor, List[_Tensor]]:
    """A multidimensional histogram of the (N, D) tensor `sample`. Returns the counts and the bin edges of every dimension.
    `range` is a flat sequence (min0, max0, min1, max1, ...). Computed in Rust; no graph."""
    from .oxideon import _histogramdd
    s = sample if isinstance(sample, _Tensor) else _t(np.asarray(sample))
    d = int(s.shape[1])
    nb = [int(bins)] * d if isinstance(bins, int) else [int(v) for v in bins]
    rng = None if range is None else [(float(range[2 * i]), float(range[2 * i + 1])) for i in builtins.range(d)]
    w = None if weight is None else (weight if isinstance(weight, _Tensor) else _t(np.asarray(weight))).reshape([-1])
    h, edges = _histogramdd(s, nb, rng, w, bool(density))
    return h, list(edges)



# ------------------------------------------------------------------------------------------------------------ linalg


def chain_matmul(*matrices: _Tensor) -> _Tensor:
    """The product of two or more matrices, multiplied left to right."""
    if len(matrices) < 1:
        raise ValueError("chain_matmul needs at least one matrix")
    return functools.reduce(lambda a, b: a.matmul(b), matrices)


def matrix_power(x: _Tensor, n: int) -> _Tensor:
    """`x` multiplied by itself `n` times (negative `n`: the inverse's power); `x` is a square matrix or a batch of them."""
    from . import linalg
    return linalg.matrix_power(x, n)


def matrix_exp(x: _Tensor) -> _Tensor:
    """The matrix exponential of a square matrix (or a batch)."""
    from . import linalg
    return linalg.matrix_exp(x)


def pinverse(x: _Tensor, rcond: float = 1e-15) -> _Tensor:
    """The Moore-Penrose pseudo-inverse."""
    from . import linalg
    return linalg.pinv(x)


def cholesky_solve(b: _Tensor, L: _Tensor, upper: bool = False) -> _Tensor:
    """Solves `A X = B` for symmetric positive-definite `A` given its Cholesky factor `L` (lower, or upper with `upper=True`)."""
    from . import linalg
    return linalg.cholesky_solve(b, L, upper=upper) if upper else linalg.cholesky_solve(b, L)


def lu(x: _Tensor):
    """The LU factorisation with partial pivoting: `(P, L, U)` with `P @ L @ U == x`."""
    from . import linalg
    return linalg.lu(x)


def det(x: _Tensor) -> _Tensor:
    """The determinant of a square matrix (or a batch)."""
    from . import linalg
    return linalg.det(x)


def inverse(x: _Tensor) -> _Tensor:
    """The inverse of a square matrix (or a batch)."""
    from . import linalg
    return linalg.inv(x)


def slogdet(x: _Tensor):
    """The sign and the natural log of the absolute determinant."""
    from . import linalg
    return linalg.slogdet(x)


def ger(a: _Tensor, b: _Tensor) -> _Tensor:
    """The outer product of two vectors (an alias of `outer`)."""
    return a.reshape([-1, 1]).matmul(b.reshape([1, -1]))


# ------------------------------------------------------------------------------------------------------ logical, bitwise


def logical_and(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise logical AND."""
    return a.logical_and(b)


def logical_or(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise logical OR."""
    return a.logical_or(b)


def logical_xor(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise logical XOR."""
    return a.logical_xor(b)


def logical_not(a: _Tensor) -> _Tensor:
    """Elementwise logical NOT."""
    return a.logical_not()


def bitwise_and(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise bitwise AND of integer or bool tensors."""
    return a.bitwise_and(b)


def bitwise_or(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise bitwise OR of integer or bool tensors."""
    return a.bitwise_or(b)


def bitwise_xor(a: _Tensor, b: _Tensor) -> _Tensor:
    """Elementwise bitwise XOR of integer or bool tensors."""
    return a.bitwise_xor(b)


def bitwise_not(a: _Tensor) -> _Tensor:
    """Elementwise bitwise NOT of an integer or bool tensor."""
    return a.bitwise_not()
