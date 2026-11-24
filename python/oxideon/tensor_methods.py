"""PyTorch-style `Tensor` methods attached at import time: method forms of the module-level functions (`x.logsumexp(1)`,
`x.amax(0)`, `x.movedim(0, -1)`, ...), comparison and division aliases (`x.gt(y)`, `x.true_divide(y)`, ...), and in-place
variants (`x.exp_()`, `x.clamp_min_(0)`, `x.lerp_(y, 0.5)`, ...).

In-place methods follow PyTorch's rules: the result must keep the tensor's shape and must be castable to its dtype (an
integer tensor cannot take a float result), a tensor that requires grad is refused while gradients are enabled (wrap the
update in `no_grad`), and a non-contiguous view is refused because writing it would not reach its base tensor. Every
other handle to the tensor sees the new values.

Nothing is exported; importing the module adds the methods. `ADDED` lists the names this module attached.
"""

from typing import Any, Callable, List, Optional

import numpy as np
from ._contig import contig as _contig

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .primitives import is_grad_enabled as _is_grad_enabled

__all__: List[str] = []
ADDED: List[str] = []


def _attach(name: str, fn: Callable[..., Any]) -> None:
    """Adds `fn` as `Tensor.<name>` unless the class already has that attribute (native or patched elsewhere)."""
    if hasattr(_Tensor, name):
        return
    fn.__name__ = name
    fn.__qualname__ = f"Tensor.{name}"
    setattr(_Tensor, name, fn)
    ADDED.append(name)


def _np(x: Any) -> np.ndarray:
    return x.detach().to_numpy() if isinstance(x, _Tensor) else np.asarray(x)


def _t(a: Any) -> _Tensor:
    return _from_numpy(_contig(a))


def _is_float(x: _Tensor) -> bool:
    return x.dtype in ("f16", "bf16", "f32", "f64")


# ------------------------------------------------------------------------------------- method forms of module functions


def _method_forms() -> None:
    from . import tensor_utils as tu
    import oxideon as ox

    for name in ("lerp", "logsumexp", "movedim", "moveaxis", "swapaxes", "swapdims", "tensor_split", "hsplit", "vsplit",
                 "dsplit", "ravel", "msort", "float_power", "signbit", "gcd", "lcm", "ldexp", "frexp", "nextafter", "sgn",
                 "positive", "negative", "fix", "renorm", "scatter_reduce", "isneginf", "isposinf", "isreal", "pinverse", "ger"):
        f = getattr(tu, name)
        _attach(name, (lambda f: lambda self, *a, **k: f(self, *a, **k))(f))
    for name in ("amax", "amin", "outer", "inner", "dot", "vdot", "mv", "allclose", "isclose", "unique",
                 "unique_consecutive", "diagflat", "deg2rad", "rad2deg", "histogram"):
        f = getattr(ox, name)
        _attach(name, (lambda f: lambda self, *a, **k: f(self, *a, **k))(f))

    def where(self, condition, y):
        """`where(condition, self, y)`: the values of `self` where `condition` holds, of `y` elsewhere."""
        return ox.where(condition, self, y)
    _attach("where", where)

    def trace(self):
        """The sum of the main diagonal of a 2-D tensor."""
        if len(self.shape) != 2:
            raise RuntimeError("trace: expected a matrix (2-D tensor)")
        return ox.linalg.diagonal(self).sum()
    _attach("trace", trace)


# ---------------------------------------------------------------------------------------- comparisons, division, misc


def _new_methods() -> None:
    import oxideon as ox

    for name, op in (("eq", lambda a, b: a == b), ("ne", lambda a, b: a != b), ("lt", lambda a, b: a < b),
                     ("le", lambda a, b: a <= b), ("gt", lambda a, b: a > b), ("ge", lambda a, b: a >= b)):
        _attach(name, (lambda op, name: lambda self, other: op(self, other))(op, name))
    _attach("not_equal", lambda self, other: self != other)
    _attach("greater", lambda self, other: self > other)
    _attach("greater_equal", lambda self, other: self >= other)
    _attach("less", lambda self, other: self < other)
    _attach("less_equal", lambda self, other: self <= other)

    def equal(self, other) -> bool:
        """True when both tensors have the same shape and the same values."""
        return tuple(self.shape) == tuple(other.shape) and bool((self == other).all().item())
    _attach("equal", equal)

    def reciprocal(self):
        """`1 / self` (as a float tensor for integer input)."""
        x = self if _is_float(self) else self.float()
        return 1.0 / x
    _attach("reciprocal", reciprocal)

    def true_divide(self, other):
        """Division that always gives a floating-point result (integer tensors are converted first)."""
        a = self if _is_float(self) else self.float()
        b = other.float() if isinstance(other, _Tensor) and not _is_float(other) else other
        return a / b
    _attach("true_divide", true_divide)

    def floor_divide(self, other):
        """`floor(self / other)`, rounding towards minus infinity (also for integers, like Python's `//`). No graph."""
        if not _is_float(self) and not (isinstance(other, _Tensor) and _is_float(other)) and not isinstance(other, float):
            b = _np(other)
            if (b == 0).any():
                raise RuntimeError("ZeroDivisionError")
            return _t(np.floor_divide(_np(self), b).astype(_np(self).dtype))
        return (self / other).floor()
    _attach("floor_divide", floor_divide)

    def divide(self, other, *, rounding_mode: Optional[str] = None):
        """`self / other`: true division by default, or rounded with `rounding_mode` "trunc" or "floor"."""
        if rounding_mode is None:
            return true_divide(self, other)
        if rounding_mode == "floor":
            return floor_divide(self, other)
        if rounding_mode == "trunc":
            if not _is_float(self) and not (isinstance(other, _Tensor) and _is_float(other)) and not isinstance(other, float):
                q = np.trunc(_np(self) / _np(other))
                return _t(q.astype(_np(self).dtype))
            return (self / other).trunc()
        raise ValueError(f"rounding_mode must be None, 'trunc' or 'floor', not {rounding_mode!r}")
    _attach("divide", divide)
    _attach("multiply", lambda self, other: self * other)

    def subtract(self, other, *, alpha: float = 1.0):
        """`self - alpha * other`."""
        return self - other * alpha if alpha != 1.0 else self - other
    _attach("subtract", subtract)
    _attach("absolute", lambda self: self.abs())

    def dist(self, other, p: float = 2.0):
        """The p-norm of `self - other`."""
        d = (self - other).abs()
        if p == float("inf"):
            return d.max()
        if p == 0:
            return (d != 0).float().sum()
        return d.pow(float(p)).sum().pow(1.0 / p)
    _attach("dist", dist)

    def new_full(self, size, fill_value, *, dtype=None, requires_grad: bool = False):
        """A tensor of `size` filled with `fill_value`, with this tensor's dtype unless `dtype` is given."""
        return ox.full(list(size), fill_value, dtype=dtype if dtype is not None else self.dtype, requires_grad=requires_grad)
    _attach("new_full", new_full)

    def reshape_as(self, other):
        """`self.reshape(other.shape)`."""
        return self.reshape(list(other.shape))
    _attach("reshape_as", reshape_as)

    def is_same_size(self, other) -> bool:
        """Whether both tensors have the same shape."""
        return tuple(self.shape) == tuple(other.shape)
    _attach("is_same_size", is_same_size)

    def is_nonzero(self) -> bool:
        """The truth value of a one-element tensor (an error for any other size, like PyTorch)."""
        if self.numel() != 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous" if self.numel() else
                               "Boolean value of Tensor with no values is ambiguous")
        return bool(self.item() != 0)
    _attach("is_nonzero", is_nonzero)

    def element_size(self) -> int:
        """Bytes per element."""
        sizes = {"bool": 1, "u8": 1, "i8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8, "f64": 8}
        return sizes.get(self.dtype) or int(_np(self).dtype.itemsize)
    _attach("element_size", element_size)

    def sum_to_size(self, *size):
        """Sums `self` down to `size`, which `self` must be broadcastable from (the reverse of `expand`)."""
        shape = list(size[0]) if len(size) == 1 and not isinstance(size[0], int) else list(size)
        own = list(self.shape)
        lead = len(own) - len(shape)
        if lead < 0:
            raise RuntimeError(f"size {shape} is not expandable to {own}")
        out = self
        for _ in range(lead):
            out = out.sum(0)
        for i, (s, t) in enumerate(zip(shape, own[lead:])):
            if s == 1 and t != 1:
                out = out.sum(i, keepdim=True)
            elif s != t:
                raise RuntimeError(f"size {shape} is not expandable to {own}")
        return out
    _attach("sum_to_size", sum_to_size)

    def split_with_sizes(self, split_sizes, dim: int = 0):
        """Splits along `dim` into pieces of the given sizes, which must add up to the dimension's length."""
        if sum(split_sizes) != self.shape[dim]:
            raise RuntimeError(f"split_with_sizes expects split_sizes to sum exactly to {self.shape[dim]}, got {sum(split_sizes)}")
        return list(self.split(list(split_sizes), dim))
    _attach("split_with_sizes", split_with_sizes)


# ---------------------------------------------------------------------------------------------------------- in-place


def _check_inplace(self: _Tensor, name: str) -> None:
    if not self.is_contiguous():
        raise RuntimeError(f"{name}: in-place updates need a contiguous tensor (writing a view would not update its base)")
    if _is_grad_enabled() and self.requires_grad:
        raise RuntimeError(
            f"{name}: a tensor that requires grad cannot be updated in place while gradients are enabled; "
            "wrap the update in oxideon.no_grad()")


def _write(self: _Tensor, out: Any, name: str) -> _Tensor:
    if not isinstance(out, _Tensor):
        out = _t(np.full(tuple(self.shape), out, dtype=_np(self).dtype))
    if tuple(out.shape) != tuple(self.shape):
        raise RuntimeError(f"{name}: the result has shape {tuple(out.shape)}, which does not match the tensor's {tuple(self.shape)}")
    if _is_float(out) and not _is_float(self):
        raise RuntimeError(f"{name}: result type {out.dtype} can't be cast to the desired output type {self.dtype}")
    self.assign(out.detach())
    return self


def _inplace(name: str, compute: Callable[..., Any]) -> None:
    def method(self, *args, **kwargs):
        _check_inplace(self, name)
        return _write(self, compute(self, *args, **kwargs), name)
    method.__doc__ = f"In-place form of `{name[:-1]}`: overwrites this tensor with the result and returns it."
    _attach(name, method)


def _as_tensor(self: _Tensor, v: Any) -> Any:
    """A scalar operand as a tensor of `self`'s shape and dtype (for native ops that take only tensors)."""
    return v if isinstance(v, _Tensor) else _t(np.full(tuple(self.shape), v, dtype=_np(self).dtype))


# unary functions whose result is floating point in PyTorch, so their in-place form is refused on integer tensors
_FLOAT_VALUED = {"exp", "log", "log2", "log10", "log1p", "expm1", "sqrt", "rsqrt", "sigmoid", "tanh", "sin", "cos", "tan",
                 "asin", "acos", "atan", "sinh", "cosh", "erf", "erfinv", "digamma", "lgamma"}


def _inplace_methods() -> None:
    from . import tensor_utils as tu

    def unary(b):
        def compute(self):
            if b in _FLOAT_VALUED and not _is_float(self):
                raise RuntimeError(f"{b}_: result type Float can't be cast to the desired output type {self.dtype}")
            return getattr(self, b)()
        return compute

    for base in ("exp", "log", "log2", "log10", "log1p", "expm1", "sqrt", "rsqrt", "abs", "sign", "floor", "ceil", "trunc",
                 "frac", "square", "sigmoid", "tanh", "relu", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
                 "erf", "erfinv", "digamma", "lgamma"):
        if hasattr(_Tensor, base):
            _inplace(base + "_", unary(base))
    _inplace("absolute_", lambda self: self.abs())
    _inplace("neg_", lambda self: self.neg())
    _inplace("negative_", lambda self: self.neg())
    _inplace("sgn_", lambda self: self.sign())
    _inplace("fix_", lambda self: self.trunc())
    _inplace("round_", lambda self, decimals=0: self.round(decimals) if decimals else self.round())
    _inplace("reciprocal_", lambda self: _Tensor.reciprocal(self))
    _inplace("pow_", lambda self, exponent: self.pow(exponent))
    _inplace("clamp_min_", lambda self, min: self.clamp_min(min))
    _inplace("clamp_max_", lambda self, max: self.clamp_max(max))
    _inplace("lerp_", lambda self, end, weight: tu.lerp(self, end, weight))
    _inplace("addcmul_", lambda self, t1, t2, *, value=1.0: tu.addcmul(self, t1, t2, value=value))
    _inplace("addcdiv_", lambda self, t1, t2, *, value=1.0: tu.addcdiv(self, t1, t2, value=value))
    _inplace("remainder_", lambda self, other: self.remainder(_as_tensor(self, other)))
    _inplace("fmod_", lambda self, other: self.fmod(_as_tensor(self, other)))
    _inplace("atan2_", lambda self, other: self.atan2(_as_tensor(self, other)))
    _inplace("hypot_", lambda self, other: self.hypot(_as_tensor(self, other)))
    _inplace("copysign_", lambda self, other: self.copysign(_as_tensor(self, other)))
    _inplace("xlogy_", lambda self, other: self.xlogy(_as_tensor(self, other)))
    _inplace("logit_", lambda self, eps=None: self.logit(eps) if eps is not None else self.logit())
    _inplace("tril_", lambda self, diagonal=0: self.tril(diagonal))
    _inplace("triu_", lambda self, diagonal=0: self.triu(diagonal))
    _inplace("true_divide_", lambda self, other: _Tensor.true_divide(self, other))
    _inplace("divide_", lambda self, other, *, rounding_mode=None: _Tensor.divide(self, other, rounding_mode=rounding_mode))
    _inplace("multiply_", lambda self, other: self * other)
    _inplace("subtract_", lambda self, other, *, alpha=1.0: _Tensor.subtract(self, other, alpha=alpha))
    _inplace("floor_divide_", lambda self, other: _Tensor.floor_divide(self, other))
    _inplace("float_power_", lambda self, exponent: tu.float_power(self, exponent))
    _inplace("renorm_", lambda self, p, dim, maxnorm: tu.renorm(self, p, dim, maxnorm))
    _inplace("nextafter_", lambda self, other: tu.nextafter(self, other))
    _inplace("ldexp_", lambda self, other: tu.ldexp(self, other))
    for name, op in (("eq_", "__eq__"), ("ne_", "__ne__"), ("lt_", "__lt__"), ("le_", "__le__"), ("gt_", "__gt__"),
                     ("ge_", "__ge__")):
        # the comparison result (bool) is written in the tensor's own dtype: 1 where it holds, 0 elsewhere
        _inplace(name, (lambda op: lambda self, other: getattr(self, op)(other).to_dtype(self.dtype))(op))
    for name, op in (("logical_and_", "logical_and"), ("logical_or_", "logical_or"), ("logical_xor_", "logical_xor")):
        _inplace(name, (lambda op: lambda self, other: getattr(self, op)(other).to_dtype(self.dtype))(op))
    _inplace("logical_not_", lambda self: self.logical_not().to_dtype(self.dtype))
    for name in ("bitwise_and", "bitwise_or", "bitwise_xor"):
        _inplace(name + "_", (lambda n: lambda self, other: getattr(self, n)(other))(name))
    _inplace("bitwise_not_", lambda self: self.bitwise_not())


# ---------------------------------------------------------------------------- scalar operands and arithmetic operators


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (bool, int, float, np.generic)) and not isinstance(v, np.ndarray)


def _scalar_operands() -> None:
    """Native binary methods only take tensors; PyTorch also accepts a Python number (`x.add(2)`, `x.maximum(0.5)`)."""
    names = ("add", "sub", "mul", "div", "atan2", "hypot", "fmod", "remainder", "maximum", "minimum", "copysign", "xlogy",
             "fmax", "fmin", "heaviside", "logaddexp", "logaddexp2", "ldexp", "nextafter")
    for name in names:
        orig = getattr(_Tensor, name, None)
        if orig is None or getattr(orig, "_scalar_ok", False):
            continue

        def make(_orig):
            def wrapped(self, other, *a, **k):
                if _is_scalar(other):
                    # a float scalar on an integer tensor promotes to float32 (like PyTorch); otherwise keep self's dtype
                    dt = _np(self).dtype
                    if not _is_float(self) and isinstance(other, (float, np.floating)):
                        dt = np.float32
                        self = self.to_dtype("f32")
                    other = _from_numpy(np.full((), other, dtype=dt))
                return _orig(self, other, *a, **k)
            return wrapped

        wrapped = make(orig)
        wrapped.__name__ = name
        wrapped._scalar_ok = True
        setattr(_Tensor, name, wrapped)


def _operators() -> None:
    """`//`, `%`, `<<`, `>>`, unary `+`, true division of integer tensors (float result, like PyTorch) and in-place
    operators (`x += 1` writes into x instead of rebinding the name)."""
    truediv, rtruediv = _Tensor.__truediv__, _Tensor.__rtruediv__

    def _int_like(x: Any) -> bool:
        return (isinstance(x, _Tensor) and not _is_float(x) and x.dtype not in ("c64", "c128")) or isinstance(x, (bool, int))

    def __truediv__(self, other):
        if _int_like(self) and _int_like(other):
            return truediv(self.float(), other.float() if isinstance(other, _Tensor) else float(other))
        return truediv(self, other)

    def __rtruediv__(self, other):
        if _int_like(self) and _int_like(other):
            return rtruediv(self.float(), float(other))
        return rtruediv(self, other)

    _Tensor.__truediv__, _Tensor.__rtruediv__ = __truediv__, __rtruediv__

    def div(self, other, *, rounding_mode: Optional[str] = None):
        """`self / other` (a float result even for integer input); `rounding_mode` "trunc" or "floor" rounds the quotient."""
        return _Tensor.divide(self, other, rounding_mode=rounding_mode)
    div.__name__ = "div"
    div._scalar_ok = True
    _Tensor.div = div
    _Tensor.__floordiv__ = lambda self, other: self.floor_divide(other)
    _Tensor.__rfloordiv__ = lambda self, other: _as_tensor(self, other).floor_divide(self)
    _Tensor.__mod__ = lambda self, other: self.remainder(other)
    _Tensor.__rmod__ = lambda self, other: _as_tensor(self, other).remainder(self)
    _Tensor.__pos__ = lambda self: self
    _Tensor.__lshift__ = lambda self, n: self * (2 ** n)
    _Tensor.__rshift__ = lambda self, n: self.floor_divide(2 ** n)
    for dunder, method in (("__iadd__", "add_"), ("__isub__", "sub_"), ("__imul__", "mul_"), ("__itruediv__", "div_"),
                           ("__ifloordiv__", "floor_divide_"), ("__imod__", "remainder_")):
        if hasattr(_Tensor, method):
            def inplace(self, other, _m=method, _op=dunder.replace("__i", "__")):
                if (_is_grad_enabled() and self.requires_grad) or not self.is_contiguous():
                    # autograd (or a view) is involved: rebind to the out-of-place result, which keeps the graph correct
                    return getattr(self, _op)(other)
                getattr(self, _m)(other)
                return self
            setattr(_Tensor, dunder, inplace)


def _torch_signatures() -> None:
    """PyTorch call forms the native methods lack: varargs dims (`x.permute(2, 0, 1)`, `x.expand(3, 4)`, `x.flip(0, 1)`),
    `T` / `mT` / `H` / `mH`, `max` / `min` returning `(values, indices)` for a dim and the elementwise maximum for a
    tensor, named tuple results (`x.sort().values`), `scatter(dim, index, value)`, and `put`."""
    import collections

    def varargs(name, kw):
        orig = getattr(_Tensor, name)

        def method(self, *dims, **k):
            if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
                dims = tuple(dims[0])
            elif not dims and kw in k:
                dims = tuple(k.pop(kw))
            return orig(self, list(dims), **k)
        method.__name__ = name
        setattr(_Tensor, name, method)

    varargs("permute", "dims")
    orig_expand = _Tensor.expand

    def expand(self, *sizes, **k):
        if not sizes and "size" in k:
            sizes = tuple(k.pop("size"))
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = tuple(sizes[0])
        sizes = [int(v) for v in sizes]
        lead = len(sizes) - self.ndim
        if lead < 0:
            raise RuntimeError(f"expand: the number of sizes provided ({len(sizes)}) must be greater or equal to the number of dimensions in the tensor ({self.ndim})")
        for i, v in enumerate(sizes):
            if v == -1:
                if i < lead:
                    raise RuntimeError(f"expand: -1 is not allowed in a leading, non-existing dimension {i}")
                sizes[i] = int(self.shape[i - lead])
        return orig_expand(self, sizes)
    _Tensor.expand = expand

    orig_flip = _Tensor.flip

    def flip(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return orig_flip(self, list(dims))
    _Tensor.flip = flip

    _Tensor.T = property(lambda self: self.permute(list(range(self.ndim - 1, -1, -1))))
    _Tensor.mT = property(lambda self: self.transpose(-2, -1))
    _Tensor.H = property(lambda self: self.T.conj() if self.dtype in ("c64", "c128") else self.T)
    _Tensor.mH = property(lambda self: self.mT.conj() if self.dtype in ("c64", "c128") else self.mT)

    names = {"sort": ("values", "indices"), "topk": ("values", "indices"), "kthvalue": ("values", "indices"),
             "mode": ("values", "indices"), "median": ("values", "indices"), "nanmedian": ("values", "indices"),
             "cummax": ("values", "indices"), "cummin": ("values", "indices"), "aminmax": ("min", "max")}
    for name, fields in names.items():
        orig = getattr(_Tensor, name, None)
        if orig is None:
            continue
        nt = collections.namedtuple(name, fields)

        def make(orig, nt):
            def method(self, *a, **k):
                r = orig(self, *a, **k)
                return nt(*r) if isinstance(r, tuple) and len(r) == len(nt._fields) else r
            return method
        m = make(orig, nt)
        m.__name__ = name
        setattr(_Tensor, name, m)

    for name, arg in (("max", "argmax"), ("min", "argmin")):
        orig = getattr(_Tensor, name)
        pick = getattr(_Tensor, arg)
        nt = collections.namedtuple(name, ("values", "indices"))

        def make(orig, pick, name, nt):
            def method(self, dim=None, keepdim=False, *, other=None):
                if isinstance(dim, _Tensor):
                    return getattr(self, "maximum" if name == "max" else "minimum")(dim)
                if other is not None:
                    return getattr(self, "maximum" if name == "max" else "minimum")(other)
                if dim is None:
                    return orig(self)
                return nt(orig(self, dim=dim, keepdim=keepdim), pick(self, dim, keepdim))
            return method
        m = make(orig, pick, name, nt)
        m.__name__ = name
        setattr(_Tensor, name, m)

    orig_scatter = _Tensor.scatter

    def scatter(self, dim, index, src=None, *, value=None):
        src = value if src is None else src
        if not isinstance(src, _Tensor):
            src = _from_numpy(np.full(tuple(index.shape), src, dtype=_np(self).dtype))
        return orig_scatter(self, dim, index, src)
    _Tensor.scatter = scatter

    def put(self, index, source, accumulate=False):
        """Writes `source` into the flattened tensor at `index` (`accumulate=True` adds)."""
        flat, idx, src = self.reshape([-1]), index.reshape([-1]), source.reshape([-1])
        n = int(flat.shape[0])
        idx = _from_numpy(np.where(_np(idx) < 0, _np(idx) + n, _np(idx)).astype(np.int64))
        out = flat.scatter_add(0, idx, src) if accumulate else flat.scatter(0, idx, src)
        return out.reshape(list(self.shape))
    _attach("put", put)


def _reductions() -> None:
    """Reductions over several dims at once (`x.sum((0, 1))`), a differentiable `amax`/`amin` that shares the gradient
    between ties like PyTorch, `var`/`std` with `correction=`, and 0-d results for full `prod`/`all`/`any`."""
    import builtins
    from . import tensor_utils as tu

    def dims_of(x, dim):
        nd = max(x.ndim, 1)
        ds = [dim] if isinstance(dim, int) else list(dim)
        out = sorted({d + nd if d < 0 else d for d in ds})
        for d, raw in zip(out, sorted(ds)):
            if d < 0 or d >= nd:
                raise IndexError(f"Dimension out of range (expected to be in range of [{-nd}, {nd - 1}], but got {raw})")
        return out

    def squeeze_dims(x, ds):
        return x.reshape([int(s) for i, s in enumerate(x.shape) if i not in ds])

    def multi(name, native_axes=None):
        orig = getattr(_Tensor, name)

        def method(self, dim=None, keepdim=False, **k):
            if isinstance(dim, (list, tuple)):
                ds = dims_of(self, dim)
                if native_axes is not None:
                    return getattr(self, native_axes)(ds, keepdim=keepdim, **k)
                out = self
                for d in reversed(ds):  # highest first so the remaining dim numbers stay valid
                    out = orig(out, d, keepdim=keepdim, **k)
                return out
            if dim is None and not k:
                r = orig(self)
                if keepdim:
                    return r.reshape([1] * self.ndim)
                return r.reshape([]) if tuple(r.shape) == (1,) else r
            return orig(self, dim, keepdim=keepdim, **k)
        method.__name__ = name
        setattr(_Tensor, name, method)

    multi("sum", "sum_axes")
    multi("mean", "mean_axes")
    multi("prod")
    for name in ("all", "any"):
        orig = getattr(_Tensor, name)

        def make(orig):
            def method(self, dim=None, keepdim=False):
                if isinstance(dim, (list, tuple)):
                    out = self
                    for d in reversed(dims_of(self, dim)):
                        out = orig(out, d, keepdim)
                    return out
                if dim is None:
                    r = orig(self)
                    return r.reshape([1] * self.ndim) if keepdim else r.reshape([])
                return orig(self, dim, keepdim)
            return method
        m = make(orig)
        m.__name__ = name
        setattr(_Tensor, name, m)
    for name, native_axes in (("nansum", "nansum_axes"), ("nanmean", "nanmean_axes")):
        orig = getattr(_Tensor, name)

        def make(orig, native_axes):
            def method(self, dim=None, keepdim=False):
                if isinstance(dim, (list, tuple)):
                    return getattr(self, native_axes)(dims_of(self, dim), keepdim=keepdim)
                return orig(self) if dim is None and not keepdim else orig(self, dim=dim, keepdim=keepdim)
            return method
        m = make(orig, native_axes)
        m.__name__ = name
        setattr(_Tensor, name, m)

    def extremum(is_max):

        def method(self, dim=None, keepdim=False):
            ds = list(range(self.ndim)) if dim is None or (isinstance(dim, (list, tuple)) and not dim) else dims_of(self, dim)
            if self.ndim == 0:
                return self.clone()
            m = self.detach()
            for d in reversed(ds):
                m = (m.max(dim=d, keepdim=True).values if is_max else m.min(dim=d, keepdim=True).values)
            if not self.requires_grad:
                return m if keepdim else squeeze_dims(m, ds)
            tie = (self.detach() == m).to_dtype(self.dtype)
            w = tie / tie.sum_axes(ds, keepdim=True)
            s = (self * w).sum_axes(ds, keepdim=True)
            out = m + (s - s.detach())  # exact value, gradient shared evenly between the tied extremes
            return out if keepdim else squeeze_dims(out, ds)
        return method

    _Tensor.count_nonzero = lambda self, dim=None: tu.count_nonzero(self, dim)
    _Tensor.amax = extremum(True)
    _Tensor.amin = extremum(False)

    orig_var, orig_std = _Tensor.var, _Tensor.std

    def spread(orig, root):
        def method(self, dim=None, unbiased=None, keepdim=False, *, correction=None):
            if isinstance(dim, bool):  # x.var(unbiased)
                dim, unbiased = None, dim
            if correction is None and not isinstance(dim, (list, tuple)):
                return orig(self, dim, unbiased=True if unbiased is None else unbiased, keepdim=keepdim) if dim is not None \
                    else orig(self, unbiased=True if unbiased is None else unbiased)
            corr = (1 if unbiased is None or unbiased else 0) if correction is None else correction
            ds = list(range(self.ndim)) if dim is None else dims_of(self, dim)
            n = 1
            for d in ds:
                n *= int(self.shape[d])
            mu = self.mean_axes(ds, keepdim=True)
            v = ((self - mu) * (self - mu)).sum_axes(ds, keepdim=keepdim) / builtins.max(n - corr, 0)
            return v.sqrt() if root else v
        return method
    _Tensor.var = spread(orig_var, False)
    _Tensor.std = spread(orig_std, True)

    orig_lse = _Tensor.logsumexp

    def logsumexp(self, dim, keepdim=False):
        if not isinstance(dim, (list, tuple)):
            return orig_lse(self, dim, keepdim=keepdim)
        ds = dims_of(self, dim)
        m = _Tensor.amax(self.detach(), ds, keepdim=True)
        m = m.masked_fill(m.abs() == float("inf"), 0.0)
        out = (self - m).exp().sum_axes(ds, keepdim=True).log() + m
        return out if keepdim else squeeze_dims(out, ds)
    _Tensor.logsumexp = logsumexp


_scalar_operands()
_torch_signatures()
_method_forms()
_reductions()
_new_methods()
_inplace_methods()
_operators()
