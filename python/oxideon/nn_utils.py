"""Neural network structural and parameterization utilities for Oxideon.

Provides:
- Weight normalization: `weight_norm`, `remove_weight_norm`
- Spectral normalization: `spectral_norm`, `remove_spectral_norm`
- Convolution / Linear + BatchNorm operator fusion: `fuse_conv_batchnorm_weights`,
  `fuse_conv_batchnorm`, `fuse_linear_batchnorm_weights`, `fuse_linear_batchnorm`
  (also grouped under `nn.utils.fusion`)
- Temporary weight substitution: `apply_with_weights`
- Generic tensor reparameterizations: `reparam` (`bind_reparam`, `unbind_reparam`,
  `has_reparam`, `cache_reparam`)
- Structured and mask-based pruning helpers under `nn.utils.prune`
- Aggregate norm computation: `total_norm`
- Validated layer construction: `fast_init`
"""

from contextlib import contextmanager
import copy
import math
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .primitives import no_grad as _no_grad, tensor as _make_tensor


# ---------------------------------------------------------------------------
# Native Layer Hook Adapter
# ---------------------------------------------------------------------------

def _ensure_hookable(module: Any) -> Any:
    """Returns `module` directly if it is an `nn.Module`, or wraps a native C/Rust layer
    so parameter registration, buffer registration, and forward pre-hooks work seamlessly.
    """
    from .nn import Module, Parameter

    if isinstance(module, Module):
        return module

    class _HookableNativeModule(Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            object.__setattr__(self, "_native_inner", inner)
            if hasattr(inner, "weight") and inner.weight is not None:
                self.register_parameter("weight", Parameter(inner.weight.detach()))
            if hasattr(inner, "bias"):
                b = inner.bias
                if b is not None:
                    self.register_parameter("bias", Parameter(b.detach()))
                else:
                    object.__setattr__(self, "bias", None)

        def train(self, mode: bool = True):
            super().train(mode)
            inner = object.__getattribute__(self, "_native_inner")
            if hasattr(inner, "train"):
                inner.train(mode)
            return self

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            inner = object.__getattribute__(self, "_native_inner")
            w = getattr(self, "weight", None)
            if w is not None and hasattr(inner, "weight"):
                inner.weight = w
            b = getattr(self, "bias", None)
            if hasattr(inner, "bias") and b is not None:
                inner.bias = b
            return inner(*args, **kwargs)

        def __getattr__(self, item: str) -> Any:
            inner = object.__getattribute__(self, "_native_inner")
            return getattr(inner, item)

    return _HookableNativeModule(module)


def _set_computed_attr(module: Any, name: str, value: _Tensor) -> None:
    """Assigns a computed non-leaf Tensor attribute onto `module` without re-registering it
    as a leaf entry in `module._parameters`.
    """
    if hasattr(module, "_parameters") and name in module._parameters:
        module._parameters.pop(name, None)
    object.__setattr__(module, name, value)


# ---------------------------------------------------------------------------
# Aggregate Norm (`total_norm`)
# ---------------------------------------------------------------------------

def total_norm(
    tensors: Union[_Tensor, Iterable[_Tensor]],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> _Tensor:
    """Computes the joint Lp norm across an iterable of tensors treated as a single concatenated vector."""
    if isinstance(tensors, _Tensor):
        tensor_list = [tensors]
    else:
        tensor_list = [t for t in tensors if t is not None]

    norm_type = float(norm_type)
    if not tensor_list:
        return _make_tensor(0.0, dtype="float32")

    if math.isnan(norm_type) or (norm_type <= 0.0 and norm_type != float("inf")):
        raise ValueError(f"total_norm requires norm_type > 0 or inf, got {norm_type}")

    with _no_grad():
        if norm_type == float("inf"):
            val = max(float(t.abs().max().item()) for t in tensor_list)
        else:
            acc = 0.0
            for t in tensor_list:
                acc += float(t.abs().powf(norm_type).sum().item())
            val = acc ** (1.0 / norm_type)

    if error_if_nonfinite and (math.isnan(val) or math.isinf(val)):
        raise RuntimeError(f"Total norm of order {norm_type} is non-finite ({val}).")
    return _make_tensor(val, dtype="float32")


# ---------------------------------------------------------------------------
# Weight Normalization (`weight_norm` / `remove_weight_norm`)
# ---------------------------------------------------------------------------

def _norm_except_dim(v: _Tensor, dim: Optional[int], eps: float = 1e-12) -> _Tensor:
    ndim = len(v.shape)
    if dim is None:
        sq_sum = v.square().sum()
        norm = (sq_sum + eps).sqrt()
        return norm.reshape([1] * ndim) if ndim > 0 else norm

    d = dim if dim >= 0 else dim + ndim
    if d < 0 or d >= ndim:
        raise IndexError(f"dim {dim} out of range for weight of rank {ndim}")

    reduce_axes = [i for i in range(ndim) if i != d]
    sq = v.square()
    for ax in sorted(reduce_axes, reverse=True):
        sq = sq.sum(ax, keepdim=True)
    return (sq + eps).sqrt()


class _WeightNormHook:
    def __init__(self, name: str, dim: Optional[int]) -> None:
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Any) -> _Tensor:
        g = getattr(module, f"{self.name}_g")
        v = getattr(module, f"{self.name}_v")
        norm = _norm_except_dim(v, self.dim)
        return v * (g / norm)

    def __call__(self, module: Any, inputs: Any) -> None:
        del inputs
        _set_computed_attr(module, self.name, self.compute_weight(module))


def weight_norm(module: Any, name: str = "weight", dim: Optional[int] = 0) -> Any:
    """Applies decoupled magnitude-direction weight normalization `w = g * (v / ||v||)` to `module.<name>`."""
    from .nn import Parameter

    module = _ensure_hookable(module)
    if not hasattr(module, name) or getattr(module, name) is None:
        raise ValueError(f"Module {type(module).__name__} has no parameter named '{name}'")

    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, _WeightNormHook) and hook.name == name:
            raise RuntimeError(f"Cannot register two weight_norm hooks on the same parameter '{name}'")

    weight = getattr(module, name)
    with _no_grad():
        g_init = _norm_except_dim(weight.detach(), dim)
        v_init = weight.detach().clone()

    delattr(module, name)
    module.register_parameter(f"{name}_g", Parameter(g_init))
    module.register_parameter(f"{name}_v", Parameter(v_init))

    fn = _WeightNormHook(name, dim)
    _set_computed_attr(module, name, fn.compute_weight(module))
    module.register_forward_pre_hook(fn)
    return module


def remove_weight_norm(module: Any, name: str = "weight") -> Any:
    """Removes weight normalization from `module.<name>` and materializes the combined weight parameter."""
    from .nn import Parameter

    if not hasattr(module, "_forward_pre_hooks"):
        raise ValueError(f"weight_norm of '{name}' not found in {module}")

    for k, hook in list(module._forward_pre_hooks.items()):
        if isinstance(hook, _WeightNormHook) and hook.name == name:
            with _no_grad():
                w = hook.compute_weight(module).detach().clone()
            del module._forward_pre_hooks[k]
            delattr(module, f"{name}_g")
            delattr(module, f"{name}_v")
            module.register_parameter(name, Parameter(w))
            return module

    raise ValueError(f"weight_norm of '{name}' not found in {module}")


# ---------------------------------------------------------------------------
# Spectral Normalization (`spectral_norm` / `remove_spectral_norm`)
# ---------------------------------------------------------------------------

def _l2_normalize_1d(vec: _Tensor, eps: float = 1e-12) -> _Tensor:
    norm = (vec.square().sum() + eps).sqrt()
    return vec / norm


class _SpectralNormHook:
    def __init__(self, name: str, n_power_iterations: int, dim: int, eps: float) -> None:
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps

    def _reshape_weight_to_matrix(self, weight: _Tensor) -> _Tensor:
        ndim = len(weight.shape)
        if ndim == 1:
            return weight.reshape([1, int(weight.shape[0])])
        w = weight
        if self.dim != 0:
            perm = [self.dim] + [i for i in range(ndim) if i != self.dim]
            w = w.permute(perm).contiguous()
        height = int(w.shape[0])
        return w.reshape([height, -1])

    def compute_weight(self, module: Any, do_power_iteration: bool) -> _Tensor:
        weight = getattr(module, f"{self.name}_orig")
        u = getattr(module, f"{self.name}_u")
        v = getattr(module, f"{self.name}_v")
        weight_mat = self._reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with _no_grad():
                w_det = weight_mat.detach()
                wt_det = w_det.transpose(0, 1)
                for _ in range(self.n_power_iterations):
                    v_new = _l2_normalize_1d(wt_det.matmul(u.reshape([-1, 1])).reshape([-1]), self.eps)
                    u_new = _l2_normalize_1d(w_det.matmul(v_new.reshape([-1, 1])).reshape([-1]), self.eps)
                    v = v_new
                    u = u_new
                module._buffers[f"{self.name}_u"] = u
                module._buffers[f"{self.name}_v"] = v
                object.__setattr__(module, f"{self.name}_u", u)
                object.__setattr__(module, f"{self.name}_v", v)

        sigma = u.detach().reshape([1, -1]).matmul(weight_mat).matmul(v.detach().reshape([-1, 1])).reshape([])
        return weight / (sigma + self.eps)

    def __call__(self, module: Any, inputs: Any) -> None:
        del inputs
        _set_computed_attr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=bool(getattr(module, "training", True))),
        )


def spectral_norm(
    module: Any,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: Optional[int] = None,
) -> Any:
    """Normalizes `module.<name>` by its largest singular value estimated via power iteration."""
    from .nn import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Parameter

    if n_power_iterations < 1:
        raise ValueError(f"n_power_iterations must be positive, got {n_power_iterations}")

    module = _ensure_hookable(module)
    if not hasattr(module, name) or getattr(module, name) is None:
        raise ValueError(f"Module {type(module).__name__} has no parameter named '{name}'")

    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, _SpectralNormHook) and hook.name == name:
            raise RuntimeError(f"Cannot register two spectral_norm hooks on the same parameter '{name}'")

    if dim is None:
        dim = 1 if isinstance(module, (ConvTranspose1d, ConvTranspose2d, ConvTranspose3d)) else 0

    weight = getattr(module, name)
    ndim = len(weight.shape)
    d = dim if dim >= 0 else dim + ndim
    if ndim > 1 and (d < 0 or d >= ndim):
        raise IndexError(f"dim {dim} out of range for weight of rank {ndim}")

    fn = _SpectralNormHook(name, n_power_iterations, d, eps)
    with _no_grad():
        w_mat = fn._reshape_weight_to_matrix(weight.detach())
        h, w_cols = int(w_mat.shape[0]), int(w_mat.shape[1])
        rng = np.random.default_rng(0)
        u_init = _l2_normalize_1d(_from_numpy(rng.standard_normal(h).astype(np.float32)), eps)
        v_init = _l2_normalize_1d(_from_numpy(rng.standard_normal(w_cols).astype(np.float32)), eps)
        orig = weight.detach().clone()

    delattr(module, name)
    module.register_parameter(f"{name}_orig", Parameter(orig))
    module.register_buffer(f"{name}_u", u_init)
    module.register_buffer(f"{name}_v", v_init)

    _set_computed_attr(module, name, fn.compute_weight(module, do_power_iteration=True))
    module.register_forward_pre_hook(fn)
    return module


def remove_spectral_norm(module: Any, name: str = "weight") -> Any:
    """Removes spectral normalization from `module.<name>` and materializes the normalized weight."""
    from .nn import Parameter

    if not hasattr(module, "_forward_pre_hooks"):
        raise ValueError(f"spectral_norm of '{name}' not found in {module}")

    for k, hook in list(module._forward_pre_hooks.items()):
        if isinstance(hook, _SpectralNormHook) and hook.name == name:
            with _no_grad():
                w = hook.compute_weight(module, do_power_iteration=False).detach().clone()
            del module._forward_pre_hooks[k]
            delattr(module, f"{name}_orig")
            delattr(module, f"{name}_u")
            delattr(module, f"{name}_v")
            module.register_parameter(name, Parameter(w))
            return module

    raise ValueError(f"spectral_norm of '{name}' not found in {module}")


# ---------------------------------------------------------------------------
# Convolution / Linear + BatchNorm Operator Fusion
# ---------------------------------------------------------------------------

def fuse_conv_batchnorm_weights(
    conv_w: _Tensor,
    conv_b: Optional[_Tensor],
    bn_rm: _Tensor,
    bn_rv: _Tensor,
    bn_eps: float,
    bn_w: Optional[_Tensor],
    bn_b: Optional[_Tensor],
    transpose: bool = False,
) -> Tuple[Any, Any]:
    """Folds BatchNorm statistics and affine parameters into convolution `(weight, bias)`."""
    from .nn import Parameter

    with _no_grad():
        c_out = int(conv_w.shape[1] if transpose else conv_w.shape[0])
        conv_b_t = _from_numpy(np.zeros(c_out, dtype=np.float32)) if conv_b is None else conv_b.detach()
        bn_w_t = _from_numpy(np.ones(c_out, dtype=np.float32)) if bn_w is None else bn_w.detach()
        bn_b_t = _from_numpy(np.zeros(c_out, dtype=np.float32)) if bn_b is None else bn_b.detach()

        bn_var_rsqrt = (bn_rv.detach() + float(bn_eps)).rsqrt()
        scale = bn_w_t * bn_var_rsqrt

        ndim = len(conv_w.shape)
        if transpose:
            reshape_dims = [1, c_out] + [1] * (ndim - 2)
        else:
            reshape_dims = [c_out] + [1] * (ndim - 1)

        fused_w = conv_w.detach() * scale.reshape(reshape_dims)
        fused_b = (conv_b_t - bn_rm.detach()) * scale + bn_b_t

    return Parameter(fused_w), Parameter(fused_b)


def fuse_conv_batchnorm(conv: Any, bn: Any, transpose: bool = False) -> Any:
    """Fuses an evaluation-mode convolution layer and BatchNorm layer into a single convolution layer."""
    if getattr(conv, "training", False) or getattr(bn, "training", False):
        raise ValueError("fuse_conv_batchnorm requires both conv and bn to be in eval() mode.")

    fused_conv = copy.deepcopy(conv)
    fused_w, fused_b = fuse_conv_batchnorm_weights(
        fused_conv.weight,
        fused_conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        transpose=transpose,
    )
    fused_conv.weight = fused_w
    fused_conv.bias = fused_b
    return fused_conv


def fuse_linear_batchnorm_weights(
    linear_w: _Tensor,
    linear_b: Optional[_Tensor],
    bn_rm: _Tensor,
    bn_rv: _Tensor,
    bn_eps: float,
    bn_w: Optional[_Tensor],
    bn_b: Optional[_Tensor],
) -> Tuple[Any, Any]:
    """Folds 1D BatchNorm statistics and affine parameters into Linear `(weight, bias)`."""
    from .nn import Parameter

    with _no_grad():
        out_features = int(linear_w.shape[0])
        lin_b_t = _from_numpy(np.zeros(out_features, dtype=np.float32)) if linear_b is None else linear_b.detach()
        bn_w_t = _from_numpy(np.ones(out_features, dtype=np.float32)) if bn_w is None else bn_w.detach()
        bn_b_t = _from_numpy(np.zeros(out_features, dtype=np.float32)) if bn_b is None else bn_b.detach()

        bn_scale = bn_w_t * (bn_rv.detach() + float(bn_eps)).rsqrt()
        fused_w = linear_w.detach() * bn_scale.reshape([-1, 1])
        fused_b = (lin_b_t - bn_rm.detach()) * bn_scale + bn_b_t

    return Parameter(fused_w), Parameter(fused_b)


def fuse_linear_batchnorm(linear: Any, bn: Any) -> Any:
    """Fuses an evaluation-mode `Linear` layer and `BatchNorm1d` layer into a single `Linear` layer."""
    from .nn import Linear

    if getattr(linear, "training", False) or getattr(bn, "training", False):
        raise ValueError("fuse_linear_batchnorm requires both linear and bn to be in eval() mode.")

    w = linear.weight
    out_features, in_features = int(w.shape[0]), int(w.shape[1])
    fused = Linear(in_features, out_features, bias=True)
    fused.eval()
    fw, fb = fuse_linear_batchnorm_weights(
        linear.weight,
        linear.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )
    fused.weight = fw
    fused.bias = fb
    return fused


fusion = types.ModuleType("oxideon.nn.utils.fusion")
fusion.fuse_conv_batchnorm_weights = fuse_conv_batchnorm_weights  # type: ignore[attr-defined]
fusion.fuse_conv_batchnorm = fuse_conv_batchnorm  # type: ignore[attr-defined]
fusion.fuse_linear_batchnorm_weights = fuse_linear_batchnorm_weights  # type: ignore[attr-defined]
fusion.fuse_linear_batchnorm = fuse_linear_batchnorm  # type: ignore[attr-defined]
fusion.__all__ = [  # type: ignore[attr-defined]
    "fuse_conv_batchnorm_weights",
    "fuse_conv_batchnorm",
    "fuse_linear_batchnorm_weights",
    "fuse_linear_batchnorm",
]


# ---------------------------------------------------------------------------
# `fast_init` & `apply_with_weights`
# ---------------------------------------------------------------------------

def fast_init(module_cls: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Validates and constructs `module_cls(*args, **kwargs)` cleanly."""
    if not callable(module_cls):
        raise TypeError(f"Expected a module class or callable constructor, got {module_cls!r}")
    kwargs.pop("device", None)
    return module_cls(*args, **kwargs)


def apply_with_weights(
    module: Any,
    weights_and_buffers: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    args: Any = (),
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    tie_weights: bool = True,
    strict: bool = False,
) -> Any:
    """Evaluates `module` using temporary parameter and buffer overrides without permanently mutating `module`."""
    del tie_weights
    if isinstance(weights_and_buffers, (list, tuple)):
        merged: Dict[str, Any] = {}
        for d in weights_and_buffers:
            merged.update(d)
    else:
        merged = dict(weights_and_buffers)

    call_args = args if isinstance(args, tuple) else (args,)
    call_kwargs = kwargs or {}

    saved_attrs: List[Tuple[Any, str, Any, bool, bool]] = []
    missing: List[str] = []

    if strict and hasattr(module, "named_parameters"):
        expected = {k for k, _ in module.named_parameters()}
        if hasattr(module, "named_buffers"):
            expected |= {k for k, _ in module.named_buffers()}
        missing = sorted(expected - set(merged.keys()))

    try:
        for full_name, tensor_val in merged.items():
            parts = full_name.split(".")
            curr = module
            ok = True
            for p in parts[:-1]:
                if hasattr(curr, p):
                    curr = getattr(curr, p)
                elif hasattr(curr, "_modules") and p in curr._modules:
                    curr = curr._modules[p]
                else:
                    ok = False
                    break
            last = parts[-1]
            if not ok or not hasattr(curr, last):
                if strict:
                    raise RuntimeError(f"apply_with_weights(strict=True): unexpected key '{full_name}'")
                continue
            was_param = hasattr(curr, "_parameters") and last in curr._parameters
            was_buf = hasattr(curr, "_buffers") and last in curr._buffers
            orig_val = getattr(curr, last)
            saved_attrs.append((curr, last, orig_val, was_param, was_buf))
            if hasattr(curr, "_parameters"):
                if was_param:
                    curr._parameters[last] = tensor_val
                if was_buf:
                    curr._buffers[last] = tensor_val
                object.__setattr__(curr, last, tensor_val)
            else:
                setattr(curr, last, tensor_val)

        if strict and missing:
            raise RuntimeError(f"apply_with_weights(strict=True): missing keys {missing}")

        return module(*call_args, **call_kwargs)
    finally:
        for curr, last, orig_val, was_param, was_buf in reversed(saved_attrs):
            if hasattr(curr, "_parameters"):
                if was_param:
                    curr._parameters[last] = orig_val
                else:
                    curr._parameters.pop(last, None)
                if was_buf:
                    curr._buffers[last] = orig_val
                object.__setattr__(curr, last, orig_val)
            else:
                setattr(curr, last, orig_val)


# ---------------------------------------------------------------------------
# Generic Tensor Reparameterization (`nn.utils.reparam`)
# ---------------------------------------------------------------------------

_REPARAM_CACHE_DEPTH = 0
_REPARAM_CACHE: Dict[Tuple[int, str], _Tensor] = {}


@contextmanager
def cache_reparam():
    """Context manager caching reparameterized tensor outputs within a forward block."""
    global _REPARAM_CACHE_DEPTH
    _REPARAM_CACHE_DEPTH += 1
    try:
        yield
    finally:
        _REPARAM_CACHE_DEPTH -= 1
        if _REPARAM_CACHE_DEPTH == 0:
            _REPARAM_CACHE.clear()


class _ReparamHook:
    def __init__(self, tensor_name: str) -> None:
        self.tensor_name = tensor_name

    def evaluate(self, module: Any) -> _Tensor:
        key = (id(module), self.tensor_name)
        if _REPARAM_CACHE_DEPTH > 0 and key in _REPARAM_CACHE:
            return _REPARAM_CACHE[key]
        val = getattr(module, f"_reparam_orig_{self.tensor_name}")
        plist = module.reparams[self.tensor_name]
        for p_mod in plist:
            val = p_mod(val)
        if _REPARAM_CACHE_DEPTH > 0:
            _REPARAM_CACHE[key] = val
        return val

    def __call__(self, module: Any, inputs: Any) -> None:
        del inputs
        _set_computed_attr(module, self.tensor_name, self.evaluate(module))


def has_reparam(module: Any, tensor_name: Optional[str] = None) -> bool:
    """Returns True if `module` (or `tensor_name` on `module`) has an active reparameterization."""
    rdict = getattr(module, "reparams", None)
    if rdict is None or len(rdict) == 0:
        return False
    if tensor_name is None:
        return True
    return tensor_name in rdict and len(rdict[tensor_name]) > 0


def bind_reparam(
    module: Any,
    tensor_name: str,
    transform: Any,
    *,
    unsafe: bool = False,
) -> Any:
    """Binds a differentiable reparameterization module `transform` onto `module.<tensor_name>`."""
    from .nn import ModuleDict, ModuleList, Parameter

    del unsafe
    module = _ensure_hookable(module)
    if not has_reparam(module, tensor_name):
        if not hasattr(module, tensor_name) or getattr(module, tensor_name) is None:
            raise ValueError(f"Module {type(module).__name__} has no parameter or buffer '{tensor_name}'")
        orig = getattr(module, tensor_name)
        if hasattr(transform, "right_inverse"):
            with _no_grad():
                orig_val = transform.right_inverse(orig.detach())
        else:
            orig_val = orig.detach().clone()

        delattr(module, tensor_name)
        module.register_parameter(f"_reparam_orig_{tensor_name}", Parameter(orig_val))
        if not hasattr(module, "reparams"):
            module.reparams = ModuleDict()
        module.reparams[tensor_name] = ModuleList([transform])
        hook = _ReparamHook(tensor_name)
        _set_computed_attr(module, tensor_name, hook.evaluate(module))
        module.register_forward_pre_hook(hook)
    else:
        plist = module.reparams[tensor_name]
        if hasattr(transform, "right_inverse"):
            orig_param = getattr(module, f"_reparam_orig_{tensor_name}")
            with _no_grad():
                current_eval = orig_param.detach()
                for p_mod in plist:
                    current_eval = p_mod(current_eval)
                inv = transform.right_inverse(current_eval)
                module.register_parameter(f"_reparam_orig_{tensor_name}", Parameter(inv))
        plist.append(transform)
        for h in module._forward_pre_hooks.values():
            if isinstance(h, _ReparamHook) and h.tensor_name == tensor_name:
                _set_computed_attr(module, tensor_name, h.evaluate(module))
                break

    return module


def unbind_reparam(
    module: Any,
    tensor_name: str,
    leave_transformed: bool = True,
) -> Any:
    """Removes active reparameterizations on `tensor_name`, baking either the transformed or raw parameter."""
    from .nn import Parameter

    if not has_reparam(module, tensor_name):
        raise ValueError(f"Module {type(module).__name__} does not have a reparameterization on '{tensor_name}'")

    hook_key = None
    hook_obj = None
    for k, h in module._forward_pre_hooks.items():
        if isinstance(h, _ReparamHook) and h.tensor_name == tensor_name:
            hook_key = k
            hook_obj = h
            break

    with _no_grad():
        if leave_transformed and hook_obj is not None:
            final_t = hook_obj.evaluate(module).detach().clone()
        else:
            final_t = getattr(module, f"_reparam_orig_{tensor_name}").detach().clone()

    if hook_key is not None:
        del module._forward_pre_hooks[hook_key]
    delattr(module, f"_reparam_orig_{tensor_name}")
    del module.reparams[tensor_name]
    if len(module.reparams) == 0:
        delattr(module, "reparams")
    module.register_parameter(tensor_name, Parameter(final_t))
    return module


reparam = types.ModuleType("oxideon.nn.utils.reparam")
reparam.bind_reparam = bind_reparam  # type: ignore[attr-defined]
reparam.unbind_reparam = unbind_reparam  # type: ignore[attr-defined]
reparam.has_reparam = has_reparam  # type: ignore[attr-defined]
reparam.cache_reparam = cache_reparam  # type: ignore[attr-defined]
reparam.__all__ = [  # type: ignore[attr-defined]
    "bind_reparam",
    "unbind_reparam",
    "has_reparam",
    "cache_reparam",
]


# ---------------------------------------------------------------------------
# Structured & Mask-Based Pruning (`nn.utils.prune`)
# ---------------------------------------------------------------------------

def ln_structured_prune(
    module: Any,
    name: str,
    amount: Union[float, int],
    n: float = 2.0,
    dim: int = -1,
) -> Any:
    """Prunes entire slices along `dim` with the smallest Ln norm."""
    from .prune import _store_pruning_mask

    param = getattr(module, name, None)
    if param is None:
        raise AttributeError(f"Module {module} has no parameter named '{name}'")

    arr = param.to_numpy().copy()
    d = dim if dim >= 0 else dim + arr.ndim
    if d < 0 or d >= arr.ndim:
        raise IndexError(f"dim {dim} out of bounds for rank {arr.ndim}")

    num_channels = arr.shape[d]
    if isinstance(amount, int) and not isinstance(amount, bool):
        if not (0 <= amount <= num_channels):
            raise ValueError(f"integer amount must be in [0, {num_channels}], got {amount}")
        k = amount
    else:
        if not (0.0 <= float(amount) <= 1.0):
            raise ValueError(f"float amount must be in [0, 1], got {amount}")
        k = int(round(num_channels * float(amount)))

    axes = tuple(i for i in range(arr.ndim) if i != d)
    if n == float("inf"):
        norms = np.max(np.abs(arr), axis=axes)
    elif n == float("-inf"):
        norms = np.min(np.abs(arr), axis=axes)
    else:
        norms = np.sum(np.abs(arr) ** float(n), axis=axes) ** (1.0 / float(n))

    if k >= num_channels:
        full_mask = np.zeros_like(arr, dtype=np.float32)
    elif k > 0:
        pruned_idx = np.argsort(norms)[:k]
        mask_1d = np.ones(num_channels, dtype=np.float32)
        mask_1d[pruned_idx] = 0.0
        shape = [1] * arr.ndim
        shape[d] = num_channels
        full_mask = np.broadcast_to(mask_1d.reshape(shape), arr.shape).copy()
    else:
        full_mask = np.ones_like(arr, dtype=np.float32)

    return _store_pruning_mask(module, name, full_mask, arr)


def mask_prune(module: Any, name: str, mask: Union[_Tensor, np.ndarray]) -> Any:
    """Applies a user-supplied binary `mask` directly to `module.<name>`."""
    from .prune import _store_pruning_mask

    param = getattr(module, name, None)
    if param is None:
        raise AttributeError(f"Module {module} has no parameter named '{name}'")
    arr = param.to_numpy().copy()
    mask_np = mask.to_numpy() if isinstance(mask, _Tensor) else np.asarray(mask)
    if mask_np.shape != arr.shape:
        raise ValueError(f"Mask shape {mask_np.shape} does not match parameter shape {arr.shape}")
    return _store_pruning_mask(module, name, mask_np.astype(np.float32), arr)


def identity_prune(module: Any, name: str) -> Any:
    """Attaches an all-ones pruning mask to `module.<name>` without zeroing entries."""
    param = getattr(module, name, None)
    if param is None:
        raise AttributeError(f"Module {module} has no parameter named '{name}'")
    arr = param.to_numpy()
    return mask_prune(module, name, np.ones_like(arr, dtype=np.float32))


def is_pruned(module: Any) -> bool:
    """Returns True if `module` or any of its submodules currently holds an active pruning mask."""
    from .prune import has_pruning_mask

    if hasattr(module, "_parameters"):
        for pname in list(module._parameters.keys()):
            if has_pruning_mask(module, pname):
                return True
    for attr in ("weight", "bias"):
        if has_pruning_mask(module, attr):
            return True
    if hasattr(module, "children"):
        for ch in module.children():
            if is_pruned(ch):
                return True
    return False


def _build_prune_submodule() -> types.ModuleType:
    mod = types.ModuleType("oxideon.nn.utils.prune")

    def _l1(module: Any, name: str = "weight", amount: float = 0.3) -> Any:
        from . import prune as _ox_prune
        return _ox_prune.prune_l1_unstructured(module, name=name, amount=amount)

    def _rand(module: Any, name: str = "weight", amount: float = 0.3, seed: Optional[int] = None) -> Any:
        from . import prune as _ox_prune
        return _ox_prune.prune_random_unstructured(module, name=name, amount=amount, seed=seed)

    def _global(parameters: Sequence[Tuple[Any, str]], amount: float = 0.3, method: str = "l1", seed: Optional[int] = None) -> None:
        from . import prune as _ox_prune
        _ox_prune.global_unstructured_pruning(parameters, amount=amount, method=method, seed=seed)

    def _remove(module: Any, name: str = "weight") -> Any:
        from . import prune as _ox_prune
        return _ox_prune.remove_pruning(module, name=name)

    mod.prune_l1_unstructured = _l1  # type: ignore[attr-defined]
    mod.prune_random_unstructured = _rand  # type: ignore[attr-defined]
    mod.ln_structured_prune = ln_structured_prune  # type: ignore[attr-defined]
    mod.global_unstructured_pruning = _global  # type: ignore[attr-defined]
    mod.mask_prune = mask_prune  # type: ignore[attr-defined]
    mod.identity_prune = identity_prune  # type: ignore[attr-defined]
    mod.remove_pruning = _remove  # type: ignore[attr-defined]
    mod.is_pruned = is_pruned  # type: ignore[attr-defined]
    mod.__all__ = [  # type: ignore[attr-defined]
        "prune_l1_unstructured",
        "prune_random_unstructured",
        "ln_structured_prune",
        "global_unstructured_pruning",
        "mask_prune",
        "identity_prune",
        "remove_pruning",
        "is_pruned",
    ]
    return mod


prune = _build_prune_submodule()


__all__ = [
    "total_norm",
    "weight_norm",
    "remove_weight_norm",
    "spectral_norm",
    "remove_spectral_norm",
    "fuse_conv_batchnorm_weights",
    "fuse_conv_batchnorm",
    "fuse_linear_batchnorm_weights",
    "fuse_linear_batchnorm",
    "fast_init",
    "apply_with_weights",
    "bind_reparam",
    "unbind_reparam",
    "has_reparam",
    "cache_reparam",
    "ln_structured_prune",
    "mask_prune",
    "identity_prune",
    "is_pruned",
    "fusion",
    "reparam",
    "prune",
]
