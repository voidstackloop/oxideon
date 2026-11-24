"""High-level functional programming transforms for Oxideon (vmap, grad, value_and_grad, jacrev, jacfwd, hessian).

Compatible with standard functorch and PyTorch 2.0 functional transforms.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import functools
import numpy as np

from .oxideon import Tensor as _Tensor
from .primitives import tensor, stack, unbind, _from_numpy, _to_np
from .primitives import _AutogradModule, _FunctionalAutograd


def vmap(
    func: Callable[..., Any],
    in_dims: Union[int, Tuple[Optional[int], ...]] = 0,
    out_dims: int = 0,
) -> Callable[..., Any]:
    """Vectorizing map that batches execution of `func` across dimension `in_dims`."""
    @functools.wraps(func)
    def _vmapped(*args, **kwargs):
        if not args:
            return func(**kwargs)

        if isinstance(in_dims, int):
            norm_in_dims = tuple(in_dims if isinstance(a, _Tensor) else None for a in args)
        else:
            if len(in_dims) != len(args):
                raise ValueError(f"in_dims length ({len(in_dims)}) must match args length ({len(args)})")
            norm_in_dims = in_dims

        batch_size = None
        for a, dim in zip(args, norm_in_dims):
            if isinstance(a, _Tensor) and dim is not None:
                dim_idx = dim if dim >= 0 else dim + len(a.shape)
                if dim_idx < 0 or dim_idx >= len(a.shape):
                    raise ValueError(f"in_dims {dim} is out of bounds for tensor of rank {len(a.shape)}")
                curr_bs = a.shape[dim_idx]
                if batch_size is None:
                    batch_size = curr_bs
                elif batch_size != curr_bs:
                    raise ValueError(f"Inconsistent batch sizes across mapped arguments: expected {batch_size}, got {curr_bs}")

        if batch_size is None:
            return func(*args, **kwargs)

        results = []
        for i in range(batch_size):
            slice_args = []
            for a, dim in zip(args, norm_in_dims):
                if isinstance(a, _Tensor) and dim is not None:
                    dim_idx = dim if dim >= 0 else dim + len(a.shape)
                    slc = [slice(None)] * len(a.shape)
                    slc[dim_idx] = i
                    slice_args.append(a[tuple(slc)])
                else:
                    slice_args.append(a)
            results.append(func(*slice_args, **kwargs))

        if isinstance(results[0], tuple):
            return tuple(stack([r[j] for r in results], dim=out_dims) for j in range(len(results[0])))
        elif isinstance(results[0], _Tensor):
            return stack(results, dim=out_dims)
        return results

    return _vmapped


def grad(
    func: Callable[..., Any],
    argnums: Union[int, Tuple[int, ...]] = 0,
    has_aux: bool = False,
    create_graph: bool = True,
) -> Callable[..., Any]:
    """Returns a function that computes the gradient of `func` w.r.t specified arguments."""
    @functools.wraps(func)
    def _grad_fn(*args, **kwargs):
        arg_indices = (argnums,) if isinstance(argnums, int) else argnums
        diff_args = []
        for i, a in enumerate(args):
            if i in arg_indices and isinstance(a, _Tensor):
                diff_args.append(a if a.requires_grad else a.clone().detach().requires_grad_(True))
            else:
                diff_args.append(a)

        raw_out = func(*diff_args, **kwargs)
        if has_aux:
            if not isinstance(raw_out, tuple) or len(raw_out) != 2:
                raise ValueError("When has_aux=True, func must return a pair (loss, aux)")
            out, aux = raw_out
        else:
            out, aux = raw_out, None

        target_inps = [diff_args[i] for i in arg_indices]
        from .oxideon import grad as _native_grad
        try:
            grads = _native_grad([out], target_inps, create_graph=create_graph)
        except Exception:
            grads = _AutogradModule.grad(out, target_inps, create_graph=create_graph)

        g_out = grads[0] if isinstance(argnums, int) else grads
        if has_aux:
            return g_out, aux
        return g_out

    return _grad_fn


def value_and_grad(
    func: Callable[..., Any],
    argnums: Union[int, Tuple[int, ...]] = 0,
    has_aux: bool = False,
    create_graph: bool = True,
) -> Callable[..., Tuple[Any, Any]]:
    """Returns a function that computes both the output and gradient of `func`."""
    @functools.wraps(func)
    def _vag_fn(*args, **kwargs):
        arg_indices = (argnums,) if isinstance(argnums, int) else argnums
        diff_args = []
        for i, a in enumerate(args):
            if i in arg_indices and isinstance(a, _Tensor):
                diff_args.append(a if a.requires_grad else a.clone().detach().requires_grad_(True))
            else:
                diff_args.append(a)

        raw_out = func(*diff_args, **kwargs)
        if has_aux:
            if not isinstance(raw_out, tuple) or len(raw_out) != 2:
                raise ValueError("When has_aux=True, func must return a pair (loss, aux)")
            out, aux = raw_out
        else:
            out, aux = raw_out, None

        target_inps = [diff_args[i] for i in arg_indices]
        from .oxideon import grad as _native_grad
        try:
            grads = _native_grad([out], target_inps, create_graph=create_graph)
        except Exception:
            grads = _AutogradModule.grad(out, target_inps, create_graph=create_graph)

        g_out = grads[0] if isinstance(argnums, int) else grads
        if has_aux:
            return (out, aux), g_out
        return out, g_out

    return _vag_fn


def jacrev(func: Callable[..., Any], argnums: Union[int, Tuple[int, ...]] = 0) -> Callable[..., Any]:
    """Computes the Jacobian using reverse-mode automatic differentiation."""
    @functools.wraps(func)
    def _jacrev_fn(*args, **kwargs):
        inputs = args[argnums] if isinstance(argnums, int) else tuple(args[i] for i in argnums)
        return _FunctionalAutograd.jacobian(lambda *inps: func(*inps, **kwargs), inputs)
    return _jacrev_fn


jacfwd = jacrev


def hessian(func: Callable[..., Any], argnums: Union[int, Tuple[int, ...]] = 0) -> Callable[..., Any]:
    """Computes the Hessian matrix of a scalar function."""
    @functools.wraps(func)
    def _hessian_fn(*args, **kwargs):
        inputs = args[argnums] if isinstance(argnums, int) else tuple(args[i] for i in argnums)
        return _FunctionalAutograd.hessian(lambda *inps: func(*inps, **kwargs), inputs)
    return _hessian_fn


vjp = _FunctionalAutograd.vjp


def jvp(
    func: Callable[..., Any],
    primals: Union[_Tensor, Tuple[_Tensor, ...], List[_Tensor]],
    tangents: Union[_Tensor, Tuple[_Tensor, ...], List[_Tensor]],
) -> Tuple[Any, Any]:
    """Computes the Jacobian-vector product (primal_out, tangent_out) at `primals` along `tangents`."""
    single_input = isinstance(primals, _Tensor)
    p_tuple = (primals,) if single_input else tuple(primals)
    t_tuple = (tangents,) if isinstance(tangents, _Tensor) else tuple(tangents)
    if len(p_tuple) != len(t_tuple):
        raise ValueError(f"primals length ({len(p_tuple)}) must match tangents length ({len(t_tuple)})")

    p_leaves = [p.clone().detach().requires_grad_(True) for p in p_tuple]
    out = func(*p_leaves) if not single_input else func(p_leaves[0])
    jac = _FunctionalAutograd.jacobian(func, primals)

    if single_input:
        jac_np = _to_np(jac)
        v_np = _to_np(t_tuple[0]).reshape(-1)
        out_shape = out.shape
        out_numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        out_dt = _to_np(out).dtype
        jvp_np = (jac_np.reshape(out_numel, -1) @ v_np).reshape(out_shape).astype(out_dt)
        return out, _from_numpy(np.ascontiguousarray(jvp_np))

    out_shape = out.shape
    out_numel = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
    out_dt = _to_np(out).dtype
    total_jvp = np.zeros(out_numel, dtype=out_dt)
    for j_i, v_i in zip(jac, t_tuple):
        j_np = _to_np(j_i).reshape(out_numel, -1)
        v_np = _to_np(v_i).reshape(-1)
        total_jvp += (j_np @ v_np).astype(out_dt)
    return out, _from_numpy(np.ascontiguousarray(total_jvp.reshape(out_shape)))


def hvp(
    func: Callable[..., Any],
    primals: Union[_Tensor, Tuple[_Tensor, ...], List[_Tensor]],
    tangents: Union[_Tensor, Tuple[_Tensor, ...], List[_Tensor]],
) -> Tuple[Any, Any]:
    """Computes the Hessian-vector product (f(primals), H(primals) @ tangents) via double backward.

    Complexity:
      - Memory: O(D) (never materializes the full D x D Hessian matrix)
      - Compute: Exactly 2 backward passes, optimal for Newton-CG, TRPO, and curvature estimation.
    """
    single_input = isinstance(primals, _Tensor)
    p_tuple = (primals,) if single_input else tuple(primals)
    t_tuple = (tangents,) if isinstance(tangents, _Tensor) else tuple(tangents)
    if len(p_tuple) != len(t_tuple):
        raise ValueError(f"primals length ({len(p_tuple)}) must match tangents length ({len(t_tuple)})")

    from .oxideon import grad as _native_grad

    p_leaves = [p.clone().detach().requires_grad_(True) for p in p_tuple]
    out = func(*p_leaves) if not single_input else func(p_leaves[0])

    grads = _native_grad([out], p_leaves, create_graph=True)

    inner_prod = None
    for g_i, v_i in zip(grads, t_tuple):
        term = (g_i * v_i).sum()
        inner_prod = term if inner_prod is None else inner_prod + term

    hv_grads = _native_grad([inner_prod], p_leaves, create_graph=False)

    hv_out = hv_grads[0] if single_input else tuple(hv_grads)
    return out, hv_out


def linearize(
    func: Callable[..., Any],
    *primals: Any,
) -> Tuple[Any, Callable[..., Any]]:
    """Evaluates `func(*primals)` and returns a closure that computes JVP along any tangent vector."""
    primal_out = func(*primals)

    def jvp_fn(*tangents: Any) -> Any:
        t_arg = tangents if len(tangents) > 1 else tangents[0]
        _, tangent_out = jvp(func, primals if len(primals) > 1 else primals[0], t_arg)
        return tangent_out

    return primal_out, jvp_fn


def functional_call(
    module: Any,
    parameter_and_buffer_dicts: Dict[str, Any],
    args: Any = (),
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Invokes `module(*args, **kwargs)` with parameters temporarily substituted by `parameter_and_buffer_dicts`."""
    kwargs = kwargs or {}
    saved_attrs = {}

    try:
        # Swap parameters/buffers
        for name, tensor_val in parameter_and_buffer_dicts.items():
            parts = name.split(".")
            curr = module
            for p in parts[:-1]:
                curr = getattr(curr, p)
            last_part = parts[-1]
            if hasattr(curr, last_part):
                saved_attrs[name] = getattr(curr, last_part)
                setattr(curr, last_part, tensor_val)

        return module(*args, **kwargs)
    finally:
        # Restore original parameters/buffers
        for name, orig_val in saved_attrs.items():
            parts = name.split(".")
            curr = module
            for p in parts[:-1]:
                curr = getattr(curr, p)
            last_part = parts[-1]
            setattr(curr, last_part, orig_val)


def make_functional(module: Any) -> Tuple[Callable[..., Any], Dict[str, Any]]:
    """Extracts parameters of `module` into a dictionary and returns `(func, params)`."""
    params = {}
    if hasattr(module, "named_parameters"):
        for name, param in module.named_parameters():
            params[name] = param
    else:
        for attr in ["weight", "bias"]:
            if hasattr(module, attr) and getattr(module, attr) is not None:
                params[attr] = getattr(module, attr)

    def _func(params_dict: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return functional_call(module, params_dict, args=args, kwargs=kwargs)

    return _func, params


__all__ = [
    "vmap",
    "grad",
    "value_and_grad",
    "jacrev",
    "jacfwd",
    "hessian",
    "vjp",
    "jvp",
    "hvp",
    "linearize",
    "functional_call",
    "make_functional",
]
