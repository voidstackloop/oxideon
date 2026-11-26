"""Testing assertion and synthetic tensor creation utilities for Oxideon."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import math
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy
from .primitives import tensor


def assert_close(
    actual: Any,
    expected: Any,
    *,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    msg: Optional[str] = None,
) -> None:
    """Asserts that `actual` and `expected` tensors or collections are numerically close within tolerances."""
    if isinstance(actual, _Tensor) and isinstance(expected, _Tensor):
        if check_device and actual.device != expected.device:
            raise AssertionError(f"Device mismatch: actual {actual.device} != expected {expected.device}")
        if check_dtype and actual.dtype != expected.dtype:
            raise AssertionError(f"Dtype mismatch: actual {actual.dtype} != expected {expected.dtype}")
        if list(actual.shape) != list(expected.shape):
            raise AssertionError(f"Shape mismatch: actual {actual.shape} != expected {expected.shape}")

    arr_act = actual.to_numpy() if isinstance(actual, _Tensor) else np.asarray(actual)
    arr_exp = expected.to_numpy() if isinstance(expected, _Tensor) else np.asarray(expected)

    if arr_act.shape != arr_exp.shape:
        raise AssertionError(f"Shape mismatch: {arr_act.shape} != {arr_exp.shape}")

    if rtol is None:
        if arr_act.dtype in (np.float64,):
            rtol = 1e-7
        elif arr_act.dtype in (np.float32,):
            rtol = 1e-5
        else:
            rtol = 1e-3

    if atol is None:
        if arr_act.dtype in (np.float64,):
            atol = 1e-12
        elif arr_act.dtype in (np.float32,):
            atol = 1e-8
        else:
            atol = 1e-5

    close = np.isclose(arr_act, arr_exp, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not np.all(close):
        abs_diff = np.abs(arr_act - arr_exp)
        if equal_nan:
            abs_diff = np.where(np.isnan(arr_act) & np.isnan(arr_exp), 0.0, abs_diff)
        max_abs = np.nanmax(abs_diff)
        rel_diff = abs_diff / np.maximum(1e-15, np.abs(arr_exp))
        max_rel = np.nanmax(rel_diff)
        header = f"{msg}: " if msg else ""
        raise AssertionError(
            f"{header}Tensors not close within rtol={rtol}, atol={atol}.\n"
            f"Max absolute difference: {max_abs:.6e} (atol={atol})\n"
            f"Max relative difference: {max_rel:.6e} (rtol={rtol})"
        )


assert_allclose = assert_close


def assert_equal(actual: Any, expected: Any, msg: Optional[str] = None) -> None:
    """Asserts that two tensors or collections have identical values and shapes."""
    if isinstance(actual, _Tensor) and isinstance(expected, _Tensor):
        if list(actual.shape) != list(expected.shape):
            raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    arr_act = actual.to_numpy() if isinstance(actual, _Tensor) else np.asarray(actual)
    arr_exp = expected.to_numpy() if isinstance(expected, _Tensor) else np.asarray(expected)
    if not np.array_equal(arr_act, arr_exp):
        header = f"{msg}: " if msg else ""
        raise AssertionError(f"{header}Arrays not strictly equal: {arr_act} != {arr_exp}")


def gradcheck(
    func: Callable[..., Any],
    inputs: Union[_Tensor, Sequence[_Tensor]],
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> bool:
    """Numerically verifies analytical autograd gradients against central finite differences."""
    tensors = [inputs] if isinstance(inputs, _Tensor) else list(inputs)
    for t in tensors:
        if not getattr(t, "requires_grad", False):
            t.requires_grad_(True)

    # Compute analytical gradients
    out = func(*tensors)
    if hasattr(out, "backward"):
        out.backward()
    else:
        raise ValueError("func output must be a Tensor supporting backward()")

    for t in tensors:
        g_call = getattr(t, "grad", None)
        g_anal = g_call() if callable(g_call) else g_call
        if g_anal is None:
            raise AssertionError("Tensor did not receive analytical gradients during gradcheck")
        g_anal_np = g_anal.to_numpy()

        # Central finite difference for every element
        x0 = t.to_numpy().copy()
        g_num = np.zeros_like(x0)
        it = np.nditer(x0, flags=["multi_index"])

        while not it.finished:
            idx = it.multi_index
            val = float(x0[idx])

            # x + eps
            x_plus = x0.copy()
            x_plus[idx] = val + eps
            t_plus = _from_numpy(x_plus)
            # Reconstruct arguments replacing t with t_plus
            args_plus = [t_plus if arg is t else arg for arg in tensors]
            out_p = func(*args_plus)
            y_p = float(out_p.to_numpy().flat[0])

            # x - eps
            x_minus = x0.copy()
            x_minus[idx] = val - eps
            t_minus = _from_numpy(x_minus)
            args_minus = [t_minus if arg is t else arg for arg in tensors]
            out_m = func(*args_minus)
            y_m = float(out_m.to_numpy().flat[0])

            dx = float(x_plus[idx]) - float(x_minus[idx])
            step = dx if abs(dx) > 1e-12 else (2.0 * eps)
            g_num[idx] = (y_p - y_m) / step
            it.iternext()

        if not np.allclose(g_anal_np, g_num, rtol=rtol, atol=atol):
            max_diff = float(np.max(np.abs(g_anal_np - g_num)))
            raise AssertionError(
                f"gradcheck failed: max abs difference {max_diff:.6e} > atol={atol}\n"
                f"Analytical:\n{g_anal_np}\nNumerical:\n{g_num}"
            )

    return True


def make_tensor(
    shape: Sequence[int],
    dtype: str = "f32",
    device: str = "cpu",
    low: Optional[float] = None,
    high: Optional[float] = None,
    requires_grad: bool = False,
    seed: int = 0,
) -> _Tensor:
    """Constructs a deterministic synthetic tensor for unit testing and numerical verification."""
    rng = np.random.default_rng(seed)
    shape_tuple = tuple(shape)
    if "int" in dtype or dtype in ("i32", "i64"):
        min_v = int(low) if low is not None else 0
        max_v = int(high) if high is not None else 100
        arr = rng.integers(min_v, max_v, size=shape_tuple).astype(np.int64 if dtype == "i64" else np.int32)
    elif dtype in ("bool",):
        arr = rng.choice([True, False], size=shape_tuple)
    else:
        min_v = float(low) if low is not None else 0.0
        max_v = float(high) if high is not None else 1.0
        arr = rng.uniform(min_v, max_v, size=shape_tuple).astype(np.float64 if dtype == "f64" else np.float32)

    res = _from_numpy(np.ascontiguousarray(arr))
    if device != "cpu":
        res = res.to(device)
    if requires_grad:
        res = res.requires_grad_(True)
    return res


def gradgradcheck(
    func: Callable[..., Any],
    inputs: Union[_Tensor, Sequence[_Tensor]],
    grad_outputs: Optional[Union[_Tensor, Sequence[_Tensor]]] = None,
    eps: float = 1e-4,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> bool:
    """Numerically verifies second-order autograd derivatives against central finite differences of first-order gradients."""
    from .oxideon import grad as _native_grad

    tensors = [inputs] if isinstance(inputs, _Tensor) else list(inputs)
    clean_inputs = [_from_numpy(np.ascontiguousarray(t.to_numpy())).requires_grad_(True) for t in tensors]

    def _first_grad_directional(args_list: List[_Tensor], create_graph: bool) -> List[_Tensor]:
        out = func(*args_list)
        out_list = [out] if isinstance(out, _Tensor) else list(out)
        if grad_outputs is None:
            scalar = sum((o.sum() for o in out_list[1:]), out_list[0].sum())
        else:
            gos = [grad_outputs] if isinstance(grad_outputs, _Tensor) else list(grad_outputs)
            terms = [(o * go).sum() for o, go in zip(out_list, gos)]
            scalar = sum(terms[1:], terms[0])
        return list(_native_grad([scalar], args_list, create_graph=create_graph))

    first_grads = _first_grad_directional(clean_inputs, create_graph=True)
    # Sum of first-order gradients forms scalar objective whose gradient w.r.t. inputs is sum_i H_{i,j}
    second_obj = sum((g.sum() for g in first_grads[1:]), first_grads[0].sum())
    anal_second = list(_native_grad([second_obj], clean_inputs, create_graph=False))

    for inp_idx, (t_in, g2_anal_t) in enumerate(zip(clean_inputs, anal_second)):
        g2_anal = g2_anal_t.to_numpy()
        x0 = t_in.to_numpy().copy()
        g2_num = np.zeros_like(x0)
        it = np.nditer(x0, flags=["multi_index"])

        while not it.finished:
            idx = it.multi_index
            val = float(x0[idx])

            x_plus = x0.copy()
            x_plus[idx] = val + eps
            args_plus = [
                _from_numpy(x_plus if j == inp_idx else clean_inputs[j].to_numpy().copy()).requires_grad_(True)
                for j in range(len(clean_inputs))
            ]
            fg_plus = _first_grad_directional(args_plus, create_graph=False)
            s_plus = sum(float(g.to_numpy().sum()) for g in fg_plus)

            x_minus = x0.copy()
            x_minus[idx] = val - eps
            args_minus = [
                _from_numpy(x_minus if j == inp_idx else clean_inputs[j].to_numpy().copy()).requires_grad_(True)
                for j in range(len(clean_inputs))
            ]
            fg_minus = _first_grad_directional(args_minus, create_graph=False)
            s_minus = sum(float(g.to_numpy().sum()) for g in fg_minus)

            dx = float(x_plus[idx]) - float(x_minus[idx])
            step = dx if abs(dx) > 1e-12 else (2.0 * eps)
            g2_num[idx] = (s_plus - s_minus) / step
            it.iternext()

        if not np.allclose(g2_anal, g2_num, rtol=rtol, atol=atol):
            max_diff = float(np.max(np.abs(g2_anal - g2_num)))
            raise AssertionError(
                f"gradgradcheck failed on input {inp_idx}: max abs difference {max_diff:.6e} > atol={atol}\n"
                f"Analytical 2nd derivative:\n{g2_anal}\nNumerical 2nd derivative:\n{g2_num}"
            )

    return True


def assert_finite(tensor_or_tree: Any, msg: Optional[str] = None) -> None:
    """Asserts that every `Tensor` or NumPy array inside `tensor_or_tree` contains only finite values (no NaN or Inf)."""
    from .treemap import tree_leaves

    prefix = f"{msg}: " if msg else ""
    for idx, leaf in enumerate(tree_leaves(tensor_or_tree)):
        if isinstance(leaf, _Tensor):
            arr = leaf.to_numpy()
        elif isinstance(leaf, np.ndarray):
            arr = leaf
        elif isinstance(leaf, (int, float)):
            if not math.isfinite(leaf):
                raise AssertionError(f"{prefix}Non-finite scalar value at leaf {idx}: {leaf}")
            continue
        else:
            continue

        if arr.dtype.kind in ("f", "c") and not np.all(np.isfinite(arr)):
            nan_cnt = int(np.isnan(arr).sum())
            inf_cnt = int(np.isinf(arr).sum())
            raise AssertionError(
                f"{prefix}Non-finite tensor values at leaf {idx} with shape {tuple(arr.shape)}: "
                f"{nan_cnt} NaN(s), {inf_cnt} Inf(s)"
            )


def assert_shapes(
    specs: Sequence[Any],
    msg: Optional[str] = None,
) -> Dict[str, int]:
    """Verifies tensor ranks, literal dimensions, and symbolic dimension bindings across multiple `(tensor, pattern)` pairs.

    Each element of `specs` is `(tensor, expected_dims)` where `expected_dims` is a sequence of:
    - `int`: exact dimension size required
    - `str`: symbolic dimension name (e.g. `'B'`, `'T'`, `'D'`) bound on first occurrence and checked across all occurrences
      (`'*'` or `'_'` acts as a single-axis wildcard)
    - `None`: single-axis wildcard
    - `...` (`Ellipsis`): variable-length prefix/infix wildcard (at most one per pattern)

    Returns a dictionary mapping bound symbolic names to their resolved integer sizes.
    """
    prefix = f"{msg}: " if msg else ""
    bindings: Dict[str, int] = {}

    for item_idx, pair in enumerate(specs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"assert_shapes expects (tensor, pattern) pairs, got {pair!r} at index {item_idx}")
        t, pat_raw = pair
        actual_shape = tuple(t.shape) if hasattr(t, "shape") else tuple(np.asarray(t).shape)
        pattern = list(pat_raw)

        ellipsis_positions = [i for i, tok in enumerate(pattern) if tok is Ellipsis]
        if len(ellipsis_positions) > 1:
            raise ValueError(f"Pattern at index {item_idx} may contain at most one Ellipsis (...)")

        if ellipsis_positions:
            ep = ellipsis_positions[0]
            head = pattern[:ep]
            tail = pattern[ep + 1 :]
            if len(actual_shape) < len(head) + len(tail):
                raise AssertionError(
                    f"{prefix}Tensor {item_idx} rank mismatch: shape {actual_shape} has fewer than "
                    f"{len(head) + len(tail)} dims required by pattern {tuple(pattern)}"
                )
            aligned_pairs = list(zip(actual_shape[: len(head)], head))
            if tail:
                aligned_pairs.extend(zip(actual_shape[-len(tail) :], tail))
        else:
            if len(actual_shape) != len(pattern):
                raise AssertionError(
                    f"{prefix}Tensor {item_idx} rank mismatch: actual shape {actual_shape} != pattern {tuple(pattern)}"
                )
            aligned_pairs = list(zip(actual_shape, pattern))

        for dim_val, tok in aligned_pairs:
            dim_int = int(dim_val)
            if tok is None or tok in ("*", "_"):
                continue
            if isinstance(tok, int) and not isinstance(tok, bool):
                if dim_int != tok:
                    raise AssertionError(
                        f"{prefix}Tensor {item_idx} shape {actual_shape} failed literal dimension check: "
                        f"expected {tok}, got {dim_int} (pattern {tuple(pattern)})"
                    )
            elif isinstance(tok, str):
                if tok in bindings:
                    if bindings[tok] != dim_int:
                        raise AssertionError(
                            f"{prefix}Symbolic dimension '{tok}' mismatch at tensor {item_idx} (shape {actual_shape}): "
                            f"previously bound to {bindings[tok]}, got {dim_int}"
                        )
                else:
                    bindings[tok] = dim_int
            else:
                raise TypeError(f"Unsupported dimension token {tok!r} in assert_shapes pattern")

    return bindings


__all__ = [
    "assert_close",
    "assert_allclose",
    "assert_equal",
    "assert_finite",
    "assert_shapes",
    "gradcheck",
    "gradgradcheck",
    "make_tensor",
]
