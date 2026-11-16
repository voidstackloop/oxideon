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


__all__ = [
    "assert_close",
    "assert_allclose",
    "assert_equal",
    "gradcheck",
    "make_tensor",
]
