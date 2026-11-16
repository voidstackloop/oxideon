"""Differentiable numerical root finding and nonlinear optimization for Oxideon."""

from typing import Callable, NamedTuple, Optional, Tuple, Any
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, grad as _native_grad


class RootResult(NamedTuple):
    root: float
    iterations: int
    converged: bool
    residual: float


def _eval_scalar(func: Callable[[Any], Any], x_val: float) -> float:
    """Evaluates a scalar function accepting either a Python float or a 1-element Tensor."""
    try:
        out = func(x_val)
    except (TypeError, AttributeError):
        out = func(_from_numpy(np.array([x_val], dtype=np.float32)))
    if isinstance(out, _Tensor):
        return float(out.to_numpy().flat[0])
    if isinstance(out, np.ndarray):
        return float(out.flat[0])
    return float(out)


def newton_raphson(
    func: Callable[[_Tensor], _Tensor],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> RootResult:
    """Finds a root f(x) = 0 using exact analytical automatic differentiation."""
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    x_val = float(x0)
    converged = False

    for iteration in range(1, max_iter + 1):
        x_tensor = _from_numpy(np.array([x_val], dtype=np.float32)).detach().requires_grad_(True)

        y_tensor = func(x_tensor)
        y_val = float(y_tensor.to_numpy().flat[0])

        if abs(y_val) < tol:
            converged = True
            return RootResult(root=x_val, iterations=iteration, converged=True, residual=abs(y_val))

        y_tensor.backward()
        g_call = getattr(x_tensor, "grad", None)
        g = g_call() if callable(g_call) else g_call
        if g is None:
            break
        deriv_val = float(g.to_numpy().flat[0])

        if abs(deriv_val) < 1e-12:
            break

        step = y_val / deriv_val
        x_val = x_val - step

        if abs(step) < tol:
            converged = True
            new_res = abs(float(func(_from_numpy(np.array([x_val], dtype=np.float32))).to_numpy().flat[0]))
            return RootResult(root=x_val, iterations=iteration, converged=True, residual=new_res)

    final_residual = abs(float(func(_from_numpy(np.array([x_val], dtype=np.float32))).to_numpy().flat[0]))
    return RootResult(root=x_val, iterations=max_iter, converged=converged or final_residual < tol, residual=final_residual)


def secant_method(
    func: Callable[[Any], Any],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> RootResult:
    """Finds a root f(x) = 0 using the secant method without derivatives."""
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")

    x_prev = float(x0)
    x_curr = float(x1)
    f_prev = _eval_scalar(func, x_prev)
    f_curr = _eval_scalar(func, x_curr)

    if abs(f_prev) < tol:
        return RootResult(root=x_prev, iterations=0, converged=True, residual=abs(f_prev))
    if abs(f_curr) < tol:
        return RootResult(root=x_curr, iterations=0, converged=True, residual=abs(f_curr))

    for iteration in range(1, max_iter + 1):
        denom = f_curr - f_prev
        if abs(denom) < 1e-14:
            break

        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        f_next = _eval_scalar(func, x_next)

        if abs(f_next) < tol or abs(x_next - x_curr) < tol:
            return RootResult(root=x_next, iterations=iteration, converged=True, residual=abs(f_next))

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    return RootResult(root=x_curr, iterations=max_iter, converged=abs(f_curr) < tol, residual=abs(f_curr))


def halley_method(
    func: Callable[[Any], Any],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> RootResult:
    """Finds a root f(x) = 0 using Halley's second-order root finding method."""
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")

    x_val = float(x0)
    h = 1e-4

    for iteration in range(1, max_iter + 1):
        f_val = _eval_scalar(func, x_val)
        if abs(f_val) < tol:
            return RootResult(root=x_val, iterations=iteration, converged=True, residual=abs(f_val))

        f_p = _eval_scalar(func, x_val + h)
        f_m = _eval_scalar(func, x_val - h)
        f_prime = (f_p - f_m) / (2.0 * h)
        f_double_prime = (f_p - 2.0 * f_val + f_m) / (h * h)

        denom = 2.0 * (f_prime ** 2) - f_val * f_double_prime
        if abs(denom) < 1e-14:
            step = f_val / (f_prime + 1e-12)
        else:
            step = (2.0 * f_val * f_prime) / denom

        x_val = x_val - step
        if abs(step) < tol:
            final_res = abs(_eval_scalar(func, x_val))
            return RootResult(root=x_val, iterations=iteration, converged=True, residual=final_res)

    final_residual = abs(_eval_scalar(func, x_val))
    return RootResult(root=x_val, iterations=max_iter, converged=final_residual < tol, residual=final_residual)


def bisection(
    func: Callable[[Any], Any],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> RootResult:
    """Finds a root f(x) = 0 on bracket [a, b] using the bisection method."""
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")
    if a > b:
        a, b = b, a
    fa = _eval_scalar(func, a)
    fb = _eval_scalar(func, b)
    if abs(fa) < tol:
        return RootResult(root=a, iterations=0, converged=True, residual=abs(fa))
    if abs(fb) < tol:
        return RootResult(root=b, iterations=0, converged=True, residual=abs(fb))
    if fa * fb > 0:
        raise ValueError(f"f(a) and f(b) must have opposite signs. Got f({a})={fa}, f({b})={fb}")

    mid = a
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        fmid = _eval_scalar(func, mid)
        if abs(fmid) < tol or (b - a) * 0.5 < tol:
            return RootResult(root=mid, iterations=iteration, converged=True, residual=abs(fmid))

        if fa * fmid < 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid

    final_res = abs(_eval_scalar(func, mid))
    is_conv = (final_res < tol) or ((b - a) * 0.5 < tol)
    return RootResult(root=mid, iterations=max_iter, converged=is_conv, residual=final_res)


def differentiable_root(
    func: Callable[[Any], Any],
    x0: float,
    solver: str = "newton",
    tol: float = 1e-6,
    max_iter: int = 50,
) -> _Tensor:
    """Computes root f(x, theta) = 0 and preserves autograd backpropagation via the Implicit Function Theorem."""
    # Find numerical root x*
    curr = float(x0)
    for _ in range(max_iter):
        f_val = _eval_scalar(func, curr)
        if abs(f_val) < tol:
            break
        # numerical derivative wrt x
        h = 1e-5
        f_h = _eval_scalar(func, curr + h)
        df_dx = (f_h - f_val) / h
        if abs(df_dx) < 1e-12:
            break
        curr = curr - f_val / df_dx

    # Derivative wrt x at x*
    h = 1e-4
    f_p = _eval_scalar(func, curr + h)
    f_m = _eval_scalar(func, curr - h)
    df_dx = float((f_p - f_m) / (2.0 * h))
    if abs(df_dx) < 1e-12:
        df_dx = 1e-12 if df_dx >= 0 else -1e-12

    # By Implicit Function Theorem:
    # x(theta) = x* - f(x*, theta) / (df/dx)
    # dx/dtheta = - (1 / (df/dx)) * (df/dtheta)
    x_const = _from_numpy(np.array([curr], dtype=np.float32))
    f_at_root = func(x_const)
    scale = _from_numpy(np.array([1.0 / df_dx], dtype=np.float32))

    return x_const - f_at_root * scale


__all__ = [
    "RootResult",
    "newton_raphson",
    "secant_method",
    "halley_method",
    "bisection",
    "differentiable_root",
]
