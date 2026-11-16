"""Classical numerical methods: quadrature, interpolation, optimization, and physical constants.

Implements:
53. quad: Adaptive Simpson quadrature with Richardson error control
54. gauss_legendre / fixed_quad: Golub-Welsch nodes/weights and fixed-order quadrature
55. dblquad: Iterated adaptive quadrature over a 2-D region
56. Interp1D: Piecewise 1-D interpolation (linear/nearest/previous/next)
57. CubicSpline: C2 cubic spline with derivatives and exact integration
58. minimize_scalar: Brent and golden-section scalar minimization
59. curve_fit: Levenberg-Marquardt nonlinear least squares
60. physical_constants: CODATA physical constants with scipy-compatible accessors
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple, Union
import math
import operator
import warnings
import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


__all__ = [
    "quad",
    "fixed_quad",
    "gauss_legendre",
    "dblquad",
    "Interp1D",
    "CubicSpline",
    "minimize_scalar",
    "ScalarOptResult",
    "curve_fit",
    "physical_constants",
    "Constant",
]


def _call_scalar_func(f: Callable[[float], float], x: float, name: str) -> float:
    """Evaluates a scalar integrand, rejecting non-scalar/non-finite results."""
    try:
        v = float(f(x))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, FloatingPointError) as exc:
        raise ValueError(f"{name}: integrand must return a real scalar, failed at x={x} ({exc})") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name}: integrand returned non-finite value {v} at x={x}")
    return v


def _check_tolerances(epsabs: float, epsrel: float, name: str) -> None:
    if not math.isfinite(epsabs) or epsabs < 0.0:
        raise ValueError(f"{name}: epsabs must be a finite non-negative float, got {epsabs!r}")
    if not math.isfinite(epsrel) or epsrel < 0.0:
        raise ValueError(f"{name}: epsrel must be a finite non-negative float, got {epsrel!r}")


def _check_limit(limit: int, name: str) -> int:
    if isinstance(limit, bool):
        raise ValueError(f"{name}: limit must be a positive integer, got {limit!r}")
    try:
        limit = operator.index(limit)
    except TypeError as exc:
        raise TypeError(f"{name}: limit must be an integer, got {limit!r}") from exc
    if limit < 1:
        raise ValueError(f"{name}: limit must be a positive integer, got {limit!r}")
    return limit


def _check_positive_int(raw: Any, label: str, name: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{name}: {label} must be a positive integer, got {raw!r}")
    try:
        val = operator.index(raw)
    except TypeError as exc:
        raise TypeError(f"{name}: {label} must be an integer, got {raw!r}") from exc
    if val < 1:
        raise ValueError(f"{name}: {label} must be a positive integer, got {raw!r}")
    return val


def _check_finite_interval(a: Any, b: Any, name: str) -> Tuple[float, float]:
    try:
        a_f, b_f = float(a), float(b)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: interval ends must be real scalars, got {a!r}, {b!r}") from exc
    if not (math.isfinite(a_f) and math.isfinite(b_f)):
        raise ValueError(f"{name}: only finite intervals are supported, got [{a!r}, {b!r}]")
    return a_f, b_f


def quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 200,
) -> Tuple[float, float]:
    """Adaptive Simpson quadrature of `f` over `[a, b]`.

    Returns `(value, abserr)` like `scipy.integrate.quad`. Only finite
    intervals are supported; reversed limits integrate with a negated sign.
    Warns (RuntimeWarning) and returns the best estimate when `limit`
    subdivisions are exhausted before the tolerance is met.
    """
    a_f, b_f = _check_finite_interval(a, b, "quad")
    _check_tolerances(epsabs, epsrel, "quad")
    limit = _check_limit(limit, "quad")
    if a_f == b_f:
        return 0.0, 0.0
    if a_f > b_f:
        val, err = quad(f, b_f, a_f, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return -val, err

    fa = _call_scalar_func(f, a_f, "quad")
    fb = _call_scalar_func(f, b_f, "quad")
    m = 0.5 * (a_f + b_f)
    fm = _call_scalar_func(f, m, "quad")
    whole = (b_f - a_f) / 6.0 * (fa + 4.0 * fm + fb)
    span = b_f - a_f

    total = 0.0
    err_total = 0.0
    stack = [(a_f, fa, m, fm, b_f, fb, whole)]
    intervals = 1
    hit_limit = False
    while stack:
        a0, fa0, m0, fm0, b0, fb0, whole0 = stack.pop()
        ml = 0.5 * (a0 + m0)
        mr = 0.5 * (m0 + b0)
        fml = _call_scalar_func(f, ml, "quad")
        fmr = _call_scalar_func(f, mr, "quad")
        left = (m0 - a0) / 6.0 * (fa0 + 4.0 * fml + fm0)
        right = (b0 - m0) / 6.0 * (fm0 + 4.0 * fmr + fb0)
        delta = left + right - whole0
        tol = epsabs * (b0 - a0) / span + epsrel * abs(left + right)
        if abs(delta) <= 15.0 * tol or intervals >= limit:
            if intervals >= limit and abs(delta) > 15.0 * tol:
                hit_limit = True
            total += left + right + delta / 15.0
            err_total += abs(delta) / 15.0
        else:
            intervals += 1
            stack.append((a0, fa0, ml, fml, m0, fm0, left))
            stack.append((m0, fm0, mr, fmr, b0, fb0, right))
    if hit_limit:
        warnings.warn(
            f"quad: maximum subdivisions (limit={limit}) reached; "
            "result may not meet the requested tolerance",
            RuntimeWarning,
            stacklevel=2,
        )
    return total, err_total


def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the `n`-point Gauss-Legendre rule on [-1, 1].

    Computed with the Golub-Welsch eigen-decomposition of the Jacobi matrix.
    """
    n = _check_positive_int(n, "n", "gauss_legendre")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    k = np.arange(1, n, dtype=np.float64)
    off = k / np.sqrt(4.0 * k * k - 1.0)
    jacobi = np.diag(off, 1) + np.diag(off, -1)
    vals, vecs = np.linalg.eigh(jacobi)
    order = np.argsort(vals)
    return vals[order], 2.0 * vecs[0, order] ** 2


def fixed_quad(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 5,
) -> float:
    """Fixed-order Gauss-Legendre quadrature of `f` over `[a, b]` (exact for degree < 2n)."""
    a_f, b_f = _check_finite_interval(a, b, "fixed_quad")
    if a_f == b_f:
        return 0.0
    if a_f > b_f:
        return -fixed_quad(f, b_f, a_f, n=n)
    x, w = gauss_legendre(n)
    mid = 0.5 * (b_f + a_f)
    half = 0.5 * (b_f - a_f)
    return float(half * sum(wi * _call_scalar_func(f, mid + half * xi, "fixed_quad") for xi, wi in zip(x, w)))


def dblquad(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    gfun: Callable[[float], float],
    hfun: Callable[[float], float],
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
) -> Tuple[float, float]:
    """Iterated integral of `f(y, x)` for x in [a, b], y in [gfun(x), hfun(x)].

    Argument order matches `scipy.integrate.dblquad`. Returns `(value,
    abserr)` where the error estimate covers the outer integral; inner
    tolerances are inherited from `epsabs`/`epsrel`.
    """
    a_f, b_f = _check_finite_interval(a, b, "dblquad")
    _check_tolerances(epsabs, epsrel, "dblquad")
    if a_f == b_f:
        return 0.0, 0.0
    if a_f > b_f:
        val, err = dblquad(f, b_f, a_f, gfun, hfun, epsabs=epsabs, epsrel=epsrel)
        return -val, err

    def inner(x: float) -> float:
        try:
            lo, hi = float(gfun(x)), float(hfun(x))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dblquad: bound functions must return real scalars at x={x} ({exc})") from exc
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"dblquad: non-finite inner bounds [{lo}, {hi}] at x={x}")
        val, _ = quad(lambda y: f(y, x), lo, hi, epsabs=epsabs, epsrel=epsrel)
        return val

    return quad(inner, a_f, b_f, epsabs=epsabs, epsrel=epsrel)


def _as_numpy_1d(x: Any, name: str) -> np.ndarray:
    arr = x.to_numpy() if isinstance(x, _Tensor) else np.asarray(x, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


class Interp1D:
    """Piecewise 1-D interpolation over strictly increasing nodes.

    `kind` is one of 'linear' ('slinear' alias), 'nearest', 'previous'
    ('zero' alias), or 'next'. `y` may be shape (n,) or (n, m) with columns
    interpolated independently. Out-of-domain queries raise unless
    `bounds_error=False`, in which case they return `fill_value`.
    """

    _KINDS = {"linear": "linear", "slinear": "linear", "nearest": "nearest",
              "previous": "previous", "zero": "previous", "next": "next"}

    def __init__(
        self,
        x: Any,
        y: Any,
        kind: str = "linear",
        bounds_error: bool = True,
        fill_value: float = np.nan,
    ):
        if kind not in self._KINDS:
            raise ValueError(f"Interp1D: unknown kind {kind!r} (expected one of {sorted(self._KINDS)})")
        self.kind = self._KINDS[kind]
        self.x = _as_numpy_1d(x, "Interp1D x")
        yy = y.to_numpy() if isinstance(y, _Tensor) else np.asarray(y, dtype=np.float64)
        yy = np.asarray(yy, dtype=np.float64)
        n = self.x.shape[0]
        if n < 1:
            raise ValueError("Interp1D: need at least one node")
        if yy.shape[0] != n or yy.ndim > 2:
            raise ValueError(f"Interp1D: y must have shape ({n},) or ({n}, m), got {yy.shape}")
        if n > 1 and not bool(np.all(np.diff(self.x) > 0.0)):
            raise ValueError("Interp1D: x must be strictly increasing")
        self._y1d = yy.ndim == 1
        self.y = yy if yy.ndim == 2 else yy[:, None]
        self.bounds_error = bool(bounds_error)
        self.fill_value = float(fill_value)

    def __call__(self, x_new: Any) -> Any:
        is_tensor = isinstance(x_new, _Tensor)
        xn = x_new.to_numpy() if is_tensor else np.asarray(x_new, dtype=np.float64)
        scalar = xn.ndim == 0
        flat = np.asarray(xn, dtype=np.float64).ravel()
        out = np.empty((flat.shape[0], self.y.shape[1]), dtype=np.float64)
        lo, hi = self.x[0], self.x[-1]
        oob = (flat < lo) | (flat > hi)
        if self.bounds_error and bool(np.any(oob)):
            bad = float(flat[oob][0])
            raise ValueError(f"Interp1D: query {bad} outside domain [{lo}, {hi}]")
        out[oob] = self.fill_value
        ok = ~oob
        if bool(np.any(ok)):
            out[ok] = self._evaluate(np.clip(flat[ok], lo, hi))
        if self._y1d:
            out = out[:, 0]
        if scalar:
            return out.reshape(())
        if is_tensor:
            return _from_numpy(np.ascontiguousarray(out.reshape(xn.shape + (() if self._y1d else (self.y.shape[1],))), dtype=xn.dtype))
        return out.reshape(xn.shape + (() if self._y1d else (self.y.shape[1],)))

    def _evaluate(self, flat: np.ndarray) -> np.ndarray:
        n = self.x.shape[0]
        if n == 1:
            return np.repeat(self.y, flat.shape[0], axis=0)
        if self.kind == "nearest":
            idx = np.searchsorted(self.x, flat, side="left")
            idx = np.clip(idx, 1, n - 1)
            left = self.x[idx - 1]
            right = self.x[idx]
            pick = np.where(flat - left <= right - flat, idx - 1, idx)
            return self.y[pick]
        if self.kind == "previous":
            idx = np.clip(np.searchsorted(self.x, flat, side="right") - 1, 0, n - 1)
            return self.y[idx]
        if self.kind == "next":
            idx = np.clip(np.searchsorted(self.x, flat, side="left"), 0, n - 1)
            return self.y[idx]
        idx = np.clip(np.searchsorted(self.x, flat, side="right") - 1, 0, n - 2)
        x0 = self.x[idx]
        x1 = self.x[idx + 1]
        t = (flat - x0) / (x1 - x0)
        return self.y[idx] * (1.0 - t)[:, None] + self.y[idx + 1] * t[:, None]


class CubicSpline:
    """C2 cubic spline through (x, y) with 'natural' or 'clamped' end conditions.

    `y` may be shape (n,) or (n, m). Exact on cubic polynomials. `extrapolate`
    mirrors scipy (end-polynomial evaluation outside the domain); with
    `extrapolate=False` out-of-domain queries raise ValueError.
    """

    def __init__(self, x: Any, y: Any, bc_type: str = "natural", extrapolate: bool = True):
        if bc_type not in ("natural", "clamped"):
            raise ValueError(f"CubicSpline: bc_type must be 'natural' or 'clamped', got {bc_type!r}")
        xs = _as_numpy_1d(x, "CubicSpline x")
        yy = y.to_numpy() if isinstance(y, _Tensor) else np.asarray(y, dtype=np.float64)
        yy = np.asarray(yy, dtype=np.float64)
        n = xs.shape[0]
        if n < 2:
            raise ValueError(f"CubicSpline: need at least 2 nodes, got {n}")
        if yy.shape[0] != n or yy.ndim > 2:
            raise ValueError(f"CubicSpline: y must have shape ({n},) or ({n}, m), got {yy.shape}")
        if not bool(np.all(np.diff(xs) > 0.0)):
            raise ValueError("CubicSpline: x must be strictly increasing")
        self.x = xs
        self._y1d = yy.ndim == 1
        self.extrapolate = bool(extrapolate)
        h = np.diff(xs)
        rhs = np.empty((n, yy.shape[1] if yy.ndim == 2 else 1), dtype=np.float64)
        target = yy if yy.ndim == 2 else yy[:, None]
        rhs[1:-1] = 6.0 * ((target[2:] - target[1:-1]) / h[1:, None] - (target[1:-1] - target[:-2]) / h[:-1, None])
        lower = np.zeros(n - 1)
        diag = np.zeros(n)
        upper = np.zeros(n - 1)
        lower[:-1] = h[:-1]
        diag[1:-1] = 2.0 * (h[:-1] + h[1:])
        upper[1:] = h[1:]
        if bc_type == "natural":
            diag[0] = diag[-1] = 1.0
            rhs[0] = rhs[-1] = 0.0
        else:  # clamped: zero end slopes
            diag[0] = 2.0 * h[0]
            upper[0] = h[0]
            rhs[0] = 6.0 * ((target[1] - target[0]) / h[0])
            lower[-1] = h[-1]
            diag[-1] = 2.0 * h[-1]
            rhs[-1] = 6.0 * (-(target[-1] - target[-2]) / h[-1])
        moments = CubicSpline._thomas(lower, diag, upper, rhs)
        self._coeffs = np.empty((n - 1, 4, moments.shape[1]), dtype=np.float64)
        m0, m1 = moments[:-1], moments[1:]
        y0, y1 = target[:-1], target[1:]
        hh = h[:, None]
        self._coeffs[:, 0] = y0
        self._coeffs[:, 1] = (y1 - y0) / hh - hh * (2.0 * m0 + m1) / 6.0
        self._coeffs[:, 2] = m0 / 2.0
        self._coeffs[:, 3] = (m1 - m0) / (6.0 * hh)

    @staticmethod
    def _thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        n = diag.shape[0]
        c = np.zeros(n - 1)
        d = rhs.copy()
        c[0] = upper[0] / diag[0]
        d[0] = d[0] / diag[0]
        for i in range(1, n - 1):
            denom = diag[i] - lower[i - 1] * c[i - 1]
            c[i] = upper[i] / denom
            d[i] = (d[i] - lower[i - 1] * d[i - 1]) / denom
        d[n - 1] = (d[n - 1] - lower[n - 2] * d[n - 2]) / (diag[n - 1] - lower[n - 2] * c[n - 2])
        for i in range(n - 2, -1, -1):
            d[i] = d[i] - c[i] * d[i + 1]
        return d

    @classmethod
    def _from_coeffs(cls, x: np.ndarray, coeffs: np.ndarray, y1d: bool, extrapolate: bool) -> "CubicSpline":
        obj = cls.__new__(cls)
        obj.x = np.asarray(x, dtype=np.float64)
        obj._coeffs = np.asarray(coeffs, dtype=np.float64)
        obj._y1d = bool(y1d)
        obj.extrapolate = bool(extrapolate)
        return obj

    def _interval(self, flat: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.x, flat, side="right") - 1, 0, self.x.shape[0] - 2)

    def __call__(self, x_new: Any) -> Any:
        is_tensor = isinstance(x_new, _Tensor)
        xn = x_new.to_numpy() if is_tensor else np.asarray(x_new, dtype=np.float64)
        scalar = xn.ndim == 0
        flat = np.asarray(xn, dtype=np.float64).ravel()
        if not self.extrapolate:
            lo, hi = self.x[0], self.x[-1]
            oob = (flat < lo) | (flat > hi)
            if bool(np.any(oob)):
                raise ValueError(f"CubicSpline: query {float(flat[oob][0])} outside domain [{lo}, {hi}]")
        idx = self._interval(np.clip(flat, self.x[0], self.x[-1]) if not self.extrapolate else flat)
        idx = np.clip(idx, 0, self._coeffs.shape[0] - 1)
        t = (flat - self.x[idx])[:, None]
        c = self._coeffs[idx]
        out = ((c[:, 3] * t + c[:, 2]) * t + c[:, 1]) * t + c[:, 0]
        if self._y1d:
            out = out[:, 0]
        shape = xn.shape + (() if self._y1d else (self._coeffs.shape[2],))
        if scalar:
            return out.reshape(())
        if is_tensor:
            return _from_numpy(np.ascontiguousarray(out.reshape(shape), dtype=xn.dtype))
        return out.reshape(shape)

    def derivative(self, nu: int = 1) -> Any:
        """nu-th derivative: CubicSpline for nu in (1, 2), Interp1D for nu=3."""
        if isinstance(nu, bool):
            raise ValueError(f"CubicSpline.derivative: nu must be 1, 2, or 3, got {nu!r}")
        try:
            nu = operator.index(nu)
        except TypeError as exc:
            raise TypeError(f"CubicSpline.derivative: nu must be an integer, got {nu!r}") from exc
        if nu not in (1, 2, 3):
            raise ValueError(f"CubicSpline.derivative: nu must be 1, 2, or 3, got {nu!r}")
        c = self._coeffs
        if nu == 1:
            dc = np.zeros_like(c)
            dc[:, 0] = c[:, 1]
            dc[:, 1] = 2.0 * c[:, 2]
            dc[:, 2] = 3.0 * c[:, 3]
            return CubicSpline._from_coeffs(self.x, dc, self._y1d, self.extrapolate)
        if nu == 2:
            dc = np.zeros_like(c)
            dc[:, 0] = 2.0 * c[:, 2]
            dc[:, 1] = 6.0 * c[:, 3]
            return CubicSpline._from_coeffs(self.x, dc, self._y1d, self.extrapolate)
        d3 = 6.0 * c[:, 3]
        d3_full = np.concatenate([d3, d3[-1:]], axis=0)
        if self._y1d:
            d3_full = d3_full[:, 0]
        return Interp1D(self.x, d3_full, kind="previous", bounds_error=not self.extrapolate)

    def integrate(self, a: float, b: float) -> Union[float, np.ndarray]:
        """Exact definite integral of the spline over [a, b] (must lie in the domain)."""
        try:
            a_f, b_f = float(a), float(b)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CubicSpline.integrate: bounds must be real scalars, got {a!r}, {b!r}") from exc
        lo, hi = float(self.x[0]), float(self.x[-1])
        if not (lo <= a_f <= hi and lo <= b_f <= hi):
            raise ValueError(f"CubicSpline.integrate: [{a_f}, {b_f}] outside domain [{lo}, {hi}]")
        if a_f == b_f:
            return 0.0 if self._y1d else np.zeros(self._coeffs.shape[2])
        if a_f > b_f:
            res = self.integrate(b_f, a_f)
            return -res
        knots = np.concatenate(([a_f], self.x[(self.x > a_f) & (self.x < b_f)], [b_f]))
        total = np.zeros(self._coeffs.shape[2])
        for s, e in zip(knots[:-1], knots[1:]):
            i = int(np.clip(np.searchsorted(self.x, s, side="right") - 1, 0, self._coeffs.shape[0] - 1))
            t0, t1 = s - self.x[i], e - self.x[i]
            a0, b0, c0, d0 = (self._coeffs[i, j] for j in range(4))
            total += (a0 * (t1 - t0) + b0 * (t1 ** 2 - t0 ** 2) / 2.0
                      + c0 * (t1 ** 3 - t0 ** 3) / 3.0 + d0 * (t1 ** 4 - t0 ** 4) / 4.0)
        return float(total[0]) if self._y1d else total


class ScalarOptResult(NamedTuple):
    """Result of `minimize_scalar`: location, value, evals, iterations, converged."""

    x: float
    fun: float
    nfev: int
    nit: int
    success: bool


def minimize_scalar(
    f: Callable[[float], float],
    a: float,
    b: float,
    method: str = "brent",
    tol: float = 1.48e-8,
    maxiter: int = 500,
) -> ScalarOptResult:
    """Minimizes unimodal `f` over the bracket [a, b] ('brent' or 'golden').

    The minimum is assumed to lie strictly inside (a, b). Returns a
    ScalarOptResult; `success` is False (no exception) if `maxiter` is hit.
    """
    a_f, b_f = _check_finite_interval(a, b, "minimize_scalar")
    if not a_f < b_f:
        raise ValueError(f"minimize_scalar: need a < b, got [{a_f}, {b_f}]")
    if method not in ("brent", "golden"):
        raise ValueError(f"minimize_scalar: method must be 'brent' or 'golden', got {method!r}")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"minimize_scalar: tol must be positive, got {tol!r}")
    maxiter = _check_positive_int(maxiter, "maxiter", "minimize_scalar")

    def fval(x: float) -> float:
        return _call_scalar_func(f, x, "minimize_scalar")

    if method == "golden":
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        lo, hi = a_f, b_f
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc, fd = fval(c), fval(d)
        nfev = 2
        for nit in range(1, maxiter + 1):
            if abs(hi - lo) < tol * (abs(c) + abs(d)):
                x = 0.5 * (lo + hi)
                return ScalarOptResult(x, fval(x), nfev + 1, nit, True)
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = fval(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = fval(d)
            nfev += 1
        x = 0.5 * (lo + hi)
        return ScalarOptResult(x, fval(x), nfev + 1, maxiter, False)

    gr = 0.3819660112501051
    lo, hi = a_f, b_f
    x = w = v = lo + gr * (hi - lo)
    fx = fw = fv = fval(x)
    nfev = 1
    d = e = 0.0
    for nit in range(1, maxiter + 1):
        mid = 0.5 * (lo + hi)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1
        if abs(x - mid) <= tol2 - 0.5 * (hi - lo):
            return ScalarOptResult(x, fx, nfev, nit, True)
        take_parabolic = abs(e) > tol1
        if take_parabolic:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            e_tmp = e
            e = d
            if abs(p) >= abs(0.5 * q * e_tmp) or p <= q * (lo - x) or p >= q * (hi - x):
                take_parabolic = False
            else:
                d = p / q
                u = x + d
                if u - lo < tol2 or hi - u < tol2:
                    d = math.copysign(tol1, mid - x)
        if not take_parabolic:
            e = lo - x if x >= mid else hi - x
            d = gr * e
        u = x + d if (take_parabolic and abs(d) >= tol1) else x + math.copysign(tol1, d if d != 0.0 else (mid - x))
        fu = fval(u)
        nfev += 1
        if fu <= fx:
            if u >= x:
                lo = x
            else:
                hi = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                lo = u
            else:
                hi = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    return ScalarOptResult(x, fx, nfev, maxiter, False)


def curve_fit(
    f: Callable[..., Any],
    xdata: Any,
    ydata: Any,
    p0: Any,
    sigma: Optional[Any] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    absolute_sigma: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt nonlinear least squares (scipy `curve_fit` calling convention).

    Fits `ydata = f(xdata, *params)` starting from `p0`. Returns `(popt,
    pcov)`; raises RuntimeError if the iteration budget is exhausted.
    """
    xd = np.asarray(xdata.to_numpy() if isinstance(xdata, _Tensor) else xdata, dtype=np.float64)
    yd = np.asarray(ydata.to_numpy() if isinstance(ydata, _Tensor) else ydata, dtype=np.float64)
    p = np.asarray(p0, dtype=np.float64).ravel().copy()
    if xd.ndim != 1 or yd.ndim != 1 or xd.shape[0] != yd.shape[0]:
        raise ValueError("curve_fit: xdata and ydata must be 1-D with matching lengths, "
                         f"got {xd.shape} and {yd.shape}")
    m, n = yd.shape[0], p.shape[0]
    if n < 1:
        raise ValueError("curve_fit: p0 must contain at least one parameter")
    if m < n:
        raise ValueError(f"curve_fit: underdetermined problem ({m} points, {n} parameters)")
    if sigma is not None:
        sw = np.asarray(sigma, dtype=np.float64).ravel()
        if sw.shape[0] != m:
            raise ValueError(f"curve_fit: sigma must have length {m}, got {sw.shape[0]}")
        if bool(np.any(sw <= 0.0)) or bool(np.any(~np.isfinite(sw))):
            raise ValueError("curve_fit: sigma must be finite and positive")
    else:
        sw = np.ones(m)
    maxiter = _check_positive_int(maxiter, "maxiter", "curve_fit")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"curve_fit: tol must be positive, got {tol!r}")

    def residuals(pp: np.ndarray) -> np.ndarray:
        try:
            pred = np.asarray(f(xd, *pp), dtype=np.float64).ravel()
        except Exception as exc:
            raise ValueError(f"curve_fit: model function failed ({exc})") from exc
        if pred.shape[0] != m:
            raise ValueError(f"curve_fit: model must return {m} values, got {pred.shape[0]}")
        if bool(np.any(~np.isfinite(pred))):
            raise ValueError("curve_fit: model returned non-finite values")
        return (yd - pred) / sw

    def jacobian(pp: np.ndarray) -> np.ndarray:
        jac = np.empty((m, n))
        step = np.sqrt(np.finfo(float).eps) * (1.0 + np.abs(pp))
        for j in range(n):
            dp = np.zeros(n)
            dp[j] = step[j]
            jac[:, j] = (residuals(pp + dp) - residuals(pp - dp)) / (2.0 * step[j])
        return jac

    r = residuals(p)
    cost = float(r @ r)
    lam = 0.01
    nu = 2.0
    jac = jacobian(p)
    jt_r = jac.T @ r
    if bool(np.all(np.abs(jt_r) < tol)):
        return p.copy(), _curve_fit_pcov(jac, r, m, n, absolute_sigma)
    for _ in range(maxiter):
        jt_j = jac.T @ jac
        diag = np.diag(jt_j).copy()
        damped = jt_j + lam * np.diag(diag)
        try:
            olm = np.linalg.solve(damped, -jt_r)
        except np.linalg.LinAlgError:
            olm = np.linalg.lstsq(damped, -jt_r, rcond=None)[0]
        p_new = p + olm
        try:
            r_new = residuals(p_new)
        except ValueError:
            lam *= nu
            nu *= 2.0
            continue
        cost_new = float(r_new @ r_new)
        rho = (cost - cost_new) / float(olm @ (lam * diag * olm - jt_r)) if cost != cost_new else 0.0
        if rho > 0.0:
            p = p_new
            r = r_new
            cost = cost_new
            jac = jacobian(p)
            jt_r = jac.T @ r
            lam = max(lam / 3.0, 1e-12)
            nu = 2.0
            if bool(np.all(np.abs(jt_r) < tol)) or float(olm @ olm) < tol * tol * (1.0 + float(p @ p)):
                return p.copy(), _curve_fit_pcov(jac, r, m, n, absolute_sigma)
        else:
            lam *= nu
            nu *= 2.0
    raise RuntimeError(f"curve_fit: optimal parameters not found within maxiter={maxiter}")


def _curve_fit_pcov(jac: np.ndarray, r: np.ndarray, m: int, n: int, absolute_sigma: bool) -> np.ndarray:
    jt_j = jac.T @ jac
    try:
        cov = np.linalg.inv(jt_j)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jt_j)
    if not absolute_sigma and m > n:
        cov = cov * (float(r @ r) / (m - n))
    return cov


class Constant(NamedTuple):
    """A physical constant: value, unit string, and standard uncertainty."""

    value: float
    unit: str
    uncertainty: float


physical_constants: Dict[str, Constant] = {
    "speed of light in vacuum": Constant(299792458.0, "m s^-1", 0.0),
    "Planck constant": Constant(6.62607015e-34, "J Hz^-1", 0.0),
    "reduced Planck constant": Constant(1.0545718176461565e-34, "J s", 0.0),
    "elementary charge": Constant(1.602176634e-19, "C", 0.0),
    "Boltzmann constant": Constant(1.380649e-23, "J K^-1", 0.0),
    "Avogadro constant": Constant(6.02214076e23, "mol^-1", 0.0),
    "electron mass": Constant(9.1093837139e-31, "kg", 2.8e-40),
    "proton mass": Constant(1.67262192595e-27, "kg", 5.2e-37),
    "neutron mass": Constant(1.67492750056e-27, "kg", 8.5e-37),
    "atomic mass constant": Constant(1.66053906892e-27, "kg", 5.2e-37),
    "Newtonian constant of gravitation": Constant(6.6743e-11, "m^3 kg^-1 s^-2", 1.5e-15),
    "fine-structure constant": Constant(0.0072973525643, "", 1.1e-12),
    "Rydberg constant": Constant(10973731.568157, "m^-1", 1.2e-05),
    "Bohr radius": Constant(5.29177210544e-11, "m", 8.2e-21),
    "vacuum electric permittivity": Constant(8.8541878188e-12, "F m^-1", 1.4e-21),
    "vacuum mag. permeability": Constant(1.25663706127e-06, "N A^-2", 2e-16),
    "molar gas constant": Constant(8.31446261815324, "J mol^-1 K^-1", 0.0),
    "Faraday constant": Constant(96485.33212331001, "C mol^-1", 0.0),
    "Stefan-Boltzmann constant": Constant(5.6703744191844314e-08, "W m^-2 K^-4", 0.0),
    "electron volt": Constant(1.602176634e-19, "J", 0.0),
}


def value(key: str) -> float:
    """Value of the named physical constant (KeyError if unknown, like scipy)."""
    return physical_constants[key].value


def unit(key: str) -> str:
    """Unit string of the named physical constant (KeyError if unknown, like scipy)."""
    return physical_constants[key].unit


def precision(key: str) -> float:
    """Relative standard uncertainty of the named constant (KeyError if unknown)."""
    c = physical_constants[key]
    return abs(c.uncertainty / c.value) if c.value != 0.0 else 0.0


# Convenience aliases (module attributes only, intentionally not star-exported).
c = value("speed of light in vacuum")
h = value("Planck constant")
hbar = value("reduced Planck constant")
e = value("elementary charge")
k = value("Boltzmann constant")
N_A = value("Avogadro constant")
m_e = value("electron mass")
m_p = value("proton mass")
G = value("Newtonian constant of gravitation")
R = value("molar gas constant")
eV = value("electron volt")
