"""Comprehensive Special Mathematical Functions for Oxideon.

Provides tensor-friendly, broadcasting special mathematical functions,
including error functions, gamma family,
modified Bessel functions, normal distribution CDFs, and information-theoretic operators.
"""

import math
from typing import Any, Optional, Sequence, Union
import numpy as np
from ._contig import contig as _contig

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, _Tensor):
        return x.to_numpy()
    return np.asarray(x, dtype=np.float32)


def _wrap(arr: np.ndarray, orig_t: Optional[_Tensor] = None) -> _Tensor:
    dtype = orig_t.to_numpy().dtype if orig_t is not None else np.float32
    return _from_numpy(_contig(arr, dtype=dtype))


def _differentiable_eval(input: _Tensor, forward_fn, deriv_fn) -> _Tensor:
    if getattr(input, "requires_grad", False):
        x0 = input.to_numpy()
        dtype = x0.dtype if x0.dtype in (np.float32, np.float64) else np.float32
        x0 = x0.astype(dtype)
        y0 = forward_fn(x0).astype(dtype)
        deriv = deriv_fn(x0).astype(dtype)
        c0 = _from_numpy(_contig(y0 - deriv * x0))
        c1 = _from_numpy(_contig(deriv))
        return c0 + c1 * input
    return _wrap(forward_fn(_to_np(input)), input)


# Vectorized standard math functions
_erf_vec = np.vectorize(math.erf)
_erfc_vec = np.vectorize(math.erfc)
_lgamma_vec = np.vectorize(math.lgamma)


def erf(input: _Tensor) -> _Tensor:
    """Computes the error function of each element in input."""
    if hasattr(input, "erf"):
        return input.erf()
    return _differentiable_eval(
        input,
        lambda a: _erf_vec(a.astype(np.float64)),
        lambda a: (2.0 / math.sqrt(math.pi)) * np.exp(-a * a),
    )


def erfc(input: _Tensor) -> _Tensor:
    """Computes the complementary error function of each element in input."""
    if hasattr(input, "erfc"):
        return input.erfc()
    arr = _to_np(input).astype(np.float64)
    res = _erfc_vec(arr)
    return _wrap(res, input)


def erfcx(input: _Tensor) -> _Tensor:
    """Computes the scaled complementary error function exp(x^2) * erfc(x)."""
    if hasattr(input, "erfcx"):
        return input.erfcx()
    x = _to_np(input).astype(np.float64)
    return _wrap(np.exp(x * x) * _erfc_vec(x), input)


def _erfinv_core(y: np.ndarray) -> np.ndarray:
    """Inverse erf on the open interval (-1, 1); callers set domain edges."""
    # Winitzki approximation as initial seed
    a = 0.147
    sgn = np.sign(y)
    ln1_y2 = np.log(np.maximum(1e-15, 1.0 - y * y))
    term1 = 2.0 / (np.pi * a) + ln1_y2 / 2.0
    inner = term1 * term1 - ln1_y2 / a
    x = sgn * np.sqrt(np.sqrt(np.maximum(0.0, inner)) - term1)

    # Newton-Raphson refinement (5 iterations for f64 tightness)
    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    with np.errstate(all="ignore"):
        for _ in range(5):
            err = _erf_vec(x) - y
            deriv = two_over_sqrt_pi * np.exp(-x * x)
            x = x - err / (deriv + 1e-18)
    return x


def erfinv(input: _Tensor) -> _Tensor:
    """Computes the inverse error function of each element in input."""
    if hasattr(input, "erfinv"):
        return input.erfinv()
    y = _to_np(input).astype(np.float64)
    interior = (y > -1.0) & (y < 1.0)
    x = np.full_like(y, np.nan)
    if np.any(interior):
        x[interior] = _erfinv_core(y[interior])
    x[y == 1.0] = np.inf
    x[y == -1.0] = -np.inf
    # |y| > 1 stays NaN; NaN input flows through as NaN.
    return _wrap(x, input)


def exp2(input: _Tensor) -> _Tensor:
    """Computes the base two exponential function of each element in input."""
    if hasattr(input, "exp2"):
        return input.exp2()
    arr = _to_np(input)
    res = np.exp2(arr)
    return _wrap(res, input)


def expm1(input: _Tensor) -> _Tensor:
    """Computes the exponential of each element minus 1: exp(x) - 1."""
    if hasattr(input, "expm1"):
        return input.expm1()
    return _differentiable_eval(
        input,
        np.expm1,
        np.exp,
    )


def expit(input: _Tensor) -> _Tensor:
    """Computes the logistic sigmoid function: 1 / (1 + exp(-x))."""
    def _sig(a):
        return np.where(a >= 0, 1.0 / (1.0 + np.exp(-a)), np.exp(a) / (1.0 + np.exp(a)))
    return _differentiable_eval(
        input,
        _sig,
        lambda a: _sig(a) * (1.0 - _sig(a)),
    )


def logit(input: _Tensor, eps: Optional[float] = None) -> _Tensor:
    """Computes the logit (inverse sigmoid) of each element in input."""
    if hasattr(input, "logit"):
        return input.logit(eps)
    def _logit_fwd(a):
        clipped = np.clip(a, eps, 1.0 - eps) if eps is not None else a
        with np.errstate(all="ignore"):
            return np.log(clipped / (1.0 - clipped))
    def _logit_deriv(a):
        clipped = np.clip(a, eps, 1.0 - eps) if eps is not None else a
        with np.errstate(all="ignore"):
            return 1.0 / np.maximum(clipped * (1.0 - clipped), 1e-15)
    return _differentiable_eval(input, _logit_fwd, _logit_deriv)


def _gammaln_np(arr: np.ndarray) -> np.ndarray:
    """Full-domain ln|Gamma| ourselves: +inf at poles, reflection left of 0."""
    res = np.full_like(arr, np.nan, dtype=np.float64)
    # Poles at 0 and the negative integers read +inf.
    res[(arr == 0.0) | ((arr < 0.0) & (arr == np.floor(arr)))] = np.inf
    pos = arr > 0.0
    if np.any(pos):
        res[pos] = _lgamma_vec(arr[pos])
    # Negative non-integers via Euler reflection:
    # ln|G(x)| = ln(pi) - ln|sin(pi x)| - lgamma(1 - x).
    neg = (arr < 0.0) & (arr != np.floor(arr))
    if np.any(neg):
        an = arr[neg]
        with np.errstate(all="ignore"):
            res[neg] = (math.log(math.pi) - np.log(np.abs(np.sin(math.pi * an)))
                        - _lgamma_vec(1.0 - an))
    return res


def gammaln(input: _Tensor) -> _Tensor:
    """Computes the natural logarithm of the absolute value of the gamma function on input."""
    if hasattr(input, "lgamma"):
        return input.lgamma()
    arr = _to_np(input).astype(np.float64)
    return _wrap(_gammaln_np(arr), input)


loggamma = gammaln


def _digamma_pos(x: np.ndarray) -> np.ndarray:
    """Digamma for strictly positive x: recurrence shift + asymptotic series."""
    flat_x = x.flatten()
    flat_res = np.zeros_like(flat_x)

    for i, val in enumerate(flat_x):
        shift = 0.0
        while val < 7.0:
            shift -= 1.0 / max(val, 1e-15)
            val += 1.0
        inv_val = 1.0 / val
        inv_val2 = inv_val * inv_val
        val_res = math.log(val) - 0.5 * inv_val - inv_val2 * (
            1.0 / 12.0 - inv_val2 * (
                1.0 / 120.0 - inv_val2 * (1.0 / 252.0)
            )
        )
        flat_res[i] = val_res + shift

    return flat_res.reshape(x.shape)


def digamma(input: _Tensor) -> _Tensor:
    """Computes the logarithmic derivative of the gamma function on input."""
    if hasattr(input, "digamma"):
        return input.digamma()
    x = _to_np(input).astype(np.float64)
    res = np.full_like(x, np.nan)
    pos = x > 0.0
    res[pos] = _digamma_pos(x[pos])
    res[x == 0.0] = -np.inf
    # Negative non-integers via reflection psi(x) = psi(1-x) - pi/tan(pi x);
    # negative integers are poles (NaN).
    neg = (x < 0.0) & (x != np.floor(x))
    if np.any(neg):
        xn = x[neg]
        with np.errstate(all="ignore"):
            res[neg] = _digamma_pos(1.0 - xn) - math.pi / np.tan(math.pi * xn)
    return _wrap(res, input)


psi = digamma


def polygamma(n: int, input: _Tensor) -> _Tensor:
    """Computes the n-th derivative of the digamma function on input."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if hasattr(input, "polygamma"):
        return input.polygamma(n)
    if n == 0:
        return digamma(input)
    x = _to_np(input).astype(np.float64)
    # Direct sum plus Euler-Maclaurin tail: without the tail the N=100
    # truncation reads 0.2% low at x=0.5 for n=1.
    n_terms = 1000
    factorial_n = math.factorial(n)
    sign = (-1) ** (n + 1)
    res = np.zeros_like(x)
    with np.errstate(all="ignore"):
        for k in range(n_terms):
            res += 1.0 / ((x + k) ** (n + 1))
        t = x + n_terms
        res += 1.0 / (n * t ** n) + 0.5 / t ** (n + 1)
    return _wrap(sign * factorial_n * res, input)


def multigammaln(input: _Tensor, p: int) -> _Tensor:
    """Computes the multivariate log-gamma function of dimension p."""
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    if hasattr(input, "mvlgamma"):
        return input.mvlgamma(p)
    a = _to_np(input).astype(np.float64)
    res = np.full_like(a, 0.25 * p * (p - 1) * math.log(math.pi))
    for j in range(1, p + 1):
        res = res + _gammaln_np(a + 0.5 * (1.0 - j))
    return _wrap(res, input)


def i0(input: _Tensor) -> _Tensor:
    """Computes the zeroth order modified Bessel function of the first kind."""
    if hasattr(input, "i0"):
        return input.i0()
    arr = _to_np(input)
    res = np.i0(arr)
    return _wrap(res, input)


def i0e(input: _Tensor) -> _Tensor:
    """Computes the exponentially scaled zeroth order modified Bessel function of the first kind."""
    if hasattr(input, "i0e"):
        return input.i0e()
    arr = _to_np(input)
    res = np.i0(arr) * np.exp(-np.abs(arr))
    return _wrap(res, input)


def i1(input: _Tensor) -> _Tensor:
    """Computes the first order modified Bessel function of the first kind."""
    if hasattr(input, "i1"):
        return input.i1()
    x = _to_np(input).astype(np.float64)
    res = np.zeros_like(x)
    tiny = np.abs(x) < 1e-8
    res[tiny] = x[tiny] / 2.0
    gen = ~tiny
    if np.any(gen):
        # Miller's downward recurrence, normalized against numpy's exact I0:
        # I_{n-1} = I_{n+1} + (2n/x) I_n, seeded b_{M+1}=0, b_M=1.
        xg = x[gen]
        # Miller needs M past e|x|/2 for the seed transient to die.
        m = min(int(1.5 * np.max(np.abs(xg))) + 40, 2000)
        b_next2 = np.zeros_like(xg)
        b_next1 = np.ones_like(xg)
        with np.errstate(all="ignore"):
            for n in range(m, 0, -1):
                b = b_next2 + (2.0 * n / xg) * b_next1
                if n == 2:
                    b1 = b
                if n == 1:
                    b0 = b
                b_next2, b_next1 = b_next1, b
                # The seed scale is arbitrary (fixed by normalization), so
                # keep the downward values away from overflow for huge x.
                if n > 1 and np.any(np.abs(b_next1) > 1e200):
                    b_next1 = b_next1 / 1e200
                    b_next2 = b_next2 / 1e200
                    if n <= 2:
                        b1 = b1 / 1e200
        scale = np.i0(xg) / b0
        res[gen] = scale * b1
    return _wrap(res, input)


def i1e(input: _Tensor) -> _Tensor:
    """Computes the exponentially scaled first order modified Bessel function of the first kind."""
    if hasattr(input, "i1e"):
        return input.i1e()
    x = _to_np(input)
    i1_val = _to_np(i1(input))
    res = i1_val * np.exp(-np.abs(x))
    return _wrap(res, input)


def ndtr(input: _Tensor) -> _Tensor:
    """Computes the area under the standard Gaussian probability density function from minus infinity to input."""
    if hasattr(input, "ndtr"):
        return input.ndtr()
    x = _to_np(input).astype(np.float64)
    res = 0.5 * (1.0 + _erf_vec(x / math.sqrt(2.0)))
    return _wrap(res, input)


def ndtri(input: _Tensor) -> _Tensor:
    """Computes the argument x for which the area under the Gaussian PDF from minus infinity to x is input."""
    if hasattr(input, "ndtri"):
        return input.ndtri()
    p = _to_np(input).astype(np.float64)
    res = np.full_like(p, np.nan)
    interior = (p > 0.0) & (p < 1.0)
    if np.any(interior):
        res[interior] = math.sqrt(2.0) * _erfinv_core(2.0 * p[interior] - 1.0)
    res[p == 0.0] = -np.inf
    res[p == 1.0] = np.inf
    # p outside [0, 1] stays NaN; NaN input flows through as NaN.
    return _wrap(res, input)


def log1p(input: _Tensor) -> _Tensor:
    """Computes the natural logarithm of (1 + input): log(1 + x)."""
    if hasattr(input, "log1p"):
        return input.log1p()
    arr = _to_np(input)
    res = np.log1p(arr)
    return _wrap(res, input)


def sinc(input: _Tensor) -> _Tensor:
    """Computes the normalized sinc of input: sin(pi * x) / (pi * x)."""
    if hasattr(input, "sinc"):
        return input.sinc()
    def _sinc_deriv(a):
        px = np.pi * a
        with np.errstate(all="ignore"):
            return np.where(np.abs(a) < 1e-7, 0.0, (px * np.cos(px) - np.sin(px)) / (np.pi * a * a + 1e-18))
    return _differentiable_eval(input, np.sinc, _sinc_deriv)


def zeta(input: _Tensor, q: Union[_Tensor, float] = 1.0) -> _Tensor:
    """Computes the Hurwitz zeta function.

    Direct summation with an integral tail correction; requires x > 1
    (x == 1 diverges to +inf; x < 1 needs analytic continuation, which
    this shim does not implement, so it reads NaN there).
    """
    if hasattr(input, "zeta"):
        if not isinstance(q, _Tensor):
            from .primitives import tensor
            q = tensor(q, dtype=input.dtype, device=input.device)
        return input.zeta(q)
    x = _to_np(input).astype(np.float64)
    q_val = _to_np(q).astype(np.float64) if isinstance(q, _Tensor) else float(q)
    res = np.full_like(x, np.nan)
    conv = x > 1.0
    res[x == 1.0] = np.inf
    if np.any(conv):
        xc = x[conv]
        n_terms = 2000
        with np.errstate(all="ignore"):
            s = np.zeros_like(xc)
            for k in range(n_terms):
                s += 1.0 / ((k + q_val) ** xc)
            t = n_terms + q_val
            s += t ** (1.0 - xc) / (xc - 1.0) + t ** (-xc) / 2.0
            res[conv] = s
    return _wrap(res, input)


def softmax(input: _Tensor, dim: Optional[int] = None) -> _Tensor:
    """Computes the softmax along the specified dimension."""
    from .functional import softmax as _f_softmax
    return _f_softmax(input, dim=dim if dim is not None else -1)


def log_softmax(input: _Tensor, dim: Optional[int] = None) -> _Tensor:
    """Computes the log_softmax along the specified dimension."""
    from .functional import log_softmax as _f_log_softmax
    return _f_log_softmax(input, dim=dim if dim is not None else -1)


def entr(input: _Tensor) -> _Tensor:
    """Computes the entropy elementwise: -x * ln(x)."""
    if hasattr(input, "entr"):
        return input.entr()
    x = _to_np(input).astype(np.float64)
    with np.errstate(all="ignore"):
        res = np.where(x > 0, -x * np.log(x), np.where(x == 0, 0.0, -np.inf))
    return _wrap(res, input)


def rel_entr(input: _Tensor, other: _Tensor) -> _Tensor:
    """Computes the relative entropy elementwise: x * ln(x / y).

    Matches the convention ``xlogy(x, x) - xlogy(x, y)``: 0 when
    x == 0, +inf when x > 0 and y == 0, NaN for negative arguments.
    """
    x = _to_np(input).astype(np.float64)
    y = _to_np(other).astype(np.float64)
    res = np.full(np.broadcast_shapes(x.shape, y.shape), np.nan)
    with np.errstate(all="ignore"):
        both_pos = (x > 0) & (y > 0)
        res = np.where(both_pos, x * (np.log(x) - np.log(y)), res)
        res = np.where(x == 0, 0.0, res)
        res = np.where((x > 0) & (y == 0), np.inf, res)
    return _wrap(res, input)


def round(input: _Tensor) -> _Tensor:
    """Rounds elements of input to the nearest integer."""
    arr = _to_np(input)
    res = np.round(arr)
    return _wrap(res, input)


def log_ndtr(input: _Tensor) -> _Tensor:
    """Computes the natural log of the standard normal cumulative distribution function: ln(Phi(x))."""
    if hasattr(input, "log_ndtr"):
        return input.log_ndtr()
    x = _to_np(input).astype(np.float64)
    neg_mask = x < -10.0
    res = np.zeros_like(x)
    if np.any(neg_mask):
        xm = x[neg_mask]
        res[neg_mask] = -0.5 * math.log(2.0 * math.pi) - 0.5 * xm * xm - np.log(-xm) + np.log(1.0 - 1.0 / (xm * xm) + 3.0 / (xm ** 4))
    if np.any(~neg_mask):
        xp = x[~neg_mask]
        phi = 0.5 * (1.0 + _erf_vec(xp / math.sqrt(2.0)))
        res[~neg_mask] = np.log(np.maximum(phi, 1e-300))
    return _wrap(res, input)


def _bessel_j01(x: np.ndarray) -> tuple:
    """J0 and J1 via Miller's downward recurrence (accurate ~1e-13).

    Falls back to the leading-order Hankel expansion past |x| = 200, where
    the recurrence would need a long chain; accuracy there is ~1e-2.
    """
    j0 = np.zeros_like(x)
    j1 = np.zeros_like(x)
    tiny = np.abs(x) < 1e-8
    j0[tiny] = 1.0 - x[tiny] * x[tiny] / 4.0
    j1[tiny] = x[tiny] / 2.0
    gen = (~tiny) & (np.abs(x) <= 200.0)
    if np.any(gen):
        # J_{n-1} = (2n/x) J_n - J_{n+1}, seeded b_{M+1}=0, b_M=1,
        # normalized by J0 + 2 J2 + 2 J4 + ... = 1.
        xg = x[gen]
        # Miller needs M past e|x|/2 for the seed transient to die.
        m = int(1.5 * np.max(np.abs(xg))) + 40
        b_next2 = np.zeros_like(xg)
        b_next1 = np.ones_like(xg)
        norm = np.zeros_like(xg)
        with np.errstate(all="ignore"):
            for n in range(m, 0, -1):
                # b holds b_{n-1} here; b_next1 still holds b_n.
                b = (2.0 * n / xg) * b_next1 - b_next2
                if n % 2 == 0:
                    norm = norm + 2.0 * b_next1
                if n == 2:
                    b1 = b
                if n == 1:
                    b0 = b
                b_next2, b_next1 = b_next1, b
        # norm = 2*b_2 + 2*b_4 + ... ; adding b_0 completes the identity
        # J0 + 2*J2 + 2*J4 + ... = 1 used for the scale.
        scale = 1.0 / (b0 + norm)
        j0[gen] = scale * b0
        j1[gen] = scale * b1
    far = np.abs(x) > 200.0
    if np.any(far):
        xl = np.abs(x[far])
        s = np.sign(x[far])
        j0[far] = np.sqrt(2.0 / (np.pi * xl)) * np.cos(xl - np.pi / 4.0)
        j1[far] = np.sqrt(2.0 / (np.pi * xl)) * np.cos(xl - 3.0 * np.pi / 4.0) * s
    return j0, j1


def bessel_j0(input: _Tensor) -> _Tensor:
    """Computes the Bessel function of the first kind of order 0: J_0(x)."""
    if hasattr(input, "bessel_j0"):
        return input.bessel_j0()
    x = _to_np(input).astype(np.float64)
    j0, _ = _bessel_j01(x)
    return _wrap(j0, input)


def bessel_j1(input: _Tensor) -> _Tensor:
    """Computes the Bessel function of the first kind of order 1: J_1(x)."""
    if hasattr(input, "bessel_j1"):
        return input.bessel_j1()
    x = _to_np(input).astype(np.float64)
    _, j1 = _bessel_j01(x)
    return _wrap(j1, input)


def lambert_w(input: _Tensor) -> _Tensor:
    """Computes the principal branch of the Lambert W function W_0(x), where W(x) * exp(W(x)) = x."""
    if hasattr(input, "lambert_w"):
        return input.lambert_w()
    x = _to_np(input).astype(np.float64)
    res = np.full_like(x, np.nan)
    valid = x >= -1.0 / math.e
    # x < -1/e has no real value (NaN); NaN input flows through as NaN.
    if np.any(valid):
        xv = x[valid]
        # Initial guess
        w = np.where(xv < 1.0, xv, np.log(np.maximum(xv, 1e-12)))

        # Halley's method; 25 iterations because convergence degrades to
        # linear near the x = -1/e branch point.
        with np.errstate(all="ignore"):
            for _ in range(25):
                ew = np.exp(w)
                wew = w * ew
                wew_x = wew - xv
                deriv = ew * (w + 1.0) - (w + 2.0) * wew_x / (2.0 * w + 2.0)
                w = w - wew_x / (deriv + 1e-18)
        res[valid] = w

    return _wrap(res, input)


def xlogy(input: Any, other: Any) -> Any:
    """Computes input * log(other), returning 0 when input == 0."""
    if isinstance(input, _Tensor):
        if not isinstance(other, _Tensor):
            from .primitives import tensor
            other = tensor(other, dtype=input.dtype, device=input.device)
        return input.xlogy(other)
    x = _to_np(input).astype(np.float64)
    y = _to_np(other).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.log(y)
        res = np.where(x == 0.0, 0.0, x * log_y)
    return _wrap(res, input if isinstance(input, _Tensor) else (other if isinstance(other, _Tensor) else None))


def xlog1py(input: _Tensor, other: Union[_Tensor, float]) -> _Tensor:
    """Computes input * log(1 + other), returning 0 when input == 0."""
    if hasattr(input, "xlog1py"):
        if not isinstance(other, _Tensor):
            from .primitives import tensor
            other = tensor(other, dtype=input.dtype, device=input.device)
        return input.xlog1py(other)
    x = _to_np(input).astype(np.float64)
    y = _to_np(other).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log1p_y = np.log1p(y)
        res = np.where(x == 0.0, 0.0, x * log1p_y)
    return _wrap(res, input if isinstance(input, _Tensor) else (other if isinstance(other, _Tensor) else None))


def log1pexp(input: _Tensor) -> _Tensor:
    """Computes log(1 + exp(x)) in a numerically stable manner."""
    x = _to_np(input).astype(np.float64)
    with np.errstate(over="ignore"):
        res = np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -50.0, 20.0))))
    return _wrap(res, input)


def gammainc(input: Any, other: Union[_Tensor, float]) -> _Tensor:
    """Computes the regularized lower incomplete gamma function P(a, x)."""
    if not isinstance(input, _Tensor) and isinstance(other, _Tensor):
        from .primitives import tensor
        input = tensor(input, dtype=other.dtype, device=other.device)
    if isinstance(input, _Tensor) and not isinstance(other, _Tensor):
        from .primitives import tensor
        other = tensor(other, dtype=input.dtype, device=input.device)
    if hasattr(input, "igamma") and isinstance(other, _Tensor):
        return input.igamma(other)
    a = _to_np(input).astype(np.float64)
    x = _to_np(other).astype(np.float64)
    a, x = np.broadcast_arrays(a, x)
    
    flat_a = a.reshape(-1)
    flat_x = x.reshape(-1)
    flat_out = np.zeros_like(flat_a)
    for i in range(len(flat_a)):
        ai, xi = flat_a[i], flat_x[i]
        if xi <= 0.0:
            flat_out[i] = 0.0
        elif ai <= 0.0:
            flat_out[i] = float("nan")
        elif math.isinf(xi):
            flat_out[i] = 1.0
        else:
            gln = math.lgamma(ai)
            if xi < ai + 1.0:
                ap = ai
                sum_val = 1.0 / ai
                del_val = sum_val
                for _ in range(1, 300):
                    ap += 1.0
                    del_val *= xi / ap
                    sum_val += del_val
                    if abs(del_val) < abs(sum_val) * 1e-15:
                        break
                v = sum_val * math.exp(-xi + ai * math.log(xi) - gln)
                flat_out[i] = min(max(v, 0.0), 1.0)
            else:
                tiny = 1e-30
                b = xi + 1.0 - ai
                c = 1.0 / tiny
                d = 1.0 / b
                h = d
                for k in range(1, 300):
                    an = -k * (k - ai)
                    b += 2.0
                    d = an * d + b
                    if abs(d) < tiny:
                        d = tiny
                    c = b + an / c
                    if abs(c) < tiny:
                        c = tiny
                    d = 1.0 / d
                    del_h = d * c
                    h *= del_h
                    if abs(del_h - 1.0) < 1e-15:
                        break
                q = math.exp(-xi + ai * math.log(xi) - gln) * h
                flat_out[i] = min(max(1.0 - q, 0.0), 1.0)
    return _wrap(flat_out.reshape(a.shape), input if isinstance(input, _Tensor) else (other if isinstance(other, _Tensor) else None))


def gammaincc(input: Any, other: Union[_Tensor, float]) -> _Tensor:
    """Computes the regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x)."""
    if not isinstance(input, _Tensor) and isinstance(other, _Tensor):
        from .primitives import tensor
        input = tensor(input, dtype=other.dtype, device=other.device)
    if isinstance(input, _Tensor) and not isinstance(other, _Tensor):
        from .primitives import tensor
        other = tensor(other, dtype=input.dtype, device=input.device)
    if hasattr(input, "igammac") and isinstance(other, _Tensor):
        return input.igammac(other)
    return _wrap(1.0 - _to_np(gammainc(input, other)), input if isinstance(input, _Tensor) else (other if isinstance(other, _Tensor) else None))


def bessel_y0(input: _Tensor) -> _Tensor:
    """Computes the Bessel function of the second kind of order 0: Y_0(x)."""
    if hasattr(input, "bessel_y0"):
        return input.bessel_y0()
    x = _to_np(input).astype(np.float64)
    y = np.full_like(x, np.nan, dtype=np.float64)
    pos = x > 0.0
    if not np.any(pos):
        return _wrap(y, input)
    xp = x[pos]
    res = np.zeros_like(xp)
    near = xp < 8.0
    if np.any(near):
        xn = xp[near]
        z = (xn / 2.0)**2
        j0 = 1.0 - z + (z**2)/4.0 - (z**3)/36.0 + (z**4)/576.0
        term = -z
        sum_terms = term * 1.0
        h_k = 1.0
        fact_sq = 1.0
        for k in range(2, 25):
            h_k += 1.0 / k
            fact_sq *= (k * k)
            term = ((-1.0)**k) * (z**k) / fact_sq
            sum_terms += term * h_k
            if np.max(np.abs(term * h_k)) < 1e-15:
                break
        res[near] = (2.0 / np.pi) * ((np.log(xn / 2.0) + 0.5772156649015329) * j0 - sum_terms)
    far = ~near
    if np.any(far):
        xf = xp[far]
        res[far] = np.sqrt(2.0 / (np.pi * xf)) * (
            np.sin(xf - np.pi / 4.0) + (1.0 / (8.0 * xf)) * np.cos(xf - np.pi / 4.0)
        )
    y[pos] = res
    return _wrap(y, input)


def bessel_y1(input: _Tensor) -> _Tensor:
    """Computes the Bessel function of the second kind of order 1: Y_1(x)."""
    if hasattr(input, "bessel_y1"):
        return input.bessel_y1()
    x = _to_np(input).astype(np.float64)
    y = np.full_like(x, np.nan, dtype=np.float64)
    pos = x > 0.0
    if not np.any(pos):
        return _wrap(y, input)
    xp = x[pos]
    res = np.zeros_like(xp)
    near = xp < 8.0
    if np.any(near):
        xn = xp[near]
        z = (xn / 2.0)**2
        # J1(x) series
        j1 = np.zeros_like(xn)
        fact_k = 1.0
        fact_kp1 = 1.0
        for k in range(20):
            if k > 0:
                fact_k *= k
                fact_kp1 *= (k + 1)
            j1 += (((-1.0)**k) * (z**k) / (fact_k * fact_kp1))
        j1 *= (xn / 2.0)

        sum_part = np.zeros_like(xn)
        hk = 0.0
        fact_k = 1.0
        fact_kp1 = 1.0
        for k in range(20):
            if k > 0:
                hk += 1.0 / k
                fact_k *= k
                fact_kp1 *= (k + 1)
            hk1 = hk + 1.0 / (k + 1)
            term = (((-1.0)**k) * (z**k) / (fact_k * fact_kp1)) * (hk + hk1)
            sum_part += term
        res[near] = -2.0 / (np.pi * xn) + (2.0 / np.pi) * (np.log(xn / 2.0) + 0.5772156649015329) * j1 - (xn / (2.0 * np.pi)) * sum_part
    far = ~near
    if np.any(far):
        xf = xp[far]
        res[far] = -np.sqrt(2.0 / (np.pi * xf)) * np.cos(xf - np.pi / 4.0)
    y[pos] = res
    return _wrap(y, input)


def modified_bessel_k0(input: _Tensor) -> _Tensor:
    """Computes the modified Bessel function of the second kind of order 0: K_0(x)."""
    if hasattr(input, "modified_bessel_k0"):
        return input.modified_bessel_k0()
    x = _to_np(input).astype(np.float64)
    k0 = np.full_like(x, np.nan, dtype=np.float64)
    pos = x > 0.0
    if not np.any(pos):
        return _wrap(k0, input)
    xp = x[pos]
    res = np.zeros_like(xp)
    near = xp <= 2.0
    if np.any(near):
        xn = xp[near]
        z = (xn / 2.0)**2
        i0 = 1.0 + z + (z**2)/4.0 + (z**3)/36.0 + (z**4)/576.0
        term = z
        sum_terms = term * 1.0
        h_k = 1.0
        fact_sq = 1.0
        for k in range(2, 25):
            h_k += 1.0 / k
            fact_sq *= (k * k)
            term = (z**k) / fact_sq
            sum_terms += term * h_k
            if np.max(np.abs(term * h_k)) < 1e-15:
                break
        res[near] = -(np.log(xn / 2.0) + 0.5772156649015329) * i0 + sum_terms
    far = ~near
    if np.any(far):
        xf = xp[far]
        res[far] = np.sqrt(np.pi / (2.0 * xf)) * np.exp(-xf) * (
            1.0 - 1.0 / (8.0 * xf) + 9.0 / (128.0 * xf**2)
        )
    k0[pos] = res
    return _wrap(k0, input)


def modified_bessel_k1(input: _Tensor) -> _Tensor:
    """Computes the modified Bessel function of the second kind of order 1: K_1(x)."""
    if hasattr(input, "modified_bessel_k1"):
        return input.modified_bessel_k1()
    x = _to_np(input).astype(np.float64)
    k1 = np.full_like(x, np.nan, dtype=np.float64)
    pos = x > 0.0
    if not np.any(pos):
        return _wrap(k1, input)
    xp = x[pos]
    res = np.zeros_like(xp)
    near = xp <= 2.0
    if np.any(near):
        xn = xp[near]
        z = (xn / 2.0)**2
        # I1(x) series
        i1 = np.zeros_like(xn)
        fact_k = 1.0
        fact_kp1 = 1.0
        for k in range(20):
            if k > 0:
                fact_k *= k
                fact_kp1 *= (k + 1)
            i1 += ((z**k) / (fact_k * fact_kp1))
        i1 *= (xn / 2.0)

        sum_part = np.zeros_like(xn)
        hk = 0.0
        fact_k = 1.0
        fact_kp1 = 1.0
        for k in range(20):
            if k > 0:
                hk += 1.0 / k
                fact_k *= k
                fact_kp1 *= (k + 1)
            hk1 = hk + 1.0 / (k + 1)
            term = ((z**k) / (fact_k * fact_kp1)) * (hk + hk1)
            sum_part += term
        res[near] = 1.0 / xn + (np.log(xn / 2.0) + 0.5772156649015329) * i1 - (xn / 4.0) * sum_part
    far = ~near
    if np.any(far):
        xf = xp[far]
        res[far] = np.sqrt(np.pi / (2.0 * xf)) * np.exp(-xf) * (
            1.0 + 3.0 / (8.0 * xf) - 15.0 / (128.0 * xf**2)
        )
    k1[pos] = res
    return _wrap(k1, input)


def scaled_modified_bessel_k0(input: _Tensor) -> _Tensor:
    """Computes exp(x) * K_0(x)."""
    if hasattr(input, "scaled_modified_bessel_k0"):
        return input.scaled_modified_bessel_k0()
    x = _to_np(input).astype(np.float64)
    k0 = _to_np(modified_bessel_k0(input)).astype(np.float64)
    return _wrap(np.exp(x) * k0, input)


def scaled_modified_bessel_k1(input: _Tensor) -> _Tensor:
    """Computes exp(x) * K_1(x)."""
    if hasattr(input, "scaled_modified_bessel_k1"):
        return input.scaled_modified_bessel_k1()
    x = _to_np(input).astype(np.float64)
    k1 = _to_np(modified_bessel_k1(input)).astype(np.float64)
    return _wrap(np.exp(x) * k1, input)


def airy_ai(input: _Tensor) -> _Tensor:
    """Computes the Airy function Ai(x)."""
    x = _to_np(input).astype(np.float64)
    c1 = 0.3550280538878172
    c2 = 0.2588194037928068
    res = np.zeros_like(x)
    near = np.abs(x) <= 4.0
    if np.any(near):
        xn = x[near]
        f = 1.0 + (xn**3)/6.0 + (xn**6)/180.0
        g = xn + (xn**4)/12.0 + (xn**7)/504.0
        res[near] = c1 * f - c2 * g
    far_pos = x > 4.0
    if np.any(far_pos):
        xf = x[far_pos]
        zeta = (2.0 / 3.0) * (xf ** 1.5)
        res[far_pos] = (1.0 / (2.0 * np.sqrt(np.pi) * (xf ** 0.25))) * np.exp(-zeta)
    far_neg = x < -4.0
    if np.any(far_neg):
        xf = -x[far_neg]
        zeta = (2.0 / 3.0) * (xf ** 1.5)
        res[far_neg] = (1.0 / (np.sqrt(np.pi) * (xf ** 0.25))) * np.sin(zeta + np.pi / 4.0)
    return _wrap(res, input)


def spherical_bessel_j0(input: _Tensor) -> _Tensor:
    """Computes the spherical Bessel function of the first kind of order 0: j_0(x) = sin(x) / x."""
    if hasattr(input, "spherical_bessel_j0"):
        return input.spherical_bessel_j0()
    x = _to_np(input).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(np.abs(x) < 1e-8, 1.0 - x * x / 6.0, np.sin(x) / x)
    return _wrap(res, input)


def ellipk(input: _Tensor) -> _Tensor:
    """Computes the complete elliptic integral of the first kind K(m)."""
    return input.ellipk()


def ellipe(input: _Tensor) -> _Tensor:
    """Computes the complete elliptic integral of the second kind E(m)."""
    return input.ellipe()


def expint_e1(input: _Tensor) -> _Tensor:
    """Computes the exponential integral E_1(x)."""
    return input.expint_e1()


def expint_ei(input: _Tensor) -> _Tensor:
    """Computes the exponential integral Ei(x)."""
    return input.expint_ei()


def dawsn(input: _Tensor) -> _Tensor:
    """Computes Dawson's integral D_+(x)."""
    return input.dawsn()


def spence(input: _Tensor) -> _Tensor:
    """Computes Spence's function (dilogarithm)."""
    return input.spence()


def fresnel(input: _Tensor) -> tuple:
    """Computes Fresnel sine and cosine integrals (S(x), C(x))."""
    return input.fresnel()


def sici(input: _Tensor) -> tuple:
    """Computes sine and cosine integrals (Si(x), Ci(x))."""
    return input.sici()


def gegenbauer(input: _Tensor, n: int, alpha: float) -> _Tensor:
    """Computes the Gegenbauer polynomial C_n^{(alpha)}(x)."""
    return input.gegenbauer(n, alpha)


def jacobi(input: _Tensor, n: int, alpha: float, beta: float) -> _Tensor:
    """Computes the Jacobi polynomial P_n^{(alpha, beta)}(x)."""
    return input.jacobi(n, alpha, beta)


__all__ = [
    "xlogy",
    "xlog1py",
    "log1pexp",
    "gammainc",
    "gammaincc",
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "laguerre_polynomial_l",
    "bessel_y0",
    "bessel_y1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1",
    "airy_ai",
    "erf",
    "erfc",
    "erfcx",
    "erfinv",
    "exp2",
    "expm1",
    "expit",
    "logit",
    "gammaln",
    "loggamma",
    "digamma",
    "psi",
    "polygamma",
    "multigammaln",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "ndtr",
    "ndtri",
    "log_ndtr",
    "bessel_j0",
    "bessel_j1",
    "spherical_bessel_j0",
    "ellipk",
    "ellipe",
    "expint_e1",
    "expint_ei",
    "dawsn",
    "spence",
    "fresnel",
    "sici",
    "gegenbauer",
    "jacobi",
    "lambert_w",
    "hermite_polynomial_h",
    "legendre_polynomial_p",
    "log1p",
    "sinc",
    "zeta",
    "softmax",
    "log_softmax",
    "entr",
    "rel_entr",
    "round",
]


# ---------------------------------------------------------------------------------------------------------------------
# Orthogonal polynomials (degree n an int or a tensor, broadcast against x), all evaluated by one Rust kernel
# (numeric_kernels.rs, three-term recurrences); shifted Chebyshev polynomials; modified Bessel I0 / I1
# ---------------------------------------------------------------------------------------------------------------------

def _poly(kind: str, input: Any, n: Any) -> _Tensor:
    from .oxideon import _orthogonal_polynomial
    x = input if isinstance(input, _Tensor) else _from_numpy(_contig(np.asarray(input, dtype=np.float32)))
    nt = n if isinstance(n, _Tensor) else _from_numpy(_contig(np.asarray(n, dtype=np.float64)))
    shape = tuple(int(v) for v in np.broadcast_shapes(tuple(x.shape), tuple(nt.shape)))
    if tuple(x.shape) != shape:
        x = x.expand(list(shape)).contiguous()
    if tuple(nt.shape) != shape:
        nt = nt.expand(list(shape)).contiguous()
    return _orthogonal_polynomial(kind, x, nt.double() if nt.dtype != "f64" else nt)


def chebyshev_polynomial_t(input: Any, n: Any) -> _Tensor:
    """The Chebyshev polynomial of the first kind T_n(x)."""
    return _poly("chebyshev_t", input, n)


def chebyshev_polynomial_u(input: Any, n: Any) -> _Tensor:
    """The Chebyshev polynomial of the second kind U_n(x)."""
    return _poly("chebyshev_u", input, n)


def chebyshev_polynomial_v(input: Any, n: Any) -> _Tensor:
    """The Chebyshev polynomial of the third kind V_n(x)."""
    return _poly("chebyshev_v", input, n)


def chebyshev_polynomial_w(input: Any, n: Any) -> _Tensor:
    """The Chebyshev polynomial of the fourth kind W_n(x)."""
    return _poly("chebyshev_w", input, n)


def hermite_polynomial_h(input: Any, n: Any) -> _Tensor:
    """The physicists' Hermite polynomial H_n(x)."""
    return _poly("hermite_h", input, n)


def hermite_polynomial_he(input: Any, n: Any) -> _Tensor:
    """The probabilists' Hermite polynomial He_n(x) (He_0 = 1, He_1 = x, He_{k+1} = x He_k - k He_{k-1}); 0 for n < 0."""
    return _poly("hermite_he", input, n)


def laguerre_polynomial_l(input: Any, n: Any) -> _Tensor:
    """The Laguerre polynomial L_n(x)."""
    return _poly("laguerre_l", input, n)


def legendre_polynomial_p(input: Any, n: Any) -> _Tensor:
    """The Legendre polynomial P_n(x)."""
    return _poly("legendre_p", input, n)


def _shifted(kind: str):
    def poly(input: Any, n: Any) -> _Tensor:
        x = input if isinstance(input, _Tensor) else _from_numpy(_contig(np.asarray(input, dtype=np.float32)))
        return _poly(kind, x * 2.0 - 1.0, n)
    poly.__name__ = "shifted_" + kind.replace("chebyshev_", "chebyshev_polynomial_")
    poly.__doc__ = f"The shifted Chebyshev polynomial on [0, 1]: `{kind}(2 x - 1, n)`."
    return poly


shifted_chebyshev_polynomial_t = _shifted("chebyshev_t")
shifted_chebyshev_polynomial_u = _shifted("chebyshev_u")
shifted_chebyshev_polynomial_v = _shifted("chebyshev_v")
shifted_chebyshev_polynomial_w = _shifted("chebyshev_w")


def modified_bessel_i0(input: Any) -> _Tensor:
    """The modified Bessel function of the first kind of order 0 (the same function as `i0`)."""
    return i0(input)


def modified_bessel_i1(input: Any) -> _Tensor:
    """The modified Bessel function of the first kind of order 1 (the same function as `i1`)."""
    return i1(input)


__all__ += [
    "hermite_polynomial_he", "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w", "modified_bessel_i0", "modified_bessel_i1",
]
