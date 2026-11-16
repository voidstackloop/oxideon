"""Tests for oxideon.numerical: quadrature, interpolation, optimization, constants.

Analytic identities are the primary oracles; scipy cross-checks are
supplementary (skipped when scipy is unavailable).
"""

import math

import numpy as np
import pytest

import oxideon as ox
from oxideon.numerical import (
    Constant,
    CubicSpline,
    Interp1D,
    ScalarOptResult,
    curve_fit,
    dblquad,
    fixed_quad,
    gauss_legendre,
    minimize_scalar,
    physical_constants,
    precision,
    quad,
    unit,
    value,
)

try:
    import scipy.integrate as _sci_int
    import scipy.interpolate as _sci_interp
    import scipy.optimize as _sci_opt
    from scipy.constants import physical_constants as _sci_pc

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")


# --- quad -----------------------------------------------------------------


def test_quad_analytic_cases():
    v, e = quad(lambda x: x * x, 0.0, 1.0)
    assert v == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert e < 1e-9
    v, _ = quad(math.sin, 0.0, math.pi)
    assert v == pytest.approx(2.0, rel=1e-9)
    v, _ = quad(lambda x: math.exp(-x), 0.0, 10.0)
    assert v == pytest.approx(1.0 - math.exp(-10.0), rel=1e-8)


def test_quad_sign_and_degenerate_limits():
    v, e = quad(lambda x: x, 1.0, 0.0)
    assert v == pytest.approx(-0.5, rel=1e-12)
    assert e >= 0.0
    assert quad(math.sin, 2.0, 2.0) == (0.0, 0.0)


def test_quad_limit_warning():
    f = lambda x: 1.0 / (1.0 + 1000.0 * (x - 0.5) ** 2)  # noqa: E731
    with pytest.warns(RuntimeWarning, match="limit"):
        v, e = quad(f, 0.0, 1.0, epsabs=1e-14, epsrel=1e-14, limit=3)
    assert math.isfinite(v) and e >= 0.0


def test_quad_validation():
    with pytest.raises(ValueError, match="finite"):
        quad(math.sin, 0.0, float("inf"))
    with pytest.raises(ValueError, match="scalar"):
        quad(lambda x: np.array([x, x]), 0.0, 1.0)
    with pytest.raises(ValueError, match="failed"):
        quad(lambda x: 1.0 / x, -1.0, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        quad(lambda x: float("inf"), 0.0, 1.0)
    with pytest.raises(ValueError, match="epsabs"):
        quad(math.sin, 0.0, 1.0, epsabs=-1.0)
    with pytest.raises(ValueError, match="limit"):
        quad(math.sin, 0.0, 1.0, limit=0)


@needs_scipy
def test_quad_matches_scipy():
    for f, a, b in [(math.sin, 0.0, 2.0), (lambda x: x**3 - 2 * x, -1.0, 3.0)]:
        v, _ = quad(f, a, b)
        ref, _ = _sci_int.quad(f, a, b)
        assert v == pytest.approx(ref, rel=1e-8)


# --- gauss_legendre / fixed_quad / dblquad --------------------------------


def test_gauss_legendre_properties():
    for n in (1, 2, 5, 10):
        x, w = gauss_legendre(n)
        assert x.shape == (n,) and w.shape == (n,)
        assert bool(np.all(x > -1.0)) and bool(np.all(x < 1.0))
        assert float(w.sum()) == pytest.approx(2.0, rel=1e-12)
        assert np.allclose(x, -x[::-1], atol=1e-12)
    with pytest.raises(ValueError, match="positive integer"):
        gauss_legendre(0)
    with pytest.raises(ValueError, match="positive integer"):
        gauss_legendre(True)


@needs_scipy
def test_gauss_legendre_matches_scipy():
    from scipy.special import roots_legendre

    for n in (3, 7):
        x, w = gauss_legendre(n)
        rx, rw = roots_legendre(n)
        np.testing.assert_allclose(x, rx, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(w, rw, rtol=1e-12, atol=1e-14)


def test_fixed_quad_exactness():
    poly = lambda x: x**9 - 3 * x**4 + x - 1.0  # noqa: E731
    assert fixed_quad(poly, -2.0, 1.0, n=5) == pytest.approx(-126.6, rel=1e-9)
    assert quad(poly, -2.0, 1.0)[0] == pytest.approx(-126.6, rel=1e-8)
    assert fixed_quad(math.sin, 1.0, 1.0) == 0.0


def test_dblquad_analytic():
    v, _ = dblquad(lambda y, x: y, 0.0, 1.0, lambda x: 0.0, lambda x: x)
    assert v == pytest.approx(1.0 / 6.0, rel=1e-8)
    v, _ = dblquad(lambda y, x: x * y, 0.0, 2.0, lambda x: 0.0, lambda x: 1.0)
    assert v == pytest.approx(1.0, rel=1e-8)
    with pytest.raises(ValueError, match="bound"):
        dblquad(lambda y, x: 1.0, 0.0, 1.0, lambda x: 0.0, lambda x: float("nan"))


@needs_scipy
def test_dblquad_matches_scipy():
    v, _ = dblquad(lambda y, x: x * y, 0.0, 2.0, lambda x: 0.0, lambda x: 1.0)
    ref, _ = _sci_int.dblquad(lambda y, x: x * y, 0.0, 2.0, lambda x: 0.0, lambda x: 1.0)
    assert v == pytest.approx(ref, rel=1e-8)


# --- Interp1D ---------------------------------------------------------------


def test_interp1d_kinds():
    x = np.array([0.0, 1.0, 3.0])
    y = np.array([0.0, 2.0, 2.0])
    assert Interp1D(x, y)(np.array([0.5]))[0] == pytest.approx(1.0)
    assert Interp1D(x, y, kind="nearest")(np.array([2.4]))[0] == pytest.approx(2.0)
    assert Interp1D(x, y, kind="previous")(np.array([2.9]))[0] == pytest.approx(2.0)
    assert Interp1D(x, y, kind="next")(np.array([0.1]))[0] == pytest.approx(2.0)
    assert Interp1D(x, y, kind="slinear")(np.array([0.5]))[0] == pytest.approx(1.0)
    assert Interp1D(x, y, kind="zero")(np.array([2.9]))[0] == pytest.approx(2.0)


def test_interp1d_nodes_shapes_bounds():
    x = np.array([0.0, 1.0, 2.0])
    f = Interp1D(x, np.array([1.0, 3.0, 2.0]))
    np.testing.assert_allclose(f(x), [1.0, 3.0, 2.0])
    g = Interp1D(x, np.column_stack([x, 2 * x]))
    np.testing.assert_allclose(g(np.array([0.5])), [[0.5, 1.0]])
    assert float(f(1.5)) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="outside domain"):
        f(np.array([-0.1]))
    h = Interp1D(x, np.array([1.0, 3.0, 2.0]), bounds_error=False, fill_value=-1.0)
    assert float(h(-5.0)) == -1.0
    t = Interp1D(x, np.array([1.0, 3.0, 2.0]))(ox.tensor(np.array([0.5], dtype=np.float32)))
    assert isinstance(t, ox.Tensor)
    np.testing.assert_allclose(t.to_numpy().ravel(), [2.0], rtol=1e-6)


def test_interp1d_validation():
    with pytest.raises(ValueError, match="kind"):
        Interp1D([0.0, 1.0], [0.0, 1.0], kind="cubic")
    with pytest.raises(ValueError, match="strictly increasing"):
        Interp1D([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        Interp1D([0.0, 1.0], [0.0, 1.0, 2.0])


# --- CubicSpline ------------------------------------------------------------


def test_cubic_spline_exactness_on_bc_compatible_data():
    # Natural BCs (M=0 at ends) reproduce lines; clamped zero-slope BCs
    # reproduce cubics with zero end slopes. A general cubic satisfies
    # neither, so exactness must be tested on compatible data.
    xl = np.array([0.0, 0.7, 1.9, 3.1, 4.0])
    sl = CubicSpline(xl, 2 * xl + 1.0, bc_type="natural")
    grid = np.linspace(0.0, 4.0, 25)
    np.testing.assert_allclose(sl(grid), 2 * grid + 1.0, rtol=1e-12, atol=1e-12)
    xc = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yc = xc**3 / 3.0 - 2 * xc**2 + 5.0  # y'(0) = y'(4) = 0
    sc = CubicSpline(xc, yc, bc_type="clamped")
    np.testing.assert_allclose(sc(grid), grid**3 / 3.0 - 2 * grid**2 + 5.0, rtol=1e-10, atol=1e-10)


def test_cubic_spline_derivative_and_integral():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    s = CubicSpline(x, x**3 / 3.0 - 2 * x**2 + 5.0, bc_type="clamped")
    grid = np.linspace(0.0, 4.0, 9)
    np.testing.assert_allclose(s.derivative(1)(grid), grid**2 - 4 * grid, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(s.derivative(2)(grid), 2 * grid - 4, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(s.derivative(3)(grid[1:-1]), 2.0, rtol=1e-9, atol=1e-9)
    assert s.integrate(0.0, 2.0) == pytest.approx(6.0, rel=1e-9)
    assert s.integrate(2.0, 0.0) == pytest.approx(-6.0, rel=1e-9)
    sc = CubicSpline(x, (x - 2.0) ** 2, bc_type="clamped")
    assert float(sc.derivative(1)(np.array([0.0]))[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(sc.derivative(1)(np.array([4.0]))[0]) == pytest.approx(0.0, abs=1e-9)


def test_cubic_spline_validation_and_domain():
    s = CubicSpline([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], extrapolate=False)
    with pytest.raises(ValueError, match="outside domain"):
        s(np.array([5.0]))
    with pytest.raises(ValueError, match="bc_type"):
        CubicSpline([0.0, 1.0], [0.0, 1.0], bc_type="periodic")
    with pytest.raises(ValueError, match="at least 2"):
        CubicSpline([0.0], [1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        CubicSpline([0.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="nu"):
        s.derivative(4)
    with pytest.raises(ValueError, match="outside domain"):
        s.integrate(-1.0, 1.0)


@needs_scipy
def test_cubic_spline_matches_scipy():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 5.0, 8))
    y = np.sin(x) + 0.1 * np.cos(3 * x)
    grid = np.linspace(0.0, 5.0, 30)
    for bc in ("natural", "clamped"):
        mine = CubicSpline(x, y, bc_type=bc)
        ref = _sci_interp.CubicSpline(x, y, bc_type=bc)
        np.testing.assert_allclose(mine(grid), ref(grid), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(mine.derivative(1)(grid), ref.derivative(1)(grid), rtol=1e-8, atol=1e-8)


# --- minimize_scalar --------------------------------------------------------


@pytest.mark.parametrize("method", ["brent", "golden"])
def test_minimize_scalar_methods(method):
    r = minimize_scalar(lambda x: (x - 2.0) ** 2 + 1.0, -5.0, 5.0, method=method)
    assert isinstance(r, ScalarOptResult)
    assert r.x == pytest.approx(2.0, abs=1e-6)
    assert r.fun == pytest.approx(1.0, abs=1e-8)
    assert r.success and r.nfev > 0
    r2 = minimize_scalar(lambda x: (x - 1.0) ** 4 + (x - 1.0) ** 2, -2.0, 4.0, method=method)
    assert r2.x == pytest.approx(1.0, abs=1e-5)


def test_minimize_scalar_nonconvergence_and_validation():
    r = minimize_scalar(lambda x: (x - 2.0) ** 2, -5.0, 5.0, maxiter=1)
    assert r.success is False
    with pytest.raises(ValueError, match="a < b"):
        minimize_scalar(lambda x: x, 1.0, 1.0)
    with pytest.raises(ValueError, match="method"):
        minimize_scalar(lambda x: x, 0.0, 1.0, method="nelder")
    with pytest.raises(ValueError, match="tol"):
        minimize_scalar(lambda x: x, 0.0, 1.0, tol=0.0)


@needs_scipy
def test_minimize_scalar_matches_scipy():
    f = lambda x: (x - 0.3) ** 2 + math.sin(3 * x)  # noqa: E731
    mine = minimize_scalar(f, -1.0, 1.5)
    ref = _sci_opt.minimize_scalar(f, bracket=(0.0, 0.5), method="brent")
    assert mine.x == pytest.approx(ref.x, abs=1e-6)


# --- curve_fit --------------------------------------------------------------


def test_curve_fit_exponential():
    rng = np.random.default_rng(0)
    xd = np.linspace(0.0, 3.0, 25)
    yd = 2.5 * np.exp(-1.3 * xd) + 0.5 + rng.normal(0.0, 1e-4, xd.shape[0])
    popt, pcov = curve_fit(lambda x, a, b, c: a * np.exp(-b * x) + c, xd, yd, p0=[1.0, 1.0, 0.0])
    np.testing.assert_allclose(popt, [2.5, 1.3, 0.5], rtol=1e-2, atol=1e-2)
    assert pcov.shape == (3, 3)
    assert bool(np.all(np.linalg.eigvalsh(pcov) > -1e-12))


def test_curve_fit_validation_and_nonconvergence():
    xd = np.linspace(0.0, 1.0, 6)
    with pytest.raises(ValueError, match="underdetermined"):
        curve_fit(lambda x, a, b, c: a * x + b + c, xd, xd, p0=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="sigma"):
        curve_fit(lambda x, a: a * x, xd, xd, p0=[1.0], sigma=np.zeros(6))
    with pytest.raises(ValueError, match="matching lengths"):
        curve_fit(lambda x, a: a * x, xd, xd[:-1], p0=[1.0])
    def _blowup(x, a, b):
        with np.errstate(over="ignore", invalid="ignore"):
            return a * np.exp(-b * x)

    with pytest.raises(RuntimeError, match="maxiter"):
        curve_fit(_blowup, xd, np.exp(-xd), p0=[50.0, 50.0], maxiter=1)


@needs_scipy
def test_curve_fit_matches_scipy():
    rng = np.random.default_rng(1)
    xd = np.linspace(0.0, 2.0, 15)
    yd = 1.5 * np.exp(-0.7 * xd) + rng.normal(0.0, 0.01, xd.shape[0])
    f = lambda x, a, b: a * np.exp(-b * x)  # noqa: E731
    popt, _ = curve_fit(f, xd, yd, p0=[1.0, 1.0])
    ref, _ = _sci_opt.curve_fit(f, xd, yd, p0=[1.0, 1.0])
    np.testing.assert_allclose(popt, ref, rtol=1e-4)


# --- physical constants -----------------------------------------------------


def test_constants_definitional_values():
    assert value("speed of light in vacuum") == 299792458.0
    assert value("Planck constant") == 6.62607015e-34
    assert value("elementary charge") == 1.602176634e-19
    assert value("Boltzmann constant") == 1.380649e-23
    assert value("Avogadro constant") == 6.02214076e23
    assert precision("speed of light in vacuum") == 0.0
    assert unit("electron volt") == "J"
    assert isinstance(physical_constants["Bohr radius"], Constant)
    with pytest.raises(KeyError):
        value("bogus constant")


@needs_scipy
def test_constants_match_scipy():
    for key, const in physical_constants.items():
        ref_v, ref_u, ref_un = _sci_pc[key]
        assert const.value == pytest.approx(ref_v, rel=1e-12)
        assert const.unit == ref_u
        assert const.uncertainty == pytest.approx(ref_un, rel=1e-9, abs=1e-30)
