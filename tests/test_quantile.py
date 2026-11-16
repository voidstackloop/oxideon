"""Z38: quantile/nanquantile vs torch ground truth.

The native engine kernels (sort + gather + in-width lerp, all five
interpolations, differentiable) serve both the Tensor methods and, via pure
delegation in `primitives.py`, the module functions. Values, shapes, edges,
and gradients are pinned against torch.

Deliberate divergences from torch (all pinned below):
- `q` given as a Python list/tuple is accepted (torch: `TypeError`); it is
  converted with the input's dtype so the dtype-match rule still holds.
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy

X = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], dtype=np.float32)
METHODS = ("linear", "lower", "higher", "nearest", "midpoint")


def test_quantile_interpolations_match_torch():
    for m in METHODS:
        for q in (0.0, 0.3, 0.5, 1.0):
            o = ox.quantile(T(X), q, interpolation=m).to_numpy()
            t = torch.quantile(th(X), q, interpolation=m).numpy()
            assert o.shape == t.shape == ()
            np.testing.assert_allclose(o, t, rtol=1e-6)
    xf = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    for m in METHODS:
        o = ox.quantile(T(xf), 0.5, interpolation=m).to_numpy()
        t = torch.quantile(th(xf), 0.5, interpolation=m).numpy()
        np.testing.assert_allclose(o, t, rtol=1e-12)


def test_quantile_q_tensor_shapes_and_values():
    q = T(np.array([0.25, 0.75], dtype=np.float32))
    tq = th(np.array([0.25, 0.75], dtype=np.float32))
    o = ox.quantile(T(X), q).to_numpy()
    t = torch.quantile(th(X), tq).numpy()
    assert o.shape == t.shape == (2,)
    np.testing.assert_allclose(o, t, rtol=1e-6)
    m = T(X).quantile(q, interpolation="midpoint").to_numpy()
    tm = torch.quantile(th(X), tq, interpolation="midpoint").numpy()
    np.testing.assert_allclose(m, tm, rtol=1e-6)
    # from_numpy folds 0-d to [1], so a genuine 0-d q comes from a scalar result.
    q0 = ox.quantile(T(np.array([0.0, 0.5, 1.0], dtype=np.float32)), 0.5)
    assert q0.to_numpy().shape == ()
    z = ox.quantile(T(X), q0).to_numpy()
    assert z.shape == ()
    np.testing.assert_allclose(z, torch.quantile(th(X), 0.5).numpy(), rtol=1e-6)
    # Python int q coerces like torch.
    assert ox.quantile(T(X), 1).to_numpy() == 9.0
    # Empty q reads empty.
    e = ox.quantile(T(X), T(np.array([], dtype=np.float32))).to_numpy()
    assert e.shape == (0,)


def test_quantile_dim_and_keepdim():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    q = T(np.array([0.25, 0.75], dtype=np.float32))
    tq = th(np.array([0.25, 0.75], dtype=np.float32))
    for dim in (None, 0, 1):
        for keepdim in (False, True):
            o = ox.quantile(T(a), q, dim=dim, keepdim=keepdim).to_numpy()
            t = torch.quantile(th(a), tq, dim=dim, keepdim=keepdim).numpy()
            assert o.shape == t.shape
            np.testing.assert_allclose(o, t, rtol=1e-6)
    o = ox.quantile(T(a), 0.5, dim=0).to_numpy()
    assert o.shape == (4,)
    np.testing.assert_allclose(o, torch.quantile(th(a), 0.5, dim=0).numpy(), rtol=1e-6)


def test_quantile_rejects_non_float_input():
    with pytest.raises(RuntimeError):
        ox.quantile(T(np.array([1, 2, 3, 4])), 0.5)
    with pytest.raises(RuntimeError):
        torch.quantile(th(np.array([1, 2, 3, 4])), 0.5)
    with pytest.raises(RuntimeError):
        ox.quantile(T(np.array([1, 2, 3, 4], dtype=np.float16)), 0.5)


def test_quantile_validates_q_and_interpolation():
    for bad_q in (1.5, -0.1, float("nan")):
        with pytest.raises(RuntimeError):
            ox.quantile(T(X), bad_q)
    with pytest.raises(RuntimeError):
        ox.quantile(T(X), 0.5, interpolation="foo")
    with pytest.raises(RuntimeError):
        torch.quantile(th(X), 0.5, interpolation="foo")
    # q dtype must match the input dtype.
    with pytest.raises(RuntimeError):
        ox.quantile(T(X), T(np.array([0.5], dtype=np.float64)))
    # q must be a scalar or 1-D.
    with pytest.raises(RuntimeError):
        ox.quantile(T(X), T(np.array([[0.5]], dtype=np.float32)))


def test_quantile_accepts_list_q():
    o = ox.quantile(T(X), [0.25, 0.75]).to_numpy()
    t = torch.quantile(th(X), th(np.array([0.25, 0.75], dtype=np.float32))).numpy()
    assert o.shape == (2,)
    np.testing.assert_allclose(o, t, rtol=1e-6)
    with pytest.raises(TypeError):
        torch.quantile(th(X), [0.25, 0.75])


def test_nanquantile_skips_nan():
    xn = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    for m in METHODS:
        o = ox.nanquantile(T(xn), 0.5, interpolation=m).to_numpy()
        t = torch.nanquantile(th(xn), 0.5, interpolation=m).numpy()
        np.testing.assert_allclose(o, t, rtol=1e-6)
    xa = np.full(4, np.nan, dtype=np.float32)
    assert np.isnan(ox.nanquantile(T(xa), 0.5).to_numpy())
    x2 = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    for dim in (0, 1):
        o = ox.nanquantile(T(x2), 0.25, dim=dim).to_numpy()
        t = torch.nanquantile(th(x2), 0.25, dim=dim).numpy()
        np.testing.assert_allclose(o, t, equal_nan=True, rtol=1e-6)


def test_quantile_nan_propagates_and_grad_hits_first_nan():
    xn = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    for m in METHODS:
        assert np.isnan(ox.quantile(T(xn), 0.5, interpolation=m).to_numpy())
    o = T(xn).requires_grad_(True)
    t = th(xn).requires_grad_(True)
    ox.quantile(o, 0.5).sum().backward()
    torch.quantile(t, 0.5).sum().backward()
    np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy())


def test_quantile_grads_match_torch():
    for m in ("linear", "midpoint", "lower", "higher", "nearest"):
        o = T(X).requires_grad_(True)
        t = th(X).requires_grad_(True)
        ox.quantile(o, 0.3, interpolation=m).sum().backward()
        torch.quantile(t, 0.3, interpolation=m).sum().backward()
        np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy(), rtol=1e-6)


def test_quantile_overflow_safe_midpoint_and_linear():
    big = np.array([3e38, 3.4e38], dtype=np.float32)
    for m in ("linear", "midpoint"):
        o = ox.quantile(T(big), 0.5, interpolation=m).to_numpy()
        t = torch.quantile(th(big), 0.5, interpolation=m).numpy()
        assert np.isfinite(o)
        np.testing.assert_allclose(o, t, rtol=1e-6)


def test_quantile_midpoint_integral_and_near_hits():
    # Exactly integral positions read the element (ceil rule).
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    for q, want in ((0.25, 2.0), (0.5, 3.0), (0.75, 4.0)):
        o = ox.quantile(T(f), q, interpolation="midpoint").to_numpy()
        t = torch.quantile(th(f), q, interpolation="midpoint").numpy()
        assert o == want == t
        n = ox.nanquantile(T(f), q, interpolation="midpoint").to_numpy()
        assert n == want
    # Near-integral f32 hits bracket downwards (positions run in f64).
    rng = np.random.default_rng(37)
    x = (rng.standard_normal(141) * 10).astype(np.float32)
    x = np.sort(x)
    o = ox.quantile(T(x), 0.9, interpolation="midpoint").to_numpy()
    t = torch.quantile(th(x), 0.9, interpolation="midpoint").numpy()
    assert o != x[126]
    np.testing.assert_allclose(o, t, rtol=1e-6)


def test_quantile_randomized_fuzz_vs_torch():
    rng = np.random.default_rng(37)
    for trial in range(5):
        n = int(rng.integers(2, 200))
        x = (rng.standard_normal(n) * 10).astype(np.float32)
        for m in ("linear", "midpoint"):
            for q in (0.1, 0.5, 0.9):
                o = ox.quantile(T(x), q, interpolation=m).to_numpy()
                t = torch.quantile(th(x), q, interpolation=m).numpy()
                np.testing.assert_allclose(o, t, rtol=1e-5)


def test_quantile_empty_input_raises():
    e = np.empty((0,), dtype=np.float32)
    with pytest.raises(RuntimeError):
        ox.quantile(T(e), 0.5)
    with pytest.raises(RuntimeError):
        ox.nanquantile(T(e), 0.5)
    with pytest.raises(RuntimeError):
        torch.quantile(th(e), 0.5)
