"""Z33: bincount/histc/histogram vs torch ground truth.

bincount (new weights support) and histc (new min == max data-range rule)
are native engine methods; the module functions delegate. histogram keeps
its value-correct numpy path (counts exact; edges carry f32 noise).
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy


def test_bincount_plain_and_minlength():
    b = np.array([0, 1, 1, 3, 3, 3], dtype=np.int64)
    np.testing.assert_array_equal(ox.bincount(T(b)).to_numpy(),
                                  torch.bincount(th(b)).numpy())
    np.testing.assert_array_equal(ox.bincount(T(b), minlength=8).to_numpy(),
                                  torch.bincount(th(b), minlength=8).numpy())
    np.testing.assert_array_equal(
        ox.bincount(T(np.array([], dtype=np.int64))).to_numpy(),
        torch.bincount(th(np.array([], dtype=np.int64))).numpy())


def test_bincount_weights_values_and_dtypes():
    b = np.array([0, 1, 1, 3, 3, 3], dtype=np.int64)
    w32 = np.array([0.5, 1.0, 1.5, 2.0, 2.0, 2.0], dtype=np.float32)
    got = ox.bincount(T(b), weights=T(w32), minlength=6).to_numpy()
    want = torch.bincount(th(b), weights=th(w32), minlength=6).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6)
    assert got.dtype == want.dtype == np.float32
    for dt in (np.float64, np.int64, np.bool_):
        w = np.array([1, 2, 1, 0, 1, 2]).astype(dt)
        got = ox.bincount(T(b), weights=T(w)).to_numpy()
        want = torch.bincount(th(b), weights=th(w)).numpy()
        np.testing.assert_allclose(got, want)
        assert got.dtype == want.dtype


def test_bincount_rejections_match_torch():
    b = np.array([0, 1, 1, 3, 3, 3], dtype=np.int64)
    with pytest.raises(RuntimeError):
        ox.bincount(T(np.array([0, -1])))
    with pytest.raises(RuntimeError):
        torch.bincount(th(np.array([0, -1])))
    with pytest.raises(RuntimeError):
        ox.bincount(T(np.array([[0, 1]])))
    with pytest.raises(RuntimeError):
        torch.bincount(th(np.array([[0, 1]])))
    with pytest.raises(RuntimeError):
        ox.bincount(T(np.array([0.0, 1.0])))
    with pytest.raises(RuntimeError, match="minlength should be >= 0"):
        ox.bincount(T(b), minlength=-1)
    with pytest.raises(RuntimeError, match="minlength should be >= 0"):
        torch.bincount(th(b), minlength=-1)
    with pytest.raises(RuntimeError, match="1-d and have the same length"):
        ox.bincount(T(b), weights=T(np.array([1.0, 2.0])))
    with pytest.raises(RuntimeError, match="1-d and have the same length"):
        torch.bincount(th(b), weights=th(np.array([1.0, 2.0])))
    with pytest.raises(RuntimeError, match="1-d and have the same length"):
        ox.bincount(T(b), weights=T(np.ones((6, 1))))


def test_histc_ranges_and_edges():
    x = np.array([0.5, 1.5, 2.5, 9.0], dtype=np.float32)
    np.testing.assert_allclose(ox.histc(T(x), 4, 0.0, 10.0).to_numpy(),
                               th(x).histc(4, 0.0, 10.0).numpy())
    np.testing.assert_allclose(ox.histc(T(x), 4, 0.0, 0.0).to_numpy(),
                               th(x).histc(4, 0.0, 0.0).numpy())
    eq = np.ones(5, dtype=np.float32)
    np.testing.assert_allclose(ox.histc(T(eq), 4, 0.0, 0.0).to_numpy(),
                               th(eq).histc(4, 0.0, 0.0).numpy())
    np.testing.assert_allclose(
        ox.histc(T(np.array([], dtype=np.float32)), 4, 0.0, 0.0).to_numpy(),
        th(np.array([], dtype=np.float32)).histc(4, 0.0, 0.0).numpy())
    n = np.array([0.5, float("nan"), 2.5], dtype=np.float32)
    np.testing.assert_allclose(ox.histc(T(n), 2, 0.0, 3.0).to_numpy(),
                               th(n).histc(2, 0.0, 3.0).numpy())
    with pytest.raises(RuntimeError, match="not finite"):
        ox.histc(T(n), 2, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="not finite"):
        th(n).histc(2, 0.0, 0.0)
    with pytest.raises(RuntimeError):
        ox.histc(T(x), 4, 2.0, 1.0)
    with pytest.raises(RuntimeError):
        ox.histc(T(x), 0, 0.0, 1.0)


def test_histc_int_input_extension():
    # Deliberate superset: torch rejects integer input; oxideon bins it.
    xi = np.array([0, 1, 1, 3], dtype=np.int64)
    got = ox.histc(T(xi), 4, 0.0, 4.0).to_numpy()
    np.testing.assert_array_equal(got, [1, 2, 0, 1])
    assert got.dtype == np.int64


def test_histogram_values_with_edge_tolerance():
    h = (np.random.default_rng(0).standard_normal(50) * 3).astype(np.float32)
    for kw in [{}, {"bins": 5}, {"range": (-2.0, 2.0)}, {"density": True}]:
        got, edges = ox.histogram(T(h), **kw)
        want, wedge = torch.histogram(th(h), **kw)
        # Raw counts exact; density=True divides by the bin width, which
        # inherits f32 noise, so normalized counts get a tolerance too.
        # Edges differ at f32-noise level between numpy's and torch's
        # linspace, so they always get a tolerance.
        if kw.get("density"):
            np.testing.assert_allclose(got.to_numpy(), want.numpy(),
                                       rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(got.to_numpy(), want.numpy())
        np.testing.assert_allclose(edges.to_numpy(), wedge.numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
