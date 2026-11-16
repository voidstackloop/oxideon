"""Z31: cumsum/cumprod/diff vs torch ground truth (values, edges, gradients).

cumprod/diff are native engine methods (differentiable); the module fns
delegate, with prepend/append joined via cat. Engine diff reads empty on
short dims like torch instead of erroring.
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy


def test_cumulative_values_and_int_dtypes():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((3, 5)) * 3).astype(np.float32)
    for dim in (0, 1, -1):
        np.testing.assert_allclose(T(x).cumsum(dim).to_numpy(),
                                   th(x).cumsum(dim).numpy(), rtol=1e-6)
        np.testing.assert_allclose(T(x).cumprod(dim).to_numpy(),
                                   th(x).cumprod(dim).numpy(), rtol=1e-6)
        np.testing.assert_allclose(ox.cumprod(T(x), dim).to_numpy(),
                                   torch.cumprod(th(x), dim).numpy(), rtol=1e-6)
        np.testing.assert_allclose(T(x).diff(2, dim).to_numpy(),
                                   th(x).diff(2, dim).numpy(), rtol=1e-6)
    xi = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(T(xi).cumsum(1).to_numpy(), th(xi).cumsum(1).numpy())
    np.testing.assert_array_equal(T(xi).cumprod(1).to_numpy(), th(xi).cumprod(1).numpy())
    np.testing.assert_array_equal(T(xi).diff(1, 1).to_numpy(), th(xi).diff(1, 1).numpy())


def test_diff_edges_match_torch():
    one = np.array([5.0], dtype=np.float32)
    np.testing.assert_array_equal(T(one).diff(1, -1).to_numpy(),
                                  th(one).diff(1, -1).numpy())
    assert T(one).diff(1, -1).to_numpy().shape == (0,)
    two = np.array([1.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(T(two).diff(2, -1).to_numpy(),
                                  th(two).diff(2, -1).numpy())
    np.testing.assert_array_equal(T(two).diff(0, -1).to_numpy(), two)
    with pytest.raises(RuntimeError, match="order must be non-negative"):
        ox.diff(T(two), n=-1)
    with pytest.raises(RuntimeError, match="order must be non-negative"):
        th(two).diff(n=-1)


def test_diff_prepend_append_and_module():
    x = np.array([[1.0, 3.0, 6.0], [2.0, 4.0, 8.0]], dtype=np.float32)
    p = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    a = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    np.testing.assert_allclose(
        ox.diff(T(x), 1, 0, prepend=T(p), append=T(a)).to_numpy(),
        th(x).diff(1, 0, prepend=th(p), append=th(a)).numpy())
    np.testing.assert_allclose(
        T(x).diff(1, 0, prepend=T(p)).to_numpy(),
        th(x).diff(1, 0, prepend=th(p)).numpy())


def test_cumulative_gradients_match_torch():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((3, 5)) * 2).astype(np.float32)
    for name in ("cumsum", "cumprod"):
        xr = T(x).requires_grad_(True)
        getattr(xr, name)(1).sum().backward()
        xt = th(x).requires_grad_(True)
        getattr(xt, name)(1).sum().backward()
        np.testing.assert_allclose(xr.grad.numpy(), xt.grad.numpy(), rtol=1e-5, atol=1e-6)
    xr = T(x).requires_grad_(True)
    xr.diff(1, 1).sum().backward()
    xt = th(x).requires_grad_(True)
    xt.diff(1, 1).sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), xt.grad.numpy())
    # cumprod stays correct with a zero in the row (no division in backward).
    z = np.array([[2.0, 0.0, 3.0]], dtype=np.float32)
    xr = T(z).requires_grad_(True)
    xr.cumprod(1).sum().backward()
    xt = th(z).requires_grad_(True)
    xt.cumprod(1).sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), xt.grad.numpy())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
