"""torch.linalg parity regressions: batched-vector solve, increasing vander, differentiable LU."""
import numpy as np
import pytest

import oxideon as ox

torch = pytest.importorskip("torch")
rng = np.random.default_rng(9)


def O(a):
    return ox.from_numpy(np.ascontiguousarray(a))


def T(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def test_solve_treats_batch_shaped_rhs_as_vectors():
    A = (rng.standard_normal((2, 4, 4)) + 3 * np.eye(4)).astype(np.float32)
    b = rng.standard_normal((2, 4)).astype(np.float32)
    got = ox.linalg.solve(O(A), O(b))
    assert tuple(got.shape) == (2, 4)
    np.testing.assert_allclose(got.to_numpy(), torch.linalg.solve(T(A), T(b)).numpy(), rtol=1e-4, atol=1e-4)


def test_linalg_vander_is_increasing_like_torch():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(ox.linalg.vander(O(x)).to_numpy(), torch.linalg.vander(T(x)).numpy())
    np.testing.assert_allclose(ox.linalg.vander(O(x), N=2).to_numpy(), torch.linalg.vander(T(x), N=2).numpy())


@pytest.mark.parametrize("shape", [(4, 4), (2, 3, 3)])
def test_lu_and_lu_factor_gradients_match_torch(shape):
    A = (rng.standard_normal(shape) + 3 * np.eye(shape[-1])).astype(np.float32)
    wl, wu = (rng.standard_normal(shape).astype(np.float32) for _ in range(2))
    ao, at = O(A).requires_grad_(True), T(A).requires_grad_(True)
    P, L, U = ox.linalg.lu(ao)
    ((L * O(wl)).sum() + (U * O(wu)).sum()).backward()
    Pt, Lt, Ut = torch.linalg.lu(at)
    ((Lt * T(wl)).sum() + (Ut * T(wu)).sum()).backward()
    np.testing.assert_allclose(ao.grad.to_numpy(), at.grad.numpy(), rtol=2e-3, atol=2e-3)
    ao.grad = None
    lu, _ = ox.linalg.lu_factor(ao)
    (lu * O(wl)).sum().backward()
    at.grad = None
    torch.linalg.lu_factor(at)[0].mul(T(wl)).sum().backward()
    np.testing.assert_allclose(ao.grad.to_numpy(), at.grad.numpy(), rtol=2e-3, atol=2e-3)
