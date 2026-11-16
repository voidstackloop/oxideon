"""Z27: kthvalue vs torch ground truth (values, NaN, ties, keepdim, grad)."""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy


def test_kthvalue_matches_torch_values_and_nan():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 9)).astype(np.float32)
    x[0, 0] = np.nan
    x[2, 5] = np.nan
    for dim, ks in ((0, (1, 2, 4)), (1, (1, 2, 5, 8, 9)), (-1, (1, 2, 5, 8, 9))):
        for k in ks:
            for keepdim in (False, True):
                ov, oi = T(x).kthvalue(k, dim=dim, keepdim=keepdim)
                tv, ti = th(x).kthvalue(k, dim=dim, keepdim=keepdim)
                np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
                assert list(ov.shape) == list(tv.shape)
                assert list(oi.shape) == list(ti.shape)


def test_kthvalue_indices_valid_and_exact_without_ties():
    a = np.array([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]], dtype=np.float32)
    for k in (1, 2, 3):
        ov, oi = T(a).kthvalue(k, -1)
        tv, ti = th(a).kthvalue(k, -1)
        np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
    # Ties: any occurrence is valid, but it must reproduce the value.
    t = np.array([[2.0, 1.0, 2.0, 1.0]], dtype=np.float32)
    for k in (1, 2, 3, 4):
        ov, oi = T(t).kthvalue(k, -1)
        tv, _ = th(t).kthvalue(k, -1)
        np.testing.assert_allclose(ov.to_numpy(), tv.numpy())
        np.testing.assert_array_equal(t[0, oi.to_numpy()], ov.to_numpy())


def test_kthvalue_module_dtypes_and_grad():
    mv, mi = ox.kthvalue(T(np.array([3.0, 1.0, 2.0])), 2)
    np.testing.assert_allclose(mv.to_numpy(), [2.0])
    np.testing.assert_array_equal(mi.to_numpy(), [2])
    big = np.array([2**53 + 1, 2**53, 0], dtype=np.int64)
    ov, oi = T(big).kthvalue(2, -1)
    tv, ti = th(big).kthvalue(2, -1)
    np.testing.assert_array_equal(ov.to_numpy(), tv.numpy())
    np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
    xr = T(np.array([3.0, 1.0, 2.0], dtype=np.float32)).requires_grad_(True)
    xr.kthvalue(2, -1)[0].sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), [0.0, 0.0, 1.0])


def test_kthvalue_rejects_bad_k_and_dim():
    x = T(np.array([3.0, 1.0, 2.0], dtype=np.float32))
    for bad in (0, 4):
        with pytest.raises(RuntimeError):
            x.kthvalue(bad, -1)
        with pytest.raises(RuntimeError):
            th(np.array([3.0, 1.0, 2.0])).kthvalue(bad, -1)
    with pytest.raises(RuntimeError):
        x.kthvalue(1, 5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
