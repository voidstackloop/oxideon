"""Z35: nanmean/nansum/nanmedian vs torch ground truth.

The native engine reductions (NaN-skipping, differentiable) serve both the
Tensor methods and, via pure delegation in `primitives.py`, the module
functions. The lower-middle rule for even counts, (values, indices) tuples,
all-NaN/empty edges, and gradients are pinned against torch.

Deliberate divergences from torch (all pinned below):
- `nanmean` on integer input raises `RuntimeError` (torch: `NotImplementedError`).
- `nanmedian` on an empty reduction raises `RuntimeError` (torch: `IndexError`),
  matching the engine's other empty-reduction errors.
- `nanmedian` with `dim` returns a plain tuple (torch: namedtuple), like `median`.
- `nanmedian` accepts `keepdim` without `dim` (ones-shaped output); torch
  rejects that combination with `TypeError`.
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy

X = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)


def test_nanmean_matches_torch_all_dims():
    for dim in (None, 0, 1):
        for keepdim in (False, True):
            o = ox.nanmean(T(X), dim=dim, keepdim=keepdim).to_numpy()
            t = torch.nanmean(th(X), dim=dim, keepdim=keepdim).numpy()
            assert o.shape == t.shape
            np.testing.assert_allclose(o, t, equal_nan=True)
            m = T(X).nanmean(dim=dim, keepdim=keepdim).to_numpy()
            np.testing.assert_allclose(m, t, equal_nan=True)


def test_nanmean_tuple_dim_module():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x[0, 1, 2] = np.nan
    x[1, 0, 0] = np.nan
    for keepdim in (False, True):
        o = ox.nanmean(T(x), dim=(0, 2), keepdim=keepdim).to_numpy()
        t = torch.nanmean(th(x), dim=(0, 2), keepdim=keepdim).numpy()
        assert o.shape == t.shape
        np.testing.assert_allclose(o, t, equal_nan=True)


def test_nanmean_all_nan_is_nan_and_empty_is_nan():
    a = np.full((2, 3), np.nan, dtype=np.float32)
    assert np.isnan(ox.nanmean(T(a)).to_numpy())
    assert np.isnan(ox.nanmean(T(a), dim=1).to_numpy()).all()
    e = np.empty((0,), dtype=np.float32)
    assert np.isnan(ox.nanmean(T(e)).to_numpy())


def test_nanmean_rejects_integer():
    with pytest.raises(RuntimeError):
        ox.nanmean(T(np.array([1, 2, 3])))
    with pytest.raises(RuntimeError):
        T(np.array([[1, 2], [3, 4]])).nanmean(dim=1)


def test_nanmean_grad_skips_nan():
    o = T(X).requires_grad_(True)
    t = th(X).requires_grad_(True)
    ox.nanmean(o, dim=1).sum().backward()
    torch.nanmean(t, dim=1).sum().backward()
    np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy())


def test_nansum_matches_torch_float_and_int():
    for dim in (None, 0, 1):
        for keepdim in (False, True):
            o = ox.nansum(T(X), dim=dim, keepdim=keepdim).to_numpy()
            t = torch.nansum(th(X), dim=dim, keepdim=keepdim).numpy()
            assert o.shape == t.shape
            np.testing.assert_allclose(o, t, equal_nan=True)
    xi = np.array([[1, 2, 3], [4, 5, 6]])
    o = ox.nansum(T(xi), dim=1).to_numpy()
    np.testing.assert_array_equal(o, [6, 15])
    assert o.dtype == np.int64
    m = T(X).nansum(dim=0, keepdim=True).to_numpy()
    np.testing.assert_allclose(m, torch.nansum(th(X), dim=0, keepdim=True).numpy())


def test_nansum_tuple_dim_all_nan_and_empty():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    x[0, 1, 2] = np.nan
    o = ox.nansum(T(x), dim=(1, 2)).to_numpy()
    np.testing.assert_allclose(o, torch.nansum(th(x), dim=(1, 2)).numpy())
    a = np.full((2, 3), np.nan, dtype=np.float32)
    assert ox.nansum(T(a)).to_numpy() == 0.0
    assert (ox.nansum(T(a), dim=0).to_numpy() == 0.0).all()
    assert ox.nansum(T(np.empty((0,), dtype=np.float32))).to_numpy() == 0.0


def test_nansum_grad_skips_nan():
    o = T(X).requires_grad_(True)
    t = th(X).requires_grad_(True)
    ox.nansum(o, dim=0).sum().backward()
    torch.nansum(t, dim=0).sum().backward()
    np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy())


def test_nanmedian_values_indices_vs_torch():
    cases = [
        np.array([3.0, 1.0, 2.0], dtype=np.float32),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([3.0, np.nan, 1.0, 2.0], dtype=np.float32),
        np.array([np.nan, 5.0, np.nan, 1.0, 9.0], dtype=np.float32),
    ]
    for x in cases:
        ov, oi = T(x).nanmedian(dim=0)
        tv, ti = torch.nanmedian(th(x), dim=0)
        np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
        np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
        mv, mi = ox.nanmedian(T(x), dim=0)
        np.testing.assert_allclose(mv.to_numpy(), tv.numpy(), equal_nan=True)
        np.testing.assert_array_equal(mi.to_numpy(), ti.numpy())
    for dim in (0, 1):
        for keepdim in (False, True):
            ov, oi = ox.nanmedian(T(X), dim=dim, keepdim=keepdim)
            tv, ti = torch.nanmedian(th(X), dim=dim, keepdim=keepdim)
            assert ov.to_numpy().shape == tv.numpy().shape
            np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
            np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())


def test_nanmedian_none_returns_values_only():
    o = ox.nanmedian(T(X)).to_numpy()
    t = torch.nanmedian(th(X)).numpy()
    assert o.shape == t.shape == ()
    np.testing.assert_allclose(o, t, equal_nan=True)
    m = T(X).nanmedian().to_numpy()
    np.testing.assert_allclose(m, t, equal_nan=True)
    k = ox.nanmedian(T(X), keepdim=True).to_numpy()
    assert k.shape == (1, 1)
    np.testing.assert_allclose(k, t.reshape(1, 1), equal_nan=True)
    # Divergence: torch rejects keepdim without dim outright; oxideon extends
    # the ones-shape rule from nanmean/nansum instead (ox.median ignores
    # keepdim when dim is None; nanmedian follows the nanmean/nansum rule).
    with pytest.raises(TypeError):
        torch.nanmedian(th(X), keepdim=True)


def test_nanmedian_int_all_nan_and_empty():
    xi = np.array([[1, 2, 3], [4, 5, 6]])
    v = ox.nanmedian(T(xi)).to_numpy()
    assert v == 3
    assert v.dtype == np.int64
    vv, ii = ox.nanmedian(T(xi), dim=1)
    np.testing.assert_array_equal(vv.to_numpy(), [2, 5])
    np.testing.assert_array_equal(ii.to_numpy(), [1, 1])
    a = np.full(4, np.nan, dtype=np.float32)
    assert np.isnan(ox.nanmedian(T(a)).to_numpy())
    with pytest.raises(RuntimeError):
        ox.nanmedian(T(np.empty((0,), dtype=np.float32)), dim=0)


def test_nanmedian_grad_flows_to_chosen_only():
    o = T(X).requires_grad_(True)
    t = th(X).requires_grad_(True)
    ox.nanmedian(o, dim=1)[0].sum().backward()
    torch.nanmedian(t, dim=1).values.sum().backward()
    np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy())


def test_nan_keepdim_none_matches_torch_shapes():
    for fn, tfn in ((ox.nanmean, torch.nanmean), (ox.nansum, torch.nansum)):
        o = fn(T(X), keepdim=True).to_numpy()
        t = tfn(th(X), keepdim=True).numpy()
        assert o.shape == t.shape == (1, 1)
        np.testing.assert_allclose(o, t, equal_nan=True)
