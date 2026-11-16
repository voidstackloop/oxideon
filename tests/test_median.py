"""Z37: median vs torch ground truth.

The native engine median (lower middle for even counts, NaN-propagating,
differentiable) serves both the Tensor method and, via pure delegation in
`primitives.py`, the module function. Values, indices, and gradients are
pinned against torch.

Deliberate divergences from torch (all pinned below):
- `median` on an empty reduction raises `RuntimeError` (torch: `IndexError`),
  matching the engine's other empty-reduction errors.
- `median` with `dim` returns a plain tuple (torch: namedtuple).
- `median` accepts `keepdim` without `dim` (ones-shaped output); torch
  rejects that combination with `TypeError`.
- `median` accepts bool input (torch: `NotImplementedError`), consistent
  with the engine's bool `sort`/`topk` extension.
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy

X = np.array([[3.0, 1.0, 2.0], [0.5, -1.0, 4.0]], dtype=np.float32)


def test_median_matches_torch_even_takes_lower_middle():
    for dim in (None, 0, 1):
        for keepdim in (False, True):
            if dim is None:
                o = ox.median(T(X), keepdim=keepdim).to_numpy()
                t = torch.median(th(X)).numpy()
                want_shape = (1, 1) if keepdim else ()
                assert o.shape == want_shape
                np.testing.assert_allclose(o, t.reshape(want_shape), equal_nan=True)
                continue
            ov, oi = ox.median(T(X), dim=dim, keepdim=keepdim)
            tv, ti = torch.median(th(X), dim=dim, keepdim=keepdim)
            assert ov.to_numpy().shape == tv.numpy().shape
            np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
            np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
    # Even counts take the LOWER middle (torch never averages; numpy does).
    e = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v, i = T(e).median(dim=0)
    assert v.to_numpy() == 2.0
    assert i.to_numpy() == 1
    mv, mi = ox.median(T(e), dim=0)
    assert mv.to_numpy() == 2.0
    assert mi.to_numpy() == 1


def test_median_int_and_bool():
    xi = np.array([[1, 2, 3], [4, 5, 6]])
    v = ox.median(T(xi)).to_numpy()
    assert v == 3
    assert v.dtype == np.int64
    vv, ii = ox.median(T(xi), dim=1)
    np.testing.assert_array_equal(vv.to_numpy(), [2, 5])
    np.testing.assert_array_equal(ii.to_numpy(), [1, 1])
    tv, ti = torch.median(th(xi), dim=1)
    np.testing.assert_array_equal(vv.to_numpy(), tv.numpy())
    # Extension: bool sorts (False < True) although torch rejects it.
    b = np.array([True, False, True])
    bv, bi = ox.median(T(b), dim=0)
    assert bv.to_numpy() == True
    assert bi.to_numpy() == 0
    with pytest.raises(NotImplementedError):
        torch.median(th(b))


def test_median_nan_propagates_to_first_nan():
    cases = [
        np.array([1.0, np.nan, 2.0], dtype=np.float32),
        np.array([np.nan, 1.0], dtype=np.float32),
        np.array([1.0, np.nan, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32),
        np.array([np.nan, np.nan], dtype=np.float32),
    ]
    for x in cases:
        ov, oi = T(x).median(dim=0)
        tv, ti = torch.median(th(x), dim=0)
        np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
        np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())
        assert np.isnan(ov.to_numpy())
    xn = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    for dim in (0, 1):
        ov, oi = ox.median(T(xn), dim=dim)
        tv, ti = torch.median(th(xn), dim=dim)
        np.testing.assert_allclose(ov.to_numpy(), tv.numpy(), equal_nan=True)
        np.testing.assert_array_equal(oi.to_numpy(), ti.numpy())


def test_median_none_returns_values_only():
    o = ox.median(T(X)).to_numpy()
    t = torch.median(th(X)).numpy()
    assert o.shape == t.shape == ()
    np.testing.assert_allclose(o, t, equal_nan=True)
    assert ox.median(T(np.array([3.0, 1.0, 2.0], dtype=np.float32))).to_numpy() == 2.0


def test_median_grad_flows_to_winner_only():
    for x in (X, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
              np.array([1.0, np.nan, 3.0], dtype=np.float32)):
        o = T(x).requires_grad_(True)
        t = th(x).requires_grad_(True)
        ox.median(o, dim=-1)[0].sum().backward()
        torch.median(t, dim=-1).values.sum().backward()
        np.testing.assert_allclose(o.grad.to_numpy(), t.grad.numpy())


def test_median_empty_raises_and_keepdim_none_extends():
    with pytest.raises(RuntimeError):
        ox.median(T(np.empty((0,), dtype=np.float32)), dim=0)
    with pytest.raises(IndexError):
        torch.median(th(np.empty((0,), dtype=np.float32)), dim=0)
    k = ox.median(T(X), keepdim=True).to_numpy()
    assert k.shape == (1, 1)
    with pytest.raises(TypeError):
        torch.median(th(X), keepdim=True)
