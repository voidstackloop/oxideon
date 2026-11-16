"""Z29: masked/gather/scatter ops vs torch ground truth (values + gradients).

Native engine methods serve masked_fill (unpatched), gather/scatter_add
(delegated module fns), and the new engine scatter. masked_select,
masked_scatter, and index_fill keep value-correct numpy shims; their
autograd gap is a documented engine follow-up.

Z40: masked_select/masked_scatter go native (index_select/scatter
composition, differentiable); index_fill stays a shim (follow-up).

Z41: index_fill goes native (mask + where composition, differentiable).
Out-of-range indices and bad dims raise RuntimeError (torch: IndexError).
"""
import numpy as np
import pytest
import torch

import oxideon as ox

T = ox.from_numpy
th = torch.from_numpy
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
M = np.array([[True, False, True], [False, True, False]])


def test_masked_fill_values_broadcast_and_grad():
    np.testing.assert_allclose(T(A).masked_fill(T(M), 9.0).to_numpy(),
                               th(A).masked_fill(th(M), 9.0).numpy())
    np.testing.assert_allclose(ox.masked_fill(T(A), T(M), 9.0).to_numpy(),
                               torch.masked_fill(th(A), th(M), 9.0).numpy())
    mb = np.array([[True, False, True]])
    np.testing.assert_allclose(T(A).masked_fill(T(mb), 9.0).to_numpy(),
                               th(A).masked_fill(th(mb), 9.0).numpy())
    xr = T(A).requires_grad_(True)
    xr.masked_fill(T(M), 9.0).sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(),
                               [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    xi = T(A)
    xi.masked_fill_(T(M), 9.0)
    np.testing.assert_allclose(xi.to_numpy(), [[9.0, 2.0, 9.0], [4.0, 9.0, 6.0]])


def test_gather_scatter_add_values_and_grad():
    idx = np.array([[0, 2], [1, 0]])
    s = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    np.testing.assert_allclose(ox.gather(T(A), 1, T(idx)).to_numpy(),
                               torch.gather(th(A), 1, th(idx)).numpy())
    np.testing.assert_allclose(ox.scatter_add(T(A), 1, T(idx), T(s)).to_numpy(),
                               torch.scatter_add(th(A), 1, th(idx), th(s)).numpy())
    xr = T(A).requires_grad_(True)
    ox.gather(xr, 1, T(idx)).sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    xr = T(A).requires_grad_(True)
    sr = T(s).requires_grad_(True)
    ox.scatter_add(xr, 1, T(idx), sr).sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), np.ones_like(A))
    np.testing.assert_allclose(sr.grad.numpy(), np.ones_like(s))


def test_scatter_values_dups_negatives_and_grad():
    idx = np.array([[0, 2], [1, 0]])
    s = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    np.testing.assert_allclose(T(A).scatter(1, T(idx), T(s)).to_numpy(),
                               torch.scatter(th(A), 1, th(idx), th(s)).numpy())
    np.testing.assert_allclose(ox.scatter(T(A), 1, T(idx), T(s)).to_numpy(),
                               torch.scatter(th(A), 1, th(idx), th(s)).numpy())
    # Duplicate indices: last write wins, matching torch CPU.
    r = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(
        ox.scatter(T(r), 1, T(np.array([[0, 0]])),
                   T(np.array([[7.0, 8.0]], dtype=np.float32))).to_numpy(),
        torch.scatter(th(r), 1, th(np.array([[0, 0]])),
                      th(np.array([[7.0, 8.0]], dtype=np.float32))).numpy())
    # Deliberate divergence (pinned for the whole gather family): negative
    # indices wrap instead of raising like torch.
    np.testing.assert_allclose(
        ox.scatter(T(r), 1, T(np.array([[-1, 0]])),
                   T(np.array([[7.0, 8.0]], dtype=np.float32))).to_numpy(),
        [[8.0, 2.0, 7.0]])
    xr = T(A).requires_grad_(True)
    sr = T(s).requires_grad_(True)
    ox.scatter(xr, 1, T(idx), sr).sum().backward()
    np.testing.assert_allclose(xr.grad.numpy(), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(sr.grad.numpy(), np.ones_like(s))


def test_masked_select_scatter_index_fill_values():
    np.testing.assert_allclose(ox.masked_select(T(A), T(M)).to_numpy(),
                               torch.masked_select(th(A), th(M)).numpy())
    np.testing.assert_allclose(T(A).masked_select(T(M)).to_numpy(),
                               torch.masked_select(th(A), th(M)).numpy())
    src = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    np.testing.assert_allclose(ox.masked_scatter(T(A), T(M), T(src)).to_numpy(),
                               torch.masked_scatter(th(A), th(M), th(src)).numpy())
    np.testing.assert_allclose(
        ox.index_fill(T(A), 1, T(np.array([0, 2])), 9.0).to_numpy(),
        torch.index_fill(th(A), 1, th(np.array([0, 2])), 9.0).numpy())


def test_masked_select_broadcast_and_validation():
    for ms in (np.array([True, False, True]), np.array(True), np.array([[True]])):
        np.testing.assert_array_equal(ox.masked_select(T(A), T(ms)).to_numpy(),
                                      torch.masked_select(th(A), th(ms)).numpy())
    np.testing.assert_array_equal(T(A).masked_select(T(M)).to_numpy(), [1.0, 3.0, 5.0])
    # Masks must be boolean, like torch.
    with pytest.raises(RuntimeError):
        ox.masked_select(T(A), T(M.astype(np.int64)))
    with pytest.raises(RuntimeError):
        torch.masked_select(th(A), th(M.astype(np.int64)))
    # Non-broadcastable masks fail closed on both sides.
    bad = np.array([True, False])
    with pytest.raises(RuntimeError):
        ox.masked_select(T(A), T(bad))
    with pytest.raises(RuntimeError):
        torch.masked_select(th(A), th(bad))


def test_masked_select_grad_and_edges():
    xr = T(A).requires_grad_(True)
    ox.masked_select(xr, T(M)).sum().backward()
    tr = th(A).requires_grad_(True)
    torch.masked_select(tr, th(M)).sum().backward()
    np.testing.assert_array_equal(xr.grad.numpy(), tr.grad.numpy())
    z = np.array(5.0, dtype=np.float32)
    o = ox.masked_select(T(z), T(np.array(True))).to_numpy()
    assert o.shape == (1,) and o[0] == 5.0
    o = ox.masked_select(T(z), T(np.array(False))).to_numpy()
    assert o.shape == (0,)
    e = np.zeros((0, 3), dtype=np.float32)
    o = ox.masked_select(T(e), T(np.zeros((0, 3), dtype=bool))).to_numpy()
    assert o.shape == (0,)
    o = ox.masked_select(T(A), T(np.zeros_like(M))).to_numpy()
    assert o.shape == (0,)


def test_masked_scatter_source_rules_values_and_grad():
    exact = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    long = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    want = torch.masked_scatter(th(A), th(M), th(exact)).numpy()
    np.testing.assert_array_equal(ox.masked_scatter(T(A), T(M), T(exact)).to_numpy(), want)
    # Longer sources silently drop the tail, like torch.
    np.testing.assert_array_equal(ox.masked_scatter(T(A), T(M), T(long)).to_numpy(), want)
    # N-D sources flatten in order, like torch.
    np.testing.assert_array_equal(
        ox.masked_scatter(T(A), T(M), T(exact.reshape(1, 3))).to_numpy(), want)
    # Too few elements fail closed on both sides.
    with pytest.raises(RuntimeError):
        ox.masked_scatter(T(A), T(M), T(exact[:2]))
    with pytest.raises(RuntimeError):
        torch.masked_scatter(th(A), th(M), th(exact[:2]))
    # Dtypes must match, and masks must be boolean, on both sides.
    with pytest.raises(RuntimeError):
        ox.masked_scatter(T(A), T(M), T(np.array([10, 20, 30])))
    with pytest.raises(RuntimeError):
        torch.masked_scatter(th(A), th(M), th(np.array([10, 20, 30])))
    with pytest.raises(RuntimeError):
        ox.masked_scatter(T(A), T(M.astype(np.int64)), T(exact))
    # Gradients: base keeps the complement, source takes ones (tail zeros).
    xr = T(A).requires_grad_(True)
    sr = T(long).requires_grad_(True)
    ox.masked_scatter(xr, T(M), sr).sum().backward()
    tr = th(A).requires_grad_(True)
    ts = th(long).requires_grad_(True)
    torch.masked_scatter(tr, th(M), ts).sum().backward()
    np.testing.assert_array_equal(xr.grad.numpy(), tr.grad.numpy())
    np.testing.assert_array_equal(sr.grad.numpy(), ts.grad.numpy())
    # 0-d and all-false edges (genuine 0-d via reduction: from_numpy folds).
    z5 = T(np.array([5.0], dtype=np.float32)).sum()
    assert z5.to_numpy().shape == ()
    zv = ox.masked_scatter(z5, z5 == z5,
                           T(np.array([10.0], dtype=np.float32))).to_numpy()
    assert zv.shape == () and zv == 10.0
    e = ox.masked_scatter(T(A), T(np.zeros_like(M)),
                          T(np.array([], dtype=np.float32))).to_numpy()
    np.testing.assert_array_equal(e, A)


def test_index_fill_values_wrap_and_validation():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    for idx in ([1, 3], [-1], [1, 1]):
        o = ox.index_fill(T(a), 1, T(np.array(idx)), 9.0).to_numpy()
        t = torch.index_fill(th(a), 1, th(np.array(idx)), 9.0).numpy()
        np.testing.assert_array_equal(o, t)
    m = T(a).index_fill(0, T(np.array([0, 2])), -1.0).to_numpy()
    np.testing.assert_array_equal(m, torch.index_fill(th(a), 0, th(np.array([0, 2])), -1.0).numpy())
    # Empty index is a no-op; 0-d index behaves as a scalar.
    np.testing.assert_array_equal(
        ox.index_fill(T(a), 1, T(np.array([], dtype=np.int64)), 9.0).to_numpy(), a)
    q0 = T(np.array([1])).sum()
    assert q0.to_numpy().shape == ()
    o = ox.index_fill(T(a), 1, q0, 9.0).to_numpy()
    np.testing.assert_array_equal(o, torch.index_fill(th(a), 1, th(np.array(1)), 9.0).numpy())
    # Index must be int64 vector/scalar, like torch.
    with pytest.raises(RuntimeError):
        ox.index_fill(T(a), 1, T(np.array([1], dtype=np.int32)), 9.0)
    with pytest.raises(IndexError):
        torch.index_fill(th(a), 1, th(np.array([1], dtype=np.int32)), 9.0)
    with pytest.raises(RuntimeError):
        ox.index_fill(T(a), 1, T(np.array([[1]], dtype=np.int64)), 9.0)
    # Out-of-range indices and bad dims fail closed (torch: IndexError).
    with pytest.raises(RuntimeError):
        ox.index_fill(T(a), 1, T(np.array([4])), 9.0)
    with pytest.raises(IndexError):
        torch.index_fill(th(a), 1, th(np.array([4])), 9.0)
    with pytest.raises(RuntimeError):
        ox.index_fill(T(a), 5, T(np.array([1])), 9.0)


def test_index_fill_dtypes_edges_and_grad():
    xi = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(ox.index_fill(T(xi), 0, T(np.array([1])), 9).to_numpy(),
                                  torch.index_fill(th(xi), 0, th(np.array([1])), 9).numpy())
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(ox.index_fill(T(b), 0, T(np.array([1])), True).to_numpy(),
                                  torch.index_fill(th(b), 0, th(np.array([1])), True).numpy())
    # 0-d input behaves as size-1 (genuine 0-d via reduction).
    z5 = T(np.array([5.0], dtype=np.float32)).sum()
    for d in (0, -1):
        zv = ox.index_fill(z5, d, T(np.array([0])), 9.0).to_numpy()
        assert zv.shape == () and zv == 9.0
    with pytest.raises(RuntimeError):
        ox.index_fill(z5, 0, T(np.array([5])), 9.0)
    # Tensor values coerce through item(), like torch Scalar.
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    v = T(np.array([9.0], dtype=np.float32)).sum()
    np.testing.assert_array_equal(ox.index_fill(T(a), 1, T(np.array([2])), v).to_numpy(),
                                  torch.index_fill(th(a), 1, th(np.array([2])), 9.0).numpy())
    # Gradients are zero at filled positions only.
    xr = T(a).requires_grad_(True)
    ox.index_fill(xr, 1, T(np.array([1, 3])), 9.0).sum().backward()
    tr = th(a).requires_grad_(True)
    torch.index_fill(tr, 1, th(np.array([1, 3])), 9.0).sum().backward()
    np.testing.assert_array_equal(xr.grad.numpy(), tr.grad.numpy())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
