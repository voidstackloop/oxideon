"""Tests of the remaining PyTorch-parity additions: max pooling with indices / dilation / ceil_mode, more losses,
in-place functional activations, linalg (*_ex, LDL^T, LU pivots, matrix_sqrth, polar), special polynomials, Hermitian
FFTs, lazy convolutions, CrossMapLRN2d, Buffer, SparseAdam, DistributedSampler and StackDataset.

The first part needs only NumPy; the PyTorch comparisons at the end are skipped where torch is not installed."""
import numpy as np
import pytest

import oxideon as ox

F, L, S = ox.functional, ox.linalg, ox.special
rng = np.random.default_rng(0)


def T(a):
    return ox.from_numpy(np.ascontiguousarray(a))


def A(t):
    return t.to_numpy() if hasattr(t, "to_numpy") else np.asarray(t)


# ---- NumPy-only checks
def test_max_pool2d_indices_point_at_the_maxima():
    x = rng.standard_normal((2, 3, 9, 10)).astype(np.float32)
    vals, idx = F.max_pool2d(T(x), 3, stride=2, padding=1, dilation=1, ceil_mode=True, return_indices=True)
    flat = x.reshape(2, 3, -1)
    np.testing.assert_array_equal(np.take_along_axis(flat, A(idx).reshape(2, 3, -1), -1).reshape(A(vals).shape), A(vals))
    assert A(idx).dtype == np.int64
    assert F.max_pool2d(T(x), 2).shape == ox.nn.MaxPool2d(2)(T(x)).shape


def test_max_pool_gradient_goes_to_the_maxima():
    x = T(np.array([[[1.0, 5.0, 2.0, 2.0]]], np.float32)).requires_grad_(True)
    F.max_pool1d(x, 2).sum().backward()
    np.testing.assert_array_equal(A(x.grad), [[[0, 1, 1, 0]]])  # ties go to the first maximum


def test_adaptive_max_pool_and_lp_pool3d_shapes():
    x = T(rng.standard_normal((1, 2, 7, 8, 9)).astype(np.float32))
    v, i = F.adaptive_max_pool3d(x, (3, 4, 2), return_indices=True)
    assert tuple(v.shape) == tuple(i.shape) == (1, 2, 3, 4, 2)
    assert tuple(F.lp_pool3d(x.abs() + 0.1, 2.0, 2).shape) == (1, 2, 3, 4, 4)


def test_losses_reduce_correctly():
    x = T(rng.standard_normal((4, 3)).astype(np.float32))
    y = T(np.array([0, 2, 1, 1]))
    none = A(F.multi_margin_loss(x, y, reduction="none"))
    assert none.shape == (4,)
    assert np.isclose(A(F.multi_margin_loss(x, y)), none.mean()) and np.isclose(A(F.multi_margin_loss(x, y, reduction="sum")), none.sum())
    with pytest.raises(ValueError):
        F.multi_margin_loss(x, y, p=3)
    assert A(F.triplet_margin_with_distance_loss(x, x, x + 10.0)) == pytest.approx(0.0)


def test_functional_inplace_activation():
    x = T(np.array([-1.0, 2.0], np.float32))
    assert F.relu_(x) is x and A(x).tolist() == [0.0, 2.0]
    with pytest.raises(RuntimeError):
        F.relu_(T(np.array([1.0], np.float32)).requires_grad_(True))


def test_ldl_and_lu_solve_systems():
    m = rng.standard_normal((6, 6))
    sym = m + m.T
    b = rng.standard_normal((6, 2))
    ld, piv = L.ldl_factor(sym)
    np.testing.assert_allclose(sym @ A(L.ldl_solve(ld, piv, b)), b, atol=1e-9)
    lu, p = L.lu_factor(T(m))
    assert A(p).min() >= 1 and A(p).dtype == np.int32  # 1-based, as in PyTorch
    np.testing.assert_allclose(m @ A(L.lu_solve(lu, p, T(b))), b, atol=1e-9)
    sing = np.zeros((3, 3))
    assert int(A(L.ldl_factor_ex(sing).info)) == 1
    assert int(A(L.cholesky_ex(-np.eye(2)).info)) == 1 and int(A(L.inv_ex(sing).info)) == 1
    with pytest.raises(RuntimeError):
        L.cholesky_ex(-np.eye(2), check_errors=True)


def test_matrix_sqrth_and_polar():
    m = rng.standard_normal((4, 4))
    spd = m @ m.T + np.eye(4)
    r = A(L.matrix_sqrth(spd))
    np.testing.assert_allclose(r @ r, spd, atol=1e-9)
    u, p = L.polar(m)
    np.testing.assert_allclose(A(u) @ A(p), m, atol=1e-9)
    np.testing.assert_allclose(A(u) @ A(u).T, np.eye(4), atol=1e-9)


def test_special_polynomials():
    x = np.linspace(-2, 2, 9)
    np.testing.assert_allclose(A(S.hermite_polynomial_he(T(x), 3)), x ** 3 - 3 * x, atol=1e-12)
    assert np.all(A(S.hermite_polynomial_he(T(x), -1)) == 0)
    u = np.linspace(0, 1, 5).astype(np.float32)
    np.testing.assert_allclose(A(S.shifted_chebyshev_polynomial_t(T(u), 2)), 2 * (2 * u - 1) ** 2 - 1, atol=1e-6)


def test_hermitian_fft_round_trip():
    x = rng.standard_normal((4, 6, 8)).astype(np.float32)
    back = ox.fft.hfftn(ox.fft.ihfftn(T(x)), s=x.shape)
    np.testing.assert_allclose(A(back), x, atol=1e-5)


def test_lazy_convolutions_infer_in_channels():
    for cls, shape, wshape in [(ox.nn.LazyConv1d, (2, 3, 10), (4, 3, 3)), (ox.nn.LazyConv3d, (1, 3, 5, 5, 5), (4, 3, 3, 3, 3)),
                               (ox.nn.LazyConvTranspose2d, (2, 3, 5, 6), (3, 4, 3, 3))]:
        m = cls(4, 3)
        assert m.weight is None
        m(T(rng.standard_normal(shape).astype(np.float32)))
        assert tuple(m.weight.shape) == wshape and m.in_channels == 3


def test_buffer_registration():
    class M(ox.nn.Module):
        def __init__(self):
            super().__init__()
            self.running = ox.nn.Buffer(T(np.zeros(3, np.float32)))
            self.scratch = ox.nn.Buffer(T(np.ones(2, np.float32)), persistent=False)

    m = M()
    assert "running" in dict(m.named_buffers()) and isinstance(m.running, ox.Tensor)
    p = ox.nn.UninitializedParameter().materialize((2, 3))
    assert tuple(p.shape) == (2, 3) and p.requires_grad and ox.nn.is_lazy(ox.nn.UninitializedParameter())


def test_cross_map_lrn_shape_and_rnn_cell_base():
    x = T(rng.standard_normal((2, 5, 4, 4)).astype(np.float32))
    assert tuple(ox.nn.CrossMapLRN2d(3)(x).shape) == (2, 5, 4, 4)
    with pytest.raises(ValueError):
        ox.nn.CrossMapLRN2d(3)(T(np.zeros((5, 4, 4), np.float32)))
    cell = ox.nn.RNNCellBase(4, 6, True, num_chunks=4)
    assert tuple(cell.weight_ih.shape) == (24, 4) and tuple(cell.bias_hh.shape) == (24,)


def test_sparse_adam_leaves_untouched_rows_alone():
    w = T(np.ones((5, 2), np.float32)).requires_grad_(True)
    opt = ox.optim.SparseAdam([w], lr=0.1)
    g = np.zeros((5, 2), np.float32)
    g[1] = 1.0
    g[3, 0] = -2.0
    w.set_grad(T(g))
    opt.step()
    out = A(w)
    assert np.all(out[[0, 2, 4]] == 1.0)
    np.testing.assert_allclose(out[1], 0.9, atol=1e-6)
    assert out[3, 0] > 1.0 and out[3, 1] == 1.0  # the row is touched, the zero entry moves by 0


def test_distributed_sampler_partitions_the_data():
    ds = list(range(10))
    parts = [list(ox.data.DistributedSampler(ds, 3, r, shuffle=False)) for r in range(3)]
    assert [len(p) for p in parts] == [4, 4, 4] and sorted(set(sum(parts, []))) == ds
    shuffled = [list(ox.data.DistributedSampler(ds, 2, r, shuffle=True, seed=1)) for r in range(2)]
    assert sorted(sum(shuffled, [])) == ds
    s = ox.data.DistributedSampler(ds, 2, 0, shuffle=True)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first
    assert len(ox.data.DistributedSampler(ds, 3, 0, drop_last=True)) == 3
    with pytest.raises(ValueError):
        ox.data.DistributedSampler(ds, 2, 2)


def test_stack_dataset():
    sd = ox.data.StackDataset([1, 2, 3], "abc")
    assert sd[2] == (3, "c") and len(sd) == 3
    assert ox.data.StackDataset(x=[1, 2], y=[3, 4])[0] == {"x": 1, "y": 3}
    with pytest.raises(ValueError):
        ox.data.StackDataset([1], [1, 2])


# ---- against PyTorch
torch = pytest.importorskip("torch")
TF = torch.nn.functional


def P(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def tclose(o, t, tol=1e-5):
    if isinstance(o, (tuple, list)):
        assert len(o) == len(t)
        for a, b in zip(o, t):
            tclose(a, b, tol)
        return
    a, b = A(o), t.detach().numpy()
    assert a.shape == b.shape
    if a.dtype.kind in "iub":
        np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol, equal_nan=True)


@pytest.mark.parametrize("nd,shape,kw", [
    (1, (2, 3, 11), dict(kernel_size=3, stride=2, padding=1, dilation=2, ceil_mode=True)),
    (2, (2, 3, 9, 10), dict(kernel_size=3, stride=2, padding=1)),
    (2, (2, 3, 9, 10), dict(kernel_size=(3, 2), stride=(2, 1), dilation=(1, 2), ceil_mode=True)),
    (3, (1, 2, 7, 8, 9), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True)),
])
def test_max_pool_against_torch(nd, shape, kw):
    x = rng.standard_normal(shape).astype(np.float32)
    tclose(getattr(F, f"max_pool{nd}d")(T(x), return_indices=True, **kw), getattr(TF, f"max_pool{nd}d")(P(x), return_indices=True, **kw), 0)
    tclose(getattr(F, f"adaptive_max_pool{nd}d")(T(x), 3 if nd == 1 else [3] * nd, True),
           getattr(TF, f"adaptive_max_pool{nd}d")(P(x), 3 if nd == 1 else [3] * nd, True), 0)


def test_losses_against_torch():
    x = rng.standard_normal((5, 4)).astype(np.float32)
    y = np.array([0, 3, 1, 2, 3])
    w = np.array([0.5, 1.0, 2.0, 1.5], np.float32)
    for p in (1, 2):
        tclose(F.multi_margin_loss(T(x), T(y), p=p, margin=0.7, weight=T(w), reduction="none"),
               TF.multi_margin_loss(P(x), P(y), p=p, margin=0.7, weight=P(w), reduction="none"))
    tm = np.array([[3, 0, -1, 1], [1, 2, 3, -1], [0, -1, 2, 2], [2, 1, 0, 3], [-1, 0, 0, 0]])
    tclose(F.multilabel_margin_loss(T(x), T(tm), reduction="none"), TF.multilabel_margin_loss(P(x), P(tm), reduction="none"))
    yb = (rng.random((5, 4)) > 0.5).astype(np.float32)
    tclose(F.multilabel_soft_margin_loss(T(x), T(yb), weight=T(w)), TF.multilabel_soft_margin_loss(P(x), P(yb), weight=P(w)))
    a, p_, n = (rng.standard_normal((6, 5)).astype(np.float32) for _ in range(3))
    tclose(F.triplet_margin_with_distance_loss(T(a), T(p_), T(n), swap=True, reduction="none"),
           TF.triplet_margin_with_distance_loss(P(a), P(p_), P(n), swap=True, reduction="none"))


def test_linalg_against_torch():
    m = rng.standard_normal((5, 5))
    sym = m + m.T
    b = rng.standard_normal((5, 2))
    ld, piv = L.ldl_factor(sym)
    tld, tpiv = torch.linalg.ldl_factor(P(sym))
    tclose((ld, piv), (tld, tpiv), 1e-10)
    tclose(L.ldl_solve(ld, piv, b), torch.linalg.ldl_solve(tld, tpiv, P(b)), 1e-9)
    tclose(L.lu_factor(T(m)), torch.linalg.lu_factor(P(m)), 1e-10)
    tclose(L.matrix_sqrth(sym), torch.linalg.matrix_sqrth(P(sym)), 1e-9)
    tclose(L.polar(m), torch.linalg.polar(P(m)), 1e-9)
    spd = m @ m.T + np.eye(5)
    tclose(L.cholesky_ex(spd), torch.linalg.cholesky_ex(P(spd)), 1e-10)
    tclose(L.solve_ex(spd, b), torch.linalg.solve_ex(P(spd), P(b)), 1e-10)


def test_special_and_fft_against_torch():
    x = np.linspace(-2.5, 2.5, 21)
    for n in (0, 2, 5):
        tclose(S.hermite_polynomial_he(T(x), n), torch.special.hermite_polynomial_he(P(x), n), 1e-9)
    u = np.linspace(0, 1, 11).astype(np.float32)
    for kind in "tuvw":
        tclose(getattr(S, f"shifted_chebyshev_polynomial_{kind}")(T(u), 4), getattr(torch.special, f"shifted_chebyshev_polynomial_{kind}")(P(u), 4), 1e-4)
    r = rng.standard_normal((4, 6, 8)).astype(np.float32)
    for norm in (None, "forward", "ortho"):
        tclose(ox.fft.ihfftn(T(r), norm=norm), torch.view_as_real(torch.fft.ihfftn(P(r), norm=norm)), 1e-5)
        tclose(ox.fft.hfft2(ox.fft.ihfft2(T(r), norm=norm), s=(6, 8), norm=norm),
               torch.fft.hfft2(torch.fft.ihfft2(P(r), norm=norm), s=(6, 8), norm=norm), 1e-4)


@pytest.mark.parametrize("size", [3, 4, 5])
def test_cross_map_lrn_against_torch(size):
    x = rng.standard_normal((2, 6, 5, 5)).astype(np.float32)
    tclose(ox.nn.CrossMapLRN2d(size, 1e-2, 0.75, 2.0)(T(x)), torch.nn.CrossMapLRN2d(size, 1e-2, 0.75, 2.0)(P(x)))


def test_sparse_adam_against_torch():
    w0 = rng.standard_normal((10, 3)).astype(np.float32)
    emb = torch.nn.Embedding(10, 3, sparse=True)
    emb.weight.data = P(w0.copy())
    topt = torch.optim.SparseAdam(list(emb.parameters()), lr=0.05)
    w = T(w0.copy()).requires_grad_(True)
    opt = ox.optim.SparseAdam([w], lr=0.05)
    for idx in ([1, 2, 2, 5], [5, 7], [1, 9, 9]):
        topt.zero_grad()
        (emb(torch.tensor(idx)) ** 2).sum().backward()
        topt.step()
        g = np.zeros_like(w0)
        cur = A(w)
        for i in idx:
            g[i] += 2 * cur[i]
        w.set_grad(T(g))
        opt.step()
    tclose(w, emb.weight)


def test_distributed_sampler_against_torch():
    ds = list(range(11))
    for drop in (False, True):
        for r in range(3):
            assert list(ox.data.DistributedSampler(ds, 3, r, shuffle=False, drop_last=drop)) == \
                list(torch.utils.data.DistributedSampler(ds, 3, r, shuffle=False, drop_last=drop))
