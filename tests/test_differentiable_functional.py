"""Differentiability of the functional/nn layers (no silent autograd detach).

Regression tests: several ``functional`` ops and ``nn`` modules used to
round-trip through NumPy, so outputs never required grad (pooling, deconv,
layer_norm, embedding, SDPA) or silently trained only the last layer
(MultiheadAttention, LSTM/GRU cells). Every case below checks that gradients
flow and match PyTorch (values and grads).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as TF

import oxideon as ox
import oxideon.nn as nn
from oxideon import functional as F


def O(arr):
    return ox.from_numpy(np.ascontiguousarray(arr)).requires_grad_(True)


def T(arr):
    return torch.from_numpy(np.ascontiguousarray(arr)).requires_grad_(True)


def assert_close(name, o, t, tol=1e-4):
    o = o.to_numpy() if hasattr(o, "to_numpy") else np.asarray(o, dtype=np.float32)
    t = t.detach().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    assert o.shape == tuple(t.shape), (name, o.shape, tuple(t.shape))
    assert np.allclose(o, t, rtol=tol, atol=tol), (name, np.abs(o - t).max())


def test_outputs_require_grad():
    rng = np.random.default_rng(0)
    x4 = O(rng.standard_normal([2, 3, 8, 8]).astype(np.float32))
    assert F.avg_pool2d(x4, 2).requires_grad
    assert F.max_pool2d(x4, 2).requires_grad
    assert F.adaptive_avg_pool2d(x4, (4, 4)).requires_grad
    assert nn.AvgPool2d(2)(x4).requires_grad
    assert nn.MaxPool2d(2)(x4).requires_grad
    w = ox.from_numpy((rng.standard_normal([3, 2, 3, 3]) * 0.5).astype(np.float32))
    x = O(rng.standard_normal([1, 3, 5, 5]).astype(np.float32))
    assert F.conv_transpose2d(x, w).requires_grad
    assert nn.ConvTranspose2d(3, 2, 3)(x).requires_grad
    xl = O(rng.standard_normal([4, 6]).astype(np.float32))
    lw = ox.from_numpy(rng.standard_normal([6]).astype(np.float32))
    assert F.layer_norm(xl, [6], lw).requires_grad
    emb_w = O(rng.standard_normal([10, 8]).astype(np.float32))
    idx = ox.from_numpy(np.array([[1, 2], [3, 4]]))
    assert F.embedding(idx, emb_w).requires_grad
    q = O((rng.standard_normal([1, 2, 4, 8]) * 0.5).astype(np.float32))
    k = ox.from_numpy((rng.standard_normal([1, 2, 4, 8]) * 0.5).astype(np.float32))
    v = ox.from_numpy(rng.standard_normal([1, 2, 4, 8]).astype(np.float32))
    assert F.scaled_dot_product_attention(q, k, v).requires_grad
    cell = nn.LSTMCell(3, 4)
    h, c = cell(O(rng.standard_normal([2, 3]).astype(np.float32)))
    assert h.requires_grad and c.requires_grad
    g = nn.GRUCell(3, 4)
    assert g(O(rng.standard_normal([2, 3]).astype(np.float32))).requires_grad


def test_sdpa_matches_torch():
    rng = np.random.default_rng(1)
    Q = (rng.standard_normal([2, 4, 8, 16]) * 0.5).astype(np.float32)
    K = (rng.standard_normal([2, 4, 8, 16]) * 0.5).astype(np.float32)
    V = rng.standard_normal([2, 4, 8, 16]).astype(np.float32)
    oq, tq = O(Q), T(Q)
    ok, tk = O(K), T(K)
    ov, tv = O(V), T(V)
    yo = F.scaled_dot_product_attention(oq, ok, ov)
    yt = TF.scaled_dot_product_attention(tq, tk, tv)
    assert_close("sdpa-fwd", yo, yt)
    yo.sum().backward()
    yt.sum().backward()
    assert_close("sdpa-qgrad", oq.grad, tq.grad)
    assert_close("sdpa-kgrad", ok.grad, tk.grad)
    assert_close("sdpa-vgrad", ov.grad, tv.grad)


def test_pooling_matches_torch():
    rng = np.random.default_rng(2)
    X = rng.standard_normal([2, 3, 8, 8]).astype(np.float32)
    for name, of, tf in [("avg", F.avg_pool2d, TF.avg_pool2d), ("max", F.max_pool2d, TF.max_pool2d)]:
        oa, ta = O(X), T(X)
        assert_close(f"{name}-fwd", of(oa, 2), tf(ta, 2))
        oa2, ta2 = O(X), T(X)
        of(oa2, 2).sum().backward()
        tf(ta2, 2).sum().backward()
        assert_close(f"{name}-xgrad", oa2.grad, ta2.grad, tol=2e-4)


def test_deconv_matches_torch():
    rng = np.random.default_rng(3)
    X = rng.standard_normal([1, 2, 5, 5]).astype(np.float32)
    W = (rng.standard_normal([2, 3, 3, 3]) * 0.5).astype(np.float32)
    oa, ta = O(X), T(X)
    ow = ox.from_numpy(W.copy())
    tw = torch.from_numpy(W.copy())
    assert_close("deconv-fwd", F.conv_transpose2d(oa, ow), TF.conv_transpose2d(ta, tw))
    oa2, ta2 = O(X), T(X)
    ow2 = O(W.copy())
    tw2 = T(W.copy())
    F.conv_transpose2d(oa2, ow2).sum().backward()
    TF.conv_transpose2d(ta2, tw2).sum().backward()
    assert_close("deconv-xgrad", oa2.grad, ta2.grad)
    assert_close("deconv-wgrad", ow2.grad, tw2.grad)


def test_layer_norm_matches_torch():
    rng = np.random.default_rng(4)
    X = rng.standard_normal([4, 6]).astype(np.float32)
    W = rng.standard_normal([6]).astype(np.float32)
    B = rng.standard_normal([6]).astype(np.float32)
    oa, ta = O(X), T(X)
    ow, tw = O(W.copy()), T(W.copy())
    ob, tb = O(B.copy()), T(B.copy())
    assert_close("ln-fwd", F.layer_norm(oa, [6], ow, ob, 1e-5), TF.layer_norm(ta, [6], tw, tb, 1e-5))
    oa2, ta2 = O(X), T(X)
    F.layer_norm(oa2, [6], ow, ob, 1e-5).sum().backward()
    TF.layer_norm(ta2, [6], tw, tb, 1e-5).sum().backward()
    assert_close("ln-xgrad", oa2.grad, ta2.grad)


def test_embedding_matches_torch():
    rng = np.random.default_rng(5)
    W = rng.standard_normal([10, 8]).astype(np.float32)
    I = rng.integers(0, 10, size=[4, 5])
    ow, tw = O(W.copy()), T(W.copy())
    oi = ox.from_numpy(I.copy())
    ti = torch.from_numpy(I.copy())
    assert_close("emb-fwd", F.embedding(oi, ow), TF.embedding(ti, tw))
    ow2, tw2 = O(W.copy()), T(W.copy())
    F.embedding(oi, ow2).sum().backward()
    TF.embedding(ti, tw2).sum().backward()
    assert_close("emb-wgrad", ow2.grad, tw2.grad)


def test_lstm_cell_matches_torch():
    rng = np.random.default_rng(6)
    X = rng.standard_normal([2, 3]).astype(np.float32)
    H = rng.standard_normal([2, 4]).astype(np.float32)
    C = rng.standard_normal([2, 4]).astype(np.float32)
    Wi = rng.standard_normal([16, 3]).astype(np.float32) * 0.5
    Wh = rng.standard_normal([16, 4]).astype(np.float32) * 0.5
    Bi = rng.standard_normal([16]).astype(np.float32) * 0.5
    Bh = rng.standard_normal([16]).astype(np.float32) * 0.5
    oc = nn.LSTMCell(3, 4)
    oc.weight_ih = O(Wi.copy())
    oc.weight_hh = O(Wh.copy())
    oc.bias_ih = O(Bi.copy())
    oc.bias_hh = O(Bh.copy())
    tc = torch.nn.LSTMCell(3, 4)
    tc.weight_ih.data = torch.from_numpy(Wi.copy())
    tc.weight_hh.data = torch.from_numpy(Wh.copy())
    tc.bias_ih.data = torch.from_numpy(Bi.copy())
    tc.bias_hh.data = torch.from_numpy(Bh.copy())
    xo, xt = O(X.copy()), T(X.copy())
    ho = (O(H.copy()), O(C.copy()))
    ht = (torch.from_numpy(H.copy()).requires_grad_(True), torch.from_numpy(C.copy()).requires_grad_(True))
    (oh, oc_), (th, tc_) = oc(xo, ho), tc(xt, ht)
    assert_close("lstm-h", oh, th)
    assert_close("lstm-c", oc_, tc_)
    (oh.sum() + oc_.sum()).backward()
    (th.sum() + tc_.sum()).backward()
    assert_close("lstm-xgrad", xo.grad, xt.grad, tol=2e-4)
    assert_close("lstm-wih", oc.weight_ih.grad, tc.weight_ih.grad, tol=2e-4)


def test_gru_cell_matches_torch():
    rng = np.random.default_rng(7)
    X = rng.standard_normal([2, 3]).astype(np.float32)
    H = rng.standard_normal([2, 4]).astype(np.float32)
    Wi = rng.standard_normal([12, 3]).astype(np.float32) * 0.5
    Wh = rng.standard_normal([12, 4]).astype(np.float32) * 0.5
    Bi = rng.standard_normal([12]).astype(np.float32) * 0.5
    Bh = rng.standard_normal([12]).astype(np.float32) * 0.5
    oc = nn.GRUCell(3, 4)
    oc.weight_ih = O(Wi.copy())
    oc.weight_hh = O(Wh.copy())
    oc.bias_ih = O(Bi.copy())
    oc.bias_hh = O(Bh.copy())
    tc = torch.nn.GRUCell(3, 4)
    tc.weight_ih.data = torch.from_numpy(Wi.copy())
    tc.weight_hh.data = torch.from_numpy(Wh.copy())
    tc.bias_ih.data = torch.from_numpy(Bi.copy())
    tc.bias_hh.data = torch.from_numpy(Bh.copy())
    xo, xt = O(X.copy()), T(X.copy())
    ho = O(H.copy())
    ht = torch.from_numpy(H.copy()).requires_grad_(True)
    oh, th = oc(xo, ho), tc(xt, ht)
    assert_close("gru-h", oh, th)
    oh.sum().backward()
    th.sum().backward()
    assert_close("gru-xgrad", xo.grad, xt.grad, tol=2e-4)
    assert_close("gru-wih", oc.weight_ih.grad, tc.weight_ih.grad, tol=2e-4)


def test_mha_trains_every_projection():
    rng = np.random.default_rng(8)
    mha = nn.MultiheadAttention(8, 2, batch_first=True)
    x = O(rng.standard_normal([1, 4, 8]).astype(np.float32))
    y = mha(x, x, x)[0]
    assert y.requires_grad
    y.sum().backward()
    assert x.grad is not None
    for p in mha.parameters():
        g = p.grad if not callable(p.grad) else p.grad()
        assert g is not None, "a projection received no gradient"
    # Numeric gradcheck on one q_proj element (finite differences).
    w = mha.q_proj.weight.to_numpy().copy()
    flat_idx = 0
    eps = 1e-3
    projs = {}
    for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        mod = getattr(mha, name)
        projs[name] = (mod.weight.to_numpy().copy(), mod.bias.to_numpy().copy())

    def run(mat):
        m = nn.MultiheadAttention(8, 2, batch_first=True)
        m.q_proj.weight = ox.from_numpy(mat)
        m.q_proj.bias = ox.from_numpy(projs["q_proj"][1])
        for name in ["k_proj", "v_proj", "out_proj"]:
            getattr(m, name).weight = ox.from_numpy(projs[name][0])
            getattr(m, name).bias = ox.from_numpy(projs[name][1])
        return m(x, x, x)[0].sum().to_numpy().item()
    f0 = run(w)
    wp = w.copy().reshape(-1)
    wp[flat_idx] += eps
    fp = run(wp.reshape(w.shape))
    wm = w.copy().reshape(-1)
    wm[flat_idx] -= eps
    fm = run(wm.reshape(w.shape))
    num = (fp - fm) / (2 * eps)
    got = mha.q_proj.weight.grad.to_numpy().reshape(-1)[flat_idx]
    assert abs(got - num) / max(1.0, abs(num)) < 5e-2, (got, num)


def test_pool_1d_3d_match_torch():
    rng = np.random.default_rng(9)
    x1 = rng.standard_normal([2, 3, 9]).astype(np.float32)
    for name, of, tf in [("avg1d", F.avg_pool1d, TF.avg_pool1d), ("max1d", F.max_pool1d, TF.max_pool1d)]:
        oa, ta = O(x1), T(x1)
        assert_close(f"{name}-fwd", of(oa, 3, stride=2), tf(ta, 3, stride=2))
        oa2, ta2 = O(x1), T(x1)
        of(oa2, 3, stride=2).sum().backward()
        tf(ta2, 3, stride=2).sum().backward()
        assert_close(f"{name}-xgrad", oa2.grad, ta2.grad, tol=2e-4)
    x3 = rng.standard_normal([1, 2, 5, 5, 5]).astype(np.float32)
    for name, of, tf in [("avg3d", F.avg_pool3d, TF.avg_pool3d), ("max3d", F.max_pool3d, TF.max_pool3d)]:
        oa, ta = O(x3), T(x3)
        assert_close(f"{name}-fwd", of(oa, 2), tf(ta, 2))
        oa2, ta2 = O(x3), T(x3)
        of(oa2, 2).sum().backward()
        tf(ta2, 2).sum().backward()
        assert_close(f"{name}-xgrad", oa2.grad, ta2.grad, tol=2e-4)
    oa, ta = O(x1), T(x1)
    assert_close("adap1d-fwd", F.adaptive_avg_pool1d(oa, 4), TF.adaptive_avg_pool1d(ta, 4))
    oa2, ta2 = O(x1), T(x1)
    F.adaptive_avg_pool1d(oa2, 4).sum().backward()
    TF.adaptive_avg_pool1d(ta2, 4).sum().backward()
    assert_close("adap1d-xgrad", oa2.grad, ta2.grad, tol=2e-4)
    oa, ta = O(x3), T(x3)
    assert_close("adap3d-fwd", F.adaptive_avg_pool3d(oa, (2, 2, 2)), TF.adaptive_avg_pool3d(ta, (2, 2, 2)))
    oa2, ta2 = O(x3), T(x3)
    F.adaptive_avg_pool3d(oa2, (2, 2, 2)).sum().backward()
    TF.adaptive_avg_pool3d(ta2, (2, 2, 2)).sum().backward()
    assert_close("adap3d-xgrad", oa2.grad, ta2.grad, tol=2e-4)


def test_deconv_1d_3d_match_torch():
    rng = np.random.default_rng(10)
    x1 = rng.standard_normal([1, 2, 6]).astype(np.float32)
    w1 = (rng.standard_normal([2, 3, 3]) * 0.5).astype(np.float32)
    oa, ta = O(x1), T(x1)
    ow = ox.from_numpy(w1.copy())
    tw = torch.from_numpy(w1.copy())
    assert_close("deconv1d-fwd", F.conv_transpose1d(oa, ow), TF.conv_transpose1d(ta, tw))
    oa2, ta2 = O(x1), T(x1)
    ow2, tw2 = O(w1.copy()), T(w1.copy())
    F.conv_transpose1d(oa2, ow2).sum().backward()
    TF.conv_transpose1d(ta2, tw2).sum().backward()
    assert_close("deconv1d-xgrad", oa2.grad, ta2.grad)
    assert_close("deconv1d-wgrad", ow2.grad, tw2.grad)
    x3 = rng.standard_normal([1, 1, 4, 4, 4]).astype(np.float32)
    w3 = (rng.standard_normal([1, 1, 2, 2, 2]) * 0.5).astype(np.float32)
    oa, ta = O(x3), T(x3)
    ow = ox.from_numpy(w3.copy())
    tw = torch.from_numpy(w3.copy())
    assert_close("deconv3d-fwd", F.conv_transpose3d(oa, ow), TF.conv_transpose3d(ta, tw))
    oa2, ta2 = O(x3), T(x3)
    ow2, tw2 = O(w3.copy()), T(w3.copy())
    F.conv_transpose3d(oa2, ow2).sum().backward()
    TF.conv_transpose3d(ta2, tw2).sum().backward()
    assert_close("deconv3d-xgrad", oa2.grad, ta2.grad)
    assert_close("deconv3d-wgrad", ow2.grad, tw2.grad)
