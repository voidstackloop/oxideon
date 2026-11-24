"""Tests of the Rust-backed optimizer steps and linear algebra: every torch.optim optimizer oxideon mirrors gives the
same parameters as PyTorch over several steps, and qr / svd / lu / norm / pinv / eig agree with PyTorch or NumPy."""
import numpy as np
import pytest

import oxideon as ox

torch = pytest.importorskip("torch")
rng = np.random.default_rng(0)


def T(a):
    return ox.from_numpy(np.ascontiguousarray(a))


def A(t):
    return t.to_numpy() if hasattr(t, "to_numpy") else t.detach().numpy()


W0 = rng.standard_normal((4, 3)).astype(np.float32)
GRADS = [rng.standard_normal((4, 3)).astype(np.float32) for _ in range(6)]

CASES = [
    ("RMSprop", dict(lr=0.01)),
    ("RMSprop", dict(lr=0.01, momentum=0.9, centered=True, weight_decay=0.1)),
    ("Adagrad", dict(lr=0.1, lr_decay=0.01, weight_decay=0.05, initial_accumulator_value=0.2)),
    ("Adadelta", dict(lr=1.0, rho=0.8, weight_decay=0.1)),
    ("Adamax", dict(lr=0.02, weight_decay=0.1)),
    ("NAdam", dict(lr=0.01, weight_decay=0.1)),
    ("RAdam", dict(lr=0.01)),
    ("RAdam", dict(lr=0.01, betas=(0.9, 0.5))),  # rho_t > 5 from the start: the rectified branch
    ("ASGD", dict(lr=0.05, lambd=0.1, t0=2, weight_decay=0.1)),
    ("Rprop", dict(lr=0.05)),
]


@pytest.mark.parametrize("name,kw", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_optimizers_match_torch(name, kw):
    w_o = T(W0.copy()).requires_grad_(True)
    w_t = torch.from_numpy(W0.copy()).requires_grad_(True)
    opt_o = getattr(ox.optim, name)([w_o], **kw)
    opt_t = getattr(torch.optim, name)([w_t], **kw)
    for g in GRADS:
        w_o.set_grad(T(g))
        w_t.grad = torch.from_numpy(g.copy())
        opt_o.step()
        opt_t.step()
    np.testing.assert_allclose(A(w_o), A(w_t), rtol=2e-5, atol=2e-6)


def test_optimizer_state_dict_round_trip_keeps_rprop_state():
    w = T(W0.copy()).requires_grad_(True)
    opt = ox.optim.Rprop([w], lr=0.05)
    w.set_grad(T(GRADS[0]))
    opt.step()
    sd = opt.state_dict()
    w2 = T(A(w).copy()).requires_grad_(True)
    opt2 = ox.optim.Rprop([w2], lr=0.05)
    opt2.load_state_dict(sd)
    for o, p in ((opt, w), (opt2, w2)):
        p.set_grad(T(GRADS[1]))
        o.step()
    np.testing.assert_allclose(A(w), A(w2), rtol=1e-6)


def _reference_lion(w, grads, lr, b1, b2, wd):
    m = np.zeros_like(w)
    for g in grads:
        w = w - lr * wd * w
        w = w - lr * np.sign(b1 * m + (1 - b1) * g)
        m = b2 * m + (1 - b2) * g
    return w


def _reference_lamb(w, grads, lr, b1, b2, eps, wd):
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        upd = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps) + wd * w
        wn, rn = np.linalg.norm(w), np.linalg.norm(upd)
        w = w - lr * (wn / rn if wn > 0 and rn > 0 else 1.0) * upd
    return w


def test_lion_and_lamb_match_their_formulas():
    w64, grads = W0.astype(np.float64), [g.astype(np.float64) for g in GRADS]
    for name, kw, ref in [("Lion", dict(lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1), _reference_lion(w64, grads, 1e-2, 0.9, 0.99, 0.1)),
                          ("Lamb", dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01), _reference_lamb(w64, grads, 1e-2, 0.9, 0.999, 1e-6, 0.01))]:
        w = T(W0.copy()).requires_grad_(True)
        opt = getattr(ox.optim, name)([w], **kw)
        for g in GRADS:
            w.set_grad(T(g))
            opt.step()
        np.testing.assert_allclose(A(w), ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kw", [dict(lr=1e-2), dict(lr=1e-2, beta1=0.9, weight_decay=0.1)])
def test_adafactor_reduces_a_quadratic(kw):
    target = rng.standard_normal((5, 4)).astype(np.float32)
    w = T(np.zeros((5, 4), np.float32)).requires_grad_(True)
    opt = ox.optim.Adafactor([w], **kw)
    start = float(((A(w) - target) ** 2).sum())
    for _ in range(300):
        w.set_grad(T(2 * (A(w) - target)))
        opt.step()
    assert float(((A(w) - target) ** 2).sum()) < 0.5 * start


def test_lbfgs_minimizes_a_quadratic():
    a = rng.standard_normal((6, 6))
    h = (a @ a.T + 6 * np.eye(6)).astype(np.float32)
    b = rng.standard_normal(6).astype(np.float32)
    x = T(np.zeros(6, np.float32)).requires_grad_(True)
    opt = ox.optim.LBFGS([x], lr=0.1, max_iter=200, history_size=10)
    hx, bx = T(h), T(b)

    def closure():
        opt.zero_grad()
        loss = (x.reshape([1, 6]).matmul(hx).matmul(x.reshape([6, 1])).sum() * 0.5) - (x * bx).sum()
        loss.backward()
        return float(loss.item())

    for _ in range(5):
        opt.step(closure)
    np.testing.assert_allclose(A(x), np.linalg.solve(h, b), rtol=1e-3, atol=1e-3)


M = rng.standard_normal((5, 3))
SQ = rng.standard_normal((4, 4))


@pytest.mark.parametrize("shape,mode", [((5, 3), "reduced"), ((5, 3), "complete"), ((3, 5), "reduced"), ((3, 5), "complete"), ((2, 4, 4), "complete")])
def test_qr_matches_torch(shape, mode):
    a = rng.standard_normal(shape)
    q, r = ox.linalg.qr(T(a), mode=mode)
    tq, tr = torch.linalg.qr(torch.from_numpy(a), mode=mode)
    np.testing.assert_allclose(A(q), A(tq), atol=1e-10)
    np.testing.assert_allclose(A(r), A(tr), atol=1e-10)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_svd_full_matrices_is_orthogonal_and_reconstructs(shape):
    a = rng.standard_normal(shape)
    u, s, vh = ox.linalg.svd(T(a), full_matrices=True)
    m, n = shape
    assert tuple(u.shape) == (m, m) and tuple(vh.shape) == (n, n)
    np.testing.assert_allclose(A(u) @ A(u).T, np.eye(m), atol=1e-9)
    np.testing.assert_allclose(A(vh) @ A(vh).T, np.eye(n), atol=1e-9)
    k = min(m, n)
    np.testing.assert_allclose(A(u)[:, :k] * A(s) @ A(vh)[:k], a, atol=1e-9)
    np.testing.assert_allclose(A(s), np.linalg.svd(a, compute_uv=False), atol=1e-9)


def test_lu_norm_pinv_eig():
    p, l, u = ox.linalg.lu(T(SQ))
    np.testing.assert_allclose(A(p) @ A(l) @ A(u), SQ, atol=1e-10)
    for ord_ in (None, "fro", "nuc", 2, -2, 1, -1, float("inf"), -float("inf")):
        np.testing.assert_allclose(A(ox.linalg.norm(T(SQ), ord_)), np.linalg.norm(SQ, ord_), rtol=1e-9)
    v = rng.standard_normal(7)
    for ord_ in (None, 1, 3, float("inf"), 0):
        np.testing.assert_allclose(A(ox.linalg.norm(T(v), ord_)), np.linalg.norm(v, ord_), rtol=1e-9)
    np.testing.assert_allclose(A(ox.linalg.norm(T(SQ), dim=1)), np.linalg.norm(SQ, axis=1), rtol=1e-9)
    np.testing.assert_allclose(A(ox.linalg.pinv(T(M))), np.linalg.pinv(M), atol=1e-9)
    np.testing.assert_allclose(A(ox.linalg.pinv(T(SQ + SQ.T), hermitian=True)), np.linalg.pinv(SQ + SQ.T), atol=1e-8)
    np.testing.assert_allclose(A(ox.linalg.pinv(T(M), atol=0.5)), A(torch.linalg.pinv(torch.from_numpy(M), atol=0.5)), atol=1e-9)
    w, vecs = ox.linalg.eig(T(SQ))
    tw = np.linalg.eigvals(SQ)
    assert sorted(np.round(np.asarray(A(w)).astype(complex), 8).tolist(), key=lambda z: (z.real, z.imag)) == \
        sorted(np.round(tw, 8).tolist(), key=lambda z: (z.real, z.imag))
    wv = np.asarray(A(w)).astype(complex)
    vv = np.asarray(A(vecs)).astype(complex)
    np.testing.assert_allclose(SQ @ vv, vv * wv, atol=1e-8)
    sym = SQ + SQ.T
    np.testing.assert_allclose(np.sort(A(ox.linalg.eigvals(T(sym)))), np.linalg.eigvalsh(sym), atol=1e-9)
