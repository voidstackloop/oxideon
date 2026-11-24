"""Tests of oxideon.diagnostics and of the second-order autograd it relies on."""
import math

import numpy as np
import pytest

import oxideon as ox
from oxideon import diagnostics as dg


def T(a):
    return ox.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def mse(o, t):
    return ((o - t) * (o - t)).mean()


class Net(ox.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = ox.nn.Linear(4, 8)
        self.act = ox.nn.ReLU()
        self.fc2 = ox.nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LogLayer(ox.nn.Module):
    def forward(self, x):
        return x.log()


class SqrtOfSquare(ox.nn.Module):
    def forward(self, x):
        return (x * x).sqrt()  # finite forward, 0 / 0 in the backward pass at x == 0


# ---- autograd.grad (fixed alongside: it used to accumulate into .grad)
def test_autograd_grad_does_not_accumulate_or_touch_grad():
    w = T([1.0, 2.0, 3.0]).requires_grad_(True)
    for _ in range(3):
        g = ox.autograd.grad((w * w).sum(), [w])[0]
        np.testing.assert_allclose(g.to_numpy(), [2, 4, 6])
    assert w.grad is None


def test_hessian_vector_products_are_exact():
    w = T([1.0, 2.0, 3.0]).requires_grad_(True)
    g = ox.autograd.grad((w * w * w).sum(), [w], create_graph=True)[0]
    hv = ox.autograd.grad(g.sum(), [w])[0]
    np.testing.assert_allclose(hv.to_numpy(), [6, 12, 18], rtol=1e-6)
    g = ox.autograd.grad(w.exp().sum(), [w], create_graph=True)[0]
    np.testing.assert_allclose(ox.autograd.grad(g.sum(), [w])[0].to_numpy(), np.exp([1, 2, 3]), rtol=1e-6)


def test_autograd_grad_with_grad_outputs():
    w = T([1.0, 2.0]).requires_grad_(True)
    y = w * 3.0
    g = ox.autograd.grad(y, [w], grad_outputs=T([1.0, 10.0]))[0]
    np.testing.assert_allclose(g.to_numpy(), [3, 30])


# ---- detect_anomaly
def test_anomaly_forward_names_the_layer():
    bad = ox.nn.Sequential(ox.nn.Linear(4, 4), LogLayer(), ox.nn.Linear(4, 2))
    with pytest.raises(dg.AnomalyError) as e:
        with dg.detect_anomaly(bad):
            bad(T(np.random.default_rng(0).standard_normal((5, 4))))
    assert e.value.module == "1" and e.value.phase == "forward" and "LogLayer" in str(e.value)
    assert type(bad._modules["1"]).__name__ == "LogLayer"  # probes removed


def test_anomaly_backward_names_the_layer():
    net = ox.nn.Sequential(ox.nn.Linear(4, 4), SqrtOfSquare(), ox.nn.Linear(4, 1))
    with ox.no_grad():
        net[0].weight.assign(T(np.eye(4)))
        net[0].bias.assign(T(np.zeros(4)))
    with pytest.raises(dg.AnomalyError) as e:
        with dg.detect_anomaly(net):
            net(T([[0.0, 1.0, 2.0, 3.0]])).sum().backward()
    assert e.value.module == "1" and e.value.phase == "backward"


def test_anomaly_in_the_loss_is_reported_as_the_loss():
    net = Net()
    with pytest.raises(dg.AnomalyError) as e:
        with dg.detect_anomaly(net):
            out = net(T(np.ones((2, 4))))
            (out * 0.0).sqrt().sum().backward()  # d sqrt / dx at 0 is inf
    assert e.value.module == "<loss>"


def test_anomaly_clean_model_and_bad_parameters():
    net = Net()
    with dg.detect_anomaly(net):
        mse(net(T(np.ones((3, 4)))), T(np.zeros((3, 3)))).backward()
    assert all(p.grad is not None for p in net.parameters())
    with ox.no_grad():
        net.fc2.bias.assign(T([np.nan, 0, 0]))
    with pytest.raises(dg.AnomalyError) as e:
        with dg.detect_anomaly(net):
            pass
    assert e.value.phase == "parameters"


# ---- precision_audit
def test_precision_audit_finds_the_lossy_layer_and_restores_weights():
    class Model(ox.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = ox.nn.Linear(32, 32)
            self.exp = Exp()

        def forward(self, x):
            return self.exp(self.fc(x) * 40.0)

    class Exp(ox.nn.Module):
        def forward(self, x):
            return x.exp()

    m = Model()
    w = m.fc.weight.to_numpy().copy()
    r = dg.precision_audit(m, T(np.random.default_rng(1).standard_normal((4, 32))))
    assert [row.name for row in r.rows] == ["fc", "exp"]
    assert m.fc.weight.dtype == "f32" and np.array_equal(m.fc.weight.to_numpy(), w)
    exp_row = r.rows[1]
    assert exp_row.total_error > r.rows[0].total_error  # exp of large values amplifies the error of its input
    assert r.output_error == pytest.approx(exp_row.total_error)
    assert "exp" in str(r)


# ---- estimate_memory
def test_estimate_memory_counts():
    m = Net()
    n_params = 4 * 8 + 8 + 8 * 3 + 3
    est = dg.estimate_memory(m, T(np.zeros((32, 4))), optimizer="adam")
    assert est.parameters == 4 * n_params
    assert est.gradients == 4 * n_params
    assert est.optimizer_state == 2 * 4 * n_params
    assert est.activations == 4 * (32 * 4 + 32 * 8 + 32 * 8 + 32 * 3)
    assert est.total == est.parameters + est.gradients + est.optimizer_state + est.activations
    assert dg.estimate_memory(m, T(np.zeros((32, 4))), optimizer="sgd").optimizer_state == 0
    with pytest.raises(ValueError):
        dg.estimate_memory(m, T(np.zeros((1, 4))), optimizer="magic")


# ---- lr_find
def test_lr_find_restores_weights_and_suggests_a_useful_rate():
    rng = np.random.default_rng(1)
    w_true = rng.standard_normal((4, 3)).astype(np.float32)
    data = []
    for _ in range(20):
        xb = rng.standard_normal((64, 4)).astype(np.float32)
        data.append((T(xb), T(xb @ w_true)))
    ox.manual_seed(0)
    m = ox.nn.Sequential(ox.nn.Linear(4, 16), ox.nn.ReLU(), ox.nn.Linear(16, 3))
    before = m[0].weight.to_numpy().copy()
    r = dg.lr_find(m, mse, data, num_iter=80)
    assert np.array_equal(before, m[0].weight.to_numpy())
    assert len(r.lrs) == len(r.losses) and r.suggestion is not None and r.valley is not None
    assert 1e-4 < r.suggestion < 1.0

    def final_loss(lr):
        ox.manual_seed(5)
        mm = ox.nn.Sequential(ox.nn.Linear(4, 16), ox.nn.ReLU(), ox.nn.Linear(16, 3))
        opt = ox.optim.Adam(list(mm.parameters()), lr=lr)
        for i in range(200):
            xb, yb = data[i % 20]
            opt.zero_grad()
            loss = mse(mm(xb), yb)
            loss.backward()
            opt.step()
        return loss.item()

    assert final_loss(r.suggestion) < final_loss(r.suggestion / 100)


# ---- check_determinism
def test_check_determinism():
    def train():
        m = Net()
        opt = ox.optim.SGD(list(m.parameters()), lr=0.1)
        for _ in range(3):
            opt.zero_grad()
            mse(m(ox.randn([8, 4])), ox.zeros([8, 3])).backward()
            opt.step()
        return m

    assert dg.check_determinism(train)
    calls = []

    def leaky():
        calls.append(1)
        return {"a": T([1.0, 2.0]), "b": T([float(len(calls)), 0.0])}

    r = dg.check_determinism(leaky)
    assert not r and r.first_difference == "b" and r.differing_elements == 1 and r.max_abs_difference == 1.0
    with pytest.raises(ValueError):
        dg.check_determinism(train, runs=1)


# ---- diff_checkpoints
def test_diff_checkpoints(tmp_path):
    a, b = Net(), Net()
    assert dg.diff_checkpoints(a, a).identical
    d = dg.diff_checkpoints(a, b)
    assert len(d.changed) == 4 and not d.unchanged
    sa = dict(a.state_dict())
    sb = dict(sa)
    sb["fc1.bias"] = T(sa["fc1.bias"].to_numpy() + 1.0)
    sb["extra"] = T([1.0])
    del sb["fc2.bias"]
    d = dg.diff_checkpoints(sa, sb)
    assert d.only_in_a == ["fc2.bias"] and d.only_in_b == ["extra"]
    assert [k for k, _, _ in d.changed] == ["fc1.bias"] and d.changed[0][2] == pytest.approx(1.0)
    assert dg.diff_checkpoints(sa, sb, atol=2.0).changed == []
    path = tmp_path / "a.oxs"
    ox.save(a.state_dict(), str(path))
    assert dg.diff_checkpoints(str(path), a).identical


# ---- TrainingMonitor
def test_training_monitor_records_and_warns():
    m = Net()
    opt = ox.optim.SGD(list(m.parameters()), lr=0.01)
    mon = dg.TrainingMonitor(m)
    x, y = ox.randn([16, 4]), ox.randn([16, 3])
    for _ in range(2):
        with mon.watch():
            loss = mse(m(x), y)
        loss.backward()
        opt.step()
        rec = mon.step(loss)
        opt.zero_grad()
    assert set(rec["params"]) == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
    assert all(s["grad_norm"] > 0 and s["update_ratio"] > 0 for s in rec["params"].values())
    assert set(rec["activations"]) == {"fc1", "act", "fc2"}
    assert "fc1.weight" in mon.report()
    # a huge learning rate makes updates far larger than the weights
    big = ox.optim.SGD(list(m.parameters()), lr=1e4)
    loss = mse(m(x), y)
    loss.backward()
    big.step()
    mon.step(loss)
    assert any("update is" in w for w in mon.warnings())


def test_training_monitor_dead_relu_warning():
    m = Net()
    with ox.no_grad():
        m.fc1.bias.assign(T(np.full(8, -100.0)))  # every unit negative -> ReLU outputs 0
    mon = dg.TrainingMonitor(m)
    with mon.watch():
        loss = mse(m(ox.randn([16, 4])), ox.zeros([16, 3]))
    loss.backward()
    mon.step(loss)
    assert any("dead" in w for w in mon.warnings())


# ---- sharpness
class Quadratic(ox.nn.Module):
    def __init__(self, diag):
        super().__init__()
        self.w = T(np.ones(len(diag))).requires_grad_(True)
        self.diag = np.asarray(diag, dtype=np.float32)

    def forward(self, _):
        a = T(np.diag(self.diag))
        n = len(self.diag)
        return 0.5 * self.w.reshape([1, n]).matmul(a).matmul(self.w.reshape([n, 1])).sum()


def test_hessian_top_eigenvalue_and_trace_on_a_known_hessian():
    q = Quadratic([3.0, 7.0, 1.0, 2.0])
    assert dg.hessian_top_eigenvalue(q, lambda o: o, None, iters=100, tol=1e-7) == pytest.approx(7.0, rel=1e-3)
    assert dg.hessian_trace(q, lambda o: o, None, samples=5) == pytest.approx(13.0, rel=1e-5)  # exact for a diagonal H


def test_sharpness_runs_on_a_network():
    m = Net()
    x, y = ox.randn([8, 4]), ox.randn([8, 3])
    eig = dg.hessian_top_eigenvalue(m, mse, x, y, iters=10)
    assert math.isfinite(eig) and eig > 0
    assert all(p.grad is None for p in m.parameters())  # no side effects on .grad
