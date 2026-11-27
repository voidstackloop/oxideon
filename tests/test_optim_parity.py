"""Optimizer / scheduler parity with PyTorch: parameter groups, live hyper-parameter edits, momentum-cycling schedulers."""
import numpy as np
import pytest

import oxideon as ox

torch = pytest.importorskip("torch")
rng = np.random.default_rng(3)
A0, B0, C = (rng.standard_normal(s).astype(np.float32) for s in ((3, 2), (2,), (3, 2)))


def tensors(lib):
    if lib == "ox":
        return ox.from_numpy(A0.copy()).requires_grad_(True), ox.from_numpy(B0.copy()).requires_grad_(True), ox.from_numpy(C)
    return torch.tensor(A0.copy(), requires_grad=True), torch.tensor(B0.copy(), requires_grad=True), torch.tensor(C)


def values(t):
    return t.to_numpy() if lib_of(t) == "ox" else t.detach().numpy()


def lib_of(t):
    return "ox" if isinstance(t, ox.Tensor) else "torch"


def train(lib, name, build, steps=6, hook=None, sched=None):
    a, b, c = tensors(lib)
    P = ox.optim if lib == "ox" else torch.optim
    opt = build(getattr(P, name), a, b)
    sc = sched(P, opt) if sched else None
    for i in range(steps):
        if hook:
            hook(opt, i)
        opt.zero_grad()
        loss = ((a * c) ** 2).sum() + (a * c).sum() + (b ** 3).sum() * 0.1 + b.sum()
        loss.backward()
        opt.step()
        if sc:
            sc.step()
    return values(a), values(b), [g["lr"] for g in opt.param_groups]


def same(name, build, **kw):
    got, want = train("ox", name, build, **kw), train("torch", name, build, **kw)
    for x, y in zip(got[:2], want[:2]):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got[2], want[2], rtol=1e-6)


def lr_decay(opt, i):
    for g in opt.param_groups:
        g["lr"] = 0.1 / (1 + i)


def momentum_ramp(opt, i):
    for g in opt.param_groups:
        if "momentum" in g:
            g["momentum"] = 0.9 - 0.1 * i
        else:
            g["betas"] = (0.9 - 0.1 * i, 0.99)


CASES = [("SGD", dict(momentum=0.9)), ("SGD", dict(momentum=0.9, nesterov=True, weight_decay=0.1)), ("Adam", {}), ("AdamW", dict(amsgrad=True))]


@pytest.mark.parametrize("name,kw", CASES)
def test_parameter_groups_with_their_own_hyperparameters(name, kw):
    same(name, lambda P, a, b: P([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01, "weight_decay": 0.5}], **kw))

    def add(P, a, b):
        o = P([a], lr=0.1, **kw)
        o.add_param_group({"params": [b], "lr": 0.02})
        return o
    same(name, add)


@pytest.mark.parametrize("name,kw", CASES)
def test_editing_param_groups_changes_the_next_step(name, kw):
    same(name, lambda P, a, b: P([a, b], lr=0.1, **kw), hook=lr_decay)
    same(name, lambda P, a, b: P([a, b], lr=0.05, **kw), hook=momentum_ramp)


@pytest.mark.parametrize("name,kw", [("SGD", dict(momentum=0.9)), ("Adam", {})])
def test_scheduler_over_two_groups(name, kw):
    same(name, lambda P, a, b: P([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01}], **kw),
         sched=lambda P, o: P.lr_scheduler.StepLR(o, step_size=2, gamma=0.5))


@pytest.mark.parametrize("name,okw", [("SGD", dict(lr=0.05, momentum=0.9)), ("Adam", dict(lr=0.05))])
@pytest.mark.parametrize("skw", [
    ("OneCycleLR", dict(max_lr=0.1, total_steps=8)), ("OneCycleLR", dict(max_lr=0.1, total_steps=8, anneal_strategy="linear", three_phase=True)),
    ("OneCycleLR", dict(max_lr=0.1, epochs=2, steps_per_epoch=4, cycle_momentum=False)),
    ("CyclicLR", dict(base_lr=0.01, max_lr=0.1, step_size_up=2)), ("CyclicLR", dict(base_lr=0.01, max_lr=0.1, step_size_up=2, step_size_down=3, mode="triangular2")),
    ("CyclicLR", dict(base_lr=0.01, max_lr=0.1, step_size_up=2, mode="exp_range", gamma=0.9)),
])
def test_one_cycle_and_cyclic_cycle_the_momentum_like_torch(name, okw, skw):
    same(name, lambda P, a, b: P([a, b], **okw), steps=8, sched=lambda P, o: getattr(P.lr_scheduler, skw[0])(o, **skw[1]))


@pytest.mark.parametrize("name", ["NAdam", "RAdam"])
def test_decoupled_weight_decay(name):
    same(name, lambda P, a, b: P([a, b], lr=0.05, weight_decay=0.1, decoupled_weight_decay=True))


@pytest.mark.parametrize("name", ["SGD", "Adam", "AdamW"])
@pytest.mark.parametrize("groups", [False, True])
def test_state_dict_resume_is_exact(name, groups):
    def build(a, b):
        cls = getattr(ox.optim, name)
        return cls([{"params": [a], "lr": 0.1}, {"params": [b], "lr": 0.01}]) if groups else cls([a, b], lr=0.05)

    def run(opt, a, b, n):
        for _ in range(n):
            opt.zero_grad()
            (((a * ox.from_numpy(C)) ** 2).sum() + (b ** 3).sum()).backward()
            opt.step()
    a, b, _ = tensors("ox")
    o = build(a, b)
    run(o, a, b, 3)
    a2, b2 = ox.from_numpy(a.to_numpy().copy()).requires_grad_(True), ox.from_numpy(b.to_numpy().copy()).requires_grad_(True)
    o2 = build(a2, b2)
    o2.load_state_dict(o.state_dict())
    run(o, a, b, 3)
    run(o2, a2, b2, 3)
    np.testing.assert_array_equal(a.to_numpy(), a2.to_numpy())
    np.testing.assert_array_equal(b.to_numpy(), b2.to_numpy())


def test_cycle_momentum_requires_a_momentum_optimizer():
    a, _, _ = tensors("ox")
    with pytest.raises(ValueError, match="momentum"):
        ox.optim.OneCycleLR(ox.optim.Adagrad([a], lr=0.1), max_lr=0.1, total_steps=4)
