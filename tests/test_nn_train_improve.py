"""NN/training hardening: scheduler torch-parity, loss-module reductions,
BCE weight placement, and module state_dict round-trips."""

import math

import numpy as np
import pytest

import oxideon as ox
import oxideon.nn as nn
import oxideon.optim as optim
from oxideon.optim import lr_scheduler as lrs


def _sgd(lr=0.1):
    return optim.SGD([ox.from_numpy(np.ones(4, dtype=np.float32))], lr=lr)


def _cos_warmrestarts(base_lr, eta_min, t_cur, t_i):
    return eta_min + (base_lr - eta_min) * (1.0 + math.cos(math.pi * t_cur / t_i)) / 2.0


def test_warm_restarts_matches_torch_sequence():
    s = lrs.CosineAnnealingWarmRestarts(_sgd(0.1), T_0=4, T_mult=1, eta_min=0.0)
    # torch: construction leaves last_epoch=0, T_cur=0 -> base_lr.
    assert s.T_cur == 0
    assert s.get_last_lr()[0] == pytest.approx(0.1)
    for step in range(1, 9):
        s.step()
        assert s.get_last_lr()[0] == pytest.approx(
            _cos_warmrestarts(0.1, 0.0, step % 4, 4)
        ), f"step {step}"


def test_warm_restarts_explicit_epoch_with_t_mult():
    s = lrs.CosineAnnealingWarmRestarts(_sgd(0.1), T_0=4, T_mult=2, eta_min=0.0)
    s.step(6)  # Second cycle: T_cur=2, T_i=8 (torch closed form).
    assert s.T_cur == 2
    assert s.T_i == 8
    assert s.get_last_lr()[0] == pytest.approx(_cos_warmrestarts(0.1, 0.0, 2, 8))
    s.step(2)  # First cycle still exact.
    assert s.T_cur == 2
    assert s.T_i == 4
    with pytest.raises(ValueError, match="non-negative"):
        s.step(-1)


def test_elementwise_loss_modules_honor_sum_reduction():
    x = ox.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    y = ox.from_numpy(np.array([[1.5, 1.0], [3.0, 5.0]], dtype=np.float32))
    for cls, kw in [
        (nn.MSELoss, {}),
        (nn.L1Loss, {}),
        (nn.HuberLoss, {"delta": 1.0}),
    ]:
        mean = cls(**kw)(x, y).item()
        total = cls(reduction="sum", **kw)(x, y).item()
        assert total == pytest.approx(mean * 4), cls.__name__
    # BCE needs 0/1 targets.
    logits = ox.from_numpy(np.array([[0.5, -1.0], [2.0, 0.0]], dtype=np.float32))
    bits = ox.from_numpy(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    mean = nn.BCEWithLogitsLoss()(logits, bits).item()
    total = nn.BCEWithLogitsLoss(reduction="sum")(logits, bits).item()
    assert total == pytest.approx(mean * 4)
    # The sum rescale must preserve autograd (d sum(x-y)^2 / dx = 2(x-y)).
    xr = ox.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)).requires_grad_(
        True
    )
    nn.MSELoss(reduction="sum")(xr, y).backward()
    assert np.allclose(
        xr.grad.numpy(), 2.0 * (np.array([[1.0, 2.0], [3.0, 4.0]]) - y.numpy())
    )


def test_batch_mean_loss_modules_honor_sum_reduction():
    logits = ox.from_numpy(
        np.array([[2.0, 1.0, 0.5], [0.5, 3.0, 1.0]], dtype=np.float32)
    )
    labels = ox.from_numpy(np.array([0, 1], dtype=np.int64))
    mean = nn.CrossEntropyLoss()(logits, labels).item()
    total = nn.CrossEntropyLoss(reduction="sum")(logits, labels).item()
    assert total == pytest.approx(mean * 2)
    log_probs = ox.from_numpy(
        np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32))
    )
    mean = nn.NLLLoss()(log_probs, labels).item()
    total = nn.NLLLoss(reduction="sum")(log_probs, labels).item()
    assert total == pytest.approx(mean * 2)
    probs = ox.from_numpy(np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32))
    mean = nn.KLDivLoss()(log_probs, probs).item()
    total = nn.KLDivLoss(reduction="sum")(log_probs, probs).item()
    assert total == pytest.approx(mean * 2)


def test_loss_modules_reject_bad_reduction_loudly():
    x = ox.from_numpy(np.ones((2, 2), dtype=np.float32))
    y = ox.from_numpy(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="reduction"):
        nn.MSELoss(reduction="bogus")(x, y)
    # Native kernels return reduced scalars, so 'none' cannot be honored.
    with pytest.raises(ValueError, match="reduction"):
        nn.MSELoss(reduction="none")(x, y)
    # Mean-only modules must not silently swallow a 'sum' request either.
    with pytest.raises(ValueError, match="reduction"):
        nn.TripletMarginLoss(reduction="sum")(x, y, x)


def test_bce_loss_applies_tensor_weight_before_reduction():
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(0)
    inp = rng.rand(4, 3).astype(np.float32)
    tgt = (rng.rand(4, 3) > 0.5).astype(np.float32)
    w = rng.rand(4, 3).astype(np.float32) + 0.5
    got = (
        nn.BCELoss(weight=ox.from_numpy(w))(ox.from_numpy(inp), ox.from_numpy(tgt))
        .numpy()
        .item()
    )
    want = torch.nn.functional.binary_cross_entropy(
        torch.from_numpy(inp), torch.from_numpy(tgt), weight=torch.from_numpy(w)
    ).item()
    assert got == pytest.approx(want, rel=1e-4)


def test_module_state_dict_roundtrip_and_optimizer_identity():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)
            self.register_buffer("scale", ox.from_numpy(np.ones(2, dtype=np.float32)))

        def forward(self, x):
            return self.fc(x) * self.scale

    m = Tiny()
    handles = m.parameters()
    names = [n for n, _ in m.named_parameters()]
    opt = optim.RMSprop(handles, lr=0.01)
    sd = m.state_dict()
    # Perturb params and buffers, then restore: values must match, and the
    # very same handle objects the optimizer holds must show restored values
    # (in-place copy, shared storage).
    for p in handles:
        p.copy_(p * 0.0)
    m.scale.copy_(m.scale * 0.0)
    m.load_state_dict(sd)
    after = m.state_dict()
    for k in sd:
        assert np.allclose(after[k].numpy(), sd[k].numpy()), k
    for n, p in zip(names, handles):
        assert np.allclose(p.numpy(), sd[n].numpy()), n
    assert [id(p) for g in opt.param_groups for p in g["params"]] == [
        id(p) for p in handles
    ]
