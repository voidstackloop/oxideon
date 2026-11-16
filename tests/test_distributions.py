"""Z13: distributions (native `oxideon.oxideon` engine) vs torch ground truth.

NOTE: top-level `RustNormal` is the numpy-backed `oxideon.distributions` shim
module (18 distribution types, no autograd); these tests pin the NATIVE Rust
engine classes (`oxideon.oxideon.Normal/...`) — the only differentiable
distribution path in Python — covering closed-form values, torch
broadcasting/batching shapes, Gumbel-Softmax straight-through gradients, and
fail-closed validation.
"""
import numpy as np
import pytest
import torch

import oxideon as ox
from oxideon.oxideon import Categorical as RustCategorical
from oxideon.oxideon import Laplace as RustLaplace
from oxideon.oxideon import Normal as RustNormal

T = ox.from_numpy


def test_normal_matches_torch_values_and_broadcast():
    loc = np.array([0.0, 1.0, -2.0])
    d = RustNormal(T(loc), T(np.array(2.0)))
    th = torch.distributions.Normal(torch.from_numpy(loc), torch.tensor(2.0))
    v = np.array([0.5, -1.0, 3.0])
    np.testing.assert_allclose(d.log_prob(T(v)).to_numpy(),
                               th.log_prob(torch.from_numpy(v)).numpy(), rtol=1e-6)
    np.testing.assert_allclose(d.entropy().to_numpy(), th.entropy().numpy(), rtol=1e-6)
    assert d.entropy().to_numpy().shape == (3,)
    assert d.rsample().to_numpy().shape == (3,)
    p = RustNormal(T(np.array([1.0])), T(np.array([2.0])))
    q = RustNormal(T(np.array([0.0])), T(np.array([1.0])))
    thp = torch.distributions.Normal(torch.tensor([1.0]), torch.tensor([2.0]))
    thq = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    np.testing.assert_allclose(p.kl_divergence(q).to_numpy(),
                               torch.distributions.kl_divergence(thp, thq).numpy(),
                               rtol=1e-6)


def test_laplace_matches_torch_values():
    d = RustLaplace(T(np.array([0.0, 1.0])), T(np.array([1.0, 0.5])))
    th = torch.distributions.Laplace(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.5]))
    v = np.array([0.0, 2.0])
    np.testing.assert_allclose(d.log_prob(T(v)).to_numpy(),
                               th.log_prob(torch.from_numpy(v)).numpy(), rtol=1e-6)
    np.testing.assert_allclose(d.entropy().to_numpy(), th.entropy().numpy(), rtol=1e-6)
    assert d.rsample().to_numpy().shape == (2,)


def test_categorical_matches_torch():
    logits = np.array([[1.0, 2.5, -0.5], [0.2, -1.0, 1.8]])
    c = RustCategorical(T(logits))
    th = torch.distributions.Categorical(logits=torch.from_numpy(logits))
    np.testing.assert_allclose(c.probs().to_numpy(), th.probs.numpy(), rtol=1e-6)
    np.testing.assert_allclose(c.entropy().to_numpy(), th.entropy().numpy(), rtol=1e-6)
    idx = np.array([2, 0], dtype=np.int64)
    np.testing.assert_allclose(c.log_prob(T(idx)).to_numpy(),
                               th.log_prob(torch.from_numpy(idx)).numpy(), rtol=1e-6)
    s = c.sample().to_numpy()
    assert s.shape == (2,)
    assert set(s.tolist()) <= {0, 1, 2}
    # 1D logits sample a scalar like torch.
    c1 = RustCategorical(T(np.array([1.0, 2.0])))
    assert c1.sample().to_numpy().shape == ()
    th1 = torch.distributions.Categorical(logits=torch.tensor([1.0, 2.0]))
    assert tuple(th1.sample().shape) == ()


def test_gumbel_softmax_sums_and_straight_through():
    logits = T(np.array([[1.0, 2.5, -0.5], [0.2, -1.0, 1.8]])).requires_grad_(True)
    c = RustCategorical(logits)
    soft = c.gumbel_softmax(0.5, False).to_numpy()
    np.testing.assert_allclose(soft.sum(axis=-1), [1.0, 1.0], rtol=1e-6)
    hard = c.gumbel_softmax(0.5, True)
    hv = hard.to_numpy()
    np.testing.assert_allclose(hv.sum(axis=-1), [1.0, 1.0], rtol=1e-6)
    assert set(np.unique(hv).tolist()) <= {0.0, 1.0}
    # Weighted (not plain) sum: grad of sum(softmax) is identically zero, while
    # distinct weights provably keep at least two grad entries nonzero.
    w = T(np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]))
    (hard * w).sum().backward()
    assert logits.grad is not None
    assert float(np.abs(logits.grad.to_numpy()).sum()) > 0.0


def test_distributions_reject_bad_params():
    loc = T(np.zeros(3))
    with pytest.raises(Exception, match="positive"):
        RustNormal(loc, T(np.zeros(3)))
    with pytest.raises(Exception, match="positive"):
        RustLaplace(loc, T(np.array([1.0, -1.0, 1.0])))
    with pytest.raises(Exception, match="broadcast|shape|mismatch|incompatible"):
        RustNormal(loc, T(np.ones(2)))
    c = RustCategorical(T(np.zeros((2, 4))))
    for bad in [0.0, -1.0, float("nan"), float("inf")]:
        with pytest.raises(Exception, match="temperature"):
            c.gumbel_softmax(bad, False)
