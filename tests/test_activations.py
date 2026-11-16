"""Z21: activations vs torch ground truth.

Covers the newly bound activations (relu6, hardtanh, log_sigmoid,
softshrink, hardshrink, elu, erf), f64 exactness of erf/GELU after the
libm upgrade, softplus(beta) stability on large inputs, and a gradient
smoke test proving the new bindings stay differentiable.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import oxideon as ox

T = ox.from_numpy

XS = np.array([-30.0, -8.0, -2.0, -0.5, -0.0, 0.0, 0.5, 2.0, 8.0, 30.0])


def check(ox_fn, th_fn, dtype=np.float32, rtol=1e-6, atol=1e-7):
    x = XS.astype(dtype)
    got = ox_fn(T(x)).to_numpy()
    want = th_fn(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


def test_relu6_matches_torch():
    check(lambda t: t.relu6(), F.relu6)


def test_hardtanh_matches_torch_default_and_custom():
    check(lambda t: t.hardtanh(), F.hardtanh)
    check(lambda t: t.hardtanh(-0.5, 2.0),
          lambda t: F.hardtanh(t, min_val=-0.5, max_val=2.0))


def test_log_sigmoid_matches_torch():
    check(lambda t: t.log_sigmoid(), F.logsigmoid)


def test_softshrink_matches_torch_default_and_custom():
    check(lambda t: t.softshrink(), F.softshrink)
    check(lambda t: t.softshrink(1.0), lambda t: F.softshrink(t, lambd=1.0))


def test_hardshrink_matches_torch_default_and_custom():
    check(lambda t: t.hardshrink(), F.hardshrink)
    check(lambda t: t.hardshrink(1.0), lambda t: F.hardshrink(t, lambd=1.0))


def test_elu_matches_torch_default_and_custom():
    check(lambda t: t.elu(), F.elu)
    check(lambda t: t.elu(2.0), lambda t: F.elu(t, alpha=2.0))


def test_erf_matches_torch_f32_and_f64_exact():
    check(lambda t: t.erf(), torch.special.erf)
    check(lambda t: t.erf(), torch.special.erf,
          dtype=np.float64, rtol=1e-12, atol=1e-12)


def test_gelu_exact_f64_after_libm_upgrade():
    check(lambda t: t.gelu(), lambda t: F.gelu(t, approximate="none"),
          dtype=np.float64, rtol=1e-12, atol=1e-12)
    # Deliberate divergence: the f32 kernel evaluates erf in f64 internally,
    # so at x=-2 it lands ~500x closer to the true value (-0.04550026389...)
    # than torch's pure-f32 gelu (-0.04550010). Tolerance covers torch's
    # own f32 noise, not ours.
    check(lambda t: t.gelu(), lambda t: F.gelu(t, approximate="none"),
          rtol=2e-5, atol=1e-6)


def test_softplus_beta_matches_torch_and_stays_finite():
    check(lambda t: t.softplus(2.0), lambda t: F.softplus(t, beta=2.0))
    check(lambda t: t.softplus(), lambda t: F.softplus(t))
    big = np.array([100.0, 500.0, 1000.0], dtype=np.float32)
    got = T(big).softplus(1.0).to_numpy()
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, F.softplus(torch.from_numpy(big), beta=1.0).numpy(),
                               rtol=1e-6)


def test_new_activations_stay_differentiable():
    x = np.array([-1.5, -0.25, 0.75, 3.0], dtype=np.float32)
    for name, kw in [("relu6", {}), ("hardtanh", {}), ("log_sigmoid", {}),
                     ("softshrink", {}), ("hardshrink", {}),
                     ("elu", {"alpha": 1.5}), ("erf", {})]:
        xr = T(x).requires_grad_(True)
        getattr(xr, name)(**kw).sum().backward()
        g = xr.grad.numpy()
        assert np.all(np.isfinite(g)), name
    # ELU gradient has a closed form: 1 for x > 0, alpha*exp(x) for x <= 0.
    xr = T(x).requires_grad_(True)
    xr.elu(1.5).sum().backward()
    want = np.where(x > 0, 1.0, 1.5 * np.exp(x))
    np.testing.assert_allclose(xr.grad.numpy(), want, rtol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
