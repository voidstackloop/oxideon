"""Z23: oxideon.special vs torch ground truth (values + domain edges).

The numpy-backed torch.special shims had fail-open clamps (erfinv, gammaln,
ndtri, lambert_w), a garbage digamma for x<=0, truncated polygamma/zeta
series, a broken bessel_j1 small branch, and weak large-x Bessel/I1
approximations. Interior behavior of previously-OK functions is pinned.
"""
import math

import numpy as np
import pytest
import torch

import oxideon as ox
from oxideon import special as S

T = ox.from_numpy
th = torch.from_numpy


def check(name, got, want, rtol=1e-6, atol=1e-7):
    np.testing.assert_allclose(np.asarray(got, dtype=np.float64),
                               np.asarray(want, dtype=np.float64),
                               rtol=rtol, atol=atol, equal_nan=True)


def test_erf_family_interior():
    x = np.array([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0, 8.0])
    check("erf", S.erf(T(x)).to_numpy(), torch.special.erf(th(x)).numpy())
    check("erfc", S.erfc(T(x)).to_numpy(), torch.special.erfc(th(x)).numpy())


def test_erfinv_matches_torch_values_and_domain():
    y = np.array([-0.999, -0.5, 0.0, 0.5, 0.999], dtype=np.float64)
    check("erfinv", S.erfinv(T(y)).to_numpy(), torch.special.erfinv(th(y)).numpy(),
          rtol=1e-9, atol=1e-10)
    e = np.array([-2.0, -1.0, 1.0, 2.0])
    check("erfinv-edges", S.erfinv(T(e)).to_numpy(),
          torch.special.erfinv(th(e)).numpy())


def test_exp_family():
    x = np.array([-3.0, -0.5, 0.0, 0.5, 2.0, 8.0])
    check("exp2", S.exp2(T(x)).to_numpy(), torch.special.exp2(th(x)).numpy())
    check("expm1", S.expm1(T(x)).to_numpy(), torch.special.expm1(th(x)).numpy())
    check("expit", S.expit(T(x)).to_numpy(), torch.special.expit(th(x)).numpy())


def test_logit_with_and_without_eps():
    x = np.array([-0.5, 0.0, 0.1, 0.5, 0.9, 1.0, 1.5])
    check("logit", S.logit(T(x)).to_numpy(), torch.special.logit(th(x)).numpy())
    check("logit-eps", S.logit(T(x), eps=0.1).to_numpy(),
          torch.special.logit(th(x), eps=0.1).numpy())


def test_gammaln_negative_domain_via_reflection():
    g = np.array([-2.5, -2.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    check("gammaln", S.gammaln(T(g)).to_numpy(), torch.special.gammaln(th(g)).numpy())


def test_digamma_negative_domain_poles_and_reflection():
    g = np.array([-2.5, -2.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    check("digamma", S.digamma(T(g)).to_numpy(),
          torch.special.digamma(th(g)).numpy(), rtol=1e-9, atol=1e-9)


def test_polygamma_tail_correction():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    for n in (0, 1, 2, 3):
        check(f"polygamma{n}", S.polygamma(n, T(x)).to_numpy(),
              torch.special.polygamma(n, th(x)).numpy(), rtol=1e-6, atol=1e-8)
    a = np.array([1.5, 2.0, 3.0])
    check("multigammaln", S.multigammaln(T(a), 2).to_numpy(),
          torch.special.multigammaln(th(a), 2).numpy())
    bad = np.array([0.5, 1.0])
    check("multigammaln-invalid", S.multigammaln(T(bad), 3).to_numpy(),
          torch.special.multigammaln(th(bad), 3).numpy())


def test_bessel_i_family():
    b = np.array([-5.0, -1.0, 0.0, 1.0, 3.0, 5.0, 8.0, 15.0])
    check("i0", S.i0(T(b)).to_numpy(), torch.special.i0(th(b)).numpy())
    check("i0e", S.i0e(T(b)).to_numpy(), torch.special.i0e(th(b)).numpy())
    check("i1", S.i1(T(b)).to_numpy(), torch.special.i1(th(b)).numpy())
    check("i1e", S.i1e(T(b)).to_numpy(), torch.special.i1e(th(b)).numpy())


def test_ndtr_ndtri_values_domain_and_f64():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 10.0])
    check("ndtr", S.ndtr(T(x)).to_numpy(), torch.special.ndtr(th(x)).numpy())
    check("log_ndtr", S.log_ndtr(T(x)).to_numpy(),
          torch.special.log_ndtr(th(x)).numpy())
    p = np.array([-0.1, 0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0, 1.1])
    check("ndtri", S.ndtri(T(p)).to_numpy(), torch.special.ndtri(th(p)).numpy(),
          rtol=1e-9, atol=1e-10)
    f = np.array([0.25, 0.75], dtype=np.float64)
    check("ndtri-f64", S.ndtri(T(f)).to_numpy(),
          torch.special.ndtri(th(f)).numpy(), rtol=1e-12, atol=1e-13)


def test_entr_torch_negative_convention():
    x = np.array([-0.5, 0.0, 0.1, 0.5, 1.0])
    check("entr", S.entr(T(x)).to_numpy(), torch.special.entr(th(x)).numpy())


def test_rel_entr_matches_xlogy_composition():
    x = np.array([0.0, 0.5, 1.0, 2.0, 1.0, -1.0])
    y = np.array([0.0, 1.0, 0.5, 0.0, -1.0, 1.0])
    want = (torch.special.xlogy(th(x), th(x)) - torch.special.xlogy(th(x), th(y))).numpy()
    check("rel_entr", S.rel_entr(T(x), T(y)).to_numpy(), want)


def test_zeta_with_tail_correction():
    z = np.array([1.5, 2.0, 3.0, 5.0])
    check("zeta", S.zeta(T(z)).to_numpy(),
          torch.special.zeta(th(z), th(np.ones(4))).numpy(), rtol=1e-6)


def test_bessel_j_miller_accuracy():
    j = np.array([-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, 15.0, 30.0])
    # Deliberate divergence: Miller agrees with the Taylor series to 1e-15
    # at x=8 (J0=0.17165080713755, J1=0.23463634685391) while torch's cephes
    # port reads ~1e-8 high there; tolerance covers torch's noise, not ours.
    check("bessel_j0", S.bessel_j0(T(j)).to_numpy(),
          torch.special.bessel_j0(th(j)).numpy(), rtol=2e-7, atol=1e-8)
    check("bessel_j1", S.bessel_j1(T(j)).to_numpy(),
          torch.special.bessel_j1(th(j)).numpy(), rtol=2e-7, atol=1e-8)
    check("bessel_j0(8)", S.bessel_j0(T(np.array([8.0]))).to_numpy(),
          np.array([0.17165080713755]), rtol=1e-12, atol=0.0)
    check("bessel_j1(8)", S.bessel_j1(T(np.array([8.0]))).to_numpy(),
          np.array([0.23463634685391]), rtol=1e-12, atol=0.0)


def test_lambert_w_identities_and_domain():
    # Exact identities: W(x)e^W(x) = x pins the value without a reference lib.
    pts = {0.0: 0.0, math.e: 1.0, 1.0: 0.5671432904097838,
           -math.log(2) / 2: -math.log(2), -1.0 / math.e: -1.0}
    xs = np.array(sorted(pts))
    got = S.lambert_w(T(xs)).to_numpy()
    for x, w in zip(xs, got):
        assert abs(w * math.exp(w) - x) <= 1e-9 * max(1.0, abs(x)), (x, w)
    check("lambert_w-omega", S.lambert_w(T(np.array([1.0]))).to_numpy(),
          np.array([pts[1.0]]), rtol=1e-12, atol=0.0)
    edge = np.array([-1.0, -0.5])
    check("lambert_w-domain", S.lambert_w(T(edge)).to_numpy(),
          np.array([np.nan, np.nan]))


def test_misc_pins():
    check("sinc", S.sinc(T(np.array([-1.0, 0.0, 0.5]))).to_numpy(),
          torch.sinc(th(np.array([-1.0, 0.0, 0.5]))).numpy())
    check("round", S.round(T(np.array([-2.5, -0.5, 0.5, 2.5]))).to_numpy(),
          torch.round(th(np.array([-2.5, -0.5, 0.5, 2.5]))).numpy())
    check("log1p", S.log1p(T(np.array([0.0, 1.0, 10.0]))).to_numpy(),
          torch.log1p(th(np.array([0.0, 1.0, 10.0]))).numpy())
    from numpy.polynomial import hermite as H, legendre as Le
    hx = np.array([-1.0, 0.0, 0.5, 2.0])
    check("hermite3", S.hermite_polynomial_h(T(hx), 3).to_numpy(),
          H.hermval(hx, [0, 0, 0, 1]))
    check("legendre3", S.legendre_polynomial_p(T(hx), 3).to_numpy(),
          Le.legval(hx, [0, 0, 0, 1]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
