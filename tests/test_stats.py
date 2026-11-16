"""Tests for oxideon.stats: classical hypothesis tests and corrections.

scipy.stats is the oracle wherever it applies (skipped if unavailable);
degenerate and validation behavior is asserted directly.
"""

import math

import numpy as np
import pytest

import oxideon as ox
from oxideon.stats import (
    ECDF,
    chi2_contingency,
    chisquare,
    ecdf,
    jarque_bera,
    kstest,
    kurtosis,
    mannwhitneyu,
    p_adjust,
    pearsonr,
    skew,
    spearmanr,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
)

try:
    from scipy import stats as _ss

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

needs_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")

_rng = np.random.default_rng(42)
N1 = _rng.normal(0.0, 1.0, 50)
N2 = _rng.normal(0.5, 1.5, 40)
U1 = _rng.uniform(-1.0, 1.0, 60)
BIG = _rng.normal(0.0, 1.0, 500)
U500 = np.random.default_rng(7).uniform(-1.0, 1.0, 500)


# --- distribution-function core ---------------------------------------------


@needs_scipy
def test_private_core_matches_scipy():
    from scipy import special as _sp
    from oxideon.stats import _betainc_scalar, _chi2_sf, _gammaincc_scalar, _norm_sf, _t_sf

    assert _norm_sf(1.96) == pytest.approx(_sp.ndtr(-1.96), rel=1e-12)
    assert _chi2_sf(7.5, 4.0) == pytest.approx(_sp.chdtrc(4.0, 7.5), rel=1e-10)
    assert _t_sf(2.3, 9.0) == pytest.approx(_ss.t.sf(2.3, 9), rel=1e-10)
    assert _betainc_scalar(2.5, 3.0, 0.4) == pytest.approx(_sp.betainc(2.5, 3.0, 0.4), rel=1e-10)
    assert _gammaincc_scalar(50.0, 40.0) == pytest.approx(_sp.gammaincc(50.0, 40.0), rel=1e-10)


# --- t-tests ------------------------------------------------------------------


@needs_scipy
@pytest.mark.parametrize("alt", ["two-sided", "less", "greater"])
def test_ttest_1samp_matches_scipy(alt):
    r = ttest_1samp(N1, 0.1, alternative=alt)
    ref = _ss.ttest_1samp(N1, 0.1, alternative=alt)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert r.df == 49.0


def test_ttest_1samp_degenerate_and_validation():
    r = ttest_1samp([1.0, 1.0, 1.0], 1.0)
    assert math.isnan(r.statistic) and math.isnan(r.pvalue)
    r = ttest_1samp([1.0, 1.0, 1.0], 0.0)
    assert r.statistic == float("inf") and r.pvalue == 0.0
    r = ttest_1samp([1.0, 1.0, 1.0], 0.0, alternative="less")
    assert r.pvalue == 1.0
    with pytest.raises(ValueError, match="at least 2"):
        ttest_1samp([1.0], 0.0)
    with pytest.raises(ValueError, match="finite"):
        ttest_1samp([1.0, float("nan"), 2.0], 0.0)
    with pytest.raises(ValueError, match="alternative"):
        ttest_1samp(N1, 0.0, alternative="bogus")
    with pytest.raises(ValueError, match="popmean"):
        ttest_1samp(N1, float("inf"))


@needs_scipy
@pytest.mark.parametrize("equal_var", [True, False])
def test_ttest_ind_matches_scipy(equal_var):
    r = ttest_ind(N1, N2, equal_var=equal_var)
    ref = _ss.ttest_ind(N1, N2, equal_var=equal_var)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert r.df == pytest.approx(float(ref.df), rel=1e-12)


def test_ttest_ind_degenerate():
    r = ttest_ind([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert math.isnan(r.statistic) and math.isnan(r.pvalue)
    r = ttest_ind([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert r.statistic == float("-inf") and r.pvalue == 0.0
    rw = ttest_ind([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], equal_var=False)
    assert rw.statistic == float("-inf") and rw.pvalue == 0.0


@needs_scipy
def test_ttest_rel_matches_scipy():
    r = ttest_rel(N1, N1 + U1[:50] * 0.1)
    ref = _ss.ttest_rel(N1, N1 + U1[:50] * 0.1)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    with pytest.raises(ValueError, match="must match"):
        ttest_rel(N1, N2)


def test_ttest_accepts_tensors():
    r = ttest_1samp(ox.tensor(N1.astype(np.float32)), 0.0)
    assert r.statistic == pytest.approx(ttest_1samp(N1, 0.0).statistic, rel=1e-6)


# --- chi-square ---------------------------------------------------------------


@needs_scipy
def test_chisquare_matches_scipy():
    obs = np.array([16.0, 18.0, 16.0, 14.0, 12.0, 12.0])
    r = chisquare(obs)
    ref = _ss.chisquare(obs)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    exp = np.array([16.0, 16.0, 16.0, 16.0, 12.0, 12.0])
    r = chisquare(obs, exp)
    ref = _ss.chisquare(obs, exp)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    r = chisquare(obs, ddof=1)
    ref = _ss.chisquare(obs, ddof=1)
    assert r.dof == 4 and r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)


def test_chisquare_validation():
    with pytest.raises(ValueError, match="non-negative"):
        chisquare([1.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        chisquare([1.0, 2.0, 3.0], [1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="ddof"):
        chisquare([1.0, 2.0, 3.0], ddof=-1)
    r = chisquare([0.0, 0.0, 0.0])
    assert math.isnan(r.statistic) and math.isnan(r.pvalue)


@needs_scipy
@pytest.mark.parametrize("correction", [True, False])
def test_chi2_contingency_matches_scipy(correction):
    tab2 = np.array([[10.0, 10.0], [20.0, 20.0]])
    r = chi2_contingency(tab2, correction=correction)
    ref = _ss.chi2_contingency(tab2, correction=correction)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert r.dof == ref.dof
    np.testing.assert_allclose(r.expected_freq, ref.expected_freq, rtol=1e-12)
    tab3 = np.array([[10.0, 20.0, 30.0], [6.0, 9.0, 17.0], [12.0, 8.0, 4.0]])
    r = chi2_contingency(tab3, correction=correction)
    ref = _ss.chi2_contingency(tab3, correction=correction)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)


def test_chi2_contingency_validation():
    with pytest.raises(ValueError, match="2-D"):
        chi2_contingency([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-negative"):
        chi2_contingency([[1.0, -1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="zero"):
        chi2_contingency([[0.0, 0.0], [1.0, 2.0]])


# --- correlations -------------------------------------------------------------


@needs_scipy
def test_pearsonr_matches_scipy():
    x = N1[:40]
    y = 2.0 * x + U1[:40]
    r = pearsonr(x, y)
    ref = _ss.pearsonr(x, y)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)


def test_pearsonr_edges():
    r = pearsonr([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert r.statistic == pytest.approx(1.0) and r.pvalue == 0.0
    r = pearsonr([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
    assert math.isnan(r.statistic) and math.isnan(r.pvalue)
    r = pearsonr([1.0, 2.0], [3.0, 5.0])
    assert math.isnan(r.pvalue)
    with pytest.raises(ValueError, match="must match"):
        pearsonr([1.0, 2.0], [1.0, 2.0, 3.0])


@needs_scipy
def test_spearmanr_matches_scipy():
    x = np.array([1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 9.0, 9.0, 10.0, 12.0])
    y = np.array([2.0, 1.0, 3.0, 3.0, 4.0, 9.0, 7.0, 8.0, 11.0, 10.0])
    r = spearmanr(x, y)
    ref = _ss.spearmanr(x, y)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    r = spearmanr([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])
    assert math.isnan(r.statistic) and math.isnan(r.pvalue)


# --- mann-whitney ---------------------------------------------------------------


@needs_scipy
@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
@pytest.mark.parametrize("continuity", [True, False])
def test_mannwhitneyu_matches_scipy(alternative, continuity):
    for x, y in ((N1, N2), (N2, N1)):
        r = mannwhitneyu(x, y, continuity=continuity, alternative=alternative)
        ref = _ss.mannwhitneyu(x, y, alternative=alternative, method="asymptotic",
                               use_continuity=continuity)
        assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
        assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)


def test_mannwhitneyu_all_ties():
    r = mannwhitneyu([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert r.statistic == pytest.approx(4.5) and r.pvalue == 1.0


# --- KS test --------------------------------------------------------------------


@needs_scipy
def test_kstest_statistic_exact_pvalue_close():
    # D is exact; the p-value is asymptotic while scipy applies a
    # finite-sample refinement (observed gap ~2% at n=500), hence the
    # loose bound.
    r = kstest(BIG)
    ref = _ss.kstest(BIG, "norm")
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=0.05)
    r = kstest(BIG, "norm", args=(0.0, 2.0))
    ref = _ss.kstest(BIG, "norm", args=(0.0, 2.0))
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-12)


def test_kstest_callable_and_validation():
    r = kstest([0.1, 0.4, 0.35, 0.8], lambda t: t)
    assert 0.0 <= r.statistic <= 1.0 and 0.0 <= r.pvalue <= 1.0
    with pytest.raises(ValueError, match="cdf"):
        kstest([0.1, 0.2], "expon")
    with pytest.raises(ValueError, match="scale"):
        kstest([0.1, 0.2], "norm", args=(0.0, -1.0))
    with pytest.raises(ValueError, match="callable"):
        kstest([0.1, 0.2], lambda t: [0.5])


# --- moments / JB / ECDF / p_adjust ----------------------------------------------


@needs_scipy
@pytest.mark.parametrize("bias", [True, False])
def test_moments_match_scipy(bias):
    assert skew(U1, bias=bias) == pytest.approx(_ss.skew(U1, bias=bias), rel=1e-12)
    assert kurtosis(U1, bias=bias) == pytest.approx(_ss.kurtosis(U1, bias=bias), rel=1e-12)
    assert kurtosis(U1, fisher=False, bias=bias) == pytest.approx(
        _ss.kurtosis(U1, fisher=False, bias=bias), rel=1e-12)


def test_moments_constant_and_minimums():
    assert math.isnan(skew([1.0, 1.0, 1.0]))
    assert math.isnan(kurtosis([1.0, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="at least 3"):
        skew([1.0, 2.0])
    with pytest.raises(ValueError, match="at least 4"):
        kurtosis([1.0, 2.0, 3.0])


@needs_scipy
def test_jarque_bera_matches_scipy():
    r = jarque_bera(BIG)
    ref = _ss.jarque_bera(BIG)
    assert r.statistic == pytest.approx(ref.statistic, rel=1e-10)
    assert r.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert jarque_bera(BIG).pvalue > 0.05
    assert jarque_bera(U500).pvalue < 1e-6
    r = jarque_bera([1.0, 1.0, 1.0, 1.0])
    assert math.isnan(r.statistic) and math.isnan(r.pvalue)


def test_ecdf_steps():
    e = ecdf([3.0, 1.0, 2.0])
    assert isinstance(e, ECDF)
    np.testing.assert_allclose(e.quantiles, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(e([0.5, 1.0, 2.5, 3.0]), [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    with pytest.raises(ValueError, match="at least 1"):
        ecdf([])


def test_p_adjust_hand_computed():
    p = np.array([0.01, 0.04, 0.03])
    rej, adj = p_adjust(p, method="holm")
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06], rtol=1e-12)
    assert rej.tolist() == [True, False, False]
    rej, adj = p_adjust(p, method="bh")
    np.testing.assert_allclose(adj, [0.03, 0.04, 0.04], rtol=1e-12)
    assert rej.tolist() == [True, True, True]
    _, adj = p_adjust(p, method="bonferroni")
    np.testing.assert_allclose(adj, [0.03, 0.12, 0.09], rtol=1e-12)
    _, adj = p_adjust(p, method="none")
    np.testing.assert_allclose(adj, p)
    _, adj = p_adjust(p, method="fdr_bh")
    np.testing.assert_allclose(adj, [0.03, 0.04, 0.04], rtol=1e-12)
    with pytest.raises(ValueError, match="method"):
        p_adjust(p, method="bogus")
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        p_adjust([0.1, 1.5])
    with pytest.raises(ValueError, match="alpha"):
        p_adjust(p, alpha=2.0)
