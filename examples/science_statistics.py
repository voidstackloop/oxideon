"""Classical statistics with ox.stats.

Covers t-tests, chi-square, correlation, non-parametric tests, moments,
ECDF, and multiple-testing correction. No SciPy at runtime.
Run: python examples/science_statistics.py
"""

import numpy as np

import oxideon as ox


def main() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(0.5, 1.0, 200)

    # --- t-tests ---
    r1 = ox.stats.ttest_1samp(a, popmean=0.0)
    print(f"ttest_1samp: t = {r1.statistic:.4f}, p = {r1.pvalue:.4f}, df = {r1.df}")
    assert r1.pvalue > 0.05  # same distribution it was drawn from

    rw = ox.stats.ttest_ind(a, b, equal_var=False)
    print(f"ttest_ind (Welch): t = {rw.statistic:.4f}, p = {rw.pvalue:.2e}")
    assert rw.pvalue < 1e-4  # means differ by 0.5 sigma

    rp = ox.stats.ttest_rel(a, a + rng.normal(0.0, 0.1, 200))
    print(f"ttest_rel: t = {rp.statistic:.4f}, p = {rp.pvalue:.4f}")
    assert rp.pvalue > 0.05

    # --- Chi-square ---
    gof = ox.stats.chisquare([16, 18, 16, 14, 12, 12])
    print(f"chisquare: chi2 = {gof.statistic:.4f}, p = {gof.pvalue:.4f}")
    assert gof.pvalue > 0.05
    ct = ox.stats.chi2_contingency([[10, 10, 20], [20, 20, 10]])
    print(f"chi2_contingency: chi2 = {ct.statistic:.4f}, p = {ct.pvalue:.4f}, dof = {ct.dof}")
    assert ct.dof == 2

    # --- Correlation ---
    pr = ox.stats.pearsonr(a, b)
    sr = ox.stats.spearmanr(a, b)
    print(f"pearson r = {pr.statistic:.4f} (p = {pr.pvalue:.3f}), spearman r = {sr.statistic:.4f}")
    assert abs(pr.statistic) < 0.2  # independent draws

    # --- Non-parametric ---
    mw = ox.stats.mannwhitneyu(a, b)
    print(f"mannwhitneyu: U = {mw.statistic:.1f}, p = {mw.pvalue:.2e}")
    assert mw.pvalue < 1e-3
    ks = ox.stats.kstest(rng.normal(size=300), "norm")
    print(f"kstest vs Normal: D = {ks.statistic:.4f}, p = {ks.pvalue:.4f}")
    assert ks.pvalue > 0.05

    # --- Moments, ECDF, multiple testing ---
    print(f"skew = {ox.stats.skew(a):.4f}, kurtosis = {ox.stats.kurtosis(a):.4f}")
    jb = ox.stats.jarque_bera(a)
    print(f"jarque_bera: JB = {jb.statistic:.4f}, p = {jb.pvalue:.4f}")
    assert jb.pvalue > 0.05

    e = ox.stats.ecdf(a)
    print(f"ecdf median bracket: {e.quantiles[99]:.4f} .. {e.quantiles[100]:.4f}")
    assert abs(float(e([0.0])[0]) - 0.5) < 0.1

    rejected, adjusted = ox.stats.p_adjust([0.01, 0.04, 0.30], method="holm")
    print(f"holm: rejected = {rejected}, adjusted = {np.asarray(adjusted).round(4)}")
    assert list(rejected) == [True, False, False]

    print("science_statistics: OK")


if __name__ == "__main__":
    main()
