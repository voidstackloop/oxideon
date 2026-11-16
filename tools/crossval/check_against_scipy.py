"""Compare Rust-computed statistics/metrics (crates/oxideon-core/tests/crossval_emit.rs) with SciPy / scikit-learn.

    cargo test -p oxideon-core --test crossval_emit -- --nocapture --ignored > rust_values.txt
    py -3.11 tools/crossval/check_against_scipy.py rust_values.txt
"""
import sys

import numpy as np
from scipy import signal, special, stats
from sklearn import metrics

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass

a = np.array([5.1, 4.9, 5.6, 5.8, 6.0, 5.2, 5.5, 6.3, 4.7, 5.9])
b = np.array([6.2, 6.8, 5.9, 7.1, 6.5, 6.0, 7.3, 6.7])
x = np.arange(1, 13, dtype=float)
y = np.array([2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 6.0, 9.0, 8.0, 12.0, 10.0, 11.0])
l1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
l2 = [0, 0, 1, 1, 1, 2, 2, 2, 0, 2]
pts = np.array([[(i * i % 7) + 0.1 * i, ((i * 3) % 5) - 0.05 * i] for i in range(10)])
scores = np.array([0.1, 0.4, 0.35, 0.8, 0.65, 0.2, 0.9, 0.55, 0.3, 0.7])
tg = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1])

ref = {}
r = stats.ttest_1samp(a, 5.0); ref["ttest_1samp.stat"], ref["ttest_1samp.p"] = r.statistic, r.pvalue
r = stats.ttest_ind(a, b, equal_var=False); ref["ttest_ind.stat"], ref["ttest_ind.p"] = r.statistic, r.pvalue
r = stats.mannwhitneyu(a, b, use_continuity=True, alternative="two-sided", method="asymptotic"); ref["mannwhitneyu.U"], ref["mannwhitneyu.p"] = r.statistic, r.pvalue
ref["ks_2samp.stat"] = stats.ks_2samp(a, b).statistic
r = stats.pearsonr(x, y); ref["pearsonr.r"], ref["pearsonr.p"] = r.statistic, r.pvalue
r = stats.spearmanr(x, y); ref["spearmanr.r"], ref["spearmanr.p"] = r.statistic, r.pvalue
ref["kendalltau.tau"] = stats.kendalltau(x, y).statistic
r = stats.f_oneway(a[:5], a[5:], b); ref["f_oneway.F"], ref["f_oneway.p"] = r.statistic, r.pvalue
c = stats.chi2_contingency(np.array([[12, 5, 9], [7, 14, 6]]), correction=False); ref["chi2_indep.stat"], ref["chi2_indep.p"] = c[0], c[1]
ref["t_cdf(1.7,6)"] = stats.t.cdf(1.7, 6)
ref["t_cdf(-0.4,3.5)"] = stats.t.cdf(-0.4, 3.5)
ref["chi2_cdf(7.2,4)"] = stats.chi2.cdf(7.2, 4)
ref["f_cdf(2.3,3,17)"] = stats.f.cdf(2.3, 3, 17)
ref["norm_cdf(-1.3)"] = stats.norm.cdf(-1.3)
ref["beta_inc(2.5,1.5,0.35)"] = special.betainc(2.5, 1.5, 0.35)
ref["gamma_p(3.2,2.1)"] = special.gammainc(3.2, 2.1)
ref["ari"] = metrics.adjusted_rand_score(l1, l2)
ref["nmi"] = metrics.normalized_mutual_info_score(l1, l2)
h, c_, v = metrics.homogeneity_completeness_v_measure(l1, l2)
ref["homogeneity"], ref["completeness"], ref["v_measure"] = h, c_, v
ref["silhouette"] = metrics.silhouette_score(pts, l1)
ref["davies_bouldin"] = metrics.davies_bouldin_score(pts, l1)
ref["calinski_harabasz"] = metrics.calinski_harabasz_score(pts, l1)
ref["roc_auc"] = metrics.roc_auc_score(tg, scores)
ref["log_loss"] = metrics.log_loss(tg, scores)
ref["brier"] = metrics.brier_score_loss(tg, scores)
bb, aa = signal.butter(3, 0.35)
for i, v in enumerate(bb): ref[f"butter3_b{i}"] = v
for i, v in enumerate(aa): ref[f"butter3_a{i}"] = v
bh, ah = signal.butter(4, 0.3, btype="high")
for i, v in enumerate(bh): ref[f"butter4hp_b{i}"] = v
for i, v in enumerate(ah): ref[f"butter4hp_a{i}"] = v

worst = 0.0
bad = []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    got = rust[k]
    err = abs(got - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 1e-6:
        bad.append((k, got, want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
for k, got, want in bad:
    print(f"MISMATCH {k}: rust={got} reference={want}")
sys.exit(1 if bad else 0)
