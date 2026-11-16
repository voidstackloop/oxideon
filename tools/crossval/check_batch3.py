"""Third cross-validation batch vs scikit-learn / SciPy / NetworkX / NumPy.

    cargo test -p oxideon-core --test crossval_emit3 -- --nocapture --ignored > rust_values3.txt
    py -3.11 tools/crossval/check_batch3.py rust_values3.txt
"""
import sys

import networkx as nx
import numpy as np
from scipy import integrate, interpolate, optimize, stats
from scipy.sparse.linalg import cg as sp_cg
from scipy.spatial.transform import Rotation, Slerp
from sklearn import cluster, ensemble, linear_model, metrics, naive_bayes, neighbors, tree
from sklearn.feature_extraction.text import TfidfVectorizer

rust, text = {}, {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            text[parts[0]] = parts[1]

ref, label_checks = {}, []


def ari_one(key, want):
    got = [int(v) for v in text[key].split(",")]
    label_checks.append((key, metrics.adjusted_rand_score(want, got)))


P = np.array([[([(0, 0), (5, 1), (2, 6)][i % 3][0]) + ((i * 37 % 17) - 8) / 9.0 + 1e-4 * np.sin(i * 1.3), ([(0, 0), (5, 1), (2, 6)][i % 3][1]) + ((i * 53 % 19) - 9) / 9.0 + 1e-4 * np.cos(i * 2.1)] for i in range(45)])
y = np.arange(45) % 3
Q = np.array([[i * 0.5 - 1.0, (i * i) % 7] for i in range(12)])

# clustering (label agreement, i.e. adjusted Rand index of 1.0 means identical partitions)
ari_one("dbscan", cluster.DBSCAN(eps=0.9, min_samples=4).fit_predict(P))
for name, link in [("single", "single"), ("complete", "complete"), ("average", "average"), ("ward", "ward")]:
    ari_one(f"agglo_{name}", cluster.AgglomerativeClustering(n_clusters=3, linkage=link).fit_predict(P))
ari_one("spectral", cluster.SpectralClustering(3, affinity="rbf", gamma=0.5, random_state=0).fit_predict(P))

# classifiers
np.testing.assert_equal(text["knn"], ",".join(map(str, neighbors.KNeighborsClassifier(5).fit(P, y).predict(Q))))
gnb = naive_bayes.GaussianNB(var_smoothing=1e-9).fit(P, y)
label_checks.append(("gnb", float(text["gnb"] == ",".join(map(str, gnb.predict(Q))))))
jl = gnb.predict_joint_log_proba(Q)
for i in range(4):
    for c in range(3):
        ref[f"gnb_ll_{i}_{c}"] = jl[i][c]
counts = np.array([[((i * (j + 3) + j * 5) % 7) for j in range(6)] for i in range(40)], dtype=float)
cl = np.array([(i * 7) % 3 for i in range(40)])
mnb = naive_bayes.MultinomialNB(alpha=1.0).fit(counts, cl)
label_checks.append(("mnb", float(text["mnb"] == ",".join(map(str, mnb.predict(counts))))))
pr = mnb.predict_proba(counts)
for i in range(3):
    for c in range(3):
        ref[f"mnb_p_{i}_{c}"] = pr[i][c]
label_checks.append(("bnb", float(text["bnb"] == ",".join(map(str, naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0).fit(counts, cl).predict(counts))))))
label_checks.append(("cnb", float(text["cnb"] == ",".join(map(str, naive_bayes.ComplementNB(alpha=1.0).fit(counts, cl).predict(counts))))))

# GLMs
bx = np.array([[np.sin(i * 0.17) * 2.0, ((i * 11) % 9) / 4.0 - 1.0] for i in range(60)])
by = np.array([float(r[0] + 0.5 * r[1] + 1.5 * np.sin(i * 1.7) > 0.1) for i, r in enumerate(bx)])
lg = linear_model.LogisticRegression(penalty=None, solver="newton-cg", tol=1e-12, max_iter=1000).fit(bx, by)
ref["glm_logit_c0"] = lg.intercept_[0]
ref["glm_logit_c1"], ref["glm_logit_c2"] = lg.coef_[0]
cnt = np.array([round(np.exp(0.4 + 0.5 * r[0] - 0.3 * r[1]) + (i % 3)) for i, r in enumerate(bx)])
pg = linear_model.PoissonRegressor(alpha=0.0, tol=1e-12, max_iter=1000).fit(bx, cnt)
ref["glm_pois_c0"] = pg.intercept_
ref["glm_pois_c1"], ref["glm_pois_c2"] = pg.coef_

# trees / boosting
gx = np.array([[i / 10.0, ((i * 13) % 17) / 5.0] for i in range(80)])
gy = np.sin(gx[:, 0]) * 2.0 + gx[:, 1] * 0.5
gbr = ensemble.GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=3, criterion="squared_error", random_state=0).fit(gx, gy)
for i, v in enumerate(gbr.predict(gx)): ref[f"gbr_{i}"] = v
rt = tree.DecisionTreeRegressor(max_depth=4, random_state=0).fit(gx, gy)
for i, v in enumerate(rt.predict(gx)): ref[f"rtree_{i}"] = v

# outlier scores
lof = neighbors.LocalOutlierFactor(n_neighbors=6).fit(P)
for i, v in enumerate(-lof.negative_outlier_factor_): ref[f"lof_{i}"] = v
mu = P.mean(0)
cov = np.cov(P.T)
inv = np.linalg.inv(cov + 1e-9 * np.eye(2))
for i, r in enumerate(P): ref[f"maha_{i}"] = float(np.sqrt((r - mu) @ inv @ (r - mu)))

# tf-idf (documents without single-character tokens so the default sklearn token pattern agrees)
docs = ["the cat sat on the mat", "the dog chased the cat", "dogs and cats are pets", "quantum field theory lectures"]
tv = TfidfVectorizer(norm="l2", smooth_idf=True, sublinear_tf=False).fit(docs)
vocab_sorted = sorted(tv.vocabulary_, key=tv.vocabulary_.get)
mat = tv.transform(docs).toarray()
for d in range(4):
    for ix, term in enumerate(vocab_sorted):
        if mat[d][ix] != 0:
            ref[f"tfidf_{d}_{ix}"] = mat[d][ix]

# graph algorithms vs networkx
G = nx.Graph()
for a, b, w in [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15), (2, 3, 11), (2, 5, 2), (3, 4, 6), (4, 5, 9), (6, 7, 1), (7, 8, 2)]:
    G.add_edge(a, b, weight=float(w))
for k, v in nx.single_source_dijkstra_path_length(G, 0).items(): ref[f"dijkstra_{k}"] = v
for k, v in nx.pagerank(G, alpha=0.85, tol=1e-14, max_iter=1000).items(): ref[f"pagerank_{k}"] = v
Gu = nx.Graph(G)
for k, v in nx.betweenness_centrality(Gu, normalized=True, weight=None).items(): ref[f"betweenness_{k}"] = v
for k, v in nx.clustering(Gu).items(): ref[f"clustering_{k}"] = v
ref["mst_total"] = sum(d["weight"] for _, _, d in nx.minimum_spanning_edges(G, data=True))
comp = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
got = [int(v) for v in text["components"].split(",")]
label_checks.append(("components", metrics.adjusted_rand_score([comp[i] for i in range(9)], got)))
D = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)])
scc = {n: i for i, c in enumerate(nx.strongly_connected_components(D)) for n in c}
got = [int(v) for v in text["scc"].split(",")]
label_checks.append(("scc", metrics.adjusted_rand_score([scc[i] for i in range(6)], got)))
F = nx.DiGraph()
for a, b, c in [(0, 1, 16), (0, 2, 13), (1, 2, 10), (2, 1, 4), (1, 3, 12), (3, 2, 9), (2, 4, 14), (4, 3, 7), (3, 5, 20), (4, 5, 4)]:
    F.add_edge(a, b, capacity=c)
ref["maxflow"] = nx.maximum_flow_value(F, 0, 5)

# rotations / alignment
r1 = Rotation.from_euler("ZYX", [0.4, -0.3, 0.9])
qx, qy, qz, qw = r1.as_quat()
# the Rust quaternion and scipy's may differ by an overall sign; compare after fixing the sign of w
sgn = 1.0 if (rust["quat_w"] * qw >= 0) else -1.0
ref["quat_w"], ref["quat_x"], ref["quat_y"], ref["quat_z"] = sgn * qw, sgn * qx, sgn * qy, sgn * qz
for i, v in enumerate(r1.as_matrix().ravel()): ref[f"rotm_{i}"] = v
r2 = Rotation.from_euler("ZYX", [-1.1, 0.7, 0.2])
sl = Slerp([0, 1], Rotation.concatenate([r1, r2]))(0.37)
sx, sy, sz, sw = sl.as_quat()
sg = 1.0 if (rust["slerp_w"] * sw >= 0) else -1.0
ref["slerp_w"], ref["slerp_x"], ref["slerp_y"], ref["slerp_z"] = sg * sw, sg * sx, sg * sy, sg * sz
src = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1], [-1, 0.5, 2]], dtype=float)
M = r1.as_matrix()
dst = np.array([M @ s + np.array([1.0, -2.0, 0.5]) + np.array([0.01 * np.sin(i), -0.02 * np.cos(i), 0.015 * np.sin(2.0 * i)]) for i, s in enumerate(src)])
sc = src - src.mean(0)
dc = dst - dst.mean(0)
R, rssd = Rotation.align_vectors(dc, sc)
for i, v in enumerate(R.as_matrix().ravel()): ref[f"kabsch_r{i}"] = v
t = dst.mean(0) - R.as_matrix() @ src.mean(0)
for i, v in enumerate(t): ref[f"kabsch_t{i}"] = v
ref["kabsch_rms"] = float(np.sqrt(((src @ R.as_matrix().T + t - dst) ** 2).sum(1).mean()))

# polynomials
coef = [-6.0, 11.0, -6.0, 1.0, 0.5]
rts = sorted(np.roots(coef[::-1]), key=lambda z: (round(z.real * 1e8), z.imag))
for i, z in enumerate(rts):
    ref[f"root_re{i}"] = z.real
    ref[f"root_im{i}"] = z.imag
qd, rd = np.polydiv(coef[::-1], [1.0, -1.0, 2.0])
for i, v in enumerate(qd[::-1]): ref[f"pdiv_q{i}"] = v
for i, v in enumerate(np.trim_zeros(rd[::-1], "b") if rd.size else [0.0]): ref[f"pdiv_r{i}"] = v

# interpolation
from numpy.polynomial import chebyshev as C
cheb = C.Chebyshev.interpolate(lambda x: np.sin(3 * x) * np.exp(-x), 11, domain=[-1, 2])
chd = cheb.deriv()
for t_ in [-0.9, -0.2, 0.4, 1.1, 1.9]:
    ref[f"cheb_{t_}"] = float(cheb(t_))
    ref[f"chebd_{t_}"] = float(chd(t_))
ref["lagrange"] = float(interpolate.lagrange([0.0, 1.0, 2.5, 4.0, 5.5], [1.0, -1.0, 2.0, 0.5, 3.0])(3.3))
vals = np.array([((i * 7) % 5) + 0.25 * i for i in range(12)], dtype=float).reshape(3, 4)
rgi = interpolate.RegularGridInterpolator(([0.0, 1.0, 3.0], [-1.0, 0.5, 2.0, 4.0]), vals, method="linear")
for a, b in [(0.5, 1.0), (2.2, 3.3), (1.0, -0.4)]:
    ref[f"grid_{a:.1f}_{b:.1f}"] = float(rgi([[a, b]])[0])

# iterative solvers / eigenvalues
n = 40
A = np.zeros((n, n))
for i in range(n):
    A[i, i] = 2.0 + 0.01 * i
    if i > 0: A[i, i - 1] = -1.0
    if i + 1 < n: A[i, i + 1] = -1.0
rhs = np.cos(np.arange(n) * 0.3)
x_ref = np.linalg.solve(A, rhs)
for i, v in enumerate(x_ref): ref[f"cg_{i}"] = v
w = np.linalg.eigvalsh((A + A.T) / 2)
ref["lanczos_min"], ref["lanczos_max"] = w[0], w[-1]

# Sobol (unscrambled) vs scipy.stats.qmc
sob = stats.qmc.Sobol(d=6, scramble=False).random(32)
for i in range(32):
    for d in range(6): ref[f"sobol_{i}_{d}"] = sob[i][d]

# normal quantiles, Black-Scholes, financial maths
for pp in [0.001, 0.025, 0.3, 0.5, 0.84, 0.975, 0.9999]: ref[f"ppf_{pp}"] = stats.norm.ppf(pp)
S, K, r, q, sig, T = 105.0, 98.0, 0.03, 0.01, 0.27, 0.75
d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
d2 = d1 - sig * np.sqrt(T)
N, n_ = stats.norm.cdf, stats.norm.pdf
ref["bs_price"] = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
ref["bs_delta"] = np.exp(-q * T) * N(d1)
ref["bs_gamma"] = np.exp(-q * T) * n_(d1) / (S * sig * np.sqrt(T))
ref["bs_vega"] = S * np.exp(-q * T) * n_(d1) * np.sqrt(T)
ref["bs_theta"] = -S * np.exp(-q * T) * n_(d1) * sig / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2) + q * S * np.exp(-q * T) * N(d1)
ref["bs_rho"] = K * T * np.exp(-r * T) * N(d2)
ref["bs_put"] = K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)
cf = [-1000.0, 300.0, 400.0, 500.0, 120.0]
ref["irr"] = optimize.brentq(lambda rr: sum(c / (1 + rr) ** t for t, c in enumerate(cf)), -0.9, 5.0, xtol=1e-14)
i_ = 0.045 / 12
ref["loan"] = 250000.0 * i_ / (1 - (1 + i_) ** -300)

pv = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
m_ = len(pv)
holm = np.maximum.accumulate((m_ - np.arange(m_)) * pv)
for i, v in enumerate(np.minimum(holm, 1.0)): ref[f"holm_{i}"] = v
for i, v in enumerate(stats.false_discovery_control(pv, method="bh")): ref[f"bh_{i}"] = v

worst, bad = 0.0, []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 2e-6:
        bad.append((k, rust[k], want))
print(f"compared {len(ref)} numeric values, worst relative error {worst:.2e}")
for k, ari in label_checks:
    flag = "" if ari > 0.999 else "   <-- DIFFERENT"
    print(f"  labels {k:14s} agreement {ari:.4f}{flag}")
    if ari <= 0.999:
        bad.append((k, ari, 1.0))
for k, got, want in bad[:40]:
    print(f"MISMATCH {k}: rust={got} reference={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
