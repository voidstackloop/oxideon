"""Batch 16: multiple testing, power, causal estimators, clustering, feature selection, optimal transport,
numerics, root finding / optimization / quadrature, geometry, Soundex."""
import math, re, sys
import numpy as np
from scipy import stats, optimize, integrate, interpolate, spatial
from scipy.cluster import hierarchy
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.feature_selection import f_classif as sk_f
from sklearn.decomposition import NMF
from sklearn.cluster import SpectralClustering, MeanShift

raw = {}
for line in open(sys.argv[1]):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if rest and not line.startswith(("warning", "running")): raw[k] = rest
# data rows may be wrapped by the console; rejoin is unnecessary because the file is written untruncated
F = lambda k: [float(x) for x in raw[k].split()]
S = lambda k: raw[k].replace(" ", "")
bad = n = 0
def check(name, ok, detail=""):
    global bad, n
    n += 1
    if not ok:
        bad += 1; print("MISMATCH", name, detail)
def close(name, got, want, tol=1e-9):
    check(name, abs(got - want) <= tol * max(1.0, abs(want)), f"{got} vs {want}")

# ---- multiple testing
p = np.array([F(f"p_{i}")[0] for i in range(25)]); m = len(p)
o = np.argsort(p)
holm = np.empty(m); run = 0
for r, i in enumerate(o): run = max(run, min(1, (m - r) * p[i])); holm[i] = run
bh = stats.false_discovery_control(p, method="bh")
for i in range(m):
    close(f"bonf_{i}", F(f"bonf_{i}")[0], min(1, p[i] * m), 1e-13)
    close(f"holm_{i}", F(f"holm_{i}")[0], holm[i], 1e-13)
    close(f"bh_{i}", F(f"bh_{i}")[0], bh[i], 1e-13)

# ---- norm_ppf (Acklam, rel. error < 1.2e-9)
for i in range(14):
    q, v = F(f"ppf_{i}")
    if q == 0.5: check("ppf_median", abs(v) < 1e-9)
    else: close(f"ppf_{i}", v, stats.norm.ppf(q), 2e-9)

# ---- two-proportion z-test, sample size, power
for i, (x1, n1, x2, n2) in enumerate([(120,1000,150,1000),(30,200,55,220),(5,50,9,60),(480,5000,600,5000)]):
    pp = (x1 + x2) / (n1 + n2); se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)); z = (x1 / n1 - x2 / n2) / se
    zs, pv = F(f"zt_{i}")
    close(f"zt_{i}_z", zs, z, 1e-12); close(f"zt_{i}_p", pv, 2 * stats.norm.sf(abs(z)), 1e-10)
za = lambda a: stats.norm.ppf(1 - a / 2)
for i, (d, sd, a, pw) in enumerate([(.5,1,.05,.8),(2,10,.01,.9),(.1,1,.05,.95)]):
    want = math.ceil(2 * ((za(a) + stats.norm.ppf(pw)) * sd / d) ** 2)
    check(f"nm_{i}", int(F(f"nm_{i}")[0]) == want, f"{F(f'nm_{i}')[0]} vs {want}")
for i, (p1, p2, a, pw) in enumerate([(.1,.12,.05,.8),(.5,.55,.01,.9),(.02,.03,.05,.8)]):
    pb = (p1 + p2) / 2
    num = za(a) * math.sqrt(2 * pb * (1 - pb)) + stats.norm.ppf(pw) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    check(f"np_{i}", int(F(f"np_{i}")[0]) == math.ceil(num * num / (p1 - p2) ** 2))
try:
    from statsmodels.stats.power import NormalIndPower
    HAVE_SM = True
except Exception:
    HAVE_SM = False
for i, (d, sd, nn, a) in enumerate([(.5,1,64,.05),(.2,1,100,.05),(1,2,30,.01)]):
    se = sd * math.sqrt(2 / nn); z = za(a)
    want = stats.norm.cdf(d / se - z) + stats.norm.cdf(-d / se - z)
    close(f"pw_{i}", F(f"pw_{i}")[0], want, 1e-9)
    if HAVE_SM: close(f"pw_{i}_statsmodels", F(f"pw_{i}")[0], NormalIndPower().power(d / sd, nn, a), 1e-6)

# ---- IPW / 2SLS / DiD
from sklearn.linear_model import LogisticRegression
rows = np.array([F(f"ipx_{i}") for i in range(400)])
X, T, Y = rows[:, :2], rows[:, 2], rows[:, 3]
lr = LogisticRegression(penalty=None, tol=1e-12, max_iter=1000).fit(X, T)
e = np.clip(lr.predict_proba(X)[:, 1], 0.01, 0.99)
close("ipw", F("ipw")[0], (T * Y / e - (1 - T) * Y / (1 - e)).mean(), 1e-6)
check("ipw_near_truth", abs(F("ipw")[0] - 2.0) < 0.35, F("ipw")[0])
iv = np.array([F(f"iv_{i}") for i in range(400)])
y, endog, ctrl, z1, z2 = iv[:, 0], iv[:, 1], iv[:, 2], iv[:, 3], iv[:, 4]
Z = np.column_stack([np.ones(400), z1, z2, ctrl])
fitted = Z @ np.linalg.lstsq(Z, endog, rcond=None)[0]
b = np.linalg.lstsq(np.column_stack([np.ones(400), fitted, ctrl]), y, rcond=None)[0]
for i in range(3): close(f"2sls_{i}", F(f"2sls_{i}")[0], b[i], 1e-8)
check("2sls_recovers_truth", abs(b[1] - 2.0) < 0.1 and abs(b[2] - 0.5) < 0.15, b)
close("did", F("did")[0], (np.mean([4, 5]) - np.mean([1, 2, 3.5])) - (np.mean([1, 2, 2.5]) - np.mean([.5, 1.5])), 1e-12)

# ---- clustering
pts = np.array([F(f"pt_{i}") for i in range(90)]); truth = np.arange(90) % 3
lab = lambda k: np.array([int(x) for x in S(k).split(",")])
for name, method in [("single", "single"), ("complete", "complete"), ("average", "average"), ("ward", "ward")]:
    want = hierarchy.fcluster(hierarchy.linkage(pts, method), 3, "maxclust")
    check(f"agg_{name}_vs_scipy", ari(lab(f"agg_{name}"), want) == 1.0, ari(lab(f"agg_{name}"), want))
check("spectral_vs_truth", ari(lab("spectral"), truth) > 0.95, ari(lab("spectral"), truth))
check("meanshift_vs_truth", ari(lab("meanshift"), truth) > 0.95, ari(lab("meanshift"), truth))
ms = MeanShift(bandwidth=2.5).fit(pts)
check("meanshift_vs_sklearn", ari(lab("meanshift"), ms.labels_) > 0.95)
xm = np.array([F(f"nmfx_{i}") for i in range(30)])
for rank, key in [(3, "nmf_err"), (1, "nmf_err_rank1")]:
    sk = NMF(n_components=rank, init="nndsvda", max_iter=5000, tol=1e-10, random_state=0).fit(xm).reconstruction_err_
    got = F(key)[0]
    # exact-rank data: both near zero; rank-1: both are (near-)optimal nonnegative rank-1 fits
    check(f"nmf_rank{rank}", got < max(1.05 * sk, 0.02), f"{got} vs sklearn {sk}")

# ---- feature selection
xf = np.array([F(f"fx_{i}") for i in range(120)]); yf = np.arange(120) % 3
fs = sk_f(xf, yf)[0]
for j in range(3): close(f"fc_{j}", F(f"fc_{j}")[0], fs[j], 1e-9)
def mi_binned(x, y, bins):
    lo, hi = x.min(), x.max(); b = np.minimum(((x - lo) / (hi - lo) * bins).astype(int), bins - 1)
    j = np.zeros((bins, 3)); [j.__setitem__((bb, yy), j[bb, yy] + 1) for bb, yy in zip(b, y)]
    j /= j.sum(); px, py = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return (j[nz] * np.log(j[nz] / (px @ py)[nz])).sum()
for j in range(3): close(f"mi_{j}", F(f"mi_{j}")[0], mi_binned(xf[:, j], yf, 8), 1e-9)
from sklearn.feature_selection import VarianceThreshold
check("vt", S("vt") == ",".join(str(i) for i in np.flatnonzero(VarianceThreshold(1.0).fit(xf).get_support())))

# ---- Sinkhorn vs an independent numpy Sinkhorn and the exact LP
a = np.array([.1, .3, .2, .25, .15]); bb = np.array([.2, .1, .25, .15, .2, .1])
C = np.array([F(f"cost_{i}") for i in range(5)])
c_lp = optimize.linprog(C.ravel(), A_eq=np.vstack([np.kron(np.eye(5), np.ones(6)), np.kron(np.ones(5), np.eye(6))]),
                        b_eq=np.concatenate([a, bb]), bounds=(0, None)).fun
def sink(eps, it=20000):
    K = np.exp(-C / eps); v = np.ones(6)
    for _ in range(it):
        u_ = a / (K @ v); v = bb / (K.T @ u_)
    P = u_[:, None] * K * v[None]; return (P * C).sum()
for eps in (0.5, 0.05, 0.01):
    got = F(f"sink_{eps}_cost")[0]
    close(f"sink_{eps}", got, sink(eps), 1e-5)
    check(f"sink_{eps}_marginals", F(f"sink_{eps}_marg")[0] < 1e-9)
    check(f"sink_{eps}_above_exact", got >= c_lp - 1e-9)
check("sink_converges_to_ot", abs(F("sink_0.01_cost")[0] - c_lp) < 0.005, (F("sink_0.01_cost")[0], c_lp))

# ---- numerics
close("lagr", F("lagr")[0], interpolate.lagrange([0, 1, 2, 3.5], [1, 3, 2, 7])(1.7), 1e-10)
cn = np.sort(np.array(F("cheb"))); want = np.sort(np.polynomial.chebyshev.chebpts1(6) * 2.5 + 0.5)
check("cheb", np.allclose(cn, want, atol=1e-12))
fx = lambda x: math.sin(x) * math.exp(x); d_true = math.exp(.8) * (math.sin(.8) + math.cos(.8))
close("fd1", F("fd1")[0], d_true, 1e-8); close("rich", F("rich")[0], d_true, 1e-9)
v = (1.1, .7, -.4)
check("grad", np.allclose(F("grad"), [3 * v[0]**2 * v[1], v[0]**3 + v[2] * math.cos(v[1] * v[2]), v[1] * math.cos(v[1] * v[2])], atol=1e-6))
x0, x1 = .9, -.6
check("jac", np.allclose(F("jac"), [x1, x0, math.cos(x0), 2 * x1, math.exp(x0), 0], atol=1e-6))
s = math.sin(v[1] * v[2]); c = math.cos(v[1] * v[2])
Ht = [[6 * v[0] * v[1], 3 * v[0]**2, 0], [3 * v[0]**2, -v[2]**2 * s, c - v[1] * v[2] * s], [0, c - v[1] * v[2] * s, -v[1]**2 * s]]
check("hess", np.allclose(F("hess"), np.array(Ht).ravel(), atol=1e-5), (F("hess"), np.array(Ht).ravel()))

# ---- root finding / minimization / quadrature
close("bisect", F("bisect")[0], optimize.bisect(lambda x: x**3 - 2 * x - 5, 2, 3, xtol=1e-14), 1e-11)
close("brentq", F("brentq")[0], optimize.brentq(lambda x: math.cos(x) - x, 0, 1, xtol=1e-14), 1e-11)
close("newton", F("newton")[0], math.sqrt(612), 1e-11)
bm = optimize.minimize_scalar(lambda x: (x - 2)**2 + math.sin(3 * x), bounds=(1, 3), method="bounded", options={"xatol": 1e-10})
close("brentmin_f", F("brentmin")[1], bm.fun, 1e-8); close("brentmin_x", F("brentmin")[0], bm.x, 1e-6)
for key in ("nm", "bfgs"):
    r = F(key); check(key, abs(r[0] - 1) < 1e-4 and abs(r[1] - 1) < 1e-4 and r[2] < 1e-7, r)
qv = integrate.quad(lambda x: math.exp(-x * x) * math.cos(x), -2, 3, epsabs=1e-12)[0]
close("quad", F("quad")[0], qv, 1e-9)
sv = integrate.quad(lambda x: math.sin(x)**2 / (1 + x), 0, 4, epsabs=1e-13)[0]
close("simpson", F("simpson")[0], sv, 1e-8); close("romberg", F("romberg")[0], sv, 1e-7)
cf = np.array([F(f"cf_{i}") for i in range(30)])
pf, _ = optimize.curve_fit(lambda x, A, k, c0: A * np.exp(-k * x) + c0, cf[:, 0], cf[:, 1], p0=[1, 1, 0])
check("curvefit", np.allclose(F("curvefit"), pf, atol=1e-5), (F("curvefit"), pf))

# ---- geometry
gp = np.array([F(f"gp_{i}") for i in range(60)])
h = spatial.ConvexHull(gp)
check("hull_n", int(F("hull_n")[0]) == len(h.vertices), (F("hull_n")[0], len(h.vertices)))
close("hull_area", F("hull_area")[0], h.volume, 1e-9)
close("signed_area", F("signed_area_poly")[0], 16.0, 1e-12)
poly = [(0, 0), (5, 0), (5, 5), (2.5, 2), (0, 5)]
def pip(pt):
    x, y = pt; ins = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1: ins = not ins
    return int(ins)
def u(i):
    z = (i + 0x9E3779B97F4A7C15) * 0xBF58476D1CE4E5B9 & (2**64 - 1)
    z = ((z ^ (z >> 30)) * 0x94D049BB133111EB) & (2**64 - 1); z ^= z >> 31
    return ((z >> 11) + 0.5) / 2**53
wantp = [pip((u(110000 + i) * 7 - 1, u(111000 + i) * 7 - 1)) for i in range(40)]
check("pip", [int(x) for x in S("pip").split(",")] == wantp)
d = spatial.distance.squareform(spatial.distance.pdist(gp)); np.fill_diagonal(d, np.inf)
i, j = np.unravel_index(d.argmin(), d.shape)
ci, cj, cd = raw["closest"].split()
check("closest_pair", (int(ci), int(cj)) == (min(i, j), max(i, j)) and abs(float(cd) - d.min()) < 1e-9, (ci, cj, cd, i, j, d.min()))

# ---- Soundex canonical vectors
for w, code in {"Robert":"R163","Rupert":"R163","Rubin":"R150","Ashcraft":"A261","Ashcroft":"A261","Tymczak":"T522","Pfister":"P236",
                "Honeyman":"H555","Lee":"L000","Gutierrez":"G362","Jackson":"J250","VanDeusen":"V532","Ekzampul":"E251"}.items():
    check(f"sdx_{w}", raw[f"sdx_{w}"] == code, raw[f"sdx_{w}"])
print(f"{n} checks, {bad} mismatches")
sys.exit(1 if bad else 0)
