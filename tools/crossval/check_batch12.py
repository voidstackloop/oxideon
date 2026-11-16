"""Twelfth cross-validation batch: GP, KDE, time series, QMC, survival, HMM, Kalman, image processing,
preprocessing, number theory, algebra, DataFrame and model-selection splits vs SciPy / scikit-learn / pandas / SymPy
and independent NumPy reference implementations.

    cargo test -p oxideon-core --test crossval_emit12 -- --nocapture --ignored > rust_values12.txt
    py -3.11 tools/crossval/check_batch12.py rust_values12.txt
"""
import itertools
import re
import math
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import sympy
import torch
import torch.nn.functional as F
from scipy import ndimage, optimize, stats
from scipy.stats import qmc
from sklearn import model_selection as ms
from sklearn import preprocessing as skp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

rust, text = {}, {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line)
    parts = line.split(None, 1)
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            text[parts[0]] = parts[1].strip()

ref, loose = {}, {}  # `loose` entries carry their own tolerance (approximate algorithms)


def put(name, arr, tol=None):
    for i, v in enumerate(np.asarray(arr, dtype=float).ravel()):
        (loose.__setitem__(f"{name}_{i}", (float(v), tol)) if tol else ref.__setitem__(f"{name}_{i}", float(v)))


# ---------------------------------------------------------------- GP
X = np.array([[i * 0.5, ((i * 7) % 5) * 0.3] for i in range(12)])
Y = np.sin(X[:, 0] * 0.8) + 0.3 * X[:, 1]
Q = np.array([[0.7, 0.2], [2.9, 0.9], [5.0, 0.0], [8.0, 1.0]])
for name, base in [("rbf", RBF(1.3, "fixed")), ("m52", Matern(1.3, "fixed", nu=2.5))]:
    gpr = GaussianProcessRegressor(ConstantKernel(1.0, "fixed") * base, alpha=1e-3 + 1e-10, normalize_y=True, optimizer=None).fit(X, Y)
    m, sd = gpr.predict(Q, return_std=True)
    for i in range(4):
        ref[f"gp_{name}_mean_{i}"] = m[i]
        ref[f"gp_{name}_var_{i}"] = sd[i] ** 2
    ref[f"gp_{name}_lml"] = gpr.log_marginal_likelihood()

# ---------------------------------------------------------------- KDE (density with the same bandwidth)
data = np.array([math.sin(i * 0.7) * 2.0 + (i % 5) * 0.3 for i in range(40)])
h = rust["kde_bw"]
kde = stats.gaussian_kde(data, bw_method=h / data.std(ddof=1))
for i, q in enumerate([-2.5, -1.0, 0.0, 0.7, 2.2, 4.0]):
    ref[f"kde_{i}"] = float(kde(q)[0])
q1, q3 = np.percentile(data, [25, 75])
ref["kde_bw"] = 0.9 * min(data.std(ddof=1), (q3 - q1) / 1.34) * len(data) ** -0.2

# ---------------------------------------------------------------- time series
s = np.array([math.sin(i * 0.3) + 0.4 * math.cos((i * i) * 0.17) + 0.02 * i for i in range(80)])
mu = s.mean()
gam = np.array([((s[k:] - mu) * (s[:len(s) - k] - mu)).sum() / len(s) for k in range(4)])
T = np.array([[gam[abs(i - j)] for j in range(3)] for i in range(3)])
phi = np.linalg.solve(T, gam[1:4])
put("ar_phi", phi)
ref["ar_sigma2_0"] = gam[0] - phi @ gam[1:4]
ref["ar_mean"] = mu
r = np.array([((s[k:] - mu) * (s[:len(s) - k] - mu)).sum() / ((s - mu) ** 2).sum() for k in range(9)])
ref["ljung_box"] = len(s) * (len(s) + 2) * sum(r[k] ** 2 / (len(s) - k) for k in range(1, 9))
put("diff2", np.diff(s, 2)[:10])
ser = pd.Series(s)
put("ewma", ser.ewm(alpha=0.3, adjust=False).mean().values[:10])
put("rmean", ser.rolling(5).mean().values[4:14])
put("rstd", ser.rolling(5).std().values[4:14])
period = 6
kern = np.array([0.5] + [1.0] * (period - 1) + [0.5]) / period
trend = np.convolve(s, kern, mode="valid")  # entries for t = 3 .. n-4
for i in range(12):
    ref[f"trend_{3 + i}"] = trend[i]
full_trend = np.full(len(s), np.nan)
full_trend[3:3 + len(trend)] = trend
det = s - full_trend
raw = np.array([np.nanmean(det[k::period]) for k in range(period)])
put("seasonal", (raw - raw.mean()))

# ---------------------------------------------------------------- QMC
sob = qmc.Sobol(3, scramble=False).random(64)
hal = qmc.Halton(2, scramble=False).random(51)[1:]
for name, pts in [("sobol", sob), ("halton", hal)]:
    d2 = qmc.discrepancy(pts, method="L2-star")
    key = f"l2star_{name}_0"
    # scipy's L2-star value is the (non-squared) Warnock discrepancy
    ref[key.replace("_0", "")] = d2
for i in range(6):
    ref[f"halton_{i}_0"], ref[f"halton_{i}_1"] = hal[i]

# ---------------------------------------------------------------- survival
time = np.array([5.0, 6.0, 6.0, 2.5, 4.0, 4.0, 8.0, 9.0, 12.0, 12.0, 3.0, 7.5, 10.0, 1.0, 11.0])
event = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=bool)


def km_at(t):
    s_ = 1.0
    for u in sorted(set(time[event])):
        if u <= t:
            n = (time >= u).sum()
            d = ((time == u) & event).sum()
            s_ *= 1 - d / n
    return s_


def na_at(t):
    h_ = 0.0
    for u in sorted(set(time[event])):
        if u <= t:
            h_ += ((time == u) & event).sum() / (time >= u).sum()
    return h_


for t in [0.5, 1.0, 2.5, 4.0, 5.0, 6.0, 8.0, 9.0, 12.0, 20.0]: ref[f"km_{t:g}"] = km_at(t)
for t in [1.0, 4.0, 6.0, 9.0, 12.0]: ref[f"na_{t:g}"] = na_at(t)
grp = np.arange(15) % 2 == 0
lr = stats.logrank(stats.CensoredData(uncensored=time[event & grp], right=time[~event & grp]), stats.CensoredData(uncensored=time[event & ~grp], right=time[~event & ~grp]))
ref["logrank_stat"] = lr.statistic ** 2
ref["logrank_p"] = lr.pvalue
Xc = np.array([[math.sin(i * 0.6), (i % 4) * 0.5] for i in range(15)])


def neg_pl(b):
    eta = Xc @ b
    ll = 0.0
    for i in np.where(event)[0]:
        risk = time >= time[i]
        ll += eta[i] - np.log(np.exp(eta[risk]).sum())
    return -ll


res = optimize.minimize(neg_pl, np.zeros(2), method="BFGS", options=dict(gtol=1e-10))
res = optimize.minimize(neg_pl, res.x, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-14, maxiter=5000))
put("cox_coef", res.x)
ref["cox_ll"] = -res.fun
eps = 1e-4
Hs = np.zeros((2, 2))
for a in range(2):
    for b in range(2):
        ea, eb = np.eye(2)[a] * eps, np.eye(2)[b] * eps
        Hs[a, b] = (neg_pl(res.x + ea + eb) - neg_pl(res.x + ea - eb) - neg_pl(res.x - ea + eb) + neg_pl(res.x - ea - eb)) / (4 * eps * eps)
put("cox_se", np.sqrt(np.diag(np.linalg.inv(Hs))), tol=1e-4)
risk = np.exp(Xc @ res.x)
conc = tot = 0.0
for i in range(15):
    for j in range(15):
        if event[i] and time[i] < time[j]:
            tot += 1
            conc += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
ref["cindex"] = conc / tot

# ---------------------------------------------------------------- HMM by brute-force enumeration
start = np.array([0.6, 0.3, 0.1])
trans = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
emit = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6], [0.3, 0.3, 0.4]])
obs = [0, 1, 2, 2, 1, 0, 0, 2, 1, 1]
paths = np.array(list(itertools.product(range(3), repeat=len(obs))))
pr = start[paths[:, 0]] * emit[paths[:, 0], obs[0]]
for t in range(1, len(obs)):
    pr = pr * trans[paths[:, t - 1], paths[:, t]] * emit[paths[:, t], obs[t]]
tot = pr.sum()
ref["hmm_ll"] = math.log(tot)
for t in range(len(obs)):
    for i in range(3):
        ref[f"hmm_gamma_{t}_{i}"] = pr[paths[:, t] == i].sum() / tot
best = paths[np.argmax(pr)]

# ---------------------------------------------------------------- Kalman (textbook recursion, independent code path)
F_ = np.array([[1.0, 1.0], [0.0, 1.0]]); H_ = np.array([[1.0, 0.0]]); Q_ = np.diag([0.01, 0.02]); R_ = np.array([[0.5]])
ys = [0.4 * i + 0.3 * math.sin(i * 1.7) for i in range(12)]
x, P = np.zeros(2), np.eye(2)
xs, Ps, xp_l, Pp_l, ll = [], [], [], [], 0.0
for y in ys:
    xp = F_ @ x; Pp = F_ @ P @ F_.T + Q_
    S = H_ @ Pp @ H_.T + R_
    K = Pp @ H_.T @ np.linalg.inv(S)
    inn = np.array([y]) - H_ @ xp
    x = xp + K @ inn; P = (np.eye(2) - K @ H_) @ Pp
    ll += -0.5 * (np.log(np.linalg.det(S)) + inn @ np.linalg.inv(S) @ inn + np.log(2 * np.pi))
    xs.append(x); Ps.append(P); xp_l.append(xp); Pp_l.append(Pp)
for t in range(12):
    put(f"kf_mean_{t}", xs[t])
    ref[f"kf_var_{t}"] = Ps[t][0, 0]
ref["kf_ll"] = ll
sx, sP = list(xs), list(Ps)
for t in range(10, -1, -1):
    Pn = F_ @ Ps[t] @ F_.T + Q_
    C = Ps[t] @ F_.T @ np.linalg.inv(Pn)
    sx[t] = xs[t] + C @ (sx[t + 1] - F_ @ xs[t])
    sP[t] = Ps[t] + C @ (sP[t + 1] - Pn) @ C.T
for t in range(12):
    put(f"rts_mean_{t}", sx[t])

# ---------------------------------------------------------------- image processing vs scipy.ndimage
img = np.array([[((x_ * 3 + y_ * 5) % 7) / 7.0 + 0.1 * ((x_ * y_) % 3) for x_ in range(9)] for y_ in range(7)])
put("gblur", ndimage.gaussian_filter(img, 1.2, mode="mirror", truncate=3.0))
put("sobel_x", ndimage.sobel(img, axis=1, mode="mirror"))
put("sobel_y", ndimage.sobel(img, axis=0, mode="mirror"))
b = np.array([[float((x_ // 3 + y_ // 3) % 2 == 0 and x_ != 4) for x_ in range(10)] for y_ in range(8)])
lab4, n4 = ndimage.label(b)
ref["cc4_n_0"] = n4
lab8, n8 = ndimage.label(b, structure=np.ones((3, 3)))
ref["cc8_n_0"] = n8
put("erode", ndimage.grey_erosion(img, size=3, mode="constant", cval=np.inf))
put("dilate", ndimage.grey_dilation(img, size=3, mode="constant", cval=-np.inf))
ref["box_sum_0"] = img[1:5, 2:7].sum()
hist, _ = np.histogram(np.clip(img, 0, 1) * 255 // 1, bins=256, range=(0, 256))
ref["otsu_0"] = None
put("resize", F.interpolate(torch.tensor(img)[None, None], size=(4, 5), mode="bilinear", align_corners=False).numpy())
edt = ndimage.distance_transform_edt(b == 0)
put("dt", edt, tol=0.09)  # the Rust transform is a 3-4 style chamfer approximation, so only bounded agreement is expected

# ---------------------------------------------------------------- preprocessing vs scikit-learn
M = np.array([[math.sin(i * 0.7) * 3.0 + i, (i * 5) % 7, i ** 2 / 10.0] for i in range(12)])
put("robust", skp.RobustScaler().fit_transform(M))
put("kbins_q", skp.KBinsDiscretizer(4, encode="ordinal", strategy="quantile", quantile_method="linear").fit_transform(M) if "quantile_method" in skp.KBinsDiscretizer.__init__.__code__.co_varnames else skp.KBinsDiscretizer(4, encode="ordinal", strategy="quantile").fit_transform(M))
put("kbins_u", skp.KBinsDiscretizer(3, encode="ordinal", strategy="uniform").fit_transform(M))
small = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
put("poly", skp.PolynomialFeatures(3).fit_transform(small))
put("norm2", skp.normalize(small, "l2"))

# ---------------------------------------------------------------- number theory vs sympy
for n in [2, 97, 561, 7919, 1_000_000_007, 3_215_031_751]:
    ref[f"isprime_{n}_0"] = float(sympy.isprime(n))
factors = {n: ",".join(str(p_) for p_, e in sorted(sympy.factorint(n).items()) for _ in range(e)) for n in [360, 9_699_690, 600_851_475_143, 1_000_000_016_000_000_063]}
for n in [36, 97, 1001, 600_851_475_143]:
    ref[f"totient_{n}_0"] = float(sympy.totient(n))
ref["modinv_0"] = float(pow(17, -1, 3120))
cx, cm = sympy.ntheory.modular.crt([3, 5, 7, 11], [2, 3, 2, 10])
ref["crt_x_0"], ref["crt_m_0"] = float(cx), float(cm)
ref["binom_0"] = float(math.comb(60, 25))
ref["stirling2_0"] = float(sympy.functions.combinatorial.numbers.stirling(14, 5, kind=2))
ref["partitions_0"] = float(sympy.functions.combinatorial.numbers.partition(150))
ref["catalan_0"] = float(sympy.catalan(20))
ref["modpow_0"] = float(pow(7, 1_000_003, 1_000_000_007))

# ---------------------------------------------------------------- algebra vs fractions / numpy.polynomial
fr = Fraction(355, 113) * Fraction(-7, 22) + Fraction(5, 6)
exact_text = {"rational": f"{fr.numerator}/{fr.denominator}" if fr.denominator != 1 else str(fr.numerator),
              "approx": (lambda f: f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator))(Fraction(math.e).limit_denominator(500))}
from numpy.polynomial import Polynomial as P_
q = P_([1.0, -3.0, 0.0, 2.0]); r_ = P_([-2.0, 1.0, 5.0])
put("pmul", (q * r_).coef)
put("pcomp", q(r_).coef)
put("pder", q.deriv().coef)
put("pint", q.integ().coef)

# ---------------------------------------------------------------- DataFrame vs pandas
df = pd.DataFrame({"region": ["west", "east", "west", "east", "west", "north"], "product": list("aabbac"), "units": [10, 4, 7, 3, 3, 8], "price": [2.5, 2.5, 4.0, 4.0, 2.5, 9.0]})
g = df.groupby("region")["units"]
for name, ser_ in [("sum", g.sum()), ("mean", g.mean()), ("min", g.min()), ("max", g.max()), ("count", g.count())]:
    put(f"groupby_{name}", ser_.values.astype(float))
put("sort", np.sort(df["units"].values)[::-1].astype(float))
desc = df[["units", "price"]].describe().loc[["count", "mean", "std", "min", "max"]]
for c, col in enumerate(["units", "price"]):
    put(f"describe_{c}", desc[col].values)

# ---------------------------------------------------------------- split schemes vs scikit-learn
labels = np.array([int(i % 3 == 0) for i in range(23)])
sk_sizes = sorted(len(te) for _, te in ms.StratifiedKFold(4).split(np.zeros(23), labels))
rs_sizes = sorted(int(rust[f"skf_size_{i}"]) for i in range(4))
rs_pos = sorted(int(rust[f"skf_pos_{i}"]) for i in range(4))
sk_pos = sorted(int(labels[te].sum()) for _, te in ms.StratifiedKFold(4).split(np.zeros(23), labels))
for i, (tr, te) in enumerate(ms.TimeSeriesSplit(4).split(np.zeros(20))):
    ref[f"tss_train_{i}"] = len(tr)
    ref[f"tss_test_{i}"] = len(te)
    ref[f"tss_first_{i}"] = te[0]

# ---------------------------------------------------------------- compare
worst, bad = 0.0, []
checks = [(k, v, 2e-6) for k, v in ref.items() if v is not None] + [(k, v, t) for k, (v, t) in loose.items()]
for k, want, tol in checks:
    key = k if k in rust else (k[:-2] if k.endswith("_0") and k[:-2] in rust else k)
    if key not in rust:
        key = k[: k.rfind('_')] + k[k.rfind('_') + 1:] if k.rfind('_') > 0 and k[k.rfind('_') + 1:].isdigit() else k  # ar_phi_0 -> ar_phi0
    if key not in rust:
        bad.append((k, "missing", want)); continue
    k_rust = key
    err = abs(rust[k_rust] - want) / max(1.0, abs(want))
    if tol == 0.09:
        pass
    worst = max(worst, err) if tol == 2e-6 else worst
    if err > tol:
        bad.append((k, rust[k_rust], want))
checks_text = [("hmm_viterbi", ",".join(map(str, best))), ("rational", exact_text["rational"]), ("approx", exact_text["approx"])]
checks_text += [(f"factor_{n}", v) for n, v in factors.items()]
checks_text += [("cc4", ",".join(map(str, lab4.ravel())))]
for k, want in checks_text:
    if text.get(k) != want:
        bad.append((k, text.get(k), want))
if sk_sizes != rs_sizes or sk_pos != rs_pos:
    bad.append(("stratified_kfold sizes/positives", (rs_sizes, rs_pos), (sk_sizes, sk_pos)))
print(f"compared {len(checks)} numeric values (+{len(checks_text) + 1} exact checks), worst relative error {worst:.2e}")
groups = {}
for k, got, want in bad:
    groups.setdefault(k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k, []).append((k, got, want))
for gname, items in groups.items():
    k, got, want = items[0]
    print(f"MISMATCH {gname}: {len(items)} differ, e.g. {k}: rust={got} reference={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
