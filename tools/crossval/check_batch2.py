"""Second cross-validation batch (signal processing, interpolation, optimization, linear models) vs SciPy / sklearn.

    cargo test -p oxideon-core --test crossval_emit emit_batch2 -- --nocapture --ignored > rust_values2.txt
    py -3.11 tools/crossval/check_batch2.py rust_values2.txt
"""
import sys

import numpy as np
from scipy import integrate, interpolate, optimize, signal
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge

rust, text = {}, {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            text[parts[0]] = parts[1]
ref = {}

sig = np.array([np.sin(i * 0.3) + 0.2 * np.cos((i * i) * 0.7) for i in range(60)])
for i, v in enumerate(signal.savgol_filter(sig, 7, 2, deriv=0, mode="interp")): ref[f"savgol0_{i}"] = v
for i, v in enumerate(signal.savgol_filter(sig, 9, 3, deriv=1, mode="interp")): ref[f"savgol1_{i}"] = v
med = signal.medfilt(sig, 5)
for i in range(2, 57): ref[f"medfilt_{i}"] = med[i]
for i, v in enumerate(signal.firwin(41, 0.25)): ref[f"firwin_lp_{i}"] = v
for i, v in enumerate(signal.firwin(41, [0.2, 0.5], pass_zero=False)): ref[f"firwin_bp_{i}"] = v
for i, v in enumerate(signal.firwin(41, 0.3, pass_zero=False)): ref[f"firwin_hp_{i}"] = v
long = np.array([np.sin(i * 0.4) + 0.5 * np.sin(i * 1.3) + 0.1 * ((i * 7919) % 13) for i in range(512)])
f, p = signal.welch(long, fs=100.0, window="hann", nperseg=128, noverlap=64, detrend="constant", scaling="density")
for i in range(len(f)):
    ref[f"welch_f{i}"] = f[i]
    ref[f"welch_p{i}"] = p[i]
xs = np.array([0.0, 0.7, 1.5, 2.1, 3.4, 4.0, 5.5])
ys = np.array([1.0, 0.2, 1.9, -0.5, 2.2, 1.1, 0.3])
cs = interpolate.CubicSpline(xs, ys, bc_type="natural")
for q in [0.3, 1.0, 1.8, 2.7, 3.7, 4.8, 5.4]:
    ref[f"spline_{q:.1f}"] = float(cs(q))
    ref[f"splined_{q:.1f}"] = float(cs(q, 1))
cost = np.array([[4.0, 1.0, 3.0, 7.0], [2.0, 0.0, 5.0, 3.0], [3.0, 2.0, 2.0, 9.0], [6.0, 4.0, 8.0, 1.0]])
r, c = optimize.linear_sum_assignment(cost)
ref["assign_total"] = cost[r, c].sum()
lp = optimize.linprog([-3, -5, -1], A_ub=[[1, 0, 1], [0, 2, 1], [3, 2, 2]], b_ub=[4, 12, 18], A_eq=[[1, 1, 1]], b_eq=[7], bounds=(0, None))
ref["linprog_obj"] = lp.fun
nn_a = np.array([[1.0, 0.5, 2.0], [0.3, 1.2, 0.1], [2.0, 0.1, 0.4], [0.6, 0.9, 1.1], [1.5, 0.2, 0.7]])
nx, nr = optimize.nnls(nn_a, np.array([1.0, -0.5, 2.0, 0.3, 1.1]))
for i, v in enumerate(nx): ref[f"nnls_x{i}"] = v
ref["nnls_res"] = nr
ref["quad"] = integrate.quad(lambda x: np.sin(x * x) * np.exp(-x), 0, 4, epsabs=1e-13)[0]
ref["brentq"] = optimize.brentq(lambda x: np.cos(x) - x ** 3, 0, 1, xtol=1e-14)
cx = np.arange(30) * 0.2
cy = 2.0 * np.exp(-0.7 * cx) + 0.3 + 0.01 * np.sin(cx * 37.0)
popt, _ = optimize.curve_fit(lambda x, a, b, c: a * np.exp(-b * x) + c, cx, cy, p0=[1.0, 1.0, 0.0])
for i, v in enumerate(popt): ref[f"curvefit_p{i}"] = v
ref["curvefit_sse"] = float(((popt[0] * np.exp(-popt[1] * cx) + popt[2] - cy) ** 2).sum())
X = np.array([[i / 5.0, ((i * 7) % 11) / 3.0, np.sin(i * 0.37)] for i in range(30)])
Y = 1.5 + 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2] + 0.1 * np.cos(X[:, 0] * 13.0)
ols = LinearRegression().fit(X, Y)
for i, v in enumerate(ols.coef_): ref[f"ols_c{i}"] = v
ref["ols_b"] = ols.intercept_
rd = Ridge(alpha=2.5).fit(X, Y)
for i, v in enumerate(rd.coef_): ref[f"ridge_c{i}"] = v
ref["ridge_b"] = rd.intercept_
pca = PCA(n_components=2).fit(X)
for i, v in enumerate(pca.explained_variance_): ref[f"pca_var{i}"] = v
for i, v in enumerate(pca.explained_variance_ratio_): ref[f"pca_ratio{i}"] = v
iso = IsotonicRegression().fit_transform(np.arange(1, 9, dtype=float), [1.0, 3.0, 2.0, 4.0, 3.5, 6.0, 5.0, 8.0])
for i, v in enumerate(iso): ref[f"iso_{i}"] = v
m = sig.mean(); c0 = ((sig - m) ** 2).sum()
r = [((sig[k:] - m) * (sig[:len(sig) - k] - m)).sum() / c0 for k in range(9)]
for k, v in enumerate(r): ref[f"acf_{k}"] = v
for k in range(0, 7):
    if k == 0:
        ref["pacf_0"] = 1.0
        continue
    toe = np.array([[r[abs(i - j)] for j in range(k)] for i in range(k)])
    ref[f"pacf_{k}"] = np.linalg.solve(toe, np.array(r[1:k + 1]))[-1]

worst, bad = 0.0, []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 2e-6:
        bad.append((k, rust[k], want))
pk, _ = signal.find_peaks(sig, height=-10.0)
if text.get("peaks") != ",".join(map(str, pk)):
    bad.append(("peaks", text.get("peaks"), ",".join(map(str, pk))))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
for k, got, want in bad[:40]:
    print(f"MISMATCH {k}: rust={got} reference={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
