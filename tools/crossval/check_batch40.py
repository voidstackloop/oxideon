"""Batch 40: closed-form / small-algorithm engines in `ode` vs SciPy / NumPy / quadrature."""
import math, re, sys, warnings
import numpy as np
from scipy import linalg, integrate, stats
warnings.filterwarnings("ignore")
arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass
bad = n = 0
def check(name, want, tol=1e-9):
    global bad, n
    n += 1
    want = np.asarray(want, dtype=float).ravel(); got = arrs.get(name)
    if got is None or got.shape != want.shape: bad += 1; print("SHAPE/MISSING", name, None if got is None else got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1, np.abs(want)))
    if not err <= tol: bad += 1; print("MISMATCH", name, err, got, want)

# Kalman-Bucy: P solves 2aP + q - (c^2/r) P^2 = 0 (dual ARE), K = P c / r
want = []
for a, c, q, r in [(-0.5, 1.0, 2.0, 0.7), (0.8, 2.0, 0.3, 1.5), (0.0, 1.0, 1.0, 1.0), (3.0, 0.5, 4.0, 0.2)]:
    P = linalg.solve_continuous_are(np.array([[a]]).T, np.array([[c]]).T, np.array([[q]]), np.array([[r]]))[0, 0]
    want += [P, P * c / r]
check("kb", want)

# H-infinity scalar ARE from the stable invariant subspace of the Hamiltonian matrix
def hinf(a, b1, b2, q, r, g):
    G = b2 * b2 / r - b1 * b1 / g ** 2
    H = np.array([[a, -G], [-q, -a]]); w, V = np.linalg.eig(H)
    v = V[:, np.argmin(w.real)]
    return (v[1] / v[0]).real
check("hinf", [hinf(-0.5, 0.4, 1.0, 2.0, 0.7, 2.0), hinf(0.8, 0.2, 1.5, 1.0, 1.0, 3.0)])

# Malliavin weights: E[f(S_T) w_delta] = d/dS0 E f, E[f(S_T) w_gamma] = d2/dS0^2 E f (Black-Scholes call, r = 0.05)
S0, sig, T, K, r = 100.0, 0.3, 1.5, 95.0, 0.05
ws = np.linspace(-12, 12, 6001); dens = stats.norm.pdf(ws, scale=math.sqrt(T))
ST = S0 * np.exp((r - sig ** 2 / 2) * T + sig * ws); pay = np.maximum(ST - K, 0) * math.exp(-r * T)
mw = arrs["malliavin"].reshape(-1, 2)
d1 = (math.log(S0 / K) + (r + sig ** 2 / 2) * T) / (sig * math.sqrt(T))
bs_delta = stats.norm.cdf(d1); bs_gamma = stats.norm.pdf(d1) / (S0 * sig * math.sqrt(T))
dw = ws[1] - ws[0]
for nm, col, want_v in [("delta", 0, bs_delta), ("gamma", 1, bs_gamma)]:
    n += 1
    got = np.sum(pay * mw[:, col] * dens) * dw
    if abs(got - want_v) > 2e-4 * max(1, abs(want_v)): bad += 1; print("MISMATCH malliavin", nm, got, want_v)

# McKean-Vlasov variance ODE dV/dt = -2 k V + s^2
k_, s_ = 0.8, 0.5
sol = integrate.solve_ivp(lambda t, v: -2 * k_ * v + s_ ** 2, [0, 4], [2.0], t_eval=[0, 0.3, 1.0, 4.0], rtol=1e-12, atol=1e-14)
check("mckean", sol.y[0], 1e-9)

# Levy exponent: Re Psi(u) = -sigma^2 u^2 / 2 + lam (E cos(u J) - 1), J ~ N(mu, s^2) by quadrature
sg, lam, mu, js = 0.4, 1.3, 0.2, 0.6
def re_psi(u):
    ec = integrate.quad(lambda x: math.cos(u * x) * stats.norm.pdf(x, mu, js), mu - 12 * js, mu + 12 * js, epsabs=1e-13, epsrel=1e-13, limit=200)[0]
    return -0.5 * sg * sg * u * u + lam * (ec - 1)
check("levy", [re_psi(u) for u in (0.0, 0.5, 1.0, 2.5, 6.0)], 1e-9)

# Step-2 signature of a piecewise-linear path: S2_ij = sum_{s<t} dx_i(s) dx_j(t) + 1/2 sum_s dx_i(s) dx_j(s)
pts = np.array([[math.sin(0.9 * t), math.cos(0.5 * t) * t * 0.3, math.sin(1.7 * t) * 0.5] for t in range(7)])
dx = np.diff(pts, axis=0); cum = np.cumsum(dx, axis=0) - dx
S2 = cum.T @ dx + 0.5 * dx.T @ dx
check("sig1", dx.sum(0)); check("sig2", S2)
check("levy_area", [0.5 * (S2[i, j] - S2[j, i]) for i, j in [(0, 1), (1, 2), (2, 0)]])

# Sontag universal formula, with the closed-loop Lyapunov rate a + b u = -sqrt(a^2 + b^4)
w = []
for a, b in [(1.0, 2.0), (-1.0, 0.5), (3.0, -1.5), (0.0, 1.0), (2.0, 0.0)]:
    w.append(0.0 if b == 0 else -(a + math.sqrt(a * a + b ** 4)) / b)
    if b != 0: assert abs(a + b * w[-1] + math.sqrt(a * a + b ** 4)) < 1e-12
check("sontag", w)

# CBF filter: closest point to u_nom in {u : lf + lg u + alpha >= 0}, by brute force
us = np.linspace(-20, 20, 4_000_001); w = []
for un, f, g, al in [(1.0, 0.5, 1.0, 0.2), (-2.0, 0.5, 1.0, 0.2), (0.3, -1.0, -2.0, 0.5), (5.0, 1.0, 2.0, 0.1)]:
    ok = f + g * us + al >= 0; w.append(us[ok][np.argmin(np.abs(us[ok] - un))])
check("cbf", w, 1e-4)

# Pontryagin: maximise drift_dot_p + g u - R u^2 / 2 over |u| <= 1 (R = 0.8) by brute force
us = np.linspace(-1, 1, 2_000_001); w = []
for d, g in [(0.5, 0.3), (0.5, 2.0), (-1.0, -3.0)]:
    H = d + g * us - 0.5 * 0.8 * us ** 2; i = np.argmax(H); w += [us[i], H[i]]
check("pontryagin", w, 1e-6)

# Stein-Chen: formula, and it must dominate the exact TV distance between Poisson-binomial and Poisson(lambda)
bounds = []
for ps in ([0.1, 0.2, 0.05, 0.3, 0.15], [0.5, 0.5]):
    lam = sum(ps); bounds.append((1 - math.exp(-lam)) / lam * sum(p * p for p in ps))
    pm = np.array([1.0])
    for p in ps: pm = np.convolve(pm, [1 - p, p])
    pois = stats.poisson.pmf(np.arange(200), lam)
    tv = 0.5 * np.abs(np.pad(pm, (0, 200 - len(pm))) - pois).sum()
    n += 1
    if tv > bounds[-1] + 1e-12: bad += 1; print("BOUND VIOLATED", ps, tv, bounds[-1])
check("steinchen", bounds)
check("fourth", [2 * math.sqrt((q - 1) / (3 * q) * abs(m4 - 3)) for q, m4 in [(2, 3.5), (4, 4.2)]])

# Koopman diagonal EDMD vs per-column least squares, and K^5 prediction
px = np.array([[math.sin(0.3 * i), math.cos(0.7 * i) + 0.2, 0.1 * i] for i in range(20)])
py = np.column_stack([0.9 * px[:, 0] + 0.01, -0.4 * px[:, 1] + 0.02, 1.1 * px[:, 2]])
kd = [np.linalg.lstsq(px[:, [j]], py[:, j], rcond=None)[0][0] for j in range(3)]
Km = np.diag(kd); check("koopman_k", Km); check("koopman_pred", np.linalg.matrix_power(Km, 5) @ np.array([1.0, 2.0, -1.5]))

# Birkhoff average of the logistic map (invariant mean 1/2), loose tolerance
n += 1
if abs(arrs["birkhoff"][0] - 0.5) > 5e-3: bad += 1; print("BIRKHOFF", arrs["birkhoff"])

# Heisenberg: a loop enclosing area z has length 2 sqrt(pi z) (isoperimetric); Eyring-Kramers prefactor
check("carnot", [2 * math.sqrt(math.pi * 2.5), 2 * math.sqrt(math.pi * 0.7)])
check("ek", [2 * math.pi / math.sqrt(2.0 * 3.0) * math.exp(1.0 / 0.1)])

# Mean-field game, one explicit step on a periodic ring (mass must stay 1)
N = 16; pot = np.cos(2 * np.pi * np.arange(N) / N)
m0 = 1 + 0.3 * np.sin(np.arange(N) * 0.8); m0 /= m0.sum(); u0 = np.cos(np.arange(N) * 0.5)
dt, dxg, nu, gam = 0.01, 0.2, 0.1, 0.5
up, um = np.roll(u0, -1), np.roll(u0, 1)
ux = (up - um) / (2 * dxg); lap = (up - 2 * u0 + um) / dxg ** 2
u1 = u0 + dt * (nu * lap - 0.5 * ux ** 2 + pot + gam * m0); al = -ux
mp, mm = np.roll(m0, -1), np.roll(m0, 1); ap, am = np.roll(al, -1), np.roll(al, 1)
m1 = np.maximum(m0 + dt * (nu * (mp - 2 * m0 + mm) / dxg ** 2 - (mp * ap - mm * am) / (2 * dxg)), 1e-12)
m1 *= m0.sum() / m1.sum()
check("mfg_u1", u1); check("mfg_m1", m1)
print(f"{n} checks, {bad} mismatches"); sys.exit(1 if bad else 0)
