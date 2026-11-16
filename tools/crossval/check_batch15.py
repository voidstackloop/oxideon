"""Batch 15: expected improvement vs scipy.stats.norm, particle filter vs exact Kalman, TPE / Hyperband / BO behaviour."""
import math, re, sys
import numpy as np
from scipy.stats import norm

vals = {}
for line in open(sys.argv[1]):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    p = line.split()
    if len(p) == 2:
        try: vals[p[0]] = float(p[1])
        except ValueError: pass
bad = n = 0
def check(name, ok, detail=""):
    global bad, n
    n += 1
    if not ok:
        bad += 1; print("MISMATCH", name, detail)

# ---- expected improvement (minimization)
for i, (m, v, b) in enumerate([(0,1,.5),(1,.25,0),(-1,4,-.5),(5,.01,0),(10,1,0),(2,.5,8),(.3,1e-4,.35),(-3,1,0),(3,1,0),(4.5,1,0)]):
    sd = math.sqrt(v); z = (b - m) / sd
    want = (b - m) * norm.cdf(z) + sd * norm.pdf(z)
    got = vals[f"ei_{i}"]
    check(f"ei_{i}", abs(got - want) <= 1e-12 + 1e-9 * abs(want), f"{got} vs {want}")
    check(f"ei_{i}_nonneg", got >= 0)

# ---- particle filter vs the exact Kalman filter (agree to Monte-Carlo error)
T = 60
pf = np.array([vals[f"pf_{t}"] for t in range(T)]); kf = np.array([vals[f"kf_{t}"] for t in range(T)])
ys = np.array([vals[f"y_{t}"] for t in range(T)])
a, q, r = 0.9, 0.3 ** 2, 0.5 ** 2
x, p, ref = 0.0, 1.0, []
for y in ys:
    xp, pp = a * x, a * a * p + q
    k = pp / (pp + r); x = xp + k * (y - xp); p = (1 - k) * pp
    ref.append(x)
check("kalman_reference", np.allclose(kf, ref, atol=1e-8), np.abs(kf - ref).max())
err = np.abs(pf - kf)
check("pf_matches_kalman_max", err.max() < 0.03, err.max())
check("pf_matches_kalman_rmse", math.sqrt((err ** 2).mean()) < 0.012, math.sqrt((err ** 2).mean()))

# ---- TPE beats random search at the same budget
check("tpe_beats_random", vals["tpe_mean_best"] < vals["random_mean_best"], f"{vals['tpe_mean_best']} vs {vals['random_mean_best']}")

# ---- Hyperband (R=81, eta=3): brackets s=4..0
# n_i, r_i from the paper, per bracket: n = ceil(5/(s+1)*3^s), r = 81*3^-s; rungs i=0..: n_i=floor(n*3^-i), r_i=r*3^i
tot_trials = tot_budget = full = 0
for s in range(4, -1, -1):
    nn = math.ceil(5 / (s + 1) * 3 ** s); rr = 81 * 3.0 ** -s
    alive, budget = nn, rr
    while True:
        tot_trials += alive; tot_budget += alive * budget
        if budget >= 81 - 1e-9: full += alive
        if budget >= 81 - 1e-9 or alive == 1: break
        alive = max(1, math.floor(alive / 3)); budget = min(budget * 3, 81)
check("hb_ntrials", vals["hb_ntrials"] == tot_trials, f"{vals['hb_ntrials']} vs {tot_trials}")
check("hb_total_budget", abs(vals["hb_total_budget"] - tot_budget) < 1e-6, f"{vals['hb_total_budget']} vs {tot_budget}")
check("hb_full_budget_evals", vals["hb_max_budget_evals"] == full, f"{vals['hb_max_budget_evals']} vs {full}")
check("hb_finds_good_config", vals["hb_best"] < 1.0, vals["hb_best"])

# ---- Bayesian optimization beats random search at an equal budget (30 evaluations)
check("bo_beats_random", vals["bo_mean_best"] < vals["rs30_mean_best"], f"{vals['bo_mean_best']} vs {vals['rs30_mean_best']}")
# true optimum of the BO test function, by dense grid
g = np.linspace(-3, 3, 1201); X, Y = np.meshgrid(g, g)
opt = ((X - .7) ** 2 * 3 + (Y + 1.2) ** 2 + .5 * np.sin(3 * X)).min()
check("bo_close_to_optimum", vals["bo_mean_best"] < opt + 0.1, f"{vals['bo_mean_best']} vs optimum {opt}")
print(f"{n} checks, {bad} mismatches")
sys.exit(1 if bad else 0)
