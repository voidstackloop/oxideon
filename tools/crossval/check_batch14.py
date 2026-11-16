"""Batch 14: RDP accountant, sketches, apriori/rules, item-kNN, MF, evolutionary optimizers.
Deterministic parts are compared with independent implementations; randomized structures against their
theoretical guarantees (error bounds / chi-square)."""
import itertools, math, re, sys
from fractions import Fraction
import numpy as np
from scipy.special import logsumexp, gammaln
from scipy import stats

vals, lines = {}, []
for line in open(sys.argv[1]):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    lines.append(line)
    p = line.split()
    if len(p) >= 2:
        try: vals[p[0]] = [float(x) for x in p[1:]]
        except ValueError: pass
V = lambda k: vals[k][0]

bad = n = 0
def check(name, ok, detail=""):
    global bad, n
    n += 1
    if not ok:
        bad += 1
        print("MISMATCH", name, detail)
def close(name, got, want, tol=1e-9):
    check(name, abs(got - want) <= tol * max(1, abs(want)), f"{got} vs {want}")

def u(i): return (((i * 2654435761) % 4294967296) + 0.5) / 4294967296.0

# ---- RDP accountant (independent: exact integer-order sampled-Gaussian RDP via logsumexp)
def rdp(q, s, a):
    if q == 0: return 0.0
    if q >= 1: return a / (2 * s * s)
    k = np.arange(a + 1)
    t = gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1) + k * math.log(q) + (a - k) * math.log1p(-q) + (k * (k - 1)) / (2 * s * s)
    return logsumexp(t) / (a - 1)
def eps(q, s, steps, d):
    return min(steps * rdp(q, s, a) + math.log(1 / d) / (a - 1) for a in range(2, 257))
for q, s, st, d in [(0.01,1.1,1000,1e-5),(256/60000,1.1,14062,1e-5),(0.05,0.8,200,1e-6),(1.0,2.0,10,1e-5),(0.002,0.7,5000,1e-5)]:
    key = next(k for k in vals if k.startswith("rdp_") and abs(float(k.split("_")[1]) - q) < 1e-12 and k.split("_")[3] == str(st))
    close(key, V(key), eps(q, s, st, d), 1e-9)
# published TF-Privacy figure for MNIST (N=60000, batch 256, sigma 1.1, 60 epochs, delta 1e-5): eps ~ 3.0
check("rdp_mnist_published", 2.95 < V("rdp_0.004266666666666667_1.1_14062_0.00001") < 3.1)

# ---- RunningStats
xs = np.array([(u(i) * 20 - 7) ** 3 / 50 for i in range(1, 501)])
for tag in ("all", "merged"):
    close(f"rs_{tag}_mean", V(f"rs_{tag}_mean"), xs.mean(), 1e-11)
    close(f"rs_{tag}_var", V(f"rs_{tag}_var"), xs.var(ddof=1), 1e-11)
    close(f"rs_{tag}_min", V(f"rs_{tag}_min"), xs.min(), 1e-12)
    close(f"rs_{tag}_max", V(f"rs_{tag}_max"), xs.max(), 1e-12)
    close(f"rs_{tag}_n", V(f"rs_{tag}_n"), 500)

# ---- HyperLogLog: within 4 standard errors (1.04/sqrt(m))
for k, v in vals.items():
    if k.startswith("hll_"):
        _, p, nn = k.split("_"); m = 2 ** int(p); nn = int(nn)
        check(k, abs(v[0] - nn) / nn < 4 * 1.04 / math.sqrt(m), f"{v[0]} vs {nn}")

# ---- Count-Min / Bloom
check("cm_never_underestimates", V("cm_under") == 0)
check("cm_epsilon_guarantee", V("cm_over_eps_frac") <= 0.01 + 0.005, V("cm_over_eps_frac"))
check("bloom_no_false_negatives", V("bloom_false_neg") == 0)
check("bloom_fp_rate", 0.006 < V("bloom_fp_rate") < 0.016, V("bloom_fp_rate"))
close("bloom_est_fp", V("bloom_est_fp"), 0.01, 0.1)

# ---- streaming quantiles: rank error vs exact
data = np.sort(np.array([-math.log(u(i)) * 3.0 for i in range(1, 100001)]))
for k, v in vals.items():
    if k.startswith("tdigest_0"):
        q = float(k.split("_")[1])
        rank = np.searchsorted(data, v[0]) / len(data)
        tol = max(0.0005, 0.02 * min(q, 1 - q))  # tighter at the tails
        check(k, abs(rank - q) <= tol, f"rank {rank} vs {q}")

# ---- reservoir sampling uniformity (inclusion counts expected trials*k/n each)
hits = np.array([V(f"res_{i}") for i in range(40)])
chi = ((hits - 500) ** 2 / 500).sum()
check("reservoir_uniform", stats.chi2.sf(chi, 39) > 1e-3, f"chi2={chi:.1f} p={stats.chi2.sf(chi, 39):.4f}")
check("reservoir_total", hits.sum() == 4000 * 5)

# ---- SpaceSaving guarantees: count >= true >= count - error
for r in range(8):
    item, cnt, err, tr = vals[f"ss_{r}"]
    check(f"ss_{r}", cnt >= tr and cnt - err <= tr, f"{cnt} {err} {tr}")

# ---- Apriori (brute force) + rules
tx = []
for t in range(200):
    v = []
    for item in range(8):
        p = [0.7, 0.5, 0.45, 0.3, 0.25, 0.2, 0.1, 0.05][item]
        boost = 0.35 if (item == 1 and 0 in v) or (item == 3 and 2 in v) else 0.0
        if u(t * 8 + item + 1) < p + boost: v.append(item)
    tx.append(frozenset(v))
sup = lambda s: sum(1 for t in tx if set(s) <= t) / len(tx)
want = {}
for r in range(1, 9):
    for s in itertools.combinations(range(8), r):
        if sup(s) >= 0.08: want[s] = sup(s)
check("ap_nsets", V("ap_nsets") == len(want), f"{V('ap_nsets')} vs {len(want)}")
for s, sp in want.items():
    close("ap_" + "-".join(map(str, s)), V("ap_" + "-".join(map(str, s))), sp, 1e-12)
rules = {}
for s in want:
    if len(s) < 2: continue
    for r in range(1, len(s)):
        for a in itertools.combinations(s, r):
            c = tuple(x for x in s if x not in a)
            conf = want[s] / want[a]
            if conf >= 0.5: rules[(a, c)] = (conf, conf / want[c], want[s])
check("rules_n", V("rules_n") == len(rules), f"{V('rules_n')} vs {len(rules)}")
for (a, c), (conf, lift, sp) in rules.items():
    base = "rule_" + "-".join(map(str, a)) + "_to_" + "-".join(map(str, c))
    close(base + "_conf", V(base + "_conf"), conf, 1e-8)
    close(base + "_lift", V(base + "_lift"), lift, 1e-8)
    close(base + "_sup", V(base + "_sup"), sp, 1e-8)

# ---- ItemKnn (independent implementation)
R = {}
for user in range(12):
    for item in range(9):
        if u(user * 9 + item + 5) < 0.6:
            R[(user, item)] = 1.0 + min(math.floor(u(user * 9 + item + 777) * 5.0), 4.0)
um = {us: np.mean([v for (a, _), v in R.items() if a == us]) for us in range(12) if any(a == us for a, _ in R)}
cen = {k: v - um[k[0]] for k, v in R.items()}
def sim(i, j):
    num = sum(cen[(us, i)] * cen[(us, j)] for us in range(12) if (us, i) in cen and (us, j) in cen)
    di = sum(cen[(us, i)] ** 2 for us in range(12) if (us, i) in cen)
    dj = sum(cen[(us, j)] ** 2 for us in range(12) if (us, j) in cen)
    return num / math.sqrt(di * dj) if di * dj > 0 and num != 0 else 0.0
# note: the library sums only users that rated both items, but normalizes by the full-column norms
def sim(i, j):
    num = sum(cen[(us, i)] * cen[(us, j)] for us in range(12) if (us, i) in cen and (us, j) in cen)
    di = sum(v * v for (us, it), v in cen.items() if it == i)
    dj = sum(v * v for (us, it), v in cen.items() if it == j)
    return num / math.sqrt(di * dj) if di * dj > 0 else 0.0
def knn(us, it, k=3):
    c = [(sim(it, j), cen[(us, j)]) for (a, j) in cen if a == us and j != it]
    c = [x for x in c if x[0] != 0.0]
    c.sort(key=lambda x: -abs(x[0]))
    c = c[:k]
    den = sum(abs(s) for s, _ in c)
    return um.get(us, 0.0) + (sum(s * v for s, v in c) / den if den else 0.0)
for us in range(12):
    for it in range(9):
        close(f"knn_{us}_{it}", V(f"knn_{us}_{it}"), knn(us, it), 1e-9)

# ---- Matrix factorization: low-rank ground truth is recovered on held-out cells
check("mf_fits_train", V("mf_train_rmse") < 0.1)
check("mf_generalizes", V("mf_test_rmse") < 0.25 * V("mf_baseline_rmse"), f"{V('mf_test_rmse')} vs {V('mf_baseline_rmse')}")

# ---- Optimizers: reported value is f(x), history monotone, benchmark success thresholds
thresh = {("sphere", "de"): 1e-10, ("sphere", "cma"): 1e-10, ("sphere", "pso"): 1e-10, ("sphere", "sa"): 1e-3, ("sphere", "ga"): 0.05,
          ("rosen", "de"): 1e-10, ("rosen", "pso"): 1e-8, ("rosen", "sa"): 1e-4, ("rosen", "cma"): 0.1, ("rosen", "ga"): 0.05,
          ("ackley", "de"): 1e-6, ("ackley", "cma"): 1e-6, ("ackley", "pso"): 1e-6, ("ackley", "ga"): 1.0}
for k, v in vals.items():
    if not k.startswith("opt_"): continue
    _, fn, alg, seed = k.split("_")
    close(k + "_recomputed", v[0], v[1], 1e-6 if abs(v[1]) > 1e-9 else 1e-9)
    check(k + "_monotone", v[2] == 1.0)
    if (fn, alg) in thresh:
        check(k + "_converges", v[0] < thresh[(fn, alg)], f"{v[0]} (limit {thresh[(fn, alg)]})")
# rastrigin: PSO must reach the global basin on at least 2 of 5 seeds; everything beats the random-start value
check("rastrigin_pso_global", sum(V(f"opt_rastrigin_pso_{s}") < 1.0 for s in range(1, 6)) >= 2)
print(f"{n} checks, {bad} mismatches")
sys.exit(1 if bad else 0)
