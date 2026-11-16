"""Batch 17: crates/oxideon-core/src/special_ext.rs vs scipy.special."""
import math, re, sys, warnings
import numpy as np
from scipy import special as sp
warnings.filterwarnings("ignore")

vals = {}
for line in open(sys.argv[1]):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    p = line.split()
    if len(p) == 2:
        try: vals[p[0]] = float(p[1])
        except ValueError: pass
bad = n = 0
worst = {}
def cmp(key, want, tol=1e-12, floor=1e-300):
    global bad, n
    if key not in vals: return
    got = vals[key]; n += 1
    fam = key.split("_")[0]
    if math.isnan(want) and math.isnan(got): return
    if math.isinf(want) and got == want: return
    rel = abs(got - want) / max(abs(want), floor) if want != 0 else abs(got)
    worst[fam] = max(worst.get(fam, 0), rel)
    if rel > tol and abs(got - want) > 1e-300:
        bad += 1; print("MISMATCH", key, got, want, f"rel={rel:.2e}")

xs = [1e-8, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.9, 1.0, 1.5, 1.99, 2.0, 2.01, 3.0, 5.0, 8.0, 12.0, 20.0, 26.0, 40.0, 100.0]
fmt = lambda x: f"{x:e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
# Rust prints `{:e}` as e.g. 1e-8, 1e0, 1.5e0 ; reproduce
def rs(x):
    s = f"{x:e}"; m, e = s.split("e"); m = m.rstrip("0").rstrip("."); return f"{m}e{int(e)}"
for x in xs:
    k = rs(x)
    cmp(f"erfc_{k}", sp.erfc(x), 1e-13); cmp(f"erfcx_{k}", sp.erfcx(x), 1e-13); cmp(f"e1_{k}", sp.exp1(x), 1e-13)
    s, c = sp.fresnel(x)
    cmp(f"fresS_{k}", s, 1e-12); cmp(f"fresC_{k}", c, 1e-12)
    cmp(f"dawsn_{k}", sp.dawsn(x), 1e-12)
    si, ci = sp.sici(x)
    cmp(f"Si_{k}", si, 1e-12); cmp(f"Ci_{k}", ci, 1e-12)
    cmp(f"spence_{k}", sp.spence(x), 1e-12)
    cmp(f"lw0_{k}", sp.lambertw(x, 0).real, 1e-13)
for x in [-5.0, -2.0, -1.0, -0.5, -0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 40.0, 100.0]:
    k = rs(x)
    cmp(f"ei_{k}", sp.expi(x), 1e-13); cmp(f"erfc_{k}", sp.erfc(x), 1e-13); cmp(f"erfcx_{k}", sp.erfcx(x), 1e-13)
    cmp(f"fresS_{k}", sp.fresnel(x)[0], 1e-12); cmp(f"dawsn_{k}", sp.dawsn(x), 1e-12)
for m in [-10.0, -1.0, -0.3, 0.0, 0.1, 0.5, 0.9, 0.99, 0.999999]:
    k = rs(m)
    cmp(f"ellipk_{k}", sp.ellipk(m), 1e-13); cmp(f"ellipe_{k}", sp.ellipe(m), 1e-13)
for x in [-0.35, -0.3, -0.2, -0.05, -1e-3]:
    k = rs(x)
    cmp(f"lwm1_{k}", sp.lambertw(x, -1).real, 1e-12); cmp(f"lw0_{k}", sp.lambertw(x, 0).real, 1e-12)
for nn in [0, 1, 2, 5, 10, -3]:
    for x in [0.1, 1.0, 3.5, 10.0, 25.0, 60.0]:
        k = rs(x)
        cmp(f"jn_{nn}_{k}", sp.jv(nn, x), 1e-11, floor=1e-3)   # absolute-ish near zeros
        cmp(f"ive_{nn}_{k}", sp.ive(abs(nn), x), 1e-11)
for nn, x in [(10, 0.1), (25, 0.5), (3, 1e-6), (40, 2.0), (0, 30.0), (1, 100.0), (2, 1e4), (5, 1e6), (0, 1e9), (3, 1e12), (7, 80.0), (3, -2.5), (-3, -40.0), (4, -1e5)]:
    # pure relative error for the series region; near-zero-crossing values elsewhere get a small absolute floor
    cmp(f"jnx_{nn}_{rs(x)}", sp.jv(nn, x), 1e-10, floor=1e-300 if x * x <= 4 * (abs(nn) + 1) else 1e-4 / math.sqrt(abs(x)))
for v, x in [(40.0, 1.0), (100.0, 1.0), (150.0, 5.0), (60.5, 0.3)]:
    cmp(f"kvex_{rs(v)}_{rs(x)}", sp.kve(v, x), 1e-11)
cmp("inf_fresS", 0.5); cmp("inf_fresCneg", -0.5); cmp("inf_Si", math.pi / 2); cmp("inf_Ci", 0.0)
cmp("inf_lw", float("inf")); cmp("inf_ellipk", 0.0)
for v in [0.0, 0.5, 1.0, 2.5, 7.0]:
    for x in [0.01, 0.5, 1.0, 5.0, 30.0]:
        cmp(f"kve_{rs(v)}_{rs(x)}", sp.kve(v, x), 1e-12)
for nn in [0, 1, 2, 3, 6, 10, 25]:
    for x in [-0.9, -0.3, 0.2, 0.7, 1.0, 1.7]:
        k = f"{nn}_{rs(x)}"
        cmp(f"leg_{k}", sp.eval_legendre(nn, x), 1e-11, floor=1e-3)
        cmp(f"herm_{k}", sp.eval_hermite(nn, x), 1e-11, floor=1e-3)
        cmp(f"herme_{k}", sp.eval_hermitenorm(nn, x), 1e-11, floor=1e-3)
        cmp(f"lag_{k}", sp.eval_laguerre(nn, x), 1e-11, floor=1e-3)
        cmp(f"glag_{k}", sp.eval_genlaguerre(nn, 1.5, x), 1e-11, floor=1e-3)
        cmp(f"cheT_{k}", sp.eval_chebyt(nn, x), 1e-11, floor=1e-3)
        cmp(f"cheU_{k}", sp.eval_chebyu(nn, x), 1e-11, floor=1e-3)
        cmp(f"geg_{k}", sp.eval_gegenbauer(nn, 0.75, x), 1e-11, floor=1e-3)
        cmp(f"jac_{k}", sp.eval_jacobi(nn, 0.5, 1.5, x), 1e-11, floor=1e-3)
print(f"{n} comparisons, {bad} mismatches")
print("worst relative error per family:", {k: f"{v:.1e}" for k, v in sorted(worst.items())})
sys.exit(1 if bad else 0)
