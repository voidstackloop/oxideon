"""Batch 37: elementwise ops at edge values (halves, signed zeros, inf, NaN, subnormals, zero divisors) vs PyTorch."""
import math, re, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

inf, nan = float("inf"), float("nan")
xs = [-2.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 2.5, 3.7, -3.7, 0.49999999999999994, 4503599627370497.5, inf, -inf, nan, 1e-320, -1e-320, 1e308]
x = torch.tensor(xs, dtype=torch.float64)
bad = n = 0
def check(name, want, signed_zero=True, tol=1e-12):
    global bad, n
    n += 1
    want = want.double().numpy().ravel(); got = arrs[name]
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    for i, (g, w) in enumerate(zip(got, want)):
        ok = (math.isnan(g) and math.isnan(w)) or g == w or (math.isfinite(w) and abs(g - w) <= tol * max(1.0, abs(w)))
        if ok and signed_zero and g == 0 and w == 0 and math.copysign(1, g) != math.copysign(1, w): ok = False
        if not ok:
            bad += 1; print("MISMATCH", name, f"index {i}: got {g!r} want {w!r}"); break

for name, f in [("round", torch.round), ("floor", torch.floor), ("ceil", torch.ceil), ("trunc", torch.trunc), ("frac", torch.frac),
                ("sign", torch.sign), ("signbit", torch.signbit), ("abs", torch.abs), ("neg", torch.neg), ("recip", torch.reciprocal),
                ("sqrt", torch.sqrt), ("exp", torch.exp), ("ln", torch.log), ("log1p", torch.log1p), ("expm1", torch.expm1),
                ("tanh", torch.tanh), ("sigmoid", torch.sigmoid), ("relu", torch.relu), ("isfinite", torch.isfinite),
                ("isinf", torch.isinf), ("isnan", torch.isnan), ("square", torch.square)]:
    check(name, f(x))
check("nan_to_num", torch.nan_to_num(x, 7.0, 1e9, -1e9)); check("heaviside", torch.heaviside(x, torch.tensor([0.25], dtype=torch.float64)))
check("clamp", torch.clamp(x, -1.0, 2.0)); check("count_nonzero", torch.count_nonzero(x).reshape(1))

nums = [-7.5, -3.0, -0.0, 0.0, 2.0, 5.5, 7.0, inf, -inf, nan]; dens = [-2.0, -0.5, 0.0, -0.0, 0.3, 2.0, inf, nan]
a = torch.tensor([p for p in nums for q in dens], dtype=torch.float64); b = torch.tensor([q for p in nums for q in dens], dtype=torch.float64)
for name, f in [("div", torch.div), ("floor_divide", torch.floor_divide), ("fmod", torch.fmod), ("remainder", torch.remainder), ("pow", torch.pow),
                ("atan2", torch.atan2), ("maximum", torch.maximum), ("minimum", torch.minimum), ("copysign", torch.copysign), ("hypot", torch.hypot),
                ("eq", torch.eq), ("lt", torch.lt), ("le", torch.le), ("gt", torch.gt), ("ge", torch.ge)]:
    # the sign of a zero result from max / min of (+0, -0) is unspecified in IEEE 754 and differs between libraries
    check(name, f(a, b), signed_zero=name not in ("maximum", "minimum"))
ia = torch.tensor([-7, -7, 7, 7, -8, 8, 0, 5, -5, 9]); ib = torch.tensor([2, -2, 2, -2, 3, -3, 5, 5, 5, 4])
check("i_floor_divide", torch.floor_divide(ia, ib)); check("i_remainder", torch.remainder(ia, ib)); check("i_fmod", torch.fmod(ia, ib))
check("i_and", torch.bitwise_and(ia, ib)); check("i_xor", torch.bitwise_xor(ia, ib))
tv = torch.tensor([7, 0, 9])
check("isin", torch.isin(ia, tv)); check("isin_inv", torch.isin(ia, tv, invert=True))
check("nonzero", torch.nonzero(ia.reshape(2, 5))); check("count_nz_axis", torch.count_nonzero(ia.reshape(2, 5), dim=1))
print(f"{n} ops compared, {bad} mismatches")
sys.exit(1 if bad else 0)
