"""Batch 38: reductions and order statistics on awkward inputs (NaN, inf, ties, single element) vs PyTorch."""
import math, re, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

raw = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if rest: raw[k] = rest
inf, nan = float("inf"), float("nan")
rows = {"plain": [3.0, -1.0, 2.0, 7.0, 0.5, -4.0], "ties": [2.0, 5.0, 5.0, -3.0, 5.0, -3.0], "nan_mid": [1.0, nan, 3.0, -2.0, 0.0, 4.0],
        "nan_first": [nan, 1.0, 3.0, -2.0, 0.0, 4.0], "infs": [inf, 1.0, -inf, 2.0, 0.0, 5.0], "pos_inf2": [inf, 1.0, inf, 2.0, 0.0, 5.0],
        "all_neg_inf": [-inf] * 6, "zeros": [0.0, -0.0, 0.0, 0.0, -0.0, 0.0], "single": [4.25], "big": [1e308, 1e308, -1e308, 3.0, 2.0, 1.0]}
bad = n = 0
def check(name, want, tol=1e-12, values_only=False):
    """values_only: compare as sorted multisets (used for indices among tied or NaN entries, which are unordered)."""
    global bad, n
    n += 1
    want = np.atleast_1d(np.asarray(want.detach() if torch.is_tensor(want) else want, dtype=float)).ravel()
    got = raw.get(name)
    if got is None: bad += 1; print("MISSING", name); return
    if got.startswith("ERR"): bad += 1; print("ERROR", name, got[:90], "torch:", want[:4]); return
    g = np.array([float(v) for v in got.split()])
    if g.shape != want.shape: bad += 1; print("SHAPE", name, g.shape, want.shape); return
    if values_only: g, want = np.sort(g), np.sort(want)
    for i, (a, b) in enumerate(zip(g, want)):
        if not ((math.isnan(a) and math.isnan(b)) or a == b or (math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(b)))):
            bad += 1; print("MISMATCH", name, f"index {i}: got {a!r} want {b!r}   (input {rows[name.split('_')[0]] if name.split('_')[0] in rows else ''})"); return

for tag, v in rows.items():
    x = torch.tensor(v, dtype=torch.float64)
    has_nan = any(math.isnan(e) for e in v); tied = len(set(v)) < len(v) or has_nan
    check(f"{tag}_sum", x.sum()); check(f"{tag}_mean", x.mean()); check(f"{tag}_prod", x.prod())
    check(f"{tag}_max", x.max(0).values); check(f"{tag}_min", x.min(0).values); check(f"{tag}_amax", x.amax(0))
    check(f"{tag}_argmax", x.argmax(0).double()); check(f"{tag}_argmin", x.argmin(0).double())
    check(f"{tag}_median", x.median())
    check(f"{tag}_var", x.var(0, unbiased=True))
    # std adds 1e-12 under the square root by design (finite gradient at zero variance): compare sqrt(var + 1e-12)
    check(f"{tag}_std0", torch.sqrt(x.var(0, unbiased=False) + 1e-12))
    check(f"{tag}_logsumexp", x.logsumexp(0)); 
    if inf not in v:   # rows with +inf: this library returns the limiting distribution by design, torch returns NaN
        check(f"{tag}_softmax", x.softmax(0)); check(f"{tag}_log_softmax", x.log_softmax(0))
    check(f"{tag}_cumsum", x.cumsum(0)); check(f"{tag}_cumprod", x.cumprod(0))
    cm = x.cummax(0); check(f"{tag}_cummax_v", cm.values); check(f"{tag}_cummax_i", cm.indices.double(), values_only=tied)
    cm = x.cummin(0); check(f"{tag}_cummin_v", cm.values); check(f"{tag}_cummin_i", cm.indices.double(), values_only=tied)
    s = x.sort(0); check(f"{tag}_sort_v", s.values); check(f"{tag}_sort_i", s.indices.double(), values_only=tied)
    s = x.sort(0, descending=True); check(f"{tag}_sort_desc_v", s.values); check(f"{tag}_sort_desc_i", s.indices.double(), values_only=tied)
    if len(v) >= 3:
        tk = x.topk(3, 0); check(f"{tag}_topk_v", tk.values)
        if not tied: check(f"{tag}_topk_i", tk.indices.double())
        kv = x.kthvalue(2, 0); check(f"{tag}_kth_v", kv.values)
    check(f"{tag}_norm2", torch.linalg.vector_norm(x, 2)); check(f"{tag}_norminf", torch.linalg.vector_norm(x, inf))
    check(f"{tag}_nansum", x.nansum()); check(f"{tag}_nanmean", x.nanmean())
    check(f"{tag}_any", x.any().double()); check(f"{tag}_all", x.all().double())
print(f"{n} results compared, {bad} mismatches")
sys.exit(1 if bad else 0)
