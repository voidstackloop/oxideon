"""Batch 26: indexing / selection / shape ops, values and gradients, vs PyTorch (float64)."""
import math, re, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")

arrs = {}; nograd = set()
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if not rest: continue
    if rest == "NOGRAD": nograd.add(k); continue
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

D = torch.float64
def base(r, c, salt): return [math.sin(0.77 * k + salt) * 1.5 for k in range(r * c)]
def wts(shape):
    n = int(np.prod(shape)); return torch.tensor([math.sin(1.3 * k + 0.4) for k in range(n)], dtype=D).reshape(shape)
def T(v, shape): return torch.tensor(v, dtype=D).reshape(shape)

xv = base(4, 5, 0.3); xv[1] = 2.5; xv[3] = 2.5; xv[10] = -2.0; xv[12] = -2.0
xt = base(4, 5, 0.3); yv = base(4, 5, 1.7)
mask = torch.tensor([k % 3 == 0 for k in range(20)]).reshape(4, 5)
I = lambda v, shape: torch.tensor(v, dtype=torch.long).reshape(shape)

bad = n = 0; worst = 0.0
def check(name, got, want, tol=1e-9):
    global bad, n, worst
    want = want.detach().double().flatten().numpy(); n += 1
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    both_nan = np.isnan(got) & np.isnan(want)
    g, w = np.where(both_nan, 0, got), np.where(both_nan, 0, want)
    err = np.max(np.abs(g - w) / np.maximum(1.0, np.abs(w))) if len(w) else 0.0
    worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:5], want[:5])

def case(name, specs, fn, tol=1e-9):
    leaves = [T(v, s).requires_grad_(True) for v, s in specs]
    outs = fn(*leaves)
    if not isinstance(outs, (list, tuple)): outs = [outs]
    loss = sum((o * wts(list(o.shape))).sum() for o in outs)
    loss.backward()
    for k, o in enumerate(outs): check(f"{name}_out{k}", arrs[f"{name}_out{k}"], o, tol)
    for k, x in enumerate(leaves):
        if f"{name}_g{k}" in nograd:
            bad_ = x.grad is not None and bool((x.grad != 0).any());
            if bad_: print("NOGRAD but torch has gradient:", name, k)
            globals()["bad"] += int(bad_); continue
        check(f"{name}_g{k}", arrs[f"{name}_g{k}"], x.grad if x.grad is not None else torch.zeros_like(x), tol)

X = (xv, (4, 5)); XT = (xt, (4, 5)); Y = (yv, (4, 5))
case("gather", [X], lambda x: x.gather(1, I([0, 4, 2, 1, 3, 3, 0, 0, 2, 2, 4, 1], (4, 3))))
case("index_select", [X], lambda x: x.index_select(0, I([2, 0, 3, 0], (4,))))
case("masked_fill", [X], lambda x: x.masked_fill(mask, 7.0))
case("where", [X, Y], lambda x, y: torch.where(mask, x, y))
case("masked_select", [X], lambda x: x.masked_select(mask))
case("scatter", [X, (base(4, 3, 2.2), (4, 3))], lambda x, s: x.scatter(1, I([0, 4, 2, 1, 3, 0, 4, 0, 2, 1, 2, 3], (4, 3)), s))
case("scatter_add", [X, (base(4, 3, 2.2), (4, 3))], lambda x, s: x.scatter_add(1, I([0, 0, 2, 1, 3, 3, 4, 0, 2, 1, 1, 1], (4, 3)), s))
case("index_add", [X, (base(3, 5, 3.3), (3, 5))], lambda x, s: x.index_add(0, I([1, 3, 1], (3,)), s))
case("index_copy", [X, (base(2, 5, 3.9), (2, 5))], lambda x, s: x.index_copy(0, I([3, 0], (2,)), s))
case("cat0", [X, Y], lambda x, y: torch.cat([x, y], 0)); case("cat1", [X, Y], lambda x, y: torch.cat([x, y], 1))
case("stack", [X, Y], lambda x, y: torch.stack([x, y], 1))
case("repeat", [X], lambda x: x.repeat(2, 3)); case("tile", [X], lambda x: x.tile((2, 1, 2)))
case("repeat_interleave", [X], lambda x: x.repeat_interleave(3, dim=1))
case("tril", [X], lambda x: x.tril(1)); case("triu", [X], lambda x: x.triu(-1))
case("diag_embed", [X], lambda x: torch.diag_embed(x))
case("diagonal", [X], lambda x: x.diagonal(1, 0, 1))
case("sort_desc", [XT], lambda x: x.sort(1, descending=True).values)
case("sort_asc0", [XT], lambda x: x.sort(0).values)
case("topk", [XT], lambda x: x.topk(3, 1, largest=True).values)
case("topk_small", [XT], lambda x: x.topk(2, 0, largest=False).values)
case("cummax", [XT], lambda x: x.cummax(1).values); case("cummin", [XT], lambda x: x.cummin(0).values)
case("kthvalue", [XT], lambda x: x.kthvalue(2, 1).values)
case("median", [XT], lambda x: x.median().reshape(1))
case("max_axis_ties", [X], lambda x: x.max(1).values); case("min_axis_ties", [X], lambda x: x.min(1, keepdim=True).values)
case("amax_ties", [X], lambda x: x.amax(1)); case("amin_ties", [X], lambda x: x.amin(1))
case("narrow", [X], lambda x: x.narrow(1, 1, 3))
case("expand", [(base(1, 5, 0.9), (1, 5))], lambda x: x.expand(4, 5))
case("take", [X], lambda x: x.take(I([19, 0, 7, 7, 3], (5,))))
case("clamp", [X], lambda x: x.clamp(-1.0, 1.2)); case("clamp_edge", [X], lambda x: x.clamp(2.5, 2.5)); case("clamp_min", [X], lambda x: x.clamp(min=0.3))
case("permute", [(base(6, 4, 0.1), (2, 3, 4))], lambda x: x.permute(2, 0, 1))
case("transpose", [X], lambda x: x.transpose(0, 1))
case("movedim", [(base(6, 4, 0.1), (2, 3, 4))], lambda x: x.movedim(0, 2))
case("chunk", [X], lambda x: list(x.chunk(3, 1))); case("split", [X], lambda x: list(x.split(2, 0)))
case("std_axis", [X], lambda x: x.std(1, unbiased=True)); case("var_axis0", [X], lambda x: x.var(0, unbiased=False, keepdim=True))
case("pad", [X], lambda x: F.pad(x, (0, 3, 1, 2), value=0.5))
case("unfold_dim", [X], lambda x: x.unfold(1, 3, 1))
xn = list(xv); xn[7] = float("nan")
case("nanmean", [(xn, (4, 5))], lambda x: x.nanmean().reshape(1)); case("nansum", [(xn, (4, 5))], lambda x: x.nansum().reshape(1))
case("clamp_nan", [(xn, (4, 5))], lambda x: x.clamp(-1.0, 1.0))
case("norm_inf_ties", [X], lambda x: torch.linalg.vector_norm(x, float("inf"), dim=1))
case("aminmax_ties", [X], lambda x: list(torch.aminmax(x, dim=1)))
case("matrix_norm_inf_ties", [([1.0, -2.0, 2.0, 1.0, 0.5, 0.25], (3, 2))], lambda x: torch.linalg.matrix_norm(x, float("inf")).reshape(1))
IMG = (base(2 * 3 * 4, 5, 0.6), (2, 3, 4, 5))
case("interp_nearest", [IMG], lambda x: F.interpolate(x, size=(7, 9), mode="nearest"))
case("interp_bilinear_f", [IMG], lambda x: F.interpolate(x, size=(7, 9), mode="bilinear", align_corners=False))
case("interp_bilinear_t", [IMG], lambda x: F.interpolate(x, size=(6, 8), mode="bilinear", align_corners=True))
case("interp_bicubic", [IMG], lambda x: F.interpolate(x, size=(7, 9), mode="bicubic", align_corners=False))
case("interp_down", [IMG], lambda x: F.interpolate(x, size=(2, 3), mode="bilinear", align_corners=False))
case("interp_area", [IMG], lambda x: F.interpolate(x, size=(2, 3), mode="area"))
print(f"{n} tensors compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
