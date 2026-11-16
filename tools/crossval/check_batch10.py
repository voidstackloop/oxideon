"""Tenth cross-validation batch: general tensor ops vs PyTorch.

    cargo test -p oxideon-core --test crossval_emit10 -- --nocapture --ignored > rust_values10.txt
    py -3.11 tools/crossval/check_batch10.py rust_values10.txt
"""
import math
import sys

import numpy as np
import torch

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass

D = torch.float64
T = lambda v, shape=None: (torch.tensor(v, dtype=D).reshape(shape) if shape else torch.tensor(v, dtype=D))
I = lambda v, shape=None: (torch.tensor(v, dtype=torch.int64).reshape(shape) if shape else torch.tensor(v, dtype=torch.int64))
ref = {}


def put(name, x):
    for i, v in enumerate(torch.as_tensor(x).to(D).reshape(-1).tolist()):
        ref[f"{name}_{i}"] = v


a = T([0.5, -1.2, 3.3, 2.0, -0.7, 1.1], (2, 3))
b = T([1.5, 0.4, -2.2, 0.9, 3.0, -0.1], (2, 3))
put("logaddexp", torch.logaddexp(a, b))
put("logaddexp2", torch.logaddexp2(a, b))
put("float_power", torch.float_power(a.abs(), b))
put("ldexp", torch.ldexp(a, I([1, -2, 3, 0, 4, -1], (2, 3))))
m_, e_ = torch.frexp(a)
put("frexp_m", m_)
put("frexp_e", e_)
put("nextafter", torch.nextafter(T([1.0, -1.0, 0.0, 2.5]), T([2.0, -2.0, 1.0, -3.0])))
put("gcd", torch.gcd(I([12, -18, 7, 0, 100]), I([18, 12, 5, 9, 75])))
put("lcm", torch.lcm(I([4, 6, 7, 3]), I([6, 8, 5, 0])))
put("hypot", torch.hypot(a, b))
put("copysign", torch.copysign(a, b))
put("trapezoid", torch.trapezoid(b, dx=0.5, dim=1))
put("cumtrapz", torch.cumulative_trapezoid(b, dx=0.5, dim=1))
for k in [1, 2, 3, -1]:
    put(f"rot90_{k if k > 0 else 'neg' + str(-k)}", torch.rot90(a, k, (0, 1)))
put("take_along_dim", torch.take_along_dim(a, I([2, 0, 1, 1, 1, 0], (2, 3)), dim=1))
z = torch.zeros(6, dtype=D)
put("index_put_acc", z.index_put((I([1, 3, 1, 5]),), T([1.0, 2.0, 3.0, 4.0]), accumulate=True))
put("index_copy", torch.zeros(3, 2, dtype=D).index_copy(0, I([2, 0]), T([1.0, 2.0, 3.0, 4.0], (2, 2))))
idx = I([0, 1, 0, 1, 2, 1, 0, 2, 2], (3, 3))
src = T([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (3, 3))
base = T([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], (3, 3))
for r in ["sum", "prod", "mean", "amax", "amin"]:
    for inc in [True, False]:
        put(f"scatter_reduce_{r}_{str(inc).lower()}", base.scatter_reduce(0, idx, src, reduce=r, include_self=inc))
put("unique", torch.unique(T([3.0, 1.0, 2.0, 3.0, 1.0, 5.0])))
bounds, vals = T([1.0, 2.5, 4.0]), T([0.5, 1.0, 2.5, 3.0, 4.0, 5.0])
put("bucketize_left", torch.bucketize(vals, bounds, right=False))
put("bucketize_right", torch.bucketize(vals, bounds, right=True))
put("searchsorted_left", torch.searchsorted(bounds, vals, right=False))
put("searchsorted_right", torch.searchsorted(bounds, vals, right=True))
put("tril_idx", torch.tril_indices(4, 3, 0))
put("tril_idx_off", torch.tril_indices(3, 4, 1))
put("triu_idx", torch.triu_indices(4, 3, 0))
put("triu_idx_off", torch.triu_indices(3, 4, -1))
put("combinations", torch.combinations(T([1.0, 2.0, 3.0, 4.0]), 2))
put("combinations_rep", torch.combinations(T([1.0, 2.0, 3.0]), 2, with_replacement=True))
put("renorm", a.renorm(2, 0, 1.5))
x8 = T([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.0, 3.5], (2, 4))
v_, i_ = x8.topk(2, 1, largest=True)
put("topk_v", v_); put("topk_i", i_)
v_, i_ = x8.topk(3, 1, largest=False)
put("topk_small_v", v_); put("topk_small_i", i_)
v_, i_ = x8.kthvalue(2, 1)
put("kth_v", v_); put("kth_i", i_)
put("diff2", torch.diff(T([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]), n=2))
put("bincount", torch.bincount(I([0, 1, 1, 3, 3, 3, 6])))
put("bincount_w", torch.bincount(I([0, 1, 1, 3]), weights=T([0.5, 1.0, 2.0, 0.25]), minlength=6))
put("histc", torch.histc(T([0.1, 0.5, 0.9, 1.4, 2.2, 2.9, 3.0, 3.5]), bins=4, min=0.0, max=3.5))
cm = T([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], (2, 4))
v_, i_ = torch.cummax(cm, 1)
put("cummax_v", v_); put("cummax_i", i_)
v_, i_ = torch.cummin(cm, 1)
put("cummin_v", v_); put("cummin_i", i_)
put("logcumsumexp", torch.logcumsumexp(cm, 1))
put("cross", torch.linalg.cross(T([1.0, 2.0, 3.0, -1.0, 0.5, 2.0], (2, 3)), T([0.5, -1.0, 2.0, 3.0, 1.0, -2.0], (2, 3)), dim=1))
t3 = T([math.sin(i * 0.37) for i in range(24)], (2, 3, 4))
t2 = T([math.cos(i * 0.21) for i in range(12)], (3, 4))
put("tensordot", torch.tensordot(t3, t2, dims=([1, 2], [0, 1])))
put("block_diag", torch.block_diag(T([1.0, 2.0, 3.0, 4.0], (2, 2)), T([5.0, 6.0, 7.0], (1, 3))))
put("vander", torch.vander(T([1.0, 2.0, 3.0]), N=4, increasing=True))
put("cartesian", torch.cartesian_prod(T([1.0, 2.0]), T([3.0, 4.0, 5.0])))
put("kron", torch.kron(T([1.0, 2.0, 3.0, 4.0], (2, 2)), T([0.0, 5.0, 6.0, 7.0], (2, 2))))
obs = T([((i * 7 % 11) / 3.0) - 1.0 for i in range(15)], (3, 5))
put("cov", torch.cov(obs, correction=1))
put("cov0", torch.cov(obs, correction=0))
put("corrcoef", torch.corrcoef(obs))
put("cdist", torch.cdist(T([0.0, 0.0, 1.0, 2.0, -1.0, 0.5], (3, 2)), T([1.0, 1.0, 0.0, -2.0], (2, 2))))
e1 = T([math.sin(i * 0.3) for i in range(12)], (3, 4))
e2 = T([math.cos(i * 0.5) for i in range(8)], (4, 2))
put("einsum_mm", torch.einsum("ij,jk->ik", e1, e2))
put("einsum_trace", torch.einsum("ii->", T([1.0, 2.0, 3.0, 4.0], (2, 2))))
put("einsum_outer", torch.einsum("i,j->ij", T([1.0, 2.0]), T([3.0, 4.0, 5.0])))
put("einsum_batch", torch.einsum("bij,bjk->bik", T([i * 0.1 for i in range(12)], (2, 2, 3)), T([1.0 - i * 0.05 for i in range(12)], (2, 3, 2))))
put("nanmedian", torch.nanmedian(T([3.0, float("nan"), 1.0, 2.0, 7.0])))

worst, bad = 0.0, []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 1e-6:
        bad.append((k, rust[k], want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
groups = {}
for k, got, want in bad:
    groups.setdefault(k.rsplit("_", 1)[0], []).append((k, got, want))
for g, items in groups.items():
    k, got, want = items[0]
    print(f"MISMATCH {g}: {len(items)} values differ, e.g. {k}: rust={got} torch={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
