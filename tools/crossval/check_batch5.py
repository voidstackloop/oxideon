"""Fifth cross-validation batch: tensor special functions (values and gradients) vs SciPy / PyTorch.

    cargo test -p oxideon-core --test crossval_emit5 -- --nocapture --ignored > rust_values5.txt
    py -3.11 tools/crossval/check_batch5.py rust_values5.txt
"""
import sys

import numpy as np
import torch
from scipy import special as sp

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass

pos = np.array([0.05, 0.3, 0.9, 1.0, 1.7, 2.5, 4.2, 7.5, 15.0, 40.0])
unit = np.array([-0.97, -0.6, -0.2, 0.0, 0.15, 0.5, 0.8, 0.99])
anyv = np.array([-12.0, -6.5, -3.1, -1.2, -0.3, 0.0, 0.4, 1.3, 3.6, 7.2, 11.0])
ref = {}


def put(name, arr):
    for i, v in enumerate(np.atleast_1d(arr)):
        ref[f"{name}_{i}"] = float(v)


put("erfc", sp.erfc(anyv))
put("erfinv", sp.erfinv(unit))
put("lgamma", sp.gammaln(pos))
put("digamma", sp.digamma(pos))
put("polygamma1", sp.polygamma(1, pos))
put("polygamma2", sp.polygamma(2, pos))
put("polygamma3", sp.polygamma(3, pos))
put("ndtr", sp.ndtr(anyv))
put("log_ndtr", sp.log_ndtr(anyv))
put("ndtri", sp.ndtri(np.array([0.001, 0.02, 0.2, 0.5, 0.77, 0.95, 0.9999])))
put("entr", sp.entr(np.array([0.0, 0.1, 0.5, 1.0, 2.7])))
put("zeta", sp.zeta(np.array([1.5, 2.0, 3.5, 6.0]), np.array([1.0, 2.0, 0.5, 3.0])))
put("igamma", sp.gammainc(np.array([0.5, 1.0, 2.5, 5.0, 12.0]), np.array([0.3, 1.0, 3.0, 4.0, 15.0])))
put("igammac", sp.gammaincc(np.array([0.5, 1.0, 2.5, 5.0, 12.0]), np.array([0.3, 1.0, 3.0, 4.0, 15.0])))
put("i0", sp.i0(anyv))
put("i0e", sp.i0e(anyv))
put("i1", sp.i1(anyv))
put("i1e", sp.i1e(anyv))
put("j0", sp.j0(anyv))
put("j1", sp.j1(anyv))
put("y0", sp.y0(pos))
put("y1", sp.y1(pos))
put("mvlgamma", sp.multigammaln(np.array([2.5, 3.3, 6.0]), 3) if False else np.array([float(sp.multigammaln(a, 3)) for a in [2.5, 3.3, 6.0]]))
put("xlog1py", sp.xlog1py(np.array([0.0, 1.5, 2.0, 3.0]), np.array([0.5, -0.4, 2.0, 9.0])))


def grad_of(fn, vals):
    x = torch.tensor(vals, dtype=torch.float64, requires_grad=True)
    fn(x).sum().backward()
    return x.grad.numpy()


put("g_lgamma", grad_of(torch.lgamma, [0.4, 1.1, 2.3, 5.0]))
put("g_digamma", grad_of(torch.digamma, [0.4, 1.1, 2.3, 5.0]))
put("g_erfinv", grad_of(torch.erfinv, [-0.8, -0.1, 0.4, 0.9]))
put("g_i0", grad_of(torch.special.i0, [0.4, 1.1, 2.3, 5.0]))
put("g_log_ndtr", grad_of(torch.special.log_ndtr, [-6.0, -1.0, 0.5, 3.0]))

worst, bad = 0.0, []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    tol = 1e-8 * max(1.0, abs(want)) + 1e-12
    err = abs(rust[k] - want) / max(1e-300, max(abs(want), 1e-12))
    worst = max(worst, min(err, 1.0))
    if abs(rust[k] - want) > max(2e-7 * abs(want), 1e-9):
        bad.append((k, rust[k], want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
for k, got, want in bad[:40]:
    print(f"MISMATCH {k}: rust={got} reference={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
