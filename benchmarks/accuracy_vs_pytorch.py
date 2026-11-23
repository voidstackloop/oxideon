"""Numerical accuracy of oxideon against PyTorch, both float32 CPU, measured against a float64 reference on identical inputs.

    cargo build --release --example accuracy_dump
    python benchmarks/accuracy_vs_pytorch.py

The error of one element is |result - reference| / scale, in units of the float32 machine epsilon (2^-24, half an ulp at 1.0):
  * element-wise functions: scale = |reference|, so the value is the relative error (0.5 = correctly rounded, 1.0 = one ulp);
  * reductions and matrix products: scale = the sum of the absolute terms (sum |x_i|, |A| @ |B|), the condition-aware error
    that is independent of cancellation and is what summation order controls.
Each case reports the worst and the mean element together, for both libraries, with "x better" = torch error / oxideon error
(above 1 means oxideon is more accurate); the verdict is the smaller of the two figures.
Set OXIDEON_PRECISION=fastest|balanced|highest to choose oxideon's matmul mode (default balanced).
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "target/release/examples/accuracy_dump.exe"
if not EXE.exists():
    EXE = ROOT / "target/release/examples/accuracy_dump"
EPS = 2.0 ** -24
rng = np.random.default_rng(2026)


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def erf64(x):
    return np.array([math.erf(v) for v in x.ravel()]).reshape(x.shape)


# case -> (inputs, torch function, float64 reference function returning (reference, scale))
def elementwise(fn64):
    def ref(i):
        r = fn64(i["x"].astype(np.float64))
        return r, np.abs(r)
    return ref


def abs_sum(i):
    return np.abs(i["x"].astype(np.float64)).sum()


cases = {}


def case(name, inputs, torch_fn, ref_fn):
    cases[name] = (inputs, torch_fn, ref_fn)


case("sum_1M", {"x": f32(rng.standard_normal(1_000_000))}, lambda i: i["x"].sum(),
     lambda i: (i["x"].astype(np.float64).sum(), abs_sum(i)))
case("sum_16M_offset", {"x": f32(rng.standard_normal(16_000_000) + 100)}, lambda i: i["x"].sum(),
     lambda i: (i["x"].astype(np.float64).sum(), abs_sum(i)))
case("mean_4M_offset", {"x": f32(rng.standard_normal(4_000_000) + 1e4)}, lambda i: i["x"].mean(),
     lambda i: (i["x"].astype(np.float64).mean(), abs_sum(i) / 4_000_000))
case("sum_axis0", {"x": f32(rng.standard_normal((1024, 1024)) + 50)}, lambda i: i["x"].sum(0),
     lambda i: (i["x"].astype(np.float64).sum(0), np.abs(i["x"].astype(np.float64)).sum(0)))
case("sum_axis1", {"x": f32(rng.standard_normal((1024, 1024)) + 50)}, lambda i: i["x"].sum(1),
     lambda i: (i["x"].astype(np.float64).sum(1), np.abs(i["x"].astype(np.float64)).sum(1)))
a4, b4 = f32(rng.standard_normal(4_000_000)), f32(rng.standard_normal(4_000_000))
case("dot_4M", {"a": a4, "b": b4}, lambda i: torch.dot(i["a"], i["b"]),
     lambda i: ((i["a"].astype(np.float64) * i["b"]).sum(), np.abs(i["a"].astype(np.float64) * i["b"]).sum()))
case("norm_4M", {"x": a4}, lambda i: torch.linalg.vector_norm(i["x"]),
     lambda i: (math.sqrt((i["x"].astype(np.float64) ** 2).sum()), math.sqrt((i["x"].astype(np.float64) ** 2).sum())))


def mm(i):
    a, b = i["a"].astype(np.float64), i["b"].astype(np.float64)
    return a @ b, np.abs(a) @ np.abs(b)


case("matmul_512", {"a": f32(rng.standard_normal((512, 512))), "b": f32(rng.standard_normal((512, 512)))}, lambda i: i["a"] @ i["b"], mm)
case("matmul_k8192", {"a": f32(rng.standard_normal((64, 8192))), "b": f32(rng.standard_normal((8192, 64)))}, lambda i: i["a"] @ i["b"], mm)


def softmax64(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def log_softmax64(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


logits = f32(rng.standard_normal((1024, 1000)) * 10)
case("softmax", {"x": logits}, lambda i: torch.softmax(i["x"], -1), elementwise(softmax64))
case("log_softmax", {"x": logits}, lambda i: torch.log_softmax(i["x"], -1),
     lambda i: (log_softmax64(i["x"].astype(np.float64)), np.abs(log_softmax64(i["x"].astype(np.float64))) + 1e-3))
case("logsumexp", {"x": logits}, lambda i: torch.logsumexp(i["x"], -1),
     lambda i: (lambda m: (m.squeeze(-1) + np.log(np.exp(i["x"].astype(np.float64) - m).sum(-1)), None))(i["x"].astype(np.float64).max(-1, keepdims=True)))

ln_x = f32(rng.standard_normal((256, 1024)) + 1000)
ln_w, ln_b = f32(rng.uniform(0.5, 1.5, 1024)), f32(rng.standard_normal(1024))


def ln64(i):
    x = i["x"].astype(np.float64)
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + 1e-5) * i["w"] + i["b"]
    return y, np.abs(y) + 0.5


case("layer_norm_offset", {"x": ln_x, "w": ln_w, "b": ln_b}, lambda i: torch.nn.functional.layer_norm(i["x"], (1024,), i["w"], i["b"], 1e-5), ln64)

xs = f32(rng.uniform(-20, 20, 1_000_000))
case("exp", {"x": f32(rng.uniform(-30, 30, 1_000_000))}, lambda i: torch.exp(i["x"]), elementwise(np.exp))
case("tanh", {"x": xs}, lambda i: torch.tanh(i["x"]), elementwise(np.tanh))
case("sigmoid", {"x": xs}, lambda i: torch.sigmoid(i["x"]), elementwise(lambda x: 1 / (1 + np.exp(-x))))
case("gelu", {"x": f32(rng.uniform(-8, 8, 1_000_000))}, lambda i: torch.nn.functional.gelu(i["x"]),
     elementwise(lambda x: 0.5 * x * (1 + erf64(x / math.sqrt(2)))))
case("erf", {"x": f32(rng.uniform(-4, 4, 1_000_000))}, lambda i: torch.erf(i["x"]), elementwise(erf64))
case("ln", {"x": f32(np.exp(rng.uniform(-14, 14, 1_000_000)))}, lambda i: torch.log(i["x"]), elementwise(np.log))
case("sqrt", {"x": f32(np.exp(rng.uniform(-14, 14, 1_000_000)))}, lambda i: torch.sqrt(i["x"]), elementwise(np.sqrt))


def conv64(i):
    x, w = torch.from_numpy(i["x"]).double(), torch.from_numpy(i["w"]).double()
    r = torch.nn.functional.conv2d(x, w, padding=1).numpy()
    s = torch.nn.functional.conv2d(x.abs(), w.abs(), padding=1).numpy()
    return r, s


case("conv2d_16ch", {"x": f32(rng.standard_normal((8, 16, 64, 64))), "w": f32(rng.standard_normal((16, 16, 3, 3)) * 0.2)},
     lambda i: torch.nn.functional.conv2d(i["x"], i["w"], padding=1), conv64)
case("conv2d_64ch", {"x": f32(rng.standard_normal((4, 64, 32, 32))), "w": f32(rng.standard_normal((64, 64, 3, 3)) * 0.1)},
     lambda i: torch.nn.functional.conv2d(i["x"], i["w"], padding=1), conv64)
cs = f32(rng.standard_normal(1_000_000) + 0.5)
case("cumsum_1M", {"x": cs}, lambda i: torch.cumsum(i["x"], 0),
     lambda i: (np.cumsum(i["x"].astype(np.float64)), np.cumsum(np.abs(i["x"].astype(np.float64)))))


# ---- second group: more element-wise functions, statistics and normalisations ----
F = torch.nn.functional
case("sin", {"x": f32(rng.uniform(-100, 100, 1_000_000))}, lambda i: torch.sin(i["x"]), elementwise(np.sin))
case("cos", {"x": f32(rng.uniform(-100, 100, 1_000_000))}, lambda i: torch.cos(i["x"]), elementwise(np.cos))
case("log1p", {"x": f32(np.exp(rng.uniform(-18, 2, 1_000_000)))}, lambda i: torch.log1p(i["x"]), elementwise(np.log1p))
case("expm1", {"x": f32(rng.uniform(-5, 5, 1_000_000) * np.exp(rng.uniform(-12, 0, 1_000_000)))}, lambda i: torch.expm1(i["x"]), elementwise(np.expm1))
case("softplus", {"x": f32(rng.uniform(-20, 20, 1_000_000))}, lambda i: F.softplus(i["x"]), elementwise(lambda x: np.log1p(np.exp(x))))
case("silu", {"x": f32(rng.uniform(-20, 20, 1_000_000))}, lambda i: F.silu(i["x"]), elementwise(lambda x: x / (1 + np.exp(-x))))
case("sinh", {"x": f32(rng.uniform(-10, 10, 1_000_000))}, lambda i: torch.sinh(i["x"]), elementwise(np.sinh))
case("cosh", {"x": f32(rng.uniform(-10, 10, 1_000_000))}, lambda i: torch.cosh(i["x"]), elementwise(np.cosh))
case("atan", {"x": f32(rng.uniform(-50, 50, 1_000_000))}, lambda i: torch.atan(i["x"]), elementwise(np.arctan))
case("rsqrt", {"x": f32(np.exp(rng.uniform(-14, 14, 1_000_000)))}, lambda i: torch.rsqrt(i["x"]), elementwise(lambda x: 1 / np.sqrt(x)))
case("pow_2_5", {"x": f32(np.exp(rng.uniform(-6, 6, 1_000_000)))}, lambda i: torch.pow(i["x"], 2.5), elementwise(lambda x: x ** 2.5))

off = f32(rng.standard_normal((2048, 512)) + 1000)
case("var_offset", {"x": off}, lambda i: i["x"].var(1, unbiased=False),
     lambda i: (i["x"].astype(np.float64).var(1), None))
bn = f32(rng.standard_normal((16, 8, 64)) + 1000)
gn = f32(rng.standard_normal((16, 8, 64)) + 1000)


def bn64(i):
    x = i["x"].astype(np.float64)
    m, v = x.mean((0, 2), keepdims=True), x.var((0, 2), keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5), np.ones_like(x)


def gn64(i):
    x = i["x"].astype(np.float64).reshape(16, 4, -1)
    m, v = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return ((x - m) / np.sqrt(v + 1e-5)).reshape(16, 8, 64), np.ones((16, 8, 64))


case("batch_norm_offset", {"x": bn}, lambda i: F.batch_norm(i["x"], None, None, None, None, True, 0.1, 1e-5), bn64)
case("group_norm_offset", {"x": gn}, lambda i: F.group_norm(i["x"], 4, None, None, 1e-5), gn64)
ca, cb = f32(rng.standard_normal((4096, 256))), f32(rng.standard_normal((4096, 256)) + 0.3)
case("cosine_sim", {"a": ca, "b": cb}, lambda i: F.cosine_similarity(i["a"], i["b"], 1, 1e-8),
     lambda i: (lambda a, b: ((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)), None))(i["a"].astype(np.float64), i["b"].astype(np.float64)))
case("normalize_l2", {"x": ca}, lambda i: F.normalize(i["x"], 2.0, 1, 1e-12),
     lambda i: (lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True), None))(i["x"].astype(np.float64)))
case("mse", {"a": f32(rng.standard_normal(2_000_000) + 100), "b": f32(rng.standard_normal(2_000_000) + 100)}, lambda i: F.mse_loss(i["a"], i["b"]),
     lambda i: (lambda d: ((d ** 2).mean(), (d ** 2).mean()))(i["a"].astype(np.float64) - i["b"].astype(np.float64)))


def better(t, o):
    """How many times smaller oxideon's error is than torch's (> 1 = oxideon more accurate)."""
    if t < 1e-3 and o < 1e-3:
        return float("nan")  # both exact to the last bit: nothing to compare
    return t / o if o > 0 else float("inf")


def verdict_of(bw, bm):
    both = [x for x in (bw, bm) if x == x]
    lo = min(both) if both else float("nan")
    return "exact" if not both else ">=2x" if lo >= 2 else ">=1.5x" if lo >= 1.5 else "better" if lo > 1.05 else "tie" if lo >= 0.95 else "WORSE"


def measure(only=()):
    """Runs the cases (all, or the names in `only`); returns {name: (worst_ox, worst_torch, x_better, mean_ox, mean_torch, x_better, verdict)}."""
    work = Path(tempfile.mkdtemp(prefix="oxideon_accuracy_"))
    manifest = []
    selected = {k: v for k, v in cases.items() if not only or k in only}
    for name, (inputs, _, _) in selected.items():
        for iname, arr in inputs.items():
            fn = f"{name}.{iname}.in.f32"
            arr.tofile(work / fn)
            manifest.append(f"{name} {iname} {','.join(map(str, arr.shape))} {fn}")
    (work / "manifest.txt").write_text("\n".join(manifest))
    subprocess.run([str(EXE), str(work)], check=True, stdout=subprocess.DEVNULL)
    rows = {}
    for name, (inputs, torch_fn, ref_fn) in selected.items():
        ref, scale = ref_fn(inputs)
        ref = np.asarray(ref, dtype=np.float64)
        scale = np.abs(np.asarray(ref if scale is None else scale, dtype=np.float64)) + 1e-300
        ox = np.fromfile(work / f"{name}.oxideon.f32", dtype=np.float32).astype(np.float64).reshape(ref.shape)
        with torch.no_grad():
            th = torch_fn({k: torch.from_numpy(v) for k, v in inputs.items()}).numpy().astype(np.float64).reshape(ref.shape)
        e_ox, e_th = np.abs(ox - ref) / scale / EPS, np.abs(th - ref) / scale / EPS
        wo, wt, mo, mt = e_ox.max(), e_th.max(), e_ox.mean(), e_th.mean()
        bw, bm = better(wt, wo), better(mt, mo)
        rows[name] = (wo, wt, bw, mo, mt, bm, verdict_of(bw, bm))
    return rows


def fmt(x):
    return "  tie" if x != x else "  inf" if x == float("inf") else f"{x:5.2f}"


def main():
    rows = measure(set(sys.argv[1:]))
    print(f"{'case':20} | {'worst: oxideon':>14} {'torch':>10} {'x better':>9} | {'mean: oxideon':>13} {'torch':>10} {'x better':>9} | verdict")
    for name, (wo, wt, bw, mo, mt, bm, verdict) in rows.items():
        print(f"{name:20} | {wo:14.4f} {wt:10.4f} {fmt(bw):>9} | {mo:13.5f} {mt:10.5f} {fmt(bm):>9} | {verdict}")
    print()
    for v in ("exact", ">=2x", ">=1.5x", "better", "tie", "WORSE"):
        names = [n for n, r in rows.items() if r[6] == v]
        if names:
            print(f"{v:>7} ({len(names)}): {', '.join(names)}")
    print("\nverdict =the smaller of the two 'x better' figures (worst and mean); tie = within 5%; exact = both below 0.001 half-ulps.")


if __name__ == "__main__":
    main()
