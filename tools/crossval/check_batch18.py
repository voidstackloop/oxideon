"""Batch 18: learning-rate schedulers vs torch.optim.lr_scheduler."""
import math, re, sys, warnings
import torch
from torch.optim import lr_scheduler as L
warnings.filterwarnings("ignore")

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    p = re.sub(r"^test \S+ \.\.\. ", "", line.strip()).split()
    if len(p) == 2:
        try: rust[p[0]] = float(p[1])
        except ValueError: pass

def seq(base, make, n=30, momentum=False):
    p = torch.zeros(1, requires_grad=True)
    o = torch.optim.SGD([p], lr=base, momentum=0.9 if momentum else 0.0)
    s = make(o)
    out = [o.param_groups[0]["lr"]]
    for _ in range(n - 1):
        o.step(); s.step(); out.append(o.param_groups[0]["lr"])
    return out

cases = {
    "constant": (0.1, lambda o: L.ConstantLR(o, factor=0.3, total_iters=4)),
    "linear": (0.1, lambda o: L.LinearLR(o, start_factor=0.2, end_factor=0.9, total_iters=7)),
    "multistep": (0.1, lambda o: L.MultiStepLR(o, milestones=[4, 9, 9, 15], gamma=0.5)),
    "poly": (0.1, lambda o: L.PolynomialLR(o, total_iters=12, power=2.0)),
    "lambda": (0.1, lambda o: L.LambdaLR(o, lambda e: 0.95 ** e)),
    "cyc_tri": (0.01, lambda o: L.CyclicLR(o, 0.01, 0.1, step_size_up=4, step_size_down=6, mode="triangular", cycle_momentum=False)),
    "cyc_tri2": (0.01, lambda o: L.CyclicLR(o, 0.01, 0.1, step_size_up=4, step_size_down=6, mode="triangular2", cycle_momentum=False)),
    "cyc_exp": (0.01, lambda o: L.CyclicLR(o, 0.01, 0.1, step_size_up=3, step_size_down=3, mode="exp_range", gamma=0.95, cycle_momentum=False)),
    "seq": (0.1, lambda o: L.SequentialLR(o, [L.LinearLR(o, start_factor=0.1, end_factor=1.0, total_iters=4), L.MultiStepLR(o, [3, 8], 0.5)], milestones=[4])),
    "chain": (0.1, lambda o: L.ChainedScheduler([L.ConstantLR(o, factor=0.5, total_iters=5), L.ExponentialLR(o, gamma=0.95)])),  # LambdaLR does not chain in torch
    "onecycle_a": (0.1, lambda o: L.OneCycleLR(o, max_lr=0.1, total_steps=30, pct_start=0.3, div_factor=25, final_div_factor=1e4, cycle_momentum=False)),
    "onecycle_b": (0.5, lambda o: L.OneCycleLR(o, max_lr=0.5, total_steps=30, pct_start=0.45, div_factor=10, final_div_factor=100, cycle_momentum=False)),
}
bad = n = 0
worst = 0.0
for name, (base, make) in cases.items():
    try:
        ref = seq(base, make)
    except Exception as e:
        print("SKIP", name, e); continue
    for e, want in enumerate(ref):
        k = f"{name}_{e}"
        if k not in rust: print("MISSING", k); bad += 1; continue
        n += 1
        err = abs(rust[k] - want) / max(abs(want), 1e-12); worst = max(worst, err)
        if err > 1e-9:
            bad += 1
            if bad < 40: print("MISMATCH", k, rust[k], want)

# warmup + cosine reference (closed form)
base, warm, total, lo = 0.1, 5, 25, 0.001
for e in range(1, 30):
    if e <= warm: want = base * e / warm
    else:
        prog = min(e - warm, total - warm) / (total - warm)
        want = lo + (base - lo) * 0.5 * (1 + math.cos(math.pi * prog))
    k = f"warmcos_{e}"; n += 1
    if abs(rust[k] - want) > 1e-12: bad += 1; print("MISMATCH", k, rust[k], want)

# ReduceLROnPlateau
metrics = [1.0, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
p = torch.zeros(1, requires_grad=True); o = torch.optim.SGD([p], lr=0.1)
s = L.ReduceLROnPlateau(o, mode="min", factor=0.5, patience=2, min_lr=1e-4)
for e, m in enumerate(metrics):
    s.step(m); want = o.param_groups[0]["lr"]; k = f"plateau_{e}"; n += 1
    if abs(rust[k] - want) > 1e-12: bad += 1; print("MISMATCH", k, rust[k], want)
for tag, mx, th, rel, cd in [("pa", False, 0.02, False, 0), ("pb", False, 1e-4, True, 3), ("pc", True, 0.05, True, 0), ("pd", True, 0.01, False, 2)]:
    p = torch.zeros(1, requires_grad=True); o = torch.optim.SGD([p], lr=0.1)
    s = L.ReduceLROnPlateau(o, mode="max" if mx else "min", factor=0.5, patience=1, min_lr=1e-3, threshold=th,
                            threshold_mode="rel" if rel else "abs", cooldown=cd)
    for e, m in enumerate(metrics):
        s.step(2.0 - m if mx else m); want = o.param_groups[0]["lr"]; k = f"{tag}_{e}"; n += 1
        if abs(rust[k] - want) > 1e-12: bad += 1; print("MISMATCH", k, rust[k], want)
print(f"{n} values compared, worst relative error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
