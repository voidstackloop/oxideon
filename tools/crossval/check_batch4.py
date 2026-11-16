"""Fourth cross-validation batch: optimizers and learning-rate schedules vs PyTorch.

    cargo test -p oxideon-core --test crossval_emit4 -- --nocapture --ignored > rust_values4.txt
    py -3.11 tools/crossval/check_batch4.py rust_values4.txt
"""
import sys

import torch

rust = {}
import re
for line in open(sys.argv[1], encoding="utf-8"):
    parts = re.sub(r"^test \S+ \.\.\. ", "", line.strip()).split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass

target = torch.tensor([1.5, -0.7, 0.2, 2.5], dtype=torch.float64)
makers = {
    "sgd": lambda p: torch.optim.SGD(p, lr=0.05),
    "sgd_mom": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9),
    "sgd_nesterov": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True),
    "sgd_damp": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.8, dampening=0.3, weight_decay=0.01),
    "sgd_max": lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.5, maximize=True),
    "adam": lambda p: torch.optim.Adam(p, lr=0.05),
    "adam_ams": lambda p: torch.optim.Adam(p, lr=0.05, amsgrad=True),
    "adam_wd_ams": lambda p: torch.optim.Adam(p, lr=0.05, weight_decay=0.02, amsgrad=True, betas=(0.8, 0.9)),
    "adamw_ams": lambda p: torch.optim.AdamW(p, lr=0.05, weight_decay=0.05, amsgrad=True),
    "adam_max": lambda p: torch.optim.Adam(p, lr=0.05, maximize=True),
    "adagrad": lambda p: torch.optim.Adagrad(p, lr=0.1),
    "rmsprop": lambda p: torch.optim.RMSprop(p, lr=0.01),
    "rmsprop_centered": lambda p: torch.optim.RMSprop(p, lr=0.01, centered=True),
    "rmsprop_mom": lambda p: torch.optim.RMSprop(p, lr=0.01, momentum=0.9, alpha=0.9, eps=1e-6, weight_decay=0.01),
    "rmsprop_cmom": lambda p: torch.optim.RMSprop(p, lr=0.01, centered=True, momentum=0.5),
    "adagrad_decay": lambda p: torch.optim.Adagrad(p, lr=0.1, lr_decay=0.05, initial_accumulator_value=0.2, weight_decay=0.01, eps=1e-8),
    "adadelta": lambda p: torch.optim.Adadelta(p, lr=1.0),
    "adamax": lambda p: torch.optim.Adamax(p, lr=0.05),
    "nadam": lambda p: torch.optim.NAdam(p, lr=0.05),
    "radam": lambda p: torch.optim.RAdam(p, lr=0.05),
    "rprop": lambda p: torch.optim.Rprop(p, lr=0.05),
    "asgd": lambda p: torch.optim.ASGD(p, lr=0.05, lambd=1e-4, alpha=0.75, t0=1e6),
}
ref = {}
for name, make in makers.items():
    p = torch.tensor([0.9, 1.4, -1.1, 0.3], dtype=torch.float64, requires_grad=True)
    opt = make([p])
    for s in range(1, 31):
        p.grad = 2 * (p.detach() - target) + 0.3 * p.detach() ** 3
        opt.step()
        if s in (1, 2, 5, 12, 30):
            for i, v in enumerate(p.detach().tolist()):
                ref[f"opt_{name}_s{s}_{i}"] = v

def sched_values(make_sched):
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=0.1)
    sch = make_sched(opt)
    out = [opt.param_groups[0]["lr"]]
    for _ in range(29):
        opt.step()
        sch.step()
        out.append(opt.param_groups[0]["lr"])
    return out

from torch.optim import lr_scheduler as L
for name, make in {
    "step": lambda o: L.StepLR(o, step_size=7, gamma=0.5),
    "cosine": lambda o: L.CosineAnnealingLR(o, T_max=20, eta_min=0.001),
    "exp": lambda o: L.ExponentialLR(o, gamma=0.93),
    "warm": lambda o: L.CosineAnnealingWarmRestarts(o, T_0=6, T_mult=2, eta_min=0.001),
}.items():
    for e, v in enumerate(sched_values(make)):
        ref[f"sched_{name}_{e}"] = v

worst, bad = 0.0, []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1e-3, abs(want))
    worst = max(worst, err)
    if err > 1e-6:
        bad.append((k, rust[k], want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
by_opt = {}
for k, got, want in bad:
    by_opt.setdefault(k.rsplit("_s", 1)[0] if k.startswith("opt_") else k.rsplit("_", 1)[0], []).append((k, got, want))
for group, items in by_opt.items():
    k, got, want = items[0]
    print(f"MISMATCH {group}: {len(items)} values differ, e.g. {k}: rust={got} torch={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
