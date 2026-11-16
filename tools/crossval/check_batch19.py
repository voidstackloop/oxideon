"""Batch 19: LBFGS (max_iter=1 per step) and Adafactor vs PyTorch."""
import re, sys
import torch

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    p = re.sub(r"^test \S+ \.\.\. ", "", line.strip()).split()
    if len(p) == 2:
        try: rust[p[0]] = float(p[1])
        except ValueError: pass

target = torch.tensor([1.5, -0.7, 0.2, 2.5, -1.2, 0.9, 0.4, -2.0, 1.1, 0.3, -0.5, 1.8], dtype=torch.float64)
init = [0.9, 1.4, -1.1, 0.3, 0.7, -0.4, 1.9, 0.2, -0.8, 1.2, 0.5, -1.5]
def grad(p, shape):
    t = target[: p.numel()].reshape(shape)
    d = p.detach()
    return 2 * (d - t) + 0.3 * d ** 3 + 0.1 * (torch.cos(d) * d + torch.sin(d))

cases = {
    "lbfgs": ([12], lambda p: torch.optim.LBFGS(p, lr=1.0, max_iter=1, history_size=10)),
    "lbfgs_h3": ([12], lambda p: torch.optim.LBFGS(p, lr=0.5, max_iter=1, history_size=3)),
    "lbfgs_lr": ([4], lambda p: torch.optim.LBFGS(p, lr=0.2, max_iter=1, history_size=5)),
    "adafactor_2d": ([3, 4], lambda p: torch.optim.Adafactor(p, lr=0.05)),
    "adafactor_1d": ([12], lambda p: torch.optim.Adafactor(p, lr=0.05)),
    "adafactor_wd": ([3, 4], lambda p: torch.optim.Adafactor(p, lr=0.1, weight_decay=0.05, beta2_decay=-0.6, d=1.5)),
    "adafactor_eps": ([4, 3], lambda p: torch.optim.Adafactor(p, lr=0.02, eps=(1e-6, 1e-2))),
}
bad = n = 0; worst = 0.0
for name, (shape, make) in cases.items():
    numel = 1
    for s in shape: numel *= s
    p = torch.tensor(init[:numel], dtype=torch.float64).reshape(shape).requires_grad_(True)
    # LBFGS's target slice uses the first `numel` entries just like the Rust helper (it reshapes `target()` to the shape)
    opt = make([p])
    for s in range(1, 26):
        if name.startswith("lbfgs"):
            def closure():
                opt.zero_grad(); p.grad = grad(p, shape); return torch.tensor(0.0, dtype=torch.float64)
            opt.step(closure)
        else:
            p.grad = grad(p, shape); opt.step()
        if s in (1, 2, 3, 5, 10, 25):
            for i, v in enumerate(p.detach().flatten().tolist()):
                k = f"{name}_s{s}_{i}"; n += 1
                if k not in rust: print("MISSING", k); bad += 1; continue
                err = abs(rust[k] - v) / max(abs(v), 1e-3); worst = max(worst, err)
                if err > 1e-7:
                    bad += 1
                    if bad < 25: print("MISMATCH", k, rust[k], v)
print(f"{n} values compared, worst relative error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
