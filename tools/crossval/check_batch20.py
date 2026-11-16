"""Batch 20: activation functions (values and gradients, incl. boundary points) vs torch.nn.functional."""
import re, sys
import torch
import torch.nn.functional as F

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    p = re.sub(r"^test \S+ \.\.\. ", "", line.strip()).split()
    if len(p) == 2:
        try: rust[p[0]] = float(p[1])
        except ValueError: pass

grid = [-30.0, -8.0, -6.0, -4.5, -3.0, -2.0, -1.5, -1.0, -0.6, -0.5, -0.1, -1e-9, 0.0, 1e-9, 0.1, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 6.0, 8.0, 19.0, 21.0, 30.0]
acts = {
    "relu": F.relu, "relu6": F.relu6, "leaky": lambda x: F.leaky_relu(x, 0.07), "elu": lambda x: F.elu(x, 1.3),
    "selu": F.selu, "celu": lambda x: F.celu(x, 0.7), "celu_neg": lambda x: F.celu(x, -0.7), "gelu": F.gelu, "silu": F.silu, "mish": F.mish,
    "softplus": F.softplus, "softplus_b2": lambda x: F.softplus(x, beta=2.0), "softplus_b05": lambda x: F.softplus(x, beta=0.5),
    "softsign": F.softsign, "hardsigmoid": F.hardsigmoid, "hardswish": F.hardswish,
    "hardtanh": lambda x: F.hardtanh(x, -1.5, 2.0), "hardshrink": lambda x: F.hardshrink(x, 0.6),
    "softshrink": lambda x: F.softshrink(x, 0.6), "tanhshrink": F.tanhshrink, "logsigmoid": F.logsigmoid,
    "sigmoid": torch.sigmoid, "tanh": torch.tanh,
}
bad = n = 0; worst = 0.0
for name, f in acts.items():
    x = torch.tensor(grid, dtype=torch.float64, requires_grad=True)
    y = f(x); y.sum().backward()
    for i in range(len(grid)):
        for kind, want in (("v", y[i].item()), ("g", x.grad[i].item())):
            k = f"{name}_{kind}_{i}"; n += 1
            got = rust[k]
            err = abs(got - want) / max(abs(want), 1e-6) if abs(want) > 1e-6 else abs(got - want)
            worst = max(worst, err)
            # torch evaluates the constants of celu / hardsigmoid backward in float32 (1/6 -> 0.1666666716): allow 1e-6
            tol = 1e-6 if (name, kind) in {("celu", "g"), ("celu_neg", "g"), ("hardsigmoid", "g")} else 1e-9
            if err > tol:
                bad += 1; print("MISMATCH", name, kind, f"x={grid[i]}", got, want)
print(f"{n} values compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
