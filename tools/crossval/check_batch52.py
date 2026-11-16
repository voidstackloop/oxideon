"""Batch 52: function transforms (value_and_grad, jacrev, jacfwd, hessian, hvp, jvp, vjp, per-sample gradients) vs torch.func."""
import math, re, sys, warnings
import numpy as np, torch
from torch.func import grad, jacrev, jacfwd, hessian, jvp, vjp, vmap, functional_call
warnings.filterwarnings("ignore")
arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass
def genv(n, salt): return [math.sin(0.37 * i + salt) * 1.3 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)]
def t(shape, salt): return torch.tensor(genv(int(np.prod(shape)), salt), dtype=torch.float64).reshape(shape)
bad = n = 0
def check(name, want, tol=1e-9):
    global bad, n
    n += 1
    want = want.detach().double().numpy().ravel(); got = arrs.get(name)
    if got is None or got.shape != want.shape: bad += 1; print("SHAPE/MISSING", name, None if got is None else got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1, np.abs(want)))
    if not err <= tol: bad += 1; print("MISMATCH", name, err)
x, y = t((4,), 0.4), t((4,), 1.1)
f = lambda a, b: (torch.sin(a) * b).sum() + (a ** 2).mean() * torch.exp(b).mean()
check("value", f(x, y)); gx, gy = grad(f, argnums=(0, 1))(x, y); check("grad_x", gx); check("grad_y", gy)
w = t((3, 4), 2.0); xm = t((2, 4), 0.9)
check("jacrev", jacrev(lambda a: torch.tanh(a @ w.T) * a.sum(1, keepdim=True))(xm))
check("jacfwd", jacfwd(lambda a: torch.tanh(a @ w.T) * a.sum(1, keepdim=True))(xm))
scalar = lambda a: (torch.sin(a) * a).sum() + (a ** (a ** 2 + 1.0)).sum()
check("hessian", hessian(scalar)(x))
v = t((4,), 3.3); check("hvp", jvp(grad(scalar), (x,), (v,))[1])
vec_f = lambda a: torch.tanh(a @ w.T); xv = t((2, 4), 0.9)
check("jvp", jvp(vec_f, (xv,), (t((2, 4), 1.5),))[1]); check("vjp", vjp(vec_f, xv)[1](t((2, 3), 0.2))[0])
lin = torch.nn.Linear(4, 3).double()
with torch.no_grad(): lin.weight.copy_(t((3, 4), 0.6)); lin.bias.copy_(t((3,), 1.4))
xb, yb = t((5, 4), 2.2), t((5, 3), 0.7)
params = {k: v.detach() for k, v in lin.named_parameters()}
loss = lambda p, xi, yi: ((functional_call(lin, p, (xi.unsqueeze(0),)) - yi.unsqueeze(0)) ** 2).sum()
g = vmap(grad(loss), in_dims=(None, 0, 0))(params, xb, yb)
check("psg_w", g["weight"], 1e-5); check("psg_b", g["bias"], 1e-5)
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
