"""Batch 50: autocast matmul / Linear numerics and the GradScaler trajectory vs torch.autocast and torch.amp.GradScaler (CPU)."""
import math, re, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass
def genv(n, salt): return np.array([math.sin(0.37 * i + salt) * 1.3 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)], dtype=np.float32)
a = torch.tensor(genv(32 * 64, 0.4)).reshape(32, 64); b = torch.tensor(genv(64 * 48, 1.7)).reshape(64, 48)
bad = n = 0
def close(name, want, rel, got=None):
    global bad, n
    n += 1
    got = arrs[name] if got is None else got; want = want.detach().float().numpy().ravel()
    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want)))
    if not err <= rel: bad += 1; print("MISMATCH", name, "max rel err", err, "allowed", rel)
    else: print("ok  ", name, f"max rel err {err:.2e}")
for name, dt in [("bf16", torch.bfloat16), ("f16", torch.float16)]:
    with torch.autocast("cpu", dtype=dt): y = torch.matmul(a, b)
    # both round the f32 accumulation to the half type; accumulation order may differ by one ulp
    close(f"matmul_{name}", y, 2 ** -7 if name == "bf16" else 2 ** -10)
    assert y.dtype == dt
lin = torch.nn.Linear(64, 48)
with torch.no_grad(): lin.weight.copy_(torch.tensor(genv(48 * 64, 0.9)).reshape(48, 64)); lin.bias.copy_(torch.tensor(genv(48, 2.2)))
with torch.autocast("cpu", dtype=torch.bfloat16): y = lin(a)
close("linear_bf16", y, 2 ** -6)
# GradScaler trajectory
w = torch.nn.Parameter(torch.tensor([1.0, 2.0])); opt = torch.optim.SGD([w], lr=0.01)
sc = torch.amp.GradScaler("cpu", init_scale=1024.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=3)
sc.scale(torch.tensor(1.0)); scales, taken = [], []
for step in range(16):
    g = float("inf") if step in (4, 5, 11) else 0.5 * sc.get_scale()
    w.grad = torch.tensor([g, g]); before = w.detach().clone()
    sc.step(opt); sc.update()
    scales.append(sc.get_scale()); taken.append(0.0 if torch.equal(before, w.detach()) else 1.0)
close("scale_trajectory", torch.tensor(scales), 0.0); close("steps_taken", torch.tensor(taken), 0.0); close("final_weight", w, 1e-6)
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
