"""Batch 39: GeGlu/ReGlu/SquaredRelu/Maxout, CosineLinear, WeightNormLinear, ConvPadded, T5 relative-position bucket vs PyTorch."""
import math, re, sys, warnings
import numpy as np, torch, torch.nn.functional as F
warnings.filterwarnings("ignore")
arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass
def genv(n, salt): return np.array([math.sin(0.37*i+salt)*1.3 + 0.1*math.cos(0.11*i*salt) for i in range(n)])
def t(shape, salt): return torch.tensor(genv(int(np.prod(shape)), salt).reshape(shape))
bad = n = 0
def check(name, want, tol=1e-9):
    global bad, n
    n += 1
    want = want.detach().double().numpy().ravel(); got = arrs.get(name)
    if got is None or got.shape != want.shape: bad += 1; print("SHAPE/MISSING", name, None if got is None else got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1, np.abs(want)))
    if not err <= tol: bad += 1; print("MISMATCH", name, err)
def run(name, x, f):
    x = x.clone().requires_grad_(True); y = f(x)
    check(name, y); (y * t(y.shape, 2.9)).sum().backward(); check(name + "_grad", x.grad)
x = t((3, 8), 0.4)
def glu(act): return lambda x: x[:, :4] * act(x[:, 4:])
run("geglu", x, glu(F.gelu)); run("reglu", x, glu(F.relu)); run("sqrelu", x, lambda x: F.relu(x) ** 2)
run("maxout", x, lambda x: x.reshape(3, 2, 4).max(-1).values)
W = t((5, 8), 1.7)
run("cosine_linear", x, lambda x: 12.0 * F.normalize(x, dim=-1) @ F.normalize(W, dim=-1).T)
V = t((5, 8), 0.9); g = t((5, 1), 3.3).abs() + 0.5; b = t((5,), 4.1)
ew = g * V / V.norm(dim=1, keepdim=True); check("wn_weight", ew)
run("wn_forward", x, lambda x: F.linear(x, ew, b))
img = t((2, 3, 6, 7), 0.2)
for nm, mode in [("reflect", "reflect"), ("replicate", "replicate"), ("circular", "circular")]:
    run(f"convpad_{nm}", img, lambda x: F.conv2d(F.pad(x, (1, 1, 2, 2), mode=mode), t((4, 3, 3, 3), 1.1), t((4,), 0.3), stride=(1, 2)))
def bucket(rel, bi, nb, md):  # transformers T5Attention._relative_position_bucket
    rel = torch.tensor(rel); out = 0
    if bi:
        nb //= 2; out = (rel > 0).long() * nb; rel = rel.abs()
    else:
        rel = -torch.min(rel, torch.zeros_like(rel))
    me = nb // 2; small = rel < me
    large = me + (torch.log(rel.float() / me) / math.log(md / me) * (nb - me)).long()
    large = torch.min(large, torch.full_like(large, nb - 1))
    return out + torch.where(small, rel, large)
for bi, nb, md in [(1, 32, 128), (0, 32, 128), (1, 16, 50), (0, 8, 20)]:
    check(f"t5bucket_{bi}_{nb}_{md}", bucket(list(range(-40, 41)), bi, nb, md).double())
emb = t((8, 2), 0.6)
idx = bucket([k - q for q in range(3) for k in range(4)], 1, 8, 20)
check("t5bias", emb[idx].reshape(3, 4, 2).permute(2, 0, 1).unsqueeze(0))
print(f"{n} checks, {bad} mismatches"); sys.exit(1 if bad else 0)
