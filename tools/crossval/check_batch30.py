"""Batch 30: Maxout, Snake, Swish, ConvLstmCell, SpectralNormLinear (values and gradients) and the structure of
DropPath / AlphaDropout outputs, vs PyTorch (float32)."""
import math, re, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass
def genv(n, salt): return [math.sin(0.37 * i + salt) * 0.8 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)]
def t(shape, salt): return torch.tensor(genv(int(np.prod(shape)), salt), dtype=torch.float32).reshape(shape)
R = lambda x: x.requires_grad_(True)
bad = n = 0; worst = 0.0
def check(name, tensor, tol=5e-5):
    global bad, n, worst
    n += 1
    if name not in arrs: bad += 1; print("MISSING", name); return
    want = tensor.detach().double().flatten().numpy(); got = arrs[name]
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want)) / max(1.0, np.abs(want).max()); worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:4], want[:4])
def ok(name, cond, detail=""):
    global bad, n
    n += 1
    if not cond: bad += 1; print("FAILED", name, detail)
def bwd(*outs): sum((o * t(list(o.shape), 90.0 + k)).sum() for k, o in enumerate(outs)).backward()

xv = genv(24, 1.0); xv[1] = xv[0]
x = R(torch.tensor(xv, dtype=torch.float32).reshape(3, 8)); out = x.reshape(3, 4, 2).max(-1).values; bwd(out)
check("maxout_out", out); check("maxout_dx", x.grad)

x = R(t((3, 4), 2.0)); a = R(t((4,), 2.5).abs() + 0.4); out = x + torch.sin(a * x) ** 2 / a; bwd(out)
check("snake_out", out); check("snake_dx", x.grad); check("snake_da", a.grad)
x = R(t((3, 4), 3.0)); b = R(torch.tensor([1.3])); out = x * torch.sigmoid(b * x); bwd(out)
check("swish_out", out); check("swish_dx", x.grad); check("swish_db", b.grad)

for tag in ("state", "zero"):
    w, bb = R(t((12, 5, 3, 3), 4.0)), R(t((12,), 4.5))
    x, h, c = R(t((2, 2, 4, 5), 5.0)), R(t((2, 3, 4, 5), 6.0)), R(t((2, 3, 4, 5), 7.0))
    h0, c0 = (h, c) if tag == "state" else (torch.zeros(2, 3, 4, 5), torch.zeros(2, 3, 4, 5))
    i, f, o, g = F.conv2d(torch.cat([x, h0], 1), w, bb, padding=1).chunk(4, 1)
    cn = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g); hn = torch.sigmoid(o) * torch.tanh(cn); bwd(hn, cn)
    check(f"clstm_{tag}_h", hn); check(f"clstm_{tag}_c", cn); check(f"clstm_{tag}_dx", x.grad)
    check(f"clstm_{tag}_dw", w.grad); check(f"clstm_{tag}_db", bb.grad)
    if tag == "state": check("clstm_state_dh", h.grad); check("clstm_state_dc", c.grad)

# spectral norm: one power iteration from the given u, v (the algorithm of torch.nn.utils.spectral_norm)
W, bias = R(t((3, 5), 8.0)), t((3,), 8.5)
u, v = F.normalize(t((3, 1), 9.0), dim=0), F.normalize(t((5, 1), 9.5), dim=0)
with torch.no_grad():
    v = F.normalize(W.t() @ u, dim=0, eps=1e-12); u = F.normalize(W @ v, dim=0, eps=1e-12)
sigma = (u.t() @ W @ v).squeeze()
x = R(t((4, 5), 10.0)); out = F.linear(x, W / sigma, bias); bwd(out)
check("sn_train_out", out); check("sn_train_dx", x.grad); check("sn_train_dw", W.grad); check("sn_u", u); check("sn_v", v)
check("sn_eval_out", F.linear(t((4, 5), 11.0), W / sigma, bias))
# and against torch's own parametrization, started from the same u / v
lin = torch.nn.Linear(5, 3)
with torch.no_grad(): lin.weight.copy_(t((3, 5), 8.0)); lin.bias.copy_(bias)
sn = torch.nn.utils.spectral_norm(lin, n_power_iterations=1)
with torch.no_grad():
    sn.weight_u.copy_(F.normalize(t((3, 1), 9.0), dim=0).flatten()); sn.weight_v.copy_(F.normalize(t((5, 1), 9.5), dim=0).flatten())
sn.train(); check("sn_train_out", sn(t((4, 5), 10.0)))

xin = arrs["drop_in"].reshape(64, 16)
dp = arrs["droppath_train"].reshape(64, 16)
rows_zero = np.all(dp == 0, axis=1); rows_kept = np.all(np.abs(dp - xin / 0.7) < 1e-5, axis=1)
ok("droppath_rows_all_or_nothing", bool(np.all(rows_zero | rows_kept)))
ok("droppath_keep_fraction", 0.45 < rows_kept.mean() < 0.9, rows_kept.mean())
ok("droppath_eval_identity", np.allclose(arrs["droppath_eval"], arrs["drop_in"]))
ok("droppath_p1_zero", np.all(arrs["droppath_p1"] == 0))
p, ap = 0.25, -1.7580993408473766; q = 1 - p
a = (q + ap * ap * q * p) ** -0.5; b = -a * p * ap
ad = arrs["alpha_train"].reshape(64, 16)
kept = np.abs(ad - (a * xin + b)) < 1e-5; dropped = np.abs(ad - (a * ap + b)) < 1e-5
ok("alpha_values_are_affine_or_saturated", bool(np.all(kept | dropped)))
ok("alpha_keep_fraction", abs(kept.mean() - q) < 0.06, kept.mean())
# same constants as torch: feeding standard-normal data keeps mean ~0 and variance ~1
ok("alpha_eval_identity", np.allclose(arrs["alpha_eval"], arrs["drop_in"])); ok("alpha_p1_zero", np.all(arrs["alpha_p1"] == 0))
torch.manual_seed(0); z = torch.randn(200000); zo = F.alpha_dropout(z, 0.25, training=True)
ref_drop = float(zo[(zo - (a * z + b)).abs() > 1e-4].mean())
ok("alpha_constants_match_torch", abs(ref_drop - (a * ap + b)) < 1e-4, (ref_drop, a * ap + b))
print(f"{n} checks, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
