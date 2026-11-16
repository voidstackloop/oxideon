"""Batch 27: MultiheadAttention backward, Linear and Embedding gradients vs PyTorch (float32 modules)."""
import math, re, sys, warnings
import numpy as np
import torch
import torch.nn.functional as F
warnings.filterwarnings("ignore")

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if not rest: continue
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

def genv(n, salt): return [math.sin(0.37 * i + salt) * 0.8 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)]
def t(shape, salt): return torch.tensor(genv(int(np.prod(shape)), salt), dtype=torch.float32).reshape(shape)
bad = n = 0; worst = 0.0
def check(name, tensor, tol=5e-5):
    global bad, n, worst
    want = tensor.detach().double().flatten().numpy(); got = arrs[name]; n += 1
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want)) / max(1.0, np.abs(want).max()); worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:3], want[:3])
def bwd(out): (out * t(list(out.shape), 99.0)).sum().backward()

def build():
    m = torch.nn.MultiheadAttention(8, 2, dropout=0.0, bias=True, batch_first=True)
    with torch.no_grad():
        m.in_proj_weight.copy_(torch.cat([t((8, 8), 10.0), t((8, 8), 11.0), t((8, 8), 12.0)], 0))
        m.in_proj_bias.copy_(torch.cat([t((8,), 10.5), t((8,), 11.5), t((8,), 12.5)], 0))
        m.out_proj.weight.copy_(t((8, 8), 13.0)); m.out_proj.bias.copy_(t((8,), 13.5))
    return m
def params(tag, m):
    gw, gb = m.in_proj_weight.grad, m.in_proj_bias.grad
    for i, nme in enumerate("qkv"):
        check(f"{tag}_dw{nme}", gw[8 * i: 8 * i + 8]); check(f"{tag}_db{nme}", gb[8 * i: 8 * i + 8])
    check(f"{tag}_dwo", m.out_proj.weight.grad); check(f"{tag}_dbo", m.out_proj.bias.grad)

def inputs(): return [x.requires_grad_(True) for x in (t((2, 5, 8), 20.0), t((2, 7, 8), 21.0), t((2, 7, 8), 22.0))]
m = build(); q, k, v = inputs()
bwd(m(q, k, v, need_weights=False)[0])
check("x_dq", q.grad); check("x_dk", k.grad); check("x_dv", v.grad); params("x", m)

m = build(); q, k, v = inputs()
kpm = torch.tensor([[0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0]], dtype=torch.bool)
bwd(m(q, k, v, key_padding_mask=kpm, need_weights=False)[0])
check("kpm_dq", q.grad); check("kpm_dk", k.grad); check("kpm_dv", v.grad); params("kpm", m)

m = build(); s = t((2, 5, 8), 23.0).requires_grad_(True)
bwd(m(s, s, s, attn_mask=torch.triu(torch.ones(5, 5, dtype=torch.bool), 1), need_weights=False)[0])
check("causal_dx", s.grad); params("causal", m)

w, b, x = t((3, 8), 30.0).requires_grad_(True), t((3,), 30.5).requires_grad_(True), t((2, 5, 8), 31.0).requires_grad_(True)
bwd(F.linear(x, w, b)); check("lin_dx", x.grad); check("lin_dw", w.grad); check("lin_db", b.grad)

W = t((7, 4), 40.0).requires_grad_(True)
e = F.embedding(torch.tensor([[1, 3, 3, 0], [6, 1, 1, 2]]), W)
check("emb_out", e); bwd(e); check("emb_dw", W.grad)
print(f"{n} tensors compared, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
