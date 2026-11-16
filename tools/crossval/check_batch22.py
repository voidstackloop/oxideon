"""Batch 22: backward passes of conv / pool / norm / recurrent layers vs PyTorch (float32 modules)."""
import math, re, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    if not rest: continue
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

def genv(n, salt):
    return [math.sin(0.37 * i + salt) * 0.8 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)]
def t(shape, salt):
    n = int(np.prod(shape))
    return torch.tensor(genv(n, salt), dtype=torch.float32).reshape(shape)

bad = n = 0; worst = 0.0
def check(name, tensor, tol=5e-5):
    global bad, n, worst
    want = tensor.detach().double().flatten().numpy(); got = arrs[name]; n += 1
    if got.shape != want.shape:
        bad += 1; print("SHAPE", name, got.shape, want.shape); return
    scale = max(1.0, np.abs(want).max())
    err = np.max(np.abs(got - want)) / scale; worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:3], want[:3])

def bwd(out):
    (out * t(list(out.shape), 99.0)).sum().backward()

x = t((2, 4, 7, 6), 0.3)
def conv_case(name, fn, wshape, wsalt, bn, bsalt):
    xi = x.clone().requires_grad_(True); w = t(wshape, wsalt).requires_grad_(True); b = t((bn,), bsalt).requires_grad_(True)
    bwd(fn(xi, w, b))
    check(name + "_dx", xi.grad); check(name + "_dw", w.grad); check(name + "_db", b.grad)
conv_case("conv2d", lambda a, w, b: F.conv2d(a, w, b), (6, 4, 3, 3), 1.0, 6, 1.5)
conv_case("conv2d_sp", lambda a, w, b: F.conv2d(a, w, b, stride=(2, 1), padding=(1, 2)), (6, 4, 3, 2), 2.0, 6, 2.5)
conv_case("conv2d_groups", lambda a, w, b: F.conv2d(a, w, b, padding=1, groups=2), (6, 2, 3, 3), 3.0, 6, 3.5)
conv_case("conv2d_dil", lambda a, w, b: F.conv2d(a, w, b, padding=2, dilation=2), (3, 4, 3, 3), 4.0, 3, 4.5)

x1 = t((2, 3, 11), 5.0).requires_grad_(True); w = t((4, 3, 3), 6.0).requires_grad_(True); b = t((4,), 6.5).requires_grad_(True)
bwd(F.conv1d(x1, w, b, stride=2, padding=1)); check("conv1d_dx", x1.grad); check("conv1d_dw", w.grad); check("conv1d_db", b.grad)
xt = t((2, 3, 4, 5), 7.0).requires_grad_(True); w = t((3, 2, 3, 3), 8.0).requires_grad_(True); b = t((2,), 8.5).requires_grad_(True)
bwd(F.conv_transpose2d(xt, w, b, stride=2, padding=1)); check("convt2d_dx", xt.grad); check("convt2d_dw", w.grad); check("convt2d_db", b.grad)

def pool(name, fn):
    xi = x.clone().requires_grad_(True); bwd(fn(xi)); check(name + "_dx", xi.grad)
pool("maxpool", lambda a: F.max_pool2d(a, 2, 2)); pool("maxpool_k3s1", lambda a: F.max_pool2d(a, 3, 1))
pool("avgpool", lambda a: F.avg_pool2d(a, (3, 2), (2, 2))); pool("adaptive_avg", lambda a: F.adaptive_avg_pool2d(a, (3, 4)))

xl = t((3, 5, 6), 10.0).requires_grad_(True); w = t((6,), 9.0).requires_grad_(True); b = t((6,), 9.5).requires_grad_(True)
bwd(F.layer_norm(xl, (6,), w, b, 1e-5)); check("layernorm_dx", xl.grad); check("layernorm_dw", w.grad); check("layernorm_db", b.grad)
xr = t((3, 5, 6), 12.0).requires_grad_(True); w = t((6,), 11.0).requires_grad_(True)
bwd(F.rms_norm(xr, (6,), w, 1e-6)); check("rmsnorm_dx", xr.grad); check("rmsnorm_dw", w.grad)
xg = x.clone().requires_grad_(True); w = t((4,), 13.0).requires_grad_(True); b = t((4,), 13.5).requires_grad_(True)
bwd(F.group_norm(xg, 2, w, b, 1e-5)); check("groupnorm_dx", xg.grad); check("groupnorm_dw", w.grad); check("groupnorm_db", b.grad)
xb = x.clone().requires_grad_(True); w = t((4,), 15.0).requires_grad_(True); b = t((4,), 15.5).requires_grad_(True)
rm, rv = torch.zeros(4), torch.ones(4)
bwd(F.batch_norm(xb, rm, rv, w, b, training=True, momentum=0.1, eps=1e-5))
check("batchnorm_dx", xb.grad); check("batchnorm_dw", w.grad); check("batchnorm_db", b.grad)
check("batchnorm_rm", rm); check("batchnorm_rv", rv)

seq = t((2, 5, 3), 14.0)
def fill(mod, ih0, hh0, gates):
    with torch.no_grad():
        for l in range(mod.num_layers):
            ind = 3 if l == 0 else 4
            getattr(mod, f"weight_ih_l{l}").copy_(t((gates * 4, ind), ih0 + 4.0 * l)); getattr(mod, f"bias_ih_l{l}").copy_(t((gates * 4,), ih0 + 4.0 * l + 0.5))
            getattr(mod, f"weight_hh_l{l}").copy_(t((gates * 4, 4), hh0 + 4.0 * l)); getattr(mod, f"bias_hh_l{l}").copy_(t((gates * 4,), hh0 + 4.0 * l + 0.5))
lstm = nn.LSTM(3, 4, 2, batch_first=True); fill(lstm, 20.0, 22.0, 4)
si = seq.clone().requires_grad_(True)
o, (h, c) = lstm(si)
((o * t(list(o.shape), 99.0)).sum() + (h * t(list(h.shape), 98.0)).sum() + (c * t(list(c.shape), 97.0)).sum()).backward()
check("lstm_dx", si.grad)
for l in range(2):
    check(f"lstm_dwih{l}", getattr(lstm, f"weight_ih_l{l}").grad); check(f"lstm_dwhh{l}", getattr(lstm, f"weight_hh_l{l}").grad)
    check(f"lstm_dbih{l}", getattr(lstm, f"bias_ih_l{l}").grad)
gru = nn.GRU(3, 4, 2, batch_first=True); fill(gru, 30.0, 32.0, 3)
sg = seq.clone().requires_grad_(True); go, _ = gru(sg); bwd(go)
check("gru_dx", sg.grad)
for l in range(2):
    check(f"gru_dwih{l}", getattr(gru, f"weight_ih_l{l}").grad); check(f"gru_dwhh{l}", getattr(gru, f"weight_hh_l{l}").grad)
    check(f"gru_dbhh{l}", getattr(gru, f"bias_hh_l{l}").grad)
rnn = nn.RNN(3, 4, 1, batch_first=True, nonlinearity="tanh"); fill(rnn, 40.0, 42.0, 1)
sr = seq.clone().requires_grad_(True); ro, _ = rnn(sr); bwd(ro)
check("rnn_dx", sr.grad); check("rnn_dwih", rnn.weight_ih_l0.grad); check("rnn_dwhh", rnn.weight_hh_l0.grad)
print(f"{n} gradient tensors compared, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
