"""Batch 30: RNN / GRU / LSTM cells and Upsample2d vs PyTorch (float32)."""
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
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:3], want[:3])
def bwd(*outs): sum((o * t(list(o.shape), 90.0 + k)).sum() for k, o in enumerate(outs)).backward()
def load(cell, s1, s2):
    with torch.no_grad():
        cell.weight_ih.copy_(t(tuple(cell.weight_ih.shape), s1)); cell.bias_ih.copy_(t((cell.bias_ih.shape[0],), s1 + 0.5))
        cell.weight_hh.copy_(t(tuple(cell.weight_hh.shape), s2)); cell.bias_hh.copy_(t((cell.bias_hh.shape[0],), s2 + 0.5))

for tag in ("state", "zero"):
    ws = tag == "state"
    c = torch.nn.RNNCell(3, 4); load(c, 1.0, 2.0)
    x, h = R(t((5, 3), 3.0)), R(t((5, 4), 4.0)); out = c(x, h if ws else None); bwd(out)
    check(f"rnn_{tag}_out", out); check(f"rnn_{tag}_dx", x.grad); check(f"rnn_{tag}_dwih", c.weight_ih.grad); check(f"rnn_{tag}_dwhh", c.weight_hh.grad)
    c = torch.nn.GRUCell(3, 4); load(c, 5.0, 6.0)
    x, h = R(t((5, 3), 7.0)), R(t((5, 4), 8.0)); out = c(x, h if ws else None); bwd(out)
    check(f"gru_{tag}_out", out); check(f"gru_{tag}_dx", x.grad); check(f"gru_{tag}_dwih", c.weight_ih.grad); check(f"gru_{tag}_dbhh", c.bias_hh.grad)
    c = torch.nn.LSTMCell(3, 4); load(c, 9.0, 10.0)
    x, h, cc = R(t((5, 3), 11.0)), R(t((5, 4), 12.0)), R(t((5, 4), 13.0))
    ho, co = c(x, (h, cc) if ws else None); bwd(ho, co)
    check(f"lstm_{tag}_h", ho); check(f"lstm_{tag}_c", co); check(f"lstm_{tag}_dx", x.grad)
    check(f"lstm_{tag}_dwih", c.weight_ih.grad); check(f"lstm_{tag}_dwhh", c.weight_hh.grad)
    if ws: check("lstm_state_dh", h.grad); check("lstm_state_dc", cc.grad)
for name, kw in [("up_nearest", dict(size=(7, 9), mode="nearest")), ("up_nearest_down", dict(size=(2, 3), mode="nearest")),
                 ("up_bilinear_f", dict(size=(7, 9), mode="bilinear", align_corners=False)), ("up_bilinear_t", dict(size=(6, 8), mode="bilinear", align_corners=True)),
                 ("up_bilinear_down", dict(size=(2, 3), mode="bilinear", align_corners=False))]:
    x = R(t((2, 3, 4, 5), 14.0)); out = F.interpolate(x, **kw); bwd(out)
    check(name + "_out", out); check(name + "_dx", x.grad)
print(f"{n} tensors compared, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
