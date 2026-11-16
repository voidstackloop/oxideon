"""Batch 36: fake quantization and min-max observers vs torch / torch.ao.quantization."""
import math, re, sys, warnings
import numpy as np
import torch
from torch.ao.quantization import MinMaxObserver, MovingAverageMinMaxObserver, PerChannelMinMaxObserver
warnings.filterwarnings("ignore")

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass
def data(n, salt, amp): return [(math.sin(0.61 * i + salt) + 0.3 * math.cos(0.23 * i)) * amp for i in range(n)]
bad = n = 0; worst = 0.0
def check(name, want, tol=1e-9):
    global bad, n, worst
    n += 1
    want = np.asarray(want.detach() if torch.is_tensor(want) else want, dtype=float).ravel(); got = arrs[name]
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want) / np.maximum(1.0, np.abs(want))); worst = max(worst, err)
    if err > tol:
        bad += 1; i = int(np.argmax(np.abs(got - want))); print("MISMATCH", name, f"err {err:.2e} at {i}: got {got[i]} want {want[i]}")

xv = data(24, 0.4, 3.0) + [0.125, 0.375, -0.125, -0.375, 0.625, 2.5 * 0.25, -1.5 * 0.25, 40.0, -40.0, 31.0, 31.125, -32.75, -32.875]
for tag, scale, zp, (qmin, qmax) in [("u8", 0.25, 3, (0, 255)), ("i8", 0.25, 0, (-128, 127)), ("i4", 0.6, -2, (-8, 7))]:
    x = torch.tensor(xv, dtype=torch.float64, requires_grad=True)
    y = torch.fake_quantize_per_tensor_affine(x, scale, zp, qmin, qmax)
    check(f"fq_{tag}_out", y, 1e-6)   # torch holds the scale in float32
    w = torch.tensor([0.5 + 0.1 * i for i in range(len(xv))], dtype=torch.float64)
    (y * w).sum().backward(); check(f"fq_{tag}_dx", x.grad)

batches = [torch.tensor(data(30, 1.0 + k, 1.0 + 0.7 * k), dtype=torch.float64).reshape(5, 6) for k in range(4)]
SCH = {"affine_u8": (torch.per_tensor_affine, torch.quint8, 0, 255), "sym_i8": (torch.per_tensor_symmetric, torch.qint8, -128, 127),
       "sym_u8": (torch.per_tensor_symmetric, torch.quint8, 0, 255), "affine_i8": (torch.per_tensor_affine, torch.qint8, -128, 127)}
for tag, (qs, dt, qmin, qmax) in SCH.items():
    o = MinMaxObserver(dtype=dt, qscheme=qs, eps=1e-12)   # default integer range of the dtype
    for b in batches: o(b)
    s, z = o.calculate_qparams(); check(f"obs_{tag}_scale", s.double(), 1e-6); check(f"obs_{tag}_zp", z.double())
    o = MovingAverageMinMaxObserver(averaging_constant=0.2, dtype=dt, qscheme=qs, eps=1e-12)
    for b in batches: o(b)
    s, z = o.calculate_qparams(); check(f"ma_{tag}_scale", s.double(), 1e-6); check(f"ma_{tag}_zp", z.double())
for tag, qs, dt, qmin, qmax in [("pc_sym_i8", torch.per_channel_symmetric, torch.qint8, -128, 127), ("pc_affine_u8", torch.per_channel_affine, torch.quint8, 0, 255)]:
    o = PerChannelMinMaxObserver(ch_axis=1, dtype=dt, qscheme=qs, eps=1e-12)
    for b in batches: o(b)
    s, z = o.calculate_qparams(); check(f"{tag}_scale", s.double(), 1e-6); check(f"{tag}_zp", z.double())
    x = batches[3].clone().float().requires_grad_(True)
    y = torch.fake_quantize_per_channel_affine(x, s.float(), z.int(), 1, qmin, qmax)
    check(f"{tag}_fq", y, 1e-6); y.sum().backward()
    # the observed extreme of a symmetric range maps to exactly +-127.5 grid steps, so whether it saturates is decided
    # by float32 rounding in torch; exclude those elements from the gradient comparison
    all_b = torch.stack(batches); amax = all_b.abs().amax(dim=(0, 1))
    edge = (batches[3].abs() >= amax * (1 - 1e-6)).numpy().ravel()
    g = x.grad.double().numpy().ravel().copy(); got = arrs[f"{tag}_fq_dx"]
    g[edge] = got[edge]
    check(f"{tag}_fq_dx", g)
print(f"{n} tensors compared, worst error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
