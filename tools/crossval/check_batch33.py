"""Batch 33: FFT family, 2-D FFT, STFT, complex helpers: values and gradients vs torch.fft (float64)."""
import math, re, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

arrs = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

def genv(n, salt): return [math.sin(0.53 * i + salt) * 1.1 + 0.2 * math.cos(0.17 * i * salt) for i in range(n)]
def t(shape, salt): return torch.tensor(genv(int(np.prod(shape)), salt), dtype=torch.float64).reshape(tuple(shape))
bad = n = 0; worst = 0.0
def check(name, tensor, tol=1e-9):
    global bad, n, worst
    n += 1
    if name not in arrs: bad += 1; print("MISSING", name); return
    want = tensor.detach().double().flatten().numpy(); got = arrs[name]
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want)) / max(1.0, np.abs(want).max()); worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:4], want[:4])

C = torch.view_as_complex; Rl = torch.view_as_real
def case(name, x, out_fn):
    out = out_fn(x)
    check(name + "_out", out)
    (out * t(list(out.shape), 77.0)).sum().backward()
    check(name + "_dx", x.grad)

for nn in [8, 16, 12, 7, 10, 1, 2, 9]:
    x = t((3, nn, 2), 1.0 + nn).requires_grad_(True); case(f"fft{nn}", x, lambda v: Rl(torch.fft.fft(C(v.contiguous()), dim=1)))
    x = t((3, nn, 2), 2.0 + nn).requires_grad_(True); case(f"ifft{nn}", x, lambda v: Rl(torch.fft.ifft(C(v.contiguous()), dim=1)))
    x = t((2, nn), 3.0 + nn).requires_grad_(True); case(f"rfft{nn}", x, lambda v: Rl(torch.fft.fft(torch.complex(v, torch.zeros_like(v)), dim=1)))
    # irfft here = real part of the full inverse transform of a two-sided spectrum
    x = t((2, nn, 2), 4.0 + nn).requires_grad_(True); case(f"irfft{nn}", x, lambda v: torch.fft.ifft(C(v.contiguous()), dim=1).real)
for h, w in [(4, 8), (6, 5), (3, 4)]:
    x = t((2, h, w, 2), 5.0 + h).requires_grad_(True); case(f"fft2_{h}x{w}", x, lambda v: Rl(torch.fft.fft2(C(v.contiguous()), dim=(1, 2))))
    x = t((2, h, w, 2), 6.0 + h).requires_grad_(True); case(f"ifft2_{h}x{w}", x, lambda v: Rl(torch.fft.ifft2(C(v.contiguous()), dim=(1, 2))))
b = t((4, 5, 2), 8.0)
x = t((4, 5, 2), 7.0).requires_grad_(True); case("cmul", x, lambda v: Rl(C(v.contiguous()) * C(b.contiguous())))
x = t((4, 5, 2), 9.0).requires_grad_(True); case("cabs", x, lambda v: torch.sqrt((v ** 2).sum(-1)))
for length, n_fft, hop, win in [(40, 8, 4, True), (37, 12, 5, True), (32, 8, 8, False), (20, 16, 1, True)]:
    x = t((2, length), 10.0 + length).requires_grad_(True)
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64) if win else torch.ones(n_fft, dtype=torch.float64)
    # frames [B, F, n_fft] -> full two-sided spectrum [B, F, n_fft, 2], like the Rust layout
    case(f"stft_{length}_{n_fft}_{hop}_{int(win)}", x, lambda v: Rl(torch.fft.fft(v.unfold(1, n_fft, hop) * window, dim=-1)))
print(f"{n} tensors compared, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
