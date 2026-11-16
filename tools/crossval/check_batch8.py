"""Eighth cross-validation batch: linear algebra, windows/DCT, polynomial/interpolation/convolution vs NumPy / SciPy.

    cargo test -p oxideon-core --test crossval_emit8 -- --nocapture --ignored > rust_values8.txt
    py -3.11 tools/crossval/check_batch8.py rust_values8.txt
"""
import sys

import numpy as np
from scipy import fft, linalg, signal

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass

A = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, -0.2], [0.5, -0.2, 2.0]])
G = np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 2.2], [4.0, 0.1, 1.5]])
ref = {}


def put(name, arr):
    for i, v in enumerate(np.asarray(arr, dtype=float).ravel()):
        ref[f"{name}_{i}"] = float(v)


put("svdvals", np.linalg.svd(G, compute_uv=False))
put("eigvalsh", np.linalg.eigvalsh(A))
put("logdet", np.linalg.slogdet(A)[1])
for k, o in [("fro", "fro"), ("nuc", "nuc"), ("2", 2), ("1", 1), ("inf", np.inf)]:
    put(f"norm_{k}", np.linalg.norm(G, o))
put("vnorm3", np.linalg.norm(np.array([1.0, -2.0, 3.0, 0.5]), 3))
ev = np.linalg.eigvals(np.array([[0.0, 1.0, 0.0], [-2.0, 0.0, 1.0], [0.5, 0.3, 0.1]]))
ev = sorted(ev, key=lambda z: (round(z.real * 1e8), z.imag))
for i, z in enumerate(ev):
    ref[f"eigvals_re_{i}"] = z.real
    ref[f"eigvals_im_{i}"] = z.imag
put("chol_inv", np.linalg.inv(A))
put("sqrtm", linalg.sqrtm(A).real)
put("logm", linalg.logm(A).real)
put("hann_p", signal.windows.hann(9, sym=False))
put("hann_s", signal.windows.hann(9, sym=True))
put("hamming_p", signal.windows.hamming(9, sym=False))
put("hamming_s", signal.windows.hamming(9, sym=True))
put("blackman_s", signal.windows.blackman(9, sym=True))
put("bartlett_s", signal.windows.bartlett(9, sym=True))
put("kaiser_s", signal.windows.kaiser(9, 8.6, sym=True))
put("fftfreq", np.fft.fftfreq(8, 0.5))
put("rfftfreq", np.fft.rfftfreq(8, 0.5))
sig = np.array([0.3, -1.2, 2.5, 0.9, -0.4, 1.8, 0.0, 0.7])
put("dct", fft.dct(sig, type=2))
put("dct_ortho", fft.dct(sig, type=2, norm="ortho"))
put("idct_ortho", sig)
put("idct", fft.idct(sig, type=2))
xs = np.array([0.0, 1.0, 2.0, 3.5, 5.0, 6.0])
ys = np.array([1.0, 0.5, 2.2, 1.9, 4.0, 3.1])
put("polyfit2", np.polyfit(xs, ys, 2))
put("polyfit3", np.polyfit(xs, ys, 3))
put("polyval", np.polyval([2.0, -1.0, 0.5, 3.0], [-1.5, 0.0, 2.2]))
put("interp", np.interp([-1.0, 0.5, 2.7, 4.2, 7.0], xs, ys))
put("roots", np.sort(np.roots([1.0, -6.0, 11.0, -6.0]).real))
ref["roots_n_0"] = 3.0
rust["roots_n_0"] = rust.get("roots_n", float("nan"))
u, v = np.array([1.0, 2.0, 3.0, -1.0, 0.5]), np.array([0.5, -1.0, 2.0])
for mode in ("full", "same", "valid"):
    put(f"conv_{mode}", np.convolve(u, v, mode))
    put(f"corr_{mode}", np.correlate(u, v, mode))
put("fftconv", signal.fftconvolve(u, v))

worst, bad = 0.0, []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 1e-6:
        bad.append((k, rust[k], want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
for k, got, want in bad:
    print(f"MISMATCH {k}: rust={got} reference={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
