"""Runs every benchmark of examples/core_ops_bench.rs and benchmarks/core_ops_bench_torch.py in its own process (with a pause
in between, so power throttling of long-running processes does not distort later rows) and prints both side by side, together with
the numerical error of the same operation against a float64 reference (from `accuracy_vs_pytorch.py`; "x better" = PyTorch's error
divided by oxideon's, so above 1 oxideon is more accurate).

    cargo build --release --example core_ops_bench --example accuracy_dump
    python benchmarks/core_ops_bench_compare.py [--no-accuracy]

The accuracy case is the one named in the last column; its shape is not always the benchmark's. Operations whose result is one
correctly rounded IEEE operation (add, mul, relu, copies ...) are `exact` in both libraries.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
EXE = ROOT / "target/release/examples/core_ops_bench.exe"
if not EXE.exists():
    EXE = ROOT / "target/release/examples/core_ops_bench"
NAMES = re.findall(r'bench\("([^"]+)"', (ROOT / "examples/core_ops_bench.rs").read_text(encoding="utf-8"))

# benchmark -> case of accuracy_vs_pytorch.py measuring the same operation; "exact" = a single correctly rounded operation
ACCURACY = {
    "add 1M": "exact", "mul 1M": "exact", "add broadcast row 1M": "exact", "mul_scalar 1M": "exact", "relu 1M": "exact",
    "max axis1 1M": "exact", "transpose+contiguous 1M": "exact", "add transposed 1M": "exact", "index_select 4096 rows": "exact",
    "cat 2x1M": "exact", "max_pool2d": "exact",
    "exp 1M": "exp", "tanh 1M": "tanh", "gelu 1M": "gelu", "sum 1M": "sum_1M", "sum axis0 1M": "sum_axis0", "sum axis1 1M": "sum_axis1",
    "softmax 1M": "softmax", "matmul 512": "matmul_512", "matmul 256x4096x256": "matmul_k8192",
    "conv2d 8x16x64x64 k3": "conv2d_16ch", "conv2d 4x64x56x56 k3": "conv2d_64ch", "layer_norm 32x128x256": "layer_norm_offset",
}


def one(cmd, name):
    out = subprocess.run(cmd + [name], capture_output=True, text=True).stdout
    m = re.search(r"^BENCH .{28}\s*([\d.]+)", out, re.M)
    time.sleep(1.5)
    return float(m.group(1)) if m else float("nan")


acc = {}
if "--no-accuracy" not in sys.argv:
    import accuracy_vs_pytorch

    acc = accuracy_vs_pytorch.measure({c for c in ACCURACY.values() if c != "exact"})


def fx(v):
    return "tie" if v != v else "inf" if v == float("inf") else f"{v:.2f}x"


def error_columns(name):
    case = ACCURACY.get(name)
    if case is None:
        return f"{'-':>9}{'-':>9}  -"
    if case == "exact":
        return f"{'exact':>9}{'exact':>9}  both libraries exact"
    if case not in acc:
        return f"{'?':>9}{'?':>9}  {case}"
    _, _, bw, _, _, bm, _ = acc[case]
    return f"{fx(bw):>9}{fx(bm):>9}  {case}"


print(f"{'op':<28}{'oxideon us':>12}{'torch us':>12}{'ratio':>8}" + (f"{'err worst':>11}{'err mean':>9}  accuracy case" if acc else ""))
for n in NAMES:
    o = one([str(EXE)], n)
    t = one([sys.executable, str(ROOT / "benchmarks/core_ops_bench_torch.py")], n)
    print(f"{n:<28}{o:>12.1f}{t:>12.1f}{o / t:>8.2f}{error_columns(n) if acc else ''}", flush=True)
