"""Speed and numerical error of Oxideon against PyTorch, for every precision mode.

Runs every benchmark of examples/core_ops_bench.rs and benchmarks/core_ops_bench_torch.py in its own process (with a pause in
between, so power throttling of long-running processes does not distort later rows), and measures the numerical error of the same
operations against a float64 reference (`accuracy_vs_pytorch.py`; "x better" = PyTorch's error divided by Oxideon's, so above 1
Oxideon is more accurate).

Oxideon has three precision modes (`oxideon.set_precision_mode`, or the `OXIDEON_PRECISION` environment variable):

  fastest   PyTorch-class accuracy at PyTorch-class speed (plain f32 accumulation, cheaper gelu)
  balanced  the default: 1.5-2.4x more accurate than PyTorch on matrix products and convolutions, about the same speed
  highest   f64 accumulation: 6-19x more accurate, 2-8x slower

Only matrix products, convolutions and gelu depend on the mode; they are run once per mode. Every other operation is identical in
all three modes and is run once.

    cargo build --release --example core_ops_bench --example accuracy_dump
    python benchmarks/core_ops_bench_compare.py [--markdown] [--no-accuracy] [--modes fastest,balanced,highest] [--repeat N]

`--repeat N` measures every time N times, spread over the run, and keeps the best (a laptop that throttles after minutes of load
would otherwise give one row a bad moment).

The accuracy case is the one named in the last column; its shape is not always the benchmark's. Operations whose result is one
correctly rounded IEEE operation (add, mul, relu, copies ...) are `exact` in both libraries.
"""
import os
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

ALL_MODES = ["fastest", "balanced", "highest"]
MARKDOWN = "--markdown" in sys.argv
WITH_ACCURACY = "--no-accuracy" not in sys.argv
MODES = ALL_MODES
REPEAT = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
if "--modes" in sys.argv:
    MODES = sys.argv[sys.argv.index("--modes") + 1].split(",")

# benchmarks whose time depends on the precision mode (the others are measured once, in the default mode)
MODE_DEPENDENT = {"gelu 1M", "matmul 512", "matmul 1024", "matmul 256x4096x256", "conv2d 8x16x64x64 k3", "conv2d 4x64x56x56 k3", "mlp train step b256"}

# benchmark -> case of accuracy_vs_pytorch.py measuring the same operation; "exact" = a single correctly rounded operation
ACCURACY = {
    "add 1M": "exact", "mul 1M": "exact", "add broadcast row 1M": "exact", "mul_scalar 1M": "exact", "relu 1M": "exact",
    "max axis1 1M": "exact", "transpose+contiguous 1M": "exact", "add transposed 1M": "exact", "index_select 4096 rows": "exact",
    "cat 2x1M": "exact", "max_pool2d": "exact",
    "exp 1M": "exp", "tanh 1M": "tanh", "gelu 1M": "gelu", "sum 1M": "sum_1M", "sum axis0 1M": "sum_axis0", "sum axis1 1M": "sum_axis1",
    "softmax 1M": "softmax", "matmul 512": "matmul_512", "matmul 256x4096x256": "matmul_k8192",
    "conv2d 8x16x64x64 k3": "conv2d_16ch", "conv2d 4x64x56x56 k3": "conv2d_64ch", "layer_norm 32x128x256": "layer_norm_offset",
}


def once(cmd, name, mode=None):
    env = dict(os.environ)
    if mode:
        env["OXIDEON_PRECISION"] = mode
    out = subprocess.run(cmd + [name], capture_output=True, text=True, env=env).stdout
    m = re.search(r"^BENCH .{28}\s*([\d.]+)", out, re.M)
    time.sleep(1.5)
    return float(m.group(1)) if m else float("nan")


def one(cmd, name, mode=None):
    """Best of `REPEAT` separate processes."""
    return min(once(cmd, name, mode) for _ in range(REPEAT))


def fx(v):
    return "tie" if v != v else "inf" if v == float("inf") else f"{v:.2f}x"


# error[mode][case] = (x better worst, x better mean)
error = {}
if WITH_ACCURACY:
    import accuracy_vs_pytorch

    cases = {c for c in ACCURACY.values() if c != "exact"}
    dependent_cases = {ACCURACY[n] for n in MODE_DEPENDENT if ACCURACY.get(n, "exact") != "exact"}
    for mode in MODES:
        os.environ["OXIDEON_PRECISION"] = mode
        rows = accuracy_vs_pytorch.measure(dependent_cases if mode != "balanced" else cases)
        error[mode] = {c: (r[2], r[5]) for c, r in rows.items()}
    os.environ.pop("OXIDEON_PRECISION", None)
    if "balanced" not in error:  # the independent operations are measured in the default mode
        os.environ["OXIDEON_PRECISION"] = "balanced"
        rows = accuracy_vs_pytorch.measure(cases)
        error["balanced"] = {c: (r[2], r[5]) for c, r in rows.items()}
        os.environ.pop("OXIDEON_PRECISION", None)


def err(mode, name):
    """(worst, mean) 'x better' text for a benchmark in a mode."""
    case = ACCURACY.get(name)
    if case is None:
        return "-", "-"
    if case == "exact":
        return "exact", "exact"
    if not WITH_ACCURACY or case not in error.get(mode, {}):
        return "-", "-"
    w, m = error[mode][case]
    return fx(w), fx(m)


dependent = [n for n in NAMES if n in MODE_DEPENDENT]
others = [n for n in NAMES if n not in MODE_DEPENDENT]
torch_cmd = [sys.executable, str(ROOT / "benchmarks/core_ops_bench_torch.py")]

# 1. the operations that depend on the mode: one run per mode
t_dep = {}
o_dep = {}
for n in dependent:
    t_dep[n] = one(torch_cmd, n)
    for mode in MODES:
        o_dep[n, mode] = one([str(EXE)], n, mode)
# 2. everything else, once, in the default mode
o_oth, t_oth = {}, {}
for n in others:
    o_oth[n] = one([str(EXE)], n, "balanced")
    t_oth[n] = one(torch_cmd, n)

if MARKDOWN:
    print("| operation | precision mode | Oxideon us | PyTorch us | time ratio | error worst (x better) | error mean (x better) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for n in dependent:
        for i, mode in enumerate(MODES):
            w, m = err(mode, n)
            name = n if i == 0 else ""
            print(f"| {name} | {mode}{' (default)' if mode == 'balanced' else ''} | {o_dep[n, mode]:.1f} | {t_dep[n]:.1f} | {o_dep[n, mode] / t_dep[n]:.2f} | {w} | {m} |")
    print()
    print("| operation (same in every mode) | Oxideon us | PyTorch us | time ratio | error worst (x better) | error mean (x better) |")
    print("|---|---:|---:|---:|---:|---:|")
    for n in others:
        w, m = err("balanced", n)
        print(f"| {n} | {o_oth[n]:.1f} | {t_oth[n]:.1f} | {o_oth[n] / t_oth[n]:.2f} | {w} | {m} |")
else:
    print("operations that depend on the precision mode (us, ratio to PyTorch, error x better worst / mean)")
    head = f"{'op':<24}{'torch us':>10}"
    for mode in MODES:
        head += f"{mode + ' us':>14}{'ratio':>7}{'err w/m':>16}"
    print(head)
    for n in dependent:
        line = f"{n:<24}{t_dep[n]:>10.1f}"
        for mode in MODES:
            w, m = err(mode, n)
            line += f"{o_dep[n, mode]:>14.1f}{o_dep[n, mode] / t_dep[n]:>7.2f}{w + ' / ' + m:>16}"
        print(line)
    print("\noperations that are the same in every mode")
    print(f"{'op':<28}{'oxideon us':>12}{'torch us':>12}{'ratio':>8}{'err worst':>11}{'err mean':>9}")
    for n in others:
        w, m = err("balanced", n)
        print(f"{n:<28}{o_oth[n]:>12.1f}{t_oth[n]:>12.1f}{o_oth[n] / t_oth[n]:>8.2f}{w:>11}{m:>9}")
