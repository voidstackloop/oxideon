"""Runs every benchmark of examples/core_ops_bench.rs and benchmarks/core_ops_bench_torch.py in its own process (with a pause
in between, so power throttling of long-running processes does not distort later rows) and prints both side by side.

    cargo build --release --example core_ops_bench
    python benchmarks/core_ops_bench_compare.py
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "target/release/examples/core_ops_bench.exe"
if not EXE.exists():
    EXE = ROOT / "target/release/examples/core_ops_bench"
NAMES = re.findall(r'bench\("([^"]+)"', (ROOT / "examples/core_ops_bench.rs").read_text(encoding="utf-8"))


def one(cmd, name):
    out = subprocess.run(cmd + [name], capture_output=True, text=True).stdout
    m = re.search(r"^BENCH .{28}\s*([\d.]+)", out, re.M)
    time.sleep(1.5)
    return float(m.group(1)) if m else float("nan")


print(f"{'op':<28}{'oxideon us':>12}{'torch us':>12}{'ratio':>8}")
for n in NAMES:
    o = one([str(EXE)], n)
    t = one([sys.executable, str(ROOT / "benchmarks/core_ops_bench_torch.py")], n)
    print(f"{n:<28}{o:>12.1f}{t:>12.1f}{o / t:>8.2f}", flush=True)
