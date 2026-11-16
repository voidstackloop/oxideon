"""GPU soak test: forward + backward on the wgpu device against the CPU for many shapes, a VRAM leak check, and a probe of
very large allocations. Needs an installed oxideon wheel built with the `wgpu` feature and a real (not software) adapter.

    python tools/gpu_soak.py [SECONDS]        (default 300)

Exit code 0 only if every GPU result matched the CPU, VRAM stayed flat and no large allocation crashed the process.
"""
import subprocess
import sys
import time

import oxideon as ox

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
SHAPES = [(8, 16, 8), (37, 129, 53), (64, 64, 64), (128, 256, 64), (512, 512, 512), (1024, 1024, 256), (2048, 512, 512)]


def vram_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:  # no nvidia-smi: the leak check is skipped
        return None


def max_diff(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0), max((abs(y) for y in b), default=0.0)


def step(a, b):
    """tanh(a @ b) scaled and shifted, summed, with gradients for both inputs. (Smooth on purpose: with a ReLU, an entry
    within rounding error of zero takes a different branch on the GPU and the gradients legitimately differ.)"""
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    y = ((a @ b).tanh() * 0.5 + 0.1).sum()
    y.backward()
    return y.cpu().item(), a.grad().cpu().to_list(), b.grad().cpu().to_list()


summary = ox.HardwareManager().summary()
print(summary)
if "software renderer" in summary:
    sys.exit("only a software renderer was found: not a GPU result")

ox.manual_seed(1234)
problems = []
rounds, t_cpu, t_gpu = 0, 0.0, 0.0
vram = [vram_mib()]
t_end = time.time() + SECONDS
while time.time() < t_end:
    rounds += 1
    worst = 0.0
    for m, k, n in SHAPES:
        a, b = ox.Tensor.randn([m, k]), ox.Tensor.randn([k, n])
        t0 = time.time()
        yc, gac, gbc = step(a, b)
        t_cpu += time.time() - t0
        t0 = time.time()
        yg, gag, gbg = step(a.to_device("wgpu"), b.to_device("wgpu"))
        t_gpu += time.time() - t0
        dy = abs(yc - yg) / max(1.0, abs(yc))
        da, sa = max_diff(gag, gac)
        db, sb = max_diff(gbg, gbc)
        rel = max(dy, da / max(1.0, sa), db / max(1.0, sb))
        worst = max(worst, rel)
        if rel > 1e-2 or any(v != v for v in (yg, *gag[:8], *gbg[:8])):
            problems.append(f"round {rounds} shape {(m, k, n)}: GPU and CPU differ by {rel:.2e} (loss {yc} vs {yg})")
    vram.append(vram_mib())
    print(f"round {rounds}: worst relative difference {worst:.1e}, cpu {t_cpu:.1f}s total, gpu {t_gpu:.1f}s total, VRAM {vram[-1]} MiB", flush=True)

if vram[0] is not None and len(vram) > 4 and vram[-1] - max(vram[1:3]) > 1024:
    problems.append(f"VRAM grew from {max(vram[1:3])} MiB (after warm-up) to {vram[-1]} MiB over {rounds} rounds: possible leak")

# Very large buffers go to a child process: a wgpu validation error may abort it, and that must not end this report.
CHILD = ("import oxideon as ox, sys\n"
         "n = int(sys.argv[1])\n"
         "a = ox.Tensor.randn([n, 64]).to_device('wgpu'); b = ox.Tensor.randn([64, n]).to_device('wgpu')\n"
         "print((a @ b).sum().cpu().item())\n")
for n in (4096, 8192, 16384, 32768):
    r = subprocess.run([sys.executable, "-c", CHILD, str(n)], capture_output=True, text=True, timeout=600)
    lines = r.stderr.strip().splitlines()
    err = next((ln for ln in reversed(lines) if "Error" in ln), lines[-1] if lines else "")[:160]
    if r.returncode == 0:
        print(f"[{n} x {n}] result {r.stdout.strip()}")
    elif "Traceback" in r.stderr and "panicked at" not in r.stderr:
        print(f"[{n} x {n}] refused with an ordinary Python error: {err}")  # fine: an error, not a panic or a crash
    else:
        problems.append(f"{n} x {n} matmul: exit {r.returncode}, Rust panic or crash: {err}")
        print(f"[{n} x {n}] PANIC OR CRASH, exit {r.returncode}: {err}")

print(f"{rounds} rounds; cpu {t_cpu:.1f}s, gpu {t_gpu:.1f}s")
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
