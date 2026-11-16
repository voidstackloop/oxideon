"""Timings for the operations a training step is made of, on the CPU and on the wgpu device (when there is one).

    python tools/ops_bench.py

Every GPU run is ended with a scalar sum read back to the host, which is how the benchmark knows the kernels have finished.
"""
import statistics
import time

import oxideon as ox


def timed(fn, reps=5):
    fn()  # warm-up: shader compilation, allocations, thread pool
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def has_gpu():
    try:
        ox.Tensor.zeros([1]).to_device("wgpu")
        return "software renderer" not in ox.HardwareManager().summary()
    except Exception:
        return False


GPU = has_gpu()
print(ox.HardwareManager().summary())
print(f"{'operation':<34}{'cpu ms':>10}" + (f"{'gpu ms':>10}{'gpu speed-up':>14}" if GPU else ""))


def row(name, make, run):
    cpu_args = make()
    cpu = timed(lambda: run(*cpu_args).sum().item())
    line = f"{name:<34}{cpu:>10.2f}"
    if GPU:
        gpu_args = [a.to_device("wgpu") for a in cpu_args]
        gpu = timed(lambda: run(*gpu_args).sum().item())
        line += f"{gpu:>10.2f}{cpu / gpu:>13.2f}x"
    print(line, flush=True)


N = 10_000_000
row("add, 10M f32", lambda: [ox.Tensor.randn([N]), ox.Tensor.randn([N])], lambda a, b: a + b)
row("exp, 10M f32", lambda: [ox.Tensor.randn([N])], lambda a: a.exp())
row("gelu, 10M f32", lambda: [ox.Tensor.randn([N])], lambda a: a.gelu())
row("cmp gt, 10M f32", lambda: [ox.Tensor.randn([N]), ox.Tensor.randn([N])], lambda a, b: a > b)
row("sum, 10M f32", lambda: [ox.Tensor.randn([N])], lambda a: a.sum())
row("sum over axis 0, 4096 x 4096", lambda: [ox.Tensor.randn([4096, 4096])], lambda a: a.sum(dim=0))
row("sum over axis 1, 4096 x 4096", lambda: [ox.Tensor.randn([4096, 4096])], lambda a: a.sum(dim=1))
row("bias add, 8192 x 1024", lambda: [ox.Tensor.randn([8192, 1024]), ox.Tensor.randn([1024])], lambda a, b: a + b)
row("bias add (sym), 1024 + 8192x1024", lambda: [ox.Tensor.randn([1024]), ox.Tensor.randn([8192, 1024])], lambda a, b: a + b)
row("transpose + relu, 4096 x 4096", lambda: [ox.Tensor.randn([4096, 4096])], lambda a: a.t().relu())
row("softmax, 4096 x 1024", lambda: [ox.Tensor.randn([4096, 1024])], lambda a: a.softmax(1))
for n in (512, 1024, 2048, 4096):
    row(f"matmul {n}x{n} f32", lambda n=n: [ox.Tensor.randn([n, n]), ox.Tensor.randn([n, n])], lambda a, b: a @ b)
for n in (512, 1024, 2048):
    a, b = ox.Tensor.randn([n, n], dtype="f64"), ox.Tensor.randn([n, n], dtype="f64")
    print(f"{f'matmul {n}x{n} f64 (CPU only)':<34}{timed(lambda: (a @ b).sum().item(), 3):>10.2f}", flush=True)
