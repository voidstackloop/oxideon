"""Square matmul timings, CPU against the wgpu device, for a range of sizes.

    python tools/gpu_bench.py

The GPU time includes copying the result back to the host (the only way to know the kernel has finished), which is a
real cost for anyone who needs the result on the CPU.
"""
import time

import oxideon as ox


def timed(fn, reps):
    fn()  # warm-up: shader compilation, allocations
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


print(ox.HardwareManager().summary())
print(f"{'n':>6} {'cpu ms':>10} {'gpu ms':>10} {'cpu GFLOP/s':>12} {'gpu GFLOP/s':>12} {'GPU speed-up':>13}")
for n in (128, 256, 512, 1024, 2048, 4096):
    a, b = ox.Tensor.randn([n, n]), ox.Tensor.randn([n, n])
    ga, gb = a.to_device("wgpu"), b.to_device("wgpu")
    reps = 3 if n >= 2048 else 10
    cpu = timed(lambda: (a @ b).sum().item(), reps)
    gpu = timed(lambda: (ga @ gb).cpu().sum().item(), reps)
    flops = 2.0 * n ** 3
    print(f"{n:>6} {cpu * 1e3:>10.2f} {gpu * 1e3:>10.2f} {flops / cpu / 1e9:>12.1f} {flops / gpu / 1e9:>12.1f} {cpu / gpu:>12.2f}x", flush=True)
