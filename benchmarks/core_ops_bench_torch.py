"""PyTorch timings for the same operations as examples/core_ops_bench.rs (best of several runs, microseconds)."""
import sys
import time

import torch
import torch.nn.functional as F


def rnd(shape, seed):
    n = 1
    for d in shape:
        n *= d
    return torch.sin((torch.arange(n, dtype=torch.float32) + seed) * 0.37).reshape(shape)


def bench(name, f):
    if len(sys.argv) > 1 and name not in sys.argv[1].split(","):
        return
    f()
    best, runs, start = float("inf"), 0, time.perf_counter()
    while runs < 200 and (runs < 5 or time.perf_counter() - start < 0.5):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
        runs += 1
    print(f"BENCH {name:<28} {best * 1e6:>12.1f}")


a, b, row, small = rnd([1024, 1024], 0), rnd([1024, 1024], 1), rnd([1024], 2), rnd([64], 3)
img, w, seq = rnd([8, 16, 64, 64], 4), rnd([32, 16, 3, 3], 5), rnd([32, 128, 256], 6)
g, bb, m = rnd([256], 7), rnd([256], 8), rnd([512, 512], 9)
idx = (torch.arange(4096) * 7) % 1024
lk_a, lk_b = rnd([256, 4096], 10), rnd([4096, 256], 11)

bench("add 1M", lambda: a + b)
bench("mul 1M", lambda: a * b)
bench("add broadcast row 1M", lambda: a + row)
bench("mul_scalar 1M", lambda: a * 1.5)
bench("exp 1M", lambda: a.exp())
bench("tanh 1M", lambda: a.tanh())
bench("gelu 1M", lambda: F.gelu(a))
bench("relu 1M", lambda: a.relu())
bench("sum 1M", lambda: a.sum())
bench("sum axis0 1M", lambda: a.sum(0))
bench("sum axis1 1M", lambda: a.sum(1))
bench("max axis1 1M", lambda: a.max(1).values)
bench("softmax 1M", lambda: a.softmax(-1))
bench("transpose+contiguous 1M", lambda: a.t().contiguous())
bench("add transposed 1M", lambda: a.t() + b)
bench("matmul 512", lambda: m @ m)
bench("matmul 1024", lambda: a @ b)
bench("matmul 256x4096x256", lambda: lk_a @ lk_b)
bench("conv2d 8x16x64x64 k3", lambda: F.conv2d(img, w, padding=1))
img64, w64 = rnd([4, 64, 56, 56], 12), rnd([64, 64, 3, 3], 13)
bench("conv2d 4x64x56x56 k3", lambda: F.conv2d(img64, w64, padding=1))
bench("max_pool2d", lambda: F.max_pool2d(img, 2))
bench("layer_norm 32x128x256", lambda: F.layer_norm(seq, [256], g, bb))
bench("index_select 4096 rows", lambda: a.index_select(0, idx))
bench("cat 2x1M", lambda: torch.cat([a, b], 0))
bench("small add 64 (overhead)", lambda: small + small)
bench("small matmul 8x8", lambda: small.reshape(8, 8) @ small.reshape(8, 8))

x = rnd([256, 784], 10)
w1 = (rnd([784, 512], 11) * 0.05).requires_grad_(True)
w2 = (rnd([512, 10], 12) * 0.05).requires_grad_(True)


def step():
    w1.grad = None
    w2.grad = None
    ((x @ w1).relu() @ w2).log_softmax(-1).mean().backward()


bench("mlp train step b256", step)
ag = rnd([1024, 1024], 13).requires_grad_(True)


def chain():
    ag.grad = None
    ((ag * b).tanh() + b).square().sum().backward()


bench("autograd elementwise chain", chain)
