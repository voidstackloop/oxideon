"""Runs tensors on the first wgpu adapter and compares them with the CPU; fails on a software renderer or any mismatch.

    python tools/gpu_probe.py
"""
import sys
import time

import oxideon as ox

print("oxideon", ox.__version__)
summary = ox.HardwareManager().summary()
print(summary)
if "software renderer" in summary:
    sys.exit("only a software renderer was found: not a GPU result")

a = ox.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
b = ox.Tensor([0.5, -1.0, 2.0, 0.0, 1.5, -2.5], shape=[3, 2])
cpu = [round(v, 4) for v in (a @ b).to_list()]

ga, gb = a.to_device("wgpu"), b.to_device("wgpu")
print("device:", ga.device, "| is_gpu:", ga.is_gpu)
assert ga.is_gpu, "to_device('wgpu') did not land on a GPU"
assert [round(v, 4) for v in (ga @ gb).cpu().to_list()] == cpu, "GPU and CPU matmul disagree"

x = ox.Tensor([0.1 * i for i in range(1000)], shape=[1000]).to_device("wgpu")
got = ((x * 2.0 + 1.0).relu()).sum().cpu().item()
ref = sum(max(0.0, 0.1 * i * 2.0 + 1.0) for i in range(1000))
print(f"element-wise + sum on the GPU: {got:.3f}  python reference: {ref:.3f}")
assert abs(got - ref) / ref < 1e-4

n = 512
big_a, big_b = ox.Tensor.randn([n, n]), ox.Tensor.randn([n, n])
t0 = time.time()
c = (big_a @ big_b).sum().item()
t_cpu = time.time() - t0
ga2, gb2 = big_a.to_device("wgpu"), big_b.to_device("wgpu")
t0 = time.time()
g = (ga2 @ gb2).sum().cpu().item()
t_gpu = time.time() - t0
print(f"{n}x{n} matmul sum: cpu {c:.2f} ({t_cpu * 1000:.0f} ms)  gpu {g:.2f} ({t_gpu * 1000:.0f} ms)")
assert abs(c - g) <= 1e-3 * max(1.0, abs(c)), "large GPU and CPU matmul disagree"
print("GPU probe OK")
