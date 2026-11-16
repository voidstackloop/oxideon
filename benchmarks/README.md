# Oxideon Benchmarks

Reproducible performance benchmarks comparing Oxideon (`--release`) against PyTorch on CPU and GPU.

## Scripts in `benchmarks/`

| Script | Scope |
|---|---|
| [`bench_vs_pytorch.py`](bench_vs_pytorch.py) | Core ops, norms, `Conv2d`, and `MLP` forward+backward vs PyTorch |
| [`core_ops_bench_compare.py`](core_ops_bench_compare.py) | Core Rust vs PyTorch operation comparison runner |
| [`core_ops_bench_torch.py`](core_ops_bench_torch.py) | PyTorch reference timings for core operations |
| [`ops_bench.py`](ops_bench.py) | Tensor op microbenchmarks |
| [`gpu_bench.py`](gpu_bench.py) | GPU (`wgpu` / CUDA) throughput benchmarks |
| [`efficiency_bench.py`](efficiency_bench.py) | Memory & execution efficiency benchmarks |
| [`hardware_kernel_matrix_bench.py`](hardware_kernel_matrix_bench.py) | Cross-device kernel matrix benchmark |

## Rust Benchmarks

```bash
cargo run --release --example core_ops_bench
cargo run --release --example compile_bench
cargo run --release --features wgpu --example gpu_matmul_bench
```

## Running Python Benchmarks

```bash
maturin develop --release
python benchmarks/bench_vs_pytorch.py
```

## Latest Measured Results (`Oxideon 0.0.1` vs `PyTorch 2.15.0+cpu`, `--release`)

| Benchmark | Oxideon | PyTorch | Ratio (`Ox / Th`) |
|---|---:|---:|---:|
| `matmul 512x512` | **0.646 ms** | 0.776 ms | **0.83×** |
| `matmul 2048x2048` | 34.581 ms | 33.429 ms | **1.03×** |
| `add [64, 128, 768]` | **0.692 ms** | 0.753 ms | **0.92×** |
| `mul [64, 128, 768]` | **0.655 ms** | 0.791 ms | **0.83×** |
| `relu [64, 128, 768]` | **0.678 ms** | 0.773 ms | **0.88×** |
| `sigmoid [64, 128, 768]` | **0.924 ms** | 1.667 ms | **0.55×** |
| `silu [64, 128, 768]` | **1.005 ms** | 1.293 ms | **0.78×** |
| `gelu [64, 128, 768]` | 1.565 ms | 1.474 ms | **1.06×** |
| `exp [64, 128, 768]` | **0.818 ms** | 1.073 ms | **0.76×** |
| `softmax [64, 128, 768]` | **1.398 ms** | 1.732 ms | **0.81×** |
| `sum(-1) [64, 128, 768]` | **0.230 ms** | 0.236 ms | **0.98×** |
| `sum(0) [64, 128, 768]` | **0.391 ms** | 0.448 ms | **0.87×** |
| `layer_norm [64, 128, 768]` | **0.755 ms** | 0.877 ms | **0.86×** |
| `conv2d [16, 64, 32, 32]` | 7.762 ms | 2.844 ms | **2.73×** |
| `MLP fwd+bwd [128,768]` | 13.637 ms | 8.719 ms | **1.56×** |
| `AdamW step [10M params]` | **4.323 ms** | 12.620 ms | **0.34×** |

### Native CUDA PTX (`NVIDIA GeForce RTX 4060 Laptop GPU`, `ox.cuda.is_available() == True`)

- `CUDA matmul 1024x1024` (32x32 2x2 register-blocked PTX): **1.404 ms** (**1529.9 GFLOP/s**)
- `CUDA matmul 2048x2048` (32x32 2x2 register-blocked PTX): **10.148 ms** (**1692.9 GFLOP/s**)
