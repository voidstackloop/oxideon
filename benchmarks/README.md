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
| `matmul 512x512` | **0.699 ms** | 0.706 ms | **0.99×** |
| `matmul 2048x2048` | 30.258 ms | 27.211 ms | **1.11×** |
| `add [64, 128, 768]` | **0.645 ms** | 0.715 ms | **0.90×** |
| `mul [64, 128, 768]` | 0.658 ms | 0.648 ms | **1.01×** |
| `relu [64, 128, 768]` | 0.687 ms | 0.666 ms | **1.03×** |
| `sigmoid [64, 128, 768]` | **0.918 ms** | 1.048 ms | **0.88×** |
| `silu [64, 128, 768]` | **0.872 ms** | 1.086 ms | **0.80×** |
| `gelu [64, 128, 768]` | 1.319 ms | 0.903 ms | **1.46×** |
| `exp [64, 128, 768]` | **0.798 ms** | 0.902 ms | **0.89×** |
| `softmax [64, 128, 768]` | **1.201 ms** | 5.048 ms | **0.24×** |
| `sum(-1) [64, 128, 768]` | **1.210 ms** | 1.828 ms | **0.66×** |
| `sum(0) [64, 128, 768]` | **1.656 ms** | 2.088 ms | **0.79×** |
| `layer_norm [64, 128, 768]` | **2.917 ms** | 3.631 ms | **0.80×** |
| `conv2d [16, 64, 32, 32]` | 21.853 ms | 10.044 ms | **2.18×** |
| `MLP fwd+bwd [128, 768]` | 36.408 ms | 34.993 ms | **1.04×** |

- **GPU detected**: `NVIDIA GeForce RTX 4060 Laptop GPU` (`ox.cuda.is_available() == True`).
