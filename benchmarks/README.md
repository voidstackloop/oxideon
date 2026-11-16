# Oxideon Benchmarks

Reproducible performance benchmarks comparing Oxideon (`--release`) against PyTorch on CPU and GPU.

## Running

Build the release extension first, then run the benchmark suite:

```bash
maturin develop --release
python benchmarks/bench_vs_pytorch.py
```

## Included Workloads

- **GEMM**: `512x512` and `2048x2048` `f32` matrix multiplication
- **Elementwise & Activations (`[64, 128, 768]`)**: `add`, `mul`, `relu`, `sigmoid`, `silu`, `gelu`, `exp`
- **Reductions & Norms (`[64, 128, 768]`)**: `softmax(-1)`, `sum(-1)`, `sum(0)`, `LayerNorm(768)`
- **Vision**: `Conv2d(64, 64, kernel_size=3, padding=1)` on `[16, 64, 32, 32]`
- **Training Step**: `MLP(768 -> 3072 -> 768)` forward + backward on `[128, 768]`
