<div align="center">

# Oxideon

**A fast, Hackable Deep Learning & Tensor Library in Rust — with a PyTorch-friendly Python API.**

[![Version](https://img.shields.io/badge/version-v0.0.1-blue.svg)](https://github.com/voidstackloop/oxideon/releases/tag/v0.0.1)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13%20(abi3)-blue.svg)](pyproject.toml)
[![Rust](https://img.shields.io/badge/rust-2024%20(1.85%2B)-orange.svg)](https://www.rust-lang.org)
[![Clippy](https://img.shields.io/badge/clippy-0%20warnings-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue.svg)](LICENSE-MIT)

[Why Oxideon?](#why-oxideon) · [Getting Started](#getting-started) · [Quick Tour](#quick-tour) · [PyTorch Cheat Sheet](#pytorch-cheat-sheet) · [Interop](#ecosystem-interop) · [Architecture](#how-its-organized) · [Docs](docs/)

</div>

---

## Why Oxideon?

Most deep learning stacks force a trade-off: either you use Python on top of a massive C++ runtime that's hard to inspect or embed, or you write everything in Rust and give up the ergonomics of NumPy and PyTorch.

**Oxideon** tries to give you both:
- **A clean Rust core (`oxideon-core`)** with multi-dtype tensors (`f32`, `f64`, `f16`, `bf16`, `fp8`, `c64`, `c128`, `i64`, `bool`), zero-copy strided views, higher-order autograd, blocked GEMM, and a caching buffer pool.
- **Native GPU & JIT execution** via `wgpu` (Vulkan / DX12 / Metal) and dynamic NVIDIA CUDA Driver PTX loading (`nvcuda`), plus an SSA graph compiler with operator fusion and WGSL/PTX codegen.
- **Familiar Python bindings (`PyO3` `abi3`)** — one wheel works across Python 3.9–3.13, releases the GIL during heavy compute, and talks zero-copy with NumPy and PyTorch via DLPack.
- **Batteries included for modern ML** — `nn.Module`, `AdamW`/schedule-free/2nd-order optimizers, SafeTensors & ONNX import/export, GGUF & 1.58-bit/4-bit quantization, PagedAttention & speculative decoding, DoRA/LoRA+, DPO/ORPO, and TCP/in-process distributed training (DDP + FSDP).

---

## Getting Started

### Build from source (Python + Rust)

```bash
git clone https://github.com/voidstackloop/oxideon.git
cd oxideon

# Create a virtualenv and install maturin
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install maturin numpy pytest

# Build and install the optimized Python extension
maturin develop --release
```

### Use from Rust

Add `oxideon-core` to your `Cargo.toml` or run the built-in tests and examples directly:

```bash
cargo test -p oxideon-core --lib
cargo run --release --example showcase
```

---

## Quick Tour

### 1. Tensors & Zero-Copy Strided Views

Oxideon tensors use signed `isize` strides, so transposes, flips, and stepped slices share underlying memory with zero copies:

```python
import oxideon as ox

x = ox.Tensor.arange(0.0, 24.0, 1.0).reshape([2, 3, 4])
v = x.transpose(0, 2)
assert v.shares_storage(x)

flipped = x[:, ::2, ::-1]
y = (x * 2.0 + 1.0).sum(dim=1)
```

### 2. Higher-Order Autograd

Need 1st, 2nd, or 3rd derivatives, Jacobians, or Hessians? Just set `create_graph=True`:

```python
import oxideon as ox

x = ox.Tensor.from_slice([3.0], [1], requires_grad=True)
y = x.powf(4.0)

dy  = ox.grad([y], [x], create_graph=True)[0]   # 4x^3  = 108.0
d2y = ox.grad([dy], [x], create_graph=True)[0]  # 12x^2 = 108.0
d3y = ox.grad([d2y], [x])[0]                    # 24x   = 72.0
```

### 3. Neural Networks (`ox.nn`) & Training

If you've ever written a PyTorch training loop, you already know how to write one in Oxideon:

```python
import oxideon as ox
import oxideon.nn as nn
import oxideon.optim as optim

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.norm(self.act(self.fc1(x))))

model = MLP(16, 64, 4)
opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

x = ox.randn([32, 16])
target = ox.randn([32, 4])

loss = ((model(x) - target) ** 2).mean()
opt.zero_grad()
loss.backward()
opt.step()
```

### 4. LLM Inference, Quantization & PEFT

Oxideon ships first-class primitives for running and fine-tuning language models:

```python
import oxideon as ox

# Paged KV cache (16-token blocks)
kv = ox.PagedKVCache(num_blocks=128, block_size=16, num_heads=8, head_dim=64)
table = kv.allocate(num_tokens=24)

# DoRA adapter fine-tuning that merges back into a plain Linear for inference
dora = ox.DoRALinear(in_features=256, out_features=256, r=8, lora_alpha=16.0)
out = dora.forward(ox.randn([16, 256]))
dora.merge()
```

You also get `FlashAttention-2` tiled online softmax, `SpeculativeDraftDecoder`, `Q4_0`/`Q8_0`/K-quants/GGUF loading, `BitLinear` 1.58-bit ternary layers, and alignment losses (`DPOLoss`, `ORPOLoss`, `KTOLoss`).

### 5. Distributed Training (DDP & FSDP)

Run multi-rank collectives over TCP or in-process threads with `NativeProcessGroup` and `FullyShardedParam` (`ox.distributed`), or launch with `RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` via `ProcessGroup.from_env()`:

```python
import oxideon as ox
import oxideon.distributed as dist

pg = dist.NativeProcessGroup.from_env()
w = ox.randn([1024, 1024])
sharded = dist.FullyShardedParam.from_tensor(w, pg)
full_w = sharded.gather(pg)
```

---

## PyTorch Cheat Sheet

| PyTorch (`torch`) | Oxideon (`ox`) |
|---|---|
| `torch.randn(3, 4)` | `ox.randn([3, 4])` |
| `torch.arange(0, 10)` | `ox.Tensor.arange(0.0, 10.0, 1.0)` |
| `x.transpose(0, 1)` / `x.t()` | `x.transpose(0, 1)` / `x.t()` |
| `y.backward()` | `y.backward()` |
| `torch.autograd.grad(y, x, create_graph=True)` | `ox.grad([y], [x], create_graph=True)` |
| `torch.nn.Module` / `Linear` | `ox.nn.Module` / `ox.nn.Linear` |
| `torch.optim.AdamW` | `ox.optim.AdamW` |
| `torch.cuda.is_available()` | `ox.cuda.is_available()` |
| `safetensors.save_file` | `ox.save_file` / `ox.load_file` |

---

## Ecosystem Interop

Oxideon plays nicely with the libraries you already use:

- **NumPy**: `ox.from_numpy(arr)`, `t.numpy()`, `np.asarray(t)` (`__array__` protocol)
- **PyTorch**: `ox.from_torch(t)`, `t.to_torch()`, zero-copy CPU DLPack (`__dlpack__`), and bidirectional `state_dict` helpers
- **SafeTensors & ONNX**: zero-copy SafeTensors I/O and ONNX export (both sequential models and compiled SSA graphs)
- **Data & Science**: Pandas, Polars, Apache Arrow, SciPy sparse matrices (`CSRMatrix`), Pillow (`from_pil` / `to_pil`), NetworkX, SymPy, and JAX

---

## How It's Organized

```text
oxideon/
├── crates/
│   ├── oxideon-core/      # Tensors, autograd, nn, optim, quant, JIT, compiler, CUDA/wgpu backends
│   └── oxideon-engines/   # Scientific, physics, PDE, and domain simulation engines
├── python/oxideon/        # High-level Python package (nn, optim, distributed, cuda, evals, vision, ...)
├── src/                   # PyO3 native extension bindings
├── examples/              # Runnable Python and Rust showcases
├── docs/                  # Architecture, tutorials, benchmarks, API stability & reference guides
└── tests/                 # Python test suite
```

---

## Accuracy and Speed vs PyTorch

Measured against PyTorch 2.15 (CPU) on the same machine: float32 on the CPU (13th-gen Core i7, 10 cores, AVX2/FMA, a thermally limited laptop), Oxideon in its default `Balanced` precision mode. Every number below is reproducible with the commands at the end of this section.

### Numerical error

The error of each result is measured against a float64 reference on identical inputs, in units of 2^-24 (half an ulp at 1.0): relative to the true value for element-wise functions, and relative to the sum of the absolute terms for sums, matrix products and convolutions (so cancellation cannot hide an error). "x better" is PyTorch's error divided by Oxideon's, so above 1 Oxideon is more accurate.

**40 operations: Oxideon is more accurate in 21 (12 of them by 2x or more), equal within 5% in 19, and less accurate in 0.**

<details>
<summary>All 40 operations (worst and mean error)</summary>

| operation | worst: Oxideon | PyTorch | x better | mean: Oxideon | PyTorch | x better |
|---|---:|---:|---:|---:|---:|---:|
| `sum_1M` | 0.0001 | 0.0004 | tie | 0.0001 | 0.0004 | tie |
| `sum_16M_offset` | 0.423 | 1.76 | 4.17x | 0.423 | 1.76 | 4.17x |
| `mean_4M_offset` | 0.253 | 1.39 | 5.47x | 0.253 | 1.39 | 5.47x |
| `sum_axis0` | 0.640 | 2.26 | 3.53x | 0.313 | 0.624 | 2.00x |
| `sum_axis1` | 0.639 | 2.44 | 3.82x | 0.316 | 0.660 | 2.09x |
| `dot_4M` | 0.0001 | 0.0033 | 55.75x | 0.0001 | 0.0033 | 55.75x |
| `norm_4M` | 0.056 | 1271 | 22680.31x | 0.056 | 1271 | 22680.31x |
| `matmul_512` | 1.02 | 2.40 | 2.35x | 0.127 | 0.211 | 1.66x |
| `matmul_k8192` | 0.180 | 0.295 | 1.64x | 0.038 | 0.060 | 1.58x |
| `softmax` | 8.93 | 68.78 | 7.70x | 1.40 | 11.98 | 8.54x |
| `log_softmax` | 1.42 | 2814 | 1981.95x | 0.366 | 0.602 | 1.65x |
| `logsumexp` | 0.972 | 1.02 | 1.05x | 0.368 | 0.369 | 1.00x |
| `layer_norm_offset` | 12.91 | 4969 | 384.80x | 0.534 | 363 | 680.76x |
| `exp` | 1.10 | 1.17 | 1.06x | 0.368 | 0.365 | 0.99x |
| `tanh` | 0.996 | 0.996 | 1.00x | 0.133 | 0.133 | 1.00x |
| `sigmoid` | 1.000 | 2.46 | 2.46x | 0.307 | 0.506 | 1.65x |
| `gelu` | 7.55e+05 | 1.69e+07 | 22.32x | 2960 | 2.68e+06 | 905.37x |
| `erf` | 0.998 | 1.12 | 1.13x | 0.283 | 0.284 | 1.00x |
| `ln` | 1.01 | 1.32 | 1.30x | 0.358 | 0.360 | 1.01x |
| `sqrt` | 1.000 | 1.000 | 1.00x | 0.360 | 0.360 | 1.00x |
| `conv2d_16ch` | 2.06 | 5.56 | 2.69x | 0.218 | 0.336 | 1.54x |
| `conv2d_64ch` | 1.09 | 4.97 | 4.58x | 0.124 | 0.331 | 2.67x |
| `cumsum_1M` | 0.727 | 0.727 | 1.00x | 0.196 | 0.196 | 1.00x |
| `sin` | 1.000 | 1.09 | 1.09x | 0.325 | 0.329 | 1.01x |
| `cos` | 1.000 | 1.09 | 1.09x | 0.325 | 0.329 | 1.01x |
| `log1p` | 1.00 | 1.04 | 1.04x | 0.360 | 0.360 | 1.00x |
| `expm1` | 0.999 | 1.38 | 1.38x | 0.359 | 0.359 | 1.00x |
| `softplus` | 0.999 | 2.00 | 2.00x | 0.323 | 0.389 | 1.20x |
| `silu` | 1.00 | 2.79 | 2.78x | 0.353 | 0.523 | 1.48x |
| `sinh` | 0.999 | 0.999 | 1.00x | 0.361 | 0.361 | 1.00x |
| `cosh` | 1.000 | 1.000 | 1.00x | 0.368 | 0.368 | 1.00x |
| `atan` | 0.997 | 1.38 | 1.38x | 0.335 | 0.337 | 1.01x |
| `rsqrt` | 1.50 | 1.50 | 1.00x | 0.496 | 0.496 | 1.00x |
| `pow_2_5` | 1.00 | 1.00 | 1.00x | 0.360 | 0.361 | 1.00x |
| `var_offset` | 0.993 | 0.993 | 1.00x | 0.373 | 0.373 | 1.00x |
| `batch_norm_offset` | 7.29 | 2034 | 278.83x | 0.533 | 480 | 900.52x |
| `group_norm_offset` | 5.27 | 1882 | 357.13x | 0.521 | 521 | 1000.91x |
| `cosine_sim` | 0.994 | 2294 | 2307.97x | 0.362 | 6.78 | 18.71x |
| `normalize_l2` | 0.999 | 3.22 | 3.22x | 0.361 | 0.625 | 1.73x |
| `mse` | 0.261 | 0.261 | 1.00x | 0.261 | 0.261 | 1.00x |

The `gelu` row is dominated by the relative error deep in the negative tail, where the result is about 1e-15; both libraries are far from the floor there.

</details>

Matrix products and convolutions trade accuracy for speed through one switch, `oxideon.set_precision_mode("fastest" | "balanced" | "highest")` (or `OXIDEON_PRECISION`): `Fastest` is PyTorch-class in accuracy and time, `Balanced` (default) is 1.5-2.4x more accurate, `Highest` 6-19x. Details: [docs/NUMERICAL_ACCURACY.md](docs/NUMERICAL_ACCURACY.md).

### Speed

Best of several runs, microseconds, each benchmark in its own process (so power throttling of a long run does not distort later rows). `ratio` is Oxideon's time divided by PyTorch's (below 1 is faster). The last columns repeat the numerical error of the same operation from the table above.

op                            oxideon us    torch us   ratio
add 1M                              40.6        47.2    0.86
mul 1M                              39.2        48.0    0.82
add broadcast row 1M                36.6        26.5    1.38
mul_scalar 1M                       33.1        29.6    1.12
exp 1M                              68.3        42.6    1.60
tanh 1M                            182.6       247.1    0.74
gelu 1M                            182.8       108.3    1.69
relu 1M                             33.3        28.2    1.18
sum 1M                              21.1        24.4    0.86
sum axis0 1M                        38.1        37.9    1.01
sum axis1 1M                        24.9        28.1    0.89
max axis1 1M                        54.4       105.4    0.52
softmax 1M                         120.2       129.1    0.93
transpose+contiguous 1M            484.9       764.5    0.63
add transposed 1M                  258.2       344.9    0.75
matmul 512                         390.8       392.0    1.00
matmul 1024                       3312.6      3045.3    1.09
matmul 256x4096x256                955.6       803.5    1.19
conv2d 8x16x64x64 k3               685.5       521.9    1.31
max_pool2d                          96.8       206.6    0.47
layer_norm 32x128x256               51.6        79.5    0.65
index_select 4096 rows             155.6       127.7    1.22
cat 2x1M                            57.3        72.4    0.79
small add 64 (overhead)              0.6         1.2    0.50
small matmul 8x8                     1.1         3.5    0.31
mlp train step b256                992.9       832.3    1.19
autograd elementwise chain         661.5       892.7    0.74

Operations whose result is a single correctly rounded IEEE operation (add, mul, relu, copies ...) are `exact` in both libraries. Run-to-run noise is about 10-15%.

### Reproduce

```bash
cargo build --release --example core_ops_bench --example accuracy_dump
python benchmarks/core_ops_bench_compare.py        # speed and error, side by side
python benchmarks/accuracy_vs_pytorch.py           # all error cases
OXIDEON_PRECISION=highest python benchmarks/accuracy_vs_pytorch.py matmul_512 conv2d_64ch
```

---

## Running Tests

```bash
# Rust core tests
cargo test -p oxideon-core --lib

# Workspace lint (zero warnings)
cargo clippy --workspace --all-targets -- -D warnings

# Python tests
pytest tests/ -q
```

Environment notes (mostly Windows):

- **Embedded Python.** With `--features python`, the tests that start an interpreter (`complete_dl`, `complete_millennium`) need to
  find the standard library; set `PYTHONHOME` to your Python installation, or they abort with `No module named 'encodings'`.
- **UTF-8 output.** The Python examples print characters such as `✓`. When their output is redirected on a console whose code page
  cannot show them, run Python with `-X utf8` (or set `PYTHONUTF8=1`).
- **CUDA kernels.** The `.cu` kernels need NVRTC. Without it the CUDA tests print `skipped: no CUDA device / NVRTC` and pass, and the
  operations fall back to the host. Install it with `pip install --target <dir> nvidia-cuda-nvrtc-cu12` and set
  `OXIDEON_NVRTC_SITE=<dir>` (see `backend/nvrtc.rs` for the search order); `cuda_bench` then reports its launched kernels.
- **Interop example.** `examples/python_interop_showcase.py` also needs `pillow`, `pandas`, `polars`, `networkx` and `sympy`.

---

## Note on Cryptography Modules

The `crypto` / `quantum` modules included in the scientific engines are **educational and complexity-estimation utilities only**. Do not use them to protect sensitive production data.

---

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
