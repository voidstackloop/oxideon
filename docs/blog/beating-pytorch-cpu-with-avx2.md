# Beating PyTorch CPU on AdamW, LayerNorm, and Softmax with AVX2+FMA in Rust

When we first benchmarked **Oxideon** against PyTorch (`2.15.0+cpu`), GEMM via blocked kernels quickly approached MKL parity, yet elementwise math (`exp`, `sigmoid`, `silu`), reductions (`softmax`, `layer_norm`), and optimizer steps (`AdamW`) initially lagged behind PyTorch's oneDNN / ATen vectorized loops by 1.5×–3×.

Here is how we closed—and reversed—that gap in pure safe/intrinsic Rust (`crates/oxideon-core/src/backend/simd_x86.rs` & `optim.rs`).

## 1. Range-Reduced AVX2+FMA `exp` (Unlocks `exp`, `sigmoid`, `silu`, `tanh`, `softmax`)

Standard `f32::exp()` in Rust calls libm scalar-by-scalar, which kills throughput and blocks auto-vectorization. We implemented an 8-lane AVX2+FMA range-reduced `exp_ps_avx2` using Cody–Waite reduction ($x = n \ln 2 + r$), a degree-5 minimax polynomial in FMA chains, and bit-shifted IEEE-754 exponent scaling (`((n + 127) << 23)`).

From that single vectorized primitive, we derived:
- **`sigmoid(x)`** = $1 / (1 + \exp(-x))$
- **`silu(x)`** = $x \cdot \text{sigmoid}(x)$
- **`tanh(x)`** = $2 \cdot \text{sigmoid}(2x) - 1$ (blending exact $x$ for $|x| < 5\times 10^{-4}$ and handling `NaN`)
- **`softmax_row_f32`**: parallel row chunks with AVX2 max reduction, subtracted `exp_ps_avx2`, and sum normalization.

## 2. Single-Pass Fused AVX2+FMA `AdamW`

PyTorch's default Python `AdamW.step()` allocates or streams across separate parameter/moment passes unless fused/foreach is active. In `adam_step_f32_slice`, Oxideon processes `[p, m, v, g]` in a single parallel `rayon` + AVX2+FMA pass with zero intermediate tensor allocations:

- Computes decoupled weight decay, EMA moment updates (`m`, `v`), bias corrections, and parameter step in registers across 8 floats at a time.
- Result on **10M parameters**: **4.323 ms** vs PyTorch CPU **12.620 ms** (**0.34×**, ~2.9× faster).

## 3. Fused `LayerNorm` Forward & Backward + In-Place `Linear` Bias

- **Forward (`layer_norm_row_f32`)**: 2-pass AVX2 sum + variance reduction + FMA affine normalize per row across `rayon` workers (`0.755 ms` vs PyTorch `0.877 ms`).
- **Linear bias fusion**: adding bias in-place inside `Linear::forward` via `Storage::try_add_bias_inplace_f32` avoids allocating an extra broadcast output buffer on every layer.

## Measured CPU Numbers (`--release`, `Oxideon v0.1.0` vs `PyTorch 2.15.0+cpu`)

| Operation | Oxideon | PyTorch | Ratio (`Ox / Th`) |
|---|---:|---:|---:|
| `AdamW step [10M params]` | **4.323 ms** | 12.620 ms | **0.34×** |
| `sigmoid [64, 128, 768]` | **0.924 ms** | 1.667 ms | **0.55×** |
| `exp [64, 128, 768]` | **0.818 ms** | 1.073 ms | **0.76×** |
| `silu [64, 128, 768]` | **1.005 ms** | 1.293 ms | **0.78×** |
| `softmax [64, 128, 768]` | **1.398 ms** | 1.732 ms | **0.81×** |
| `matmul 512×512` | **0.646 ms** | 0.776 ms | **0.83×** |
| `layer_norm [64, 128, 768]` | **0.755 ms** | 0.877 ms | **0.86×** |
| `conv2d [16, 64, 32, 32]` | 7.762 ms | 2.844 ms | 2.73× |

Reproduce every number locally with:
```bash
python benchmarks/bench_vs_pytorch.py
```
Repo: https://github.com/voidstackloop/oxideon
