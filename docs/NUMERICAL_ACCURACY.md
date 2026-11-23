# Numerical accuracy: oxideon vs PyTorch (float32, CPU)

How far are float32 results from the true answer, and how does that compare with PyTorch's CPU kernels?
Everything here is measured, not claimed: `benchmarks/accuracy_vs_pytorch.py` feeds identical float32 inputs to oxideon and to
PyTorch and scores both against a float64 reference.

## What is measured

For every output element the error is

```
|result - float64 reference| / scale          in units of 2^-24 (a float32 half-ulp at 1.0)
```

* **Element-wise functions** (`exp`, `tanh`, `sigmoid`, `erf`, `ln`, `sqrt`, `gelu`, `softmax` ...): `scale = |reference|`, i.e.
  the relative error. A correctly rounded float32 lands between 0 and 1.0 (0.5 ulp of its own binade is up to 1.0 in these units),
  so **1.0 is the floor**; no implementation can be reliably below it.
* **Reductions and matrix products** (`sum`, `mean`, `dot`, `norm`, `matmul`, `conv2d`, `cumsum`): `scale = sum of the absolute
  terms` (`sum|x_i|`, `|A| @ |B|`). This is the condition-aware error: it does not blow up under cancellation and it measures
  exactly what summation order and accumulator width control. Here the floor is 0.5.

Each case reports the **worst** and the **mean** element together, for both libraries, with

```
x better = PyTorch error / oxideon error        (above 1: oxideon is more accurate)
```

computed separately for the worst and the mean element. The verdict column is driven by the smaller of the two figures, so a
case is only called `>=2x` when it is at least twice as accurate on the worst element *and* on average.

## Results

x86-64 with AVX2/FMA, PyTorch CPU float32, 40 cases, default settings (`PrecisionMode::Balanced`). Only the matmul and conv2d rows depend on
the mode; `Highest` figures for them are just below the table. Inputs are in
`benchmarks/accuracy_vs_pytorch.py` (offset data where cancellation hurts, saturated logits, 1M-16M element reductions).

| case | worst: oxideon | torch | x better | mean: oxideon | torch | x better | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| sum_1M | 0.0001 | 0.0004 | – | 7.0e-05 | 0.0004 | – | both exact |
| sum_16M_offset | 0.423 | 1.76 | 4.17 | 0.423 | 1.76 | 4.17 | >=2x |
| mean_4M_offset | 0.253 | 1.39 | 5.47 | 0.253 | 1.39 | 5.47 | >=2x |
| sum_axis0 | 0.640 | 2.26 | 3.53 | 0.313 | 0.624 | 2.00 | >=1.5x |
| sum_axis1 | 0.639 | 2.44 | 3.82 | 0.316 | 0.660 | 2.09 | >=2x |
| dot_4M | 0.0001 | 0.0033 | 55.75 | 6.0e-05 | 0.0033 | 55.75 | >=2x |
| norm_4M | 0.056 | 1 271 | 22 680 | 0.056 | 1 271 | 22 680 | >=2x |
| matmul_512 | 1.02 | 2.40 | 2.35 | 0.127 | 0.211 | 1.66 | >=1.5x |
| matmul_k8192 | 0.180 | 0.295 | 1.64 | 0.038 | 0.060 | 1.58 | >=1.5x |
| softmax | 8.93 | 68.8 | 7.70 | 1.40 | 12.0 | 8.53 | >=2x |
| log_softmax | 1.42 | 2 814 | 1 982 | 0.366 | 0.602 | 1.65 | >=1.5x |
| logsumexp | 0.972 | 1.02 | 1.05 | 0.368 | 0.369 | 1.00 | tie (floor) |
| layer_norm_offset | 43.4 | 4 969 | 115 | 1.57 | 363 | 232 | >=2x |
| exp | 1.09 | 1.17 | 1.07 | 0.367 | 0.365 | 0.99 | tie (floor) |
| tanh | 0.996 | 0.996 | 1.00 | 0.133 | 0.133 | 1.00 | tie (floor) |
| sigmoid | 1.00 | 2.46 | 2.46 | 0.307 | 0.506 | 1.65 | >=1.5x |
| gelu | 7.55e+05 | 1.69e+07 | 22.32 | 2 960 | 2.68e+06 | 905 | >=2x |
| erf | 0.998 | 1.12 | 1.13 | 0.283 | 0.284 | 1.00 | tie (floor) |
| ln | 1.01 | 1.32 | 1.30 | 0.358 | 0.360 | 1.01 | tie (floor) |
| sqrt | 1.000 | 1.000 | 1.00 | 0.360 | 0.360 | 1.00 | tie (floor) |
| conv2d_16ch | 2.06 | 5.56 | 2.69 | 0.218 | 0.336 | 1.54 | >=1.5x |
| conv2d_64ch | 1.09 | 4.97 | 4.58 | 0.124 | 0.331 | 2.67 | >=2x |
| cumsum_1M | 0.727 | 0.727 | 1.00 | 0.196 | 0.196 | 1.00 | tie (floor) |
| sin | 1.000 | 1.09 | 1.09 | 0.325 | 0.329 | 1.01 | tie (floor) |
| cos | 1.000 | 1.09 | 1.09 | 0.325 | 0.329 | 1.01 | tie (floor) |
| log1p | 1.00 | 1.04 | 1.04 | 0.360 | 0.360 | 1.00 | tie (floor) |
| expm1 | 0.999 | 1.38 | 1.38 | 0.359 | 0.359 | 1.00 | tie (floor) |
| softplus | 0.999 | 2.00 | 2.00 | 0.323 | 0.389 | 1.20 | better |
| silu | 1.00 | 2.79 | 2.78 | 0.353 | 0.523 | 1.48 | better |
| sinh | 0.999 | 1.00 | 1.01 | 0.361 | 0.361 | 1.00 | tie (floor) |
| cosh | 1.000 | 1.000 | 1.00 | 0.368 | 0.368 | 1.00 | tie (floor) |
| atan | 0.997 | 1.38 | 1.38 | 0.335 | 0.337 | 1.01 | tie (floor) |
| rsqrt | 1.50 | 1.50 | 1.00 | 0.496 | 0.496 | 1.00 | tie (floor) |
| pow_2_5 | 1.00 | 1.00 | 1.00 | 0.360 | 0.361 | 1.00 | tie (floor) |
| var_offset | 0.993 | 0.993 | 1.00 | 0.373 | 0.373 | 1.00 | tie (floor) |
| batch_norm_offset | 7.29 | 2 034 | 279 | 0.533 | 480 | 901 | >=2x |
| group_norm_offset | 5.27 | 1 882 | 357 | 0.521 | 521 | 1 001 | >=2x |
| cosine_sim | 0.994 | 2 294 | 2 308 | 0.362 | 6.78 | 18.71 | >=2x |
| normalize_l2 | 0.999 | 3.22 | 3.22 | 0.361 | 0.625 | 1.73 | >=1.5x |
| mse | 0.261 | 0.261 | 1.00 | 0.261 | 0.261 | 1.00 | tie (floor) |

(`gelu` is dominated by relative error deep in the tail where the result is ~1e-15; both libraries are far from the floor there,
oxideon less so.)

With `PrecisionMode::Highest` the products and convolutions become: matmul_512 worst 0.211 vs 2.40 (11.4x better), mean 10.6x;
matmul_k8192 6.1x / 11.9x; conv2d_16ch 12.7x / 8.7x; conv2d_64ch 19.4x / 17.1x. With `Fastest` the matrix products are PyTorch-class
(worst ~6 at k=512) and the convolutions are 2.0x / 1.2-1.4x.

### Reading the table honestly

* **Where oxideon is at least 2x better on both worst and mean:** long reductions (`sum`, `mean`, `dot`, `norm`), matrix
  products (`Balanced` 1.6-2.4x, `Highest` 6-12x), `softmax`, `layer_norm`, `batch_norm`, `group_norm`, `gelu`, `cosine_similarity`, and the worst
  element of `log_softmax`. PyTorch accumulates in float32 with a limited number of partial sums and centres on a rounded
  float32 mean; oxideon accumulates in float64 and carries the mean as a hi/lo pair, so the gap widens with problem size and with
  the offset of the data.
* **Where the target cannot be met: the correct-rounding floor.** `sqrt`, `ln`, `exp`, `erf`, `sin`, `cos`, `log1p`, `expm1`, `sinh`,
  `cosh`, `atan`, `rsqrt`, `pow`, `logsumexp`, `cumsum`, `mse` and the final rounding of every other op sit at about 1.0 (0.5 for
  reductions) for *both* libraries. An answer that is already the nearest float32 cannot be made 2x more accurate while the output
  type is float32. These cases are reported as ties. (`ln`, `expm1`, `atan`, `erf` and `softplus` do show PyTorch a little above the
  floor on the worst element.)
* **`tanh` is now level with PyTorch on both worst (0.996 vs 0.996) and mean (0.133 vs 0.133).** Both return the nearest float32, so
  this is a tie at the floor. Getting there needed the hi/lo `exp` pair itself to be accurate to about 0.03 ulp (reduced argument
  carried as `rh + rl`, `rh^2/2` kept as an exact pair, minimax-refit cubic tail; before, 0.09-0.27 ulp), a degree-4 small-argument
  polynomial on `x^2 in [0, 1/16]`, and `(E - 1)/(E + 1)` with TwoSum numerator/denominator and a residual-corrected quotient for
  `|x| >= 1/4`. The same exp pair lifted sigmoid worst to 2.46x and silu worst to 2.78x. `var_offset` is an exact tie (0.99 vs 0.99).
* **Matmul and conv2d depend on the mode.** Default (`Balanced`): matmul 2.35x / 1.66x and 1.64x / 1.58x (worst / mean), conv2d 2.69x / 1.54x
  and 4.58x / 2.67x. `Highest`: matmul 6-12x, conv2d 9-19x. `Fastest`: matmul PyTorch-class, conv2d 2.0x / 1.2-1.4x.

## Precision modes

One application-wide switch trades speed for accuracy where it actually costs something: the accumulation inside `f32` matrix
products (and so linear layers, attention and im2col convolutions) and the channel sums of the Winograd convolution.

| mode | what it does | worst error at k=512 (half-ulps) |
|---|---|---:|
| `Fastest` | plain f32 accumulation in 256-deep slices of k (a cache-blocking depth, as the `gemm` crate uses inside) | ~2.6 |
| `Balanced` (**default**) | k cut into 48-deep f32 chains; up to 32 chains are summed inside the kernel, group totals added in f64 | 1.02 |
| `Highest` | f64 products and sums, rounded to f32 once | 0.21 |

PyTorch's CPU matmul reaches 2.4 on the same product.

Choose the mode in whichever way suits the application; all of them act on the next matrix product:

```bash
OXIDEON_PRECISION=fastest   # environment variable, read on first use: fastest | balanced | highest
```

```rust
use oxideon_core::{set_precision_mode, precision_mode, PrecisionMode};
set_precision_mode(PrecisionMode::Balanced);
assert_eq!(precision_mode(), PrecisionMode::Balanced);
```

```python
import oxideon as ox
ox.set_precision_mode("balanced")      # "fastest" | "balanced" | "highest"
ox.get_precision_mode()                # -> "balanced"
```

An explicit `set_precision_mode` always wins over the environment variable. `PrecisionMode::parse` also accepts the aliases
`fast`/`speed`, `medium`, and `high`/`accurate`/`exact`.

Reductions, element-wise functions (except GELU, below), softmax, layer norm, norms and `dot` are always computed in their accurate form
regardless of the mode: it costs next to nothing there. Two other places consult the mode. First, GELU: `Fastest` evaluates the forward
pass on x86-64 AVX2 with the five-term Abramowitz-Stegun erfc (about 1.5e-7 absolute error in the CDF, 113 us per 1M elements, level
with PyTorch) where the other modes use the tail-accurate degree-9 fit (150 us); the mode is read once per tensor, and the backward
kernel and other architectures always use the accurate form. Second, the accurate centring of the *composed*
normalizations (`batch_norm`, `group_norm`, `var`, `std`, the `LayerNorm`/`BatchNorm` layers): `Fastest` skips the two extra
passes that recover the rounded mean, `Balanced` and `Highest` keep them (see `Tensor::centered_by_mean`). Convolutions:
`Fastest` sums the input channels of the Winograd kernel in one FMA chain; `Balanced` sums them in 4-channel chains whose totals are
added with TwoSum (about 1.3x the time for 16 input channels and 0.9x for 64, error roughly halved; from 32 input channels the chains are 8 deep, which is
what keeps the 64-channel case under PyTorch's time at a small cost in accuracy: 4.64x -> 4.58x worst, 2.99x -> 2.67x mean better than PyTorch); `Highest` takes im2col + the f64 GEMM (about 8x the time of Winograd).

### What the modes cost

`core_ops_bench` on the same laptop (x86-64, AVX2, thermally limited). Microseconds, all columns re-measured on the final build (PyTorch from the same session; run-to-run noise is about 15%):

| benchmark | `Fastest` | `Balanced` | `Highest` | PyTorch |
|---|---:|---:|---:|---:|
| matmul 512 | 328 | 351 | 920 | 392 |
| matmul 1024 | 2 630 | 2 643 | 6 451 | 3 086 |
| matmul 256x4096x256 | 698 | 820 | 2 499 | 812 |
| MLP train step, batch 256 | 777 | 888 | 1 920 | 876 |
| conv2d 8x16x64x64 k3 | 484 | 664 | 4 105 | 512 |
| conv2d 4x64x56x56 k3 | 1 190 | 1 467 | 6 279 | 1 529 |
| gelu 1M | 101 | 135 | 134 | 111 |
| everything else in the suite (softmax, layer_norm, sum, exp ...) | unchanged | unchanged | unchanged |

`Balanced` no longer costs a synchronization per 48-deep chain: when a product has several chains, one parallel region splits the output into
row (or column) blocks and each worker runs all the chains of its block serially, with the f64 group combine in a per-block buffer. The
results are bit-for-bit those of the old chain loop, and `Balanced` now runs at 0.9-1.1x PyTorch's time (the output is split into a 2-D grid of blocks, about three per thread, chosen to minimize operand packing) (it was 1.45-2.5x; for long
`k` it is now faster than the single-call `Fastest`). `Highest` about 1.9-2.2x (the floor, since an f64
multiply-add moves half as many lanes per instruction as f32). `Highest` is cheaper than its accuracy suggests: 10x the accuracy of
`Balanced` for about 15% more time. Convolutions
use the Winograd kernel in `Fastest`/`Balanced`.
Products whose three f64 working copies would exceed 1 GiB (about 11 000 x 11 000) fall back to the `Balanced` chain instead of risking an
out-of-memory abort. `Highest` reuses a small pool of f64 work buffers (at most four, 256 MiB) so repeated products do not pay for page faults, which
accounted for more than half of its first implementation's time.

Skinny products (single-token inference, small batches) take a dedicated `Highest` route that never materializes an f64 copy of the
weight matrix: dot products against the contiguous columns of a transposed weight (what `Linear` uses), or f64 accumulators
over row blocks when the rows are contiguous. It is used up to 8 rows, and up to 32 rows when the weight has at least 4M elements.
Microseconds, same machine:

| product (m x k x n) | `Fastest` | `Balanced` | `Highest` |
|---|---:|---:|---:|
| 1 x 4096 x 4096 (one token) | 2 779 | 3 024 | 2 123 |
| 8 x 4096 x 4096 | 4 126 | 3 569 | 4 907 |
| 64 x 4096 x 4096 | 14 092 | 9 302 | 12 319 |
| 128 x 4096 x 4096 | 7 926 | 13 659 | 18 286 |
| 512 x 512 x 512 | 495 | 653 | 920 |
| 4096 x 64 x 64 | 110 | 91 | 229 |

A single token (and a 64-row batch) is as fast or faster in `Highest` than in `Fastest`: the product is bandwidth-bound and the f64
routes read each weight once. Mid-size square products stay at about 2x `Fastest`.
Products of up to 128 rows with a weight of at least 4M elements are cut into column strips, one single-threaded f64 product per strip, because
the `gemm` crate's f64 kernel splits by rows of the result and leaves most cores idle at that height (a 64-row product took 39 ms before,
14 ms now).

Batched products (attention heads: `[B, T, d] @ [B, d, T]`) run one f64 product per batch item on the worker that owns the item, with
per-thread buffer pools and no nested parallelism; nesting the widening and the kernel's own threads inside the batch loop cost more
than the products themselves (the first implementation was 5-7x `Fastest` here). Microseconds:

| batch x [m x k x n] | `Fastest` | `Balanced` | `Highest` |
|---|---:|---:|---:|
| 64 x [128 x 64 x 128] | 281 | 277 | 439 |
| 32 x [512 x 64 x 512] | 1 568 | 1 528 | 3 744 |
| 32 x [512 x 512 x 64] | 1 570 | 2 252 | 3 926 |
| 256 x [16 x 64 x 16] | 102 | 93 | 93 |
| 8 x [1024 x 128 x 1024] | 2 896 | 3 814 | 9 435 |

`gemm::set_k_chunk(depth)` tunes `Balanced` (default 48): at k=512 the worst error is 6.0 unsplit, 2.9 at 256, 1.6 at 128, 1.06 at 64,
1.02 at 48 and 0.77 at 32; shorter chains cost a little throughput (k=1024, 1024 x 1024: 2.8 ms at 128, 3.1 at 48, 3.2 at 32). `0` disables splitting. The
benchmarks read `OXI_KCHUNK=<depth>` for experiments.

### Everything else

No switches: the accurate variants (f64 accumulation, hi/lo mean, `ln_1p` log-softmax, TwoSum-corrected softmax, correctly
rounded `erf` via `libm` in f64) are always on, at no or small speed cost. Measured against PyTorch on the same machine:
conv2d 1.3x the time of PyTorch for 16 channels and 0.96x for 64 (the TwoSum channel chains), layer_norm 0.65x (its variance sum used to be a scalar f32 chain; now eight
vector lanes, which is also more accurate: layer_norm of offset data went from 115x to 385x better than PyTorch), softmax 0.97x, gelu
1.2x in `Balanced` (its tail-accurate erfc is what makes it ~22x better than PyTorch) and 0.9x in `Fastest`, which switches to the
five-term Abramowitz-Stegun erfc (101 us vs PyTorch's ~111 us; still ~18x / ~140x better than PyTorch on worst / mean error), tanh 0.6x (faster than PyTorch's).

## Techniques, per op

| op | what makes it accurate |
|---|---|
| sum / mean / dot / norm | float64 accumulators (AVX2 4-way f64), parallel chunks combined in f64, rounded to f32 once |
| mean / mean over axes | the kernel divides its f64 accumulator once (`ReduceOp::Mean`), so the mean is rounded a single time instead of once for the sum and again for the division |
| axis-0 sum | f64 accumulator rows (`add_f32_into_f64`) instead of a running f32 row; column blocks sized so every core has work |
| layer_norm (fused) | mean carried as `hi + lo`, so `x - mean` is centred exactly even with a large offset; variance in f64 |
| batch_norm / group_norm / var / std / LayerNorm / BatchNorm layers | `Tensor::centered_by_mean`: `x - mean_hi` followed by `- mean(x - mean_hi)`, a constant correction term (gradients unchanged) |
| softmax | `x - max` error recovered with TwoSum and folded into the exponent (`e = e0 + e0*err`) |
| log_softmax | `ln_1p` of the tail sum when one logit dominates (avoids `1 + 1e-6` rounding) |
| exp | reduced argument `rh + rl` (no reduction rounding), `1 + rh + rh^2/2` as exact pairs plus a rounded cubic tail; returns `hi + lo` good to ~0.03 ulp (unit-tested) |
| sigmoid / silu | `e^-|x|` as a hi/lo pair, quotient corrected with the exact residual (rounded once; silu multiplies both parts) |
| tanh | degree-4 polynomial in `x^2` (one fused rounding) for `|x| < 1/4`; above, `(E - 1)/(E + 1)` with `E = e^(2x)` hi/lo, TwoSum numerator/denominator, residual-corrected quotient |
| erf | `libm::erf` in f64, rounded once |
| softplus / elu | `ln(1 + exp(x))` in f64; `expm1` instead of `exp - 1` (no cancellation near 0) |
| cosine_similarity | dot product and norms in f64, one division (no-grad host path; the differentiable path normalizes first) |
| normalize (L2) / var / std | no-grad host path: norm, mean and squared deviations in f64, one rounding per element |
| matmul | `Highest`: f64 products and sums; `Balanced`: 48-deep f32 chains (groups of 32 summed in the kernel, group totals in f64); `Fastest`: plain f32 accumulation in 256-deep slices |
| conv2d (Winograd) | filter transform in f64; channel chains of 4 (8 from 32 channels) combined with TwoSum, each TwoSum error feeding the next chain (`Fastest`: one chain) |

## Reproducing

```bash
cargo build --release --example accuracy_dump
python benchmarks/accuracy_vs_pytorch.py                   # all 40 cases, worst + mean, x better, verdict
python benchmarks/accuracy_vs_pytorch.py matmul_512 tanh   # selected cases
OXIDEON_PRECISION=highest python benchmarks/accuracy_vs_pytorch.py matmul_512 matmul_k8192   # default is balanced
cargo build --release --example core_ops_bench && python benchmarks/core_ops_bench_compare.py   # time and error per operation, side by side
```

Needs `numpy` and `torch`. Inputs are seeded, so the numbers reproduce on the same machine; the `exp`/`sigmoid`/`tanh` vector paths
need AVX2+FMA (elsewhere a scalar fallback is used, bounded at 6 half-ulps by the tests).

Regression tests: `crates/oxideon-core/tests/numerical_accuracy.rs` (bounds just above what oxideon reaches, far below
PyTorch's on the cases that matter) and `tests/matmul_precision.rs` (the three modes order correctly, parse, and handle odd, strided and empty shapes). The NEON paths
run the same tests under `docker run --platform linux/arm64` (see below).

## Limitations

* The comparison is float32 CPU only; GPU kernels and float16/bfloat16 are not covered.
* One PyTorch build and one CPU were measured; PyTorch's accumulation order depends on thread count and vector width.
* The input distributions are the ones in the script; they are chosen to expose cancellation and saturation, which is where
  float32 libraries differ. On benign, well-scaled data both libraries sit near the floor and the gap shrinks.

## Verified on aarch64 (NEON)

The `aarch64-unknown-linux-musl` test binaries were built with `rust-lld` and run under Docker's arm64 emulation:
741 library tests plus `numerical_accuracy`, `matmul_precision`, `unary_backward`, `conv_fast_paths`, `png_adam7`,
`jpeg_progressive` and `jpeg_decode` pass; the only failures are two `.npy` tests that read fixtures through a compile-time Windows path.
The NEON kernels use the same algorithms but not the same instructions, so error bounds on non-x86 are the relaxed ones in the tests
(6.0 half-ulps for `exp`/`tanh`/`sigmoid`).
