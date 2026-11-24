//! Pure-Rust multi-dtype CPU backend with strided kernels, blocked GEMM, and `rayon`.

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::backend::{BinaryOp, BitOp, CmpOp, LogicOp, PredOp, ReduceOp, UnaryOp};
use crate::dtype::{DType, Element, bf16, f16, f8e4m3, f8e5m2};
use crate::error::{Error, Result};
use crate::gemm::{batched_gemm_f32, batched_gemm_f64};
use crate::layout::{DimMap, Layout, ViewSpec, broadcast_shapes};

/// Elements per parallel task of the elementwise `f32` kernels: big enough that the loop inside a task vectorizes and the
/// scheduling cost disappears, small enough that a 1M-element tensor still spreads over every core.
const EW_CHUNK: usize = 8 * 1024;

fn ew_out(n: usize) -> Vec<f32> {
    crate::pool::take_f32(n).unwrap_or_else(|| vec![0.0; n])
}

/// `out[i] = f(a[i])` with `f` monomorphized per call site, so each operation gets its own vectorized loop.
#[inline]
fn f32_map(a: &[f32], f: impl Fn(f32) -> f32 + Sync) -> Vec<f32> {
    let mut out = ew_out(a.len());
    let kernel = |(o, a): (&mut [f32], &[f32])| o.iter_mut().zip(a).for_each(|(o, &x)| *o = f(x));
    #[cfg(feature = "rayon")]
    if a.len() > EW_CHUNK {
        out.par_chunks_mut(EW_CHUNK).zip(a.par_chunks(EW_CHUNK)).for_each(kernel);
        return out;
    }
    kernel((&mut out, a));
    out
}

/// `out[i] = f(a[i], b[i])`, see [`f32_map`].
#[inline]
fn f32_zip(a: &[f32], b: &[f32], f: impl Fn(f32, f32) -> f32 + Sync) -> Vec<f32> {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut out = ew_out(n);
    let kernel = |((o, a), b): ((&mut [f32], &[f32]), &[f32])| o.iter_mut().zip(a).zip(b).for_each(|((o, &x), &y)| *o = f(x, y));
    #[cfg(feature = "rayon")]
    if n > EW_CHUNK {
        out.par_chunks_mut(EW_CHUNK).zip(a.par_chunks(EW_CHUNK)).zip(b.par_chunks(EW_CHUNK)).for_each(kernel);
        return out;
    }
    kernel(((&mut out, a), b));
    out
}

/// Contiguous `f32` unary operation. The `match` is outside the loops: every arm instantiates [`f32_map`] with a constant
/// operation, which the per-element evaluator folds to a single expression.
fn f32_unary_contiguous(a: &[f32], op: UnaryOp) -> Vec<f32> {
    macro_rules! arm {
        ($e:expr) => {
            f32_map(a, |x| eval_unary_f32(x, $e))
        };
    }
    match op {
        UnaryOp::Neg => arm!(UnaryOp::Neg),
        UnaryOp::Abs => arm!(UnaryOp::Abs),
        UnaryOp::Sign => arm!(UnaryOp::Sign),
        UnaryOp::Exp => {
            if super::simd_x86::has_avx2_fma() {
                let mut out = crate::pool::take::<f32>(a.len()).unwrap_or_else(|| vec![0.0; a.len()]);
                #[cfg(feature = "rayon")]
                if a.len() >= 16384 {
                    out.par_chunks_mut(8192).zip(a.par_chunks(8192)).for_each(|(o, s)| {
                        super::simd_x86::exp_f32_slice(s, o);
                    });
                    return out;
                }
                super::simd_x86::exp_f32_slice(a, &mut out);
                return out;
            }
            arm!(UnaryOp::Exp)
        }
        UnaryOp::Ln => arm!(UnaryOp::Ln),
        UnaryOp::Sqrt => arm!(UnaryOp::Sqrt),
        UnaryOp::Rsqrt => arm!(UnaryOp::Rsqrt),
        UnaryOp::Sin => arm!(UnaryOp::Sin),
        UnaryOp::Cos => arm!(UnaryOp::Cos),
        UnaryOp::Tanh => {
            if super::simd_x86::has_avx2_fma() {
                let mut out = crate::pool::take::<f32>(a.len()).unwrap_or_else(|| vec![0.0; a.len()]);
                #[cfg(feature = "rayon")]
                if a.len() >= 16384 {
                    out.par_chunks_mut(8192).zip(a.par_chunks(8192)).for_each(|(o, s)| {
                        super::simd_x86::tanh_f32_slice(s, o);
                    });
                    return out;
                }
                super::simd_x86::tanh_f32_slice(a, &mut out);
                return out;
            }
            arm!(UnaryOp::Tanh)
        }
        UnaryOp::Sigmoid => {
            if super::simd_x86::has_avx2_fma() {
                let mut out = crate::pool::take::<f32>(a.len()).unwrap_or_else(|| vec![0.0; a.len()]);
                #[cfg(feature = "rayon")]
                if a.len() >= 16384 {
                    out.par_chunks_mut(8192).zip(a.par_chunks(8192)).for_each(|(o, s)| {
                        super::simd_x86::sigmoid_f32_slice(s, o);
                    });
                    return out;
                }
                super::simd_x86::sigmoid_f32_slice(a, &mut out);
                return out;
            }
            arm!(UnaryOp::Sigmoid)
        }
        UnaryOp::Relu => {
            if super::simd_x86::has_avx2_fma() {
                let mut out = crate::pool::take::<f32>(a.len()).unwrap_or_else(|| vec![0.0; a.len()]);
                #[cfg(feature = "rayon")]
                if a.len() >= 16384 {
                    out.par_chunks_mut(8192).zip(a.par_chunks(8192)).for_each(|(o, s)| {
                        super::simd_x86::relu_f32_slice(s, o);
                    });
                    return out;
                }
                super::simd_x86::relu_f32_slice(a, &mut out);
                return out;
            }
            arm!(UnaryOp::Relu)
        }
        UnaryOp::LeakyRelu(s) => arm!(UnaryOp::LeakyRelu(s)),
        UnaryOp::Gelu => {
            if super::simd_x86::has_avx2_fma() {
                let mut out = crate::pool::take::<f32>(a.len()).unwrap_or_else(|| vec![0.0; a.len()]);
                // the mode is read once for the whole tensor, so a concurrent `set_precision_mode` cannot mix two error levels in one result
                let fast = crate::gemm::precision_mode() == crate::gemm::PrecisionMode::Fastest;
                #[cfg(feature = "rayon")]
                if a.len() >= 16384 {
                    out.par_chunks_mut(8192).zip(a.par_chunks(8192)).for_each(|(o, s)| {
                        super::simd_x86::gelu_f32_slice(s, o, fast);
                    });
                    return out;
                }
                super::simd_x86::gelu_f32_slice(a, &mut out, fast);
                return out;
            }
            arm!(UnaryOp::Gelu)
        }
        UnaryOp::Silu => {
            if super::simd_x86::has_avx2_fma() {
                let mut out = crate::pool::take::<f32>(a.len()).unwrap_or_else(|| vec![0.0; a.len()]);
                #[cfg(feature = "rayon")]
                if a.len() >= 16384 {
                    out.par_chunks_mut(8192).zip(a.par_chunks(8192)).for_each(|(o, s)| {
                        super::simd_x86::silu_f32_slice(s, o);
                    });
                    return out;
                }
                super::simd_x86::silu_f32_slice(a, &mut out);
                return out;
            }
            arm!(UnaryOp::Silu)
        }
        UnaryOp::Elu(s) => arm!(UnaryOp::Elu(s)),
        UnaryOp::Softplus => arm!(UnaryOp::Softplus),
        UnaryOp::Square => arm!(UnaryOp::Square),
        UnaryOp::Recip => arm!(UnaryOp::Recip),
        UnaryOp::Floor => arm!(UnaryOp::Floor),
        UnaryOp::Ceil => arm!(UnaryOp::Ceil),
        UnaryOp::Round => arm!(UnaryOp::Round),
        UnaryOp::Erf => arm!(UnaryOp::Erf),
        UnaryOp::Powf(p) => arm!(UnaryOp::Powf(p)),
        UnaryOp::AddScalar(v) => arm!(UnaryOp::AddScalar(v)),
        UnaryOp::MulScalar(v) => arm!(UnaryOp::MulScalar(v)),
        UnaryOp::Clamp(lo, hi) => arm!(UnaryOp::Clamp(lo, hi)),
        UnaryOp::LogicalNot => arm!(UnaryOp::LogicalNot),
    }
}

/// Columns per task of a reduction over leading axes: 256 at most (a block's accumulators stay in L1), narrower when the matrix is
/// so slim that 256-wide blocks would leave most cores idle (1024 columns would otherwise make only four tasks).
fn leading_block(outer: usize) -> usize {
    #[cfg(feature = "rayon")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "rayon"))]
    let threads = 1usize;
    ((outer / (2 * threads).max(1)) / 16 * 16).clamp(64, 256).min(outer).max(1)
}

/// [`add_f32_into_f64`] over a whole matrix, in parallel when it is large.
pub(crate) fn add_f32_into_f64_par(acc: &mut [f64], row: &[f32]) {
    #[cfg(feature = "rayon")]
    if acc.len() >= 1 << 16 {
        acc.par_chunks_mut(1 << 14).zip(row.par_chunks(1 << 14)).for_each(|(a, r)| add_f32_into_f64(a, r));
        return;
    }
    add_f32_into_f64(acc, row);
}

/// `acc[j] += row[j]` with the `f32` row converted to `f64`: a sum over many rows accumulates without the rounding of a long `f32` run.
#[inline]
fn add_f32_into_f64(acc: &mut [f64], row: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if super::simd_x86::has_avx2_fma() {
        // SAFETY: AVX2 was detected.
        unsafe { add_f32_into_f64_avx2(acc, row) };
        return;
    }
    for (a, &v) in acc.iter_mut().zip(row) {
        *a += v as f64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_f32_into_f64_avx2(acc: &mut [f64], row: &[f32]) {
    use std::arch::x86_64::*;
    let n = acc.len().min(row.len());
    let (ap, rp) = (acc.as_mut_ptr(), row.as_ptr());
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: i + 4 <= n for both slices.
        unsafe { _mm256_storeu_pd(ap.add(i), _mm256_add_pd(_mm256_loadu_pd(ap.add(i)), _mm256_cvtps_pd(_mm_loadu_ps(rp.add(i))))) };
        i += 4;
    }
    while i < n {
        unsafe { *ap.add(i) += *rp.add(i) as f64 };
        i += 1;
    }
}

/// Mean of an `f32` slice accumulated in `f64`: no rounding of a long `f32` running sum, so a row such as `1000 + noise` keeps its
/// mean to the last bit instead of to a few thousand ulps.
pub(crate) fn mean_f64(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    sum_f64_of_f32(x) / x.len() as f64
}

/// `sum(((x - hi) - lo)^2)` over a row, in `f32` lanes (eight partial sums, vectorized where AVX2 is available).
pub(crate) fn centered_sumsq(x: &[f32], hi: f32, lo: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if super::simd_x86::has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected.
        return unsafe { centered_sumsq_avx2(x, hi, lo) };
    }
    let mut acc = [0.0f32; 8];
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        for k in 0..8 {
            let d = (c[k] - hi) - lo;
            acc[k] += d * d;
        }
    }
    let tail: f32 = chunks.remainder().iter().map(|&v| ((v - hi) - lo) * ((v - hi) - lo)).sum();
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn centered_sumsq_avx2(x: &[f32], hi: f32, lo: f32) -> f32 {
    use std::arch::x86_64::*;
    let (n, p) = (x.len(), x.as_ptr());
    let (vhi, vlo) = (_mm256_set1_ps(hi), _mm256_set1_ps(lo));
    let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0usize;
    while i + 16 <= n {
        // SAFETY: i + 16 <= n.
        let d0 = _mm256_sub_ps(_mm256_sub_ps(unsafe { _mm256_loadu_ps(p.add(i)) }, vhi), vlo);
        let d1 = _mm256_sub_ps(_mm256_sub_ps(unsafe { _mm256_loadu_ps(p.add(i + 8)) }, vhi), vlo);
        a0 = _mm256_fmadd_ps(d0, d0, a0);
        a1 = _mm256_fmadd_ps(d1, d1, a1);
        i += 16;
    }
    if i + 8 <= n {
        // SAFETY: i + 8 <= n.
        let d0 = _mm256_sub_ps(_mm256_sub_ps(unsafe { _mm256_loadu_ps(p.add(i)) }, vhi), vlo);
        a0 = _mm256_fmadd_ps(d0, d0, a0);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    // SAFETY: `lanes` holds eight f32.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(a0, a1)) };
    let mut total = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    while i < n {
        let d = (unsafe { *p.add(i) } - hi) - lo;
        total += d * d;
        i += 1;
    }
    total
}

/// Sum of an `f32` slice accumulated in `f64` (vectorized with AVX2 where available).
pub(crate) fn sum_f64_of_f32(x: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if super::simd_x86::has_avx2_fma() {
        // SAFETY: AVX2 was detected.
        return unsafe { sum_f64_avx2(x) };
    }
    let mut acc = [0.0f64; 4];
    let mut chunks = x.chunks_exact(4);
    for c in &mut chunks {
        for k in 0..4 {
            acc[k] += c[k] as f64;
        }
    }
    let tail: f64 = chunks.remainder().iter().map(|&v| v as f64).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f64_avx2(x: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let (n, p) = (x.len(), x.as_ptr());
    let mut i = 0usize;
    let mut acc = [_mm256_setzero_pd(); 4];
    while i + 16 <= n {
        for (k, a) in acc.iter_mut().enumerate() {
            // SAFETY: i + 16 <= n.
            *a = _mm256_add_pd(*a, _mm256_cvtps_pd(unsafe { _mm_loadu_ps(p.add(i + 4 * k)) }));
        }
        i += 16;
    }
    let sum4 = _mm256_add_pd(_mm256_add_pd(acc[0], acc[1]), _mm256_add_pd(acc[2], acc[3]));
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` holds four f64.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum4) };
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        total += unsafe { *p.add(i) } as f64;
        i += 1;
    }
    total
}

/// `sum(a[i] * b[i])` with the products and the sum carried out in `f64` (one rounding of the final result instead of one per
/// product and one per addition).
pub(crate) fn dot_f64_of_f32(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    #[cfg(feature = "rayon")]
    if n >= 65536 {
        return a.par_chunks(16384).zip(b.par_chunks(16384)).map(|(x, y)| dot_f64_chunk(x, y)).sum::<f64>();
    }
    dot_f64_chunk(a, b)
}

fn dot_f64_chunk(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if super::simd_x86::has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected; both slices have the same length.
        return unsafe { dot_f64_avx2(a, b) };
    }
    let mut acc = [0.0f64; 4];
    let mut ca = a.chunks_exact(4);
    let mut cb = b.chunks_exact(4);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for k in 0..4 {
            acc[k] += x[k] as f64 * y[k] as f64;
        }
    }
    let tail: f64 = ca.remainder().iter().zip(cb.remainder()).map(|(&x, &y)| x as f64 * y as f64).sum();
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f64_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;
    let (n, ap, bp) = (a.len().min(b.len()), a.as_ptr(), b.as_ptr());
    let mut i = 0usize;
    let mut acc = [_mm256_setzero_pd(); 4];
    while i + 16 <= n {
        for (k, s) in acc.iter_mut().enumerate() {
            // SAFETY: i + 16 <= n for both slices.
            let (x, y) = unsafe { (_mm256_cvtps_pd(_mm_loadu_ps(ap.add(i + 4 * k))), _mm256_cvtps_pd(_mm_loadu_ps(bp.add(i + 4 * k)))) };
            *s = _mm256_fmadd_pd(x, y, *s);
        }
        i += 16;
    }
    let sum4 = _mm256_add_pd(_mm256_add_pd(acc[0], acc[1]), _mm256_add_pd(acc[2], acc[3]));
    let mut lanes = [0.0f64; 4];
    // SAFETY: `lanes` holds four f64.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), sum4) };
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        total += unsafe { *ap.add(i) as f64 * *bp.add(i) as f64 };
        i += 1;
    }
    total
}

/// `acc[j] += a * x[j]` with the `f32` row widened to `f64` (the building block of a skinny matrix product that never materializes
/// an `f64` copy of the right-hand matrix).
pub(crate) fn axpy_f64_of_f32(acc: &mut [f64], x: &[f32], a: f64) {
    #[cfg(target_arch = "x86_64")]
    if super::simd_x86::has_avx2_fma() {
        // SAFETY: AVX2 and FMA were detected.
        unsafe { axpy_f64_avx2(acc, x, a) };
        return;
    }
    for (d, &v) in acc.iter_mut().zip(x) {
        *d += a * v as f64;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f64_avx2(acc: &mut [f64], x: &[f32], a: f64) {
    use std::arch::x86_64::*;
    let n = acc.len().min(x.len());
    let (dp, xp) = (acc.as_mut_ptr(), x.as_ptr());
    let av = _mm256_set1_pd(a);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: i + 4 <= n for both slices.
        unsafe { _mm256_storeu_pd(dp.add(i), _mm256_fmadd_pd(av, _mm256_cvtps_pd(_mm_loadu_ps(xp.add(i))), _mm256_loadu_pd(dp.add(i)))) };
        i += 4;
    }
    while i < n {
        unsafe { *dp.add(i) += a * *xp.add(i) as f64 };
        i += 1;
    }
}

/// The mean of a row as an unevaluated sum `hi + lo` of two `f32`s. `(x - hi) - lo` then centers a value with the accuracy of
/// the `f64` mean: `x - hi` is exact for values near the mean and `lo` carries what `f32` cannot hold of the mean.
pub(crate) fn center_split(x: &[f32]) -> (f32, f32) {
    let m = mean_f64(x);
    let hi = m as f32;
    (hi, (m - hi as f64) as f32)
}

/// `a op row` / `row op a` for a contiguous full-shape operand and a `[.., cols]` row broadcast over the leading dimensions
/// (a bias add, a per-feature scale): one vectorized slice kernel per row, rows in parallel. `None` when the layouts are
/// anything else, for the operation, or when no vector kernel exists on this machine.
fn f32_row_broadcast(a: &[f32], la: &Layout, b: &[f32], lb: &Layout, op: BinaryOp) -> Option<Vec<f32>> {
    let nd = la.ndim();
    if nd < 2 || lb.ndim() != nd || la.shape() != lb.shape() {
        return None;
    }
    // the vector kernels of this architecture (the x86 stubs do nothing on ARM and vice versa)
    #[cfg(target_arch = "aarch64")]
    use super::simd_arm as lanes;
    #[cfg(not(target_arch = "aarch64"))]
    use super::simd_x86 as lanes;
    let simd: fn(&[f32], &[f32], &mut [f32]) -> bool = match op {
        BinaryOp::Add => lanes::add_f32_slice,
        BinaryOp::Sub => lanes::sub_f32_slice,
        BinaryOp::Mul => lanes::mul_f32_slice,
        BinaryOp::Div => lanes::div_f32_slice,
        _ => return None,
    };
    #[cfg(target_arch = "aarch64")]
    let available = lanes::has_neon();
    #[cfg(not(target_arch = "aarch64"))]
    let available = lanes::has_avx2_fma();
    if !available {
        return None;
    }
    let cols = *la.shape().last()?;
    let total: usize = la.shape().iter().product();
    if cols < 8 || total < 4096 {
        return None;
    }
    let is_row = |l: &Layout| l.strides()[..nd - 1].iter().all(|&s| s == 0) && l.strides()[nd - 1] == 1;
    // (full operand, its range, row operand slice, the full operand is the left one)
    let (full, row, full_is_lhs): (&[f32], &[f32], bool) = if let Some(r) = la.contiguous_range_in(a.len())
        && is_row(lb)
        && lb.offset() + cols <= b.len()
    {
        (&a[r], &b[lb.offset()..lb.offset() + cols], true)
    } else if let Some(r) = lb.contiguous_range_in(b.len())
        && is_row(la)
        && la.offset() + cols <= a.len()
    {
        (&b[r], &a[la.offset()..la.offset() + cols], false)
    } else {
        return None;
    };
    let mut out = crate::pool::take::<f32>(total).unwrap_or_else(|| vec![0.0; total]);
    // a kernel that declines a row leaves it unwritten, so any refusal abandons the whole fast path
    let declined = std::sync::atomic::AtomicBool::new(false);
    let kernel = |(o, f): (&mut [f32], &[f32])| {
        let handled = if full_is_lhs { simd(f, row, o) } else { simd(row, f, o) };
        if !handled {
            declined.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    };
    #[cfg(feature = "rayon")]
    if total >= 16384 {
        out.par_chunks_mut(cols).zip(full.par_chunks(cols)).with_min_len((16384 / cols).max(1)).for_each(kernel);
        return (!declined.load(std::sync::atomic::Ordering::Relaxed)).then_some(out);
    }
    out.chunks_mut(cols).zip(full.chunks(cols)).for_each(kernel);
    (!declined.load(std::sync::atomic::Ordering::Relaxed)).then_some(out)
}

/// Contiguous `f32` binary operation, one vectorized loop per operation (see [`f32_unary_contiguous`]).
fn f32_binary_contiguous(a: &[f32], b: &[f32], op: BinaryOp) -> Vec<f32> {
    macro_rules! arm {
        ($e:expr) => {
            f32_zip(a, b, |x, y| eval_bin_f32(x, y, $e))
        };
    }
    macro_rules! simd_bin {
        ($simd_fn:path) => {{
            let mut out = crate::pool::take::<f32>(a.len()).unwrap_or_else(|| vec![0.0; a.len()]);
            #[cfg(feature = "rayon")]
            if a.len() >= 16384 {
                out.par_chunks_mut(8192)
                    .zip(a.par_chunks(8192))
                    .zip(b.par_chunks(8192))
                    .for_each(|((o, sa), sb)| {
                        $simd_fn(sa, sb, o);
                    });
                return out;
            }
            $simd_fn(a, b, &mut out);
            return out;
        }};
    }
    match op {
        BinaryOp::Add => {
            if super::simd_x86::has_avx2_fma() {
                simd_bin!(super::simd_x86::add_f32_slice);
            }
            if cfg!(target_arch = "aarch64") {
                simd_bin!(super::simd_arm::add_f32_slice);
            }
            arm!(BinaryOp::Add)
        }
        BinaryOp::Sub => {
            if super::simd_x86::has_avx2_fma() {
                simd_bin!(super::simd_x86::sub_f32_slice);
            }
            if cfg!(target_arch = "aarch64") {
                simd_bin!(super::simd_arm::sub_f32_slice);
            }
            arm!(BinaryOp::Sub)
        }
        BinaryOp::Mul => {
            if super::simd_x86::has_avx2_fma() {
                simd_bin!(super::simd_x86::mul_f32_slice);
            }
            if cfg!(target_arch = "aarch64") {
                simd_bin!(super::simd_arm::mul_f32_slice);
            }
            arm!(BinaryOp::Mul)
        }
        BinaryOp::Div => {
            if super::simd_x86::has_avx2_fma() {
                simd_bin!(super::simd_x86::div_f32_slice);
            }
            if cfg!(target_arch = "aarch64") {
                simd_bin!(super::simd_arm::div_f32_slice);
            }
            arm!(BinaryOp::Div)
        }
        BinaryOp::Maximum => arm!(BinaryOp::Maximum),
        BinaryOp::Minimum => arm!(BinaryOp::Minimum),
        BinaryOp::Pow => arm!(BinaryOp::Pow),
        BinaryOp::Fmod => arm!(BinaryOp::Fmod),
        BinaryOp::Remainder => arm!(BinaryOp::Remainder),
        BinaryOp::FloorDiv => arm!(BinaryOp::FloorDiv),
    }
}

/// The smallest and largest storage index a layout can reach, or `None` if it has no elements.
fn layout_extent(l: &Layout) -> Option<(isize, isize)> {
    let (mut lo, mut hi) = (l.offset() as isize, l.offset() as isize);
    for (&d, &st) in l.shape().iter().zip(l.strides()) {
        if d == 0 {
            return None;
        }
        let span = (d as isize - 1) * st;
        if span < 0 { lo += span } else { hi += span }
    }
    Some((lo, hi))
}

/// Binary kernel for two arbitrary layouts of one shape (broadcast dims have stride 0, views any stride): the leading
/// dimensions are walked row by row (in parallel for large outputs) and the last dimension runs as a tight loop, with
/// slice-based fast paths for the stride pairs that occur in practice. `None` if a layout does not fit its storage.
fn strided_binary<T, F>(a: &[T], la: &Layout, b: &[T], lb: &Layout, f: F) -> Option<Vec<T>>
where
    T: Copy + Default + Send + Sync + 'static,
    F: Fn(T, T) -> T + Sync,
{
    let shape = la.shape();
    let nd = shape.len();
    if nd == 0 || shape != lb.shape() {
        return None;
    }
    let (alo, ahi) = layout_extent(la)?;
    let (blo, bhi) = layout_extent(lb)?;
    if alo < 0 || blo < 0 || ahi as usize >= a.len() || bhi as usize >= b.len() {
        return None;
    }
    let cols = shape[nd - 1];
    let total: usize = shape.iter().product();
    if let (Some(ra), Some(rb)) = (la.contiguous_range_in(a.len()), lb.contiguous_range_in(b.len())) {
        let sa = &a[ra];
        let sb = &b[rb];
        let mut out = crate::pool::take::<T>(total).unwrap_or_else(|| vec![T::default(); total]);
        #[cfg(feature = "rayon")]
        if total >= 16384 {
            out.par_chunks_mut(8192)
                .zip(sa.par_chunks(8192))
                .zip(sb.par_chunks(8192))
                .for_each(|((o, x), y)| {
                    for ((dst, &lhs), &rhs) in o.iter_mut().zip(x).zip(y) {
                        *dst = f(lhs, rhs);
                    }
                });
            return Some(out);
        }
        for ((dst, &lhs), &rhs) in out.iter_mut().zip(sa).zip(sb) {
            *dst = f(lhs, rhs);
        }
        return Some(out);
    }
    let (sa, sb) = (la.strides()[nd - 1], lb.strides()[nd - 1]);
    let lead = &shape[..nd - 1];
    let (lsa, lsb) = (&la.strides()[..nd - 1], &lb.strides()[..nd - 1]);
    let (oa, ob) = (la.offset() as isize, lb.offset() as isize);
    let row = |r: usize, out: &mut [T]| {
        // storage offsets of the first element of row `r`
        let (mut pa, mut pb, mut rem) = (oa, ob, r);
        for d in (0..lead.len()).rev() {
            let c = (rem % lead[d]) as isize;
            rem /= lead[d];
            pa += c * lsa[d];
            pb += c * lsb[d];
        }
        let (pa, pb) = (pa as usize, pb as usize);
        match (sa, sb) {
            (1, 1) => out.iter_mut().zip(&a[pa..pa + cols]).zip(&b[pb..pb + cols]).for_each(|((o, &x), &y)| *o = f(x, y)),
            (1, 0) => {
                let y = b[pb];
                out.iter_mut().zip(&a[pa..pa + cols]).for_each(|(o, &x)| *o = f(x, y));
            }
            (0, 1) => {
                let x = a[pa];
                out.iter_mut().zip(&b[pb..pb + cols]).for_each(|(o, &y)| *o = f(x, y));
            }
            (sa, 1) if sa > 1 => out.iter_mut().zip(a[pa..].iter().step_by(sa as usize)).zip(&b[pb..pb + cols]).for_each(|((o, &x), &y)| *o = f(x, y)),
            (1, sb) if sb > 1 => out.iter_mut().zip(&a[pa..pa + cols]).zip(b[pb..].iter().step_by(sb as usize)).for_each(|((o, &x), &y)| *o = f(x, y)),
            _ => {
                for (j, o) in out.iter_mut().enumerate() {
                    let j = j as isize;
                    *o = f(a[(pa as isize + j * sa) as usize], b[(pb as isize + j * sb) as usize]);
                }
            }
        }
    };
    let mut out = crate::pool::take::<T>(total).unwrap_or_else(|| vec![T::default(); total]);
    #[cfg(feature = "rayon")]
    if total >= 16384 {
        let group = (EW_CHUNK / cols).max(1);
        out.par_chunks_mut(cols * group).enumerate().for_each(|(gi, block)| block.chunks_mut(cols).enumerate().for_each(|(k, o)| row(gi * group + k, o)));
        return Some(out);
    }
    out.chunks_mut(cols).enumerate().for_each(|(r, o)| row(r, o));
    Some(out)
}

/// Host-memory storage supporting all 8 [`DType`] variants.
#[derive(Clone, Debug, PartialEq)]
pub enum CpuStorage {
    Bool(Vec<bool>),
    U8(Vec<u8>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    F8E4M3(Vec<f8e4m3>),
    F8E5M2(Vec<f8e5m2>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    C64(Vec<[f32; 2]>),
    C128(Vec<[f64; 2]>),
}

/// Exact `base^exp` for integer-pow validation, computed in i128 so any
/// overflow of the true value is observable. None when `exp < 0` (no
/// integer value) or the true value exceeds i128 (hence every dtype).
fn int_pow_exact(base: i64, exp: i64) -> Option<i128> {
    if exp < 0 {
        return None;
    }
    if exp == 0 {
        return Some(1);
    }
    if base == 0 || base == 1 {
        return Some(base as i128);
    }
    if base == -1 {
        return Some(if exp % 2 == 0 { 1 } else { -1 });
    }
    let mut k = u32::try_from(exp).ok()?;
    let mut result: i128 = 1;
    let mut b = base as i128;
    while k > 0 {
        if k & 1 == 1 {
            result = result.checked_mul(b)?;
        }
        k >>= 1;
        if k > 0 {
            b = b.checked_mul(b)?;
        }
    }
    Some(result)
}

/// Exact non-negative integer power for the pow kernels. The binary-op
/// pre-scan guarantees a valid exponent and a fitting result, so plain
/// arithmetic is correct — and would panic loudly in debug builds if
/// validation ever disagreed with the kernel.
fn int_pow_kernel(base: i64, exp: i64) -> i64 {
    debug_assert!(exp >= 0);
    let mut result = 1i64;
    let mut b = base;
    let mut e = exp as u64;
    while e > 0 {
        if e & 1 == 1 {
            result *= b;
        }
        e >>= 1;
        // Skip the last squaring: b^(2*2^floor(log2 e)) can exceed a
        // fitting result (e.g. (2^20)^3 fits i64, (2^20)^4 does not).
        if e > 0 {
            b *= b;
        }
    }
    result
}

/// Which unary op [`CpuStorage::unary_backward_f32`] differentiates, and whether its derivative is a function of the op's
/// input or of its output.
#[derive(Clone, Copy, Debug)]
pub(crate) enum UnaryBwd {
    TanhFromOutput,
    SigmoidFromOutput,
    ExpFromOutput,
    ReluFromInput,
    SquareFromInput,
}

impl CpuStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::F16(_) => DType::F16,
            Self::BF16(_) => DType::BF16,
            Self::F8E4M3(_) => DType::F8E4M3,
            Self::F8E5M2(_) => DType::F8E5M2,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::C64(_) => DType::C64,
            Self::C128(_) => DType::C128,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bool(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F16(v) => v.len(),
            Self::BF16(v) => v.len(),
            Self::F8E4M3(v) => v.len(),
            Self::F8E5M2(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::C64(v) => v.len(),
            Self::C128(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Detaches the buffer from its `Vec` without freeing it (the storage then holds an empty `U8` vector). Used for storage
    /// that aliases foreign memory.
    pub(crate) fn forget_buffer(&mut self) {
        let old = std::mem::replace(self, Self::U8(Vec::new()));
        std::mem::forget(old);
    }

    /// Raw pointer to the start of the underlying CPU buffer (never reallocates while pinned).
    pub fn raw_cpu_ptr(&self) -> *mut u8 {
        match self {
            Self::Bool(v) => v.as_ptr() as *mut u8,
            Self::U8(v) => v.as_ptr() as *mut u8,
            Self::I8(v) => v.as_ptr() as *mut u8,
            Self::I16(v) => v.as_ptr() as *mut u8,
            Self::U16(v) => v.as_ptr() as *mut u8,
            Self::U32(v) => v.as_ptr() as *mut u8,
            Self::U64(v) => v.as_ptr() as *mut u8,
            Self::I32(v) => v.as_ptr() as *mut u8,
            Self::I64(v) => v.as_ptr() as *mut u8,
            Self::F16(v) => v.as_ptr() as *mut u8,
            Self::BF16(v) => v.as_ptr() as *mut u8,
            Self::F8E4M3(v) => v.as_ptr() as *mut u8,
            Self::F8E5M2(v) => v.as_ptr() as *mut u8,
            Self::F32(v) => v.as_ptr() as *mut u8,
            Self::F64(v) => v.as_ptr() as *mut u8,
            Self::C64(v) => v.as_ptr() as *mut u8,
            Self::C128(v) => v.as_ptr() as *mut u8,
        }
    }

    /// Copies elements from `src` into `self` in-place without reallocating if variant and length match.
    pub fn copy_in_place_from(&mut self, src: &Self) -> bool {
        match (self, src) {
            (Self::Bool(d), Self::Bool(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::U8(d), Self::U8(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::I8(d), Self::I8(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::I16(d), Self::I16(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::U16(d), Self::U16(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::U32(d), Self::U32(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::U64(d), Self::U64(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::I32(d), Self::I32(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::I64(d), Self::I64(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::F16(d), Self::F16(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::BF16(d), Self::BF16(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::F8E4M3(d), Self::F8E4M3(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::F8E5M2(d), Self::F8E5M2(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::F32(d), Self::F32(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::F64(d), Self::F64(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::C64(d), Self::C64(s)) if d.len() == s.len() => d.copy_from_slice(s),
            (Self::C128(d), Self::C128(s)) if d.len() == s.len() => d.copy_from_slice(s),
            _ => return false,
        }
        true
    }

    pub fn all_finite(&self, layout: &Layout) -> bool {
        match self {
            Self::F32(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    #[cfg(feature = "rayon")]
                    if r.len() >= 16384 {
                        return v[r].par_iter().all(|x| x.is_finite());
                    }
                    v[r].iter().all(|x| x.is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i].is_finite())
                }
            }
            Self::F64(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    #[cfg(feature = "rayon")]
                    if r.len() >= 16384 {
                        return v[r].par_iter().all(|x| x.is_finite());
                    }
                    v[r].iter().all(|x| x.is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i].is_finite())
                }
            }
            Self::C64(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    v[r].iter().all(|x| x[0].is_finite() && x[1].is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i][0].is_finite() && v[i][1].is_finite())
                }
            }
            Self::C128(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    v[r].iter().all(|x| x[0].is_finite() && x[1].is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i][0].is_finite() && v[i][1].is_finite())
                }
            }
            Self::F16(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    v[r].iter().all(|x| x.to_f32().is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i].to_f32().is_finite())
                }
            }
            Self::BF16(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    v[r].iter().all(|x| x.to_f32().is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i].to_f32().is_finite())
                }
            }
            Self::F8E4M3(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    v[r].iter().all(|x| x.to_f32().is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i].to_f32().is_finite())
                }
            }
            Self::F8E5M2(v) => {
                if let Some(r) = layout.contiguous_range_in(v.len()) {
                    v[r].iter().all(|x| x.to_f32().is_finite())
                } else {
                    layout.storage_offsets().all(|i| v[i].to_f32().is_finite())
                }
            }
            _ => true,
        }
    }

    pub fn from_slice<T: Element>(data: &[T]) -> Result<Self> {
        crate::limits::check_elements("cpu storage element count", &[data.len()])?;
        Ok(match T::DTYPE {
            DType::Bool => Self::Bool(data.iter().map(|x| x.to_f64() != 0.0).collect()),
            DType::U8 => Self::U8(data.iter().map(|x| x.to_f64() as u8).collect()),
            DType::I8 => Self::I8(data.iter().map(|x| x.to_f64() as i8).collect()),
            DType::I16 => Self::I16(data.iter().map(|x| x.to_f64() as i16).collect()),
            DType::U16 => Self::U16(data.iter().map(|x| x.to_f64() as u16).collect()),
            DType::U32 => Self::U32(data.iter().map(|x| x.to_f64() as u32).collect()),
            DType::U64 => Self::U64(data.iter().map(|x| x.to_u64()).collect()),
            DType::I32 => Self::I32(data.iter().map(|x| x.to_f64() as i32).collect()),
            DType::I64 => Self::I64(data.iter().map(|x| x.to_i64()).collect()),
            DType::F16 => Self::F16(data.iter().map(|x| f16::from_f32(x.to_f32())).collect()),
            DType::BF16 => Self::BF16(data.iter().map(|x| bf16::from_f32(x.to_f32())).collect()),
            DType::F8E4M3 => Self::F8E4M3(data.iter().map(|x| f8e4m3::from_f32(x.to_f32())).collect()),
            DType::F8E5M2 => Self::F8E5M2(data.iter().map(|x| f8e5m2::from_f32(x.to_f32())).collect()),
            DType::F32 => Self::F32(data.iter().map(|x| x.to_f32()).collect()),
            DType::F64 => Self::F64(data.iter().map(|x| x.to_f64()).collect()),
            DType::C64 => Self::C64(data.iter().map(|x| [x.to_f32(), 0.0]).collect()),
            DType::C128 => Self::C128(data.iter().map(|x| [x.to_f64(), 0.0]).collect()),
        })
    }

    pub fn full(numel: usize, val: f64, dtype: DType) -> Result<Self> {
        crate::limits::check_elements("cpu storage element count", &[numel])?;
        Ok(match dtype {
            DType::Bool => Self::Bool(vec![val != 0.0; numel]),
            DType::U8 => Self::U8(vec![val as u8; numel]),
            DType::I8 => Self::I8(vec![val as i8; numel]),
            DType::I16 => Self::I16(vec![val as i16; numel]),
            DType::U16 => Self::U16(vec![val as u16; numel]),
            DType::U32 => Self::U32(vec![val as u32; numel]),
            DType::U64 => Self::U64(vec![val as u64; numel]),
            DType::I32 => Self::I32(vec![val as i32; numel]),
            DType::I64 => Self::I64(vec![val as i64; numel]),
            DType::F16 => Self::F16(vec![f16::from_f32(val as f32); numel]),
            DType::BF16 => Self::BF16(vec![bf16::from_f32(val as f32); numel]),
            DType::F8E4M3 => Self::F8E4M3(vec![f8e4m3::from_f32(val as f32); numel]),
            DType::F8E5M2 => Self::F8E5M2(vec![f8e5m2::from_f32(val as f32); numel]),
            DType::F32 => Self::F32(vec![val as f32; numel]),
            DType::F64 => Self::F64(vec![val; numel]),
            DType::C64 => Self::C64(vec![[val as f32, 0.0]; numel]),
            DType::C128 => Self::C128(vec![[val, 0.0]; numel]),
        })
    }

    /// Reads the tensor view described by `layout` into a contiguous `Vec<T>`.
    pub fn to_vec<T: Element>(&self, layout: &Layout) -> Vec<T> {
        macro_rules! extract {
            ($vec:expr, $conv:expr) => {{
                if let Some(range) = layout.contiguous_range_in($vec.len()) {
                    $vec[range].iter().map($conv).collect()
                } else if layout.shape().len() == 2 {
                    let rows = layout.shape()[0];
                    let cols = layout.shape()[1];
                    let s0 = layout.strides()[0];
                    let s1 = layout.strides()[1];
                    let offset = layout.offset();
                    let n = rows * cols;
                    let min_idx = offset as isize
                        + (if s0 < 0 { (rows as isize - 1) * s0 } else { 0 })
                        + (if s1 < 0 { (cols as isize - 1) * s1 } else { 0 });
                    let max_idx = offset as isize
                        + (if s0 > 0 { (rows as isize - 1) * s0 } else { 0 })
                        + (if s1 > 0 { (cols as isize - 1) * s1 } else { 0 });
                    if min_idx >= 0 && (max_idx as usize) < $vec.len() {
                        let mut out = Vec::with_capacity(n);
                        for r in 0..rows {
                            let mut idx = offset as isize + r as isize * s0;
                            for _ in 0..cols {
                                out.push(($conv)(&$vec[idx as usize]));
                                idx += s1;
                            }
                        }
                        out
                    } else {
                        layout
                            .storage_offsets()
                            .map(|i| ($conv)(&$vec[i]))
                            .collect()
                    }
                } else {
                    layout
                        .storage_offsets()
                        .map(|i| ($conv)(&$vec[i]))
                        .collect()
                }
            }};
        }
        match self {
            Self::Bool(v) => extract!(v, |&x: &bool| T::from_f64(if x { 1.0 } else { 0.0 })),
            Self::U8(v) => extract!(v, |&x: &u8| T::from_f64(x as f64)),
            Self::I8(v) => extract!(v, |&x: &i8| T::from_f64(x as f64)),
            Self::I16(v) => extract!(v, |&x: &i16| T::from_f64(x as f64)),
            Self::U16(v) => extract!(v, |&x: &u16| T::from_f64(x as f64)),
            Self::U32(v) => extract!(v, |&x: &u32| T::from_f64(x as f64)),
            Self::U64(v) => extract!(v, |&x: &u64| T::from_u64(x)),
            Self::I32(v) => extract!(v, |&x: &i32| T::from_f64(x as f64)),
            Self::I64(v) => extract!(v, |&x: &i64| T::from_i64(x)),
            Self::F16(v) => extract!(v, |&x: &f16| T::from_f32(x.to_f32())),
            Self::BF16(v) => extract!(v, |&x: &bf16| T::from_f32(x.to_f32())),
            Self::F8E4M3(v) => extract!(v, |&x: &f8e4m3| T::from_f32(x.to_f32())),
            Self::F8E5M2(v) => extract!(v, |&x: &f8e5m2| T::from_f32(x.to_f32())),
            Self::F32(v) => extract!(v, |&x: &f32| T::from_f32(x)),
            Self::F64(v) => extract!(v, |&x: &f64| T::from_f64(x)),
            Self::C64(v) => extract!(v, |&x: &[f32; 2]| T::from_f32(x[0])),
            Self::C128(v) => extract!(v, |&x: &[f64; 2]| T::from_f64(x[0])),
        }
    }

    pub fn to_contiguous(&self, layout: &Layout) -> Result<Self> {
        macro_rules! copy_typed {
            ($variant:ident, $v:expr) => {{
                if let Some(range) = layout.contiguous_range_in($v.len()) {
                    Ok(Self::$variant($v[range].to_vec()))
                } else if layout.shape().len() == 2 {
                    let rows = layout.shape()[0];
                    let cols = layout.shape()[1];
                    let s0 = layout.strides()[0];
                    let s1 = layout.strides()[1];
                    let offset = layout.offset();
                    let n = rows * cols;
                    let min_idx = offset as isize
                        + (if s0 < 0 { (rows as isize - 1) * s0 } else { 0 })
                        + (if s1 < 0 { (cols as isize - 1) * s1 } else { 0 });
                    let max_idx = offset as isize
                        + (if s0 > 0 { (rows as isize - 1) * s0 } else { 0 })
                        + (if s1 > 0 { (cols as isize - 1) * s1 } else { 0 });
                    if min_idx >= 0 && (max_idx as usize) < $v.len() {
                        let mut out = vec![Default::default(); n];
                        #[cfg(feature = "rayon")]
                        if n >= 16384 {
                            out.par_chunks_mut(cols).enumerate().for_each(|(r, row_out)| {
                                let mut idx = offset as isize + r as isize * s0;
                                for dst in row_out.iter_mut() {
                                    *dst = $v[idx as usize];
                                    idx += s1;
                                }
                            });
                        } else {
                            for (r, row_out) in out.chunks_mut(cols).enumerate() {
                                let mut idx = offset as isize + r as isize * s0;
                                for dst in row_out.iter_mut() {
                                    *dst = $v[idx as usize];
                                    idx += s1;
                                }
                            }
                        }
                        #[cfg(not(feature = "rayon"))]
                        for (r, row_out) in out.chunks_mut(cols).enumerate() {
                            let mut idx = offset as isize + r as isize * s0;
                            for dst in row_out.iter_mut() {
                                *dst = $v[idx as usize];
                                idx += s1;
                            }
                        }
                        Ok(Self::$variant(out))
                    } else {
                        Ok(Self::$variant(
                            layout.storage_offsets().map(|i| $v[i]).collect(),
                        ))
                    }
                } else {
                    Ok(Self::$variant(
                        layout.storage_offsets().map(|i| $v[i]).collect(),
                    ))
                }
            }};
        }
        match self {
            Self::Bool(v) => copy_typed!(Bool, v),
            Self::U8(v) => copy_typed!(U8, v),
            Self::I8(v) => copy_typed!(I8, v),
            Self::I16(v) => copy_typed!(I16, v),
            Self::U16(v) => copy_typed!(U16, v),
            Self::U32(v) => copy_typed!(U32, v),
            Self::U64(v) => copy_typed!(U64, v),
            Self::I32(v) => copy_typed!(I32, v),
            Self::I64(v) => copy_typed!(I64, v),
            Self::F16(v) => copy_typed!(F16, v),
            Self::BF16(v) => copy_typed!(BF16, v),
            Self::F8E4M3(v) => copy_typed!(F8E4M3, v),
            Self::F8E5M2(v) => copy_typed!(F8E5M2, v),
            Self::F32(v) => copy_typed!(F32, v),
            Self::F64(v) => copy_typed!(F64, v),
            Self::C64(v) => copy_typed!(C64, v),
            Self::C128(v) => copy_typed!(C128, v),
        }
    }

    pub fn to_dtype(&self, layout: &Layout, dst: DType) -> Result<Self> {
        if self.dtype() == dst {
            return self.to_contiguous(layout);
        }
        match dst {
            DType::Bool => Ok(Self::Bool(self.to_vec::<bool>(layout))),
            DType::U8 => Ok(Self::U8(self.to_vec::<u8>(layout))),
            DType::I8 => Ok(Self::I8(self.to_vec::<i8>(layout))),
            DType::I16 => Ok(Self::I16(self.to_vec::<i16>(layout))),
            DType::U16 => Ok(Self::U16(self.to_vec::<u16>(layout))),
            DType::U32 => Ok(Self::U32(self.to_vec::<u32>(layout))),
            DType::U64 => Ok(Self::U64(self.to_vec::<u64>(layout))),
            DType::I32 => Ok(Self::I32(self.to_vec::<i32>(layout))),
            DType::I64 => Ok(Self::I64(self.to_vec::<i64>(layout))),
            DType::F16 => Ok(Self::F16(self.to_vec::<f16>(layout))),
            DType::BF16 => Ok(Self::BF16(self.to_vec::<bf16>(layout))),
            DType::F8E4M3 => Ok(Self::F8E4M3(self.to_vec::<f8e4m3>(layout))),
            DType::F8E5M2 => Ok(Self::F8E5M2(self.to_vec::<f8e5m2>(layout))),
            DType::F32 => Ok(Self::F32(self.to_vec::<f32>(layout))),
            DType::F64 => Ok(Self::F64(self.to_vec::<f64>(layout))),
            DType::C64 => match self {
                Self::C128(v) => {
                    let out = if let Some(r) = layout.contiguous_range_in(v.len()) {
                        v[r].iter().map(|x| [x[0] as f32, x[1] as f32]).collect()
                    } else {
                        layout.storage_offsets().map(|i| [v[i][0] as f32, v[i][1] as f32]).collect()
                    };
                    Ok(Self::C64(out))
                }
                _ => Ok(Self::C64(
                    self.to_vec::<f32>(layout).into_iter().map(|r| [r, 0.0]).collect(),
                )),
            },
            DType::C128 => match self {
                Self::C64(v) => {
                    let out = if let Some(r) = layout.contiguous_range_in(v.len()) {
                        v[r].iter().map(|x| [x[0] as f64, x[1] as f64]).collect()
                    } else {
                        layout.storage_offsets().map(|i| [v[i][0] as f64, v[i][1] as f64]).collect()
                    };
                    Ok(Self::C128(out))
                }
                _ => Ok(Self::C128(
                    self.to_vec::<f64>(layout).into_iter().map(|r| [r, 0.0]).collect(),
                )),
            },
        }
    }

    /// Scatters elements from a `src_layout` tensor into a zero tensor of `spec.src_shape`.
    pub fn view_scatter(&self, src_layout: &Layout, spec: &ViewSpec) -> Result<Self> {
        if src_layout.shape() != spec.dst_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                op: "view_scatter",
                expected: spec.dst_shape.clone(),
                got: src_layout.shape().to_vec(),
            });
        }
        let dst_layout = Layout::contiguous(spec.src_shape.clone());
        let scattered_view = spec.apply_layout(&dst_layout)?;
        // `ViewSpec` fields are pub: a hand-built spec can carry strides that
        // scatter past the fresh output below (index panic) instead of erroring.
        scattered_view.check_storage_bounds(dst_layout.numel(), "view_scatter")?;

        macro_rules! scatter_typed {
            ($variant:ident, $v:expr, $zero:expr, $add:expr) => {{
                let mut out = vec![$zero; dst_layout.numel()];
                for (src_off, dst_off) in src_layout
                    .storage_offsets()
                    .zip(scattered_view.storage_offsets())
                {
                    $add(&mut out[dst_off], $v[src_off]);
                }
                Ok(Self::$variant(out))
            }};
        }

        match self {
            Self::Bool(v) => scatter_typed!(Bool, v, false, |a: &mut bool, b: bool| *a |= b),
            Self::U8(v) => scatter_typed!(U8, v, 0u8, |a: &mut u8, b: u8| *a = a.wrapping_add(b)),
            Self::I8(v) => scatter_typed!(I8, v, 0i8, |a: &mut i8, b: i8| *a = a.wrapping_add(b)),
            Self::I16(v) => scatter_typed!(I16, v, 0i16, |a: &mut i16, b: i16| *a = a.wrapping_add(b)),
            Self::U16(v) => scatter_typed!(U16, v, 0u16, |a: &mut u16, b: u16| *a = a.wrapping_add(b)),
            Self::U32(v) => scatter_typed!(U32, v, 0u32, |a: &mut u32, b: u32| *a = a.wrapping_add(b)),
            Self::U64(v) => scatter_typed!(U64, v, 0u64, |a: &mut u64, b: u64| *a = a.wrapping_add(b)),
            Self::I32(v) => {
                scatter_typed!(I32, v, 0i32, |a: &mut i32, b: i32| *a = a.wrapping_add(b))
            }
            Self::I64(v) => {
                scatter_typed!(I64, v, 0i64, |a: &mut i64, b: i64| *a = a.wrapping_add(b))
            }
            Self::F16(v) => scatter_typed!(F16, v, f16::ZERO, |a: &mut f16, b: f16| {
                *a = f16::from_f32(a.to_f32() + b.to_f32())
            }),
            Self::BF16(v) => scatter_typed!(BF16, v, bf16::ZERO, |a: &mut bf16, b: bf16| {
                *a = bf16::from_f32(a.to_f32() + b.to_f32())
            }),
            Self::F8E4M3(v) => scatter_typed!(F8E4M3, v, f8e4m3::ZERO, |a: &mut f8e4m3, b: f8e4m3| {
                *a = f8e4m3::from_f32(a.to_f32() + b.to_f32())
            }),
            Self::F8E5M2(v) => scatter_typed!(F8E5M2, v, f8e5m2::ZERO, |a: &mut f8e5m2, b: f8e5m2| {
                *a = f8e5m2::from_f32(a.to_f32() + b.to_f32())
            }),
            Self::F32(v) => scatter_typed!(F32, v, 0.0f32, |a: &mut f32, b: f32| *a += b),
            Self::F64(v) => scatter_typed!(F64, v, 0.0f64, |a: &mut f64, b: f64| *a += b),
            Self::C64(v) => scatter_typed!(C64, v, [0.0f32, 0.0f32], |a: &mut [f32; 2], b: [f32; 2]| {
                a[0] += b[0];
                a[1] += b[1];
            }),
            Self::C128(v) => scatter_typed!(C128, v, [0.0f64, 0.0f64], |a: &mut [f64; 2], b: [f64; 2]| {
                a[0] += b[0];
                a[1] += b[1];
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Unary operations
    // -----------------------------------------------------------------------
    /// Fused single-pass last-axis softmax on contiguous `F32`/`F64` buffers when all rows are finite.
    /// Returns `None` on non-contiguous layouts, unsupported dtypes, or non-finite inputs (`+inf`/`NaN`)
    /// so the general path handles IEEE-754 edge cases identically.
    /// Fused last-axis `log_softmax` on contiguous `F32` storage: `(x - max) - ln(sum(exp(x - max)))` in two passes per row
    /// instead of the eight tensor operations of the composed form. `None` (use the composed path) for any non-finite
    /// element, other dtypes or non-contiguous layouts.
    pub fn log_softmax_last_axis(&self, layout: &Layout) -> Option<Self> {
        let Self::F32(v) = self else { return None };
        let cols = *layout.shape().last()?;
        if cols == 0 {
            return None;
        }
        let total: usize = layout.shape().iter().product();
        let src = &v[layout.contiguous_range_in(v.len())?];
        let non_finite = |c: &[f32]| c.iter().fold(false, |bad, x| bad | (x.to_bits() & 0x7f80_0000 == 0x7f80_0000));
        #[cfg(feature = "rayon")]
        let bad = if total >= 16384 { src.par_chunks(EW_CHUNK).any(non_finite) } else { non_finite(src) };
        #[cfg(not(feature = "rayon"))]
        let bad = non_finite(src);
        if bad {
            return None;
        }
        let mut out = crate::pool::take::<f32>(total).unwrap_or_else(|| vec![0.0; total]);
        let row_fn = |(src_row, dst_row): (&[f32], &mut [f32])| {
            let m = src_row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
            // ln(sum exp(x - m)) with the elements equal to the maximum counted separately: their terms are exactly 1, so the
            // rest (`tail`) is summed on its own in f64 and a dominant element gives ln(1 + tail) = ln_1p(tail) instead of the
            // ln of a rounded 1.00001, which has lost most of its digits
            let (mut tail, mut at_max) = (0.0f64, 0usize);
            for (d, &x) in dst_row.iter_mut().zip(src_row) {
                if x < m {
                    tail += fast_exp_f32(x - m) as f64;
                } else {
                    at_max += 1;
                }
                *d = x;
            }
            let ln_sum = if at_max == 1 { tail.ln_1p() } else { (at_max as f64 + tail).ln() };
            // (x - m) is exact in f64, so the output is rounded once
            for d in dst_row.iter_mut() {
                *d = ((*d as f64 - m as f64) - ln_sum) as f32;
            }
        };
        #[cfg(feature = "rayon")]
        if total >= 16384 {
            let block = cols * (EW_CHUNK / cols).max(1);
            src.par_chunks(block)
                .zip(out.par_chunks_mut(block))
                .for_each(|(s, d)| s.chunks(cols).zip(d.chunks_mut(cols)).for_each(row_fn));
            return Some(Self::F32(out));
        }
        src.chunks(cols).zip(out.chunks_mut(cols)).for_each(row_fn);
        Some(Self::F32(out))
    }

    pub fn softmax_last_axis(&self, layout: &Layout) -> Option<Self> {
        let shape = layout.shape();
        let cols = *shape.last()?;
        if cols == 0 {
            return None;
        }
        let total: usize = shape.iter().product();
        match self {
            Self::F32(v) => {
                let range = layout.contiguous_range_in(v.len())?;
                let src = &v[range];
                // Fast scan: fall back if any element is non-finite (+inf/-inf/NaN)
                // (an OR over the exponent bits: unlike a short-circuiting `any` it vectorizes, and it runs in parallel)
                let non_finite = |c: &[f32]| c.iter().fold(false, |bad, x| bad | (x.to_bits() & 0x7f80_0000 == 0x7f80_0000));
                #[cfg(feature = "rayon")]
                let bad = src.par_chunks(EW_CHUNK).any(non_finite);
                #[cfg(not(feature = "rayon"))]
                let bad = non_finite(src);
                if bad {
                    return None;
                }
                let mut out = crate::pool::take::<f32>(total).unwrap_or_else(|| vec![0.0; total]);
                let row_fn = |(src_row, dst_row): (&[f32], &mut [f32])| {
                    if super::simd_x86::softmax_row_f32(src_row, dst_row) {
                        return;
                    }
                    let mut m = f32::NEG_INFINITY;
                    for &x in src_row {
                        if x > m {
                            m = x;
                        }
                    }
                    // the exponentials in their own loop (it vectorizes); the ordered sum stays a separate scalar pass
                    for (d, &x) in dst_row.iter_mut().zip(src_row.iter()) {
                        *d = fast_exp_f32(x - m);
                    }
                    let mut sum = 0.0f32;
                    for &e in dst_row.iter() {
                        sum += e;
                    }
                    let inv = 1.0f32 / sum;
                    for d in dst_row.iter_mut() {
                        *d *= inv;
                    }
                };
                #[cfg(feature = "rayon")]
                if total >= 16384 {
                    // several rows per task (about EW_CHUNK elements), not one task per row
                    let block = cols * (EW_CHUNK / cols).max(1);
                    src.par_chunks(block)
                        .zip(out.par_chunks_mut(block))
                        .for_each(|(s, d)| s.chunks(cols).zip(d.chunks_mut(cols)).for_each(row_fn));
                    return Some(Self::F32(out));
                }
                src.chunks(cols).zip(out.chunks_mut(cols)).for_each(row_fn);
                Some(Self::F32(out))
            }
            Self::F64(v) => {
                let range = layout.contiguous_range_in(v.len())?;
                let src = &v[range];
                if src.iter().any(|&x| !x.is_finite()) {
                    return None;
                }
                let mut out = vec![0.0f64; total];
                let row_fn = |(src_row, dst_row): (&[f64], &mut [f64])| {
                    let mut m = f64::NEG_INFINITY;
                    for &x in src_row {
                        if x > m {
                            m = x;
                        }
                    }
                    let mut sum = 0.0f64;
                    for (d, &x) in dst_row.iter_mut().zip(src_row.iter()) {
                        let e = (x - m).exp();
                        *d = e;
                        sum += e;
                    }
                    let inv = 1.0f64 / sum;
                    for d in dst_row.iter_mut() {
                        *d *= inv;
                    }
                };
                #[cfg(feature = "rayon")]
                if total >= 16384 {
                    src.par_chunks(cols)
                        .zip(out.par_chunks_mut(cols))
                        .for_each(row_fn);
                    return Some(Self::F64(out));
                }
                src.chunks(cols).zip(out.chunks_mut(cols)).for_each(row_fn);
                Some(Self::F64(out))
            }
            _ => None,
        }
    }

    /// Fused single-pass last-axis `LayerNorm` on contiguous `F32` storage.
    pub fn layer_norm_last_axis(
        &self,
        layout: &Layout,
        weight: &Self,
        w_layout: &Layout,
        bias: &Self,
        b_layout: &Layout,
        eps: f64,
    ) -> Option<Self> {
        let d = w_layout.numel();
        if d == 0 || b_layout.numel() != d {
            return None;
        }
        let total = layout.numel();
        if total == 0 || total % d != 0 {
            return None;
        }
        match (self, weight, bias) {
            (Self::F32(xv), Self::F32(wv), Self::F32(bv)) => {
                let x_range = layout.contiguous_range_in(xv.len())?;
                let w_range = w_layout.contiguous_range_in(wv.len())?;
                let b_range = b_layout.contiguous_range_in(bv.len())?;
                if w_range.len() != d || b_range.len() != d {
                    return None;
                }
                let src = &xv[x_range];
                let ws = &wv[w_range];
                let bs = &bv[b_range];
                let mut out = crate::pool::take::<f32>(total).unwrap_or_else(|| vec![0.0; total]);
                let row_fn = |(src_row, dst_row): (&[f32], &mut [f32])| {
                    if super::simd_x86::layer_norm_row_f32(src_row, ws, bs, dst_row, eps as f32) {
                        return;
                    }
                    let (hi, lo) = center_split(src_row);
                    let mut var_sum = 0.0f64;
                    for &x in src_row {
                        let diff = ((x - hi) - lo) as f64;
                        var_sum += diff * diff;
                    }
                    let inv32 = (1.0 / (var_sum / (d as f64) + eps).sqrt()) as f32;
                    for ((dst, &x), (&w, &b)) in dst_row.iter_mut().zip(src_row).zip(ws.iter().zip(bs)) {
                        *dst = ((x - hi) - lo) * inv32 * w + b;
                    }
                };
                #[cfg(feature = "rayon")]
                if total >= 16384 {
                    // blocks of rows (about 16K elements): a task per row is mostly scheduling for the usual 256-4096 wide rows
                    let block = d * (16384 / d).max(1);
                    src.par_chunks(block).zip(out.par_chunks_mut(block)).for_each(|(s, o)| s.chunks(d).zip(o.chunks_mut(d)).for_each(row_fn));
                    return Some(Self::F32(out));
                }
                src.chunks(d).zip(out.chunks_mut(d)).for_each(row_fn);
                Some(Self::F32(out))
            }
            _ => None,
        }
    }

    /// Fused single-pass GELU backward on contiguous `F32` storage:
    /// `g * (0.5 * (1 + erf(x / sqrt(2))) + x * inv_sqrt2pi * exp(-0.5 * x^2))`.
    pub fn gelu_backward_f32(&self, g_layout: &Layout, x: &Self, x_layout: &Layout) -> Option<Self> {
        let total = g_layout.numel();
        if total == 0 || g_layout.shape() != x_layout.shape() {
            return None;
        }
        match (self, x) {
            (Self::F32(gv), Self::F32(xv)) => {
                let gr = g_layout.contiguous_range_in(gv.len())?;
                let xr = x_layout.contiguous_range_in(xv.len())?;
                let gs = &gv[gr];
                let xs = &xv[xr];
                let mut out = crate::pool::take::<f32>(total).unwrap_or_else(|| vec![0.0; total]);
                const INV_SQRT2PI: f32 = 0.398_942_3;
                let chunk_fn = |((dst, g_c), x_c): ((&mut [f32], &[f32]), &[f32])| {
                    if super::simd_x86::gelu_backward_f32_slice(g_c, x_c, dst) {
                        return;
                    }
                    for ((o, &g), &xv) in dst.iter_mut().zip(g_c).zip(x_c) {
                        let cdf = fast_norm_cdf_f32(xv);
                        let pdf = fast_exp_f32(-0.5 * xv * xv) * INV_SQRT2PI;
                        *o = g * (cdf + xv * pdf);
                    }
                };
                #[cfg(feature = "rayon")]
                if total >= 16384 {
                    out.par_chunks_mut(8192)
                        .zip(gs.par_chunks(8192))
                        .zip(xs.par_chunks(8192))
                        .for_each(chunk_fn);
                    return Some(Self::F32(out));
                }
                chunk_fn(((&mut out, gs), xs));
                Some(Self::F32(out))
            }
            _ => None,
        }
    }

    /// Fused single-pass backward of a unary op on `F32` storage: `out = g * f(a)`, where `a` is the op's input or output
    /// (see [`UnaryBwd`]). `g` may also be a broadcast scalar (all strides 0), which is what `sum().backward()` hands down.
    pub(crate) fn unary_backward_f32(&self, g_layout: &Layout, a: &Self, a_layout: &Layout, kind: UnaryBwd) -> Option<Self> {
        let total = g_layout.numel();
        if total == 0 || g_layout.shape() != a_layout.shape() {
            return None;
        }
        let (Self::F32(gv), Self::F32(av)) = (self, a) else { return None };
        let a_s = &av[a_layout.contiguous_range_in(av.len())?];
        let scalar_g = g_layout.strides().iter().all(|&s| s == 0).then(|| gv.get(g_layout.offset()).copied()).flatten();
        let g_s: &[f32] = match scalar_g {
            Some(_) => &[],
            None => &gv[g_layout.contiguous_range_in(gv.len())?],
        };
        let mut out = crate::pool::take::<f32>(total).unwrap_or_else(|| vec![0.0; total]);
        fn pass(out: &mut [f32], g_s: &[f32], scalar_g: Option<f32>, a_s: &[f32], f: impl Fn(f32) -> f32 + Sync) {
            const CHUNK: usize = 8192;
            let body = |(i, dst): (usize, &mut [f32])| {
                let a_c = &a_s[i * CHUNK..i * CHUNK + dst.len()];
                match scalar_g {
                    Some(g) => dst.iter_mut().zip(a_c).for_each(|(o, &a)| *o = g * f(a)),
                    None => dst.iter_mut().zip(&g_s[i * CHUNK..i * CHUNK + a_c.len()]).zip(a_c).for_each(|((o, &g), &a)| *o = g * f(a)),
                }
            };
            #[cfg(feature = "rayon")]
            if out.len() >= 16384 {
                out.par_chunks_mut(CHUNK).enumerate().for_each(body);
                return;
            }
            out.chunks_mut(CHUNK).enumerate().for_each(body);
        }
        match kind {
            UnaryBwd::TanhFromOutput => pass(&mut out, g_s, scalar_g, a_s, |y| 1.0 - y * y),
            UnaryBwd::SigmoidFromOutput => pass(&mut out, g_s, scalar_g, a_s, |y| y * (1.0 - y)),
            UnaryBwd::ExpFromOutput => pass(&mut out, g_s, scalar_g, a_s, |y| y),
            UnaryBwd::ReluFromInput => pass(&mut out, g_s, scalar_g, a_s, |x| if x > 0.0 { 1.0 } else { 0.0 }),
            UnaryBwd::SquareFromInput => pass(&mut out, g_s, scalar_g, a_s, |x| 2.0 * x),
        }
        Some(Self::F32(out))
    }

    pub fn unary_op(&self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        if op == UnaryOp::LogicalNot {
            let b = self.to_vec::<bool>(layout);
            return Ok(Self::Bool(b.into_iter().map(|x| !x).collect()));
        }
        macro_rules! run_unary {
            ($variant:ident, $v:expr, $map_fn:expr) => {{
                if let Some(range) = layout.contiguous_range_in($v.len()) {
                    let src = &$v[range];
                    if let Some(mut out) = crate::pool::take(src.len()) {
                        #[cfg(feature = "rayon")]
                        if src.len() >= 16384 {
                            out.par_chunks_mut(8192).zip(src.par_chunks(8192)).for_each(|(o, s)| {
                                for (dst, &x) in o.iter_mut().zip(s) {
                                    *dst = ($map_fn)(x);
                                }
                            });
                            return Ok(Self::$variant(out));
                        }
                        for (dst, &x) in out.iter_mut().zip(src) {
                            *dst = ($map_fn)(x);
                        }
                        return Ok(Self::$variant(out));
                    }
                    #[cfg(feature = "rayon")]
                    if src.len() >= 16384 {
                        let mut out = vec![Default::default(); src.len()];
                        out.par_chunks_mut(8192).zip(src.par_chunks(8192)).for_each(|(o, s)| {
                            for (dst, &x) in o.iter_mut().zip(s) {
                                *dst = ($map_fn)(x);
                            }
                        });
                        return Ok(Self::$variant(out));
                    }
                    return Ok(Self::$variant(src.iter().map(|&x| ($map_fn)(x)).collect()));
                }
                if layout.shape().len() == 2 {
                    let rows = layout.shape()[0];
                    let cols = layout.shape()[1];
                    let s0 = layout.strides()[0];
                    let s1 = layout.strides()[1];
                    let offset = layout.offset();
                    let n = rows * cols;
                    let min_idx = offset as isize
                        + (if s0 < 0 { (rows as isize - 1) * s0 } else { 0 })
                        + (if s1 < 0 { (cols as isize - 1) * s1 } else { 0 });
                    let max_idx = offset as isize
                        + (if s0 > 0 { (rows as isize - 1) * s0 } else { 0 })
                        + (if s1 > 0 { (cols as isize - 1) * s1 } else { 0 });
                    if min_idx >= 0 && (max_idx as usize) < $v.len() {
                        let mut out = vec![Default::default(); n];
                        #[cfg(feature = "rayon")]
                        if n >= 16384 {
                            out.par_chunks_mut(cols).enumerate().for_each(|(r, row_out)| {
                                let mut idx = offset as isize + r as isize * s0;
                                for dst in row_out.iter_mut() {
                                    *dst = ($map_fn)($v[idx as usize]);
                                    idx += s1;
                                }
                            });
                            return Ok(Self::$variant(out));
                        }
                        for (r, row_out) in out.chunks_mut(cols).enumerate() {
                            let mut idx = offset as isize + r as isize * s0;
                            for dst in row_out.iter_mut() {
                                *dst = ($map_fn)($v[idx as usize]);
                                idx += s1;
                            }
                        }
                        return Ok(Self::$variant(out));
                    }
                }
                let out = layout.storage_offsets().map(|i| ($map_fn)($v[i])).collect();
                Ok(Self::$variant(out))
            }};
        }
        match self {
            Self::F32(v) => {
                if op != UnaryOp::LogicalNot
                    && let Some(range) = layout.contiguous_range_in(v.len())
                {
                    return Ok(Self::F32(f32_unary_contiguous(&v[range], op)));
                }
                run_unary!(F32, v, |x: f32| eval_unary_f32(x, op))
            }
            Self::F64(v) => run_unary!(F64, v, |x: f64| eval_unary_f64(x, op)),
            Self::C64(v) => {
                if op == UnaryOp::Abs {
                    let out = if let Some(r) = layout.contiguous_range_in(v.len()) {
                        v[r].iter().map(|&[re, im]| re.hypot(im)).collect()
                    } else {
                        layout.storage_offsets().map(|i| v[i][0].hypot(v[i][1])).collect()
                    };
                    return Ok(Self::F32(out));
                }
                run_unary!(C64, v, |x: [f32; 2]| eval_unary_c64(x, op))
            }
            Self::C128(v) => {
                if op == UnaryOp::Abs {
                    let out = if let Some(r) = layout.contiguous_range_in(v.len()) {
                        v[r].iter().map(|&[re, im]| re.hypot(im)).collect()
                    } else {
                        layout.storage_offsets().map(|i| v[i][0].hypot(v[i][1])).collect()
                    };
                    return Ok(Self::F64(out));
                }
                run_unary!(C128, v, |x: [f64; 2]| eval_unary_c128(x, op))
            }
            Self::F16(v) => run_unary!(F16, v, |x: f16| f16::from_f32(eval_unary_f32(x.to_f32(), op))),
            Self::BF16(v) => run_unary!(BF16, v, |x: bf16| bf16::from_f32(eval_unary_f32(x.to_f32(), op))),
            Self::F8E4M3(v) => run_unary!(F8E4M3, v, |x: f8e4m3| f8e4m3::from_f32(eval_unary_f32(x.to_f32(), op))),
            Self::F8E5M2(v) => run_unary!(F8E5M2, v, |x: f8e5m2| f8e5m2::from_f32(eval_unary_f32(x.to_f32(), op))),
            Self::I32(v) => run_unary!(I32, v, |x: i32| eval_unary_f64(x as f64, op) as i64 as i32),
            Self::I64(v) => run_unary!(I64, v, |x: i64| eval_unary_f64(x as f64, op) as i64),
            Self::U8(v) => run_unary!(U8, v, |x: u8| eval_unary_f64(x as f64, op) as i64 as u8),
            Self::I8(v) => run_unary!(I8, v, |x: i8| eval_unary_f64(x as f64, op) as i64 as i8),
            Self::I16(v) => run_unary!(I16, v, |x: i16| eval_unary_f64(x as f64, op) as i64 as i16),
            Self::U16(v) => run_unary!(U16, v, |x: u16| eval_unary_f64(x as f64, op) as i64 as u16),
            Self::U32(v) => run_unary!(U32, v, |x: u32| eval_unary_f64(x as f64, op) as i64 as u32),
            Self::U64(v) => run_unary!(U64, v, |x: u64| eval_unary_f64(x as f64, op) as i64 as u64),
            Self::Bool(_) => Err(Error::UnsupportedDType {
                op: "unary_op",
                dtype: DType::Bool,
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Binary operations (assuming `lhs_layout` and `rhs_layout` share shape)
    // -----------------------------------------------------------------------
    pub fn binary_op(
        &self,
        lhs_l: &Layout,
        rhs: &Self,
        rhs_l: &Layout,
        op: BinaryOp,
    ) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                op: "binary_op",
                expected: self.dtype(),
                got: rhs.dtype(),
            });
        }
        if lhs_l.shape() != rhs_l.shape() {
            return Err(Error::ShapeMismatch {
                op: "binary_op",
                expected: lhs_l.shape().to_vec(),
                got: rhs_l.shape().to_vec(),
            });
        }
        // Boolean div/pow have no meaning; returning lhs silently hid type errors.
        if matches!(self, Self::Bool(_)) && matches!(op, BinaryOp::Div | BinaryOp::Pow | BinaryOp::Fmod | BinaryOp::Remainder | BinaryOp::FloorDiv) {
            return Err(Error::Msg("boolean div, pow and remainders are not supported".into()));
        }
        // Integer division by zero is an error, not a panic. Only denominators
        // inside the (pre-broadcast) view count. MIN/-1 keeps wrapping
        // semantics via wrapping_div below (the wrapped quotient is MIN).
        if matches!(op, BinaryOp::Div | BinaryOp::Fmod | BinaryOp::Remainder | BinaryOp::FloorDiv) && matches!(self, Self::I32(_) | Self::I64(_) | Self::U8(_) | Self::I8(_) | Self::I16(_) | Self::U16(_) | Self::U32(_) | Self::U64(_))
        {
            let denoms = rhs.to_vec::<i64>(rhs_l);
            if denoms.contains(&0) {
                return Err(Error::Msg("integer division by zero".into()));
            }
        }
        // Integer pow is exact: negative exponents have no integer value and
        // silently saturated/wrapped huge powers hid bugs, so both are errors.
        if matches!(op, BinaryOp::Pow) && matches!(self, Self::I32(_) | Self::I64(_) | Self::U8(_) | Self::I8(_) | Self::I16(_) | Self::U16(_) | Self::U32(_) | Self::U64(_))
        {
            let bases = self.to_vec::<i64>(lhs_l);
            let exps = rhs.to_vec::<i64>(rhs_l);
            let (lo, hi) = match self {
                Self::I32(_) => (i32::MIN as i128, i32::MAX as i128),
                Self::I64(_) => (i64::MIN as i128, i64::MAX as i128),
                Self::U8(_) => (0, u8::MAX as i128),
                Self::I8(_) => (i8::MIN as i128, i8::MAX as i128),
                Self::I16(_) => (i16::MIN as i128, i16::MAX as i128),
                Self::U16(_) => (0, u16::MAX as i128),
                Self::U32(_) => (0, u32::MAX as i128),
                Self::U64(_) => (0, u64::MAX as i128),
                _ => unreachable!("guarded by the outer matches"),
            };
            for (&b, &e) in bases.iter().zip(exps.iter()) {
                if e < 0 {
                    return Err(Error::Msg(format!(
                        "integer pow with negative exponent ({b}^{e}) is not supported"
                    )));
                }
                match int_pow_exact(b, e) {
                    Some(v) if v >= lo && v <= hi => {}
                    _ => {
                        return Err(Error::Msg(format!(
                            "integer pow overflow ({b}^{e} exceeds the dtype range)"
                        )));
                    }
                }
            }
        }

        macro_rules! run_bin {
            ($variant:ident, $a:expr, $b:expr, $f:expr) => {{
                match (
                    lhs_l.contiguous_range_in($a.len()),
                    rhs_l.contiguous_range_in($b.len()),
                ) {
                    (Some(ra), Some(rb)) => {
                        let sa = &$a[ra];
                        let sb = &$b[rb];
                        if let Some(mut out) = crate::pool::take(sa.len().min(sb.len())) {
                            #[cfg(feature = "rayon")]
                            out.par_iter_mut().zip(sa.par_iter().zip(sb.par_iter())).for_each(|(o, (&x, &y))| *o = ($f)(x, y));
                            #[cfg(not(feature = "rayon"))]
                            out.iter_mut().zip(sa.iter().zip(sb.iter())).for_each(|(o, (&x, &y))| *o = ($f)(x, y));
                            return Ok(Self::$variant(out));
                        }
                        #[cfg(feature = "rayon")]
                        if sa.len() >= 16384 {
                            return Ok(Self::$variant(
                                sa.par_iter()
                                    .zip(sb.par_iter())
                                    .map(|(&x, &y)| ($f)(x, y))
                                    .collect(),
                            ));
                        }
                        Ok(Self::$variant(
                            sa.iter()
                                .zip(sb.iter())
                                .map(|(&x, &y)| ($f)(x, y))
                                .collect(),
                        ))
                    }
                    _ => {
                        let shape = lhs_l.shape();
                        let total_n = shape.iter().product::<usize>();
                        // 1. Scalar broadcast RHS: RHS has all strides 0
                        if rhs_l.strides().iter().all(|&s| s == 0)
                            && lhs_l.contiguous_range_in($a.len()).is_some()
                        {
                            let ra = lhs_l.contiguous_range_in($a.len()).unwrap();
                            let b_val = $b[rhs_l.offset()];
                            let sa = &$a[ra];
                            if let Some(mut out) = crate::pool::take(sa.len()) {
                                #[cfg(feature = "rayon")]
                                out.par_iter_mut().zip(sa.par_iter()).for_each(|(o, &x)| *o = ($f)(x, b_val));
                                #[cfg(not(feature = "rayon"))]
                                out.iter_mut().zip(sa.iter()).for_each(|(o, &x)| *o = ($f)(x, b_val));
                                return Ok(Self::$variant(out));
                            }
                            #[cfg(feature = "rayon")]
                            if sa.len() >= 16384 {
                                return Ok(Self::$variant(
                                    sa.par_iter().map(|&x| ($f)(x, b_val)).collect(),
                                ));
                            }
                            return Ok(Self::$variant(
                                sa.iter().map(|&x| ($f)(x, b_val)).collect(),
                            ));
                        }
                        // 2. Scalar broadcast LHS: LHS has all strides 0
                        if lhs_l.strides().iter().all(|&s| s == 0)
                            && rhs_l.contiguous_range_in($b.len()).is_some()
                        {
                            let rb = rhs_l.contiguous_range_in($b.len()).unwrap();
                            let a_val = $a[lhs_l.offset()];
                            let sb = &$b[rb];
                            if let Some(mut out) = crate::pool::take(sb.len()) {
                                #[cfg(feature = "rayon")]
                                out.par_iter_mut().zip(sb.par_iter()).for_each(|(o, &y)| *o = ($f)(a_val, y));
                                #[cfg(not(feature = "rayon"))]
                                out.iter_mut().zip(sb.iter()).for_each(|(o, &y)| *o = ($f)(a_val, y));
                                return Ok(Self::$variant(out));
                            }
                            #[cfg(feature = "rayon")]
                            if sb.len() >= 16384 {
                                return Ok(Self::$variant(
                                    sb.par_iter().map(|&y| ($f)(a_val, y)).collect(),
                                ));
                            }
                            return Ok(Self::$variant(
                                sb.iter().map(|&y| ($f)(a_val, y)).collect(),
                            ));
                        }
                        // Any other pair of strided / broadcast layouts: one tight loop per row of the last dimension
                        if let Some(out) = strided_binary(&$a[..], lhs_l, &$b[..], rhs_l, $f) {
                            return Ok(Self::$variant(out));
                        }
                        // 3. Trailing row broadcast RHS: [.., cols] where RHS has stride 1 on last dim and 0 on preceding dims
                        if !shape.is_empty()
                            && rhs_l.strides().last() == Some(&1)
                            && rhs_l.strides()[..shape.len() - 1].iter().all(|&s| s == 0)
                            && lhs_l.contiguous_range_in($a.len()).is_some()
                        {
                            let ra = lhs_l.contiguous_range_in($a.len()).unwrap();
                            let cols = shape[shape.len() - 1];
                            if cols > 0 && rhs_l.offset() + cols <= $b.len() {
                                let mut out = vec![Default::default(); total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                        let b_row = &$b[rhs_l.offset()..rhs_l.offset() + cols];
                                        for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                            *o = ($f)(x, y);
                                        }
                                    });
                                    return Ok(Self::$variant(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                    let b_row = &$b[rhs_l.offset()..rhs_l.offset() + cols];
                                    for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                        *o = ($f)(x, y);
                                    }
                                }
                                return Ok(Self::$variant(out));
                            }
                        }
                        // 4. Trailing row broadcast LHS: symmetric case
                        if !shape.is_empty()
                            && lhs_l.strides().last() == Some(&1)
                            && lhs_l.strides()[..shape.len() - 1].iter().all(|&s| s == 0)
                            && rhs_l.contiguous_range_in($b.len()).is_some()
                        {
                            let rb = rhs_l.contiguous_range_in($b.len()).unwrap();
                            let cols = shape[shape.len() - 1];
                            if cols > 0 && lhs_l.offset() + cols <= $a.len() {
                                let mut out = vec![Default::default(); total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let a_row = &$a[lhs_l.offset()..lhs_l.offset() + cols];
                                        let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                        for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                            *o = ($f)(x, y);
                                        }
                                    });
                                    return Ok(Self::$variant(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let a_row = &$a[lhs_l.offset()..lhs_l.offset() + cols];
                                    let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                    for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                        *o = ($f)(x, y);
                                    }
                                }
                                return Ok(Self::$variant(out));
                            }
                        }
                        // 5. Column broadcast RHS: 2D [rows, cols] where RHS has stride 0 on the trailing dim
                        if shape.len() == 2
                            && rhs_l.strides()[1] == 0
                            && lhs_l.contiguous_range_in($a.len()).is_some()
                        {
                            let ra = lhs_l.contiguous_range_in($a.len()).unwrap();
                            let rows = shape[0];
                            let cols = shape[1];
                            let s0 = rhs_l.strides()[0];
                            let min_b = rhs_l.offset() as isize + (if s0 < 0 { (rows as isize - 1) * s0 } else { 0 });
                            let max_b = rhs_l.offset() as isize + (if s0 > 0 { (rows as isize - 1) * s0 } else { 0 });
                            if min_b >= 0 && (max_b as usize) < $b.len() {
                                let mut out = vec![Default::default(); total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let b_idx = (rhs_l.offset() as isize + r as isize * s0) as usize;
                                        let b_val = $b[b_idx];
                                        let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                        for (o, &x) in out_row.iter_mut().zip(a_row) {
                                            *o = ($f)(x, b_val);
                                        }
                                    });
                                    return Ok(Self::$variant(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let b_idx = (rhs_l.offset() as isize + r as isize * s0) as usize;
                                    let b_val = $b[b_idx];
                                    let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                    for (o, &x) in out_row.iter_mut().zip(a_row) {
                                        *o = ($f)(x, b_val);
                                    }
                                }
                                return Ok(Self::$variant(out));
                            }
                        }
                        // 6. Column broadcast LHS: symmetric case
                        if shape.len() == 2
                            && lhs_l.strides()[1] == 0
                            && rhs_l.contiguous_range_in($b.len()).is_some()
                        {
                            let rb = rhs_l.contiguous_range_in($b.len()).unwrap();
                            let rows = shape[0];
                            let cols = shape[1];
                            let s0 = lhs_l.strides()[0];
                            let min_a = lhs_l.offset() as isize + (if s0 < 0 { (rows as isize - 1) * s0 } else { 0 });
                            let max_a = lhs_l.offset() as isize + (if s0 > 0 { (rows as isize - 1) * s0 } else { 0 });
                            if min_a >= 0 && (max_a as usize) < $a.len() {
                                let mut out = vec![Default::default(); total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let a_idx = (lhs_l.offset() as isize + r as isize * s0) as usize;
                                        let a_val = $a[a_idx];
                                        let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                        for (o, &y) in out_row.iter_mut().zip(b_row) {
                                            *o = ($f)(a_val, y);
                                        }
                                    });
                                    return Ok(Self::$variant(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let a_idx = (lhs_l.offset() as isize + r as isize * s0) as usize;
                                    let a_val = $a[a_idx];
                                    let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                    for (o, &y) in out_row.iter_mut().zip(b_row) {
                                        *o = ($f)(a_val, y);
                                    }
                                }
                                return Ok(Self::$variant(out));
                            }
                        }
                        // General fallback
                        Ok(Self::$variant(
                            lhs_l
                                .storage_offsets()
                                .zip(rhs_l.storage_offsets())
                                .map(|(ia, ib)| ($f)($a[ia], $b[ib]))
                                .collect(),
                        ))
                    }
                }
            }};
        }

        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => {
                if let (Some(ra), Some(rb)) = (lhs_l.contiguous_range_in(a.len()), rhs_l.contiguous_range_in(b.len())) {
                    return Ok(Self::F32(f32_binary_contiguous(&a[ra], &b[rb], op)));
                }
                if let Some(out) = f32_row_broadcast(a, lhs_l, b, rhs_l, op) {
                    return Ok(Self::F32(out));
                }
                run_bin!(F32, a, b, |x: f32, y: f32| eval_bin_f32(x, y, op))
            }
            (Self::F64(a), Self::F64(b)) => {
                run_bin!(F64, a, b, |x: f64, y: f64| eval_bin_f64(x, y, op))
            }
            (Self::C64(a), Self::C64(b)) => {
                run_bin!(C64, a, b, |x: [f32; 2], y: [f32; 2]| eval_bin_c64(x, y, op))
            }
            (Self::C128(a), Self::C128(b)) => {
                run_bin!(C128, a, b, |x: [f64; 2], y: [f64; 2]| eval_bin_c128(x, y, op))
            }
            (Self::F16(a), Self::F16(b)) => run_bin!(F16, a, b, |x: f16, y: f16| {
                f16::from_f32(eval_bin_f32(x.to_f32(), y.to_f32(), op))
            }),
            (Self::BF16(a), Self::BF16(b)) => run_bin!(BF16, a, b, |x: bf16, y: bf16| {
                bf16::from_f32(eval_bin_f32(x.to_f32(), y.to_f32(), op))
            }),
            (Self::F8E4M3(a), Self::F8E4M3(b)) => run_bin!(F8E4M3, a, b, |x: f8e4m3, y: f8e4m3| {
                f8e4m3::from_f32(eval_bin_f32(x.to_f32(), y.to_f32(), op))
            }),
            (Self::F8E5M2(a), Self::F8E5M2(b)) => run_bin!(F8E5M2, a, b, |x: f8e5m2, y: f8e5m2| {
                f8e5m2::from_f32(eval_bin_f32(x.to_f32(), y.to_f32(), op))
            }),
            (Self::I32(a), Self::I32(b)) => run_bin!(I32, a, b, |x: i32, y: i32| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x as i64, y as i64) as i32,
                BinaryOp::Fmod => x.wrapping_rem(y),
                BinaryOp::Remainder => int_remainder_i64(i64::from(x), i64::from(y)) as i32,
                BinaryOp::FloorDiv => int_floor_div_i64(i64::from(x), i64::from(y)) as i32,
            }),
            (Self::I64(a), Self::I64(b)) => run_bin!(I64, a, b, |x: i64, y: i64| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x, y),
                BinaryOp::Fmod => x.wrapping_rem(y),
                BinaryOp::Remainder => int_remainder_i64(x, y),
                BinaryOp::FloorDiv => int_floor_div_i64(x, y),
            }),
            (Self::U8(a), Self::U8(b)) => run_bin!(U8, a, b, |x: u8, y: u8| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x as i64, y as i64) as u8,
                BinaryOp::Fmod | BinaryOp::Remainder => x % y,
                BinaryOp::FloorDiv => x / y,
            }),
            (Self::I8(a), Self::I8(b)) => run_bin!(I8, a, b, |x: i8, y: i8| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x as i64, y as i64) as i8,
                BinaryOp::Fmod | BinaryOp::Remainder => x % y,
                BinaryOp::FloorDiv => x / y,
            }),
            (Self::I16(a), Self::I16(b)) => run_bin!(I16, a, b, |x: i16, y: i16| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x as i64, y as i64) as i16,
                BinaryOp::Fmod | BinaryOp::Remainder => x % y,
                BinaryOp::FloorDiv => x / y,
            }),
            (Self::U16(a), Self::U16(b)) => run_bin!(U16, a, b, |x: u16, y: u16| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x as i64, y as i64) as u16,
                BinaryOp::Fmod | BinaryOp::Remainder => x % y,
                BinaryOp::FloorDiv => x / y,
            }),
            (Self::U32(a), Self::U32(b)) => run_bin!(U32, a, b, |x: u32, y: u32| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x as i64, y as i64) as u32,
                BinaryOp::Fmod | BinaryOp::Remainder => x % y,
                BinaryOp::FloorDiv => x / y,
            }),
            (Self::U64(a), Self::U64(b)) => run_bin!(U64, a, b, |x: u64, y: u64| match op {
                BinaryOp::Add => x.wrapping_add(y),
                BinaryOp::Sub => x.wrapping_sub(y),
                BinaryOp::Mul => x.wrapping_mul(y),
                BinaryOp::Div => x.wrapping_div(y),
                BinaryOp::Maximum => x.max(y),
                BinaryOp::Minimum => x.min(y),
                BinaryOp::Pow => int_pow_kernel(x as i64, y as i64) as u64,
                BinaryOp::Fmod | BinaryOp::Remainder => x % y,
                BinaryOp::FloorDiv => x / y,
            }),
            (Self::Bool(a), Self::Bool(b)) => run_bin!(Bool, a, b, |x: bool, y: bool| match op {
                BinaryOp::Add | BinaryOp::Maximum => x || y,
                BinaryOp::Mul | BinaryOp::Minimum => x && y,
                BinaryOp::Sub => x && !y,
                _ => x,
            }),
            _ => unreachable!(),
        }
    }

    pub fn cmp_op(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout, op: CmpOp) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                op: "cmp_op",
                expected: self.dtype(),
                got: rhs.dtype(),
            });
        }
        // Like binary_op, callers pre-broadcast; unequal shapes would zip-truncate.
        if lhs_l.shape() != rhs_l.shape() {
            return Err(Error::ShapeMismatch {
                op: "cmp_op",
                expected: lhs_l.shape().to_vec(),
                got: rhs_l.shape().to_vec(),
            });
        }

        macro_rules! run_cmp {
            ($a:expr, $b:expr, $cmp_fn:expr) => {{
                match (
                    lhs_l.contiguous_range_in($a.len()),
                    rhs_l.contiguous_range_in($b.len()),
                ) {
                    (Some(ra), Some(rb)) => {
                        let sa = &$a[ra];
                        let sb = &$b[rb];
                        #[cfg(feature = "rayon")]
                        if sa.len() >= 16384 {
                            return Ok(Self::Bool(
                                sa.par_iter()
                                    .zip(sb.par_iter())
                                    .map(|(&x, &y)| ($cmp_fn)(x, y))
                                    .collect(),
                            ));
                        }
                        Ok(Self::Bool(
                            sa.iter()
                                .zip(sb.iter())
                                .map(|(&x, &y)| ($cmp_fn)(x, y))
                                .collect(),
                        ))
                    }
                    _ => {
                        let shape = lhs_l.shape();
                        let total_n = shape.iter().product::<usize>();
                        // 1. Scalar broadcast RHS
                        if rhs_l.strides().iter().all(|&s| s == 0)
                            && lhs_l.contiguous_range_in($a.len()).is_some()
                        {
                            let ra = lhs_l.contiguous_range_in($a.len()).unwrap();
                            let b_val = $b[rhs_l.offset()];
                            let sa = &$a[ra];
                            #[cfg(feature = "rayon")]
                            if sa.len() >= 16384 {
                                return Ok(Self::Bool(
                                    sa.par_iter().map(|&x| ($cmp_fn)(x, b_val)).collect(),
                                ));
                            }
                            return Ok(Self::Bool(
                                sa.iter().map(|&x| ($cmp_fn)(x, b_val)).collect(),
                            ));
                        }
                        // 2. Scalar broadcast LHS
                        if lhs_l.strides().iter().all(|&s| s == 0)
                            && rhs_l.contiguous_range_in($b.len()).is_some()
                        {
                            let rb = rhs_l.contiguous_range_in($b.len()).unwrap();
                            let a_val = $a[lhs_l.offset()];
                            let sb = &$b[rb];
                            #[cfg(feature = "rayon")]
                            if sb.len() >= 16384 {
                                return Ok(Self::Bool(
                                    sb.par_iter().map(|&y| ($cmp_fn)(a_val, y)).collect(),
                                ));
                            }
                            return Ok(Self::Bool(
                                sb.iter().map(|&y| ($cmp_fn)(a_val, y)).collect(),
                            ));
                        }
                        // 3. Trailing row broadcast RHS
                        if !shape.is_empty()
                            && rhs_l.strides().last() == Some(&1)
                            && rhs_l.strides()[..shape.len() - 1].iter().all(|&s| s == 0)
                            && lhs_l.contiguous_range_in($a.len()).is_some()
                        {
                            let ra = lhs_l.contiguous_range_in($a.len()).unwrap();
                            let cols = shape[shape.len() - 1];
                            if cols > 0 && rhs_l.offset() + cols <= $b.len() {
                                let mut out = vec![false; total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                        let b_row = &$b[rhs_l.offset()..rhs_l.offset() + cols];
                                        for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                            *o = ($cmp_fn)(x, y);
                                        }
                                    });
                                    return Ok(Self::Bool(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                    let b_row = &$b[rhs_l.offset()..rhs_l.offset() + cols];
                                    for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                        *o = ($cmp_fn)(x, y);
                                    }
                                }
                                return Ok(Self::Bool(out));
                            }
                        }
                        // 4. Trailing row broadcast LHS
                        if !shape.is_empty()
                            && lhs_l.strides().last() == Some(&1)
                            && lhs_l.strides()[..shape.len() - 1].iter().all(|&s| s == 0)
                            && rhs_l.contiguous_range_in($b.len()).is_some()
                        {
                            let rb = rhs_l.contiguous_range_in($b.len()).unwrap();
                            let cols = shape[shape.len() - 1];
                            if cols > 0 && lhs_l.offset() + cols <= $a.len() {
                                let mut out = vec![false; total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let a_row = &$a[lhs_l.offset()..lhs_l.offset() + cols];
                                        let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                        for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                            *o = ($cmp_fn)(x, y);
                                        }
                                    });
                                    return Ok(Self::Bool(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let a_row = &$a[lhs_l.offset()..lhs_l.offset() + cols];
                                    let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                    for ((o, &x), &y) in out_row.iter_mut().zip(a_row).zip(b_row) {
                                        *o = ($cmp_fn)(x, y);
                                    }
                                }
                                return Ok(Self::Bool(out));
                            }
                        }
                        // 5. Column broadcast RHS
                        if shape.len() == 2
                            && rhs_l.strides()[1] == 0
                            && lhs_l.contiguous_range_in($a.len()).is_some()
                        {
                            let ra = lhs_l.contiguous_range_in($a.len()).unwrap();
                            let rows = shape[0];
                            let cols = shape[1];
                            let s0 = rhs_l.strides()[0];
                            let min_b = rhs_l.offset() as isize + (if s0 < 0 { (rows as isize - 1) * s0 } else { 0 });
                            let max_b = rhs_l.offset() as isize + (if s0 > 0 { (rows as isize - 1) * s0 } else { 0 });
                            if min_b >= 0 && (max_b as usize) < $b.len() {
                                let mut out = vec![false; total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let b_idx = (rhs_l.offset() as isize + r as isize * s0) as usize;
                                        let b_val = $b[b_idx];
                                        let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                        for (o, &x) in out_row.iter_mut().zip(a_row) {
                                            *o = ($cmp_fn)(x, b_val);
                                        }
                                    });
                                    return Ok(Self::Bool(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let b_idx = (rhs_l.offset() as isize + r as isize * s0) as usize;
                                    let b_val = $b[b_idx];
                                    let a_row = &$a[ra.start + r * cols..ra.start + (r + 1) * cols];
                                    for (o, &x) in out_row.iter_mut().zip(a_row) {
                                        *o = ($cmp_fn)(x, b_val);
                                    }
                                }
                                return Ok(Self::Bool(out));
                            }
                        }
                        // 6. Column broadcast LHS
                        if shape.len() == 2
                            && lhs_l.strides()[1] == 0
                            && rhs_l.contiguous_range_in($b.len()).is_some()
                        {
                            let rb = rhs_l.contiguous_range_in($b.len()).unwrap();
                            let rows = shape[0];
                            let cols = shape[1];
                            let s0 = lhs_l.strides()[0];
                            let min_a = lhs_l.offset() as isize + (if s0 < 0 { (rows as isize - 1) * s0 } else { 0 });
                            let max_a = lhs_l.offset() as isize + (if s0 > 0 { (rows as isize - 1) * s0 } else { 0 });
                            if min_a >= 0 && (max_a as usize) < $a.len() {
                                let mut out = vec![false; total_n];
                                #[cfg(feature = "rayon")]
                                if total_n >= 16384 {
                                    out.par_chunks_mut(cols).enumerate().for_each(|(r, out_row)| {
                                        let a_idx = (lhs_l.offset() as isize + r as isize * s0) as usize;
                                        let a_val = $a[a_idx];
                                        let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                        for (o, &y) in out_row.iter_mut().zip(b_row) {
                                            *o = ($cmp_fn)(a_val, y);
                                        }
                                    });
                                    return Ok(Self::Bool(out));
                                }
                                for (r, out_row) in out.chunks_mut(cols).enumerate() {
                                    let a_idx = (lhs_l.offset() as isize + r as isize * s0) as usize;
                                    let a_val = $a[a_idx];
                                    let b_row = &$b[rb.start + r * cols..rb.start + (r + 1) * cols];
                                    for (o, &y) in out_row.iter_mut().zip(b_row) {
                                        *o = ($cmp_fn)(a_val, y);
                                    }
                                }
                                return Ok(Self::Bool(out));
                            }
                        }
                        // General fallback
                        Ok(Self::Bool(
                            lhs_l
                                .storage_offsets()
                                .zip(rhs_l.storage_offsets())
                                .map(|(ia, ib)| ($cmp_fn)($a[ia], $b[ib]))
                                .collect(),
                        ))
                    }
                }
            }};
        }

        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => {
                run_cmp!(a, b, |x: f32, y: f32| eval_cmp(x, y, op))
            }
            (Self::F64(a), Self::F64(b)) => {
                run_cmp!(a, b, |x: f64, y: f64| eval_cmp(x, y, op))
            }
            (Self::C64(a), Self::C64(b)) if matches!(op, CmpOp::Eq | CmpOp::Ne) => {
                let eq = op == CmpOp::Eq;
                run_cmp!(a, b, |x: [f32; 2], y: [f32; 2]| (x == y) == eq)
            }
            (Self::C128(a), Self::C128(b)) if matches!(op, CmpOp::Eq | CmpOp::Ne) => {
                let eq = op == CmpOp::Eq;
                run_cmp!(a, b, |x: [f64; 2], y: [f64; 2]| (x == y) == eq)
            }
            (Self::I64(a), Self::I64(b)) => {
                run_cmp!(a, b, |x: i64, y: i64| eval_cmp(x, y, op))
            }
            (Self::I32(a), Self::I32(b)) => {
                run_cmp!(a, b, |x: i32, y: i32| eval_cmp(x, y, op))
            }
            (Self::U8(a), Self::U8(b)) => {
                run_cmp!(a, b, |x: u8, y: u8| eval_cmp(x, y, op))
            }
            (Self::I8(a), Self::I8(b)) => {
                run_cmp!(a, b, |x: i8, y: i8| eval_cmp(x, y, op))
            }
            (Self::I16(a), Self::I16(b)) => {
                run_cmp!(a, b, |x: i16, y: i16| eval_cmp(x, y, op))
            }
            (Self::U16(a), Self::U16(b)) => {
                run_cmp!(a, b, |x: u16, y: u16| eval_cmp(x, y, op))
            }
            (Self::U32(a), Self::U32(b)) => {
                run_cmp!(a, b, |x: u32, y: u32| eval_cmp(x, y, op))
            }
            (Self::U64(a), Self::U64(b)) => {
                run_cmp!(a, b, |x: u64, y: u64| eval_cmp(x, y, op))
            }
            (Self::Bool(a), Self::Bool(b)) => {
                run_cmp!(a, b, |x: bool, y: bool| eval_cmp(x, y, op))
            }
            (Self::F16(a), Self::F16(b)) => {
                run_cmp!(a, b, |x: f16, y: f16| eval_cmp(x.to_f32(), y.to_f32(), op))
            }
            (Self::BF16(a), Self::BF16(b)) => {
                run_cmp!(a, b, |x: bf16, y: bf16| eval_cmp(x.to_f32(), y.to_f32(), op))
            }
            (Self::F8E4M3(a), Self::F8E4M3(b)) => {
                run_cmp!(a, b, |x: f8e4m3, y: f8e4m3| eval_cmp(x.to_f32(), y.to_f32(), op))
            }
            (Self::F8E5M2(a), Self::F8E5M2(b)) => {
                run_cmp!(a, b, |x: f8e5m2, y: f8e5m2| eval_cmp(x.to_f32(), y.to_f32(), op))
            }
            _ => unreachable!(),
        }
    }

    pub fn bitwise_op(
        &self,
        lhs_l: &Layout,
        rhs: &Self,
        rhs_l: &Layout,
        op: BitOp,
    ) -> Result<Self> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                op: "bitwise_op",
                expected: self.dtype(),
                got: rhs.dtype(),
            });
        }
        // Like binary_op, callers pre-broadcast; unequal shapes would zip-truncate.
        if lhs_l.shape() != rhs_l.shape() {
            return Err(Error::ShapeMismatch {
                op: "bitwise_op",
                expected: lhs_l.shape().to_vec(),
                got: rhs_l.shape().to_vec(),
            });
        }
        macro_rules! bit_typed {
            ($variant:ident, $av:expr, $bv:expr) => {{
                let out = lhs_l
                    .storage_offsets()
                    .zip(rhs_l.storage_offsets())
                    .map(|(i, j)| {
                        let x = $av[i];
                        let y = $bv[j];
                        match op {
                            BitOp::And => x & y,
                            BitOp::Or => x | y,
                            BitOp::Xor => x ^ y,
                        }
                    })
                    .collect();
                Ok(Self::$variant(out))
            }};
        }
        // Native-width ops: no f64 roundtrip, so i64 bit patterns stay exact.
        // Floats are rejected (torch raises `NotImplementedError` there too).
        match (self, rhs) {
            (Self::Bool(a), Self::Bool(b)) => bit_typed!(Bool, a, b),
            (Self::U8(a), Self::U8(b)) => bit_typed!(U8, a, b),
            (Self::I8(a), Self::I8(b)) => bit_typed!(I8, a, b),
            (Self::I16(a), Self::I16(b)) => bit_typed!(I16, a, b),
            (Self::U16(a), Self::U16(b)) => bit_typed!(U16, a, b),
            (Self::U32(a), Self::U32(b)) => bit_typed!(U32, a, b),
            (Self::U64(a), Self::U64(b)) => bit_typed!(U64, a, b),
            (Self::I32(a), Self::I32(b)) => bit_typed!(I32, a, b),
            (Self::I64(a), Self::I64(b)) => bit_typed!(I64, a, b),
            _ => Err(Error::UnsupportedDType {
                op: "bitwise_op",
                dtype: self.dtype(),
            }),
        }
    }

    pub fn logical_op(
        &self,
        lhs_l: &Layout,
        rhs: &Self,
        rhs_l: &Layout,
        op: LogicOp,
    ) -> Result<Self> {
        // Unlike value ops (add/cmp/bitwise), mixed dtypes are fine here: each
        // side reduces to truthiness independently, so no promotion rule is
        // needed. Torch allows mixed logical operands too.
        if lhs_l.shape() != rhs_l.shape() {
            return Err(Error::ShapeMismatch {
                op: "logical_op",
                expected: lhs_l.shape().to_vec(),
                got: rhs_l.shape().to_vec(),
            });
        }
        // Truthiness is `!= 0` on every dtype (NaN is truthy, like torch).
        let a = self.to_vec::<bool>(lhs_l);
        let b = rhs.to_vec::<bool>(rhs_l);
        let out: Vec<bool> = a
            .into_iter()
            .zip(b)
            .map(|(x, y)| match op {
                LogicOp::And => x && y,
                LogicOp::Or => x || y,
                LogicOp::Xor => x ^ y,
            })
            .collect();
        Ok(Self::Bool(out))
    }

    pub fn pred_op(&self, layout: &Layout, op: PredOp) -> Result<Self> {
        // Torch defines these on every dtype: integers/bools are never NaN or
        // infinite and always finite.
        if !matches!(
            self.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) {
            let n: usize = layout.shape().iter().product();
            return Ok(Self::Bool(vec![matches!(op, PredOp::IsFinite); n]));
        }
        // f16/f32 widen to f64 losslessly, preserving NaN/inf exactly.
        let out: Vec<bool> = self
            .to_vec::<f64>(layout)
            .into_iter()
            .map(|x| match op {
                PredOp::IsNan => x.is_nan(),
                PredOp::IsInf => x.is_infinite(),
                PredOp::IsFinite => x.is_finite(),
            })
            .collect();
        Ok(Self::Bool(out))
    }

    pub fn bitwise_not(&self, layout: &Layout) -> Result<Self> {
        macro_rules! not_typed {
            ($variant:ident, $v:expr) => {{
                Ok(Self::$variant(
                    layout.storage_offsets().map(|i| !$v[i]).collect(),
                ))
            }};
        }
        match self {
            Self::Bool(v) => not_typed!(Bool, v),
            Self::U8(v) => not_typed!(U8, v),
            Self::I8(v) => not_typed!(I8, v),
            Self::I16(v) => not_typed!(I16, v),
            Self::U16(v) => not_typed!(U16, v),
            Self::U32(v) => not_typed!(U32, v),
            Self::U64(v) => not_typed!(U64, v),
            Self::I32(v) => not_typed!(I32, v),
            Self::I64(v) => not_typed!(I64, v),
            _ => Err(Error::UnsupportedDType {
                op: "bitwise_not",
                dtype: self.dtype(),
            }),
        }
    }

    /// Index validation for the index_* family: negatives wrap (deliberate
    /// divergence — torch rejects negatives in these four ops, but wrapping
    /// matches take/advanced indexing and is pinned by three `ops::tests`);
    /// still-OOB indices error.
    pub(crate) fn check_index(op: &'static str, raw: i64, dim: usize) -> Result<usize> {
        let norm = if raw < 0 { dim as i64 + raw } else { raw };
        if norm < 0 || norm as usize >= dim {
            return Err(Error::IndexOutOfBounds {
                op,
                index: raw as isize,
                dim_size: dim,
            });
        }
        Ok(norm as usize)
    }

    /// Typed (exact-width, no f64 roundtrip) `index_select` gather core.
    pub fn index_select_vals(&self, layout: &Layout, ax: usize, idx: &[i64]) -> Result<Self> {
        const OP: &str = "index_select";
        let shape = layout.shape();
        let dim = shape[ax];
        let outer: usize = shape[..ax].iter().product();
        let inner: usize = shape[ax + 1..].iter().product();
        let norm: Vec<usize> = idx
            .iter()
            .map(|&r| Self::check_index(OP, r, dim))
            .collect::<Result<_>>()?;
        macro_rules! select_typed {
            ($variant:ident, $t:ty) => {{
                // read the source in place when it is contiguous; only strided views are gathered first
                let Self::$variant(v) = self else { unreachable!() };
                let owned;
                let src: &[$t] = match layout.contiguous_range_in(v.len()) {
                    Some(r) => &v[r],
                    None => {
                        owned = self.to_vec::<$t>(layout);
                        &owned
                    }
                };
                let n = outer * norm.len() * inner;
                let mut out: Vec<$t> = crate::pool::take(n).unwrap_or_default();
                if out.len() != n {
                    out = vec![Default::default(); n];
                }
                if inner > 0 {
                    // slice `k` of the output is slice `norm[k % idx]` of outer block `k / idx` of the source
                    let per = norm.len();
                    let copy = |(k, dst): (usize, &mut [$t])| {
                        let start = ((k / per) * dim + norm[k % per]) * inner;
                        dst.copy_from_slice(&src[start..start + inner]);
                    };
                    #[cfg(feature = "rayon")]
                    if n >= 65536 {
                        let group = (8 * EW_CHUNK / inner).max(1);
                        out.par_chunks_mut(inner * group).enumerate().for_each(|(gi, block)| block.chunks_mut(inner).enumerate().for_each(|(j, d)| copy((gi * group + j, d))));
                    } else {
                        out.chunks_mut(inner).enumerate().for_each(copy);
                    }
                    #[cfg(not(feature = "rayon"))]
                    out.chunks_mut(inner).enumerate().for_each(copy);
                }
                Ok(Self::$variant(out))
            }};
        }
        match self {
            Self::Bool(_) => select_typed!(Bool, bool),
            Self::U8(_) => select_typed!(U8, u8),
            Self::I8(_) => select_typed!(I8, i8),
            Self::I16(_) => select_typed!(I16, i16),
            Self::U16(_) => select_typed!(U16, u16),
            Self::U32(_) => select_typed!(U32, u32),
            Self::U64(_) => select_typed!(U64, u64),
            Self::I32(_) => select_typed!(I32, i32),
            Self::I64(_) => select_typed!(I64, i64),
            Self::F16(_) => select_typed!(F16, f16),
            Self::BF16(_) => select_typed!(BF16, bf16),
            Self::F8E4M3(_) => select_typed!(F8E4M3, f8e4m3),
            Self::F8E5M2(_) => select_typed!(F8E5M2, f8e5m2),
            Self::F32(_) => select_typed!(F32, f32),
            Self::F64(_) => select_typed!(F64, f64),
            Self::C64(_) => {
                let src = self.to_contiguous(layout)?;
                let Self::C64(src) = src else { unreachable!() };
                let mut out = Vec::with_capacity(outer * norm.len() * inner);
                for o in 0..outer {
                    for &i in &norm {
                        let start = (o * dim + i) * inner;
                        out.extend_from_slice(&src[start..start + inner]);
                    }
                }
                Ok(Self::C64(out))
            }
            Self::C128(_) => {
                let src = self.to_contiguous(layout)?;
                let Self::C128(src) = src else { unreachable!() };
                let mut out = Vec::with_capacity(outer * norm.len() * inner);
                for o in 0..outer {
                    for &i in &norm {
                        let start = (o * dim + i) * inner;
                        out.extend_from_slice(&src[start..start + inner]);
                    }
                }
                Ok(Self::C128(out))
            }
        }
    }

    /// Typed `index_add` core: same-dtype sources, wrapping integer adds and
    /// OR-combined bools (torch semantics, mirroring `view_scatter`).
    pub fn index_add_vals(
        &self,
        layout: &Layout,
        other: &Self,
        other_layout: &Layout,
        ax: usize,
        idx: &[i64],
    ) -> Result<Self> {
        const OP: &str = "index_add";
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                op: OP,
                expected: self.dtype(),
                got: other.dtype(),
            });
        }
        let dim = layout.shape()[ax];
        let outer: usize = layout.shape()[..ax].iter().product();
        let inner: usize = layout.shape()[ax + 1..].iter().product();
        let norm: Vec<usize> = idx
            .iter()
            .map(|&r| Self::check_index(OP, r, dim))
            .collect::<Result<_>>()?;
        macro_rules! add_typed {
            ($variant:ident, $t:ty, $add:expr) => {{
                let mut dst = self.to_vec::<$t>(layout);
                let src = other.to_vec::<$t>(other_layout);
                let add: fn(&mut $t, $t) = $add;
                for o in 0..outer {
                    for (k, &i) in norm.iter().enumerate() {
                        let dst_start = (o * dim + i) * inner;
                        let src_start = (o * norm.len() + k) * inner;
                        for j in 0..inner {
                            add(&mut dst[dst_start + j], src[src_start + j]);
                        }
                    }
                }
                Ok(Self::$variant(dst))
            }};
        }
        match (self, other) {
            (Self::Bool(_), Self::Bool(_)) => {
                add_typed!(Bool, bool, |a: &mut bool, b: bool| *a |= b)
            }
            (Self::U8(_), Self::U8(_)) => {
                add_typed!(U8, u8, |a: &mut u8, b: u8| *a = a.wrapping_add(b))
            }
            (Self::I8(_), Self::I8(_)) => {
                add_typed!(I8, i8, |a: &mut i8, b: i8| *a = a.wrapping_add(b))
            }
            (Self::I16(_), Self::I16(_)) => {
                add_typed!(I16, i16, |a: &mut i16, b: i16| *a = a.wrapping_add(b))
            }
            (Self::U16(_), Self::U16(_)) => {
                add_typed!(U16, u16, |a: &mut u16, b: u16| *a = a.wrapping_add(b))
            }
            (Self::U32(_), Self::U32(_)) => {
                add_typed!(U32, u32, |a: &mut u32, b: u32| *a = a.wrapping_add(b))
            }
            (Self::U64(_), Self::U64(_)) => {
                add_typed!(U64, u64, |a: &mut u64, b: u64| *a = a.wrapping_add(b))
            }
            (Self::I32(_), Self::I32(_)) => {
                add_typed!(I32, i32, |a: &mut i32, b: i32| *a = a.wrapping_add(b))
            }
            (Self::I64(_), Self::I64(_)) => {
                add_typed!(I64, i64, |a: &mut i64, b: i64| *a = a.wrapping_add(b))
            }
            (Self::F16(_), Self::F16(_)) => add_typed!(F16, f16, |a: &mut f16, b: f16| {
                *a = f16::from_f32(a.to_f32() + b.to_f32())
            }),
            (Self::BF16(_), Self::BF16(_)) => {
                add_typed!(BF16, bf16, |a: &mut bf16, b: bf16| {
                    *a = bf16::from_f32(a.to_f32() + b.to_f32())
                })
            }
            (Self::F8E4M3(_), Self::F8E4M3(_)) => {
                add_typed!(F8E4M3, f8e4m3, |a: &mut f8e4m3, b: f8e4m3| {
                    *a = f8e4m3::from_f32(a.to_f32() + b.to_f32())
                })
            }
            (Self::F8E5M2(_), Self::F8E5M2(_)) => {
                add_typed!(F8E5M2, f8e5m2, |a: &mut f8e5m2, b: f8e5m2| {
                    *a = f8e5m2::from_f32(a.to_f32() + b.to_f32())
                })
            }
            (Self::F32(_), Self::F32(_)) => {
                add_typed!(F32, f32, |a: &mut f32, b: f32| *a += b)
            }
            (Self::F64(_), Self::F64(_)) => {
                add_typed!(F64, f64, |a: &mut f64, b: f64| *a += b)
            }
            (Self::C64(_), Self::C64(_)) => {
                let Ok(Self::C64(mut dst)) = self.to_contiguous(layout) else { unreachable!() };
                let Ok(Self::C64(src)) = other.to_contiguous(other_layout) else { unreachable!() };
                for o in 0..outer {
                    for (k, &i) in norm.iter().enumerate() {
                        let dst_start = (o * dim + i) * inner;
                        let src_start = (o * norm.len() + k) * inner;
                        for j in 0..inner {
                            dst[dst_start + j][0] += src[src_start + j][0];
                            dst[dst_start + j][1] += src[src_start + j][1];
                        }
                    }
                }
                Ok(Self::C64(dst))
            }
            (Self::C128(_), Self::C128(_)) => {
                let Ok(Self::C128(mut dst)) = self.to_contiguous(layout) else { unreachable!() };
                let Ok(Self::C128(src)) = other.to_contiguous(other_layout) else { unreachable!() };
                for o in 0..outer {
                    for (k, &i) in norm.iter().enumerate() {
                        let dst_start = (o * dim + i) * inner;
                        let src_start = (o * norm.len() + k) * inner;
                        for j in 0..inner {
                            dst[dst_start + j][0] += src[src_start + j][0];
                            dst[dst_start + j][1] += src[src_start + j][1];
                        }
                    }
                }
                Ok(Self::C128(dst))
            }
            _ => Err(Error::DTypeMismatch {
                op: OP,
                expected: self.dtype(),
                got: other.dtype(),
            }),
        }
    }

    /// Typed same-rank `gather` core.
    pub fn gather_vals(
        &self,
        layout: &Layout,
        ax: usize,
        idx_shape: &[usize],
        idx: &[i64],
    ) -> Result<Self> {
        const OP: &str = "gather";
        let dim = layout.shape()[ax];
        let norm: Vec<usize> = idx
            .iter()
            .map(|&r| Self::check_index(OP, r, dim))
            .collect::<Result<_>>()?;
        macro_rules! gather_typed {
            ($variant:ident, $t:ty) => {{
                let src = self.to_vec::<$t>(layout);
                let strides = Layout::contiguous(layout.shape().to_vec())
                    .strides()
                    .to_vec();
                let mut out = Vec::with_capacity(norm.len());
                let mut coord = vec![0usize; layout.shape().len()];
                for &i in &norm {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    out.push(src[off as usize]);
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::$variant(out))
            }};
        }
        match self {
            Self::Bool(_) => gather_typed!(Bool, bool),
            Self::U8(_) => gather_typed!(U8, u8),
            Self::I8(_) => gather_typed!(I8, i8),
            Self::I16(_) => gather_typed!(I16, i16),
            Self::U16(_) => gather_typed!(U16, u16),
            Self::U32(_) => gather_typed!(U32, u32),
            Self::U64(_) => gather_typed!(U64, u64),
            Self::I32(_) => gather_typed!(I32, i32),
            Self::I64(_) => gather_typed!(I64, i64),
            Self::F16(_) => gather_typed!(F16, f16),
            Self::BF16(_) => gather_typed!(BF16, bf16),
            Self::F8E4M3(_) => gather_typed!(F8E4M3, f8e4m3),
            Self::F8E5M2(_) => gather_typed!(F8E5M2, f8e5m2),
            Self::F32(_) => gather_typed!(F32, f32),
            Self::F64(_) => gather_typed!(F64, f64),
            Self::C64(_) => {
                let Ok(Self::C64(src)) = self.to_contiguous(layout) else { unreachable!() };
                let strides = Layout::contiguous(layout.shape().to_vec()).strides().to_vec();
                let mut out = Vec::with_capacity(norm.len());
                let mut coord = vec![0usize; layout.shape().len()];
                for &i in &norm {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    out.push(src[off as usize]);
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::C64(out))
            }
            Self::C128(_) => {
                let Ok(Self::C128(src)) = self.to_contiguous(layout) else { unreachable!() };
                let strides = Layout::contiguous(layout.shape().to_vec()).strides().to_vec();
                let mut out = Vec::with_capacity(norm.len());
                let mut coord = vec![0usize; layout.shape().len()];
                for &i in &norm {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    out.push(src[off as usize]);
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::C128(out))
            }
        }
    }

    /// Typed same-rank `scatter_add` core (same add rules as `index_add_vals`).
    pub fn scatter_add_vals(
        &self,
        layout: &Layout,
        other: &Self,
        other_layout: &Layout,
        idx_shape: &[usize],
        ax: usize,
        idx: &[i64],
    ) -> Result<Self> {
        const OP: &str = "scatter_add";
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                op: OP,
                expected: self.dtype(),
                got: other.dtype(),
            });
        }
        let dim = layout.shape()[ax];
        let norm: Vec<usize> = idx
            .iter()
            .map(|&r| Self::check_index(OP, r, dim))
            .collect::<Result<_>>()?;
        macro_rules! scatter_typed {
            ($variant:ident, $t:ty, $add:expr) => {{
                let mut dst = self.to_vec::<$t>(layout);
                let src = other.to_vec::<$t>(other_layout);
                let strides = Layout::contiguous(layout.shape().to_vec())
                    .strides()
                    .to_vec();
                let add: fn(&mut $t, $t) = $add;
                let mut coord = vec![0usize; layout.shape().len()];
                for (&i, &v) in norm.iter().zip(src.iter()) {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    add(&mut dst[off as usize], v);
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::$variant(dst))
            }};
        }
        match (self, other) {
            (Self::Bool(_), Self::Bool(_)) => {
                scatter_typed!(Bool, bool, |a: &mut bool, b: bool| *a |= b)
            }
            (Self::U8(_), Self::U8(_)) => {
                scatter_typed!(U8, u8, |a: &mut u8, b: u8| *a = a.wrapping_add(b))
            }
            (Self::I8(_), Self::I8(_)) => {
                scatter_typed!(I8, i8, |a: &mut i8, b: i8| *a = a.wrapping_add(b))
            }
            (Self::I16(_), Self::I16(_)) => {
                scatter_typed!(I16, i16, |a: &mut i16, b: i16| *a = a.wrapping_add(b))
            }
            (Self::U16(_), Self::U16(_)) => {
                scatter_typed!(U16, u16, |a: &mut u16, b: u16| *a = a.wrapping_add(b))
            }
            (Self::U32(_), Self::U32(_)) => {
                scatter_typed!(U32, u32, |a: &mut u32, b: u32| *a = a.wrapping_add(b))
            }
            (Self::U64(_), Self::U64(_)) => {
                scatter_typed!(U64, u64, |a: &mut u64, b: u64| *a = a.wrapping_add(b))
            }
            (Self::I32(_), Self::I32(_)) => {
                scatter_typed!(I32, i32, |a: &mut i32, b: i32| *a = a.wrapping_add(b))
            }
            (Self::I64(_), Self::I64(_)) => {
                scatter_typed!(I64, i64, |a: &mut i64, b: i64| *a = a.wrapping_add(b))
            }
            (Self::F16(_), Self::F16(_)) => {
                scatter_typed!(F16, f16, |a: &mut f16, b: f16| {
                    *a = f16::from_f32(a.to_f32() + b.to_f32())
                })
            }
            (Self::BF16(_), Self::BF16(_)) => {
                scatter_typed!(BF16, bf16, |a: &mut bf16, b: bf16| {
                    *a = bf16::from_f32(a.to_f32() + b.to_f32())
                })
            }
            (Self::F8E4M3(_), Self::F8E4M3(_)) => {
                scatter_typed!(F8E4M3, f8e4m3, |a: &mut f8e4m3, b: f8e4m3| {
                    *a = f8e4m3::from_f32(a.to_f32() + b.to_f32())
                })
            }
            (Self::F8E5M2(_), Self::F8E5M2(_)) => {
                scatter_typed!(F8E5M2, f8e5m2, |a: &mut f8e5m2, b: f8e5m2| {
                    *a = f8e5m2::from_f32(a.to_f32() + b.to_f32())
                })
            }
            (Self::F32(_), Self::F32(_)) => {
                scatter_typed!(F32, f32, |a: &mut f32, b: f32| *a += b)
            }
            (Self::F64(_), Self::F64(_)) => {
                scatter_typed!(F64, f64, |a: &mut f64, b: f64| *a += b)
            }
            (Self::C64(_), Self::C64(_)) => {
                let Ok(Self::C64(mut dst)) = self.to_contiguous(layout) else { unreachable!() };
                let Ok(Self::C64(src)) = other.to_contiguous(other_layout) else { unreachable!() };
                let strides = Layout::contiguous(layout.shape().to_vec()).strides().to_vec();
                let mut coord = vec![0usize; layout.shape().len()];
                for (&i, &v) in norm.iter().zip(src.iter()) {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    dst[off as usize][0] += v[0];
                    dst[off as usize][1] += v[1];
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::C64(dst))
            }
            (Self::C128(_), Self::C128(_)) => {
                let Ok(Self::C128(mut dst)) = self.to_contiguous(layout) else { unreachable!() };
                let Ok(Self::C128(src)) = other.to_contiguous(other_layout) else { unreachable!() };
                let strides = Layout::contiguous(layout.shape().to_vec()).strides().to_vec();
                let mut coord = vec![0usize; layout.shape().len()];
                for (&i, &v) in norm.iter().zip(src.iter()) {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    dst[off as usize][0] += v[0];
                    dst[off as usize][1] += v[1];
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::C128(dst))
            }
            _ => Err(Error::DTypeMismatch {
                op: OP,
                expected: self.dtype(),
                got: other.dtype(),
            }),
        }
    }

    /// Typed same-rank `scatter` core: plain writes, sequential order, so
    /// duplicate indices resolve last-write-wins deterministically.
    pub fn scatter_vals(
        &self,
        layout: &Layout,
        other: &Self,
        other_layout: &Layout,
        idx_shape: &[usize],
        ax: usize,
        idx: &[i64],
    ) -> Result<Self> {
        const OP: &str = "scatter";
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                op: OP,
                expected: self.dtype(),
                got: other.dtype(),
            });
        }
        let dim = layout.shape()[ax];
        let norm: Vec<usize> = idx
            .iter()
            .map(|&r| Self::check_index(OP, r, dim))
            .collect::<Result<_>>()?;
        macro_rules! scatter_write {
            ($variant:ident, $t:ty) => {{
                let mut dst = self.to_vec::<$t>(layout);
                let src = other.to_vec::<$t>(other_layout);
                let strides = Layout::contiguous(layout.shape().to_vec())
                    .strides()
                    .to_vec();
                let mut coord = vec![0usize; layout.shape().len()];
                for (&i, &v) in norm.iter().zip(src.iter()) {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    dst[off as usize] = v;
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::$variant(dst))
            }};
        }
        match (self, other) {
            (Self::Bool(_), Self::Bool(_)) => scatter_write!(Bool, bool),
            (Self::U8(_), Self::U8(_)) => scatter_write!(U8, u8),
            (Self::I8(_), Self::I8(_)) => scatter_write!(I8, i8),
            (Self::I16(_), Self::I16(_)) => scatter_write!(I16, i16),
            (Self::U16(_), Self::U16(_)) => scatter_write!(U16, u16),
            (Self::U32(_), Self::U32(_)) => scatter_write!(U32, u32),
            (Self::U64(_), Self::U64(_)) => scatter_write!(U64, u64),
            (Self::I32(_), Self::I32(_)) => scatter_write!(I32, i32),
            (Self::I64(_), Self::I64(_)) => scatter_write!(I64, i64),
            (Self::F16(_), Self::F16(_)) => scatter_write!(F16, f16),
            (Self::BF16(_), Self::BF16(_)) => scatter_write!(BF16, bf16),
            (Self::F8E4M3(_), Self::F8E4M3(_)) => scatter_write!(F8E4M3, f8e4m3),
            (Self::F8E5M2(_), Self::F8E5M2(_)) => scatter_write!(F8E5M2, f8e5m2),
            (Self::F32(_), Self::F32(_)) => scatter_write!(F32, f32),
            (Self::F64(_), Self::F64(_)) => scatter_write!(F64, f64),
            (Self::C64(_), Self::C64(_)) => {
                let Ok(Self::C64(mut dst)) = self.to_contiguous(layout) else { unreachable!() };
                let Ok(Self::C64(src)) = other.to_contiguous(other_layout) else { unreachable!() };
                let strides = Layout::contiguous(layout.shape().to_vec()).strides().to_vec();
                let mut coord = vec![0usize; layout.shape().len()];
                for (&i, &v) in norm.iter().zip(src.iter()) {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    dst[off as usize] = v;
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::C64(dst))
            }
            (Self::C128(_), Self::C128(_)) => {
                let Ok(Self::C128(mut dst)) = self.to_contiguous(layout) else { unreachable!() };
                let Ok(Self::C128(src)) = other.to_contiguous(other_layout) else { unreachable!() };
                let strides = Layout::contiguous(layout.shape().to_vec()).strides().to_vec();
                let mut coord = vec![0usize; layout.shape().len()];
                for (&i, &v) in norm.iter().zip(src.iter()) {
                    let mut off = 0isize;
                    for d in 0..layout.shape().len() {
                        let c = if d == ax { i } else { coord[d] };
                        off += (c as isize) * strides[d];
                    }
                    dst[off as usize] = v;
                    for d in (0..idx_shape.len()).rev() {
                        coord[d] += 1;
                        if coord[d] < idx_shape[d] {
                            break;
                        }
                        coord[d] = 0;
                    }
                }
                Ok(Self::C128(dst))
            }
            _ => Err(Error::DTypeMismatch {
                op: OP,
                expected: self.dtype(),
                got: other.dtype(),
            }),
        }
    }

    pub fn where_cond(
        cond: &Self,
        cond_l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        if t.dtype() != f.dtype() {
            return Err(Error::DTypeMismatch {
                op: "where_cond",
                expected: t.dtype(),
                got: f.dtype(),
            });
        }
        // Like binary_op, callers pre-broadcast; unequal shapes would zip-truncate.
        if cond_l.shape() != t_l.shape() || t_l.shape() != f_l.shape() {
            return Err(Error::ShapeMismatch {
                op: "where_cond",
                expected: cond_l.shape().to_vec(),
                got: t_l.shape().to_vec(),
            });
        }
        let c_vec = cond.to_vec::<bool>(cond_l);
        macro_rules! select_typed {
            ($variant:ident, $tv:expr, $fv:expr) => {{
                let out = c_vec
                    .iter()
                    .zip(t_l.storage_offsets().zip(f_l.storage_offsets()))
                    .map(|(&c, (it, jf))| if c { $tv[it] } else { $fv[jf] })
                    .collect();
                Ok(Self::$variant(out))
            }};
        }
        match (t, f) {
            (Self::Bool(tv), Self::Bool(fv)) => select_typed!(Bool, tv, fv),
            (Self::U8(tv), Self::U8(fv)) => select_typed!(U8, tv, fv),
            (Self::I8(tv), Self::I8(fv)) => select_typed!(I8, tv, fv),
            (Self::I16(tv), Self::I16(fv)) => select_typed!(I16, tv, fv),
            (Self::U16(tv), Self::U16(fv)) => select_typed!(U16, tv, fv),
            (Self::U32(tv), Self::U32(fv)) => select_typed!(U32, tv, fv),
            (Self::U64(tv), Self::U64(fv)) => select_typed!(U64, tv, fv),
            (Self::I32(tv), Self::I32(fv)) => select_typed!(I32, tv, fv),
            (Self::I64(tv), Self::I64(fv)) => select_typed!(I64, tv, fv),
            (Self::F16(tv), Self::F16(fv)) => select_typed!(F16, tv, fv),
            (Self::BF16(tv), Self::BF16(fv)) => select_typed!(BF16, tv, fv),
            (Self::F8E4M3(tv), Self::F8E4M3(fv)) => select_typed!(F8E4M3, tv, fv),
            (Self::F8E5M2(tv), Self::F8E5M2(fv)) => select_typed!(F8E5M2, tv, fv),
            (Self::F32(tv), Self::F32(fv)) => select_typed!(F32, tv, fv),
            (Self::F64(tv), Self::F64(fv)) => select_typed!(F64, tv, fv),
            (Self::C64(tv), Self::C64(fv)) => select_typed!(C64, tv, fv),
            (Self::C128(tv), Self::C128(fv)) => select_typed!(C128, tv, fv),
            _ => unreachable!(),
        }
    }

    // -----------------------------------------------------------------------
    // Multi-axis reductions
    // -----------------------------------------------------------------------
    pub fn reduce_op(&self, layout: &Layout, axes: &[usize], op: ReduceOp) -> Result<Self> {
        let shape = layout.shape();
        let ndim = shape.len();
        let mut is_red = vec![false; ndim];
        for &a in axes {
            if a >= ndim {
                return Err(Error::Msg(format!(
                    "reduce_op: axis {a} out of range for rank {ndim}"
                )));
            }
            is_red[a] = true;
        }
        let mut kept_dims = Vec::new();
        let mut red_dims = Vec::new();
        for (d, &red) in is_red.iter().enumerate() {
            if red {
                red_dims.push(d);
            } else {
                kept_dims.push(d);
            }
        }

        // Permute layout so kept dims come first, reduced dims come last.
        let mut perm = kept_dims.clone();
        perm.extend_from_slice(&red_dims);
        let perm_layout = layout.permute(&perm)?;
        let outer: usize = kept_dims.iter().map(|&d| shape[d]).product();
        let inner: usize = red_dims.iter().map(|&d| shape[d]).product();

        if let Some(range) = layout.contiguous_range_in(self.len()) {
            // Fast-path 1: Full reduction across all axes
            if kept_dims.is_empty() {
                match self {
                    Self::F32(v) => {
                        let slice = &v[range];
                        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                            let mut best_idx = 0usize;
                            let mut best_val = slice.first().copied().unwrap_or(0.0);
                            for (k, &x) in slice.iter().enumerate().skip(1) {
                                let better = match op {
                                    ReduceOp::ArgMax => x > best_val || (x.is_nan() && !best_val.is_nan()),
                                    ReduceOp::ArgMin => x < best_val || (x.is_nan() && !best_val.is_nan()),
                                    _ => false,
                                };
                                if better {
                                    best_val = x;
                                    best_idx = k;
                                }
                            }
                            return Ok(Self::I64(vec![best_idx as i64]));
                        }
                        return Ok(Self::F32(vec![reduce_slice_f32(slice, op)]));
                    }
                    Self::F64(v) => {
                        let slice = &v[range];
                        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                            let mut best_idx = 0usize;
                            let mut best_val = slice.first().copied().unwrap_or(0.0);
                            for (k, &x) in slice.iter().enumerate().skip(1) {
                                let better = match op {
                                    ReduceOp::ArgMax => x > best_val || (x.is_nan() && !best_val.is_nan()),
                                    ReduceOp::ArgMin => x < best_val || (x.is_nan() && !best_val.is_nan()),
                                    _ => false,
                                };
                                if better {
                                    best_val = x;
                                    best_idx = k;
                                }
                            }
                            return Ok(Self::I64(vec![best_idx as i64]));
                        }
                        return Ok(Self::F64(vec![reduce_slice_f64(slice, op)]));
                    }
                    _ => {}
                }
            }

            // Fast-path 2: Trailing reduction (all kept dims precede all reduced dims)
            let is_trailing = kept_dims.iter().enumerate().all(|(i, &d)| d == i)
                && red_dims.iter().enumerate().all(|(i, &d)| d == kept_dims.len() + i);
            if is_trailing {
                match self {
                    Self::F32(v) => {
                        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                            let mut out = vec![0i64; outer];
                            let run_o = |o: usize| {
                                let slice = &v[range.start + o * inner..range.start + (o + 1) * inner];
                                let mut best_idx = 0usize;
                                let mut best_val = slice[0];
                                for (k, &val) in slice.iter().enumerate().skip(1) {
                                    let better = match op {
                                        ReduceOp::ArgMax => val > best_val || (val.is_nan() && !best_val.is_nan()),
                                        ReduceOp::ArgMin => val < best_val || (val.is_nan() && !best_val.is_nan()),
                                        _ => false,
                                    };
                                    if better {
                                        best_val = val;
                                        best_idx = k;
                                    }
                                }
                                best_idx as i64
                            };
                            #[cfg(feature = "rayon")]
                            if outer >= 64 {
                                out.par_iter_mut().enumerate().for_each(|(o, dst)| *dst = run_o(o));
                                return Ok(Self::I64(out));
                            }
                            for (o, dst) in out.iter_mut().enumerate() {
                                *dst = run_o(o);
                            }
                            return Ok(Self::I64(out));
                        }
                        let mut out = vec![0.0f32; outer];
                        let run_o = |o: usize| {
                            let slice = &v[range.start + o * inner..range.start + (o + 1) * inner];
                            reduce_slice_f32(slice, op)
                        };
                        #[cfg(feature = "rayon")]
                        if outer >= 64 {
                            out.par_iter_mut().enumerate().for_each(|(o, dst)| *dst = run_o(o));
                            return Ok(Self::F32(out));
                        }
                        for (o, dst) in out.iter_mut().enumerate() {
                            *dst = run_o(o);
                        }
                        return Ok(Self::F32(out));
                    }
                    Self::F64(v) => {
                        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                            let mut out = vec![0i64; outer];
                            let run_o = |o: usize| {
                                let slice = &v[range.start + o * inner..range.start + (o + 1) * inner];
                                let mut best_idx = 0usize;
                                let mut best_val = slice[0];
                                for (k, &val) in slice.iter().enumerate().skip(1) {
                                    let better = match op {
                                        ReduceOp::ArgMax => val > best_val || (val.is_nan() && !best_val.is_nan()),
                                        ReduceOp::ArgMin => val < best_val || (val.is_nan() && !best_val.is_nan()),
                                        _ => false,
                                    };
                                    if better {
                                        best_val = val;
                                        best_idx = k;
                                    }
                                }
                                best_idx as i64
                            };
                            #[cfg(feature = "rayon")]
                            if outer >= 64 {
                                out.par_iter_mut().enumerate().for_each(|(o, dst)| *dst = run_o(o));
                                return Ok(Self::I64(out));
                            }
                            for (o, dst) in out.iter_mut().enumerate() {
                                *dst = run_o(o);
                            }
                            return Ok(Self::I64(out));
                        }
                        let mut out = vec![0.0f64; outer];
                        let run_o = |o: usize| {
                            let slice = &v[range.start + o * inner..range.start + (o + 1) * inner];
                            reduce_slice_f64(slice, op)
                        };
                        #[cfg(feature = "rayon")]
                        if outer >= 64 {
                            out.par_iter_mut().enumerate().for_each(|(o, dst)| *dst = run_o(o));
                            return Ok(Self::F64(out));
                        }
                        for (o, dst) in out.iter_mut().enumerate() {
                            *dst = run_o(o);
                        }
                        return Ok(Self::F64(out));
                    }
                    _ => {}
                }
            }

            // Fast-path 3: Leading reduction (all reduced dims precede all kept dims)
            let is_leading = red_dims.iter().enumerate().all(|(i, &d)| d == i)
                && kept_dims.iter().enumerate().all(|(i, &d)| d == red_dims.len() + i);
            if is_leading {
                match self {
                    Self::F32(v) => {
                        let block_size = leading_block(outer);
                        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                            let mut out = vec![0i64; outer];
                            let run_chunk = |b_idx: usize, chunk: &mut [i64]| {
                                let c_start = b_idx * block_size;
                                let w = chunk.len();
                                let mut best_val = vec![if op == ReduceOp::ArgMax { f32::NEG_INFINITY } else { f32::INFINITY }; w];
                                let mut best_idx = vec![0i64; w];
                                for r in 0..inner {
                                    let row_offset = range.start + r * outer + c_start;
                                    let row_slice = &v[row_offset..row_offset + w];
                                    for (j, &val) in row_slice.iter().enumerate() {
                                        let better = (if op == ReduceOp::ArgMax { val > best_val[j] } else { val < best_val[j] }) || (val.is_nan() && !best_val[j].is_nan());
                                        if better {
                                            best_val[j] = val;
                                            best_idx[j] = r as i64;
                                        }
                                    }
                                }
                                chunk.copy_from_slice(&best_idx);
                            };
                            #[cfg(feature = "rayon")]
                            if outer >= 64 {
                                out.par_chunks_mut(block_size).enumerate().for_each(|(b_idx, chunk)| run_chunk(b_idx, chunk));
                                return Ok(Self::I64(out));
                            }
                            for (b_idx, chunk) in out.chunks_mut(block_size).enumerate() {
                                run_chunk(b_idx, chunk);
                            }
                            return Ok(Self::I64(out));
                        }
                        let mut out = vec![0.0f32; outer];
                        let run_chunk = |b_idx: usize, chunk: &mut [f32]| {
                            let c_start = b_idx * block_size;
                            let w = chunk.len();
                            match op {
                                ReduceOp::Sum | ReduceOp::Mean => {
                                    // f64 accumulators: summing 1000 rows in f32 loses several bits (and far more with a common offset)
                                    let mut acc = vec![0.0f64; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        add_f32_into_f64(&mut acc, &v[row_offset..row_offset + w]);
                                    }
                                    let denom = if op == ReduceOp::Mean { inner as f64 } else { 1.0 };
                                    for (dst, sum) in chunk.iter_mut().zip(acc) {
                                        *dst = (sum / denom) as f32;
                                    }
                                }
                                ReduceOp::Max => {
                                    let mut acc = vec![f32::NEG_INFINITY; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a = a.nmax(val);
                                        }
                                    }
                                    chunk.copy_from_slice(&acc);
                                }
                                ReduceOp::Min => {
                                    let mut acc = vec![f32::INFINITY; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a = a.nmin(val);
                                        }
                                    }
                                    chunk.copy_from_slice(&acc);
                                }
                                ReduceOp::Prod => {
                                    let mut acc = vec![1.0f32; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a *= val;
                                        }
                                    }
                                    chunk.copy_from_slice(&acc);
                                }
                                _ => unreachable!(),
                            }
                        };
                        #[cfg(feature = "rayon")]
                        if outer >= 64 {
                            out.par_chunks_mut(block_size).enumerate().for_each(|(b_idx, chunk)| run_chunk(b_idx, chunk));
                            return Ok(Self::F32(out));
                        }
                        for (b_idx, chunk) in out.chunks_mut(block_size).enumerate() {
                            run_chunk(b_idx, chunk);
                        }
                        return Ok(Self::F32(out));
                    }
                    Self::F64(v) => {
                        let block_size = leading_block(outer);
                        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
                            let mut out = vec![0i64; outer];
                            let run_chunk = |b_idx: usize, chunk: &mut [i64]| {
                                let c_start = b_idx * block_size;
                                let w = chunk.len();
                                let mut best_val = vec![if op == ReduceOp::ArgMax { f64::NEG_INFINITY } else { f64::INFINITY }; w];
                                let mut best_idx = vec![0i64; w];
                                for r in 0..inner {
                                    let row_offset = range.start + r * outer + c_start;
                                    let row_slice = &v[row_offset..row_offset + w];
                                    for (j, &val) in row_slice.iter().enumerate() {
                                        let better = (if op == ReduceOp::ArgMax { val > best_val[j] } else { val < best_val[j] }) || (val.is_nan() && !best_val[j].is_nan());
                                        if better {
                                            best_val[j] = val;
                                            best_idx[j] = r as i64;
                                        }
                                    }
                                }
                                chunk.copy_from_slice(&best_idx);
                            };
                            #[cfg(feature = "rayon")]
                            if outer >= 64 {
                                out.par_chunks_mut(block_size).enumerate().for_each(|(b_idx, chunk)| run_chunk(b_idx, chunk));
                                return Ok(Self::I64(out));
                            }
                            for (b_idx, chunk) in out.chunks_mut(block_size).enumerate() {
                                run_chunk(b_idx, chunk);
                            }
                            return Ok(Self::I64(out));
                        }
                        let mut out = vec![0.0f64; outer];
                        let run_chunk = |b_idx: usize, chunk: &mut [f64]| {
                            let c_start = b_idx * block_size;
                            let w = chunk.len();
                            match op {
                                ReduceOp::Sum => {
                                    let mut acc = vec![0.0f64; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a += val;
                                        }
                                    }
                                    chunk.copy_from_slice(&acc);
                                }
                                ReduceOp::Mean => {
                                    let mut acc = vec![0.0f64; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a += val;
                                        }
                                    }
                                    let denom = inner as f64;
                                    for (dst, sum) in chunk.iter_mut().zip(acc) {
                                        *dst = sum / denom;
                                    }
                                }
                                ReduceOp::Max => {
                                    let mut acc = vec![f64::NEG_INFINITY; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a = a.nmax(val);
                                        }
                                    }
                                    chunk.copy_from_slice(&acc);
                                }
                                ReduceOp::Min => {
                                    let mut acc = vec![f64::INFINITY; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a = a.nmin(val);
                                        }
                                    }
                                    chunk.copy_from_slice(&acc);
                                }
                                ReduceOp::Prod => {
                                    let mut acc = vec![1.0f64; w];
                                    for r in 0..inner {
                                        let row_offset = range.start + r * outer + c_start;
                                        let row_slice = &v[row_offset..row_offset + w];
                                        for (a, &val) in acc.iter_mut().zip(row_slice) {
                                            *a *= val;
                                        }
                                    }
                                    chunk.copy_from_slice(&acc);
                                }
                                _ => unreachable!(),
                            }
                        };
                        #[cfg(feature = "rayon")]
                        if outer >= 64 {
                            out.par_chunks_mut(block_size).enumerate().for_each(|(b_idx, chunk)| run_chunk(b_idx, chunk));
                            return Ok(Self::F64(out));
                        }
                        for (b_idx, chunk) in out.chunks_mut(block_size).enumerate() {
                            run_chunk(b_idx, chunk);
                        }
                        return Ok(Self::F64(out));
                    }
                    _ => {}
                }
            }
        }

        if matches!(self, Self::C64(_) | Self::C128(_)) {
            if !matches!(op, ReduceOp::Sum | ReduceOp::Mean | ReduceOp::Prod) {
                return Err(Error::UnsupportedDType {
                    op: "reduce_op",
                    dtype: self.dtype(),
                });
            }
            match self {
                Self::C64(v) => {
                    let data: Vec<[f32; 2]> = perm_layout.storage_offsets().map(|i| v[i]).collect();
                    let mut out = Vec::with_capacity(outer);
                    for o in 0..outer {
                        let slice = &data[o * inner..(o + 1) * inner];
                        let val = match op {
                            ReduceOp::Sum => {
                                let (mut sr, mut si) = (0.0f64, 0.0f64);
                                for &[r, i] in slice {
                                    sr += r as f64;
                                    si += i as f64;
                                }
                                [sr as f32, si as f32]
                            }
                            ReduceOp::Mean => {
                                let (mut sr, mut si) = (0.0f64, 0.0f64);
                                for &[r, i] in slice {
                                    sr += r as f64;
                                    si += i as f64;
                                }
                                let n = inner as f64;
                                [(sr / n) as f32, (si / n) as f32]
                            }
                            ReduceOp::Prod => {
                                let mut acc = [1.0f64, 0.0f64];
                                for &[r, i] in slice {
                                    let (ar, ai) = (acc[0], acc[1]);
                                    acc = [ar * r as f64 - ai * i as f64, ar * i as f64 + ai * r as f64];
                                }
                                [acc[0] as f32, acc[1] as f32]
                            }
                            _ => unreachable!(),
                        };
                        out.push(val);
                    }
                    return Ok(Self::C64(out));
                }
                Self::C128(v) => {
                    let data: Vec<[f64; 2]> = perm_layout.storage_offsets().map(|i| v[i]).collect();
                    let mut out = Vec::with_capacity(outer);
                    for o in 0..outer {
                        let slice = &data[o * inner..(o + 1) * inner];
                        let val = match op {
                            ReduceOp::Sum => {
                                let (mut sr, mut si) = (0.0f64, 0.0f64);
                                for &[r, i] in slice {
                                    sr += r;
                                    si += i;
                                }
                                [sr, si]
                            }
                            ReduceOp::Mean => {
                                let (mut sr, mut si) = (0.0f64, 0.0f64);
                                for &[r, i] in slice {
                                    sr += r;
                                    si += i;
                                }
                                let n = inner as f64;
                                [sr / n, si / n]
                            }
                            ReduceOp::Prod => {
                                let mut acc = [1.0f64, 0.0f64];
                                for &[r, i] in slice {
                                    let (ar, ai) = (acc[0], acc[1]);
                                    acc = [ar * r - ai * i, ar * i + ai * r];
                                }
                                acc
                            }
                            _ => unreachable!(),
                        };
                        out.push(val);
                    }
                    return Ok(Self::C128(out));
                }
                _ => unreachable!(),
            }
        }

        if matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin) {
            let data = self.to_vec::<f64>(&perm_layout);
            let mut out = vec![0i64; outer];
            for o in 0..outer {
                let slice = &data[o * inner..(o + 1) * inner];
                let mut best_idx = 0usize;
                let mut best_val = slice[0];
                for (k, &v) in slice.iter().enumerate().skip(1) {
                    let better = match op {
                        ReduceOp::ArgMax => v > best_val || (v.is_nan() && !best_val.is_nan()),
                        ReduceOp::ArgMin => v < best_val || (v.is_nan() && !best_val.is_nan()),
                        _ => false,
                    };
                    if better {
                        best_val = v;
                        best_idx = k;
                    }
                }
                out[o] = best_idx as i64;
            }
            return Ok(Self::I64(out));
        }

        if self.dtype() == DType::F32 {
            let data = self.to_vec::<f32>(&perm_layout);
            let mut out = vec![0.0f32; outer];
            for (o, dst) in out.iter_mut().enumerate() {
                let slice = &data[o * inner..(o + 1) * inner];
                *dst = match op {
                    // Accumulate in f64: a running f32 total drifts ~1% over
                    // a million values; the cast back is a single rounding.
                    ReduceOp::Sum => slice.iter().map(|&v| v as f64).sum::<f64>() as f32,
                    ReduceOp::Mean => {
                        (slice.iter().map(|&v| v as f64).sum::<f64>() / (inner as f64)) as f32
                    }
                    ReduceOp::Max => slice.iter().copied().fold(f32::NEG_INFINITY, NanProp::nmax),
                    ReduceOp::Min => slice.iter().copied().fold(f32::INFINITY, NanProp::nmin),
                    ReduceOp::Prod => slice.iter().product(),
                    _ => unreachable!(),
                };
            }
            return Ok(Self::F32(out));
        }

        let data = self.to_vec::<f64>(&perm_layout);
        let mut out = vec![0.0f64; outer];
        for (o, dst) in out.iter_mut().enumerate() {
            let slice = &data[o * inner..(o + 1) * inner];
            *dst = match op {
                ReduceOp::Sum => slice.iter().sum(),
                ReduceOp::Mean => slice.iter().sum::<f64>() / (inner as f64),
                ReduceOp::Max => slice.iter().copied().fold(f64::NEG_INFINITY, NanProp::nmax),
                ReduceOp::Min => slice.iter().copied().fold(f64::INFINITY, NanProp::nmin),
                ReduceOp::Prod => slice.iter().product(),
                _ => unreachable!(),
            };
        }
        let res_storage = Self::F64(out);
        res_storage.to_dtype(&Layout::contiguous(vec![outer]), self.dtype())
    }

    // -----------------------------------------------------------------------
    // Strided Batched Matmul
    // -----------------------------------------------------------------------
    pub fn matmul(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<(Self, Vec<usize>)> {
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                op: "matmul",
                expected: self.dtype(),
                got: rhs.dtype(),
            });
        }
        let (ar, br) = (lhs_l.ndim(), rhs_l.ndim());
        if ar < 2 || br < 2 {
            return Err(Error::Msg(format!(
                "matmul requires rank >= 2, got {:?} and {:?}",
                lhs_l.shape(),
                rhs_l.shape()
            )));
        }
        let (m, k) = (lhs_l.shape()[ar - 2], lhs_l.shape()[ar - 1]);
        let (k2, n) = (rhs_l.shape()[br - 2], rhs_l.shape()[br - 1]);
        if k != k2 {
            return Err(Error::Msg(format!(
                "matmul: shape mismatch ({:?} and {:?}, inner dims {k} != {k2})",
                lhs_l.shape(),
                rhs_l.shape()
            )));
        }

        let batch_shape =
            broadcast_shapes("matmul", &lhs_l.shape()[..ar - 2], &rhs_l.shape()[..br - 2])?;
        let mut lhs_target = batch_shape.clone();
        lhs_target.extend([m, k]);
        let mut rhs_target = batch_shape.clone();
        rhs_target.extend([k, n]);

        let lhs_b = lhs_l.broadcast_to(&lhs_target)?;
        let rhs_b = rhs_l.broadcast_to(&rhs_target)?;

        let b_rank = batch_shape.len();
        let rs_a = lhs_b.strides()[b_rank];
        let cs_a = lhs_b.strides()[b_rank + 1];
        let rs_b = rhs_b.strides()[b_rank];
        let cs_b = rhs_b.strides()[b_rank + 1];

        let batch_layout_a = Layout::new(
            batch_shape.clone(),
            lhs_b.strides()[..b_rank].to_vec(),
            lhs_b.offset(),
        );
        let batch_layout_b = Layout::new(
            batch_shape.clone(),
            rhs_b.strides()[..b_rank].to_vec(),
            rhs_b.offset(),
        );
        let batch_offsets: Vec<(usize, usize)> = batch_layout_a
            .storage_offsets()
            .zip(batch_layout_b.storage_offsets())
            .collect();

        let mut out_shape = batch_shape;
        out_shape.extend([m, n]);
        // Refuse outputs that can never fit BEFORE the GEMM reserves nb*m*n:
        // broadcast batch x independent m/n can exceed memory on validated
        // operands.
        crate::limits::check_elements("matmul", &out_shape)?;

        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => {
                let c = batched_gemm_f32(a, rs_a, cs_a, b, rs_b, cs_b, &batch_offsets, m, k, n);
                Ok((Self::F32(c), out_shape))
            }
            (Self::F64(a), Self::F64(b)) => {
                let c = batched_gemm_f64(a, rs_a, cs_a, b, rs_b, cs_b, &batch_offsets, m, k, n);
                Ok((Self::F64(c), out_shape))
            }
            (Self::C64(a), Self::C64(b)) => {
                let mut out = vec![[0.0f32; 2]; batch_offsets.len() * m * n];
                for (bi, &(off_a, off_b)) in batch_offsets.iter().enumerate() {
                    let out_base = bi * m * n;
                    for i in 0..m {
                        for j in 0..n {
                            let (mut sr, mut si) = (0.0f64, 0.0f64);
                            for p in 0..k {
                                let va = a[(off_a as isize + i as isize * rs_a + p as isize * cs_a) as usize];
                                let vb = b[(off_b as isize + p as isize * rs_b + j as isize * cs_b) as usize];
                                sr += va[0] as f64 * vb[0] as f64 - va[1] as f64 * vb[1] as f64;
                                si += va[0] as f64 * vb[1] as f64 + va[1] as f64 * vb[0] as f64;
                            }
                            out[out_base + i * n + j] = [sr as f32, si as f32];
                        }
                    }
                }
                Ok((Self::C64(out), out_shape))
            }
            (Self::C128(a), Self::C128(b)) => {
                let mut out = vec![[0.0f64; 2]; batch_offsets.len() * m * n];
                for (bi, &(off_a, off_b)) in batch_offsets.iter().enumerate() {
                    let out_base = bi * m * n;
                    for i in 0..m {
                        for j in 0..n {
                            let (mut sr, mut si) = (0.0f64, 0.0f64);
                            for p in 0..k {
                                let va = a[(off_a as isize + i as isize * rs_a + p as isize * cs_a) as usize];
                                let vb = b[(off_b as isize + p as isize * rs_b + j as isize * cs_b) as usize];
                                sr += va[0] * vb[0] - va[1] * vb[1];
                                si += va[0] * vb[1] + va[1] * vb[0];
                            }
                            out[out_base + i * n + j] = [sr, si];
                        }
                    }
                }
                Ok((Self::C128(out), out_shape))
            }
            (Self::F16(_), Self::F16(_)) | (Self::BF16(_), Self::BF16(_)) => {
                let a_f32 = self.to_dtype(lhs_l, DType::F32)?;
                let b_f32 = rhs.to_dtype(rhs_l, DType::F32)?;
                let la = Layout::contiguous(lhs_l.shape().to_vec());
                let lb = Layout::contiguous(rhs_l.shape().to_vec());
                let (c_f32, shape) = a_f32.matmul(&la, &b_f32, &lb)?;
                let lc = Layout::contiguous(shape.clone());
                Ok((c_f32.to_dtype(&lc, self.dtype())?, shape))
            }
            (Self::F8E4M3(_), Self::F8E4M3(_)) => {
                let a_f32 = self.to_dtype(lhs_l, DType::F32)?;
                let b_f32 = rhs.to_dtype(rhs_l, DType::F32)?;
                let la = Layout::contiguous(lhs_l.shape().to_vec());
                let lb = Layout::contiguous(rhs_l.shape().to_vec());
                let (c_f32, shape) = a_f32.matmul(&la, &b_f32, &lb)?;
                let lc = Layout::contiguous(shape.clone());
                Ok((c_f32.to_dtype(&lc, self.dtype())?, shape))
            }
            (Self::F8E5M2(_), Self::F8E5M2(_)) => {
                let a_f32 = self.to_dtype(lhs_l, DType::F32)?;
                let b_f32 = rhs.to_dtype(rhs_l, DType::F32)?;
                let la = Layout::contiguous(lhs_l.shape().to_vec());
                let lb = Layout::contiguous(rhs_l.shape().to_vec());
                let (c_f32, shape) = a_f32.matmul(&la, &b_f32, &lb)?;
                let lc = Layout::contiguous(shape.clone());
                Ok((c_f32.to_dtype(&lc, self.dtype())?, shape))
            }
            _ => {
                let a_f64 = self.to_dtype(lhs_l, DType::F64)?;
                let b_f64 = rhs.to_dtype(rhs_l, DType::F64)?;
                let la = Layout::contiguous(lhs_l.shape().to_vec());
                let lb = Layout::contiguous(rhs_l.shape().to_vec());
                let (c_f64, shape) = a_f64.matmul(&la, &b_f64, &lb)?;
                let lc = Layout::contiguous(shape.clone());
                Ok((c_f64.to_dtype(&lc, self.dtype())?, shape))
            }
        }
    }
}

/// `exp` for `f32` without a library call or a branch, so loops over it vectorize (the system `expf` is a scalar call
/// that pins every exponential, sigmoid and softmax to one element at a time). Cephes' `expf` scheme: `x = n ln2 + r`
/// with `|r| <= ln2 / 2`, a degree-5 polynomial for `e^r - 1 - r`, and the scale `2^n` applied in two halves so results
/// in the subnormal range underflow gradually. Within 1 ulp of the correctly rounded value on the normal range;
/// `exp(NaN) = NaN`, overflow gives `inf`, large negative inputs give 0.
#[inline(always)]
#[allow(clippy::manual_clamp)] // max/min also map NaN into range; `clamp` would keep it and the int cast below needs a finite value
pub(crate) fn fast_exp_f32(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const C1: f32 = 0.693_359_4; // ln 2 split in a short high part and a correction
    const C2: f32 = -2.121_944_4e-4;
    const ROUND: f32 = 12_582_912.0; // 1.5 * 2^23: adding and subtracting it rounds to the nearest integer
    let input = x;
    let x = x.max(-150.0).min(89.0); // also maps NaN into range; it is restored at the end
    let n = (x * LOG2E + ROUND) - ROUND;
    let r = x - n * C1 - n * C2;
    let mut p = 1.987_569_1e-4f32;
    p = p * r + 1.398_2e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 5.0e-1;
    let y = p * r * r + r + 1.0;
    // SAFETY: after the clamp `n` is an integer in [-217, 129], which fits an `i32` (the checked `as` cast would add a
    // saturation sequence to every element for a case that cannot occur).
    let ni = unsafe { n.to_int_unchecked::<i32>() };
    let n1 = ni >> 1;
    let n2 = ni - n1;
    let s1 = f32::from_bits(((n1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((n2 + 127) as u32) << 23);
    let out = y * s1 * s2;
    if input.is_nan() { input } else { out }
}

/// Logistic function on top of [`fast_exp_f32`], branch-free: `e = exp(-|x|)` never overflows, and the two halves
/// `1 / (1 + e)` and `e / (1 + e)` are chosen by sign.
#[inline(always)]
pub(crate) fn fast_sigmoid_f32(x: f32) -> f32 {
    let e = fast_exp_f32(-x.abs());
    let s = 1.0 / (1.0 + e);
    if x >= 0.0 { s } else { e * s }
}

/// Standard normal CDF `0.5 * erfc(-x / sqrt(2))` for `f32`, branch-free on top of [`fast_exp_f32`]: the Chebyshev fit
/// of `erfc` from Numerical Recipes (`erfcc`), whose *relative* error is below 1.2e-7 for every argument, so the left
/// tail (where `1 + erf` would cancel) keeps its precision.
#[inline(always)]
pub(crate) fn fast_norm_cdf_f32(x: f32) -> f32 {
    let u = -x * std::f32::consts::FRAC_1_SQRT_2; // cdf = 0.5 * erfc(u)
    let z = u.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -1.265_512_2
        + t * (1.000_023_7
            + t * (0.374_091_96
                + t * (0.096_784_18 + t * (-0.186_288_06 + t * (0.278_868_07 + t * (-1.135_204 + t * (1.488_515_9 + t * (-0.822_152_26 + t * 0.170_872_77))))))));
    let erfc_abs = t * fast_exp_f32(-z * z + poly); // erfc(|u|)
    0.5 * if u >= 0.0 { erfc_abs } else { 2.0 - erfc_abs }
}

#[inline(always)]
pub(crate) fn eval_unary_f32(x: f32, op: UnaryOp) -> f32 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryOp::Exp => fast_exp_f32(x),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sigmoid => fast_sigmoid_f32(x),
        // not `x.max(0.0)`: that maps NaN to 0 and hides a diverged activation (torch.relu propagates NaN)
        UnaryOp::Relu => if x < 0.0 { 0.0 } else { x },
        UnaryOp::LeakyRelu(s) => {
            if x > 0.0 {
                x
            } else {
                (s as f32) * x
            }
        }
        UnaryOp::Gelu => {
            // Exact GELU via erf: 0.5 * x * (1 + erf(x / sqrt(2)))
            x * fast_norm_cdf_f32(x)
        }
        UnaryOp::Silu => {
            // the sigmoid is written out (not a recursive call): a recursive function cannot be inlined into the
            // per-operation loops, and every element would pay a call plus the whole `match`
            x * fast_sigmoid_f32(x)
        }
        UnaryOp::Elu(alpha) => {
            if x > 0.0 {
                x
            } else {
                (alpha as f32) * x.exp_m1()
            }
        }
        UnaryOp::Softplus => {
            if x > 20.0 {
                x
            } else {
                // f64 inside: a rounded f32 exp(x) puts up to a full ulp of error into ln(1 + exp(x))
                ((x as f64).exp().ln_1p()) as f32
            }
        }
        UnaryOp::Square => x * x,
        UnaryOp::Recip => 1.0 / x,
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => x.round_ties_even(), // half to even, like torch.round / numpy.round / WGSL round
        UnaryOp::Erf => libm::erf(x as f64) as f32 /* double precision, then one rounding: erff alone is ~1.3 half-ulps */,
        UnaryOp::Powf(p) => x.powf(p as f32),
        UnaryOp::AddScalar(s) => x + (s as f32),
        UnaryOp::MulScalar(s) => x * (s as f32),
        UnaryOp::Clamp(lo, hi) => x.clamp(lo as f32, hi as f32),
        UnaryOp::LogicalNot => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

#[inline]
pub(crate) fn eval_unary_f64(x: f64, op: UnaryOp) -> f64 {
    match op {
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Rsqrt => 1.0 / x.sqrt(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sigmoid => {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        }
        // not `x.max(0.0)`: that maps NaN to 0 and hides a diverged activation (torch.relu propagates NaN)
        UnaryOp::Relu => if x < 0.0 { 0.0 } else { x },
        UnaryOp::LeakyRelu(s) => {
            if x > 0.0 {
                x
            } else {
                s * x
            }
        }
        UnaryOp::Gelu => 0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2)),
        UnaryOp::Silu => x * eval_unary_f64(x, UnaryOp::Sigmoid),
        UnaryOp::Elu(alpha) => {
            if x > 0.0 {
                x
            } else {
                alpha * x.exp_m1()
            }
        }
        UnaryOp::Softplus => {
            if x > 20.0 {
                x
            } else {
                x.exp().ln_1p()
            }
        }
        UnaryOp::Square => x * x,
        UnaryOp::Recip => 1.0 / x,
        UnaryOp::Floor => x.floor(),
        UnaryOp::Ceil => x.ceil(),
        UnaryOp::Round => x.round_ties_even(), // half to even, like torch.round / numpy.round / WGSL round
        UnaryOp::Erf => libm::erf(x),
        UnaryOp::Powf(p) => x.powf(p),
        UnaryOp::AddScalar(s) => x + s,
        UnaryOp::MulScalar(s) => x * s,
        UnaryOp::Clamp(lo, hi) => x.clamp(lo, hi),
        UnaryOp::LogicalNot => {
            if x == 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn reduce_slice_f32(slice: &[f32], op: ReduceOp) -> f32 {
    // sums and means accumulate in f64 (the final rounding to f32 is then the only rounding of the result)
    let sum_f64 = || -> f64 {
        if slice.len() >= 65536 {
            slice.par_chunks(16384).map(sum_f64_of_f32).sum::<f64>()
        } else {
            sum_f64_of_f32(slice)
        }
    };
    match op {
        ReduceOp::Sum => sum_f64() as f32,
        ReduceOp::Mean => (sum_f64() / slice.len() as f64) as f32,
        ReduceOp::Max => {
            if slice.len() >= 16384 {
                slice.par_iter().copied().reduce(|| f32::NEG_INFINITY, NanProp::nmax)
            } else {
                slice.iter().copied().fold(f32::NEG_INFINITY, NanProp::nmax)
            }
        }
        ReduceOp::Min => {
            if slice.len() >= 16384 {
                slice.par_iter().copied().reduce(|| f32::INFINITY, NanProp::nmin)
            } else {
                slice.iter().copied().fold(f32::INFINITY, NanProp::nmin)
            }
        }
        ReduceOp::Prod => slice.iter().product(),
        _ => unreachable!(),
    }
}

#[cfg(not(feature = "rayon"))]
fn reduce_slice_f32(slice: &[f32], op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum => slice.iter().map(|&x| x as f64).sum::<f64>() as f32,
        ReduceOp::Mean => (slice.iter().map(|&x| x as f64).sum::<f64>() / slice.len() as f64) as f32,
        ReduceOp::Max => slice.iter().copied().fold(f32::NEG_INFINITY, NanProp::nmax),
        ReduceOp::Min => slice.iter().copied().fold(f32::INFINITY, NanProp::nmin),
        ReduceOp::Prod => slice.iter().product(),
        _ => unreachable!(),
    }
}

#[cfg(feature = "rayon")]
fn reduce_slice_f64(slice: &[f64], op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum => {
            if slice.len() >= 16384 {
                slice.par_iter().sum::<f64>()
            } else {
                slice.iter().sum::<f64>()
            }
        }
        ReduceOp::Mean => {
            let sum: f64 = if slice.len() >= 16384 {
                slice.par_iter().sum::<f64>()
            } else {
                slice.iter().sum::<f64>()
            };
            sum / slice.len() as f64
        }
        ReduceOp::Max => {
            if slice.len() >= 16384 {
                slice.par_iter().copied().reduce(|| f64::NEG_INFINITY, NanProp::nmax)
            } else {
                slice.iter().copied().fold(f64::NEG_INFINITY, NanProp::nmax)
            }
        }
        ReduceOp::Min => {
            if slice.len() >= 16384 {
                slice.par_iter().copied().reduce(|| f64::INFINITY, NanProp::nmin)
            } else {
                slice.iter().copied().fold(f64::INFINITY, NanProp::nmin)
            }
        }
        ReduceOp::Prod => slice.iter().product(),
        _ => unreachable!(),
    }
}

#[cfg(not(feature = "rayon"))]
fn reduce_slice_f64(slice: &[f64], op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum => slice.iter().sum(),
        ReduceOp::Mean => slice.iter().sum::<f64>() / slice.len() as f64,
        ReduceOp::Max => slice.iter().copied().fold(f64::NEG_INFINITY, NanProp::nmax),
        ReduceOp::Min => slice.iter().copied().fold(f64::INFINITY, NanProp::nmin),
        ReduceOp::Prod => slice.iter().product(),
        _ => unreachable!(),
    }
}

#[inline]
fn eval_cmp<T: PartialOrd>(x: T, y: T, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => x == y,
        CmpOp::Ne => x != y,
        CmpOp::Lt => x < y,
        CmpOp::Le => x <= y,
        CmpOp::Gt => x > y,
        CmpOp::Ge => x >= y,
    }
}

#[inline]
pub(crate) fn eval_bin_f32(x: f32, y: f32, op: BinaryOp) -> f32 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
        // NaN propagates (torch.maximum/minimum parity). Rust's `f32::max`
        // would silently return the non-NaN operand instead.
        BinaryOp::Maximum => {
            if x.is_nan() || y.is_nan() {
                f32::NAN
            } else {
                x.max(y)
            }
        }
        BinaryOp::Minimum => {
            if x.is_nan() || y.is_nan() {
                f32::NAN
            } else {
                x.min(y)
            }
        }
        BinaryOp::Pow => x.powf(y),
        BinaryOp::Fmod => x % y,
        BinaryOp::Remainder => remainder_f32(x, y),
        BinaryOp::FloorDiv => floor_div_f32(x, y),
    }
}

#[inline]
pub(crate) fn eval_bin_f64(x: f64, y: f64, op: BinaryOp) -> f64 {
    match op {
        BinaryOp::Add => x + y,
        BinaryOp::Sub => x - y,
        BinaryOp::Mul => x * y,
        BinaryOp::Div => x / y,
        // NaN propagates (torch.maximum/minimum parity). Rust's `f64::max`
        // would silently return the non-NaN operand instead.
        BinaryOp::Maximum => {
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else {
                x.max(y)
            }
        }
        BinaryOp::Minimum => {
            if x.is_nan() || y.is_nan() {
                f64::NAN
            } else {
                x.min(y)
            }
        }
        BinaryOp::Pow => x.powf(y),
        BinaryOp::Fmod => x % y,
        BinaryOp::Remainder => remainder_f64(x, y),
        BinaryOp::FloorDiv => floor_div_f64(x, y),
    }
}

impl Default for DimMap {
    fn default() -> Self {
        Self::Selected { index: 0 }
    }
}

/// Maximum / minimum that propagate NaN (`f64::max` ignores it), matching `torch.max` / `torch.amax`: a slice
/// containing NaN reduces to NaN instead of looking like a normal extremum.
trait NanProp: Copy {
    fn nmax(self, other: Self) -> Self;
    fn nmin(self, other: Self) -> Self;
}

macro_rules! impl_nan_prop {
    ($t:ty) => {
        impl NanProp for $t {
            #[inline]
            fn nmax(self, other: Self) -> Self {
                // exactly one of the comparisons holds unless a NaN is involved, which then falls through to `+`
                if self >= other { self } else if other > self { other } else { self + other }
            }
            #[inline]
            fn nmin(self, other: Self) -> Self {
                if self <= other { self } else if other < self { other } else { self + other }
            }
        }
    };
}
impl_nan_prop!(f32);
impl_nan_prop!(f64);

// `torch.remainder` / `torch.floor_divide` kernels (c10::div_floor_floating and the modulo adjustment), exact per element
// instead of `a - floor(a / b) * b`, which is off by one near exact multiples.
macro_rules! float_division_kernels {
    ($t:ty, $remainder:ident, $floor_div:ident) => {
        #[inline]
        fn $remainder(x: $t, y: $t) -> $t {
            let m = x % y;
            if m != 0.0 && ((m < 0.0) != (y < 0.0)) { m + y } else { m }
        }

        #[inline]
        fn $floor_div(x: $t, y: $t) -> $t {
            if y == 0.0 {
                return x / y;
            }
            let m = x % y;
            let mut div = (x - m) / y;
            if m != 0.0 && ((y < 0.0) != (m < 0.0)) {
                div -= 1.0;
            }
            if div != 0.0 {
                let fl = div.floor();
                if div - fl > 0.5 { fl + 1.0 } else { fl }
            } else {
                (0.0 as $t).copysign(x / y)
            }
        }
    };
}
float_division_kernels!(f32, remainder_f32, floor_div_f32);
float_division_kernels!(f64, remainder_f64, floor_div_f64);

#[inline]
fn int_remainder_i64(x: i64, y: i64) -> i64 {
    let r = x.wrapping_rem(y);
    if r != 0 && ((r < 0) != (y < 0)) { r + y } else { r }
}

#[inline]
fn int_floor_div_i64(x: i64, y: i64) -> i64 {
    let (q, r) = (x.wrapping_div(y), x.wrapping_rem(y));
    if r != 0 && ((r < 0) != (y < 0)) { q - 1 } else { q }
}

#[inline]
fn eval_unary_c128(x: [f64; 2], op: UnaryOp) -> [f64; 2] {
    let (re, im) = (x[0], x[1]);
    match op {
        UnaryOp::Neg => [-re, -im],
        UnaryOp::Square => [re * re - im * im, 2.0 * re * im],
        UnaryOp::Recip => {
            let d = re * re + im * im;
            [re / d, -im / d]
        }
        UnaryOp::Exp => {
            let e = re.exp();
            [e * im.cos(), e * im.sin()]
        }
        UnaryOp::Ln => [re.hypot(im).ln(), im.atan2(re)],
        UnaryOp::Sqrt => {
            let r = re.hypot(im);
            let sr = ((r + re) * 0.5).max(0.0).sqrt();
            let si = ((r - re) * 0.5).max(0.0).sqrt().copysign(im);
            [sr, si]
        }
        UnaryOp::Sin => [re.sin() * im.cosh(), re.cos() * im.sinh()],
        UnaryOp::Cos => [re.cos() * im.cosh(), -re.sin() * im.sinh()],
        UnaryOp::Tanh => {
            let (s, c) = (re.sin() * im.cosh(), re.cos() * im.sinh());
            let (cs, cc) = (re.cos() * im.cosh(), -re.sin() * im.sinh());
            let (sr, si) = (re.sinh() * im.cos(), re.cosh() * im.sin());
            let (cr, ci) = (re.cosh() * im.cos(), re.sinh() * im.sin());
            let _ = (s, c, cs, cc);
            let d = cr * cr + ci * ci;
            [(sr * cr + si * ci) / d, (si * cr - sr * ci) / d]
        }
        UnaryOp::AddScalar(s) => [re + s, im],
        UnaryOp::MulScalar(s) => [re * s, im * s],
        _ => [re, im],
    }
}

#[inline]
fn eval_unary_c64(x: [f32; 2], op: UnaryOp) -> [f32; 2] {
    let [r, i] = eval_unary_c128([x[0] as f64, x[1] as f64], op);
    [r as f32, i as f32]
}

#[inline]
fn eval_bin_c128(a: [f64; 2], b: [f64; 2], op: BinaryOp) -> [f64; 2] {
    match op {
        BinaryOp::Add => [a[0] + b[0], a[1] + b[1]],
        BinaryOp::Sub => [a[0] - b[0], a[1] - b[1]],
        BinaryOp::Mul => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
        BinaryOp::Div => {
            let d = b[0] * b[0] + b[1] * b[1];
            [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
        }
        BinaryOp::Pow => {
            if a[0] == 0.0 && a[1] == 0.0 && b[0] == 0.0 && b[1] == 0.0 {
                return [1.0, 0.0];
            }
            let ln_r = a[0].hypot(a[1]).ln();
            let ln_i = a[1].atan2(a[0]);
            let exp_r = b[0] * ln_r - b[1] * ln_i;
            let exp_i = b[0] * ln_i + b[1] * ln_r;
            let e = exp_r.exp();
            [e * exp_i.cos(), e * exp_i.sin()]
        }
        _ => [f64::NAN, f64::NAN],
    }
}

#[inline]
fn eval_bin_c64(a: [f32; 2], b: [f32; 2], op: BinaryOp) -> [f32; 2] {
    let [r, i] = eval_bin_c128([a[0] as f64, a[1] as f64], [b[0] as f64, b[1] as f64], op);
    [r as f32, i as f32]
}
