//! High-performance strided Batched GEMM (`f32` and `f64`).
//!
//! Features:
//! - Accepts arbitrary signed row/column strides `(rs_a, cs_a)` and `(rs_b, cs_b)` directly,
//!   so transposed, permuted, or sliced inputs (`A.t() @ B`, `A @ B.t()`) are multiplied
//!   without allocating temporary transpose copies.
//! - Cache-blocked GOTO-style packing (`MC x KC` and `KC x NC` panels) with an `8x8` AVX2+FMA
//!   register-blocked microkernel on x86_64 and a portable `4x8` microkernel fallback.
//! - Parallelizes across batches and M-tiles with `rayon` when the `rayon` feature is enabled.

#![allow(clippy::too_many_arguments)]

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::error::{Error, Result};

const MC: usize = 128;
const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 8;
const NR: usize = 8;

/// Largest product (`m * k * n`) the in-house kernels handle without packing; below it neither they nor the `gemm` crate pay off.
const SMALL: usize = 64 * 64 * 64;

/// Threads the `gemm` crate may use for one product: every rayon thread when the `rayon` feature is on (it keeps small
/// products on one thread by itself), otherwise one.
#[cfg(feature = "fast-gemm")]
fn parallelism() -> ::gemm::Parallelism {
    #[cfg(feature = "rayon")]
    {
        ::gemm::Parallelism::Rayon(0)
    }
    #[cfg(not(feature = "rayon"))]
    {
        ::gemm::Parallelism::None
    }
}

/// Whether every element `off + i * rs + j * cs` (`i < rows`, `j < cols`) lies inside a slice of `len` elements.
fn strided_within(len: usize, off: usize, rs: isize, cs: isize, rows: usize, cols: usize) -> bool {
    if rows == 0 || cols == 0 {
        return true;
    }
    let (Some(r), Some(c)) = (
        (rows as isize - 1).checked_mul(rs),
        (cols as isize - 1).checked_mul(cs),
    ) else {
        return false;
    };
    let (Some(lo), Some(hi)) = (
        (off as isize)
            .checked_add(r.min(0))
            .and_then(|v| v.checked_add(c.min(0))),
        (off as isize)
            .checked_add(r.max(0))
            .and_then(|v| v.checked_add(c.max(0))),
    ) else {
        return false;
    };
    lo >= 0 && (hi as usize) < len
}

/// The application-wide accuracy/speed profile for `f32` matrix products and everything built on them (linear layers, attention,
/// im2col convolutions). Pick one with [`set_precision_mode`], with the `OXIDEON_PRECISION` environment variable
/// (`fastest`, `balanced` or `highest`), or from Python with `oxideon.set_precision_mode("balanced")`.
///
/// Element-wise functions, reductions, softmax and layer norm are always computed in their accurate form: it costs next to nothing
/// there (the one exception is GELU, see [`PrecisionMode::Fastest`]). The modes otherwise trade speed for accuracy only where it really
/// costs: the matrix-product accumulation and the channel sums of the Winograd convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionMode {
    /// Speed first: plain `f32` accumulation, in 256-deep slices of `k` that each add onto the previous ones (a cache-blocking depth, as
    /// the `gemm` crate uses inside). The error of a `k = 512` product is about 2.6 half-ulps (normalized by the sum of the absolute
    /// terms), PyTorch-class. Also selects the cheaper five-term erfc for GELU on x86-64 AVX2 (about 1.5e-7 absolute error in the CDF;
    /// the backward kernel and other architectures always use their accurate form).
    Fastest,
    /// The default: the `k` loop is cut into 48-deep `f32` chains whose results are summed afterwards (groups of 32 chains inside the
    /// kernel, the group totals in `f64`). About 2.3x less worst-case and 1.7x less mean error than PyTorch's CPU matmul, at about the
    /// time of [`PrecisionMode::Fastest`] on most shapes (up to 1.3x on some).
    Balanced,
    /// Accuracy first: products and sums are carried out in `f64` and rounded to `f32` once. The error is that of the final rounding,
    /// an order of magnitude below PyTorch's, at about 2-2.5x the time of [`PrecisionMode::Fastest`] on square products.
    Highest,
}

impl PrecisionMode {
    /// Parses `fastest`, `balanced` or `highest` (case-insensitive; `fast`, `speed`, `high`, `accurate` and `exact` are accepted too).
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "fastest" | "fast" | "speed" => Some(Self::Fastest),
            "balanced" | "medium" => Some(Self::Balanced),
            "highest" | "high" | "accurate" | "exact" => Some(Self::Highest),
            _ => None,
        }
    }

    /// The canonical lower-case name, the inverse of [`PrecisionMode::parse`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Fastest => "fastest",
            Self::Balanced => "balanced",
            Self::Highest => "highest",
        }
    }
}

const MODE_UNSET: u8 = 255;
static PRECISION: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(MODE_UNSET);
/// Chains per group in [`gemm_chunked`]: the partials of this many consecutive chains are summed in f32 before the f64 combine.
const CHAIN_GROUP: usize = 32;
/// Depth of the `k` slices of `Fastest` (a cache-blocking depth: each slice accumulates onto the previous ones in `f32`, as the kernel does inside).
const FASTEST_SLICE: usize = 256;
static K_CHUNK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(48);

/// Selects the application-wide [`PrecisionMode`]. Process-wide and takes effect on the next matrix product; the default is
/// [`PrecisionMode::Balanced`] unless `OXIDEON_PRECISION` names another mode.
pub fn set_precision_mode(mode: PrecisionMode) {
    PRECISION.store(mode as u8, std::sync::atomic::Ordering::Relaxed);
}

/// The current [`PrecisionMode`]. On first use it is read from the `OXIDEON_PRECISION` environment variable
/// (`fastest` / `balanced` / `highest`), falling back to [`PrecisionMode::Balanced`] (an unrecognised value prints a warning to stderr).
pub fn precision_mode() -> PrecisionMode {
    let from = |v: u8| match v {
        0 => PrecisionMode::Fastest,
        2 => PrecisionMode::Highest,
        _ => PrecisionMode::Balanced,
    };
    let v = PRECISION.load(std::sync::atomic::Ordering::Relaxed);
    if v != MODE_UNSET {
        return from(v);
    }
    let mode = match std::env::var("OXIDEON_PRECISION") {
        Ok(n) => PrecisionMode::parse(&n).unwrap_or_else(|| {
            // a typo must not silently change the numerics
            eprintln!("oxideon: ignoring OXIDEON_PRECISION={n:?} (expected fastest, balanced or highest); using balanced");
            PrecisionMode::Balanced
        }),
        Err(_) => PrecisionMode::Balanced,
    };
    // a concurrent explicit `set_precision_mode` wins over the environment
    let _ = PRECISION.compare_exchange(MODE_UNSET, mode as u8, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed);
    from(PRECISION.load(std::sync::atomic::Ordering::Relaxed))
}

/// Sets the depth of the accumulation chains of [`PrecisionMode::Balanced`] (default 48; `0` never splits `k`).
///
/// For a `k = 512` product the worst error (float32 half-ulps, normalized by the sum of the absolute terms) is about 6.0
/// unsplit, 2.9 at 256, 1.6 at 128, 1.1 at 64 and 1.0 at 48; PyTorch's CPU matmul lands at about 2.3. Shorter chains cost throughput.
pub fn set_k_chunk(depth: usize) {
    K_CHUNK.store(depth, std::sync::atomic::Ordering::Relaxed);
}

/// The current chain depth of [`PrecisionMode::Balanced`], see [`set_k_chunk`].
pub fn k_chunk() -> usize {
    K_CHUNK.load(std::sync::atomic::Ordering::Relaxed)
}

// Work buffers of `gemm_chain_loop`'s wide mode, one set per thread and kept between calls: a fresh allocation per block and per
// product would fault in new pages every time.
#[cfg(feature = "fast-gemm")]
thread_local! {
    static CHAIN_BUFFERS: std::cell::RefCell<(Vec<f32>, Vec<f64>)> = const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Largest per-thread work buffer (elements) [`CHAIN_BUFFERS`] keeps; a bigger one is freed after use instead of being hoarded.
#[cfg(feature = "fast-gemm")]
const CHAIN_BUFFER_KEEP: usize = 1 << 18;

/// The serial chain loop of [`gemm_chunked`] for one block of the output (`chunks` slices of `k`, each accumulated onto the
/// previous ones). When `wide`, groups of [`CHAIN_GROUP`] chains are summed in `f32` and the group totals in `f64`.
///
/// # Safety
/// As for [`gemm_chunked`]; `c` is addressed with row stride `ldc`.
#[cfg(feature = "fast-gemm")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_chain_loop(
    m: usize,
    n: usize,
    ldc: isize,
    k: usize,
    c: *mut f32,
    a: *const f32,
    cs_a: isize,
    rs_a: isize,
    b: *const f32,
    cs_b: isize,
    rs_b: isize,
    chunks: usize,
    wide: bool,
) {
    let (base, extra) = (k / chunks, k % chunks);
    let (mut partial, mut acc) = if wide { CHAIN_BUFFERS.with(|b| std::mem::take(&mut *b.borrow_mut())) } else { (Vec::new(), Vec::new()) };
    if wide {
        // `partial` is fully written by the first chain of every group; `acc` starts from zero
        if partial.len() < m * n {
            partial.resize(m * n, 0.0);
        }
        acc.clear();
        acc.resize(m * n, 0.0);
    }
    let mut k0 = 0usize;
    for ci in 0..chunks {
        let kl = base + usize::from(ci < extra);
        let (dst, dst_ld, accumulate) = if wide { (partial.as_mut_ptr(), n as isize, ci % CHAIN_GROUP > 0) } else { (c, ldc, ci > 0) };
        // SAFETY: the slice of `k` [k0, k0 + kl) lies inside the extents the caller validated; `partial` holds m * n elements.
        unsafe {
            ::gemm::gemm(
                m,
                n,
                kl,
                dst,
                1,
                dst_ld,
                accumulate,
                a.offset(k0 as isize * cs_a),
                cs_a,
                rs_a,
                b.offset(k0 as isize * rs_b),
                cs_b,
                rs_b,
                if accumulate { 1.0 } else { 0.0 },
                1.0,
                false,
                false,
                false,
                ::gemm::Parallelism::None,
            );
        }
        if wide && (ci % CHAIN_GROUP == CHAIN_GROUP - 1 || ci + 1 == chunks) {
            for (x, &p) in acc.iter_mut().zip(&partial) {
                *x += f64::from(p);
            }
        }
        k0 += kl;
    }
    if wide {
        for i in 0..m {
            for j in 0..n {
                // SAFETY: (i, j) is inside the m x n block of `c`.
                unsafe { *c.offset(i as isize * ldc + j as isize) = acc[i * n + j] as f32 };
            }
        }
        if partial.len() <= CHAIN_BUFFER_KEEP && acc.len() <= CHAIN_BUFFER_KEEP {
            CHAIN_BUFFERS.with(|b| *b.borrow_mut() = (partial, acc));
        }
    }
}

/// `c = a @ b` through the `gemm` crate, accumulating `f32` products over `K_CHUNK`-deep slices of `k`.
///
/// # Safety
/// The extents of `a`, `b` and `c` must have been checked by the caller (`strided_within`), exactly as for `::gemm::gemm`.
#[cfg(feature = "fast-gemm")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_chunked<T: Copy + 'static>(
    m: usize,
    n: usize,
    k: usize,
    c: *mut T,
    a: *const T,
    cs_a: isize,
    rs_a: isize,
    b: *const T,
    cs_b: isize,
    rs_b: isize,
    zero: T,
    one: T,
    parallelism: ::gemm::Parallelism,
) {
    let is_f32 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>();
    let mut mode = precision_mode();
    // the f64 path holds three f64 copies of the operands and the result: beyond 1 GiB of them the products are chained in f32
    // instead (still accumulated in f64 across chunks) rather than risking an out-of-memory abort
    if mode == PrecisionMode::Highest && m.saturating_mul(k).saturating_add(k.saturating_mul(n)).saturating_add(m.saturating_mul(n)) > (1 << 27) {
        mode = PrecisionMode::Balanced;
    }
    if is_f32 && mode == PrecisionMode::Highest {
        // SAFETY: T is f32 here, and the caller validated the extents.
        unsafe { gemm_via_f64(m, n, k, c as *mut f32, a as *const f32, cs_a, rs_a, b as *const f32, cs_b, rs_b, parallelism) };
        return;
    }
    let kc = if mode == PrecisionMode::Fastest { FASTEST_SLICE } else { k_chunk() };
    let chunks = if is_f32 && kc > 0 && k > kc + kc / 2 { k.div_ceil(kc) } else { 1 };
    let (base, extra) = (k / chunks, k % chunks);
    // with many chunks the sum of their partial products would itself accumulate rounding error in f32 (a k = 8192 product
    // of 64-deep chunks has 128 of them): `Balanced` adds the partials in f64 instead. `Fastest` stays plain f32 throughout.
    let wide_chains = chunks > CHAIN_GROUP && mode != PrecisionMode::Fastest;
    // Inside a rayon worker (a batch of products, a parallel layer) the chains of a moderate product run serially on this thread: a
    // nested parallel region per chain would cost a synchronization every ~45 depth steps
    #[cfg(feature = "rayon")]
    if chunks > 1
        && is_f32
        && !matches!(parallelism, ::gemm::Parallelism::None)
        && rayon::current_thread_index().is_some()
        && m.saturating_mul(n).saturating_mul(k) <= 1 << 30
    {
        // SAFETY: T is f32, and the caller validated the extents; `c` is the m x n row-major output.
        unsafe { gemm_chain_loop(m, n, n as isize, k, c as *mut f32, a as *const f32, cs_a, rs_a, b as *const f32, cs_b, rs_b, chunks, wide_chains) };
        return;
    }
    // A big product: one parallel region over a grid of output blocks, each worker running all the chains of its block serially
    // (a parallel region per chain costs a synchronization every ~45 depth steps and keeps the block out of cache). Also used when
    // the whole depth is one chain (`Fastest`): a single `gemm` call splits skinny or deep products poorly.
    #[cfg(feature = "rayon")]
    if is_f32 && !matches!(parallelism, ::gemm::Parallelism::None) && rayon::current_thread_index().is_none() {
        let threads = rayon::current_num_threads();
        if threads > 1 && m.saturating_mul(n).saturating_mul(k) >= 1 << 20 {
            // a grid of gm x gn output blocks, about three per thread: the packing of the operand panels is paid once per block and
            // grows with (rows + columns) of the block, so the grid is the one that minimizes gn * m + gm * n
            let target = threads * 3;
            let (gm_max, gn_max) = ((m / 32).max(1), (n / 32).max(1));
            let (mut gm, mut gn, mut best) = (1usize, 1usize, f64::INFINITY);
            for g in 1..=gm_max.min(target) {
                let h = target.div_ceil(g).min(gn_max);
                let cost = (h * m + g * n) as f64 * target as f64 / (g * h).min(target) as f64;
                if cost < best {
                    (gm, gn, best) = (g, h, cost);
                }
            }
            let bm = m.div_ceil(gm).div_ceil(8) * 8;
            let bn = n.div_ceil(gn).div_ceil(16) * 16;
            let (gm, gn) = (m.div_ceil(bm), n.div_ceil(bn));
            if gm * gn > 1 {
                let (cp, ap, bp) = (c as usize, a as usize, b as usize);
                (0..gm * gn).into_par_iter().for_each(|t| {
                    let (i0, j0) = ((t / gn) * bm, (t % gn) * bn);
                    let (rows, cols) = (bm.min(m - i0), bn.min(n - j0));
                    let (c, a, b) = (cp as *mut f32, ap as *const f32, bp as *const f32);
                    // SAFETY: the blocks cover disjoint parts of `c` and read inside the validated extents of `a` and `b`.
                    unsafe {
                        gemm_chain_loop(rows, cols, n as isize, k, c.add(i0 * n + j0), a.offset(i0 as isize * rs_a), cs_a, rs_a, b.offset(j0 as isize * cs_b), cs_b, rs_b, chunks, wide_chains);
                    }
                });
                return;
            }
        }
    }
    let wide = wide_chains && m.saturating_mul(n) > 0;
    let mut partial: Vec<f32> = Vec::new();
    let mut acc: Vec<f64> = Vec::new();
    if wide {
        // recycled buffers: fresh multi-megabyte allocations cost more in page faults than the combine itself
        partial = crate::pool::take::<f32>(m * n).unwrap_or_else(|| vec![0.0; m * n]);
        acc = f64_buffer(m * n);
        acc.resize(m * n, 0.0);
    }
    let mut k0 = 0usize;
    for ci in 0..chunks {
        let kl = base + usize::from(ci < extra);
        // the first chunk (or every chunk, when summing in f64) overwrites its destination; later f32 chunks add to `c`
        let (dst, accumulate) = if wide { (partial.as_mut_ptr() as *mut T, ci % CHAIN_GROUP > 0) } else { (c, ci > 0) };
        // SAFETY: the slice of `k` [k0, k0 + kl) lies inside the extents the caller validated; `partial` holds m * n elements
        // (T is f32 whenever `wide` is set).
        unsafe {
            ::gemm::gemm(
                m,
                n,
                kl,
                dst,
                1,
                n as isize,
                accumulate,
                a.offset(k0 as isize * cs_a),
                cs_a,
                rs_a,
                b.offset(k0 as isize * rs_b),
                cs_b,
                rs_b,
                if accumulate { one } else { zero },
                one,
                false,
                false,
                false,
                parallelism,
            );
        }
        // a group of chains is summed inside the kernel (in f32, where the partial sums stay far below the sum of magnitudes); only
        // the group totals are added in f64, which keeps the cost of the combine at one pass per group
        if wide && (ci % CHAIN_GROUP == CHAIN_GROUP - 1 || ci + 1 == chunks) {
            crate::backend::cpu::add_f32_into_f64_par(&mut acc, &partial);
        }
        k0 += kl;
    }
    if wide {
        // SAFETY: `wide` implies f32, and `c` has room for m * n elements (the caller checked the extents).
        let out = unsafe { std::slice::from_raw_parts_mut(c as *mut f32, m * n) };
        #[cfg(feature = "rayon")]
        if m * n >= 1 << 15 {
            out.par_chunks_mut(8192).zip(acc.par_chunks(8192)).for_each(|(o, v)| {
                for (o, &x) in o.iter_mut().zip(v) {
                    *o = x as f32;
                }
            });
        } else {
            for (o, &v) in out.iter_mut().zip(&acc) {
                *o = v as f32;
            }
        }
        #[cfg(not(feature = "rayon"))]
        for (o, &v) in out.iter_mut().zip(&acc) {
            *o = v as f32;
        }
        crate::pool::give(partial);
        f64_recycle(acc);
    }
}

/// Recycled `f64` work buffers for [`gemm_via_f64`]: fresh multi-megabyte allocations spend more time in page faults than the
/// widening copies themselves. Small buffers (the per-item products of a batched matmul, one thread each) live in a per-thread
/// pool, so worker threads never contend for a lock; big ones go to a shared pool of at most four buffers and 256 MiB.
#[cfg(feature = "fast-gemm")]
static F64_POOL: std::sync::Mutex<Vec<Vec<f64>>> = std::sync::Mutex::new(Vec::new());

/// Largest buffer (in elements) kept in a thread's own pool: 16 MiB, at most four of them per thread.
#[cfg(feature = "fast-gemm")]
const LOCAL_POOL_ELEMS: usize = 2 << 20;

#[cfg(feature = "fast-gemm")]
thread_local! {
    static F64_LOCAL: std::cell::RefCell<Vec<Vec<f64>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The smallest pooled buffer that fits, so one huge buffer is not burned on a small product.
#[cfg(feature = "fast-gemm")]
fn take_fitting(pool: &mut Vec<Vec<f64>>, len: usize) -> Option<Vec<f64>> {
    let best = pool.iter().enumerate().filter(|(_, v)| v.capacity() >= len).min_by_key(|(_, v)| v.capacity()).map(|(i, _)| i)?;
    let mut v = pool.swap_remove(best);
    v.clear();
    Some(v)
}

/// An empty buffer with room for `len` elements, from a pool when one is large enough.
#[cfg(feature = "fast-gemm")]
fn f64_buffer(len: usize) -> Vec<f64> {
    if len <= LOCAL_POOL_ELEMS {
        if let Some(v) = F64_LOCAL.with(|p| take_fitting(&mut p.borrow_mut(), len)) {
            return v;
        }
        // round small requests up so a thread's buffers fit the next, slightly larger item too
        return Vec::with_capacity(len.next_power_of_two().min(LOCAL_POOL_ELEMS).max(len));
    }
    take_fitting(&mut F64_POOL.lock().unwrap_or_else(|e| e.into_inner()), len).unwrap_or_else(|| Vec::with_capacity(len))
}

/// Hands a work buffer back to its pool.
#[cfg(feature = "fast-gemm")]
fn f64_recycle(v: Vec<f64>) {
    if v.capacity() <= LOCAL_POOL_ELEMS {
        F64_LOCAL.with(|p| {
            let mut p = p.borrow_mut();
            if p.len() < 6 {
                p.push(v);
            }
        });
        return;
    }
    let mut pool = F64_POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|b| std::cmp::Reverse(b.capacity()));
    let mut total = 0usize;
    let mut keep = 0usize;
    for b in pool.iter() {
        total += b.capacity() * 8;
        if keep == 4 || total > 256 << 20 {
            break;
        }
        keep += 1;
    }
    pool.truncate(keep);
}

/// Largest `m` handled by [`gemm_small_m_f64`].
#[cfg(feature = "fast-gemm")]
const SMALL_M: usize = 8;

/// Up to this many rows the skinny route also wins when the right-hand matrix is big (at least 4M elements): the product is then
/// limited by the bytes read, and the `f64` copy the general route makes of `b` moves five times as many.
#[cfg(feature = "fast-gemm")]
const MEDIUM_M: usize = 32;

/// `c = a @ b` for a handful of rows (single-token inference, small batches) in `f64`, without widening the big right-hand matrix:
/// a skinny product is bandwidth-bound, and an `f64` copy of `b` would move five times the bytes the product needs. `false` when `b`
/// has neither contiguous rows nor contiguous columns (the caller then takes the general route).
///
/// # Safety
/// Same contract as [`gemm_chunked`].
#[cfg(feature = "fast-gemm")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_small_m_f64(m: usize, n: usize, k: usize, c: *mut f32, a: *const f32, cs_a: isize, rs_a: isize, b: *const f32, cs_b: isize, rs_b: isize, parallelism: ::gemm::Parallelism) -> bool {
    use crate::backend::cpu::{axpy_f64_of_f32, dot_f64_of_f32};
    let serial = matches!(parallelism, ::gemm::Parallelism::None);
    // the few rows of `a`, contiguous
    let mut abuf: Vec<f32> = Vec::with_capacity(m * k);
    for i in 0..m {
        for p in 0..k {
            // SAFETY: (i, p) lies inside the validated extent of `a`.
            abuf.push(unsafe { *a.offset(i as isize * rs_a + p as isize * cs_a) });
        }
    }
    let (cp, bp) = (c as usize, b as usize);
    if rs_b == 1 {
        // columns of `b` are contiguous (a transposed weight): one f64 dot per output element
        let col_task = |j: usize| {
            // SAFETY: column j of `b` holds k contiguous elements inside the validated extent; each task owns column j of `c`.
            let col = unsafe { std::slice::from_raw_parts((bp as *const f32).offset(j as isize * cs_b), k) };
            for i in 0..m {
                unsafe { *(cp as *mut f32).add(i * n + j) = dot_f64_of_f32(&abuf[i * k..(i + 1) * k], col) as f32 };
            }
        };
        #[cfg(feature = "rayon")]
        if !serial && n * k >= 1 << 15 {
            (0..n).into_par_iter().with_min_len(16).for_each(col_task);
            return true;
        }
        let _ = serial;
        (0..n).for_each(col_task);
        return true;
    }
    if cs_b == 1 {
        // rows of `b` are contiguous: f64 accumulators over a block of columns, one axpy per (row of b, row of a)
        const BLOCK: usize = 512;
        let block_task = |jb: usize| {
            let j0 = jb * BLOCK;
            let w = BLOCK.min(n - j0);
            let mut acc = vec![0.0f64; m * w];
            for p in 0..k {
                // SAFETY: row p of `b`, columns j0..j0 + w, is inside the validated extent.
                let brow = unsafe { std::slice::from_raw_parts((bp as *const f32).offset(p as isize * rs_b + j0 as isize), w) };
                for i in 0..m {
                    axpy_f64_of_f32(&mut acc[i * w..(i + 1) * w], brow, abuf[i * k + p] as f64);
                }
            }
            for i in 0..m {
                for (j, &v) in acc[i * w..(i + 1) * w].iter().enumerate() {
                    // SAFETY: each task owns columns j0..j0 + w of every row of `c`.
                    unsafe { *(cp as *mut f32).add(i * n + j0 + j) = v as f32 };
                }
            }
        };
        let blocks = n.div_ceil(BLOCK);
        #[cfg(feature = "rayon")]
        if !serial && n * k >= 1 << 15 {
            (0..blocks).into_par_iter().for_each(block_task);
            return true;
        }
        let _ = serial;
        (0..blocks).for_each(block_task);
        return true;
    }
    false
}

/// `c = a @ b` for `f32` operands computed in `f64` and rounded once.
///
/// # Safety
/// Same contract as [`gemm_chunked`].
#[cfg(feature = "fast-gemm")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_via_f64(m: usize, n: usize, k: usize, c: *mut f32, a: *const f32, cs_a: isize, rs_a: isize, b: *const f32, cs_b: isize, rs_b: isize, parallelism: ::gemm::Parallelism) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // SAFETY: `c` has room for m * n elements.
        unsafe { std::slice::from_raw_parts_mut(c, m * n) }.fill(0.0);
        return;
    }
    if (m <= SMALL_M || (m <= MEDIUM_M && n.saturating_mul(k) >= 1 << 22)) && unsafe { gemm_small_m_f64(m, n, k, c, a, cs_a, rs_a, b, cs_b, rs_b, parallelism) } {
        return;
    }
    // inside a rayon worker (the items of a batched matmul) a product of modest size stays on its thread: nesting the widening and
    // the kernel's own parallelism there costs several times what the product does
    #[cfg(feature = "rayon")]
    let in_worker = rayon::current_thread_index().is_some() && m.saturating_mul(n).saturating_mul(k) <= 1 << 30;
    #[cfg(not(feature = "rayon"))]
    let in_worker = false;
    let serial = in_worker || matches!(parallelism, ::gemm::Parallelism::None);
    let parallelism = if serial { ::gemm::Parallelism::None } else { parallelism };
    // widen the (possibly strided) operands into contiguous row-major f64 matrices; every element is written before it is read
    let (ap, bp) = (a as usize, b as usize);
    let widen = |rows: usize, cols: usize, base: usize, rs: isize, cs: isize| -> Vec<f64> {
        let mut v: Vec<f64> = f64_buffer(rows * cols);
        let fill = |(i, row): (usize, &mut [f64])| {
            let src = base as *const f32;
            if cs == 1 {
                // SAFETY: row i of the operand holds `cols` contiguous elements inside the validated extent.
                let s = unsafe { std::slice::from_raw_parts(src.offset(i as isize * rs), cols) };
                for (d, &x) in row.iter_mut().zip(s) {
                    *d = x as f64;
                }
            } else {
                for (j, d) in row.iter_mut().enumerate() {
                    // SAFETY: (i, j) lies inside the validated extent of the operand.
                    *d = unsafe { *src.offset(i as isize * rs + j as isize * cs) } as f64;
                }
            }
        };
        // SAFETY: the spare capacity is fully overwritten by `fill` below (f64 has no invalid bit patterns).
        let spare = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr(), rows * cols) };
        #[cfg(feature = "rayon")]
        if !serial && rows * cols >= 1 << 15 {
            // tasks of ~16k elements: a task per row would drown a skinny matrix (4096 rows of 64) in scheduling
            let rows_per = (1usize << 14).div_ceil(cols).max(1);
            spare.par_chunks_mut(rows_per * cols).enumerate().for_each(|(bi, blk)| {
                for (r, row) in blk.chunks_mut(cols).enumerate() {
                    fill((bi * rows_per + r, row));
                }
            });
            unsafe { v.set_len(rows * cols) };
            return v;
        }
        spare.chunks_mut(cols).enumerate().for_each(fill);
        unsafe { v.set_len(rows * cols) };
        v
    };
    if serial {
        // one thread (the items of a batched matmul): row tiles keep the f64 result in cache instead of writing the whole `m x n` f64
        // matrix out and reading it back, which dominates the cost when `k` is small
        let tile = (65536 / n).clamp(32, 256);
        if m > tile {
            let b64 = widen(k, n, bp, rs_b, cs_b);
            let out = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
            let mut i0 = 0usize;
            while i0 < m {
                let h = tile.min(m - i0);
                // SAFETY: rows i0..i0 + h of `a` lie inside the validated extent.
                let a64 = widen(h, k, unsafe { a.offset(i0 as isize * rs_a) } as usize, rs_a, cs_a);
                let mut c64: Vec<f64> = f64_buffer(h * n);
                // SAFETY: contiguous row-major buffers of the stated shapes; `read_dst = false` never reads c64.
                unsafe {
                    ::gemm::gemm(h, n, k, c64.as_mut_ptr(), 1, n as isize, false, a64.as_ptr(), 1, k as isize, b64.as_ptr(), 1, n as isize, 0.0f64, 1.0f64, false, false, false, ::gemm::Parallelism::None);
                    c64.set_len(h * n);
                }
                for (o, &v) in out[i0 * n..(i0 + h) * n].iter_mut().zip(&c64) {
                    *o = v as f32;
                }
                f64_recycle(a64);
                f64_recycle(c64);
                i0 += h;
            }
            f64_recycle(b64);
            return;
        }
    }
    let a64 = widen(m, k, ap, rs_a, cs_a);
    let b64 = widen(k, n, bp, rs_b, cs_b);
    let mut c64: Vec<f64> = f64_buffer(m * n);
    // SAFETY: the three buffers are contiguous row-major of the stated shapes; with `read_dst = false` the product never reads c64.
    unsafe {
        let (c64p, a64p, b64p) = (c64.as_mut_ptr() as usize, a64.as_ptr() as usize, b64.as_ptr() as usize);
        let strip = |j0: usize, w: usize, par: ::gemm::Parallelism| {
            // the strip covers columns j0..j0 + w of every row of c64 and b64 (row stride n); strips never overlap
            ::gemm::gemm(m, w, k, (c64p as *mut f64).add(j0), 1, n as isize, false, a64p as *const f64, 1, k as isize, (b64p as *const f64).add(j0), 1, n as isize, 0.0f64, 1.0f64, false, false, false, par);
        };
        // the f64 kernel splits its work by rows of `c`, which leaves most cores idle for a few dozen rows: such products are cut into
        // column strips instead, one single-threaded product per strip
        #[cfg(feature = "rayon")]
        if m <= 128 && n.saturating_mul(k) >= 1 << 22 && !matches!(parallelism, ::gemm::Parallelism::None) {
            let w = n.div_ceil(rayon::current_num_threads().max(1)).clamp(128, 1024).next_multiple_of(8);
            (0..n.div_ceil(w)).into_par_iter().for_each(|t| strip(t * w, w.min(n - t * w), ::gemm::Parallelism::None));
        } else {
            strip(0, n, parallelism);
        }
        #[cfg(not(feature = "rayon"))]
        strip(0, n, parallelism);
        c64.set_len(m * n);
    }
    // SAFETY: `c` has room for m * n elements (the caller checked the extents).
    let out = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
    #[cfg(feature = "rayon")]
    if !serial && m * n >= 1 << 15 {
        out.par_chunks_mut(8192).zip(c64.par_chunks(8192)).for_each(|(o, v)| {
            for (o, &x) in o.iter_mut().zip(v) {
                *o = x as f32;
            }
        });
    } else {
        for (o, &v) in out.iter_mut().zip(&c64) {
            *o = v as f32;
        }
    }
    #[cfg(not(feature = "rayon"))]
    for (o, &v) in out.iter_mut().zip(&c64) {
        *o = v as f32;
    }
    f64_recycle(a64);
    f64_recycle(b64);
    f64_recycle(c64);
}

/// `c = a @ b` through the `gemm` crate (row-major `c`, arbitrary signed strides for `a` and `b`). `false` when the strided
/// extents do not fit the slices, so the caller falls back to the checked in-house kernels.
#[cfg(feature = "fast-gemm")]
fn gemm_fast<T: Copy + 'static>(
    a: &[T],
    off_a: usize,
    rs_a: isize,
    cs_a: isize,
    b: &[T],
    off_b: usize,
    rs_b: isize,
    cs_b: isize,
    c: &mut [T],
    m: usize,
    k: usize,
    n: usize,
    zero: T,
    one: T,
) -> bool {
    // `saturating_mul`: a wrapping `m * n` here would admit a short `c` into the
    // `unsafe` kernel below (out-of-bounds write); saturation refuses instead.
    if c.len() < m.saturating_mul(n)
        || !strided_within(a.len(), off_a, rs_a, cs_a, m, k)
        || !strided_within(b.len(), off_b, rs_b, cs_b, k, n)
    {
        return false;
    }
    // SAFETY: the extents of `a`, `b` and `c` were checked above, so every address the kernel computes from the base pointers
    // and the strides lies inside the three slices; `c` is exclusively borrowed and does not overlap the inputs.
    unsafe {
        gemm_chunked(m, n, k, c.as_mut_ptr(), a.as_ptr().add(off_a), cs_a, rs_a, b.as_ptr().add(off_b), cs_b, rs_b, zero, one, parallelism());
    }
    true
}

/// `dst[i * k + j] = a[off + i + j * cs]`: copies a column-major `m x k` operand into row-major order, in 32-row bands (in
/// parallel when there is enough work) so both the reads and the writes stay inside cache lines.
fn transpose_pack_lhs(a: &[f32], off: usize, cs: usize, m: usize, k: usize, dst: &mut [f32]) {
    const BAND: usize = 32;
    let band = |bi: usize, rows: &mut [f32]| {
        let i0 = bi * BAND;
        let nrows = rows.len() / k;
        for jb in (0..k).step_by(BAND) {
            let j1 = (jb + BAND).min(k);
            for j in jb..j1 {
                let col = &a[off + j * cs + i0..off + j * cs + i0 + nrows];
                for (r, &v) in col.iter().enumerate() {
                    rows[r * k + j] = v;
                }
            }
        }
    };
    #[cfg(feature = "rayon")]
    if m * k >= 1 << 16 {
        dst[..m * k].par_chunks_mut(BAND * k).enumerate().for_each(|(bi, rows)| band(bi, rows));
        return;
    }
    for (bi, rows) in dst[..m * k].chunks_mut(BAND * k).enumerate() {
        band(bi, rows);
    }
}

/// Computes `C = A @ B` for a single 2D matrix pair `A: [m, k]` (with strides `rs_a, cs_a`
/// and base offset `off_a`), `B: [k, n]` (with strides `rs_b, cs_b` and base offset `off_b`),
/// writing into contiguous row-major `c: &mut [f32]` of shape `[m, n]`.
///
/// Callers must pass in-bounds offsets/strides/dims (checked in debug builds);
/// out-of-bounds access panics on slice indexing rather than corrupting memory.
///
/// # Panics
/// Panics when an input view reads outside its slice or `c` is short. Prefer
/// [`try_gemm_f32_strided`] when the geometry comes from untrusted input.
pub fn gemm_f32_strided(
    a: &[f32],
    off_a: usize,
    rs_a: isize,
    cs_a: isize,
    b: &[f32],
    off_b: usize,
    rs_b: isize,
    cs_b: isize,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_strided_bounds("gemm A", a.len(), off_a, rs_a, cs_a, m, k);
    debug_assert_strided_bounds("gemm B", b.len(), off_b, rs_b, cs_b, k, n);
    debug_assert!(
        c.len() >= m.saturating_mul(n),
        "gemm C holds {} elements, needs {}",
        c.len(),
        m.saturating_mul(n)
    );
    if m == 0 || k == 0 || n == 0 {
        c.fill(0.0);
        return;
    }

    // Small matrix fast-path: avoid packing overhead when total work is tiny.
    if m * k * n <= SMALL {
        c.fill(0.0);
        gemm_f32_small(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n);
        return;
    }

    // A column-major left operand (a transposed view, as in `x^T @ grad` of every linear layer's backward pass) makes the
    // packing inside the kernel read with a large stride; one blocked transpose up front is cheaper than that for big products
    #[cfg(feature = "fast-gemm")]
    if rs_a == 1 && cs_a >= m as isize && m >= 64 && k >= 64 && m.saturating_mul(k).saturating_mul(n) >= (1 << 23) {
        let mut packed = crate::pool::take::<f32>(m * k).unwrap_or_else(|| vec![0.0; m * k]);
        transpose_pack_lhs(a, off_a, cs_a as usize, m, k, &mut packed);
        let done = gemm_fast::<f32>(&packed, 0, k as isize, 1, b, off_b, rs_b, cs_b, c, m, k, n, 0.0, 1.0);
        crate::pool::give(packed);
        if done {
            return;
        }
    }

    #[cfg(feature = "fast-gemm")]
    if gemm_fast::<f32>(
        a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n, 0.0, 1.0,
    ) {
        return;
    }

    c.fill(0.0);

    #[cfg(feature = "rayon")]
    if m >= 128 && m * k * n >= 128 * 128 * 128 {
        c.par_chunks_mut(MC * n)
            .enumerate()
            .for_each(|(blk_idx, c_chunk)| {
                let ic = blk_idx * MC;
                let mc = (m - ic).min(MC);
                let off_a_tile = (off_a as isize + (ic as isize) * rs_a) as usize;
                gemm_f32_macro_m(
                    a, off_a_tile, rs_a, cs_a, b, off_b, rs_b, cs_b, c_chunk, mc, k, n,
                );
            });
        return;
    }

    gemm_f32_macro_m(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n);
}

/// Single-threaded strided `f32` GEMM for use inside outer `rayon` parallel loops (avoids thread oversubscription).
pub fn gemm_f32_strided_seq(
    a: &[f32],
    off_a: usize,
    rs_a: isize,
    cs_a: isize,
    b: &[f32],
    off_b: usize,
    rs_b: isize,
    cs_b: isize,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if m == 0 || k == 0 || n == 0 {
        c.fill(0.0);
        return;
    }
    if m * k * n <= SMALL {
        c.fill(0.0);
        gemm_f32_small(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n);
        return;
    }
    #[cfg(feature = "fast-gemm")]
    {
        let a_ok = strided_within(a.len(), off_a, rs_a, cs_a, m, k);
        let b_ok = strided_within(b.len(), off_b, rs_b, cs_b, k, n);
        if a_ok && b_ok {
            unsafe {
                gemm_chunked(m, n, k, c.as_mut_ptr(), a.as_ptr().add(off_a), cs_a, rs_a, b.as_ptr().add(off_b), cs_b, rs_b, 0.0f32, 1.0f32, ::gemm::Parallelism::None);
            }
            return;
        }
    }
    c.fill(0.0);
    gemm_f32_macro_m(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n);
}

/// Validates one strided product's extents against its slices: the output must
/// hold `m * n` elements and every element `off + i * rs + j * cs` of both
/// inputs must lie inside its slice. All arithmetic is checked, so overflowing
/// dims/strides are refused instead of wrapping into a wrong verdict.
#[allow(clippy::too_many_arguments)]
fn check_strided_args(
    op: &'static str,
    a_len: usize,
    off_a: usize,
    rs_a: isize,
    cs_a: isize,
    b_len: usize,
    off_b: usize,
    rs_b: isize,
    cs_b: isize,
    c_len: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    let out_n = m
        .checked_mul(n)
        .ok_or_else(|| Error::Msg(format!("{op}: output size {m} x {n} overflows usize")))?;
    if c_len < out_n {
        return Err(Error::Msg(format!(
            "{op}: output holds {c_len} elements, needs {m} x {n} = {out_n}"
        )));
    }
    if !strided_within(a_len, off_a, rs_a, cs_a, m, k) {
        return Err(Error::Msg(format!(
            "{op}: A view (offset {off_a}, strides {rs_a}/{cs_a}, {m} x {k}) reads outside its {a_len}-element slice"
        )));
    }
    if !strided_within(b_len, off_b, rs_b, cs_b, k, n) {
        return Err(Error::Msg(format!(
            "{op}: B view (offset {off_b}, strides {rs_b}/{cs_b}, {k} x {n}) reads outside its {b_len}-element slice"
        )));
    }
    Ok(())
}

/// Fallible [`gemm_f32_strided`]: validates offsets/strides/slice lengths up
/// front and returns `Err` on out-of-bounds geometry instead of panicking deep
/// in the kernels.
pub fn try_gemm_f32_strided(
    a: &[f32],
    off_a: usize,
    rs_a: isize,
    cs_a: isize,
    b: &[f32],
    off_b: usize,
    rs_b: isize,
    cs_b: isize,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    check_strided_args(
        "gemm_f32_strided",
        a.len(),
        off_a,
        rs_a,
        cs_a,
        b.len(),
        off_b,
        rs_b,
        cs_b,
        c.len(),
        m,
        k,
        n,
    )?;
    gemm_f32_strided(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n);
    Ok(())
}

/// Fallible [`batched_gemm_f32`]: validates every batch's offsets/strides up
/// front (and refuses outputs that can never fit) instead of panicking mid-run.
pub fn try_batched_gemm_f32(
    a: &[f32],
    rs_a: isize,
    cs_a: isize,
    b: &[f32],
    rs_b: isize,
    cs_b: isize,
    batch_offsets: &[(usize, usize)],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let stride_c = m.checked_mul(n).ok_or_else(|| {
        Error::Msg(format!(
            "batched_gemm_f32: output size {m} x {n} overflows usize"
        ))
    })?;
    crate::limits::check_elements("batched_gemm_f32", &[batch_offsets.len(), m, n])?;
    for &(off_a, off_b) in batch_offsets {
        check_strided_args(
            "batched_gemm_f32",
            a.len(),
            off_a,
            rs_a,
            cs_a,
            b.len(),
            off_b,
            rs_b,
            cs_b,
            stride_c,
            m,
            k,
            n,
        )?;
    }
    Ok(batched_gemm_f32(
        a,
        rs_a,
        cs_a,
        b,
        rs_b,
        cs_b,
        batch_offsets,
        m,
        k,
        n,
    ))
}

/// Fallible [`batched_gemm_f64`]: validates every batch's offsets/strides up
/// front (and refuses outputs that can never fit) instead of panicking mid-run.
pub fn try_batched_gemm_f64(
    a: &[f64],
    rs_a: isize,
    cs_a: isize,
    b: &[f64],
    rs_b: isize,
    cs_b: isize,
    batch_offsets: &[(usize, usize)],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f64>> {
    let stride_c = m.checked_mul(n).ok_or_else(|| {
        Error::Msg(format!(
            "batched_gemm_f64: output size {m} x {n} overflows usize"
        ))
    })?;
    crate::limits::check_elements("batched_gemm_f64", &[batch_offsets.len(), m, n])?;
    for &(off_a, off_b) in batch_offsets {
        check_strided_args(
            "batched_gemm_f64",
            a.len(),
            off_a,
            rs_a,
            cs_a,
            b.len(),
            off_b,
            rs_b,
            cs_b,
            stride_c,
            m,
            k,
            n,
        )?;
    }
    Ok(batched_gemm_f64(
        a,
        rs_a,
        cs_a,
        b,
        rs_b,
        cs_b,
        batch_offsets,
        m,
        k,
        n,
    ))
}

fn gemm_f32_small(
    a: &[f32],
    off_a: usize,
    rs_a: isize,
    cs_a: isize,
    b: &[f32],
    off_b: usize,
    rs_b: isize,
    cs_b: isize,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    if cs_b == 1 {
        // Inner row of B is contiguous -> i-p-j loop auto-vectorizes cleanly.
        for i in 0..m {
            let c_row = &mut c[i * n..(i + 1) * n];
            let a_row_base = off_a as isize + (i as isize) * rs_a;
            for p in 0..k {
                let a_ip = a[(a_row_base + (p as isize) * cs_a) as usize];
                let b_row_start = (off_b as isize + (p as isize) * rs_b) as usize;
                let b_row = &b[b_row_start..b_row_start + n];
                for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                    *c_ij += a_ip * b_pj;
                }
            }
        }
    } else {
        for i in 0..m {
            let c_row = &mut c[i * n..(i + 1) * n];
            let a_row_base = off_a as isize + (i as isize) * rs_a;
            for p in 0..k {
                let a_ip = a[(a_row_base + (p as isize) * cs_a) as usize];
                let b_row_base = off_b as isize + (p as isize) * rs_b;
                for (j, c_ij) in c_row.iter_mut().enumerate() {
                    let b_pj = b[(b_row_base + (j as isize) * cs_b) as usize];
                    *c_ij += a_ip * b_pj;
                }
            }
        }
    }
}

fn gemm_f32_macro_m(
    a: &[f32],
    off_a: usize,
    rs_a: isize,
    cs_a: isize,
    b: &[f32],
    off_b: usize,
    rs_b: isize,
    cs_b: isize,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let m_padded = m.div_ceil(MR) * MR;
    let n_padded = NC.div_ceil(NR) * NR;
    let mut pack_a = vec![0.0f32; MC.max(m_padded) * KC];
    let mut pack_b = vec![0.0f32; KC * n_padded];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");

    let mut ic = 0;
    while ic < m {
        let mc = (m - ic).min(MC);
        let mut pc = 0;
        while pc < k {
            let kc = (k - pc).min(KC);
            pack_panel_a_f32(
                a,
                (off_a as isize + (ic as isize) * rs_a + (pc as isize) * cs_a) as usize,
                rs_a,
                cs_a,
                mc,
                kc,
                &mut pack_a,
            );

            let mut jc = 0;
            while jc < n {
                let nc = (n - jc).min(NC);
                pack_panel_b_f32(
                    b,
                    (off_b as isize + (pc as isize) * rs_b + (jc as isize) * cs_b) as usize,
                    rs_b,
                    cs_b,
                    kc,
                    nc,
                    &mut pack_b,
                );

                let ir_blocks = mc.div_ceil(MR);
                let jr_blocks = nc.div_ceil(NR);
                for ir in 0..ir_blocks {
                    let i0 = ir * MR;
                    let mr = (mc - i0).min(MR);
                    let pa = &pack_a[ir * kc * MR..(ir + 1) * kc * MR];
                    for jr in 0..jr_blocks {
                        let j0 = jr * NR;
                        let nr = (nc - j0).min(NR);
                        let pb = &pack_b[jr * kc * NR..(jr + 1) * kc * NR];

                        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                        if use_avx2 && mr == MR && nr == NR {
                            // SAFETY: `use_avx2` verified AVX2 + FMA support at runtime, and
                            // `mr == 8`, `nr == 8` guarantees all row slices are within `c`.
                            unsafe {
                                microkernel_avx2_8x8(
                                    kc,
                                    pa.as_ptr(),
                                    pb.as_ptr(),
                                    c.as_mut_ptr().add((ic + i0) * n + (jc + j0)),
                                    n,
                                );
                            }
                            continue;
                        }

                        microkernel_portable_8x8(
                            kc,
                            pa,
                            pb,
                            &mut c[(ic + i0) * n + (jc + j0)..],
                            n,
                            mr,
                            nr,
                        );
                    }
                }
                jc += NC;
            }
            pc += KC;
        }
        ic += MC;
    }
}

fn pack_panel_a_f32(
    a: &[f32],
    base: usize,
    rs_a: isize,
    cs_a: isize,
    mc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    let blocks = mc.div_ceil(MR);
    for br in 0..blocks {
        let i0 = br * MR;
        let mr = (mc - i0).min(MR);
        let mut out_idx = br * kc * MR;
        for p in 0..kc {
            let col_off = (p as isize) * cs_a;
            for r in 0..mr {
                let idx = (base as isize + ((i0 + r) as isize) * rs_a + col_off) as usize;
                dst[out_idx + r] = a[idx];
            }
            for r in mr..MR {
                dst[out_idx + r] = 0.0;
            }
            out_idx += MR;
        }
    }
}

fn pack_panel_b_f32(
    b: &[f32],
    base: usize,
    rs_b: isize,
    cs_b: isize,
    kc: usize,
    nc: usize,
    dst: &mut [f32],
) {
    let blocks = nc.div_ceil(NR);
    for bc in 0..blocks {
        let j0 = bc * NR;
        let nr = (nc - j0).min(NR);
        let mut out_idx = bc * kc * NR;
        for p in 0..kc {
            let row_off = (p as isize) * rs_b;
            for c in 0..nr {
                let idx = (base as isize + row_off + ((j0 + c) as isize) * cs_b) as usize;
                dst[out_idx + c] = b[idx];
            }
            for c in nr..NR {
                dst[out_idx + c] = 0.0;
            }
            out_idx += NR;
        }
    }
}

#[inline(always)]
fn microkernel_portable_8x8(
    kc: usize,
    pa: &[f32],
    pb: &[f32],
    c: &mut [f32],
    ldc: usize,
    mr: usize,
    nr: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..kc {
        let a_col = &pa[p * MR..p * MR + MR];
        let b_row = &pb[p * NR..p * NR + NR];
        for i in 0..MR {
            let a_ip = a_col[i];
            for j in 0..NR {
                acc[i][j] = a_ip.mul_add(b_row[j], acc[i][j]);
            }
        }
    }
    for i in 0..mr {
        let row = &mut c[i * ldc..i * ldc + nr];
        for j in 0..nr {
            row[j] += acc[i][j];
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_avx2_8x8(
    kc: usize,
    mut pa: *const f32,
    mut pb: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut c2 = _mm256_setzero_ps();
        let mut c3 = _mm256_setzero_ps();
        let mut c4 = _mm256_setzero_ps();
        let mut c5 = _mm256_setzero_ps();
        let mut c6 = _mm256_setzero_ps();
        let mut c7 = _mm256_setzero_ps();

        for _ in 0..kc {
            let b_vec = _mm256_loadu_ps(pb);
            c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(0)), b_vec, c0);
            c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(1)), b_vec, c1);
            c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(2)), b_vec, c2);
            c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(3)), b_vec, c3);
            c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(4)), b_vec, c4);
            c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(5)), b_vec, c5);
            c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(6)), b_vec, c6);
            c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*pa.add(7)), b_vec, c7);
            pa = pa.add(MR);
            pb = pb.add(NR);
        }

        let r0 = c;
        let r1 = c.add(ldc);
        let r2 = c.add(2 * ldc);
        let r3 = c.add(3 * ldc);
        let r4 = c.add(4 * ldc);
        let r5 = c.add(5 * ldc);
        let r6 = c.add(6 * ldc);
        let r7 = c.add(7 * ldc);

        _mm256_storeu_ps(r0, _mm256_add_ps(_mm256_loadu_ps(r0), c0));
        _mm256_storeu_ps(r1, _mm256_add_ps(_mm256_loadu_ps(r1), c1));
        _mm256_storeu_ps(r2, _mm256_add_ps(_mm256_loadu_ps(r2), c2));
        _mm256_storeu_ps(r3, _mm256_add_ps(_mm256_loadu_ps(r3), c3));
        _mm256_storeu_ps(r4, _mm256_add_ps(_mm256_loadu_ps(r4), c4));
        _mm256_storeu_ps(r5, _mm256_add_ps(_mm256_loadu_ps(r5), c5));
        _mm256_storeu_ps(r6, _mm256_add_ps(_mm256_loadu_ps(r6), c6));
        _mm256_storeu_ps(r7, _mm256_add_ps(_mm256_loadu_ps(r7), c7));
    }
}

/// Debug-only bounds contract for strided GEMM inputs (extreme access is always
/// at a corner since the index is linear in both loop variables).
fn debug_assert_strided_bounds(
    what: &str,
    len: usize,
    off: usize,
    rs: isize,
    cs: isize,
    rows: usize,
    cols: usize,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    let corners = [
        off as i128,
        off as i128 + (rows as i128 - 1) * rs as i128,
        off as i128 + (cols as i128 - 1) * cs as i128,
        off as i128 + (rows as i128 - 1) * rs as i128 + (cols as i128 - 1) * cs as i128,
    ];
    let lo = corners.iter().min().copied().unwrap_or(0);
    let hi = corners.iter().max().copied().unwrap_or(0);
    debug_assert!(
        lo >= 0 && hi < len as i128,
        "{what}: strided access [{lo}, {hi}] out of bounds for len {len}"
    );
}

/// Batched strided `f32` GEMM across a slice of `(off_a, off_b)` pairs.
///
/// # Panics
/// Panics on out-of-bounds batch offsets or overflowing output sizes; prefer
/// [`try_batched_gemm_f32`] for fallible validation up front.
pub fn batched_gemm_f32(
    a: &[f32],
    rs_a: isize,
    cs_a: isize,
    b: &[f32],
    rs_b: isize,
    cs_b: isize,
    batch_offsets: &[(usize, usize)],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let nb = batch_offsets.len();
    let stride_c = m.checked_mul(n).expect("batched_gemm_f32: m * n overflows");
    let out_len = nb
        .checked_mul(stride_c)
        .expect("batched_gemm_f32: batch output size overflows");
    // a recycled buffer is fine: every GEMM path below writes (or first zeroes) all of its output
    let mut out = crate::pool::take::<f32>(out_len).unwrap_or_else(|| vec![0.0f32; out_len]);
    if nb == 0 || stride_c == 0 {
        return out;
    }

    #[cfg(feature = "rayon")]
    if nb > 1 && nb * m * k * n >= 32 * 32 * 32 {
        out.par_chunks_mut(stride_c)
            .zip(batch_offsets.par_iter())
            .for_each(|(c_slice, &(off_a, off_b))| {
                gemm_f32_strided(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c_slice, m, k, n);
            });
        return out;
    }

    for (c_slice, &(off_a, off_b)) in out.chunks_mut(stride_c).zip(batch_offsets) {
        gemm_f32_strided(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c_slice, m, k, n);
    }
    out
}

/// Batched strided `f64` GEMM.
///
/// # Panics
/// Panics on out-of-bounds batch offsets or overflowing output sizes; prefer
/// [`try_batched_gemm_f64`] for fallible validation up front.
pub fn batched_gemm_f64(
    a: &[f64],
    rs_a: isize,
    cs_a: isize,
    b: &[f64],
    rs_b: isize,
    cs_b: isize,
    batch_offsets: &[(usize, usize)],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    let nb = batch_offsets.len();
    let stride_c = m.checked_mul(n).expect("batched_gemm_f64: m * n overflows");
    let out_len = nb
        .checked_mul(stride_c)
        .expect("batched_gemm_f64: batch output size overflows");
    let mut out = vec![0.0f64; out_len];
    if nb == 0 || stride_c == 0 || k == 0 {
        return out;
    }

    let run_one = |c_slice: &mut [f64], off_a: usize, off_b: usize| {
        #[cfg(feature = "fast-gemm")]
        if m * k * n > SMALL
            && gemm_fast::<f64>(
                a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c_slice, m, k, n, 0.0, 1.0,
            )
        {
            return;
        }
        for i in 0..m {
            let c_row = &mut c_slice[i * n..(i + 1) * n];
            let a_row_base = off_a as isize + (i as isize) * rs_a;
            for p in 0..k {
                let a_ip = a[(a_row_base + (p as isize) * cs_a) as usize];
                let b_row_base = off_b as isize + (p as isize) * rs_b;
                if cs_b == 1 {
                    let b_start = b_row_base as usize;
                    for (c_ij, &b_pj) in c_row.iter_mut().zip(&b[b_start..b_start + n]) {
                        *c_ij += a_ip * b_pj;
                    }
                } else {
                    for (j, c_ij) in c_row.iter_mut().enumerate() {
                        *c_ij += a_ip * b[(b_row_base + (j as isize) * cs_b) as usize];
                    }
                }
            }
        }
    };

    #[cfg(feature = "rayon")]
    if nb > 1 && nb * m * k * n >= 32 * 32 * 32 {
        out.par_chunks_mut(stride_c)
            .zip(batch_offsets.par_iter())
            .for_each(|(c_slice, &(off_a, off_b))| run_one(c_slice, off_a, off_b));
        return out;
    }

    for (c_slice, &(off_a, off_b)) in out.chunks_mut(stride_c).zip(batch_offsets) {
        run_one(c_slice, off_a, off_b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in `[-1, 1)`.
    fn values(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f64 / (1u64 << 23) as f64) - 1.0
            })
            .collect()
    }

    /// A `rows x cols` matrix stored four ways: `(data, offset, row stride, column stride)`.
    fn layouts(rows: usize, cols: usize, seed: u64) -> Vec<(Vec<f64>, usize, isize, isize)> {
        let plain = values(rows * cols, seed);
        let (r, c) = (rows as isize, cols as isize);
        let mut padded = vec![9.0; 7];
        padded.extend(&plain);
        vec![
            (plain.clone(), 0, c, 1),         // row-major
            (plain.clone(), 0, 1, r),         // the transpose's storage: column-major
            (padded, 7, c, 1),                // an offset into a larger buffer
            (plain, rows * cols - 1, -c, -1), // both axes flipped (negative strides)
        ]
    }

    fn reference(
        a: &(Vec<f64>, usize, isize, isize),
        b: &(Vec<f64>, usize, isize, isize),
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f64> {
        let at = |d: &(Vec<f64>, usize, isize, isize), i: usize, j: usize| {
            d.0[(d.1 as isize + i as isize * d.2 + j as isize * d.3) as usize]
        };
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k).map(|p| at(a, i, p) * at(b, p, j)).sum();
            }
        }
        c
    }

    /// Sizes on both sides of the small-matrix threshold (`m * k * n` = 262144), with odd edges.
    const SIZES: [(usize, usize, usize); 9] = [
        (1, 1, 1),
        (5, 7, 3),
        (65, 100, 33),
        (70, 70, 70),
        (130, 257, 129),
        (1, 600, 600),
        (600, 1, 600),
        (300, 300, 300),
        (129, 513, 65),
    ];

    #[test]
    fn f32_product_matches_a_float64_reference_for_every_layout() {
        for (m, k, n) in SIZES {
            for (ia, a) in layouts(m, k, 1).iter().enumerate() {
                for (ib, b) in layouts(k, n, 2).iter().enumerate() {
                    let a32: Vec<f32> = a.0.iter().map(|&x| x as f32).collect();
                    let b32: Vec<f32> = b.0.iter().map(|&x| x as f32).collect();
                    let mut c = vec![f32::NAN; m * n]; // must be fully overwritten
                    gemm_f32_strided(&a32, a.1, a.2, a.3, &b32, b.1, b.2, b.3, &mut c, m, k, n);
                    let want = reference(a, b, m, k, n);
                    for (idx, (&got, &w)) in c.iter().zip(&want).enumerate() {
                        assert!(
                            (got as f64 - w).abs() <= 2e-3,
                            "f32 {m}x{k}x{n} layouts ({ia},{ib}) element {idx}: {got} vs {w}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn f64_batched_product_matches_the_reference_for_every_layout() {
        for (m, k, n) in SIZES {
            for (ia, a) in layouts(m, k, 3).iter().enumerate() {
                for (ib, b) in layouts(k, n, 4).iter().enumerate() {
                    let c =
                        batched_gemm_f64(&a.0, a.2, a.3, &b.0, b.2, b.3, &[(a.1, b.1)], m, k, n);
                    let want = reference(a, b, m, k, n);
                    for (idx, (&got, &w)) in c.iter().zip(&want).enumerate() {
                        assert!(
                            (got - w).abs() <= 1e-9,
                            "f64 {m}x{k}x{n} layouts ({ia},{ib}) element {idx}: {got} vs {w}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn batches_are_independent() {
        let (m, k, n, nb) = (70, 90, 80, 3);
        let a: Vec<f32> = values(nb * m * k, 5).iter().map(|&x| x as f32).collect();
        let b: Vec<f32> = values(nb * k * n, 6).iter().map(|&x| x as f32).collect();
        let offsets: Vec<(usize, usize)> = (0..nb).map(|i| (i * m * k, i * k * n)).collect();
        let c = batched_gemm_f32(&a, k as isize, 1, &b, n as isize, 1, &offsets, m, k, n);
        for (i, &(oa, ob)) in offsets.iter().enumerate() {
            let da = (
                a[oa..oa + m * k]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
                0,
                k as isize,
                1,
            );
            let db = (
                b[ob..ob + k * n]
                    .iter()
                    .map(|&x| x as f64)
                    .collect::<Vec<_>>(),
                0,
                n as isize,
                1,
            );
            let want = reference(&da, &db, m, k, n);
            for (j, &w) in want.iter().enumerate() {
                assert!(
                    (c[i * m * n + j] as f64 - w).abs() <= 2e-3,
                    "batch {i} element {j}"
                );
            }
        }
    }

    #[test]
    fn views_that_do_not_fit_the_slice_are_refused_not_read() {
        assert!(strided_within(12, 0, 4, 1, 3, 4));
        assert!(
            !strided_within(12, 0, 5, 1, 3, 4),
            "the last element is at index 13"
        );
        assert!(
            !strided_within(12, 0, -4, 1, 3, 4),
            "a negative stride from offset 0 leaves the slice"
        );
        assert!(strided_within(12, 11, -4, -1, 3, 4));
        assert!(
            strided_within(0, 0, 4, 1, 0, 4),
            "an empty view reads nothing"
        );
        assert!(
            !strided_within(12, 0, isize::MAX, 1, 3, 4),
            "overflowing strides are refused"
        );
    }
}
