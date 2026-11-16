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
        ::gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            n as isize,
            false, // overwrite `c` instead of accumulating into it
            a.as_ptr().add(off_a),
            cs_a,
            rs_a,
            b.as_ptr().add(off_b),
            cs_b,
            rs_b,
            zero,
            one,
            false,
            false,
            false,
            parallelism(),
        );
    }
    true
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
    c.fill(0.0);
    if m == 0 || k == 0 || n == 0 {
        return;
    }

    // Small matrix fast-path: avoid packing overhead when total work is tiny.
    if m * k * n <= SMALL {
        gemm_f32_small(a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n);
        return;
    }

    #[cfg(feature = "fast-gemm")]
    if gemm_fast::<f32>(
        a, off_a, rs_a, cs_a, b, off_b, rs_b, cs_b, c, m, k, n, 0.0, 1.0,
    ) {
        return;
    }

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
    let mut out = vec![0.0f32; out_len];
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
