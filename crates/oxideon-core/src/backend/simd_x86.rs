//! Runtime-dispatched x86_64 AVX2 + FMA vector kernels for `f32`.
//!
//! All public functions check `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`
//! at runtime and return `false` / `None` on CPUs without AVX2+FMA so callers fall back cleanly.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Vectorized `exp(x)` on 8 `f32` lanes using AVX2 (matching `fast_exp_f32` constants & range).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp_ps_avx2(x: __m256) -> __m256 {
    let hi_bound = _mm256_set1_ps(89.0);
    let lo_bound = _mm256_set1_ps(-150.0);
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let c1 = _mm256_set1_ps(0.693_359_4);
    let c2 = _mm256_set1_ps(-2.121_944_4e-4);
    let round = _mm256_set1_ps(12_582_912.0);
    let bias = _mm256_set1_epi32(127);

    // Match `x.max(-150.0).min(89.0)` (clamp finite range; NaN restored at end)
    let xc = _mm256_min_ps(_mm256_max_ps(x, lo_bound), hi_bound);
    let n = _mm256_sub_ps(_mm256_add_ps(_mm256_mul_ps(xc, log2e), round), round);
    let r = _mm256_sub_ps(_mm256_sub_ps(xc, _mm256_mul_ps(n, c1)), _mm256_mul_ps(n, c2));

    let mut p = _mm256_set1_ps(1.987_569_1e-4);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.398_2e-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(8.333_452e-3));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(4.166_579_6e-2));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.666_666_5e-1));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(5.0e-1));
    let y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(p, r), r), r), _mm256_set1_ps(1.0));

    let ni = _mm256_cvttps_epi32(n);
    let n1 = _mm256_srai_epi32(ni, 1);
    let n2 = _mm256_sub_epi32(ni, n1);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, bias), 23));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, bias), 23));
    let out = _mm256_mul_ps(_mm256_mul_ps(y, s1), s2);

    let isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    _mm256_blendv_ps(out, x, isnan)
}

/// Vectorized standard normal CDF `0.5 * erfc(-x / sqrt(2))` on 8 `f32` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn norm_cdf_ps_avx2(x: __m256) -> __m256 {
    let neg_inv_sqrt2 = _mm256_set1_ps(-std::f32::consts::FRAC_1_SQRT_2);
    let u = _mm256_mul_ps(x, neg_inv_sqrt2);
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let z = _mm256_and_ps(u, abs_mask);

    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let t = _mm256_div_ps(one, _mm256_fmadd_ps(half, z, one));

    let mut poly = _mm256_set1_ps(0.170_872_77);
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-0.822_152_26));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.488_515_9));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-1.135_204));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(0.278_868_07));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-0.186_288_06));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(0.096_784_18));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(0.374_091_96));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(1.000_023_7));
    poly = _mm256_fmadd_ps(poly, t, _mm256_set1_ps(-1.265_512_2));

    let arg = _mm256_fnmadd_ps(z, z, poly);
    let erfc_abs = _mm256_mul_ps(t, unsafe { exp_ps_avx2(arg) });
    let neg_erfc = _mm256_sub_ps(two, erfc_abs);
    let u_neg = _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_LT_OQ);
    let erfc = _mm256_blendv_ps(erfc_abs, neg_erfc, u_neg);
    _mm256_mul_ps(half, erfc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let mut i = 0usize;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    while i + 8 <= n {
        let v = unsafe { _mm256_loadu_ps(sp.add(i)) };
        let r = unsafe { exp_ps_avx2(v) };
        unsafe { _mm256_storeu_ps(dp.add(i), r) };
        i += 8;
    }
    while i < n {
        unsafe { *dp.add(i) = super::cpu::fast_exp_f32(*sp.add(i)) };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let mut i = 0usize;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        let cdf = unsafe { norm_cdf_ps_avx2(x) };
        let r = _mm256_mul_ps(x, cdf);
        unsafe { _mm256_storeu_ps(dp.add(i), r) };
        i += 8;
    }
    while i < n {
        let x = unsafe { *sp.add(i) };
        unsafe { *dp.add(i) = x * super::cpu::fast_norm_cdf_f32(x) };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_backward_f32_avx2_impl(g: &[f32], x: &[f32], dst: &mut [f32]) {
    let n = x.len();
    let mut i = 0usize;
    let gp = g.as_ptr();
    let xp = x.as_ptr();
    let dp = dst.as_mut_ptr();
    let neg_half = _mm256_set1_ps(-0.5);
    let inv_sqrt2pi = _mm256_set1_ps(0.398_942_3);
    while i + 8 <= n {
        let gv = unsafe { _mm256_loadu_ps(gp.add(i)) };
        let xv = unsafe { _mm256_loadu_ps(xp.add(i)) };
        let cdf = unsafe { norm_cdf_ps_avx2(xv) };
        let x2 = _mm256_mul_ps(xv, xv);
        let pdf = _mm256_mul_ps(unsafe { exp_ps_avx2(_mm256_mul_ps(neg_half, x2)) }, inv_sqrt2pi);
        let deriv = _mm256_fmadd_ps(xv, pdf, cdf);
        let out = _mm256_mul_ps(gv, deriv);
        unsafe { _mm256_storeu_ps(dp.add(i), out) };
        i += 8;
    }
    while i < n {
        let xv = unsafe { *xp.add(i) };
        let gv = unsafe { *gp.add(i) };
        let cdf = super::cpu::fast_norm_cdf_f32(xv);
        let pdf = super::cpu::fast_exp_f32(-0.5 * xv * xv) * 0.398_942_3;
        unsafe { *dp.add(i) = gv * (cdf + xv * pdf) };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hmax256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(m128);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_movehl_ps(m64, m64);
    _mm_cvtss_f32(_mm_max_ss(m64, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f32_avx2_impl(src: &[f32]) -> f32 {
    let n = src.len();
    let sp = src.as_ptr();
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    while i + 32 <= n {
        acc0 = _mm256_add_ps(acc0, unsafe { _mm256_loadu_ps(sp.add(i)) });
        acc1 = _mm256_add_ps(acc1, unsafe { _mm256_loadu_ps(sp.add(i + 8)) });
        acc2 = _mm256_add_ps(acc2, unsafe { _mm256_loadu_ps(sp.add(i + 16)) });
        acc3 = _mm256_add_ps(acc3, unsafe { _mm256_loadu_ps(sp.add(i + 24)) });
        i += 32;
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    while i + 8 <= n {
        acc0 = _mm256_add_ps(acc0, unsafe { _mm256_loadu_ps(sp.add(i)) });
        i += 8;
    }
    let mut total = unsafe { hsum256_ps(acc0) };
    while i < n {
        total += unsafe { *sp.add(i) };
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_row_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();

    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 8 <= n {
        vmax = _mm256_max_ps(vmax, unsafe { _mm256_loadu_ps(sp.add(i)) });
        i += 8;
    }
    let mut m = unsafe { hmax256_ps(vmax) };
    while i < n {
        let x = unsafe { *sp.add(i) };
        if x > m {
            m = x;
        }
        i += 1;
    }

    let vm = _mm256_set1_ps(m);
    let mut vsum = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        let e = unsafe { exp_ps_avx2(_mm256_sub_ps(x, vm)) };
        unsafe { _mm256_storeu_ps(dp.add(i), e) };
        vsum = _mm256_add_ps(vsum, e);
        i += 8;
    }
    let mut sum = unsafe { hsum256_ps(vsum) };
    while i < n {
        let e = super::cpu::fast_exp_f32(unsafe { *sp.add(i) } - m);
        unsafe { *dp.add(i) = e };
        sum += e;
        i += 1;
    }

    let inv = 1.0f32 / sum;
    let vinv = _mm256_set1_ps(inv);
    i = 0;
    while i + 8 <= n {
        let e = unsafe { _mm256_loadu_ps(dp.add(i)) };
        unsafe { _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(e, vinv)) };
        i += 8;
    }
    while i < n {
        unsafe { *dp.add(i) *= inv };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_row_f32_avx2_impl(src: &[f32], w: &[f32], b: &[f32], dst: &mut [f32], eps: f32) {
    let n = src.len();
    let sp = src.as_ptr();
    let wp = w.as_ptr();
    let bp = b.as_ptr();
    let dp = dst.as_mut_ptr();

    let sum = unsafe { sum_f32_avx2_impl(src) };
    let mean = sum / (n as f32);
    let vmean = _mm256_set1_ps(mean);

    let mut vvar0 = _mm256_setzero_ps();
    let mut vvar1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let d0 = _mm256_sub_ps(unsafe { _mm256_loadu_ps(sp.add(i)) }, vmean);
        let d1 = _mm256_sub_ps(unsafe { _mm256_loadu_ps(sp.add(i + 8)) }, vmean);
        vvar0 = _mm256_fmadd_ps(d0, d0, vvar0);
        vvar1 = _mm256_fmadd_ps(d1, d1, vvar1);
        i += 16;
    }
    vvar0 = _mm256_add_ps(vvar0, vvar1);
    while i + 8 <= n {
        let d0 = _mm256_sub_ps(unsafe { _mm256_loadu_ps(sp.add(i)) }, vmean);
        vvar0 = _mm256_fmadd_ps(d0, d0, vvar0);
        i += 8;
    }
    let mut var_sum = unsafe { hsum256_ps(vvar0) };
    while i < n {
        let d = unsafe { *sp.add(i) } - mean;
        var_sum += d * d;
        i += 1;
    }

    let inv_std = 1.0f32 / (var_sum / (n as f32) + eps).sqrt();
    let vinv = _mm256_set1_ps(inv_std);
    i = 0;
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        let wv = unsafe { _mm256_loadu_ps(wp.add(i)) };
        let bv = unsafe { _mm256_loadu_ps(bp.add(i)) };
        let xhat = _mm256_mul_ps(_mm256_sub_ps(x, vmean), vinv);
        let out = _mm256_fmadd_ps(xhat, wv, bv);
        unsafe { _mm256_storeu_ps(dp.add(i), out) };
        i += 8;
    }
    while i < n {
        let x = unsafe { *sp.add(i) };
        let wv = unsafe { *wp.add(i) };
        let bv = unsafe { *bp.add(i) };
        unsafe { *dp.add(i) = (x - mean) * inv_std * wv + bv };
        i += 1;
    }
}

pub fn exp_f32_slice(src: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { exp_f32_avx2_impl(src, dst) };
        return true;
    }
    false
}

pub fn gelu_f32_slice(src: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { gelu_f32_avx2_impl(src, dst) };
        return true;
    }
    false
}

pub fn gelu_backward_f32_slice(g: &[f32], x: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(g.len(), x.len());
    debug_assert_eq!(x.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { gelu_backward_f32_avx2_impl(g, x, dst) };
        return true;
    }
    false
}

pub fn sum_f32_slice(src: &[f32]) -> Option<f32> {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        return Some(unsafe { sum_f32_avx2_impl(src) });
    }
    None
}

pub fn softmax_row_f32(src: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { softmax_row_f32_avx2_impl(src, dst) };
        return true;
    }
    false
}

pub fn layer_norm_row_f32(src: &[f32], w: &[f32], b: &[f32], dst: &mut [f32], eps: f32) -> bool {
    debug_assert_eq!(src.len(), w.len());
    debug_assert_eq!(src.len(), b.len());
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { layer_norm_row_f32_avx2_impl(src, w, b, dst, eps) };
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv2d_oc_block4_s1d1_avx2_impl(
    src: &[f32],
    w_block: &[f32],
    b_block: [f32; 4],
    out_block: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    ph: usize,
    pw: usize,
    out_h: usize,
    out_w: usize,
) {
    let l = out_h * out_w;
    let hw = h * w;
    let kdim = c_in * kh * kw;
    let sp = src.as_ptr();
    let wp = w_block.as_ptr();
    let op = out_block.as_mut_ptr();

    let w0 = wp;
    let w1 = unsafe { wp.add(kdim) };
    let w2 = unsafe { wp.add(2 * kdim) };
    let w3 = unsafe { wp.add(3 * kdim) };
    let (b0, b1, b2, b3) = (b_block[0], b_block[1], b_block[2], b_block[3]);
    let vb0 = _mm256_set1_ps(b0);
    let vb1 = _mm256_set1_ps(b1);
    let vb2 = _mm256_set1_ps(b2);
    let vb3 = _mm256_set1_ps(b3);

    let ow_safe_lo = pw;
    let ow_safe_hi = (w + pw).saturating_sub(kw - 1).min(out_w);
    let left_end = ow_safe_lo.min(out_w);

    for oh in 0..out_h {
        let out_row_off = oh * out_w;

        // Left border (scalar)
        for ow in 0..left_end {
            let (mut s0, mut s1, mut s2, mut s3) = (b0, b1, b2, b3);
            for ic in 0..c_in {
                let in_plane = unsafe { sp.add(ic * hw) };
                let k_off = ic * kh * kw;
                for r in 0..kh {
                    let ih = (oh + r) as isize - ph as isize;
                    if ih < 0 || ih as usize >= h {
                        continue;
                    }
                    let in_row = unsafe { in_plane.add(ih as usize * w) };
                    for s in 0..kw {
                        let iw = (ow + s) as isize - pw as isize;
                        if iw >= 0 && (iw as usize) < w {
                            let x = unsafe { *in_row.add(iw as usize) };
                            let ki = k_off + r * kw + s;
                            s0 += x * unsafe { *w0.add(ki) };
                            s1 += x * unsafe { *w1.add(ki) };
                            s2 += x * unsafe { *w2.add(ki) };
                            s3 += x * unsafe { *w3.add(ki) };
                        }
                    }
                }
            }
            unsafe {
                *op.add(out_row_off + ow) = s0;
                *op.add(l + out_row_off + ow) = s1;
                *op.add(2 * l + out_row_off + ow) = s2;
                *op.add(3 * l + out_row_off + ow) = s3;
            }
        }

        // Vectorized interior
        let mut ow = left_end;
        let row_inside = oh >= ph && oh + kh - 1 - ph < h;
        if kh == 3 && kw == 3 && row_inside {
            let ih0 = oh - ph;
            while ow + 24 <= ow_safe_hi {
                let (mut a00, mut a01, mut a02) = (vb0, vb0, vb0);
                let (mut a10, mut a11, mut a12) = (vb1, vb1, vb1);
                let (mut a20, mut a21, mut a22) = (vb2, vb2, vb2);
                let (mut a30, mut a31, mut a32) = (vb3, vb3, vb3);
                let iw_base = ow - pw;
                for ic in 0..c_in {
                    let in_plane = unsafe { sp.add(ic * hw + ih0 * w + iw_base) };
                    let k_off = ic * 9;
                    for r in 0..3 {
                        let in_row = unsafe { in_plane.add(r * w) };
                        for s in 0..3 {
                            let x0 = unsafe { _mm256_loadu_ps(in_row.add(s)) };
                            let x1 = unsafe { _mm256_loadu_ps(in_row.add(s + 8)) };
                            let x2 = unsafe { _mm256_loadu_ps(in_row.add(s + 16)) };
                            let ki = k_off + r * 3 + s;
                            let vw0 = _mm256_set1_ps(unsafe { *w0.add(ki) });
                            a00 = _mm256_fmadd_ps(x0, vw0, a00);
                            a01 = _mm256_fmadd_ps(x1, vw0, a01);
                            a02 = _mm256_fmadd_ps(x2, vw0, a02);
                            let vw1 = _mm256_set1_ps(unsafe { *w1.add(ki) });
                            a10 = _mm256_fmadd_ps(x0, vw1, a10);
                            a11 = _mm256_fmadd_ps(x1, vw1, a11);
                            a12 = _mm256_fmadd_ps(x2, vw1, a12);
                            let vw2 = _mm256_set1_ps(unsafe { *w2.add(ki) });
                            a20 = _mm256_fmadd_ps(x0, vw2, a20);
                            a21 = _mm256_fmadd_ps(x1, vw2, a21);
                            a22 = _mm256_fmadd_ps(x2, vw2, a22);
                            let vw3 = _mm256_set1_ps(unsafe { *w3.add(ki) });
                            a30 = _mm256_fmadd_ps(x0, vw3, a30);
                            a31 = _mm256_fmadd_ps(x1, vw3, a31);
                            a32 = _mm256_fmadd_ps(x2, vw3, a32);
                        }
                    }
                }
                unsafe {
                    _mm256_storeu_ps(op.add(out_row_off + ow), a00);
                    _mm256_storeu_ps(op.add(out_row_off + ow + 8), a01);
                    _mm256_storeu_ps(op.add(out_row_off + ow + 16), a02);
                    _mm256_storeu_ps(op.add(l + out_row_off + ow), a10);
                    _mm256_storeu_ps(op.add(l + out_row_off + ow + 8), a11);
                    _mm256_storeu_ps(op.add(l + out_row_off + ow + 16), a12);
                    _mm256_storeu_ps(op.add(2 * l + out_row_off + ow), a20);
                    _mm256_storeu_ps(op.add(2 * l + out_row_off + ow + 8), a21);
                    _mm256_storeu_ps(op.add(2 * l + out_row_off + ow + 16), a22);
                    _mm256_storeu_ps(op.add(3 * l + out_row_off + ow), a30);
                    _mm256_storeu_ps(op.add(3 * l + out_row_off + ow + 8), a31);
                    _mm256_storeu_ps(op.add(3 * l + out_row_off + ow + 16), a32);
                }
                ow += 24;
            }
        }
        while ow + 8 <= ow_safe_hi {
            let (mut acc0, mut acc1, mut acc2, mut acc3) = (vb0, vb1, vb2, vb3);
            let iw_base = ow - pw;
            for ic in 0..c_in {
                let in_plane = unsafe { sp.add(ic * hw) };
                let k_off = ic * kh * kw;
                for r in 0..kh {
                    let ih = (oh + r) as isize - ph as isize;
                    if ih < 0 || ih as usize >= h {
                        continue;
                    }
                    let in_row = unsafe { in_plane.add(ih as usize * w + iw_base) };
                    for s in 0..kw {
                        let xv = unsafe { _mm256_loadu_ps(in_row.add(s)) };
                        let ki = k_off + r * kw + s;
                        acc0 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w0.add(ki) }), acc0);
                        acc1 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w1.add(ki) }), acc1);
                        acc2 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w2.add(ki) }), acc2);
                        acc3 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w3.add(ki) }), acc3);
                    }
                }
            }
            unsafe {
                _mm256_storeu_ps(op.add(out_row_off + ow), acc0);
                _mm256_storeu_ps(op.add(l + out_row_off + ow), acc1);
                _mm256_storeu_ps(op.add(2 * l + out_row_off + ow), acc2);
                _mm256_storeu_ps(op.add(3 * l + out_row_off + ow), acc3);
            }
            ow += 8;
        }

        // Right tail (scalar)
        while ow < out_w {
            let (mut s0, mut s1, mut s2, mut s3) = (b0, b1, b2, b3);
            for ic in 0..c_in {
                let in_plane = unsafe { sp.add(ic * hw) };
                let k_off = ic * kh * kw;
                for r in 0..kh {
                    let ih = (oh + r) as isize - ph as isize;
                    if ih < 0 || ih as usize >= h {
                        continue;
                    }
                    let in_row = unsafe { in_plane.add(ih as usize * w) };
                    for s in 0..kw {
                        let iw = (ow + s) as isize - pw as isize;
                        if iw >= 0 && (iw as usize) < w {
                            let x = unsafe { *in_row.add(iw as usize) };
                            let ki = k_off + r * kw + s;
                            s0 += x * unsafe { *w0.add(ki) };
                            s1 += x * unsafe { *w1.add(ki) };
                            s2 += x * unsafe { *w2.add(ki) };
                            s3 += x * unsafe { *w3.add(ki) };
                        }
                    }
                }
            }
            unsafe {
                *op.add(out_row_off + ow) = s0;
                *op.add(l + out_row_off + ow) = s1;
                *op.add(2 * l + out_row_off + ow) = s2;
                *op.add(3 * l + out_row_off + ow) = s3;
            }
            ow += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_oc_block4_s1d1_f32(
    src: &[f32],
    w_block: &[f32],
    b_block: [f32; 4],
    out_block: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    ph: usize,
    pw: usize,
    out_h: usize,
    out_w: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() && out_w >= 8 {
        unsafe {
            conv2d_oc_block4_s1d1_avx2_impl(src, w_block, b_block, out_block, c_in, h, w, kh, kw, ph, pw, out_h, out_w);
        }
        return true;
    }
    false
}

/// Direct L2-resident AVX2+FMA 2D convolution for `stride=(1,1), dilation=(1,1), out_w >= 8`
/// on a single image (`src`: `[c_in, h, w]`, `wv`: `[c_out, c_in, kh, kw]`, `out`: `[c_out, out_h, out_w]`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv2d_image_s1d1_avx2_impl(
    src: &[f32],
    wv: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    ph: usize,
    pw: usize,
    out_h: usize,
    out_w: usize,
) {
    let l = out_h * out_w;
    let hw = h * w;
    let kdim = c_in * kh * kw;
    let sp = src.as_ptr();
    let wp = wv.as_ptr();
    let op = out.as_mut_ptr();

    let mut oc = 0usize;
    while oc + 4 <= c_out {
        let w0 = unsafe { wp.add(oc * kdim) };
        let w1 = unsafe { wp.add((oc + 1) * kdim) };
        let w2 = unsafe { wp.add((oc + 2) * kdim) };
        let w3 = unsafe { wp.add((oc + 3) * kdim) };
        let b0 = bias.map_or(0.0, |bv| bv[oc]);
        let b1 = bias.map_or(0.0, |bv| bv[oc + 1]);
        let b2 = bias.map_or(0.0, |bv| bv[oc + 2]);
        let b3 = bias.map_or(0.0, |bv| bv[oc + 3]);
        let vb0 = _mm256_set1_ps(b0);
        let vb1 = _mm256_set1_ps(b1);
        let vb2 = _mm256_set1_ps(b2);
        let vb3 = _mm256_set1_ps(b3);

        for oh in 0..out_h {
            let out_row_off = oh * out_w;
            // Fast 8-wide interior columns where `ow + s - pw` is in `[0, w - 1]` for all `s in 0..kw`
            let ow_safe_lo = pw;
            let ow_safe_hi = (w + pw).saturating_sub(kw - 1).min(out_w);

            // Left border (scalar)
            let left_end = ow_safe_lo.min(out_w);
            for ow in 0..left_end {
                let mut s0 = b0;
                let mut s1 = b1;
                let mut s2 = b2;
                let mut s3 = b3;
                for ic in 0..c_in {
                    let in_plane = unsafe { sp.add(ic * hw) };
                    let k_off = ic * kh * kw;
                    for r in 0..kh {
                        let ih = (oh + r) as isize - ph as isize;
                        if ih < 0 || ih as usize >= h {
                            continue;
                        }
                        let in_row = unsafe { in_plane.add(ih as usize * w) };
                        for s in 0..kw {
                            let iw = (ow + s) as isize - pw as isize;
                            if iw >= 0 && (iw as usize) < w {
                                let x = unsafe { *in_row.add(iw as usize) };
                                let ki = k_off + r * kw + s;
                                s0 += x * unsafe { *w0.add(ki) };
                                s1 += x * unsafe { *w1.add(ki) };
                                s2 += x * unsafe { *w2.add(ki) };
                                s3 += x * unsafe { *w3.add(ki) };
                            }
                        }
                    }
                }
                unsafe {
                    *op.add(oc * l + out_row_off + ow) = s0;
                    *op.add((oc + 1) * l + out_row_off + ow) = s1;
                    *op.add((oc + 2) * l + out_row_off + ow) = s2;
                    *op.add((oc + 3) * l + out_row_off + ow) = s3;
                }
            }

            // Vectorized 24-wide (3x8) + 8-wide interior (zero bounds checks!)
            let mut ow = left_end;
            while ow + 24 <= ow_safe_hi {
                let (mut a00, mut a01, mut a02) = (vb0, vb0, vb0);
                let (mut a10, mut a11, mut a12) = (vb1, vb1, vb1);
                let (mut a20, mut a21, mut a22) = (vb2, vb2, vb2);
                let (mut a30, mut a31, mut a32) = (vb3, vb3, vb3);
                for ic in 0..c_in {
                    let in_plane = unsafe { sp.add(ic * hw) };
                    let k_off = ic * kh * kw;
                    for r in 0..kh {
                        let ih = (oh + r) as isize - ph as isize;
                        if ih < 0 || ih as usize >= h {
                            continue;
                        }
                        let in_row = unsafe { in_plane.add(ih as usize * w) };
                        let iw_base = ow - pw;
                        for s in 0..kw {
                            let x0 = unsafe { _mm256_loadu_ps(in_row.add(iw_base + s)) };
                            let x1 = unsafe { _mm256_loadu_ps(in_row.add(iw_base + s + 8)) };
                            let x2 = unsafe { _mm256_loadu_ps(in_row.add(iw_base + s + 16)) };
                            let ki = k_off + r * kw + s;
                            let vw0 = _mm256_set1_ps(unsafe { *w0.add(ki) });
                            a00 = _mm256_fmadd_ps(x0, vw0, a00);
                            a01 = _mm256_fmadd_ps(x1, vw0, a01);
                            a02 = _mm256_fmadd_ps(x2, vw0, a02);
                            let vw1 = _mm256_set1_ps(unsafe { *w1.add(ki) });
                            a10 = _mm256_fmadd_ps(x0, vw1, a10);
                            a11 = _mm256_fmadd_ps(x1, vw1, a11);
                            a12 = _mm256_fmadd_ps(x2, vw1, a12);
                            let vw2 = _mm256_set1_ps(unsafe { *w2.add(ki) });
                            a20 = _mm256_fmadd_ps(x0, vw2, a20);
                            a21 = _mm256_fmadd_ps(x1, vw2, a21);
                            a22 = _mm256_fmadd_ps(x2, vw2, a22);
                            let vw3 = _mm256_set1_ps(unsafe { *w3.add(ki) });
                            a30 = _mm256_fmadd_ps(x0, vw3, a30);
                            a31 = _mm256_fmadd_ps(x1, vw3, a31);
                            a32 = _mm256_fmadd_ps(x2, vw3, a32);
                        }
                    }
                }
                unsafe {
                    _mm256_storeu_ps(op.add(oc * l + out_row_off + ow), a00);
                    _mm256_storeu_ps(op.add(oc * l + out_row_off + ow + 8), a01);
                    _mm256_storeu_ps(op.add(oc * l + out_row_off + ow + 16), a02);
                    _mm256_storeu_ps(op.add((oc + 1) * l + out_row_off + ow), a10);
                    _mm256_storeu_ps(op.add((oc + 1) * l + out_row_off + ow + 8), a11);
                    _mm256_storeu_ps(op.add((oc + 1) * l + out_row_off + ow + 16), a12);
                    _mm256_storeu_ps(op.add((oc + 2) * l + out_row_off + ow), a20);
                    _mm256_storeu_ps(op.add((oc + 2) * l + out_row_off + ow + 8), a21);
                    _mm256_storeu_ps(op.add((oc + 2) * l + out_row_off + ow + 16), a22);
                    _mm256_storeu_ps(op.add((oc + 3) * l + out_row_off + ow), a30);
                    _mm256_storeu_ps(op.add((oc + 3) * l + out_row_off + ow + 8), a31);
                    _mm256_storeu_ps(op.add((oc + 3) * l + out_row_off + ow + 16), a32);
                }
                ow += 24;
            }
            while ow + 8 <= ow_safe_hi {
                let mut acc0 = vb0;
                let mut acc1 = vb1;
                let mut acc2 = vb2;
                let mut acc3 = vb3;
                for ic in 0..c_in {
                    let in_plane = unsafe { sp.add(ic * hw) };
                    let k_off = ic * kh * kw;
                    for r in 0..kh {
                        let ih = (oh + r) as isize - ph as isize;
                        if ih < 0 || ih as usize >= h {
                            continue;
                        }
                        let in_row = unsafe { in_plane.add(ih as usize * w) };
                        let iw_base = ow - pw;
                        for s in 0..kw {
                            let xv = unsafe { _mm256_loadu_ps(in_row.add(iw_base + s)) };
                            let ki = k_off + r * kw + s;
                            acc0 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w0.add(ki) }), acc0);
                            acc1 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w1.add(ki) }), acc1);
                            acc2 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w2.add(ki) }), acc2);
                            acc3 = _mm256_fmadd_ps(xv, _mm256_set1_ps(unsafe { *w3.add(ki) }), acc3);
                        }
                    }
                }
                unsafe {
                    _mm256_storeu_ps(op.add(oc * l + out_row_off + ow), acc0);
                    _mm256_storeu_ps(op.add((oc + 1) * l + out_row_off + ow), acc1);
                    _mm256_storeu_ps(op.add((oc + 2) * l + out_row_off + ow), acc2);
                    _mm256_storeu_ps(op.add((oc + 3) * l + out_row_off + ow), acc3);
                }
                ow += 8;
            }

            // Right tail (scalar)
            while ow < out_w {
                let mut s0 = b0;
                let mut s1 = b1;
                let mut s2 = b2;
                let mut s3 = b3;
                for ic in 0..c_in {
                    let in_plane = unsafe { sp.add(ic * hw) };
                    let k_off = ic * kh * kw;
                    for r in 0..kh {
                        let ih = (oh + r) as isize - ph as isize;
                        if ih < 0 || ih as usize >= h {
                            continue;
                        }
                        let in_row = unsafe { in_plane.add(ih as usize * w) };
                        for s in 0..kw {
                            let iw = (ow + s) as isize - pw as isize;
                            if iw >= 0 && (iw as usize) < w {
                                let x = unsafe { *in_row.add(iw as usize) };
                                let ki = k_off + r * kw + s;
                                s0 += x * unsafe { *w0.add(ki) };
                                s1 += x * unsafe { *w1.add(ki) };
                                s2 += x * unsafe { *w2.add(ki) };
                                s3 += x * unsafe { *w3.add(ki) };
                            }
                        }
                    }
                }
                unsafe {
                    *op.add(oc * l + out_row_off + ow) = s0;
                    *op.add((oc + 1) * l + out_row_off + ow) = s1;
                    *op.add((oc + 2) * l + out_row_off + ow) = s2;
                    *op.add((oc + 3) * l + out_row_off + ow) = s3;
                }
                ow += 1;
            }
        }
        oc += 4;
    }

    // Remaining < 4 output channels
    while oc < c_out {
        let w0 = unsafe { wp.add(oc * kdim) };
        let b0 = bias.map_or(0.0, |bv| bv[oc]);
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut s0 = b0;
                for ic in 0..c_in {
                    let in_plane = unsafe { sp.add(ic * hw) };
                    let k_off = ic * kh * kw;
                    for r in 0..kh {
                        let ih = (oh + r) as isize - ph as isize;
                        if ih < 0 || ih as usize >= h {
                            continue;
                        }
                        let in_row = unsafe { in_plane.add(ih as usize * w) };
                        for s in 0..kw {
                            let iw = (ow + s) as isize - pw as isize;
                            if iw >= 0 && (iw as usize) < w {
                                s0 += unsafe { *in_row.add(iw as usize) * *w0.add(k_off + r * kw + s) };
                            }
                        }
                    }
                }
                unsafe { *op.add(oc * l + oh * out_w + ow) = s0 };
            }
        }
        oc += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn conv2d_image_s1d1_f32(
    src: &[f32],
    wv: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    ph: usize,
    pw: usize,
    out_h: usize,
    out_w: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() && out_w >= 8 {
        unsafe {
            conv2d_image_s1d1_avx2_impl(src, wv, bias, out, c_in, h, w, c_out, kh, kw, ph, pw, out_h, out_w);
        }
        return true;
    }
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_into_f32_avx2_impl(dst: &mut [f32], src: &[f32]) {
    let n = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let d0 = unsafe { _mm256_loadu_ps(dp.add(i)) };
        let s0 = unsafe { _mm256_loadu_ps(sp.add(i)) };
        let d1 = unsafe { _mm256_loadu_ps(dp.add(i + 8)) };
        let s1 = unsafe { _mm256_loadu_ps(sp.add(i + 8)) };
        unsafe {
            _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d0, s0));
            _mm256_storeu_ps(dp.add(i + 8), _mm256_add_ps(d1, s1));
        }
        i += 16;
    }
    while i + 8 <= n {
        let d0 = unsafe { _mm256_loadu_ps(dp.add(i)) };
        let s0 = unsafe { _mm256_loadu_ps(sp.add(i)) };
        unsafe { _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d0, s0)) };
        i += 8;
    }
    while i < n {
        unsafe { *dp.add(i) += *sp.add(i) };
        i += 1;
    }
}

pub fn add_into_f32_slice(dst: &mut [f32], src: &[f32]) -> bool {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { add_into_f32_avx2_impl(dst, src) };
        return true;
    }
    false
}

macro_rules! impl_bin_avx2 {
    ($fn_impl:ident, $fn_pub:ident, $mm:ident, $op:tt) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $fn_impl(a: &[f32], b: &[f32], dst: &mut [f32]) {
            let n = a.len();
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let dp = dst.as_mut_ptr();
            let mut i = 0usize;
            while i + 16 <= n {
                let a0 = unsafe { _mm256_loadu_ps(ap.add(i)) };
                let b0 = unsafe { _mm256_loadu_ps(bp.add(i)) };
                let a1 = unsafe { _mm256_loadu_ps(ap.add(i + 8)) };
                let b1 = unsafe { _mm256_loadu_ps(bp.add(i + 8)) };
                unsafe {
                    _mm256_storeu_ps(dp.add(i), $mm(a0, b0));
                    _mm256_storeu_ps(dp.add(i + 8), $mm(a1, b1));
                }
                i += 16;
            }
            while i + 8 <= n {
                let a0 = unsafe { _mm256_loadu_ps(ap.add(i)) };
                let b0 = unsafe { _mm256_loadu_ps(bp.add(i)) };
                unsafe { _mm256_storeu_ps(dp.add(i), $mm(a0, b0)) };
                i += 8;
            }
            while i < n {
                unsafe { *dp.add(i) = *ap.add(i) $op *bp.add(i) };
                i += 1;
            }
        }

        pub fn $fn_pub(a: &[f32], b: &[f32], dst: &mut [f32]) -> bool {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), dst.len());
            #[cfg(target_arch = "x86_64")]
            if has_avx2_fma() {
                unsafe { $fn_impl(a, b, dst) };
                return true;
            }
            false
        }
    };
}

impl_bin_avx2!(add_f32_avx2_impl, add_f32_slice, _mm256_add_ps, +);
impl_bin_avx2!(sub_f32_avx2_impl, sub_f32_slice, _mm256_sub_ps, -);
impl_bin_avx2!(mul_f32_avx2_impl, mul_f32_slice, _mm256_mul_ps, *);
impl_bin_avx2!(div_f32_avx2_impl, div_f32_slice, _mm256_div_ps, /);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sigmoid_ps_avx2(x: __m256) -> __m256 {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let neg_abs = _mm256_xor_ps(_mm256_and_ps(x, abs_mask), _mm256_set1_ps(-0.0));
    let e = unsafe { exp_ps_avx2(neg_abs) };
    let one = _mm256_set1_ps(1.0);
    let denom = _mm256_add_ps(one, e);
    let pos = _mm256_div_ps(one, denom);
    let neg = _mm256_mul_ps(e, pos);
    let is_neg = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    let res = _mm256_blendv_ps(pos, neg, is_neg);
    let isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    _mm256_blendv_ps(res, x, isnan)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        unsafe { _mm256_storeu_ps(dp.add(i), sigmoid_ps_avx2(x)) };
        i += 8;
    }
    while i < n {
        unsafe { *dp.add(i) = super::cpu::fast_sigmoid_f32(*sp.add(i)) };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        let s = unsafe { sigmoid_ps_avx2(x) };
        unsafe { _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(x, s)) };
        i += 8;
    }
    while i < n {
        let x = unsafe { *sp.add(i) };
        unsafe { *dp.add(i) = x * super::cpu::fast_sigmoid_f32(x) };
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn relu_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        let m = _mm256_max_ps(x, zero);
        let isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
        unsafe { _mm256_storeu_ps(dp.add(i), _mm256_blendv_ps(m, x, isnan)) };
        i += 8;
    }
    while i < n {
        let x = unsafe { *sp.add(i) };
        unsafe { *dp.add(i) = if x < 0.0 { 0.0 } else { x } };
        i += 1;
    }
}

pub fn sigmoid_f32_slice(src: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { sigmoid_f32_avx2_impl(src, dst) };
        return true;
    }
    false
}

pub fn silu_f32_slice(src: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { silu_f32_avx2_impl(src, dst) };
        return true;
    }
    false
}

pub fn relu_f32_slice(src: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { relu_f32_avx2_impl(src, dst) };
        return true;
    }
    false
}
