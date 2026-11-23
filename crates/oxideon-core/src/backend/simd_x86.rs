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
    // fused multiply-adds shorten the dependent chain (the function is compiled with FMA enabled)
    let n = _mm256_sub_ps(_mm256_fmadd_ps(xc, log2e, round), round);
    let r = _mm256_fnmadd_ps(n, c2, _mm256_fnmadd_ps(n, c1, xc));

    // Estrin form of the degree-5 polynomial (a shorter dependency chain than Horner, same coefficients)
    let r2 = _mm256_mul_ps(r, r);
    let r4 = _mm256_mul_ps(r2, r2);
    let e0 = _mm256_fmadd_ps(_mm256_set1_ps(1.666_666_5e-1), r, _mm256_set1_ps(5.0e-1));
    let e1 = _mm256_fmadd_ps(_mm256_set1_ps(8.333_452e-3), r, _mm256_set1_ps(4.166_579_6e-2));
    let e2 = _mm256_fmadd_ps(_mm256_set1_ps(1.987_569_1e-4), r, _mm256_set1_ps(1.398_2e-3));
    let p = _mm256_fmadd_ps(e2, r4, _mm256_fmadd_ps(e1, r2, e0));
    // 1 + r + r^2 p with a single final rounding: `1 + r` is split into its rounded sum and the exact residual (|r| < 1) instead
    // of letting the rounding of `1 + r` add half an ulp to the result
    let one = _mm256_set1_ps(1.0);
    let s1 = _mm256_add_ps(r, one);
    let resid = _mm256_sub_ps(r, _mm256_sub_ps(s1, one));
    let y = _mm256_add_ps(s1, _mm256_fmadd_ps(_mm256_mul_ps(p, r), r, resid));

    let ni = _mm256_cvttps_epi32(n);
    let n1 = _mm256_srai_epi32(ni, 1);
    let n2 = _mm256_sub_epi32(ni, n1);
    let s1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, bias), 23));
    let s2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, bias), 23));
    let out = _mm256_mul_ps(_mm256_mul_ps(y, s1), s2);

    let isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    _mm256_blendv_ps(out, x, isnan)
}

/// [`exp_ps_avx2`] for arguments that are never positive (the gelu tail exponent `poly - z^2`): no upper clamp, and no NaN restore, so
/// a NaN argument gives a finite value here and the caller must let the NaN come from elsewhere (`x * cdf` does).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp_nonpos_ps_avx2(x: __m256) -> __m256 {
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let c1 = _mm256_set1_ps(0.693_359_4);
    let c2 = _mm256_set1_ps(-2.121_944_4e-4);
    let round = _mm256_set1_ps(12_582_912.0);
    let bias = _mm256_set1_epi32(127);
    let xc = _mm256_max_ps(x, _mm256_set1_ps(-150.0));
    let n = _mm256_sub_ps(_mm256_fmadd_ps(xc, log2e, round), round);
    let r = _mm256_fnmadd_ps(n, c2, _mm256_fnmadd_ps(n, c1, xc));
    let r2 = _mm256_mul_ps(r, r);
    let r4 = _mm256_mul_ps(r2, r2);
    let e0 = _mm256_fmadd_ps(_mm256_set1_ps(1.666_666_5e-1), r, _mm256_set1_ps(5.0e-1));
    let e1 = _mm256_fmadd_ps(_mm256_set1_ps(8.333_452e-3), r, _mm256_set1_ps(4.166_579_6e-2));
    let e2 = _mm256_fmadd_ps(_mm256_set1_ps(1.987_569_1e-4), r, _mm256_set1_ps(1.398_2e-3));
    let p = _mm256_fmadd_ps(e2, r4, _mm256_fmadd_ps(e1, r2, e0));
    let one = _mm256_set1_ps(1.0);
    let s1 = _mm256_add_ps(r, one);
    let resid = _mm256_sub_ps(r, _mm256_sub_ps(s1, one));
    let y = _mm256_add_ps(s1, _mm256_fmadd_ps(_mm256_mul_ps(p, r), r, resid));
    let ni = _mm256_cvttps_epi32(n);
    let n1 = _mm256_srai_epi32(ni, 1);
    let n2 = _mm256_sub_epi32(ni, n1);
    let sc1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, bias), 23));
    let sc2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, bias), 23));
    _mm256_mul_ps(_mm256_mul_ps(y, sc1), sc2)
}

/// `e^x` as an unevaluated sum `hi + lo` of two `f32` lanes (`hi` is the correctly rounded value up to the last bit, `lo` what the
/// rounding left behind), for callers that divide or subtract it and would otherwise add its half-ulp error to their own. Valid
/// where the result is a normal number (`-87 < x < 88`); outside it `lo` is meaningless.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp_hilo_ps_avx2(x: __m256) -> (__m256, __m256) {
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let c1 = _mm256_set1_ps(0.693_359_4);
    let c2 = _mm256_set1_ps(-2.121_944_4e-4);
    let round = _mm256_set1_ps(12_582_912.0);
    let bias = _mm256_set1_epi32(127);
    let xc = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0)), _mm256_set1_ps(88.0));
    let n = _mm256_sub_ps(_mm256_fmadd_ps(xc, log2e, round), round);
    // r = rh + rl: the reduced argument as a pair, so the reduction itself adds no rounding error
    let r1 = _mm256_fnmadd_ps(n, c1, xc);
    let rh = _mm256_fnmadd_ps(n, c2, r1);
    let rl = _mm256_fnmadd_ps(n, c2, _mm256_sub_ps(r1, rh)); // exact: (r1 - rh) - n*c2
    // e^rh = 1 + rh + rh^2/2 + rh^3 * q(rh); the leading terms are kept as exact pairs, only the (small) rh^3 term is rounded
    let mut q = _mm256_set1_ps(1.989_925_4e-4);
    q = _mm256_fmadd_ps(q, rh, _mm256_set1_ps(1.392_615_1e-3));
    q = _mm256_fmadd_ps(q, rh, _mm256_set1_ps(8.333_301e-3));
    q = _mm256_fmadd_ps(q, rh, _mm256_set1_ps(4.166_655_6e-2));
    q = _mm256_fmadd_ps(q, rh, _mm256_set1_ps(1.666_666_7e-1));
    let one = _mm256_set1_ps(1.0);
    let r2h = _mm256_mul_ps(rh, rh);
    let r2l = _mm256_fmsub_ps(rh, rh, r2h);
    let half = _mm256_set1_ps(0.5);
    let s1 = _mm256_add_ps(rh, one);
    let e1 = _mm256_sub_ps(rh, _mm256_sub_ps(s1, one)); // exact: s1 + e1 == 1 + rh (the rounding error of the sum; |1| >= |rh|)
    let h2 = _mm256_mul_ps(half, r2h);
    let y1 = _mm256_add_ps(s1, h2);
    let e2 = _mm256_sub_ps(h2, _mm256_sub_ps(y1, s1)); // Fast2Sum: |s1| >= |h2|
    let c3 = _mm256_mul_ps(_mm256_mul_ps(r2h, rh), q);
    // everything below the f32 grid of y1: rounding of s1 and h2, the low half of rh^2/2, the low part of r, and the cubic term
    let small = _mm256_add_ps(
        _mm256_add_ps(_mm256_add_ps(e1, e2), _mm256_fmadd_ps(half, r2l, _mm256_mul_ps(rl, s1))),
        c3,
    );
    let y = _mm256_add_ps(y1, small);
    let ylo = _mm256_add_ps(_mm256_sub_ps(y1, y), small); // Fast2Sum: |y1| >= |small|
    let ni = _mm256_cvttps_epi32(n);
    let n1 = _mm256_srai_epi32(ni, 1);
    let n2 = _mm256_sub_epi32(ni, n1);
    let sc1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n1, bias), 23));
    let sc2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(n2, bias), 23));
    (_mm256_mul_ps(_mm256_mul_ps(y, sc1), sc2), _mm256_mul_ps(_mm256_mul_ps(ylo, sc1), sc2))
}

/// `num / (1 + e)` for `e = eh + el` and a numerator `num` (itself exact), as `(q, corr)` with `q + corr` correct to well below a
/// half-ulp: one quotient, then a correction from the exact residual that also carries the rounding error of `1 + eh` and `el`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn div_one_plus_hilo(num: __m256, eh: __m256, el: __m256) -> (__m256, __m256) {
    let one = _mm256_set1_ps(1.0);
    let d = _mm256_add_ps(one, eh);
    let d_lo = _mm256_add_ps(_mm256_sub_ps(eh, _mm256_sub_ps(d, one)), el); // 1 + eh + el = d + d_lo
    let inv = _mm256_div_ps(one, d);
    let q0 = _mm256_mul_ps(num, inv);
    let resid = _mm256_fnmadd_ps(q0, d_lo, _mm256_fnmadd_ps(q0, d, num)); // num - q0 * (d + d_lo)
    (q0, _mm256_mul_ps(resid, inv))
}

/// sigmoid as `(hi, lo)` with `hi` the nearest `f32` to `hi + lo` (about 0.5 ulp from the true value in the worst case).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn sigmoid_hilo_ps_avx2(x: __m256) -> (__m256, __m256) {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let neg_abs = _mm256_xor_ps(_mm256_and_ps(x, abs_mask), _mm256_set1_ps(-0.0));
    let (eh0, el0) = unsafe { exp_hilo_ps_avx2(neg_abs) };
    // below -87 the exponential is subnormal: the plain kernel (whose clamp reaches the subnormal range) takes over, without a low part
    let deep = _mm256_cmp_ps(neg_abs, _mm256_set1_ps(-87.0), _CMP_LT_OQ);
    let eh = _mm256_blendv_ps(eh0, unsafe { exp_ps_avx2(neg_abs) }, deep);
    let el = _mm256_andnot_ps(deep, el0);
    let one = _mm256_set1_ps(1.0);
    // x >= 0: 1 / (1 + e^-x); x < 0: e^x / (1 + e^x) with the numerator e = eh + el carried as two terms
    let is_neg = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
    let (q0, c0) = unsafe { div_one_plus_hilo(_mm256_blendv_ps(one, eh, is_neg), eh, el) };
    // the low half of a negative-side numerator: el / (1 + e), to first order el * inv
    let inv = _mm256_div_ps(one, _mm256_add_ps(one, eh));
    let c = _mm256_add_ps(c0, _mm256_and_ps(_mm256_mul_ps(el, inv), is_neg));
    let hi = _mm256_add_ps(q0, c);
    let lo = _mm256_add_ps(_mm256_sub_ps(q0, hi), c);
    (hi, lo)
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
    let t = _mm256_div_ps(one, _mm256_fmadd_ps(half, z, one));

    // degree-9 polynomial in t, Estrin form: the same coefficients as a Horner chain, but a dependency chain of about half the length
    let t2 = _mm256_mul_ps(t, t);
    let t4 = _mm256_mul_ps(t2, t2);
    let t8 = _mm256_mul_ps(t4, t4);
    let pair = |hi: f32, lo: f32| _mm256_fmadd_ps(_mm256_set1_ps(hi), t, _mm256_set1_ps(lo));
    let q0 = pair(1.000_023_7, -1.265_512_2);
    let q1 = pair(0.096_784_18, 0.374_091_96);
    let q2 = pair(0.278_868_07, -0.186_288_06);
    let q3 = pair(1.488_515_9, -1.135_204);
    let q4 = pair(0.170_872_77, -0.822_152_26);
    let r0 = _mm256_fmadd_ps(q1, t2, q0);
    let r1 = _mm256_fmadd_ps(q3, t2, q2);
    let poly = _mm256_fmadd_ps(q4, t8, _mm256_fmadd_ps(r1, t4, r0));

    let arg = _mm256_fnmadd_ps(z, z, poly);
    // the cdf is half of erfc: the 0.5 goes into the factor `t` (a power-of-two scaling, so exact) that is ready long before the exponential
    let h = _mm256_mul_ps(_mm256_mul_ps(half, t), unsafe { exp_nonpos_ps_avx2(arg) });
    let u_neg = _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_LT_OQ);
    _mm256_blendv_ps(h, _mm256_sub_ps(one, h), u_neg)
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

/// Standard normal CDF for `PrecisionMode::Fastest`: the five-term Abramowitz-Stegun erfc (absolute error about 1.5e-7) instead of the
/// degree-9 fit with relative accuracy in the tails.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn norm_cdf_fast_ps_avx2(x: __m256) -> __m256 {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let u = _mm256_mul_ps(x, _mm256_set1_ps(-std::f32::consts::FRAC_1_SQRT_2));
    let z = _mm256_and_ps(u, abs_mask);
    let one = _mm256_set1_ps(1.0);
    let t = _mm256_div_ps(one, _mm256_fmadd_ps(_mm256_set1_ps(0.327_591_1), z, one));
    let mut p = _mm256_set1_ps(1.061_405_4);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-1.453_152));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.421_413_8));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(-0.284_496_74));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.254_829_6));
    let e = unsafe { exp_nonpos_ps_avx2(_mm256_xor_ps(_mm256_mul_ps(z, z), _mm256_set1_ps(-0.0))) };
    let erfc_abs = _mm256_mul_ps(_mm256_mul_ps(p, t), e);
    let neg = _mm256_sub_ps(_mm256_set1_ps(2.0), erfc_abs);
    let erfc = _mm256_blendv_ps(erfc_abs, neg, _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_LT_OQ));
    _mm256_mul_ps(_mm256_set1_ps(0.5), erfc)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_fast_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let mut i = 0usize;
    let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
    while i + 16 <= n {
        let (x0, x1) = unsafe { (_mm256_loadu_ps(sp.add(i)), _mm256_loadu_ps(sp.add(i + 8))) };
        let (c0, c1) = unsafe { (norm_cdf_fast_ps_avx2(x0), norm_cdf_fast_ps_avx2(x1)) };
        unsafe {
            _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(x0, c0));
            _mm256_storeu_ps(dp.add(i + 8), _mm256_mul_ps(x1, c1));
        }
        i += 16;
    }
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        unsafe { _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(x, norm_cdf_fast_ps_avx2(x))) };
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
unsafe fn gelu_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let mut i = 0usize;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    while i + 16 <= n {
        let (x0, x1) = unsafe { (_mm256_loadu_ps(sp.add(i)), _mm256_loadu_ps(sp.add(i + 8))) };
        let (c0, c1) = unsafe { (norm_cdf_ps_avx2(x0), norm_cdf_ps_avx2(x1)) };
        unsafe {
            _mm256_storeu_ps(dp.add(i), _mm256_mul_ps(x0, c0));
            _mm256_storeu_ps(dp.add(i + 8), _mm256_mul_ps(x1, c1));
        }
        i += 16;
    }
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
        // x - m rounds; its exact rounding error (TwoSum) scales the result: exp(x - m) = exp(d) (1 + err). Without it the
        // relative error of a probability grows with |x - m| (about |x - m| half-ulps, i.e. dozens for logits spread over 100)
        let d = _mm256_sub_ps(x, vm);
        let bb = _mm256_sub_ps(d, x);
        let err = _mm256_add_ps(_mm256_sub_ps(x, _mm256_sub_ps(d, bb)), _mm256_sub_ps(_mm256_setzero_ps(), _mm256_add_ps(vm, bb)));
        let e0 = unsafe { exp_ps_avx2(d) };
        let e = _mm256_fmadd_ps(e0, err, e0);
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

    // the mean as hi + lo (accumulated in f64): centring a row such as 1000 + noise would otherwise lose the noise
    let (hi, lo) = super::cpu::center_split(src);
    let (vhi, vlo) = (_mm256_set1_ps(hi), _mm256_set1_ps(lo));

    let mut vvar0 = _mm256_setzero_ps();
    let mut vvar1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let d0 = _mm256_sub_ps(_mm256_sub_ps(unsafe { _mm256_loadu_ps(sp.add(i)) }, vhi), vlo);
        let d1 = _mm256_sub_ps(_mm256_sub_ps(unsafe { _mm256_loadu_ps(sp.add(i + 8)) }, vhi), vlo);
        vvar0 = _mm256_fmadd_ps(d0, d0, vvar0);
        vvar1 = _mm256_fmadd_ps(d1, d1, vvar1);
        i += 16;
    }
    vvar0 = _mm256_add_ps(vvar0, vvar1);
    while i + 8 <= n {
        let d0 = _mm256_sub_ps(_mm256_sub_ps(unsafe { _mm256_loadu_ps(sp.add(i)) }, vhi), vlo);
        vvar0 = _mm256_fmadd_ps(d0, d0, vvar0);
        i += 8;
    }
    let mut var_sum = unsafe { hsum256_ps(vvar0) };
    while i < n {
        let d = (unsafe { *sp.add(i) } - hi) - lo;
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
        let xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_sub_ps(x, vhi), vlo), vinv);
        let out = _mm256_fmadd_ps(xhat, wv, bv);
        unsafe { _mm256_storeu_ps(dp.add(i), out) };
        i += 8;
    }
    while i < n {
        let x = unsafe { *sp.add(i) };
        let wv = unsafe { *wp.add(i) };
        let bv = unsafe { *bp.add(i) };
        unsafe { *dp.add(i) = ((x - hi) - lo) * inv_std * wv + bv };
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
    #[cfg(target_arch = "aarch64")]
    {
        return super::simd_arm::exp_f32_slice(src, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    false
}

/// GELU forward; `fast` selects the cheaper five-term erfc used by `PrecisionMode::Fastest` (the caller reads the mode once per tensor).
pub fn gelu_f32_slice(src: &[f32], dst: &mut [f32], fast: bool) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        if fast {
            unsafe { gelu_fast_f32_avx2_impl(src, dst) };
        } else {
            unsafe { gelu_f32_avx2_impl(src, dst) };
        }
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
    #[cfg(target_arch = "aarch64")]
    {
        return super::simd_arm::sum_f32_slice(src);
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    let (hi, _) = unsafe { sigmoid_hilo_ps_avx2(x) };
    let isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    _mm256_blendv_ps(hi, x, isnan)
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
        let (sh, sl) = unsafe { sigmoid_hilo_ps_avx2(x) };
        // x * (sh + sl) with one rounding: the product's rounding error is recovered by an FMA
        let t = _mm256_mul_ps(x, sh);
        let terr = _mm256_fmsub_ps(x, sh, t);
        let r = _mm256_add_ps(t, _mm256_fmadd_ps(x, sl, terr));
        // an infinite input makes the FMA residual NaN; the plain product (inf, or NaN for a NaN input) is the answer there
        let bad = _mm256_cmp_ps(r, r, _CMP_UNORD_Q);
        unsafe { _mm256_storeu_ps(dp.add(i), _mm256_blendv_ps(r, t, bad)) };
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_ps_avx2(x: __m256) -> __m256 {
    let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let ax = _mm256_and_ps(x, abs_mask);
    let sign = _mm256_andnot_ps(abs_mask, x);
    // |x| < 1/4: x + x^3 * Q(x^2), Q a degree-4 fit of (tanh(x)/x - 1)/x^2 on x^2 in [0, 1/16] (relative error 0.011 float32 half-ulps)
    let x2 = _mm256_mul_ps(x, x);
    let mut p = _mm256_set1_ps(-8.324_3e-3);
    for c in [2.184_04e-2, -5.396_761e-2, 1.333_333_3e-1, -3.333_333_4e-1] {
        p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(c));
    }
    let small = _mm256_fmadd_ps(x, _mm256_mul_ps(x2, p), x);
    // |x| >= 1/4: (E - 1) / (E + 1) with E = e^(2|x|) carried as a hi/lo pair. Numerator and denominator are formed with TwoSum so
    // neither loses bits, and the quotient is corrected by its residual, so the result is rounded once. The clamp makes E finite
    // and the result exactly 1 where tanh rounds to 1.
    let z = _mm256_min_ps(ax, _mm256_set1_ps(9.1));
    let (eh, el) = unsafe { exp_hilo_ps_avx2(_mm256_add_ps(z, z)) };
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    let two_sum = |a: __m256, b: __m256| {
        let s = _mm256_add_ps(a, b);
        let bb = _mm256_sub_ps(s, a);
        (s, _mm256_add_ps(_mm256_sub_ps(a, _mm256_sub_ps(s, bb)), _mm256_sub_ps(b, bb)))
    };
    let (n_hi, n_e) = two_sum(eh, neg_one);
    let (d_hi, d_e) = two_sum(eh, one);
    let n_lo = _mm256_add_ps(n_e, el);
    let d_lo = _mm256_add_ps(d_e, el);
    let inv = _mm256_div_ps(one, d_hi);
    let q0 = _mm256_mul_ps(n_hi, inv);
    let resid = _mm256_fnmadd_ps(q0, d_lo, _mm256_add_ps(_mm256_fnmadd_ps(q0, d_hi, n_hi), n_lo));
    let large = _mm256_or_ps(_mm256_fmadd_ps(resid, inv, q0), sign);
    let is_small = _mm256_cmp_ps(ax, _mm256_set1_ps(0.25), _CMP_LT_OQ);
    let isnan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    _mm256_blendv_ps(_mm256_blendv_ps(large, small, is_small), x, isnan)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tanh_f32_avx2_impl(src: &[f32], dst: &mut [f32]) {
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        let x = unsafe { _mm256_loadu_ps(sp.add(i)) };
        unsafe { _mm256_storeu_ps(dp.add(i), tanh_ps_avx2(x)) };
        i += 8;
    }
    while i < n {
        unsafe { *dp.add(i) = (*sp.add(i)).tanh() };
        i += 1;
    }
}

pub fn tanh_f32_slice(src: &[f32], dst: &mut [f32]) -> bool {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { tanh_f32_avx2_impl(src, dst) };
        return true;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return super::simd_arm::tanh_f32_slice(src, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    false
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
    #[cfg(target_arch = "aarch64")]
    {
        return super::simd_arm::relu_f32_slice(src, dst);
    }
    #[cfg(not(target_arch = "aarch64"))]
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_bias_row_f32_avx2_impl(row: &mut [f32], bias: &[f32]) {
    let n = row.len();
    let rp = row.as_mut_ptr();
    let bp = bias.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let r0 = unsafe { _mm256_loadu_ps(rp.add(i)) };
        let b0 = unsafe { _mm256_loadu_ps(bp.add(i)) };
        let r1 = unsafe { _mm256_loadu_ps(rp.add(i + 8)) };
        let b1 = unsafe { _mm256_loadu_ps(bp.add(i + 8)) };
        unsafe {
            _mm256_storeu_ps(rp.add(i), _mm256_add_ps(r0, b0));
            _mm256_storeu_ps(rp.add(i + 8), _mm256_add_ps(r1, b1));
        }
        i += 16;
    }
    while i + 8 <= n {
        let r0 = unsafe { _mm256_loadu_ps(rp.add(i)) };
        let b0 = unsafe { _mm256_loadu_ps(bp.add(i)) };
        unsafe { _mm256_storeu_ps(rp.add(i), _mm256_add_ps(r0, b0)) };
        i += 8;
    }
    while i < n {
        unsafe { *rp.add(i) += *bp.add(i) };
        i += 1;
    }
}

pub fn add_bias_row_f32(row: &mut [f32], bias: &[f32]) -> bool {
    debug_assert_eq!(row.len(), bias.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { add_bias_row_f32_avx2_impl(row, bias) };
        return true;
    }
    false
}

/// Single-pass fused Adam / AdamW parameter update in AVX2+FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn adam_step_f32_avx2_impl(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled: bool,
    inv_bc1: f32,
    inv_bc2: f32,
) {
    let n = p.len();
    let pp = p.as_mut_ptr();
    let gp = g.as_ptr();
    let mp = m.as_mut_ptr();
    let vp = v.as_mut_ptr();

    let vb1 = _mm256_set1_ps(beta1);
    let v1mb1 = _mm256_set1_ps(1.0 - beta1);
    let vb2 = _mm256_set1_ps(beta2);
    let v1mb2 = _mm256_set1_ps(1.0 - beta2);
    let v_inv_bc1 = _mm256_set1_ps(inv_bc1);
    let v_inv_bc2 = _mm256_set1_ps(inv_bc2);
    let v_lr = _mm256_set1_ps(lr);
    let v_eps = _mm256_set1_ps(eps);
    let v_wd = _mm256_set1_ps(weight_decay);
    let v_decay = _mm256_set1_ps(1.0 - lr * weight_decay);

    let mut i = 0usize;
    while i + 8 <= n {
        let mut pv = unsafe { _mm256_loadu_ps(pp.add(i)) };
        let mut gv = unsafe { _mm256_loadu_ps(gp.add(i)) };
        if weight_decay != 0.0 {
            if decoupled {
                pv = _mm256_mul_ps(pv, v_decay);
            } else {
                gv = _mm256_fmadd_ps(pv, v_wd, gv);
            }
        }
        let mv = unsafe { _mm256_loadu_ps(mp.add(i)) };
        let vv = unsafe { _mm256_loadu_ps(vp.add(i)) };
        let m_new = _mm256_fmadd_ps(vb1, mv, _mm256_mul_ps(v1mb1, gv));
        let g2 = _mm256_mul_ps(gv, gv);
        let v_new = _mm256_fmadd_ps(vb2, vv, _mm256_mul_ps(v1mb2, g2));
        unsafe {
            _mm256_storeu_ps(mp.add(i), m_new);
            _mm256_storeu_ps(vp.add(i), v_new);
        }
        let m_hat = _mm256_mul_ps(m_new, v_inv_bc1);
        let v_hat = _mm256_mul_ps(v_new, v_inv_bc2);
        let denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), v_eps);
        let step = _mm256_mul_ps(_mm256_div_ps(m_hat, denom), v_lr);
        unsafe { _mm256_storeu_ps(pp.add(i), _mm256_sub_ps(pv, step)) };
        i += 8;
    }
    while i < n {
        let mut pv = unsafe { *pp.add(i) };
        let mut gv = unsafe { *gp.add(i) };
        if weight_decay != 0.0 {
            if decoupled {
                pv *= 1.0 - lr * weight_decay;
            } else {
                gv += pv * weight_decay;
            }
        }
        let m_new = beta1 * unsafe { *mp.add(i) } + (1.0 - beta1) * gv;
        let v_new = beta2 * unsafe { *vp.add(i) } + (1.0 - beta2) * gv * gv;
        unsafe {
            *mp.add(i) = m_new;
            *vp.add(i) = v_new;
            *pp.add(i) = pv - lr * (m_new * inv_bc1) / ((v_new * inv_bc2).sqrt() + eps);
        }
        i += 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn adam_step_f32_slice(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled: bool,
    inv_bc1: f32,
    inv_bc2: f32,
) -> bool {
    debug_assert_eq!(p.len(), g.len());
    debug_assert_eq!(p.len(), m.len());
    debug_assert_eq!(p.len(), v.len());
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe {
            adam_step_f32_avx2_impl(p, g, m, v, lr, beta1, beta2, eps, weight_decay, decoupled, inv_bc1, inv_bc2);
        }
        return true;
    }
    false
}

/// Fused single-row LayerNorm backward (`dx`, accumulating `dw` & `db` into thread-local buffers).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_row_bwd_f32_avx2_impl(
    x: &[f32],
    g: &[f32],
    w: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
    db: &mut [f32],
    eps: f32,
) {
    let n = x.len();
    let xp = x.as_ptr();
    let gp = g.as_ptr();
    let wp = w.as_ptr();
    let dxp = dx.as_mut_ptr();
    let dwp = dw.as_mut_ptr();
    let dbp = db.as_mut_ptr();

    let (hi, lo) = super::cpu::center_split(x);
    let (vhi, vlo) = (_mm256_set1_ps(hi), _mm256_set1_ps(lo));

    let mut vvar = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let d = _mm256_sub_ps(_mm256_sub_ps(unsafe { _mm256_loadu_ps(xp.add(i)) }, vhi), vlo);
        vvar = _mm256_fmadd_ps(d, d, vvar);
        i += 8;
    }
    let mut var_sum = unsafe { hsum256_ps(vvar) };
    while i < n {
        let d = (unsafe { *xp.add(i) } - hi) - lo;
        var_sum += d * d;
        i += 1;
    }
    let inv_std = 1.0f32 / (var_sum / (n as f32) + eps).sqrt();
    let vinv = _mm256_set1_ps(inv_std);

    // Pass 2: compute sum1 = sum(g * w) and sum2 = sum(g * w * x_hat), plus accumulate dw & db
    let mut vs1 = _mm256_setzero_ps();
    let mut vs2 = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let xv = unsafe { _mm256_loadu_ps(xp.add(i)) };
        let gv = unsafe { _mm256_loadu_ps(gp.add(i)) };
        let wv = unsafe { _mm256_loadu_ps(wp.add(i)) };
        let xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_sub_ps(xv, vhi), vlo), vinv);
        let gxhat = _mm256_mul_ps(gv, wv);
        vs1 = _mm256_add_ps(vs1, gxhat);
        vs2 = _mm256_fmadd_ps(gxhat, xhat, vs2);
        let cur_dw = unsafe { _mm256_loadu_ps(dwp.add(i)) };
        let cur_db = unsafe { _mm256_loadu_ps(dbp.add(i)) };
        unsafe {
            _mm256_storeu_ps(dwp.add(i), _mm256_fmadd_ps(gv, xhat, cur_dw));
            _mm256_storeu_ps(dbp.add(i), _mm256_add_ps(cur_db, gv));
        }
        i += 8;
    }
    let mut s1 = unsafe { hsum256_ps(vs1) };
    let mut s2 = unsafe { hsum256_ps(vs2) };
    while i < n {
        let xhat = ((unsafe { *xp.add(i) } - hi) - lo) * inv_std;
        let gv = unsafe { *gp.add(i) };
        let gxhat = gv * unsafe { *wp.add(i) };
        s1 += gxhat;
        s2 += gxhat * xhat;
        unsafe {
            *dwp.add(i) += gv * xhat;
            *dbp.add(i) += gv;
        }
        i += 1;
    }

    // Pass 3: dx = inv_std * (g * w - (s1 + x_hat * s2) / n)
    let inv_n = 1.0f32 / (n as f32);
    let vm1 = _mm256_set1_ps(s1 * inv_n);
    let vm2 = _mm256_set1_ps(s2 * inv_n);
    i = 0;
    while i + 8 <= n {
        let xv = unsafe { _mm256_loadu_ps(xp.add(i)) };
        let gv = unsafe { _mm256_loadu_ps(gp.add(i)) };
        let wv = unsafe { _mm256_loadu_ps(wp.add(i)) };
        let xhat = _mm256_mul_ps(_mm256_sub_ps(_mm256_sub_ps(xv, vhi), vlo), vinv);
        let gxhat = _mm256_mul_ps(gv, wv);
        let sub = _mm256_fmadd_ps(xhat, vm2, vm1);
        let dxv = _mm256_mul_ps(_mm256_sub_ps(gxhat, sub), vinv);
        unsafe { _mm256_storeu_ps(dxp.add(i), dxv) };
        i += 8;
    }
    while i < n {
        let xhat = ((unsafe { *xp.add(i) } - hi) - lo) * inv_std;
        let gxhat = unsafe { *gp.add(i) * *wp.add(i) };
        unsafe { *dxp.add(i) = (gxhat - s1 * inv_n - xhat * (s2 * inv_n)) * inv_std };
        i += 1;
    }
}

pub fn layer_norm_row_bwd_f32(
    x: &[f32],
    g: &[f32],
    w: &[f32],
    dx: &mut [f32],
    dw: &mut [f32],
    db: &mut [f32],
    eps: f32,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { layer_norm_row_bwd_f32_avx2_impl(x, g, w, dx, dw, db, eps) };
        return true;
    }
    false
}

#[cfg(all(test, target_arch = "x86_64"))]
/// The `(hi, lo)` exponential feeds sigmoid/silu/tanh: `hi + lo` must be within a few hundredths of an ulp of the true value.
mod exp_hilo_tests {
    use super::*;
    #[test]
    fn exp_hilo_pair_is_accurate() {
        if !has_avx2_fma() {
            return;
        }
        for (lo, hi) in [(0.5f32, 1.0f32), (1.0, 2.0), (2.0, 4.0), (4.0, 9.0)] {
            let mut worst = 0f64;
            let mut mean = 0f64;
            let n = 20000;
            for i in 0..n {
                let x = lo + (hi - lo) * (i as f32) / (n as f32);
                let (h, l) = unsafe {
                    let (h, l) = exp_hilo_ps_avx2(_mm256_set1_ps(x));
                    let mut a = [0f32; 8];
                    let mut b = [0f32; 8];
                    _mm256_storeu_ps(a.as_mut_ptr(), h);
                    _mm256_storeu_ps(b.as_mut_ptr(), l);
                    (a[0], b[0])
                };
                let r = (x as f64).exp();
                let ulp = (h.to_bits() as i64 - 1) as u32;
                let ulp = (h - f32::from_bits(ulp)) as f64;
                let e = ((h as f64 + l as f64) - r).abs() / ulp;
                worst = worst.max(e);
                mean += e / n as f64;
            }
            assert!(worst < 0.06, "exp_hilo [{lo},{hi}): worst {worst:.4} ulp(hi), mean {mean:.4}");
        }
    }
}
