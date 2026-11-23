//! ARM64 NEON SIMD kernels for `f32` elementwise and reduction ops.
//!
//! The `*_f32_slice` functions here are the aarch64 branch of the dispatchers in `simd_x86` (which `cpu.rs` calls on every
//! architecture): on aarch64, where NEON is mandatory, they take over; elsewhere they report "not handled".
//!
//! The vectorized `exp` and `tanh` follow the scalar models [`exp_model`] and [`tanh_model`] step by step. The models run (and
//! are tested for accuracy against `libm`) on every architecture; the NEON code itself only executes on aarch64.

#[inline]
pub fn has_neon() -> bool {
    cfg!(target_arch = "aarch64")
}

// Cody-Waite split of ln 2 (the high part has trailing zero bits so `n * LN2_HI` is exact for |n| < 2^11).
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const INV_LN2: f32 = std::f32::consts::LOG2_E;
/// Above this `exp` overflows to +inf (ln of `f32::MAX`); below `EXP_FLUSH` the result is flushed to zero.
const EXP_OVERFLOW: f32 = 88.722_84;
const EXP_FLUSH: f32 = -87.336_55;
/// |x| below this uses the odd Taylor series for `tanh` (error about 2e-7), above it `1 - 2 / (exp(2|x|) + 1)`.
const TANH_SMALL: f32 = 0.35;

/// `exp` as the vector code computes it: `x = n ln2 + r`, a degree-6 polynomial for `e^r`, and `2^n` applied in two steps so
/// that results near `f32::MAX` do not overflow the exponent field.
pub fn exp_model(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > EXP_OVERFLOW {
        return f32::INFINITY;
    }
    if x < EXP_FLUSH {
        return 0.0;
    }
    let n = (x * INV_LN2).round_ties_even();
    let r = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let mut p = (1.0f32 / 720.0).mul_add(r, 1.0 / 120.0);
    p = p.mul_add(r, 1.0 / 24.0);
    p = p.mul_add(r, 1.0 / 6.0);
    p = p.mul_add(r, 0.5);
    p = p.mul_add(r, 1.0);
    p = p.mul_add(r, 1.0);
    let n_i = n as i32;
    let n1 = n_i >> 1;
    let n2 = n_i - n1;
    let pow2 = |k: i32| f32::from_bits(((k + 127) as u32) << 23);
    (p * pow2(n1)) * pow2(n2)
}

/// `tanh` as the vector code computes it.
pub fn tanh_model(x: f32) -> f32 {
    let ax = x.abs();
    if ax < TANH_SMALL {
        let x2 = x * x;
        let mut p = x2.mul_add(62.0 / 2835.0, -17.0 / 315.0);
        p = x2.mul_add(p, 2.0 / 15.0);
        p = x2.mul_add(p, -1.0 / 3.0);
        p = x2.mul_add(p, 1.0);
        return x * p;
    }
    if x.is_nan() {
        return x;
    }
    let t = 1.0 - 2.0 / (exp_model(2.0 * ax) + 1.0);
    t.copysign(x)
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_op_in_unsafe_fn)] // every body is a run of NEON intrinsics, each of them unsafe
mod neon_impl {
    use super::*;
    use std::arch::aarch64::*;

    #[inline]
    pub unsafe fn sum_f32_neon(src: &[f32]) -> f32 {
        let mut i = 0usize;
        let n = src.len();
        let p = src.as_ptr();
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        while i + 8 <= n {
            unsafe {
                acc0 = vaddq_f32(acc0, vld1q_f32(p.add(i)));
                acc1 = vaddq_f32(acc1, vld1q_f32(p.add(i + 4)));
            }
            i += 8;
        }
        let mut sum = unsafe { vaddvq_f32(vaddq_f32(acc0, acc1)) };
        while i < n {
            unsafe { sum += *p.add(i) };
            i += 1;
        }
        sum
    }

    #[inline]
    pub unsafe fn relu_f32_neon(src: &[f32], dst: &mut [f32]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let zero = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= n {
            unsafe {
                let x = vld1q_f32(sp.add(i));
                vst1q_f32(dp.add(i), vmaxq_f32(x, zero)); // NaN propagates, like the scalar `if x < 0 { 0 } else { x }`
            }
            i += 4;
        }
        while i < n {
            let x = unsafe { *sp.add(i) };
            unsafe { *dp.add(i) = if x < 0.0 { 0.0 } else { x } };
            i += 1;
        }
    }

    /// Four lanes of [`exp_model`].
    #[inline]
    unsafe fn exp_ps_neon(x: float32x4_t) -> float32x4_t {
        let overflow = vcgtq_f32(x, vdupq_n_f32(EXP_OVERFLOW));
        let flush = vcltq_f32(x, vdupq_n_f32(EXP_FLUSH));
        // keep the arithmetic below inside the range where `2^n` is representable; the masks fix the ends up afterwards
        let xc = vminq_f32(vmaxq_f32(x, vdupq_n_f32(EXP_FLUSH)), vdupq_n_f32(EXP_OVERFLOW));
        let n_f = vrndnq_f32(vmulq_f32(xc, vdupq_n_f32(INV_LN2)));
        // r = x - n * ln2_hi - n * ln2_lo  (vfmsq_f32(a, b, c) = a - b * c)
        let r = vfmsq_f32(vfmsq_f32(xc, n_f, vdupq_n_f32(LN2_HI)), n_f, vdupq_n_f32(LN2_LO));
        let mut p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), vdupq_n_f32(1.0 / 720.0), r); // vfmaq_f32(a, b, c) = a + b * c
        p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        let n_i = vcvtq_s32_f32(n_f);
        let n1 = vshrq_n_s32::<1>(n_i);
        let n2 = vsubq_s32(n_i, n1);
        let bias = vdupq_n_s32(127);
        let pow2a = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n1, bias)));
        let pow2b = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n2, bias)));
        let y = vmulq_f32(vmulq_f32(p, pow2a), pow2b);
        let y = vbslq_f32(overflow, vdupq_n_f32(f32::INFINITY), y);
        vbslq_f32(flush, vdupq_n_f32(0.0), y)
    }

    #[inline]
    pub unsafe fn exp_f32_neon(src: &[f32], dst: &mut [f32]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0usize;
        while i + 4 <= n {
            unsafe {
                let x = vld1q_f32(sp.add(i));
                vst1q_f32(dp.add(i), exp_ps_neon(x));
            }
            i += 4;
        }
        while i < n {
            unsafe { *dp.add(i) = exp_model(*sp.add(i)) };
            i += 1;
        }
    }

    /// Four lanes of [`tanh_model`].
    #[inline]
    unsafe fn tanh_ps_neon(x: float32x4_t) -> float32x4_t {
        let ax = vabsq_f32(x);
        let x2 = vmulq_f32(x, x);
        let mut p = vfmaq_f32(vdupq_n_f32(-17.0 / 315.0), x2, vdupq_n_f32(62.0 / 2835.0));
        p = vfmaq_f32(vdupq_n_f32(2.0 / 15.0), x2, p);
        p = vfmaq_f32(vdupq_n_f32(-1.0 / 3.0), x2, p);
        p = vfmaq_f32(vdupq_n_f32(1.0), x2, p);
        let small = vmulq_f32(x, p);
        let e = exp_ps_neon(vmulq_f32(vdupq_n_f32(2.0), ax));
        let t = vsubq_f32(vdupq_n_f32(1.0), vdivq_f32(vdupq_n_f32(2.0), vaddq_f32(e, vdupq_n_f32(1.0))));
        let sign_mask = vdupq_n_u32(0x8000_0000);
        let t = vbslq_f32(sign_mask, x, t); // copy the sign of x onto t
        vbslq_f32(vcltq_f32(ax, vdupq_n_f32(TANH_SMALL)), small, t)
    }

    #[inline]
    pub unsafe fn tanh_f32_neon(src: &[f32], dst: &mut [f32]) {
        let n = src.len();
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();
        let mut i = 0usize;
        while i + 4 <= n {
            unsafe {
                let x = vld1q_f32(sp.add(i));
                vst1q_f32(dp.add(i), tanh_ps_neon(x));
            }
            i += 4;
        }
        while i < n {
            unsafe { *dp.add(i) = tanh_model(*sp.add(i)) };
            i += 1;
        }
    }

    macro_rules! neon_binop {
        ($name:ident, $intrinsic:ident, $scalar:expr) => {
            #[inline]
            pub unsafe fn $name(a: &[f32], b: &[f32], dst: &mut [f32]) {
                let n = a.len();
                let ap = a.as_ptr();
                let bp = b.as_ptr();
                let dp = dst.as_mut_ptr();
                let mut i = 0usize;
                while i + 4 <= n {
                    unsafe {
                        vst1q_f32(dp.add(i), $intrinsic(vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i))));
                    }
                    i += 4;
                }
                let op: fn(f32, f32) -> f32 = $scalar;
                while i < n {
                    unsafe { *dp.add(i) = op(*ap.add(i), *bp.add(i)) };
                    i += 1;
                }
            }
        };
    }

    neon_binop!(add_f32_neon, vaddq_f32, |a, b| a + b);
    neon_binop!(sub_f32_neon, vsubq_f32, |a, b| a - b);
    neon_binop!(mul_f32_neon, vmulq_f32, |a, b| a * b);
    neon_binop!(div_f32_neon, vdivq_f32, |a, b| a / b);
}

pub fn sum_f32_slice(src: &[f32]) -> Option<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        Some(unsafe { neon_impl::sum_f32_neon(src) })
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = src;
        None
    }
}

macro_rules! slice_fn {
    ($name:ident, $imp:ident, ($($arg:ident),*)) => {
        pub fn $name($($arg: &[f32],)* dst: &mut [f32]) -> bool {
            #[cfg(target_arch = "aarch64")]
            {
                unsafe { neon_impl::$imp($($arg,)* dst) };
                true
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let _ = ($($arg,)* dst);
                false
            }
        }
    };
}

slice_fn!(exp_f32_slice, exp_f32_neon, (src));
slice_fn!(tanh_f32_slice, tanh_f32_neon, (src));
slice_fn!(relu_f32_slice, relu_f32_neon, (src));
slice_fn!(add_f32_slice, add_f32_neon, (a, b));
slice_fn!(sub_f32_slice, sub_f32_neon, (a, b));
slice_fn!(mul_f32_slice, mul_f32_neon, (a, b));
slice_fn!(div_f32_slice, div_f32_neon, (a, b));

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(got: f32, want: f64) -> f64 {
        if want == 0.0 { got.abs() as f64 } else { ((got as f64 - want) / want).abs() }
    }

    #[test]
    fn exp_model_is_accurate_and_handles_the_ends() {
        let mut worst = 0.0f64;
        let mut x = -87.0f32;
        while x < 88.7 {
            worst = worst.max(rel_err(exp_model(x), (x as f64).exp()));
            x += 0.0137;
        }
        assert!(worst < 3e-7, "worst relative error {worst:e}");
        for x in [0.0f32, -0.0, 1e-8, -1e-8, 1.0, -1.0, 0.5, 20.0, -20.0] {
            assert!(rel_err(exp_model(x), (x as f64).exp()) < 3e-7, "x = {x}");
        }
        assert_eq!(exp_model(89.0), f32::INFINITY);
        assert_eq!(exp_model(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_model(-90.0), 0.0);
        assert_eq!(exp_model(f32::NEG_INFINITY), 0.0);
        assert!(exp_model(f32::NAN).is_nan());
        // close to the overflow threshold the two-step scaling keeps the result finite and right
        assert!(rel_err(exp_model(88.5), 88.5f64.exp()) < 3e-7);
        assert!(exp_model(88.7).is_finite());
    }

    #[test]
    fn tanh_model_is_accurate_for_tiny_moderate_and_huge_inputs() {
        let mut worst = 0.0f64;
        let mut x = -12.0f32;
        while x < 12.0 {
            worst = worst.max(rel_err(tanh_model(x), (x as f64).tanh()));
            x += 0.00731;
        }
        assert!(worst < 5e-7, "worst relative error {worst:e}");
        // the failure mode of (e - 1) / (e + 1): tiny arguments
        for x in [1e-8f32, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.34, 0.36] {
            assert!(rel_err(tanh_model(x), (x as f64).tanh()) < 5e-7, "x = {x}");
            assert!(rel_err(tanh_model(-x), (-x as f64).tanh()) < 5e-7, "x = -{x}");
        }
        assert_eq!(tanh_model(50.0), 1.0);
        assert_eq!(tanh_model(-50.0), -1.0);
        assert_eq!(tanh_model(f32::INFINITY), 1.0);
        assert!(tanh_model(f32::NAN).is_nan());
        assert_eq!(tanh_model(-0.0).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn the_dispatchers_report_whether_neon_handled_the_call() {
        let src = [1.0f32, -2.0, 3.0, 0.5, 9.0];
        let mut dst = [0.0f32; 5];
        let handled = relu_f32_slice(&src, &mut dst);
        assert_eq!(handled, has_neon());
        if handled {
            assert_eq!(dst, [1.0, 0.0, 3.0, 0.5, 9.0]);
        }
    }
}
