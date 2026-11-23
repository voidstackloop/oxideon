//! True integer-arithmetic `int8` quantized inference kernels (`qlinear_i8`, `qconv2d_i8`).
//!
//! Accumulates `i8 * i8 -> i32` directly in integer registers (AVX2 `vpmaddwd` on `x86_64`)
//! without dequantizing weights or activations to `f32` in the inner loop.

use crate::error::{Error, Result};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i32_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut i = 0usize;
    let mut acc = _mm256_setzero_si256();
    while i + 16 <= n {
        unsafe {
            let va8 = _mm_loadu_si128(ap.add(i) as *const __m128i);
            let vb8 = _mm_loadu_si128(bp.add(i) as *const __m128i);
            let va16 = _mm256_cvtepi8_epi16(va8);
            let vb16 = _mm256_cvtepi8_epi16(vb8);
            let prod = _mm256_madd_epi16(va16, vb16);
            acc = _mm256_add_epi32(acc, prod);
        }
        i += 16;
    }
    let hi = _mm256_extracti128_si256::<1>(acc);
    let lo = _mm256_castsi256_si128(acc);
    let sum128 = _mm_add_epi32(lo, hi);
    let mut tmp = [0i32; 4];
    unsafe { _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, sum128) };
    let mut total = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < n {
        unsafe {
            total += (*ap.add(i) as i32) * (*bp.add(i) as i32);
        }
        i += 1;
    }
    total
}

#[inline]
pub fn dot_i8_i32(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { dot_i8_i32_avx2(a, b) };
    }
    let mut acc = 0i32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        acc += (x as i32) * (y as i32);
    }
    acc
}

/// Symmetric per-tensor `f32 -> i8` quantization returning `(quantized, scale)`.
pub fn quantize_symmetric_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
    let inv = 1.0 / scale;
    let q = x
        .iter()
        .map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// True integer matrix multiply `C[m, n] = A[m, k] @ B_T[n, k]^T` with `i32` accumulation.
pub fn int8_matmul_i32(a: &[i8], b_t: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
            let arow = &a[i * k..(i + 1) * k];
            for j in 0..n {
                crow[j] = dot_i8_i32(arow, &b_t[j * k..(j + 1) * k]);
            }
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for i in 0..m {
            let arow = &a[i * k..(i + 1) * k];
            for j in 0..n {
                c[i * n + j] = dot_i8_i32(arow, &b_t[j * k..(j + 1) * k]);
            }
        }
    }
    c
}

/// Int8 quantized Linear layer: `y = (x_q @ w_q^T) * (x_scale * w_scale) + bias`.
/// `x_q: [m, k]`, `w_q: [n, k]`. Computes strictly in `i8 -> i32` integers.
#[allow(clippy::too_many_arguments)]
pub fn qlinear_i8(
    x_q: &[i8],
    x_scale: f32,
    w_q: &[i8],
    w_scale: f32,
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    if x_q.len() != m * k || w_q.len() != n * k {
        return Err(Error::Msg("qlinear_i8: shape mismatch".into()));
    }
    if let Some(b) = bias
        && b.len() != n
    {
        return Err(Error::Msg("qlinear_i8: bias length mismatch".into()));
    }
    let acc = int8_matmul_i32(x_q, w_q, m, k, n);
    let combined_scale = x_scale * w_scale;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let b = bias.map_or(0.0, |bv| bv[j]);
            out[i * n + j] = (acc[i * n + j] as f32) * combined_scale + b;
        }
    }
    Ok(out)
}

/// Int8 quantized 2D convolution (`input: [N, C_in, H, W]`, `weight: [C_out, C_in, kH, kW]`).
/// Builds `im2col` patches in `i8` and accumulates strictly in `i32` integers.
#[allow(clippy::too_many_arguments)]
pub fn qconv2d_i8(
    x_q: &[i8],
    x_scale: f32,
    w_q: &[i8],
    w_scale: f32,
    bias: Option<&[f32]>,
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Vec<f32>> {
    if x_q.len() != n * c_in * h * w || w_q.len() != c_out * c_in * kh * kw {
        return Err(Error::Msg("qconv2d_i8: shape mismatch".into()));
    }
    if stride.0 == 0 || stride.1 == 0 {
        return Err(Error::Msg("qconv2d_i8: stride must be positive".into()));
    }
    if kh == 0 || kw == 0 || h + 2 * padding.0 < kh || w + 2 * padding.1 < kw {
        return Err(Error::Msg("qconv2d_i8: kernel does not fit the padded input".into()));
    }
    if let Some(b) = bias
        && b.len() != c_out
    {
        return Err(Error::Msg("qconv2d_i8: bias length mismatch".into()));
    }
    let out_h = (h + 2 * padding.0 - kh) / stride.0 + 1;
    let out_w = (w + 2 * padding.1 - kw) / stride.1 + 1;
    let kdim = c_in * kh * kw;
    let l = out_h * out_w;
    let combined_scale = x_scale * w_scale;
    let mut out = vec![0.0f32; n * c_out * l];
    let img_elems = c_in * h * w;

    for b in 0..n {
        let img = &x_q[b * img_elems..(b + 1) * img_elems];
        // Build row-major patch matrix [l, kdim] in i8 so each row is a contiguous kdim dot product!
        let mut patches = vec![0i8; l * kdim];
        for oh in 0..out_h {
            for ow in 0..out_w {
                let p_row = &mut patches[(oh * out_w + ow) * kdim..(oh * out_w + ow + 1) * kdim];
                for c in 0..c_in {
                    for r in 0..kh {
                        let ih = (oh * stride.0 + r) as isize - padding.0 as isize;
                        if ih < 0 || ih as usize >= h {
                            continue;
                        }
                        for s in 0..kw {
                            let iw = (ow * stride.1 + s) as isize - padding.1 as isize;
                            if iw >= 0 && (iw as usize) < w {
                                p_row[(c * kh + r) * kw + s] =
                                    img[(c * h + ih as usize) * w + iw as usize];
                            }
                        }
                    }
                }
            }
        }
        // acc_l_oc[l, c_out] = patches[l, kdim] @ w_q[c_out, kdim]^T
        let acc = int8_matmul_i32(&patches, w_q, l, kdim, c_out);
        let obase = b * c_out * l;
        for oc in 0..c_out {
            let bv = bias.map_or(0.0, |bs| bs[oc]);
            for pos in 0..l {
                out[obase + oc * l + pos] = (acc[pos * c_out + oc] as f32) * combined_scale + bv;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qlinear_and_qconv2d_compute_exact_integer_products() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let w: Vec<f32> = (0..48).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
        let (xq, xs) = quantize_symmetric_i8(&x);
        let (wq, ws) = quantize_symmetric_i8(&w);
        let y = qlinear_i8(&xq, xs, &wq, ws, Some(&[0.1, -0.2, 0.3]), 2, 16, 3).unwrap();
        assert_eq!(y.len(), 6);
        for i in 0..2 {
            for j in 0..3 {
                let mut ref_acc = 0i32;
                for k in 0..16 {
                    ref_acc += (xq[i * 16 + k] as i32) * (wq[j * 16 + k] as i32);
                }
                let want = ref_acc as f32 * xs * ws + [0.1, -0.2, 0.3][j];
                assert!((y[i * 3 + j] - want).abs() < 1e-6);
            }
        }

        // qconv2d_i8: [1, 2, 4, 4] * [3, 2, 3, 3]
        let cx: Vec<f32> = (0..32).map(|i| (i as f32 - 15.0) * 0.1).collect();
        let cw: Vec<f32> = (0..54).map(|i| ((i % 5) as f32 - 2.0) * 0.15).collect();
        let (cxq, cxs) = quantize_symmetric_i8(&cx);
        let (cwq, cws) = quantize_symmetric_i8(&cw);
        let cy = qconv2d_i8(&cxq, cxs, &cwq, cws, None, 1, 2, 4, 4, 3, 3, 3, (1, 1), (1, 1)).unwrap();
        assert_eq!(cy.len(), 3 * 4 * 4);
        // exact integer reference, strided and padded, with bias
        for &(stride, pad) in &[((1usize, 1usize), (1usize, 1usize)), ((2, 1), (0, 2)), ((1, 2), (1, 0))] {
            let bias = [0.5f32, -1.0, 2.0];
            let (kh, kw, h, w, cin, cout) = (3usize, 3usize, 4usize, 4usize, 2usize, 3usize);
            let oh = (h + 2 * pad.0 - kh) / stride.0 + 1;
            let ow = (w + 2 * pad.1 - kw) / stride.1 + 1;
            let got = qconv2d_i8(&cxq, cxs, &cwq, cws, Some(&bias), 1, cin, h, w, cout, kh, kw, stride, pad).unwrap();
            assert_eq!(got.len(), cout * oh * ow);
            for oc in 0..cout {
                for y in 0..oh {
                    for x in 0..ow {
                        let mut acc = 0i32;
                        for c in 0..cin {
                            for r in 0..kh {
                                for s in 0..kw {
                                    let (iy, ix) = ((y * stride.0 + r) as isize - pad.0 as isize, (x * stride.1 + s) as isize - pad.1 as isize);
                                    if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                        acc += cxq[(c * h + iy as usize) * w + ix as usize] as i32 * cwq[((oc * cin + c) * kh + r) * kw + s] as i32;
                                    }
                                }
                            }
                        }
                        let want = acc as f32 * cxs * cws + bias[oc];
                        assert!((got[(oc * oh + y) * ow + x] - want).abs() < 1e-5, "stride {stride:?} pad {pad:?} at ({oc},{y},{x})");
                    }
                }
            }
        }
        // degenerate shapes are errors / empty results, not panics
        assert!(qconv2d_i8(&cxq, cxs, &cwq, cws, None, 1, 2, 4, 4, 3, 9, 9, (1, 1), (0, 0)).is_err());
        assert!(int8_matmul_i32(&[], &[], 0, 4, 3).is_empty());
        assert!(int8_matmul_i32(&[1, 2, 3, 4], &[], 1, 4, 0).is_empty());
    }
}
