//! Element-wise numeric kernels behind `oxideon.tensor_utils` and `oxideon.special` (`gcd`, `lcm`, `frexp`, `ldexp`,
//! `nextafter`, `signbit`, `isposinf` / `isneginf`), the multidimensional histogram `histogramdd`, and the classical
//! orthogonal polynomials with a per-element degree (`torch.special.*_polynomial_*`). Binary kernels take operands of
//! one shape (the Python side broadcasts first).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

fn same_shape(a: &Tensor, b: &Tensor, op: &str) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!("{op}: operands must have one shape (broadcast first), got {:?} and {:?}", a.shape(), b.shape())));
    }
    Ok(())
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    a = a.wrapping_abs();
    b = b.wrapping_abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Element-wise greatest common divisor (`lcm == false`) or least common multiple of two integer tensors, as `i64`
/// (never negative; `lcm(0, x) == 0`).
pub fn gcd_lcm(a: &Tensor, b: &Tensor, lcm: bool) -> Result<Tensor> {
    same_shape(a, b, if lcm { "lcm" } else { "gcd" })?;
    if a.dtype().is_floating_point() || b.dtype().is_floating_point() {
        return Err(Error::Msg("gcd / lcm: expected integer tensors".into()));
    }
    let (x, y) = (a.to_vec::<i64>()?, b.to_vec::<i64>()?);
    let out: Vec<i64> = x
        .iter()
        .zip(&y)
        .map(|(&p, &q)| {
            let g = gcd_i64(p, q);
            if !lcm {
                g
            } else if g == 0 {
                0
            } else {
                (p / g).wrapping_mul(q).wrapping_abs()
            }
        })
        .collect();
    Tensor::from_slice_on(&out, a.shape(), &a.device())?.to_dtype(a.dtype())
}

/// Splits each element into a mantissa in [0.5, 1) (with the sign of x) and an `i32` exponent: x = m * 2^e
/// (`frexp(0) == (0, 0)`; inf and NaN keep their value with exponent 0).
pub fn frexp(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let v = x.to_vec::<f64>()?;
    let mut m = Vec::with_capacity(v.len());
    let mut e = Vec::with_capacity(v.len());
    for &val in &v {
        if val == 0.0 || !val.is_finite() {
            m.push(val);
            e.push(0i32);
            continue;
        }
        let mut exp = val.abs().log2().floor() as i32 + 1;
        let mut mant = val / 2f64.powi(exp);
        // correct rounding at the edges of the range
        while mant.abs() >= 1.0 {
            mant /= 2.0;
            exp += 1;
        }
        while mant.abs() < 0.5 {
            mant *= 2.0;
            exp -= 1;
        }
        m.push(mant);
        e.push(exp);
    }
    let dt = if x.dtype().is_floating_point() { x.dtype() } else { DType::F32 };
    Ok((Tensor::from_slice_on(&m, x.shape(), &x.device())?.to_dtype(dt)?, Tensor::from_slice_on(&e, x.shape(), &x.device())?))
}

/// `x * 2^e` element-wise, exact while the result is representable (`e` is rounded to an integer).
pub fn ldexp(x: &Tensor, e: &Tensor) -> Result<Tensor> {
    same_shape(x, e, "ldexp")?;
    let (v, k) = (x.to_vec::<f64>()?, e.to_vec::<f64>()?);
    let out: Vec<f64> = v
        .iter()
        .zip(&k)
        .map(|(&a, &b)| {
            // apply the power in two halves so 2^b itself cannot overflow while the product is finite
            let b = b.round().clamp(-2200.0, 2200.0) as i32;
            let (h1, h2) = (b / 2, b - b / 2);
            a * 2f64.powi(h1) * 2f64.powi(h2)
        })
        .collect();
    let dt = if x.dtype().is_floating_point() { x.dtype() } else { DType::F32 };
    Tensor::from_slice_on(&out, x.shape(), &x.device())?.to_dtype(dt)
}

fn next_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return b;
    }
    if a == 0.0 {
        let tiny = f64::from_bits(1);
        return if b > 0.0 { tiny } else { -tiny };
    }
    let bits = a.to_bits();
    let up = (b > a) == (a > 0.0);
    f64::from_bits(if up { bits + 1 } else { bits - 1 })
}

fn next_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if a == b {
        return b;
    }
    if a == 0.0 {
        let tiny = f32::from_bits(1);
        return if b > 0.0 { tiny } else { -tiny };
    }
    let bits = a.to_bits();
    let up = (b > a) == (a > 0.0);
    f32::from_bits(if up { bits + 1 } else { bits - 1 })
}

/// The next representable value after `x` towards `other`, in `x`'s dtype (`f32` or `f64`).
pub fn nextafter(x: &Tensor, other: &Tensor) -> Result<Tensor> {
    same_shape(x, other, "nextafter")?;
    match x.dtype() {
        DType::F64 => {
            let (a, b) = (x.to_vec::<f64>()?, other.to_vec::<f64>()?);
            let out: Vec<f64> = a.iter().zip(&b).map(|(&p, &q)| next_f64(p, q)).collect();
            Tensor::from_slice_on(&out, x.shape(), &x.device())
        }
        DType::F32 => {
            let (a, b) = (x.to_vec::<f32>()?, other.to_dtype(DType::F32)?.to_vec::<f32>()?);
            let out: Vec<f32> = a.iter().zip(&b).map(|(&p, &q)| next_f32(p, q)).collect();
            Tensor::from_slice_on(&out, x.shape(), &x.device())
        }
        other => Err(Error::Msg(format!("nextafter: expected f32 or f64, got {other:?}"))),
    }
}

/// Element-wise float classification as a `bool` tensor: `"signbit"` (sign bit set, including -0.0 and -inf),
/// `"posinf"` or `"neginf"`.
pub fn float_class(x: &Tensor, what: &str) -> Result<Tensor> {
    let v = x.to_vec::<f64>()?;
    let out: Vec<bool> = match what {
        "signbit" => v.iter().map(|a| a.is_sign_negative()).collect(),
        "posinf" => v.iter().map(|a| *a == f64::INFINITY).collect(),
        "neginf" => v.iter().map(|a| *a == f64::NEG_INFINITY).collect(),
        other => return Err(Error::Msg(format!("float_class: unknown class '{other}'"))),
    };
    Tensor::from_slice_on(&out, x.shape(), &x.device())
}

/// The histogram of the rows of an `[N, D]` sample over `bins[d]` equal-width bins per dimension spanning
/// `ranges[d] = (low, high)` (the data range when `None`; the last bin includes its right edge, values outside are
/// dropped), optionally weighted; with `density` the counts are divided by the total and the bin volume. Returns the
/// `D`-dimensional counts and the `bins[d] + 1` edges of every dimension.
pub fn histogramdd(
    sample: &Tensor,
    bins: &[usize],
    ranges: Option<&[(f64, f64)]>,
    weight: Option<&Tensor>,
    density: bool,
) -> Result<(Tensor, Vec<Tensor>)> {
    if sample.ndim() != 2 {
        return Err(Error::Msg("histogramdd: expected an [N, D] sample".into()));
    }
    let (n, d) = (sample.shape()[0], sample.shape()[1]);
    if bins.len() != d || bins.contains(&0) {
        return Err(Error::Msg(format!("histogramdd: need {d} positive bin counts, got {bins:?}")));
    }
    let x = sample.to_vec::<f64>()?;
    let w = match weight {
        Some(t) => {
            if t.numel() != n {
                return Err(Error::Msg("histogramdd: weight needs one value per sample".into()));
            }
            Some(t.to_vec::<f64>()?)
        }
        None => None,
    };
    let mut lims = Vec::with_capacity(d);
    for j in 0..d {
        let (lo, hi) = match ranges {
            Some(r) => r[j],
            None => {
                let col = (0..n).map(|i| x[i * d + j]).filter(|v| v.is_finite());
                let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
                for v in col {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if lo > hi {
                    (0.0, 1.0)
                } else if lo == hi {
                    (lo - 0.5, hi + 0.5)
                } else {
                    (lo, hi)
                }
            }
        };
        if lo.is_nan() || hi.is_nan() || lo >= hi {
            return Err(Error::Msg(format!("histogramdd: range {j} must have low < high, got ({lo}, {hi})")));
        }
        lims.push((lo, hi));
    }
    let total_bins: usize = bins.iter().product();
    let mut hist = vec![0.0f64; total_bins];
    'rows: for i in 0..n {
        let mut flat = 0usize;
        for j in 0..d {
            let v = x[i * d + j];
            let (lo, hi) = lims[j];
            if !(v >= lo && v <= hi) {
                continue 'rows;
            }
            let mut b = ((v - lo) / (hi - lo) * bins[j] as f64).floor() as usize;
            if b >= bins[j] {
                b = bins[j] - 1;
            }
            flat = flat * bins[j] + b;
        }
        hist[flat] += w.as_ref().map_or(1.0, |w| w[i]);
    }
    if density {
        let total: f64 = hist.iter().sum();
        let vol: f64 = (0..d).map(|j| (lims[j].1 - lims[j].0) / bins[j] as f64).product();
        if total > 0.0 {
            hist.iter_mut().for_each(|h| *h /= total * vol);
        }
    }
    let dt = if sample.dtype().is_floating_point() { sample.dtype() } else { DType::F32 };
    let dev = sample.device();
    let h = Tensor::from_slice_on(&hist, bins, &dev)?.to_dtype(dt)?;
    let edges = (0..d)
        .map(|j| {
            let (lo, hi) = lims[j];
            let e: Vec<f64> = (0..=bins[j]).map(|k| lo + (hi - lo) * k as f64 / bins[j] as f64).collect();
            Tensor::from_slice_on(&e, &[bins[j] + 1], &dev)?.to_dtype(dt)
        })
        .collect::<Result<_>>()?;
    Ok((h, edges))
}

/// Evaluates an orthogonal polynomial of per-element degree `n` (broadcast to `x`'s shape by the caller) by its
/// three-term recurrence: `"chebyshev_t" | "chebyshev_u" | "chebyshev_v" | "chebyshev_w" | "hermite_h" |
/// "hermite_he" | "laguerre_l" | "legendre_p"`. Negative degrees give 0, as in PyTorch.
pub fn orthogonal_polynomial(kind: &str, x: &Tensor, n: &Tensor) -> Result<Tensor> {
    same_shape(x, n, "orthogonal_polynomial")?;
    let xs = x.to_vec::<f64>()?;
    let ns = n.to_vec::<f64>()?;
    let first = |x: f64| -> Option<f64> {
        Some(match kind {
            "chebyshev_t" | "legendre_p" | "hermite_he" => x,
            "chebyshev_u" | "hermite_h" => 2.0 * x,
            "chebyshev_v" => 2.0 * x - 1.0,
            "chebyshev_w" => 2.0 * x + 1.0,
            "laguerre_l" => 1.0 - x,
            _ => return None,
        })
    };
    if first(0.0).is_none() {
        return Err(Error::Msg(format!("orthogonal_polynomial: unknown kind '{kind}'")));
    }
    let out: Vec<f64> = xs
        .iter()
        .zip(&ns)
        .map(|(&x, &nf)| {
            if nf < 0.0 {
                return 0.0;
            }
            let n = nf as usize;
            if n == 0 {
                return 1.0;
            }
            let (mut p0, mut p1) = (1.0, first(x).unwrap_or(0.0));
            for k in 1..n {
                let kf = k as f64;
                let p2 = match kind {
                    "chebyshev_t" | "chebyshev_u" | "chebyshev_v" | "chebyshev_w" => 2.0 * x * p1 - p0,
                    "hermite_h" => 2.0 * x * p1 - 2.0 * kf * p0,
                    "hermite_he" => x * p1 - kf * p0,
                    "laguerre_l" => ((2.0 * kf + 1.0 - x) * p1 - kf * p0) / (kf + 1.0),
                    _ => ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0), // legendre_p
                };
                p0 = p1;
                p1 = p2;
            }
            p1
        })
        .collect();
    let dt = if x.dtype().is_floating_point() { x.dtype() } else { DType::F32 };
    Tensor::from_slice_on(&out, x.shape(), &x.device())?.to_dtype(dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }

    #[test]
    fn integer_and_float_helpers() {
        let a = Tensor::from_slice(&[12i64, -18, 0, 7], &[4]).unwrap();
        let b = Tensor::from_slice(&[18i64, 12, 5, 0], &[4]).unwrap();
        assert_eq!(gcd_lcm(&a, &b, false).unwrap().to_vec::<i64>().unwrap(), vec![6, 6, 5, 7]);
        assert_eq!(gcd_lcm(&a, &b, true).unwrap().to_vec::<i64>().unwrap(), vec![36, 36, 0, 0]);
        let (m, e) = frexp(&t(&[8.0, -0.75, 0.0, 3.0])).unwrap();
        assert_eq!(m.to_vec::<f64>().unwrap(), vec![0.5, -0.75, 0.0, 0.75]);
        assert_eq!(e.to_vec::<i32>().unwrap(), vec![4, 0, 0, 2]);
        assert_eq!(ldexp(&m, &e.to_dtype(DType::F64).unwrap()).unwrap().to_vec::<f64>().unwrap(), vec![8.0, -0.75, 0.0, 3.0]);
        let up = nextafter(&t(&[1.0, 0.0, -1.0]), &t(&[2.0, -1.0, -2.0])).unwrap().to_vec::<f64>().unwrap();
        assert_eq!(up[0], f64::from_bits(1.0f64.to_bits() + 1));
        assert!(up[1] < 0.0 && up[1] > -1e-300);
        assert_eq!(float_class(&t(&[-0.0, 0.0, f64::NEG_INFINITY]), "signbit").unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
    }

    #[test]
    fn histogram_and_polynomials() {
        let s = Tensor::from_slice(&[0.1, 0.1, 0.9, 0.9, 0.6, 0.2, 1.0, 1.0], &[4, 2]).unwrap();
        let (h, e) = histogramdd(&s, &[2, 2], Some(&[(0.0, 1.0), (0.0, 1.0)]), None, false).unwrap();
        assert_eq!(h.to_vec::<f64>().unwrap(), vec![1.0, 0.0, 1.0, 2.0]);
        assert_eq!(e[0].to_vec::<f64>().unwrap(), vec![0.0, 0.5, 1.0]);
        let x = t(&[0.3, -1.2]);
        let n = t(&[3.0, 3.0]);
        let he = orthogonal_polynomial("hermite_he", &x, &n).unwrap().to_vec::<f64>().unwrap();
        assert!((he[0] - (0.027 - 0.9)).abs() < 1e-12);
        let t3 = orthogonal_polynomial("chebyshev_t", &x, &n).unwrap().to_vec::<f64>().unwrap();
        assert!((t3[0] - (4.0 * 0.027 - 0.9)).abs() < 1e-12);
        assert_eq!(orthogonal_polynomial("legendre_p", &x, &t(&[-1.0, 0.0])).unwrap().to_vec::<f64>().unwrap(), vec![0.0, 1.0]);
    }
}
