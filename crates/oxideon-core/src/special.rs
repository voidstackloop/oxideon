//! `torch.special`-style functions with autograd: `erfc`, `erfinv`, `lgamma`, `digamma`, `polygamma`, `zeta`,
//! `igamma` / `igammac`, the normal distribution helpers (`ndtr`, `log_ndtr`, `ndtri`), `entr`, `xlog1py`, modified
//! and ordinary Bessel functions.
//!
//! Forward values are computed in `f64` on the host (with `libm` where it has the function); gradients are built from
//! other differentiable ops, so higher-order derivatives work for the chains `lgamma -> digamma -> polygamma`,
//! `erfc`, `erfinv`, `ndtr`, `log_ndtr`, `ndtri` and `entr`.

use crate::backend::{CpuStorage, Storage};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::{Layout, broadcast_shapes};
use crate::tensor::Tensor;
use std::f64::consts::{FRAC_2_SQRT_PI, PI, SQRT_2};

// ---------------------------------------------------------------------------------------------------------------
// Scalar kernels
// ---------------------------------------------------------------------------------------------------------------

/// `psi(x)`, the logarithmic derivative of the gamma function.
pub fn digamma_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return if x.is_sign_negative() { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    if x < 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // Reflection: psi(1 - x) - psi(x) = pi cot(pi x).
        return digamma_f64(1.0 - x) - PI / (PI * x).tan();
    }
    let mut x = x;
    let mut acc = 0.0;
    while x < 10.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // ln x - 1/(2x) - sum B_2k / (2k x^2k)
    let series = inv2
        * (1.0 / 12.0
            - inv2
                * (1.0 / 120.0
                    - inv2
                        * (1.0 / 252.0
                            - inv2
                                * (1.0 / 240.0
                                    - inv2
                                        * (1.0 / 132.0
                                            - inv2 * (691.0 / 32760.0 - inv2 / 12.0))))));
    acc + x.ln() - 0.5 * inv - series
}

/// Hurwitz zeta `sum_{k>=0} (q + k)^-s` by Euler-Maclaurin summation (`s > 1`, `q > 0`; other inputs follow Cephes).
pub fn zeta_f64(s: f64, q: f64) -> f64 {
    const A: [f64; 12] = [
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.892_437_580_318_379_2e9,
        7.472_424_96e10,
        -2.950_130_727_918_164e12,
        1.164_678_281_435_006_7e14,
        -4.597_978_722_407_473e15,
        1.815_210_540_194_354_7e17,
        -7.166_165_256_175_667e18,
    ];
    if s.is_nan() || q.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY;
    }
    if s < 1.0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        if q == q.floor() {
            return f64::INFINITY;
        }
        if s != s.floor() {
            return f64::NAN; // q^-s is complex
        }
    }
    let mut sum = q.powf(-s);
    let mut a = q;
    let mut i = 0;
    let mut b = 0.0;
    while i < 9 || a <= 9.0 {
        i += 1;
        a += 1.0;
        b = a.powf(-s);
        sum += b;
        if (b / sum).abs() < 1e-17 {
            return sum;
        }
    }
    let w = a;
    sum += b * w / (s - 1.0);
    sum -= 0.5 * b;
    let mut fac = 1.0;
    let mut k = 0.0;
    for ai in A {
        fac *= s + k;
        b /= w;
        let t = fac * b / ai;
        sum += t;
        if (t / sum).abs() < 1e-17 {
            break;
        }
        k += 1.0;
        fac *= s + k;
        b /= w;
        k += 1.0;
    }
    sum
}

/// `psi^(n)(x)`, the `n`-th derivative of the digamma function.
pub fn polygamma_f64(n: u32, x: f64) -> f64 {
    if n == 0 {
        return digamma_f64(x);
    }
    if x.is_nan() {
        return x;
    }
    let nf = f64::from(n);
    let fact = libm::tgamma(nf + 1.0);
    let sign = if n % 2 == 0 { -1.0 } else { 1.0 }; // (-1)^(n+1)
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.0 {
        // psi^(n)(x) = psi^(n)(x + 1) - (-1)^n n! x^-(n+1)
        let mut acc = 0.0;
        let mut x = x;
        while x < 0.0 {
            acc -= (-1.0f64).powi(n as i32) * fact * x.powf(-(nf + 1.0));
            x += 1.0;
        }
        return acc + polygamma_f64(n, x);
    }
    sign * fact * zeta_f64(nf + 1.0, x)
}

/// Inverse error function: Giles' approximation refined by Newton steps (on `erfc` for the tails).
pub fn erfinv_f64(y: f64) -> f64 {
    if y.is_nan() || !(-1.0..=1.0).contains(&y) {
        return f64::NAN;
    }
    if y == 1.0 {
        return f64::INFINITY;
    }
    if y == -1.0 {
        return f64::NEG_INFINITY;
    }
    if y == 0.0 {
        return y;
    }
    let (sign, a) = if y < 0.0 { (-1.0, -y) } else { (1.0, y) };
    // initial guess (Giles 2010, single precision polynomial)
    let w = -((1.0 - a) * (1.0 + a)).ln();
    let mut x = if w < 5.0 {
        let w = w - 2.5;
        let mut p = 2.810_226_36e-8;
        for c in [
            3.432_739_39e-7,
            -3.523_387_7e-6,
            -4.391_506_54e-6,
            2.185_808_7e-4,
            -1.253_725_03e-3,
            -4.177_681_64e-3,
            2.466_407_27e-1,
            1.501_409_41,
        ] {
            p = c + p * w;
        }
        p * a
    } else {
        let w = w.sqrt() - 3.0;
        let mut p = -2.002_142_57e-4;
        for c in [
            1.009_505_58e-4,
            1.349_343_22e-3,
            -3.673_428_44e-3,
            5.739_507_73e-3,
            -7.622_461_3e-3,
            9.438_870_47e-3,
            1.001_674_06,
            2.832_976_82,
        ] {
            p = c + p * w;
        }
        p * a
    };
    let c = 1.0 - a; // exact in double, so the tail keeps its precision
    for _ in 0..3 {
        let (err, d) = if a > 0.9 {
            (libm::erfc(x) - c, -FRAC_2_SQRT_PI * (-x * x).exp())
        } else {
            (libm::erf(x) - a, FRAC_2_SQRT_PI * (-x * x).exp())
        };
        if d == 0.0 {
            break;
        }
        let step = err / d;
        // Halley correction
        x -= step / (1.0 + x * step);
    }
    sign * x
}

/// Standard normal quantile function.
pub fn ndtri_f64(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    // Acklam's rational approximation, then one Halley step with the exact cdf.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let lo = 0.02425;
    let mut x = if p < lo {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - lo {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    for _ in 0..2 {
        let e = 0.5 * libm::erfc(-x / SQRT_2) - p;
        let u = e * (2.0 * PI).sqrt() * (x * x / 2.0).exp();
        x -= u / (1.0 + x * u / 2.0);
    }
    x
}

/// `ln Phi(x)` without underflow for very negative `x`.
pub fn log_ndtr_f64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 5.0 {
        return (-0.5 * libm::erfc(x / SQRT_2)).ln_1p();
    }
    if x > -20.0 {
        return (0.5 * libm::erfc(-x / SQRT_2)).ln();
    }
    let x2 = x * x;
    let series = 1.0 - 1.0 / x2 + 3.0 / (x2 * x2) - 15.0 / (x2 * x2 * x2) + 105.0 / (x2 * x2 * x2 * x2);
    -0.5 * x2 - (-x).ln() - 0.5 * (2.0 * PI).ln() + series.ln()
}

/// Regularized lower incomplete gamma `P(a, x)` and its complement `Q(a, x)`.
pub fn gammainc_f64(a: f64, x: f64) -> (f64, f64) {
    if a.is_nan() || x.is_nan() || a < 0.0 || x < 0.0 {
        return (f64::NAN, f64::NAN);
    }
    if a == 0.0 {
        return if x > 0.0 { (1.0, 0.0) } else { (f64::NAN, f64::NAN) };
    }
    if x == 0.0 {
        return (0.0, 1.0);
    }
    if x.is_infinite() {
        return (1.0, 0.0);
    }
    let lg = libm::lgamma(a);
    if x < a + 1.0 {
        // power series
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..10_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        let p = sum * (-x + a * x.ln() - lg).exp();
        (p, 1.0 - p)
    } else {
        // continued fraction (modified Lentz)
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        let q = (-x + a * x.ln() - lg).exp() * h;
        (1.0 - q, q)
    }
}

/// Modified Bessel function of the first kind scaled by `exp(-|x|)`: `order` 0 or 1.
fn bessel_i_scaled(order: u32, x: f64) -> f64 {
    let ax = x.abs();
    let nu2 = 4.0 * f64::from(order * order);
    let v = if ax <= 30.0 {
        let q = ax * ax / 4.0;
        let mut term = if order == 0 { 1.0 } else { ax / 2.0 };
        let mut sum = term;
        let mut k = 1.0;
        loop {
            term *= q / (k * (k + f64::from(order)));
            sum += term;
            if term <= sum * 1e-17 {
                break;
            }
            k += 1.0;
        }
        sum * (-ax).exp()
    } else {
        // asymptotic expansion in 1 / (8 x)
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..60 {
            let kf = f64::from(k);
            let next = -term * (nu2 - (2.0 * kf - 1.0).powi(2)) / (kf * 8.0 * ax);
            if next.abs() >= term.abs() {
                break;
            }
            term = next;
            sum += term;
            if term.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        sum / (2.0 * PI * ax).sqrt()
    };
    if order == 1 && x < 0.0 { -v } else { v }
}

// ---------------------------------------------------------------------------------------------------------------
// Host-side op builders
// ---------------------------------------------------------------------------------------------------------------

pub(crate) fn float_dtype(t: &Tensor) -> DType {
    if matches!(
        t.dtype(),
        DType::F16 | DType::BF16 | DType::F32 | DType::F64
    ) {
        t.dtype()
    } else {
        DType::F32
    }
}

fn storage_from(vals: Vec<f64>, shape: &[usize], dt: DType, like: &Tensor) -> Result<(Storage, Layout)> {
    let layout = Layout::contiguous(shape.to_vec());
    let cpu = CpuStorage::F64(vals).to_dtype(&layout, dt)?;
    let storage = Storage::new_cpu(cpu).to_device(&layout, &like.device())?;
    Ok((storage, layout))
}

/// Elementwise `f64` map with a tensor-valued backward `(grad, input, output) -> grad_input`.
pub(crate) fn unary(
    x: &Tensor,
    name: &'static str,
    fwd: impl Fn(f64) -> f64,
    bwd: impl Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + Send + Sync + 'static,
) -> Result<Tensor> {
    let dt = float_dtype(x);
    let vals: Vec<f64> = x.to_vec::<f64>()?.into_iter().map(fwd).collect();
    let (storage, layout) = storage_from(vals, x.shape(), dt, x)?;
    let inp = x.to_dtype(dt)?;
    let out = Tensor::from_storage(storage.clone(), layout.clone());
    Ok(Tensor::from_op(storage, layout, name, vec![x.clone()], move |g| {
        Ok(vec![Some(bwd(g, &inp, &out)?)])
    }))
}

/// Broadcasting binary `f64` map; `da` / `db` are the partial derivatives (first-order gradients only).
pub(crate) fn binary(
    a: &Tensor,
    b: &Tensor,
    name: &'static str,
    fwd: impl Fn(f64, f64) -> f64,
    da: Option<fn(f64, f64) -> f64>,
    db: Option<fn(f64, f64) -> f64>,
) -> Result<Tensor> {
    let shape = broadcast_shapes(name, a.shape(), b.shape())?;
    let av = a.broadcast_to(&shape)?.to_vec::<f64>()?;
    let bv = b.broadcast_to(&shape)?.to_vec::<f64>()?;
    let dt = if float_dtype(a) == DType::F64 || float_dtype(b) == DType::F64 {
        DType::F64
    } else {
        float_dtype(a)
    };
    let vals: Vec<f64> = av.iter().zip(&bv).map(|(&x, &y)| fwd(x, y)).collect();
    let (storage, layout) = storage_from(vals, &shape, dt, a)?;
    let (sa, sb) = (a.shape().to_vec(), b.shape().to_vec());
    let (ra, rb) = (a.requires_grad(), b.requires_grad());
    let dev = a.device();
    Ok(Tensor::from_op(storage, layout, name, vec![a.clone(), b.clone()], move |g| {
        let gv = g.to_vec::<f64>()?;
        let grad = |d: fn(f64, f64) -> f64, target: &[usize]| -> Result<Tensor> {
            let v: Vec<f64> = gv
                .iter()
                .zip(av.iter().zip(&bv))
                .map(|(&gi, (&x, &y))| gi * d(x, y))
                .collect();
            Tensor::from_slice_on(&v, &shape, &dev)?
                .to_dtype(dt)?
                .sum_to_shape(target)
        };
        Ok(vec![
            match (ra, da) {
                (true, Some(d)) => Some(grad(d, &sa)?),
                _ => None,
            },
            match (rb, db) {
                (true, Some(d)) => Some(grad(d, &sb)?),
                _ => None,
            },
        ])
    }))
}

// ---------------------------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------------------------

impl Tensor {
    /// Complementary error function `1 - erf(x)`, accurate in the upper tail.
    pub fn erfc(&self) -> Result<Tensor> {
        unary(self, "erfc", libm::erfc, |g, x, _| {
            g.mul(&x.square()?.neg()?.exp()?.mul_scalar(-FRAC_2_SQRT_PI)?)
        })
    }

    /// Inverse error function (`torch.erfinv`).
    pub fn erfinv(&self) -> Result<Tensor> {
        unary(self, "erfinv", erfinv_f64, |g, _, y| {
            // d/dy erfinv = sqrt(pi) / 2 * exp(erfinv(y)^2)
            g.mul(&y.square()?.exp()?.mul_scalar(PI.sqrt() / 2.0)?)
        })
    }

    /// `ln |Gamma(x)|` (`torch.lgamma`).
    pub fn lgamma(&self) -> Result<Tensor> {
        unary(self, "lgamma", libm::lgamma, |g, x, _| g.mul(&x.digamma()?))
    }

    /// Logarithmic derivative of the gamma function (`torch.digamma`).
    pub fn digamma(&self) -> Result<Tensor> {
        unary(self, "digamma", digamma_f64, |g, x, _| g.mul(&x.polygamma(1)?))
    }

    /// `n`-th derivative of the digamma function (`torch.polygamma(n, x)`).
    pub fn polygamma(&self, n: u32) -> Result<Tensor> {
        if n == 0 {
            return self.digamma();
        }
        unary(
            self,
            "polygamma",
            move |x| polygamma_f64(n, x),
            move |g, x, _| g.mul(&x.polygamma(n + 1)?),
        )
    }

    /// Multivariate log-gamma with dimension `p` (`torch.mvlgamma`).
    pub fn mvlgamma(&self, p: usize) -> Result<Tensor> {
        if p == 0 {
            return Err(Error::Msg("mvlgamma: p must be at least 1".into()));
        }
        let c = (p * (p - 1)) as f64 / 4.0 * PI.ln();
        let mut acc = self.lgamma()?;
        for j in 1..p {
            acc = acc.add(&self.sub_scalar(j as f64 / 2.0)?.lgamma()?)?;
        }
        acc.add_scalar(c)
    }

    /// Hurwitz zeta function `zeta(self, q)` (`torch.special.zeta`); gradients flow to `q` only.
    pub fn zeta(&self, q: &Tensor) -> Result<Tensor> {
        binary(
            self,
            q,
            "zeta",
            zeta_f64,
            None,
            Some(|s, q| -s * zeta_f64(s + 1.0, q)),
        )
    }

    /// Standard normal cdf (`torch.special.ndtr`).
    pub fn ndtr(&self) -> Result<Tensor> {
        unary(
            self,
            "ndtr",
            |x| 0.5 * libm::erfc(-x / SQRT_2),
            |g, x, _| {
                let pdf = x.square()?.mul_scalar(-0.5)?.exp()?.mul_scalar(1.0 / (2.0 * PI).sqrt())?;
                g.mul(&pdf)
            },
        )
    }

    /// Log of the standard normal cdf, stable far into the left tail (`torch.special.log_ndtr`).
    pub fn log_ndtr(&self) -> Result<Tensor> {
        unary(self, "log_ndtr", log_ndtr_f64, |g, x, y| {
            // d/dx ln Phi = phi / Phi = exp(ln phi - ln Phi)
            let log_pdf = x.square()?.mul_scalar(-0.5)?.add_scalar(-0.5 * (2.0 * PI).ln())?;
            g.mul(&log_pdf.sub(y)?.exp()?)
        })
    }

    /// Standard normal quantile function (`torch.special.ndtri`).
    pub fn ndtri(&self) -> Result<Tensor> {
        unary(self, "ndtri", ndtri_f64, |g, _, y| {
            // d/dp ndtri = 1 / phi(ndtri(p))
            let pdf = y.square()?.mul_scalar(-0.5)?.exp()?.mul_scalar(1.0 / (2.0 * PI).sqrt())?;
            g.div(&pdf)
        })
    }

    /// Elementwise entropy `-x ln x` (`0` at `0`, `-inf` for negative inputs; `torch.special.entr`).
    pub fn entr(&self) -> Result<Tensor> {
        unary(
            self,
            "entr",
            |x| {
                if x.is_nan() {
                    x
                } else if x > 0.0 {
                    -x * x.ln()
                } else if x == 0.0 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                }
            },
            |g, x, _| g.mul(&x.ln()?.add_scalar(1.0)?.neg()?),
        )
    }

    /// `self * log1p(other)` with `0 * log1p(-1) = 0` (`torch.special.xlog1py`).
    pub fn xlog1py(&self, other: &Tensor) -> Result<Tensor> {
        let prod = self.mul(&other.log1p()?)?;
        let zero = self.eq_tensor(&self.zeros_like()?)?;
        let nan = other.isnan()?;
        let keep = zero.logical_and(&nan.logical_not()?)?;
        Tensor::where_cond(&keep, &prod.zeros_like()?, &prod)
    }

    /// Regularized lower incomplete gamma function `P(a, x)` (`torch.special.gammainc`); gradients flow to `x` only.
    pub fn igamma(&self, x: &Tensor) -> Result<Tensor> {
        if self.requires_grad() {
            return Err(Error::Msg(
                "igamma: the derivative with respect to the shape parameter `a` is not implemented".into(),
            ));
        }
        binary(
            self,
            x,
            "igamma",
            |a, x| gammainc_f64(a, x).0,
            None,
            Some(|a, x| {
                if x == 0.0 && a == 1.0 {
                    1.0
                } else {
                    ((a - 1.0) * x.ln() - x - libm::lgamma(a)).exp()
                }
            }),
        )
    }

    /// Regularized upper incomplete gamma function `Q(a, x)` (`torch.special.gammaincc`).
    pub fn igammac(&self, x: &Tensor) -> Result<Tensor> {
        if self.requires_grad() {
            return Err(Error::Msg(
                "igammac: the derivative with respect to the shape parameter `a` is not implemented".into(),
            ));
        }
        binary(
            self,
            x,
            "igammac",
            |a, x| gammainc_f64(a, x).1,
            None,
            Some(|a, x| {
                if x == 0.0 && a == 1.0 {
                    -1.0
                } else {
                    -((a - 1.0) * x.ln() - x - libm::lgamma(a)).exp()
                }
            }),
        )
    }

    /// Modified Bessel function of the first kind, order 0 (`torch.special.i0`).
    pub fn i0(&self) -> Result<Tensor> {
        unary(self, "i0", |x| bessel_i_scaled(0, x) * x.abs().exp(), |g, x, _| g.mul(&x.i1()?))
    }

    /// Exponentially scaled `i0(x) * exp(-|x|)` (`torch.special.i0e`).
    pub fn i0e(&self) -> Result<Tensor> {
        unary(self, "i0e", |x| bessel_i_scaled(0, x), |g, x, y| {
            g.mul(&x.i1e()?.sub(&x.sign()?.mul(y)?)?)
        })
    }

    /// Modified Bessel function of the first kind, order 1 (`torch.special.i1`).
    pub fn i1(&self) -> Result<Tensor> {
        unary(self, "i1", |x| bessel_i_scaled(1, x) * x.abs().exp(), |g, x, y| {
            // i1' = i0 - i1 / x, with the limit 1/2 at 0
            let safe = x.abs()?.lt(&x.full_like(1e-12)?)?;
            let one = x.full_like(1.0)?;
            let inv = Tensor::where_cond(&safe, &one, x)?;
            let d = x.i0()?.sub(&y.div(&inv)?)?;
            g.mul(&Tensor::where_cond(&safe, &x.full_like(0.5)?, &d)?)
        })
    }

    /// Exponentially scaled `i1(x) * exp(-|x|)` (`torch.special.i1e`).
    pub fn i1e(&self) -> Result<Tensor> {
        unary(self, "i1e", |x| bessel_i_scaled(1, x), |g, x, y| {
            let safe = x.abs()?.lt(&x.full_like(1e-12)?)?;
            let one = x.full_like(1.0)?;
            let inv = Tensor::where_cond(&safe, &one, x)?;
            let d = x.i0e()?.sub(&y.mul(&x.sign()?.add(&inv.recip()?)?)?)?;
            g.mul(&Tensor::where_cond(&safe, &x.full_like(0.5)?, &d)?)
        })
    }

    /// Bessel function of the first kind, order 0 (`torch.special.bessel_j0`).
    pub fn bessel_j0(&self) -> Result<Tensor> {
        unary(self, "bessel_j0", libm::j0, |g, x, _| g.mul(&x.bessel_j1()?.neg()?))
    }

    /// Bessel function of the first kind, order 1 (`torch.special.bessel_j1`).
    pub fn bessel_j1(&self) -> Result<Tensor> {
        unary(self, "bessel_j1", libm::j1, |g, x, y| {
            let safe = x.abs()?.lt(&x.full_like(1e-12)?)?;
            let inv = Tensor::where_cond(&safe, &x.full_like(1.0)?, x)?;
            let d = x.bessel_j0()?.sub(&y.div(&inv)?)?;
            g.mul(&Tensor::where_cond(&safe, &x.full_like(0.5)?, &d)?)
        })
    }

    /// Bessel function of the second kind, order 0 (`torch.special.bessel_y0`).
    pub fn bessel_y0(&self) -> Result<Tensor> {
        unary(self, "bessel_y0", libm::y0, |g, x, _| g.mul(&x.bessel_y1()?.neg()?))
    }

    /// Bessel function of the second kind, order 1 (`torch.special.bessel_y1`).
    pub fn bessel_y1(&self) -> Result<Tensor> {
        unary(self, "bessel_y1", libm::y1, |g, x, y| {
            g.mul(&x.bessel_y0()?.sub(&y.div(x)?)?)
        })
    }

    /// `sin(x) / x` with value 1 at 0 (`torch.special.spherical_bessel_j0`).
    pub fn spherical_bessel_j0(&self) -> Result<Tensor> {
        self.mul_scalar(1.0 / PI)?.sinc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }
    fn close(a: &Tensor, want: &[f64], tol: f64) {
        let got = a.to_vec::<f64>().unwrap();
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g - w).abs() <= tol * (1.0 + w.abs()),
                "got {g}, want {w} (all: {got:?})"
            );
        }
    }

    #[test]
    fn modified_bessel_i1_is_defined_at_zero() {
        // regression: the series loop never terminated at x = 0 for order 1 (term == sum == 0)
        close(&t(&[0.0, -0.0, 1e-300]).i1().unwrap(), &[0.0, 0.0, 0.0], 1e-12);
        close(&t(&[0.0]).i1e().unwrap(), &[0.0], 1e-12);
        close(&t(&[0.0]).i0().unwrap(), &[1.0], 1e-12);
    }

    #[test]
    fn known_values() {
        close(&t(&[0.5, 1.0, 5.0]).lgamma().unwrap(), &[0.572_364_942_924_700_1, 0.0, 3.178_053_830_347_945_8], 1e-12);
        close(&t(&[1.0, 0.5, -0.5]).digamma().unwrap(), &[-0.577_215_664_901_532_9, -1.963_510_026_021_423_5, 0.036_489_973_978_576_5], 1e-12);
        close(&t(&[1.0, 2.0]).polygamma(1).unwrap(), &[1.644_934_066_848_226_4, 0.644_934_066_848_226_4], 1e-12);
        close(&t(&[1.0]).polygamma(2).unwrap(), &[-2.404_113_806_319_188_6], 1e-12);
        close(&t(&[2.0, 4.0]).zeta(&t(&[1.0, 1.0])).unwrap(), &[1.644_934_066_848_226_4, 1.082_323_233_711_138_2], 1e-12);
        close(&t(&[0.0, 0.3, 0.9]).erfinv().unwrap(), &[0.0, 0.272_462_714_726_754_4, 1.163_087_153_676_674], 1e-10);
        for y in [1e-9, 0.5, 0.999_999, 1.0 - 1e-12, -0.99] {
            let x = erfinv_f64(y);
            let back = if y > 0.9 { 1.0 - libm::erfc(x) } else { libm::erf(x) };
            assert!((back - y).abs() < 1e-15 * 4.0, "erfinv({y}) = {x}, erf back = {back}");
        }
        close(&t(&[0.5, 0.975, 1e-10]).ndtri().unwrap(), &[0.0, 1.959_963_984_540_054, -6.361_340_902_404_056], 1e-10);
        close(&t(&[0.0, 1.0, 3.0]).erfc().unwrap(), &[1.0, 0.157_299_207_050_285_13, 2.209_049_699_858_544e-5], 1e-10);
        close(&t(&[0.0, -40.0]).log_ndtr().unwrap(), &[-std::f64::consts::LN_2, -804.608_442_013_754_4], 1e-9);
        close(&t(&[0.0, 1.0, 0.5]).i0().unwrap(), &[1.0, 1.266_065_877_752_008_4, 1.063_483_370_741_323], 1e-12);
        close(&t(&[1.0, 0.5, -2.0]).i1().unwrap(), &[0.565_159_103_992_485, 0.257_894_305_390_021, -1.590_636_854_637_329], 1e-12);
        close(&t(&[50.0]).i0e().unwrap(), &[0.056_562_172_5], 1e-6);
        close(&t(&[0.0, 1.0]).bessel_j0().unwrap(), &[1.0, 0.765_197_686_557_966_6], 1e-12);
        // P(1, x) = 1 - e^-x
        close(&t(&[1.0, 1.0]).igamma(&t(&[0.5, 3.0])).unwrap(), &[1.0 - (-0.5f64).exp(), 1.0 - (-3.0f64).exp()], 1e-12);
        close(&t(&[2.5]).igammac(&t(&[7.0])).unwrap(), &[0.015_609_416_100_266_9], 1e-9);
        close(&t(&[0.0, 0.5]).entr().unwrap(), &[0.0, 0.346_573_590_279_972_65], 1e-12);
        close(&t(&[3.0]).mvlgamma(2).unwrap(), &[(PI).ln() / 2.0 + libm::lgamma(3.0) + libm::lgamma(2.5)], 1e-12);
    }

    /// Central-difference check of `d/dx sum(f(x))` against the autograd gradient.
    fn grad_check(f: impl Fn(&Tensor) -> Tensor, xs: &[f64]) {
        let x = t(xs).requires_grad_(true);
        f(&x).sum().unwrap().backward().unwrap();
        let g = x.grad().unwrap().to_vec::<f64>().unwrap();
        for (i, &x0) in xs.iter().enumerate() {
            let h = 1e-6;
            let at = |v: f64| {
                let mut p = xs.to_vec();
                p[i] = v;
                f(&t(&p)).sum().unwrap().item::<f64>().unwrap()
            };
            let num = (at(x0 + h) - at(x0 - h)) / (2.0 * h);
            assert!((g[i] - num).abs() <= 1e-5 * (1.0 + num.abs()), "x={x0}: {} vs {num}", g[i]);
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        grad_check(|x| x.lgamma().unwrap(), &[0.3, 1.7, 4.2]);
        grad_check(|x| x.digamma().unwrap(), &[0.3, 1.7, 4.2]);
        grad_check(|x| x.polygamma(2).unwrap(), &[0.9, 2.2]);
        grad_check(|x| x.erfc().unwrap(), &[-0.5, 0.3, 1.4]);
        grad_check(|x| x.erfinv().unwrap(), &[-0.6, 0.1, 0.8]);
        grad_check(|x| x.ndtr().unwrap(), &[-1.0, 0.2, 2.0]);
        grad_check(|x| x.log_ndtr().unwrap(), &[-8.0, -1.0, 0.5, 3.0]);
        grad_check(|x| x.ndtri().unwrap(), &[0.1, 0.5, 0.93]);
        grad_check(|x| x.entr().unwrap(), &[0.2, 0.5, 0.9]);
        grad_check(|x| x.i0().unwrap(), &[-1.5, 0.4, 2.0]);
        grad_check(|x| x.i1().unwrap(), &[-1.5, 0.4, 2.0]);
        grad_check(|x| x.i0e().unwrap(), &[-1.5, 0.4, 2.0]);
        grad_check(|x| x.i1e().unwrap(), &[-1.5, 0.4, 2.0]);
        grad_check(|x| x.bessel_j0().unwrap(), &[0.4, 2.0, 5.5]);
        grad_check(|x| x.bessel_j1().unwrap(), &[0.4, 2.0, 5.5]);
        grad_check(|x| x.bessel_y0().unwrap(), &[0.4, 2.0, 5.5]);
        grad_check(|x| x.bessel_y1().unwrap(), &[0.4, 2.0, 5.5]);
        grad_check(|x| t(&[2.5, 0.7]).igamma(x).unwrap(), &[0.5, 3.0]);
        grad_check(|x| t(&[3.0, 2.0]).zeta(x).unwrap(), &[0.5, 2.5]);
    }

    #[test]
    fn second_derivative_through_the_gamma_chain() {
        // d2/dx2 lgamma = polygamma(1, x)
        let x = t(&[1.3]).requires_grad_(true);
        let y = x.lgamma().unwrap().sum().unwrap();
        let g = crate::tensor::grad(&[y], std::slice::from_ref(&x), None, true).unwrap();
        g[0].as_ref().unwrap().sum().unwrap().backward().unwrap();
        close(&x.grad().unwrap(), &[polygamma_f64(1, 1.3)], 1e-9);
    }
}
