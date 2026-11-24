//! More probability distributions in the style of `torch.distributions`: `Bernoulli`, `Binomial`, `Poisson`,
//! `Geometric`, `Uniform`, `Exponential`, `Gamma`, `Chi2`, `Beta`, `Dirichlet`, `Cauchy`, `StudentT`, `LogNormal`,
//! `HalfNormal`, `Gumbel`, `MultivariateNormal`, and the [`Independent`] wrapper, all behind a common
//! [`Distribution`] trait (also implemented for the existing `Normal` and `Laplace`).
//!
//! Continuous families expose `rsample` with pathwise gradients wherever PyTorch does; gamma-based families use the
//! implicit reparameterization gradient. Closed-form `kl_divergence` is provided for same-family pairs.


use crate::distributions::{Laplace, Normal};
use crate::error::{Error, Result};
use crate::layout::broadcast_shapes;
use crate::rng;
use crate::special::{float_dtype, gammainc_f64, unary};
use crate::tensor::Tensor;
use std::f64::consts::{E, PI, TAU};

/// Common interface of the distributions in this module.
pub trait Distribution {
    /// Draws a sample without gradients.
    fn sample(&self) -> Result<Tensor>;
    /// Draws a sample that is differentiable with respect to the parameters, where the family allows it.
    fn rsample(&self) -> Result<Tensor> {
        Err(Error::Msg("this distribution has no reparameterized sampler".into()))
    }
    fn log_prob(&self, value: &Tensor) -> Result<Tensor>;
    fn entropy(&self) -> Result<Tensor>;
    fn mean(&self) -> Result<Tensor>;
    fn variance(&self) -> Result<Tensor>;
    /// `sqrt(variance)`.
    fn stddev(&self) -> Result<Tensor> {
        self.variance()?.sqrt()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------------------------

fn bshape(name: &'static str, ts: &[&Tensor]) -> Result<Vec<usize>> {
    let mut s = ts[0].shape().to_vec();
    for t in &ts[1..] {
        s = broadcast_shapes(name, &s, t.shape())?;
    }
    Ok(s)
}

fn host(t: &Tensor, shape: &[usize]) -> Result<Vec<f64>> {
    t.broadcast_to(shape)?.to_vec::<f64>()
}

fn from_host(v: &[f64], shape: &[usize], like: &Tensor) -> Result<Tensor> {
    Tensor::from_slice_on(v, shape, &like.device())?.to_dtype(float_dtype(like))
}

/// A host-computed value `v` of `param` (broadcast to `shape`) that stays differentiable: its gradient w.r.t. `param`
/// is the host-computed derivative `dv`. Built as `v + (param - param.detach()) * dv`.
fn with_host_grad(v: &[f64], dv: &[f64], shape: &[usize], param: &Tensor) -> Result<Tensor> {
    let p = param.broadcast_to(shape)?;
    from_host(v, shape, param)?.add(&p.sub(&p.detach())?.mul(&from_host(dv, shape, param)?)?)
}

/// Tensor of i.i.d. draws of `f` with the shape/dtype/device of `like`.
fn draw(shape: &[usize], like: &Tensor, mut f: impl FnMut(&mut rng::Rng, usize) -> f64) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let v: Vec<f64> = rng::with_rng(|r| (0..n).map(|i| f(r, i)).collect());
    from_host(&v, shape, like)
}

fn u01(r: &mut rng::Rng) -> f64 {
    loop {
        let u = r.next_f64();
        if u > 0.0 {
            return u;
        }
    }
}

fn normal64(r: &mut rng::Rng) -> f64 {
    (-2.0 * u01(r).ln()).sqrt() * (TAU * r.next_f64()).cos()
}

fn check_positive(op: &'static str, t: &Tensor) -> Result<()> {
    if !t.to_vec::<f64>()?.iter().all(|&v| v.is_finite() && v > 0.0) {
        return Err(Error::Msg(format!("{op}: parameter must be finite and positive everywhere")));
    }
    Ok(())
}

fn lbeta(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.lgamma()?.add(&b.lgamma()?)?.sub(&a.add(b)?.lgamma()?)
}

/// Marsaglia-Tsang gamma sampler with unit rate.
fn gamma_draw(r: &mut rng::Rng, alpha: f64) -> f64 {
    if alpha < 1.0 {
        let g = gamma_draw(r, alpha + 1.0);
        return g * u01(r).powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal64(r);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = u01(r);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// Standard gamma sample with the implicit reparameterization gradient `dz/d alpha = -(dP/d alpha) / pdf(z)`.
fn standard_gamma(alpha: &Tensor) -> Result<Tensor> {
    let shape = alpha.shape().to_vec();
    let a = alpha.to_vec::<f64>()?;
    let z: Vec<f64> = rng::with_rng(|r| a.iter().map(|&al| gamma_draw(r, al)).collect());
    // `unary` maps elementwise, so feed the pre-drawn samples through an index
    let idx = std::cell::Cell::new(0usize);
    let samples = z.clone();
    unary(
        alpha,
        "standard_gamma",
        move |_| {
            let i = idx.get();
            idx.set(i + 1);
            samples[i]
        },
        move |g, x, out| {
            let (xa, zv) = (x.to_vec::<f64>()?, out.to_vec::<f64>()?);
            let dz: Vec<f64> = xa
                .iter()
                .zip(&zv)
                .map(|(&al, &z)| {
                    let h = 1e-5 * al.max(1.0);
                    let dp = (gammainc_f64(al + h, z).0 - gammainc_f64((al - h).max(1e-12), z).0)
                        / (al + h - (al - h).max(1e-12));
                    let pdf = ((al - 1.0) * z.ln() - z - libm::lgamma(al)).exp();
                    if pdf > 0.0 && pdf.is_finite() { -dp / pdf } else { 0.0 }
                })
                .collect();
            g.mul(&Tensor::from_slice_on(&dz, &shape, &g.device())?.to_dtype(g.dtype())?)
        },
    )
}

fn poisson_draw(r: &mut rng::Rng, lam: f64) -> f64 {
    if lam <= 0.0 {
        return 0.0;
    }
    if lam < 10.0 {
        let l = (-lam).exp();
        let (mut k, mut p) = (0.0, 1.0);
        loop {
            k += 1.0;
            p *= r.next_f64();
            if p <= l {
                return k - 1.0;
            }
        }
    }
    // PTRS (Hoermann 1993)
    let smu = lam.sqrt();
    let b = 0.931 + 2.53 * smu;
    let a = -0.059 + 0.02483 * b;
    let inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = r.next_f64() - 0.5;
        let v = r.next_f64();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + inv_alpha.ln() - (a / (us * us) + b).ln() <= -lam + k * lam.ln() - libm::lgamma(k + 1.0) {
            return k;
        }
    }
}

fn binomial_draw(r: &mut rng::Rng, n: f64, p: f64) -> f64 {
    let (mut n, mut p) = (n.round() as i64, p);
    let mut acc = 0i64;
    while n > 0 {
        if p <= 0.0 {
            break;
        }
        if p >= 1.0 {
            acc += n;
            break;
        }
        if n <= 64 {
            acc += (0..n).filter(|_| r.next_f64() < p).count() as i64;
            break;
        }
        // split by the order statistic X ~ Beta(a, b) (Knuth, TAOCP vol. 2)
        let a = 1 + n / 2;
        let b = 1 + n - a;
        let (ga, gb) = (gamma_draw(r, a as f64), gamma_draw(r, b as f64));
        let x = ga / (ga + gb);
        if x >= p {
            n = a - 1;
            p /= x;
        } else {
            acc += a;
            n = b - 1;
            p = (p - x) / (1.0 - x);
        }
    }
    acc as f64
}

// ---------------------------------------------------------------------------------------------------------------
// Bernoulli, Binomial, Poisson, Geometric
// ---------------------------------------------------------------------------------------------------------------

/// Bernoulli distribution parameterized by `logits`.
#[derive(Clone)]
pub struct Bernoulli {
    pub logits: Tensor,
}

impl Bernoulli {
    pub fn from_logits(logits: Tensor) -> Self {
        Self { logits }
    }
    pub fn from_probs(probs: &Tensor) -> Result<Self> {
        let p = probs.clamp(1e-12, 1.0 - 1e-12)?;
        Ok(Self { logits: p.logit(None)? })
    }
    pub fn probs(&self) -> Result<Tensor> {
        self.logits.sigmoid()
    }
    /// `KL(self || other)`.
    pub fn kl_divergence(&self, other: &Bernoulli) -> Result<Tensor> {
        let (p, q) = (self.probs()?, other.probs()?);
        let one = |t: &Tensor| t.neg()?.add_scalar(1.0);
        p.xlogy(&p.div(&q)?)?.add(&one(&p)?.xlogy(&one(&p)?.div(&one(&q)?)?)?)
    }
}

impl Distribution for Bernoulli {
    fn sample(&self) -> Result<Tensor> {
        self.probs()?.detach().bernoulli()
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        v.mul(&self.logits)?.sub(&self.logits.softplus()?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.logits.softplus()?.sub(&self.probs()?.mul(&self.logits)?)
    }
    fn mean(&self) -> Result<Tensor> {
        self.probs()
    }
    fn variance(&self) -> Result<Tensor> {
        let p = self.probs()?;
        p.mul(&p.neg()?.add_scalar(1.0)?)
    }
}

/// Binomial distribution with `total_count` trials and success `logits`.
#[derive(Clone)]
pub struct Binomial {
    pub total_count: Tensor,
    pub logits: Tensor,
}

impl Binomial {
    pub fn new(total_count: Tensor, probs: &Tensor) -> Result<Self> {
        Ok(Self { total_count, logits: Bernoulli::from_probs(probs)?.logits })
    }
    pub fn probs(&self) -> Result<Tensor> {
        self.logits.sigmoid()
    }
}

impl Distribution for Binomial {
    fn sample(&self) -> Result<Tensor> {
        let shape = bshape("Binomial", &[&self.total_count, &self.logits])?;
        let (n, p) = (host(&self.total_count, &shape)?, host(&self.probs()?, &shape)?);
        draw(&shape, &self.logits, |r, i| binomial_draw(r, n[i], p[i]))
    }
    fn log_prob(&self, k: &Tensor) -> Result<Tensor> {
        let n = &self.total_count;
        let lchoose = n.add_scalar(1.0)?.lgamma()?.sub(&k.add_scalar(1.0)?.lgamma()?)?.sub(&n.sub(k)?.add_scalar(1.0)?.lgamma()?)?;
        lchoose.add(&k.mul(&self.logits)?)?.sub(&n.mul(&self.logits.softplus()?)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let shape = bshape("Binomial", &[&self.total_count, &self.logits])?;
        let (n, l) = (host(&self.total_count, &shape)?, host(&self.logits, &shape)?);
        // entropy and its derivative w.r.t. the logit: dH/dl = -sum_k p_k (k - n sigmoid(l)) ln p_k
        let (out, dout): (Vec<f64>, Vec<f64>) = n
            .iter()
            .zip(&l)
            .map(|(&n, &l)| {
                let sp = if l > 30.0 { l } else { l.exp().ln_1p() };
                let mean = n / (1.0 + (-l).exp());
                (0..=n as i64).fold((0.0, 0.0), |(h, dh), k| {
                    let k = k as f64;
                    let lp = libm::lgamma(n + 1.0) - libm::lgamma(k + 1.0) - libm::lgamma(n - k + 1.0) + k * l - n * sp;
                    let plp = lp.exp() * lp;
                    (h - plp, dh - plp * (k - mean))
                })
            })
            .unzip();
        with_host_grad(&out, &dout, &shape, &self.logits)
    }
    fn mean(&self) -> Result<Tensor> {
        self.total_count.mul(&self.probs()?)
    }
    fn variance(&self) -> Result<Tensor> {
        let p = self.probs()?;
        self.total_count.mul(&p)?.mul(&p.neg()?.add_scalar(1.0)?)
    }
}

/// Poisson distribution with `rate`.
#[derive(Clone)]
pub struct Poisson {
    pub rate: Tensor,
}

impl Poisson {
    pub fn new(rate: Tensor) -> Result<Self> {
        check_positive("Poisson::new", &rate)?;
        Ok(Self { rate })
    }
    pub fn kl_divergence(&self, other: &Poisson) -> Result<Tensor> {
        self.rate.mul(&self.rate.div(&other.rate)?.ln()?)?.sub(&self.rate)?.add(&other.rate)
    }
}

impl Distribution for Poisson {
    fn sample(&self) -> Result<Tensor> {
        let lam = self.rate.to_vec::<f64>()?;
        draw(self.rate.shape(), &self.rate, |r, i| poisson_draw(r, lam[i]))
    }
    fn log_prob(&self, k: &Tensor) -> Result<Tensor> {
        k.mul(&self.rate.ln()?)?.sub(&self.rate)?.sub(&k.add_scalar(1.0)?.lgamma()?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let lam = self.rate.to_vec::<f64>()?;
        // entropy and its derivative: dH/d(rate) = -sum_k p_k (k / rate - 1) ln p_k
        let (out, dout): (Vec<f64>, Vec<f64>) = lam
            .iter()
            .map(|&l| {
                let hi = (l + 12.0 * l.sqrt().max(1.0) + 20.0) as i64;
                (0..=hi).fold((0.0, 0.0), |(h, dh), k| {
                    let k = k as f64;
                    let lp = k * l.ln() - l - libm::lgamma(k + 1.0);
                    let plp = lp.exp() * lp;
                    (h - plp, dh - plp * (k / l - 1.0))
                })
            })
            .unzip();
        with_host_grad(&out, &dout, self.rate.shape(), &self.rate)
    }
    fn mean(&self) -> Result<Tensor> {
        Ok(self.rate.clone())
    }
    fn variance(&self) -> Result<Tensor> {
        Ok(self.rate.clone())
    }
}

/// Number of failures before the first success, with success probability `probs`.
#[derive(Clone)]
pub struct Geometric {
    pub probs: Tensor,
}

impl Geometric {
    pub fn new(probs: Tensor) -> Result<Self> {
        let v = probs.to_vec::<f64>()?;
        if !v.iter().all(|&p| p > 0.0 && p <= 1.0) {
            return Err(Error::Msg("Geometric::new: probs must lie in (0, 1]".into()));
        }
        Ok(Self { probs })
    }
}

impl Distribution for Geometric {
    fn sample(&self) -> Result<Tensor> {
        let p = self.probs.to_vec::<f64>()?;
        draw(self.probs.shape(), &self.probs, |r, i| {
            if p[i] >= 1.0 { 0.0 } else { (u01(r).ln() / (-p[i]).ln_1p()).floor() }
        })
    }
    fn log_prob(&self, k: &Tensor) -> Result<Tensor> {
        k.mul(&self.probs.neg()?.log1p()?)?.add(&self.probs.ln()?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let p = &self.probs;
        let q = p.neg()?.add_scalar(1.0)?;
        q.xlogy(&q)?.add(&p.xlogy(p)?)?.neg()?.div(p)
    }
    fn mean(&self) -> Result<Tensor> {
        self.probs.recip()?.sub_scalar(1.0)
    }
    fn variance(&self) -> Result<Tensor> {
        self.probs.neg()?.add_scalar(1.0)?.div(&self.probs.square()?)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Continuous families on the real line / half line
// ---------------------------------------------------------------------------------------------------------------

/// Uniform distribution on `[low, high)`.
#[derive(Clone)]
pub struct Uniform {
    pub low: Tensor,
    pub high: Tensor,
}

impl Uniform {
    pub fn new(low: Tensor, high: Tensor) -> Result<Self> {
        bshape("Uniform::new", &[&low, &high])?;
        let (l, h) = (low.to_vec::<f64>()?, high.to_vec::<f64>()?);
        let n = l.len().max(h.len());
        if (0..n).any(|i| l[i % l.len()] >= h[i % h.len()]) {
            return Err(Error::Msg("Uniform::new: low must be below high".into()));
        }
        Ok(Self { low, high })
    }
    pub fn cdf(&self, v: &Tensor) -> Result<Tensor> {
        v.sub(&self.low)?.div(&self.high.sub(&self.low)?)?.clamp(0.0, 1.0)
    }
    pub fn icdf(&self, p: &Tensor) -> Result<Tensor> {
        self.low.add(&p.mul(&self.high.sub(&self.low)?)?)
    }
    pub fn kl_divergence(&self, other: &Uniform) -> Result<Tensor> {
        // finite only when the support of self lies inside the support of other
        let inside = self.low.ge(&other.low)?.logical_and(&self.high.le(&other.high)?)?;
        let kl = other.high.sub(&other.low)?.div(&self.high.sub(&self.low)?)?.ln()?;
        Tensor::where_cond(&inside, &kl, &kl.full_like(f64::INFINITY)?)
    }
}

impl Distribution for Uniform {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape("Uniform", &[&self.low, &self.high])?;
        let u = draw(&shape, &self.low, |r, _| r.next_f64())?;
        self.low.add(&u.mul(&self.high.sub(&self.low)?)?)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let inside = v.ge(&self.low)?.logical_and(&v.lt(&self.high)?)?;
        let lp = self.high.sub(&self.low)?.ln()?.neg()?.add(&v.zeros_like()?)?;
        Tensor::where_cond(&inside, &lp, &lp.full_like(f64::NEG_INFINITY)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.high.sub(&self.low)?.ln()
    }
    fn mean(&self) -> Result<Tensor> {
        self.low.add(&self.high)?.mul_scalar(0.5)
    }
    fn variance(&self) -> Result<Tensor> {
        self.high.sub(&self.low)?.square()?.div_scalar(12.0)
    }
}

/// Exponential distribution with `rate`.
#[derive(Clone)]
pub struct Exponential {
    pub rate: Tensor,
}

impl Exponential {
    pub fn new(rate: Tensor) -> Result<Self> {
        check_positive("Exponential::new", &rate)?;
        Ok(Self { rate })
    }
    pub fn cdf(&self, v: &Tensor) -> Result<Tensor> {
        v.mul(&self.rate)?.neg()?.exp()?.neg()?.add_scalar(1.0)
    }
    pub fn kl_divergence(&self, other: &Exponential) -> Result<Tensor> {
        let r = other.rate.div(&self.rate)?;
        r.ln()?.neg()?.add(&r)?.sub_scalar(1.0)
    }
}

impl Distribution for Exponential {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let e = draw(self.rate.shape(), &self.rate, |r, _| -u01(r).ln())?;
        e.div(&self.rate)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let lp = self.rate.ln()?.sub(&self.rate.mul(v)?)?;
        Tensor::where_cond(&v.ge(&v.zeros_like()?)?, &lp, &lp.full_like(f64::NEG_INFINITY)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.rate.ln()?.neg()?.add_scalar(1.0)
    }
    fn mean(&self) -> Result<Tensor> {
        self.rate.recip()
    }
    fn variance(&self) -> Result<Tensor> {
        self.rate.square()?.recip()
    }
}

/// Gamma distribution with `concentration` (shape) and `rate`.
#[derive(Clone)]
pub struct Gamma {
    pub concentration: Tensor,
    pub rate: Tensor,
}

impl Gamma {
    pub fn new(concentration: Tensor, rate: Tensor) -> Result<Self> {
        check_positive("Gamma::new", &concentration)?;
        check_positive("Gamma::new", &rate)?;
        bshape("Gamma::new", &[&concentration, &rate])?;
        Ok(Self { concentration, rate })
    }
    pub fn kl_divergence(&self, other: &Gamma) -> Result<Tensor> {
        let (a1, b1, a2, b2) = (&self.concentration, &self.rate, &other.concentration, &other.rate);
        a2.mul(&b1.div(b2)?.ln()?)?
            .sub(&a1.lgamma()?)?
            .add(&a2.lgamma()?)?
            .add(&a1.sub(a2)?.mul(&a1.digamma()?)?)?
            .add(&a1.mul(&b2.sub(b1)?.div(b1)?)?)
    }
}

impl Distribution for Gamma {
    fn sample(&self) -> Result<Tensor> {
        Ok(self.rsample()?.detach())
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape("Gamma", &[&self.concentration, &self.rate])?;
        let alpha = self.concentration.broadcast_to(&shape)?.contiguous()?;
        standard_gamma(&alpha)?.div(&self.rate)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let (a, b) = (&self.concentration, &self.rate);
        a.mul(&b.ln()?)?.add(&a.sub_scalar(1.0)?.mul(&v.ln()?)?)?.sub(&b.mul(v)?)?.sub(&a.lgamma()?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let (a, b) = (&self.concentration, &self.rate);
        a.sub(&b.ln()?)?.add(&a.lgamma()?)?.add(&a.neg()?.add_scalar(1.0)?.mul(&a.digamma()?)?)
    }
    fn mean(&self) -> Result<Tensor> {
        self.concentration.div(&self.rate)
    }
    fn variance(&self) -> Result<Tensor> {
        self.concentration.div(&self.rate.square()?)
    }
}

/// Chi-squared distribution with `df` degrees of freedom (`Gamma(df / 2, 1 / 2)`).
#[derive(Clone)]
pub struct Chi2 {
    pub df: Tensor,
}

impl Chi2 {
    pub fn new(df: Tensor) -> Result<Self> {
        check_positive("Chi2::new", &df)?;
        Ok(Self { df })
    }
    fn gamma(&self) -> Result<Gamma> {
        Gamma::new(self.df.mul_scalar(0.5)?, self.df.full_like(0.5)?)
    }
}

impl Distribution for Chi2 {
    fn sample(&self) -> Result<Tensor> {
        self.gamma()?.sample()
    }
    fn rsample(&self) -> Result<Tensor> {
        self.gamma()?.rsample()
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        self.gamma()?.log_prob(v)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.gamma()?.entropy()
    }
    fn mean(&self) -> Result<Tensor> {
        Ok(self.df.clone())
    }
    fn variance(&self) -> Result<Tensor> {
        self.df.mul_scalar(2.0)
    }
}

/// Beta distribution on `(0, 1)`.
#[derive(Clone)]
pub struct Beta {
    pub concentration1: Tensor,
    pub concentration0: Tensor,
}

impl Beta {
    pub fn new(concentration1: Tensor, concentration0: Tensor) -> Result<Self> {
        check_positive("Beta::new", &concentration1)?;
        check_positive("Beta::new", &concentration0)?;
        bshape("Beta::new", &[&concentration1, &concentration0])?;
        Ok(Self { concentration1, concentration0 })
    }
    pub fn kl_divergence(&self, other: &Beta) -> Result<Tensor> {
        let (a1, b1, a2, b2) = (&self.concentration1, &self.concentration0, &other.concentration1, &other.concentration0);
        let s1 = a1.add(b1)?;
        lbeta(a2, b2)?
            .sub(&lbeta(a1, b1)?)?
            .add(&a1.sub(a2)?.mul(&a1.digamma()?)?)?
            .add(&b1.sub(b2)?.mul(&b1.digamma()?)?)?
            .add(&a2.add(b2)?.sub(&s1)?.mul(&s1.digamma()?)?)
    }
}

impl Distribution for Beta {
    fn sample(&self) -> Result<Tensor> {
        Ok(self.rsample()?.detach())
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape("Beta", &[&self.concentration1, &self.concentration0])?;
        let ga = standard_gamma(&self.concentration1.broadcast_to(&shape)?.contiguous()?)?;
        let gb = standard_gamma(&self.concentration0.broadcast_to(&shape)?.contiguous()?)?;
        ga.div(&ga.add(&gb)?)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let (a, b) = (&self.concentration1, &self.concentration0);
        a.sub_scalar(1.0)?
            .mul(&v.ln()?)?
            .add(&b.sub_scalar(1.0)?.mul(&v.neg()?.log1p()?)?)?
            .sub(&lbeta(a, b)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let (a, b) = (&self.concentration1, &self.concentration0);
        let s = a.add(b)?;
        lbeta(a, b)?
            .sub(&a.sub_scalar(1.0)?.mul(&a.digamma()?)?)?
            .sub(&b.sub_scalar(1.0)?.mul(&b.digamma()?)?)?
            .add(&s.sub_scalar(2.0)?.mul(&s.digamma()?)?)
    }
    fn mean(&self) -> Result<Tensor> {
        let (a, b) = (&self.concentration1, &self.concentration0);
        a.div(&a.add(b)?)
    }
    fn variance(&self) -> Result<Tensor> {
        let (a, b) = (&self.concentration1, &self.concentration0);
        let s = a.add(b)?;
        a.mul(b)?.div(&s.square()?.mul(&s.add_scalar(1.0)?)?)
    }
}

/// Dirichlet distribution; the last dimension of `concentration` indexes the categories.
#[derive(Clone)]
pub struct Dirichlet {
    pub concentration: Tensor,
}

impl Dirichlet {
    pub fn new(concentration: Tensor) -> Result<Self> {
        if concentration.ndim() == 0 {
            return Err(Error::Msg("Dirichlet::new: concentration needs at least one dimension".into()));
        }
        check_positive("Dirichlet::new", &concentration)?;
        Ok(Self { concentration })
    }
    pub fn kl_divergence(&self, other: &Dirichlet) -> Result<Tensor> {
        let (a, b) = (&self.concentration, &other.concentration);
        let a0 = a.sum_axis(-1, true)?;
        let t1 = a0.squeeze_axis(-1)?.lgamma()?.sub(&b.sum_axis(-1, false)?.lgamma()?)?;
        let t2 = a.lgamma()?.sub(&b.lgamma()?)?.sum_axis(-1, false)?;
        let t3 = a.sub(b)?.mul(&a.digamma()?.sub(&a0.digamma()?)?)?.sum_axis(-1, false)?;
        t1.sub(&t2)?.add(&t3)
    }
}

impl Distribution for Dirichlet {
    fn sample(&self) -> Result<Tensor> {
        Ok(self.rsample()?.detach())
    }
    fn rsample(&self) -> Result<Tensor> {
        let g = standard_gamma(&self.concentration.contiguous()?)?;
        g.div(&g.sum_axis(-1, true)?)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let a = &self.concentration;
        a.sub_scalar(1.0)?
            .mul(&v.ln()?)?
            .sum_axis(-1, false)?
            .add(&a.sum_axis(-1, false)?.lgamma()?)?
            .sub(&a.lgamma()?.sum_axis(-1, false)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let a = &self.concentration;
        let k = a.shape()[a.ndim() - 1] as f64;
        let a0 = a.sum_axis(-1, false)?;
        a.lgamma()?
            .sum_axis(-1, false)?
            .sub(&a0.lgamma()?)?
            .add(&a0.sub_scalar(k)?.mul(&a0.digamma()?)?)?
            .sub(&a.sub_scalar(1.0)?.mul(&a.digamma()?)?.sum_axis(-1, false)?)
    }
    fn mean(&self) -> Result<Tensor> {
        self.concentration.div(&self.concentration.sum_axis(-1, true)?)
    }
    fn variance(&self) -> Result<Tensor> {
        let a = &self.concentration;
        let a0 = a.sum_axis(-1, true)?;
        a.mul(&a0.sub(a)?)?.div(&a0.square()?.mul(&a0.add_scalar(1.0)?)?)
    }
}

/// Cauchy distribution.
#[derive(Clone)]
pub struct Cauchy {
    pub loc: Tensor,
    pub scale: Tensor,
}

impl Cauchy {
    pub fn new(loc: Tensor, scale: Tensor) -> Result<Self> {
        check_positive("Cauchy::new", &scale)?;
        bshape("Cauchy::new", &[&loc, &scale])?;
        Ok(Self { loc, scale })
    }
    pub fn cdf(&self, v: &Tensor) -> Result<Tensor> {
        v.sub(&self.loc)?.div(&self.scale)?.atan()?.mul_scalar(1.0 / PI)?.add_scalar(0.5)
    }
    pub fn icdf(&self, p: &Tensor) -> Result<Tensor> {
        self.loc.add(&self.scale.mul(&p.sub_scalar(0.5)?.mul_scalar(PI)?.tan()?)?)
    }
    pub fn kl_divergence(&self, other: &Cauchy) -> Result<Tensor> {
        let num = self.scale.add(&other.scale)?.square()?.add(&self.loc.sub(&other.loc)?.square()?)?;
        num.div(&self.scale.mul(&other.scale)?.mul_scalar(4.0)?)?.ln()
    }
}

impl Distribution for Cauchy {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape("Cauchy", &[&self.loc, &self.scale])?;
        let eps = draw(&shape, &self.loc, |r, _| (PI * (r.next_f64() - 0.5)).tan())?;
        self.loc.add(&self.scale.mul(&eps)?)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let z = v.sub(&self.loc)?.div(&self.scale)?;
        z.square()?.log1p()?.add(&self.scale.ln()?)?.add_scalar(PI.ln())?.neg()
    }
    fn entropy(&self) -> Result<Tensor> {
        self.scale.ln()?.add_scalar((4.0 * PI).ln())?.add(&self.loc.zeros_like()?)
    }
    fn mean(&self) -> Result<Tensor> {
        self.loc.full_like(f64::NAN)
    }
    fn variance(&self) -> Result<Tensor> {
        self.loc.full_like(f64::INFINITY)
    }
}

/// Student's t distribution.
#[derive(Clone)]
pub struct StudentT {
    pub df: Tensor,
    pub loc: Tensor,
    pub scale: Tensor,
}

impl StudentT {
    pub fn new(df: Tensor, loc: Tensor, scale: Tensor) -> Result<Self> {
        check_positive("StudentT::new", &df)?;
        check_positive("StudentT::new", &scale)?;
        bshape("StudentT::new", &[&df, &loc, &scale])?;
        Ok(Self { df, loc, scale })
    }
}

impl Distribution for StudentT {
    fn sample(&self) -> Result<Tensor> {
        Ok(self.rsample()?.detach())
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape("StudentT", &[&self.df, &self.loc, &self.scale])?;
        let normal = draw(&shape, &self.loc, |r, _| normal64(r))?;
        let chi2 = standard_gamma(&self.df.mul_scalar(0.5)?.broadcast_to(&shape)?.contiguous()?)?.mul_scalar(2.0)?;
        let t = normal.mul(&self.df.div(&chi2)?.sqrt()?)?;
        self.loc.add(&self.scale.mul(&t)?)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let nu = &self.df;
        let z = v.sub(&self.loc)?.div(&self.scale)?;
        let half = nu.add_scalar(1.0)?.mul_scalar(0.5)?;
        half.lgamma()?
            .sub(&nu.mul_scalar(0.5)?.lgamma()?)?
            .sub(&nu.mul_scalar(PI)?.ln()?.mul_scalar(0.5)?)?
            .sub(&self.scale.ln()?)?
            .sub(&half.mul(&z.square()?.div(nu)?.log1p()?)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let nu = &self.df;
        let half = nu.add_scalar(1.0)?.mul_scalar(0.5)?;
        let lb = nu.mul_scalar(0.5)?.lgamma()?.add_scalar(0.5 * PI.ln())?
            .sub(&nu.mul_scalar(0.5)?.add_scalar(0.5)?.lgamma()?)?;
        half.mul(&half.digamma()?.sub(&nu.mul_scalar(0.5)?.digamma()?)?)?
            .add(&nu.sqrt()?.ln()?)?
            .add(&lb)?
            .add(&self.scale.ln()?)
    }
    fn mean(&self) -> Result<Tensor> {
        let ok = self.df.gt(&self.df.ones_like()?)?;
        let m = self.loc.add(&self.df.zeros_like()?)?;
        Tensor::where_cond(&ok, &m, &m.full_like(f64::NAN)?)
    }
    fn variance(&self) -> Result<Tensor> {
        let nu = &self.df;
        let finite = self.scale.square()?.mul(&nu.div(&nu.sub_scalar(2.0)?)?)?;
        let big = nu.gt(&nu.full_like(2.0)?)?;
        let mid = nu.gt(&nu.ones_like()?)?;
        let inf = finite.full_like(f64::INFINITY)?;
        Tensor::where_cond(&big, &finite, &Tensor::where_cond(&mid, &inf, &finite.full_like(f64::NAN)?)?)
    }
}

/// Log-normal: `exp(N(loc, scale))`.
#[derive(Clone)]
pub struct LogNormal {
    pub base: Normal,
}

impl LogNormal {
    pub fn new(loc: Tensor, scale: Tensor) -> Result<Self> {
        Ok(Self { base: Normal::new(loc, scale)? })
    }
}

impl Distribution for LogNormal {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        self.base.rsample()?.exp()
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let lv = v.ln()?;
        self.base.log_prob(&lv)?.sub(&lv)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.base.entropy()?.add(&self.base.loc)
    }
    fn mean(&self) -> Result<Tensor> {
        self.base.loc.add(&self.base.scale.square()?.mul_scalar(0.5)?)?.exp()
    }
    fn variance(&self) -> Result<Tensor> {
        let s2 = self.base.scale.square()?;
        s2.exp()?.sub_scalar(1.0)?.mul(&self.base.loc.mul_scalar(2.0)?.add(&s2)?.exp()?)
    }
}

/// Half-normal: `|N(0, scale)|`.
#[derive(Clone)]
pub struct HalfNormal {
    pub scale: Tensor,
}

impl HalfNormal {
    pub fn new(scale: Tensor) -> Result<Self> {
        check_positive("HalfNormal::new", &scale)?;
        Ok(Self { scale })
    }
}

impl Distribution for HalfNormal {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let eps = draw(self.scale.shape(), &self.scale, |r, _| normal64(r))?;
        self.scale.mul(&eps)?.abs()
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let n = Normal::new(self.scale.zeros_like()?, self.scale.clone())?;
        let lp = n.log_prob(v)?.add_scalar(std::f64::consts::LN_2)?;
        Tensor::where_cond(&v.ge(&v.zeros_like()?)?, &lp, &lp.full_like(f64::NEG_INFINITY)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.scale.ln()?.add_scalar(0.5 * (PI * E / 2.0).ln())
    }
    fn mean(&self) -> Result<Tensor> {
        self.scale.mul_scalar((2.0 / PI).sqrt())
    }
    fn variance(&self) -> Result<Tensor> {
        self.scale.square()?.mul_scalar(1.0 - 2.0 / PI)
    }
}

/// Gumbel (type-I extreme value) distribution.
#[derive(Clone)]
pub struct Gumbel {
    pub loc: Tensor,
    pub scale: Tensor,
}

impl Gumbel {
    pub fn new(loc: Tensor, scale: Tensor) -> Result<Self> {
        check_positive("Gumbel::new", &scale)?;
        bshape("Gumbel::new", &[&loc, &scale])?;
        Ok(Self { loc, scale })
    }
}

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

impl Distribution for Gumbel {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape("Gumbel", &[&self.loc, &self.scale])?;
        let g = draw(&shape, &self.loc, |r, _| -(-u01(r).ln()).ln())?;
        self.loc.add(&self.scale.mul(&g)?)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let z = v.sub(&self.loc)?.div(&self.scale)?;
        z.add(&z.neg()?.exp()?)?.neg()?.sub(&self.scale.ln()?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.scale.ln()?.add_scalar(EULER_GAMMA + 1.0)?.add(&self.loc.zeros_like()?)
    }
    fn mean(&self) -> Result<Tensor> {
        self.loc.add(&self.scale.mul_scalar(EULER_GAMMA)?)
    }
    fn variance(&self) -> Result<Tensor> {
        self.scale.square()?.mul_scalar(PI * PI / 6.0)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Multivariate normal and Independent
// ---------------------------------------------------------------------------------------------------------------

/// Multivariate normal `N(loc, L L^T)` parameterized by its lower-triangular Cholesky factor.
#[derive(Clone)]
pub struct MultivariateNormal {
    pub loc: Tensor,
    pub scale_tril: Tensor,
}

impl MultivariateNormal {
    pub fn from_scale_tril(loc: Tensor, scale_tril: Tensor) -> Result<Self> {
        let d = loc.shape().last().copied().unwrap_or(0);
        let n = scale_tril.ndim();
        if loc.ndim() == 0 || n < 2 || scale_tril.shape()[n - 1] != d || scale_tril.shape()[n - 2] != d {
            return Err(Error::Msg("MultivariateNormal: loc must be [..., d] and scale_tril [..., d, d]".into()));
        }
        Ok(Self { loc, scale_tril })
    }
    pub fn from_covariance(loc: Tensor, covariance: &Tensor) -> Result<Self> {
        Self::from_scale_tril(loc, covariance.cholesky()?)
    }
    pub fn covariance(&self) -> Result<Tensor> {
        self.scale_tril.matmul(&self.scale_tril.transpose(-2, -1)?)
    }
    fn dim(&self) -> f64 {
        self.loc.shape()[self.loc.ndim() - 1] as f64
    }
    fn half_log_det(&self) -> Result<Tensor> {
        self.scale_tril.diagonal(0, -2, -1)?.ln()?.sum_axis(-1, false)
    }
    pub fn kl_divergence(&self, other: &MultivariateNormal) -> Result<Tensor> {
        let d = self.dim();
        let m = other.scale_tril.triangular_solve(&self.scale_tril, true, false)?; // Lq^-1 Lp
        let tr = m.square()?.sum_axes(&[-2, -1], false)?;
        let diff = other.loc.sub(&self.loc)?.unsqueeze(-1)?;
        let maha = other.scale_tril.triangular_solve(&diff, true, false)?.square()?.sum_axes(&[-2, -1], false)?;
        other.half_log_det()?.sub(&self.half_log_det()?)?.add(&tr.add(&maha)?.sub_scalar(d)?.mul_scalar(0.5)?)
    }
}

impl Distribution for MultivariateNormal {
    fn sample(&self) -> Result<Tensor> {
        Ok(self.rsample()?.detach())
    }
    fn rsample(&self) -> Result<Tensor> {
        let eps = draw(self.loc.shape(), &self.loc, |r, _| normal64(r))?;
        let z = self.scale_tril.matmul(&eps.unsqueeze(-1)?)?.squeeze_axis(-1)?;
        self.loc.add(&z)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        let diff = v.sub(&self.loc)?.unsqueeze(-1)?;
        let y = self.scale_tril.triangular_solve(&diff, true, false)?;
        let maha = y.square()?.sum_axes(&[-2, -1], false)?;
        maha.add_scalar(self.dim() * TAU.ln())?.mul_scalar(-0.5)?.sub(&self.half_log_det()?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.half_log_det()?.add_scalar(0.5 * self.dim() * (1.0 + TAU.ln()))
    }
    fn mean(&self) -> Result<Tensor> {
        Ok(self.loc.clone())
    }
    fn variance(&self) -> Result<Tensor> {
        self.scale_tril.square()?.sum_axis(-1, false)
    }
}

/// Treats the last `n` batch dimensions of `base` as event dimensions, so `log_prob` sums over them.
pub struct Independent<D: Distribution> {
    pub base: D,
    pub reinterpreted_batch_ndims: usize,
}

impl<D: Distribution> Independent<D> {
    pub fn new(base: D, reinterpreted_batch_ndims: usize) -> Self {
        Self { base, reinterpreted_batch_ndims }
    }
    fn sum_tail(&self, t: Tensor) -> Result<Tensor> {
        let mut t = t;
        for _ in 0..self.reinterpreted_batch_ndims {
            t = t.sum_axis(-1, false)?;
        }
        Ok(t)
    }
}

impl<D: Distribution> Distribution for Independent<D> {
    fn sample(&self) -> Result<Tensor> {
        self.base.sample()
    }
    fn rsample(&self) -> Result<Tensor> {
        self.base.rsample()
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        self.sum_tail(self.base.log_prob(v)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.sum_tail(self.base.entropy()?)
    }
    fn mean(&self) -> Result<Tensor> {
        self.base.mean()
    }
    fn variance(&self) -> Result<Tensor> {
        self.base.variance()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The existing families join the trait
// ---------------------------------------------------------------------------------------------------------------

impl Distribution for Normal {
    fn sample(&self) -> Result<Tensor> {
        Normal::sample(self)
    }
    fn rsample(&self) -> Result<Tensor> {
        Normal::rsample(self)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        Normal::log_prob(self, v)
    }
    fn entropy(&self) -> Result<Tensor> {
        Normal::entropy(self)
    }
    fn mean(&self) -> Result<Tensor> {
        self.loc.add(&self.scale.zeros_like()?)
    }
    fn variance(&self) -> Result<Tensor> {
        self.scale.square()?.add(&self.loc.zeros_like()?)
    }
}

impl Distribution for Laplace {
    fn sample(&self) -> Result<Tensor> {
        Laplace::sample(self)
    }
    fn rsample(&self) -> Result<Tensor> {
        Laplace::rsample(self)
    }
    fn log_prob(&self, v: &Tensor) -> Result<Tensor> {
        Laplace::log_prob(self, v)
    }
    fn entropy(&self) -> Result<Tensor> {
        Laplace::entropy(self)
    }
    fn mean(&self) -> Result<Tensor> {
        self.loc.add(&self.scale.zeros_like()?)
    }
    fn variance(&self) -> Result<Tensor> {
        self.scale.square()?.mul_scalar(2.0)?.add(&self.loc.zeros_like()?)
    }
}


// ---------------------------------------------------------------------------------------------------------------
// Sampling primitives for the Python distributions (drawn from the global, `manual_seed`-able generator)
// ---------------------------------------------------------------------------------------------------------------

/// Standard gamma draws `Gamma(alpha, 1)` for every element of `alpha`, differentiable in `alpha` (implicit
/// reparameterization).
pub fn sample_standard_gamma(alpha: &Tensor) -> Result<Tensor> {
    standard_gamma(&alpha.contiguous()?)
}

/// Poisson draws with the given (non-negative) rates, as floats.
pub fn sample_poisson(rate: &Tensor) -> Result<Tensor> {
    let lam = rate.to_vec::<f64>()?;
    if lam.iter().any(|&l| l.is_nan() || l < 0.0) {
        return Err(Error::Msg("sample_poisson: rates must be non-negative".into()));
    }
    let out: Vec<f64> = rng::with_rng(|r| lam.iter().map(|&l| poisson_draw(r, l)).collect());
    Tensor::from_slice_on(&out, rate.shape(), &rate.device())?.to_dtype(if rate.dtype().is_floating_point() { rate.dtype() } else { crate::dtype::DType::F32 })
}

/// Von Mises draws (Best & Fisher 1979) with the given means and concentrations (one shape), wrapped to [-pi, pi).
pub fn sample_von_mises(loc: &Tensor, concentration: &Tensor) -> Result<Tensor> {
    if loc.shape() != concentration.shape() {
        return Err(Error::Msg("sample_von_mises: loc and concentration must have one shape".into()));
    }
    let (mu, k) = (loc.to_vec::<f64>()?, concentration.to_vec::<f64>()?);
    let out: Vec<f64> = rng::with_rng(|r| {
        mu.iter()
            .zip(&k)
            .map(|(&m, &kappa)| {
                let theta = if kappa < 1e-8 {
                    std::f64::consts::PI * (2.0 * r.next_f64() - 1.0)
                } else {
                    let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
                    let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
                    let rr = (1.0 + rho * rho) / (2.0 * rho);
                    loop {
                        let (u1, u2, u3) = (r.next_f64(), r.next_f64(), r.next_f64());
                        let z = (std::f64::consts::PI * u1).cos();
                        let f = (1.0 + rr * z) / (rr + z);
                        let c = kappa * (rr - f);
                        if c * (2.0 - c) - u2 > 0.0 || (c / u2).ln() + 1.0 - c >= 0.0 {
                            let sign = if u3 > 0.5 { 1.0 } else { -1.0 };
                            break sign * f.clamp(-1.0, 1.0).acos();
                        }
                    }
                };
                let x = theta + m + std::f64::consts::PI;
                x - (x / std::f64::consts::TAU).floor() * std::f64::consts::TAU - std::f64::consts::PI
            })
            .collect()
    });
    Tensor::from_slice_on(&out, loc.shape(), &loc.device())?.to_dtype(if loc.dtype().is_floating_point() { loc.dtype() } else { crate::dtype::DType::F32 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    fn f(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn near(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{got:?} vs {want:?}");
        }
    }
    fn moments(d: &dyn Distribution, n: usize) -> (f64, f64) {
        // draw n samples one by one into a vector (parameters are scalars)
        let s: Vec<f64> = (0..n).map(|_| vals(&d.sample().unwrap())[0]).collect();
        let m = s.iter().sum::<f64>() / n as f64;
        let v = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
        (m, v)
    }

    #[test]
    fn log_prob_values() {
        near(&vals(&Bernoulli::from_probs(&f(&[0.8])).unwrap().log_prob(&f(&[1.0])).unwrap()), &[0.8f64.ln()], 1e-9);
        near(&vals(&Poisson::new(f(&[3.0])).unwrap().log_prob(&f(&[2.0])).unwrap()), &[(9.0f64 / 2.0).ln() - 3.0], 1e-9);
        near(&vals(&Uniform::new(f(&[0.0]), f(&[4.0])).unwrap().log_prob(&f(&[1.0])).unwrap()), &[-(4.0f64.ln())], 1e-9);
        near(&vals(&Exponential::new(f(&[2.0])).unwrap().log_prob(&f(&[0.5])).unwrap()), &[2.0f64.ln() - 1.0], 1e-9);
        // Gamma(2, 3) at 1: 2 ln 3 + ln 1 - 3 - lgamma(2)
        near(&vals(&Gamma::new(f(&[2.0]), f(&[3.0])).unwrap().log_prob(&f(&[1.0])).unwrap()), &[2.0 * 3.0f64.ln() - 3.0], 1e-9);
        // Beta(2, 3) at 0.4: pdf = 12 * 0.4 * 0.6^2
        near(&vals(&Beta::new(f(&[2.0]), f(&[3.0])).unwrap().log_prob(&f(&[0.4])).unwrap()), &[(12.0f64 * 0.4 * 0.36).ln()], 1e-9);
        // Dirichlet(1,1,1): density 2
        let d = Dirichlet::new(Tensor::from_slice(&[1.0, 1.0, 1.0], &[1, 3]).unwrap()).unwrap();
        near(&vals(&d.log_prob(&Tensor::from_slice(&[0.2, 0.3, 0.5], &[1, 3]).unwrap()).unwrap()), &[2.0f64.ln()], 1e-9);
        near(&vals(&Cauchy::new(f(&[0.0]), f(&[1.0])).unwrap().log_prob(&f(&[0.0])).unwrap()), &[-(PI.ln())], 1e-9);
        // StudentT(df=1) is Cauchy
        near(
            &vals(&StudentT::new(f(&[1.0]), f(&[0.0]), f(&[1.0])).unwrap().log_prob(&f(&[0.7])).unwrap()),
            &vals(&Cauchy::new(f(&[0.0]), f(&[1.0])).unwrap().log_prob(&f(&[0.7])).unwrap()),
            1e-9,
        );
        near(&vals(&Gumbel::new(f(&[0.0]), f(&[1.0])).unwrap().log_prob(&f(&[0.0])).unwrap()), &[-1.0], 1e-9);
        near(&vals(&LogNormal::new(f(&[0.0]), f(&[1.0])).unwrap().log_prob(&f(&[1.0])).unwrap()), &[-0.5 * TAU.ln()], 1e-9);
        near(&vals(&HalfNormal::new(f(&[1.0])).unwrap().log_prob(&f(&[0.0])).unwrap()), &[2.0f64.ln() - 0.5 * TAU.ln()], 1e-9);
        near(&vals(&Geometric::new(f(&[0.25])).unwrap().log_prob(&f(&[2.0])).unwrap()), &[2.0 * 0.75f64.ln() + 0.25f64.ln()], 1e-9);
        // Binomial(10, 0.3) at 4
        let choose = 210.0f64;
        near(&vals(&Binomial::new(f(&[10.0]), &f(&[0.3])).unwrap().log_prob(&f(&[4.0])).unwrap()), &[(choose * 0.3f64.powi(4) * 0.7f64.powi(6)).ln()], 1e-9);
        // Chi2(2) is Exponential(1/2)
        near(
            &vals(&Chi2::new(f(&[2.0])).unwrap().log_prob(&f(&[1.5])).unwrap()),
            &vals(&Exponential::new(f(&[0.5])).unwrap().log_prob(&f(&[1.5])).unwrap()),
            1e-9,
        );
    }

    #[test]
    fn entropies_match_numerical_integration() {
        // -integral p ln p on a fine grid
        fn numeric(d: &dyn Distribution, lo: f64, hi: f64) -> f64 {
            let n = 40_000;
            let xs: Vec<f64> = (0..n).map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n as f64).collect();
            let lp = vals(&d.log_prob(&f(&xs)).unwrap());
            lp.iter().map(|l| -l.exp() * l).sum::<f64>() * (hi - lo) / n as f64
        }
        let cases: Vec<(&str, Box<dyn Distribution>, f64, f64)> = vec![
            ("gamma", Box::new(Gamma::new(f(&[2.5]), f(&[1.5])).unwrap()), 1e-6, 30.0),
            ("beta", Box::new(Beta::new(f(&[2.0]), f(&[3.5])).unwrap()), 1e-7, 1.0 - 1e-7),
            ("exp", Box::new(Exponential::new(f(&[1.7])).unwrap()), 1e-9, 30.0),
            ("cauchy", Box::new(Cauchy::new(f(&[0.5]), f(&[1.2])).unwrap()), -3000.0, 3000.0),
            ("student", Box::new(StudentT::new(f(&[4.0]), f(&[0.0]), f(&[1.3])).unwrap()), -400.0, 400.0),
            ("lognormal", Box::new(LogNormal::new(f(&[0.2]), f(&[0.7])).unwrap()), 1e-9, 60.0),
            ("halfnormal", Box::new(HalfNormal::new(f(&[1.4])).unwrap()), 0.0, 20.0),
            ("gumbel", Box::new(Gumbel::new(f(&[0.3]), f(&[0.9])).unwrap()), -15.0, 60.0),
            ("uniform", Box::new(Uniform::new(f(&[-1.0]), f(&[2.0])).unwrap()), -1.0, 1.999_999),
        ];
        for (name, d, lo, hi) in &cases {
            let want = numeric(d.as_ref(), *lo, *hi);
            let got = vals(&d.entropy().unwrap())[0];
            assert!((got - want).abs() < 2e-3 * (1.0 + want.abs()), "{name}: {got} vs {want}");
        }
        // discrete entropies against direct sums
        let h = vals(&Poisson::new(f(&[2.0])).unwrap().entropy().unwrap())[0];
        let direct: f64 = (0..60).map(|k| { let lp = k as f64 * 2.0f64.ln() - 2.0 - libm::lgamma(k as f64 + 1.0); -lp.exp() * lp }).sum();
        assert!((h - direct).abs() < 1e-9);
        let hb = vals(&Bernoulli::from_probs(&f(&[0.3])).unwrap().entropy().unwrap())[0];
        assert!((hb + 0.3 * 0.3f64.ln() + 0.7 * 0.7f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn sample_moments() {
        rng::manual_seed(7);
        let n = 20_000;
        let cases: Vec<(&str, Box<dyn Distribution>)> = vec![
            ("bernoulli", Box::new(Bernoulli::from_probs(&f(&[0.3])).unwrap())),
            ("binomial-small", Box::new(Binomial::new(f(&[12.0]), &f(&[0.35])).unwrap())),
            ("binomial-large", Box::new(Binomial::new(f(&[500.0]), &f(&[0.2])).unwrap())),
            ("poisson-small", Box::new(Poisson::new(f(&[3.5])).unwrap())),
            ("poisson-large", Box::new(Poisson::new(f(&[40.0])).unwrap())),
            ("geometric", Box::new(Geometric::new(f(&[0.3])).unwrap())),
            ("uniform", Box::new(Uniform::new(f(&[-2.0]), f(&[3.0])).unwrap())),
            ("exponential", Box::new(Exponential::new(f(&[2.0])).unwrap())),
            ("gamma-big", Box::new(Gamma::new(f(&[4.0]), f(&[2.0])).unwrap())),
            ("gamma-small", Box::new(Gamma::new(f(&[0.4]), f(&[1.0])).unwrap())),
            ("beta", Box::new(Beta::new(f(&[2.0]), f(&[5.0])).unwrap())),
            ("lognormal", Box::new(LogNormal::new(f(&[0.0]), f(&[0.5])).unwrap())),
            ("halfnormal", Box::new(HalfNormal::new(f(&[2.0])).unwrap())),
            ("gumbel", Box::new(Gumbel::new(f(&[1.0]), f(&[2.0])).unwrap())),
            ("studentt", Box::new(StudentT::new(f(&[7.0]), f(&[1.0]), f(&[2.0])).unwrap())),
            ("chi2", Box::new(Chi2::new(f(&[5.0])).unwrap())),
        ];
        for (name, d) in &cases {
            let (m, v) = moments(d.as_ref(), n);
            let (wm, wv) = (vals(&d.mean().unwrap())[0], vals(&d.variance().unwrap())[0]);
            assert!((m - wm).abs() < 0.05 * (1.0 + wm.abs()) + 4.0 * (wv / n as f64).sqrt(), "{name}: mean {m} vs {wm}");
            assert!((v - wv).abs() < 0.12 * (1.0 + wv), "{name}: var {v} vs {wv}");
        }
    }

    #[test]
    fn dirichlet_and_mvn() {
        rng::manual_seed(3);
        let d = Dirichlet::new(Tensor::from_slice(&[2.0, 3.0, 5.0], &[3]).unwrap()).unwrap();
        let s = d.sample().unwrap();
        assert!((vals(&s).iter().sum::<f64>() - 1.0).abs() < 1e-6);
        near(&vals(&d.mean().unwrap()), &[0.2, 0.3, 0.5], 1e-9);
        // MVN with identity covariance factorizes into independent normals
        let mvn = MultivariateNormal::from_scale_tril(f(&[1.0, -1.0]), Tensor::eye(2, DType::F64, &crate::backend::Device::Cpu).unwrap()).unwrap();
        let x = f(&[0.5, 0.25]);
        let ind = Independent::new(Normal::new(f(&[1.0, -1.0]), f(&[1.0, 1.0])).unwrap(), 1);
        near(&vals(&mvn.log_prob(&x).unwrap()), &vals(&ind.log_prob(&x).unwrap()), 1e-9);
        near(&vals(&mvn.entropy().unwrap()), &vals(&ind.entropy().unwrap()), 1e-9);
        // correlated: against the explicit 2-d formula
        let cov = Tensor::from_slice(&[2.0, 0.6, 0.6, 1.0], &[2, 2]).unwrap();
        let m = MultivariateNormal::from_covariance(f(&[0.0, 0.0]), &cov).unwrap();
        let det = 2.0 * 1.0 - 0.36;
        let (a, b) = (0.7, -0.4);
        let maha = (1.0 * a * a - 2.0 * 0.6 * a * b + 2.0 * b * b) / det;
        near(&vals(&m.log_prob(&f(&[a, b])).unwrap()), &[-0.5 * (2.0 * TAU.ln() + det.ln() + maha)], 1e-9);
        near(&vals(&m.variance().unwrap()), &[2.0, 1.0], 1e-9);
        // sample covariance
        let n = 20_000;
        let big = MultivariateNormal::from_covariance(Tensor::from_slice(&vec![0.0; n * 2], &[n, 2]).unwrap(), &cov).unwrap();
        let sm = vals(&big.sample().unwrap());
        let c01: f64 = (0..n).map(|i| sm[2 * i] * sm[2 * i + 1]).sum::<f64>() / n as f64;
        assert!((c01 - 0.6).abs() < 0.06, "{c01}");
    }

    #[test]
    fn kl_divergences() {
        // KL(p || p) = 0, and positive otherwise; compare against Monte-Carlo-free numeric integration
        let g1 = Gamma::new(f(&[2.0]), f(&[1.0])).unwrap();
        let g2 = Gamma::new(f(&[3.0]), f(&[2.0])).unwrap();
        assert!(vals(&g1.kl_divergence(&g1).unwrap())[0].abs() < 1e-9);
        let n = 80_000;
        let xs: Vec<f64> = (0..n).map(|i| 1e-6 + 40.0 * (i as f64 + 0.5) / n as f64).collect();
        let (l1, l2) = (vals(&g1.log_prob(&f(&xs)).unwrap()), vals(&g2.log_prob(&f(&xs)).unwrap()));
        let num: f64 = (0..n).map(|i| l1[i].exp() * (l1[i] - l2[i])).sum::<f64>() * 40.0 / n as f64;
        assert!((vals(&g1.kl_divergence(&g2).unwrap())[0] - num).abs() < 1e-3, "{} vs {num}", vals(&g1.kl_divergence(&g2).unwrap())[0]);
        let b1 = Beta::new(f(&[2.0]), f(&[3.0])).unwrap();
        let b2 = Beta::new(f(&[1.5]), f(&[4.0])).unwrap();
        let xs: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect();
        let (l1, l2) = (vals(&b1.log_prob(&f(&xs)).unwrap()), vals(&b2.log_prob(&f(&xs)).unwrap()));
        let num: f64 = (0..n).map(|i| l1[i].exp() * (l1[i] - l2[i])).sum::<f64>() / n as f64;
        assert!((vals(&b1.kl_divergence(&b2).unwrap())[0] - num).abs() < 1e-3);
        near(&vals(&Exponential::new(f(&[1.0])).unwrap().kl_divergence(&Exponential::new(f(&[2.0])).unwrap()).unwrap()), &[0.5f64.ln() + 2.0 - 1.0], 1e-9);
        near(&vals(&Bernoulli::from_probs(&f(&[0.3])).unwrap().kl_divergence(&Bernoulli::from_probs(&f(&[0.6])).unwrap()).unwrap()), &[0.3 * (0.3f64 / 0.6).ln() + 0.7 * (0.7f64 / 0.4).ln()], 1e-6);
        near(&vals(&Poisson::new(f(&[2.0])).unwrap().kl_divergence(&Poisson::new(f(&[3.0])).unwrap()).unwrap()), &[2.0 * (2.0f64 / 3.0).ln() - 2.0 + 3.0], 1e-9);
        // identical MVNs
        let a = MultivariateNormal::from_covariance(f(&[0.5, 0.1]), &Tensor::from_slice(&[2.0, 0.6, 0.6, 1.0], &[2, 2]).unwrap()).unwrap();
        assert!(vals(&a.kl_divergence(&a).unwrap())[0].abs() < 1e-9);
        let d1 = Dirichlet::new(Tensor::from_slice(&[2.0, 3.0], &[2]).unwrap()).unwrap();
        assert!(vals(&d1.kl_divergence(&d1).unwrap())[0].abs() < 1e-9);
        assert!(vals(&d1.kl_divergence(&Dirichlet::new(Tensor::from_slice(&[1.0, 5.0], &[2]).unwrap()).unwrap()).unwrap())[0] > 0.0);
    }

    #[test]
    fn rsample_gradients() {
        rng::manual_seed(11);
        // d E[Gamma(a, b)] / da = 1 / b, estimated by averaging pathwise gradients
        let a = Tensor::from_slice(&vec![2.5f64; 20_000], &[20_000]).unwrap().requires_grad_(true);
        let b = Tensor::from_slice(&vec![2.0f64; 20_000], &[20_000]).unwrap();
        Gamma::new(a.clone(), b).unwrap().rsample().unwrap().mean().unwrap().backward().unwrap();
        let g = vals(&a.grad().unwrap()).iter().sum::<f64>();
        assert!((g - 0.5).abs() < 0.03, "{g}");
        // d E[Exp(r)] / dr = -1 / r^2
        let r = Tensor::from_slice(&vec![2.0f64; 20_000], &[20_000]).unwrap().requires_grad_(true);
        Exponential::new(r.clone()).unwrap().rsample().unwrap().mean().unwrap().backward().unwrap();
        let g = vals(&r.grad().unwrap()).iter().sum::<f64>();
        assert!((g + 0.25).abs() < 0.02, "{g}");
        // MVN rsample carries gradients to the loc and the factor
        let loc = f(&[0.0, 1.0]).requires_grad_(true);
        let l = Tensor::eye(2, DType::F64, &crate::backend::Device::Cpu).unwrap().requires_grad_(true);
        MultivariateNormal::from_scale_tril(loc.clone(), l.clone()).unwrap().rsample().unwrap().sum().unwrap().backward().unwrap();
        near(&vals(&loc.grad().unwrap()), &[1.0, 1.0], 1e-9);
        assert!(l.grad().is_some());
        // log_prob gradient of Gamma w.r.t. the value
        let x = f(&[1.3]).requires_grad_(true);
        Gamma::new(f(&[2.0]), f(&[3.0])).unwrap().log_prob(&x).unwrap().sum().unwrap().backward().unwrap();
        near(&vals(&x.grad().unwrap()), &[1.0 / 1.3 - 3.0], 1e-9);
    }
}
