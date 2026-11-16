//! Bijective transforms and [`TransformedDistribution`] (normalizing-flow building blocks), plus further families:
//! `Weibull`, `Pareto`, `Logistic`, `Kumaraswamy`, `FisherSnedecor`, `NegativeBinomial` and `Multinomial`, all behind
//! the [`Distribution`] trait of [`crate::distributions_ext`].

use crate::distributions_ext::{Chi2, Distribution, Gamma, Poisson};
use crate::error::{Error, Result};
use crate::rng;
use crate::tensor::Tensor;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------------------------------------------

/// An invertible, differentiable map between random variables.
pub trait Bijector {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn inverse(&self, y: &Tensor) -> Result<Tensor>;
    /// `log |dy/dx|` evaluated at `x` (with `y = forward(x)`), elementwise.
    fn log_abs_det_jacobian(&self, x: &Tensor, y: &Tensor) -> Result<Tensor>;
}

/// `y = exp(x)`.
pub struct ExpTransform;

impl Bijector for ExpTransform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.exp()
    }
    fn inverse(&self, y: &Tensor) -> Result<Tensor> {
        y.ln()
    }
    fn log_abs_det_jacobian(&self, x: &Tensor, _y: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

/// `y = loc + scale * x`.
pub struct AffineTransform {
    pub loc: Tensor,
    pub scale: Tensor,
}

impl Bijector for AffineTransform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.loc.add(&self.scale.mul(x)?)
    }
    fn inverse(&self, y: &Tensor) -> Result<Tensor> {
        y.sub(&self.loc)?.div(&self.scale)
    }
    fn log_abs_det_jacobian(&self, x: &Tensor, _y: &Tensor) -> Result<Tensor> {
        self.scale.abs()?.ln()?.add(&x.zeros_like()?)
    }
}

/// `y = sigmoid(x)`.
pub struct SigmoidTransform;

impl Bijector for SigmoidTransform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.sigmoid()
    }
    fn inverse(&self, y: &Tensor) -> Result<Tensor> {
        y.clamp(1e-12, 1.0 - 1e-12)?.logit(None)
    }
    fn log_abs_det_jacobian(&self, x: &Tensor, _y: &Tensor) -> Result<Tensor> {
        // log sigmoid(x) + log sigmoid(-x)
        x.neg()?.softplus()?.neg()?.sub(&x.softplus()?)
    }
}

/// `y = tanh(x)`.
pub struct TanhTransform;

impl Bijector for TanhTransform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.tanh()
    }
    fn inverse(&self, y: &Tensor) -> Result<Tensor> {
        let y = y.clamp(-1.0 + 1e-12, 1.0 - 1e-12)?;
        y.add_scalar(1.0)?.div(&y.neg()?.add_scalar(1.0)?)?.ln()?.mul_scalar(0.5)
    }
    fn log_abs_det_jacobian(&self, x: &Tensor, _y: &Tensor) -> Result<Tensor> {
        // 2 (ln 2 - x - softplus(-2x)), a numerically stable log(1 - tanh^2)
        x.neg()?.sub(&x.mul_scalar(-2.0)?.softplus()?)?.add_scalar(std::f64::consts::LN_2)?.mul_scalar(2.0)
    }
}

/// `y = softplus(x)`.
pub struct SoftplusTransform;

impl Bijector for SoftplusTransform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.softplus()
    }
    fn inverse(&self, y: &Tensor) -> Result<Tensor> {
        // log(expm1(y)) computed stably
        y.add(&y.neg()?.exp()?.neg()?.log1p()?)
    }
    fn log_abs_det_jacobian(&self, x: &Tensor, _y: &Tensor) -> Result<Tensor> {
        x.neg()?.softplus()?.neg()
    }
}

/// `y = x ^ exponent` on the positive reals.
pub struct PowerTransform {
    pub exponent: f64,
}

impl Bijector for PowerTransform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.powf(self.exponent)
    }
    fn inverse(&self, y: &Tensor) -> Result<Tensor> {
        y.powf(1.0 / self.exponent)
    }
    fn log_abs_det_jacobian(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        y.div(x)?.mul_scalar(self.exponent)?.abs()?.ln()
    }
}

/// Applies the transforms in order.
pub struct ComposeTransform(pub Vec<Box<dyn Bijector>>);

impl Bijector for ComposeTransform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.0.iter().try_fold(x.clone(), |acc, t| t.forward(&acc))
    }
    fn inverse(&self, y: &Tensor) -> Result<Tensor> {
        self.0.iter().rev().try_fold(y.clone(), |acc, t| t.inverse(&acc))
    }
    fn log_abs_det_jacobian(&self, x: &Tensor, _y: &Tensor) -> Result<Tensor> {
        let mut cur = x.clone();
        let mut total: Option<Tensor> = None;
        for t in &self.0 {
            let next = t.forward(&cur)?;
            let l = t.log_abs_det_jacobian(&cur, &next)?;
            total = Some(match total {
                Some(acc) => acc.add(&l)?,
                None => l,
            });
            cur = next;
        }
        total.ok_or_else(|| Error::Msg("ComposeTransform: no transforms".into()))
    }
}

/// `y = T(x)` for `x ~ base`: samples are pushed through the transform, densities follow the change of variables.
pub struct TransformedDistribution<D: Distribution> {
    pub base: D,
    pub transform: Box<dyn Bijector>,
}

impl<D: Distribution> TransformedDistribution<D> {
    pub fn new(base: D, transform: Box<dyn Bijector>) -> Self {
        Self { base, transform }
    }
}

impl<D: Distribution> Distribution for TransformedDistribution<D> {
    fn sample(&self) -> Result<Tensor> {
        self.transform.forward(&self.base.sample()?)
    }
    fn rsample(&self) -> Result<Tensor> {
        self.transform.forward(&self.base.rsample()?)
    }
    fn log_prob(&self, y: &Tensor) -> Result<Tensor> {
        let x = self.transform.inverse(y)?;
        self.base.log_prob(&x)?.sub(&self.transform.log_abs_det_jacobian(&x, y)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        Err(Error::Msg("TransformedDistribution has no closed-form entropy".into()))
    }
    fn mean(&self) -> Result<Tensor> {
        Err(Error::Msg("TransformedDistribution has no closed-form mean".into()))
    }
    fn variance(&self) -> Result<Tensor> {
        Err(Error::Msg("TransformedDistribution has no closed-form variance".into()))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// More families
// ---------------------------------------------------------------------------------------------------------------

fn positive(op: &str, t: &Tensor) -> Result<()> {
    if t.to_vec::<f64>()?.iter().all(|&v| v.is_finite() && v > 0.0) {
        Ok(())
    } else {
        Err(Error::Msg(format!("{op}: parameters must be finite and positive")))
    }
}

fn uniform_like(shape: &[usize], like: &Tensor) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let v: Vec<f64> = rng::with_rng(|r| {
        (0..n)
            .map(|_| loop {
                let u = r.next_f64();
                if u > 0.0 {
                    break u;
                }
            })
            .collect()
    });
    Tensor::from_slice_on(&v, shape, &like.device())?.to_dtype(like.dtype())
}

fn bshape(ts: &[&Tensor]) -> Result<Vec<usize>> {
    let mut s = ts[0].shape().to_vec();
    for t in &ts[1..] {
        s = crate::layout::broadcast_shapes("distribution", &s, t.shape())?;
    }
    Ok(s)
}

/// Weibull distribution with `scale` and `concentration`.
#[derive(Clone)]
pub struct Weibull {
    pub scale: Tensor,
    pub concentration: Tensor,
}

impl Weibull {
    pub fn new(scale: Tensor, concentration: Tensor) -> Result<Self> {
        positive("Weibull::new", &scale)?;
        positive("Weibull::new", &concentration)?;
        Ok(Self { scale, concentration })
    }
}

impl Distribution for Weibull {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape(&[&self.scale, &self.concentration])?;
        let e = uniform_like(&shape, &self.scale)?.ln()?.neg()?;
        self.scale.mul(&e.pow(&self.concentration.recip()?)?)
    }
    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let (l, k) = (&self.scale, &self.concentration);
        let z = x.div(l)?;
        k.ln()?.sub(&l.ln()?)?.add(&k.sub_scalar(1.0)?.mul(&z.ln()?)?)?.sub(&z.pow(k)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let (l, k) = (&self.scale, &self.concentration);
        k.recip()?.neg()?.add_scalar(1.0)?.mul_scalar(0.577_215_664_901_532_9)?.add(&l.div(k)?.ln()?)?.add_scalar(1.0)
    }
    fn mean(&self) -> Result<Tensor> {
        self.scale.mul(&self.concentration.recip()?.add_scalar(1.0)?.lgamma()?.exp()?)
    }
    fn variance(&self) -> Result<Tensor> {
        let k = &self.concentration;
        let g1 = k.recip()?.add_scalar(1.0)?.lgamma()?.exp()?;
        let g2 = k.recip()?.mul_scalar(2.0)?.add_scalar(1.0)?.lgamma()?.exp()?;
        self.scale.square()?.mul(&g2.sub(&g1.square()?)?)
    }
}

/// Pareto (type I) distribution with `scale` (minimum) and shape `alpha`.
#[derive(Clone)]
pub struct Pareto {
    pub scale: Tensor,
    pub alpha: Tensor,
}

impl Pareto {
    pub fn new(scale: Tensor, alpha: Tensor) -> Result<Self> {
        positive("Pareto::new", &scale)?;
        positive("Pareto::new", &alpha)?;
        Ok(Self { scale, alpha })
    }
}

impl Distribution for Pareto {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape(&[&self.scale, &self.alpha])?;
        let u = uniform_like(&shape, &self.scale)?;
        self.scale.div(&u.pow(&self.alpha.recip()?)?)
    }
    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let (m, a) = (&self.scale, &self.alpha);
        let lp = a.ln()?.add(&a.mul(&m.ln()?)?)?.sub(&a.add_scalar(1.0)?.mul(&x.ln()?)?)?;
        Tensor::where_cond(&x.ge(m)?, &lp, &lp.full_like(f64::NEG_INFINITY)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        let (m, a) = (&self.scale, &self.alpha);
        m.div(a)?.ln()?.add(&a.recip()?)?.add_scalar(1.0)
    }
    fn mean(&self) -> Result<Tensor> {
        let (m, a) = (&self.scale, &self.alpha);
        let v = a.mul(m)?.div(&a.sub_scalar(1.0)?)?;
        Tensor::where_cond(&a.gt(&a.ones_like()?)?, &v, &v.full_like(f64::INFINITY)?)
    }
    fn variance(&self) -> Result<Tensor> {
        let (m, a) = (&self.scale, &self.alpha);
        let v = m.square()?.mul(a)?.div(&a.sub_scalar(1.0)?.square()?.mul(&a.sub_scalar(2.0)?)?)?;
        Tensor::where_cond(&a.gt(&a.full_like(2.0)?)?, &v, &v.full_like(f64::INFINITY)?)
    }
}

/// Logistic distribution.
#[derive(Clone)]
pub struct Logistic {
    pub loc: Tensor,
    pub scale: Tensor,
}

impl Logistic {
    pub fn new(loc: Tensor, scale: Tensor) -> Result<Self> {
        positive("Logistic::new", &scale)?;
        Ok(Self { loc, scale })
    }
}

impl Distribution for Logistic {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape(&[&self.loc, &self.scale])?;
        let u = uniform_like(&shape, &self.loc)?.clamp(1e-12, 1.0 - 1e-12)?;
        let noise = u.ln()?.sub(&u.neg()?.log1p()?)?;
        self.loc.add(&self.scale.mul(&noise)?)
    }
    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let z = x.sub(&self.loc)?.div(&self.scale)?;
        z.neg()?.sub(&self.scale.ln()?)?.sub(&z.neg()?.softplus()?.mul_scalar(2.0)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        self.scale.ln()?.add_scalar(2.0)?.add(&self.loc.zeros_like()?)
    }
    fn mean(&self) -> Result<Tensor> {
        self.loc.add(&self.scale.zeros_like()?)
    }
    fn variance(&self) -> Result<Tensor> {
        self.scale.square()?.mul_scalar(PI * PI / 3.0)
    }
}

/// Kumaraswamy distribution on `(0, 1)`.
#[derive(Clone)]
pub struct Kumaraswamy {
    pub a: Tensor,
    pub b: Tensor,
}

impl Kumaraswamy {
    pub fn new(a: Tensor, b: Tensor) -> Result<Self> {
        positive("Kumaraswamy::new", &a)?;
        positive("Kumaraswamy::new", &b)?;
        Ok(Self { a, b })
    }
}

impl Distribution for Kumaraswamy {
    fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape(&[&self.a, &self.b])?;
        let u = uniform_like(&shape, &self.a)?;
        let inner = u.neg()?.add_scalar(1.0)?.pow(&self.b.recip()?)?.neg()?.add_scalar(1.0)?;
        inner.pow(&self.a.recip()?)
    }
    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let (a, b) = (&self.a, &self.b);
        a.ln()?
            .add(&b.ln()?)?
            .add(&a.sub_scalar(1.0)?.mul(&x.ln()?)?)?
            .add(&b.sub_scalar(1.0)?.mul(&x.pow(a)?.neg()?.log1p()?)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        Err(Error::Msg("Kumaraswamy entropy is not implemented".into()))
    }
    fn mean(&self) -> Result<Tensor> {
        // b * B(1 + 1/a, b)
        let (a, b) = (&self.a, &self.b);
        let t = a.recip()?.add_scalar(1.0)?;
        b.ln()?.add(&t.lgamma()?)?.add(&b.lgamma()?)?.sub(&t.add(b)?.lgamma()?)?.exp()
    }
    fn variance(&self) -> Result<Tensor> {
        let (a, b) = (&self.a, &self.b);
        let m2 = {
            let t = a.recip()?.mul_scalar(2.0)?.add_scalar(1.0)?;
            b.ln()?.add(&t.lgamma()?)?.add(&b.lgamma()?)?.sub(&t.add(b)?.lgamma()?)?.exp()?
        };
        m2.sub(&self.mean()?.square()?)
    }
}

/// F distribution with `df1` and `df2` degrees of freedom.
#[derive(Clone)]
pub struct FisherSnedecor {
    pub df1: Tensor,
    pub df2: Tensor,
}

impl FisherSnedecor {
    pub fn new(df1: Tensor, df2: Tensor) -> Result<Self> {
        positive("FisherSnedecor::new", &df1)?;
        positive("FisherSnedecor::new", &df2)?;
        Ok(Self { df1, df2 })
    }
}

impl Distribution for FisherSnedecor {
    fn sample(&self) -> Result<Tensor> {
        Ok(self.rsample()?.detach())
    }
    fn rsample(&self) -> Result<Tensor> {
        let shape = bshape(&[&self.df1, &self.df2])?;
        let (d1, d2) = (self.df1.broadcast_to(&shape)?.contiguous()?, self.df2.broadcast_to(&shape)?.contiguous()?);
        let x1 = Chi2::new(d1.clone())?.rsample()?.div(&d1)?;
        let x2 = Chi2::new(d2.clone())?.rsample()?.div(&d2)?;
        x1.div(&x2)
    }
    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let (d1, d2) = (&self.df1, &self.df2);
        let half = |t: &Tensor| t.mul_scalar(0.5);
        let lbeta = half(d1)?.lgamma()?.add(&half(d2)?.lgamma()?)?.sub(&half(&d1.add(d2)?)?.lgamma()?)?;
        let d1x = d1.mul(x)?;
        half(&d1.mul(&d1x.ln()?)?.add(&d2.mul(&d2.ln()?)?)?.sub(&d1.add(d2)?.mul(&d1x.add(d2)?.ln()?)?)?)?
            .sub(&x.ln()?)?
            .sub(&lbeta)
    }
    fn entropy(&self) -> Result<Tensor> {
        Err(Error::Msg("FisherSnedecor entropy is not implemented".into()))
    }
    fn mean(&self) -> Result<Tensor> {
        let d2 = &self.df2;
        let v = d2.div(&d2.sub_scalar(2.0)?)?;
        Tensor::where_cond(&d2.gt(&d2.full_like(2.0)?)?, &v, &v.full_like(f64::NAN)?)
    }
    fn variance(&self) -> Result<Tensor> {
        let (d1, d2) = (&self.df1, &self.df2);
        let v = d2.square()?.mul(&d1.add(&d2.sub_scalar(2.0)?)?)?.mul_scalar(2.0)?.div(&d1.mul(&d2.sub_scalar(2.0)?.square()?)?.mul(&d2.sub_scalar(4.0)?)?)?;
        Tensor::where_cond(&d2.gt(&d2.full_like(4.0)?)?, &v, &v.full_like(f64::NAN)?)
    }
}

/// Number of successes before `total_count` failures, each trial succeeding with probability `probs`.
#[derive(Clone)]
pub struct NegativeBinomial {
    pub total_count: Tensor,
    pub probs: Tensor,
}

impl NegativeBinomial {
    pub fn new(total_count: Tensor, probs: Tensor) -> Result<Self> {
        positive("NegativeBinomial::new", &total_count)?;
        if !probs.to_vec::<f64>()?.iter().all(|&p| p > 0.0 && p < 1.0) {
            return Err(Error::Msg("NegativeBinomial::new: probs must lie in (0, 1)".into()));
        }
        Ok(Self { total_count, probs })
    }
}

impl Distribution for NegativeBinomial {
    fn sample(&self) -> Result<Tensor> {
        let shape = bshape(&[&self.total_count, &self.probs])?;
        let (r, p) = (self.total_count.broadcast_to(&shape)?.contiguous()?, self.probs.broadcast_to(&shape)?.contiguous()?);
        // Gamma-Poisson mixture: rate ~ Gamma(r, (1 - p) / p)
        let rate = Gamma::new(r, p.neg()?.add_scalar(1.0)?.div(&p)?)?.sample()?;
        Poisson::new(rate.clamp(1e-300, f64::INFINITY)?)?.sample()
    }
    fn log_prob(&self, k: &Tensor) -> Result<Tensor> {
        let (r, p) = (&self.total_count, &self.probs);
        k.add(r)?
            .lgamma()?
            .sub(&k.add_scalar(1.0)?.lgamma()?)?
            .sub(&r.lgamma()?)?
            .add(&k.mul(&p.ln()?)?)?
            .add(&r.mul(&p.neg()?.log1p()?)?)
    }
    fn entropy(&self) -> Result<Tensor> {
        Err(Error::Msg("NegativeBinomial entropy is not implemented".into()))
    }
    fn mean(&self) -> Result<Tensor> {
        let p = &self.probs;
        self.total_count.mul(p)?.div(&p.neg()?.add_scalar(1.0)?)
    }
    fn variance(&self) -> Result<Tensor> {
        let p = &self.probs;
        self.total_count.mul(p)?.div(&p.neg()?.add_scalar(1.0)?.square()?)
    }
}

/// Counts of `total_count` draws over the categories of `probs` (last dimension).
#[derive(Clone)]
pub struct Multinomial {
    pub total_count: usize,
    pub probs: Tensor,
}

impl Multinomial {
    pub fn new(total_count: usize, probs: Tensor) -> Result<Self> {
        if probs.ndim() == 0 {
            return Err(Error::Msg("Multinomial::new: probs needs a category dimension".into()));
        }
        let norm = probs.sum_axis(-1, true)?;
        Ok(Self { total_count, probs: probs.div(&norm)? })
    }
}

impl Distribution for Multinomial {
    fn sample(&self) -> Result<Tensor> {
        let k = self.probs.shape()[self.probs.ndim() - 1];
        let p = self.probs.to_vec::<f64>()?;
        let rows = p.len() / k;
        let mut out = Vec::with_capacity(p.len());
        rng::with_rng(|r| {
            for row in 0..rows {
                let (mut left, mut mass) = (self.total_count as i64, 1.0f64);
                for j in 0..k {
                    let pj = p[row * k + j];
                    let c = if j == k - 1 || mass <= 0.0 {
                        left
                    } else {
                        let q = (pj / mass).clamp(0.0, 1.0);
                        (0..left).filter(|_| r.next_f64() < q).count() as i64
                    };
                    out.push(c as f64);
                    left -= c;
                    mass -= pj;
                }
            }
        });
        Tensor::from_slice_on(&out, self.probs.shape(), &self.probs.device())?.to_dtype(self.probs.dtype())
    }
    fn log_prob(&self, counts: &Tensor) -> Result<Tensor> {
        let n = self.total_count as f64;
        let coeff = Tensor::from_slice(&[n + 1.0], &[1])?.to_dtype(counts.dtype())?.lgamma()?.reshape(&[])?;
        let lfact = counts.add_scalar(1.0)?.lgamma()?.sum_axis(-1, false)?;
        let logits = counts.xlogy(&self.probs)?.sum_axis(-1, false)?;
        logits.sub(&lfact)?.add(&coeff)
    }
    fn entropy(&self) -> Result<Tensor> {
        Err(Error::Msg("Multinomial entropy is not implemented".into()))
    }
    fn mean(&self) -> Result<Tensor> {
        self.probs.mul_scalar(self.total_count as f64)
    }
    fn variance(&self) -> Result<Tensor> {
        self.probs.mul(&self.probs.neg()?.add_scalar(1.0)?)?.mul_scalar(self.total_count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::Normal;
    use crate::distributions_ext::LogNormal;

    fn f(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn near(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn transforms_round_trip_and_have_correct_jacobians() {
        let ts: Vec<(&str, Box<dyn Bijector>, Vec<f64>)> = vec![
            ("exp", Box::new(ExpTransform), vec![-1.0, 0.3, 1.7]),
            ("affine", Box::new(AffineTransform { loc: f(&[1.0]), scale: f(&[-2.5]) }), vec![-1.0, 0.3]),
            ("sigmoid", Box::new(SigmoidTransform), vec![-2.0, 0.3, 3.0]),
            ("tanh", Box::new(TanhTransform), vec![-1.5, 0.3, 0.9]),
            ("softplus", Box::new(SoftplusTransform), vec![-2.0, 0.3, 3.0]),
            ("power", Box::new(PowerTransform { exponent: 3.0 }), vec![0.5, 1.5]),
            ("compose", Box::new(ComposeTransform(vec![Box::new(AffineTransform { loc: f(&[0.0]), scale: f(&[0.5]) }), Box::new(SigmoidTransform)])), vec![-1.0, 2.0]),
        ];
        for (name, t, xs) in &ts {
            let x = f(xs);
            let y = t.forward(&x).unwrap();
            near(&vals(&t.inverse(&y).unwrap()), xs, 1e-6);
            // log |dy/dx| against a central difference
            let h = 1e-6;
            let num: Vec<f64> = xs
                .iter()
                .map(|&v| {
                    let (a, b) = (vals(&t.forward(&f(&[v + h])).unwrap())[0], vals(&t.forward(&f(&[v - h])).unwrap())[0]);
                    ((a - b) / (2.0 * h)).abs().ln()
                })
                .collect();
            let got = t.log_abs_det_jacobian(&x, &y).unwrap();
            near(&vals(&got), &num, 1e-5);
            let _ = name;
        }
    }

    #[test]
    fn transformed_distribution_matches_known_families() {
        // exp of a normal is log-normal
        let td = TransformedDistribution::new(Normal::new(f(&[0.2]), f(&[0.7])).unwrap(), Box::new(ExpTransform));
        let ln = LogNormal::new(f(&[0.2]), f(&[0.7])).unwrap();
        let y = f(&[0.5, 1.3, 2.9]);
        near(&vals(&td.log_prob(&y).unwrap()), &vals(&ln.log_prob(&y).unwrap()), 1e-9);
        // tanh-squashed normal integrates to one
        let sq = TransformedDistribution::new(Normal::new(f(&[0.0]), f(&[1.0])).unwrap(), Box::new(TanhTransform));
        let n = 20_000;
        let ys: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64).collect();
        let total: f64 = vals(&sq.log_prob(&f(&ys)).unwrap()).iter().map(|l| l.exp()).sum::<f64>() * 2.0 / n as f64;
        assert!((total - 1.0).abs() < 1e-3, "{total}");
        assert!(td.sample().unwrap().numel() == 1 && td.entropy().is_err());
    }

    fn integrates_to_one(d: &dyn Distribution, lo: f64, hi: f64) {
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n as f64).collect();
        let total: f64 = vals(&d.log_prob(&f(&xs)).unwrap()).iter().map(|l| l.exp()).sum::<f64>() * (hi - lo) / n as f64;
        assert!((total - 1.0).abs() < 2e-3, "integral {total}");
    }

    #[test]
    fn densities_normalize_and_match_moments() {
        integrates_to_one(&Weibull::new(f(&[1.5]), f(&[2.2])).unwrap(), 1e-9, 12.0);
        integrates_to_one(&Pareto::new(f(&[1.0]), f(&[3.0])).unwrap(), 1.0, 4000.0);
        integrates_to_one(&Logistic::new(f(&[0.3]), f(&[0.8])).unwrap(), -40.0, 40.0);
        integrates_to_one(&Kumaraswamy::new(f(&[2.0]), f(&[3.0])).unwrap(), 1e-9, 1.0 - 1e-9);
        integrates_to_one(&FisherSnedecor::new(f(&[5.0]), f(&[8.0])).unwrap(), 1e-9, 400.0);
        rng::manual_seed(17);
        let n = 30_000;
        let cases: Vec<(&str, Box<dyn Distribution>)> = vec![
            ("weibull", Box::new(Weibull::new(f(&[1.5]), f(&[2.2])).unwrap())),
            ("logistic", Box::new(Logistic::new(f(&[0.3]), f(&[0.8])).unwrap())),
            ("kumaraswamy", Box::new(Kumaraswamy::new(f(&[2.0]), f(&[3.0])).unwrap())),
            ("fisher", Box::new(FisherSnedecor::new(f(&[6.0]), f(&[12.0])).unwrap())),
            ("negbin", Box::new(NegativeBinomial::new(f(&[4.0]), f(&[0.4])).unwrap())),
            ("pareto", Box::new(Pareto::new(f(&[1.0]), f(&[5.0])).unwrap())),
        ];
        for (name, d) in &cases {
            let s: Vec<f64> = (0..n).map(|_| vals(&d.sample().unwrap())[0]).collect();
            let m = s.iter().sum::<f64>() / n as f64;
            let var = s.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n as f64;
            let (wm, wv) = (vals(&d.mean().unwrap())[0], vals(&d.variance().unwrap())[0]);
            assert!((m - wm).abs() < 0.05 * (1.0 + wm.abs()), "{name}: mean {m} vs {wm}");
            assert!((var - wv).abs() < 0.25 * (1.0 + wv), "{name}: var {var} vs {wv}");
        }
    }

    #[test]
    fn discrete_families() {
        // negative binomial pmf sums to one
        let nb = NegativeBinomial::new(f(&[3.0]), f(&[0.35])).unwrap();
        let ks: Vec<f64> = (0..200).map(f64::from).collect();
        let total: f64 = vals(&nb.log_prob(&f(&ks)).unwrap()).iter().map(|l| l.exp()).sum();
        assert!((total - 1.0).abs() < 1e-6, "{total}");
        // multinomial: pmf of (1, 2, 1) for n = 4 with probs (.2, .5, .3)
        let m = Multinomial::new(4, Tensor::from_slice(&[0.2, 0.5, 0.3], &[3]).unwrap()).unwrap();
        let want = (12.0f64 * 0.2 * 0.25 * 0.3).ln(); // 4! / (1! 2! 1!) = 12
        near(&vals(&m.log_prob(&Tensor::from_slice(&[1.0, 2.0, 1.0], &[3]).unwrap()).unwrap()), &[want], 1e-9);
        rng::manual_seed(8);
        let mut acc = [0.0; 3];
        let n = 4000;
        for _ in 0..n {
            let s = vals(&m.sample().unwrap());
            assert_eq!(s.iter().sum::<f64>(), 4.0);
            for (a, v) in acc.iter_mut().zip(&s) {
                *a += v / n as f64;
            }
        }
        near(&acc, &[0.8, 2.0, 1.2], 0.05);
        near(&vals(&m.mean().unwrap()), &[0.8, 2.0, 1.2], 1e-9);
    }
}
