//! Probabilistic distributions (`Normal`, `Categorical`, `Laplace`), analytical KL divergence
//! & Shannon entropy, pathwise reparameterized sampling (`rsample`), and Straight-Through
//! Gumbel-Softmax (`gumbel_softmax`).

use std::f64::consts::{E, PI, TAU};

use crate::error::{Error, Result};
use crate::layout::broadcast_shapes;
use crate::rng;
use crate::tensor::Tensor;

/// Univariate Normal (Gaussian) distribution $\mathcal{N}(\mu, \sigma^2)$ with broadcastable
/// `loc` ($\mu$) and `scale` ($\sigma > 0$) tensors.
#[derive(Clone)]
pub struct Normal {
    pub loc: Tensor,
    pub scale: Tensor,
}

fn check_positive_scale(op: &'static str, scale: &Tensor) -> Result<()> {
    let vals = scale.to_vec::<f64>()?;
    if !vals.iter().all(|&v| v.is_finite() && v > 0.0) {
        return Err(Error::Msg(format!(
            "{op}: scale must be finite and positive everywhere"
        )));
    }
    Ok(())
}

impl Normal {
    /// Builds `N(loc, scale)`; `scale` must be finite and positive everywhere
    /// (a zero/negative scale would silently yield NaN log-densities).
    /// `loc`/`scale` broadcast like torch (only genuinely incompatible shapes
    /// fail here). Fields stay `pub`, so mutating `scale` afterwards bypasses
    /// this check.
    pub fn new(loc: Tensor, scale: Tensor) -> Result<Self> {
        broadcast_shapes("Normal::new", loc.shape(), scale.shape())?;
        check_positive_scale("Normal::new", &scale)?;
        Ok(Self { loc, scale })
    }

    /// Pathwise reparameterized sample $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$,
    /// allowing exact gradients with respect to both `loc` and `scale`.
    pub fn rsample(&self) -> Result<Tensor> {
        let eps = Tensor::randn(self.loc.shape())?
            .to_dtype(self.loc.dtype())?
            .to_device(&self.loc.device())?;
        self.loc.add(&self.scale.mul(&eps)?)
    }

    /// Draws a sample (identical to [`Normal::rsample`]: the Normal is fully
    /// reparameterized, so no detach is needed — same as torch).
    pub fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }

    /// Elementwise log-density $\log p(x) = -\frac{(x - \mu)^2}{2 \sigma^2} - \ln \sigma - \frac{1}{2}\ln(2\pi)$.
    pub fn log_prob(&self, value: &Tensor) -> Result<Tensor> {
        let z = value.sub(&self.loc)?.div(&self.scale)?;
        let quad = z.square()?.mul_scalar(-0.5)?;
        let log_norm = self.scale.ln()?.add_scalar(0.5 * TAU.ln())?;
        quad.sub(&log_norm)
    }

    /// Elementwise differential entropy $\mathcal{H}(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2}\ln(2\pi e) + \ln \sigma$,
    /// broadcast to the `loc`/`scale` union batch shape like torch.
    pub fn entropy(&self) -> Result<Tensor> {
        let term = self.scale.ln()?.add_scalar(0.5 * (TAU * E).ln())?;
        term.add(&self.loc.zeros_like()?.to_dtype(self.scale.dtype())?)
    }

    /// Elementwise analytical KL divergence $\operatorname{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \,\|\, \mathcal{N}(\mu_2, \sigma_2^2))$.
    pub fn kl_divergence(&self, other: &Normal) -> Result<Tensor> {
        let var_ratio = self.scale.div(&other.scale)?.square()?;
        let t1 = self.loc.sub(&other.loc)?.div(&other.scale)?.square()?;
        let log_ratio = other.scale.div(&self.scale)?.ln()?;
        log_ratio
            .add(&var_ratio.add(&t1)?.mul_scalar(0.5)?)?
            .sub_scalar(0.5)
    }
}

/// Categorical distribution parameterized by unnormalized `logits` along the last dimension.
#[derive(Clone)]
pub struct Categorical {
    pub logits: Tensor,
}

impl Categorical {
    pub fn from_logits(logits: Tensor) -> Self {
        Self { logits }
    }

    pub fn probs(&self) -> Result<Tensor> {
        self.logits.softmax(-1)
    }

    pub fn log_probs(&self) -> Result<Tensor> {
        self.logits.log_softmax(-1)
    }

    /// Draws discrete category indices (`I64`) along the last axis: a scalar for
    /// 1D logits, one index per row otherwise (torch `sample` shapes).
    pub fn sample(&self) -> Result<Tensor> {
        let p = self.probs()?;
        let s = p.multinomial(1, true)?;
        if s.shape().last() == Some(&1) {
            let new_shape = s.shape()[..s.ndim() - 1].to_vec();
            s.reshape(&new_shape)
        } else {
            Ok(s)
        }
    }

    /// Evaluates $\log P(X = \text{indices})$ for `indices` (`I64`), with the
    /// output reshaped back to the indices shape (torch broadcasts batch shapes;
    /// mixed-rank batch/index pairs that `gather` cannot express stay an error).
    pub fn log_prob(&self, indices: &Tensor) -> Result<Tensor> {
        let lp = self.log_probs()?;
        let out_shape = indices.shape().to_vec();
        let idx_flat = if lp.ndim() == 1 {
            indices.reshape(&[indices.numel()])?
        } else {
            indices.unsqueeze(-1)?
        };
        let gathered = lp.gather(-1, &idx_flat)?;
        gathered.reshape(&out_shape)
    }

    /// Shannon entropy $\mathcal{H}(p) = -\sum_k p_k \log p_k$ along the last dimension.
    pub fn entropy(&self) -> Result<Tensor> {
        let p = self.probs()?;
        let lp = self.log_probs()?;
        p.mul(&lp)?.sum_axis(-1, false)?.neg()
    }

    /// Differentiable Gumbel-Softmax (Concrete) distribution sample (Jang et al. 2017):
    /// $$y = \operatorname{softmax}\!\left(\frac{\text{logits} + g}{\tau}\right), \quad g_k = -\ln(-\ln(u_k)), \; u_k \sim \mathcal{U}(0, 1)$$
    /// When `hard = true`, returns a discrete one-hot vector in the forward pass while
    /// preserving the continuous Gumbel-Softmax gradient in the backward pass (Straight-Through Estimator)!
    /// Gumbel-Softmax sample at `temperature` (must be finite and positive;
    /// clamping `tau <= 0` would silently return a near-one-hot regardless of
    /// the requested temperature).
    pub fn gumbel_softmax(&self, temperature: f64, hard: bool) -> Result<Tensor> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Error::Msg(format!(
                "gumbel_softmax requires a finite positive temperature, got {temperature}"
            )));
        }
        let n = self.logits.numel();
        let gumbels: Vec<f64> = rng::with_rng(|r| {
            (0..n)
                .map(|_| {
                    let u = (r.next_f32() as f64).clamp(1e-7, 1.0 - 1e-7);
                    -(-u.ln()).ln()
                })
                .collect()
        });
        let g_tensor = Tensor::from_slice_on(&gumbels, self.logits.shape(), &self.logits.device())?
            .to_dtype(self.logits.dtype())?;
        let y_soft = self
            .logits
            .add(&g_tensor)?
            .div_scalar(temperature)?
            .softmax(-1)?;

        if hard {
            let num_classes = *self.logits.shape().last().unwrap_or(&1);
            let idx = y_soft.argmax_axis(-1, false)?;
            let y_hard = idx
                .one_hot(num_classes, y_soft.dtype())?
                .to_device(&y_soft.device())?;
            // Straight-Through Estimator: (y_hard - y_soft.detach()) + y_soft
            y_hard.sub(&y_soft.detach())?.add(&y_soft)
        } else {
            Ok(y_soft)
        }
    }
}

/// Laplace (double-exponential) distribution $\operatorname{Laplace}(\mu, b)$.
#[derive(Clone)]
pub struct Laplace {
    pub loc: Tensor,
    pub scale: Tensor,
}

impl Laplace {
    /// Builds `Laplace(loc, scale)`; `scale` must be finite and positive
    /// everywhere. `loc`/`scale` broadcast like torch (only genuinely
    /// incompatible shapes fail here). Fields stay `pub`, so mutating `scale`
    /// afterwards bypasses this check.
    pub fn new(loc: Tensor, scale: Tensor) -> Result<Self> {
        broadcast_shapes("Laplace::new", loc.shape(), scale.shape())?;
        check_positive_scale("Laplace::new", &scale)?;
        Ok(Self { loc, scale })
    }

    /// Reparameterized inverse-CDF sample $z = \mu - b \operatorname{sign}(u) \ln(1 - 2|u|)$ for $u \sim \mathcal{U}(-0.5, 0.5)$.
    pub fn rsample(&self) -> Result<Tensor> {
        let n = self.loc.numel();
        let noise: Vec<f64> = rng::with_rng(|r| {
            (0..n)
                .map(|_| {
                    let u = ((r.next_f32() as f64) - 0.5).clamp(-0.499_999, 0.499_999);
                    -u.signum() * (1.0 - 2.0 * u.abs()).ln()
                })
                .collect()
        });
        let eps = Tensor::from_slice_on(&noise, self.loc.shape(), &self.loc.device())?
            .to_dtype(self.loc.dtype())?;
        self.loc.add(&self.scale.mul(&eps)?)
    }

    /// Draws a sample (identical to [`Laplace::rsample`]: fully reparameterized).
    pub fn sample(&self) -> Result<Tensor> {
        self.rsample()
    }

    /// Elementwise log-density $\log p(x) = -\frac{|x - \mu|}{b} - \ln(2 b)$.
    pub fn log_prob(&self, value: &Tensor) -> Result<Tensor> {
        let z = value.sub(&self.loc)?.abs()?.div(&self.scale)?.neg()?;
        let log_norm = self.scale.mul_scalar(2.0)?.ln()?;
        z.sub(&log_norm)
    }

    /// Elementwise differential entropy $\mathcal{H}(\operatorname{Laplace}(\mu, b)) = 1 + \ln(2 b)$,
    /// broadcast to the `loc`/`scale` union batch shape like torch.
    pub fn entropy(&self) -> Result<Tensor> {
        let _ = PI;
        let term = self.scale.mul_scalar(2.0)?.ln()?.add_scalar(1.0)?;
        term.add(&self.loc.zeros_like()?.to_dtype(self.scale.dtype())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::LN_2;

    fn t(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape).unwrap()
    }

    fn item(t: &Tensor) -> f64 {
        t.to_vec::<f64>().unwrap()[0]
    }

    /// Closed-form values cross-checked against `torch.distributions`: N(0,1)
    /// log_prob(0) = -ln(2pi)/2, entropy = ln(2pi*e)/2, KL(N(1,4)||N(0,1)).
    #[test]
    fn normal_values_match_torch() {
        let d = Normal::new(t(&[0.0], &[1]), t(&[1.0], &[1])).unwrap();
        assert!((item(&d.log_prob(&t(&[0.0], &[1])).unwrap()) - -0.9189385332).abs() < 1e-9);
        assert!((item(&d.entropy().unwrap()) - 1.4189385332).abs() < 1e-9);
        let p = Normal::new(t(&[1.0], &[1]), t(&[2.0], &[1])).unwrap();
        let q = Normal::new(t(&[0.0], &[1]), t(&[1.0], &[1])).unwrap();
        assert!((item(&p.kl_divergence(&q).unwrap()) - 1.3068528194).abs() < 1e-9);
        assert!(item(&p.kl_divergence(&p).unwrap()).abs() < 1e-12);
    }

    /// Laplace(0,1): log_prob(0) = -ln 2, entropy = 1 + ln 2 (torch-verified).
    #[test]
    fn laplace_values_match_torch() {
        let d = Laplace::new(t(&[0.0], &[1]), t(&[1.0], &[1])).unwrap();
        assert!((item(&d.log_prob(&t(&[0.0], &[1])).unwrap()) - -LN_2).abs() < 1e-9);
        assert!((item(&d.entropy().unwrap()) - (1.0 + LN_2)).abs() < 1e-9);
    }

    /// Uniform-binary entropy = ln 2; log_prob(2) of logits [1,2,3] (torch-verified).
    #[test]
    fn categorical_values_match_torch() {
        let c = Categorical::from_logits(t(&[0.0, 0.0], &[2]));
        assert!((item(&c.entropy().unwrap()) - LN_2).abs() < 1e-9);
        let c3 = Categorical::from_logits(t(&[1.0, 2.0, 3.0], &[3]));
        let idx = Tensor::from_slice(&[2i64], &[1]).unwrap();
        // f64 truth 3 - ln(e + e^2 + e^3) (torch float32 prints -0.40760588…).
        assert!((item(&c3.log_prob(&idx).unwrap()) - -0.4076059644).abs() < 1e-9);
    }

    /// Like torch, 1D logits sample a scalar and batched logits sample per-row.
    #[test]
    fn categorical_sample_shape_matches_torch() {
        let c1 = Categorical::from_logits(t(&[1.0, 2.0], &[2]));
        assert_eq!(c1.sample().unwrap().shape(), &[] as &[usize]);
        let c2 = Categorical::from_logits(t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let s = c2.sample().unwrap();
        assert_eq!(s.shape(), &[2]);
        for v in s.to_vec::<i64>().unwrap() {
            assert!((0..2).contains(&v));
        }
    }

    /// Torch broadcasts loc/scale (the struct doc already promised
    /// "broadcastable" tensors, but `new` rejected mismatched shapes):
    /// loc [3] + scalar scale acts with batch shape [3] everywhere, while
    /// genuinely incompatible shapes still fail fast at construction.
    #[test]
    fn normal_broadcasts_loc_and_scale_like_torch() {
        let d = Normal::new(t(&[0.0, 1.0, 2.0], &[3]), t(&[2.0], &[])).unwrap();
        assert_eq!(d.entropy().unwrap().shape(), &[3]);
        assert_eq!(
            d.log_prob(&t(&[0.0, 0.0, 0.0], &[3])).unwrap().shape(),
            &[3]
        );
        assert_eq!(d.rsample().unwrap().shape(), &[3]);
        assert_eq!(d.sample().unwrap().shape(), &[3]);
        let want = 0.5 * (TAU * E).ln() + 2.0f64.ln();
        for v in d.entropy().unwrap().to_vec::<f64>().unwrap() {
            assert!((v - want).abs() < 1e-12, "entropy {v} vs {want}");
        }
        assert!(Normal::new(t(&[0.0, 0.0, 0.0], &[3]), t(&[1.0, 1.0], &[2])).is_err());
    }

    /// Same broadcast contract for Laplace (entry shape [3] + scalar scale).
    #[test]
    fn laplace_broadcasts_loc_and_scale_like_torch() {
        let d = Laplace::new(t(&[0.0, 1.0, 2.0], &[3]), t(&[2.0], &[])).unwrap();
        assert_eq!(d.entropy().unwrap().shape(), &[3]);
        assert_eq!(
            d.log_prob(&t(&[0.0, 0.0, 0.0], &[3])).unwrap().shape(),
            &[3]
        );
        assert_eq!(d.rsample().unwrap().shape(), &[3]);
        assert_eq!(d.sample().unwrap().shape(), &[3]);
        let want = 1.0 + (2.0 * 2.0f64).ln();
        for v in d.entropy().unwrap().to_vec::<f64>().unwrap() {
            assert!((v - want).abs() < 1e-12, "entropy {v} vs {want}");
        }
        assert!(Laplace::new(t(&[0.0, 0.0, 0.0], &[3]), t(&[1.0, 1.0], &[2])).is_err());
    }

    /// `rsample` gradients: d sum(z)/d loc = 1 and d sum(z)/d scale = eps,
    /// with eps recovered exactly from the same forward pass.
    #[test]
    fn normal_rsample_grad_flows_to_loc_and_scale() {
        let loc = t(&[0.5, -1.0], &[2]).requires_grad_(true);
        let scale = t(&[1.2, 0.8], &[2]).requires_grad_(true);
        let d = Normal::new(loc.clone(), scale.clone()).unwrap();
        let z = d.rsample().unwrap();
        z.sum().unwrap().backward().unwrap();
        assert_eq!(loc.grad().unwrap().to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
        let eps = z
            .sub(&loc)
            .unwrap()
            .div(&scale)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        let gs = scale.grad().unwrap().to_vec::<f64>().unwrap();
        for (g, e) in gs.iter().zip(eps.iter()) {
            assert!((g - e).abs() < 1e-12, "grad {g} vs eps {e}");
        }
    }
}
