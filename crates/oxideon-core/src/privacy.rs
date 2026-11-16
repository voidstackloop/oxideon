//! Per-sample gradients and DP-SGD aggregation (clip each sample's gradient to a norm bound, sum, add Gaussian noise).

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Gradient of `loss_fn(x_i, y_i)` for every sample `i` with respect to every tensor in `params`.
/// Returns one tensor per parameter with shape `[batch, ...param.shape]`. Samples are processed one at a time
/// (`x.narrow(0, i, 1)`), and `params` keep their `.grad` cleared afterwards.
pub fn per_sample_grads(params: &[Tensor], loss_fn: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>, x: &Tensor, y: &Tensor) -> Result<Vec<Tensor>> {
    let b = x.shape()[0];
    if y.shape()[0] != b {
        return Err(Error::Msg("per_sample_grads: x and y must share the batch size".into()));
    }
    let mut rows: Vec<Vec<Tensor>> = vec![Vec::with_capacity(b); params.len()];
    for i in 0..b {
        params.iter().for_each(Tensor::zero_grad);
        loss_fn(&x.narrow(0, i, 1)?, &y.narrow(0, i, 1)?)?.backward()?;
        for (k, p) in params.iter().enumerate() {
            rows[k].push(match p.grad() {
                Some(g) => g.detach(),
                None => p.zeros_like()?,
            });
        }
    }
    params.iter().for_each(Tensor::zero_grad);
    rows.iter().map(|r| Tensor::stack(r, 0)).collect()
}

/// Global L2 norm of each sample's gradient across all parameters (`[batch]` values).
pub fn per_sample_norms(per_sample: &[Tensor]) -> Result<Vec<f64>> {
    let b = per_sample.first().map_or(0, |t| t.shape()[0]);
    let mut sq = vec![0.0; b];
    for t in per_sample {
        let flat = t.reshape(&[b, t.numel() / b.max(1)])?.square()?.sum_axis(1, false)?.to_vec::<f64>()?;
        sq.iter_mut().zip(flat).for_each(|(s, v)| *s += v);
    }
    Ok(sq.into_iter().map(f64::sqrt).collect())
}

/// DP-SGD aggregation: scales every sample's gradient so its global norm is at most `clip`, sums over the batch, adds
/// `N(0, (noise_multiplier * clip)^2)` noise and divides by the batch size. Returns one gradient per parameter.
pub fn dp_aggregate(per_sample: &[Tensor], clip: f64, noise_multiplier: f64) -> Result<Vec<Tensor>> {
    if clip <= 0.0 {
        return Err(Error::Msg("dp_aggregate: clip must be positive".into()));
    }
    let b = per_sample.first().map_or(0, |t| t.shape()[0]);
    let norms = per_sample_norms(per_sample)?;
    let mut out = Vec::with_capacity(per_sample.len());
    for t in per_sample {
        let scale: Vec<f64> = norms.iter().map(|&n| (clip / (n + 1e-12)).min(1.0)).collect();
        let mut shape = vec![1; t.ndim()];
        shape[0] = b;
        let s = Tensor::from_slice_on(&scale, &[b], &t.device())?.to_dtype(t.dtype())?.reshape(&shape)?;
        let mut g = t.mul(&s)?.sum_axis(0, false)?;
        if noise_multiplier > 0.0 {
            g = g.add(&Tensor::randn_with(g.shape(), g.dtype(), &g.device())?.mul_scalar(noise_multiplier * clip)?)?;
        }
        out.push(g.div_scalar(b.max(1) as f64)?);
    }
    Ok(out)
}

/// Computes the DP-SGD gradients for one batch and stores them in `params[k].grad`, ready for any optimizer.
pub fn dp_backward(params: &[Tensor], loss_fn: &dyn Fn(&Tensor, &Tensor) -> Result<Tensor>, x: &Tensor, y: &Tensor, clip: f64, noise_multiplier: f64) -> Result<()> {
    let grads = dp_aggregate(&per_sample_grads(params, loss_fn, x, y)?, clip, noise_multiplier)?;
    params.iter().zip(grads).for_each(|(p, g)| p.set_grad(Some(g)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, Module};

    fn setup() -> (Linear, Tensor, Tensor) {
        let lin = Linear::new(3, 2).unwrap();
        (lin, Tensor::randn(&[5, 3]).unwrap(), Tensor::randn(&[5, 2]).unwrap())
    }

    #[test]
    fn per_sample_grads_average_to_the_batch_gradient() {
        let (lin, x, y) = setup();
        let loss = |a: &Tensor, b: &Tensor| lin.forward(a)?.sub(b)?.square()?.mean();
        let ps = per_sample_grads(&lin.parameters(), &loss, &x, &y).unwrap();
        assert_eq!(ps[0].shape(), [5, 2, 3]);
        assert_eq!(ps[1].shape(), [5, 2]);
        // mean over samples equals the gradient of the mean loss (every sample has equal weight)
        lin.parameters().iter().for_each(Tensor::zero_grad);
        lin.forward(&x).unwrap().sub(&y).unwrap().square().unwrap().sum_axis(1, false).unwrap().div_scalar(2.0).unwrap().mean().unwrap().backward().unwrap();
        for (p, g) in lin.parameters().iter().zip(&ps) {
            let want = p.grad().unwrap().to_vec::<f32>().unwrap();
            let got = g.mean_axis(0, false).unwrap().to_vec::<f32>().unwrap();
            assert!(want.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-5), "{want:?} vs {got:?}");
        }
    }

    #[test]
    fn dp_clipping_bounds_every_sample_and_noise_is_optional() {
        let (lin, x, y) = setup();
        let loss = |a: &Tensor, b: &Tensor| lin.forward(a)?.sub(b)?.square()?.mean();
        let ps = per_sample_grads(&lin.parameters(), &loss, &x, &y).unwrap();
        let norms = per_sample_norms(&ps).unwrap();
        let clip = norms.iter().cloned().fold(f64::INFINITY, f64::min) * 0.5; // clips every sample
        let g = dp_aggregate(&ps, clip, 0.0).unwrap();
        // the mean of five vectors each of norm <= clip has norm <= clip
        let total: f64 = g.iter().map(|t| t.square().unwrap().sum().unwrap().item::<f64>().unwrap()).sum::<f64>().sqrt();
        assert!(total <= clip * 1.0001, "{total} vs {clip}");
        // no clipping needed: result is the plain mean
        let loose = dp_aggregate(&ps, 1e9, 0.0).unwrap();
        let mean = ps[1].mean_axis(0, false).unwrap().to_vec::<f32>().unwrap();
        assert!(loose[1].to_vec::<f32>().unwrap().iter().zip(&mean).all(|(a, b)| (a - b).abs() < 1e-5));
        // noise changes the result; dp_backward fills .grad
        let noisy = dp_aggregate(&ps, clip, 1.0).unwrap();
        assert_ne!(noisy[1].to_vec::<f32>().unwrap(), g[1].to_vec::<f32>().unwrap());
        dp_backward(&lin.parameters(), &loss, &x, &y, clip, 0.0).unwrap();
        assert!(lin.parameters().iter().all(|p| p.grad().is_some()));
        assert!(dp_aggregate(&ps, 0.0, 0.0).is_err());
    }
}

/// Rényi-DP accountant for the (Poisson-)subsampled Gaussian mechanism with integer orders
/// (Mironov et al. 2019): `eps` after `steps` steps at sampling rate `q` and noise multiplier `sigma`.
pub struct RdpAccountant {
    pub q: f64,
    pub sigma: f64,
    pub steps: usize,
}

fn log_add(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    if lo == f64::NEG_INFINITY { hi } else { hi + (lo - hi).exp().ln_1p() }
}

fn ln_binom(n: u64, k: u64) -> f64 {
    (1..=k).map(|i| ((n - k + i) as f64).ln() - (i as f64).ln()).sum()
}

impl RdpAccountant {
    pub fn new(q: f64, sigma: f64) -> Self {
        Self { q, sigma, steps: 0 }
    }

    pub fn step(&mut self, n: usize) {
        self.steps += n;
    }

    /// RDP of one step at integer order `alpha >= 2`.
    fn rdp_one_step(&self, alpha: u64) -> f64 {
        if self.q == 0.0 {
            return 0.0;
        }
        if self.q >= 1.0 {
            return alpha as f64 / (2.0 * self.sigma * self.sigma);
        }
        let mut acc = f64::NEG_INFINITY;
        for k in 0..=alpha {
            let term = ln_binom(alpha, k) + k as f64 * self.q.ln() + (alpha - k) as f64 * (1.0 - self.q).ln() + (k * k.saturating_sub(1)) as f64 / (2.0 * self.sigma * self.sigma);
            acc = log_add(acc, term);
        }
        acc / (alpha as f64 - 1.0)
    }

    /// The tightest `eps` for the given `delta` over orders 2..=256.
    pub fn epsilon(&self, delta: f64) -> f64 {
        (2..=256u64)
            .map(|a| self.steps as f64 * self.rdp_one_step(a) + (1.0 / delta).ln() / (a as f64 - 1.0))
            .fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod accountant_tests {
    use super::*;

    #[test]
    fn epsilon_behaves_like_dp() {
        let mut a = RdpAccountant::new(0.01, 1.1);
        a.step(1000);
        let e1 = a.epsilon(1e-5);
        a.step(1000);
        let e2 = a.epsilon(1e-5);
        assert!(e1 > 0.0 && e2 > e1, "more steps cost more privacy: {e1} {e2}");
        let mut quiet = RdpAccountant::new(0.01, 2.0);
        quiet.step(1000);
        assert!(quiet.epsilon(1e-5) < e1, "more noise costs less privacy");
        let mut full = RdpAccountant::new(1.0, 1.0);
        full.step(1);
        // without subsampling: eps(alpha) = alpha / (2 sigma^2); tight value near sqrt(2 ln(1/delta)) / sigma
        assert!(full.epsilon(1e-5) > 4.0 && full.epsilon(1e-5) < 6.0, "{}", full.epsilon(1e-5));
        assert!(RdpAccountant::new(0.0, 1.0).epsilon(1e-5) < 0.1);
    }
}
