//! Gaussian-process regression ([`GaussianProcess`]) with RBF / Matern-5/2 kernels, log marginal likelihood, a simple
//! length-scale search, and Bayesian optimization of a black-box function with expected improvement
//! ([`bayes_opt`]).

use crate::classical::Matrix;
use crate::error::{Error, Result};
use crate::rng::Rng;

/// Stationary kernel with unit signal variance (scaled by `signal_var` in the model).
#[derive(Debug, Clone, Copy)]
pub enum GpKernel {
    Rbf,
    Matern52,
}

fn kern(k: GpKernel, a: &[f64], b: &[f64], ls: f64) -> f64 {
    let r2 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / (ls * ls);
    match k {
        GpKernel::Rbf => (-0.5 * r2).exp(),
        GpKernel::Matern52 => {
            let r = (5.0 * r2).sqrt();
            (1.0 + r + 5.0 * r2 / 3.0) * (-r).exp()
        }
    }
}

fn cholesky(a: &Matrix) -> Result<Matrix> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d <= 0.0 {
                    return Err(Error::Msg("GaussianProcess: kernel matrix is not positive definite (increase noise)".into()));
                }
                l[i][j] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    Ok(l)
}

fn forward(l: &Matrix, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; b.len()];
    for i in 0..b.len() {
        y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    y
}

fn backward(l: &Matrix, y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    x
}

/// Zero-mean GP on standardized targets.
pub struct GaussianProcess {
    x: Matrix,
    l: Matrix,
    alpha: Vec<f64>,
    y_mean: f64,
    y_std: f64,
    kernel: GpKernel,
    pub length_scale: f64,
    pub signal_var: f64,
    pub noise: f64,
    log_marginal: f64,
}

impl GaussianProcess {
    pub fn fit(x: &Matrix, y: &[f64], kernel: GpKernel, length_scale: f64, signal_var: f64, noise: f64) -> Result<Self> {
        let n = x.len();
        if n == 0 || y.len() != n || x.iter().any(|r| r.len() != x[0].len()) || length_scale <= 0.0 || signal_var <= 0.0 || noise < 0.0 {
            return Err(Error::Msg("GaussianProcess: need matching non-empty data and positive hyperparameters".into()));
        }
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let y_std = (y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64).sqrt().max(1e-12);
        let ys: Vec<f64> = y.iter().map(|v| (v - y_mean) / y_std).collect();
        let mut k: Matrix = (0..n).map(|i| (0..n).map(|j| signal_var * kern(kernel, &x[i], &x[j], length_scale)).collect()).collect();
        (0..n).for_each(|i| k[i][i] += noise + 1e-10);
        let l = cholesky(&k)?;
        let alpha = backward(&l, &forward(&l, &ys));
        let log_marginal = -0.5 * ys.iter().zip(&alpha).map(|(a, b)| a * b).sum::<f64>() - (0..n).map(|i| l[i][i].ln()).sum::<f64>() - 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
        Ok(Self { x: x.clone(), l, alpha, y_mean, y_std, kernel, length_scale, signal_var, noise, log_marginal })
    }

    /// Fits with the length scale chosen from `candidates` by maximizing the log marginal likelihood.
    pub fn fit_best_length_scale(x: &Matrix, y: &[f64], kernel: GpKernel, candidates: &[f64], signal_var: f64, noise: f64) -> Result<Self> {
        let mut best: Option<Self> = None;
        for &ls in candidates {
            if let Ok(gp) = Self::fit(x, y, kernel, ls, signal_var, noise) {
                if best.as_ref().is_none_or(|b| gp.log_marginal > b.log_marginal) {
                    best = Some(gp);
                }
            }
        }
        best.ok_or_else(|| Error::Msg("GaussianProcess: no candidate length scale produced a valid model".into()))
    }

    pub fn log_marginal_likelihood(&self) -> f64 {
        self.log_marginal
    }

    /// Posterior `(mean, variance)` at a query point, in the original target units.
    pub fn predict(&self, q: &[f64]) -> (f64, f64) {
        let ks: Vec<f64> = self.x.iter().map(|xi| self.signal_var * kern(self.kernel, xi, q, self.length_scale)).collect();
        let mean = ks.iter().zip(&self.alpha).map(|(a, b)| a * b).sum::<f64>();
        let v = forward(&self.l, &ks);
        let var = (self.signal_var - v.iter().map(|t| t * t).sum::<f64>()).max(0.0);
        (mean * self.y_std + self.y_mean, var * self.y_std * self.y_std)
    }
}

fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// Full-precision normal CDF (an erf approximation with 1e-7 error makes expected improvement negative in the tails).
use crate::stats_tests::norm_cdf;

/// Expected improvement over `best` when minimizing, for a posterior `(mean, variance)`.
pub fn expected_improvement(mean: f64, var: f64, best: f64) -> f64 {
    let sd = var.sqrt();
    if sd < 1e-12 {
        return (best - mean).max(0.0);
    }
    let z = (best - mean) / sd;
    (best - mean) * norm_cdf(z) + sd * norm_pdf(z)
}

/// Result of [`bayes_opt`].
#[derive(Debug, Clone)]
pub struct BoResult {
    pub x: Vec<f64>,
    pub value: f64,
    pub history: Vec<(Vec<f64>, f64)>,
}

/// Minimizes `f` over the box `bounds` with `n_init` random points followed by `n_iter` expected-improvement steps
/// (candidates: `n_candidates` random points per step).
pub fn bayes_opt<F>(f: F, bounds: &[(f64, f64)], n_init: usize, n_iter: usize, n_candidates: usize, seed: u64) -> Result<BoResult>
where
    F: Fn(&[f64]) -> f64,
{
    if bounds.is_empty() || bounds.iter().any(|&(lo, hi)| !matches!(lo.partial_cmp(&hi), Some(std::cmp::Ordering::Less))) || n_init < 2 {
        return Err(Error::Msg("bayes_opt: need non-degenerate bounds and n_init >= 2".into()));
    }
    let mut rng = Rng::new(seed);
    let draw = |rng: &mut Rng| -> Vec<f64> { bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect() };
    let unit = |p: &[f64]| -> Vec<f64> { p.iter().zip(bounds).map(|(v, &(lo, hi))| (v - lo) / (hi - lo)).collect() };
    let mut history: Vec<(Vec<f64>, f64)> = (0..n_init).map(|_| { let p = draw(&mut rng); let v = f(&p); (p, v) }).collect();
    for _ in 0..n_iter {
        let xs: Matrix = history.iter().map(|(p, _)| unit(p)).collect();
        let ys: Vec<f64> = history.iter().map(|(_, v)| *v).collect();
        let gp = GaussianProcess::fit_best_length_scale(&xs, &ys, GpKernel::Matern52, &[0.05, 0.1, 0.2, 0.4, 0.8], 1.0, 1e-6)?;
        let best = ys.iter().cloned().fold(f64::INFINITY, f64::min);
        let next = (0..n_candidates.max(1))
            .map(|_| draw(&mut rng))
            .max_by(|a, b| {
                let (ma, va) = gp.predict(&unit(a));
                let (mb, vb) = gp.predict(&unit(b));
                expected_improvement(ma, va, best).total_cmp(&expected_improvement(mb, vb, best))
            })
            .unwrap();
        let v = f(&next);
        history.push((next, v));
    }
    let (x, value) = history.iter().min_by(|a, b| a.1.total_cmp(&b.1)).cloned().unwrap();
    Ok(BoResult { x, value, history })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_and_reports_uncertainty() {
        let x: Matrix = (0..8).map(|i| vec![i as f64 * 0.5]).collect();
        let y: Vec<f64> = x.iter().map(|r| r[0].sin()).collect();
        let gp = GaussianProcess::fit(&x, &y, GpKernel::Rbf, 1.0, 1.0, 1e-8).unwrap();
        // exact at training points with ~zero variance
        let (m, v) = gp.predict(&[1.5]);
        assert!((m - 1.5f64.sin()).abs() < 1e-4 && v < 1e-6, "{m} {v}");
        // accurate between points, uncertain far away
        let (m, v_near) = gp.predict(&[1.75]);
        assert!((m - 1.75f64.sin()).abs() < 0.01);
        let (_, v_far) = gp.predict(&[20.0]);
        assert!(v_far > 100.0 * v_near.max(1e-9), "{v_near} {v_far}");
        assert!(GaussianProcess::fit(&x, &y[..3], GpKernel::Rbf, 1.0, 1.0, 0.0).is_err());
        assert!(GaussianProcess::fit(&x, &y, GpKernel::Rbf, -1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn marginal_likelihood_picks_a_sensible_length_scale() {
        let x: Matrix = (0..30).map(|i| vec![i as f64 / 29.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| (6.0 * r[0]).sin() + 0.01 * ((r[0] * 997.0).sin())).collect();
        let gp = GaussianProcess::fit_best_length_scale(&x, &y, GpKernel::Matern52, &[0.01, 0.05, 0.2, 0.5, 2.0, 10.0], 1.0, 1e-4).unwrap();
        assert!(gp.length_scale >= 0.05 && gp.length_scale <= 2.0, "{}", gp.length_scale);
        let tiny = GaussianProcess::fit(&x, &y, GpKernel::Matern52, 0.01, 1.0, 1e-4).unwrap();
        assert!(gp.log_marginal_likelihood() > tiny.log_marginal_likelihood());
    }

    #[test]
    fn expected_improvement_properties() {
        assert!(expected_improvement(0.0, 1.0, 0.0) > 0.0); // uncertainty alone has value
        assert!(expected_improvement(-1.0, 0.0, 0.0) == 1.0); // certain improvement of 1
        assert!(expected_improvement(5.0, 0.0, 0.0) == 0.0);
        assert!(expected_improvement(0.0, 4.0, 0.0) > expected_improvement(0.0, 1.0, 0.0));
        assert!((norm_cdf(1.96) - 0.975).abs() < 1e-3 && (norm_cdf(0.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn bayes_opt_finds_the_minimum_with_few_evaluations() {
        // 2-D Branin-like bowl with a unique minimum at (1.0, -2.0)
        let f = |p: &[f64]| (p[0] - 1.0).powi(2) + 0.5 * (p[1] + 2.0).powi(2) + 0.3 * (3.0 * p[0]).sin().powi(2);
        let r = bayes_opt(f, &[(-5.0, 5.0), (-5.0, 5.0)], 5, 25, 400, 1).unwrap();
        assert_eq!(r.history.len(), 30);
        assert!(r.value < 0.5, "{r:?}");
        // random search with the same 30 evaluations does not do better than this on average; check it is not trivial
        let mut rng = Rng::new(11);
        let rand_best = (0..30).map(|_| f(&[rng.next_f64() * 10.0 - 5.0, rng.next_f64() * 10.0 - 5.0])).fold(f64::INFINITY, f64::min);
        assert!(r.value <= rand_best + 0.5, "{} vs random {rand_best}", r.value);
        assert!(bayes_opt(f, &[(1.0, 1.0)], 5, 5, 10, 0).is_err());
    }
}
