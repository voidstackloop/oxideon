//! Probabilistic inference on plain `f64` data: [`metropolis_hastings`], Hamiltonian Monte Carlo ([`hmc`]),
//! [`effective_sample_size`] and [`gelman_rubin`] diagnostics, a linear [`KalmanFilter`] (with RTS smoothing), a
//! discrete [`Hmm`] (forward-backward, Viterbi, Baum-Welch), and a bootstrap [`particle_filter`].

use crate::error::{Error, Result};
use crate::rng::Rng;

fn gauss(rng: &mut Rng) -> f64 {
    let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Samples and acceptance rate of an MCMC run.
#[derive(Debug, Clone)]
pub struct Chain {
    pub samples: Vec<Vec<f64>>,
    pub accept_rate: f64,
}

/// Random-walk Metropolis-Hastings with Gaussian proposals of scale `step` (after `burn_in` discarded draws).
pub fn metropolis_hastings<F: Fn(&[f64]) -> f64>(log_prob: F, x0: &[f64], step: f64, n: usize, burn_in: usize, seed: u64) -> Result<Chain> {
    if x0.is_empty() || step <= 0.0 || n == 0 || !log_prob(x0).is_finite() {
        return Err(Error::Msg("metropolis_hastings: need a start with finite log-probability, step > 0 and n >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let (mut x, mut lp) = (x0.to_vec(), log_prob(x0));
    let (mut out, mut accepted) = (Vec::with_capacity(n), 0usize);
    for i in 0..burn_in + n {
        let prop: Vec<f64> = x.iter().map(|v| v + step * gauss(&mut rng)).collect();
        let lq = log_prob(&prop);
        if lq.is_finite() && rng.next_f64().max(1e-300).ln() < lq - lp {
            x = prop;
            lp = lq;
            if i >= burn_in {
                accepted += 1;
            }
        }
        if i >= burn_in {
            out.push(x.clone());
        }
    }
    Ok(Chain { samples: out, accept_rate: accepted as f64 / n as f64 })
}

/// Hamiltonian Monte Carlo with leapfrog integration; `grad` is the gradient of `log_prob`.
#[allow(clippy::too_many_arguments)]
pub fn hmc<F, G>(log_prob: F, grad: G, x0: &[f64], eps: f64, leapfrog_steps: usize, n: usize, burn_in: usize, seed: u64) -> Result<Chain>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    if x0.is_empty() || eps <= 0.0 || leapfrog_steps == 0 || n == 0 || !log_prob(x0).is_finite() {
        return Err(Error::Msg("hmc: need a valid start, eps > 0, leapfrog_steps >= 1 and n >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let mut x = x0.to_vec();
    let (mut out, mut accepted) = (Vec::with_capacity(n), 0usize);
    for i in 0..burn_in + n {
        let p0: Vec<f64> = (0..x.len()).map(|_| gauss(&mut rng)).collect();
        let (mut q, mut p) = (x.clone(), p0.clone());
        let mut g = grad(&q);
        p.iter_mut().zip(&g).for_each(|(pi, gi)| *pi += 0.5 * eps * gi);
        for s in 0..leapfrog_steps {
            q.iter_mut().zip(&p).for_each(|(qi, pi)| *qi += eps * pi);
            g = grad(&q);
            let k = if s + 1 == leapfrog_steps { 0.5 } else { 1.0 };
            p.iter_mut().zip(&g).for_each(|(pi, gi)| *pi += k * eps * gi);
        }
        let (h0, h1) = (-log_prob(&x) + 0.5 * p0.iter().map(|v| v * v).sum::<f64>(), -log_prob(&q) + 0.5 * p.iter().map(|v| v * v).sum::<f64>());
        if h1.is_finite() && rng.next_f64().max(1e-300).ln() < h0 - h1 {
            x = q;
            if i >= burn_in {
                accepted += 1;
            }
        }
        if i >= burn_in {
            out.push(x.clone());
        }
    }
    Ok(Chain { samples: out, accept_rate: accepted as f64 / n as f64 })
}

/// Effective sample size of one scalar series from its autocorrelation (Geyer's initial positive sequence).
pub fn effective_sample_size(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return n as f64;
    }
    let m = x.iter().sum::<f64>() / n as f64;
    let c0: f64 = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64;
    if c0 == 0.0 {
        return n as f64;
    }
    let rho = |k: usize| (k..n).map(|t| (x[t] - m) * (x[t - k] - m)).sum::<f64>() / (n as f64 * c0);
    let mut sum = 0.0;
    let mut k = 1;
    while k + 1 < n {
        let pair = rho(k) + rho(k + 1);
        if pair < 0.0 {
            break;
        }
        sum += pair;
        k += 2;
    }
    n as f64 / (1.0 + 2.0 * sum).max(1.0)
}

/// Gelman-Rubin R-hat of several chains of one scalar (values near 1 indicate convergence).
pub fn gelman_rubin(chains: &[Vec<f64>]) -> Result<f64> {
    let m = chains.len();
    let n = chains.first().map_or(0, Vec::len);
    if m < 2 || n < 2 || chains.iter().any(|c| c.len() != n) {
        return Err(Error::Msg("gelman_rubin: need >= 2 chains of equal length >= 2".into()));
    }
    let means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
    let grand = means.iter().sum::<f64>() / m as f64;
    let b = n as f64 / (m as f64 - 1.0) * means.iter().map(|v| (v - grand).powi(2)).sum::<f64>();
    let w = chains.iter().zip(&means).map(|(c, mu)| c.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sum::<f64>() / m as f64;
    if w == 0.0 {
        return Ok(1.0);
    }
    let var_plus = (n as f64 - 1.0) / n as f64 * w + b / n as f64;
    Ok((var_plus / w).sqrt())
}

type Mat = Vec<Vec<f64>>;

fn matmul(a: &Mat, b: &Mat) -> Mat {
    (0..a.len()).map(|i| (0..b[0].len()).map(|j| (0..b.len()).map(|k| a[i][k] * b[k][j]).sum()).collect()).collect()
}

fn transpose(a: &Mat) -> Mat {
    (0..a[0].len()).map(|j| a.iter().map(|r| r[j]).collect()).collect()
}

fn add(a: &Mat, b: &Mat) -> Mat {
    a.iter().zip(b).map(|(r, s)| r.iter().zip(s).map(|(x, y)| x + y).collect()).collect()
}

fn inverse(a: &Mat) -> Option<Mat> {
    let n = a.len();
    let mut m: Mat = a.iter().enumerate().map(|(i, r)| r.iter().cloned().chain((0..n).map(|j| f64::from(u8::from(i == j)))).collect()).collect();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))?;
        if m[p][c].abs() < 1e-14 {
            return None;
        }
        m.swap(c, p);
        let d = m[c][c];
        m[c].iter_mut().for_each(|v| *v /= d);
        let pivot = m[c].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r != c {
                let w = row[c];
                row.iter_mut().zip(&pivot).for_each(|(v, p)| *v -= w * p);
            }
        }
    }
    Some(m.into_iter().map(|r| r[n..].to_vec()).collect())
}

/// Linear-Gaussian state space model `x' = F x + w (Q)`, `y = H x + v (R)`.
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    pub f: Mat,
    pub h: Mat,
    pub q: Mat,
    pub r: Mat,
}

/// Filtered (and optionally smoothed) state means and covariances per time step.
#[derive(Debug, Clone)]
pub struct KalmanOutput {
    pub means: Vec<Vec<f64>>,
    pub covs: Vec<Mat>,
    pub log_likelihood: f64,
}

impl KalmanFilter {
    /// Runs the filter over observations `ys` from the prior `(x0, p0)`.
    pub fn filter(&self, ys: &[Vec<f64>], x0: &[f64], p0: &Mat) -> Result<KalmanOutput> {
        let (mut x, mut p) = (x0.to_vec(), p0.clone());
        let (mut means, mut covs, mut ll) = (Vec::new(), Vec::new(), 0.0);
        for y in ys {
            let xp: Vec<f64> = self.f.iter().map(|r| r.iter().zip(&x).map(|(a, b)| a * b).sum()).collect();
            let pp = add(&matmul(&matmul(&self.f, &p), &transpose(&self.f)), &self.q);
            let s = add(&matmul(&matmul(&self.h, &pp), &transpose(&self.h)), &self.r);
            let s_inv = inverse(&s).ok_or_else(|| Error::Msg("KalmanFilter: singular innovation covariance".into()))?;
            let k = matmul(&matmul(&pp, &transpose(&self.h)), &s_inv);
            let innov: Vec<f64> = y.iter().zip(&self.h).map(|(yi, hr)| yi - hr.iter().zip(&xp).map(|(a, b)| a * b).sum::<f64>()).collect();
            x = xp.iter().enumerate().map(|(i, v)| v + k[i].iter().zip(&innov).map(|(a, b)| a * b).sum::<f64>()).collect();
            let kh = matmul(&k, &self.h);
            let i_kh: Mat = (0..p.len()).map(|i| (0..p.len()).map(|j| f64::from(u8::from(i == j)) - kh[i][j]).collect()).collect();
            p = matmul(&i_kh, &pp);
            let det_ln = {
                // log-determinant via the inverse's Cholesky-free route: product of eigen-free LU pivots
                let mut a = s.clone();
                let mut ld = 0.0;
                for c in 0..a.len() {
                    ld += a[c][c].abs().ln();
                    for r in c + 1..a.len() {
                        let w = a[r][c] / a[c][c];
                        for kk in c..a.len() {
                            a[r][kk] -= w * a[c][kk];
                        }
                    }
                }
                ld
            };
            let quad: f64 = innov.iter().enumerate().map(|(i, v)| v * s_inv[i].iter().zip(&innov).map(|(a, b)| a * b).sum::<f64>()).sum();
            ll += -0.5 * (det_ln + quad + innov.len() as f64 * (2.0 * std::f64::consts::PI).ln());
            means.push(x.clone());
            covs.push(p.clone());
        }
        Ok(KalmanOutput { means, covs, log_likelihood: ll })
    }

    /// Rauch-Tung-Striebel smoother applied to a filter output.
    pub fn smooth(&self, out: &KalmanOutput) -> Result<KalmanOutput> {
        let n = out.means.len();
        let (mut means, mut covs) = (out.means.clone(), out.covs.clone());
        for t in (0..n.saturating_sub(1)).rev() {
            let pp = add(&matmul(&matmul(&self.f, &out.covs[t]), &transpose(&self.f)), &self.q);
            let c = matmul(&matmul(&out.covs[t], &transpose(&self.f)), &inverse(&pp).ok_or_else(|| Error::Msg("KalmanFilter::smooth: singular prediction covariance".into()))?);
            let xp: Vec<f64> = self.f.iter().map(|r| r.iter().zip(&out.means[t]).map(|(a, b)| a * b).sum()).collect();
            let dx: Vec<f64> = means[t + 1].iter().zip(&xp).map(|(a, b)| a - b).collect();
            means[t] = out.means[t].iter().enumerate().map(|(i, v)| v + c[i].iter().zip(&dx).map(|(a, b)| a * b).sum::<f64>()).collect();
            let dp: Mat = covs[t + 1].iter().zip(&pp).map(|(r, s)| r.iter().zip(s).map(|(a, b)| a - b).collect()).collect();
            covs[t] = add(&out.covs[t], &matmul(&matmul(&c, &dp), &transpose(&c)));
        }
        Ok(KalmanOutput { means, covs, log_likelihood: out.log_likelihood })
    }
}

/// Discrete hidden Markov model.
#[derive(Debug, Clone)]
pub struct Hmm {
    pub start: Vec<f64>,
    pub trans: Mat,
    pub emit: Mat,
}

impl Hmm {
    fn check(&self, obs: &[usize]) -> Result<()> {
        let k = self.start.len();
        if k == 0 || self.trans.len() != k || self.emit.len() != k || obs.iter().any(|&o| o >= self.emit[0].len()) || obs.is_empty() {
            return Err(Error::Msg("Hmm: inconsistent dimensions or observation out of range".into()));
        }
        Ok(())
    }

    /// Scaled forward-backward: returns posterior state probabilities per step and the log-likelihood.
    pub fn forward_backward(&self, obs: &[usize]) -> Result<(Mat, f64)> {
        self.check(obs)?;
        let (k, t) = (self.start.len(), obs.len());
        let (mut alpha, mut scale) = (vec![vec![0.0; k]; t], vec![0.0; t]);
        for i in 0..k {
            alpha[0][i] = self.start[i] * self.emit[i][obs[0]];
        }
        for s in 0..t {
            if s > 0 {
                for j in 0..k {
                    alpha[s][j] = (0..k).map(|i| alpha[s - 1][i] * self.trans[i][j]).sum::<f64>() * self.emit[j][obs[s]];
                }
            }
            scale[s] = alpha[s].iter().sum::<f64>();
            if scale[s] == 0.0 {
                return Err(Error::Msg("Hmm: the observation sequence has zero probability".into()));
            }
            alpha[s].iter_mut().for_each(|v| *v /= scale[s]);
        }
        let mut beta = vec![vec![1.0; k]; t];
        for s in (0..t - 1).rev() {
            for i in 0..k {
                beta[s][i] = (0..k).map(|j| self.trans[i][j] * self.emit[j][obs[s + 1]] * beta[s + 1][j]).sum::<f64>() / scale[s + 1];
            }
        }
        let gamma = (0..t).map(|s| { let g: Vec<f64> = (0..k).map(|i| alpha[s][i] * beta[s][i]).collect(); let z: f64 = g.iter().sum(); g.iter().map(|v| v / z).collect() }).collect();
        Ok((gamma, scale.iter().map(|s| s.ln()).sum()))
    }

    /// Most likely state sequence.
    pub fn viterbi(&self, obs: &[usize]) -> Result<Vec<usize>> {
        self.check(obs)?;
        let (k, t) = (self.start.len(), obs.len());
        let ln = |v: f64| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY };
        let mut delta: Vec<f64> = (0..k).map(|i| ln(self.start[i]) + ln(self.emit[i][obs[0]])).collect();
        let mut back = vec![vec![0usize; k]; t];
        for s in 1..t {
            let prev = delta.clone();
            for j in 0..k {
                let (best, arg) = (0..k).map(|i| (prev[i] + ln(self.trans[i][j]), i)).fold((f64::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a });
                delta[j] = best + ln(self.emit[j][obs[s]]);
                back[s][j] = arg;
            }
        }
        let mut state = (0..k).max_by(|&a, &b| delta[a].total_cmp(&delta[b])).unwrap();
        let mut path = vec![state];
        for s in (1..t).rev() {
            state = back[s][state];
            path.push(state);
        }
        path.reverse();
        Ok(path)
    }

    /// Baum-Welch (EM) re-estimation from one observation sequence; returns the log-likelihood trace.
    pub fn baum_welch(&mut self, obs: &[usize], iters: usize) -> Result<Vec<f64>> {
        self.check(obs)?;
        let (k, m, t) = (self.start.len(), self.emit[0].len(), obs.len());
        let mut trace = Vec::new();
        for _ in 0..iters {
            let (gamma, ll) = self.forward_backward(obs)?;
            trace.push(ll);
            // expected transitions xi via a fresh scaled pass
            let mut xi_sum = vec![vec![0.0; k]; k];
            let (mut alpha, mut scale) = (vec![vec![0.0; k]; t], vec![0.0; t]);
            for i in 0..k {
                alpha[0][i] = self.start[i] * self.emit[i][obs[0]];
            }
            for s in 0..t {
                if s > 0 {
                    for j in 0..k {
                        alpha[s][j] = (0..k).map(|i| alpha[s - 1][i] * self.trans[i][j]).sum::<f64>() * self.emit[j][obs[s]];
                    }
                }
                scale[s] = alpha[s].iter().sum();
                alpha[s].iter_mut().for_each(|v| *v /= scale[s]);
            }
            let mut beta = vec![vec![1.0; k]; t];
            for s in (0..t - 1).rev() {
                for i in 0..k {
                    beta[s][i] = (0..k).map(|j| self.trans[i][j] * self.emit[j][obs[s + 1]] * beta[s + 1][j]).sum::<f64>() / scale[s + 1];
                }
            }
            for s in 0..t - 1 {
                for i in 0..k {
                    for j in 0..k {
                        xi_sum[i][j] += alpha[s][i] * self.trans[i][j] * self.emit[j][obs[s + 1]] * beta[s + 1][j] / scale[s + 1];
                    }
                }
            }
            self.start = gamma[0].clone();
            for i in 0..k {
                let denom: f64 = xi_sum[i].iter().sum::<f64>().max(1e-300);
                self.trans[i] = xi_sum[i].iter().map(|v| v / denom).collect();
                let g_sum: f64 = gamma.iter().map(|g| g[i]).sum::<f64>().max(1e-300);
                self.emit[i] = (0..m).map(|o| (0..t).filter(|&s| obs[s] == o).map(|s| gamma[s][i]).sum::<f64>() / g_sum).collect();
            }
        }
        Ok(trace)
    }
}

/// Bootstrap particle filter for a scalar state; returns the filtered posterior mean at each step.
/// `transition(x, rng_noise)` propagates a particle, `likelihood(x, y)` weights it.
pub fn particle_filter<T, L>(ys: &[f64], n_particles: usize, init: &[f64], transition: T, likelihood: L, seed: u64) -> Result<Vec<f64>>
where
    T: Fn(f64, f64) -> f64,
    L: Fn(f64, f64) -> f64,
{
    if ys.is_empty() || init.len() != n_particles || n_particles < 2 {
        return Err(Error::Msg("particle_filter: need observations and one initial value per particle (>= 2)".into()));
    }
    let mut rng = Rng::new(seed);
    let mut parts = init.to_vec();
    let mut means = Vec::with_capacity(ys.len());
    for &y in ys {
        parts = parts.iter().map(|&x| transition(x, gauss(&mut rng))).collect();
        let w: Vec<f64> = parts.iter().map(|&x| likelihood(x, y)).collect();
        let total: f64 = w.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::Msg("particle_filter: all particle weights vanished".into()));
        }
        means.push(parts.iter().zip(&w).map(|(x, wi)| x * wi).sum::<f64>() / total);
        // systematic resampling
        let step = total / n_particles as f64;
        let (mut u, mut cum, mut idx) = (rng.next_f64() * step, w[0], 0);
        let mut next = Vec::with_capacity(n_particles);
        for _ in 0..n_particles {
            while u > cum && idx + 1 < n_particles {
                idx += 1;
                cum += w[idx];
            }
            next.push(parts[idx]);
            u += step;
        }
        parts = next;
    }
    Ok(means)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_var(chain: &Chain, d: usize) -> (f64, f64) {
        let v: Vec<f64> = chain.samples.iter().map(|s| s[d]).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (m, v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64)
    }

    #[test]
    fn mcmc_samples_a_gaussian() {
        // target N(3, 2^2) in dimension 0 and N(-1, 1) in dimension 1
        let lp = |x: &[f64]| -0.5 * ((x[0] - 3.0) / 2.0).powi(2) - 0.5 * (x[1] + 1.0).powi(2);
        let mh = metropolis_hastings(lp, &[0.0, 0.0], 1.5, 20000, 2000, 1).unwrap();
        let (m0, v0) = mean_var(&mh, 0);
        assert!((m0 - 3.0).abs() < 0.2 && (v0 - 4.0).abs() < 0.8, "{m0} {v0}");
        assert!(mh.accept_rate > 0.1 && mh.accept_rate < 0.8);
        let grad = |x: &[f64]| vec![-(x[0] - 3.0) / 4.0, -(x[1] + 1.0)];
        let h = hmc(lp, grad, &[0.0, 0.0], 0.3, 10, 4000, 500, 2).unwrap();
        let (m1, v1) = mean_var(&h, 1);
        assert!((m1 + 1.0).abs() < 0.1 && (v1 - 1.0).abs() < 0.2, "{m1} {v1}");
        assert!(h.accept_rate > 0.6, "HMC accepts most proposals: {}", h.accept_rate);
        // HMC mixes better than a random walk: higher effective sample size per draw
        let ess_h = effective_sample_size(&h.samples.iter().map(|s| s[0]).collect::<Vec<_>>()) / 4000.0;
        let ess_m = effective_sample_size(&mh.samples.iter().map(|s| s[0]).collect::<Vec<_>>()) / 20000.0;
        assert!(ess_h > ess_m, "{ess_h} vs {ess_m}");
        assert!(metropolis_hastings(|_| f64::NEG_INFINITY, &[0.0], 1.0, 10, 0, 0).is_err());
    }

    #[test]
    fn diagnostics() {
        let iid: Vec<f64> = { let mut r = Rng::new(3); (0..2000).map(|_| gauss(&mut r)).collect() };
        assert!(effective_sample_size(&iid) > 1500.0);
        let sticky: Vec<f64> = iid.iter().scan(0.0, |s, v| { *s = 0.95 * *s + v; Some(*s) }).collect();
        assert!(effective_sample_size(&sticky) < 300.0);
        let chains: Vec<Vec<f64>> = (0..4).map(|s| { let mut r = Rng::new(10 + s); (0..1000).map(|_| gauss(&mut r)).collect() }).collect();
        assert!((gelman_rubin(&chains).unwrap() - 1.0).abs() < 0.02);
        let shifted: Vec<Vec<f64>> = chains.iter().enumerate().map(|(i, c)| c.iter().map(|v| v + 3.0 * i as f64).collect()).collect();
        assert!(gelman_rubin(&shifted).unwrap() > 2.0);
        assert!(gelman_rubin(&chains[..1]).is_err());
    }

    #[test]
    fn kalman_tracks_and_smooths_constant_velocity() {
        let kf = KalmanFilter { f: vec![vec![1.0, 1.0], vec![0.0, 1.0]], h: vec![vec![1.0, 0.0]], q: vec![vec![1e-4, 0.0], vec![0.0, 1e-4]], r: vec![vec![1.0]] };
        let mut rng = Rng::new(4);
        let truth: Vec<f64> = (0..60).map(|t| 2.0 + 0.5 * t as f64).collect();
        let ys: Vec<Vec<f64>> = truth.iter().map(|x| vec![x + gauss(&mut rng)]).collect();
        let out = kf.filter(&ys, &[0.0, 0.0], &vec![vec![100.0, 0.0], vec![0.0, 100.0]]).unwrap();
        let est_vel = out.means.last().unwrap()[1];
        assert!((est_vel - 0.5).abs() < 0.1, "{est_vel}");
        let raw_err: f64 = ys.iter().zip(&truth).map(|(y, x)| (y[0] - x).powi(2)).sum::<f64>() / 60.0;
        let filt_err: f64 = out.means.iter().zip(&truth).skip(10).map(|(m, x)| (m[0] - x).powi(2)).sum::<f64>() / 50.0;
        assert!(filt_err < raw_err, "filtering reduces noise: {filt_err} vs {raw_err}");
        let sm = kf.smooth(&out).unwrap();
        let sm_err: f64 = sm.means.iter().zip(&truth).skip(10).map(|(m, x)| (m[0] - x).powi(2)).sum::<f64>() / 50.0;
        assert!(sm_err <= filt_err + 1e-9, "{sm_err} vs {filt_err}");
        assert!(out.log_likelihood.is_finite());
    }

    #[test]
    fn hmm_decoding_and_learning() {
        // sticky two-state chain with distinct emissions
        let hmm = Hmm { start: vec![0.5, 0.5], trans: vec![vec![0.9, 0.1], vec![0.1, 0.9]], emit: vec![vec![0.9, 0.1], vec![0.1, 0.9]] };
        let obs = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0];
        assert_eq!(hmm.viterbi(&obs).unwrap(), vec![0, 0, 0, 0, 1, 1, 1, 1, 0, 0]);
        let (gamma, ll) = hmm.forward_backward(&obs).unwrap();
        assert!(gamma.iter().all(|g| (g.iter().sum::<f64>() - 1.0).abs() < 1e-9) && ll < 0.0);
        assert!(gamma[1][0] > 0.9 && gamma[5][1] > 0.9);
        // Baum-Welch never decreases the likelihood and learns sticky structure from a start with weak asymmetry
        let mut rng = Rng::new(8);
        let (mut state, mut seq) = (0usize, Vec::new());
        for _ in 0..400 {
            if rng.next_f64() < 0.1 { state = 1 - state; }
            seq.push(if rng.next_f64() < 0.85 { state } else { 1 - state });
        }
        let mut learner = Hmm { start: vec![0.6, 0.4], trans: vec![vec![0.6, 0.4], vec![0.4, 0.6]], emit: vec![vec![0.6, 0.4], vec![0.4, 0.6]] };
        let trace = learner.baum_welch(&seq, 30).unwrap();
        assert!(trace.windows(2).all(|w| w[1] >= w[0] - 1e-9), "EM must not decrease the likelihood");
        assert!(trace.last().unwrap() > &trace[0]);
        assert!(learner.trans[0][0] > 0.75 || learner.trans[1][1] > 0.75, "{:?}", learner.trans);
        assert!(hmm.viterbi(&[5]).is_err());
    }

    #[test]
    fn particle_filter_tracks_a_random_walk() {
        let mut rng = Rng::new(6);
        let (mut x, mut truth, mut ys) = (0.0, Vec::new(), Vec::new());
        for _ in 0..80 {
            x += 0.3 * gauss(&mut rng);
            truth.push(x);
            ys.push(x + 0.5 * gauss(&mut rng));
        }
        let init: Vec<f64> = { let mut r = Rng::new(1); (0..500).map(|_| gauss(&mut r)).collect() };
        let est = particle_filter(&ys, 500, &init, |x, n| x + 0.3 * n, |x, y| (-0.5 * ((y - x) / 0.5).powi(2)).exp(), 7).unwrap();
        let err: f64 = est.iter().zip(&truth).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 80.0;
        let raw: f64 = ys.iter().zip(&truth).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 80.0;
        assert!(err < raw, "{err} vs {raw}");
        assert!(particle_filter(&[], 10, &[0.0; 10], |x, _| x, |_, _| 1.0, 0).is_err());
    }
}
