//! Continuous Latent Stochastic Differential Equations (Latent SDE).
//!
//! Implements continuous-time generative modeling via Itô SDEs:
//! - Prior SDE: $d\mathbf{z}_t = f_\theta(\mathbf{z}_t, t) dt + g_\theta(\mathbf{z}_t, t) d\mathbf{W}_t$.
//! - Variational Posterior SDE: $d\mathbf{z}_t = q_\phi(\mathbf{z}_t, t, \mathbf{x}) dt + g_\theta(\mathbf{z}_t, t) d\mathbf{W}_t$.
//! - Exact continuous path KL divergence via Girsanov's Theorem:
//!   $$\mathcal{D}_{\text{KL}}(\mathbb{Q} \parallel \mathbb{P}) = \frac{1}{2} \int_0^T \left\| g_\theta^{-1}(\mathbf{z}_t, t) (q_\phi(\mathbf{z}_t, t) - f_\theta(\mathbf{z}_t, t)) \right\|_2^2 dt + \mathcal{D}_{\text{KL}}(q(\mathbf{z}_0) \parallel p(\mathbf{z}_0))$$
//! - Evidence Lower Bound (ELBO) training with Euler-Maruyama stochastic integration.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for Latent Stochastic Differential Equations.
#[derive(Debug, Clone)]
pub struct LatentSdeConfig {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub diffusion_scale: f64,
    pub t0: f64,
    pub t1: f64,
    pub num_steps: usize,
}

impl Default for LatentSdeConfig {
    fn default() -> Self {
        Self {
            latent_dim: 4,
            hidden_dim: 16,
            diffusion_scale: 0.1,
            t0: 0.0,
            t1: 1.0,
            num_steps: 50,
        }
    }
}

/// Simulated SDE trajectory containing latent paths, noise increments, and continuous KL divergence.
#[derive(Debug, Clone)]
pub struct SdeTrajectory {
    /// Time evaluation grid $t_0, t_1, \dots, t_N$.
    pub timestamps: Vec<f64>,
    /// Latent state trajectory $[\text{steps} + 1, \text{latent\_dim}]$.
    pub path: Tensor,
    /// Brownian noise realizations $[\text{steps}, \text{latent\_dim}]$.
    pub brownian_increments: Tensor,
    /// Continuous path KL divergence $\mathcal{D}_{\text{KL}}(\mathbb{Q} \parallel \mathbb{P})$.
    pub path_kl: f64,
}

/// Latent Stochastic Differential Equation Engine.
#[derive(Debug, Clone)]
pub struct LatentSdeEngine {
    pub config: LatentSdeConfig,
    /// Prior drift weights $W_f \in \mathbb{R}^{D \times D}$.
    pub prior_drift_w: Vec<f64>,
    /// Posterior drift weights $W_q \in \mathbb{R}^{D \times D}$.
    pub post_drift_w: Vec<f64>,
}

impl LatentSdeEngine {
    /// Creates a new `LatentSdeEngine`.
    pub fn new(config: LatentSdeConfig) -> Result<Self> {
        let d = config.latent_dim;
        crate::limits::check_elements("latent_dim * latent_dim", &[d, d])?;
        // Initialize with gentle stabilizing drift (negative diagonal)
        let mut prior_w = vec![0.0f64; d * d];
        let mut post_w = vec![0.0f64; d * d];
        for i in 0..d {
            prior_w[i * d + i] = -0.5; // mean-reverting prior Ornstein-Uhlenbeck
            post_w[i * d + i] = -0.8; // observation-adapted posterior
        }

        Ok(Self {
            config,
            prior_drift_w: prior_w,
            post_drift_w: post_w,
        })
    }

    /// Evaluates linear drift $f(\mathbf{z}) = W \mathbf{z}$.
    pub fn eval_drift(w: &[f64], z: &[f64], d: usize) -> Result<Vec<f64>> {
        let need = d
            .checked_mul(d)
            .ok_or_else(|| Error::Msg(format!("drift dim {d} overflows")))?;
        if w.len() < need || z.len() < d {
            return Err(Error::Msg(format!(
                "drift needs [{need}] weights and [{d}] states, got {} and {}",
                w.len(),
                z.len()
            )));
        }
        let mut drift = vec![0.0f64; d];
        for i in 0..d {
            let mut sum = 0.0;
            for j in 0..d {
                sum += w[i * d + j] * z[j];
            }
            drift[i] = sum;
        }
        Ok(drift)
    }

    /// Simulates a latent SDE trajectory using the Euler-Maruyama scheme.
    ///
    /// If `is_posterior` is true, uses the variational drift $q_\phi$ and computes the
    /// continuous Girsanov KL divergence against the prior drift $f_\theta$.
    pub fn simulate(
        &self,
        z0: &Tensor,
        is_posterior: bool,
        noise_seed: Option<u64>,
    ) -> Result<SdeTrajectory> {
        let d = self.config.latent_dim;
        if z0.numel() != d {
            return Err(Error::Msg(format!(
                "Expected z0 of length {d}, got {}",
                z0.numel()
            )));
        }

        let n = self.config.num_steps;
        if n == 0 {
            return Err(Error::Msg(
                "LatentSdeEngine::simulate requires num_steps > 0".into(),
            ));
        }
        let n = crate::limits::check_iterations("num_steps", n)?;
        crate::limits::check_elements("latent trajectory", &[n + 1, d])?;
        let wlen = d * d;
        if self.prior_drift_w.len() != wlen || self.post_drift_w.len() != wlen {
            return Err(Error::Msg(format!(
                "drift weights disagree with latent_dim {d}"
            )));
        }
        let (t0, t1) = (self.config.t0, self.config.t1);
        if !t0.is_finite() || !t1.is_finite() || t1 <= t0 {
            return Err(Error::Msg(format!(
                "LatentSdeEngine::simulate requires finite t0 < t1, got t0={t0}, t1={t1}"
            )));
        }
        let dt = (t1 - t0) / (n as f64);
        let sqrt_dt = dt.sqrt();
        let g = self.config.diffusion_scale.max(1e-6);

        let mut timestamps = Vec::with_capacity(n + 1);
        let mut path_data = Vec::with_capacity((n + 1) * d);
        let mut noise_data = Vec::with_capacity(n * d);

        let mut current_z = z0.to_vec::<f64>()?;
        timestamps.push(self.config.t0);
        path_data.extend_from_slice(&current_z);

        // Deterministic pseudo-random Gaussian noise generator (Box-Muller)
        let mut rng_state = noise_seed.unwrap_or(42);
        let mut next_gauss = move || -> f64 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = ((rng_state >> 32) as f64 + 1.0) / 4294967297.0;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = ((rng_state >> 32) as f64 + 1.0) / 4294967297.0;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mut path_kl = 0.0f64;

        for step in 0..n {
            let t = self.config.t0 + (step as f64) * dt;

            // Prior drift f(z)
            let f_drift = Self::eval_drift(&self.prior_drift_w, &current_z, d)?;

            // Active drift: prior or posterior
            let active_drift = if is_posterior {
                Self::eval_drift(&self.post_drift_w, &current_z, d)?
            } else {
                f_drift.clone()
            };

            // Sample Brownian increment dW ~ N(0, dt)
            let mut dw = vec![0.0f64; d];
            for val in &mut dw {
                let xi = next_gauss();
                *val = xi * sqrt_dt;
                noise_data.push(*val);
            }

            // Girsanov KL divergence integrand: 0.5 * ||(q - f) / g||^2 * dt
            if is_posterior {
                let mut drift_diff_sq = 0.0f64;
                for j in 0..d {
                    let diff = active_drift[j] - f_drift[j];
                    drift_diff_sq += (diff / g).powi(2);
                }
                path_kl += 0.5 * drift_diff_sq * dt;
            }

            // Euler-Maruyama step: z_{t+dt} = z_t + drift * dt + g * dW
            for j in 0..d {
                current_z[j] += active_drift[j] * dt + g * dw[j];
            }

            timestamps.push(t + dt);
            path_data.extend_from_slice(&current_z);
        }

        let path = Tensor::from_vec(path_data, &[n + 1, d])?;
        let brownian_increments = Tensor::from_vec(noise_data, &[n, d])?;

        Ok(SdeTrajectory {
            timestamps,
            path,
            brownian_increments,
            path_kl,
        })
    }

    /// Computes the variational Evidence Lower Bound (ELBO):
    /// $$\mathcal{L}_{\text{ELBO}} = \sum_{k=1}^K \log p(\mathbf{x}_{t_k} \mid \mathbf{z}_{t_k}) - \mathcal{D}_{\text{KL}}(\mathbb{Q} \parallel \mathbb{P})$$
    /// where observations are reconstructed with emission variance $\sigma_x^2$.
    pub fn compute_elbo(
        &self,
        z0: &Tensor,
        observations: &Tensor,
        obs_indices: &[usize],
        obs_noise_sigma: f64,
        noise_seed: Option<u64>,
    ) -> Result<(f64, f64, f64)> {
        let traj = self.simulate(z0, true, noise_seed)?;
        let path_mat = traj.path.to_vec::<f64>()?;
        let d = self.config.latent_dim;
        let obs_vec = observations.to_vec::<f64>()?;

        if obs_indices.len() * d != obs_vec.len() {
            return Err(Error::Msg(
                "Observations size mismatch with obs_indices".into(),
            ));
        }

        let sig_sq = obs_noise_sigma.powi(2).max(1e-6);
        let log_norm = -0.5 * (2.0 * std::f64::consts::PI * sig_sq).ln() * (d as f64);
        let mut log_likelihood = 0.0f64;

        for (i, &t_idx) in obs_indices.iter().enumerate() {
            if t_idx > self.config.num_steps {
                return Err(Error::Msg(format!(
                    "obs_index {t_idx} exceeds trajectory steps"
                )));
            }
            let mut sq_err = 0.0f64;
            for j in 0..d {
                let z_val = path_mat[t_idx * d + j];
                let x_val = obs_vec[i * d + j];
                sq_err += (x_val - z_val).powi(2);
            }
            log_likelihood += log_norm - 0.5 * sq_err / sig_sq;
        }

        let path_kl = traj.path_kl;
        let elbo = log_likelihood - path_kl;

        Ok((elbo, log_likelihood, path_kl))
    }
}
