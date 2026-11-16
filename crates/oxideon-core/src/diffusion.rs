//! Denoising Diffusion Probabilistic Models (`DDPM` & `DDIM` noise schedulers and sinusoidal
//! timestep embeddings, Ho et al. 2020 / Song et al. 2020 / Nichol & Dhariwal 2021).

pub mod k_diffusion;
pub mod rectified_flow;

pub use k_diffusion::{EDMConfig, KDiffusionSampler};
pub use rectified_flow::{RectifiedFlowConfig, RectifiedFlowEngine};

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Variance schedule $\beta_1, \dots, \beta_T$ for diffusion processes.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// Linear schedule from `beta_start` (e.g. `1e-4`) to `beta_end` (e.g. `0.02`).
    Linear { beta_start: f64, beta_end: f64 },
    /// Improved cosine $\bar{\alpha}_t$ schedule (Nichol & Dhariwal 2021) with offset `s` (e.g. `0.008`).
    Cosine { s: f64 },
}

impl Default for BetaSchedule {
    fn default() -> Self {
        Self::Linear {
            beta_start: 1e-4,
            beta_end: 0.02,
        }
    }
}

/// Precomputed noise schedule for forward diffusion $q(x_t \mid x_0)$ and reverse sampling
/// via ancestral DDPM or accelerated DDIM.
#[derive(Debug, Clone)]
pub struct DDPMScheduler {
    pub num_train_timesteps: usize,
    pub betas: Vec<f64>,
    pub alphas: Vec<f64>,
    pub alphas_cumprod: Vec<f64>,
    pub sqrt_alphas_cumprod: Vec<f64>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,
    pub posterior_variance: Vec<f64>,
}

impl DDPMScheduler {
    pub fn new(num_train_timesteps: usize, schedule: BetaSchedule) -> Result<Self> {
        if num_train_timesteps == 0 {
            return Err(Error::Msg(
                "DDPMScheduler needs num_train_timesteps > 0".into(),
            ));
        }
        crate::limits::check_repeats("DDPMScheduler num_train_timesteps", num_train_timesteps)?;
        let betas: Vec<f64> = match schedule {
            BetaSchedule::Linear {
                beta_start,
                beta_end,
            } => (0..num_train_timesteps)
                .map(|t| {
                    let frac = if num_train_timesteps > 1 {
                        (t as f64) / ((num_train_timesteps - 1) as f64)
                    } else {
                        0.0
                    };
                    beta_start + frac * (beta_end - beta_start)
                })
                .collect(),
            BetaSchedule::Cosine { s } => {
                let f = |t: usize| -> f64 {
                    let u = ((t as f64) / (num_train_timesteps as f64) + s) / (1.0 + s);
                    (u * PI * 0.5).cos().powi(2)
                };
                let f0 = f(0);
                (0..num_train_timesteps)
                    .map(|t| {
                        let ac_t = f(t + 1) / f0;
                        let ac_prev = f(t) / f0;
                        (1.0 - ac_t / ac_prev).clamp(1e-5, 0.999)
                    })
                    .collect()
            }
        };

        let mut alphas = Vec::with_capacity(num_train_timesteps);
        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut sqrt_alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut sqrt_one_minus_alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut posterior_variance = Vec::with_capacity(num_train_timesteps);

        let mut cumprod = 1.0f64;
        for (t, &beta) in betas.iter().enumerate() {
            let alpha = 1.0 - beta;
            let prev_cumprod = cumprod;
            cumprod *= alpha;
            alphas.push(alpha);
            alphas_cumprod.push(cumprod);
            sqrt_alphas_cumprod.push(cumprod.sqrt());
            sqrt_one_minus_alphas_cumprod.push((1.0 - cumprod).max(0.0).sqrt());
            let post_var = if t == 0 {
                0.0
            } else {
                beta * (1.0 - prev_cumprod) / (1.0 - cumprod).max(1e-12)
            };
            posterior_variance.push(post_var);
        }

        Ok(Self {
            num_train_timesteps,
            betas,
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_variance,
        })
    }

    /// Forward diffusion process:
    /// $$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$
    /// Supports per-batch-element `timesteps` of length `batch_size = x_start.shape()[0]`.
    pub fn add_noise(
        &self,
        x_start: &Tensor,
        noise: &Tensor,
        timesteps: &[usize],
    ) -> Result<Tensor> {
        if x_start.shape() != noise.shape() {
            return Err(Error::ShapeMismatch {
                op: "DDPMScheduler::add_noise",
                expected: x_start.shape().to_vec(),
                got: noise.shape().to_vec(),
            });
        }
        if x_start.ndim() < 1 {
            return Err(Error::Msg(
                "DDPMScheduler::add_noise needs rank >= 1 inputs".into(),
            ));
        }
        let b = x_start.shape()[0];
        if timesteps.len() != b {
            return Err(Error::Msg(format!(
                "add_noise expected {b} timesteps, got {}",
                timesteps.len()
            )));
        }

        let mut broadcast_shape = vec![1usize; x_start.ndim()];
        broadcast_shape[0] = b;

        let mut c1 = Vec::with_capacity(b);
        let mut c2 = Vec::with_capacity(b);
        for &t in timesteps {
            if t >= self.num_train_timesteps {
                return Err(Error::Msg(format!(
                    "timestep {t} out of range 0..{}",
                    self.num_train_timesteps
                )));
            }
            c1.push(self.sqrt_alphas_cumprod[t]);
            c2.push(self.sqrt_one_minus_alphas_cumprod[t]);
        }

        let s1 = Tensor::from_slice_on(&c1, &broadcast_shape, &x_start.device())?
            .to_dtype(x_start.dtype())?;
        let s2 = Tensor::from_slice_on(&c2, &broadcast_shape, &x_start.device())?
            .to_dtype(x_start.dtype())?;

        x_start.mul(&s1)?.add(&noise.mul(&s2)?)
    }

    /// Self-consistency guard: all schedule tables are public and hand-tamperable,
    /// and every method indexes them by timestep.
    fn check_layout(&self) -> Result<()> {
        let n = self.num_train_timesteps;
        if n == 0 {
            return Err(Error::Msg(
                "DDPMScheduler: num_train_timesteps must be > 0".into(),
            ));
        }
        for (name, table) in [
            ("betas", &self.betas),
            ("alphas", &self.alphas),
            ("alphas_cumprod", &self.alphas_cumprod),
            ("sqrt_alphas_cumprod", &self.sqrt_alphas_cumprod),
            (
                "sqrt_one_minus_alphas_cumprod",
                &self.sqrt_one_minus_alphas_cumprod,
            ),
            ("posterior_variance", &self.posterior_variance),
        ] {
            if table.len() != n {
                return Err(Error::Msg(format!(
                    "DDPMScheduler: {name} has {} entries, need {n}",
                    table.len()
                )));
            }
        }
        Ok(())
    }

    fn check_timestep(&self, op: &'static str, t: usize) -> Result<()> {
        self.check_layout()?;
        if t >= self.num_train_timesteps {
            return Err(Error::Msg(format!(
                "{op}: timestep {t} out of range 0..{}",
                self.num_train_timesteps
            )));
        }
        Ok(())
    }

    /// Reconstructs $\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t}\, \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$.
    pub fn predict_start_from_noise(
        &self,
        x_t: &Tensor,
        t: usize,
        pred_noise: &Tensor,
    ) -> Result<Tensor> {
        self.check_timestep("predict_start_from_noise", t)?;
        let s_alpha = self.sqrt_alphas_cumprod[t].max(1e-12);
        let s_one_minus = self.sqrt_one_minus_alphas_cumprod[t];
        x_t.sub(&pred_noise.mul_scalar(s_one_minus)?)?
            .div_scalar(s_alpha)
    }

    /// Single reverse ancestral DDPM step $p_\theta(x_{t-1} \mid x_t)$ from timestep `t` to `t - 1`.
    pub fn step_ddpm(&self, pred_noise: &Tensor, t: usize, x_t: &Tensor) -> Result<Tensor> {
        self.check_timestep("step_ddpm", t)?;
        let alpha_t = self.alphas[t];
        let beta_t = self.betas[t];
        let sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t].max(1e-12);
        let coef = beta_t / sqrt_one_minus_ac;

        // mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise)
        let mean = x_t
            .sub(&pred_noise.mul_scalar(coef)?)?
            .div_scalar(alpha_t.sqrt())?;

        if t == 0 {
            Ok(mean)
        } else {
            let std = self.posterior_variance[t].sqrt();
            let z = Tensor::randn(x_t.shape())?
                .to_dtype(x_t.dtype())?
                .to_device(&x_t.device())?;
            mean.add(&z.mul_scalar(std)?)
        }
    }

    /// Accelerated DDIM step (Song et al. 2020) from timestep `t` to `t_prev` (or `0` when `t_prev == None`).
    /// Setting `eta = 0.0` yields a completely deterministic ODE trajectory.
    pub fn step_ddim(
        &self,
        pred_noise: &Tensor,
        t: usize,
        t_prev: Option<usize>,
        x_t: &Tensor,
        eta: f64,
    ) -> Result<Tensor> {
        self.check_timestep("step_ddim", t)?;
        if let Some(tp) = t_prev {
            self.check_timestep("step_ddim", tp)?;
        }
        let ac_t = self.alphas_cumprod[t];
        let ac_prev = t_prev.map_or(1.0, |tp| self.alphas_cumprod[tp]);

        let pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise)?;
        let sigma_t = if eta > 0.0 && t_prev.is_some() {
            eta * ((1.0 - ac_prev) / (1.0 - ac_t).max(1e-12) * (1.0 - ac_t / ac_prev.max(1e-12)))
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };

        let dir_scale = (1.0 - ac_prev - sigma_t * sigma_t).max(0.0).sqrt();
        let mut x_prev = pred_x0
            .mul_scalar(ac_prev.sqrt())?
            .add(&pred_noise.mul_scalar(dir_scale)?)?;

        if sigma_t > 0.0 {
            let z = Tensor::randn(x_t.shape())?
                .to_dtype(x_t.dtype())?
                .to_device(&x_t.device())?;
            x_prev = x_prev.add(&z.mul_scalar(sigma_t)?)?;
        }
        Ok(x_prev)
    }
}

/// Sinusoidal timestep embedding (`[batch_size, dim]`) used to condition Diffusion `UNet`s and `DiT`s.
///
/// The first `half = dim / 2` columns hold cosines and the next `half` hold sines.
/// When `dim` is odd the final column has no pair and stays zero (the classic
/// transformer convention), which callers should account for.
pub fn timestep_embedding(timesteps: &[usize], dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let max_period = 10000.0f32;
    let n = crate::limits::check_elements("timestep_embedding", &[timesteps.len(), dim])?;
    let mut out = vec![0.0f32; n];

    for (b, &t) in timesteps.iter().enumerate() {
        let tf = t as f32;
        for i in 0..half {
            let freq = (-max_period.ln() * (i as f32) / (half.max(1) as f32)).exp();
            let angle = tf * freq;
            out[b * dim + i] = angle.cos();
            out[b * dim + half + i] = angle.sin();
        }
    }
    Tensor::from_vec(out, &[timesteps.len(), dim])
}

/// Diffusion Forcing Scheduler for variable-noise sequence and video rollouts (Chen et al.).
///
/// Permits each sequence position $k$ to have an independent noise level $t_k$.
/// This bridges autoregressive generation with continuous diffusion, eliminating rollout error compounding.
#[derive(Debug, Clone)]
pub struct DiffusionForcingScheduler {
    pub ddpm: DDPMScheduler,
}

impl DiffusionForcingScheduler {
    pub fn new(num_train_timesteps: usize, schedule: BetaSchedule) -> Result<Self> {
        Ok(Self {
            ddpm: DDPMScheduler::new(num_train_timesteps, schedule)?,
        })
    }

    /// Adds independent per-token noise to an input tensor of shape `[batch_size, seq_len, dim]`.
    /// `token_timesteps` has length `seq_len` (or `batch_size * seq_len`).
    pub fn add_noise_per_token(
        &self,
        sequence: &Tensor,
        token_timesteps: &[usize],
    ) -> Result<Tensor> {
        self.ddpm.check_layout()?;
        let shape = sequence.shape();
        if shape.len() != 3 {
            return Err(Error::Msg(
                "add_noise_per_token expects [batch_size, seq_len, dim]".into(),
            ));
        }
        let batch_size = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];

        if token_timesteps.len() != seq_len {
            return Err(Error::Msg(format!(
                "token_timesteps length {} must match seq_len {}",
                token_timesteps.len(),
                seq_len
            )));
        }

        let noise = Tensor::randn(shape)?
            .to_dtype(sequence.dtype())?
            .to_device(&sequence.device())?;

        let seq_data = sequence.to_vec::<f32>()?;
        let noise_data = noise.to_vec::<f32>()?;
        let mut out_data = vec![0.0f32; sequence.numel()];

        for b in 0..batch_size {
            for (s, &step_t) in token_timesteps.iter().enumerate().take(seq_len) {
                let offset = (b * seq_len + s) * dim;
                if step_t == 0 {
                    // Uncorrupted conditioning observation (zero noise)
                    for d in 0..dim {
                        let idx = offset + d;
                        out_data[idx] = seq_data[idx];
                    }
                } else {
                    let t = (step_t - 1).min(self.ddpm.num_train_timesteps - 1);
                    let sqrt_alpha = self.ddpm.sqrt_alphas_cumprod[t] as f32;
                    let sqrt_one_minus = self.ddpm.sqrt_one_minus_alphas_cumprod[t] as f32;

                    for d in 0..dim {
                        let idx = offset + d;
                        out_data[idx] =
                            sqrt_alpha * seq_data[idx] + sqrt_one_minus * noise_data[idx];
                    }
                }
            }
        }

        let res = Tensor::from_slice(&out_data, shape)?;
        res.to_device(&sequence.device())
    }

    /// Performs one step of reverse diffusion per token from `current_timesteps` to `prev_timesteps`.
    pub fn step_per_token(
        &self,
        pred_noise: &Tensor,
        current_timesteps: &[usize],
        prev_timesteps: &[usize],
        x_t: &Tensor,
    ) -> Result<Tensor> {
        self.ddpm.check_layout()?;
        let shape = x_t.shape();
        if shape.len() != 3 {
            return Err(Error::Msg(
                "step_per_token expects [batch_size, seq_len, dim]".into(),
            ));
        }
        let batch_size = shape[0];
        let seq_len = shape[1];
        let dim = shape[2];
        if current_timesteps.len() != seq_len || prev_timesteps.len() != seq_len {
            return Err(Error::Msg(format!(
                "step_per_token: timestep slices must each hold seq_len={seq_len} entries, got {}/{}",
                current_timesteps.len(),
                prev_timesteps.len()
            )));
        }
        if pred_noise.shape() != x_t.shape() {
            return Err(Error::ShapeMismatch {
                op: "step_per_token",
                expected: x_t.shape().to_vec(),
                got: pred_noise.shape().to_vec(),
            });
        }

        let xt_data = x_t.to_vec::<f32>()?;
        let eps_data = pred_noise.to_vec::<f32>()?;
        let mut out_data = vec![0.0f32; x_t.numel()];

        for b in 0..batch_size {
            for (s, &cur_step_t) in current_timesteps.iter().enumerate().take(seq_len) {
                let horizon = self.ddpm.num_train_timesteps;
                if cur_step_t >= horizon {
                    return Err(Error::Msg(format!(
                        "step_per_token: current timestep {} out of range 0..{}",
                        cur_step_t, horizon
                    )));
                }
                let prev_t = prev_timesteps[s];
                if prev_t >= horizon {
                    return Err(Error::Msg(format!(
                        "step_per_token: prev timestep {} out of range 0..{}",
                        prev_t, horizon
                    )));
                }

                let _ac_t = self.ddpm.alphas_cumprod[cur_step_t];
                let ac_prev = if prev_t == 0 {
                    1.0
                } else {
                    self.ddpm.alphas_cumprod[prev_t]
                };
                let cur_t = cur_step_t;

                let sqrt_ac_t = self.ddpm.sqrt_alphas_cumprod[cur_t];
                let sqrt_one_minus_ac_t = self.ddpm.sqrt_one_minus_alphas_cumprod[cur_t];

                let offset = (b * seq_len + s) * dim;
                for d in 0..dim {
                    let idx = offset + d;
                    let x_cur = xt_data[idx] as f64;
                    let eps = eps_data[idx] as f64;

                    // Predict x_0
                    let x0_pred = (x_cur - sqrt_one_minus_ac_t * eps) / sqrt_ac_t.max(1e-12);
                    // DDIM deterministic update to prev_t
                    let dir_scale = (1.0 - ac_prev).max(0.0).sqrt();
                    let x_prev = x0_pred * ac_prev.sqrt() + eps * dir_scale;

                    out_data[idx] = x_prev as f32;
                }
            }
        }

        let res = Tensor::from_slice(&out_data, shape)?;
        res.to_device(&x_t.device())
    }
}
