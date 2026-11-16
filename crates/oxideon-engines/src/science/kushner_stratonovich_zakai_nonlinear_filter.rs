//! Zakai Linear Stochastic PDE & Kushner-Stratonovich Nonlinear Filtering Equation (Stratonovich, 1960; Kushner, 1964; Zakai, 1969).
//!
//! ### 1. Kallianpur-Striebel Bayes Formula & Zakai Innovation Likelihood Factor
//! For hidden state $X_t$ with generator $\mathcal{L}$ and observation $dY_t = h(X_t) dt + \sigma_y dB_t$, the unnormalized conditional density $q_t(x)$ satisfies Zakai's linear SPDE:
//! $$d q_t(x) = \mathcal{L}^* q_t(x) \, dt + \frac{h(x)}{\sigma_y^2} q_t(x) \, dY_t$$
//! Over a short step $\Delta t$ with observation increment $\Delta Y$, the Kallianpur-Striebel exponential weight is $\Lambda(x) = \exp\!\left(\frac{h(x) \Delta Y - \frac{1}{2} h(x)^2 \Delta t}{\sigma_y^2}\right)$.

use crate::error::{Error, Result};

/// Zakai Unnormalized Nonlinear Filtering & Kallianpur-Striebel Engine.
#[derive(Debug, Clone, Copy)]
pub struct ZakaiUnnormalizedFilteringSde {
    /// Observation noise standard deviation $\sigma_y > 0$.
    pub obs_noise_sigma_y: f64,
}

impl Default for ZakaiUnnormalizedFilteringSde {
    fn default() -> Self {
        Self {
            obs_noise_sigma_y: 1.0,
        }
    }
}

impl ZakaiUnnormalizedFilteringSde {
    /// Creates a new Zakai Unnormalized Filtering SDE engine.
    pub fn new(obs_noise_sigma_y: f64) -> Result<Self> {
        if obs_noise_sigma_y <= 0.0 || !obs_noise_sigma_y.is_finite() {
            return Err(Error::Msg("obs_noise_sigma_y must be > 0".to_string()));
        }
        Ok(Self { obs_noise_sigma_y })
    }

    /// Evaluates the Kallianpur-Striebel unnormalized Radon-Nikodym weight $\Lambda(x)$ and normalized posterior mean over discrete particles.
    pub fn kallianpur_striebel_posterior_mean(
        &self,
        particle_states: &[f64],
        dy_increment: f64,
        dt: f64,
    ) -> Result<(f64, f64)> {
        if particle_states.is_empty() || dt <= 0.0 {
            return Err(Error::Msg(
                "particle_states must be non-empty and dt > 0".to_string(),
            ));
        }
        let s2 = self.obs_noise_sigma_y * self.obs_noise_sigma_y;
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        for &x in particle_states {
            let w = ((x * dy_increment - 0.5 * x * x * dt) / s2).exp();
            sum_w += w;
            sum_wx += w * x;
        }
        Ok((sum_w / particle_states.len() as f64, sum_wx / sum_w))
    }
}
