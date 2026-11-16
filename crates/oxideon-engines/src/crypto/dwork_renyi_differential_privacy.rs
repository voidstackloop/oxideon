//! Dwork $(\epsilon, \delta)$-Differential Privacy, Mironov Rényi Differential Privacy (RDP) & Abadi Moments Accountant (Dwork, 2006; Mironov, 2017).
//!
//! ### 1. Exact Composition of Rényi Differential Privacy (RDP) & Conversion to $(\epsilon, \delta)$-DP
//! A randomized mechanism $\mathcal{M}$ satisfies $(\alpha, \epsilon_{\mathrm{RDP}})$-Rényi Differential Privacy ($lpha > 1$) if
//! $D_\alpha(\mathcal{M}(D) \| \mathcal{M}(D')) \le \epsilon_{\mathrm{RDP}}$ for adjacent datasets $D, D'$. Under $T$ adaptive compositions
//! of the Gaussian mechanism ($\sigma_{\mathrm{noise}} = \sigma \Delta_2$), the total RDP cost adds linearly:
//! $$\epsilon_{\mathrm{RDP}}(\alpha; T) = \frac{T \alpha}{2 \sigma^2}, \qquad \epsilon(\delta) = \min_{\alpha > 1}\left\{\frac{T \alpha}{2 \sigma^2} + \frac{\ln(1 / \delta)}{\alpha - 1}\right\} = \frac{T}{2 \sigma^2} + \sqrt{\frac{2 T \ln(1/\delta)}{\sigma^2}}$$

use crate::error::{Error, Result};

/// Rényi Differential Privacy (RDP) & Moments Accountant Engine.
#[derive(Debug, Clone, Copy)]
pub struct RenyiDifferentialPrivacyAccountant {
    /// Noise multiplier $\sigma > 0$.
    pub noise_multiplier_sigma: f64,
}

impl RenyiDifferentialPrivacyAccountant {
    /// Creates a new Rényi Differential Privacy Accountant engine.
    pub fn new(noise_multiplier_sigma: f64) -> Result<Self> {
        if noise_multiplier_sigma <= 0.0 {
            return Err(Error::Msg("noise_multiplier_sigma must be > 0".to_string()));
        }
        Ok(Self {
            noise_multiplier_sigma,
        })
    }

    /// Computes the optimal converted $(\epsilon, \delta)$-DP privacy budget $\epsilon(\delta)$ and optimal Rényi order $\alpha^*$ after $T$ Gaussian steps.
    pub fn optimal_dp_epsilon(&self, steps_t: usize, failure_delta: f64) -> Result<(f64, f64)> {
        if steps_t == 0 || !(0.0..1.0).contains(&failure_delta) || failure_delta == 0.0 {
            return Err(Error::Msg(
                "steps_t >= 1 and failure_delta in (0, 1) required".to_string(),
            ));
        }
        let t = steps_t as f64;
        let s2 = self.noise_multiplier_sigma * self.noise_multiplier_sigma;
        let log_inv_delta = (1.0 / failure_delta).ln();
        let alpha_star = 1.0 + (2.0 * s2 * log_inv_delta / t).sqrt();
        let eps = t / (2.0 * s2) + (2.0 * t * log_inv_delta / s2).sqrt();
        Ok((eps, alpha_star))
    }
}
