//! Mironov Rényi Differential Privacy (RDP) $(\alpha, \varepsilon_{\mathrm{RDP}})$ & Abadi Moments Accountant (Abadi et al., 2016; Mironov, 2017).
//!
//! ### 1. Exact Gaussian RDP $\varepsilon_{\mathrm{RDP}}(\alpha) = \frac{\alpha \Delta_2^2}{2 \sigma^2}$ & Conversion to $(\varepsilon, \delta)$-DP
//! For order $\alpha > 1$, $k$ adaptive compositions of a Gaussian mechanism with noise multiplier $\sigma / \Delta_2$ satisfy $( \alpha, k \alpha \Delta_2^2 / (2 \sigma^2) )$-RDP, which converts to $(\varepsilon, \delta)$-DP with:
//! $$\varepsilon(\delta) = k \frac{\alpha \Delta_2^2}{2 \sigma^2} + \frac{\ln(1 / \delta)}{\alpha - 1}$$

use crate::error::{Error, Result};

/// Mironov Rényi Differential Privacy & Moments Accountant Engine.
#[derive(Debug, Clone, Copy)]
pub struct MironovRenyiDpAccountant {
    /// Noise multiplier $\sigma / \Delta_2 > 0$.
    pub noise_multiplier: f64,
}

impl Default for MironovRenyiDpAccountant {
    fn default() -> Self {
        Self {
            noise_multiplier: 2.0,
        }
    }
}

impl MironovRenyiDpAccountant {
    /// Creates a new Mironov Rényi DP Accountant engine.
    pub fn new(noise_multiplier: f64) -> Result<Self> {
        if noise_multiplier <= 0.0 || !noise_multiplier.is_finite() {
            return Err(Error::Msg("noise_multiplier must be > 0".to_string()));
        }
        Ok(Self { noise_multiplier })
    }

    /// Evaluates the $k$-step accumulated RDP budget $\varepsilon_{\mathrm{RDP}}(\alpha)$ and converted $(\varepsilon, \delta)$-DP guarantee.
    pub fn rdp_and_converted_dp_epsilon(
        &self,
        order_alpha: f64,
        steps_k: usize,
        delta: f64,
    ) -> Result<(f64, f64)> {
        if order_alpha <= 1.0 || steps_k == 0 || !(0.0..1.0).contains(&delta) || delta == 0.0 {
            return Err(Error::Msg(
                "order_alpha > 1, steps_k >= 1, delta in (0, 1) required".to_string(),
            ));
        }
        let rdp =
            (steps_k as f64) * order_alpha / (2.0 * self.noise_multiplier * self.noise_multiplier);
        let eps_dp = rdp + (1.0 / delta).ln() / (order_alpha - 1.0);
        Ok((rdp, eps_dp))
    }
}
