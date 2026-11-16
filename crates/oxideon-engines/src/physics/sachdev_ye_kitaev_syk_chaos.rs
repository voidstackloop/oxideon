//! Sachdev-Ye-Kitaev (SYK) Holographic Quantum Chaos & Maldacena-Shenker-Stanford Bound $\lambda_L \le 2 \pi k_B T / \hbar$ (Sachdev & Ye, 1993; Kitaev, 2015; Maldacena, Shenker & Stanford, 2016).
//!
//! ### 1. Maximal Quantum Lyapunov Exponent $\lambda_L = 2\pi T$ & Zero-Temperature Residual Entropy
//! In the large-$N$, strong-coupling ($\beta \mathcal{J} \gg 1$) limit of the $q$-body Majorana SYK model, out-of-time-order correlators (OTOCs) saturate the universal chaos bound:
//! $$\lambda_L^{\max}(T) = 2 \pi T \left(1 - O\!\left(\frac{1}{\beta \mathcal{J}}\right)\right), \qquad C_V(T) = \frac{4 \pi^2}{\alpha_S \mathcal{J}} N T$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Sachdev-Ye-Kitaev (SYK) & MSS Quantum Chaos Bound Engine.
#[derive(Debug, Clone, Copy)]
pub struct SachdevYeKitaevMaximalChaos {
    /// SYK disorder coupling scale $\mathcal{J} > 0$.
    pub coupling_j: f64,
}

impl Default for SachdevYeKitaevMaximalChaos {
    fn default() -> Self {
        Self { coupling_j: 10.0 }
    }
}

impl SachdevYeKitaevMaximalChaos {
    /// Creates a new Sachdev-Ye-Kitaev Maximal Chaos engine.
    pub fn new(coupling_j: f64) -> Result<Self> {
        if coupling_j <= 0.0 || !coupling_j.is_finite() {
            return Err(Error::Msg("coupling_j must be > 0".to_string()));
        }
        Ok(Self { coupling_j })
    }

    /// Evaluates the Maldacena-Shenker-Stanford chaos bound $\lambda_{\mathrm{MSS}} = 2\pi T$ and leading strong-coupling SYK Lyapunov exponent $\lambda_{\mathrm{SYK}}(T) = 2\pi T (1 - 4.28 / (\beta \mathcal{J}))$.
    pub fn mss_bound_and_syk_lyapunov(&self, temperature_t: f64) -> Result<(f64, f64)> {
        if temperature_t <= 0.0 || temperature_t >= self.coupling_j / 5.0 {
            return Err(Error::Msg(
                "0 < temperature_t < coupling_j / 5 required for strong-coupling SYK".to_string(),
            ));
        }
        let mss = 2.0 * PI * temperature_t;
        let beta_j = self.coupling_j / temperature_t;
        let syk = mss * (1.0 - 4.28 / beta_j);
        Ok((mss, syk))
    }
}
