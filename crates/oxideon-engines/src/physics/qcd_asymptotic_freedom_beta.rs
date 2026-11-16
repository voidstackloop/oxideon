//! Gross-Wilczek & Politzer Asymptotic Freedom, One-Loop QCD $\beta$-Function & Running Coupling $\alpha_s(Q^2)$ (Gross & Wilczek, 1973; Politzer, 1973).
//!
//! ### 1. One-Loop QCD Running Coupling
//! For $SU(N_c)$ gauge theory with $N_f$ quark flavors, the one-loop beta coefficient is $b_0 = \frac{11 N_c - 2 N_f}{12 \pi}$,
//! and asymptotic freedom ($b_0 > 0$) holds for $N_f < \frac{11}{2} N_c$:
//! $$\alpha_s(Q^2) = \frac{1}{b_0 \ln(Q^2 / \Lambda_{\mathrm{QCD}}^2)}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Gross-Wilczek-Politzer Asymptotic Freedom & Running $\alpha_s(Q^2)$ Engine.
#[derive(Debug, Clone, Copy)]
pub struct QcdAsymptoticFreedomEngine {
    /// Number of colors $N_c \ge 2$.
    pub num_colors_nc: usize,
    /// Number of active quark flavors $N_f$.
    pub num_flavors_nf: usize,
    /// QCD scale $\Lambda_{\mathrm{QCD}} > 0$ (GeV).
    pub lambda_qcd: f64,
}

impl Default for QcdAsymptoticFreedomEngine {
    fn default() -> Self {
        Self {
            num_colors_nc: 3,
            num_flavors_nf: 5,
            lambda_qcd: 0.217,
        }
    }
}

impl QcdAsymptoticFreedomEngine {
    /// Creates a new QCD Asymptotic Freedom engine.
    pub fn new(num_colors_nc: usize, num_flavors_nf: usize, lambda_qcd: f64) -> Result<Self> {
        if num_colors_nc < 2 || lambda_qcd <= 0.0 || !lambda_qcd.is_finite() {
            return Err(Error::Msg("Invalid QCD parameters".to_string()));
        }
        Ok(Self {
            num_colors_nc,
            num_flavors_nf,
            lambda_qcd,
        })
    }

    /// Evaluates the one-loop coefficient $\beta_0 = 11 N_c / 3 - 2 N_f / 3$, asymptotic freedom check, and $\alpha_s(Q)$ at scale $Q > \Lambda_{\mathrm{QCD}}$.
    pub fn running_alpha_s(&self, scale_q: f64) -> Result<(f64, f64, bool)> {
        if scale_q <= self.lambda_qcd {
            return Err(Error::Msg(
                "Perturbative scale Q must exceed lambda_qcd".to_string(),
            ));
        }
        let beta0 = (11.0 * (self.num_colors_nc as f64) - 2.0 * (self.num_flavors_nf as f64)) / 3.0;
        let is_af = beta0 > 0.0;
        if !is_af {
            return Err(Error::Msg(
                "Theory is not asymptotically free for given N_f".to_string(),
            ));
        }
        let b0 = beta0 / (4.0 * PI);
        let alpha_s = 1.0 / (b0 * (scale_q * scale_q / (self.lambda_qcd * self.lambda_qcd)).ln());
        Ok((beta0, alpha_s, is_af))
    }
}
