//! Kovtun-Son-Starinets (KSS) Holographic Shear-Viscosity-to-Entropy Bound $\eta / s \ge 1 / (4\pi)$ (Kovtun, Son & Starinets, 2005).
//!
//! ### 1. Universal AdS Black-Brane Horizon Absorption & Finite-$\lambda$ Stringy Correction
//! In strongly coupled $\mathcal{N}=4$ SYM dual to AdS$_5 \times S^5$ with 't Hooft coupling $\lambda = g_{\mathrm{YM}}^2 N_c \gg 1$:
//! $$\frac{\eta}{s} = \frac{1}{4 \pi} \left[1 + \frac{15 \zeta(3)}{\lambda^{3/2}} + \dots\right] \ge \frac{1}{4 \pi}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Kovtun-Son-Starinets (KSS) Holographic $\eta / s$ Bound Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct KovtunSonStarinetsViscosityBound;

impl KovtunSonStarinetsViscosityBound {
    /// Creates a new KSS Viscosity Bound engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the KSS lower bound $1 / (4\pi)$ and the $\alpha'^3$ stringy finite-coupling value $\eta / s(\lambda)$ for 't Hooft coupling $\lambda > 0$.
    pub fn kss_bound_and_finite_coupling_eta_over_s(thooft_lambda: f64) -> Result<(f64, f64)> {
        if thooft_lambda <= 0.0 || !thooft_lambda.is_finite() {
            return Err(Error::Msg("thooft_lambda must be > 0".to_string()));
        }
        let kss = 1.0 / (4.0 * PI);
        let zeta3 = 1.2020569031595942;
        let eta_s = kss * (1.0 + 15.0 * zeta3 / thooft_lambda.powf(1.5));
        Ok((kss, eta_s))
    }
}
