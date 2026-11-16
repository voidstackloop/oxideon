//! Adler-Bell-Jackiw (ABJ) Chiral Axial Anomaly, Triangle Diagram & $\pi^0 \to \gamma \gamma$ Decay Rate (Adler, 1969; Bell & Jackiw, 1969).
//!
//! ### 1. Chiral Anomaly & Neutral Pion Two-Photon Width
//! The anomalous divergence $\partial_\mu J_5^\mu = \frac{e^2 N_c}{16 \pi^2} \varepsilon^{\mu \nu \rho \sigma} F_{\mu \nu} F_{\rho \sigma}$ determines the $\pi^0 \to \gamma\gamma$ width:
//! $$\Gamma(\pi^0 \to \gamma \gamma) = \frac{\alpha_{\mathrm{em}}^2 m_\pi^3}{64 \pi^3 f_\pi^2} \left(\frac{N_c}{3}\right)^2$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Adler-Bell-Jackiw Chiral Axial Anomaly & $\pi^0 \to \gamma \gamma$ Engine.
#[derive(Debug, Clone, Copy)]
pub struct AbjChiralAnomalyEngine {
    /// Number of QCD colors $N_c \ge 1$.
    pub num_colors_nc: usize,
    /// Fine-structure constant $\alpha_{\mathrm{em}}$.
    pub alpha_em: f64,
}

impl Default for AbjChiralAnomalyEngine {
    fn default() -> Self {
        Self {
            num_colors_nc: 3,
            alpha_em: 1.0 / 137.035999,
        }
    }
}

impl AbjChiralAnomalyEngine {
    /// Creates a new ABJ Chiral Anomaly engine.
    pub fn new(num_colors_nc: usize, alpha_em: f64) -> Result<Self> {
        if num_colors_nc == 0 || alpha_em <= 0.0 || !alpha_em.is_finite() {
            return Err(Error::Msg(
                "num_colors_nc >= 1 and alpha_em > 0 required".to_string(),
            ));
        }
        Ok(Self {
            num_colors_nc,
            alpha_em,
        })
    }

    /// Evaluates the anomalous $\pi^0 \to \gamma \gamma$ decay width $\Gamma$ (in MeV) given pion mass $m_\pi$ and decay constant $f_\pi$ (in MeV).
    pub fn pion_two_photon_width(&self, m_pi_mev: f64, f_pi_mev: f64) -> Result<f64> {
        if m_pi_mev <= 0.0 || f_pi_mev <= 0.0 {
            return Err(Error::Msg("m_pi and f_pi must be > 0".to_string()));
        }
        let color_factor = (self.num_colors_nc as f64) / 3.0;
        let width = self.alpha_em.powi(2) * m_pi_mev.powi(3)
            / (64.0 * PI.powi(3) * f_pi_mev.powi(2))
            * color_factor.powi(2);
        Ok(width)
    }
}
