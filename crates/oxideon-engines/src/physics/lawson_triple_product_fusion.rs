//! Lawson Fusion Ignition Criterion & Fusion Triple Product $n T \tau_E$ (Lawson, 1957).
//!
//! ### 1. Self-Sustaining D-T Alpha-Heating Ignition Threshold
//! Balancing bremsstrahlung + thermal conduction losses $3 n k_B T / \tau_E$ against $3.5\,\mathrm{MeV}$ alpha-particle heating $\frac{1}{4} n^2 \langle \sigma v \rangle E_\alpha$ requires:
//! $$n T \tau_E \ge \frac{12 k_B T^2}{\langle \sigma v \rangle E_\alpha}$$

use crate::error::{Error, Result};

/// Lawson Fusion Triple Product & Alpha-Heating Ignition Engine.
#[derive(Debug, Clone, Copy)]
pub struct LawsonTripleProductIgnition {
    /// Alpha particle heating energy $E_\alpha > 0$ (keV, $3500$ keV for D-T).
    pub alpha_energy_kev: f64,
}

impl Default for LawsonTripleProductIgnition {
    fn default() -> Self {
        Self {
            alpha_energy_kev: 3500.0,
        }
    }
}

impl LawsonTripleProductIgnition {
    /// Creates a new Lawson Triple Product Ignition engine.
    pub fn new(alpha_energy_kev: f64) -> Result<Self> {
        if alpha_energy_kev <= 0.0 || !alpha_energy_kev.is_finite() {
            return Err(Error::Msg("alpha_energy_kev must be > 0".to_string()));
        }
        Ok(Self { alpha_energy_kev })
    }

    /// Evaluates achieved triple product $n T \tau_E$, required ignition threshold $12 T^2 / (\langle \sigma v\rangle E_\alpha)$, and fusion energy gain $Q_{\mathrm{sci}}$ check.
    pub fn triple_product_and_ignition(
        &self,
        density_n: f64,
        temp_kev: f64,
        tau_e_sec: f64,
        reactivity_sigma_v: f64,
    ) -> Result<(f64, f64, bool)> {
        if density_n <= 0.0 || temp_kev <= 0.0 || tau_e_sec <= 0.0 || reactivity_sigma_v <= 0.0 {
            return Err(Error::Msg(
                "All fusion plasma parameters must be > 0".to_string(),
            ));
        }
        let achieved = density_n * temp_kev * tau_e_sec;
        let threshold = 12.0 * temp_kev * temp_kev / (reactivity_sigma_v * self.alpha_energy_kev);
        Ok((achieved, threshold, achieved >= threshold))
    }
}
