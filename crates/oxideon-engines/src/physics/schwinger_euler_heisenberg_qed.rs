//! Schwinger Strong-Field QED Vacuum Pair Production & Euler-Heisenberg Effective Lagrangian (Heisenberg & Euler, 1936; Schwinger, 1951).
//!
//! ### 1. Schwinger Vacuum Pair Production Rate per Unit Volume
//! Above the critical field $E_{\mathrm{cr}} = m_e^2 c^3 / (e \hbar)$, the non-perturbative $e^+ e^-$ pair creation rate is:
//! $$\Gamma_{\mathrm{Schwinger}}(E) = \frac{(e E)^2}{4 \pi^3} \sum_{n=1}^{\infty} \frac{1}{n^2} \exp\!\left(-n \pi \frac{E_{\mathrm{cr}}}{E}\right)$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Schwinger Strong-Field QED Pair Production & Euler-Heisenberg Engine.
#[derive(Debug, Clone, Copy)]
pub struct SchwingerEulerHeisenbergQED {
    /// Critical electric field $E_{\mathrm{cr}} > 0$ in natural units.
    pub critical_field_ecr: f64,
}

impl Default for SchwingerEulerHeisenbergQED {
    fn default() -> Self {
        Self {
            critical_field_ecr: 1.0,
        }
    }
}

impl SchwingerEulerHeisenbergQED {
    /// Creates a new Schwinger Strong-Field QED engine.
    pub fn new(critical_field_ecr: f64) -> Result<Self> {
        if critical_field_ecr <= 0.0 || !critical_field_ecr.is_finite() {
            return Err(Error::Msg("critical_field_ecr must be > 0".to_string()));
        }
        Ok(Self { critical_field_ecr })
    }

    /// Evaluates the Schwinger vacuum pair-production rate $\Gamma(E)$ truncated at $N$ instanton harmonics.
    pub fn pair_production_rate(&self, electric_field_e: f64, harmonics_n: usize) -> Result<f64> {
        if electric_field_e <= 0.0 || harmonics_n == 0 {
            return Err(Error::Msg(
                "electric_field_e > 0 and harmonics_n >= 1 required".to_string(),
            ));
        }
        crate::limits::check_iterations("harmonics_n", harmonics_n)?;
        let pre = (electric_field_e * electric_field_e) / (4.0 * PI.powi(3));
        let mut sum = 0.0;
        for n in 1..=harmonics_n {
            let nf = n as f64;
            sum += (-nf * PI * self.critical_field_ecr / electric_field_e).exp() / (nf * nf);
        }
        Ok(pre * sum)
    }
}
