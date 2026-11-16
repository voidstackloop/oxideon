//! Chandrasekhar Radiative Transfer & Eddington Two-Stream Grey Atmosphere Approximation (Eddington, 1916; Chandrasekhar, 1950).
//!
//! ### 1. Eddington Temperature Profile $T(\tau)$ in Radiative Equilibrium
//! In a plane-parallel grey atmosphere with effective temperature $T_{\mathrm{eff}}$ and optical depth $\tau \ge 0$:
//! $$T(\tau)^4 = \frac{3}{4} T_{\mathrm{eff}}^4 \left(\tau + \frac{2}{3}\right)$$
//! where $q(\infty) = 0.710446$ replaces $2/3$ in the exact Hopf-Bronstein-Chandrasekhar solution.

use crate::error::{Error, Result};

/// Chandrasekhar-Eddington Radiative Transfer & Hopf Function Engine.
#[derive(Debug, Clone, Copy)]
pub struct EddingtonChandrasekharRadiative {
    /// Effective radiative temperature $T_{\mathrm{eff}} > 0$ (K).
    pub t_eff_kelvin: f64,
}

impl Default for EddingtonChandrasekharRadiative {
    fn default() -> Self {
        Self {
            t_eff_kelvin: 5778.0,
        }
    }
}

impl EddingtonChandrasekharRadiative {
    /// Creates a new Eddington-Chandrasekhar Radiative engine.
    pub fn new(t_eff_kelvin: f64) -> Result<Self> {
        if t_eff_kelvin <= 0.0 || !t_eff_kelvin.is_finite() {
            return Err(Error::Msg("t_eff_kelvin must be > 0".to_string()));
        }
        Ok(Self { t_eff_kelvin })
    }

    /// Evaluates the Eddington temperature $T(\tau) = T_{\mathrm{eff}} [\frac{3}{4}(\tau + 2/3)]^{1/4}$ and conservative-scattering two-stream slab albedo $R(\tau) = \frac{\frac{3}{4}(1 - g) \tau}{1 + \frac{3}{4}(1 - g) \tau}$.
    pub fn eddington_temperature_and_slab_albedo(
        &self,
        optical_depth_tau: f64,
        asymmetry_g: f64,
    ) -> Result<(f64, f64)> {
        if optical_depth_tau < 0.0 || !(-1.0..1.0).contains(&asymmetry_g) {
            return Err(Error::Msg(
                "optical_depth_tau >= 0 and asymmetry_g in [-1, 1) required".to_string(),
            ));
        }
        let t_tau = self.t_eff_kelvin * (0.75 * (optical_depth_tau + 2.0 / 3.0)).powf(0.25);
        let gamma_tau = 0.75 * (1.0 - asymmetry_g) * optical_depth_tau;
        let albedo = gamma_tau / (1.0 + gamma_tau);
        Ok((t_tau, albedo))
    }
}
