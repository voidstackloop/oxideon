//! Rankine-Hugoniot Normal Shock Jump Relations & Prandtl-Meyer Supersonic Expansion (Rankine, 1870; Hugoniot, 1889; Meyer, 1908).
//!
//! ### 1. Normal Shock Pressure & Density Jump Across Upstream Mach $M_1 > 1$
//! For an ideal calorically perfect gas with adiabatic index $\gamma > 1$:
//! $$\frac{p_2}{p_1} = 1 + \frac{2 \gamma}{\gamma + 1}(M_1^2 - 1), \qquad \frac{\rho_2}{\rho_1} = \frac{(\gamma + 1) M_1^2}{(\gamma - 1) M_1^2 + 2}, \qquad M_2^2 = \frac{1 + \frac{\gamma - 1}{2} M_1^2}{\gamma M_1^2 - \frac{\gamma - 1}{2}}$$

use crate::error::{Error, Result};

/// Rankine-Hugoniot Shock Adiabat & Supersonic Jump Engine.
#[derive(Debug, Clone, Copy)]
pub struct RankineHugoniotShockAdiabat {
    /// Adiabatic heat capacity ratio $\gamma > 1$ ($1.4$ for diatomic air).
    pub gamma_adiabatic: f64,
}

impl Default for RankineHugoniotShockAdiabat {
    fn default() -> Self {
        Self {
            gamma_adiabatic: 1.4,
        }
    }
}

impl RankineHugoniotShockAdiabat {
    /// Creates a new Rankine-Hugoniot Shock Adiabat engine.
    pub fn new(gamma_adiabatic: f64) -> Result<Self> {
        if gamma_adiabatic <= 1.0 || !gamma_adiabatic.is_finite() {
            return Err(Error::Msg("gamma_adiabatic must be > 1.0".to_string()));
        }
        Ok(Self { gamma_adiabatic })
    }

    /// Evaluates the pressure jump $p_2/p_1$, density compression $\rho_2/\rho_1$, and downstream Mach number $M_2 < 1$ for upstream $M_1 > 1$.
    pub fn shock_pressure_density_and_downstream_mach(
        &self,
        mach1: f64,
    ) -> Result<(f64, f64, f64)> {
        if mach1 <= 1.0 || !mach1.is_finite() {
            return Err(Error::Msg("mach1 must be > 1.0".to_string()));
        }
        let g = self.gamma_adiabatic;
        let m1_sq = mach1 * mach1;
        let p_ratio = 1.0 + (2.0 * g / (g + 1.0)) * (m1_sq - 1.0);
        let rho_ratio = ((g + 1.0) * m1_sq) / ((g - 1.0) * m1_sq + 2.0);
        let m2_sq = (1.0 + 0.5 * (g - 1.0) * m1_sq) / (g * m1_sq - 0.5 * (g - 1.0));
        Ok((p_ratio, rho_ratio, m2_sq.sqrt()))
    }
}
