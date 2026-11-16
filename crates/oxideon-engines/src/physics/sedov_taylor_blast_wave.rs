//! Sedov-Taylor Self-Similar Point-Explosion Blast Wave $R(t) = \xi_0 (E_0 t^2 / \rho_0)^{1/5}$ (Taylor, 1950; Sedov, 1959).
//!
//! ### 1. Five-Fifths Blast Scaling & Shock Velocity $U_s(t) = \frac{2}{5} \frac{R(t)}{t}$
//! For an intense point release of energy $E_0$ into an ambient medium of density $\rho_0$:
//! $$R(t) = \xi_0 \left(\frac{E_0 t^2}{\rho_0}\right)^{1/5}, \qquad U_s(t) = \frac{d R}{d t} = \frac{2 R(t)}{5 t}$$

use crate::error::{Error, Result};

/// Sedov-Taylor Self-Similar Blast Wave Engine.
#[derive(Debug, Clone, Copy)]
pub struct SedovTaylorBlastWave {
    /// Dimensionless similarity constant $\xi_0 \approx 1.15$ (for $\gamma = 1.4$).
    pub similarity_xi0: f64,
    /// Ambient medium density $\rho_0 > 0$ ($\mathrm{kg/m^3}$).
    pub ambient_rho0: f64,
}

impl Default for SedovTaylorBlastWave {
    fn default() -> Self {
        Self {
            similarity_xi0: 1.15,
            ambient_rho0: 1.225,
        }
    }
}

impl SedovTaylorBlastWave {
    /// Creates a new Sedov-Taylor Blast Wave engine.
    pub fn new(similarity_xi0: f64, ambient_rho0: f64) -> Result<Self> {
        if similarity_xi0 <= 0.0 || ambient_rho0 <= 0.0 {
            return Err(Error::Msg(
                "similarity_xi0 and ambient_rho0 must be > 0".to_string(),
            ));
        }
        Ok(Self {
            similarity_xi0,
            ambient_rho0,
        })
    }

    /// Evaluates the blast radius $R(t)$ and shock front velocity $U_s(t) = 2 R(t) / (5 t)$.
    pub fn blast_radius_and_shock_velocity(
        &self,
        energy_e0: f64,
        time_t: f64,
    ) -> Result<(f64, f64)> {
        if energy_e0 <= 0.0 || time_t <= 0.0 {
            return Err(Error::Msg("energy_e0 and time_t must be > 0".to_string()));
        }
        let r_t = self.similarity_xi0 * (energy_e0 * time_t * time_t / self.ambient_rho0).powf(0.2);
        let u_s = 0.4 * r_t / time_t;
        Ok((r_t, u_s))
    }
}
