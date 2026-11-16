//! Safronov-Toomre Protoplanetary Disk Gravitational Instability Parameter $Q$ & Hill Sphere Radius (Safronov, 1960; Toomre, 1964).
//!
//! ### 1. Toomre Stability Parameter $Q$ & Planetary Hill Radius $R_H$
//! A differentially rotating gas disk with sound speed $c_s$, epicyclic frequency $\kappa$, and surface density $\Sigma$ is gravitationally stable iff:
//! $$Q = \frac{c_s \kappa}{\pi G \Sigma} > 1, \qquad R_H = a \left(\frac{M_p}{3 M_*}\right)^{1/3}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Safronov-Toomre Protoplanetary Disk & Hill Sphere Engine.
#[derive(Debug, Clone, Copy)]
pub struct ToomreHillPlanetaryDisk {
    /// Gravitational constant $G > 0$.
    pub grav_g: f64,
}

impl Default for ToomreHillPlanetaryDisk {
    fn default() -> Self {
        Self { grav_g: 1.0 }
    }
}

impl ToomreHillPlanetaryDisk {
    /// Creates a new Toomre-Hill Planetary Disk engine.
    pub fn new(grav_g: f64) -> Result<Self> {
        if grav_g <= 0.0 || !grav_g.is_finite() {
            return Err(Error::Msg("grav_g must be > 0".to_string()));
        }
        Ok(Self { grav_g })
    }

    /// Evaluates the Toomre parameter $Q = c_s \kappa / (\pi G \Sigma)$, stability check ($Q > 1$), and Hill sphere radius $R_H = a (m_p / (3 M_*))^{1/3}$.
    pub fn toomre_q_and_hill_radius(
        &self,
        sound_speed_cs: f64,
        epicyclic_kappa: f64,
        surface_density_sigma: f64,
        semi_major_a: f64,
        mass_ratio_mp_mstar: f64,
    ) -> Result<(f64, bool, f64)> {
        if sound_speed_cs <= 0.0
            || epicyclic_kappa <= 0.0
            || surface_density_sigma <= 0.0
            || semi_major_a <= 0.0
            || mass_ratio_mp_mstar <= 0.0
        {
            return Err(Error::Msg(
                "All disk and orbital parameters must be > 0".to_string(),
            ));
        }
        let q = sound_speed_cs * epicyclic_kappa / (PI * self.grav_g * surface_density_sigma);
        let r_h = semi_major_a * (mass_ratio_mp_mstar / 3.0).cbrt();
        Ok((q, q > 1.0, r_h))
    }
}
