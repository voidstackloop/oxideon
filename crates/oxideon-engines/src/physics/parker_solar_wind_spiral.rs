//! Parker Isothermal Transonic Solar Wind Critical Point $r_c$ & Archimedean Spiral Angle $\psi(r)$ (Parker, 1958).
//!
//! ### 1. Sonic Critical Radius $r_c = G M_\star / (2 c_s^2)$ & Parker Spiral Garden-Hose Angle
//! In an isothermal corona with sound speed $c_s$ around a star of mass $M_\star$ and rotation rate $\Omega_\star$,
//! the wind passes smoothly from subsonic to supersonic at $r_c = \frac{G M_\star}{2 c_s^2}$, winding the heliospheric magnetic field at angle $\tan \psi(r) = \frac{\Omega_\star r \sin\theta}{v_w}$.

use crate::error::{Error, Result};

/// Parker Transonic Solar Wind & Heliospheric Spiral Engine.
#[derive(Debug, Clone, Copy)]
pub struct ParkerSolarWindSpiralEngine {
    /// Gravitational parameter $\mu = G M_\star > 0$.
    pub gm_star: f64,
    /// Stellar angular rotation rate $\Omega_\star > 0$.
    pub omega_star: f64,
}

impl Default for ParkerSolarWindSpiralEngine {
    fn default() -> Self {
        Self {
            gm_star: 2.0,
            omega_star: 1.0,
        }
    }
}

impl ParkerSolarWindSpiralEngine {
    /// Creates a new Parker Solar Wind Spiral engine.
    pub fn new(gm_star: f64, omega_star: f64) -> Result<Self> {
        if gm_star <= 0.0 || omega_star <= 0.0 {
            return Err(Error::Msg("gm_star and omega_star must be > 0".to_string()));
        }
        Ok(Self {
            gm_star,
            omega_star,
        })
    }

    /// Evaluates the sonic critical radius $r_c = G M_\star / (2 c_s^2)$ and equatorial Parker spiral pitch angle $\psi = \arctan(\Omega_\star r / v_w)$ in radians.
    pub fn sonic_radius_and_spiral_angle(
        &self,
        sound_speed_cs: f64,
        radius_r: f64,
        wind_speed_vw: f64,
    ) -> Result<(f64, f64)> {
        if sound_speed_cs <= 0.0 || radius_r <= 0.0 || wind_speed_vw <= 0.0 {
            return Err(Error::Msg(
                "sound_speed_cs, radius_r, and wind_speed_vw must be > 0".to_string(),
            ));
        }
        let r_c = self.gm_star / (2.0 * sound_speed_cs * sound_speed_cs);
        let psi = (self.omega_star * radius_r / wind_speed_vw).atan();
        Ok((r_c, psi))
    }
}
