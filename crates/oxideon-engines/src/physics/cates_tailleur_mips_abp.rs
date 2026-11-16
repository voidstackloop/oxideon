//! Cates-Tailleur Motility-Induced Phase Separation (MIPS) in Active Brownian Particles (Tailleur & Cates, 2008; Cates & Tailleur, 2015).
//!
//! ### 1. Density-Dependent Swim Speed $v(\rho)$ & MIPS Spinodal Criterion
//! For repulsive Active Brownian Particles whose effective propulsion speed decreases with crowding as $v(\rho) = v_0 (1 - \rho / \rho_{\mathrm{jam}})$,
//! the effective chemical potential derivative vanishes and triggers spinodal MIPS when $\frac{d}{d\rho}[\rho v(\rho)] < 0$, i.e., $\rho > \rho_{\mathrm{jam}} / 2$ at high Péclet number $\mathrm{Pe} = v_0 / (\sigma D_r)$.

use crate::error::{Error, Result};

/// Cates-Tailleur Motility-Induced Phase Separation (MIPS) Engine.
#[derive(Debug, Clone, Copy)]
pub struct MotilityInducedPhaseSeparation {
    /// Bare self-propulsion speed $v_0 > 0$.
    pub bare_speed_v0: f64,
    /// Jamming close-packing density $\rho_{\mathrm{jam}} > 0$.
    pub rho_jam: f64,
}

impl Default for MotilityInducedPhaseSeparation {
    fn default() -> Self {
        Self {
            bare_speed_v0: 10.0,
            rho_jam: 1.0,
        }
    }
}

impl MotilityInducedPhaseSeparation {
    /// Creates a new Cates-Tailleur MIPS engine.
    pub fn new(bare_speed_v0: f64, rho_jam: f64) -> Result<Self> {
        if bare_speed_v0 <= 0.0 || rho_jam <= 0.0 {
            return Err(Error::Msg(
                "bare_speed_v0 and rho_jam must be > 0".to_string(),
            ));
        }
        Ok(Self {
            bare_speed_v0,
            rho_jam,
        })
    }

    /// Evaluates the local swim speed $v(\rho)$, effective flux derivative $\frac{d}{d\rho}[\rho v(\rho)] = v_0 (1 - 2\rho/\rho_{\mathrm{jam}})$, and spinodal MIPS instability flag.
    pub fn swim_speed_and_mips_spinodal(&self, density_rho: f64) -> Result<(f64, f64, bool)> {
        if !(0.0..=self.rho_jam).contains(&density_rho) {
            return Err(Error::Msg(
                "density_rho must be in [0, rho_jam]".to_string(),
            ));
        }
        let v_rho = self.bare_speed_v0 * (1.0 - density_rho / self.rho_jam);
        let d_flux = self.bare_speed_v0 * (1.0 - 2.0 * density_rho / self.rho_jam);
        Ok((v_rho, d_flux, d_flux < 0.0))
    }
}
