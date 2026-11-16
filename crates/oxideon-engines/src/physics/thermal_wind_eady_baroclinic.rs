//! Thermal Wind Balance & Eady Baroclinic Instability Maximum Growth Rate (Eady, 1949).
//!
//! ### 1. Eady Baroclinic Growth Rate & Rossby Deformation Wavelength
//! In a stratified rotating atmosphere with Coriolis parameter $f$, Brunt-Väisälä frequency $N$, and vertical wind shear $\Lambda = \partial U / \partial z$:
//! $$\sigma_{\mathrm{Eady,max}} = 0.3098 \frac{|f| |\Lambda|}{N}, \qquad L_R = \frac{N H}{|f|}$$

use crate::error::{Error, Result};

/// Thermal Wind Balance & Eady Baroclinic Instability Engine.
#[derive(Debug, Clone, Copy)]
pub struct ThermalWindEadyBaroclinic {
    /// Coriolis parameter $f > 0$ ($\mathrm{s^{-1}}$).
    pub coriolis_f: f64,
    /// Brunt-Väisälä buoyancy frequency $N > 0$ ($\mathrm{s^{-1}}$).
    pub brunt_vaisala_n: f64,
}

impl Default for ThermalWindEadyBaroclinic {
    fn default() -> Self {
        Self {
            coriolis_f: 1.0e-4,
            brunt_vaisala_n: 1.0e-2,
        }
    }
}

impl ThermalWindEadyBaroclinic {
    /// Creates a new Thermal Wind & Eady Baroclinic engine.
    pub fn new(coriolis_f: f64, brunt_vaisala_n: f64) -> Result<Self> {
        if coriolis_f <= 0.0 || brunt_vaisala_n <= 0.0 {
            return Err(Error::Msg(
                "coriolis_f and brunt_vaisala_n must be > 0".to_string(),
            ));
        }
        Ok(Self {
            coriolis_f,
            brunt_vaisala_n,
        })
    }

    /// Evaluates the maximum Eady baroclinic growth rate $\sigma_{\mathrm{max}} = 0.3098 f |\Lambda| / N$ and internal Rossby radius $L_R = N H / f$.
    pub fn eady_growth_rate_and_rossby_radius(
        &self,
        vertical_shear_lambda: f64,
        tropopause_height_h: f64,
    ) -> Result<(f64, f64)> {
        if tropopause_height_h <= 0.0 {
            return Err(Error::Msg("tropopause_height_h must be > 0".to_string()));
        }
        let sigma_max =
            0.3098 * self.coriolis_f * vertical_shear_lambda.abs() / self.brunt_vaisala_n;
        let l_r = self.brunt_vaisala_n * tropopause_height_h / self.coriolis_f;
        Ok((sigma_max, l_r))
    }
}
