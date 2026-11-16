//! Fermi First-Order Diffusive Shock Acceleration (DSA) Spectral Index & Hillas Cosmic-Ray Confinement Limit (Fermi, 1949; Bell, 1978; Hillas, 1984).
//!
//! ### 1. Universal DSA Power-Law Spectral Index $s = (r + 2)/(r - 1)$
//! Across an astrophysical shock with compression ratio $r = u_1 / u_2 > 1$, diffusive shock acceleration produces a differential energy spectrum
//! $N(E) \propto E^{-s}$ with $s = \frac{r + 2}{r - 1}$ ($s = 2$ for a strong $\gamma = 5/3$ shock with $r = 4$), up to the Hillas gyroradius limit $E_{\mathrm{max}} = Z e B R u_s / c$.

use crate::error::{Error, Result};

/// Fermi Diffusive Shock Acceleration & Hillas Cosmic-Ray Confinement Engine.
#[derive(Debug, Clone, Copy)]
pub struct FermiHillasCosmicRayShock {
    /// Shock compression ratio $r = u_1 / u_2 > 1$.
    pub compression_ratio_r: f64,
}

impl Default for FermiHillasCosmicRayShock {
    fn default() -> Self {
        Self {
            compression_ratio_r: 4.0,
        }
    }
}

impl FermiHillasCosmicRayShock {
    /// Creates a new Fermi-Hillas Cosmic-Ray Shock engine.
    pub fn new(compression_ratio_r: f64) -> Result<Self> {
        if compression_ratio_r <= 1.0 || !compression_ratio_r.is_finite() {
            return Err(Error::Msg(
                "Shock compression_ratio_r must be > 1".to_string(),
            ));
        }
        Ok(Self {
            compression_ratio_r,
        })
    }

    /// Evaluates the differential energy spectral index $s = (r + 2)/(r - 1)$ and Hillas maximum energy $E_{\mathrm{max}}$ (in EeV $= 10^{18}\,\mathrm{eV}$) for charge $Z$, magnetic field $B$ ($\mu\mathrm{G}$), size $R$ (kpc), and shock beta $\beta_s = u_s / c$.
    pub fn spectral_index_and_hillas_emax_eev(
        &self,
        charge_z: usize,
        b_microgauss: f64,
        radius_kpc: f64,
        shock_beta: f64,
    ) -> Result<(f64, f64)> {
        if charge_z == 0
            || b_microgauss <= 0.0
            || radius_kpc <= 0.0
            || !(0.0..=1.0).contains(&shock_beta)
        {
            return Err(Error::Msg("Invalid Hillas parameters".to_string()));
        }
        let s_index = (self.compression_ratio_r + 2.0) / (self.compression_ratio_r - 1.0);
        // E_max(EeV) approx 0.925 * Z * B(uG) * R(kpc) * beta_s
        let e_max_eev = 0.925 * (charge_z as f64) * b_microgauss * radius_kpc * shock_beta;
        Ok((s_index, e_max_eev))
    }
}
