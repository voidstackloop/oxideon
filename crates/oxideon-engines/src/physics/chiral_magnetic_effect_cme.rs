//! Chiral Magnetic Effect (CME) & Chiral Vortical Effect in Topological QCD Matter (Fukushima, Kharzeev & Warringa, 2008).
//!
//! ### 1. Anomaly-Induced Chiral Magnetic Current $\mathbf{J}_{\mathrm{CME}}$
//! In the presence of a chiral chemical potential $\mu_5 = (\mu_R - \mu_L)/2$ induced by topological $SU(3)_c$ sphaleron transitions and an external magnetic field $\mathbf{B}$,
//! the lowest Landau level generates the dissipationless vector current:
//! $$\mathbf{J}_{\mathrm{CME}} = \frac{N_c \sum_f q_f^2}{2 \pi^2} \mu_5 \mathbf{B} = \sigma_\chi \mathbf{B}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Fukushima-Kharzeev-Warringa Chiral Magnetic Effect (CME) Engine.
#[derive(Debug, Clone, Copy)]
pub struct ChiralMagneticEffectKharzeev {
    /// Sum of squared electric charges $N_c \sum_f (q_f / e)^2$ (equals $3 \times (4/9 + 1/9 + 1/9) = 2$ for $u, d, s$).
    pub charge_factor_nc_qf2: f64,
}

impl Default for ChiralMagneticEffectKharzeev {
    fn default() -> Self {
        Self {
            charge_factor_nc_qf2: 2.0,
        }
    }
}

impl ChiralMagneticEffectKharzeev {
    /// Creates a new Chiral Magnetic Effect engine.
    pub fn new(charge_factor_nc_qf2: f64) -> Result<Self> {
        if charge_factor_nc_qf2 <= 0.0 || !charge_factor_nc_qf2.is_finite() {
            return Err(Error::Msg("charge_factor_nc_qf2 must be > 0".to_string()));
        }
        Ok(Self {
            charge_factor_nc_qf2,
        })
    }

    /// Evaluates the chiral magnetic conductivity $\sigma_\chi = \frac{N_c \sum q_f^2}{2 \pi^2} \mu_5$ and CME current density $J_{\mathrm{CME}} = \sigma_\chi B$ (in natural units $e = \hbar = c = 1$).
    pub fn chiral_conductivity_and_current(&self, mu5_gev: f64, b_field_gev2: f64) -> (f64, f64) {
        let sigma_chi = self.charge_factor_nc_qf2 * mu5_gev / (2.0 * PI * PI);
        let j_cme = sigma_chi * b_field_gev2;
        (sigma_chi, j_cme)
    }
}
