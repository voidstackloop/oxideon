//! Derjaguin-Landau-Verwey-Overbeek (DLVO) Colloidal Stability & Debye-Hückel Screening (Derjaguin & Landau, 1941; Verwey & Overbeek, 1948).
//!
//! ### 1. Electrostatic Double-Layer Repulsion vs. Hamaker Van der Waals Attraction
//! For two spheres of radius $R$ at surface separation $D > 0$ with inverse Debye screening length $\kappa$ and Hamaker constant $A_H$:
//! $$V_{\mathrm{DLVO}}(D) = R Z e^{-\kappa D} - \frac{A_H R}{12 D}$$

use crate::error::{Error, Result};

/// DLVO Colloidal Stability & Debye-Hückel Screening Engine.
#[derive(Debug, Clone, Copy)]
pub struct DlvoColloidalStabilityEngine {
    /// Inverse Debye screening length $\kappa > 0$ ($\mathrm{nm^{-1}}$).
    pub kappa_inv_nm: f64,
    /// Hamaker constant $A_H > 0$ (in $k_B T$ units).
    pub hamaker_ah_kt: f64,
}

impl Default for DlvoColloidalStabilityEngine {
    fn default() -> Self {
        Self {
            kappa_inv_nm: 0.5,
            hamaker_ah_kt: 12.0,
        }
    }
}

impl DlvoColloidalStabilityEngine {
    /// Creates a new DLVO Colloidal Stability engine.
    pub fn new(kappa_inv_nm: f64, hamaker_ah_kt: f64) -> Result<Self> {
        if kappa_inv_nm <= 0.0 || hamaker_ah_kt <= 0.0 {
            return Err(Error::Msg(
                "kappa_inv_nm and hamaker_ah_kt must be > 0".to_string(),
            ));
        }
        Ok(Self {
            kappa_inv_nm,
            hamaker_ah_kt,
        })
    }

    /// Evaluates the Debye length $\lambda_D = \kappa^{-1}$ and total DLVO interaction energy $V_{\mathrm{DLVO}}(D) / k_B T$ at separation $D > 0$.
    pub fn debye_length_and_dlvo_potential(
        &self,
        radius_r_nm: f64,
        electro_prefactor_z: f64,
        separation_d_nm: f64,
    ) -> Result<(f64, f64)> {
        if radius_r_nm <= 0.0 || electro_prefactor_z < 0.0 || separation_d_nm <= 0.0 {
            return Err(Error::Msg(
                "radius_r_nm > 0, electro_prefactor_z >= 0, separation_d_nm > 0 required"
                    .to_string(),
            ));
        }
        let debye_l = 1.0 / self.kappa_inv_nm;
        let v_rep =
            radius_r_nm * electro_prefactor_z * (-self.kappa_inv_nm * separation_d_nm).exp();
        let v_att = -self.hamaker_ah_kt * radius_r_nm / (12.0 * separation_d_nm);
        Ok((debye_l, v_rep + v_att))
    }
}
