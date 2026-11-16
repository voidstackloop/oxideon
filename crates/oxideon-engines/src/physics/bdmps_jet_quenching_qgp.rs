//! BDMPS-Z Radiative Parton Jet Quenching & Landau-Pomeranchuk-Migdal Interference in QGP (Baier, Dokshitzer, Mueller, Peigné & Schiff, 1997; Zakharov, 1996).
//!
//! ### 1. Coherent LPM Quadratic Path-Length Energy Loss $\Delta E_{\mathrm{BDMPS}} \propto \hat{q} L^2$
//! A hard parton of Casimir $C_R$ ($C_F = 4/3$ for quarks, $C_A = 3$ for gluons) traversing QGP medium of length $L$ and transport coefficient $\hat{q}$ loses radiative energy:
//! $$\Delta E_{\mathrm{BDMPS}} = \frac{\alpha_s C_R}{4} \hat{q} L^2, \qquad \omega_c = \frac{1}{2} \hat{q} L^2$$

use crate::error::{Error, Result};

/// BDMPS-Z Radiative Jet Quenching & Transport Coefficient $\hat{q}$ Engine.
#[derive(Debug, Clone, Copy)]
pub struct BdmpsJetQuenchingEnergyLoss {
    /// Strong coupling constant $\alpha_s > 0$.
    pub alpha_s: f64,
    /// Jet quenching transport coefficient $\hat{q} > 0$ ($\mathrm{GeV^2/fm}$).
    pub q_hat_gev2_fm: f64,
}

impl Default for BdmpsJetQuenchingEnergyLoss {
    fn default() -> Self {
        Self {
            alpha_s: 0.3,
            q_hat_gev2_fm: 1.5,
        }
    }
}

impl BdmpsJetQuenchingEnergyLoss {
    /// Creates a new BDMPS Jet Quenching engine.
    pub fn new(alpha_s: f64, q_hat_gev2_fm: f64) -> Result<Self> {
        if alpha_s <= 0.0 || q_hat_gev2_fm <= 0.0 {
            return Err(Error::Msg(
                "alpha_s and q_hat_gev2_fm must be > 0".to_string(),
            ));
        }
        Ok(Self {
            alpha_s,
            q_hat_gev2_fm,
        })
    }

    /// Evaluates the characteristic gluon frequency $\omega_c = \frac{1}{2} \hat{q} L^2 / (\hbar c)$ and BDMPS radiative energy loss $\Delta E$ (in GeV, using $\hbar c = 0.197327\,\mathrm{GeV \cdot fm}$).
    pub fn characteristic_frequency_and_energy_loss_gev(
        &self,
        path_length_fm: f64,
        casimir_cr: f64,
    ) -> Result<(f64, f64)> {
        if path_length_fm <= 0.0 || casimir_cr <= 0.0 {
            return Err(Error::Msg(
                "path_length_fm and casimir_cr must be > 0".to_string(),
            ));
        }
        let hbar_c = 0.19732698;
        let ql2_gev = self.q_hat_gev2_fm * path_length_fm.powi(2) / hbar_c;
        let omega_c_gev = 0.5 * ql2_gev;
        let delta_e_gev = 0.25 * self.alpha_s * casimir_cr * ql2_gev;
        Ok((omega_c_gev, delta_e_gev))
    }
}
