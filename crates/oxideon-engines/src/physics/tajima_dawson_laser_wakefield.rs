//! Tajima-Dawson Relativistic Laser Wakefield Acceleration (LWFA) & Cold Non-Relativistic Wavebreaking Field (Tajima & Dawson, 1979).
//!
//! ### 1. Cold Relativistic Wavebreaking Field $E_0$ & Dephasing Energy Gain
//! An ultrashort laser pulse with normalized vector potential $a_0$ propagating through underdense plasma of Lorentz factor $\gamma_p = \omega_0 / \omega_p \gg 1$
//! excites a plasma wakefield of characteristic field $E_0 \approx 96 \sqrt{n_e(\mathrm{cm^{-3}})} \, \mathrm{V/m}$ and maximum electron energy gain $\Delta W \approx 2 \gamma_p^2 m_e c^2$.

use crate::error::{Error, Result};

/// Tajima-Dawson Laser Wakefield Accelerator (LWFA) Engine.
#[derive(Debug, Clone, Copy)]
pub struct TajimaDawsonLaserWakefield {
    /// Plasma electron density $n_e$ in units of $10^{18}\,\mathrm{cm^{-3}}$.
    pub density_1e18_cm3: f64,
}

impl Default for TajimaDawsonLaserWakefield {
    fn default() -> Self {
        Self {
            density_1e18_cm3: 1.0,
        }
    }
}

impl TajimaDawsonLaserWakefield {
    /// Creates a new Tajima-Dawson Laser Wakefield engine.
    pub fn new(density_1e18_cm3: f64) -> Result<Self> {
        if density_1e18_cm3 <= 0.0 || !density_1e18_cm3.is_finite() {
            return Err(Error::Msg("density_1e18_cm3 must be > 0".to_string()));
        }
        Ok(Self { density_1e18_cm3 })
    }

    /// Evaluates the cold wavebreaking field $E_0 \approx 96 \sqrt{n_{18}} \, \mathrm{GV/m}$ and linear dephasing energy gain $\Delta W = 2 \gamma_p^2 m_e c^2$ (in MeV, with $m_e c^2 = 0.511\,\mathrm{MeV}$).
    pub fn wavebreaking_gv_m_and_energy_gain_mev(&self, gamma_p: f64) -> Result<(f64, f64)> {
        if gamma_p <= 1.0 {
            return Err(Error::Msg(
                "Wakefield phase Lorentz factor gamma_p must be > 1".to_string(),
            ));
        }
        let e0_gv_m = 96.0 * self.density_1e18_cm3.sqrt();
        let dw_mev = 2.0 * 0.51099895 * gamma_p * gamma_p;
        Ok((e0_gv_m, dw_mev))
    }
}
