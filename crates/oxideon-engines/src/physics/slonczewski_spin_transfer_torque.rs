//! Slonczewski-Berger Spin-Transfer Torque (STT) & Landau-Lifshitz-Gilbert Critical Switching Current (Slonczewski, 1996; Berger, 1996).
//!
//! ### 1. Critical Spin-Torque Switching Current $I_{c0}$
//! For a nanomagnet of saturation magnetization $M_s$, volume $V$, uniaxial anisotropy field $H_k$, demagnetizing field $H_d$,
//! Gilbert damping $\alpha_G$, and spin polarization efficiency $\eta_s$:
//! $$I_{c0} = \frac{2 e \alpha_G M_s V}{\hbar \eta_s} \left(H_k + \frac{H_d}{2}\right)$$

use crate::error::{Error, Result};

/// Slonczewski-Berger Spin-Transfer Torque (STT-MRAM) Engine.
#[derive(Debug, Clone, Copy)]
pub struct SlonczewskiSpinTransferTorque {
    /// Gilbert damping constant $\alpha_G > 0$.
    pub gilbert_alpha: f64,
    /// Spin-polarization torque efficiency $\eta_s \in (0, 1]$.
    pub spin_efficiency_eta: f64,
}

impl Default for SlonczewskiSpinTransferTorque {
    fn default() -> Self {
        Self {
            gilbert_alpha: 0.01,
            spin_efficiency_eta: 0.5,
        }
    }
}

impl SlonczewskiSpinTransferTorque {
    /// Creates a new Slonczewski Spin-Transfer Torque engine.
    pub fn new(gilbert_alpha: f64, spin_efficiency_eta: f64) -> Result<Self> {
        if gilbert_alpha <= 0.0
            || !(0.0..=1.0).contains(&spin_efficiency_eta)
            || spin_efficiency_eta == 0.0
        {
            return Err(Error::Msg(
                "gilbert_alpha > 0 and spin_efficiency_eta in (0, 1] required".to_string(),
            ));
        }
        Ok(Self {
            gilbert_alpha,
            spin_efficiency_eta,
        })
    }

    /// Evaluates the dimensionless critical switching current $I_{c0}$ (in natural units $2e / \hbar = 1$) and thermal stability factor $\Delta = M_s V H_k / (2 k_B T)$.
    pub fn critical_current_and_thermal_stability(
        &self,
        ms_volume: f64,
        hk: f64,
        hd: f64,
        k_b_t: f64,
    ) -> Result<(f64, f64)> {
        if ms_volume <= 0.0 || hk <= 0.0 || hd < 0.0 || k_b_t <= 0.0 {
            return Err(Error::Msg(
                "ms_volume, hk, and k_b_t must be > 0, hd >= 0".to_string(),
            ));
        }
        let i_c0 = (self.gilbert_alpha * ms_volume / self.spin_efficiency_eta) * (hk + 0.5 * hd);
        let delta_thermal = 0.5 * ms_volume * hk / k_b_t;
        Ok((i_c0, delta_thermal))
    }
}
