//! Bell-Evans Force-Induced Receptor-Ligand Unbinding & Dynamic Force Spectroscopy Rupture Force (Bell, 1978; Evans & Ritchie, 1997).
//!
//! ### 1. Bell Slip-Bond Off-Rate & Most Probable Rupture Force $F^*$
//! Under pulling force $F$ along transition-state distance $x^\ddagger$, the off-rate increases exponentially as $k_{\mathrm{off}}(F) = k_0 \exp(F / F_\beta)$ where $F_\beta = k_B T / x^\ddagger$.
//! At constant loading rate $r_f = dF/dt$, Evans & Ritchie derived the most probable rupture force:
//! $$F^*(r_f) = F_\beta \ln\!\left(\frac{r_f}{k_0 F_\beta}\right)$$

use crate::error::{Error, Result};

/// Bell-Evans Mechanobiology & Dynamic Force Spectroscopy Engine.
#[derive(Debug, Clone, Copy)]
pub struct BellEvansMechanobiologyBond {
    /// Zero-force dissociation rate $k_0 > 0$ ($\mathrm{s^{-1}}$).
    pub k0: f64,
    /// Thermal force scale $F_\beta = k_B T / x^\ddagger > 0$ (pN).
    pub thermal_force_f_beta: f64,
}

impl Default for BellEvansMechanobiologyBond {
    fn default() -> Self {
        Self {
            k0: 0.1,
            thermal_force_f_beta: 10.0,
        }
    }
}

impl BellEvansMechanobiologyBond {
    /// Creates a new Bell-Evans Mechanobiology Bond engine.
    pub fn new(k0: f64, thermal_force_f_beta: f64) -> Result<Self> {
        if k0 <= 0.0 || thermal_force_f_beta <= 0.0 {
            return Err(Error::Msg(
                "k0 and thermal_force_f_beta must be > 0".to_string(),
            ));
        }
        Ok(Self {
            k0,
            thermal_force_f_beta,
        })
    }

    /// Evaluates the Bell off-rate $k_{\mathrm{off}}(F)$ at force $F \ge 0$ and the Evans-Ritchie most probable rupture force $F^*(r_f)$ at loading rate $r_f > k_0 F_\beta$.
    pub fn off_rate_and_rupture_force(
        &self,
        force_pn: f64,
        loading_rate_rf: f64,
    ) -> Result<(f64, f64)> {
        if force_pn < 0.0 || loading_rate_rf <= self.k0 * self.thermal_force_f_beta {
            return Err(Error::Msg(
                "force_pn >= 0 and loading_rate_rf > k0 * F_beta required".to_string(),
            ));
        }
        let k_off = self.k0 * (force_pn / self.thermal_force_f_beta).exp();
        let f_star = self.thermal_force_f_beta
            * (loading_rate_rf / (self.k0 * self.thermal_force_f_beta)).ln();
        Ok((k_off, f_star))
    }
}
