//! Bjorken Boost-Invariant Relativistic Hydrodynamic Expansion of Quark-Gluon Plasma (Bjorken, 1983).
//!
//! ### 1. Longitudinal Proper-Time Cooling $\varepsilon(\tau) = \varepsilon_0 (\tau_0 / \tau)^{1 + c_s^2}$
//! For longitudinal boost-invariant flow $v_z = z / t$ in proper time $\tau = \sqrt{t^2 - z^2}$ with equation of state $p = c_s^2 \varepsilon$
//! ($c_s^2 = 1/3$ for an ideal ultrarelativistic QGP):
//! $$\varepsilon(\tau) = \varepsilon_0 \left(\frac{\tau_0}{\tau}\right)^{1 + c_s^2}, \qquad T(\tau) = T_0 \left(\frac{\tau_0}{\tau}\right)^{c_s^2}$$

use crate::error::{Error, Result};

/// Bjorken Boost-Invariant QGP Relativistic Hydrodynamics Engine.
#[derive(Debug, Clone, Copy)]
pub struct BjorkenQgpHydroExpansion {
    /// Initial proper thermalization time $\tau_0 > 0$ ($\mathrm{fm}/c$).
    pub tau0_fm: f64,
    /// Initial energy density $\varepsilon_0 > 0$ ($\mathrm{GeV/fm^3}$).
    pub eps0_gev_fm3: f64,
}

impl Default for BjorkenQgpHydroExpansion {
    fn default() -> Self {
        Self {
            tau0_fm: 0.6,
            eps0_gev_fm3: 15.0,
        }
    }
}

impl BjorkenQgpHydroExpansion {
    /// Creates a new Bjorken QGP Hydro Expansion engine.
    pub fn new(tau0_fm: f64, eps0_gev_fm3: f64) -> Result<Self> {
        if tau0_fm <= 0.0 || eps0_gev_fm3 <= 0.0 {
            return Err(Error::Msg(
                "tau0_fm and eps0_gev_fm3 must be > 0".to_string(),
            ));
        }
        Ok(Self {
            tau0_fm,
            eps0_gev_fm3,
        })
    }

    /// Evaluates the energy density $\varepsilon(\tau)$ and temperature ratio $T(\tau)/T_0$ at proper time $\tau \ge \tau_0$ for squared speed of sound $c_s^2 \in (0, 1]$.
    pub fn energy_density_and_temp_ratio(&self, tau_fm: f64, cs_sq: f64) -> Result<(f64, f64)> {
        if tau_fm < self.tau0_fm || !(0.0..=1.0).contains(&cs_sq) || cs_sq == 0.0 {
            return Err(Error::Msg(
                "tau_fm >= tau0_fm and cs_sq in (0, 1] required".to_string(),
            ));
        }
        let ratio = self.tau0_fm / tau_fm;
        let eps = self.eps0_gev_fm3 * ratio.powf(1.0 + cs_sq);
        let t_ratio = ratio.powf(cs_sq);
        Ok((eps, t_ratio))
    }
}
