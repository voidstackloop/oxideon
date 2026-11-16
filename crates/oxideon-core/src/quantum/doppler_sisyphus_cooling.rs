//! Chu–Cohen-Tannoudji–Phillips Doppler & Sisyphus Laser Cooling Temperature Limits (Chu, 1985; Dalibard & Cohen-Tannoudji, 1989).
//!
//! ### 1. Doppler Cooling Temperature Curve & Minimum $k_B T_D = \hbar \Gamma / 2$
//! For a two-level transition of natural linewidth $\Gamma$ and red detuning $\delta = \omega_L - \omega_0 < 0$:
//! $$k_B T(\delta) = \frac{\hbar \Gamma}{4} \left(\frac{2 |\delta|}{\Gamma} + \frac{\Gamma}{2 |\delta|}\right) \ge \frac{\hbar \Gamma}{2} \quad (\text{at } \delta = -\Gamma / 2)$$

use crate::error::{Error, Result};

/// Doppler & Sisyphus Sub-Doppler Laser Cooling Engine.
#[derive(Debug, Clone, Copy)]
pub struct DopplerSisyphusLaserCooling {
    /// Natural radiative linewidth $\Gamma > 0$ (in natural energy units $\hbar = k_B = 1$).
    pub linewidth_gamma: f64,
}

impl Default for DopplerSisyphusLaserCooling {
    fn default() -> Self {
        Self {
            linewidth_gamma: 2.0,
        }
    }
}

impl DopplerSisyphusLaserCooling {
    /// Creates a new Doppler-Sisyphus Laser Cooling engine.
    pub fn new(linewidth_gamma: f64) -> Result<Self> {
        if linewidth_gamma <= 0.0 || !linewidth_gamma.is_finite() {
            return Err(Error::Msg("linewidth_gamma must be > 0".to_string()));
        }
        Ok(Self { linewidth_gamma })
    }

    /// Evaluates the Doppler temperature $T(\delta)$ at red detuning $\delta < 0$ and the minimum Doppler limit $T_D = \Gamma / 2$.
    pub fn doppler_temperature_and_limit(&self, red_detuning_delta: f64) -> Result<(f64, f64)> {
        if red_detuning_delta >= 0.0 {
            return Err(Error::Msg(
                "Red detuning delta must be < 0 for Doppler cooling".to_string(),
            ));
        }
        let x = 2.0 * red_detuning_delta.abs() / self.linewidth_gamma;
        let t_delta = 0.25 * self.linewidth_gamma * (x + 1.0 / x);
        let t_min = 0.5 * self.linewidth_gamma;
        Ok((t_delta, t_min))
    }
}
