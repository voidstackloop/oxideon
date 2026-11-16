//! Budyko-Sellers Energy Balance Climate Model, Ice-Albedo Feedback & Milankovitch Insolation (Budyko, 1969; Sellers, 1969).
//!
//! ### 1. Zero-Dimensional Energy Balance with Temperature-Dependent Albedo
//! Global mean surface temperature $T$ evolves via $C \frac{dT}{dt} = \frac{S_0}{4}(1 - \alpha(T)) - (A + B T)$,
//! where step/ramp albedo $\alpha(T) \in [\alpha_{\mathrm{warm}}, \alpha_{\mathrm{ice}}]$ creates a saddle-node Snowball Earth bifurcation.

use crate::error::{Error, Result};

/// Budyko-Sellers Ice-Albedo Climate Bifurcation & Milankovitch Engine.
#[derive(Debug, Clone, Copy)]
pub struct BudykoSellersClimateBifurcation {
    /// Outgoing longwave radiation intercept $A$ ($\mathrm{W/m^2}$).
    pub olr_a: f64,
    /// Outgoing longwave radiation climate feedback slope $B > 0$ ($\mathrm{W/(m^2 \cdot {}^\circ C)}$).
    pub olr_b: f64,
}

impl Default for BudykoSellersClimateBifurcation {
    fn default() -> Self {
        Self {
            olr_a: 202.0,
            olr_b: 1.9,
        }
    }
}

impl BudykoSellersClimateBifurcation {
    /// Creates a new Budyko-Sellers Climate Bifurcation engine.
    pub fn new(olr_a: f64, olr_b: f64) -> Result<Self> {
        if olr_b <= 0.0 || !olr_b.is_finite() {
            return Err(Error::Msg(
                "Climate feedback slope olr_b must be > 0".to_string(),
            ));
        }
        Ok(Self { olr_a, olr_b })
    }

    /// Evaluates the warm and Snowball-Earth equilibrium temperatures $(T_{\mathrm{warm}}, T_{\mathrm{ice}})$ in ${}^\circ\mathrm{C}$ for solar constant $S_0$ and albedos $(\alpha_w, \alpha_i)$.
    pub fn warm_and_snowball_equilibria(
        &self,
        solar_s0: f64,
        alpha_warm: f64,
        alpha_ice: f64,
    ) -> Result<(f64, f64)> {
        if solar_s0 <= 0.0
            || !(0.0..1.0).contains(&alpha_warm)
            || !(0.0..1.0).contains(&alpha_ice)
            || alpha_ice <= alpha_warm
        {
            return Err(Error::Msg(
                "Invalid solar constant or ice/warm albedos".to_string(),
            ));
        }
        let q = 0.25 * solar_s0;
        let t_warm = (q * (1.0 - alpha_warm) - self.olr_a) / self.olr_b;
        let t_ice = (q * (1.0 - alpha_ice) - self.olr_a) / self.olr_b;
        Ok((t_warm, t_ice))
    }
}
