//! Merton Structural Credit Risk Model, Distance-to-Default & Credit Spread (Merton, 1974).
//!
//! ### 1. Distance-to-Default $DD$ under Asset Geometric Brownian Motion
//! Treating firm equity as a European call option on firm asset value $V_0$ with zero-coupon debt face value $D$ maturing at $T$:
//! $$DD = \frac{\ln(V_0 / D) + (r - \frac{1}{2} \sigma_V^2) T}{\sigma_V \sqrt{T}}$$

use crate::error::{Error, Result};

/// Merton Structural Credit Default & Distance-to-Default Engine.
#[derive(Debug, Clone, Copy)]
pub struct MertonStructuralDefaultEngine {
    /// Firm asset volatility $\sigma_V > 0$.
    pub asset_vol_sigma: f64,
    /// Risk-free rate $r$.
    pub risk_free_r: f64,
}

impl Default for MertonStructuralDefaultEngine {
    fn default() -> Self {
        Self {
            asset_vol_sigma: 0.25,
            risk_free_r: 0.05,
        }
    }
}

impl MertonStructuralDefaultEngine {
    /// Creates a new Merton Structural Default engine.
    pub fn new(asset_vol_sigma: f64, risk_free_r: f64) -> Result<Self> {
        if asset_vol_sigma <= 0.0 || !asset_vol_sigma.is_finite() {
            return Err(Error::Msg("asset_vol_sigma must be > 0".to_string()));
        }
        Ok(Self {
            asset_vol_sigma,
            risk_free_r,
        })
    }

    /// Evaluates Merton's Distance-to-Default $DD = d_2$ and $d_1 = d_2 + \sigma_V \sqrt{T}$ for asset value $V_0$, debt $D$, and maturity $T > 0$.
    pub fn distance_to_default(
        &self,
        asset_v0: f64,
        debt_d: f64,
        maturity_t: f64,
    ) -> Result<(f64, f64)> {
        if asset_v0 <= 0.0 || debt_d <= 0.0 || maturity_t <= 0.0 {
            return Err(Error::Msg(
                "asset_v0, debt_d, and maturity_t must be > 0".to_string(),
            ));
        }
        let vol_sqrt_t = self.asset_vol_sigma * maturity_t.sqrt();
        let d1 = ((asset_v0 / debt_d).ln()
            + (self.risk_free_r + 0.5 * self.asset_vol_sigma.powi(2)) * maturity_t)
            / vol_sqrt_t;
        let d2 = d1 - vol_sqrt_t;
        Ok((d2, d1))
    }
}
