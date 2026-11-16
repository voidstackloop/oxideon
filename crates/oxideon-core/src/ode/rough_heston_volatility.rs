//! Gatheral-Jaisson-Rosenbaum Rough Volatility ($H \in (0, 1/2)$), Volterra Kernel & Power-Law ATM Skew (Gatheral, Jaisson & Rosenbaum, 2018).
//!
//! ### 1. Fractional Volterra Kernel $K_H(t) = \frac{t^{H - 1/2}}{\Gamma(H + 1/2)}$ & ATM Implied Volatility Skew
//! Empirical high-frequency log-volatility behaves like fractional Brownian motion with Hurst exponent $H \approx 0.1 < 0.5$,
//! producing the universal short-maturity **Power-Law ATM Implied Volatility Skew Explosion**:
//! $$\mathcal{S}(T) = \left|\frac{\partial \sigma_{\text{BS}}(k, T)}{\partial k}\right|_{k=0} \propto T^{H - 1/2}$$

use crate::error::{Error, Result};

/// Rough Volatility ($H \in (0, 0.5)$) & Power-Law ATM Skew Engine.
#[derive(Debug, Clone, Copy)]
pub struct RoughVolatilityHestonEngine {
    /// Hurst roughness exponent $H \in (0, 0.5)$.
    pub hurst_h: f64,
    /// Vol-of-vol parameter $\nu > 0$.
    pub vol_of_vol_nu: f64,
    /// Spot-vol correlation $\rho \in [-1, 1]$.
    pub correlation_rho: f64,
}

impl RoughVolatilityHestonEngine {
    /// Creates a new Rough Volatility engine.
    pub fn new(hurst_h: f64, vol_of_vol_nu: f64, correlation_rho: f64) -> Result<Self> {
        if !(0.0..0.5).contains(&hurst_h)
            || vol_of_vol_nu <= 0.0
            || !(-1.0..=1.0).contains(&correlation_rho)
        {
            return Err(Error::Msg(
                "Requires H in (0, 0.5), vol_of_vol_nu > 0, and correlation_rho in [-1, 1]"
                    .to_string(),
            ));
        }
        Ok(Self {
            hurst_h,
            vol_of_vol_nu,
            correlation_rho,
        })
    }

    /// Evaluates the asymptotic Rough Bergomi / Rough Heston ATM implied volatility skew $\mathcal{S}(T) = \frac{\rho \nu}{2} T^{H - 1/2}$ at maturity $T > 0$.
    pub fn atm_implied_vol_skew(&self, maturity_t: f64) -> Result<f64> {
        if maturity_t <= 0.0 {
            return Err(Error::Msg("maturity_t must be > 0".to_string()));
        }
        Ok(0.5 * self.correlation_rho * self.vol_of_vol_nu * maturity_t.powf(self.hurst_h - 0.5))
    }
}
