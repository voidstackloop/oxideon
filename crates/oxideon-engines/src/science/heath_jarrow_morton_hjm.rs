//! Heath-Jarrow-Morton (HJM) Forward-Rate No-Arbitrage Drift Condition (Heath, Jarrow & Morton, 1992).
//!
//! ### 1. HJM Risk-Neutral No-Arbitrage Drift Identity
//! Under the risk-neutral measure $\mathbb{Q}$, the instantaneous forward rate $f(t, T)$ with volatility $\sigma(t, T)$ has drift uniquely determined by:
//! $$\alpha(t, T) = \sigma(t, T) \int_t^T \sigma(t, u) \, du$$
//! For the Ho-Lee constant volatility $\sigma_0$, $\alpha(t, T) = \sigma_0^2 (T - t)$; for Hull-White $\sigma(t, T) = \sigma_0 e^{-\kappa (T - t)}$, $\alpha(t, T) = \frac{\sigma_0^2}{\kappa} e^{-\kappa \tau}(1 - e^{-\kappa \tau})$.

use crate::error::{Error, Result};

/// Heath-Jarrow-Morton (HJM) Arbitrage-Free Forward Curve Engine.
#[derive(Debug, Clone, Copy)]
pub struct HeathJarrowMortonTermStructure {
    /// Base forward-rate volatility $\sigma_0 > 0$.
    pub sigma0: f64,
    /// Mean-reversion rate $\kappa > 0$.
    pub kappa: f64,
}

impl Default for HeathJarrowMortonTermStructure {
    fn default() -> Self {
        Self {
            sigma0: 0.01,
            kappa: 0.1,
        }
    }
}

impl HeathJarrowMortonTermStructure {
    /// Creates a new Heath-Jarrow-Morton Term Structure engine.
    pub fn new(sigma0: f64, kappa: f64) -> Result<Self> {
        if sigma0 <= 0.0 || kappa <= 0.0 {
            return Err(Error::Msg("sigma0 and kappa must be > 0".to_string()));
        }
        Ok(Self { sigma0, kappa })
    }

    /// Evaluates the exact HJM arbitrage-free drift $\alpha(t, T)$ and bond volatility $B(t, T) \sigma_0$ at time-to-maturity $\tau = T - t \ge 0$.
    pub fn no_arbitrage_drift_and_bond_vol(&self, maturity_tau: f64) -> Result<(f64, f64)> {
        if maturity_tau < 0.0 {
            return Err(Error::Msg("maturity_tau must be >= 0".to_string()));
        }
        let exp_k = (-self.kappa * maturity_tau).exp();
        let bond_vol = self.sigma0 * (1.0 - exp_k) / self.kappa;
        let alpha_drift = (self.sigma0 * exp_k) * bond_vol;
        Ok((alpha_drift, bond_vol))
    }
}
