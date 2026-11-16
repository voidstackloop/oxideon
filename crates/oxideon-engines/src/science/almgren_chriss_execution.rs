//! Almgren-Chriss Optimal Portfolio Execution with Temporary & Permanent Market Impact (Almgren & Chriss, 2001).
//!
//! ### 1. Hyperbolic Inventory Liquidation Trajectory & Urgency Parameter $\kappa$
//! Minimizing $\mathbb{E}[\text{Cost}] + \lambda \mathrm{Var}(\text{Cost})$ for volatility $\sigma$, temporary impact $\eta$, and risk aversion $\lambda$ yields urgency
//! $\kappa = \sqrt{\lambda \sigma^2 / \eta}$ and the continuous-time inventory curve over horizon $[0, T]$:
//! $$x(t) = X_0 \frac{\sinh(\kappa (T - t))}{\sinh(\kappa T)}$$

use crate::error::{Error, Result};

/// Almgren-Chriss Optimal Liquidation & Market Impact Engine.
#[derive(Debug, Clone, Copy)]
pub struct AlmgrenChrissOptimalExecution {
    /// Asset volatility $\sigma > 0$.
    pub sigma: f64,
    /// Temporary market impact coefficient $\eta > 0$.
    pub temp_impact_eta: f64,
    /// Mean-variance risk aversion $\lambda > 0$.
    pub risk_aversion_lambda: f64,
}

impl Default for AlmgrenChrissOptimalExecution {
    fn default() -> Self {
        Self {
            sigma: 0.02,
            temp_impact_eta: 1e-4,
            risk_aversion_lambda: 1e-2,
        }
    }
}

impl AlmgrenChrissOptimalExecution {
    /// Creates a new Almgren-Chriss Optimal Execution engine.
    pub fn new(sigma: f64, temp_impact_eta: f64, risk_aversion_lambda: f64) -> Result<Self> {
        if sigma <= 0.0 || temp_impact_eta <= 0.0 || risk_aversion_lambda <= 0.0 {
            return Err(Error::Msg(
                "sigma, temp_impact_eta, and risk_aversion_lambda must be > 0".to_string(),
            ));
        }
        Ok(Self {
            sigma,
            temp_impact_eta,
            risk_aversion_lambda,
        })
    }

    /// Evaluates the urgency parameter $\kappa = \sqrt{\lambda \sigma^2 / \eta}$ and remaining inventory fraction $x(t)/X_0$ at time $t \in [0, T]$.
    pub fn urgency_and_inventory_fraction(
        &self,
        time_t: f64,
        horizon_t: f64,
    ) -> Result<(f64, f64)> {
        if horizon_t <= 0.0 || !(0.0..=horizon_t).contains(&time_t) {
            return Err(Error::Msg(
                "0 <= time_t <= horizon_t and horizon_t > 0 required".to_string(),
            ));
        }
        let kappa =
            (self.risk_aversion_lambda * self.sigma * self.sigma / self.temp_impact_eta).sqrt();
        let frac = (kappa * (horizon_t - time_t)).sinh() / (kappa * horizon_t).sinh();
        Ok((kappa, frac))
    }
}
