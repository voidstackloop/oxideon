//! Avellaneda-Stoikov High-Frequency Market Making Reservation Price & Optimal Spread (Avellaneda & Stoikov, 2008).
//!
//! ### 1. Inventory Reservation Indifference Price $r(s, q, t)$ & Optimal Bid-Ask Spread
//! For mid-price $s$, inventory $q$, risk aversion $\gamma > 0$, volatility $\sigma$, remaining horizon $\tau = T - t$, and order-book intensity decay $k > 0$:
//! $$r(s, q, \tau) = s - q \gamma \sigma^2 \tau, \qquad \delta^a + \delta^b = \gamma \sigma^2 \tau + \frac{2}{\gamma} \ln\!\left(1 + \frac{\gamma}{k}\right)$$

use crate::error::{Error, Result};

/// Avellaneda-Stoikov Limit Order Book Market-Making Engine.
#[derive(Debug, Clone, Copy)]
pub struct AvellanedaStoikovMarketMaker {
    /// Inventory CARA risk aversion $\gamma > 0$.
    pub gamma: f64,
    /// Mid-price volatility $\sigma > 0$.
    pub sigma: f64,
    /// Order book liquidity exponential decay $k > 0$.
    pub kappa_k: f64,
}

impl Default for AvellanedaStoikovMarketMaker {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            sigma: 2.0,
            kappa_k: 1.5,
        }
    }
}

impl AvellanedaStoikovMarketMaker {
    /// Creates a new Avellaneda-Stoikov Market Maker engine.
    pub fn new(gamma: f64, sigma: f64, kappa_k: f64) -> Result<Self> {
        if gamma <= 0.0 || sigma <= 0.0 || kappa_k <= 0.0 {
            return Err(Error::Msg(
                "gamma, sigma, and kappa_k must be > 0".to_string(),
            ));
        }
        Ok(Self {
            gamma,
            sigma,
            kappa_k,
        })
    }

    /// Evaluates the inventory reservation price $r(s, q, \tau)$ and optimal total bid-ask spread $\delta^a + \delta^b$.
    pub fn reservation_price_and_spread(
        &self,
        mid_price_s: f64,
        inventory_q: f64,
        remaining_tau: f64,
    ) -> Result<(f64, f64)> {
        if remaining_tau < 0.0 {
            return Err(Error::Msg("remaining_tau must be >= 0".to_string()));
        }
        let risk_term = self.gamma * self.sigma * self.sigma * remaining_tau;
        let r_price = mid_price_s - inventory_q * risk_term;
        let spread = risk_term + (2.0 / self.gamma) * (1.0 + self.gamma / self.kappa_k).ln();
        Ok((r_price, spread))
    }
}
