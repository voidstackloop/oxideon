//! Kelly Criterion Log-Optimal Portfolio Growth & Thorp Continuous-Time Merton-Kelly Fraction (Kelly, 1956; Thorp, 1969).
//!
//! ### 1. Continuous-Time Kelly Fraction & Fractional Kelly Growth Rate
//! For an asset with excess drift $\mu - r$ and variance $\sigma^2$, the full Kelly leverage is $f^* = \frac{\mu - r}{\sigma^2}$.
//! Investing a fraction $c \in [0, 2]$ of full Kelly ($f = c f^*$) achieves long-run geometric excess growth rate:
//! $$g(c) = c (2 - c) \frac{(\mu - r)^2}{2 \sigma^2}$$

use crate::error::{Error, Result};

/// Kelly-Thorp Log-Optimal Growth & Fractional Drawdown Engine.
#[derive(Debug, Clone, Copy)]
pub struct KellyLogOptimalPortfolio {
    /// Excess return $\mu - r$.
    pub excess_return: f64,
    /// Return volatility $\sigma > 0$.
    pub volatility_sigma: f64,
}

impl Default for KellyLogOptimalPortfolio {
    fn default() -> Self {
        Self {
            excess_return: 0.08,
            volatility_sigma: 0.2,
        }
    }
}

impl KellyLogOptimalPortfolio {
    /// Creates a new Kelly Log-Optimal Portfolio engine.
    pub fn new(excess_return: f64, volatility_sigma: f64) -> Result<Self> {
        if volatility_sigma <= 0.0 || !volatility_sigma.is_finite() {
            return Err(Error::Msg("volatility_sigma must be > 0".to_string()));
        }
        Ok(Self {
            excess_return,
            volatility_sigma,
        })
    }

    /// Evaluates the full Kelly leverage $f^* = (\mu - r)/\sigma^2$ and fractional Kelly geometric excess growth rate $g(c)$ for multiplier $c \ge 0$ (e.g. $c = 0.5$ half-Kelly achieves $75\%$ of max growth with $25\%$ variance).
    pub fn kelly_leverage_and_growth_rate(&self, fraction_c: f64) -> Result<(f64, f64)> {
        if fraction_c < 0.0 {
            return Err(Error::Msg("fraction_c must be >= 0".to_string()));
        }
        let var = self.volatility_sigma * self.volatility_sigma;
        let f_star = self.excess_return / var;
        let max_g = 0.5 * self.excess_return * self.excess_return / var;
        let g_c = fraction_c * (2.0 - fraction_c) * max_g;
        Ok((f_star, g_c))
    }
}
