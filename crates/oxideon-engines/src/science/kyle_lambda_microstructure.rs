//! Kyle Insider Auction Equilibrium, Kyle's Lambda $\lambda$ & Market Depth (Kyle, 1985).
//!
//! ### 1. Linear Bayesian Nash Equilibrium of Informed Trading
//! Given fundamental liquidation value $v \sim \mathcal{N}(p_0, \Sigma_0)$ and noise trader order flow $u \sim \mathcal{N}(0, \sigma_u^2)$:
//! $$\lambda = \frac{\sqrt{\Sigma_0}}{2 \sigma_u}, \qquad \beta = \frac{\sigma_u}{\sqrt{\Sigma_0}}, \qquad \Sigma_1 = \frac{1}{2} \Sigma_0$$

use crate::error::{Error, Result};

/// Kyle Continuous Auction & Market Depth $\lambda^{-1}$ Engine.
#[derive(Debug, Clone, Copy)]
pub struct KyleLambdaAuctionEquilibrium {
    /// Prior fundamental variance $\Sigma_0 > 0$.
    pub fundamental_var_sigma0: f64,
    /// Noise trader order-flow standard deviation $\sigma_u > 0$.
    pub noise_std_sigma_u: f64,
}

impl Default for KyleLambdaAuctionEquilibrium {
    fn default() -> Self {
        Self {
            fundamental_var_sigma0: 4.0,
            noise_std_sigma_u: 10.0,
        }
    }
}

impl KyleLambdaAuctionEquilibrium {
    /// Creates a new Kyle Lambda Auction Equilibrium engine.
    pub fn new(fundamental_var_sigma0: f64, noise_std_sigma_u: f64) -> Result<Self> {
        if fundamental_var_sigma0 <= 0.0 || noise_std_sigma_u <= 0.0 {
            return Err(Error::Msg(
                "fundamental_var_sigma0 and noise_std_sigma_u must be > 0".to_string(),
            ));
        }
        Ok(Self {
            fundamental_var_sigma0,
            noise_std_sigma_u,
        })
    }

    /// Evaluates Kyle's price-impact $\lambda = \sqrt{\Sigma_0} / (2 \sigma_u)$, insider trading intensity $\beta = \sigma_u / \sqrt{\Sigma_0}$, and expected insider profit $\frac{1}{2} \sqrt{\Sigma_0} \sigma_u$.
    pub fn lambda_beta_and_insider_profit(&self) -> (f64, f64, f64) {
        let s0_sqrt = self.fundamental_var_sigma0.sqrt();
        let lambda = s0_sqrt / (2.0 * self.noise_std_sigma_u);
        let beta = self.noise_std_sigma_u / s0_sqrt;
        let expected_profit = 0.5 * s0_sqrt * self.noise_std_sigma_u;
        (lambda, beta, expected_profit)
    }
}
