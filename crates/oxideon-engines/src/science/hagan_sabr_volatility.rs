//! Hagan SABR (Stochastic Alpha-Beta-Rho) Implied Volatility Smile Asymptotics (Hagan, Kumar, Lesniewski & Woodward, 2002).
//!
//! ### 1. At-The-Money (ATM) SABR Implied Black Volatility
//! For forward $F > 0$, initial volatility $\alpha > 0$, CEV exponent $\beta \in [0, 1]$, correlation $\rho \in (-1, 1)$, and vol-of-vol $\nu \ge 0$:
//! $$\sigma_{\mathrm{ATM}}(F, T) = \frac{\alpha}{F^{1 - \beta}} \left[1 + \left(\frac{(1 - \beta)^2 \alpha^2}{24 F^{2 - 2\beta}} + \frac{\rho \beta \nu \alpha}{4 F^{1 - \beta}} + \frac{2 - 3 \rho^2}{24} \nu^2\right) T\right]$$

use crate::error::{Error, Result};

/// Hagan SABR Stochastic Volatility Smile Engine.
#[derive(Debug, Clone, Copy)]
pub struct HaganSabrVolatilityEngine {
    /// Initial volatility $\alpha > 0$.
    pub alpha: f64,
    /// Backbone CEV exponent $\beta \in [0, 1]$.
    pub beta: f64,
    /// Spot-vol correlation $\rho \in (-1, 1)$.
    pub rho: f64,
    /// Volatility of volatility $\nu \ge 0$.
    pub nu: f64,
}

impl Default for HaganSabrVolatilityEngine {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            beta: 0.5,
            rho: -0.3,
            nu: 0.4,
        }
    }
}

impl HaganSabrVolatilityEngine {
    /// Creates a new Hagan SABR Volatility engine.
    pub fn new(alpha: f64, beta: f64, rho: f64, nu: f64) -> Result<Self> {
        if alpha <= 0.0 || !(0.0..=1.0).contains(&beta) || rho.abs() >= 1.0 || nu < 0.0 {
            return Err(Error::Msg(
                "Invalid SABR parameters (alpha, beta, rho, nu)".to_string(),
            ));
        }
        Ok(Self {
            alpha,
            beta,
            rho,
            nu,
        })
    }

    /// Evaluates the Hagan asymptotic ATM Black implied volatility $\sigma_{\mathrm{ATM}}(F, T)$.
    pub fn atm_implied_volatility(&self, forward_f: f64, maturity_t: f64) -> Result<f64> {
        if forward_f <= 0.0 || maturity_t < 0.0 {
            return Err(Error::Msg(
                "forward_f > 0 and maturity_t >= 0 required".to_string(),
            ));
        }
        let one_m_b = 1.0 - self.beta;
        let f_pow = forward_f.powf(one_m_b);
        let term1 = (one_m_b * self.alpha).powi(2) / (24.0 * f_pow * f_pow);
        let term2 = 0.25 * self.rho * self.beta * self.nu * self.alpha / f_pow;
        let term3 = (2.0 - 3.0 * self.rho * self.rho) * self.nu * self.nu / 24.0;
        Ok((self.alpha / f_pow) * (1.0 + (term1 + term2 + term3) * maturity_t))
    }
}
