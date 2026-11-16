//! Lasry-Lions & Huang-Malhamé-Caines Mean-Field Game (MFG) HJB-Fokker-Planck Coupled System (Lasry & Lions, 2006; Huang, Malhamé & Caines, 2006).
//!
//! ### 1. Stationary Linear-Quadratic Mean-Field Equilibrium Riccati Gain
//! In a continuum population where each agent minimizes $\mathbb{E}\int_0^\infty e^{-\rho t} [\frac{q}{2}(X_t - m_t)^2 + \frac{r}{2} u_t^2] dt$ subject to $dX_t = u_t dt + \sigma dW_t$, the stationary mean-field HJB Riccati equation is:
//! $$\rho \Pi = q - \frac{\Pi^2}{r} \implies \Pi^* = \frac{-r \rho + \sqrt{r^2 \rho^2 + 4 q r}}{2}, \qquad u^*(x, m) = -\frac{\Pi^*}{r}(x - m)$$

use crate::error::{Error, Result};

/// Lasry-Lions Stationary Linear-Quadratic Mean-Field Game Engine.
#[derive(Debug, Clone, Copy)]
pub struct LasryLionsLinearQuadraticMfg {
    /// Discount rate $\rho \ge 0$.
    pub discount_rho: f64,
    /// Mean-tracking penalty $q > 0$.
    pub state_cost_q: f64,
    /// Control effort weight $r > 0$.
    pub control_cost_r: f64,
}

impl Default for LasryLionsLinearQuadraticMfg {
    fn default() -> Self {
        Self {
            discount_rho: 0.1,
            state_cost_q: 4.0,
            control_cost_r: 1.0,
        }
    }
}

impl LasryLionsLinearQuadraticMfg {
    /// Creates a new Lasry-Lions Mean-Field Game engine.
    pub fn new(discount_rho: f64, state_cost_q: f64, control_cost_r: f64) -> Result<Self> {
        if discount_rho < 0.0 || state_cost_q <= 0.0 || control_cost_r <= 0.0 {
            return Err(Error::Msg(
                "discount_rho >= 0, state_cost_q > 0, control_cost_r > 0 required".to_string(),
            ));
        }
        Ok(Self {
            discount_rho,
            state_cost_q,
            control_cost_r,
        })
    }

    /// Evaluates the stationary HJB Riccati coefficient $\Pi^*$ and stationary Fokker-Planck variance $\sigma^2 / (2 \Pi^* / r)$.
    pub fn equilibrium_riccati_and_stationary_variance(
        &self,
        volatility_sigma: f64,
    ) -> Result<(f64, f64)> {
        if volatility_sigma <= 0.0 || !volatility_sigma.is_finite() {
            return Err(Error::Msg("volatility_sigma must be > 0".to_string()));
        }
        let r = self.control_cost_r;
        let rp = r * self.discount_rho;
        let pi_star = 0.5 * (-rp + (rp * rp + 4.0 * self.state_cost_q * r).sqrt());
        let feedback_gain = pi_star / r;
        let var = (volatility_sigma * volatility_sigma) / (2.0 * feedback_gain);
        Ok((pi_star, var))
    }
}
