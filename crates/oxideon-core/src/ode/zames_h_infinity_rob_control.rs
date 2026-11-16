//! Zames $H_\infty$ Optimal Robust Control, Small-Gain Theorem & Game-Theoretic Riccati Attenuation $\gamma$ (Zames, 1981; Doyle, Glover, Khargonekar & Francis, 1989).
//!
//! ### 1. DGKF Game-Theoretic $H_\infty$ Riccati Equation & Disturbance Attenuation Bound $\gamma$
//! To guarantee $\|T_{zw}\|_{H_\infty} < \gamma$ against worst-case $L^2$ disturbances $w(t)$ for $\dot{x} = a x + b_1 w + b_2 u$,
//! the game-theoretic Riccati equation $2 a X + q - \left(\frac{b_2^2}{r} - \frac{b_1^2}{\gamma^2}\right) X^2 = 0$ admits a positive
//! stabilizing solution iff $\gamma > \gamma_{\text{crit}} = \frac{|b_1| \sqrt{r}}{|b_2|}$ (when $a \ge 0$).

use crate::error::{Error, Result};

/// Zames-Doyle-Glover $H_\infty$ Robust Control & Small-Gain Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct HInfinityRobustControlEngine;

impl HInfinityRobustControlEngine {
    /// Creates a new $H_\infty$ Robust Control engine.
    pub fn new() -> Self {
        Self
    }

    /// Solves the scalar game-theoretic $H_\infty$ Algebraic Riccati Equation for attenuation level $\gamma > 0$.
    pub fn solve_h_infinity_are(
        a: f64,
        b1_dist: f64,
        b2_ctrl: f64,
        q: f64,
        r: f64,
        gamma: f64,
    ) -> Result<f64> {
        if q <= 0.0 || r <= 0.0 || gamma <= 0.0 {
            return Err(Error::Msg("q, r, and gamma must be > 0".to_string()));
        }
        let eff_gain = (b2_ctrl * b2_ctrl) / r - (b1_dist * b1_dist) / (gamma * gamma);
        if eff_gain <= 0.0 {
            return Err(Error::Msg(
                "Disturbance attenuation gamma is below the achievable H_infinity threshold"
                    .to_string(),
            ));
        }
        Ok((a + (a * a + eff_gain * q).sqrt()) / eff_gain)
    }
}
