//! Kermack-McKendrick SIR Epidemic Threshold, Basic Reproduction Number $R_0$ & Final Epidemic Size (Kermack & McKendrick, 1927).
//!
//! ### 1. Herd Immunity Threshold & Final Size Transcendental Equation
//! For transmission rate $\beta$ and recovery rate $\gamma$, the basic reproduction number is $R_0 = \beta / \gamma$,
//! herd immunity threshold is $p_c = 1 - 1/R_0$, and final attack fraction $z_\infty \in (0, 1)$ for $R_0 > 1$ solves $1 - z_\infty = e^{-R_0 z_\infty}$.

use crate::error::{Error, Result};

/// Kermack-McKendrick SIR Epidemic Threshold & Final Attack Size Engine.
#[derive(Debug, Clone, Copy)]
pub struct KermackMcKendrickEpidemicEngine {
    /// Transmission rate $\beta > 0$.
    pub beta: f64,
    /// Recovery rate $\gamma > 0$.
    pub gamma: f64,
}

impl Default for KermackMcKendrickEpidemicEngine {
    fn default() -> Self {
        Self {
            beta: 0.5,
            gamma: 0.25,
        }
    }
}

impl KermackMcKendrickEpidemicEngine {
    /// Creates a new Kermack-McKendrick SIR engine.
    pub fn new(beta: f64, gamma: f64) -> Result<Self> {
        if beta <= 0.0 || gamma <= 0.0 {
            return Err(Error::Msg("beta and gamma must be > 0".to_string()));
        }
        Ok(Self { beta, gamma })
    }

    /// Evaluates $R_0 = \beta / \gamma$, herd immunity threshold $\max(0, 1 - 1/R_0)$, and Newton-Raphson final epidemic size $z_\infty$.
    pub fn r0_herd_threshold_and_final_size(&self) -> (f64, f64, f64) {
        let r0 = self.beta / self.gamma;
        if r0 <= 1.0 {
            return (r0, 0.0, 0.0);
        }
        let herd = 1.0 - 1.0 / r0;
        let mut z = 0.9;
        for _ in 0..30 {
            let f = z - 1.0 + (-r0 * z).exp();
            let df = 1.0 - r0 * (-r0 * z).exp();
            if df.abs() > 1e-15 {
                z = (z - f / df).clamp(1e-9, 1.0 - 1e-12);
            }
        }
        (r0, herd, z)
    }
}
