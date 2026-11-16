//! Gallager Random Coding Error Exponent $E_r(R) = \max_{\rho \in [0, 1]} [E_0(\rho) - \rho R]$ & LDPC Iterative Decoding (Gallager, 1962/1965).
//!
//! ### 1. Gallager Exponent $E_0(\rho)$ for the Binary Symmetric Channel (BSC)
//! Over a BSC with crossover probability $p \in (0, 1/2)$, uniform input distribution gives:
//! $$E_0(\rho) = \rho - (1 + \rho) \log_2\!\left(p^{1/(1+\rho)} + (1 - p)^{1/(1+\rho)}\right), \qquad P_e(N) \le 2^{-N E_r(R)}$$

use crate::error::{Error, Result};

/// Gallager Random Coding Error Exponent & LDPC Engine.
#[derive(Debug, Clone, Copy)]
pub struct GallagerRandomCodingExponent {
    /// BSC crossover probability $p \in (0, 0.5)$.
    pub crossover_p: f64,
}

impl Default for GallagerRandomCodingExponent {
    fn default() -> Self {
        Self { crossover_p: 0.05 }
    }
}

impl GallagerRandomCodingExponent {
    /// Creates a new Gallager Random Coding Exponent engine.
    pub fn new(crossover_p: f64) -> Result<Self> {
        if !(0.0..0.5).contains(&crossover_p) || crossover_p == 0.0 {
            return Err(Error::Msg("crossover_p must be in (0, 0.5)".to_string()));
        }
        Ok(Self { crossover_p })
    }

    /// Evaluates the Gallager function $E_0(\rho)$ and the error exponent lower bound $E_0(\rho) - \rho R$ for $\rho \in [0, 1]$.
    pub fn e0_and_error_exponent(&self, rho: f64, rate_r: f64) -> Result<(f64, f64)> {
        if !(0.0..=1.0).contains(&rho) || rate_r < 0.0 {
            return Err(Error::Msg(
                "rho in [0, 1] and rate_r >= 0 required".to_string(),
            ));
        }
        let inv = 1.0 / (1.0 + rho);
        let sum_term = self.crossover_p.powf(inv) + (1.0 - self.crossover_p).powf(inv);
        let e0 = rho - (1.0 + rho) * sum_term.log2();
        Ok((e0, (e0 - rho * rate_r).max(0.0)))
    }
}
