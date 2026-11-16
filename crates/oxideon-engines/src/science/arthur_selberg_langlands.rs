//! Satake Isomorphism, Hecke L-Polynomials & Deligne-Ramanujan-Petersson Bound (Satake, 1963; Deligne, 1974).
//!
//! ### 1. Ramanujan-Petersson Bound & Satake Parameters
//! For a weight-$k$ holomorphic cusp Hecke eigenform $f$ at prime $p$, the Hecke eigenvalue $a_p$ satisfies Deligne's theorem:
//! $$|a_p| \le 2 p^{(k-1)/2}, \qquad L_p(s, f)^{-1} = 1 - a_p p^{-s} + p^{k-1-2s}$$

use crate::error::{Error, Result};

/// Satake Parameter, Local Euler Factor & Ramanujan-Petersson Hecke Engine.
#[derive(Debug, Clone, Copy)]
pub struct SatakeHeckeLanglandsEngine {
    /// Modular weight $k \ge 2$.
    pub weight_k: usize,
}

impl Default for SatakeHeckeLanglandsEngine {
    fn default() -> Self {
        Self { weight_k: 12 }
    }
}

impl SatakeHeckeLanglandsEngine {
    /// Creates a new Satake-Hecke Langlands engine.
    pub fn new(weight_k: usize) -> Result<Self> {
        if weight_k < 2 {
            return Err(Error::Msg("Modular weight k must be >= 2".to_string()));
        }
        Ok(Self { weight_k })
    }

    /// Evaluates Deligne's Ramanujan bound $2 p^{(k-1)/2}$ and normalized Satake angle $\theta_p = \arccos(a_p / (2 p^{(k-1)/2}))$ if tempered.
    pub fn ramanujan_bound_and_satake_angle(
        &self,
        prime_p: u64,
        hecke_ap: f64,
    ) -> Result<(f64, f64, bool)> {
        if prime_p < 2 {
            return Err(Error::Msg("prime_p must be >= 2".to_string()));
        }
        let bound = 2.0 * (prime_p as f64).powf(0.5 * ((self.weight_k - 1) as f64));
        let ratio = (hecke_ap / bound).clamp(-1.0, 1.0);
        let is_tempered = hecke_ap.abs() <= bound + 1e-9;
        Ok((bound, ratio.acos(), is_tempered))
    }
}
