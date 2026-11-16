//! Ben-Sasson et al. Fast Reed-Solomon Interactive Oracle Proof of Proximity (FRI) & Transparent STARKs (Ben-Sasson, Bentov, Horesh & Riabzev, 2018).
//!
//! ### 1. Johnson-Unique-Decoding Proximity Gap & Query Soundness $(1 - \delta_{\mathrm{FRI}})^s$
//! For a Reed-Solomon evaluation domain $D \subset \mathbb{F}_q$ of rate $\rho = k / |D| \in (0, 1)$, $s$ independent FRI query chains achieve soundness error:
//! $$\epsilon_{\mathrm{query}} \le (1 - \delta)^s, \qquad \delta \le 1 - \sqrt{\rho}$$

use crate::error::{Error, Result};

/// Ben-Sasson FRI Reed-Solomon Proximity & STARK Soundness Engine.
#[derive(Debug, Clone, Copy)]
pub struct FriReedSolomonProximityStark {
    /// Reed-Solomon rate $\rho = 1 / B \in (0, 1)$, where $B$ is the blowup factor.
    pub rate_rho: f64,
}

impl Default for FriReedSolomonProximityStark {
    fn default() -> Self {
        Self { rate_rho: 0.125 }
    }
}

impl FriReedSolomonProximityStark {
    /// Creates a new FRI Reed-Solomon Proximity STARK engine.
    pub fn new(rate_rho: f64) -> Result<Self> {
        if !(0.0..1.0).contains(&rate_rho) || rate_rho == 0.0 {
            return Err(Error::Msg("rate_rho must be in (0, 1)".to_string()));
        }
        Ok(Self { rate_rho })
    }

    /// Evaluates the Johnson/capacity proximity bound $1 - \sqrt{\rho}$ and the bits of query security $-\log_2(\rho^{s/2})$ for $s$ queries.
    pub fn johnson_proximity_and_query_security_bits(
        &self,
        num_queries_s: usize,
    ) -> Result<(f64, f64)> {
        if num_queries_s == 0 {
            return Err(Error::Msg("num_queries_s must be >= 1".to_string()));
        }
        let delta_j = 1.0 - self.rate_rho.sqrt();
        let sec_bits = -0.5 * (num_queries_s as f64) * self.rate_rho.log2();
        Ok((delta_j, sec_bits))
    }
}
