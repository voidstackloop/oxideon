//! Shamir Polynomial Secret Sharing & Ben-Or-Goldwasser-Wigderson (BGW) Information-Theoretic MPC (Shamir, 1979; Ben-Or, Goldwasser & Wigderson, 1988).
//!
//! ### 1. Information-Theoretic Honesty Thresholds: Semi-Honest $t < n/2$ & Malicious $t < n/3$
//! Over a finite field $\mathbb{F}_q$ ($q > n$), Shamir $(t+1, n)$ secret sharing hides $s = P(0)$ behind a random degree-$t$ polynomial, and BGW degree-reduction after multiplication requires $2t + 1 \le n$ (semi-honest) or $3t + 1 \le n$ (malicious without broadcast).

use crate::error::{Error, Result};

/// Shamir Secret Sharing & BGW Information-Theoretic MPC Engine.
#[derive(Debug, Clone, Copy)]
pub struct ShamirBgwThresholdMpc {
    /// Total number of parties $n \ge 3$.
    pub num_parties_n: usize,
}

impl Default for ShamirBgwThresholdMpc {
    fn default() -> Self {
        Self { num_parties_n: 7 }
    }
}

impl ShamirBgwThresholdMpc {
    /// Creates a new Shamir-BGW Threshold MPC engine.
    pub fn new(num_parties_n: usize) -> Result<Self> {
        if num_parties_n < 3 {
            return Err(Error::Msg("num_parties_n must be >= 3".to_string()));
        }
        Ok(Self { num_parties_n })
    }

    /// Evaluates the maximum tolerated corrupted parties $(t_{\mathrm{semi}}, t_{\mathrm{malicious}}) = (\lfloor(n-1)/2\rfloor, \lfloor(n-1)/3\rfloor)$ in BGW.
    pub fn max_semi_honest_and_malicious_thresholds(&self) -> (usize, usize) {
        ((self.num_parties_n - 1) / 2, (self.num_parties_n - 1) / 3)
    }
}
