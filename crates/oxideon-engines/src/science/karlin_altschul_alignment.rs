//! Karlin-Altschul Extreme-Value Local Alignment Statistics & Gumbel E-Value (Karlin & Altschul, 1990; Smith & Waterman, 1981).
//!
//! ### 1. Karlin-Altschul Bit Score & Gumbel P-Value
//! For query length $m$, database length $n$, scale parameter $\lambda > 0$, and entropy parameter $K \in (0, 1)$,
//! a raw alignment score $S$ has expected high-scoring segment pair count (E-value) and P-value:
//! $$E(S) = K m n e^{-\lambda S}, \qquad S_{\mathrm{bits}} = \frac{\lambda S - \ln K}{\ln 2}, \qquad P(S) = 1 - \exp(-E(S))$$

use crate::error::{Error, Result};

/// Karlin-Altschul Local Alignment Gumbel Statistics Engine.
#[derive(Debug, Clone, Copy)]
pub struct KarlinAltschulSequenceAlignment {
    /// Karlin-Altschul exponential scale parameter $\lambda > 0$.
    pub lambda: f64,
    /// Karlin-Altschul prefactor $K > 0$.
    pub kappa_k: f64,
}

impl Default for KarlinAltschulSequenceAlignment {
    fn default() -> Self {
        Self {
            lambda: 0.267,
            kappa_k: 0.041,
        }
    }
}

impl KarlinAltschulSequenceAlignment {
    /// Creates a new Karlin-Altschul Sequence Alignment engine.
    pub fn new(lambda: f64, kappa_k: f64) -> Result<Self> {
        if lambda <= 0.0 || kappa_k <= 0.0 {
            return Err(Error::Msg("lambda and kappa_k must be > 0".to_string()));
        }
        Ok(Self { lambda, kappa_k })
    }

    /// Evaluates bit score $S_{\mathrm{bits}}$, expected value $E = K m n e^{-\lambda S}$, and Gumbel P-value $1 - e^{-E}$.
    pub fn bit_score_evalue_and_pvalue(
        &self,
        raw_score_s: f64,
        query_m: u64,
        db_n: u64,
    ) -> Result<(f64, f64, f64)> {
        if query_m == 0 || db_n == 0 {
            return Err(Error::Msg("query_m and db_n must be >= 1".to_string()));
        }
        let bit_score = (self.lambda * raw_score_s - self.kappa_k.ln()) / 2.0_f64.ln();
        let e_val =
            self.kappa_k * (query_m as f64) * (db_n as f64) * (-self.lambda * raw_score_s).exp();
        let p_val = 1.0 - (-e_val).exp();
        Ok((bit_score, e_val, p_val))
    }
}
