//! Roth's Theorem on Diophantine Approximation, Faltings' Theorem (Mordell Conjecture) & Naive Weil Height (Roth, 1955; Faltings, 1983).
//!
//! ### 1. Roth's Irrationality Measure $\mu(\alpha) = 2$ & Weil Logarithmic Height
//! For any algebraic irrational $\alpha \in \overline{\mathbb{Q}} \setminus \mathbb{Q}$ and any $\epsilon > 0$, Roth's Theorem
//! states that $\left|\alpha - \frac{p}{q}\right| < \frac{1}{q^{2 + \epsilon}}$ has only finitely many coprime integer solutions $(p, q)$.

use crate::error::{Error, Result};

/// Diophantine Approximation, Weil Height & Faltings Mordell Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct FaltingsHeightEngine;

impl FaltingsHeightEngine {
    /// Creates a new Faltings-Roth Diophantine engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the logarithmic Weil height $h(p/q) = \ln \max(|p|, |q|)$ of a reduced rational $p/q$ and the empirical
    /// Diophantine approximation exponent $\mu = -\frac{\ln |\alpha - p/q|}{\ln q}$.
    pub fn diophantine_approximation_exponent(alpha: f64, p: i64, q: i64) -> Result<(f64, f64)> {
        if q <= 1 {
            return Err(Error::Msg("Denominator q must be > 1".to_string()));
        }
        let err = (alpha - (p as f64) / (q as f64)).abs();
        if err == 0.0 {
            return Err(Error::Msg(
                "Exact rational match has infinite exponent".to_string(),
            ));
        }
        let weil_h = (p.unsigned_abs().max(q.unsigned_abs()) as f64).ln();
        let mu_eff = -err.ln() / (q as f64).ln();
        Ok((weil_h, mu_eff))
    }
}
