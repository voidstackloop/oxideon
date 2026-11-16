//! Friedgut-Bourgain Sharp Threshold Theorem & Russo-Margulis Influence Formula (Russo, 1982; Bourgain, Kalai & Friedgut, 1999).
//!
//! ### 1. Russo-Margulis Derivative $d\mu_p(A)/dp = I_p(A)$ & Total Influence
//! For a monotone Boolean property $A \subset \{0, 1\}^n$ under product measure $\mu_p$:
//! $$\frac{d \mu_p(A)}{d p} = I_p(A) = \sum_{i=1}^n I_i^{(p)}(A), \qquad \Delta p_{\varepsilon} \le \frac{2 \ln(1/\varepsilon)}{I_p(A)}$$
//! By Friedgut-Bourgain, if $I_p(A) \ll \log n$, then $A$ is approximated by a bounded-size Junta; transitive-symmetric properties without local obstructions have sharp thresholds $\Delta p = O(1 / \log n)$.

use crate::error::{Error, Result};

/// Friedgut-Bourgain Sharp Threshold & Russo-Margulis Influence Engine.
#[derive(Debug, Clone, Copy)]
pub struct FriedgutBourgainSharpThreshold {
    /// Number of Boolean coordinates $n \ge 2$.
    pub num_vars_n: usize,
}

impl Default for FriedgutBourgainSharpThreshold {
    fn default() -> Self {
        Self { num_vars_n: 1024 }
    }
}

impl FriedgutBourgainSharpThreshold {
    /// Creates a new Friedgut-Bourgain Sharp Threshold engine.
    pub fn new(num_vars_n: usize) -> Result<Self> {
        if num_vars_n < 2 {
            return Err(Error::Msg("num_vars_n must be >= 2".to_string()));
        }
        Ok(Self { num_vars_n })
    }

    /// Evaluates the Russo-Margulis total influence $I(A) = \sum_i I_i(A)$ and the threshold window width $\Delta p \approx 1 / I(A)$.
    pub fn total_influence_and_threshold_width(&self, influences: &[f64]) -> Result<(f64, f64)> {
        if influences.is_empty() || influences.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(Error::Msg(
                "influences must be non-empty and non-negative".to_string(),
            ));
        }
        let total_inf: f64 = influences.iter().sum();
        if total_inf <= 0.0 {
            return Err(Error::Msg("total influence must be > 0".to_string()));
        }
        let width = (1.0 / total_inf).min(1.0);
        Ok((total_inf, width))
    }
}
