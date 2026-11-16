//! Backus-Gilbert Geophysical Linear Inverse Theory, Resolving Kernels & Resolution-Variance Trade-Off (Backus & Gilbert, 1968).
//!
//! ### 1. Dirichlet Resolving Spread vs. Error Variance Trade-Off
//! Given model resolution matrix $R_\alpha = (G^T G + \alpha I)^{-1} G^T G$ at Tikhonov-Backus trade-off parameter $\alpha \ge 0$,
//! increasing $\alpha$ decreases model covariance $\mathrm{Tr}(\mathrm{Cov}(m))$ at the expense of broader resolution spread $\|R_\alpha - I\|_F^2$.

use crate::error::{Error, Result};

/// Backus-Gilbert Geophysical Inverse Resolution & Trade-Off Engine.
#[derive(Debug, Clone)]
pub struct BackusGilbertInverseResolution {
    /// Singular values $\sigma_i > 0$ of the forward sensitivity operator $G$.
    pub singular_values: Vec<f64>,
}

impl Default for BackusGilbertInverseResolution {
    fn default() -> Self {
        Self {
            singular_values: vec![4.0, 2.0, 1.0, 0.5],
        }
    }
}

impl BackusGilbertInverseResolution {
    /// Creates a new Backus-Gilbert Inverse Resolution engine.
    pub fn new(singular_values: Vec<f64>) -> Result<Self> {
        if singular_values.is_empty() || singular_values.iter().any(|&s| s <= 0.0 || !s.is_finite())
        {
            return Err(Error::Msg(
                "singular_values must be non-empty and > 0".to_string(),
            ));
        }
        Ok(Self { singular_values })
    }

    /// Evaluates the effective degrees of freedom $\mathrm{Tr}(R_\alpha)$, resolution spread $\sum (1 - r_i)^2$, and unit-noise variance $\sum (\sigma_i / (\sigma_i^2 + \alpha))^2$.
    pub fn resolution_trace_spread_and_variance(
        &self,
        trade_off_alpha: f64,
    ) -> Result<(f64, f64, f64)> {
        if trade_off_alpha < 0.0 {
            return Err(Error::Msg("trade_off_alpha must be >= 0".to_string()));
        }
        let mut trace_r = 0.0;
        let mut spread = 0.0;
        let mut var = 0.0;
        for &s in &self.singular_values {
            let s2 = s * s;
            let denom = s2 + trade_off_alpha;
            let r_i = s2 / denom;
            trace_r += r_i;
            spread += (1.0 - r_i).powi(2);
            var += (s / denom).powi(2);
        }
        Ok((trace_r, spread, var))
    }
}
