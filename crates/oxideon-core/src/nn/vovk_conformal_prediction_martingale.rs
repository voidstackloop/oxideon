//! Vovk Distribution-Free Split Conformal Prediction & Finite-Sample Exchangeability Coverage Guarantee (Vovk, Gammerman & Shafer, 2005).
//!
//! ### 1. Exact Finite-Sample Marginal Coverage Theorem
//! Given $n$ exchangeable calibration nonconformity scores $s_1, \dots, s_n$ and miscoverage level $\alpha \in (0, 1)$,
//! let $\hat{q}_{1-\alpha}$ be the $k$-th smallest calibration score where $k = \lceil (n + 1)(1 - \alpha) \rceil$.
//! Then the prediction set $\widehat{C}(X_{n+1}) = \{y : s(X_{n+1}, y) \le \hat{q}_{1-\alpha}\}$ satisfies for ANY distribution $P$:
//! $$1 - \alpha \le \mathbb{P}\left(Y_{n+1} \in \widehat{C}(X_{n+1})\right) < 1 - \alpha + \frac{1}{n + 1}$$

use crate::error::{Error, Result};

/// Vovk Distribution-Free Split Conformal Prediction Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitConformalPredictionEngine;

impl SplitConformalPredictionEngine {
    /// Creates a new Split Conformal Prediction engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the exact finite-sample conformal quantile threshold $\hat{q}_{1-\alpha}$ from $n$ calibration nonconformity scores.
    pub fn conformal_quantile(calibration_scores: &[f64], miscoverage_alpha: f64) -> Result<f64> {
        let n = calibration_scores.len();
        if n == 0 || !(0.0..1.0).contains(&miscoverage_alpha) {
            return Err(Error::Msg(
                "Non-empty calibration_scores and alpha in (0, 1) required".to_string(),
            ));
        }
        let k = (((n + 1) as f64) * (1.0 - miscoverage_alpha)).ceil() as usize;
        if k > n {
            return Ok(f64::INFINITY);
        }
        let mut sorted = calibration_scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(sorted[k - 1])
    }
}
