//! Distribution-Free Conformal Prediction & Quantile Uncertainty Quantification.
//!
//! Provides finite-sample validity guarantees:
//! $$\mathbb{P}(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1 - \alpha$$
//! holding for any data distribution, without requiring asymptotic or parametric assumptions.
//!
//! Features:
//! - Split Conformal Regression with homoscedastic and locally heteroscedastic scaling.
//! - Conformalized Quantile Regression (CQR) with asymmetric intervals.
//! - Adaptive Conformal Classification with prediction sets $\hat{C}(X) \subseteq \{1, \dots, K\}$.
//! - Empirical coverage and sharpness evaluation suite.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Non-conformity score method for conformal prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformalScoreType {
    /// Absolute residual: $s_i = |y_i - \hat{y}_i|$.
    AbsoluteResidual,
    /// Normalized residual: $s_i = \frac{|y_i - \hat{y}_i|}{\hat{\sigma}_i + \epsilon}$.
    NormalizedResidual,
    /// Conformalized Quantile Regression: $s_i = \max(\hat{q}_{lo} - y_i, y_i - \hat{q}_{hi})$.
    CQR,
}

/// Conformal prediction interval $[y_{\text{lower}}, y_{\text{upper}}]$.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionInterval {
    pub lower: f64,
    pub upper: f64,
}

impl PredictionInterval {
    /// Interval width (sharpness).
    pub fn width(&self) -> f64 {
        (self.upper - self.lower).max(0.0)
    }

    /// Checks if a ground truth value is covered by this interval.
    pub fn contains(&self, y: f64) -> bool {
        y >= self.lower && y <= self.upper
    }
}

/// Split Conformal Predictor for regression models.
#[derive(Debug, Clone)]
pub struct SplitConformalRegressor {
    pub score_type: ConformalScoreType,
    pub alpha: f64,
    pub calibrated_q_hat: Option<f64>,
}

impl SplitConformalRegressor {
    /// Creates a new `SplitConformalRegressor` with target miscoverage rate $\alpha \in (0, 1)$.
    pub fn new(score_type: ConformalScoreType, alpha: f64) -> Result<Self> {
        // NaN fails both comparisons, so it slipped through and silently
        // calibrated to the minimum score (mirrors the CRC constructor).
        if alpha.is_nan() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(Error::Msg(format!("Alpha must be in (0, 1), got {alpha}")));
        }
        Ok(Self {
            score_type,
            alpha,
            calibrated_q_hat: None,
        })
    }

    /// Calibrates the conformal quantile on held-out calibration predictions and ground truth.
    ///
    /// - `y_pred`: Model point predictions $\hat{y}$ (or $[\hat{q}_{lo}, \hat{q}_{hi}]$ for CQR).
    /// - `y_true`: Ground truth calibration targets $y$.
    /// - `dispersion`: Optional variance/uncertainty estimates $\hat{\sigma}$ for normalized residuals.
    pub fn calibrate(
        &mut self,
        y_pred: &Tensor,
        y_true: &Tensor,
        dispersion: Option<&Tensor>,
    ) -> Result<f64> {
        let n = y_true.numel();
        if n == 0 {
            return Err(Error::Msg("Calibration set cannot be empty".into()));
        }

        let y_t = y_true.to_vec::<f64>()?;
        let mut scores = Vec::with_capacity(n);

        match self.score_type {
            ConformalScoreType::AbsoluteResidual => {
                let y_p = y_pred.to_vec::<f64>()?;
                if y_p.len() != n {
                    return Err(Error::Msg("y_pred and y_true length mismatch".into()));
                }
                for i in 0..n {
                    scores.push((y_t[i] - y_p[i]).abs());
                }
            }
            ConformalScoreType::NormalizedResidual => {
                let y_p = y_pred.to_vec::<f64>()?;
                let sig = match dispersion {
                    Some(d) => d.to_vec::<f64>()?,
                    None => vec![1.0; n],
                };
                if y_p.len() != n || sig.len() != n {
                    return Err(Error::Msg(
                        "Shape mismatch in normalized calibration".into(),
                    ));
                }
                for i in 0..n {
                    let s = (y_t[i] - y_p[i]).abs() / (sig[i].max(1e-6));
                    scores.push(s);
                }
            }
            ConformalScoreType::CQR => {
                // y_pred must be shape [n, 2]: col 0 = q_lo, col 1 = q_hi
                if y_pred.ndim() != 2 || y_pred.shape()[1] != 2 || y_pred.shape()[0] != n {
                    return Err(Error::Msg(format!(
                        "CQR requires y_pred of shape [{n}, 2], got {:?}",
                        y_pred.shape()
                    )));
                }
                let y_p = y_pred.to_vec::<f64>()?;
                for i in 0..n {
                    let q_lo = y_p[i * 2];
                    let q_hi = y_p[i * 2 + 1];
                    let s = (q_lo - y_t[i]).max(y_t[i] - q_hi);
                    scores.push(s);
                }
            }
        }

        // Sort non-conformity scores
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Exact conformal quantile index: p = ceil((n + 1) * (1 - alpha)) / n
        let rank = (((n + 1) as f64 * (1.0 - self.alpha)).ceil() as usize).min(n);
        let q_hat = scores[rank.saturating_sub(1)];

        self.calibrated_q_hat = Some(q_hat);
        Ok(q_hat)
    }

    /// Predicts calibrated prediction intervals for new test inputs.
    pub fn predict_intervals(
        &self,
        y_pred: &Tensor,
        dispersion: Option<&Tensor>,
    ) -> Result<Vec<PredictionInterval>> {
        let q_hat = self.calibrated_q_hat.ok_or_else(|| {
            Error::Msg("Regressor must be calibrated before predicting intervals".into())
        })?;

        match self.score_type {
            ConformalScoreType::AbsoluteResidual => {
                let y_p = y_pred.to_vec::<f64>()?;
                let intervals = y_p
                    .into_iter()
                    .map(|mu| PredictionInterval {
                        lower: mu - q_hat,
                        upper: mu + q_hat,
                    })
                    .collect();
                Ok(intervals)
            }
            ConformalScoreType::NormalizedResidual => {
                let y_p = y_pred.to_vec::<f64>()?;
                let sig = match dispersion {
                    Some(d) => d.to_vec::<f64>()?,
                    None => vec![1.0; y_p.len()],
                };
                // `calibrate` rejects this mismatch; zipping here silently
                // returned fewer intervals than inputs.
                if sig.len() != y_p.len() {
                    return Err(Error::Msg(format!(
                        "Shape mismatch in normalized predict: {} predictions vs {} dispersions",
                        y_p.len(),
                        sig.len()
                    )));
                }
                let intervals = y_p
                    .into_iter()
                    .zip(sig)
                    .map(|(mu, s)| PredictionInterval {
                        lower: mu - q_hat * s,
                        upper: mu + q_hat * s,
                    })
                    .collect();
                Ok(intervals)
            }
            ConformalScoreType::CQR => {
                if y_pred.ndim() != 2 || y_pred.shape()[1] != 2 {
                    return Err(Error::Msg("CQR requires y_pred shape [m, 2]".into()));
                }
                let m = y_pred.shape()[0];
                let y_p = y_pred.to_vec::<f64>()?;
                let mut intervals = Vec::with_capacity(m);
                for i in 0..m {
                    let q_lo = y_p[i * 2];
                    let q_hi = y_p[i * 2 + 1];
                    intervals.push(PredictionInterval {
                        lower: q_lo - q_hat,
                        upper: q_hi + q_hat,
                    });
                }
                Ok(intervals)
            }
        }
    }
}

/// Adaptive Conformal Classifier generating prediction sets with exact coverage guarantees:
/// $$\mathbb{P}(Y \in \hat{C}(X)) \ge 1 - \alpha$$
#[derive(Debug, Clone)]
pub struct AdaptiveConformalClassifier {
    pub alpha: f64,
    pub threshold_p_hat: Option<f64>,
}

impl AdaptiveConformalClassifier {
    /// Creates a new `AdaptiveConformalClassifier` with error rate $\alpha \in (0, 1)$.
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha.is_nan() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(Error::Msg(format!("Alpha must be in (0, 1), got {alpha}")));
        }
        Ok(Self {
            alpha,
            threshold_p_hat: None,
        })
    }

    /// Calibrates prediction set threshold on predicted probabilities `probs` \[n, K\] and `labels` \[n\].
    pub fn calibrate(&mut self, probs: &Tensor, labels: &[usize]) -> Result<f64> {
        let n = labels.len();
        if n == 0 {
            return Err(Error::Msg("Labels cannot be empty".into()));
        }
        let shape = probs.shape();
        if shape.len() != 2 || shape[0] != n {
            return Err(Error::Msg(format!(
                "Expected probs shape [{n}, K], got {:?}",
                shape
            )));
        }
        let k_classes = shape[1];
        let p_vec = probs.to_vec::<f64>()?;

        // Compute generalized inverse quantile non-conformity score: 1.0 - p_true_class
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let true_c = labels[i];
            if true_c >= k_classes {
                return Err(Error::Msg(format!(
                    "Label {true_c} out of bounds (0..{k_classes})"
                )));
            }
            let p_true = p_vec[i * k_classes + true_c];
            scores.push(1.0 - p_true);
        }

        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rank = (((n + 1) as f64 * (1.0 - self.alpha)).ceil() as usize).min(n);
        let s_hat = scores[rank.saturating_sub(1)];

        // Calibrated probability threshold: p_c >= 1.0 - s_hat
        let p_thresh = (1.0 - s_hat).max(0.0);
        self.threshold_p_hat = Some(p_thresh);
        Ok(p_thresh)
    }

    /// Predicts conformal prediction sets for test instances with predicted probabilities `probs` [m, K].
    pub fn predict_sets(&self, probs: &Tensor) -> Result<Vec<Vec<usize>>> {
        let p_thresh = self.threshold_p_hat.ok_or_else(|| {
            Error::Msg("Classifier must be calibrated before generating sets".into())
        })?;

        let shape = probs.shape();
        if shape.len() != 2 {
            return Err(Error::Msg("probs must be 2D [m, K]".into()));
        }
        let m = shape[0];
        let k = shape[1];
        let p_vec = probs.to_vec::<f64>()?;

        let mut sets = Vec::with_capacity(m);
        for i in 0..m {
            let mut set_i = Vec::new();
            for c in 0..k {
                let p = p_vec[i * k + c];
                if p >= p_thresh {
                    set_i.push(c);
                }
            }
            // Ensure non-empty set fallback to top-1 class if threshold is high
            if set_i.is_empty() {
                let mut best_c = 0;
                let mut best_p = -1.0;
                for c in 0..k {
                    let p = p_vec[i * k + c];
                    if p > best_p {
                        best_p = p;
                        best_c = c;
                    }
                }
                set_i.push(best_c);
            }
            sets.push(set_i);
        }

        Ok(sets)
    }
}

/// Evaluates empirical coverage and efficiency metrics for conformal predictions.
pub struct ConformalEvaluator;

impl ConformalEvaluator {
    /// Evaluates empirical interval coverage $\frac{1}{m} \sum \mathbb{I}(y_i \in [L_i, U_i])$ and mean width.
    pub fn evaluate_regression(
        intervals: &[PredictionInterval],
        y_true: &[f64],
    ) -> Result<(f64, f64)> {
        if intervals.len() != y_true.len() || intervals.is_empty() {
            return Err(Error::Msg(
                "Mismatch in intervals and ground truth count".into(),
            ));
        }
        let m = intervals.len();
        let mut covered_count = 0;
        let mut total_width = 0.0;

        for (interval, &y) in intervals.iter().zip(y_true) {
            if interval.contains(y) {
                covered_count += 1;
            }
            total_width += interval.width();
        }

        let coverage = covered_count as f64 / (m as f64);
        let mean_width = total_width / (m as f64);
        Ok((coverage, mean_width))
    }

    /// Evaluates empirical classification coverage $\frac{1}{m} \sum \mathbb{I}(y_i \in \hat{C}_i)$ and mean set size.
    pub fn evaluate_classification(sets: &[Vec<usize>], labels: &[usize]) -> Result<(f64, f64)> {
        if sets.len() != labels.len() || sets.is_empty() {
            return Err(Error::Msg("Mismatch in sets and labels count".into()));
        }
        let m = sets.len();
        let mut covered_count = 0;
        let mut total_size = 0;

        for (set, &y) in sets.iter().zip(labels) {
            if set.contains(&y) {
                covered_count += 1;
            }
            total_size += set.len();
        }

        let coverage = covered_count as f64 / (m as f64);
        let mean_size = total_size as f64 / (m as f64);
        Ok((coverage, mean_size))
    }
}
