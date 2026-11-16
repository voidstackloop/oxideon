//! Conformalized Risk Control (CRC) & Multi-Label False Discovery Rate (FDR) Guarantees.
//!
//! Extends distribution-free conformal prediction beyond 0-1 miscoverage to arbitrary
//! bounded monotone loss functions $L(\hat{C}_\lambda(X), Y) \in [0, B]$:
//!
//! $$\mathbb{E}\left[L\left(\hat{C}_{\hat{\lambda}}(X_{n+1}), Y_{n+1}\right)\right] \le \alpha$$
//!
//! References:
//! - Angelopoulos, Bates, Candès, Jordan, Wainwright (2022): "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control"
//! - Bates, Candès, Lei, Romano (2021): "Distribution-Free False Discovery Rate Control for Selection of Important Features"

use crate::error::{Error, Result};

/// Type of statistical risk metric to control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskMetric {
    /// False Negative Rate: $1 - \frac{|C \cap Y|}{|Y|}$ (fraction of true labels missed).
    FalseNegativeRate,
    /// False Discovery Proportion: $\frac{|C \setminus Y|}{\max(|C|, 1)}$ (fraction of spurious predictions).
    FalseDiscoveryProportion,
    /// Custom bounded loss in $[0, B]$.
    BoundedLoss,
}

/// Output result of Conformalized Risk Control calibration.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskCalibrationResult {
    /// Calibrated threshold $\hat{\lambda}$ guaranteeing $\mathbb{E}[R(\hat{\lambda})] \le \alpha$.
    pub lambda_hat: f64,
    /// Empirical risk $\hat{R}_n(\hat{\lambda})$ achieved on calibration set.
    pub empirical_risk: f64,
    /// Theoretical upper bound $\frac{n}{n+1} \hat{R}_n + \frac{B}{n+1}$.
    pub risk_upper_bound: f64,
    /// Target risk tolerance $\alpha$.
    pub target_alpha: f64,
    /// Upper bound $B$ on loss values.
    pub loss_bound_b: f64,
    /// Number of calibration samples $n$.
    pub num_samples: usize,
}

/// Conformalized Risk Controller for multi-label prediction sets.
#[derive(Debug, Clone)]
pub struct ConformalRiskController {
    /// Risk metric to control.
    pub metric: RiskMetric,
    /// Target risk upper bound $\alpha \in (0, 1)$.
    pub alpha: f64,
    /// Upper bound $B$ on the loss function (typically 1.0 for FNR/FDP).
    pub loss_bound_b: f64,
}

impl ConformalRiskController {
    /// Creates a new risk controller with specified metric, target risk $\alpha$, and loss bound $B$.
    pub fn new(metric: RiskMetric, alpha: f64, loss_bound_b: f64) -> Result<Self> {
        // NaN must be rejected here: it would later panic inside `f64::clamp`.
        if alpha.is_nan() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(Error::Msg(format!(
                "Target risk alpha must be in (0, 1), got {alpha}"
            )));
        }
        if loss_bound_b.is_nan() || loss_bound_b <= 0.0 {
            return Err(Error::Msg(format!(
                "Loss bound B must be strictly positive, got {loss_bound_b}"
            )));
        }
        Ok(Self {
            metric,
            alpha,
            loss_bound_b,
        })
    }

    /// Evaluates sample loss $L(C, Y) \in [0, B]$ for a predicted set $C$ and ground truth set $Y$.
    pub fn compute_loss(&self, pred_set: &[usize], true_labels: &[usize]) -> Result<f64> {
        // `loss_bound_b` is pub: NaN/negative values panicked inside
        // `f64::clamp` below (`new()` checks construction only).
        let b = self.loss_bound_b;
        if b.is_nan() || b <= 0.0 {
            return Err(Error::Msg(format!(
                "Loss bound B must be strictly positive, got {b}"
            )));
        }
        if true_labels.is_empty() {
            return Ok(0.0);
        }

        match self.metric {
            RiskMetric::FalseNegativeRate => {
                // FNR = 1 - (|pred \cap true| / |true|)
                let mut hits = 0;
                for &y in true_labels {
                    if pred_set.contains(&y) {
                        hits += 1;
                    }
                }
                let recall = hits as f64 / true_labels.len() as f64;
                Ok((1.0 - recall).clamp(0.0, b))
            }
            RiskMetric::FalseDiscoveryProportion => {
                // FDP = |pred \setminus true| / max(|pred|, 1)
                if pred_set.is_empty() {
                    return Ok(0.0);
                }
                let mut false_discoveries = 0;
                for &p in pred_set {
                    if !true_labels.contains(&p) {
                        false_discoveries += 1;
                    }
                }
                let fdp = false_discoveries as f64 / pred_set.len() as f64;
                Ok(fdp.clamp(0.0, b))
            }
            RiskMetric::BoundedLoss => {
                let mut hits = 0;
                for &y in true_labels {
                    if pred_set.contains(&y) {
                        hits += 1;
                    }
                }
                let missed = true_labels.len().saturating_sub(hits);
                Ok((missed as f64 / true_labels.len() as f64).clamp(0.0, b))
            }
        }
    }

    /// Forms prediction set $\hat{C}_\lambda(X) = \{k : s_k \ge \lambda\}$.
    pub fn predict_set(scores: &[f64], lambda: f64) -> Vec<usize> {
        scores
            .iter()
            .enumerate()
            .filter(|(_, s)| **s >= lambda)
            .map(|(k, _)| k)
            .collect()
    }

    /// Computes empirical mean risk $\hat{R}_n(\lambda)$ over $n$ calibration instances.
    pub fn evaluate_empirical_risk(
        &self,
        calibration_scores: &[Vec<f64>],
        ground_truth: &[Vec<usize>],
        lambda: f64,
    ) -> Result<f64> {
        let n = calibration_scores.len();
        if n == 0 || n != ground_truth.len() {
            return Err(Error::Msg(format!(
                "Calibration scores ({n}) and ground truth ({}) size mismatch or empty",
                ground_truth.len()
            )));
        }

        let mut total_loss = 0.0;
        for (scores, true_set) in calibration_scores.iter().zip(ground_truth) {
            let pred_set = Self::predict_set(scores, lambda);
            total_loss += self.compute_loss(&pred_set, true_set)?;
        }

        Ok(total_loss / n as f64)
    }

    /// Calibrates threshold $\hat{\lambda}$ using finite-sample Conformal Risk Control bound:
    ///
    /// $$\frac{n}{n+1} \hat{R}_n(\lambda) + \frac{B}{n+1} \le \alpha$$
    ///
    /// For FNR (which decreases with lower $\lambda$ as sets expand), searches for largest $\lambda \in [0, 1]$
    /// satisfying the CRC risk bound.
    pub fn calibrate(
        &self,
        calibration_scores: &[Vec<f64>],
        ground_truth: &[Vec<usize>],
        num_grid_steps: usize,
    ) -> Result<RiskCalibrationResult> {
        let n = calibration_scores.len();
        if n == 0 || n != ground_truth.len() {
            return Err(Error::Msg(
                "Empty or mismatched calibration set".to_string(),
            ));
        }

        let b = self.loss_bound_b;
        let alpha = self.alpha;
        let steps = num_grid_steps.max(100);
        // Each grid step costs a full empirical-risk pass; real grids are hundreds wide.
        crate::limits::check_repeats("num_grid_steps", steps)?;

        let mut best_lambda = 0.0;
        let mut best_emp_risk = 0.0;
        let mut best_upper_bound = 0.0;
        let mut found = false;

        // Search thresholds from 1.0 down to 0.0
        for step in 0..=steps {
            let lambda = 1.0 - (step as f64) / (steps as f64);
            let emp_risk =
                self.evaluate_empirical_risk(calibration_scores, ground_truth, lambda)?;
            let upper_bound = ((n as f64) * emp_risk + b) / (n as f64 + 1.0);

            if upper_bound <= alpha {
                best_lambda = lambda;
                best_emp_risk = emp_risk;
                best_upper_bound = upper_bound;
                found = true;
                break;
            }
        }

        // If even lambda=0 (all labels included) doesn't strictly satisfy bound due to small n:
        if !found {
            best_lambda = 0.0;
            best_emp_risk = self.evaluate_empirical_risk(calibration_scores, ground_truth, 0.0)?;
            best_upper_bound = ((n as f64) * best_emp_risk + b) / (n as f64 + 1.0);
        }

        Ok(RiskCalibrationResult {
            lambda_hat: best_lambda,
            empirical_risk: best_emp_risk,
            risk_upper_bound: best_upper_bound,
            target_alpha: alpha,
            loss_bound_b: b,
            num_samples: n,
        })
    }
}

/// Learn-Then-Test (LTT) multi-hypothesis calibration engine.
#[derive(Debug, Clone)]
pub struct LearnThenTest;

impl LearnThenTest {
    /// Computes Hoeffding-Bentkus p-value for testing null hypothesis $H_0: \mathbb{E}[R(\lambda)] > \alpha$:
    ///
    /// $$p(\lambda) = \exp\left(-n D_{\text{KL}}(\hat{R}_n \parallel \alpha)\right) \quad \text{for } \hat{R}_n < \alpha$$
    pub fn hoeffding_bentkus_p_value(emp_risk: f64, alpha: f64, n: usize) -> Result<f64> {
        if !emp_risk.is_finite() || !(0.0..=1.0).contains(&emp_risk) {
            return Err(Error::Msg(format!(
                "hoeffding_bentkus_p_value requires emp_risk in [0, 1], got {emp_risk}"
            )));
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(Error::Msg(format!(
                "hoeffding_bentkus_p_value requires alpha in [0, 1], got {alpha}"
            )));
        }
        if n == 0 {
            return Err(Error::Msg(
                "hoeffding_bentkus_p_value requires n > 0".into(),
            ));
        }
        if emp_risk >= alpha {
            return Ok(1.0);
        }
        let r = emp_risk.max(1e-12);
        let a = alpha.min(1.0 - 1e-12);

        // KL divergence between Bernoulli(r) and Bernoulli(a)
        let kl = r * (r / a).ln() + (1.0 - r) * ((1.0 - r) / (1.0 - a)).ln();
        let exponent = -(n as f64) * kl;
        Ok(exponent.exp().clamp(0.0, 1.0))
    }

    /// Benjamini-Hochberg step-up procedure to control False Discovery Rate at level $q \in (0, 1)$.
    ///
    /// Returns indices of rejected hypotheses.
    pub fn benjamini_hochberg(p_values: &[f64], fdr_q: f64) -> Vec<usize> {
        let m = p_values.len();
        if m == 0 {
            return Vec::new();
        }

        let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut max_k = None;
        for (k_idx, &(orig_idx, p_val)) in indexed.iter().enumerate() {
            let k = k_idx + 1;
            let threshold = (k as f64 / m as f64) * fdr_q;
            if p_val <= threshold {
                max_k = Some((k_idx, orig_idx));
            }
        }

        match max_k {
            Some((cutoff, _)) => indexed[0..=cutoff].iter().map(|&(idx, _)| idx).collect(),
            None => Vec::new(),
        }
    }
}
