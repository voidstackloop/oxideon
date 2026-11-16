//! Quantum Error Mitigation: Zero-Noise Extrapolation (ZNE) & Clifford Data Regression (CDR).
//!
//! Provides algorithmic error mitigation on noisy intermediate-scale quantum (NISQ) systems
//! without requiring physical ancilla qubits or fault-tolerant syndrome measurements:
//! - [`ZneExtrapolator`]: Zero-Noise Extrapolation using Richardson polynomial, linear, and exponential fits.
//! - [`CliffordDataRegression`]: Machine-learning error calibration using classically simulable Clifford training circuits.

use crate::error::{Error, Result};

/// Extrapolation model for Zero-Noise Extrapolation (ZNE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtrapolationModel {
    /// Linear fit: $E(\lambda) = a \lambda + b \implies E(0) = b$.
    Linear,
    /// Richardson polynomial interpolation through all scale factors: $E(0) = \sum_i E(\lambda_i) \prod_{j \ne i} \frac{-\lambda_j}{\lambda_i - \lambda_j}$.
    Richardson,
    /// Exponential decay model: $E(\lambda) = A + B e^{-\gamma \lambda}$.
    Exponential,
}

/// Zero-Noise Extrapolator for NISQ expectation values.
#[derive(Debug, Clone)]
pub struct ZneExtrapolator {
    /// Noise scale factors $\lambda_i \ge 1.0$ (e.g. $[1.0, 1.5, 2.0]$).
    pub scale_factors: Vec<f64>,
    /// Extrapolation model.
    pub model: ExtrapolationModel,
}

impl ZneExtrapolator {
    /// Constructs a new ZNE extrapolator.
    pub fn new(scale_factors: Vec<f64>, model: ExtrapolationModel) -> Result<Self> {
        if scale_factors.is_empty() {
            return Err(Error::Msg("Scale factors cannot be empty".into()));
        }
        for &s in &scale_factors {
            if !s.is_finite() || s < 1.0 {
                return Err(Error::Msg(format!(
                    "Scale factor must be finite and >= 1.0, got {s}"
                )));
            }
        }
        if model == ExtrapolationModel::Linear && scale_factors.len() < 2 {
            return Err(Error::Msg(
                "Linear extrapolation requires at least 2 points".into(),
            ));
        }

        Ok(Self {
            scale_factors,
            model,
        })
    }

    /// Extrapolates noisy expectation values measured at each noise scale to the zero-noise limit $\lambda = 0$.
    pub fn extrapolate(&self, noisy_values: &[f64]) -> Result<f64> {
        // Scales are public config: re-check (tampered empty/nonfinite
        // scales silently yield 0.0/NaN through the fits below).
        if self.scale_factors.is_empty() {
            return Err(Error::Msg("Scale factors cannot be empty".into()));
        }
        for &s in &self.scale_factors {
            if !s.is_finite() || s < 1.0 {
                return Err(Error::Msg(format!(
                    "Scale factor must be finite and >= 1.0, got {s}"
                )));
            }
        }
        let m = self.scale_factors.len();
        if noisy_values.len() != m {
            return Err(Error::ShapeMismatch {
                op: "ZneExtrapolator::extrapolate",
                expected: vec![m],
                got: vec![noisy_values.len()],
            });
        }

        match self.model {
            ExtrapolationModel::Linear => {
                // Ordinary least squares linear fit: E(lambda) = a * lambda + b
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_xx = 0.0;
                let mut sum_xy = 0.0;
                let n = m as f64;

                for (&x, &y) in self.scale_factors.iter().zip(noisy_values.iter()) {
                    sum_x += x;
                    sum_y += y;
                    sum_xx += x * x;
                    sum_xy += x * y;
                }

                let denom = n * sum_xx - sum_x * sum_x;
                if denom.abs() < 1e-12 {
                    return Err(Error::Msg(
                        "Singular scale factors in linear extrapolation".into(),
                    ));
                }

                let intercept = (sum_y * sum_xx - sum_x * sum_xy) / denom;
                Ok(intercept)
            }
            ExtrapolationModel::Richardson => {
                // Richardson polynomial extrapolation: evaluates Lagrange interpolating polynomial at lambda = 0
                let mut zero_noise_estimate = 0.0;

                for (i, &y_i) in noisy_values.iter().enumerate().take(m) {
                    let lambda_i = self.scale_factors[i];
                    let mut coeff = 1.0;

                    for (j, &lambda_j) in self.scale_factors.iter().enumerate().take(m) {
                        if i == j {
                            continue;
                        }
                        let diff = lambda_i - lambda_j;
                        if diff.abs() < 1e-12 {
                            return Err(Error::Msg(
                                "Duplicate scale factors in Richardson extrapolation".into(),
                            ));
                        }
                        coeff *= -lambda_j / diff;
                    }

                    zero_noise_estimate += y_i * coeff;
                }

                Ok(zero_noise_estimate)
            }
            ExtrapolationModel::Exponential => {
                // For exponential decay E(lambda) = A + B e^{-gamma * lambda} on
                // uniformly spaced scale factors, an exact algebraic solution
                // exists on the first three points; the zero-noise value is
                // E(0) = A + B, not the lambda -> infinity asymptote A.
                if m >= 3 {
                    let l0 = self.scale_factors[0];
                    let h1 = self.scale_factors[1] - l0;
                    let h2 = self.scale_factors[2] - self.scale_factors[1];
                    if h1 > 1e-12 && (h2 - h1).abs() <= 1e-9 * h1.max(1.0) {
                        let y0 = noisy_values[0];
                        let y1 = noisy_values[1];
                        let y2 = noisy_values[2];
                        let d1 = y1 - y0;
                        let d2 = y2 - y1;
                        if d1.abs() <= 1e-12 && d2.abs() <= 1e-12 {
                            return Ok(y0); // constant signal: E(0) equals the data
                        }
                        if d1.abs() > 1e-12 {
                            let r = d2 / d1; // per-step decay ratio e^{-gamma * h}
                            if r > 1e-12 && r < 1.0 {
                                let denom = y0 + y2 - 2.0 * y1;
                                if denom.abs() > 1e-8 {
                                    let a = (y0 * y2 - y1 * y1) / denom;
                                    let e0 = a + (y0 - a) * r.powf(-l0 / h1);
                                    if e0.is_finite() {
                                        return Ok(e0);
                                    }
                                }
                            }
                        }
                    }
                }
                // Fall back to Richardson if non-exponential or ill-conditioned
                self.clone_with_model(ExtrapolationModel::Richardson)
                    .extrapolate(noisy_values)
            }
        }
    }

    fn clone_with_model(&self, model: ExtrapolationModel) -> Self {
        Self {
            scale_factors: self.scale_factors.clone(),
            model,
        }
    }
}

/// Clifford Data Regression (CDR) engine.
///
/// Learns a linear calibration mapping $E_{\text{mitigated}} = \alpha \cdot E_{\text{noisy}} + \beta$
/// by benchmarking noisy vs exact results on classically simulable Clifford training circuits.
#[derive(Debug, Clone)]
pub struct CliffordDataRegression {
    /// Calibration slope $\alpha$.
    pub alpha: f64,
    /// Calibration intercept $\beta$.
    pub beta: f64,
    /// Whether the regressor has been fitted on training pairs.
    pub is_fitted: bool,
}

impl CliffordDataRegression {
    /// Initializes an unfitted Clifford Data Regression model.
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            is_fitted: false,
        }
    }

    /// Fits calibration parameters $(\alpha, \beta)$ using pairs of $(E_{\text{noisy}}^{(k)}, E_{\text{exact}}^{(k)})$
    /// collected from near-Clifford training circuits.
    pub fn fit(&mut self, noisy_expectations: &[f64], exact_expectations: &[f64]) -> Result<()> {
        let n = noisy_expectations.len();
        if n < 2 || n != exact_expectations.len() {
            return Err(Error::Msg(format!(
                "CDR fit requires at least 2 matching pairs, got {n} noisy and {} exact",
                exact_expectations.len()
            )));
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_xy = 0.0;
        let n_f = n as f64;

        for (&x, &y) in noisy_expectations.iter().zip(exact_expectations.iter()) {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return Err(Error::Msg(
                "Degenerate noisy data in Clifford Data Regression fit".into(),
            ));
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        self.alpha = slope;
        self.beta = intercept;
        self.is_fitted = true;

        Ok(())
    }

    /// Applies the learned CDR calibration to mitigate a noisy target circuit expectation value:
    /// $$E_{\text{mitigated}} = \alpha \cdot E_{\text{target, noisy}} + \beta$$
    pub fn mitigate(&self, noisy_target_expectation: f64) -> Result<f64> {
        if !self.is_fitted {
            return Err(Error::Msg(
                "Cannot mitigate: CDR model is not fitted yet".into(),
            ));
        }
        Ok(self.alpha * noisy_target_expectation + self.beta)
    }
}

impl Default for CliffordDataRegression {
    fn default() -> Self {
        Self::new()
    }
}
