//! Belkin-Hastie-Montanari Double-Descent Risk Curve & Benign Overparameterization at Interpolation Threshold $\gamma = p / n$ (Belkin et al., 2019; Hastie et al., 2022).
//!
//! ### 1. Exact Asymptotic Risk of Minimum-Norm Ridgeless Interpolation ($p / n \to \gamma$)
//! For isotropic Gaussian features $x_i \sim \mathcal{N}(0, I_p)$ with signal norm $r^2 = \|\beta_*\|_2^2$ and noise variance $\sigma^2$,
//! the out-of-sample excess risk $R(\gamma) = \mathbb{E}[(x^T \hat{\beta} - x^T \beta_*)^2]$ of the Moore-Penrose pseudoinverse estimator
//! $\hat{\beta} = X^+ y$ satisfies the exact **Hastie-Montanari-Rosset-Tibshirani Formula**:
//! • **Underparameterized ($\gamma < 1$)**: $R(\gamma) = \sigma^2 \frac{\gamma}{1 - \gamma}$ (diverges at $\gamma \to 1$!)
//! • **Overparameterized ($\gamma > 1$)**: $R(\gamma) = r^2 \left(1 - \frac{1}{\gamma}\right) + \sigma^2 \frac{1}{\gamma - 1}$ (variance *decreases* with $\gamma$!).

use crate::error::{Error, Result};

/// Belkin-Hastie Ridgeless Interpolation & Double-Descent Risk Engine.
#[derive(Debug, Clone, Copy)]
pub struct OverparameterizedDoubleDescent {
    /// True signal squared norm $r^2 = \|\beta_*\|_2^2 \ge 0$.
    pub signal_r2: f64,
    /// Additive observation noise variance $\sigma^2 \ge 0$.
    pub noise_sigma2: f64,
}

impl OverparameterizedDoubleDescent {
    /// Creates a new Overparameterized Double-Descent engine.
    pub fn new(signal_r2: f64, noise_sigma2: f64) -> Result<Self> {
        if signal_r2 < 0.0 || noise_sigma2 < 0.0 {
            return Err(Error::Msg(
                "signal_r2 and noise_sigma2 must be >= 0".to_string(),
            ));
        }
        Ok(Self {
            signal_r2,
            noise_sigma2,
        })
    }

    /// Evaluates the exact asymptotic out-of-sample excess risk $R(\gamma)$ at aspect ratio $\gamma = p / n \ne 1$.
    pub fn asymptotic_ridgeless_risk(&self, aspect_gamma: f64) -> Result<f64> {
        if aspect_gamma <= 0.0 || (aspect_gamma - 1.0).abs() < 1e-9 {
            return Err(Error::Msg(
                "aspect_gamma must be > 0 and != 1.0 (interpolation singularity)".to_string(),
            ));
        }
        if aspect_gamma < 1.0 {
            Ok(self.noise_sigma2 * aspect_gamma / (1.0 - aspect_gamma))
        } else {
            let bias = self.signal_r2 * (1.0 - 1.0 / aspect_gamma);
            let var = self.noise_sigma2 / (aspect_gamma - 1.0);
            Ok(bias + var)
        }
    }
}
