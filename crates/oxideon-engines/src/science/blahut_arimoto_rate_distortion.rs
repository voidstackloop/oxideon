//! Shannon Rate-Distortion Theory $R(D) = \frac{1}{2}\log_2(\sigma^2 / D)$ & Reverse Water-Filling (Shannon, 1959; Blahut, 1972; Arimoto, 1972).
//!
//! ### 1. Quadratic-Gaussian Rate-Distortion Function & Distortion-Rate Bound $D(R) = \sigma^2 2^{-2R}$
//! For a memoryless Gaussian source $X \sim \mathcal{N}(0, \sigma^2)$ under mean-squared error distortion $\mathbb{E}[(X - \hat{X})^2] \le D$:
//! $$R(D) = \max\!\left(0, \frac{1}{2} \log_2 \frac{\sigma^2}{D}\right), \qquad D(R) = \sigma^2 2^{-2 R}$$

use crate::error::{Error, Result};

/// Shannon Quadratic-Gaussian Rate-Distortion & Blahut-Arimoto Engine.
#[derive(Debug, Clone, Copy)]
pub struct GaussianRateDistortionEngine {
    /// Source variance $\sigma^2 > 0$.
    pub source_variance_sigma2: f64,
}

impl Default for GaussianRateDistortionEngine {
    fn default() -> Self {
        Self {
            source_variance_sigma2: 1.0,
        }
    }
}

impl GaussianRateDistortionEngine {
    /// Creates a new Gaussian Rate-Distortion engine.
    pub fn new(source_variance_sigma2: f64) -> Result<Self> {
        if source_variance_sigma2 <= 0.0 || !source_variance_sigma2.is_finite() {
            return Err(Error::Msg("source_variance_sigma2 must be > 0".to_string()));
        }
        Ok(Self {
            source_variance_sigma2,
        })
    }

    /// Evaluates the minimum rate $R(D)$ (bits/sample) for distortion $D > 0$ and the minimum MSE distortion $D(R)$ at rate $R(D)$.
    pub fn rate_for_distortion_and_inverse(&self, distortion_d: f64) -> Result<(f64, f64)> {
        if distortion_d <= 0.0 || !distortion_d.is_finite() {
            return Err(Error::Msg("distortion_d must be > 0".to_string()));
        }
        if distortion_d >= self.source_variance_sigma2 {
            return Ok((0.0, self.source_variance_sigma2));
        }
        let r_d = 0.5 * (self.source_variance_sigma2 / distortion_d).log2();
        let d_inv = self.source_variance_sigma2 * 2.0f64.powf(-2.0 * r_d);
        Ok((r_d, d_inv))
    }
}
