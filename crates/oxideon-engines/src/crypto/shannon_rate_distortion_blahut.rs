//! Shannon Rate-Distortion Theory $R(D)$, Reverse Water-Filling & Blahut-Arimoto Optimization (Shannon, 1959; Blahut, 1972).
//!
//! ### 1. Gaussian Source Rate-Distortion & Kolmogorov Reverse Water-Filling
//! For $m$ independent parallel Gaussian sources $X_i \sim \mathcal{N}(0, \sigma_i^2)$ under total squared-error distortion $D > 0$,
//! Kolmogorov's **Reverse Water-Filling** allocates distortion $D_i = \min(\lambda, \sigma_i^2)$ and bit rate:
//! $$R(D) = \sum_{i=1}^m \max\left(0, \frac{1}{2} \log_2\left(\frac{\sigma_i^2}{\lambda}\right)\right)$$

use crate::error::{Error, Result};

/// Shannon Rate-Distortion & Reverse Water-Filling Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct RateDistortionBlahutArimoto;

impl RateDistortionBlahutArimoto {
    /// Creates a new Shannon Rate-Distortion engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the reverse water-filling rate $R$ (in bits) and total distortion $D = \sum_i \min(\lambda, \sigma_i^2)$ at water-level $\lambda > 0$.
    pub fn reverse_waterfilling_rate_and_distortion(
        variances: &[f64],
        water_level_lambda: f64,
    ) -> Result<(f64, f64)> {
        if variances.is_empty() || water_level_lambda <= 0.0 || variances.iter().any(|&v| v <= 0.0)
        {
            return Err(Error::Msg(
                "Positive variances and water_level_lambda > 0 required".to_string(),
            ));
        }
        let mut rate = 0.0;
        let mut dist = 0.0;
        for &v in variances {
            if v > water_level_lambda {
                rate += 0.5 * (v / water_level_lambda).log2();
                dist += water_level_lambda;
            } else {
                dist += v;
            }
        }
        Ok((rate, dist))
    }
}
