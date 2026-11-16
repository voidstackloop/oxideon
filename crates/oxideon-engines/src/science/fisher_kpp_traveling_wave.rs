//! Fisher-KPP Reaction-Diffusion Equation & Pulled Traveling Wavefront Speed (Fisher, 1937; Kolmogorov, Petrovsky & Piskunov, 1937).
//!
//! ### 1. Minimum Traveling Wave Speed & Bramson Logarithmic Delay
//! For $\partial_t u = D \partial_x^2 u + r u (1 - u)$, compactly supported initial data converge to a pulled traveling wave
//! propagating at the minimum speed $c_* = 2 \sqrt{D r}$ with spatial decay rate $\lambda_* = \sqrt{r / D}$.

use crate::error::{Error, Result};

/// Fisher-KPP Reaction-Diffusion Pulled Wavefront Engine.
#[derive(Debug, Clone, Copy)]
pub struct FisherKppWavefrontEngine {
    /// Diffusion coefficient $D > 0$.
    pub diffusion_d: f64,
    /// Intrinsic growth rate $r > 0$.
    pub growth_rate_r: f64,
}

impl Default for FisherKppWavefrontEngine {
    fn default() -> Self {
        Self {
            diffusion_d: 1.0,
            growth_rate_r: 1.0,
        }
    }
}

impl FisherKppWavefrontEngine {
    /// Creates a new Fisher-KPP Wavefront engine.
    pub fn new(diffusion_d: f64, growth_rate_r: f64) -> Result<Self> {
        if diffusion_d <= 0.0 || growth_rate_r <= 0.0 {
            return Err(Error::Msg("D and r must be > 0".to_string()));
        }
        Ok(Self {
            diffusion_d,
            growth_rate_r,
        })
    }

    /// Evaluates the minimum wave speed $c_* = 2\sqrt{D r}$, critical spatial steepness $\lambda_* = \sqrt{r/D}$, and Bramson front position $X(t) = c_* t - \frac{3}{2 \lambda_*} \ln t$ for $t > 1$.
    pub fn wave_speed_and_bramson_front(&self, time_t: f64) -> Result<(f64, f64, f64)> {
        if time_t <= 1.0 {
            return Err(Error::Msg(
                "time_t must be > 1 for Bramson asymptotics".to_string(),
            ));
        }
        let c_star = 2.0 * (self.diffusion_d * self.growth_rate_r).sqrt();
        let lambda_star = (self.growth_rate_r / self.diffusion_d).sqrt();
        let x_front = c_star * time_t - (1.5 / lambda_star) * time_t.ln();
        Ok((c_star, lambda_star, x_front))
    }
}
