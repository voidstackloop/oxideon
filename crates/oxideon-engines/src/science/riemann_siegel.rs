//! Riemann-Siegel Formula, Hardy's $Z$-Function $Z(t) = e^{i \vartheta(t)} \zeta(1/2 + it)$ & Gram Points (Siegel, 1932).
//!
//! ### 1. Riemann-Siegel Theta Function $\vartheta(t)$ & Hardy's $Z(t)$
//! On the critical line $\operatorname{Re}(s) = 1/2$, Hardy's real-valued $Z$-function satisfies $|Z(t)| = |\zeta(1/2 + i t)|$,
//! so sign changes of $Z(t)$ locate zeros of $\zeta(s)$ on the critical line:
//! $$\vartheta(t) \approx \frac{t}{2} \ln\left(\frac{t}{2 \pi}\right) - \frac{t}{2} - \frac{\pi}{8} + \frac{1}{48 t}$$
//! $$Z(t) \approx 2 \sum_{n=1}^{\lfloor \sqrt{t / (2\pi)} \rfloor} \frac{\cos(\vartheta(t) - t \ln n)}{\sqrt{n}}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Riemann-Siegel Critical-Line Zeta & Hardy $Z(t)$ Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct RiemannSiegelZEngine;

impl RiemannSiegelZEngine {
    /// Creates a new Riemann-Siegel $Z$-function engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the asymptotic Riemann-Siegel phase $\vartheta(t)$ for $t \ge 2 \pi$.
    pub fn riemann_siegel_theta(t: f64) -> Result<f64> {
        if !t.is_finite() || t < 2.0 * PI {
            return Err(Error::Msg(
                "t must be finite and >= 2*pi for Riemann-Siegel asymptotic expansion".to_string(),
            ));
        }
        Ok(0.5 * t * (t / (2.0 * PI)).ln() - 0.5 * t - PI / 8.0 + 1.0 / (48.0 * t))
    }

    /// Evaluates the main sum of Hardy's $Z$-function $Z(t)$ at height $t \ge 2 \pi$.
    pub fn hardy_z_main_sum(t: f64) -> Result<f64> {
        let theta = Self::riemann_siegel_theta(t)?;
        let n_max = (t / (2.0 * PI)).sqrt().floor() as usize;
        crate::limits::check_iterations("sqrt(t / 2 pi)", n_max)?;
        let mut sum = 0.0;
        for n in 1..=n_max.max(1) {
            let nf = n as f64;
            sum += 2.0 * (theta - t * nf.ln()).cos() / nf.sqrt();
        }
        Ok(sum)
    }
}
