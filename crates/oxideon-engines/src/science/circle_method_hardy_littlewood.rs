//! Hardy-Littlewood-Ramanujan Circle Method, Weyl Exponential Sums & Waring's Problem (Hardy & Littlewood, 1920).
//!
//! ### 1. Weyl Exponential Sum $S_k(\alpha; N) = \sum_{x=1}^N e^{2 \pi i \alpha x^k}$
//! In the Hardy-Littlewood Circle Method for Waring's problem ($n = x_1^k + \dots + x_s^k$), the generating function
//! on the unit circle $\alpha \in [0, 1)$ is decomposed into **Major Arcs** $\mathfrak{M}$ (near rationals $a/q$ with small $q$)
//! and **Minor Arcs** $\mathfrak{m}$ (bounded via Weyl's inequality / Vinogradov's mean-value theorem).

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Hardy-Littlewood Circle Method & Weyl Exponential Sum Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardyLittlewoodCircleEngine;

impl HardyLittlewoodCircleEngine {
    /// Creates a new Hardy-Littlewood Circle Method engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the modulus $|S_k(\alpha; N)| = \left|\sum_{x=1}^N e^{2\pi i \alpha x^k}\right|$ of the degree-$k$ Weyl sum.
    pub fn weyl_exponential_sum_modulus(alpha: f64, power_k: u32, max_n: usize) -> Result<f64> {
        if power_k == 0 || max_n == 0 {
            return Err(Error::Msg("power_k and max_n must be >= 1".to_string()));
        }
        crate::limits::check_iterations("max_n", max_n)?;
        let k = i32::try_from(power_k)
            .map_err(|_| Error::Msg("power_k exceeds i32 range".to_string()))?;
        let mut re = 0.0;
        let mut im = 0.0;
        for x in 1..=max_n {
            let xk = (x as f64).powi(k);
            let phase = 2.0 * PI * alpha * xk;
            re += phase.cos();
            im += phase.sin();
        }
        Ok((re * re + im * im).sqrt())
    }
}
