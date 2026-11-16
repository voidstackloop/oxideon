//! Glauber Second-Order Coherence $g^{(2)}(0)$, Photon Antibunching & Mandel $Q_M$-Parameter (Glauber, 1963; Mandel, 1979).
//!
//! ### 1. Zero-Delay Second-Order Coherence & Sub-Poissonian Criterion
//! Given photon number mean $\langle n \rangle$ and variance $(\Delta n)^2$:
//! $$Q_M = \frac{(\Delta n)^2 - \langle n \rangle}{\langle n \rangle}, \qquad g^{(2)}(0) = 1 + \frac{Q_M}{\langle n \rangle} = \frac{\langle n(n - 1)\rangle}{\langle n \rangle^2}$$
//! Sub-Poissonian nonclassical light satisfies $Q_M < 0$ and $g^{(2)}(0) < 1$.

use crate::error::{Error, Result};

/// Glauber $g^{(2)}(0)$ Photon Coherence & Mandel Sub-Poissonian Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct GlauberPhotonCoherenceEngine;

impl GlauberPhotonCoherenceEngine {
    /// Creates a new Glauber Photon Coherence engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the Mandel parameter $Q_M$, zero-delay second-order correlation $g^{(2)}(0)$, and single-photon emitter check ($g^{(2)}(0) < 0.5$).
    pub fn mandel_q_and_g2_zero(mean_n: f64, var_n: f64) -> Result<(f64, f64, bool)> {
        if mean_n <= 0.0 || var_n < 0.0 {
            return Err(Error::Msg("mean_n > 0 and var_n >= 0 required".to_string()));
        }
        let q_m = (var_n - mean_n) / mean_n;
        let g2_0 = (1.0 + q_m / mean_n).max(0.0);
        Ok((q_m, g2_0, g2_0 < 0.5))
    }
}
