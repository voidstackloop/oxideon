//! Causal Dynamical Triangulations (CDT), Regge Action & Ultraviolet Spectral Dimension Flow $d_S(\sigma)$ (Ambjørn, Jurkiewicz & Loll, 2005).
//!
//! ### 1. Dynamical Dimensional Reduction in Quantum Gravity
//! In 4D Causal Dynamical Triangulations (CDT), the effective spacetime **spectral dimension** $d_S(\sigma) = -2 \frac{d \ln P(\sigma)}{d \ln \sigma}$
//! measured by a diffusion process of diffusion time $\sigma$ flows smoothly from the ultraviolet (Planckian) value
//! $d_S(\sigma \to 0) = 2$ to the classical General Relativity value $d_S(\sigma \to \infty) = 4$:
//! $$d_S(\sigma) = 4 - \frac{2}{1 + b \, \sigma}$$

use crate::error::{Error, Result};

/// Causal Dynamical Triangulations (CDT) Regge Action & Spectral Dimension Engine.
#[derive(Debug, Clone, Copy)]
pub struct CausalTriangulationEngine {
    /// Inverse Newton coupling $k_0 > 0$.
    pub k0: f64,
    /// Cosmological constant parameter $\Delta > 0$.
    pub delta_cosmological: f64,
}

impl CausalTriangulationEngine {
    /// Creates a new CDT quantum gravity engine.
    pub fn new(k0: f64, delta_cosmological: f64) -> Result<Self> {
        if k0 <= 0.0 || delta_cosmological <= 0.0 {
            return Err(Error::Msg(
                "k0 and delta_cosmological must be > 0".to_string(),
            ));
        }
        Ok(Self {
            k0,
            delta_cosmological,
        })
    }

    /// Evaluates the Wick-rotated Euclidean Regge action $S_E(N_0, N_{4}^{(4,1)}, N_4^{(3,2)})$ on a causal 4-manifold triangulation.
    pub fn regge_action(&self, n0: usize, n4_41: usize, n4_32: usize) -> f64 {
        let n4 = (n4_41 as f64) + (n4_32 as f64);
        -(self.k0 + 6.0 * self.delta_cosmological) * (n0 as f64)
            + self.k0 * (n4 + (n4_41 as f64) + 2.0 * (n4_32 as f64))
    }

    /// Evaluates the CDT running spectral dimension $d_S(\sigma) = 4 - \frac{2}{1 + b \sigma} \in [2, 4)$ at diffusion scale $\sigma \ge 0$.
    pub fn spectral_dimension(&self, diffusion_sigma: f64, scale_b: f64) -> Result<f64> {
        if diffusion_sigma < 0.0 || scale_b <= 0.0 {
            return Err(Error::Msg(
                "diffusion_sigma must be >= 0 and scale_b > 0".to_string(),
            ));
        }
        Ok(4.0 - 2.0 / (1.0 + scale_b * diffusion_sigma))
    }
}
