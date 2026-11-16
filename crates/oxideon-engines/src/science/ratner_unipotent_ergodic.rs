//! Ratner Measure Rigidity Theorem, Unipotent Horocycle Flows on $SL(2, \mathbb{R})/\Gamma$ & Equidistribution (Ratner, 1991).
//!
//! ### 1. Unipotent Horocycle Flow $u_t = \begin{pmatrix} 1 & t \\ 0 & 1 \end{pmatrix}$
//! Every ergodic invariant probability measure for a unipotent flow $u_t$ on $G/\Gamma$ is algebraic (homogeneous),
//! and quantitative horocycle ergodic averages converge at power-law rate $O(T^{-\kappa})$ controlled by the spectral gap $s_1(1 - s_1)$ of $X = \mathbb{H}/\Gamma$.

use crate::error::{Error, Result};

/// Ratner Unipotent Horocycle Flow & Homogeneous Equidistribution Engine.
#[derive(Debug, Clone, Copy)]
pub struct RatnerUnipotentFlowEngine {
    /// First non-zero Laplacian eigenvalue $\lambda_1 > 0$ on $\mathbb{H}/\Gamma$.
    pub laplacian_lambda1: f64,
}

impl Default for RatnerUnipotentFlowEngine {
    fn default() -> Self {
        Self {
            laplacian_lambda1: 0.25,
        }
    }
}

impl RatnerUnipotentFlowEngine {
    /// Creates a new Ratner Unipotent Flow engine.
    pub fn new(laplacian_lambda1: f64) -> Result<Self> {
        if laplacian_lambda1 <= 0.0 || !laplacian_lambda1.is_finite() {
            return Err(Error::Msg("laplacian_lambda1 must be > 0".to_string()));
        }
        Ok(Self { laplacian_lambda1 })
    }

    /// Evaluates the Burger-Sarnak-Strömbergsson horocycle equidistribution decay exponent $\kappa$ and error bound $T^{-\kappa}$.
    pub fn equidistribution_exponent_and_bound(
        &self,
        horocycle_length_t: f64,
    ) -> Result<(f64, f64)> {
        if horocycle_length_t <= 1.0 {
            return Err(Error::Msg("horocycle_length_t must be > 1".to_string()));
        }
        let kappa = if self.laplacian_lambda1 >= 0.25 {
            0.5
        } else {
            0.5 * (1.0 - (1.0 - 4.0 * self.laplacian_lambda1).sqrt())
        };
        Ok((kappa, horocycle_length_t.powf(-kappa)))
    }
}
