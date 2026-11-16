//! Chirikov-Taylor Symplectic Area-Preserving Standard Map, Resonance Overlap & Greene's Residue $K_c \approx 0.971635$ (Chirikov, 1979; Greene, 1979).
//!
//! ### 1. Symplectic Twist Map on the Cylinder $\mathbb{T} \times \mathbb{R}$
//! The Chirikov-Taylor standard map evolves action-angle coordinates $(\theta_n, p_n)$ with kick strength $K \ge 0$ via:
//! $$p_{n+1} = p_n + K \sin(\theta_n), \qquad \theta_{n+1} = (\theta_n + p_{n+1}) \bmod 2 \pi$$
//! With exact symplectic Jacobian $\det \frac{\partial (\theta_{n+1}, p_{n+1})}{\partial (\theta_n, p_n)} \equiv 1$. Global stochasticity
//! (destruction of the last golden-mean KAM barrier) occurs at Greene's critical parameter $K_c \approx 0.971635406$.

use std::f64::consts::PI;

use crate::error::Result;
use crate::limits::check_iterations;

/// Chirikov-Taylor Symplectic Standard Map & Resonance Overlap Engine.
#[derive(Debug, Clone, Copy)]
pub struct ChirikovStandardMapEngine {
    /// Nonlinear kick stochasticity parameter $K \ge 0$.
    pub kick_k: f64,
}

impl ChirikovStandardMapEngine {
    /// Creates a new Chirikov Standard Map engine.
    pub fn new(kick_k: f64) -> Self {
        Self { kick_k }
    }

    /// Iterates the symplectic Chirikov standard map for `steps` steps and returns `(theta_final, p_final, is_above_greene_kc)`.
    /// `steps` is limited to [`MAX_ITERATIONS`](crate::limits::MAX_ITERATIONS) per call.
    pub fn iterate_map(
        &self,
        mut theta: f64,
        mut p: f64,
        steps: usize,
    ) -> Result<(f64, f64, bool)> {
        check_iterations("steps", steps)?;
        let two_pi = 2.0 * PI;
        for _ in 0..steps {
            p += self.kick_k * theta.sin();
            theta = (theta + p).rem_euclid(two_pi);
        }
        Ok((theta, p, self.kick_k > 0.971_635_406))
    }
}
