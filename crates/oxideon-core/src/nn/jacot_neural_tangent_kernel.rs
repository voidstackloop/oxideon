//! Jacot-Gabriel-Hongler Neural Tangent Kernel (NTK) $\Theta(x, x')$ & Cho-Saul Arc-Cosine ReLU Kernels (Jacot et al., 2018; Cho & Saul, 2009).
//!
//! ### 1. Exact Infinite-Width ReLU Neural Tangent Kernel on the Unit Sphere
//! For a 2-layer infinite-width ReLU network with normalized inputs $\|x\| = \|x'\| = 1$ and cosine similarity $u = x^T x' \in [-1, 1]$,
//! the dual ReLU NNGP kernel $\Sigma^{(1)}(u)$ and Neural Tangent Kernel $\Theta^{(2)}(u)$ are given analytically by:
//! $$\kappa_0(u) = \frac{\pi - \arccos(u)}{2 \pi}, \qquad \kappa_1(u) = \frac{\sqrt{1 - u^2} + (\pi - \arccos u) u}{2 \pi}$$
//! $$\Theta^{(2)}(u) = u \, \kappa_0(u) + \kappa_1(u)$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Jacot-Gabriel-Hongler Infinite-Width Neural Tangent Kernel (NTK) Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct InfiniteWidthNtkEngine;

impl InfiniteWidthNtkEngine {
    /// Creates a new Infinite-Width NTK engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the exact 2-layer infinite-width ReLU NNGP kernel $\kappa_1(u)$ and NTK $\Theta(u)$ at cosine similarity $u \in [-1, 1]$.
    pub fn relu_nngp_and_ntk(cosine_u: f64) -> Result<(f64, f64)> {
        if !(-1.0..=1.0).contains(&cosine_u) {
            return Err(Error::Msg("cosine_u must be in [-1, 1]".to_string()));
        }
        let u = cosine_u.clamp(-1.0, 1.0);
        let theta = u.acos();
        let k0 = (PI - theta) / (2.0 * PI);
        let k1 = ((1.0 - u * u).max(0.0).sqrt() + (PI - theta) * u) / (2.0 * PI);
        let ntk = u * k0 + k1;
        Ok((k1, ntk))
    }
}
