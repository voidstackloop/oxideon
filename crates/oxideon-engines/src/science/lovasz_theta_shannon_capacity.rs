//! Lovász Theta Function $\vartheta(G)$ & Shannon Zero-Error Graph Capacity $\Theta(G)$ (Shannon, 1956; Lovász, 1979).
//!
//! ### 1. Sandwich Theorem $\alpha(G) \le \Theta(G) \le \vartheta(G) \le \bar{\chi}(G)$ & Pentagonal Capacity $\Theta(C_5) = \sqrt{5}$
//! For an odd cycle $C_n$, Lovász proved:
//! $$\vartheta(C_n) = \frac{n \cos(\pi / n)}{1 + \cos(\pi / n)}, \qquad \Theta(C_5) = \vartheta(C_5) = \sqrt{5}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Lovász Theta Function & Shannon Zero-Error Capacity Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct LovaszThetaShannonCapacity;

impl LovaszThetaShannonCapacity {
    /// Creates a new Lovász Theta & Shannon Capacity engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the exact Lovász theta $\vartheta(C_n)$ and independence number $\alpha(C_n) = (n - 1) / 2$ for an odd cycle $C_n$ ($n \ge 3$ odd).
    pub fn odd_cycle_lovasz_theta_and_alpha(n_odd: usize) -> Result<(f64, usize)> {
        if n_odd < 3 || n_odd % 2 == 0 {
            return Err(Error::Msg("n_odd must be an odd integer >= 3".to_string()));
        }
        let cos_pi_n = (PI / n_odd as f64).cos();
        let theta = (n_odd as f64 * cos_pi_n) / (1.0 + cos_pi_n);
        Ok((theta, (n_odd - 1) / 2))
    }
}
