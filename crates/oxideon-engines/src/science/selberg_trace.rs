//! Selberg Trace Formula on Hyperbolic Surfaces $\Gamma \backslash \mathbb{H}$ & Selberg Zeta Function $Z_\Gamma(s)$ (Selberg, 1956).
//!
//! ### 1. Selberg Zeta Euler Product over Prime Closed Geodesics
//! For a compact hyperbolic surface $X = \Gamma \backslash \mathbb{H}$ of genus $g \ge 2$ (area $\mathrm{Area}(X) = 4 \pi (g - 1)$),
//! the Selberg Zeta function over primitive closed geodesics $\gamma_0$ of length $\ell(\gamma_0)$ is:
//! $$Z_\Gamma(s) = \prod_{\gamma_0} \prod_{k=0}^\infty \left(1 - e^{-(s + k) \ell(\gamma_0)}\right), \qquad \operatorname{Re}(s) > 1$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Selberg Trace Formula & Hyperbolic Geodesic Zeta Engine.
#[derive(Debug, Clone)]
pub struct SelbergTraceFormulaEngine {
    /// Genus $g \ge 2$ of the hyperbolic surface.
    pub genus_g: usize,
    /// Truncated list of primitive closed geodesic lengths $\ell(\gamma_0) > 0$.
    pub prime_geodesic_lengths: Vec<f64>,
}

impl SelbergTraceFormulaEngine {
    /// Creates a new Selberg Trace Formula engine.
    pub fn new(genus_g: usize, prime_geodesic_lengths: Vec<f64>) -> Result<Self> {
        if genus_g < 2
            || prime_geodesic_lengths.is_empty()
            || prime_geodesic_lengths.iter().any(|&l| l <= 0.0)
        {
            return Err(Error::Msg(
                "genus_g >= 2 and positive geodesic lengths required".to_string(),
            ));
        }
        Ok(Self {
            genus_g,
            prime_geodesic_lengths,
        })
    }

    /// Computes the Gauss-Bonnet hyperbolic area $\mathrm{Area}(X) = 4 \pi (g - 1)$.
    pub fn hyperbolic_area(&self) -> f64 {
        4.0 * PI * ((self.genus_g - 1) as f64)
    }

    /// Evaluates the truncated Selberg Zeta Euler product $Z_\Gamma(s)$ for real $s > 1$ up to $k \le k_{\max}$.
    pub fn selberg_zeta_product(&self, s: f64, k_max: usize) -> Result<f64> {
        if s <= 1.0 {
            return Err(Error::Msg(
                "Real s must be > 1.0 for Euler product convergence".to_string(),
            ));
        }
        let mut prod = 1.0;
        for &len in &self.prime_geodesic_lengths {
            for k in 0..=k_max {
                prod *= 1.0 - (-(s + (k as f64)) * len).exp();
            }
        }
        Ok(prod)
    }
}
