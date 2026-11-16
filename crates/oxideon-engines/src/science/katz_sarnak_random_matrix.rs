//! Montgomery-Odlyzko GUE Pair Correlation of Zeta Zeros & Katz-Sarnak Random Matrix Symmetry Types (Montgomery, 1973; Katz & Sarnak, 1999).
//!
//! ### 1. Montgomery-Odlyzko GUE Pair Correlation Kernel $R_2(u)$
//! Normalized spacings of non-trivial zeros $\frac{1}{2} + i \gamma_n$ of $\zeta(s)$ obey the Dyson Gaussian Unitary Ensemble (GUE)
//! 2-point correlation function:
//! $$R_2^{\text{GUE}}(u) = 1 - \left(\frac{\sin(\pi u)}{\pi u}\right)^2$$
//! exhibiting quadratic level repulsion $R_2(u) \approx \frac{\pi^2}{3} u^2$ as $u \to 0$.

use std::f64::consts::PI;

/// Montgomery-Odlyzko GUE Pair Correlation & Katz-Sarnak Monodromy Density Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct KatzSarnakSymmetryEngine;

impl KatzSarnakSymmetryEngine {
    /// Creates a new Katz-Sarnak Random Matrix Symmetry engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates Montgomery's GUE 2-point pair correlation function $R_2(u) = 1 - \left(\frac{\sin(\pi u)}{\pi u}\right)^2$.
    pub fn montgomery_gue_pair_correlation(u: f64) -> f64 {
        if u.abs() < 1e-8 {
            (PI * PI / 3.0) * u * u
        } else {
            let x = PI * u;
            let sinc = x.sin() / x;
            1.0 - sinc * sinc
        }
    }

    /// Evaluates the Katz-Sarnak 1-level density kernel $W_{\mathrm{USp}}(x) = 1 - \frac{\sin(2\pi x)}{2\pi x}$ for symplectic $L$-function families.
    pub fn usp_one_level_density(x: f64) -> f64 {
        if x.abs() < 1e-8 {
            0.0
        } else {
            let u = 2.0 * PI * x;
            1.0 - u.sin() / u
        }
    }
}
