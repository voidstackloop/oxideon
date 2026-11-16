//! Sato-Tate Semicircle Equidistribution Theorem $\mu_{\mathrm{ST}} = \frac{2}{\pi} \sin^2 \theta \, d\theta$ (Sato & Tate, 1963; Barnet-Lamb, Geraghty, Harris & Taylor, 2011).
//!
//! ### 1. Non-CM Elliptic Curve Frobenius Angle Measure on $[0, \pi]$
//! For a non-CM elliptic curve $E / \mathbb{Q}$, the Frobenius angles $\theta_p \in [0, \pi]$ defined by $a_p(E) = 2\sqrt{p}\cos \theta_p$ are equidistributed with respect to the $SU(2)$ Haar pushforward measure:
//! $$\mu_{\mathrm{ST}}([0, \phi]) = \frac{2}{\pi} \int_0^\phi \sin^2 \theta \, d\theta = \frac{\phi - \frac{1}{2}\sin(2\phi)}{\pi}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Sato-Tate Semicircle Equidistribution Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SatoTateSemicircleEquidistribution;

impl SatoTateSemicircleEquidistribution {
    /// Creates a new Sato-Tate Equidistribution engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the Sato-Tate density $\frac{2}{\pi}\sin^2 \phi$ and cumulative distribution function $\mu_{\mathrm{ST}}([0, \phi])$ for $\phi \in [0, \pi]$.
    pub fn density_and_cdf(phi_rad: f64) -> Result<(f64, f64)> {
        if !(0.0..=PI).contains(&phi_rad) {
            return Err(Error::Msg("phi_rad must be in [0, pi]".to_string()));
        }
        let pdf = (2.0 / PI) * phi_rad.sin().powi(2);
        let cdf = (phi_rad - 0.5 * (2.0 * phi_rad).sin()) / PI;
        Ok((pdf, cdf))
    }
}
