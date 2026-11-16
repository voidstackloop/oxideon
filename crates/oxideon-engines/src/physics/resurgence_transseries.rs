//! Écalle Resurgence Theory, Borel-Padé Resummation, Stokes Phenomena & Instanton Transseries (Écalle, 1981).
//!
//! ### 1. Borel Transform $\mathcal{B}[\Phi]$ of Factorially Divergent Perturbation Series
//! Given a Gevrey-1 factorially divergent asymptotic series $\Phi(g) = \sum_{n=0}^\infty a_n g^n$ with $a_n \sim n!$,
//! the **Borel Transform** divides out the factorial divergence:
//! $$\mathcal{B}\[\Phi\](t) = \sum_{n=0}^\infty \frac{a_n}{n!} t^n$$
//! For the Euler-Stieltjes series $a_n = (-1)^n n!$, $\mathcal{B}\[\Phi\](t) = \sum_{n=0}^\infty (-t)^n = \frac{1}{1 + t}$,
//! whose [1/1] Padé approximant recovers the exact Borel transform from just the first 3 coefficients $[a_0, a_1, a_2]$!

use crate::error::{Error, Result};

/// Écalle Borel-Padé Resummation & Instanton Transseries Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct BorelResurgenceEngine;

impl BorelResurgenceEngine {
    /// Creates a new Borel Resurgence engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the $[1/1]$ Borel-Padé transform $P_{[1/1]}(t) = \frac{b_0 + (b_1 - b_0 q_1) t}{1 - q_1 t}$
    /// from the first 3 perturbation coefficients $[a_0, a_1, a_2]$ where $b_n = a_n / n!$.
    pub fn borel_pade_11_transform(coeffs: [f64; 3], t: f64) -> Result<f64> {
        let b0 = coeffs[0];
        let b1 = coeffs[1];
        let b2 = 0.5 * coeffs[2];
        if b1.abs() < 1e-15 {
            return Err(Error::Msg(
                "b1 must be non-zero for [1/1] Pade approximant".to_string(),
            ));
        }
        let q1 = b2 / b1;
        let p0 = b0;
        let p1 = b1 - b0 * q1;
        let denom = 1.0 - q1 * t;
        if denom.abs() < 1e-15 {
            return Err(Error::Msg(
                "Stokes singularity encountered on Borel integration contour".to_string(),
            ));
        }
        Ok((p0 + p1 * t) / denom)
    }

    /// Evaluates a one-instanton resurgent transseries $\Phi(g) = \Phi_0(g) + \sigma_1 e^{-S / g} \Phi_1(g)$.
    pub fn evaluate_one_instanton_transseries(
        g: f64,
        action_s: f64,
        stokes_sigma: f64,
        pert0: f64,
        pert1: f64,
    ) -> Result<f64> {
        if g <= 0.0 || action_s <= 0.0 {
            return Err(Error::Msg(
                "Coupling g and instanton action S must be > 0".to_string(),
            ));
        }
        Ok(pert0 + stokes_sigma * (-action_s / g).exp() * pert1)
    }
}
