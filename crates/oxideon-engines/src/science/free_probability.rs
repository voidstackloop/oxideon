//! Free Probability Theory, Voiculescu $R$-Transform, Free Additive Convolution $\boxplus$ &
//! Marchenko-Pastur Random Matrix Spectra (Voiculescu, 1986).
//!
//! ### 1. Wigner Semicircle Law & Free Cumulants
//! In Voiculescu's Free Probability Theory, classical independence is replaced by **free independence**
//! of non-commuting operators in a tracial $W^*$-probability space $(\mathcal{A}, \tau)$.
//! The moment-cumulant relation uses the lattice of **non-crossing partitions** $NC(n)$:
//! • $\kappa_1 = m_1$
//! • $\kappa_2 = m_2 - m_1^2$
//! • $\kappa_3 = m_3 - 3 m_2 m_1 + 2 m_1^3$
//! • $\kappa_4 = m_4 - 4 m_3 m_1 - 2 m_2^2 + 10 m_2 m_1^2 - 5 m_1^4$
//! For Wigner's Semicircle Law of variance $\sigma^2$, $\kappa_2 = \sigma^2$ and $\kappa_n = 0$ for all $n \neq 2$!
//!
//! ### 2. Voiculescu $R$-Transform & Free Additive Convolution $A \boxplus B$
//! The $R$-transform $R_A(z) = \sum_{n=1}^\infty \kappa_n(A) z^{n-1}$ linearizes free addition:
//! $$\kappa_n(A \boxplus B) = \kappa_n(A) + \kappa_n(B)$$
//!
//! ### 3. Marchenko-Pastur Law for Deep Neural Weight Matrices
//! For an $M \times N$ random weight matrix $W$ with aspect ratio $\gamma = M / N \in (0, 1]$ and element variance $\sigma^2 / N$,
//! the spectral edges of $W W^T$ are $\lambda_\pm = \sigma^2 (1 \pm \sqrt{\gamma})^2$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Voiculescu Free Probability, Non-Crossing Cumulant & Marchenko-Pastur Spectral Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct FreeProbabilityEngine;

impl FreeProbabilityEngine {
    /// Creates a new free probability engine.
    pub fn new() -> Self {
        Self
    }

    /// Converts the first 4 raw moments $[m_1, m_2, m_3, m_4]$ into exact free cumulants $[\kappa_1, \kappa_2, \kappa_3, \kappa_4]$
    /// over the lattice of non-crossing partitions $NC(4)$.
    pub fn moments_to_free_cumulants(moments: [f64; 4]) -> [f64; 4] {
        let [m1, m2, m3, m4] = moments;
        let k1 = m1;
        let k2 = m2 - m1 * m1;
        let k3 = m3 - 3.0 * m2 * m1 + 2.0 * m1.powi(3);
        let k4 = m4 - 4.0 * m3 * m1 - 2.0 * m2 * m2 + 10.0 * m2 * m1 * m1 - 5.0 * m1.powi(4);
        [k1, k2, k3, k4]
    }

    /// Converts free cumulants $[\kappa_1, \kappa_2, \kappa_3, \kappa_4]$ back to raw moments $[m_1, m_2, m_3, m_4]$.
    pub fn free_cumulants_to_moments(cumulants: [f64; 4]) -> [f64; 4] {
        let [k1, k2, k3, k4] = cumulants;
        let m1 = k1;
        let m2 = k2 + k1 * k1;
        let m3 = k3 + 3.0 * k2 * k1 + k1.powi(3);
        let m4 = k4 + 4.0 * k3 * k1 + 2.0 * k2 * k2 + 6.0 * k2 * k1 * k1 + k1.powi(4);
        [m1, m2, m3, m4]
    }

    /// Computes the moments of the free additive convolution $A \boxplus B$ given the first 4 moments of $A$ and $B$.
    pub fn free_additive_convolution_moments(moments_a: [f64; 4], moments_b: [f64; 4]) -> [f64; 4] {
        let ka = Self::moments_to_free_cumulants(moments_a);
        let kb = Self::moments_to_free_cumulants(moments_b);
        let k_sum = [ka[0] + kb[0], ka[1] + kb[1], ka[2] + kb[2], ka[3] + kb[3]];
        Self::free_cumulants_to_moments(k_sum)
    }

    /// Evaluates Wigner's Semicircle spectral density $\rho_{\text{SC}}(x) = \frac{1}{2\pi \sigma^2}\sqrt{\max(0, 4\sigma^2 - x^2)}$.
    pub fn wigner_semicircle_density(x: f64, sigma: f64) -> Result<f64> {
        if sigma <= 0.0 {
            return Err(Error::Msg("sigma must be > 0".to_string()));
        }
        let r2 = 4.0 * sigma * sigma;
        if x * x >= r2 {
            Ok(0.0)
        } else {
            Ok((r2 - x * x).sqrt() / (2.0 * PI * sigma * sigma))
        }
    }

    /// Computes the Marchenko-Pastur spectral bulk edges $(\lambda_-, \lambda_+)$ and density at $\lambda$
    /// for aspect ratio $\gamma \in (0, 1]$ and variance $\sigma^2$.
    pub fn marchenko_pastur_spectrum(
        lambda: f64,
        gamma: f64,
        sigma: f64,
    ) -> Result<(f64, f64, f64)> {
        if gamma <= 0.0 || sigma <= 0.0 {
            return Err(Error::Msg("gamma and sigma must be > 0".to_string()));
        }
        let s2 = sigma * sigma;
        let l_minus = s2 * (1.0 - gamma.sqrt()).powi(2);
        let l_plus = s2 * (1.0 + gamma.sqrt()).powi(2);

        let density = if lambda <= l_minus || lambda >= l_plus {
            0.0
        } else {
            ((l_plus - lambda) * (lambda - l_minus)).sqrt() / (2.0 * PI * s2 * gamma * lambda)
        };
        Ok((l_minus, l_plus, density))
    }
}
