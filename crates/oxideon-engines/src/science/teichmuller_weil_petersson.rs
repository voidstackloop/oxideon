//! Teichmüller Space $\mathcal{T}_g$, Grötzsch Quasiconformal Dilatation $K$ & Wolpert Weil-Petersson Volume (Teichmüller, 1939; Mirzakhani, 2007).
//!
//! ### 1. Teichmüller Metric & Beltrami Coefficient $\mu$
//! Given a Beltrami differential $\mu(z) d\bar{z}/dz$ with $\|\mu\|_\infty = k < 1$, the maximal dilatation is
//! $K = \frac{1 + k}{1 - k}$, and the Teichmüller distance is $d_{\mathcal{T}} = \frac{1}{2} \ln K$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Teichmüller Quasiconformal Dilatation & Weil-Petersson Moduli Volume Engine.
#[derive(Debug, Clone, Copy)]
pub struct TeichmullerWeilPeterssonEngine {
    /// Surface genus $g \ge 1$.
    pub genus_g: usize,
}

impl Default for TeichmullerWeilPeterssonEngine {
    fn default() -> Self {
        Self { genus_g: 1 }
    }
}

impl TeichmullerWeilPeterssonEngine {
    /// Creates a new Teichmüller & Weil-Petersson engine.
    pub fn new(genus_g: usize) -> Result<Self> {
        if genus_g == 0 {
            return Err(Error::Msg("Surface genus g must be >= 1".to_string()));
        }
        Ok(Self { genus_g })
    }

    /// Evaluates the quasiconformal dilatation $K = (1 + k)/(1 - k)$ and Teichmüller distance $d_{\mathcal{T}} = \frac{1}{2} \ln K$ for Beltrami norm $k \in [0, 1)$.
    pub fn dilatation_and_teichmuller_distance(beltrami_norm_k: f64) -> Result<(f64, f64)> {
        if !(0.0..1.0).contains(&beltrami_norm_k) {
            return Err(Error::Msg("Beltrami norm k must be in [0, 1)".to_string()));
        }
        let k_dil = (1.0 + beltrami_norm_k) / (1.0 - beltrami_norm_k);
        Ok((k_dil, 0.5 * k_dil.ln()))
    }

    /// Evaluates the Mirzakhani Weil-Petersson volume $V_{1,1}(L) = \frac{\pi^2}{6} + \frac{L^2}{24}$ for a one-holed torus with geodesic boundary length $L$.
    pub fn mirzakhani_wp_volume_v11(boundary_length_l: f64) -> f64 {
        PI * PI / 6.0 + boundary_length_l * boundary_length_l / 24.0
    }
}
