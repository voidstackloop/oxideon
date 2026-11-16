//! Eshelby's 1957 Ellipsoidal Inclusion Theorem & Mori-Tanaka Micromechanical Homogenization (Eshelby, 1957; Mori & Tanaka, 1973).
//!
//! ### 1. Exact Spherical Eshelby Tensor Components $(\alpha_S, \beta_S)$
//! If an ellipsoidal subdomain $\Omega$ of an infinite isotropic elastic matrix (Poisson ratio $\nu$) undergoes a uniform
//! eigenstrain $\boldsymbol{\varepsilon}^*$, Eshelby proved that the constrained interior strain $\boldsymbol{\varepsilon}^{\mathrm{in}} = \mathbb{S} : \boldsymbol{\varepsilon}^*$
//! is **spatially uniform**, with spherical dilatational and deviatoric Eshelby factors:
//! $$\alpha_S(\nu) = \frac{1 + \nu}{3 (1 - \nu)}, \qquad \beta_S(\nu) = \frac{2 (4 - 5 \nu)}{15 (1 - \nu)}$$

use crate::error::{Error, Result};

/// Eshelby Ellipsoidal Inclusion & Mori-Tanaka Homogenization Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct EshelbyInclusionMoriTanaka;

impl EshelbyInclusionMoriTanaka {
    /// Creates a new Eshelby Inclusion engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes Eshelby's exact spherical volumetric factor $\alpha_S(\nu)$ and deviatoric shear factor $\beta_S(\nu)$ for $\nu \in (-1.0, 0.5)$.
    pub fn spherical_eshelby_factors(poisson_nu: f64) -> Result<(f64, f64)> {
        if !(-1.0..0.5).contains(&poisson_nu) {
            return Err(Error::Msg("poisson_nu must be in [-1.0, 0.5)".to_string()));
        }
        let denom = 1.0 - poisson_nu;
        let alpha_s = (1.0 + poisson_nu) / (3.0 * denom);
        let beta_s = 2.0 * (4.0 - 5.0 * poisson_nu) / (15.0 * denom);
        Ok((alpha_s, beta_s))
    }
}
