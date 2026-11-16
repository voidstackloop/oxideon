//! Grad-Shafranov Axisymmetric Tokamak MHD Equilibrium, Solov'ev Flux Surfaces & Kruskal-Shafranov Safety Factor $q(r)$ (Grad & Rubin, 1958; Shafranov, 1966).
//!
//! ### 1. Solov'ev Analytical Solution of the Grad-Shafranov Equation $\Delta^* \psi = -\mu_0 R^2 p'(\psi) - F F'(\psi)$
//! With linear source profiles, Solov'ev's exact analytical poloidal flux surface $\psi(R, Z)$ around magnetic axis $(R_0, 0)$
//! with elongation $\kappa > 0$ is:
//! $$\psi(R, Z) = \frac{\psi_0}{2 R_0^4} \left[\left(R^2 - R_0^2\right)^2 + \frac{4 R_0^2 Z^2}{\kappa^2}\right]$$
//! and the Kruskal-Shafranov kink stability criterion requires safety factor $q(a) = \frac{a B_\phi}{R_0 B_\theta} > 1$.

use crate::error::{Error, Result};

/// Grad-Shafranov Tokamak Equilibrium & Kruskal-Shafranov Safety Factor Engine.
#[derive(Debug, Clone, Copy)]
pub struct GradShafranovTokamakEngine {
    /// Major radius of magnetic axis $R_0 > 0$.
    pub major_r0: f64,
    /// Plasma vertical elongation $\kappa > 0$.
    pub elongation_kappa: f64,
    /// Toroidal magnetic field $B_\phi > 0$ at $R_0$.
    pub toroidal_b0: f64,
}

impl GradShafranovTokamakEngine {
    /// Creates a new Grad-Shafranov Tokamak MHD Equilibrium engine.
    pub fn new(major_r0: f64, elongation_kappa: f64, toroidal_b0: f64) -> Result<Self> {
        if major_r0 <= 0.0 || elongation_kappa <= 0.0 || toroidal_b0 <= 0.0 {
            return Err(Error::Msg(
                "All tokamak equilibrium parameters must be > 0".to_string(),
            ));
        }
        Ok(Self {
            major_r0,
            elongation_kappa,
            toroidal_b0,
        })
    }

    /// Evaluates the normalized Solov'ev poloidal magnetic flux $\psi(R, Z) / \psi_0$ (vanishes on the magnetic axis $(R_0, 0)$).
    pub fn solovev_normalized_flux(&self, r: f64, z: f64) -> f64 {
        let r0_2 = self.major_r0 * self.major_r0;
        let dr2 = r * r - r0_2;
        0.5 / (r0_2 * r0_2)
            * (dr2 * dr2 + 4.0 * r0_2 * z * z / (self.elongation_kappa * self.elongation_kappa))
    }

    /// Computes the cylindrical Kruskal-Shafranov safety factor $q(r) = \frac{r B_\phi}{R_0 B_\theta} \frac{1 + \kappa^2}{2 \kappa}$.
    pub fn safety_factor_q(&self, minor_r: f64, poloidal_b_theta: f64) -> Result<f64> {
        if minor_r <= 0.0 || poloidal_b_theta <= 0.0 {
            return Err(Error::Msg(
                "minor_r and poloidal_b_theta must be > 0".to_string(),
            ));
        }
        let shape_factor =
            0.5 * (1.0 + self.elongation_kappa * self.elongation_kappa) / self.elongation_kappa;
        Ok((minor_r * self.toroidal_b0) / (self.major_r0 * poloidal_b_theta) * shape_factor)
    }
}
