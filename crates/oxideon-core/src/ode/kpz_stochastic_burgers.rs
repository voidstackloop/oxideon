//! Kardar-Parisi-Zhang (KPZ) Universality Class, Cole-Hopf Partition Function & Hairer Regularity Structures (Kardar, Parisi & Zhang, 1986; Hairer, 2014).
//!
//! ### 1. Cole-Hopf Transformation $Z(t, x) = \exp\left(\frac{\lambda}{2 \nu} h(t, x)\right)$
//! The 1+1D KPZ equation $\partial_t h = \nu \partial_x^2 h + \frac{\lambda}{2} (\partial_x h)^2 + \xi$ maps via the Cole-Hopf
//! transformation to the Multiplicative Stochastic Heat Equation (SHE) $\partial_t Z = \nu \partial_x^2 Z + \frac{\lambda}{2\nu} Z \xi$,
//! with universal 1+1D scaling exponents $\alpha = 1/2$ (roughness), $\beta = 1/3$ (growth), and $z = 3/2$ (dynamic).

use crate::error::{Error, Result};

/// KPZ Stochastic Growth & Cole-Hopf Transform Engine.
#[derive(Debug, Clone, Copy)]
pub struct KpzColeHopfEngine {
    /// Surface tension diffusivity $\nu > 0$.
    pub viscosity_nu: f64,
    /// Nonlinear lateral growth coupling $\lambda \ne 0$.
    pub lambda_coupling: f64,
}

impl KpzColeHopfEngine {
    /// Creates a new KPZ Cole-Hopf engine.
    pub fn new(viscosity_nu: f64, lambda_coupling: f64) -> Result<Self> {
        if viscosity_nu <= 0.0 || lambda_coupling.abs() < 1e-15 {
            return Err(Error::Msg(
                "viscosity_nu > 0 and non-zero lambda_coupling required".to_string(),
            ));
        }
        Ok(Self {
            viscosity_nu,
            lambda_coupling,
        })
    }

    /// Applies the Cole-Hopf exponential transform $Z = \exp(\frac{\lambda}{2\nu} h)$ and its inverse $h = \frac{2\nu}{\lambda} \ln Z$.
    pub fn cole_hopf_forward_and_inverse(&self, height_h: f64) -> (f64, f64) {
        let gamma = self.lambda_coupling / (2.0 * self.viscosity_nu);
        let z = (gamma * height_h).exp();
        let h_rec = z.ln() / gamma;
        (z, h_rec)
    }

    /// Returns the universal 1+1D KPZ scaling exponents $(\alpha, \beta, z) = (1/2, 1/3, 3/2)$ satisfying $\alpha + z = 2$.
    pub fn universal_kpz_exponents() -> (f64, f64, f64) {
        (0.5, 1.0 / 3.0, 1.5)
    }
}
