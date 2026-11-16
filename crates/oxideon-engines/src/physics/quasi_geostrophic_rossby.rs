//! Charney Quasi-Geostrophic Potential Vorticity & Planetary Rossby Wave Dispersion (Rossby, 1939; Charney, 1948).
//!
//! ### 1. Rossby Wave Phase & Group Velocity on a $\beta$-Plane
//! Conservation of Quasi-Geostrophic Potential Vorticity $q = \nabla^2 \psi - L_R^{-2} \psi + \beta y$ yields the
//! anisotropic planetary **Rossby Wave Dispersion Relation**:
//! $$\omega(k_x, k_y) = \frac{-\beta k_x}{k_x^2 + k_y^2 + L_R^{-2}}$$
//! Notice that the zonal phase speed $c_x = \omega / k_x < 0$ is **always westward**, whereas the zonal group velocity
//! $c_{gx} = \frac{\partial \omega}{\partial k_x} = \frac{\beta (k_x^2 - k_y^2 - L_R^{-2})}{(k_x^2 + k_y^2 + L_R^{-2})^2}$ can be eastward for short waves!

use crate::error::{Error, Result};

/// Charney Quasi-Geostrophic Potential Vorticity & Rossby Wave Engine.
#[derive(Debug, Clone, Copy)]
pub struct CharneyRossbyWaveEngine {
    /// Meridional Coriolis gradient $\beta = df / dy > 0$.
    pub coriolis_beta: f64,
    /// Baroclinic Rossby radius of deformation $L_R > 0$.
    pub deformation_radius_lr: f64,
}

impl CharneyRossbyWaveEngine {
    /// Creates a new Charney-Rossby planetary wave engine.
    pub fn new(coriolis_beta: f64, deformation_radius_lr: f64) -> Result<Self> {
        if coriolis_beta <= 0.0 || deformation_radius_lr <= 0.0 {
            return Err(Error::Msg(
                "coriolis_beta and deformation_radius_lr must be > 0".to_string(),
            ));
        }
        Ok(Self {
            coriolis_beta,
            deformation_radius_lr,
        })
    }

    /// Computes the Rossby wave frequency $\omega(k_x, k_y)$ and zonal group velocity $c_{gx} = \partial \omega / \partial k_x$.
    pub fn frequency_and_zonal_group_velocity(&self, kx: f64, ky: f64) -> (f64, f64) {
        let kd2 = 1.0 / (self.deformation_radius_lr * self.deformation_radius_lr);
        let denom = kx * kx + ky * ky + kd2;
        let omega = -self.coriolis_beta * kx / denom;
        let cgx = self.coriolis_beta * (kx * kx - ky * ky - kd2) / (denom * denom);
        (omega, cgx)
    }
}
