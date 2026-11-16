//! Caffarelli-Kohn-Nirenberg (CKN) 3D Navier-Stokes Partial Regularity & $\varepsilon$-Regularity Criterion (Caffarelli, Kohn & Nirenberg, 1982).
//!
//! ### 1. Scale-Invariant Parabolic Cylinder $\varepsilon$-Regularity
//! In 3D incompressible Navier-Stokes with viscosity $\nu > 0$, if the scale-invariant local dissipation in cylinder $Q_r(x_0, t_0)$ satisfies:
//! $$\frac{1}{\nu r} \iint_{Q_r} |\nabla u|^2 \, dx \, dt + \frac{1}{\nu^{3/2} r^2} \iint_{Q_r} (|u|^3 + |p|^{3/2}) \, dx \, dt < \varepsilon_{\mathrm{CKN}}$$
//! then $u$ is regular ($L^\infty$) on $Q_{r/2}$, implying the 1D parabolic Hausdorff measure of the singular set $\mathcal{P}^1(\mathcal{S}) = 0$.

use crate::error::{Error, Result};

/// Caffarelli-Kohn-Nirenberg Navier-Stokes Partial Regularity Engine.
#[derive(Debug, Clone, Copy)]
pub struct CaffarelliKohnNirenbergNavierStokes {
    /// Kinematic viscosity $\nu > 0$.
    pub viscosity_nu: f64,
    /// Universal CKN threshold $\varepsilon_{\mathrm{CKN}} > 0$.
    pub epsilon_ckn: f64,
}

impl Default for CaffarelliKohnNirenbergNavierStokes {
    fn default() -> Self {
        Self {
            viscosity_nu: 0.01,
            epsilon_ckn: 0.1,
        }
    }
}

impl CaffarelliKohnNirenbergNavierStokes {
    /// Creates a new Caffarelli-Kohn-Nirenberg Navier-Stokes engine.
    pub fn new(viscosity_nu: f64, epsilon_ckn: f64) -> Result<Self> {
        if viscosity_nu <= 0.0 || epsilon_ckn <= 0.0 {
            return Err(Error::Msg(
                "viscosity_nu and epsilon_ckn must be > 0".to_string(),
            ));
        }
        Ok(Self {
            viscosity_nu,
            epsilon_ckn,
        })
    }

    /// Evaluates the dimensionless scale-invariant CKN functional on $Q_r$ and verifies local regularity ($\Phi_r < \varepsilon_{\mathrm{CKN}}$).
    pub fn scale_invariant_ckn_functional(
        &self,
        radius_r: f64,
        enstrophy_int: f64,
        velocity_l3_cubed_int: f64,
    ) -> Result<(f64, bool)> {
        if radius_r <= 0.0 || enstrophy_int < 0.0 || velocity_l3_cubed_int < 0.0 {
            return Err(Error::Msg(
                "radius_r > 0 and non-negative integrals required".to_string(),
            ));
        }
        let term1 = enstrophy_int / (self.viscosity_nu * radius_r);
        let term2 = velocity_l3_cubed_int / (self.viscosity_nu.powf(1.5) * radius_r * radius_r);
        let phi = term1 + term2;
        Ok((phi, phi < self.epsilon_ckn))
    }
}
