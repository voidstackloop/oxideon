//! Pendry-Schurig-Smith Transformation Optics Cylindrical Cloak & Veselago Negative-Index Metamaterials (Veselago, 1968; Pendry, Schurig & Smith, 2006).
//!
//! ### 1. Pendry's Radial Coordinate Compression $r' = R_1 + r \frac{R_2 - R_1}{R_2}$
//! Mapping the electromagnetic disc $0 \le r \le R_2$ into the annular cloak shell $R_1 \le r' \le R_2$ requires
//! anisotropic radial and azimuthal relative permittivity/permeability components:
//! $$\varepsilon_{r'}(r') = \mu_{r'}(r') = \frac{r' - R_1}{r'}, \qquad \varepsilon_{\theta'}(r') = \mu_{\theta'}(r') = \frac{r'}{r' - R_1}$$

use crate::error::{Error, Result};

/// Pendry Transformation Optics Cloak & Veselago Metamaterial Engine.
#[derive(Debug, Clone, Copy)]
pub struct PendryMetamaterialCloakEngine {
    /// Inner cloaked cylinder radius $R_1 > 0$.
    pub inner_r1: f64,
    /// Outer cloak shell radius $R_2 > R_1$.
    pub outer_r2: f64,
}

impl PendryMetamaterialCloakEngine {
    /// Creates a new Pendry Transformation Optics Cloak engine.
    pub fn new(inner_r1: f64, outer_r2: f64) -> Result<Self> {
        if inner_r1 <= 0.0 || outer_r2 <= inner_r1 {
            return Err(Error::Msg("Requires 0 < inner_r1 < outer_r2".to_string()));
        }
        Ok(Self { inner_r1, outer_r2 })
    }

    /// Evaluates the radial and azimuthal metamaterial constitutive parameters $(\varepsilon_r, \varepsilon_\theta, \varepsilon_z)$ at radius $r' \in (R_1, R_2]$.
    pub fn cylindrical_cloak_permittivity(&self, r_prime: f64) -> Result<(f64, f64, f64)> {
        if r_prime <= self.inner_r1 || r_prime > self.outer_r2 {
            return Err(Error::Msg(
                "r_prime must lie in the cloak shell (R1, R2]".to_string(),
            ));
        }
        let eps_r = (r_prime - self.inner_r1) / r_prime;
        let eps_theta = 1.0 / eps_r;
        let scale = self.outer_r2 / (self.outer_r2 - self.inner_r1);
        let eps_z = scale * scale * eps_r;
        Ok((eps_r, eps_theta, eps_z))
    }
}
