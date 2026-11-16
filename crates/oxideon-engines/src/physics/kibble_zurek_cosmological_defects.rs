//! Kibble-Zurek Topological Defect Scaling across Second-Order Phase Transitions (Kibble, 1976; Zurek, 1985).
//!
//! ### 1. Kibble-Zurek Freeze-Out Correlation Length & Defect Density
//! For a quench across a critical point with correlation-length exponent $\nu$, dynamical exponent $z$, and quench timescale $\tau_Q$:
//! $$\mu_{\mathrm{KZ}} = \frac{\nu}{1 + z \nu}, \qquad \hat{\xi} = \xi_0 \left(\frac{\tau_Q}{\tau_0}\right)^{\mu_{\mathrm{KZ}}}, \qquad n_{\mathrm{def}} \propto \hat{\xi}^{-(d - p)}$$

use crate::error::{Error, Result};

/// Kibble-Zurek Freeze-Out Correlation Length & Topological Defect Scaling Engine.
#[derive(Debug, Clone, Copy)]
pub struct KibbleZurekDefectScaling {
    /// Correlation-length critical exponent $\nu > 0$.
    pub nu: f64,
    /// Dynamical critical exponent $z > 0$.
    pub z_dyn: f64,
}

impl Default for KibbleZurekDefectScaling {
    fn default() -> Self {
        Self {
            nu: 0.5,
            z_dyn: 2.0,
        }
    }
}

impl KibbleZurekDefectScaling {
    /// Creates a new Kibble-Zurek Defect Scaling engine.
    pub fn new(nu: f64, z_dyn: f64) -> Result<Self> {
        if nu <= 0.0 || z_dyn <= 0.0 {
            return Err(Error::Msg(
                "Critical exponents nu and z_dyn must be > 0".to_string(),
            ));
        }
        Ok(Self { nu, z_dyn })
    }

    /// Evaluates the Kibble-Zurek exponent $\mu_{\mathrm{KZ}} = \nu / (1 + z \nu)$, freeze-out correlation length $\hat{\xi}/\xi_0$, and point-defect density $(\hat{\xi}/\xi_0)^{-d}$.
    pub fn freeze_out_length_and_density(
        &self,
        quench_ratio_tau_q: f64,
        spatial_dim_d: usize,
    ) -> Result<(f64, f64, f64)> {
        if quench_ratio_tau_q <= 0.0 || spatial_dim_d == 0 {
            return Err(Error::Msg(
                "quench_ratio_tau_q > 0 and spatial_dim_d >= 1 required".to_string(),
            ));
        }
        let mu_kz = self.nu / (1.0 + self.z_dyn * self.nu);
        let xi_hat = quench_ratio_tau_q.powf(mu_kz);
        let density = xi_hat.powf(-(spatial_dim_d as f64));
        Ok((mu_kz, xi_hat, density))
    }
}
