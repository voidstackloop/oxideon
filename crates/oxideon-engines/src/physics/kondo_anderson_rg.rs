//! Kondo Effect, Anderson Poor Man's Scaling RG, Kondo Temperature $T_K$ & Nozières Singlet Fermi Liquid (Kondo, 1964; Anderson, 1970).
//!
//! ### 1. Poor Man's Scaling RG & Non-Perturbative Kondo Scale $T_K$
//! Antiferromagnetic spin-exchange coupling $J > 0$ between a localized magnetic impurity and a conduction band of
//! half-bandwidth $D > 0$ and density of states $\rho_0 > 0$ flows under 2-loop RG to strong coupling at the **Kondo Temperature**:
//! $$T_K = D \sqrt{\rho_0 J} \exp\left(-\frac{1}{\rho_0 J}\right)$$

use crate::error::{Error, Result};

/// Kondo Screening & Anderson Poor Man's Scaling RG Engine.
#[derive(Debug, Clone, Copy)]
pub struct KondoScreeningEngine {
    /// Conduction half-bandwidth $D > 0$.
    pub bandwidth_d: f64,
    /// Dimensionless antiferromagnetic Kondo coupling $g_0 = \rho_0 J > 0$.
    pub dimensionless_g0: f64,
}

impl KondoScreeningEngine {
    /// Creates a new Kondo Screening RG engine.
    pub fn new(bandwidth_d: f64, dimensionless_g0: f64) -> Result<Self> {
        if bandwidth_d <= 0.0 || dimensionless_g0 <= 0.0 {
            return Err(Error::Msg(
                "bandwidth_d and dimensionless_g0 must be > 0".to_string(),
            ));
        }
        Ok(Self {
            bandwidth_d,
            dimensionless_g0,
        })
    }

    /// Computes the non-perturbative 2-loop Kondo temperature $T_K = D \sqrt{g_0} e^{-1 / g_0}$.
    pub fn kondo_temperature(&self) -> f64 {
        self.bandwidth_d * self.dimensionless_g0.sqrt() * (-1.0 / self.dimensionless_g0).exp()
    }
}
