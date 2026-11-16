//! Maurice Biot's Theory of Poroelastic Consolidation, Biot-Willis Coefficient $\alpha_B$ & Skempton Coefficient $B$ (Biot, 1941).
//!
//! ### 1. Biot-Willis Effective Stress Coefficient $\alpha_B$ & Skempton Undrained Pore-Pressure Ratio
//! In a fluid-saturated porous rock with drained bulk modulus $K_d$ and solid grain bulk modulus $K_s \ge K_d$,
//! the Biot-Willis effective stress coefficient is:
//! $$\alpha_B = 1 - \frac{K_d}{K_s} \in [0, 1]$$
//! Under undrained isotropic compression $\Delta \sigma$, the induced pore pressure ratio is Skempton's coefficient
//! $B = \frac{\alpha_B M}{K_u}$ where $K_u = K_d + \alpha_B^2 M$ is Gassmann's undrained bulk modulus!

use crate::error::{Error, Result};

/// Biot Poroelasticity, Gassmann Undrained Modulus & Skempton Engine.
#[derive(Debug, Clone, Copy)]
pub struct BiotPoroelasticConsolidation {
    /// Drained frame bulk modulus $K_d > 0$.
    pub drained_kd: f64,
    /// Solid grain bulk modulus $K_s \ge K_d$.
    pub grain_ks: f64,
    /// Biot storage modulus $M > 0$.
    pub biot_modulus_m: f64,
}

impl BiotPoroelasticConsolidation {
    /// Creates a new Biot Poroelasticity engine.
    pub fn new(drained_kd: f64, grain_ks: f64, biot_modulus_m: f64) -> Result<Self> {
        if drained_kd <= 0.0 || grain_ks < drained_kd || biot_modulus_m <= 0.0 {
            return Err(Error::Msg(
                "Requires 0 < drained_kd <= grain_ks and biot_modulus_m > 0".to_string(),
            ));
        }
        Ok(Self {
            drained_kd,
            grain_ks,
            biot_modulus_m,
        })
    }

    /// Computes `(biot_willis_alpha, undrained_bulk_ku, skempton_b)`.
    pub fn biot_gassmann_skempton_parameters(&self) -> (f64, f64, f64) {
        let alpha = 1.0 - self.drained_kd / self.grain_ks;
        let ku = self.drained_kd + alpha * alpha * self.biot_modulus_m;
        let skempton_b = (alpha * self.biot_modulus_m) / ku;
        (alpha, ku, skempton_b)
    }
}
