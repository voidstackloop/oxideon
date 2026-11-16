//! Electromagnetically Induced Transparency (EIT), $\Lambda$-System Dark State Polariton & Slow Light (Harris, 1990; Fleischhauer & Lukin, 2000).
//!
//! ### 1. Dark-State Mixing Angle $\theta$ & Slow-Light Group Velocity $v_g$
//! In a three-level $\Lambda$ atomic medium with collective probe coupling $g \sqrt{N}$ and control Rabi frequency $\Omega_c$:
//! $$\tan \theta = \frac{g \sqrt{N}}{\Omega_c}, \qquad v_g = c \cos^2 \theta = \frac{c}{1 + g^2 N / \Omega_c^2}$$

use crate::error::{Error, Result};

/// Electromagnetically Induced Transparency (EIT) & Dark-State Polariton Engine.
#[derive(Debug, Clone, Copy)]
pub struct ElectromagneticallyInducedTransparency {
    /// Collective atom-field coupling $g \sqrt{N} > 0$.
    pub collective_coupling_gn: f64,
}

impl Default for ElectromagneticallyInducedTransparency {
    fn default() -> Self {
        Self {
            collective_coupling_gn: 1000.0,
        }
    }
}

impl ElectromagneticallyInducedTransparency {
    /// Creates a new EIT Dark-State Polariton engine.
    pub fn new(collective_coupling_gn: f64) -> Result<Self> {
        if collective_coupling_gn <= 0.0 || !collective_coupling_gn.is_finite() {
            return Err(Error::Msg("collective_coupling_gn must be > 0".to_string()));
        }
        Ok(Self {
            collective_coupling_gn,
        })
    }

    /// Evaluates the group index $n_g = 1 + g^2 N / \Omega_c^2$ and normalized group velocity $v_g / c = \cos^2 \theta$ for control Rabi frequency $\Omega_c > 0$.
    pub fn group_index_and_slow_light_ratio(
        &self,
        control_rabi_omega_c: f64,
    ) -> Result<(f64, f64)> {
        if control_rabi_omega_c <= 0.0 {
            return Err(Error::Msg("control_rabi_omega_c must be > 0".to_string()));
        }
        let ratio_sq = (self.collective_coupling_gn / control_rabi_omega_c).powi(2);
        let n_g = 1.0 + ratio_sq;
        let vg_over_c = 1.0 / n_g;
        Ok((n_g, vg_over_c))
    }
}
