//! Toner-Tu Hydrodynamic Theory of Active Flocking & Vicsek Polar Order Transition (Vicsek et al., 1995; Toner & Tu, 1995).
//!
//! ### 1. Spontaneous Polar Flocking Speed $v_0 = \sqrt{\alpha / \beta}$ & Giant Number Fluctuations
//! In the Toner-Tu PDE $\partial_t \mathbf{v} + \lambda (\mathbf{v} \cdot \nabla) \mathbf{v} = (\alpha - \beta |\mathbf{v}|^2)\mathbf{v} - \nabla P + D \nabla^2 \mathbf{v} + \mathbf{f}$,
//! when $\alpha > 0$ (noise below critical threshold), the homogeneous ordered flock has speed $v_0 = \sqrt{\alpha / \beta}$ and giant number fluctuations $\Delta N \propto N^{\zeta}$ with $\zeta > 0.5$.

use crate::error::{Error, Result};

/// Toner-Tu Hydrodynamic Flocking & Vicsek Polar Order Engine.
#[derive(Debug, Clone, Copy)]
pub struct TonerTuVicsekActiveMatter {
    /// Cubic saturation coefficient $\beta > 0$.
    pub saturation_beta: f64,
}

impl Default for TonerTuVicsekActiveMatter {
    fn default() -> Self {
        Self {
            saturation_beta: 2.0,
        }
    }
}

impl TonerTuVicsekActiveMatter {
    /// Creates a new Toner-Tu Vicsek Active Matter engine.
    pub fn new(saturation_beta: f64) -> Result<Self> {
        if saturation_beta <= 0.0 || !saturation_beta.is_finite() {
            return Err(Error::Msg("saturation_beta must be > 0".to_string()));
        }
        Ok(Self { saturation_beta })
    }

    /// Evaluates the ordered flock speed $v_0 = \sqrt{\max(0, \alpha)/\beta}$ and Vicsek polar order parameter $|\Phi| = |\frac{1}{N}\sum e^{i \theta_k}|$ for angles $\theta_k$.
    pub fn flock_speed_and_polar_order(
        &self,
        linear_growth_alpha: f64,
        angles_rad: &[f64],
    ) -> Result<(f64, f64)> {
        if angles_rad.is_empty() {
            return Err(Error::Msg("angles_rad must be non-empty".to_string()));
        }
        let v0 = if linear_growth_alpha > 0.0 {
            (linear_growth_alpha / self.saturation_beta).sqrt()
        } else {
            0.0
        };
        let n = angles_rad.len() as f64;
        let sum_cos: f64 = angles_rad.iter().map(|&th| th.cos()).sum();
        let sum_sin: f64 = angles_rad.iter().map(|&th| th.sin()).sum();
        let phi = (sum_cos.hypot(sum_sin)) / n;
        Ok((v0, phi))
    }
}
