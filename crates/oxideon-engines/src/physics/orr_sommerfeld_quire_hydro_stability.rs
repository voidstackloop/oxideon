//! Orr-Sommerfeld & Squire Equations for Parallel Shear Flow Stability & Transient Non-Normal Growth (Orr, 1907; Sommerfeld, 1908; Squire, 1933).
//!
//! ### 1. Squire's Transformation $\widetilde{\mathrm{Re}} = (\alpha / k) \mathrm{Re}$ & Non-Normal Kreiss Constant
//! By Squire's theorem, for any unstable 3D disturbance $(\alpha, \beta)$ at Reynolds number $\mathrm{Re}$ with $k = \sqrt{\alpha^2 + \beta^2}$,
//! there is a more unstable 2D mode $(\tilde{\alpha} = k, 0)$ at lower Reynolds number $\widetilde{\mathrm{Re}} = (\alpha / k) \mathrm{Re} < \mathrm{Re}$.

use crate::error::{Error, Result};

/// Orr-Sommerfeld & Squire Hydrodynamic Stability Engine.
#[derive(Debug, Clone, Copy)]
pub struct OrrSommerfeldSquireStability {
    /// Flow Reynolds number $\mathrm{Re} > 0$.
    pub reynolds_number: f64,
}

impl Default for OrrSommerfeldSquireStability {
    fn default() -> Self {
        Self {
            reynolds_number: 5772.22,
        }
    }
}

impl OrrSommerfeldSquireStability {
    /// Creates a new Orr-Sommerfeld & Squire Stability engine.
    pub fn new(reynolds_number: f64) -> Result<Self> {
        if reynolds_number <= 0.0 || !reynolds_number.is_finite() {
            return Err(Error::Msg("reynolds_number must be > 0".to_string()));
        }
        Ok(Self { reynolds_number })
    }

    /// Evaluates the Squire total wavenumber $k = \sqrt{\alpha^2 + \beta^2}$ and equivalent 2D Reynolds number $\widetilde{\mathrm{Re}} = (\alpha / k)\mathrm{Re}$.
    pub fn squire_2d_wavenumber_and_reynolds(
        &self,
        alpha_streamwise: f64,
        beta_spanwise: f64,
    ) -> Result<(f64, f64)> {
        if alpha_streamwise <= 0.0 || beta_spanwise < 0.0 {
            return Err(Error::Msg(
                "alpha_streamwise > 0 and beta_spanwise >= 0 required".to_string(),
            ));
        }
        let k = (alpha_streamwise * alpha_streamwise + beta_spanwise * beta_spanwise).sqrt();
        let re_2d = (alpha_streamwise / k) * self.reynolds_number;
        Ok((k, re_2d))
    }
}
