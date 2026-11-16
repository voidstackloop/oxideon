//! Zoeppritz P-SV Seismic Reflection & Aki-Richards / Shuey AVO Linearization (Zoeppritz, 1919; Aki & Richards, 1980; Shuey, 1985).
//!
//! ### 1. Shuey Three-Term AVO Reflection Coefficient $R_{PP}(\theta)$
//! For incidence angle $\theta$, normal-incidence P-wave reflectivity $R_0 = \frac{1}{2}\left(\frac{\Delta V_P}{V_P} + \frac{\Delta \rho}{\rho}\right)$, and AVO gradient $G$:
//! $$R_{PP}(\theta) \approx R_0 + G \sin^2 \theta + F (\tan^2 \theta - \sin^2 \theta)$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Aki-Richards & Shuey Seismic AVO Reflectivity Engine.
#[derive(Debug, Clone, Copy)]
pub struct AkiRichardsSeismicAvoEngine {
    /// Normal-incidence P-wave intercept reflectivity $R_0$.
    pub intercept_r0: f64,
    /// Shuey AVO gradient $G$.
    pub gradient_g: f64,
    /// Far-angle curvature term $F = \frac{1}{2} \Delta V_P / V_P$.
    pub curvature_f: f64,
}

impl Default for AkiRichardsSeismicAvoEngine {
    fn default() -> Self {
        Self {
            intercept_r0: -0.1,
            gradient_g: -0.2,
            curvature_f: 0.05,
        }
    }
}

impl AkiRichardsSeismicAvoEngine {
    /// Creates a new Aki-Richards Seismic AVO engine.
    pub fn new(intercept_r0: f64, gradient_g: f64, curvature_f: f64) -> Self {
        Self {
            intercept_r0,
            gradient_g,
            curvature_f,
        }
    }

    /// Evaluates the Shuey three-term $R_{PP}(\theta)$ reflection amplitude at incidence angle $\theta \in [0, \pi/2)$.
    pub fn shuey_rpp_reflectivity(&self, angle_theta_rad: f64) -> Result<f64> {
        if !(0.0..(0.49 * PI)).contains(&angle_theta_rad) {
            return Err(Error::Msg(
                "Incidence angle theta must be in [0, 0.49*pi)".to_string(),
            ));
        }
        let s2 = angle_theta_rad.sin().powi(2);
        let t2 = angle_theta_rad.tan().powi(2);
        Ok(self.intercept_r0 + self.gradient_g * s2 + self.curvature_f * (t2 - s2))
    }
}
