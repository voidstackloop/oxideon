//! Lucas-Washburn Capillary Imbibition Law, Young-Laplace Pressure & Capillary Length (Laplace, 1805; Washburn, 1921).
//!
//! ### 1. Lucas-Washburn Meniscus Penetration Length $L(t)$
//! In a cylindrical pore of radius $r$ wetted at contact angle $\theta \in [0, \pi/2)$ by a liquid of surface tension $\gamma$ and viscosity $\eta$:
//! $$\Delta P_{\mathrm{YL}} = \frac{2 \gamma \cos \theta}{r}, \qquad L(t) = \sqrt{\frac{\gamma r \cos \theta}{2 \eta} t}$$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Lucas-Washburn Capillary Imbibition & Young-Laplace Wetting Engine.
#[derive(Debug, Clone, Copy)]
pub struct WashburnYoungLaplaceCapillary {
    /// Liquid-vapor surface tension $\gamma > 0$.
    pub surface_tension_gamma: f64,
    /// Dynamic viscosity $\eta > 0$.
    pub viscosity_eta: f64,
}

impl Default for WashburnYoungLaplaceCapillary {
    fn default() -> Self {
        Self {
            surface_tension_gamma: 0.072,
            viscosity_eta: 0.001,
        }
    }
}

impl WashburnYoungLaplaceCapillary {
    /// Creates a new Washburn-Young-Laplace Capillary engine.
    pub fn new(surface_tension_gamma: f64, viscosity_eta: f64) -> Result<Self> {
        if surface_tension_gamma <= 0.0 || viscosity_eta <= 0.0 {
            return Err(Error::Msg(
                "surface_tension_gamma and viscosity_eta must be > 0".to_string(),
            ));
        }
        Ok(Self {
            surface_tension_gamma,
            viscosity_eta,
        })
    }

    /// Evaluates Young-Laplace capillary pressure $\Delta P = 2\gamma \cos\theta / r$ and Lucas-Washburn imbibition distance $L(t)$ at time $t \ge 0$.
    pub fn laplace_pressure_and_washburn_length(
        &self,
        pore_radius_r: f64,
        contact_angle_rad: f64,
        time_t: f64,
    ) -> Result<(f64, f64)> {
        if pore_radius_r <= 0.0 || !(0.0..(0.5 * PI)).contains(&contact_angle_rad) || time_t < 0.0 {
            return Err(Error::Msg(
                "pore_radius_r > 0, contact_angle in [0, pi/2), time_t >= 0 required".to_string(),
            ));
        }
        let cos_th = contact_angle_rad.cos();
        let dp = 2.0 * self.surface_tension_gamma * cos_th / pore_radius_r;
        let l_t = (self.surface_tension_gamma * pore_radius_r * cos_th * time_t
            / (2.0 * self.viscosity_eta))
            .sqrt();
        Ok((dp, l_t))
    }
}
