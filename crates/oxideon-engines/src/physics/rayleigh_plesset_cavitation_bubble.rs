//! Rayleigh-Plesset Cavitation Bubble Dynamics & Besant-Rayleigh Collapse Time $\tau_c \approx 0.915 R_0 \sqrt{\rho / \Delta p}$ (Rayleigh, 1917; Plesset, 1949).
//!
//! ### 1. Empty Spherical Cavity Collapse Time & Instantaneous Radial Acceleration
//! For an empty spherical vapor cavity of initial radius $R_0$ collapsing under ambient overpressure $\Delta p = p_\infty - p_v > 0$:
//! $$\tau_{\mathrm{Rayleigh}} = 0.914681 R_0 \sqrt{\frac{\rho_L}{\Delta p}}, \qquad \dot{R}^2 = \frac{2 \Delta p}{3 \rho_L}\left(\frac{R_0^3}{R^3} - 1\right)$$

use crate::error::{Error, Result};

/// Rayleigh-Plesset Cavitation Bubble Dynamics Engine.
#[derive(Debug, Clone, Copy)]
pub struct RayleighPlessetCavitationBubble {
    /// Liquid mass density $\rho_L > 0$ ($\mathrm{kg/m^3}$).
    pub liquid_density_rho: f64,
}

impl Default for RayleighPlessetCavitationBubble {
    fn default() -> Self {
        Self {
            liquid_density_rho: 1000.0,
        }
    }
}

impl RayleighPlessetCavitationBubble {
    /// Creates a new Rayleigh-Plesset Cavitation Bubble engine.
    pub fn new(liquid_density_rho: f64) -> Result<Self> {
        if liquid_density_rho <= 0.0 || !liquid_density_rho.is_finite() {
            return Err(Error::Msg("liquid_density_rho must be > 0".to_string()));
        }
        Ok(Self { liquid_density_rho })
    }

    /// Evaluates the Besant-Rayleigh total collapse time $\tau_c$ and wall speed $|\dot{R}|$ at radius $R \in (0, R_0]$.
    pub fn collapse_time_and_wall_speed(
        &self,
        r0: f64,
        r_current: f64,
        delta_p: f64,
    ) -> Result<(f64, f64)> {
        if r0 <= 0.0 || r_current <= 0.0 || r_current > r0 || delta_p <= 0.0 {
            return Err(Error::Msg(
                "0 < r_current <= r0 and delta_p > 0 required".to_string(),
            ));
        }
        let tau_c = 0.914681 * r0 * (self.liquid_density_rho / delta_p).sqrt();
        let r_dot_sq =
            (2.0 * delta_p / (3.0 * self.liquid_density_rho)) * ((r0 / r_current).powi(3) - 1.0);
        Ok((tau_c, r_dot_sq.sqrt()))
    }
}
