//! Rayleigh-Bénard Thermal Convection Instability, Critical Rayleigh Number $\mathrm{Ra}_c = \frac{27 \pi^4}{4}$ & Nusselt Scaling (Rayleigh, 1916).
//!
//! ### 1. Lord Rayleigh's Marginal Stability Criterion (1916)
//! For a horizontal fluid layer of thickness $H$ heated from below with Rayleigh number $\mathrm{Ra} = \frac{g \alpha \Delta T H^3}{\nu \kappa}$,
//! marginal stability of mode $a = k H$ under stress-free boundaries occurs at:
//! $$\mathrm{Ra}_c(a) = \frac{(\pi^2 + a^2)^3}{a^2} \ge \mathrm{Ra}_{\min} = \frac{27 \pi^4}{4} \approx 657.51136 \quad \left(a_c = \frac{\pi}{\sqrt{2}}\right)$$

use std::f64::consts::PI;

/// Rayleigh-Bénard Thermal Convection & Marginal Stability Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct RayleighBenardConvection;

impl RayleighBenardConvection {
    /// Creates a new Rayleigh-Bénard Convection engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the exact critical Rayleigh number $\mathrm{Ra}_c = \frac{27 \pi^4}{4}$ for stress-free boundaries.
    pub fn critical_rayleigh_free_free() -> f64 {
        27.0 * PI.powi(4) / 4.0
    }

    /// Evaluates the marginal Rayleigh curve $\mathrm{Ra}(a) = (\pi^2 + a^2)^3 / a^2$ at dimensionless wavenumber $a = k H > 0$.
    pub fn marginal_rayleigh_curve(wavenumber_a: f64) -> f64 {
        (PI * PI + wavenumber_a * wavenumber_a).powi(3) / (wavenumber_a * wavenumber_a)
    }
}
