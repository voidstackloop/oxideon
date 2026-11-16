//! Kolmogorov K41 Inertial Subrange Turbulence $E(k) = C_K \varepsilon^{2/3} k^{-5/3}$ & She-Leveque Intermittency (Kolmogorov, 1941; She & Leveque, 1994).
//!
//! ### 1. K41 Energy Spectrum & She-Leveque Structure Function Exponents $\zeta_p$
//! In 3D isotropic Navier-Stokes turbulence with energy dissipation rate $\varepsilon > 0$ and kinematic viscosity $\nu > 0$:
//! $$E(k) = C_K \varepsilon^{2/3} k^{-5/3}, \qquad \eta_K = \left(\frac{\nu^3}{\varepsilon}\right)^{1/4}$$
//! The She-Leveque hierarchy of vortex filaments predicts the anomalous structure function scaling exponents:
//! $$\zeta_p = \frac{p}{9} + 2 \left(1 - \left(\frac{2}{3}\right)^{p/3}\right)$$

use crate::error::{Error, Result};

/// Kolmogorov K41 Inertial Cascade & She-Leveque Intermittency Engine.
#[derive(Debug, Clone, Copy)]
pub struct KolmogorovCascadeEngine {
    /// Energy dissipation rate per unit mass $\varepsilon > 0$.
    pub dissipation_eps: f64,
    /// Kinematic viscosity $\nu > 0$.
    pub viscosity_nu: f64,
}

impl KolmogorovCascadeEngine {
    /// Creates a new Kolmogorov turbulence engine.
    pub fn new(dissipation_eps: f64, viscosity_nu: f64) -> Result<Self> {
        if dissipation_eps <= 0.0 || viscosity_nu <= 0.0 {
            return Err(Error::Msg(
                "dissipation_eps and viscosity_nu must be > 0".to_string(),
            ));
        }
        Ok(Self {
            dissipation_eps,
            viscosity_nu,
        })
    }

    /// Evaluates the K41 inertial subrange energy spectrum $E(k) = 1.5 \varepsilon^{2/3} k^{-5/3}$ and Kolmogorov microscale $\eta_K = (\nu^3 / \varepsilon)^{1/4}$.
    pub fn spectrum_and_microscale(&self, wavenumber_k: f64) -> Result<(f64, f64)> {
        if wavenumber_k <= 0.0 {
            return Err(Error::Msg("wavenumber_k must be > 0".to_string()));
        }
        let ek = 1.5 * self.dissipation_eps.powf(2.0 / 3.0) * wavenumber_k.powf(-5.0 / 3.0);
        let eta = (self.viscosity_nu.powi(3) / self.dissipation_eps).powf(0.25);
        Ok((ek, eta))
    }

    /// Evaluates the exact She-Leveque structure function exponent $\zeta_p = \frac{p}{9} + 2(1 - (2/3)^{p/3})$ (note $\zeta_3 = 1$ by Kolmogorov's 4/5 law).
    pub fn she_leveque_exponent(order_p: f64) -> f64 {
        order_p / 9.0 + 2.0 * (1.0 - (2.0_f64 / 3.0).powf(order_p / 3.0))
    }
}
