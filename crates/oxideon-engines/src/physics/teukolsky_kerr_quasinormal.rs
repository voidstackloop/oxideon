//! Teukolsky Master Equation for Kerr Black Holes, Boyer-Lindquist Horizons & Light-Ring Quasinormal Modes (Teukolsky, 1973; Ferrari & Mashhoon, 1984).
//!
//! ### 1. Boyer-Lindquist Outer/Inner Horizons $r_\pm$ & Angular Velocity $\Omega_H$
//! A rotating Kerr black hole of mass $M > 0$ and dimensionless spin $\chi = a / M \in [0, 1]$ has event and Cauchy horizons:
//! $$r_\pm = M \left(1 \pm \sqrt{1 - \chi^2}\right), \qquad \Omega_H = \frac{\chi}{2 r_+}$$

use crate::error::{Error, Result};

/// Teukolsky Kerr Black Hole Perturbation & Quasinormal Mode Engine.
#[derive(Debug, Clone, Copy)]
pub struct KerrQuasinormalModeEngine {
    /// Black hole mass $M > 0$ ($G = c = 1$).
    pub mass_m: f64,
    /// Dimensionless Kerr spin parameter $\chi = a / M \in [0, 1]$.
    pub spin_chi: f64,
}

impl KerrQuasinormalModeEngine {
    /// Creates a new Kerr Quasinormal Mode engine.
    pub fn new(mass_m: f64, spin_chi: f64) -> Result<Self> {
        if mass_m <= 0.0 || !(0.0..=1.0).contains(&spin_chi) {
            return Err(Error::Msg(
                "mass_m > 0 and spin_chi in [0, 1] required".to_string(),
            ));
        }
        Ok(Self { mass_m, spin_chi })
    }

    /// Computes the outer event horizon radius $r_+ = M(1 + \sqrt{1 - \chi^2})$ and horizon angular velocity $\Omega_H = \chi / (2 r_+)$.
    pub fn outer_horizon_and_angular_velocity(&self) -> (f64, f64) {
        let r_plus = self.mass_m * (1.0 + (1.0 - self.spin_chi * self.spin_chi).sqrt());
        let omega_h = self.spin_chi / (2.0 * r_plus);
        (r_plus, omega_h)
    }

    /// Evaluates Echeverria's analytical fit for the fundamental $(l=2, m=2, n=0)$ gravitational QNM frequency $M \omega_R \approx 1 - 0.63 (1 - \chi)^{3/10}$.
    pub fn fundamental_220_qnm_frequency(&self) -> f64 {
        (1.0 - 0.63 * (1.0 - self.spin_chi).powf(0.3)) / self.mass_m
    }
}
