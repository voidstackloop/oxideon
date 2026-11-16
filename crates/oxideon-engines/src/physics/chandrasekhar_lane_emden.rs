//! Chandrasekhar White Dwarf Mass Limit $M_{\mathrm{Ch}} \approx 1.44 M_\odot$ & Exact $n = 0, 1, 5$ Lane-Emden Polytropes (Chandrasekhar, 1931).
//!
//! ### 1. Lane-Emden Differential Equation $\frac{1}{\xi^2}\frac{d}{d\xi}\left(\xi^2 \frac{d\theta}{d\xi}\right) = -\theta^n$
//! Closed-form analytical solutions exist for polytropic indices $n = 0, 1, 5$:
//! • $n = 0$ (incompressible): $\theta_0(\xi) = 1 - \xi^2 / 6$ (surface $\xi_1 = \sqrt{6}$)
//! • $n = 1$ (linear): $\theta_1(\xi) = \frac{\sin \xi}{\xi}$ (surface $\xi_1 = \pi$)
//! • $n = 5$ (Plummer sphere): $\theta_5(\xi) = (1 + \xi^2 / 3)^{-1/2}$
//! And the ultrarelativistic ($n = 3$) degenerate electron limit yields the **Chandrasekhar Mass** $M_{\mathrm{Ch}} \approx \frac{5.836}{\mu_e^2} M_\odot$.

use crate::error::{Error, Result};

/// Chandrasekhar White Dwarf Limit & Lane-Emden Polytrope Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChandrasekharWhiteDwarfEngine;

impl ChandrasekharWhiteDwarfEngine {
    /// Creates a new Chandrasekhar White Dwarf engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the Chandrasekhar mass limit $M_{\mathrm{Ch}} = 5.836 / \mu_e^2$ (in solar masses $M_\odot$) for mean molecular weight per electron $\mu_e > 0$.
    pub fn chandrasekhar_mass_solar(mean_molecular_weight_mu_e: f64) -> Result<f64> {
        if mean_molecular_weight_mu_e <= 0.0 {
            return Err(Error::Msg(
                "mean_molecular_weight_mu_e must be > 0".to_string(),
            ));
        }
        Ok(5.836 / (mean_molecular_weight_mu_e * mean_molecular_weight_mu_e))
    }

    /// Evaluates the exact $n = 5$ Plummer-Schuster Lane-Emden solution $\theta_5(\xi) = (1 + \xi^2 / 3)^{-1/2}$.
    pub fn lane_emden_n5(xi: f64) -> f64 {
        1.0 / (1.0 + xi * xi / 3.0).sqrt()
    }

    /// Evaluates the exact $n = 0$ incompressible Lane-Emden solution $\theta_0(\xi) = 1 - \xi^2 / 6$.
    pub fn lane_emden_n0(xi: f64) -> f64 {
        1.0 - xi * xi / 6.0
    }

    /// Evaluates the exact $n = 1$ linear Lane-Emden solution $\theta_1(\xi) = \sin(\xi) / \xi$ (with $\theta_1(0) = 1$).
    pub fn lane_emden_n1(xi: f64) -> f64 {
        if xi == 0.0 { 1.0 } else { xi.sin() / xi }
    }
}
