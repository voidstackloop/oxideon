//! Press-Schechter Dark Matter Halo Mass Function, Spherical Collapse $\delta_c \approx 1.686$ & Excursion Set Theory (Press & Schechter, 1974; Bond et al., 1991).
//!
//! ### 1. Spherical Collapse Overdensity $\delta_c$ & Excursion Set Multiplicity $f_{\mathrm{PS}}(\nu)$
//! In an Einstein-de Sitter universe, a spherical top-hat perturbation virializes when its linear-theory extrapolated
//! overdensity reaches $\delta_c = \frac{3}{5}\left(\frac{3 \pi}{2}\right)^{2/3} \approx 1.68647$. For peak height $\nu = \delta_c / \sigma(M) > 0$,
//! the Press-Schechter mass fraction per unit $\ln \nu$ is:
//! $$\nu f_{\mathrm{PS}}(\nu) = \sqrt{\frac{2}{\pi}} \, \nu \, e^{-\nu^2 / 2}$$

use std::f64::consts::PI;

/// Press-Schechter Excursion Set Dark Matter Halo Mass Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct PressSchechterHaloEngine;

impl PressSchechterHaloEngine {
    /// Creates a new Press-Schechter Halo engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the exact spherical-collapse critical linear overdensity $\delta_c = \frac{3}{5} (3\pi / 2)^{2/3} \approx 1.68647$.
    pub fn spherical_collapse_delta_c() -> f64 {
        0.6 * (1.5 * PI).powf(2.0 / 3.0)
    }

    /// Evaluates the Press-Schechter halo multiplicity $\nu f_{\mathrm{PS}}(\nu) = \sqrt{2/\pi} \nu e^{-\nu^2 / 2}$ for peak height $\nu > 0$.
    pub fn press_schechter_multiplicity(peak_nu: f64) -> f64 {
        (2.0 / PI).sqrt() * peak_nu * (-0.5 * peak_nu * peak_nu).exp()
    }
}
