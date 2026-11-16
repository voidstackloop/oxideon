//! Thermal Sunyaev-Zel'dovich (tSZ) Effect, Kompaneets Equation & Compton $y$-Distortion Spectral Function $g(x)$ (Sunyaev & Zel'dovich, 1972).
//!
//! ### 1. Non-Relativistic Thermal SZ Spectral Distortion $g(x)$
//! Inverse Compton scattering of CMB photons ($x = h \nu / (k_B T_{\mathrm{CMB}})$) off hot intracluster electrons
//! shifts photon occupation number $\Delta n / n = y \frac{x e^x}{e^x - 1} g(x)$ with universal spectral function:
//! $$g(x) = x \coth\left(\frac{x}{2}\right) - 4 = x \frac{e^x + 1}{e^x - 1} - 4$$
//! which is **negative** (decrement) for $x < x_0 \approx 3.830$ ($\nu < 217\,\mathrm{GHz}$) and **positive** (increment) for $x > 3.830$!

/// Sunyaev-Zel'dovich Kompaneets Compton-$y$ Spectral Distortion Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SunyaevZeldovichKompaneetsEngine;

impl SunyaevZeldovichKompaneetsEngine {
    /// Creates a new Sunyaev-Zel'dovich engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the universal thermal Sunyaev-Zel'dovich spectral function $g(x) = x \coth(x/2) - 4$ ($g(0) = -2$).
    pub fn tsz_spectral_function_g(dimensionless_freq_x: f64) -> f64 {
        let x = dimensionless_freq_x.abs();
        if x < 1e-6 {
            -2.0 + x * x / 12.0
        } else {
            x / (0.5 * x).tanh() - 4.0
        }
    }
}
