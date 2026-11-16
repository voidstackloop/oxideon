//! Jaynes-Cummings Cavity QED Vacuum Rabi Splitting, Dressed States & Collapse-Revival Time (Jaynes & Cummings, 1963).
//!
//! ### 1. Generalized n-Photon Rabi Frequency & Revival Time
//! For single-photon dipole coupling $g$, atom-cavity detuning $\Delta = \omega_a - \omega_c$, and photon number $n$,
//! the dressed-state splitting is $\Omega_n = \sqrt{\Delta^2 + 4 g^2 (n + 1)}$, and coherent-state revival occurs at $t_{\mathrm{rev}} = \frac{2 \pi \sqrt{\bar{n}}}{g}$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Jaynes-Cummings Cavity QED & Vacuum Rabi Splitting Engine.
#[derive(Debug, Clone, Copy)]
pub struct JaynesCummingsCavityQED {
    /// Vacuum single-photon coupling strength $g > 0$.
    pub coupling_g: f64,
    /// Atom-cavity detuning $\Delta = \omega_a - \omega_c$.
    pub detuning_delta: f64,
}

impl Default for JaynesCummingsCavityQED {
    fn default() -> Self {
        Self {
            coupling_g: 1.0,
            detuning_delta: 0.0,
        }
    }
}

impl JaynesCummingsCavityQED {
    /// Creates a new Jaynes-Cummings Cavity QED engine.
    pub fn new(coupling_g: f64, detuning_delta: f64) -> Result<Self> {
        if coupling_g <= 0.0 || !coupling_g.is_finite() || !detuning_delta.is_finite() {
            return Err(Error::Msg("coupling_g must be > 0 and finite".to_string()));
        }
        Ok(Self {
            coupling_g,
            detuning_delta,
        })
    }

    /// Evaluates the $n$-photon generalized Rabi frequency $\Omega_n = \sqrt{\Delta^2 + 4 g^2 (n + 1)}$ and coherent-state revival time $t_{\mathrm{rev}} = 2 \pi \sqrt{\bar{n}} / g$.
    pub fn rabi_frequency_and_revival_time(
        &self,
        photon_n: usize,
        mean_coherent_n: f64,
    ) -> Result<(f64, f64)> {
        if mean_coherent_n <= 0.0 {
            return Err(Error::Msg("mean_coherent_n must be > 0".to_string()));
        }
        let omega_n = (self.detuning_delta.powi(2)
            + 4.0 * self.coupling_g.powi(2) * ((photon_n + 1) as f64))
            .sqrt();
        let t_rev = 2.0 * PI * mean_coherent_n.sqrt() / self.coupling_g;
        Ok((omega_n, t_rev))
    }
}
