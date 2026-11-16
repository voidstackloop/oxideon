//! Superfluid Bose-Einstein Condensate, Gross-Pitaevskii Equation, Bogoliubov Dispersion & Landau Critical Velocity (Bogoliubov, 1947; Gross, 1961; Pitaevskii, 1961).
//!
//! ### 1. Bogoliubov Quasiparticle Spectrum $E(k)$ & Healing Length $\xi$
//! In a weakly interacting BEC with condensate density $n_0 > 0$ and repulsive $s$-wave coupling $g > 0$ (chemical potential $\mu = g n_0$),
//! elementary excitations above the condensate follow the **Bogoliubov Dispersion Relation** ($\hbar = m = 1$):
//! $$E(k) = \sqrt{\frac{k^2}{2}\left(\frac{k^2}{2} + 2 g n_0\right)}, \qquad c_s = \sqrt{g n_0}, \qquad \xi = \frac{1}{\sqrt{2 g n_0}}$$

use crate::error::{Error, Result};

/// Gross-Pitaevskii BEC Superfluid & Bogoliubov Quasiparticle Engine.
#[derive(Debug, Clone, Copy)]
pub struct GrossPitaevskiiVortexEngine {
    /// Condensate density $n_0 > 0$.
    pub density_n0: f64,
    /// Repulsive interaction coupling $g > 0$.
    pub coupling_g: f64,
}

impl GrossPitaevskiiVortexEngine {
    /// Creates a new Gross-Pitaevskii BEC engine.
    pub fn new(density_n0: f64, coupling_g: f64) -> Result<Self> {
        if density_n0 <= 0.0 || coupling_g <= 0.0 {
            return Err(Error::Msg(
                "density_n0 and coupling_g must be > 0".to_string(),
            ));
        }
        Ok(Self {
            density_n0,
            coupling_g,
        })
    }

    /// Computes the Bogoliubov speed of sound $c_s = \sqrt{g n_0}$ (Landau critical velocity) and healing length $\xi = 1 / \sqrt{2 g n_0}$.
    pub fn sound_speed_and_healing_length(&self) -> (f64, f64) {
        let mu = self.coupling_g * self.density_n0;
        (mu.sqrt(), 1.0 / (2.0 * mu).sqrt())
    }

    /// Evaluates the Bogoliubov quasiparticle energy $E(k) = \sqrt{\frac{k^2}{2}(\frac{k^2}{2} + 2 g n_0)}$.
    pub fn bogoliubov_dispersion(&self, wavenumber_k: f64) -> f64 {
        let ek = 0.5 * wavenumber_k * wavenumber_k;
        let mu = self.coupling_g * self.density_n0;
        (ek * (ek + 2.0 * mu)).sqrt()
    }
}
