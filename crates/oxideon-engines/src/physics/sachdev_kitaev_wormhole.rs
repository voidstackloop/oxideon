//! Traversable Maldacena-Qi Double-SYK Wormholes & Gao-Jafferis-Wall Quantum Teleportation (Maldacena & Qi, 2018).
//!
//! ### 1. Gao-Jafferis-Wall / Maldacena-Qi Traversable Wormhole Coupling
//! Coupling two entangled boundary SYK systems ($L$ and $R$, initially in the Thermofield Double $|\mathrm{TFD}\rangle$ state)
//! via the bilinear non-local interaction $H_{\text{int}} = i \mu \sum_{j=1}^N \psi_L^j \psi_R^j$ generates negative
//! averaged null energy $\langle \int T_{uu} du \rangle < 0$ in the bulk $\text{AdS}_2$ throat, opening a
//! **Traversable Eternal Wormhole** with energy gap $E_{\text{gap}} \propto \mu^{\frac{2}{2 - 2\Delta}}$.

use crate::error::{Error, Result};

/// Traversable Maldacena-Qi Double-SYK Wormhole & Teleportation Engine.
#[derive(Debug, Clone, Copy)]
pub struct TraversableWormholeEngine {
    /// Left-Right bilinear coupling $\mu > 0$.
    pub coupling_mu: f64,
    /// Majorana fermion scaling dimension $\Delta = 1/q \in (0, 1/2)$.
    pub scaling_delta: f64,
}

impl TraversableWormholeEngine {
    /// Creates a new Traversable Wormhole engine.
    pub fn new(coupling_mu: f64, scaling_delta: f64) -> Result<Self> {
        if coupling_mu <= 0.0 || scaling_delta <= 0.0 || scaling_delta >= 0.5 {
            return Err(Error::Msg(
                "coupling_mu > 0 and scaling_delta in (0, 0.5) required".to_string(),
            ));
        }
        Ok(Self {
            coupling_mu,
            scaling_delta,
        })
    }

    /// Computes the non-perturbative Maldacena-Qi wormhole mass gap $E_{\text{gap}} = \mu^{\frac{1}{1 - \Delta}}$.
    pub fn wormhole_energy_gap(&self) -> f64 {
        self.coupling_mu.powf(1.0 / (1.0 - self.scaling_delta))
    }

    /// Evaluates the Left-to-Right graviton-assisted transmission fidelity $F(t) = \sin^2(E_{\text{gap}} t) \in [0, 1]$.
    pub fn transmission_fidelity(&self, t: f64) -> f64 {
        let gap = self.wormhole_energy_gap();
        (gap * t).sin().powi(2)
    }
}
