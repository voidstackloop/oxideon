//! Peters-Mathews Binary Inspiral Quadrupole Radiation & Gravitational-Wave Chirp Mass $\mathcal{M}_c$ (Peters & Mathews, 1963).
//!
//! ### 1. Chirp Mass $\mathcal{M}_c$ & Frequency Evolution $\dot{f} \propto \mathcal{M}_c^{5/3} f^{11/3}$
//! For a compact binary of component masses $m_1, m_2 > 0$, the leading quadrupole phase and frequency evolution
//! depend solely on the **Chirp Mass**:
//! $$\mathcal{M}_c = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}} = M \eta^{3/5}$$
//! where $\eta = \frac{m_1 m_2}{(m_1 + m_2)^2} \in (0, 1/4]$ is the symmetric mass ratio.

use crate::error::{Error, Result};

/// Peters-Mathews Binary Inspiral & Gravitational-Wave Chirp Mass Engine.
#[derive(Debug, Clone, Copy)]
pub struct GravitationalWaveChirpEngine {
    /// Primary mass $m_1 > 0$.
    pub m1: f64,
    /// Secondary mass $m_2 > 0$.
    pub m2: f64,
}

impl GravitationalWaveChirpEngine {
    /// Creates a new Gravitational-Wave Chirp engine.
    pub fn new(m1: f64, m2: f64) -> Result<Self> {
        if m1 <= 0.0 || m2 <= 0.0 {
            return Err(Error::Msg(
                "Component masses m1 and m2 must be > 0".to_string(),
            ));
        }
        Ok(Self { m1, m2 })
    }

    /// Computes the chirp mass $\mathcal{M}_c = (m_1 m_2)^{3/5} / (m_1 + m_2)^{1/5}$ and symmetric mass ratio $\eta \in (0, 0.25]$.
    pub fn chirp_mass_and_symmetric_eta(&self) -> (f64, f64) {
        let total = self.m1 + self.m2;
        let prod = self.m1 * self.m2;
        let mc = prod.powf(0.6) / total.powf(0.2);
        let eta = prod / (total * total);
        (mc, eta)
    }
}
