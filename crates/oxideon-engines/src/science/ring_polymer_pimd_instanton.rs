//! Feynman Path-Integral Ring Polymer Molecular Dynamics (RPMD) & Instanton Tunneling Crossover $T_c$ (Craig & Manolopoulos, 2004).
//!
//! ### 1. Ring Polymer Normal Mode Frequencies & Instanton Crossover Temperature
//! Discretizing the imaginary-time Feynman path integral into $P$ beads at temperature $T$ connects adjacent beads
//! with harmonic springs of frequency $\omega_P = \frac{P k_B T}{\hbar}$ and internal normal-mode frequencies:
//! $$\omega_k = 2 \omega_P \sin\left(\frac{k \pi}{P}\right) \quad (k = 0, \dots, P - 1)$$
//! Deep quantum tunneling sets in below the **Instanton Crossover Temperature** $k_B T_c = \frac{\hbar \omega_b}{2 \pi}$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Path-Integral Ring Polymer Molecular Dynamics (RPMD) & Instanton Engine.
#[derive(Debug, Clone, Copy)]
pub struct RingPolymerMolecularDynamics {
    /// Number of imaginary-time Trotter beads $P \ge 2$.
    pub num_beads_p: usize,
    /// Bead spring frequency $\omega_P = P k_B T / \hbar > 0$.
    pub omega_p: f64,
}

impl RingPolymerMolecularDynamics {
    /// Creates a new Ring Polymer Molecular Dynamics engine.
    pub fn new(num_beads_p: usize, omega_p: f64) -> Result<Self> {
        if num_beads_p < 2 || omega_p <= 0.0 {
            return Err(Error::Msg(
                "num_beads_p >= 2 and omega_p > 0 required".to_string(),
            ));
        }
        Ok(Self {
            num_beads_p,
            omega_p,
        })
    }

    /// Evaluates the $k$-th internal ring-polymer normal mode frequency $\omega_k = 2 \omega_P \sin(k \pi / P)$ (note $\omega_0 = 0$ is the centroid).
    pub fn normal_mode_frequency(&self, mode_k: usize) -> f64 {
        2.0 * self.omega_p * (PI * (mode_k as f64) / (self.num_beads_p as f64)).sin()
    }
}
