//! Datta-Das Spin Field-Effect Transistor (Spin-FET) & Bychkov-Rashba Spin-Orbit Precession (Bychkov & Rashba, 1984; Datta & Das, 1990).
//!
//! ### 1. Gate-Tunable Rashba Spin Precession Angle $\Delta \theta$
//! Under Rashba spin-orbit Hamiltonian $H_R = \frac{\alpha_R}{\hbar} (\boldsymbol{\sigma} \times \mathbf{p})_z$, an electron of effective mass $m^*$
//! traversing channel length $L$ accumulates gate-voltage-tunable differential phase (in natural units $\hbar = 1$):
//! $$\Delta \theta = 2 m^* \alpha_R L, \qquad G(\alpha_R) = G_0 \cos^2\!\left(\frac{\Delta \theta}{2}\right)$$

use crate::error::{Error, Result};

/// Datta-Das Spin-FET & Rashba Spin-Orbit Precession Engine.
#[derive(Debug, Clone, Copy)]
pub struct DattaDasRashbaSpintronics {
    /// Carrier effective mass $m^* > 0$ (in natural units $\hbar = 1$).
    pub effective_mass: f64,
    /// Channel length $L > 0$.
    pub channel_length_l: f64,
}

impl Default for DattaDasRashbaSpintronics {
    fn default() -> Self {
        Self {
            effective_mass: 1.0,
            channel_length_l: 1.0,
        }
    }
}

impl DattaDasRashbaSpintronics {
    /// Creates a new Datta-Das Rashba Spintronics engine.
    pub fn new(effective_mass: f64, channel_length_l: f64) -> Result<Self> {
        if effective_mass <= 0.0 || channel_length_l <= 0.0 {
            return Err(Error::Msg(
                "effective_mass and channel_length_l must be > 0".to_string(),
            ));
        }
        Ok(Self {
            effective_mass,
            channel_length_l,
        })
    }

    /// Evaluates the Rashba spin precession angle $\Delta \theta = 2 m^* \alpha_R L$ and normalized spin-valve modulation $\cos^2(\Delta \theta / 2)$.
    pub fn spin_precession_and_modulation(&self, rashba_alpha: f64) -> (f64, f64) {
        let delta_theta = 2.0 * self.effective_mass * rashba_alpha * self.channel_length_l;
        let modulation = (0.5 * delta_theta).cos().powi(2);
        (delta_theta, modulation)
    }
}
