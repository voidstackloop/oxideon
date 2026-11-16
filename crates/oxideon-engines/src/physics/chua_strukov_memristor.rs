//! Chua-Strukov $\mathrm{TiO}_2$ Memristor Flux-Charge Constitutive Law & Windowed State Dynamics (Chua, 1971; Strukov, Snider, Stewart & Williams, 2008).
//!
//! ### 1. Linear Ion-Drift Memristance $M(w)$ & Joglekar Window Function
//! For normalized doped region width $x = w / D \in [0, 1]$ with low/high resistances $R_{\mathrm{on}} < R_{\mathrm{off}}$:
//! $$M(x) = R_{\mathrm{on}} x + R_{\mathrm{off}} (1 - x), \qquad \frac{dx}{dt} = \frac{\mu_v R_{\mathrm{on}}}{D^2} i(t) f_p(x)$$

use crate::error::{Error, Result};

/// Chua-Strukov Memristor & Neuromorphic Synaptic Conductance Engine.
#[derive(Debug, Clone, Copy)]
pub struct ChuaStrukovMemristorSynapse {
    /// Fully doped resistance $R_{\mathrm{on}} > 0$ ($\Omega$).
    pub r_on: f64,
    /// Undoped resistance $R_{\mathrm{off}} > R_{\mathrm{on}}$ ($\Omega$).
    pub r_off: f64,
}

impl Default for ChuaStrukovMemristorSynapse {
    fn default() -> Self {
        Self {
            r_on: 100.0,
            r_off: 16000.0,
        }
    }
}

impl ChuaStrukovMemristorSynapse {
    /// Creates a new Chua-Strukov Memristor Synapse engine.
    pub fn new(r_on: f64, r_off: f64) -> Result<Self> {
        if r_on <= 0.0 || r_off <= r_on {
            return Err(Error::Msg("0 < r_on < r_off required".to_string()));
        }
        Ok(Self { r_on, r_off })
    }

    /// Evaluates memristance $M(x)$ and Joglekar window factor $f_1(x) = 1 - (2x - 1)^2 = 4x(1 - x)$ at state $x \in [0, 1]$.
    pub fn memristance_and_joglekar_window(&self, state_x: f64) -> Result<(f64, f64)> {
        if !(0.0..=1.0).contains(&state_x) {
            return Err(Error::Msg("state_x must be in [0, 1]".to_string()));
        }
        let m_x = self.r_on * state_x + self.r_off * (1.0 - state_x);
        let window = 4.0 * state_x * (1.0 - state_x);
        Ok((m_x, window))
    }
}
