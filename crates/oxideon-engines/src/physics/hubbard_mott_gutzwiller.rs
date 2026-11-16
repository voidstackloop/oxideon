//! Hubbard Model, Gutzwiller Variational Wavefunction & Brinkman-Rice Mott Metal-Insulator Transition (Gutzwiller, 1963; Brinkman & Rice, 1970).
//!
//! ### 1. Brinkman-Rice Quasiparticle Renormalization Weight $Z(U)$ at Half-Filling
//! In the half-filled Hubbard model $H = -t \sum c^\dagger c + U \sum n_\uparrow n_\downarrow$ with uncorrelated kinetic energy
//! $-|\bar{\varepsilon}_0|$ and critical Mott repulsion $U_c = 8 |\bar{\varepsilon}_0|$, the Gutzwiller projection yields
//! double occupancy $d(U) = \frac{1}{4}(1 - U / U_c)$ and Fermi-liquid quasiparticle weight:
//! $$Z(U) = \frac{m}{m^*} = 1 - \left(\frac{U}{U_c}\right)^2 \quad (U \le U_c), \qquad Z(U) = 0 \quad (U > U_c \text{ Mott Insulator})$$

use crate::error::{Error, Result};

/// Hubbard-Gutzwiller Variational & Brinkman-Rice Mott Transition Engine.
#[derive(Debug, Clone, Copy)]
pub struct BrinkmanRiceGutzwillerEngine {
    /// Critical Hubbard repulsion $U_c = 8 |\bar{\varepsilon}_0| > 0$.
    pub critical_uc: f64,
}

impl BrinkmanRiceGutzwillerEngine {
    /// Creates a new Brinkman-Rice Gutzwiller engine.
    pub fn new(critical_uc: f64) -> Result<Self> {
        if critical_uc <= 0.0 {
            return Err(Error::Msg("critical_uc must be > 0".to_string()));
        }
        Ok(Self { critical_uc })
    }

    /// Evaluates the quasiparticle renormalization factor $Z(U) \in [0, 1]$ and double occupancy $d(U) \in [0, 0.25]$ at repulsion $U \ge 0$.
    pub fn quasiparticle_z_and_double_occupancy(&self, repulsion_u: f64) -> (f64, f64) {
        if repulsion_u >= self.critical_uc {
            (0.0, 0.0)
        } else {
            let r = (repulsion_u / self.critical_uc).max(0.0);
            (1.0 - r * r, 0.25 * (1.0 - r))
        }
    }
}
