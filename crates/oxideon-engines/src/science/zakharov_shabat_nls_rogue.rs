//! Zakharov-Shabat AKNS Focusing Nonlinear Schrödinger (NLS) Integrability & Peregrine Rogue Wave (Zakharov & Shabat, 1972; Peregrine, 1983).
//!
//! ### 1. Peregrine Rational Breather Rogue Wave Amplitude
//! For the focusing 1D NLS $i \partial_t \psi + \frac{1}{2} \partial_x^2 \psi + |\psi|^2 \psi = 0$ on background amplitude $A_0$,
//! the Peregrine soliton is localized in both space and time with peak amplification factor $3$:
//! $$|\psi_P(x, t)| = A_0 \left| 1 - \frac{4(1 + 2 i A_0^2 t)}{1 + 4 A_0^2 x^2 + 4 A_0^4 t^2} \right|$$

use crate::error::{Error, Result};

/// Zakharov-Shabat Focusing NLS & Peregrine Rogue Wave Engine.
#[derive(Debug, Clone, Copy)]
pub struct ZakharovShabatNlsSoliton {
    /// Background plane-wave carrier amplitude $A_0 > 0$.
    pub background_a0: f64,
}

impl Default for ZakharovShabatNlsSoliton {
    fn default() -> Self {
        Self { background_a0: 1.0 }
    }
}

impl ZakharovShabatNlsSoliton {
    /// Creates a new Zakharov-Shabat NLS Soliton engine.
    pub fn new(background_a0: f64) -> Result<Self> {
        if background_a0 <= 0.0 || !background_a0.is_finite() {
            return Err(Error::Msg("background_a0 must be > 0".to_string()));
        }
        Ok(Self { background_a0 })
    }

    /// Evaluates the Peregrine breather modulus $|\psi_P(x, t)|$ and intensity amplification $|\psi_P / A_0|^2$ (equals $9$ at $(x, t) = (0, 0)$).
    pub fn peregrine_amplitude_and_amplification(&self, x: f64, t: f64) -> (f64, f64) {
        let a2 = self.background_a0 * self.background_a0;
        let denom = 1.0 + 4.0 * a2 * x * x + 4.0 * a2 * a2 * t * t;
        let re = 1.0 - 4.0 / denom;
        let im = -8.0 * a2 * t / denom;
        let amp_factor_sq = re * re + im * im;
        (self.background_a0 * amp_factor_sq.sqrt(), amp_factor_sq)
    }
}
