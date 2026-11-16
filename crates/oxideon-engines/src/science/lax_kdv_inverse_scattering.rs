//! Lax Pair, Gardner-Greene-Kruskal-Miura (GGKM) Inverse Scattering Transform & KdV Soliton Collisions (GGKM, 1967; Lax, 1968).
//!
//! ### 1. Exact KdV 1-Soliton Profile & 2-Soliton Collision Phase Shift
//! For $\partial_t u + 6 u \partial_x u + \partial_x^3 u = 0$, a discrete Schrödinger eigenvalue $-\kappa^2$ ($\kappa > 0$) generates
//! $u(x, t) = 2 \kappa^2 \operatorname{sech}^2(\kappa (x - 4 \kappa^2 t - x_0))$, and a fast soliton $\kappa_1 > \kappa_2$ experiences forward spatial shift:
//! $$\Delta x_1 = \frac{1}{\kappa_1} \ln\!\left(\frac{\kappa_1 + \kappa_2}{\kappa_1 - \kappa_2}\right)$$

use crate::error::{Error, Result};

/// Lax-GGKM KdV Inverse Scattering & Soliton Phase-Shift Engine.
#[derive(Debug, Clone, Copy)]
pub struct LaxKdvInverseScatteringSoliton {
    /// Spectral parameter $\kappa_1 > 0$ of the primary soliton.
    pub kappa1: f64,
}

impl Default for LaxKdvInverseScatteringSoliton {
    fn default() -> Self {
        Self { kappa1: 2.0 }
    }
}

impl LaxKdvInverseScatteringSoliton {
    /// Creates a new Lax KdV Inverse Scattering Soliton engine.
    pub fn new(kappa1: f64) -> Result<Self> {
        if kappa1 <= 0.0 || !kappa1.is_finite() {
            return Err(Error::Msg("kappa1 must be > 0".to_string()));
        }
        Ok(Self { kappa1 })
    }

    /// Evaluates the 1-soliton amplitude $2 \kappa_1^2$, speed $c_1 = 4 \kappa_1^2$, and 2-soliton collision phase shift $\Delta x_1$ against slower soliton $0 < \kappa_2 < \kappa_1$.
    pub fn amplitude_speed_and_collision_shift(&self, kappa2: f64) -> Result<(f64, f64, f64)> {
        if kappa2 <= 0.0 || kappa2 >= self.kappa1 {
            return Err(Error::Msg("0 < kappa2 < kappa1 required".to_string()));
        }
        let amp = 2.0 * self.kappa1 * self.kappa1;
        let speed = 4.0 * self.kappa1 * self.kappa1;
        let shift = ((self.kappa1 + kappa2) / (self.kappa1 - kappa2)).ln() / self.kappa1;
        Ok((amp, speed, shift))
    }
}
