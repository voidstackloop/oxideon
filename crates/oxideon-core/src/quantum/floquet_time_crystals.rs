//! Discrete Floquet Time Crystals (DTC), Subharmonic Period-Doubling & Many-Body Localization Rigidity (Else, Bauer & Nayak, 2016).
//!
//! ### 1. $\pi$-Spin-Glass Floquet Eigenstate Order
//! Under a period-$T$ Floquet drive $U_F = e^{-i H_{\text{MBL}} T} \prod_i e^{-i (\pi/2 - \epsilon) X_i}$,
//! a Discrete Time Crystal spontaneously breaks discrete time-translation symmetry $t \mapsto t + T$, oscillating
//! at subharmonic frequency $\omega / 2$ (period $2T$): $\langle Z_i(n T) \rangle = (-1)^n m_0$.

use crate::error::{Error, Result};

/// Discrete Floquet Time Crystal (DTC) Subharmonic Order Engine.
#[derive(Debug, Clone, Copy)]
pub struct FloquetTimeCrystalEngine {
    /// Pulse rotation imperfection $\epsilon \in [0, 0.5)$.
    pub pulse_imperfection_eps: f64,
    /// Ising MBL interaction coupling $J > 0$.
    pub interaction_j: f64,
}

impl FloquetTimeCrystalEngine {
    /// Creates a new Discrete Floquet Time Crystal engine.
    pub fn new(pulse_imperfection_eps: f64, interaction_j: f64) -> Result<Self> {
        if !(0.0..0.5).contains(&pulse_imperfection_eps) || interaction_j <= 0.0 {
            return Err(Error::Msg(
                "pulse_imperfection_eps in [0, 0.5) and interaction_j > 0 required".to_string(),
            ));
        }
        Ok(Self {
            pulse_imperfection_eps,
            interaction_j,
        })
    }

    /// Evaluates the stroboscopic magnetization $M(n T) = (-1)^n \cos^{2}( \epsilon / J )$ at period index $n$.
    pub fn stroboscopic_magnetization(&self, period_n: usize) -> f64 {
        let sign = if period_n % 2 == 0 { 1.0 } else { -1.0 };
        let amp = (self.pulse_imperfection_eps / self.interaction_j)
            .cos()
            .powi(2);
        sign * amp
    }
}
