//! Laughlin Fractional Quantum Hall Wavefunction $\Psi_m$, Fractional Charge $e^* = e/m$ & Anyon Braiding (Laughlin, 1983).
//!
//! ### 1. Laughlin State at Filling Factor $\nu = 1/m$ ($m$ Odd Integer)
//! In the lowest Landau level, the Laughlin trial wavefunction for $N$ electrons at filling $\nu = 1/m$ ($m = 3, 5, \dots$) is:
//! $$\Psi_m(z_1, \dots, z_N) = \prod_{1 \le j < k \le N} (z_j - z_k)^m \exp\left(-\frac{1}{4 \ell_B^2} \sum_{i=1}^N |z_i|^2\right)$$
//! Elementary quasihole excitations carry fractional electric charge $e^* = e / m$ and abelian anyon exchange phase $\theta = \pi / m$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Laughlin Fractional Quantum Hall Effect (FQHE) & Anyon Quasiparticle Engine.
#[derive(Debug, Clone, Copy)]
pub struct LaughlinFqheEngine {
    /// Odd denominator $m \in \{1, 3, 5, 7, \dots\}$.
    pub denominator_m: usize,
}

impl LaughlinFqheEngine {
    /// Creates a new Laughlin FQHE engine for odd integer $m \ge 1$.
    pub fn new(denominator_m: usize) -> Result<Self> {
        if denominator_m == 0 || denominator_m % 2 == 0 {
            return Err(Error::Msg(
                "denominator_m must be an odd positive integer (fermionic Laughlin state)"
                    .to_string(),
            ));
        }
        Ok(Self { denominator_m })
    }

    /// Returns the fractional Hall filling factor $\nu = 1/m$, quasiparticle fractional charge $e^*/e = 1/m$, and anyon exchange phase $\theta = \pi / m$.
    pub fn filling_charge_and_anyon_angle(&self) -> (f64, f64, f64) {
        let m = self.denominator_m as f64;
        (1.0 / m, 1.0 / m, PI / m)
    }
}
