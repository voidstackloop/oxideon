//! Loop Quantum Gravity (LQG), $SU(2)$ Spin Networks, Barbero-Immirzi Parameter $\gamma$ &
//! Rovelli-Smolin Discrete Area Spectrum (Rovelli & Smolin, 1995).
//!
//! ### 1. Quantized Area Operator $\hat{A}_S$ Punctured by Spin Network Edges
//! In Loop Quantum Gravity, a 2-surface $S$ punctured transversely by $N$ spin-network edges carrying $SU(2)$
//! irreducible representations $j_i \in \{\frac{1}{2}, 1, \frac{3}{2}, \dots\}$ has discrete area eigenvalues:
//! $$A(S) = 8 \pi \gamma \ell_P^2 \sum_{i=1}^N \sqrt{j_i (j_i + 1)}$$
//! where $\gamma \approx 0.2375$ is the Barbero-Immirzi parameter fixed by the Bekenstein-Hawking black hole entropy
//! $S_{\text{BH}} = \frac{A}{4 \ell_P^2}$.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Loop Quantum Gravity $SU(2)$ Spin Network & Discrete Area Spectrum Engine.
#[derive(Debug, Clone, Copy)]
pub struct SpinNetworkLqgEngine {
    /// Barbero-Immirzi parameter $\gamma > 0$.
    pub immirzi_gamma: f64,
    /// Planck length $\ell_P > 0$.
    pub planck_length: f64,
}

impl SpinNetworkLqgEngine {
    /// Creates a new Loop Quantum Gravity spin network engine.
    pub fn new(immirzi_gamma: f64, planck_length: f64) -> Result<Self> {
        if immirzi_gamma <= 0.0 || planck_length <= 0.0 {
            return Err(Error::Msg(
                "immirzi_gamma and planck_length must be > 0".to_string(),
            ));
        }
        Ok(Self {
            immirzi_gamma,
            planck_length,
        })
    }

    /// Computes the Rovelli-Smolin quantized area eigenvalue $A = 8 \pi \gamma \ell_P^2 \sum_k \sqrt{j_k(j_k + 1)}$
    /// given twice-spins `two_j` ($2 j_k \in \mathbb{N}_{\ge 1}$).
    pub fn quantized_area_eigenvalue(&self, two_j_punctures: &[u32]) -> Result<f64> {
        if two_j_punctures.is_empty() {
            return Err(Error::Msg(
                "At least one spin-network puncture is required".to_string(),
            ));
        }
        let mut casimir_sum = 0.0;
        for &two_j in two_j_punctures {
            if two_j == 0 {
                return Err(Error::Msg("Spin puncture 2j must be >= 1".to_string()));
            }
            let j = 0.5 * (two_j as f64);
            casimir_sum += (j * (j + 1.0)).sqrt();
        }
        Ok(8.0 * PI * self.immirzi_gamma * self.planck_length.powi(2) * casimir_sum)
    }

    /// Verifies the Clebsch-Gordan $SU(2)$ trivalent spin-network intertwiner admissibility at a 3-valent node $(2j_1, 2j_2, 2j_3)$.
    pub fn is_admissible_trivalent_intertwiner(two_j1: u32, two_j2: u32, two_j3: u32) -> bool {
        // Widened: the triangle checks are exact for every u32 input.
        let (a, b, c) = (two_j1 as u64, two_j2 as u64, two_j3 as u64);
        let sum = a + b + c;
        sum % 2 == 0 && a + b >= c && b + c >= a && c + a >= b
    }
}
