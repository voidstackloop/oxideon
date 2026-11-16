//! Zurek Quantum Darwinism, Pointer States, Redundancy Plateau & Quantum Mutual Information $I(S : F_k)$ (Zurek, 2009).
//!
//! ### 1. Classical Objectivity from Redundant Environmental Imprinting
//! For a system $S$ decohered into an $N$-fragment environment $E = \bigotimes_{i=1}^N E_i$ via a GHZ-like branching state
//! $|\Psi_{SE}\rangle = \frac{1}{\sqrt{2}}(|0_S\rangle |0\rangle^{\otimes N} + |1_S\rangle |\phi\rangle^{\otimes N})$ with overlap $|\langle 0|\phi\rangle| = c \in [0, 1)$,
//! the quantum mutual information $I(S : F_k) = S(\rho_S) + S(\rho_{F_k}) - S(\rho_{S F_k})$ between $S$ and a fraction
//! $f = k / N$ of the environment exhibits the universal **Classicality Plateau** $I(S : F_k) \approx H(S) = 1$ bit!

use crate::error::{Error, Result};

/// Zurek Quantum Darwinism & Redundancy Plateau Engine.
#[derive(Debug, Clone, Copy)]
pub struct QuantumDarwinismEngine {
    /// Total number of environment fragments $N \ge 2$.
    pub num_env_fragments: usize,
    /// Single-fragment pointer overlap $c = |\langle \phi_0 | \phi_1 \rangle| \in [0, 1)$.
    pub fragment_overlap_c: f64,
}

impl QuantumDarwinismEngine {
    /// Creates a new Quantum Darwinism engine.
    pub fn new(num_env_fragments: usize, fragment_overlap_c: f64) -> Result<Self> {
        if num_env_fragments < 2 || !(0.0..1.0).contains(&fragment_overlap_c) {
            return Err(Error::Msg(
                "num_env_fragments >= 2 and fragment_overlap_c in [0, 1) required".to_string(),
            ));
        }
        Ok(Self {
            num_env_fragments,
            fragment_overlap_c,
        })
    }

    fn binary_entropy(lambda_plus: f64) -> f64 {
        let p = lambda_plus.clamp(1e-15, 1.0 - 1e-15);
        let q = 1.0 - p;
        -(p * p.log2() + q * q.log2())
    }

    /// Computes the exact quantum mutual information $I(S : F_k)$ in bits for a subset of $k \in 1..=N$ environment fragments.
    pub fn mutual_information_bits(&self, fragment_count_k: usize) -> Result<f64> {
        let n = self.num_env_fragments;
        if fragment_count_k == 0 || fragment_count_k > n {
            return Err(Error::Msg("fragment_count_k must be in 1..=N".to_string()));
        }
        let s_sys = Self::binary_entropy(0.5 * (1.0 + self.fragment_overlap_c.powi(n as i32)));
        let s_fk = Self::binary_entropy(
            0.5 * (1.0 + self.fragment_overlap_c.powi(fragment_count_k as i32)),
        );
        let s_sfk = Self::binary_entropy(
            0.5 * (1.0 + self.fragment_overlap_c.powi((n - fragment_count_k) as i32)),
        );
        Ok(s_sys + s_fk - s_sfk)
    }
}
