//! Groth16 Quadratic Arithmetic Program (QAP) Zero-Knowledge SNARK Pairing Verification (Gennaro, Gentry, Parno & Raykova, 2013; Groth, 2016).
//!
//! ### 1. Optimal 3-Group-Element Proof $(\pi_A \in \mathbb{G}_1, \pi_B \in \mathbb{G}_2, \pi_C \in \mathbb{G}_1)$ & Pairing Equation
//! For an R1CS circuit with $m$ wires, $n$ multiplication gates (QAP degree $n$), and $\ell$ public inputs, Groth16 verifies:
//! $$e(\pi_A, \pi_B) = e(\alpha, \beta) \cdot e\!\left(\sum_{i=0}^\ell a_i L_i, \gamma\right) \cdot e(\pi_C, \delta)$$
//! using exactly 3 group elements and 3 bilinear pairings.

use crate::error::{Error, Result};

/// Groth16 QAP Zero-Knowledge SNARK Circuit Analyzer.
#[derive(Debug, Clone, Copy)]
pub struct Groth16QapSnarkAnalyzer {
    /// Number of R1CS multiplication constraints $n \ge 1$.
    pub num_constraints_n: usize,
    /// Number of public inputs $\ell \ge 1$.
    pub num_public_inputs_l: usize,
}

impl Default for Groth16QapSnarkAnalyzer {
    fn default() -> Self {
        Self {
            num_constraints_n: 65536,
            num_public_inputs_l: 4,
        }
    }
}

impl Groth16QapSnarkAnalyzer {
    /// Creates a new Groth16 QAP SNARK Analyzer.
    pub fn new(num_constraints_n: usize, num_public_inputs_l: usize) -> Result<Self> {
        if num_constraints_n == 0 || num_public_inputs_l == 0 {
            return Err(Error::Msg(
                "num_constraints_n and num_public_inputs_l must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            num_constraints_n,
            num_public_inputs_l,
        })
    }

    /// Returns the constant proof size (3 group elements), pairing count (3), and prover NTT quasilinear complexity $n \log_2 n$.
    pub fn proof_elements_pairings_and_ntt_ops(&self) -> (usize, usize, f64) {
        let n_f = self.num_constraints_n as f64;
        (3, 3, n_f * n_f.log2().max(1.0))
    }
}
