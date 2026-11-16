//! Groth16 zk-SNARKs, Rank-1 Constraint Systems (R1CS) & Quadratic Arithmetic Program (QAP) Divisibility (Gennaro et al., 2013; Groth, 2016).
//!
//! ### 1. R1CS Hadamard Gate Satisfaction $(A \mathbf{z}) \circ (B \mathbf{z}) \equiv C \mathbf{z} \pmod p$
//! An arithmetic circuit flattened into $m$ Rank-1 Constraint System (R1CS) gates over $\mathbb{F}_p$ is satisfied by witness
//! vector $\mathbf{z} = (1, x_1, \dots, w_k)^T$ iff for every gate $i \in 0..m$:
//! $$\left(\sum_j A_{ij} z_j\right) \cdot \left(\sum_j B_{ij} z_j\right) - \sum_j C_{ij} z_j \equiv 0 \pmod p$$
//!
//! **Not a SNARK:** this module only checks that a plaintext witness satisfies the R1CS gates. There is no trusted
//! setup, proof, pairing or zero-knowledge. See the security warning in [`crate::crypto`].

use crate::error::{Error, Result};

/// Groth16 R1CS & Quadratic Arithmetic Program (QAP) Verifier Engine.
#[derive(Debug, Clone)]
pub struct QuadraticArithmeticProgramEngine {
    /// Prime field modulus $p \ge 3$.
    pub modulus_p: i64,
    /// Number of constraints $m$.
    pub num_constraints: usize,
    /// Witness vector length $n$.
    pub num_vars: usize,
}

impl QuadraticArithmeticProgramEngine {
    /// Creates a new R1CS / QAP Verifier engine over $\mathbb{F}_p$.
    pub fn new(modulus_p: i64, num_constraints: usize, num_vars: usize) -> Result<Self> {
        if modulus_p < 3 || num_constraints == 0 || num_vars == 0 {
            return Err(Error::Msg(
                "modulus_p >= 3 and non-zero R1CS dimensions required".to_string(),
            ));
        }
        Ok(Self {
            modulus_p,
            num_constraints,
            num_vars,
        })
    }

    /// Verifies exact R1CS satisfaction $(A z)_i (B z)_i \equiv (C z)_i \pmod p$ for all $m$ gates.
    pub fn verify_r1cs_witness(&self, a: &[i64], b: &[i64], c: &[i64], z: &[i64]) -> Result<bool> {
        let m = self.num_constraints;
        let n = self.num_vars;
        let expect = m
            .checked_mul(n)
            .ok_or_else(|| Error::Msg(format!("R1CS dimensions {m}x{n} overflow usize")))?;
        if a.len() != expect || b.len() != expect || c.len() != expect || z.len() != n {
            return Err(Error::Msg(
                "R1CS matrix or witness dimension mismatch".to_string(),
            ));
        }
        let p = self.modulus_p;
        if p < 3 {
            return Err(Error::Msg("modulus_p must be >= 3".to_string()));
        }
        // i128 accumulators: matrix and witness entries are unbounded i64s,
        // so every product can overflow (debug panic / wrapped wrong verdict).
        for i in 0..m {
            let mut az = 0_i64;
            let mut bz = 0_i64;
            let mut cz = 0_i64;
            for j in 0..n {
                az = ((az as i128 + a[i * n + j] as i128 * z[j] as i128).rem_euclid(p as i128))
                    as i64;
                bz = ((bz as i128 + b[i * n + j] as i128 * z[j] as i128).rem_euclid(p as i128))
                    as i64;
                cz = ((cz as i128 + c[i * n + j] as i128 * z[j] as i128).rem_euclid(p as i128))
                    as i64;
            }
            if (az as i128 * bz as i128 - cz as i128).rem_euclid(p as i128) != 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
