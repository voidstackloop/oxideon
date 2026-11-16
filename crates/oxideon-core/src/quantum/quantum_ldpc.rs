//! Quantum Low-Density Parity-Check (qLDPC) Hypergraph Product Codes & CSS Orthogonality (Tillich & Zémor, 2014; Panteleev & Kalachev, 2022).
//!
//! ### 1. Calderbank-Shor-Steane (CSS) Commutativity Condition
//! A CSS stabilizer code defined by binary parity-check matrices $H_X \in \mathbb{F}_2^{m_X \times n}$ and
//! $H_Z \in \mathbb{F}_2^{m_Z \times n}$ has mutually commuting $X$-type and $Z$-type stabilizers iff:
//! $$H_X H_Z^T \equiv 0 \pmod{2}$$
//! The number of logical qubits is $k = n - \operatorname{rk}_{\mathbb{F}_2}(H_X) - \operatorname{rk}_{\mathbb{F}_2}(H_Z)$.

use crate::error::{Error, Result};

/// Quantum LDPC CSS Parity-Check & Commutativity Verifier Engine.
#[derive(Debug, Clone)]
pub struct QuantumLdpcCodeEngine {
    /// Number of physical qubits $n$.
    pub num_qubits: usize,
    /// Flattened binary $X$-stabilizer parity-check matrix $H_X$ of shape `m_x * n`.
    pub hx: Vec<u8>,
    /// Flattened binary $Z$-stabilizer parity-check matrix $H_Z$ of shape `m_z * n`.
    pub hz: Vec<u8>,
}

impl QuantumLdpcCodeEngine {
    /// Constructs a qLDPC CSS code and validates matrix dimensions.
    pub fn new(num_qubits: usize, hx: Vec<u8>, hz: Vec<u8>) -> Result<Self> {
        if num_qubits == 0
            || hx.is_empty()
            || hz.is_empty()
            || hx.len() % num_qubits != 0
            || hz.len() % num_qubits != 0
        {
            return Err(Error::Msg("Invalid qLDPC matrix dimensions".to_string()));
        }
        Ok(Self { num_qubits, hx, hz })
    }

    /// Verifies exact symplectic CSS stabilizer commutativity $H_X H_Z^T \equiv 0 \pmod 2$.
    /// Re-checks construction invariants (pub matrix fields are tamperable).
    fn check_layout(&self) -> Result<(usize, usize, usize)> {
        let n = self.num_qubits;
        if n == 0 || self.hx.is_empty() || self.hz.is_empty() {
            return Err(Error::Msg("Invalid qLDPC matrix dimensions".into()));
        }
        if self.hx.len() % n != 0 || self.hz.len() % n != 0 {
            return Err(Error::Msg(format!(
                "qLDPC matrix lengths ({}, {}) disagree with num_qubits {n}",
                self.hx.len(),
                self.hz.len()
            )));
        }
        Ok((n, self.hx.len() / n, self.hz.len() / n))
    }

    pub fn verify_css_commutativity(&self) -> Result<bool> {
        let (n, mx, mz) = self.check_layout()?;
        for i in 0..mx {
            for j in 0..mz {
                let mut dot = 0_u8;
                for k in 0..n {
                    dot ^= (self.hx[i * n + k] & 1) & (self.hz[j * n + k] & 1);
                }
                if dot != 0 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Evaluates the $Z$-syndrome $s_Z = H_Z e_X \pmod 2$ for a bit-flip error vector $e_X \in \{0, 1\}^n$.
    pub fn compute_z_syndrome(&self, x_error: &[u8]) -> Result<Vec<u8>> {
        let (n, _mx, mz) = self.check_layout()?;
        if x_error.len() != n {
            return Err(Error::Msg("Error vector length mismatch".to_string()));
        }
        let mut synd = vec![0_u8; mz];
        for j in 0..mz {
            let mut parity = 0_u8;
            for k in 0..n {
                parity ^= (self.hz[j * n + k] & 1) & (x_error[k] & 1);
            }
            synd[j] = parity;
        }
        Ok(synd)
    }
}
