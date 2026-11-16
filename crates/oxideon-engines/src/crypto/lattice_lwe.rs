//! Post-Quantum Lattice Cryptography: Ring Learning With Errors (Ring-LWE) & Cyclotomic Polynomial Arithmetic.
//!
//! ### 1. Cyclotomic Quotient Ring $R_q = \mathbb{Z}_q\[X\] / (X^n + 1)$
//! In NIST post-quantum lattice cryptography (ML-KEM / Kyber & ML-DSA / Dilithium), operations take place
//! over the negacyclic polynomial ring $R_q = \mathbb{Z}_q\[X\] / (X^n + 1)$ where $X^n \equiv -1 \pmod{q}$.
//! Negacyclic convolution $(a \star b)_k$ satisfies:
//! $$(a \star b)_k = \left(\sum_{i+j=k} a_i b_j - \sum_{i+j=k+n} a_i b_j\right) \bmod q$$
//!
//! ### 2. Ring-LWE Public-Key Encryption & Decryption
//! • **Key Generation**: Sample secret $s \in R_q$ and error $e \in R_q$ with small coefficients; public key is $(a, b = a \star s + e \bmod q)$.
//! • **Encryption of bit vector $m \in \{0, 1\}^n$**: Encode message as $\lfloor q / 2 \rfloor m$; sample ephemeral $r, e_1, e_2$ and output:
//! $$u = a \star r + e_1 \bmod q, \qquad v = b \star r + e_2 + \lfloor q / 2 \rfloor m \bmod q$$
//! • **Decryption**: Compute noisy plaintext $w = (v - u \star s) \bmod q$ and decode each coefficient to $1$ if closer to $\lfloor q/2 \rfloor$ than to $0$.
//!
//! **Not production crypto:** this is not ML-KEM. All randomness (`a`, `s`, `e`, `r`, ...) is caller-supplied,
//! the `i64` arithmetic is not constant-time and overflows for large `q`. See the security warning in [`crate::crypto`].

use crate::error::{Error, Result};

/// Post-Quantum Ring-LWE Cyclotomic Lattice Cryptography Engine over $R_q = \mathbb{Z}_q\[X\]/(X^n + 1)$.
#[derive(Debug, Clone)]
pub struct LatticeLweEngine {
    /// Ring degree $n$ (power of 2).
    pub degree_n: usize,
    /// Prime modulus $q$.
    pub modulus_q: i64,
}

impl LatticeLweEngine {
    /// Creates a new Ring-LWE engine over $\mathbb{Z}_q\[X\]/(X^n + 1)$.
    pub fn new(degree_n: usize, modulus_q: i64) -> Result<Self> {
        if degree_n == 0 || !degree_n.is_power_of_two() {
            return Err(Error::Msg(
                "degree_n must be a positive power of 2".to_string(),
            ));
        }
        if modulus_q < 17 {
            return Err(Error::Msg("modulus_q must be >= 17".to_string()));
        }
        Ok(Self {
            degree_n,
            modulus_q,
        })
    }

    #[inline]
    fn mod_q(&self, x: i64) -> i64 {
        x.rem_euclid(self.modulus_q)
    }

    /// Negacyclic polynomial multiplication $a(X) \cdot b(X) \bmod (X^n + 1, q)$.
    pub fn negacyclic_mul(&self, a: &[i64], b: &[i64]) -> Result<Vec<i64>> {
        let n = self.degree_n;
        if a.len() != n || b.len() != n {
            return Err(Error::Msg(
                "Polynomial length must equal degree_n".to_string(),
            ));
        }
        let mut res = vec![0_i64; n];
        for (i, &ai) in a.iter().enumerate().take(n) {
            for (j, &bj) in b.iter().enumerate().take(n) {
                let prod = ai * bj;
                if i + j < n {
                    res[i + j] = self.mod_q(res[i + j] + prod);
                } else {
                    res[i + j - n] = self.mod_q(res[i + j - n] - prod);
                }
            }
        }
        Ok(res)
    }

    /// Computes the Ring-LWE public key $b = (a \star s + e) \bmod q$.
    pub fn compute_public_key(&self, a: &[i64], s: &[i64], e: &[i64]) -> Result<Vec<i64>> {
        let n = self.degree_n;
        if e.len() != n {
            return Err(Error::Msg("Error polynomial length mismatch".to_string()));
        }
        let a_s = self.negacyclic_mul(a, s)?;
        Ok(a_s
            .iter()
            .zip(e.iter())
            .map(|(&x, &err)| self.mod_q(x + err))
            .collect())
    }

    /// Encrypts an $n$-bit message $m \in \{0, 1\}^n$ into ciphertext pair $(u, v) \in R_q \times R_q$.
    pub fn encrypt(
        &self,
        a: &[i64],
        b: &[i64],
        r: &[i64],
        e1: &[i64],
        e2: &[i64],
        message_bits: &[u8],
    ) -> Result<(Vec<i64>, Vec<i64>)> {
        let n = self.degree_n;
        if e1.len() != n || e2.len() != n || message_bits.len() != n {
            return Err(Error::Msg(
                "Vector length mismatch in Ring-LWE encrypt".to_string(),
            ));
        }
        let half_q = self.modulus_q / 2;
        let ar = self.negacyclic_mul(a, r)?;
        let br = self.negacyclic_mul(b, r)?;

        let u: Vec<i64> = ar
            .iter()
            .zip(e1.iter())
            .map(|(&x, &e)| self.mod_q(x + e))
            .collect();
        let v: Vec<i64> = br
            .iter()
            .zip(e2.iter())
            .zip(message_bits.iter())
            .map(|((&x, &e), &bit)| self.mod_q(x + e + (bit as i64) * half_q))
            .collect();
        Ok((u, v))
    }

    /// Decrypts a Ring-LWE ciphertext $(u, v)$ using secret key $s \in R_q$.
    pub fn decrypt(&self, u: &[i64], v: &[i64], s: &[i64]) -> Result<Vec<u8>> {
        let n = self.degree_n;
        if v.len() != n {
            return Err(Error::Msg("Ciphertext v length mismatch".to_string()));
        }
        let us = self.negacyclic_mul(u, s)?;
        let q = self.modulus_q;
        let q_quarter = q / 4;
        // i128 intermediate: 3*q overflows i64 for large legal q (only the
        // product overflows; 3q/4 itself always fits).
        let q_three_quarter = (q as i128 * 3 / 4) as i64;

        let mut bits = Vec::with_capacity(n);
        for (&vi, &usi) in v.iter().zip(us.iter()) {
            let diff = self.mod_q(vi - usi);
            let bit = u8::from((q_quarter..=q_three_quarter).contains(&diff));
            bits.push(bit);
        }
        Ok(bits)
    }
}
