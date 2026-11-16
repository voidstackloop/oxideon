//! Zero-Knowledge Machine Learning (zkML) Proof Engine.
//!
//! Provides verifiable inference for neural network operations over finite prime fields:
//! - Arithmetic circuit compilation with Rank-1 Constraint Systems (R1CS: $A \cdot s \circ B \cdot s = C \cdot s$).
//! - Exact integer field quantization of float tensors into prime field elements $\mathbb{F}_p$.
//! - Generation of verifiable succinct zero-knowledge execution traces and commitment proofs.
//! - Non-interactive verification of forward matrix operations without revealing private model weights.
//!
//! **Not a real proof system:** the "proof" is a hash transcript. The commitments are unsalted SHA-256 hashes of
//! low-entropy quantised values, and [`ZkMlEngine::verify_proof`] only recomputes the Fiat-Shamir hash of those
//! public commitments. It never checks a constraint or `evaluation_response`, so anyone can forge a proof that
//! verifies. It is neither sound nor zero-knowledge. See the security warning in [`crate::crypto`].

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// BN254 / Alt-bn128 scalar field prime order:
/// $p = 21888242871839275222246405745257275088548364400416034343698204186575808495617$.
/// For fast lightweight arithmetic simulation, we utilize a large safe prime modulus $p = 2^{61} - 1$
/// (Mersenne prime $M_{61} = 2305843009213693951$) supporting exact 64-bit non-overflow field operations.
pub const MERSENNE_61_PRIME: u64 = 2305843009213693951;

/// Finite Field Element $\mathbb{F}_p$ modulo $p = 2^{61} - 1$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FieldElement(pub u64);

#[allow(clippy::should_implement_trait)]
impl FieldElement {
    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);
    pub const MODULUS: u64 = MERSENNE_61_PRIME;

    pub fn new(val: i64) -> Self {
        let m = Self::MODULUS as i128;
        let v = (val as i128).rem_euclid(m);
        Self(v as u64)
    }

    pub fn from_u64(val: u64) -> Self {
        Self(val % Self::MODULUS)
    }

    pub fn add(self, other: Self) -> Self {
        let sum = (self.0 as u128) + (other.0 as u128);
        Self((sum % (Self::MODULUS as u128)) as u64)
    }

    pub fn sub(self, other: Self) -> Self {
        let m = Self::MODULUS as i128;
        let diff = (self.0 as i128) - (other.0 as i128);
        Self(diff.rem_euclid(m) as u64)
    }

    pub fn mul(self, other: Self) -> Self {
        let prod = (self.0 as u128) * (other.0 as u128);
        Self((prod % (Self::MODULUS as u128)) as u64)
    }

    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut res = Self::ONE;
        while exp > 0 {
            if exp % 2 == 1 {
                res = res.mul(base);
            }
            base = base.mul(base);
            exp /= 2;
        }
        res
    }

    pub fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            None
        } else {
            // By Fermat's Little Theorem: a^{p-2} = a^{-1} (mod p)
            Some(self.pow(Self::MODULUS - 2))
        }
    }
}

/// A linear combination of witness variables: $\sum c_i \cdot w_i$.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LinearCombination {
    /// Pairs of `(witness_index, coefficient)`. Variable 0 is reserved for the constant `1`.
    pub terms: Vec<(usize, FieldElement)>,
}

impl LinearCombination {
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    pub fn add_term(&mut self, var_idx: usize, coeff: FieldElement) {
        if coeff.0 != 0 {
            self.terms.push((var_idx, coeff));
        }
    }

    pub fn constant(val: FieldElement) -> Self {
        let mut lc = Self::new();
        lc.add_term(0, val);
        lc
    }

    pub fn variable(var_idx: usize) -> Self {
        let mut lc = Self::new();
        lc.add_term(var_idx, FieldElement::ONE);
        lc
    }

    /// Evaluates the linear combination given a full witness vector $w$.
    ///
    /// Terms whose variable index is missing from `witness` contribute zero;
    /// callers that need soundness must check [`Self::is_covered_by`] first
    /// ([`R1CSConstraint::is_satisfied`] does).
    pub fn evaluate(&self, witness: &[FieldElement]) -> FieldElement {
        let mut acc = FieldElement::ZERO;
        for &(var_idx, coeff) in &self.terms {
            if let Some(&w_val) = witness.get(var_idx) {
                acc = acc.add(coeff.mul(w_val));
            }
        }
        acc
    }

    /// Reports whether every referenced variable index exists in a witness of length `witness_len`.
    pub fn is_covered_by(&self, witness_len: usize) -> bool {
        self.terms.iter().all(|&(var_idx, _)| var_idx < witness_len)
    }
}

/// A Rank-1 Constraint System (R1CS) constraint: $\langle A, w \rangle \cdot \langle B, w \rangle = \langle C, w \rangle$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R1CSConstraint {
    pub a: LinearCombination,
    pub b: LinearCombination,
    pub c: LinearCombination,
}

impl R1CSConstraint {
    /// Checks $A \cdot B = C$ under `witness`. A witness that does not assign
    /// every referenced variable (including the constant at index 0) never
    /// satisfies: missing entries would otherwise read as zero and an empty
    /// witness would pass every constraint ($0 \cdot 0 = 0$).
    pub fn is_satisfied(&self, witness: &[FieldElement]) -> bool {
        if !self.a.is_covered_by(witness.len())
            || !self.b.is_covered_by(witness.len())
            || !self.c.is_covered_by(witness.len())
        {
            return false;
        }
        let a_val = self.a.evaluate(witness);
        let b_val = self.b.evaluate(witness);
        let c_val = self.c.evaluate(witness);
        a_val.mul(b_val) == c_val
    }
}

/// A succinct zkML inference proof containing cryptographic commitments and verification metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZkMlProof {
    /// Commitment to the private model weights $C_W$.
    pub weight_commitment: [u8; 32],
    /// Commitment to the public input tensor $C_X$.
    pub input_commitment: [u8; 32],
    /// Commitment to the verified output tensor $C_Y$.
    pub output_commitment: [u8; 32],
    /// Total number of R1CS arithmetic constraints verified.
    pub num_constraints: usize,
    /// Fiat-Shamir non-interactive challenge scalar.
    pub challenge: FieldElement,
    /// Response evaluation at the challenge point.
    pub evaluation_response: FieldElement,
}

/// Zero-Knowledge Machine Learning (zkML) Proof Engine for Verifiable Inference.
pub struct ZkMlEngine {
    pub constraints: Vec<R1CSConstraint>,
    pub num_variables: usize,
}

impl Default for ZkMlEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ZkMlEngine {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            // Variable 0 is reserved for constant ONE
            num_variables: 1,
        }
    }

    /// Allocates a new witness variable index.
    pub fn alloc_var(&mut self) -> usize {
        let idx = self.num_variables;
        self.num_variables += 1;
        idx
    }

    /// Enforces the arithmetic constraint $A \cdot B = C$.
    pub fn enforce_mul(
        &mut self,
        a: LinearCombination,
        b: LinearCombination,
        c: LinearCombination,
    ) {
        self.constraints.push(R1CSConstraint { a, b, c });
    }

    /// Quantizes a float slice into integer field elements given a fixed-point scale factor.
    pub fn quantize_tensor(tensor: &Tensor, scale: f64) -> Result<Vec<FieldElement>> {
        if !scale.is_finite() || scale == 0.0 {
            return Err(Error::Msg(format!(
                "quantize_tensor requires finite nonzero scale, got {scale}"
            )));
        }
        let vals = tensor.to_vec::<f64>()?;
        let mut elements = Vec::with_capacity(vals.len());
        for v in vals {
            if !v.is_finite() {
                return Err(Error::Msg(format!(
                    "quantize_tensor: non-finite tensor value {v}"
                )));
            }
            let scaled = (v * scale).round();
            if scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
                return Err(Error::Msg(format!(
                    "quantize_tensor: scaled value {scaled} exceeds i64 range"
                )));
            }
            elements.push(FieldElement::new(scaled as i64));
        }
        Ok(elements)
    }

    /// Builds R1CS constraints and a hash-based proof transcript for matrix multiplication $Y = X \cdot W^T$.
    /// The transcript is not sound and not zero-knowledge (see the module docs).
    pub fn prove_matmul(x: &Tensor, w: &Tensor, scale: f64) -> Result<(Tensor, ZkMlProof)> {
        let x_shape = x.shape();
        let w_shape = w.shape();
        if x_shape.len() != 2 || w_shape.len() != 2 || x_shape[1] != w_shape[1] {
            return Err(Error::Msg(
                "zkML matmul requires 2D tensors [M, K] and [N, K]".into(),
            ));
        }

        let m = x_shape[0];
        let k = x_shape[1];
        let n = w_shape[0];

        // 1. Compute true float output
        let wt = w.t()?;
        let y_float = x.matmul(&wt)?;

        // 2. Quantize inputs into field elements
        let x_fe = Self::quantize_tensor(x, scale)?;
        let w_fe = Self::quantize_tensor(w, scale)?;

        // 3. Build R1CS constraints and witness
        let mut engine = Self::new();
        // Allocate variables
        // Constant 1 is at index 0
        let mut witness = vec![FieldElement::ONE];

        let mut x_indices = Vec::with_capacity(m * k);
        for &val in &x_fe {
            let idx = engine.alloc_var();
            witness.push(val);
            x_indices.push(idx);
        }

        let mut w_indices = Vec::with_capacity(n * k);
        for &val in &w_fe {
            let idx = engine.alloc_var();
            witness.push(val);
            w_indices.push(idx);
        }

        let mut y_fe = Vec::with_capacity(m * n);

        for row in 0..m {
            for col in 0..n {
                let mut sum_fe = FieldElement::ZERO;
                for p in 0..k {
                    let x_idx = x_indices[row * k + p];
                    let w_idx = w_indices[col * k + p];

                    let term = witness[x_idx].mul(witness[w_idx]);
                    sum_fe = sum_fe.add(term);

                    // Add individual multiplier constraint
                    let prod_var = engine.alloc_var();
                    witness.push(term);

                    engine.enforce_mul(
                        LinearCombination::variable(x_idx),
                        LinearCombination::variable(w_idx),
                        LinearCombination::variable(prod_var),
                    );
                }
                let _y_var = engine.alloc_var();
                witness.push(sum_fe);
                y_fe.push(sum_fe);
            }
        }

        // Verify witness satisfies all constraints
        for (i, c) in engine.constraints.iter().enumerate() {
            if !c.is_satisfied(&witness) {
                return Err(Error::Msg(format!(
                    "zkML constraint {i} unsatisfied in witness trace"
                )));
            }
        }

        // 4. Generate commitments (using SHA-256 simulation)
        let weight_commitment = Self::hash_field_elements(&w_fe);
        let input_commitment = Self::hash_field_elements(&x_fe);
        let output_commitment = Self::hash_field_elements(&y_fe);

        // Fiat-Shamir challenge
        let mut combined_bytes = Vec::new();
        combined_bytes.extend_from_slice(&weight_commitment);
        combined_bytes.extend_from_slice(&input_commitment);
        combined_bytes.extend_from_slice(&output_commitment);
        let challenge_hash = Self::sha256(&combined_bytes);
        let challenge_u64 = u64::from_le_bytes(challenge_hash[..8].try_into().unwrap());
        let challenge = FieldElement::from_u64(challenge_u64);

        // Evaluation response: polynomial combination sum_i c_i * challenge^i
        let mut eval_resp = FieldElement::ZERO;
        let mut cur_pow = FieldElement::ONE;
        for c in &engine.constraints {
            let val = c.a.evaluate(&witness);
            eval_resp = eval_resp.add(val.mul(cur_pow));
            cur_pow = cur_pow.mul(challenge);
        }

        let proof = ZkMlProof {
            weight_commitment,
            input_commitment,
            output_commitment,
            num_constraints: engine.constraints.len(),
            challenge,
            evaluation_response: eval_resp,
        };

        Ok((y_float, proof))
    }

    /// Checks that the commitments match and the Fiat-Shamir hash was recomputed correctly.
    ///
    /// **Not a soundness check:** no constraint is evaluated, so a forged transcript passes.
    pub fn verify_proof(
        proof: &ZkMlProof,
        expected_weight_commitment: &[u8; 32],
        expected_input_commitment: &[u8; 32],
    ) -> bool {
        if proof.weight_commitment != *expected_weight_commitment {
            return false;
        }
        if proof.input_commitment != *expected_input_commitment {
            return false;
        }
        if proof.num_constraints == 0 {
            return false;
        }

        // Verify Fiat-Shamir challenge reconstruction
        let mut combined_bytes = Vec::new();
        combined_bytes.extend_from_slice(&proof.weight_commitment);
        combined_bytes.extend_from_slice(&proof.input_commitment);
        combined_bytes.extend_from_slice(&proof.output_commitment);
        let challenge_hash = Self::sha256(&combined_bytes);
        let challenge_u64 = u64::from_le_bytes(challenge_hash[..8].try_into().unwrap());
        let recomputed_challenge = FieldElement::from_u64(challenge_u64);

        recomputed_challenge == proof.challenge
    }

    /// Computes deterministic SHA-256 hash digest.
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        // Deterministic lightweight standard SHA-256 compression
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let k: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut msg = data.to_vec();
        let bit_len = (data.len() as u64) * 8;
        msg.push(0x80);
        while (msg.len() + 8) % 64 != 0 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_be_bytes());

        for chunk in msg.chunks(64) {
            let mut w = [0u32; 64];
            for (i, slot) in w.iter_mut().enumerate().take(16) {
                *slot = u32::from_be_bytes(chunk[i * 4..(i + 1) * 4].try_into().unwrap());
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let mut a = h[0];
            let mut b = h[1];
            let mut c = h[2];
            let mut d = h[3];
            let mut e = h[4];
            let mut f = h[5];
            let mut g = h[6];
            let mut hh = h[7];

            for (i, &w_val) in w.iter().enumerate() {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(k[i])
                    .wrapping_add(w_val);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut digest = [0u8; 32];
        for (i, &val) in h.iter().enumerate() {
            digest[i * 4..(i + 1) * 4].copy_from_slice(&val.to_be_bytes());
        }
        digest
    }

    /// Hashes field elements into a 32-byte commitment.
    pub fn hash_field_elements(elements: &[FieldElement]) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(elements.len() * 8);
        for elem in elements {
            bytes.extend_from_slice(&elem.0.to_le_bytes());
        }
        Self::sha256(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mul_constraint() -> R1CSConstraint {
        // w1 * w2 = w3 over variables 1..=3 (0 is the constant ONE).
        R1CSConstraint {
            a: LinearCombination::variable(1),
            b: LinearCombination::variable(2),
            c: LinearCombination::variable(3),
        }
    }

    #[test]
    fn satisfied_by_full_valid_witness() {
        let c = mul_constraint();
        let w = vec![
            FieldElement::ONE,
            FieldElement::new(6),
            FieldElement::new(7),
            FieldElement::new(42),
        ];
        assert!(c.is_satisfied(&w));
    }

    #[test]
    fn rejects_wrong_values() {
        let c = mul_constraint();
        let w = vec![
            FieldElement::ONE,
            FieldElement::new(6),
            FieldElement::new(7),
            FieldElement::new(43),
        ];
        assert!(!c.is_satisfied(&w));
    }

    #[test]
    fn empty_witness_satisfies_nothing() {
        let c = mul_constraint();
        assert!(!c.is_satisfied(&[]));
    }

    #[test]
    fn truncated_witness_satisfies_nothing() {
        // Missing w3 reads as zero under evaluate(); 6*7 != 0 must still fail.
        let c = mul_constraint();
        let w = vec![
            FieldElement::ONE,
            FieldElement::new(6),
            FieldElement::new(7),
        ];
        assert!(!c.is_satisfied(&w));
    }

    #[test]
    fn constant_term_requires_index_zero() {
        // 1 * w1 = w1 references the constant; without index 0 it must fail.
        let c = R1CSConstraint {
            a: LinearCombination::constant(FieldElement::ONE),
            b: LinearCombination::variable(1),
            c: LinearCombination::variable(1),
        };
        assert!(!c.is_satisfied(&[]));
        let w = vec![FieldElement::ONE, FieldElement::new(5)];
        assert!(c.is_satisfied(&w));
    }
}
