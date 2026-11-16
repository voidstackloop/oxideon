//! Fault-Tolerant Quantum Error Correction (QEC) & Stabilizer Codes.
//!
//! Implements stabilizer formalisms, syndrome extraction, and error recovery:
//! - 3-qubit Bit-Flip Code $[[3, 1, 3]]$ protecting against Pauli $X$ errors.
//! - 3-qubit Phase-Flip Code $[[3, 1, 3]]$ protecting against Pauli $Z$ errors.
//! - 7-qubit Steane Code $[[7, 1, 3]]$ CSS code protecting against arbitrary 1-qubit errors ($X, Y, Z$).
//! - Exact non-demolition syndrome extraction via Pauli tensor product expectations.
//! - Real-time maximum likelihood decoding and recovery.

use super::circuit::QuantumCircuit;
use crate::error::{Error, Result};

/// Pauli matrix basis operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// A multi-qubit Pauli string stabilizer operator $S = P_0 \otimes P_1 \otimes \dots \otimes P_{N-1}$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stabilizer {
    pub ops: Vec<Pauli>,
}

impl Stabilizer {
    /// Creates a stabilizer from a slice of Pauli operators.
    pub fn new(ops: Vec<Pauli>) -> Self {
        Self { ops }
    }

    /// Evaluates the expectation value $\langle \psi | S | \psi \rangle \in [-1.0, 1.0]$.
    pub fn expectation(&self, circuit: &QuantumCircuit) -> Result<f64> {
        let n = circuit.num_qubits;
        if self.ops.len() != n {
            return Err(Error::Msg(format!(
                "Stabilizer length {} does not match circuit qubits {}",
                self.ops.len(),
                n
            )));
        }

        // The circuit fields are public: re-validate layout (a tampered
        // qubit count sails through the ops-length check but indexes OOB).
        let dim = circuit.check_layout()?;
        let mut exp_val = 0.0f64;

        for (i, (&r_i, &im_i)) in circuit
            .state_real
            .iter()
            .zip(circuit.state_imag.iter())
            .enumerate()
            .take(dim)
        {
            // Apply Pauli string to computational basis state |i>
            let mut target_state = i;
            let mut phase_real = 1.0f64;
            let mut phase_imag = 0.0f64;

            for (qubit, &op) in self.ops.iter().enumerate() {
                let bit = (i >> qubit) & 1;
                match op {
                    Pauli::I => {}
                    Pauli::X => {
                        target_state ^= 1 << qubit;
                    }
                    Pauli::Z => {
                        if bit == 1 {
                            phase_real = -phase_real;
                            phase_imag = -phase_imag;
                        }
                    }
                    Pauli::Y => {
                        target_state ^= 1 << qubit;
                        // Y |0> = i |1>, Y |1> = -i |0>
                        let factor_imag = if bit == 0 { 1.0 } else { -1.0 };
                        let new_r = -phase_imag * factor_imag;
                        let new_im = phase_real * factor_imag;
                        phase_real = new_r;
                        phase_imag = new_im;
                    }
                }
            }

            // <i| S |i> contribution: psi^*(target_state) * phase * psi(i)
            let r_t = circuit.state_real[target_state];
            let im_t = circuit.state_imag[target_state];

            // (r_t - i im_t) * (phase_real + i phase_imag) * (r_i + i im_i)
            let pr = r_t * phase_real + im_t * phase_imag;
            let pi = -im_t * phase_real + r_t * phase_imag;

            let contrib = pr * r_i - pi * im_i;
            exp_val += contrib;
        }

        Ok(exp_val)
    }

    /// Measures binary syndrome eigenvalue: $+1$ if in codespace, $-1$ if error detected.
    pub fn measure_syndrome(&self, circuit: &QuantumCircuit) -> Result<i8> {
        let exp = self.expectation(circuit)?;
        if exp >= 0.0 { Ok(1) } else { Ok(-1) }
    }
}

/// Quantum Error Correcting Code specification $[[n, k, d]]$.
#[derive(Debug, Clone)]
pub struct StabilizerCode {
    pub name: String,
    pub num_physical_qubits: usize,
    pub num_logical_qubits: usize,
    pub distance: usize,
    pub generators: Vec<Stabilizer>,
}

impl StabilizerCode {
    /// 3-qubit Bit-Flip code $[[3, 1, 3]]$ with generators $S_1 = Z_0 Z_1 I_2$ and $S_2 = I_0 Z_1 Z_2$.
    pub fn bit_flip_3_qubit() -> Self {
        let s1 = Stabilizer::new(vec![Pauli::Z, Pauli::Z, Pauli::I]);
        let s2 = Stabilizer::new(vec![Pauli::I, Pauli::Z, Pauli::Z]);
        Self {
            name: "3-Qubit Bit-Flip [[3, 1, 3]]".into(),
            num_physical_qubits: 3,
            num_logical_qubits: 1,
            distance: 3,
            generators: vec![s1, s2],
        }
    }

    /// 3-qubit Phase-Flip code $[[3, 1, 3]]$ with generators $S_1 = X_0 X_1 I_2$ and $S_2 = I_0 X_1 X_2$.
    pub fn phase_flip_3_qubit() -> Self {
        let s1 = Stabilizer::new(vec![Pauli::X, Pauli::X, Pauli::I]);
        let s2 = Stabilizer::new(vec![Pauli::I, Pauli::X, Pauli::X]);
        Self {
            name: "3-Qubit Phase-Flip [[3, 1, 3]]".into(),
            num_physical_qubits: 3,
            num_logical_qubits: 1,
            distance: 3,
            generators: vec![s1, s2],
        }
    }

    /// 7-qubit Steane CSS code $[[7, 1, 3]]$ protecting against arbitrary single-qubit errors.
    pub fn steane_7_qubit() -> Self {
        // X-checks
        let x1 = Stabilizer::new(vec![
            Pauli::X,
            Pauli::I,
            Pauli::X,
            Pauli::I,
            Pauli::X,
            Pauli::I,
            Pauli::X,
        ]);
        let x2 = Stabilizer::new(vec![
            Pauli::I,
            Pauli::X,
            Pauli::X,
            Pauli::I,
            Pauli::I,
            Pauli::X,
            Pauli::X,
        ]);
        let x3 = Stabilizer::new(vec![
            Pauli::I,
            Pauli::I,
            Pauli::I,
            Pauli::X,
            Pauli::X,
            Pauli::X,
            Pauli::X,
        ]);
        // Z-checks
        let z1 = Stabilizer::new(vec![
            Pauli::Z,
            Pauli::I,
            Pauli::Z,
            Pauli::I,
            Pauli::Z,
            Pauli::I,
            Pauli::Z,
        ]);
        let z2 = Stabilizer::new(vec![
            Pauli::I,
            Pauli::Z,
            Pauli::Z,
            Pauli::I,
            Pauli::I,
            Pauli::Z,
            Pauli::Z,
        ]);
        let z3 = Stabilizer::new(vec![
            Pauli::I,
            Pauli::I,
            Pauli::I,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
            Pauli::Z,
        ]);

        Self {
            name: "Steane CSS [[7, 1, 3]]".into(),
            num_physical_qubits: 7,
            num_logical_qubits: 1,
            distance: 3,
            generators: vec![x1, x2, x3, z1, z2, z3],
        }
    }
}

/// Quantum Error Correction runtime engine.
pub struct QecEngine;

impl QecEngine {
    /// Encodes a state on `q0` into the 3-qubit bit-flip code across `(q0, q1, q2)`:
    /// $$(\alpha |0\rangle + \beta |1\rangle) \otimes |00\rangle \to \alpha |000\rangle + \beta |111\rangle$$
    pub fn encode_3qubit_bit_flip(
        circuit: &mut QuantumCircuit,
        q0: usize,
        q1: usize,
        q2: usize,
    ) -> Result<()> {
        circuit.cnot(q0, q1)?;
        circuit.cnot(q0, q2)?;
        Ok(())
    }

    /// Encodes a state on `q0` into the 3-qubit phase-flip code across `(q0, q1, q2)`:
    /// Encodes via CNOTs followed by Hadamards to map to Hadamard basis $\{|+\rangle, |-\rangle\}$.
    pub fn encode_3qubit_phase_flip(
        circuit: &mut QuantumCircuit,
        q0: usize,
        q1: usize,
        q2: usize,
    ) -> Result<()> {
        circuit.cnot(q0, q1)?;
        circuit.cnot(q0, q2)?;
        circuit.h(q0)?;
        circuit.h(q1)?;
        circuit.h(q2)?;
        Ok(())
    }

    /// Measures the syndrome vector $\mathbf{s} \in \{+1, -1\}^M$ for all stabilizer generators.
    pub fn measure_syndromes(circuit: &QuantumCircuit, code: &StabilizerCode) -> Result<Vec<i8>> {
        let mut syndromes = Vec::with_capacity(code.generators.len());
        for generator in &code.generators {
            syndromes.push(generator.measure_syndrome(circuit)?);
        }
        Ok(syndromes)
    }

    /// Decodes the 3-qubit bit-flip syndrome vector $(s_1, s_2)$ to locate corrupted qubit:
    /// - `[+1, +1]`: None (no error)
    /// - `[-1, +1]`: Qubit 0 flipped
    /// - `[-1, -1]`: Qubit 1 flipped
    /// - `[+1, -1]`: Qubit 2 flipped
    pub fn decode_3qubit_bit_flip_syndrome(syndrome: &[i8]) -> Option<usize> {
        if syndrome.len() < 2 {
            return None;
        }
        match (syndrome[0], syndrome[1]) {
            (-1, 1) => Some(0),
            (-1, -1) => Some(1),
            (1, -1) => Some(2),
            _ => None,
        }
    }

    /// Automatically measures syndromes and recovers the state from a bit-flip error.
    pub fn correct_bit_flip(
        circuit: &mut QuantumCircuit,
        code: &StabilizerCode,
    ) -> Result<Option<usize>> {
        let synd = Self::measure_syndromes(circuit, code)?;
        let corrupt_qubit = Self::decode_3qubit_bit_flip_syndrome(&synd);
        if let Some(q) = corrupt_qubit {
            // Invert the bit-flip with Pauli-X
            circuit.x(q)?;
        }
        Ok(corrupt_qubit)
    }
}
