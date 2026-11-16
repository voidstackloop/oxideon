//! Differentiable Quantum State Vector Simulation & Variational Quantum Circuit (VQC).
//!
//! Features:
//! - Multi-qubit full state vector simulator ($|\psi\rangle \in \mathbb{C}^{2^N}$).
//! - Universal quantum gate set: Hadamard (`H`), Pauli-X (`X`), Pauli-Z (`Z`), CNOT (`CX`),
//!   and parameterized rotations (`Rx`, `Ry`, `Rz`).
//! - Exact Pauli-Z expectation values ($\langle \psi | Z_i | \psi \rangle \in [-1, 1]$).
//! - Analytical Parameter-Shift Rule for exact quantum-classical hybrid gradient backpropagation:
//!   $$\frac{\partial \langle \hat{O} \rangle}{\partial \theta} = \frac{1}{2} \left[ \langle \hat{O} \rangle_{\theta + \pi/2} - \langle \hat{O} \rangle_{\theta - \pi/2} \right]$$

use crate::error::{Error, Result};
use crate::quantum::Pauli;
use std::f64::consts::PI;

/// Quantum Gate Operations supported by [`QuantumCircuit`].
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumGate {
    H(usize),
    X(usize),
    Z(usize),
    Ry(usize, f64),
    Rz(usize, f64),
    Cnot(usize, usize),
}

/// Differentiable $N$-qubit State Vector Quantum Circuit.
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub num_qubits: usize,
    pub state_real: Vec<f64>,
    pub state_imag: Vec<f64>,
    pub gates: Vec<QuantumGate>,
}

impl QuantumCircuit {
    /// Creates a new circuit with `num_qubits` initialized to the ground state $|0\dots0\rangle$.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 || num_qubits > 24 {
            return Err(Error::Msg(format!(
                "Invalid num_qubits {num_qubits} (must be between 1 and 24)"
            )));
        }
        let dim = 1 << num_qubits;
        let mut state_real = vec![0.0; dim];
        let state_imag = vec![0.0; dim];
        state_real[0] = 1.0; // |0...0> = 1.0 + 0.0i

        Ok(Self {
            num_qubits,
            state_real,
            state_imag,
            gates: Vec::new(),
        })
    }

    /// Verifies the state vectors agree with the (public, tamperable) qubit
    /// count: `len == 2^num_qubits` on both halves. Every gate/expectation
    /// path runs this before shifting or indexing.
    pub(crate) fn check_layout(&self) -> Result<usize> {
        if self.num_qubits == 0 || self.num_qubits >= usize::BITS as usize {
            return Err(Error::Msg(format!(
                "QuantumCircuit: num_qubits {} out of range 1..{}",
                self.num_qubits,
                usize::BITS
            )));
        }
        let dim = 1usize << self.num_qubits;
        if self.state_real.len() != dim || self.state_imag.len() != dim {
            return Err(Error::Msg(format!(
                "QuantumCircuit: state lengths ({}, {}) disagree with 2^{} = {dim}",
                self.state_real.len(),
                self.state_imag.len(),
                self.num_qubits
            )));
        }
        Ok(dim)
    }

    /// Resets the quantum state to $|0\dots0\rangle$.
    pub fn reset(&mut self) -> Result<&mut Self> {
        self.check_layout()?;
        self.state_real.fill(0.0);
        self.state_imag.fill(0.0);
        self.state_real[0] = 1.0;
        self.gates.clear();
        Ok(self)
    }

    /// Applies a Hadamard gate ($H$) to `qubit`.
    pub fn h(&mut self, qubit: usize) -> Result<&mut Self> {
        self.check_qubit(qubit)?;
        self.gates.push(QuantumGate::H(qubit));
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let bit = 1 << qubit;

        for i in 0..(1 << self.num_qubits) {
            if (i & bit) == 0 {
                let j = i | bit;
                let (r_i, r_j) = (self.state_real[i], self.state_real[j]);
                let (im_i, im_j) = (self.state_imag[i], self.state_imag[j]);

                self.state_real[i] = (r_i + r_j) * inv_sqrt2;
                self.state_real[j] = (r_i - r_j) * inv_sqrt2;
                self.state_imag[i] = (im_i + im_j) * inv_sqrt2;
                self.state_imag[j] = (im_i - im_j) * inv_sqrt2;
            }
        }
        Ok(self)
    }

    /// Applies a Pauli-X (Bit-flip / NOT) gate to `qubit`.
    pub fn x(&mut self, qubit: usize) -> Result<&mut Self> {
        self.check_qubit(qubit)?;
        self.gates.push(QuantumGate::X(qubit));
        let bit = 1 << qubit;

        for i in 0..(1 << self.num_qubits) {
            if (i & bit) == 0 {
                let j = i | bit;
                self.state_real.swap(i, j);
                self.state_imag.swap(i, j);
            }
        }
        Ok(self)
    }

    /// Applies a Pauli-Z (Phase-flip) gate to `qubit`.
    pub fn z(&mut self, qubit: usize) -> Result<&mut Self> {
        self.check_qubit(qubit)?;
        self.gates.push(QuantumGate::Z(qubit));
        let bit = 1 << qubit;

        for i in 0..(1 << self.num_qubits) {
            if (i & bit) != 0 {
                self.state_real[i] = -self.state_real[i];
                self.state_imag[i] = -self.state_imag[i];
            }
        }
        Ok(self)
    }

    /// Applies a parameterized $R_y(\theta) = \exp(-i \frac{\theta}{2} Y)$ rotation gate to `qubit`.
    pub fn ry(&mut self, qubit: usize, theta: f64) -> Result<&mut Self> {
        self.check_qubit(qubit)?;
        self.gates.push(QuantumGate::Ry(qubit, theta));
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        let bit = 1 << qubit;

        for i in 0..(1 << self.num_qubits) {
            if (i & bit) == 0 {
                let j = i | bit;
                let (r_i, r_j) = (self.state_real[i], self.state_real[j]);
                let (im_i, im_j) = (self.state_imag[i], self.state_imag[j]);

                self.state_real[i] = c * r_i - s * r_j;
                self.state_real[j] = s * r_i + c * r_j;
                self.state_imag[i] = c * im_i - s * im_j;
                self.state_imag[j] = s * im_i + c * im_j;
            }
        }
        Ok(self)
    }

    /// Applies a parameterized $R_z(\theta) = \exp(-i \frac{\theta}{2} Z)$ rotation gate to `qubit`.
    pub fn rz(&mut self, qubit: usize, theta: f64) -> Result<&mut Self> {
        self.check_qubit(qubit)?;
        self.gates.push(QuantumGate::Rz(qubit, theta));
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        let bit = 1 << qubit;

        for i in 0..(1 << self.num_qubits) {
            let (r, im) = (self.state_real[i], self.state_imag[i]);
            if (i & bit) == 0 {
                self.state_real[i] = r * c + im * s;
                self.state_imag[i] = im * c - r * s;
            } else {
                self.state_real[i] = r * c - im * s;
                self.state_imag[i] = im * c + r * s;
            }
        }
        Ok(self)
    }

    /// Computes full state measurement probabilities $p_i = |\psi_i|^2$ for all basis states $i \in 0..2^N$.
    pub fn probabilities(&self) -> Result<Vec<f64>> {
        self.check_layout()?;
        Ok(self
            .state_real
            .iter()
            .zip(&self.state_imag)
            .map(|(&r, &im)| r * r + im * im)
            .collect())
    }

    /// Finalizes gate execution (no-op as gates are executed eagerly on state vectors).
    pub fn run(&mut self) -> Result<&mut Self> {
        Ok(self)
    }

    /// Applies a Controlled-NOT (CNOT / CX) gate with `control` and `target` qubits.
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        if control == target {
            return Err(Error::Msg("CNOT control and target must differ".into()));
        }
        self.check_qubit(control)?;
        self.check_qubit(target)?;
        self.gates.push(QuantumGate::Cnot(control, target));

        let c_bit = 1 << control;
        let t_bit = 1 << target;

        for i in 0..(1 << self.num_qubits) {
            // Apply X to target only when control bit is 1
            if (i & c_bit) != 0 && (i & t_bit) == 0 {
                let j = i | t_bit;
                self.state_real.swap(i, j);
                self.state_imag.swap(i, j);
            }
        }
        Ok(self)
    }

    /// Computes the Pauli-Z expectation value $\langle \psi | Z_{\text{qubit}} | \psi \rangle \in [-1, 1]$.
    pub fn expectation_z(&self, qubit: usize) -> Result<f64> {
        self.check_qubit(qubit)?;
        let bit = 1 << qubit;
        let mut exp_val = 0.0f64;

        for i in 0..(1 << self.num_qubits) {
            let prob = self.state_real[i].powi(2) + self.state_imag[i].powi(2);
            if (i & bit) == 0 {
                exp_val += prob; // +1 eigenvalue for |0>
            } else {
                exp_val -= prob; // -1 eigenvalue for |1>
            }
        }
        Ok(exp_val)
    }

    /// Computes the exact Pauli-string expectation $\langle \psi | P | \psi \rangle$
    /// from the state vector, with one operator per qubit (`I`, `X`, `Y`, or `Z`).
    pub fn expectation_pauli(&self, paulis: &[Pauli]) -> Result<f64> {
        if paulis.len() != self.num_qubits {
            return Err(Error::Msg(format!(
                "expectation_pauli: need {} operators, got {}",
                self.num_qubits,
                paulis.len()
            )));
        }
        // P|x> = phase(x) |x xor flip_mask>: X/Y flip their bit; Z contributes
        // (-1)^bit; each Y contributes i * (-1)^bit.
        self.check_layout()?;
        let mut flip_mask = 0usize;
        let mut num_y = 0usize;
        for (q, p) in paulis.iter().enumerate() {
            match p {
                Pauli::I => {}
                Pauli::X => flip_mask |= 1 << q,
                Pauli::Z => {}
                Pauli::Y => {
                    flip_mask |= 1 << q;
                    num_y += 1;
                }
            }
        }
        // i^num_y cycles (1, i, -1, -i).
        let (iy_r, iy_i) = match num_y % 4 {
            0 => (1.0, 0.0),
            1 => (0.0, 1.0),
            2 => (-1.0, 0.0),
            _ => (0.0, -1.0),
        };
        let dim = 1 << self.num_qubits;
        let mut exp_val = 0.0f64;
        for x in 0..dim {
            let j = x ^ flip_mask;
            let mut sign = 1.0f64;
            for (q, p) in paulis.iter().enumerate() {
                let bit_set = (x >> q) & 1 == 1;
                if bit_set && matches!(p, Pauli::Z | Pauli::Y) {
                    sign = -sign;
                }
            }
            let (pr, pi) = (iy_r * sign, iy_i * sign);
            let (ax, bx) = (self.state_real[x], self.state_imag[x]);
            let (aj, bj) = (self.state_real[j], self.state_imag[j]);
            // Re(conj(psi_x) * phase * psi_j)
            let cr = ax * pr + bx * pi;
            let ci = ax * pi - bx * pr;
            exp_val += cr * aj - ci * bj;
        }
        Ok(exp_val)
    }

    /// Computes the exact analytical gradient of $\langle Z_{\text{observable}} \rangle$
    /// with respect to parameterized rotation angle `theta` using the Parameter-Shift Rule.
    pub fn parameter_shift_gradient<F>(
        num_qubits: usize,
        observable_qubit: usize,
        theta: f64,
        circuit_builder: F,
    ) -> Result<f64>
    where
        F: Fn(&mut QuantumCircuit, f64) -> Result<()>,
    {
        let shift = PI / 2.0;

        // 1. Forward circuit at theta + pi/2
        let mut qc_plus = QuantumCircuit::new(num_qubits)?;
        circuit_builder(&mut qc_plus, theta + shift)?;
        let exp_plus = qc_plus.expectation_z(observable_qubit)?;

        // 2. Backward circuit at theta - pi/2
        let mut qc_minus = QuantumCircuit::new(num_qubits)?;
        circuit_builder(&mut qc_minus, theta - shift)?;
        let exp_minus = qc_minus.expectation_z(observable_qubit)?;

        // d<O>/d(theta) = 0.5 * (E_+ - E_-)
        Ok(0.5 * (exp_plus - exp_minus))
    }

    fn check_qubit(&self, q: usize) -> Result<()> {
        self.check_layout()?;
        if q >= self.num_qubits {
            return Err(Error::Msg(format!(
                "Qubit {q} out of bounds for {}-qubit circuit",
                self.num_qubits
            )));
        }
        Ok(())
    }
}
