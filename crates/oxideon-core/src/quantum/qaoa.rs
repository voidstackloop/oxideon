//! Quantum Approximate Optimization Algorithm (QAOA) & Variational Quantum Eigensolver (VQE).
//!
//! Features:
//! - Multi-qubit QAOA variational state preparation for combinatorial optimization (MaxCut, Ising models):
//!   $$|\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = \prod_{l=1}^p \left( e^{-i \beta_l H_M} e^{-i \gamma_l H_C} \right) |+\rangle^{\otimes n}$$
//! - Analytical Parameter-Shift Rule for exact quantum-classical hybrid gradient descent:
//!   $$\frac{\partial \langle H_C \rangle}{\partial \theta} = \frac{1}{2} \left[ \langle H_C \rangle_{\theta + \pi/2} - \langle H_C \rangle_{\theta - \pi/2} \right]$$
//! - Variational Quantum Eigensolver ([`VqeAnsatz`]) for molecular ground state energy estimation.

use crate::error::{Error, Result};
use crate::quantum::{Pauli, QuantumCircuit};
use std::f64::consts::PI;

/// Quantum Approximate Optimization Algorithm (QAOA) Circuit.
#[derive(Debug, Clone)]
pub struct QaoaCircuit {
    pub num_qubits: usize,
    pub p_layers: usize,
    pub edges: Vec<(usize, usize, f64)>,
}

impl QaoaCircuit {
    /// Creates a new `QaoaCircuit`.
    pub fn new(
        num_qubits: usize,
        p_layers: usize,
        edges: Vec<(usize, usize, f64)>,
    ) -> Result<Self> {
        if num_qubits == 0 || num_qubits > 20 {
            return Err(Error::Msg(format!(
                "Invalid num_qubits {num_qubits} (1..20)"
            )));
        }
        if p_layers == 0 {
            return Err(Error::Msg("p_layers must be >= 1".into()));
        }
        Ok(Self {
            num_qubits,
            p_layers,
            edges,
        })
    }

    /// Builds and simulates the QAOA quantum circuit with parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$.
    pub fn build_circuit(&self, gamma: &[f64], beta: &[f64]) -> Result<QuantumCircuit> {
        if gamma.len() != self.p_layers || beta.len() != self.p_layers {
            return Err(Error::Msg(format!(
                "Expected {} parameters for gamma and beta, got {} and {}",
                self.p_layers,
                gamma.len(),
                beta.len()
            )));
        }

        // Pub edges can be tampered post-construction (new() takes them
        // unchecked): range-check before the shifts in evaluate_energy.
        for &(u, v, _) in &self.edges {
            if u >= self.num_qubits || v >= self.num_qubits || u == v {
                return Err(Error::Msg(format!(
                    "QAOA invalid edge ({u}, {v}) for {} qubits",
                    self.num_qubits
                )));
            }
        }

        let mut circuit = QuantumCircuit::new(self.num_qubits)?;

        // 1. Initial equal superposition state: |+>^{\otimes n}
        for q in 0..self.num_qubits {
            circuit.h(q)?;
        }

        // 2. Alternating p layers of Problem Unitary U(C, gamma) and Mixer Unitary U(B, beta)
        for l in 0..self.p_layers {
            let g = gamma[l];
            let b = beta[l];

            // Cost unitary: e^{-i \gamma C} for MaxCut
            for &(u, v, w) in &self.edges {
                circuit.cnot(u, v)?;
                circuit.rz(v, -2.0 * g * w)?;
                circuit.cnot(u, v)?;
            }

            // Mixer unitary: e^{-i \beta X_i} via H -> Rz(2 * beta) -> H
            for q in 0..self.num_qubits {
                circuit.h(q)?;
                circuit.rz(q, 2.0 * b)?;
                circuit.h(q)?;
            }
        }

        // Execute all gates
        circuit.run()?;
        Ok(circuit)
    }

    /// Evaluates the expected cost $\langle H_C \rangle = \sum_{z} |\psi(z)|^2 C(z)$.
    /// For MaxCut: $C(z) = \sum_{(u, v) \in E} w_{uv} \frac{1 - Z_u Z_v}{2}$.
    pub fn evaluate_energy(&self, gamma: &[f64], beta: &[f64]) -> Result<f64> {
        let circuit = self.build_circuit(gamma, beta)?;
        let probs = circuit.probabilities()?;

        let mut expected_cost = 0.0;
        let dim = 1 << self.num_qubits;

        for (bitstring, &prob) in probs.iter().enumerate().take(dim) {
            let mut cut_val = 0.0;
            for &(u, v, w) in &self.edges {
                let bit_u = (bitstring >> u) & 1;
                let bit_v = (bitstring >> v) & 1;
                if bit_u != bit_v {
                    cut_val += w;
                }
            }
            expected_cost += prob * cut_val;
        }

        Ok(expected_cost)
    }

    /// Computes analytical gradients with respect to $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$
    /// using the Parameter-Shift Rule:
    /// $$\frac{\partial \langle H_C \rangle}{\partial \theta_k} = \frac{1}{2} \left[ \langle H_C(\theta_k + \pi/2) \rangle - \langle H_C(\theta_k - \pi/2) \rangle \right]$$
    pub fn compute_gradients(&self, gamma: &[f64], beta: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        // Length agreement first: p_layers is public, and allocating before
        // checking panics on tampered counts.
        if gamma.len() != self.p_layers || beta.len() != self.p_layers {
            return Err(Error::Msg(format!(
                "Expected {} parameters for gamma and beta, got {} and {}",
                self.p_layers,
                gamma.len(),
                beta.len()
            )));
        }
        let mut grad_gamma = vec![0.0; self.p_layers];
        let mut grad_beta = vec![0.0; self.p_layers];
        let shift = PI / 4.0;

        // Gradients w.r.t gamma
        for l in 0..self.p_layers {
            let mut g_plus = gamma.to_vec();
            let mut g_minus = gamma.to_vec();
            g_plus[l] += shift;
            g_minus[l] -= shift;

            let e_plus = self.evaluate_energy(&g_plus, beta)?;
            let e_minus = self.evaluate_energy(&g_minus, beta)?;
            grad_gamma[l] = (e_plus - e_minus) / (2.0 * shift.sin());
        }

        // Gradients w.r.t beta
        for l in 0..self.p_layers {
            let mut b_plus = beta.to_vec();
            let mut b_minus = beta.to_vec();
            b_plus[l] += shift;
            b_minus[l] -= shift;

            let e_plus = self.evaluate_energy(gamma, &b_plus)?;
            let e_minus = self.evaluate_energy(gamma, &b_minus)?;
            grad_beta[l] = (e_plus - e_minus) / (2.0 * shift.sin());
        }

        Ok((grad_gamma, grad_beta))
    }

    /// Performs parameter optimization to maximize cut energy starting from `(init_gamma, init_beta)`.
    pub fn optimize_from(
        &self,
        init_gamma: &[f64],
        init_beta: &[f64],
        steps: usize,
        lr: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, f64)> {
        let steps = crate::limits::check_iterations("steps", steps)?;
        let mut gamma = init_gamma.to_vec();
        let mut beta = init_beta.to_vec();

        for _ in 0..steps {
            let (g_grad, b_grad) = self.compute_gradients(&gamma, &beta)?;
            for l in 0..self.p_layers {
                gamma[l] += lr * g_grad[l];
                beta[l] += lr * b_grad[l];
            }
        }

        let best_energy = self.evaluate_energy(&gamma, &beta)?;
        Ok((gamma, beta, best_energy))
    }

    /// Performs parameter optimization to maximize cut energy using gradient ascent from default parameters.
    pub fn optimize(&self, steps: usize, lr: f64) -> Result<(Vec<f64>, Vec<f64>, f64)> {
        let p = crate::limits::check_repeats("p_layers", self.p_layers)?;
        self.optimize_from(&vec![0.2; p], &vec![0.4; p], steps, lr)
    }
}

/// Variational Quantum Eigensolver (VQE) Hardware-Efficient Ansatz.
#[derive(Debug, Clone)]
pub struct VqeAnsatz {
    pub num_qubits: usize,
    pub num_layers: usize,
}

impl VqeAnsatz {
    pub fn new(num_qubits: usize, num_layers: usize) -> Result<Self> {
        if num_qubits == 0 || num_qubits > 20 {
            return Err(Error::Msg("Invalid num_qubits".into()));
        }
        let num_layers = crate::limits::check_repeats("num_layers", num_layers)?;
        Ok(Self {
            num_qubits,
            num_layers: num_layers.max(1),
        })
    }

    /// Number of variational parameters: $N \times (L + 1)$ (saturating:
    /// tampered counts saturate, and the length check in
    /// [`VqeAnsatz::evaluate_energy`] then rejects).
    pub fn num_parameters(&self) -> usize {
        self.num_qubits
            .saturating_mul(self.num_layers.saturating_add(1))
    }

    /// Evaluates the ground state energy expectation $\langle \psi(\boldsymbol{\theta}) | H | \psi(\boldsymbol{\theta}) \rangle$:
    pub fn evaluate_energy(
        &self,
        params: &[f64],
        pauli_terms: &[(Vec<Pauli>, f64)],
    ) -> Result<f64> {
        let expected_params = self.num_parameters();
        if params.len() != expected_params {
            return Err(Error::Msg(format!(
                "Expected {expected_params} parameters, got {}",
                params.len()
            )));
        }

        let mut circuit = QuantumCircuit::new(self.num_qubits)?;

        // Hardware-efficient layers: Ry rotations interleaved with linear CNOT entangling ladder
        let mut p_idx = 0;
        for q in 0..self.num_qubits {
            circuit.ry(q, params[p_idx])?;
            p_idx += 1;
        }

        for _ in 0..self.num_layers {
            // Entangling ladder
            for q in 0..(self.num_qubits - 1) {
                circuit.cnot(q, q + 1)?;
            }
            // Parametric rotations
            for q in 0..self.num_qubits {
                circuit.ry(q, params[p_idx])?;
                p_idx += 1;
            }
        }

        circuit.run()?;

        // Sum exact state-vector expectations over Pauli string terms
        // (factoring into single-qubit <Z> products is wrong for entangled
        // states, and X/Y terms must not be dropped).
        let mut total_energy = 0.0;
        for (string, coeff) in pauli_terms {
            let exp_val = circuit.expectation_pauli(string)?;
            total_energy += coeff * exp_val;
        }

        Ok(total_energy)
    }
}
