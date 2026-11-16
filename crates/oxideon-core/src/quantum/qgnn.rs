//! Quantum Graph Neural Networks (QGNN) & Parameterized Equivariant Quantum Hamiltonians.
//!
//! Maps graph-structured relational data onto quantum registers:
//! - Vertices $V$ map to $N$ qubits ($|V| = N$).
//! - Node attributes parameterize local rotational encodings $R_y(\theta_u), R_z(\phi_u)$.
//! - Graph edges $(u, v) \in E$ parameterize non-local entangling gates $ZZ(J_{uv}) = \exp\left(-i \frac{J_{uv}}{2} Z_u \otimes Z_v\right)$.
//! - Multi-layer variational ansatz alternating graph-topology entanglers and variational rotations.
//! - Exact multi-qubit Pauli-Z node readouts $\langle Z_u \rangle \in [-1, 1]$ and graph Hamiltonian energies $\langle H_G \rangle$.
//! - Analytical Parameter-Shift Rule for exact quantum gradients:
//!   $$\frac{\partial \langle H_G \rangle}{\partial \theta} = \frac{1}{2} \left[ \langle H_G \rangle_{\theta + \pi/2} - \langle H_G \rangle_{\theta - \pi/2} \right]$$
//!
//! References:
//! - Verdon et al. (2019): "Quantum Graph Neural Networks"
//! - Zheng et al. (2021): "Equivariant Quantum Graph Neural Networks"

use crate::error::{Error, Result};
use crate::quantum::circuit::QuantumCircuit;
use std::f64::consts::PI;

/// Analytical gradients of a QGNN with respect to variational parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct QGNNGradients {
    /// Gradients with respect to $R_y$ rotation angles $\nabla_\theta \langle H \rangle$.
    pub grad_theta: Vec<f64>,
    /// Gradients with respect to $R_z$ rotation angles $\nabla_\phi \langle H \rangle$.
    pub grad_phi: Vec<f64>,
    /// Gradients with respect to edge coupling weights $\nabla_J \langle H \rangle$.
    pub grad_j: Vec<f64>,
}

/// Quantum Graph Neural Network (QGNN).
#[derive(Debug, Clone)]
pub struct QuantumGraphNeuralNetwork {
    /// Number of nodes/qubits $N$.
    pub num_qubits: usize,
    /// Directed or undirected graph edges $(u, v)$.
    pub edges: Vec<(usize, usize)>,
    /// Number of variational ansatz layers $L$.
    pub num_layers: usize,
    /// Variational $R_y$ angles: shape $[L, N]$.
    pub theta: Vec<f64>,
    /// Variational $R_z$ angles: shape $[L, N]$.
    pub phi: Vec<f64>,
    /// Variational $ZZ$ edge coupling parameters: shape $[L, |E|]$.
    pub j_couplings: Vec<f64>,
}

impl QuantumGraphNeuralNetwork {
    /// Creates a new QGNN for a graph with $N$ vertices and given edges.
    pub fn new(num_qubits: usize, edges: Vec<(usize, usize)>, num_layers: usize) -> Result<Self> {
        if num_qubits == 0 || num_qubits > 20 {
            return Err(Error::Msg(format!(
                "Invalid num_qubits {num_qubits} (must be 1..20)"
            )));
        }
        if num_layers == 0 {
            return Err(Error::Msg("num_layers must be >= 1".to_string()));
        }
        // Each layer costs a full entangling + rotation pass; an uncapped
        // count overflows the parameter products or aborts in allocation.
        let num_layers = crate::limits::check_repeats("num_layers", num_layers)?;

        for &(u, v) in &edges {
            if u >= num_qubits || v >= num_qubits || u == v {
                return Err(Error::Msg(format!(
                    "Invalid edge ({u}, {v}) for {num_qubits} qubits"
                )));
            }
        }

        let num_edges = edges.len();
        let total_node_params = num_layers.checked_mul(num_qubits).ok_or_else(|| {
            Error::Msg(format!(
                "QGNN node parameter count {num_layers} * {num_qubits} overflows"
            ))
        })?;
        let total_edge_params = num_layers.checked_mul(num_edges).ok_or_else(|| {
            Error::Msg(format!(
                "QGNN edge parameter count {num_layers} * {num_edges} overflows"
            ))
        })?;

        // Deterministic initial parameters
        let theta = (0..total_node_params)
            .map(|i| ((i * 17 + 5) % 100) as f64 / 100.0 * PI)
            .collect();
        let phi = (0..total_node_params)
            .map(|i| ((i * 23 + 11) % 100) as f64 / 100.0 * PI)
            .collect();
        let j_couplings = (0..total_edge_params)
            .map(|i| ((i * 13 + 7) % 100) as f64 / 100.0 * PI)
            .collect();

        Ok(Self {
            num_qubits,
            edges,
            num_layers,
            theta,
            phi,
            j_couplings,
        })
    }

    /// Sets the variational parameters of the network.
    pub fn set_parameters(
        &mut self,
        theta: Vec<f64>,
        phi: Vec<f64>,
        j_couplings: Vec<f64>,
    ) -> Result<()> {
        let expected_rot = self
            .num_layers
            .checked_mul(self.num_qubits)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "QGNN node parameter count {} * {} overflows",
                    self.num_layers, self.num_qubits
                ))
            })?;
        let expected_j = self
            .num_layers
            .checked_mul(self.edges.len())
            .ok_or_else(|| {
                Error::Msg(format!(
                    "QGNN edge parameter count {} * {} overflows",
                    self.num_layers,
                    self.edges.len()
                ))
            })?;
        if theta.len() != expected_rot {
            return Err(Error::Msg(format!(
                "theta length {} does not match expected {}",
                theta.len(),
                expected_rot
            )));
        }
        if phi.len() != expected_rot {
            return Err(Error::Msg(format!(
                "phi length {} does not match expected {}",
                phi.len(),
                expected_rot
            )));
        }
        if j_couplings.len() != expected_j {
            return Err(Error::Msg(format!(
                "j_couplings length {} does not match expected {}",
                j_couplings.len(),
                expected_j
            )));
        }
        self.theta = theta;
        self.phi = phi;
        self.j_couplings = j_couplings;
        Ok(())
    }

    /// Builds and executes the quantum circuit given node feature inputs and parameter overrides.
    fn build_circuit(
        &self,
        node_features: &[f64],
        theta: &[f64],
        phi: &[f64],
        j_couplings: &[f64],
    ) -> Result<QuantumCircuit> {
        let n = self.num_qubits;
        let num_edges = self.edges.len();
        if node_features.len() != n {
            return Err(Error::ShapeMismatch {
                op: "QGNN::build_circuit",
                expected: vec![n],
                got: vec![node_features.len()],
            });
        }
        // Pub fields can be tampered post-construction: re-check endpoint
        // range (the shift sites below) and parameter-length agreement
        // (the indexing sites below) before touching either.
        for &(u, v) in &self.edges {
            if u >= n || v >= n || u == v {
                return Err(Error::Msg(format!(
                    "QGNN invalid edge ({u}, {v}) for {n} qubits"
                )));
            }
        }
        let expected_rot = self.num_layers.checked_mul(n).ok_or_else(|| {
            Error::Msg(format!(
                "QGNN node parameter count {} * {n} overflows",
                self.num_layers
            ))
        })?;
        let expected_j = self.num_layers.checked_mul(num_edges).ok_or_else(|| {
            Error::Msg(format!(
                "QGNN edge parameter count {} * {num_edges} overflows",
                self.num_layers
            ))
        })?;
        if theta.len() != expected_rot
            || phi.len() != expected_rot
            || j_couplings.len() != expected_j
        {
            return Err(Error::Msg(format!(
                "QGNN parameter lengths ({}, {}, {}) disagree with [{expected_rot}, {expected_rot}, {expected_j}]",
                theta.len(),
                phi.len(),
                j_couplings.len()
            )));
        }

        let mut qc = QuantumCircuit::new(n)?;

        // 1. Initial State Preparation: Hadamard on all qubits followed by feature encoding
        for (u, &feat) in node_features.iter().enumerate().take(n) {
            qc.h(u)?;
            qc.ry(u, feat)?;
        }

        // 2. Multi-layer Variational Layers
        for l in 0..self.num_layers {
            // (a) Graph-topology entangling layer: ZZ(J_uv) on all graph edges
            for (e_idx, &(u, v)) in self.edges.iter().enumerate() {
                let j_val = j_couplings[l * num_edges + e_idx];
                // ZZ(theta) = CNOT(u, v) * Rz(v, theta) * CNOT(u, v)
                qc.cnot(u, v)?;
                qc.rz(v, j_val)?;
                qc.cnot(u, v)?;
            }

            // (b) Single-qubit variational rotations
            for u in 0..n {
                let th = theta[l * n + u];
                let ph = phi[l * n + u];
                qc.ry(u, th)?;
                qc.rz(u, ph)?;
            }
        }

        Ok(qc)
    }

    /// Evaluates node readouts $\langle Z_u \rangle \in [-1, 1]$ for all vertices $u \in V$.
    pub fn forward(&self, node_features: &[f64]) -> Result<Vec<f64>> {
        let qc = self.build_circuit(node_features, &self.theta, &self.phi, &self.j_couplings)?;
        let mut readouts = Vec::with_capacity(self.num_qubits);
        for u in 0..self.num_qubits {
            readouts.push(qc.expectation_z(u)?);
        }
        Ok(readouts)
    }

    /// Evaluates total graph Hamiltonian energy:
    ///
    /// $$\langle H_G \rangle = \frac{1}{|V|} \sum_{u \in V} \langle Z_u \rangle + \frac{1}{|E|} \sum_{(u, v) \in E} \langle Z_u Z_v \rangle$$
    pub fn forward_graph_energy(&self, node_features: &[f64]) -> Result<f64> {
        let qc = self.build_circuit(node_features, &self.theta, &self.phi, &self.j_couplings)?;
        let n = self.num_qubits;
        let mut node_sum = 0.0;
        for u in 0..n {
            node_sum += qc.expectation_z(u)?;
        }

        // Two-qubit ZZ expectations: <Z_u Z_v> = sum_i p_i * (-1)^(bit_u + bit_v)
        let probs = qc.probabilities()?;
        let mut edge_sum = 0.0;
        for &(u, v) in &self.edges {
            let bit_u = 1 << u;
            let bit_v = 1 << v;
            let mut zz_exp = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                let sign_u = if (i & bit_u) == 0 { 1.0 } else { -1.0 };
                let sign_v = if (i & bit_v) == 0 { 1.0 } else { -1.0 };
                zz_exp += p * sign_u * sign_v;
            }
            edge_sum += zz_exp;
        }

        let node_contrib = node_sum / (n as f64);
        let edge_contrib = if self.edges.is_empty() {
            0.0
        } else {
            edge_sum / (self.edges.len() as f64)
        };

        Ok(node_contrib + edge_contrib)
    }

    /// Computes exact analytical parameter-shift gradients $\nabla \langle H_G \rangle$
    /// with respect to all trainable parameters ($\theta, \phi, J$).
    pub fn parameter_shift_gradients(&self, node_features: &[f64]) -> Result<QGNNGradients> {
        let shift = PI / 2.0;

        // Gradients for theta
        let mut grad_theta = Vec::with_capacity(self.theta.len());
        let mut theta_mut = self.theta.clone();
        for i in 0..self.theta.len() {
            let orig = theta_mut[i];

            theta_mut[i] = orig + shift;
            let qc_plus =
                self.build_circuit(node_features, &theta_mut, &self.phi, &self.j_couplings)?;
            let e_plus = self.compute_energy_from_circuit(&qc_plus)?;

            theta_mut[i] = orig - shift;
            let qc_minus =
                self.build_circuit(node_features, &theta_mut, &self.phi, &self.j_couplings)?;
            let e_minus = self.compute_energy_from_circuit(&qc_minus)?;

            theta_mut[i] = orig;
            grad_theta.push(0.5 * (e_plus - e_minus));
        }

        // Gradients for phi
        let mut grad_phi = Vec::with_capacity(self.phi.len());
        let mut phi_mut = self.phi.clone();
        for i in 0..self.phi.len() {
            let orig = phi_mut[i];

            phi_mut[i] = orig + shift;
            let qc_plus =
                self.build_circuit(node_features, &self.theta, &phi_mut, &self.j_couplings)?;
            let e_plus = self.compute_energy_from_circuit(&qc_plus)?;

            phi_mut[i] = orig - shift;
            let qc_minus =
                self.build_circuit(node_features, &self.theta, &phi_mut, &self.j_couplings)?;
            let e_minus = self.compute_energy_from_circuit(&qc_minus)?;

            phi_mut[i] = orig;
            grad_phi.push(0.5 * (e_plus - e_minus));
        }

        // Gradients for J couplings
        let mut grad_j = Vec::with_capacity(self.j_couplings.len());
        let mut j_mut = self.j_couplings.clone();
        for i in 0..self.j_couplings.len() {
            let orig = j_mut[i];

            j_mut[i] = orig + shift;
            let qc_plus = self.build_circuit(node_features, &self.theta, &self.phi, &j_mut)?;
            let e_plus = self.compute_energy_from_circuit(&qc_plus)?;

            j_mut[i] = orig - shift;
            let qc_minus = self.build_circuit(node_features, &self.theta, &self.phi, &j_mut)?;
            let e_minus = self.compute_energy_from_circuit(&qc_minus)?;

            j_mut[i] = orig;
            grad_j.push(0.5 * (e_plus - e_minus));
        }

        Ok(QGNNGradients {
            grad_theta,
            grad_phi,
            grad_j,
        })
    }

    fn compute_energy_from_circuit(&self, qc: &QuantumCircuit) -> Result<f64> {
        let n = self.num_qubits;
        for &(u, v) in &self.edges {
            if u >= n || v >= n || u == v {
                return Err(Error::Msg(format!(
                    "QGNN invalid edge ({u}, {v}) for {n} qubits"
                )));
            }
        }
        let mut node_sum = 0.0;
        for u in 0..n {
            node_sum += qc.expectation_z(u)?;
        }

        let probs = qc.probabilities()?;
        let mut edge_sum = 0.0;
        for &(u, v) in &self.edges {
            let bit_u = 1 << u;
            let bit_v = 1 << v;
            let mut zz_exp = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                let sign_u = if (i & bit_u) == 0 { 1.0 } else { -1.0 };
                let sign_v = if (i & bit_v) == 0 { 1.0 } else { -1.0 };
                zz_exp += p * sign_u * sign_v;
            }
            edge_sum += zz_exp;
        }

        let node_contrib = node_sum / (n as f64);
        let edge_contrib = if self.edges.is_empty() {
            0.0
        } else {
            edge_sum / (self.edges.len() as f64)
        };

        Ok(node_contrib + edge_contrib)
    }
}
