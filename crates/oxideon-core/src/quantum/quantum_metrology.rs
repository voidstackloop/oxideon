//! Quantum Metrology, Quantum Fisher Information (QFI), Quantum Cramér-Rao Bound & Heisenberg Limit (Braunstein & Caves, 1994).
//!
//! ### 1. Quantum Cramér-Rao Bound & Heisenberg Scaling $\Delta \theta \ge 1/N$
//! For an $N$-qubit probe state evolved under a phase generator $U(\theta) = e^{-i \theta J_z}$, the variance of any
//! unbiased phase estimator $\hat{\theta}$ is lower-bounded by the **Quantum Fisher Information (QFI)** $F_Q$:
//! $$\Delta \theta \ge \frac{1}{\sqrt{F_Q(\rho_\theta)}} = \frac{1}{2 \Delta J_z}$$
//! • **Standard Quantum Limit (SQL)** for separable product states $|+\rangle^{\otimes N}$: $F_Q = N \implies \Delta \theta = 1 / \sqrt{N}$
//! • **Heisenberg Limit (HL)** for maximally entangled GHZ / NOON states: $F_Q = N^2 \implies \Delta \theta = 1 / N$.

use crate::error::{Error, Result};

/// Quantum Metrology, QFI & Heisenberg Limit Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantumMetrologyEngine;

impl QuantumMetrologyEngine {
    /// Creates a new Quantum Metrology engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the Quantum Fisher Information $F_Q = N^2 p^2 + N (1 - p^2)$ and Cramér-Rao precision bound
    /// $\Delta \theta = 1 / \sqrt{F_Q}$ for an $N$-qubit GHZ state with visibility $p \in [0, 1]$.
    pub fn ghz_quantum_fisher_info(num_qubits_n: usize, visibility_p: f64) -> Result<(f64, f64)> {
        if num_qubits_n == 0 || !(0.0..=1.0).contains(&visibility_p) {
            return Err(Error::Msg(
                "num_qubits_n >= 1 and visibility_p in [0, 1] required".to_string(),
            ));
        }
        let n = num_qubits_n as f64;
        let p2 = visibility_p * visibility_p;
        let fq = n * n * p2 + n * (1.0 - p2);
        Ok((fq, 1.0 / fq.sqrt()))
    }
}
