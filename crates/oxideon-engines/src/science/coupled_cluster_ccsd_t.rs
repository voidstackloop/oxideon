//! Coupled-Cluster $\mathrm{CCSD(T)}$ Gold-Standard Electronic Structure, Exponential Ansatz $e^{\hat{T}}$ & Lee-Taylor $T_1$ Diagnostic (Raghavachari et al., 1989).
//!
//! ### 1. Exponential Size-Extensive Ansatz & Lee-Taylor $T_1$ Multireference Diagnostic
//! Coupled-Cluster theory parameterizes the correlated wavefunction via $|\Psi_{\mathrm{CC}}\rangle = e^{\hat{T}_1 + \hat{T}_2 + \dots} |\Phi_0\rangle$,
//! whose BCH similarity-transformed Hamiltonian $e^{-\hat{T}} \hat{H} e^{\hat{T}}$ terminates exactly at 4 nested commutators.
//! The Lee-Taylor single-reference reliability diagnostic for $N_e$ correlated electrons is:
//! $$\mathcal{T}_1 = \frac{\|t_1\|_F}{\sqrt{N_e}} \quad (\mathcal{T}_1 < 0.02 \implies \text{reliable single-reference CCSD(T)})$$

use crate::error::{Error, Result};

/// Coupled-Cluster $\mathrm{CCSD(T)}$ & Lee-Taylor $T_1$ Diagnostic Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CoupledClusterCcsdTEngine;

impl CoupledClusterCcsdTEngine {
    /// Creates a new Coupled-Cluster $\mathrm{CCSD(T)}$ engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the Lee-Taylor $\mathcal{T}_1 = \sqrt{\sum_{ia} t_i^a{}^2 / N_e}$ diagnostic and checks if $\mathcal{T}_1 < 0.02$.
    pub fn lee_taylor_t1_diagnostic(
        t1_amplitudes: &[f64],
        correlated_electrons: usize,
    ) -> Result<(f64, bool)> {
        if t1_amplitudes.is_empty() || correlated_electrons == 0 {
            return Err(Error::Msg(
                "Non-empty t1_amplitudes and correlated_electrons >= 1 required".to_string(),
            ));
        }
        let norm_sq: f64 = t1_amplitudes.iter().map(|&t| t * t).sum();
        let diag = (norm_sq / (correlated_electrons as f64)).sqrt();
        Ok((diag, diag < 0.02))
    }
}
