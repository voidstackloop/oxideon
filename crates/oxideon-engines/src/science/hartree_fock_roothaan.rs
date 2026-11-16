//! Roothaan-Hall Restricted Hartree-Fock SCF Equations $F C = S C \varepsilon$ & Pulay DIIS Convergence (Roothaan, 1951; Pulay, 1980).
//!
//! ### 1. Closed-Shell Hartree-Fock Electronic Energy & Pulay Commutator Error
//! Given one-electron core Hamiltonian $H_{\mathrm{core}}$, Fock matrix $F$, and spin-summed density matrix $P = 2 C_{\mathrm{occ}} C_{\mathrm{occ}}^T$,
//! the restricted Hartree-Fock electronic energy is:
//! $$E_{\mathrm{RHF}} = \frac{1}{2} \sum_{\mu \nu} P_{\mu \nu} \left(H_{\mu \nu}^{\mathrm{core}} + F_{\mu \nu}\right)$$

use crate::error::{Error, Result};

/// Roothaan-Hall Hartree-Fock SCF & Pulay DIIS Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoothaanHallScfEngine;

impl RoothaanHallScfEngine {
    /// Creates a new Roothaan-Hall SCF engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the closed-shell Hartree-Fock electronic energy $E_{\mathrm{elec}} = \frac{1}{2} \operatorname{Tr}[P (H_{\mathrm{core}} + F)]$.
    pub fn rhf_electronic_energy(density_p: &[f64], h_core: &[f64], fock_f: &[f64]) -> Result<f64> {
        if density_p.is_empty()
            || density_p.len() != h_core.len()
            || density_p.len() != fock_f.len()
        {
            return Err(Error::Msg(
                "Matrices P, H_core, and F must have identical non-zero length".to_string(),
            ));
        }
        let mut energy = 0.0;
        for i in 0..density_p.len() {
            energy += 0.5 * density_p[i] * (h_core[i] + fock_f[i]);
        }
        Ok(energy)
    }
}
