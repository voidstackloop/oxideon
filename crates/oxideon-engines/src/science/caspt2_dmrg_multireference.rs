//! Complete Active Space $\mathrm{CAS}(n_e, n_o)$ Weyl Dimension Formula & Quantum Chemistry DMRG Orbital Entanglement (Roos, 1980; White & Martin, 1999).
//!
//! ### 1. Weyl-Paldus Spin-Adapted Configuration State Function (CSF) Dimension
//! For $N$ active electrons in $M$ spatial orbitals coupled to total spin $S$ ($b = 2S$), the exact dimension of the
//! spin-adapted Full CI space $\mathrm{CAS}(N, M)$ is given by Weyl's $U(M)$ Irrep Dimension Formula:
//! $$\dim \mathcal{H}_{\mathrm{CSF}}(N, M, S) = \frac{2 S + 1}{M + 1} \binom{M + 1}{N/2 - S} \binom{M + 1}{N/2 + S + 1}$$

use crate::error::{Error, Result};

/// Multireference Complete Active Space $\mathrm{CAS}(N, M)$ & Weyl-Paldus Dimension Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct MultireferenceCasEngine;

impl MultireferenceCasEngine {
    /// Creates a new Multireference CAS engine.
    pub fn new() -> Self {
        Self
    }

    fn binom(n: u64, k: u64) -> u64 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut res = 1_u64;
        for i in 1..=k {
            res = res * (n + 1 - i) / i;
        }
        res
    }

    /// Computes the exact Weyl-Paldus number of spin-adapted singlet ($S = 0$) Configuration State Functions in $\mathrm{CAS}(2k, M)$.
    pub fn singlet_weyl_csf_dimension(even_electrons_n: u64, orbitals_m: u64) -> Result<u64> {
        if even_electrons_n % 2 != 0 || even_electrons_n > 2 * orbitals_m || orbitals_m > 25 {
            return Err(Error::Msg("Requires even N <= 2*M and M <= 25".to_string()));
        }
        let half_n = even_electrons_n / 2;
        let c1 = Self::binom(orbitals_m + 1, half_n);
        let c2 = Self::binom(orbitals_m + 1, half_n + 1);
        Ok((c1 * c2) / (orbitals_m + 1))
    }
}
