//! Voevodsky Univalence Axiom $(A =_{\mathcal{U}} B) \simeq (A \simeq B)$ & Synthetic Freudenthal Suspension Theorem in HoTT (Voevodsky, 2010; Univalent Foundations Program, 2013).
//!
//! ### 1. Univalence Equivalence Cardinality $|\mathrm{Aut}(A)| = n!$ & Freudenthal Stable Range $k \le 2n$
//! For a $0$-type (set) $A$ of finite cardinality $n$, Univalence identifies the loop space $\Omega(\mathcal{U}, A) = (A =_{\mathcal{U}} A)$ with $\mathrm{Aut}(A) \cong S_n$ of cardinality $n!$; for an $n$-connected space $X$, $\pi_k(X) \to \pi_{k+1}(\Sigma X)$ is an isomorphism for $k \le 2n$.

use crate::error::{Error, Result};

/// Voevodsky Univalence Axiom & Synthetic Homotopy Type Theory Engine.
#[derive(Debug, Clone, Copy)]
pub struct HomotopyTypeTheoryUnivalence {
    /// Finite set cardinality $n \ge 1$.
    pub finite_set_size_n: usize,
}

impl Default for HomotopyTypeTheoryUnivalence {
    fn default() -> Self {
        Self {
            finite_set_size_n: 5,
        }
    }
}

impl HomotopyTypeTheoryUnivalence {
    /// Creates a new Homotopy Type Theory Univalence engine.
    pub fn new(finite_set_size_n: usize) -> Result<Self> {
        if finite_set_size_n == 0 || finite_set_size_n > 20 {
            return Err(Error::Msg(
                "finite_set_size_n must be in 1..=20".to_string(),
            ));
        }
        Ok(Self { finite_set_size_n })
    }

    /// Evaluates the universe loop-space cardinality $|\Omega(\mathcal{U}, \mathrm{Fin}_n)| = n!$ by Univalence and the Freudenthal stable range $2 n_{\mathrm{conn}}$.
    pub fn loop_space_cardinality_and_freudenthal_bound(
        &self,
        connectivity_n: usize,
    ) -> Result<(u64, usize)> {
        let n = self.finite_set_size_n;
        if n == 0 || n > 20 {
            return Err(Error::Msg(format!(
                "finite_set_size_n must be in 1..=20, got {n}"
            )));
        }
        let range = connectivity_n
            .checked_mul(2)
            .ok_or_else(|| Error::Msg("2 * connectivity_n overflows".to_string()))?;
        let mut fact = 1_u64;
        for k in 2..=n {
            fact = fact
                .checked_mul(k as u64)
                .ok_or_else(|| Error::Msg("n! overflows u64".to_string()))?;
        }
        Ok((fact, range))
    }
}
