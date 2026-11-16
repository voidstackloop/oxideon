//! Voevodsky's Motivic Cohomology & Norm Residue Isomorphism (Milnor & Bloch-Kato Conjectures) (Milnor, 1970; Voevodsky, 2003/2011).
//!
//! ### 1. Galois Symbol Isomorphism $K_n^M(F)/m \xrightarrow{\sim} H^n_{\text{ét}}(F, \mu_m^{\otimes n})$
//! Voevodsky proved that the norm residue homomorphism from Milnor $K$-theory modulo $m$ to étale Galois cohomology is an isomorphism for all fields $F$ with $\operatorname{char}(F) \nmid m$,
//! and that for a quadratic Pfister form $\langle\langle a_1, \dots, a_n \rangle\rangle$ of dimension $2^n$, its Witt invariant vanishes iff its symbol vanishes in $H^n(F, \mathbb{Z}/2)$.

use crate::error::{Error, Result};

/// Voevodsky Motivic Cohomology & Norm Residue (Bloch-Kato) Engine.
#[derive(Debug, Clone, Copy)]
pub struct VoevodskyMotivicBlochKato {
    /// Milnor degree / Pfister fold $n \ge 1$.
    pub milnor_degree_n: u32,
}

impl Default for VoevodskyMotivicBlochKato {
    fn default() -> Self {
        Self { milnor_degree_n: 3 }
    }
}

impl VoevodskyMotivicBlochKato {
    /// Creates a new Voevodsky Motivic & Bloch-Kato engine.
    pub fn new(milnor_degree_n: u32) -> Result<Self> {
        if milnor_degree_n == 0 || milnor_degree_n > 30 {
            return Err(Error::Msg("milnor_degree_n must be in 1..=30".to_string()));
        }
        Ok(Self { milnor_degree_n })
    }

    /// Evaluates the Pfister quadratic form dimension $2^n$ and the motivic weight $n$ of $H^n_{\mathcal{M}}(F, \mathbb{Z}(n)) \cong K_n^M(F)$.
    pub fn pfister_dimension_and_motivic_weight(&self) -> (u64, u32) {
        (1_u64 << self.milnor_degree_n, self.milnor_degree_n)
    }
}
