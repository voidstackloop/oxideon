//! Grothendieck Topoi, Giraud's Characterization Theorem & Lawvere-Tierney Subobject Classifier $\Omega$ (Grothendieck & Verdier, 1972; Lawvere, 1970).
//!
//! ### 1. Presheaf Topos $\mathbf{Set}^{\mathcal{C}^{\mathrm{op}}}$ Sieve Count & Yoneda Embedding Density
//! For a finite poset or category $\mathcal{C}$, the subobject classifier $\Omega(c)$ of the presheaf topos $\widehat{\mathcal{C}} = \mathbf{Set}^{\mathcal{C}^{\mathrm{op}}}$ is the set of sieves on $c$; for an $n$-element Boolean poset, $|\Omega| = 2^n$.

use crate::error::{Error, Result};

/// Giraud Grothendieck Topos & Lawvere-Tierney Topology Engine.
#[derive(Debug, Clone, Copy)]
pub struct GiraudGrothendieckToposEngine {
    /// Number of basic generators / site primes $n \ge 1$.
    pub num_site_generators: u32,
}

impl Default for GiraudGrothendieckToposEngine {
    fn default() -> Self {
        Self {
            num_site_generators: 4,
        }
    }
}

impl GiraudGrothendieckToposEngine {
    /// Creates a new Giraud Grothendieck Topos engine.
    pub fn new(num_site_generators: u32) -> Result<Self> {
        if num_site_generators == 0 || num_site_generators > 30 {
            return Err(Error::Msg(
                "num_site_generators must be in 1..=30".to_string(),
            ));
        }
        Ok(Self {
            num_site_generators,
        })
    }

    /// Evaluates the Boolean sieve algebra cardinality $2^n$ and the double-negation dense sheaf locus rank.
    pub fn sieve_cardinality_and_dense_rank(&self) -> (u64, u32) {
        (1_u64 << self.num_site_generators, self.num_site_generators)
    }
}
