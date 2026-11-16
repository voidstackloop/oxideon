//! Mac Lane Coherence Theorem for Monoidal Categories & Stasheff Associahedron Pentagon Counts $C_{n-1}$ (Mac Lane, 1963; Stasheff, 1963).
//!
//! ### 1. Catalan Parenthesizations $C_{n-1} = \frac{1}{n}\binom{2n-2}{n-1}$ & Pentagon Commutativity
//! By Mac Lane's Coherence Theorem, every formal diagram built from the associator $\alpha_{A,B,C} : (A \otimes B) \otimes C \xrightarrow{\sim} A \otimes (B \otimes C)$
//! and unitors $(\lambda_A, \rho_A)$ commutes iff the pentagon and triangle axioms hold; the number of binary parenthesizations of an $n$-fold tensor product is the $(n-1)$-th Catalan number $C_{n-1}$.

use crate::error::{Error, Result};

/// Mac Lane Monoidal Coherence & Stasheff Associahedron Vertex Engine.
#[derive(Debug, Clone, Copy)]
pub struct MacLaneMonoidalCoherence {
    /// Arity $n \ge 1$ of the tensor product $A_1 \otimes \dots \otimes A_n$.
    pub tensor_arity_n: usize,
}

impl Default for MacLaneMonoidalCoherence {
    fn default() -> Self {
        Self { tensor_arity_n: 4 }
    }
}

impl MacLaneMonoidalCoherence {
    /// Creates a new Mac Lane Monoidal Coherence engine.
    pub fn new(tensor_arity_n: usize) -> Result<Self> {
        if tensor_arity_n == 0 || tensor_arity_n > 20 {
            return Err(Error::Msg("tensor_arity_n must be in 1..=20".to_string()));
        }
        Ok(Self { tensor_arity_n })
    }

    /// Evaluates the number of parenthesizations (vertices of Stasheff's associahedron $K_n$), equal to $C_{n-1}$, and the associahedron dimension $n - 2$.
    pub fn associahedron_vertices_and_dimension(&self) -> (u64, usize) {
        let m = self.tensor_arity_n - 1;
        let mut cat = 1_u64;
        for k in 0..m {
            cat = cat * (2 * (k as u64) + 1) * 2 / ((k as u64) + 2);
        }
        (cat, self.tensor_arity_n.saturating_sub(2))
    }
}
