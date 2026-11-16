//! Girard Linear Logic ($A \multimap B$, $A \otimes B$, $!A$), Danos-Regnier Proof-Net Criterion & Bounded Linear Logic (Girard, 1987; Danos & Regnier, 1989).
//!
//! ### 1. Danos-Regnier Switching Acyclicity & Connectedness Criterion
//! A multiplicative linear logic (MLL) proof structure with $L$ axiom links and $P$ par ($\parr$) / tensor ($\otimes$) gates is a valid sequentializable proof net
//! iff every Danos-Regnier par-switching graph is acyclic and connected, which requires Euler characteristic $|V| - |E| = 1$.

use crate::error::{Error, Result};

/// Girard Multiplicative Linear Logic & Danos-Regnier Proof-Net Verifier.
#[derive(Debug, Clone, Copy, Default)]
pub struct GirardLinearLogicResource;

impl GirardLinearLogicResource {
    /// Creates a new Girard Linear Logic Resource engine.
    pub fn new() -> Self {
        Self
    }

    /// Verifies exact linear resource conservation (each linear hypothesis consumed exactly once) and Danos-Regnier switching tree Euler characteristic $|V| - |E| == 1$.
    pub fn verify_linear_resource_and_switching_tree(
        usage_counts: &[usize],
        switching_vertices: usize,
        switching_edges: usize,
    ) -> Result<(bool, bool)> {
        if usage_counts.is_empty() || switching_vertices == 0 {
            return Err(Error::Msg(
                "Non-empty usage_counts and switching_vertices >= 1 required".to_string(),
            ));
        }
        let linear_exact = usage_counts.iter().all(|&c| c == 1);
        let euler_tree = switching_vertices == switching_edges + 1;
        Ok((linear_exact, euler_tree))
    }
}
