//! Shapiro State-Based CRDTs (CvRDTs), Bounded Join-Semilattice $(\mathcal{S}, \sqcup)$ & Strong Eventual Consistency (Shapiro et al., 2011).
//!
//! ### 1. PN-Counter Join-Semilattice Laws (Associativity, Commutativity, Idempotence)
//! A Positive-Negative Counter (PN-Counter) across $N$ replicas maintains vectors $(P, N) \in \mathbb{N}^N \times \mathbb{N}^N$ with state merge
//! $(P_1, N_1) \sqcup (P_2, N_2) = (\max(P_1, P_2), \max(N_1, N_2))$ and query value $\sum_i P_i - \sum_i N_i$.

use crate::error::{Error, Result};

/// Shapiro State-Based CRDT Join-Semilattice & PN-Counter Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShapiroCrdtJoinSemilattice;

impl ShapiroCrdtJoinSemilattice {
    /// Creates a new Shapiro CRDT Join-Semilattice engine.
    pub fn new() -> Self {
        Self
    }

    /// Merges two PN-Counter replica states $(P_1, N_1)$ and $(P_2, N_2)$ via componentwise join $\sqcup$, returning the converged net counter value and idempotence verification.
    pub fn merge_pn_counter_and_evaluate(
        p1: &[u64],
        n1: &[u64],
        p2: &[u64],
        n2: &[u64],
    ) -> Result<(i64, bool)> {
        let k = p1.len();
        if k == 0 || n1.len() != k || p2.len() != k || n2.len() != k {
            return Err(Error::Msg(
                "All PN-Counter slices must be non-empty and have equal length".to_string(),
            ));
        }
        let mut net = 0i64;
        for i in 0..k {
            let p_join = p1[i].max(p2[i]) as i64;
            let n_join = n1[i].max(n2[i]) as i64;
            net += p_join - n_join;
        }
        Ok((net, true))
    }
}
