//! Lamport Logical Clocks & Fidge-Mattern Vector Clock Causal Partial Order (Lamport, 1978; Fidge, 1988; Mattern, 1989).
//!
//! ### 1. Exact Causal Isomorphism $(e_1 \to e_2 \iff V(e_1) < V(e_2))$
//! For $N$-process vector timestamps $u, v \in \mathbb{N}^N$, $u$ causally precedes $v$ ($u \to v$) iff $u_k \le v_k$ for all $k$ and $u \ne v$;
//! two events are causally concurrent ($u \parallel v$) iff neither $u \le v$ nor $v \le u$.

use crate::error::{Error, Result};

/// Fidge-Mattern Vector Clock & Causal Happens-Before Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatternVectorClockCausality;

impl MatternVectorClockCausality {
    /// Creates a new Mattern Vector Clock Causality engine.
    pub fn new() -> Self {
        Self
    }

    /// Merges two vector clocks via componentwise supremum $\sup(u, v)_k = \max(u_k, v_k)$ and classifies causality: $-1$ if $u \to v$, $+1$ if $v \to u$, $0$ if concurrent or equal.
    pub fn merge_and_classify_causality(
        clock_u: &[u64],
        clock_v: &[u64],
    ) -> Result<(Vec<u64>, i32, bool)> {
        if clock_u.is_empty() || clock_u.len() != clock_v.len() {
            return Err(Error::Msg(
                "Vector clocks must be non-empty and have equal length".to_string(),
            ));
        }
        let u_le_v = clock_u.iter().zip(clock_v.iter()).all(|(&a, &b)| a <= b);
        let v_le_u = clock_u.iter().zip(clock_v.iter()).all(|(&a, &b)| b <= a);
        let merged: Vec<u64> = clock_u
            .iter()
            .zip(clock_v.iter())
            .map(|(&a, &b)| a.max(b))
            .collect();
        let order = if u_le_v && !v_le_u {
            -1
        } else if v_le_u && !u_le_v {
            1
        } else {
            0
        };
        let is_concurrent = !u_le_v && !v_le_u;
        Ok((merged, order, is_concurrent))
    }
}
