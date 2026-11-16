//! Lamport-Pease-Shostak Byzantine Generals $N \ge 3f + 1$ & Castro-Liskov PBFT Quorum Intersection (Lamport, Shostak & Pease, 1982; Castro & Liskov, 1999).
//!
//! ### 1. Asynchronous Byzantine Quorum Intersection Invariant
//! In a network of $N$ replicas tolerating $f$ Byzantine faults under partial synchrony, safety and liveness require $N \ge 3f + 1$.
//! Any two quorums of size $Q = \lceil (N + f + 1)/2 \rceil$ intersect in at least $2Q - N \ge f + 1$ replicas, guaranteeing at least one honest replica in common.

use crate::error::{Error, Result};

/// Lamport-PBFT Byzantine Quorum Intersection & Safety Engine.
#[derive(Debug, Clone, Copy)]
pub struct ByzantineQuorumConsensusEngine {
    /// Total number of replicas $N \ge 4$.
    pub num_replicas_n: usize,
}

impl Default for ByzantineQuorumConsensusEngine {
    fn default() -> Self {
        Self { num_replicas_n: 4 }
    }
}

impl ByzantineQuorumConsensusEngine {
    /// Creates a new Byzantine Quorum Consensus engine.
    pub fn new(num_replicas_n: usize) -> Result<Self> {
        if num_replicas_n < 4 {
            return Err(Error::Msg(
                "num_replicas_n must be >= 4 for BFT".to_string(),
            ));
        }
        Ok(Self { num_replicas_n })
    }

    /// Evaluates maximum tolerable Byzantine faults $f_{\mathrm{max}} = (N - 1)/3$, minimum PBFT quorum size $Q = 2 f_{\mathrm{max}} + 1$, and honest intersection margin $2Q - N - f_{\mathrm{max}} \ge 1$.
    pub fn max_faults_quorum_and_honest_intersection(&self) -> (usize, usize, usize) {
        let f_max = (self.num_replicas_n - 1) / 3;
        let quorum = 2 * f_max + 1;
        let honest_overlap = (2 * quorum).saturating_sub(self.num_replicas_n + f_max);
        (f_max, quorum, honest_overlap)
    }
}
