//! Brewer-Gilbert-Lynch CAP Theorem, Abadi PACELC Latency-Consistency Bound & Strict Quorum Read/Write ($R + W > N$) (Gilbert & Lynch, 2002; Abadi, 2012).
//!
//! ### 1. Gifford Quorum Overlap & Read-Your-Writes Linearizability
//! For $N$ replicas with write quorum $W$ and read quorum $R$, strong consistency (zero stale reads in the absence of partitions) holds iff:
//! $$R + W > N, \qquad W > \frac{N}{2}$$

use crate::error::{Error, Result};

/// Gilbert-Lynch CAP Theorem, PACELC & Gifford Quorum Engine.
#[derive(Debug, Clone, Copy)]
pub struct GilbertLynchPacelcTheorem {
    /// Total replication factor $N \ge 1$.
    pub replication_n: usize,
}

impl Default for GilbertLynchPacelcTheorem {
    fn default() -> Self {
        Self { replication_n: 5 }
    }
}

impl GilbertLynchPacelcTheorem {
    /// Creates a new Gilbert-Lynch PACELC & Quorum engine.
    pub fn new(replication_n: usize) -> Result<Self> {
        if replication_n == 0 {
            return Err(Error::Msg("replication_n must be >= 1".to_string()));
        }
        Ok(Self { replication_n })
    }

    /// Evaluates whether $(R, W)$ satisfies Gifford strong read-write overlap ($R + W > N$) and write-write conflict freedom ($2W > N$), returning the overlap size $(R + W) - N$.
    pub fn verify_strong_quorum(
        &self,
        read_quorum_r: usize,
        write_quorum_w: usize,
    ) -> Result<(i64, bool, bool)> {
        if read_quorum_r == 0
            || write_quorum_w == 0
            || read_quorum_r > self.replication_n
            || write_quorum_w > self.replication_n
        {
            return Err(Error::Msg("Quorums R and W must be in 1..=N".to_string()));
        }
        let overlap = (read_quorum_r + write_quorum_w) as i64 - (self.replication_n as i64);
        let strong_rw = overlap > 0;
        let strong_ww = 2 * write_quorum_w > self.replication_n;
        Ok((overlap, strong_rw, strong_ww))
    }
}
