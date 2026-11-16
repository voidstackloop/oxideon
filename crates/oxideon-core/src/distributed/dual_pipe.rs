//! DualPipe: Bidirectional Pipeline Parallelism (DeepSeek-V3, 2024/2026).
//!
//! Conventional 1F1B (One-Forward-One-Backward) leaves large pipeline bubbles:
//! $\text{Bubble}_{1F1B} = \frac{PP - 1}{M}$.
//!
//! DualPipe executes two streams of micro-batches flowing in opposite pipeline directions
//! (Forward stream $0 \to PP-1$ and Reverse stream $PP-1 \to 0$), fully overlapping forward
//! compute with backward gradient communication to reduce bubbles towards zero:
//!
//! $$\text{Bubble}_{\text{DualPipe}} \approx \frac{PP - 1}{2M}$$

use crate::error::{Error, Result};

/// Direction of micro-batch flow in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualPipeDirection {
    ForwardStream, // 0 -> PP-1
    ReverseStream, // PP-1 -> 0
}

/// Nature of computation execution in a pipeline chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualPipeOpKind {
    Forward,
    BackwardInput,
    BackwardWeight,
}

/// A scheduled micro-batch execution chunk.
#[derive(Debug, Clone)]
pub struct DualPipeChunk {
    pub microbatch_id: usize,
    pub direction: DualPipeDirection,
    pub op: DualPipeOpKind,
    pub stage: usize,
}

/// A scheduled step on a pipeline rank, with optional overlapped communication.
#[derive(Debug, Clone)]
pub struct DualPipeStep {
    pub step_index: usize,
    pub rank: usize,
    pub chunk: DualPipeChunk,
    pub overlapped_comm: Option<String>,
}

/// DualPipe Bidirectional Pipeline Parallel Scheduler.
#[derive(Debug, Clone)]
pub struct DualPipeSchedule {
    pub num_stages: usize,
    pub num_microbatches: usize,
}

impl DualPipeSchedule {
    pub fn new(num_stages: usize, num_microbatches: usize) -> Result<Self> {
        if num_stages == 0 || num_microbatches == 0 {
            return Err(Error::Msg(
                "DualPipeSchedule requires num_stages > 0 and num_microbatches > 0".into(),
            ));
        }
        // Generating a rank's schedule materializes one step per (stage, microbatch) pair.
        crate::limits::check_repeats("num_stages", num_stages)?;
        crate::limits::check_repeats("num_microbatches", num_microbatches)?;
        Ok(Self {
            num_stages,
            num_microbatches,
        })
    }

    /// Theoretical pipeline bubble ratio $\frac{PP - 1}{2M}$.
    pub fn bubble_ratio(&self) -> f64 {
        if self.num_microbatches == 0 {
            return 1.0;
        }
        (self.num_stages.saturating_sub(1) as f64) / (2.0 * self.num_microbatches as f64)
    }

    /// Generates execution schedule for a specific rank in the pipeline.
    pub fn generate_schedule_for_rank(&self, rank: usize) -> Result<Vec<DualPipeStep>> {
        // Fields are pub: a tampered count would hang/OOM the loops below (new() caps
        // construction, but not later mutation).
        crate::limits::check_repeats("num_microbatches", self.num_microbatches)?;
        if rank >= self.num_stages {
            return Err(Error::Msg(format!(
                "Rank {rank} exceeds num_stages {}",
                self.num_stages
            )));
        }

        let mut steps = Vec::new();
        let pp = self.num_stages;
        let m = self.num_microbatches;
        let mut step_idx = 0;

        // Warmup: forward pass of forward stream and reverse stream
        let warmup_fwd = (pp - 1 - rank).min(m);
        for mb in 0..warmup_fwd {
            steps.push(DualPipeStep {
                step_index: step_idx,
                rank,
                chunk: DualPipeChunk {
                    microbatch_id: mb,
                    direction: DualPipeDirection::ForwardStream,
                    op: DualPipeOpKind::Forward,
                    stage: rank,
                },
                overlapped_comm: Some(format!("P2P-Send-Fwd(mb={mb}, dst={})", (rank + 1) % pp)),
            });
            step_idx += 1;
        }

        let warmup_rev = rank.min(m);
        for mb in 0..warmup_rev {
            steps.push(DualPipeStep {
                step_index: step_idx,
                rank,
                chunk: DualPipeChunk {
                    microbatch_id: mb,
                    direction: DualPipeDirection::ReverseStream,
                    op: DualPipeOpKind::Forward,
                    stage: rank,
                },
                overlapped_comm: Some(format!(
                    "P2P-Send-Rev(mb={mb}, dst={})",
                    rank.checked_sub(1).unwrap_or(pp - 1)
                )),
            });
            step_idx += 1;
        }

        // Steady State: 1F1B overlapping forward compute with backward communication
        let steady_count = m.saturating_sub(warmup_fwd.max(warmup_rev));
        for i in 0..steady_count {
            let mb_fwd = warmup_fwd + i;
            let mb_rev = warmup_rev + i;

            // Forward chunk with overlapped backward communication
            steps.push(DualPipeStep {
                step_index: step_idx,
                rank,
                chunk: DualPipeChunk {
                    microbatch_id: mb_fwd,
                    direction: DualPipeDirection::ForwardStream,
                    op: DualPipeOpKind::Forward,
                    stage: rank,
                },
                overlapped_comm: Some(format!("Bwd-Grad-Comm(mb={})", mb_fwd.saturating_sub(1))),
            });
            step_idx += 1;

            // Backward chunk with overlapped forward communication
            steps.push(DualPipeStep {
                step_index: step_idx,
                rank,
                chunk: DualPipeChunk {
                    microbatch_id: mb_rev,
                    direction: DualPipeDirection::ReverseStream,
                    op: DualPipeOpKind::BackwardInput,
                    stage: rank,
                },
                overlapped_comm: Some(format!("Fwd-Activation-Comm(mb={mb_rev})")),
            });
            step_idx += 1;
        }

        // Cooldown: backward weight gradients
        for mb in 0..m {
            steps.push(DualPipeStep {
                step_index: step_idx,
                rank,
                chunk: DualPipeChunk {
                    microbatch_id: mb,
                    direction: DualPipeDirection::ForwardStream,
                    op: DualPipeOpKind::BackwardWeight,
                    stage: rank,
                },
                overlapped_comm: None,
            });
            step_idx += 1;
        }

        Ok(steps)
    }
}
