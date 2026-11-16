//! RingAttention: Distributed Ring-Exchange Attention for 1M+ Token Contexts.
//!
//! Partitions long sequences across $N$ cluster ranks in a circular ring topology:
//! Each rank holds local queries $Q_r$ and passes Key/Value blocks along the ring,
//! updating local online softmax statistics $(m, l, O)$ with zero sequence-length bottleneck.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Distributed RingAttention coordinator.
#[derive(Debug, Clone)]
pub struct RingAttention {
    pub world_size: usize,
    pub rank: usize,
    pub causal: bool,
}

impl RingAttention {
    pub fn new(world_size: usize, rank: usize, causal: bool) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(Error::Msg(format!(
                "RingAttention needs world_size > 0 and rank < world_size, got world_size={world_size}, rank={rank}"
            )));
        }
        Ok(Self {
            world_size,
            rank,
            causal,
        })
    }

    /// Simulates or executes full RingAttention across all ranks in the ring,
    /// returning the local output for `rank`.
    ///
    /// - `q_local`: local query partition `[batch, heads, seq_chunk, head_dim]`
    /// - `k_shards`: all rank key shards (or received over ring)
    /// - `v_shards`: all rank value shards (or received over ring)
    pub fn forward_simulated(
        &self,
        q_local: &Tensor,
        k_shards: &[Tensor],
        v_shards: &[Tensor],
    ) -> Result<Tensor> {
        // `rank` is pub: a tampered rank overflowed the ring arithmetic below (panic).
        if self.rank >= self.world_size {
            return Err(Error::Msg(format!(
                "RingAttention rank {} out of bounds for world_size {}",
                self.rank, self.world_size
            )));
        }
        if k_shards.len() != self.world_size || v_shards.len() != self.world_size {
            return Err(Error::Msg(format!(
                "RingAttention world_size {} mismatch with shards ({}, {})",
                self.world_size,
                k_shards.len(),
                v_shards.len()
            )));
        }
        if q_local.ndim() != 4 {
            return Err(Error::Msg(format!(
                "RingAttention expects 4D q_local [B, H, T, D], got {:?}",
                q_local.shape()
            )));
        }
        let (b, h, d) = (q_local.shape()[0], q_local.shape()[1], q_local.shape()[3]);
        for (i, (k, v)) in k_shards.iter().zip(v_shards.iter()).enumerate() {
            if k.ndim() != 4 || v.ndim() != 4 {
                return Err(Error::Msg(format!(
                    "RingAttention shard {i} must be 4D, got {:?}/{:?}",
                    k.shape(),
                    v.shape()
                )));
            }
            if k.shape()[0] != b || k.shape()[1] != h || k.shape()[3] != d {
                return Err(Error::ShapeMismatch {
                    op: "RingAttention",
                    expected: vec![b, h, k.shape()[2], d],
                    got: k.shape().to_vec(),
                });
            }
            if v.shape() != k.shape() {
                return Err(Error::ShapeMismatch {
                    op: "RingAttention",
                    expected: k.shape().to_vec(),
                    got: v.shape().to_vec(),
                });
            }
        }

        let scale = 1.0 / (d as f64).sqrt();
        let q_chunk_len = q_local.shape()[2];

        let mut m_running: Option<Tensor> = None;
        let mut l_running: Option<Tensor> = None;
        let mut o_running: Option<Tensor> = None;

        // Iterate over ring communication steps s = 0 .. world_size - 1
        for step in 0..self.world_size {
            // Source rank of incoming KV shard at step s
            let src_rank = (self.rank + self.world_size - step) % self.world_size;

            // In causal mode: if the incoming KV block is strictly in the future of our Q block, skip!
            if self.causal && src_rank > self.rank {
                continue;
            }

            let k_block = &k_shards[src_rank];
            let v_block = &v_shards[src_rank];
            let kv_chunk_len = k_block.shape()[2];

            // Q K^T
            let mut s_ij = q_local.matmul(&k_block.t()?)?.mul_scalar(scale)?;

            // If src_rank == rank (diagonal block), apply local causal mask
            if self.causal && src_rank == self.rank {
                let mut mask = vec![false; q_chunk_len * kv_chunk_len];
                for qi in 0..q_chunk_len {
                    for kj in 0..kv_chunk_len {
                        if kj > qi {
                            mask[qi * kv_chunk_len + kj] = true;
                        }
                    }
                }
                let mask_t =
                    Tensor::from_slice_on(&mask, &[q_chunk_len, kv_chunk_len], &s_ij.device())?;
                s_ij = s_ij.masked_fill(&mask_t, -1e9)?;
            }

            // Online softmax update
            let m_tile = s_ij.max_axis(-1, true)?;

            let (m_new, p_tile) = match &m_running {
                Some(m_prev) => {
                    let m_curr = m_prev.maximum(&m_tile)?;
                    let p = s_ij.sub(&m_curr)?.exp()?;
                    (m_curr, p)
                }
                None => {
                    let p = s_ij.sub(&m_tile)?.exp()?;
                    (m_tile, p)
                }
            };

            let l_tile = p_tile.sum_axis(-1, true)?;
            let pv_tile = p_tile.matmul(v_block)?;

            let (l_new, o_new) = match (&m_running, &l_running, &o_running) {
                (Some(m_prev), Some(l_prev), Some(o_prev)) => {
                    let alpha = m_prev.sub(&m_new)?.exp()?;
                    let l_rescaled = l_prev.mul(&alpha)?.add(&l_tile)?;
                    let o_rescaled = o_prev.mul(&alpha)?.add(&pv_tile)?;
                    (l_rescaled, o_rescaled)
                }
                _ => (l_tile, pv_tile),
            };

            m_running = Some(m_new);
            l_running = Some(l_new);
            o_running = Some(o_new);
        }

        match (o_running, l_running) {
            (Some(o), Some(l)) => o.div(&l.add_scalar(1e-8)?),
            _ => Tensor::zeros(q_local.shape(), q_local.dtype(), &q_local.device()),
        }
    }
}
