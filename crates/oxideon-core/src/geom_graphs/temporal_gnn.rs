//! Temporal Interaction Graph Neural Network (`TGN`, Rossi et al., Twitter / NeurIPS 2020).
//!
//! Models continuous-time dynamic interaction events $(u, v, t, e)$ with continuous
//! Fourier time encoding and persistent dynamic node memory states.

use crate::error::{Error, Result};
use crate::nn::{GRUCell, Linear, Module};
use crate::tensor::Tensor;

/// Continuous-Time Dynamic Graph Network.
pub struct TemporalInteractionGNN {
    pub num_nodes: usize,
    pub memory_dim: usize,
    pub time_dim: usize,
    pub msg_proj: Linear,
    pub gru_cell: GRUCell,
    pub node_memories: Vec<Tensor>,
    pub last_timestamps: Vec<f64>,
}

impl TemporalInteractionGNN {
    pub fn new(
        num_nodes: usize,
        memory_dim: usize,
        time_dim: usize,
        edge_dim: usize,
    ) -> Result<Self> {
        // Message input: [src_mem, dst_mem, time_enc, edge_feat] -> size: 2 * memory_dim + time_dim + edge_dim
        let msg_in_dim = 2usize
            .checked_mul(memory_dim)
            .and_then(|v| v.checked_add(time_dim))
            .and_then(|v| v.checked_add(edge_dim))
            .ok_or_else(|| {
                crate::error::Error::Msg(format!(
                    "TemporalInteractionGNN: message dim 2 * {memory_dim} + {time_dim} + {edge_dim} overflows"
                ))
            })?;
        let msg_proj = Linear::new(msg_in_dim, memory_dim)?;
        let gru_cell = GRUCell::new(memory_dim, memory_dim)?;

        // One small tensor per node: millions of nodes are plausible, billions are not.
        crate::limits::check_at_most("num_nodes", num_nodes, 1 << 22)?;
        let mut node_memories = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            node_memories.push(Tensor::zeros(
                &[1, memory_dim],
                crate::dtype::DType::F32,
                &crate::backend::Device::Cpu,
            )?);
        }

        Ok(Self {
            num_nodes,
            memory_dim,
            time_dim,
            msg_proj,
            gru_cell,
            node_memories,
            last_timestamps: vec![0.0; num_nodes],
        })
    }

    /// Continuous harmonic Fourier time encoding:
    /// $$\Phi(\Delta t) = \left[\cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \dots\right]$$
    pub fn encode_time_delta(&self, dt: f64) -> Result<Tensor> {
        let half = self.time_dim / 2;
        let mut out = vec![0.0f32; self.time_dim];
        for i in 0..half {
            let freq = (1.0 / 10000.0f64.powf((2 * i) as f64 / (self.time_dim as f64))) as f32;
            let angle = (dt as f32) * freq;
            out[i] = angle.cos();
            out[half + i] = angle.sin();
        }
        Tensor::from_vec(out, &[1, self.time_dim])
    }

    /// Processes dynamic interaction event between `src_node` and `dst_node` at `timestamp`.
    pub fn process_event(
        &mut self,
        src: usize,
        dst: usize,
        timestamp: f64,
        edge_feat: &Tensor,
    ) -> Result<Tensor> {
        if src >= self.num_nodes || dst >= self.num_nodes {
            return Err(Error::IndexOutOfBounds {
                op: "TemporalInteractionGNN::process_event",
                index: src.max(dst) as isize,
                dim_size: self.num_nodes,
            });
        }
        if self.node_memories.len() != self.num_nodes
            || self.last_timestamps.len() != self.num_nodes
        {
            return Err(Error::Msg(format!(
                "TGN state lengths ({}, {}) disagree with num_nodes {}",
                self.node_memories.len(),
                self.last_timestamps.len(),
                self.num_nodes
            )));
        }
        let dt = (timestamp - self.last_timestamps[dst]).max(0.0);
        let t_enc = self.encode_time_delta(dt)?;

        let src_mem = &self.node_memories[src];
        let dst_mem = &self.node_memories[dst];

        // Form event message
        let raw_msg = Tensor::cat(
            &[src_mem.clone(), dst_mem.clone(), t_enc, edge_feat.clone()],
            1,
        )?;
        let msg = self.msg_proj.forward(&raw_msg)?.relu()?;

        // Update destination memory state via GRU
        let new_dst_mem = self.gru_cell.step(&msg, Some(dst_mem))?;
        self.node_memories[dst] = new_dst_mem.clone();
        self.last_timestamps[dst] = timestamp;

        Ok(new_dst_mem)
    }
}
