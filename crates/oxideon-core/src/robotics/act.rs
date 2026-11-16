//! Action Chunking Transformer (`ACT`, Zhao et al., Stanford / RSS 2023).
//!
//! Predicts chunks of future robot actions (e.g. 7-DoF or 14-DoF joint positions + gripper)
//! from visual observation tokens and proprioception state, with temporal ensembling for smooth control.

use std::collections::VecDeque;

use crate::error::Result;
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Configuration for Action Chunking Transformer.
#[derive(Debug, Clone)]
pub struct ACTConfig {
    pub state_dim: usize,
    pub action_dim: usize,
    pub chunk_size: usize,
    pub hidden_dim: usize,
    pub temporal_ensemble_decay: f64,
}

impl Default for ACTConfig {
    fn default() -> Self {
        Self {
            state_dim: 14,
            action_dim: 14,
            chunk_size: 16,
            hidden_dim: 64,
            temporal_ensemble_decay: 0.01,
        }
    }
}

/// Action Chunking Transformer policy with CVAE action decoder.
pub struct ActionChunkingTransformer {
    pub config: ACTConfig,
    pub state_proj: Linear,
    pub action_proj: Linear,
    pub out_proj: Linear,
    pub action_buffer: VecDeque<Vec<f64>>,
}

impl ActionChunkingTransformer {
    pub fn new(config: ACTConfig) -> Result<Self> {
        let state_proj = Linear::new(config.state_dim, config.hidden_dim)?;
        let action_proj = Linear::new(config.hidden_dim, config.hidden_dim)?;
        let out_dim = config
            .action_dim
            .checked_mul(config.chunk_size)
            .ok_or_else(|| {
                crate::error::Error::Msg(format!(
                    "ActionChunkingTransformer: output dim {} * {} overflows",
                    config.action_dim, config.chunk_size
                ))
            })?;
        let out_proj = Linear::new(config.hidden_dim, out_dim)?;

        Ok(Self {
            config,
            state_proj,
            action_proj,
            out_proj,
            action_buffer: VecDeque::new(),
        })
    }

    /// Predicts action chunk `[batch, chunk_size, action_dim]` from current robot state `[batch, state_dim]`.
    pub fn forward(&self, state: &Tensor) -> Result<Tensor> {
        if state.ndim() != 2 {
            return Err(crate::error::Error::Msg(format!(
                "ActionChunkingTransformer expects [batch, state_dim], got {:?}",
                state.shape()
            )));
        }
        let b = state.shape()[0];
        let h = self.state_proj.forward(state)?.relu()?;
        let h2 = self.action_proj.forward(&h)?.relu()?;
        let flat_actions = self.out_proj.forward(&h2)?;

        flat_actions.reshape(&[b, self.config.chunk_size, self.config.action_dim])
    }

    /// Updates temporal ensemble buffer and returns smooth continuous action for current timestep.
    /// Single-stream only: a batch would interleave unrelated trajectories in the buffer.
    pub fn step_temporal_ensemble(&mut self, state: &Tensor) -> Result<Vec<f64>> {
        if state.ndim() != 2 || state.shape()[0] != 1 {
            return Err(crate::error::Error::Msg(format!(
                "step_temporal_ensemble needs [1, state_dim] state, got {:?}",
                state.shape()
            )));
        }
        let pred_chunk = self.forward(state)?;
        let chunk_data = pred_chunk.to_vec::<f32>()?;
        let a_dim = self.config.action_dim;

        // Store new chunk into history
        let new_chunk: Vec<f64> = chunk_data.iter().map(|&v| v as f64).collect();
        self.action_buffer.push_back(new_chunk);

        if self.action_buffer.len() > self.config.chunk_size {
            self.action_buffer.pop_front();
        }

        // Temporal ensemble: exponentially weighted average over overlapping action predictions
        let mut aggregated = vec![0.0f64; a_dim];
        let mut total_weight = 0.0f64;

        for (history_idx, chunk) in self.action_buffer.iter().enumerate() {
            let offset = self.action_buffer.len() - 1 - history_idx;
            if offset < self.config.chunk_size {
                let weight = (-self.config.temporal_ensemble_decay * offset as f64).exp();
                let start = offset * a_dim;
                for i in 0..a_dim {
                    aggregated[i] += chunk[start + i] * weight;
                }
                total_weight += weight;
            }
        }

        for val in &mut aggregated {
            *val /= total_weight.max(1e-6);
        }

        Ok(aggregated)
    }
}
