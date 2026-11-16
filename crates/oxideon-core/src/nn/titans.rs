//! Titans: Learning to Memorize at Test Time (Behrouz et al., 2024/2026).
//!
//! A neural long-term memory architecture that uses a persistent associative memory matrix $M_t$.
//! At each token step, the memory computes a "surprise" signal $\nabla \ell(M_{t-1}; k_t, v_t)$,
//! performing an online gradient-descent associative weight update:
//!
//! $$M_t = (1 - \alpha_t) M_{t-1} - \eta_t (M_{t-1} k_t - v_t) k_t^T$$
//!
//! where $\alpha_t \in (0, 1)$ is data-dependent decay and $\eta_t > 0$ is data-dependent learning rate.
//! Memory retrieval is computed as $y_t = q_t M_t$.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Configuration for Titans Neural Memory.
#[derive(Debug, Clone)]
pub struct TitansMemoryConfig {
    pub d_model: usize,
    pub memory_dim: usize,
    pub base_decay: f32,
    pub base_lr: f32,
}

impl TitansMemoryConfig {
    pub fn new(d_model: usize, memory_dim: usize) -> Self {
        Self {
            d_model,
            memory_dim,
            base_decay: 0.1,
            base_lr: 0.05,
        }
    }
}

/// Titans Neural Long-Term Memory Module.
pub struct TitansNeuralMemory {
    pub config: TitansMemoryConfig,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_alpha: Linear,
    pub w_eta: Linear,
    pub w_out: Linear,
}

impl TitansNeuralMemory {
    pub fn new(config: TitansMemoryConfig) -> Result<Self> {
        let d = config.d_model;
        let m = config.memory_dim;
        // forward_memory materializes the m x m associative matrix (and m-sized
        // work vectors) outside the checked tensor allocators; refuse up front
        // what can never fit instead of wrapping m*m or aborting on OOM.
        crate::limits::check_elements("TitansNeuralMemory memory", &[m, m])?;

        let w_q = Linear::with_bias(d, m, false)?;
        let w_k = Linear::with_bias(d, m, false)?;
        let w_v = Linear::with_bias(d, m, false)?;
        let w_alpha = Linear::with_bias(d, 1, true)?;
        let w_eta = Linear::with_bias(d, 1, true)?;
        let w_out = Linear::with_bias(m, d, false)?;

        Ok(Self {
            config,
            w_q,
            w_k,
            w_v,
            w_alpha,
            w_eta,
            w_out,
        })
    }

    /// Forward pass through the neural memory over sequence of shape `[batch_size, seq_len, d_model]`.
    /// Returns retrieved memory features of shape `[batch_size, seq_len, d_model]`.
    pub fn forward_memory(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(Error::Msg(
                "TitansNeuralMemory expects 3D tensor [batch_size, seq_len, d_model]".into(),
            ));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let d = shape[2];
        let m = self.config.memory_dim;

        if d != self.config.d_model {
            return Err(Error::ShapeMismatch {
                op: "TitansNeuralMemory",
                expected: vec![batch_size, seq_len, self.config.d_model],
                got: shape.to_vec(),
            });
        }

        // The config and projections are pub: re-validate the width agreement the slice
        // indexing and m*m scratch below assume, before any allocation happens.
        if self.w_q.out_features() != m
            || self.w_k.out_features() != m
            || self.w_v.out_features() != m
            || self.w_out.in_features() != m
        {
            return Err(Error::Msg(format!(
                "TitansNeuralMemory: projection widths disagree with memory_dim={m}"
            )));
        }

        // Project Q, K, V, Alpha, Eta
        let q = self.w_q.forward(input)?;
        let k = self.w_k.forward(input)?;
        let v = self.w_v.forward(input)?;
        let alpha = self.w_alpha.forward(input)?;
        let eta = self.w_eta.forward(input)?;

        let q_vec = q.to_vec::<f32>()?;
        let k_vec = k.to_vec::<f32>()?;
        let v_vec = v.to_vec::<f32>()?;
        let alpha_vec = alpha.to_vec::<f32>()?;
        let eta_vec = eta.to_vec::<f32>()?;

        // `m` is pinned to a real projection width above, but the width is
        // independent of the input shape, so these products can still overflow.
        let retrieved_len = batch_size
            .checked_mul(seq_len)
            .and_then(|v| v.checked_mul(m))
            .ok_or_else(|| Error::Msg("TitansNeuralMemory retrieved buffer overflows".into()))?;
        let mut retrieved_vec = vec![0.0f32; retrieved_len];

        for b in 0..batch_size {
            // Memory state M of size m x m initialized to 0
            let mem_len = m
                .checked_mul(m)
                .ok_or_else(|| Error::Msg("TitansNeuralMemory m*m scratch overflows".into()))?;
            let mut mem_state = vec![0.0f32; mem_len];

            for t in 0..seq_len {
                let token_offset = (b * seq_len + t) * m;
                let scalar_offset = b * seq_len + t;

                let q_t = &q_vec[token_offset..token_offset + m];
                let k_t = &k_vec[token_offset..token_offset + m];
                let v_t = &v_vec[token_offset..token_offset + m];

                // Gated decay and learning rate with sigmoid
                let a_raw = alpha_vec[scalar_offset];
                let decay = self.config.base_decay * (1.0 / (1.0 + (-a_raw).exp()));

                let e_raw = eta_vec[scalar_offset];
                let lr = self.config.base_lr * (1.0 / (1.0 + (-e_raw).exp()));

                // 1. Retrieve current memory output y_t = M_{t-1} * q_t
                let mut y_t = vec![0.0f32; m];
                for row in 0..m {
                    let mut sum = 0.0f32;
                    for col in 0..m {
                        sum += mem_state[row * m + col] * q_t[col];
                    }
                    y_t[row] = sum;
                }

                // Copy y_t into retrieved buffer
                retrieved_vec[token_offset..token_offset + m].copy_from_slice(&y_t);

                // 2. Compute surprise prediction error: pred_v = M_{t-1} * k_t
                let mut pred_v = vec![0.0f32; m];
                for row in 0..m {
                    let mut sum = 0.0f32;
                    for col in 0..m {
                        sum += mem_state[row * m + col] * k_t[col];
                    }
                    pred_v[row] = sum;
                }

                // Error gradient: grad_M = (pred_v - v_t) * k_t^T
                let mut err = vec![0.0f32; m];
                for row in 0..m {
                    err[row] = pred_v[row] - v_t[row];
                }

                // 3. Online associative update: M_t = (1 - decay) * M_{t-1} - lr * err * k_t^T
                let retain_factor = 1.0 - decay;
                for (row, &err_r) in err.iter().enumerate().take(m) {
                    let row_idx = row * m;
                    for col in 0..m {
                        let prev_val = mem_state[row_idx + col];
                        let delta = lr * err_r * k_t[col];
                        mem_state[row_idx + col] = retain_factor * prev_val - delta;
                    }
                }
            }
        }

        let ret_tensor = Tensor::from_slice(&retrieved_vec, &[batch_size, seq_len, m])?
            .to_device(&input.device())?;
        self.w_out.forward(&ret_tensor)
    }
}

impl Module for TitansNeuralMemory {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_memory(input)
    }
}

/// Titans Block combining persistent long-term Neural Memory with residual gated shortcuts.
pub struct TitansBlock {
    pub memory: TitansNeuralMemory,
    pub gate_linear: Linear,
}

impl TitansBlock {
    pub fn new(config: TitansMemoryConfig) -> Result<Self> {
        let d = config.d_model;
        let memory = TitansNeuralMemory::new(config)?;
        let gate_linear = Linear::with_bias(d, d, true)?;
        Ok(Self {
            memory,
            gate_linear,
        })
    }

    /// Forward pass through the Titans Block with dynamic input-conditioned residual gating.
    pub fn forward_block(&self, input: &Tensor) -> Result<Tensor> {
        let mem_out = self.memory.forward(input)?;
        let gate = self.gate_linear.forward(input)?;

        // Projections are pub: a width-swapped `w_out`/`gate_linear` shortens
        // one buffer and the gate loop below indexes past it.
        if mem_out.shape() != input.shape() || gate.shape() != input.shape() {
            return Err(Error::ShapeMismatch {
                op: "TitansBlock",
                expected: input.shape().to_vec(),
                got: mem_out.shape().to_vec(),
            });
        }

        // Sigmoid gate
        let gate_vec = gate.to_vec::<f32>()?;
        let mem_vec = mem_out.to_vec::<f32>()?;
        let in_vec = input.to_vec::<f32>()?;

        let mut out_vec = vec![0.0f32; input.numel()];
        for i in 0..out_vec.len() {
            let g = 1.0 / (1.0 + (-gate_vec[i]).exp());
            out_vec[i] = in_vec[i] + g * mem_vec[i];
        }

        Tensor::from_slice(&out_vec, input.shape())?.to_device(&input.device())
    }
}

impl Module for TitansBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_block(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_rejects_tampered_memory_dim() -> Result<()> {
        let mut mem = TitansNeuralMemory::new(TitansMemoryConfig::new(8, 4))?;
        mem.config.memory_dim = 1 << 40; // pub field: slice indexing below would panic
        let x = Tensor::zeros_f32(&[1, 2, 8])?;
        let err = mem
            .forward_memory(&x)
            .expect_err("tampered memory_dim must be an error, not a panic");
        assert!(err.to_string().contains("memory_dim"), "{err}");
        Ok(())
    }
}
