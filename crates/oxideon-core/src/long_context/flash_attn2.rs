//! FlashAttention-2: Faster and Memory-Efficient Exact Attention with Online Softmax and Autograd Backward.
//!
//! Based on Tri Dao (2023) "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning".
//! Provides:
//! - Fused forward pass with online softmax scaling and log-sum-exp accumulation ($L = m + \ln l$).
//! - Exact backward pass recomputing attention blocks on the fly without storing $N \times N$ matrices ($O(N)$ memory).

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for FlashAttention-2.
#[derive(Debug, Clone)]
pub struct FlashAttention2Config {
    pub block_size_q: usize,
    pub block_size_kv: usize,
    pub causal: bool,
    pub sm_scale: Option<f64>,
}

impl Default for FlashAttention2Config {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            causal: true,
            sm_scale: None,
        }
    }
}

/// Output of FlashAttention-2 forward pass containing the attended output tensor
/// and the per-token log-sum-exp statistics for exact autograd backprop.
#[derive(Debug, Clone)]
pub struct FlashAttention2ForwardOutput {
    pub output: Tensor,
    pub logsumexp: Tensor,
}

/// Computes the FlashAttention-2 forward pass.
///
/// Input shapes:
/// - `q`: `[batch, heads, seq_len_q, head_dim]`
/// - `k`: `[batch, heads, seq_len_kv, head_dim]`
/// - `v`: `[batch, heads, seq_len_kv, head_dim]`
pub fn flash_attention_2_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    config: &FlashAttention2Config,
) -> Result<FlashAttention2ForwardOutput> {
    if q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 {
        return Err(Error::Msg(
            "FlashAttention-2 expects 4D tensors [batch, heads, seq_len, head_dim]".into(),
        ));
    }

    let (b, h, l_q, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let l_kv = k.shape()[2];
    let bq = config.block_size_q.max(1);
    let bkv = config.block_size_kv.max(1);
    let scale = config.sm_scale.unwrap_or(1.0 / (d as f64).sqrt());
    let offset = l_kv.saturating_sub(l_q);

    let mut q_outputs = Vec::new();
    let mut q_lse = Vec::new();
    let mut q_start = 0usize;

    while q_start < l_q {
        let q_len = (l_q - q_start).min(bq);
        let q_tile = q.narrow(2, q_start, q_len)?;

        let mut m_running: Option<Tensor> = None;
        let mut l_running: Option<Tensor> = None;
        let mut o_running: Option<Tensor> = None;

        let mut kv_start = 0usize;
        while kv_start < l_kv {
            let kv_len = (l_kv - kv_start).min(bkv);

            // Causal skip
            let max_valid_kv = offset + q_start + q_len - 1;
            if config.causal && kv_start > max_valid_kv {
                break;
            }

            let k_tile = k.narrow(2, kv_start, kv_len)?;
            let v_tile = v.narrow(2, kv_start, kv_len)?;

            let mut s_ij = q_tile.matmul(&k_tile.t()?)?.mul_scalar(scale)?;

            // Apply causal mask if block overlaps the diagonal
            if config.causal && (kv_start + kv_len - 1 > offset + q_start) {
                let mut mask = vec![false; q_len * kv_len];
                for qi in 0..q_len {
                    let abs_q = offset + q_start + qi;
                    for kj in 0..kv_len {
                        let abs_k = kv_start + kj;
                        if abs_k > abs_q {
                            mask[qi * kv_len + kj] = true;
                        }
                    }
                }
                let mask_t = Tensor::from_slice_on(&mask, &[q_len, kv_len], &s_ij.device())?;
                s_ij = s_ij.masked_fill(&mask_t, -1e9)?;
            }

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
            let pv_tile = p_tile.matmul(&v_tile)?;

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

            kv_start += bkv;
        }

        let (o_final, lse_final) = match (o_running, l_running, m_running) {
            (Some(o), Some(l), Some(m)) => {
                let o_norm = o.div(&l.add_scalar(1e-8)?)?;
                // LSE = m + ln(l)
                let lse = m.add(&l.ln()?)?;
                (o_norm, lse)
            }
            _ => (
                Tensor::zeros(&[b, h, q_len, d], q.dtype(), &q.device())?,
                Tensor::zeros(&[b, h, q_len, 1], q.dtype(), &q.device())?,
            ),
        };

        q_outputs.push(o_final);
        q_lse.push(lse_final);
        q_start += bq;
    }

    let output = Tensor::cat(&q_outputs, 2)?;
    let logsumexp = Tensor::cat(&q_lse, 2)?;
    Ok(FlashAttention2ForwardOutput { output, logsumexp })
}

/// Computes the exact backward pass for FlashAttention-2.
///
/// Returns `(grad_q, grad_k, grad_v)`.
pub fn flash_attention_2_backward(
    grad_output: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    output: &Tensor,
    logsumexp: &Tensor,
    config: &FlashAttention2Config,
) -> Result<(Tensor, Tensor, Tensor)> {
    // The forward pass checks this; the backward indexed shapes directly (panic).
    if q.ndim() != 4 || k.ndim() != 4 {
        return Err(Error::Msg(
            "FlashAttention-2 backward expects 4D q/k tensors [batch, heads, seq_len, head_dim]"
                .into(),
        ));
    }
    let (b, h, l_q, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let l_kv = k.shape()[2];
    let bq = config.block_size_q.max(1);
    let bkv = config.block_size_kv.max(1);
    let scale = config.sm_scale.unwrap_or(1.0 / (d as f64).sqrt());
    let offset = l_kv.saturating_sub(l_q);

    // Di = sum(grad_output * output, axis=-1, keepdims=true)
    let d_term = grad_output.mul(output)?.sum_axis(-1, true)?;

    // 1. Compute dQ by iterating over Q blocks (outer) and accumulating over KV blocks (inner)
    let mut grad_q_tiles = Vec::new();
    let mut q_start = 0usize;
    while q_start < l_q {
        let q_len = (l_q - q_start).min(bq);
        let q_tile = q.narrow(2, q_start, q_len)?;
        let do_tile = grad_output.narrow(2, q_start, q_len)?;
        let lse_tile = logsumexp.narrow(2, q_start, q_len)?;
        let d_tile = d_term.narrow(2, q_start, q_len)?;

        let mut dq_tile = Tensor::zeros(&[b, h, q_len, d], q.dtype(), &q.device())?;

        let mut kv_start = 0usize;
        while kv_start < l_kv {
            let kv_len = (l_kv - kv_start).min(bkv);

            // Causal check: skip if entire KV block is in future
            let max_valid_kv = offset + q_start + q_len - 1;
            if config.causal && kv_start > max_valid_kv {
                break;
            }

            let k_tile = k.narrow(2, kv_start, kv_len)?;
            let v_tile = v.narrow(2, kv_start, kv_len)?;

            let mut s_ij = q_tile.matmul(&k_tile.t()?)?.mul_scalar(scale)?;

            if config.causal && (kv_start + kv_len - 1 > offset + q_start) {
                let mut mask = vec![false; q_len * kv_len];
                for qi in 0..q_len {
                    let abs_q = offset + q_start + qi;
                    for kj in 0..kv_len {
                        let abs_k = kv_start + kj;
                        if abs_k > abs_q {
                            mask[qi * kv_len + kj] = true;
                        }
                    }
                }
                let mask_t = Tensor::from_slice_on(&mask, &[q_len, kv_len], &s_ij.device())?;
                s_ij = s_ij.masked_fill(&mask_t, -1e9)?;
            }

            // P_ij = exp(S_ij - L_i)
            let p_ij = s_ij.sub(&lse_tile)?.exp()?;
            // dP_ij = dO_i * V_j^T
            let dp_ij = do_tile.matmul(&v_tile.t()?)?;
            // dS_ij = P_ij * (dP_ij - D_i)
            let ds_ij = p_ij.mul(&dp_ij.sub(&d_tile)?)?;

            // dQ_i += scale * dS_ij * K_j
            let dq_contrib = ds_ij.matmul(&k_tile)?.mul_scalar(scale)?;
            dq_tile = dq_tile.add(&dq_contrib)?;

            kv_start += bkv;
        }

        grad_q_tiles.push(dq_tile);
        q_start += bq;
    }
    let grad_q = Tensor::cat(&grad_q_tiles, 2)?;

    // 2. Compute dK and dV by iterating over KV blocks (outer) and accumulating over Q blocks (inner)
    let mut grad_k_tiles = Vec::new();
    let mut grad_v_tiles = Vec::new();
    let mut kv_start = 0usize;

    while kv_start < l_kv {
        let kv_len = (l_kv - kv_start).min(bkv);
        let k_tile = k.narrow(2, kv_start, kv_len)?;
        let v_tile = v.narrow(2, kv_start, kv_len)?;

        let mut dk_tile = Tensor::zeros(&[b, h, kv_len, d], k.dtype(), &k.device())?;
        let mut dv_tile = Tensor::zeros(&[b, h, kv_len, d], v.dtype(), &v.device())?;

        let mut q_start = 0usize;
        while q_start < l_q {
            let q_len = (l_q - q_start).min(bq);

            // Causal skip: if entire Q block is before this KV block
            let min_valid_q = kv_start.saturating_sub(offset);
            if config.causal && (q_start + q_len <= min_valid_q) {
                q_start += bq;
                continue;
            }

            let q_tile = q.narrow(2, q_start, q_len)?;
            let do_tile = grad_output.narrow(2, q_start, q_len)?;
            let lse_tile = logsumexp.narrow(2, q_start, q_len)?;
            let d_tile = d_term.narrow(2, q_start, q_len)?;

            let mut s_ij = q_tile.matmul(&k_tile.t()?)?.mul_scalar(scale)?;

            if config.causal && (kv_start + kv_len - 1 > offset + q_start) {
                let mut mask = vec![false; q_len * kv_len];
                for qi in 0..q_len {
                    let abs_q = offset + q_start + qi;
                    for kj in 0..kv_len {
                        let abs_k = kv_start + kj;
                        if abs_k > abs_q {
                            mask[qi * kv_len + kj] = true;
                        }
                    }
                }
                let mask_t = Tensor::from_slice_on(&mask, &[q_len, kv_len], &s_ij.device())?;
                s_ij = s_ij.masked_fill(&mask_t, -1e9)?;
            }

            let p_ij = s_ij.sub(&lse_tile)?.exp()?;

            // dV_j += P_ij^T * dO_i
            let dv_contrib = p_ij.t()?.matmul(&do_tile)?;
            dv_tile = dv_tile.add(&dv_contrib)?;

            let dp_ij = do_tile.matmul(&v_tile.t()?)?;
            let ds_ij = p_ij.mul(&dp_ij.sub(&d_tile)?)?;

            // dK_j += scale * dS_ij^T * Q_i
            let dk_contrib = ds_ij.t()?.matmul(&q_tile)?.mul_scalar(scale)?;
            dk_tile = dk_tile.add(&dk_contrib)?;

            q_start += bq;
        }

        grad_k_tiles.push(dk_tile);
        grad_v_tiles.push(dv_tile);
        kv_start += bkv;
    }

    let grad_k = Tensor::cat(&grad_k_tiles, 2)?;
    let grad_v = Tensor::cat(&grad_v_tiles, 2)?;

    Ok((grad_q, grad_k, grad_v))
}

/// Convenience high-level FlashAttention-2 call.
pub fn flash_attention_2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    sm_scale: Option<f64>,
) -> Result<Tensor> {
    let config = FlashAttention2Config {
        block_size_q: 64,
        block_size_kv: 64,
        causal,
        sm_scale,
    };
    let out = flash_attention_2_forward(q, k, v, &config)?;
    Ok(out.output)
}
