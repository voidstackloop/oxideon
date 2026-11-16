//! Mixture-of-Depths (`MoD`, Raposo et al., Google DeepMind 2024).
//!
//! Allocates compute dynamically across tokens: each transformer layer routes only the top-$K$
//! tokens ($K = \lfloor C \cdot T \rfloor$, capacity $C \in (0, 1]$) through heavy compute,
//! while remaining tokens bypass the layer via residual connections, saving up to 50% FLOPs.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Mixture-of-Depths dynamic compute router.
pub struct MixtureOfDepths {
    pub router: Linear,
    pub compute_block: Linear,
    pub capacity: f64,
}

impl MixtureOfDepths {
    pub fn new(embed_dim: usize, capacity: f64) -> Result<Self> {
        // NaN fails both range comparisons, then silently routes a single
        // token (`(NaN * t).round() as usize` saturates to 0, clamped to 1).
        if !capacity.is_finite() || capacity <= 0.0 || capacity > 1.0 {
            return Err(Error::Msg(format!(
                "MixtureOfDepths capacity must be in (0, 1], got {capacity}"
            )));
        }
        let router = Linear::new(embed_dim, 1)?;
        let compute_block = Linear::new(embed_dim, embed_dim)?;

        Ok(Self {
            router,
            compute_block,
            capacity,
        })
    }

    /// Evaluates dynamic token routing over 3D input `[batch, seq_len, embed_dim]`.
    ///
    /// Returns `(output_tensor, router_weights, num_routed_tokens)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, usize)> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "MixtureOfDepths requires 3D tensor [B, T, D], got {:?}",
                x.shape()
            )));
        }
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if t == 0 {
            return Err(Error::Msg("MixtureOfDepths requires seq_len >= 1".into()));
        }
        let k = ((self.capacity * t as f64).round() as usize).clamp(1, t);

        // Router scores stay in the autograd graph: only the discrete top-k
        // SELECTION reads plain values (no gradient through the choice itself),
        // while gather/modulate/scatter below are differentiable tensor ops, so
        // the router, the compute block, and the input all receive gradients.
        let x_flat = x.reshape(&[b * t, d])?;
        let router_logits = self.router.forward(&x_flat)?.sigmoid()?;
        let weights = router_logits.reshape(&[b, t, 1])?;
        let weights_vec = weights.to_vec::<f32>()?;

        let mut out_batches = Vec::with_capacity(b);
        for batch in 0..b {
            let xb = x.narrow(0, batch, 1)?.reshape(&[t, d])?;
            let wb = weights.narrow(0, batch, 1)?.reshape(&[t, 1])?;

            // Find top-k tokens based on router weights.
            let mut token_scores: Vec<(usize, f32)> = (0..t)
                .map(|tok_idx| (tok_idx, weights_vec[batch * t + tok_idx]))
                .collect();
            token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            token_scores.truncate(k);
            let topk_idx: Vec<i64> = token_scores
                .iter()
                .map(|(tok_idx, _)| *tok_idx as i64)
                .collect();
            let idx = Tensor::from_slice(&topk_idx, &[k])?;

            // Gather top-k tokens, run compute, modulate by router weights.
            let gathered = xb.index_select(0, &idx)?;
            let computed = self.compute_block.forward(&gathered)?;
            let w_topk = wb.index_select(0, &idx)?;
            let modulated = computed.mul(&w_topk)?;

            // Scatter back into original positions over the residual stream.
            let out_b = xb.index_add(0, &idx, &modulated)?;
            out_batches.push(out_b.unsqueeze(0)?);
        }

        let out = Tensor::cat(&out_batches, 0)?;

        Ok((out, weights, k))
    }
}
