//! Sparse Mixture-of-Experts (`SparseMoE`, Shazeer et al. / Switch Transformer / Mixtral).
//!
//! Routes each token to the top-$k$ out of $E$ [`SwiGLU`] experts and computes the
//! differentiable Switch-Transformer load-balancing auxiliary loss
//! $\mathcal{L}_{\text{aux}} = E \sum_{e=1}^E f_e \cdot \bar{P}_e$.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module, SwiGLU};
use crate::tensor::Tensor;

pub struct SparseMoE {
    pub router: Linear,
    pub experts: Vec<SwiGLU>,
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

impl SparseMoE {
    pub fn new(
        d_model: usize,
        ffn_hidden: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        if num_experts == 0 || top_k == 0 || top_k > num_experts || d_model == 0 {
            return Err(Error::Msg(format!(
                "SparseMoE: invalid d_model={d_model}, num_experts={num_experts}, top_k={top_k}"
            )));
        }
        let router = Linear::with_bias(d_model, num_experts, false)?;
        crate::limits::check_repeats("num_experts", num_experts)?;
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(SwiGLU::new(d_model, ffn_hidden)?);
        }
        Ok(Self {
            router,
            experts,
            d_model,
            num_experts,
            top_k,
        })
    }

    /// Runs the Sparse MoE forward pass and returns `(output, load_balancing_aux_loss)`.
    pub fn forward_with_aux_loss(&self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        // Config fields are pub: a zeroed d_model divided by zero below, a
        // zeroed top_k indexed an empty topk vec, and a tampered num_experts
        // aborted the counts reserve (`new()` checks construction only).
        if self.d_model == 0 || self.top_k == 0 || self.top_k > self.num_experts {
            return Err(Error::Msg(format!(
                "SparseMoE: invalid d_model={}, num_experts={}, top_k={}",
                self.d_model, self.num_experts, self.top_k
            )));
        }
        if self.experts.len() != self.num_experts {
            return Err(Error::Msg(format!(
                "SparseMoE: {} experts vs num_experts {}",
                self.experts.len(),
                self.num_experts
            )));
        }
        let orig_shape = input.shape().to_vec();
        let d = *orig_shape
            .last()
            .ok_or_else(|| Error::Msg("SparseMoE expects at least 2D input".into()))?;
        if d != self.d_model {
            return Err(Error::ShapeMismatch {
                op: "SparseMoE",
                expected: vec![self.d_model],
                got: orig_shape,
            });
        }
        let n_tokens = input.numel() / d;
        let x_flat = input.reshape(&[n_tokens, d])?;

        // Router probabilities: [N, E]
        let router_logits = self.router.forward(&x_flat)?;
        let router_probs = router_logits.softmax(-1)?;

        // Top-k expert weights and indices per token: [N, K]
        let (topk_vals, topk_idx) = router_probs.topk(self.top_k, -1, true, true)?;
        let denom = topk_vals.sum_axis(-1, true)?.clamp(1e-12, f64::INFINITY)?;
        let topk_weights = topk_vals.div(&denom)?;

        // Compute Switch-Transformer load-balancing auxiliary loss:
        // f_e = fraction of (token, top-1) assignments to expert e, P_e = mean router probability for expert e
        let idx_vec = topk_idx.to_vec::<i64>()?;
        let mut counts = vec![0.0f64; self.num_experts];
        for tok in 0..n_tokens {
            let e = idx_vec[tok * self.top_k] as usize;
            if e < self.num_experts {
                counts[e] += 1.0;
            }
        }
        let inv_n = 1.0 / (n_tokens.max(1) as f64);
        for c in &mut counts {
            *c *= inv_n;
        }
        let f_e = Tensor::from_slice_on(&counts, &[self.num_experts], &input.device())?
            .to_dtype(input.dtype())?;
        let p_e = router_probs.mean_axis(0, false)?;
        let aux_loss = f_e.mul(&p_e)?.sum()?.mul_scalar(self.num_experts as f64)?;

        // Build per-expert gating matrix G: [N, E] using scatter_add of normalized topk_weights
        let zeros_ne = Tensor::zeros(
            &[n_tokens, self.num_experts],
            input.dtype(),
            &input.device(),
        )?;
        let gates = zeros_ne.scatter_add(1, &topk_idx, &topk_weights)?;

        // Combine expert outputs weighted by their gate columns
        let mut combined = Tensor::zeros(&[n_tokens, d], input.dtype(), &input.device())?;
        for (e, expert) in self.experts.iter().enumerate() {
            // Skip expert computation if no token selected this expert at all
            let selected = idx_vec.iter().any(|&id| (id as usize) == e);
            if !selected {
                continue;
            }
            let gate_e = gates.narrow(1, e, 1)?; // [N, 1]
            let y_e = expert.forward(&x_flat)?; // [N, D]
            combined = combined.add(&y_e.mul(&gate_e)?)?;
        }

        Ok((combined.reshape(&orig_shape)?, aux_loss))
    }
}

impl Module for SparseMoE {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (out, _aux) = self.forward_with_aux_loss(input)?;
        Ok(out)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.router.named_parameters() {
            out.push((format!("router.{k}"), v));
        }
        for (i, exp) in self.experts.iter().enumerate() {
            for (k, v) in exp.named_parameters() {
                out.push((format!("experts.{i}.{k}"), v));
            }
        }
        out
    }
}
