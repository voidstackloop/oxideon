//! Eagle: Feature-Level Speculative Decoding (Li et al., 2024/2026).
//!
//! Unlike token-level speculation (which suffers from exponential distribution divergence at depth > 2),
//! Eagle extrapolates future tokens in the high-level **feature space** of the target LLM's top transformer layer.
//!
//! Recurrent draft heads pass predicted continuous features $h_{t+k}$ into the next speculation step,
//! achieving significantly higher token acceptance rates (typically 2.5x - 4x speedup).

use crate::error::Result;
use crate::nn::{Embedding, Linear, Module};
use crate::tensor::Tensor;

/// Configuration for Eagle Speculative Engine.
#[derive(Debug, Clone)]
pub struct EagleConfig {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub draft_depth: usize,
    pub temperature: f32,
}

impl EagleConfig {
    pub fn new(hidden_dim: usize, vocab_size: usize, draft_depth: usize) -> Self {
        Self {
            hidden_dim,
            vocab_size,
            draft_depth: draft_depth.max(1),
            temperature: 0.0,
        }
    }
}

/// A lightweight feature-level draft recurrence head.
pub struct EagleDraftHead {
    pub fusion_linear: Linear,
    pub feature_proj: Linear,
    pub lm_head: Linear,
}

impl EagleDraftHead {
    pub fn new(hidden_dim: usize, vocab_size: usize) -> Result<Self> {
        let Some(fused_dim) = hidden_dim.checked_mul(2) else {
            return Err(crate::error::Error::Msg(
                "Eagle: hidden_dim overflows".into(),
            ));
        };
        let fusion_linear = Linear::with_bias(fused_dim, hidden_dim, true)?;
        let feature_proj = Linear::with_bias(hidden_dim, hidden_dim, true)?;
        let lm_head = Linear::with_bias(hidden_dim, vocab_size, false)?;

        Ok(Self {
            fusion_linear,
            feature_proj,
            lm_head,
        })
    }

    /// Extrapolates next hidden feature and next token logits:
    /// `h_next, logits = DraftHead(h_curr, embedding(token_curr))`
    pub fn forward_step(&self, h_curr: &Tensor, token_emb: &Tensor) -> Result<(Tensor, Tensor)> {
        // Concatenate along hidden dimension: [B, 1, 2*D]
        let fused_in = Tensor::cat(&[h_curr.clone(), token_emb.clone()], 2)?;
        let fused_h = self.fusion_linear.forward(&fused_in)?.relu()?;
        let h_next = self.feature_proj.forward(&fused_h)?;
        let logits = self.lm_head.forward(&h_next)?;

        Ok((h_next, logits))
    }
}

/// Eagle Feature-Level Speculative Decoding Engine.
pub struct EagleSpeculativeEngine {
    pub config: EagleConfig,
    pub embedding: Embedding,
    pub draft_head: EagleDraftHead,
}

impl EagleSpeculativeEngine {
    pub fn new(config: EagleConfig) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim)?;
        let draft_head = EagleDraftHead::new(config.hidden_dim, config.vocab_size)?;

        Ok(Self {
            config,
            embedding,
            draft_head,
        })
    }

    /// Generates $K$ draft speculative tokens from base target model's top hidden state.
    ///
    /// - `top_hidden`: shape `[1, 1, hidden_dim]`
    /// - `last_token`: token id of current verified token
    ///
    /// Returns: `(draft_tokens, draft_features)`
    pub fn generate_draft(
        &self,
        top_hidden: &Tensor,
        last_token: usize,
    ) -> Result<(Vec<usize>, Vec<Tensor>)> {
        // `draft_depth` is a pub field: re-cap it here (tampered values aborted
        // the reserves below and hung the draft loop).
        crate::limits::check_repeats("draft_depth", self.config.draft_depth)?;
        let mut draft_tokens = Vec::with_capacity(self.config.draft_depth);
        let mut draft_features = Vec::with_capacity(self.config.draft_depth);

        let mut curr_h = top_hidden.clone();
        let mut curr_token = last_token;

        for _ in 0..self.config.draft_depth {
            // `as i64` would wrap ids >= 2^63 to negative, which embedding lookup
            // silently wraps again to a valid row (wrong draft, no error).
            let tok_i64 = i64::try_from(curr_token).map_err(|_| {
                crate::error::Error::Msg(format!(
                    "Eagle generate_draft: token id {curr_token} exceeds i64 range"
                ))
            })?;
            let tok_tensor = Tensor::from_slice(&[tok_i64], &[1, 1])?;
            let tok_emb = self.embedding.forward(&tok_tensor)?;

            let (h_next, logits) = self.draft_head.forward_step(&curr_h, &tok_emb)?;

            // Greedy argmax token selection
            let logits_vec = logits.to_vec::<f32>()?;
            let mut best_tok = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (idx, &v) in logits_vec.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    best_tok = idx;
                }
            }

            draft_tokens.push(best_tok);
            draft_features.push(h_next.clone());

            curr_h = h_next;
            curr_token = best_tok;
        }

        Ok((draft_tokens, draft_features))
    }

    /// Verifies draft tokens against target LLM verifier logits.
    /// Returns `(accepted_tokens, num_accepted)`.
    pub fn verify_and_accept(
        draft_tokens: &[usize],
        target_predicted_tokens: &[usize],
    ) -> (Vec<usize>, usize) {
        let mut accepted = Vec::new();
        for (i, &draft_tok) in draft_tokens.iter().enumerate() {
            if i < target_predicted_tokens.len() && draft_tok == target_predicted_tokens[i] {
                accepted.push(draft_tok);
            } else {
                break;
            }
        }

        // If all draft tokens matched, accept target model's bonus next token
        let n_accepted = accepted.len();
        if n_accepted < target_predicted_tokens.len() {
            accepted.push(target_predicted_tokens[n_accepted]);
        }

        (accepted, n_accepted)
    }
}
