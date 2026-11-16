//! Transfusion Unified Foundation Architecture (Zhou et al., Meta FAIR 2024).
//!
//! "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model"
//!
//! Key Properties:
//! - Unified representation for discrete text tokens and continuous visual patches.
//! - Simultaneous training on next-token prediction (LM) and diffusion trajectory prediction (Flow Matching / DDPM).
//! - Shared transformer trunk with modality-specific input projections and output heads.

use super::loss::{cross_entropy, mse_loss};
use super::transformer::TransformerEncoderLayer;
use super::{Embedding, Linear, Module, RMSNorm};
use crate::error::Result;
use crate::tensor::Tensor;

/// Transfusion Multimodal Unified Model configuration.
#[derive(Debug, Clone)]
pub struct TransfusionConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub patch_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub d_ff: usize,
    pub dropout: f32,
}

impl Default for TransfusionConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            d_model: 512,
            patch_dim: 64,
            num_layers: 4,
            num_heads: 8,
            d_ff: 2048,
            dropout: 0.0,
        }
    }
}

/// Transfusion Unified Multi-Modal Foundation Model.
pub struct TransfusionModel {
    pub config: TransfusionConfig,
    pub text_embed: Embedding,
    pub patch_proj: Linear,
    pub time_proj: Linear,
    pub layers: Vec<TransformerEncoderLayer>,
    pub norm: RMSNorm,
    pub lm_head: Linear,
    pub diff_head: Linear,
}

impl TransfusionModel {
    pub fn new(config: TransfusionConfig) -> Result<Self> {
        let text_embed = Embedding::new(config.vocab_size, config.d_model)?;
        let patch_proj = Linear::new(config.patch_dim, config.d_model)?;
        let time_proj = Linear::new(1, config.d_model)?;

        crate::limits::check_repeats("num_layers", config.num_layers)?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let layer = TransformerEncoderLayer::new(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout,
            )?
            .with_causal(false);
            layers.push(layer);
        }

        let norm = RMSNorm::new(config.d_model, 1e-5)?;
        let lm_head = Linear::new(config.d_model, config.vocab_size)?;
        let diff_head = Linear::new(config.d_model, config.patch_dim)?;

        Ok(Self {
            config,
            text_embed,
            patch_proj,
            time_proj,
            layers,
            norm,
            lm_head,
            diff_head,
        })
    }

    /// Forward pass processing multimodal sequences.
    ///
    /// Inputs:
    /// - `text_tokens`: `[batch, text_len]` token IDs
    /// - `visual_patches`: `[batch, patch_len, patch_dim]` continuous patch vectors
    /// - `timesteps`: `[batch, 1]` or `[batch]` diffusion timestep continuous values in `[0, 1]`
    ///
    /// Returns:
    /// - `(text_logits, patch_predictions)`: `([batch, text_len, vocab_size], [batch, patch_len, patch_dim])`
    pub fn forward_multimodal(
        &self,
        text_tokens: &Tensor,
        visual_patches: &Tensor,
        timesteps: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if text_tokens.ndim() != 2 || visual_patches.ndim() != 3 {
            return Err(crate::error::Error::Msg(format!(
                "TransfusionModel::forward_multimodal expects text [B, T] and patches [B, P, D], got {:?} and {:?}",
                text_tokens.shape(),
                visual_patches.shape()
            )));
        }
        let _batch_size = text_tokens.shape()[0];
        let text_len = text_tokens.shape()[1];
        let patch_len = visual_patches.shape()[1];

        // 1. Embed discrete text tokens: [B, T_text, d_model]
        let text_h = self.text_embed.forward(text_tokens)?;

        // 2. Project continuous visual patches: [B, T_patch, d_model]
        let patch_h = self.patch_proj.forward(visual_patches)?;

        // 3. Condition visual patches with diffusion timestep
        let t_2d = if timesteps.ndim() == 1 {
            timesteps.unsqueeze(1)?
        } else {
            timesteps.clone()
        };
        let t_2d = if t_2d.dtype() != self.time_proj.weight.dtype() {
            t_2d.to_dtype(self.time_proj.weight.dtype())?
        } else {
            t_2d
        };
        // [B, 1, d_model]
        let time_emb = self.time_proj.forward(&t_2d)?.unsqueeze(1)?;
        let patch_conditioned = patch_h.add(&time_emb)?;

        // 4. Concatenate unified sequence: [B, T_text + T_patch, d_model]
        let mut unified_h = Tensor::cat(&[text_h, patch_conditioned], 1)?;

        // 5. Pass through shared transformer backbone
        for layer in &self.layers {
            unified_h = layer.forward(&unified_h)?;
        }
        let unified_h = self.norm.forward(&unified_h)?;

        // 6. Split back into modality representations
        let text_out = unified_h.narrow(1, 0, text_len)?;
        let patch_out = unified_h.narrow(1, text_len, patch_len)?;

        // 7. Modality heads
        let text_logits = self.lm_head.forward(&text_out)?;
        let patch_pred = self.diff_head.forward(&patch_out)?;

        Ok((text_logits, patch_pred))
    }

    /// Computes joint Transfusion loss: L_total = L_LM + lambda_diff * L_diffusion.
    ///
    /// - `text_tokens`: `[batch, text_len]` token IDs
    /// - `visual_clean`: `[batch, patch_len, patch_dim]` ground-truth visual patches x_0
    /// - `visual_noise`: `[batch, patch_len, patch_dim]` Gaussian noise epsilon
    /// - `timesteps`: `[batch, 1]` timestep t in [0, 1]
    /// - `lambda_diff`: weight factor for the diffusion loss
    pub fn compute_loss(
        &self,
        text_tokens: &Tensor,
        visual_clean: &Tensor,
        visual_noise: &Tensor,
        timesteps: &Tensor,
        lambda_diff: f64,
    ) -> Result<Tensor> {
        // Flow matching linear interpolation: x_t = (1 - t) * x_0 + t * eps
        let t_expanded = if timesteps.ndim() == 1 {
            timesteps.unsqueeze(1)?.unsqueeze(2)?
        } else if timesteps.ndim() == 2 {
            timesteps.unsqueeze(2)?
        } else {
            timesteps.clone()
        };
        let t_expanded = if t_expanded.dtype() != visual_clean.dtype() {
            t_expanded.to_dtype(visual_clean.dtype())?
        } else {
            t_expanded
        };

        let one_minus_t =
            Tensor::ones(t_expanded.shape(), t_expanded.dtype(), &t_expanded.device())?
                .sub(&t_expanded)?;

        let x_t = visual_clean
            .mul(&one_minus_t)?
            .add(&visual_noise.mul(&t_expanded)?)?;

        // Target velocity: v = eps - x_0
        let target_v = visual_noise.sub(visual_clean)?;

        let (logits, pred_v) = self.forward_multimodal(text_tokens, &x_t, timesteps)?;

        // Diffusion MSE loss on predicted velocity
        let diff_loss = mse_loss(&pred_v, &target_v)?;

        // Next-token cross-entropy over the vocabulary (the documented L_LM):
        // logits[:, :-1, :] predict text_tokens[:, 1:]. (A mean(logits^2) term
        // here would be minimized by zero logits, i.e. maximal-entropy
        // predictions, and never consults the targets at all.)
        if text_tokens.ndim() != 2
            || !(text_tokens.dtype().is_int() || text_tokens.dtype().is_floating_point())
        {
            return Err(crate::error::Error::Msg(format!(
                "TransfusionModel::compute_loss expects numeric text_tokens [B, T], got {:?} {:?}",
                text_tokens.shape(),
                text_tokens.dtype()
            )));
        }
        let (b_tok, text_len) = (text_tokens.shape()[0], text_tokens.shape()[1]);
        if text_len < 2 {
            return Err(crate::error::Error::Msg(format!(
                "TransfusionModel::compute_loss requires text_len >= 2 for shifted LM loss, got {text_len}"
            )));
        }
        if logits.ndim() != 3 || logits.shape()[0] != b_tok || logits.shape()[1] != text_len {
            return Err(crate::error::Error::Msg(format!(
                "TransfusionModel::compute_loss expects logits [B, T, V] matching text_tokens [B, T], got {:?}",
                logits.shape()
            )));
        }
        let vocab = logits.shape()[2];
        let flat = b_tok.checked_mul(text_len - 1).ok_or_else(|| {
            crate::error::Error::Msg("TransfusionModel::compute_loss: batch x seq overflows".into())
        })?;
        let shift_logits = logits.narrow(1, 0, text_len - 1)?.reshape(&[flat, vocab])?;
        // Token IDs: ints exactly; floats only when every value is integral
        // (forward_multimodal/Embedding accepts integral floats, so the loss
        // must too — but fractional ids are rejected, never truncated).
        // Integrality covers ALL positions (position 0 feeds the embedding);
        // the shifted targets below additionally get range-checked.
        if text_tokens.dtype().is_floating_point() {
            for &v in &text_tokens.to_vec::<f64>()? {
                if !v.is_finite() || v.fract() != 0.0 {
                    return Err(crate::error::Error::Msg(format!(
                        "TransfusionModel::compute_loss: non-integral token id {v}"
                    )));
                }
            }
        }
        let tgt_shifted = text_tokens.narrow(1, 1, text_len - 1)?;
        let tgt_ids: Vec<i64> = if text_tokens.dtype().is_int() {
            tgt_shifted.to_vec::<i64>()?
        } else {
            tgt_shifted
                .to_vec::<f64>()?
                .iter()
                .map(|&v| v as i64)
                .collect()
        };
        let mut targets = Vec::with_capacity(tgt_ids.len());
        for &id in &tgt_ids {
            if id < 0 || id as u64 >= vocab as u64 {
                return Err(crate::error::Error::Msg(format!(
                    "TransfusionModel::compute_loss: token id {id} out of range for vocab {vocab}"
                )));
            }
            targets.push(id as usize);
        }
        let lm_loss = cross_entropy(&shift_logits, &targets)?;

        let total_loss = lm_loss.add(&diff_loss.mul_scalar(lambda_diff)?)?;
        Ok(total_loss)
    }
}

impl Module for TransfusionModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 2 {
            return Err(crate::error::Error::Msg(format!(
                "TransfusionModel expects 2D input [B, T], got {:?}",
                input.shape()
            )));
        }
        let dummy_patches = Tensor::zeros(
            &[input.shape()[0], 1, self.config.patch_dim],
            input.dtype(),
            &input.device(),
        )?;
        let dummy_t = Tensor::zeros(
            &[input.shape()[0], 1],
            crate::dtype::DType::F32,
            &input.device(),
        )?;
        let (logits, _) = self.forward_multimodal(input, &dummy_patches, &dummy_t)?;
        Ok(logits)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.text_embed.parameters();
        p.extend(self.patch_proj.parameters());
        p.extend(self.time_proj.parameters());
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.norm.parameters());
        p.extend(self.lm_head.parameters());
        p.extend(self.diff_head.parameters());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Result<TransfusionModel> {
        TransfusionModel::new(TransfusionConfig {
            vocab_size: 8,
            d_model: 8,
            patch_dim: 4,
            num_layers: 1,
            num_heads: 2,
            d_ff: 16,
            dropout: 0.0,
        })
    }

    #[test]
    fn multimodal_rejects_bad_ranks() -> Result<()> {
        let m = tiny()?;
        let text = Tensor::from_slice(&[1i64, 2], &[2])?;
        let patches = Tensor::zeros_f32(&[1, 2, 4])?;
        let t = Tensor::zeros_f32(&[1, 1])?;
        assert!(m.forward_multimodal(&text, &patches, &t).is_err());
        let text2 = Tensor::from_slice(&[1i64, 2], &[1, 2])?;
        let flat = Tensor::zeros_f32(&[2, 4])?;
        assert!(m.forward_multimodal(&text2, &flat, &t).is_err());
        Ok(())
    }

    #[test]
    fn module_forward_rejects_rank0() -> Result<()> {
        let m = tiny()?;
        let x = Tensor::from_slice(&[1.0f32], &[])?;
        assert!(Module::forward(&m, &x).is_err());
        Ok(())
    }
}
