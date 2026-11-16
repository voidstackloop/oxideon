//! Vision-Language-Action (VLA) Foundation Model Architecture (OpenVLA / RT-2 / Pi0).
//!
//! Features:
//! - [`GatedCrossAttention`]: Flamingo / IDEFICS style cross-attention with zero-initialized
//!   learnable $\tanh(\alpha)$ gating for stable integration into language backbones.
//! - [`VisionLanguageActionModel`]: End-to-end multimodal embodied agent architecture predicting
//!   both autoregressive text tokens and multi-step continuous robotic action trajectories ($A \in \mathbb{R}^{B \times H \times D_{action}}$).

use super::{Linear, Module, PatchEmbedding, RMSNorm, TransformerEncoder};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Learnable Gated Cross-Attention layer for multimodal conditioning.
///
/// Computes:
/// $$Y = X + \tanh(\alpha) \cdot \operatorname{CrossAttn}(X, Z_{vis})$$
/// Initialized with $\alpha = 0$ so the layer acts as an exact identity at start.
pub struct GatedCrossAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub norm: RMSNorm,
    pub alpha: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl GatedCrossAttention {
    pub fn new(dim: usize, num_heads: usize, context_dim: usize) -> Result<Self> {
        if num_heads == 0 || dim % num_heads != 0 {
            return Err(Error::Msg(format!(
                "dim {dim} must be divisible by a positive num_heads {num_heads}"
            )));
        }
        let head_dim = dim / num_heads;
        let q_proj = Linear::new(dim, dim)?;
        let k_proj = Linear::new(context_dim, dim)?;
        let v_proj = Linear::new(context_dim, dim)?;
        let out_proj = Linear::new(dim, dim)?;
        let norm = RMSNorm::new(dim, 1e-5)?;
        let alpha = Tensor::zeros(&[1], crate::dtype::DType::F32, &crate::device::Device::Cpu)?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            norm,
            alpha,
            num_heads,
            head_dim,
        })
    }

    /// Forward pass: `x` is `[batch, seq_len, dim]`, `context` is `[batch, ctx_len, context_dim]`.
    pub fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 || context.ndim() != 3 {
            return Err(Error::Msg(format!(
                "GatedCrossAttention expects x [B, S, D] and context [B, C, D], got {:?} and {:?}",
                x.shape(),
                context.shape()
            )));
        }
        let x_norm = self.norm.forward(x)?;
        let (b, seq_len, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let ctx_len = context.shape()[1];

        let q = self.q_proj.forward(&x_norm)?;
        let k = self.k_proj.forward(context)?;
        let v = self.v_proj.forward(context)?;

        let q_heads = q
            .reshape(&[b, seq_len, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let k_heads = k
            .reshape(&[b, ctx_len, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;
        let v_heads = v
            .reshape(&[b, ctx_len, self.num_heads, self.head_dim])?
            .permute(&[0, 2, 1, 3])?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q_heads.matmul(&k_heads.t()?)?.mul_scalar(scale)?;
        let attn = scores.softmax(-1)?;
        let out_heads = attn.matmul(&v_heads)?;

        let out_merged = out_heads
            .permute(&[0, 2, 1, 3])?
            .reshape(&[b, seq_len, d])?;
        let projected = self.out_proj.forward(&out_merged)?;

        // Residual gated by tanh(alpha)
        let gate = self.alpha.tanh()?;
        let gated_out = projected.mul(&gate)?;
        x.add(&gated_out)
    }
}

impl Module for GatedCrossAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward(input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.out_proj.parameters());
        p.extend(self.norm.parameters());
        p.push(self.alpha.clone());
        p
    }
}

/// Vision-Language-Action (VLA) foundation model architecture.
pub struct VisionLanguageActionModel {
    pub patch_embed: PatchEmbedding,
    pub vision_encoder: TransformerEncoder,
    pub cross_attention: GatedCrossAttention,
    pub text_embed: Linear,
    pub text_head: Linear,
    pub action_head: Linear,
    pub action_horizon: usize,
    pub action_dim: usize,
}

impl VisionLanguageActionModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        vision_dim: usize,
        text_dim: usize,
        vocab_size: usize,
        action_horizon: usize,
        action_dim: usize,
    ) -> Result<Self> {
        let patch_embed = PatchEmbedding::new(img_size, patch_size, in_channels, vision_dim)?;
        let vision_encoder = TransformerEncoder::new(2, vision_dim, 4, vision_dim * 2, 0.0)?;
        let cross_attention = GatedCrossAttention::new(text_dim, 4, vision_dim)?;
        let text_embed = Linear::new(vocab_size, text_dim)?;
        let text_head = Linear::new(text_dim, vocab_size)?;
        let action_flat = action_horizon.checked_mul(action_dim).ok_or_else(|| {
            Error::Msg(format!(
                "VLA action_head needs action_horizon ({action_horizon}) * action_dim ({action_dim}): overflows"
            ))
        })?;
        let action_head = Linear::new(text_dim, action_flat)?;

        Ok(Self {
            patch_embed,
            vision_encoder,
            cross_attention,
            text_embed,
            text_head,
            action_head,
            action_horizon,
            action_dim,
        })
    }

    /// Evaluates multimodal image + text inputs and outputs `(text_logits, continuous_actions)`.
    ///
    /// Inputs:
    /// - `images`: `[batch, channels, height, width]`
    /// - `text_one_hot`: `[batch, seq_len, vocab_size]`
    ///
    /// Returns:
    /// - `text_logits`: `[batch, seq_len, vocab_size]`
    /// - `actions`: `[batch, action_horizon, action_dim]`
    pub fn forward(&self, images: &Tensor, text_one_hot: &Tensor) -> Result<(Tensor, Tensor)> {
        if images.ndim() < 1 || text_one_hot.ndim() != 3 {
            return Err(Error::Msg(format!(
                "VLA forward expects images [B, ...] and text_one_hot [B, S, V], got {:?} and {:?}",
                images.shape(),
                text_one_hot.shape()
            )));
        }
        if text_one_hot.shape()[1] == 0 {
            return Err(Error::Msg(
                "VLA forward requires non-empty text sequence (seq_len > 0)".into(),
            ));
        }
        let b = images.shape()[0];

        // 1. Vision visual tokens: [batch, num_patches, vision_dim]
        let patches = self.patch_embed.forward(images)?;
        let visual_features = self.vision_encoder.forward(&patches)?;

        // 2. Language tokens: [batch, seq_len, text_dim]
        let text_tokens = self.text_embed.forward(text_one_hot)?;

        // 3. Multimodal cross-attention conditioning
        let fused_tokens = self
            .cross_attention
            .forward(&text_tokens, &visual_features)?;

        // 4. Autoregressive text predictions
        let text_logits = self.text_head.forward(&fused_tokens)?;

        // 5. Continuous robotic action predictions from final token representation
        let last_token = fused_tokens
            .narrow(1, fused_tokens.shape()[1] - 1, 1)?
            .squeeze_axis(1)?;
        let raw_action = self.action_head.forward(&last_token)?;
        let actions = raw_action.reshape(&[b, self.action_horizon, self.action_dim])?;

        Ok((text_logits, actions))
    }
}

impl Module for VisionLanguageActionModel {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(Error::Msg(
            "VisionLanguageActionModel requires multimodal forward(&images, &text_one_hot)".into(),
        ))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.patch_embed.parameters();
        p.extend(self.vision_encoder.parameters());
        p.extend(self.cross_attention.parameters());
        p.extend(self.text_embed.parameters());
        p.extend(self.text_head.parameters());
        p.extend(self.action_head.parameters());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_cross_attention_rejects_non_3d_input() -> Result<()> {
        let gca = GatedCrossAttention::new(8, 2, 8)?;
        let x = Tensor::zeros_f32(&[2, 8])?;
        let ctx = Tensor::zeros_f32(&[2, 3, 8])?;
        let err = gca
            .forward(&x, &ctx)
            .expect_err("rank-2 x must be an error, not a panic");
        assert!(err.to_string().contains("[B, S, D]"), "{err}");
        Ok(())
    }

    #[test]
    fn vla_rejects_overflowing_action_flat() {
        let err = match VisionLanguageActionModel::new(8, 4, 3, 8, 8, 8, usize::MAX, 2) {
            Ok(_) => panic!("overflowing horizon*dim must be an error, not a panic"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("action"), "{err}");
    }
}
