//! Complete Frontier Foundation Model Architectures:
//! 1. **[`DeepSeekMoELM`] (DeepSeek-V3 / R1 Architecture)**: Combines [`MultiHeadLatentAttention`] (`MLA`),
//!    shared + top-$k$ routed [`SharedRoutedMoE`] (DeepSeekMoE), and **Multi-Token Prediction (`MTP`)**
//!    heads (`forward_mtp`) that jointly predict future tokens $t+1, \dots, t+D$.
//! 2. **[`HybridJambaLM`] (Jamba / Nemotron-H Hybrid SSM-Attention-MoE)**: Interleaves [`MambaBlock`]
//!    Selective State Space layers with [`GroupedQueryAttention`] (`GQA`) layers and alternates
//!    dense [`SwiGLU`] with [`SparseMoE`] experts.
//! 3. **[`MultimodalVLM`] (Vision-Language Foundation Model — LLaVA / Qwen-VL / PaliGemma style)**:
//!    Fuses a ViT visual patch encoder (`PatchEmbedding` + [`TransformerEncoder`]), a 2-layer
//!    cross-modal MLP connector, and a RoPE + SwiGLU causal language model backbone for joint
//!    image + text understanding and multimodal autoregressive generation.
//! 4. **[`DiT`] & [`DiTBlock`] (Diffusion Transformer — Peebles & Xie 2023 / Flux / SD3)**:
//!    Patchified latent Diffusion / Rectified-Flow Transformer with **Adaptive LayerNorm-Zero (`AdaLN-Zero`)**
//!    conditioning $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2)$ on continuous timesteps
//!    and class/context embeddings.
//! 5. **[`RewardModel`] & [`bradley_terry_reward_loss`]**: Sequence-level reward model for RLHF
//!    and [`grpo_loss`](crate::nn::loss::grpo_loss) reasoning post-training.

use super::{
    Embedding, GELU, GroupedQueryAttention, LayerNorm, Linear, MambaBlock, Module,
    MultiHeadAttention, MultiHeadLatentAttention, PatchEmbedding, RMSNorm, RotaryEmbedding,
    SparseMoE, SwiGLU, TransformerEncoder, loss,
};
use crate::diffusion::timestep_embedding;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// ===========================================================================
// 1. DeepSeek-V3 / R1 Architecture: Shared+Routed MoE, MLA & Multi-Token Prediction (MTP)
// ===========================================================================

/// DeepSeekMoE Feed-Forward layer combining `num_shared_experts` always-active [`SwiGLU`]
/// experts (capturing common knowledge) with a top-$k$ routed [`SparseMoE`] expert bank
/// (capturing specialized knowledge).
pub struct SharedRoutedMoE {
    pub shared_experts: Vec<SwiGLU>,
    pub routed_moe: SparseMoE,
}

impl SharedRoutedMoE {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        num_shared_experts: usize,
        num_routed_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        crate::limits::check_repeats("num_shared_experts", num_shared_experts)?;
        let mut shared_experts = Vec::with_capacity(num_shared_experts);
        for _ in 0..num_shared_experts {
            shared_experts.push(SwiGLU::new(d_model, d_ff)?);
        }
        let routed_moe = SparseMoE::new(d_model, d_ff, num_routed_experts, top_k)?;
        Ok(Self {
            shared_experts,
            routed_moe,
        })
    }

    /// Returns `(output, aux_load_balance_loss)`.
    pub fn forward_with_aux(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mut out, aux_loss) = self.routed_moe.forward_with_aux_loss(x)?;
        for s_exp in &self.shared_experts {
            out = out.add(&s_exp.forward(x)?)?;
        }
        Ok((out, aux_loss))
    }
}

impl Module for SharedRoutedMoE {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_aux(input)?.0)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, s) in self.shared_experts.iter().enumerate() {
            for (k, v) in s.named_parameters() {
                out.push((format!("shared_experts.{i}.{k}"), v));
            }
        }
        for (k, v) in self.routed_moe.named_parameters() {
            out.push((format!("routed_moe.{k}"), v));
        }
        out
    }
}

/// A single decoder block of [`DeepSeekMoELM`]: pre-RMSNorm [`MultiHeadLatentAttention`] (`MLA`)
/// + pre-RMSNorm [`SharedRoutedMoE`].
pub struct DeepSeekBlock {
    pub attn_norm: RMSNorm,
    pub mla: MultiHeadLatentAttention,
    pub ffn_norm: RMSNorm,
    pub moe: SharedRoutedMoE,
}

impl DeepSeekBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: usize,
        d_ff: usize,
        num_shared_experts: usize,
        num_routed_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        Ok(Self {
            attn_norm: RMSNorm::new(d_model, 1e-5)?,
            mla: MultiHeadLatentAttention::new(d_model, num_heads, kv_lora_rank, q_lora_rank)?,
            ffn_norm: RMSNorm::new(d_model, 1e-5)?,
            moe: SharedRoutedMoE::new(
                d_model,
                d_ff,
                num_shared_experts,
                num_routed_experts,
                top_k,
            )?,
        })
    }

    pub fn forward_block(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = x.add(&self.mla.forward(&self.attn_norm.forward(x)?)?)?;
        let (moe_out, aux) = self.moe.forward_with_aux(&self.ffn_norm.forward(&h)?)?;
        Ok((h.add(&moe_out)?, aux))
    }
}

impl Module for DeepSeekBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_block(input)?.0)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.attn_norm.named_parameters() {
            out.push((format!("attn_norm.{k}"), v));
        }
        for (k, v) in self.mla.named_parameters() {
            out.push((format!("mla.{k}"), v));
        }
        for (k, v) in self.ffn_norm.named_parameters() {
            out.push((format!("ffn_norm.{k}"), v));
        }
        for (k, v) in self.moe.named_parameters() {
            out.push((format!("moe.{k}"), v));
        }
        out
    }
}

/// **Multi-Token Prediction (`MTP`) Module** (DeepSeek-V3):
/// Combines the previous depth's hidden representation $h_t^{(k-1)}$ and the embedded token
/// $e(x_{t+k})$ via RMSNorm + a `[2 * d_model -> d_model]` projection and a lightweight
/// Transformer refinement block to predict token $x_{t+k+1}$.
pub struct MultiTokenPredictionModule {
    pub norm_h: RMSNorm,
    pub norm_emb: RMSNorm,
    pub combine_proj: Linear,
    pub attn: MultiHeadLatentAttention,
    pub ffn_norm: RMSNorm,
    pub ffn: SwiGLU,
}

impl MultiTokenPredictionModule {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: usize,
        d_ff: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm_h: RMSNorm::new(d_model, 1e-5)?,
            norm_emb: RMSNorm::new(d_model, 1e-5)?,
            combine_proj: Linear::with_bias(2 * d_model, d_model, false)?,
            attn: MultiHeadLatentAttention::new(d_model, num_heads, kv_lora_rank, q_lora_rank)?,
            ffn_norm: RMSNorm::new(d_model, 1e-5)?,
            ffn: SwiGLU::new(d_model, d_ff)?,
        })
    }

    pub fn forward_mtp(&self, prev_h: &Tensor, next_tok_emb: &Tensor) -> Result<Tensor> {
        let h_n = self.norm_h.forward(prev_h)?;
        let e_n = self.norm_emb.forward(next_tok_emb)?;
        let cat = Tensor::cat(&[h_n, e_n], -1)?;
        let z = self.combine_proj.forward(&cat)?;
        let z = z.add(&self.attn.forward(&z)?)?;
        z.add(&self.ffn.forward(&self.ffn_norm.forward(&z)?)?)
    }
}

impl Module for MultiTokenPredictionModule {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_mtp(input, input)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("norm_h", &self.norm_h as &dyn Module),
            ("norm_emb", &self.norm_emb),
            ("combine_proj", &self.combine_proj),
            ("attn", &self.attn),
            ("ffn_norm", &self.ffn_norm),
            ("ffn", &self.ffn),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

/// **DeepSeek-V3 / R1 Frontier Language Model (`DeepSeekMoELM`)**:
/// Full foundation model combining:
/// - [`MultiHeadLatentAttention`] (`MLA`) low-rank KV compression in every layer
/// - [`SharedRoutedMoE`] (shared + routed SwiGLU experts)
/// - `num_mtp_heads` [`MultiTokenPredictionModule`] heads sharing `tok_emb` and `lm_head`
pub struct DeepSeekMoELM {
    pub tok_emb: Embedding,
    pub blocks: Vec<DeepSeekBlock>,
    pub mtp_modules: Vec<MultiTokenPredictionModule>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl DeepSeekMoELM {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: usize,
        d_ff: usize,
        num_shared_experts: usize,
        num_routed_experts: usize,
        top_k: usize,
        num_mtp_heads: usize,
    ) -> Result<Self> {
        let tok_emb = Embedding::new(vocab_size, d_model)?;
        crate::limits::check_repeats("num_layers", num_layers)?;
        crate::limits::check_repeats("num_mtp_heads", num_mtp_heads)?;
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(DeepSeekBlock::new(
                d_model,
                num_heads,
                kv_lora_rank,
                q_lora_rank,
                d_ff,
                num_shared_experts,
                num_routed_experts,
                top_k,
            )?);
        }
        let mut mtp_modules = Vec::with_capacity(num_mtp_heads);
        for _ in 0..num_mtp_heads {
            mtp_modules.push(MultiTokenPredictionModule::new(
                d_model,
                num_heads,
                kv_lora_rank,
                q_lora_rank,
                d_ff,
            )?);
        }
        let final_norm = RMSNorm::new(d_model, 1e-5)?;
        let lm_head = Linear::with_bias(d_model, vocab_size, false)?;
        Ok(Self {
            tok_emb,
            blocks,
            mtp_modules,
            final_norm,
            lm_head,
            vocab_size,
            d_model,
        })
    }

    /// Runs the main model trunk plus all `num_mtp_heads` Multi-Token Prediction (`MTP`) heads
    /// on `tokens` (`[B, L]`), returning `(main_logits, mtp_logits_per_depth, total_moe_aux_loss)`.
    /// Depth $k$ ($k = 1, \dots, D$) produces logits of shape `[B, L - k, vocab_size]`.
    pub fn forward_with_mtp(&self, tokens: &Tensor) -> Result<(Tensor, Vec<Tensor>, Tensor)> {
        if tokens.ndim() != 2 {
            return Err(Error::Msg(format!(
                "DeepSeekMoELM expects 2D tokens [B, L], got {:?}",
                tokens.shape()
            )));
        }
        let l = tokens.shape()[1];
        let emb = self.tok_emb.forward(tokens)?;
        let mut h = emb.clone();
        let mut total_aux: Option<Tensor> = None;

        for blk in &self.blocks {
            let (next_h, aux) = blk.forward_block(&h)?;
            h = next_h;
            total_aux = Some(match total_aux {
                Some(acc) => acc.add(&aux)?,
                None => aux,
            });
        }

        let main_logits = self.lm_head.forward(&self.final_norm.forward(&h)?)?;
        let mut mtp_logits = Vec::with_capacity(self.mtp_modules.len());
        let mut prev_h = h;

        for (depth_idx, mtp) in self.mtp_modules.iter().enumerate() {
            let k = depth_idx + 1;
            if l <= k {
                break;
            }
            let valid_len = l - k;
            let h_slice = prev_h.narrow(1, 0, valid_len)?;
            let next_emb_slice = emb.narrow(1, k, valid_len)?;
            let h_mtp = mtp.forward_mtp(&h_slice, &next_emb_slice)?;
            let logits_k = self.lm_head.forward(&self.final_norm.forward(&h_mtp)?)?;
            mtp_logits.push(logits_k);
            prev_h = h_mtp;
        }

        let aux_out = match total_aux {
            Some(a) => a,
            None => Tensor::from_slice(&[0.0f32], &[1])?,
        };
        Ok((main_logits, mtp_logits, aux_out))
    }
}

impl Module for DeepSeekMoELM {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(self.forward_with_mtp(tokens)?.0)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.tok_emb.named_parameters() {
            out.push((format!("tok_emb.{k}"), v));
        }
        for (i, blk) in self.blocks.iter().enumerate() {
            for (k, v) in blk.named_parameters() {
                out.push((format!("blocks.{i}.{k}"), v));
            }
        }
        for (i, mtp) in self.mtp_modules.iter().enumerate() {
            for (k, v) in mtp.named_parameters() {
                out.push((format!("mtp_modules.{i}.{k}"), v));
            }
        }
        for (k, v) in self.final_norm.named_parameters() {
            out.push((format!("final_norm.{k}"), v));
        }
        for (k, v) in self.lm_head.named_parameters() {
            out.push((format!("lm_head.{k}"), v));
        }
        out
    }
}

// ===========================================================================
// 2. Hybrid Mamba-Attention-MoE Foundation Model (`HybridJambaLM` — Jamba / Nemotron-H)
// ===========================================================================

pub enum HybridSequenceMixer {
    Mamba(MambaBlock),
    Attention {
        norm: RMSNorm,
        gqa: GroupedQueryAttention,
    },
}

pub enum HybridChannelMixer {
    Dense { norm: RMSNorm, ffn: SwiGLU },
    MoE { norm: RMSNorm, moe: SparseMoE },
}

/// **Hybrid Mamba-2 + GQA + SparseMoE Foundation Model (`HybridJambaLM`)**:
/// Interleaves linear-complexity [`MambaBlock`] state-space layers with periodic
/// [`GroupedQueryAttention`] (`GQA`) layers (every `attn_every_n` layers) and alternates
/// dense [`SwiGLU`] with [`SparseMoE`] (every `moe_every_n` layers).
pub struct HybridJambaLM {
    pub tok_emb: Embedding,
    pub layers: Vec<(HybridSequenceMixer, HybridChannelMixer)>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
}

impl HybridJambaLM {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        d_state: usize,
        d_ff: usize,
        num_experts: usize,
        top_k: usize,
        attn_every_n: usize,
        moe_every_n: usize,
    ) -> Result<Self> {
        let tok_emb = Embedding::new(vocab_size, d_model)?;
        let attn_period = attn_every_n.max(1);
        let moe_period = moe_every_n.max(1);
        crate::limits::check_repeats("num_layers", num_layers)?;
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let seq_mixer = if (i + 1) % attn_period == 0 {
                HybridSequenceMixer::Attention {
                    norm: RMSNorm::new(d_model, 1e-5)?,
                    gqa: GroupedQueryAttention::new(d_model, num_heads, num_kv_heads)?,
                }
            } else {
                HybridSequenceMixer::Mamba(MambaBlock::new(d_model, 2, d_state)?)
            };

            let chan_mixer = if (i + 1) % moe_period == 0 {
                HybridChannelMixer::MoE {
                    norm: RMSNorm::new(d_model, 1e-5)?,
                    moe: SparseMoE::new(d_model, d_ff, num_experts, top_k)?,
                }
            } else {
                HybridChannelMixer::Dense {
                    norm: RMSNorm::new(d_model, 1e-5)?,
                    ffn: SwiGLU::new(d_model, d_ff)?,
                }
            };
            layers.push((seq_mixer, chan_mixer));
        }

        Ok(Self {
            tok_emb,
            layers,
            final_norm: RMSNorm::new(d_model, 1e-5)?,
            lm_head: Linear::with_bias(d_model, vocab_size, false)?,
        })
    }
}

impl Module for HybridJambaLM {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut h = self.tok_emb.forward(tokens)?;
        for (seq_m, chan_m) in &self.layers {
            h = match seq_m {
                HybridSequenceMixer::Mamba(m) => m.forward(&h)?,
                HybridSequenceMixer::Attention { norm, gqa } => {
                    h.add(&gqa.forward(&norm.forward(&h)?)?)?
                }
            };
            h = match chan_m {
                HybridChannelMixer::Dense { norm, ffn } => {
                    h.add(&ffn.forward(&norm.forward(&h)?)?)?
                }
                HybridChannelMixer::MoE { norm, moe } => {
                    h.add(&moe.forward(&norm.forward(&h)?)?)?
                }
            };
        }
        self.lm_head.forward(&self.final_norm.forward(&h)?)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.tok_emb.named_parameters() {
            out.push((format!("tok_emb.{k}"), v));
        }
        for (i, (seq_m, chan_m)) in self.layers.iter().enumerate() {
            match seq_m {
                HybridSequenceMixer::Mamba(m) => {
                    for (k, v) in m.named_parameters() {
                        out.push((format!("layers.{i}.mamba.{k}"), v));
                    }
                }
                HybridSequenceMixer::Attention { norm, gqa } => {
                    for (k, v) in norm.named_parameters() {
                        out.push((format!("layers.{i}.attn_norm.{k}"), v));
                    }
                    for (k, v) in gqa.named_parameters() {
                        out.push((format!("layers.{i}.gqa.{k}"), v));
                    }
                }
            }
            match chan_m {
                HybridChannelMixer::Dense { norm, ffn } => {
                    for (k, v) in norm.named_parameters() {
                        out.push((format!("layers.{i}.ffn_norm.{k}"), v));
                    }
                    for (k, v) in ffn.named_parameters() {
                        out.push((format!("layers.{i}.ffn.{k}"), v));
                    }
                }
                HybridChannelMixer::MoE { norm, moe } => {
                    for (k, v) in norm.named_parameters() {
                        out.push((format!("layers.{i}.moe_norm.{k}"), v));
                    }
                    for (k, v) in moe.named_parameters() {
                        out.push((format!("layers.{i}.moe.{k}"), v));
                    }
                }
            }
        }
        for (k, v) in self.final_norm.named_parameters() {
            out.push((format!("final_norm.{k}"), v));
        }
        for (k, v) in self.lm_head.named_parameters() {
            out.push((format!("lm_head.{k}"), v));
        }
        out
    }
}

// ===========================================================================
// 3. Multimodal Vision-Language Foundation Model (`MultimodalVLM` — LLaVA / Qwen-VL / PaliGemma)
// ===========================================================================

/// **Multimodal Vision-Language Foundation Model (`MultimodalVLM`)**:
/// - **Vision Tower**: [`PatchEmbedding`] + learned 2D positional embedding + [`TransformerEncoder`]
///   producing `[B, num_patches, d_vision]` visual tokens.
/// - **Cross-Modal Connector**: 2-layer MLP (`Linear -> GELU -> Linear`) projecting visual tokens
///   into the language model's embedding space `[B, num_patches, d_llm]`.
/// - **Language Model Backbone**: Stack of RoPE + [`GroupedQueryAttention`] + [`SwiGLU`] causal
///   decoder layers operating over the concatenated `[visual_tokens, text_tokens]` sequence!
pub struct MultimodalVLM {
    pub patch_embed: PatchEmbedding,
    pub vision_encoder: TransformerEncoder,
    pub projector_fc1: Linear,
    pub projector_fc2: Linear,
    pub tok_emb: Embedding,
    pub rope: RotaryEmbedding,
    pub llm_layers: Vec<(RMSNorm, GroupedQueryAttention, RMSNorm, SwiGLU)>,
    pub final_norm: RMSNorm,
    pub lm_head: Linear,
    pub num_vis_tokens: usize,
    pub d_llm: usize,
}

impl MultimodalVLM {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        d_vision: usize,
        vis_layers: usize,
        vis_heads: usize,
        vocab_size: usize,
        d_llm: usize,
        llm_layers_count: usize,
        llm_heads: usize,
        llm_kv_heads: usize,
        llm_ff: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        crate::limits::check_repeats("llm_layers", llm_layers_count)?;
        super::transformer::check_heads(d_llm, llm_heads)?;
        let too_big = || Error::Msg("MultimodalVLM sizes overflow".to_string());
        let patch_embed = PatchEmbedding::new(img_size, patch_size, in_channels, d_vision)?;
        let num_vis_tokens = patch_embed.num_patches.checked_add(1).ok_or_else(too_big)?;
        let rope_len = max_seq_len
            .checked_add(num_vis_tokens)
            .ok_or_else(too_big)?;
        let d_ff = d_vision.checked_mul(2).ok_or_else(too_big)?;
        let vision_encoder = TransformerEncoder::new(vis_layers, d_vision, vis_heads, d_ff, 0.0)?;

        let projector_fc1 = Linear::new(d_vision, d_llm)?;
        let projector_fc2 = Linear::new(d_llm, d_llm)?;
        let tok_emb = Embedding::new(vocab_size, d_llm)?;
        let rope = RotaryEmbedding::new(d_llm / llm_heads, rope_len, 10000.0)?;

        let mut llm_layers = Vec::with_capacity(llm_layers_count);
        for _ in 0..llm_layers_count {
            llm_layers.push((
                RMSNorm::new(d_llm, 1e-5)?,
                GroupedQueryAttention::new(d_llm, llm_heads, llm_kv_heads)?,
                RMSNorm::new(d_llm, 1e-5)?,
                SwiGLU::new(d_llm, llm_ff)?,
            ));
        }

        Ok(Self {
            patch_embed,
            vision_encoder,
            projector_fc1,
            projector_fc2,
            tok_emb,
            rope,
            llm_layers,
            final_norm: RMSNorm::new(d_llm, 1e-5)?,
            lm_head: Linear::with_bias(d_llm, vocab_size, false)?,
            num_vis_tokens,
            d_llm,
        })
    }

    /// Encodes `images` (`[B, C, H, W]`) into `[B, num_vis_tokens, d_llm]` projected visual tokens.
    pub fn encode_images(&self, images: &Tensor) -> Result<Tensor> {
        let patches = self.patch_embed.forward(images)?;
        let vis_feat = self.vision_encoder.forward(&patches)?;
        let proj = self.projector_fc1.forward(&vis_feat)?.gelu()?;
        self.projector_fc2.forward(&proj)
    }

    /// Jointly processes `images` (`[B, C, H, W]`) and `text_tokens` (`[B, L_text]`), returning
    /// causal language modeling logits of shape `[B, L_text, vocab_size]` conditioned on the image!
    pub fn forward_multimodal(&self, images: &Tensor, text_tokens: &Tensor) -> Result<Tensor> {
        if text_tokens.ndim() != 2 {
            return Err(Error::Msg(format!(
                "MultimodalVLM expects 2D text_tokens [B, L], got {:?}",
                text_tokens.shape()
            )));
        }
        let l_txt = text_tokens.shape()[1];
        let vis_tokens = self.encode_images(images)?;
        let txt_tokens = self.tok_emb.forward(text_tokens)?;
        let mut h = Tensor::cat(&[vis_tokens, txt_tokens], 1)?;

        for (anorm, gqa, fnorm, ffn) in &self.llm_layers {
            let attn_out = gqa.forward_gqa(&anorm.forward(&h)?, Some(&self.rope), None, true)?;
            h = h.add(&attn_out)?;
            h = h.add(&ffn.forward(&fnorm.forward(&h)?)?)?;
        }

        let h_txt = h.narrow(1, self.num_vis_tokens, l_txt)?;
        self.lm_head.forward(&self.final_norm.forward(&h_txt)?)
    }

    /// Autoregressively generates `max_new_tokens` text tokens conditioned on `image` (`[1, C, H, W]`)
    /// and an initial `prompt` token sequence.
    pub fn generate_multimodal(
        &self,
        image: &Tensor,
        prompt: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(Error::Msg(
                "generate_multimodal: prompt cannot be empty".into(),
            ));
        }
        if let Some(&bad) = prompt.iter().find(|&&t| t > i64::MAX as usize) {
            return Err(Error::Msg(format!(
                "generate_multimodal: prompt token id {bad} exceeds i64::MAX"
            )));
        }
        crate::tensor::no_grad(|| {
            let mut seq: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
            for _ in 0..max_new_tokens {
                let inp = Tensor::from_slice(&seq, &[1, seq.len()])?;
                let logits = self.forward_multimodal(image, &inp)?;
                let next_tok = logits
                    .select(0, 0)?
                    .select(0, (seq.len() - 1) as isize)?
                    .argmax_axis(0, false)?
                    .item::<i64>()?;
                seq.push(next_tok);
            }
            Ok(seq.into_iter().map(|t| t as usize).collect())
        })
    }
}

impl Module for MultimodalVLM {
    fn forward(&self, text_tokens: &Tensor) -> Result<Tensor> {
        let mut h = self.tok_emb.forward(text_tokens)?;
        for (anorm, gqa, fnorm, ffn) in &self.llm_layers {
            let attn_out = gqa.forward_gqa(&anorm.forward(&h)?, Some(&self.rope), None, true)?;
            h = h.add(&attn_out)?;
            h = h.add(&ffn.forward(&fnorm.forward(&h)?)?)?;
        }
        self.lm_head.forward(&self.final_norm.forward(&h)?)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("patch_embed", &self.patch_embed as &dyn Module),
            ("vision_encoder", &self.vision_encoder),
            ("projector_fc1", &self.projector_fc1),
            ("projector_fc2", &self.projector_fc2),
            ("tok_emb", &self.tok_emb),
            ("final_norm", &self.final_norm),
            ("lm_head", &self.lm_head),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        for (i, (an, gqa, fnm, ffn)) in self.llm_layers.iter().enumerate() {
            for (k, v) in an.named_parameters() {
                out.push((format!("llm_layers.{i}.attn_norm.{k}"), v));
            }
            for (k, v) in gqa.named_parameters() {
                out.push((format!("llm_layers.{i}.gqa.{k}"), v));
            }
            for (k, v) in fnm.named_parameters() {
                out.push((format!("llm_layers.{i}.ffn_norm.{k}"), v));
            }
            for (k, v) in ffn.named_parameters() {
                out.push((format!("llm_layers.{i}.ffn.{k}"), v));
            }
        }
        out
    }
}

// ===========================================================================
// 4. Diffusion / Rectified-Flow Transformer (`DiT` with `AdaLN-Zero` — Flux / Sora / SD3)
// ===========================================================================

/// **Diffusion Transformer Block (`DiTBlock`)** with **Adaptive LayerNorm-Zero (`AdaLN-Zero`)**
/// conditioning (Peebles & Xie, ICCV 2023):
/// Regresses 6 modulation parameters $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2)$
/// from the conditioning vector `cond` (`[B, d_model]`) to scale, shift, and gate both the
/// self-attention and MLP sub-blocks!
pub struct DiTBlock {
    pub norm1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub norm2: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
    pub adaln_modulation: Linear,
    pub d_model: usize,
}

impl DiTBlock {
    pub fn new(d_model: usize, num_heads: usize, mlp_ratio: usize) -> Result<Self> {
        let d_ff = d_model.checked_mul(mlp_ratio.max(1)).ok_or_else(|| {
            Error::Msg(format!("DiTBlock: d_ff {d_model} * {mlp_ratio} overflows"))
        })?;
        Ok(Self {
            norm1: LayerNorm::new(d_model, 1e-6)?,
            attn: MultiHeadAttention::new(d_model, num_heads)?,
            norm2: LayerNorm::new(d_model, 1e-6)?,
            mlp_fc1: Linear::new(d_model, d_ff)?,
            mlp_fc2: Linear::new(d_ff, d_model)?,
            adaln_modulation: Linear::new(d_model, 6 * d_model)?,
            d_model,
        })
    }

    /// Forward pass conditioned on `cond` (`[B, d_model]`, e.g. timestep + class embedding).
    pub fn forward_conditioned(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "DiTBlock expects 3D x [B, L, D], got {:?}",
                x.shape()
            )));
        }
        let (b, l, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let mod_params = self.adaln_modulation.forward(&cond.silu()?)?; // [B, 6*D]
        let chunks = mod_params.chunk(6, -1)?;
        // chunk() yields fewer pieces when the width is short (tampered adaln);
        // indexing [0..6] would panic.
        if chunks.len() != 6 {
            return Err(Error::Msg(format!(
                "DiTBlock AdaLN needs 6 modulation chunks, got {}",
                chunks.len()
            )));
        }
        let gamma1 = chunks[0].unsqueeze(1)?.broadcast_to(&[b, l, d])?;
        let beta1 = chunks[1].unsqueeze(1)?.broadcast_to(&[b, l, d])?;
        let alpha1 = chunks[2].unsqueeze(1)?.broadcast_to(&[b, l, d])?;
        let gamma2 = chunks[3].unsqueeze(1)?.broadcast_to(&[b, l, d])?;
        let beta2 = chunks[4].unsqueeze(1)?.broadcast_to(&[b, l, d])?;
        let alpha2 = chunks[5].unsqueeze(1)?.broadcast_to(&[b, l, d])?;

        // Modulated Self-Attention: x + alpha1 * Attn(norm1(x) * (1 + gamma1) + beta1)
        let h1 = self
            .norm1
            .forward(x)?
            .mul(&gamma1.add_scalar(1.0)?)?
            .add(&beta1)?;
        let attn_out = self.attn.forward(&h1)?;
        let x = x.add(&alpha1.mul(&attn_out)?)?;

        // Modulated MLP: x + alpha2 * MLP(norm2(x) * (1 + gamma2) + beta2)
        let h2 = self
            .norm2
            .forward(&x)?
            .mul(&gamma2.add_scalar(1.0)?)?
            .add(&beta2)?;
        let mlp_out = self.mlp_fc2.forward(&self.mlp_fc1.forward(&h2)?.gelu()?)?;
        x.add(&alpha2.mul(&mlp_out)?)
    }
}

impl Module for DiTBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let cond = input.mean_axis(1, false)?;
        self.forward_conditioned(input, &cond)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("norm1", &self.norm1 as &dyn Module),
            ("attn", &self.attn),
            ("norm2", &self.norm2),
            ("mlp_fc1", &self.mlp_fc1),
            ("mlp_fc2", &self.mlp_fc2),
            ("adaln_modulation", &self.adaln_modulation),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

/// **Scalable Diffusion / Flow-Matching Transformer (`DiT`)**:
/// Patchifies `[B, C, H, W]` noisy latents, adds 2D positional embeddings, conditions every
/// [`DiTBlock`] on sinusoidal timestep embeddings + class/label embeddings via `AdaLN-Zero`,
/// and unpatchifies back to `[B, C, H, W]` predicted noise $\epsilon_\theta(x_t, t, y)$ or
/// velocity field $v_\theta(x_t, t, y)$.
pub struct DiT {
    pub patch_embed: PatchEmbedding,
    pub pos_embed: Tensor,
    pub time_mlp1: Linear,
    pub time_mlp2: Linear,
    pub class_emb: Embedding,
    pub blocks: Vec<DiTBlock>,
    pub final_norm: LayerNorm,
    pub final_adaln: Linear,
    pub final_proj: Linear,
    pub img_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub d_model: usize,
}

impl DiT {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        d_model: usize,
        depth: usize,
        num_heads: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let patch_embed = PatchEmbedding::new(img_size, patch_size, in_channels, d_model)?;
        let num_patches = patch_embed.num_patches;
        let pos_embed = Tensor::randn(&[1, num_patches, d_model])?
            .mul_scalar(0.02)?
            .detach()
            .requires_grad_(true);

        let time_mlp1 = Linear::new(d_model, d_model)?;
        let time_mlp2 = Linear::new(d_model, d_model)?;
        let class_emb = Embedding::new(num_classes.max(1), d_model)?;

        crate::limits::check_repeats("depth", depth)?;
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(DiTBlock::new(d_model, num_heads, 4)?);
        }

        let final_norm = LayerNorm::new(d_model, 1e-6)?;
        let final_adaln = Linear::new(d_model, 2 * d_model)?;
        let patch_area = in_channels * patch_size * patch_size;
        let final_proj = Linear::new(d_model, patch_area)?;

        Ok(Self {
            patch_embed,
            pos_embed,
            time_mlp1,
            time_mlp2,
            class_emb,
            blocks,
            final_norm,
            final_adaln,
            final_proj,
            img_size,
            patch_size,
            in_channels,
            d_model,
        })
    }

    /// Predicts noise $\epsilon_\theta(x_t, t, y)$ or velocity $v_\theta(x_t, t, y)$ of shape `[B, C, H, W]`.
    pub fn forward_dit(
        &self,
        x_t: &Tensor,
        timesteps: &[usize],
        class_labels: Option<&Tensor>,
    ) -> Result<Tensor> {
        if self.patch_size == 0 {
            return Err(Error::Msg("DiT patch_size must be >= 1".into()));
        }
        let num_p = self.patch_embed.num_patches;
        let d = self.d_model;

        let t_sin = timestep_embedding(timesteps, d)?.to_device(&x_t.device())?;
        let mut cond = self
            .time_mlp2
            .forward(&self.time_mlp1.forward(&t_sin)?.silu()?)?;
        if let Some(y) = class_labels {
            cond = cond.add(&self.class_emb.forward(y)?)?;
        }

        let patches = self.patch_embed.forward(x_t)?.narrow(1, 1, num_p)?;
        // Batch from the validated patches (same rows as the input): reading x_t's
        // shape first panicked on rank-0 input before patch_embed could reject it.
        let b = patches.shape()[0];
        let pos = self.pos_embed.broadcast_to(&[b, num_p, d])?;
        let mut h = patches.add(&pos)?;

        for blk in &self.blocks {
            h = blk.forward_conditioned(&h, &cond)?;
        }

        let fin_mod = self.final_adaln.forward(&cond.silu()?)?;
        let fin_chunks = fin_mod.chunk(2, -1)?;
        if fin_chunks.len() != 2 {
            return Err(Error::Msg(format!(
                "DiT final AdaLN needs 2 modulation chunks, got {}",
                fin_chunks.len()
            )));
        }
        let gamma = fin_chunks[0].unsqueeze(1)?.broadcast_to(&[b, num_p, d])?;
        let beta = fin_chunks[1].unsqueeze(1)?.broadcast_to(&[b, num_p, d])?;
        let h_norm = self
            .final_norm
            .forward(&h)?
            .mul(&gamma.add_scalar(1.0)?)?
            .add(&beta)?;

        // Unpatchify: [B, gh * gw, C * p * p] -> [B, C, gh * p, gw * p]
        let patch_out = self.final_proj.forward(&h_norm)?;
        let grid = self.img_size / self.patch_size;
        let p = self.patch_size;
        let c = self.in_channels;
        patch_out
            .reshape(&[b, grid, grid, c, p, p])?
            .permute(&[0, 3, 1, 4, 2, 5])?
            .contiguous()?
            .reshape(&[b, c, self.img_size, self.img_size])
    }
}

impl Module for DiT {
    fn forward(&self, x_t: &Tensor) -> Result<Tensor> {
        let b = x_t
            .shape()
            .first()
            .copied()
            .ok_or_else(|| Error::Msg("DiT expects rank >= 1 input".into()))?;
        let ts = vec![50usize; b];
        let _ = GELU;
        self.forward_dit(x_t, &ts, None)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![("pos_embed".into(), self.pos_embed.clone())];
        for (p, m) in [
            ("patch_embed", &self.patch_embed as &dyn Module),
            ("time_mlp1", &self.time_mlp1),
            ("time_mlp2", &self.time_mlp2),
            ("class_emb", &self.class_emb),
            ("final_norm", &self.final_norm),
            ("final_adaln", &self.final_adaln),
            ("final_proj", &self.final_proj),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        for (i, blk) in self.blocks.iter().enumerate() {
            for (k, v) in blk.named_parameters() {
                out.push((format!("blocks.{i}.{k}"), v));
            }
        }
        out
    }
}

// ===========================================================================
// 5. RLHF / Reasoning Reward Model (`RewardModel` & Bradley-Terry Preference Loss)
// ===========================================================================

/// Sequence-level **Reward Model (`RewardModel`)** for RLHF / GRPO post-training:
/// Maps token sequences `[B, L]` through an embedding + Transformer backbone and projects the
/// last token's hidden state to a scalar reward score `[B]`.
pub struct RewardModel {
    pub tok_emb: Embedding,
    pub encoder: TransformerEncoder,
    pub norm: RMSNorm,
    pub score_head: Linear,
}

impl RewardModel {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        d_ff: usize,
    ) -> Result<Self> {
        Ok(Self {
            tok_emb: Embedding::new(vocab_size, d_model)?,
            encoder: TransformerEncoder::new(num_layers, d_model, num_heads, d_ff, 0.0)?,
            norm: RMSNorm::new(d_model, 1e-5)?,
            score_head: Linear::with_bias(d_model, 1, false)?,
        })
    }
}

impl Module for RewardModel {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        if tokens.ndim() != 2 {
            return Err(Error::Msg(format!(
                "RewardModel expects 2D token tensor [B, L], got {:?}",
                tokens.shape()
            )));
        }
        let b = tokens.shape()[0];
        let l = tokens.shape()[1];
        if l == 0 {
            return Err(Error::Msg(
                "RewardModel requires non-empty sequence (L > 0)".into(),
            ));
        }
        let h = self.tok_emb.forward(tokens)?;
        let enc = self.encoder.forward(&h)?;
        let last_h = enc.select(1, (l - 1) as isize)?;
        self.score_head
            .forward(&self.norm.forward(&last_h)?)?
            .reshape(&[b])
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("tok_emb", &self.tok_emb as &dyn Module),
            ("encoder", &self.encoder),
            ("norm", &self.norm),
            ("score_head", &self.score_head),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

/// **Bradley-Terry Pairwise Preference Reward Loss** (Ouyang et al. 2022 / InstructGPT):
/// $$\mathcal{L}_{\text{RM}} = -\mathbb{E}\!\left[\log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l))\right]$$
pub fn bradley_terry_reward_loss(
    chosen_rewards: &Tensor,
    rejected_rewards: &Tensor,
) -> Result<Tensor> {
    let _ = loss::mse_loss;
    chosen_rewards
        .sub(rejected_rewards)?
        .neg()?
        .softplus()?
        .mean()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_rejects_rank1_tokens() -> Result<()> {
        let lm = DeepSeekMoELM::new(8, 8, 1, 2, 4, 4, 16, 1, 4, 2, 1)?;
        let tok = Tensor::from_slice(&[1i64, 2, 3, 4], &[4])?;
        assert!(lm.forward_with_mtp(&tok).is_err());
        Ok(())
    }

    #[test]
    fn vlm_rejects_rank1_text() -> Result<()> {
        let vlm = MultimodalVLM::new(8, 4, 3, 8, 1, 2, 8, 8, 1, 2, 2, 16, 16)?;
        let img = Tensor::zeros_f32(&[1, 3, 8, 8])?;
        let text = Tensor::from_slice(&[1i64, 2], &[2])?;
        assert!(vlm.forward_multimodal(&img, &text).is_err());
        Ok(())
    }

    #[test]
    fn dit_block_rejects_overflowing_ff() {
        let err = match DiTBlock::new(1 << 63, 2, 2) {
            Ok(_) => panic!("overflowing d_ff must be an error, not a panic"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("d_ff"), "{err}");
    }

    #[test]
    fn dit_block_conditioned_validates_shapes() -> Result<()> {
        let blk = DiTBlock::new(8, 2, 2)?;
        let x2 = Tensor::zeros_f32(&[2, 8])?;
        let cond = Tensor::zeros_f32(&[1, 8])?;
        assert!(blk.forward_conditioned(&x2, &cond).is_err());
        let mut bad = DiTBlock::new(8, 2, 2)?;
        bad.adaln_modulation = Linear::with_bias(8, 4, false)?;
        let x = Tensor::zeros_f32(&[1, 2, 8])?;
        let err = bad
            .forward_conditioned(&x, &cond)
            .expect_err("short adaln must be an error, not a panic");
        assert!(err.to_string().contains("AdaLN"), "{err}");
        Ok(())
    }

    #[test]
    fn dit_forward_validates_shapes() -> Result<()> {
        let dit = DiT::new(8, 4, 2, 8, 1, 2, 4)?;
        let x0 = Tensor::from_slice(&[1.0f32], &[])?;
        assert!(Module::forward(&dit, &x0).is_err());
        let mut r0 = DiT::new(8, 4, 2, 8, 1, 2, 4)?;
        r0.patch_size = 0;
        let x = Tensor::zeros_f32(&[1, 2, 8, 8])?;
        assert!(r0.forward_dit(&x, &[0], None).is_err());
        let mut bad = DiT::new(8, 4, 2, 8, 1, 2, 4)?;
        bad.final_adaln = Linear::with_bias(8, 1, false)?;
        assert!(bad.forward_dit(&x, &[0], None).is_err());
        Ok(())
    }

    #[test]
    fn reward_rejects_empty_sequence() -> Result<()> {
        let rm = RewardModel::new(8, 8, 1, 2, 16)?;
        let tok = Tensor::from_slice(&([] as [i64; 0]), &[2, 0])?;
        let err = Module::forward(&rm, &tok).expect_err("empty seq must be an error, not a panic");
        assert!(err.to_string().contains("L"), "{err}");
        Ok(())
    }
}
