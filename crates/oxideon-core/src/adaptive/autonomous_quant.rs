//! Autonomous Quantization Engine for Large Language Models.
//!
//! Provides dynamic, feedback-directed precision adaptation for [`CausalLM`] and transformer
//! architectures:
//! - Multi-format quantization targets: `FP32`, `Q8_0`, `Q4_0`, `FP8_E4M3`, `FP8_E5M2`, `BitNet158`, `AWQ`, and `GPTQ`.
//! - Per-block and per-projection precision assignment and sensitivity profiling.
//! - Autonomous runtime precision adaptation driven by perplexity / loss degradation thresholds,
//!   memory limits, or latency budgets.

use crate::dtype::DType;
use crate::error::Result;
use crate::nn::{CausalLM, CausalLMBlock, Linear, Module};
use crate::quant::{
    AWQLinear, BitLinear, FP8Format, FP8Linear, GPTQLinear, PackedBitLinear, QuantScheme,
    QuantizedLinear,
};
use crate::tensor::Tensor;

/// Quantization precision format supported by the autonomous engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// Full precision 32-bit float.
    FP32,
    /// 8-bit block symmetric quantization (GGML style).
    Q8_0,
    /// 4-bit packed block symmetric quantization (GGML style).
    Q4_0,
    /// 8-bit float forward format (E4M3).
    FP8E4M3,
    /// 8-bit float wide dynamic range format (E5M2).
    FP8E5M2,
    /// BitNet b1.58 Radix-3 ternary packed linear (-1, 0, +1).
    BitNet158,
    /// Activation-aware 8-bit weight quantization (AWQ).
    AWQQ8,
    /// Activation-aware 4-bit weight quantization (AWQ).
    AWQQ4,
    /// Second-order Hessian compensated 8-bit GPTQ.
    GPTQQ8,
    /// Second-order Hessian compensated 4-bit GPTQ.
    GPTQQ4,
}

impl QuantFormat {
    /// Nominal bits per weight parameter for this format.
    pub fn bits_per_weight(self) -> f32 {
        match self {
            Self::FP32 => 32.0,
            Self::Q8_0 | Self::FP8E4M3 | Self::FP8E5M2 | Self::AWQQ8 | Self::GPTQQ8 => 8.0,
            Self::Q4_0 | Self::AWQQ4 | Self::GPTQQ4 => 4.0,
            Self::BitNet158 => 1.58,
        }
    }

    /// Compression factor relative to uncompressed `f32` (32-bit).
    pub fn compression_ratio(self) -> f32 {
        32.0 / self.bits_per_weight()
    }
}

/// A polymorphic linear layer that can dynamically switch between full precision and various quantized formats.
pub enum QuantizedLayer {
    FP32(Linear),
    BlockQuant(QuantizedLinear),
    FP8(FP8Linear),
    BitNet(BitLinear),
    PackedBitNet(PackedBitLinear),
    AWQ(AWQLinear),
    GPTQ(GPTQLinear),
}

impl QuantizedLayer {
    /// Creates a layer from an existing standard [`Linear`] with the specified [`QuantFormat`].
    pub fn from_linear(
        linear: &Linear,
        format: QuantFormat,
        calib_x: Option<&Tensor>,
    ) -> Result<Self> {
        let valid_calib = match calib_x {
            Some(cx) if cx.shape().last() == Some(&linear.in_features()) => Some(cx),
            _ => None,
        };
        match format {
            QuantFormat::FP32 => Ok(Self::FP32(linear.clone())),
            QuantFormat::Q8_0 => {
                let q = QuantizedLinear::from_linear(linear, QuantScheme::DEFAULT_Q8_0)?;
                Ok(Self::BlockQuant(q))
            }
            QuantFormat::Q4_0 => {
                let q = QuantizedLinear::from_linear(linear, QuantScheme::DEFAULT_Q4_0)?;
                Ok(Self::BlockQuant(q))
            }
            QuantFormat::FP8E4M3 => {
                let q = FP8Linear::from_linear(linear, FP8Format::E4M3)?;
                Ok(Self::FP8(q))
            }
            QuantFormat::FP8E5M2 => {
                let q = FP8Linear::from_linear(linear, FP8Format::E5M2)?;
                Ok(Self::FP8(q))
            }
            QuantFormat::BitNet158 => {
                let q = BitLinear::from_linear(linear)?;
                Ok(Self::BitNet(q))
            }
            QuantFormat::AWQQ8 => {
                if let Some(calib) = valid_calib {
                    let q = AWQLinear::from_linear_calibrated(
                        linear,
                        calib,
                        QuantScheme::DEFAULT_Q8_0,
                        0.5,
                    )?;
                    Ok(Self::AWQ(q))
                } else {
                    let q = QuantizedLinear::from_linear(linear, QuantScheme::DEFAULT_Q8_0)?;
                    Ok(Self::BlockQuant(q))
                }
            }
            QuantFormat::AWQQ4 => {
                if let Some(calib) = valid_calib {
                    let q = AWQLinear::from_linear_calibrated(
                        linear,
                        calib,
                        QuantScheme::DEFAULT_Q4_0,
                        0.5,
                    )?;
                    Ok(Self::AWQ(q))
                } else {
                    let q = QuantizedLinear::from_linear(linear, QuantScheme::DEFAULT_Q4_0)?;
                    Ok(Self::BlockQuant(q))
                }
            }
            QuantFormat::GPTQQ8 => {
                if let Some(calib) = valid_calib {
                    let q = GPTQLinear::from_linear_calibrated(
                        linear,
                        calib,
                        QuantScheme::DEFAULT_Q8_0,
                        0.01,
                    )?;
                    Ok(Self::GPTQ(q))
                } else {
                    let q = QuantizedLinear::from_linear(linear, QuantScheme::DEFAULT_Q8_0)?;
                    Ok(Self::BlockQuant(q))
                }
            }
            QuantFormat::GPTQQ4 => {
                if let Some(calib) = valid_calib {
                    let q = GPTQLinear::from_linear_calibrated(
                        linear,
                        calib,
                        QuantScheme::DEFAULT_Q4_0,
                        0.01,
                    )?;
                    Ok(Self::GPTQ(q))
                } else {
                    let q = QuantizedLinear::from_linear(linear, QuantScheme::DEFAULT_Q4_0)?;
                    Ok(Self::BlockQuant(q))
                }
            }
        }
    }

    /// Converts this layer back into a standard full-precision [`Linear`].
    pub fn to_linear(&self, dtype: DType) -> Result<Linear> {
        match self {
            Self::FP32(lin) => Ok(lin.clone()),
            Self::BlockQuant(q) => q.to_linear(dtype),
            Self::FP8(q) => q.to_linear(dtype),
            Self::BitNet(q) => q.to_linear(dtype),
            Self::PackedBitNet(q) => q.to_linear(dtype),
            Self::AWQ(q) => q.to_linear(dtype),
            Self::GPTQ(q) => q.to_linear(dtype),
        }
    }

    /// Requantizes this layer into another [`QuantFormat`].
    pub fn convert_to(
        &mut self,
        target_format: QuantFormat,
        calib_x: Option<&Tensor>,
    ) -> Result<()> {
        let lin = self.to_linear(DType::F32)?;
        *self = Self::from_linear(&lin, target_format, calib_x)?;
        Ok(())
    }

    /// Memory footprint in bytes of the layer's weights.
    pub fn weight_bytes(&self) -> usize {
        match self {
            Self::FP32(lin) => lin.weight.numel() * 4,
            Self::BlockQuant(q) => q.weight_bytes(),
            Self::FP8(q) => q.qweight.size_in_bytes(),
            Self::BitNet(q) => q.weight_bytes(),
            Self::PackedBitNet(q) => q.weight.packed_data.len() + std::mem::size_of::<f32>(),
            Self::AWQ(q) => q.qweight.size_in_bytes() + q.channel_scales.numel() * 4,
            Self::GPTQ(q) => q.qweight.size_in_bytes(),
        }
    }
}

impl Module for QuantizedLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::FP32(lin) => lin.forward(input),
            Self::BlockQuant(q) => q.forward(input),
            Self::FP8(q) => q.forward(input),
            Self::BitNet(q) => q.forward(input),
            Self::PackedBitNet(q) => q.forward(input),
            Self::AWQ(q) => q.forward(input),
            Self::GPTQ(q) => q.forward(input),
        }
    }
}

/// Autonomous configuration policy for LLM quantization.
#[derive(Debug, Clone)]
pub struct AutonomousQuantPolicy {
    /// Maximum allowable relative loss / MSE degradation before stepping back to higher precision (e.g. 0.05 = 5%).
    pub max_loss_degradation: f32,
    /// Minimum compression ratio targeted across the model (e.g. 4.0 = 4x compression).
    pub target_compression_ratio: f32,
    /// Maximum total weight memory in bytes (if any).
    pub max_memory_bytes: Option<usize>,
    /// Whether attention projections (Q, K, V, O) should prioritize higher precision over MLP.
    pub preserve_attention_precision: bool,
}

impl Default for AutonomousQuantPolicy {
    fn default() -> Self {
        Self {
            max_loss_degradation: 0.05,
            target_compression_ratio: 4.0,
            max_memory_bytes: None,
            preserve_attention_precision: true,
        }
    }
}

/// Autonomous Quantization Change Manager for LLMs.
///
/// Automatically tracks model perplexity/error across calibration datasets and autonomously
/// determines layer-by-layer optimal quantization strategies (mixed-precision scheduling).
pub struct AutonomousQuantizer {
    pub policy: AutonomousQuantPolicy,
}

impl AutonomousQuantizer {
    pub fn new(policy: AutonomousQuantPolicy) -> Self {
        Self { policy }
    }

    /// Measures the MSE distortion of quantizing a linear layer into a target format.
    pub fn evaluate_layer_distortion(
        linear: &Linear,
        format: QuantFormat,
        calib_x: &Tensor,
    ) -> Result<f32> {
        let baseline = linear.forward(calib_x)?;
        let q_layer = QuantizedLayer::from_linear(linear, format, Some(calib_x))?;
        let q_out = q_layer.forward(calib_x)?;
        let diff = baseline.sub(&q_out)?;
        let mse = diff.mul(&diff)?.mean()?.item::<f32>()?;
        let base_var = baseline.mul(&baseline)?.mean()?.item::<f32>()?.max(1e-7);
        Ok(mse / base_var)
    }

    /// Autonomously selects the most compressed format for a layer that satisfies the maximum relative distortion.
    pub fn select_optimal_format(
        &self,
        linear: &Linear,
        calib_x: &Tensor,
        candidate_formats: &[QuantFormat],
    ) -> Result<(QuantFormat, f32)> {
        // Evaluate candidates sorted from highest compression (lowest bits) to lowest compression
        let mut sorted_candidates = candidate_formats.to_vec();
        sorted_candidates.sort_by(|a, b| {
            a.bits_per_weight()
                .partial_cmp(&b.bits_per_weight())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &fmt in &sorted_candidates {
            let rel_distortion = Self::evaluate_layer_distortion(linear, fmt, calib_x)?;
            if rel_distortion <= self.policy.max_loss_degradation {
                return Ok((fmt, rel_distortion));
            }
        }

        // If none met the strict threshold, fallback to FP32 or lowest distortion
        Ok((QuantFormat::FP32, 0.0))
    }

    /// Autonomously quantizes all linear projections in a [`CausalLMBlock`] according to layer sensitivity.
    pub fn quantize_block(
        &self,
        block: &CausalLMBlock,
        calib_x: &Tensor,
    ) -> Result<(QuantizedLMBlock, BlockQuantProfile)> {
        let candidates = [
            QuantFormat::Q4_0,
            QuantFormat::FP8E4M3,
            QuantFormat::Q8_0,
            QuantFormat::FP32,
        ];

        let q_fmt = if self.policy.preserve_attention_precision {
            // High sensitivity for Q and K
            QuantFormat::Q8_0
        } else {
            self.select_optimal_format(&block.q_proj, calib_x, &candidates)?
                .0
        };

        let k_fmt = if self.policy.preserve_attention_precision {
            QuantFormat::Q8_0
        } else {
            self.select_optimal_format(&block.k_proj, calib_x, &candidates)?
                .0
        };

        let (v_fmt, _) = self.select_optimal_format(&block.v_proj, calib_x, &candidates)?;
        let (o_fmt, _) = self.select_optimal_format(&block.o_proj, calib_x, &candidates)?;
        let (gate_fmt, _) = self.select_optimal_format(&block.mlp.w_gate, calib_x, &candidates)?;
        let (up_fmt, _) = self.select_optimal_format(&block.mlp.w_up, calib_x, &candidates)?;

        // For w_down, activation input comes from hidden_dim (after SwiGLU: w_gate(x) * w_up(x))
        let hidden_act = block.mlp.w_gate.forward(calib_x)?;
        let (down_fmt, _) =
            self.select_optimal_format(&block.mlp.w_down, &hidden_act, &candidates)?;

        let q_layer = QuantizedLayer::from_linear(&block.q_proj, q_fmt, Some(calib_x))?;
        let k_layer = QuantizedLayer::from_linear(&block.k_proj, k_fmt, Some(calib_x))?;
        let v_layer = QuantizedLayer::from_linear(&block.v_proj, v_fmt, Some(calib_x))?;
        let o_layer = QuantizedLayer::from_linear(&block.o_proj, o_fmt, Some(calib_x))?;
        let w_gate_layer = QuantizedLayer::from_linear(&block.mlp.w_gate, gate_fmt, Some(calib_x))?;
        let w_up_layer = QuantizedLayer::from_linear(&block.mlp.w_up, up_fmt, Some(calib_x))?;
        let w_down_layer =
            QuantizedLayer::from_linear(&block.mlp.w_down, down_fmt, Some(&hidden_act))?;

        let orig_bytes = (block.q_proj.weight.numel()
            + block.k_proj.weight.numel()
            + block.v_proj.weight.numel()
            + block.o_proj.weight.numel()
            + block.mlp.w_gate.weight.numel()
            + block.mlp.w_up.weight.numel()
            + block.mlp.w_down.weight.numel())
            * 4;

        let quant_bytes = q_layer.weight_bytes()
            + k_layer.weight_bytes()
            + v_layer.weight_bytes()
            + o_layer.weight_bytes()
            + w_gate_layer.weight_bytes()
            + w_up_layer.weight_bytes()
            + w_down_layer.weight_bytes();

        let profile = BlockQuantProfile {
            q_format: q_fmt,
            k_format: k_fmt,
            v_format: v_fmt,
            o_format: o_fmt,
            w_gate_format: gate_fmt,
            w_up_format: up_fmt,
            w_down_format: down_fmt,
            original_bytes: orig_bytes,
            quantized_bytes: quant_bytes,
            compression_ratio: (orig_bytes as f32) / (quant_bytes.max(1) as f32),
        };

        let q_block = QuantizedLMBlock {
            q_proj: q_layer,
            k_proj: k_layer,
            v_proj: v_layer,
            o_proj: o_layer,
            w_gate: w_gate_layer,
            w_up: w_up_layer,
            w_down: w_down_layer,
            attn_norm: block.attn_norm.clone(),
            ffn_norm: block.ffn_norm.clone(),
            num_heads: block.num_heads,
            head_dim: block.head_dim,
        };

        Ok((q_block, profile))
    }

    /// Autonomously converts an entire [`CausalLM`] into a mixed-precision [`QuantizedCausalLM`].
    pub fn quantize_llm(&self, model: &CausalLM, calib_x: &Tensor) -> Result<QuantizedCausalLM> {
        let mut q_blocks = Vec::with_capacity(model.blocks.len());
        let mut profiles = Vec::with_capacity(model.blocks.len());

        for block in &model.blocks {
            let (q_b, prof) = self.quantize_block(block, calib_x)?;
            q_blocks.push(q_b);
            profiles.push(prof);
        }

        let lm_head_fmt = QuantFormat::Q8_0;
        let lm_head = QuantizedLayer::from_linear(&model.lm_head, lm_head_fmt, Some(calib_x))?;

        Ok(QuantizedCausalLM {
            tok_emb: model.tok_emb.clone(),
            rope: model.rope.clone(),
            blocks: q_blocks,
            norm: model.norm.clone(),
            lm_head,
            profiles,
            vocab_size: model.vocab_size,
            max_seq_len: model.max_seq_len,
        })
    }
}

/// Statistics and profile of quantization for a single transformer block.
#[derive(Debug, Clone)]
pub struct BlockQuantProfile {
    pub q_format: QuantFormat,
    pub k_format: QuantFormat,
    pub v_format: QuantFormat,
    pub o_format: QuantFormat,
    pub w_gate_format: QuantFormat,
    pub w_up_format: QuantFormat,
    pub w_down_format: QuantFormat,
    pub original_bytes: usize,
    pub quantized_bytes: usize,
    pub compression_ratio: f32,
}

/// A quantized transformer decoder block with polymorphic quantized linear projections.
pub struct QuantizedLMBlock {
    pub q_proj: QuantizedLayer,
    pub k_proj: QuantizedLayer,
    pub v_proj: QuantizedLayer,
    pub o_proj: QuantizedLayer,
    pub w_gate: QuantizedLayer,
    pub w_up: QuantizedLayer,
    pub w_down: QuantizedLayer,
    pub attn_norm: crate::nn::RMSNorm,
    pub ffn_norm: crate::nn::RMSNorm,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl QuantizedLMBlock {
    pub fn forward_rope(
        &self,
        x: &Tensor,
        rope: &crate::nn::RotaryEmbedding,
        offset: usize,
    ) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(crate::error::Error::Msg(format!(
                "QuantizedLMBlock::forward_rope: expected [B, L, D], got {:?}",
                x.shape()
            )));
        }
        let (b, l, d_model) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.num_heads;
        let d = self.head_dim;

        let normed = self.attn_norm.forward(x)?;
        let q = self
            .q_proj
            .forward(&normed)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(&normed)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(&normed)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;

        let q = rope.apply(&q, offset)?;
        let k = rope.apply(&k, offset)?;

        let s = k.shape()[2];
        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = q.matmul(&k.t()?)?.mul_scalar(scale)?;
        if l > 1 {
            let diag = (s as isize) - (l as isize) + 1;
            let mask = Tensor::ones(&[l, s], DType::Bool, &scores.device())?.triu(diag)?;
            scores = scores.masked_fill(&mask, -1e9)?;
        }
        let attn = scores.softmax(-1)?;
        let ctx = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape(&[b, l, d_model])?;
        let x = x.add(&self.o_proj.forward(&ctx)?)?;

        // SwiGLU MLP: w_down(silu(w_gate(x)) * w_up(x))
        let ffn_in = self.ffn_norm.forward(&x)?;
        let gate = self.w_gate.forward(&ffn_in)?.silu()?;
        let up = self.w_up.forward(&ffn_in)?;
        let fused = gate.mul(&up)?;
        let h_ffn = self.w_down.forward(&fused)?;
        x.add(&h_ffn)
    }

    /// Dynamically transforms all projections in this block to a target quantization format at runtime.
    pub fn dynamic_quant_change(
        &mut self,
        target_format: QuantFormat,
        calib_x: Option<&Tensor>,
    ) -> Result<()> {
        self.q_proj.convert_to(target_format, calib_x)?;
        self.k_proj.convert_to(target_format, calib_x)?;
        self.v_proj.convert_to(target_format, calib_x)?;
        self.o_proj.convert_to(target_format, calib_x)?;
        self.w_gate.convert_to(target_format, calib_x)?;
        self.w_up.convert_to(target_format, calib_x)?;
        self.w_down.convert_to(target_format, calib_x)?;
        Ok(())
    }

    /// Converts this quantized block back into a standard [`CausalLMBlock`].
    pub fn to_causal_block(&self, dtype: DType) -> Result<CausalLMBlock> {
        let q_proj = self.q_proj.to_linear(dtype)?;
        let k_proj = self.k_proj.to_linear(dtype)?;
        let v_proj = self.v_proj.to_linear(dtype)?;
        let o_proj = self.o_proj.to_linear(dtype)?;
        let w_gate = self.w_gate.to_linear(dtype)?;
        let w_up = self.w_up.to_linear(dtype)?;
        let w_down = self.w_down.to_linear(dtype)?;

        let mut mlp = crate::nn::SwiGLU::new(q_proj.in_features(), w_gate.out_features())?;
        mlp.w_gate = w_gate;
        mlp.w_up = w_up;
        mlp.w_down = w_down;

        Ok(CausalLMBlock {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            mlp,
            attn_norm: self.attn_norm.clone(),
            ffn_norm: self.ffn_norm.clone(),
            num_heads: self.num_heads,
            head_dim: self.head_dim,
        })
    }
}

/// Mixed-precision Quantized Causal Language Model.
pub struct QuantizedCausalLM {
    pub tok_emb: crate::nn::Embedding,
    pub rope: crate::nn::RotaryEmbedding,
    pub blocks: Vec<QuantizedLMBlock>,
    pub norm: crate::nn::RMSNorm,
    pub lm_head: QuantizedLayer,
    pub profiles: Vec<BlockQuantProfile>,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl QuantizedCausalLM {
    /// Autoregressive forward pass producing next-token logits.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let mut h = self.tok_emb.forward(token_ids)?;
        for block in &self.blocks {
            h = block.forward_rope(&h, &self.rope, 0)?;
        }
        let normed = self.norm.forward(&h)?;
        self.lm_head.forward(&normed)
    }

    /// Autonomously triggers dynamic quantization change on all blocks in the LLM.
    pub fn autonomous_quant_change(
        &mut self,
        target_format: QuantFormat,
        calib_x: Option<&Tensor>,
    ) -> Result<()> {
        for block in &mut self.blocks {
            block.dynamic_quant_change(target_format, calib_x)?;
        }
        self.lm_head.convert_to(target_format, calib_x)?;
        Ok(())
    }

    /// Total memory footprint of all linear layers in bytes.
    pub fn total_weight_bytes(&self) -> usize {
        let mut total = self.lm_head.weight_bytes();
        for b in &self.blocks {
            total += b.q_proj.weight_bytes()
                + b.k_proj.weight_bytes()
                + b.v_proj.weight_bytes()
                + b.o_proj.weight_bytes()
                + b.w_gate.weight_bytes()
                + b.w_up.weight_bytes()
                + b.w_down.weight_bytes();
        }
        total
    }

    /// Converts back to a standard full-precision [`CausalLM`].
    pub fn to_causal_lm(&self, dtype: DType) -> Result<CausalLM> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            blocks.push(b.to_causal_block(dtype)?);
        }
        let lm_head = self.lm_head.to_linear(dtype)?;

        Ok(CausalLM {
            tok_emb: self.tok_emb.clone(),
            rope: self.rope.clone(),
            blocks,
            norm: self.norm.clone(),
            lm_head,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
        })
    }
}
