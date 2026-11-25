//! Complete Transformer and LLM building blocks:
//! - [`SinusoidalPositionalEncoding`] and [`RotaryEmbedding`] (RoPE)
//! - [`SwiGLU`] and [`FeedForward`]
//! - [`TransformerEncoderLayer`], [`TransformerEncoder`]
//! - [`TransformerDecoderLayer`], [`TransformerDecoder`]
//! - [`CausalLM`] (LLaMA/GPT-style decoder language model with greedy, temperature,
//!   top-k, and top-p nucleus sampling via [`CausalLM::generate`])

use super::{Dropout, Embedding, LayerNorm, Linear, Module, MultiHeadAttention, RMSNorm};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::{Tensor, no_grad};

fn check_bld(op: &'static str, x: &Tensor) -> Result<()> {
    if x.ndim() != 3 {
        return Err(Error::Msg(format!(
            "{op}: expected [B, L, D] input, got {:?}",
            x.shape()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sinusoidal Positional Encoding
// ---------------------------------------------------------------------------
pub struct SinusoidalPositionalEncoding {
    pub pe: Tensor,
    pub max_len: usize,
    pub d_model: usize,
}

impl SinusoidalPositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Result<Self> {
        let n = max_len.checked_mul(d_model).ok_or_else(|| {
            Error::Msg(format!(
                "SinusoidalPositionalEncoding: table size {max_len} * {d_model} overflows"
            ))
        })?;
        let n =
            crate::limits::check_at_most("sinusoidal table size", n, crate::limits::MAX_ELEMENTS)?;
        let mut data = crate::layout::try_filled(n, 0.0f32)?;
        for pos in 0..max_len {
            for i in (0..d_model).step_by(2) {
                let div_term = (-(i as f64) * (10000.0f64.ln()) / (d_model as f64)).exp();
                let angle = (pos as f64) * div_term;
                data[pos * d_model + i] = angle.sin() as f32;
                if i + 1 < d_model {
                    data[pos * d_model + i + 1] = angle.cos() as f32;
                }
            }
        }
        let pe = Tensor::from_vec(data, &[1, max_len, d_model])?;
        Ok(Self {
            pe,
            max_len,
            d_model,
        })
    }
}

impl Module for SinusoidalPositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 3 {
            return Err(Error::Msg(format!(
                "SinusoidalPositionalEncoding: expected [B, L, D] input, got {:?}",
                input.shape()
            )));
        }
        let seq_len = input.shape()[1];
        let slice = self
            .pe
            .narrow(1, 0, seq_len)?
            .to_device(&input.device())?
            .to_dtype(input.dtype())?;
        input.add(&slice)
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding (RoPE)
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct RotaryEmbedding {
    cos_cached: Tensor,
    sin_cached: Tensor,
    pub dim: usize,
    pub max_seq_len: usize,
}

/// Attention width must split evenly over a positive number of heads (a zero or non-dividing count used to panic).
pub(crate) fn check_heads(d_model: usize, num_heads: usize) -> Result<()> {
    if num_heads == 0 || d_model % num_heads != 0 {
        return Err(crate::error::Error::Msg(format!(
            "d_model ({d_model}) must be divisible by a positive num_heads ({num_heads})"
        )));
    }
    Ok(())
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f64) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(crate::error::Error::Msg(format!(
                "RoPE head dim must be even, got {dim}"
            )));
        }
        let half = dim / 2;
        let n = half.checked_mul(max_seq_len).ok_or_else(|| {
            Error::Msg(format!(
                "RotaryEmbedding: table size {max_seq_len} * {half} overflows"
            ))
        })?;
        let n = crate::limits::check_at_most("RoPE table size", n, crate::limits::MAX_ELEMENTS)?;
        let mut cos_data = crate::layout::try_filled(n, 0.0f32)?;
        let mut sin_data = crate::layout::try_filled(n, 0.0f32)?;

        for pos in 0..max_seq_len {
            for i in 0..half {
                let inv_freq = 1.0 / base.powf((2 * i) as f64 / (dim as f64));
                let theta = (pos as f64) * inv_freq;
                cos_data[pos * half + i] = theta.cos() as f32;
                sin_data[pos * half + i] = theta.sin() as f32;
            }
        }
        Ok(Self {
            cos_cached: Tensor::from_vec(cos_data, &[1, 1, max_seq_len, half])?,
            sin_cached: Tensor::from_vec(sin_data, &[1, 1, max_seq_len, half])?,
            dim,
            max_seq_len,
        })
    }

    /// Builds the cos/sin tables from `angle(pos, i) -> (cos, sin)` for `i in 0..dim/2` (already
    /// including any magnitude scaling), for scaled-RoPE schemes that are not plain `base^(-2i/dim)`.
    pub fn from_fn(
        dim: usize,
        max_seq_len: usize,
        angle: impl Fn(usize, usize) -> (f64, f64),
    ) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(crate::error::Error::Msg(format!(
                "RoPE head dim must be even, got {dim}"
            )));
        }
        let half = dim / 2;
        let n = half.checked_mul(max_seq_len).ok_or_else(|| {
            Error::Msg(format!(
                "RotaryEmbedding: table size {max_seq_len} * {half} overflows"
            ))
        })?;
        let n = crate::limits::check_at_most("RoPE table size", n, crate::limits::MAX_ELEMENTS)?;
        let mut cos_data = crate::layout::try_with_capacity(n)?;
        let mut sin_data = crate::layout::try_with_capacity(n)?;
        for pos in 0..max_seq_len {
            for i in 0..half {
                let (c, s) = angle(pos, i);
                cos_data.push(c as f32);
                sin_data.push(s as f32);
            }
        }
        Ok(Self {
            cos_cached: Tensor::from_vec(cos_data, &[1, 1, max_seq_len, half])?,
            sin_cached: Tensor::from_vec(sin_data, &[1, 1, max_seq_len, half])?,
            dim,
            max_seq_len,
        })
    }

    /// **YaRN (Yet another RoPE extensioN, Peng et al., ICLR 2024)**:
    /// Extends context window by scale factor `scale = max_seq_len / orig_max_seq_len` using
    /// NTK-by-parts frequency ramp interpolation and attention temperature factor
    /// $\sqrt{1/t} = 0.1 \ln(s) + 1$.
    pub fn new_yarn(
        dim: usize,
        max_seq_len: usize,
        orig_max_seq_len: usize,
        base: f64,
        beta_fast: f64,
        beta_slow: f64,
    ) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(crate::error::Error::Msg(format!(
                "RoPE head dim must be even, got {dim}"
            )));
        }
        let half = dim / 2;
        let scale = ((max_seq_len as f64) / (orig_max_seq_len.max(1) as f64)).max(1.0);
        let mscale = if scale > 1.0 {
            0.1 * scale.ln() + 1.0
        } else {
            1.0
        };
        let n = half.checked_mul(max_seq_len).ok_or_else(|| {
            Error::Msg(format!(
                "RotaryEmbedding: table size {max_seq_len} * {half} overflows"
            ))
        })?;
        let n = crate::limits::check_at_most("RoPE table size", n, crate::limits::MAX_ELEMENTS)?;
        let mut cos_data = crate::layout::try_filled(n, 0.0f32)?;
        let mut sin_data = crate::layout::try_filled(n, 0.0f32)?;

        for i in 0..half {
            let inv_freq_extrap = 1.0 / base.powf((2 * i) as f64 / (dim as f64));
            let inv_freq_interp = inv_freq_extrap / scale;
            let wavelength = 2.0 * std::f64::consts::PI / inv_freq_extrap;
            let num_rotations = (orig_max_seq_len as f64) / wavelength;
            let gamma = if (beta_fast - beta_slow).abs() < 1e-9 {
                0.5
            } else {
                ((num_rotations - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)
            };
            let inv_freq = (1.0 - gamma) * inv_freq_interp + gamma * inv_freq_extrap;
            for pos in 0..max_seq_len {
                let theta = (pos as f64) * inv_freq;
                cos_data[pos * half + i] = (theta.cos() * mscale) as f32;
                sin_data[pos * half + i] = (theta.sin() * mscale) as f32;
            }
        }
        Ok(Self {
            cos_cached: Tensor::from_vec(cos_data, &[1, 1, max_seq_len, half])?,
            sin_cached: Tensor::from_vec(sin_data, &[1, 1, max_seq_len, half])?,
            dim,
            max_seq_len,
        })
    }

    /// Applies rotary embedding to a 4D tensor `[batch, heads, seq_len, head_dim]`.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        if x.ndim() != 4 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding::apply: expected [B, H, L, D] input, got {:?}",
                x.shape()
            )));
        }
        if x.shape()[3] != self.dim {
            return Err(Error::ShapeMismatch {
                op: "RotaryEmbedding::apply",
                expected: vec![x.shape()[0], x.shape()[1], x.shape()[2], self.dim],
                got: x.shape().to_vec(),
            });
        }
        let seq_len = x.shape()[2];
        let half = self.dim / 2;
        let x1 = x.narrow(3, 0, half)?;
        let x2 = x.narrow(3, half, half)?;

        let cos = self
            .cos_cached
            .narrow(2, offset, seq_len)?
            .to_device(&x.device())?
            .to_dtype(x.dtype())?;
        let sin = self
            .sin_cached
            .narrow(2, offset, seq_len)?
            .to_device(&x.device())?
            .to_dtype(x.dtype())?;

        // [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let out1 = x1.mul(&cos)?.sub(&x2.mul(&sin)?)?;
        let out2 = x1.mul(&sin)?.add(&x2.mul(&cos)?)?;
        Tensor::cat(&[out1, out2], 3)
    }
}

// ---------------------------------------------------------------------------
// SwiGLU & Standard Feed-Forward Network
// ---------------------------------------------------------------------------
pub struct SwiGLU {
    pub w_gate: Linear,
    pub w_up: Linear,
    pub w_down: Linear,
}

impl SwiGLU {
    pub fn new(d_model: usize, hidden_dim: usize) -> Result<Self> {
        Ok(Self {
            w_gate: Linear::with_bias(d_model, hidden_dim, false)?,
            w_up: Linear::with_bias(d_model, hidden_dim, false)?,
            w_down: Linear::with_bias(hidden_dim, d_model, false)?,
        })
    }
}

impl Module for SwiGLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.w_gate.forward(input)?.silu()?;
        let up = self.w_up.forward(input)?;
        self.w_down.forward(&gate.mul(&up)?)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("w_gate", &self.w_gate),
            ("w_up", &self.w_up),
            ("w_down", &self.w_down),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
    pub dropout: Dropout,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, dropout_p: f32) -> Result<Self> {
        // `Dropout::new` asserts on NaN; reject it here with an Err instead.
        if !dropout_p.is_finite() {
            return Err(crate::error::Error::Msg(
                "FeedForward dropout_p must be finite".to_string(),
            ));
        }
        Ok(Self {
            fc1: Linear::new(d_model, d_ff)?,
            fc2: Linear::new(d_ff, d_model)?,
            dropout: Dropout::new(dropout_p),
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(input)?.gelu()?;
        let h = self.dropout.forward(&h)?;
        self.fc2.forward(&h)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [("fc1", &self.fc1), ("fc2", &self.fc2)] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        self.dropout.train(mode);
    }
}

// ---------------------------------------------------------------------------
// TransformerEncoderLayer & TransformerEncoder (Pre-Norm architecture)
// ---------------------------------------------------------------------------
pub struct TransformerEncoderLayer {
    pub self_attn: MultiHeadAttention,
    pub ffn: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dropout: Dropout,
    pub causal: bool,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout_p: f32) -> Result<Self> {
        // `Dropout::new` asserts on NaN; reject it here with an Err instead.
        if !dropout_p.is_finite() {
            return Err(crate::error::Error::Msg(
                "TransformerEncoderLayer dropout_p must be finite".to_string(),
            ));
        }
        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, num_heads)?,
            ffn: FeedForward::new(d_model, d_ff, dropout_p)?,
            norm1: LayerNorm::new(d_model, 1e-5)?,
            norm2: LayerNorm::new(d_model, 1e-5)?,
            dropout: Dropout::new(dropout_p),
            causal: false,
        })
    }

    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n1 = self.norm1.forward(x)?;
        let attn = self.self_attn.forward_qkv(&n1, &n1, &n1, self.causal)?;
        let x = x.add(&self.dropout.forward(&attn)?)?;

        let n2 = self.norm2.forward(&x)?;
        let ff = self.ffn.forward(&n2)?;
        x.add(&self.dropout.forward(&ff)?)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.self_attn.named_parameters() {
            out.push((format!("self_attn.{k}"), v));
        }
        for (k, v) in self.ffn.named_parameters() {
            out.push((format!("ffn.{k}"), v));
        }
        for (k, v) in self.norm1.named_parameters() {
            out.push((format!("norm1.{k}"), v));
        }
        for (k, v) in self.norm2.named_parameters() {
            out.push((format!("norm2.{k}"), v));
        }
        out
    }

    fn train(&self, mode: bool) {
        self.ffn.train(mode);
        self.dropout.train(mode);
    }
}

pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    pub final_norm: LayerNorm,
}

impl TransformerEncoder {
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_p: f32,
    ) -> Result<Self> {
        crate::limits::check_repeats("num_layers", num_layers)?;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model, num_heads, d_ff, dropout_p,
            )?);
        }
        Ok(Self {
            layers,
            final_norm: LayerNorm::new(d_model, 1e-5)?,
        })
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.final_norm.forward(&x)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (k, v) in l.named_parameters() {
                out.push((format!("layers.{i}.{k}"), v));
            }
        }
        for (k, v) in self.final_norm.named_parameters() {
            out.push((format!("final_norm.{k}"), v));
        }
        out
    }

    fn train(&self, mode: bool) {
        self.layers.iter().for_each(|l| l.train(mode));
    }
}

// ---------------------------------------------------------------------------
// TransformerDecoderLayer & TransformerDecoder (Self-Attn + Cross-Attn + FFN)
// ---------------------------------------------------------------------------
pub struct TransformerDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: MultiHeadAttention,
    pub ffn: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub dropout: Dropout,
}

impl TransformerDecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, dropout_p: f32) -> Result<Self> {
        // `Dropout::new` asserts on NaN; reject it here with an Err instead.
        if !dropout_p.is_finite() {
            return Err(crate::error::Error::Msg(
                "TransformerDecoderLayer dropout_p must be finite".to_string(),
            ));
        }
        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, num_heads)?,
            cross_attn: MultiHeadAttention::new(d_model, num_heads)?,
            ffn: FeedForward::new(d_model, d_ff, dropout_p)?,
            norm1: LayerNorm::new(d_model, 1e-5)?,
            norm2: LayerNorm::new(d_model, 1e-5)?,
            norm3: LayerNorm::new(d_model, 1e-5)?,
            dropout: Dropout::new(dropout_p),
        })
    }

    pub fn forward_with_memory(&self, tgt: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let n1 = self.norm1.forward(tgt)?;
        let sa = self.self_attn.forward_qkv(&n1, &n1, &n1, true)?;
        let x = tgt.add(&self.dropout.forward(&sa)?)?;

        let n2 = self.norm2.forward(&x)?;
        let ca = self.cross_attn.forward_qkv(&n2, memory, memory, false)?;
        let x = x.add(&self.dropout.forward(&ca)?)?;

        let n3 = self.norm3.forward(&x)?;
        let ff = self.ffn.forward(&n3)?;
        x.add(&self.dropout.forward(&ff)?)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.self_attn.named_parameters() {
            out.push((format!("self_attn.{k}"), v));
        }
        for (k, v) in self.cross_attn.named_parameters() {
            out.push((format!("cross_attn.{k}"), v));
        }
        for (k, v) in self.ffn.named_parameters() {
            out.push((format!("ffn.{k}"), v));
        }
        for (p, n) in [
            ("norm1", &self.norm1),
            ("norm2", &self.norm2),
            ("norm3", &self.norm3),
        ] {
            for (k, v) in n.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderLayer>,
    pub final_norm: LayerNorm,
}

impl TransformerDecoder {
    pub fn new(
        num_layers: usize,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_p: f32,
    ) -> Result<Self> {
        crate::limits::check_repeats("num_layers", num_layers)?;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerDecoderLayer::new(
                d_model, num_heads, d_ff, dropout_p,
            )?);
        }
        Ok(Self {
            layers,
            final_norm: LayerNorm::new(d_model, 1e-5)?,
        })
    }

    pub fn forward_with_memory(&self, tgt: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let mut x = tgt.clone();
        for layer in &self.layers {
            x = layer.forward_with_memory(&x, memory)?;
        }
        self.final_norm.forward(&x)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (k, v) in l.named_parameters() {
                out.push((format!("layers.{i}.{k}"), v));
            }
        }
        for (k, v) in self.final_norm.named_parameters() {
            out.push((format!("final_norm.{k}"), v));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Modern Causal Language Model (RoPE + RMSNorm + SwiGLU + Autoregressive Generate)
// ---------------------------------------------------------------------------
pub struct CausalLMBlock {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub mlp: SwiGLU,
    pub attn_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl CausalLMBlock {
    pub fn new(d_model: usize, num_heads: usize, ffn_hidden: usize) -> Result<Self> {
        check_heads(d_model, num_heads)?;
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj: Linear::with_bias(d_model, d_model, false)?,
            k_proj: Linear::with_bias(d_model, d_model, false)?,
            v_proj: Linear::with_bias(d_model, d_model, false)?,
            o_proj: Linear::with_bias(d_model, d_model, false)?,
            mlp: SwiGLU::new(d_model, ffn_hidden)?,
            attn_norm: RMSNorm::new(d_model, 1e-5)?,
            ffn_norm: RMSNorm::new(d_model, 1e-5)?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward_rope(&self, x: &Tensor, rope: &RotaryEmbedding) -> Result<Tensor> {
        self.forward_rope_cached(x, rope, None)
    }

    pub fn forward_rope_cached(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        cache: Option<&mut LayerKVCache>,
    ) -> Result<Tensor> {
        check_bld("forward_rope_cached", x)?;
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

        let offset = cache.as_ref().map_or(0, |c| c.seq_len());
        let q = rope.apply(&q, offset)?;
        let k = rope.apply(&k, offset)?;

        let (k, v) = if let Some(c) = cache {
            c.append(k, v)?
        } else {
            (k, v)
        };

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

        let h_ffn = self.mlp.forward(&self.ffn_norm.forward(&x)?)?;
        x.add(&h_ffn)
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("o_proj", &self.o_proj),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        for (k, v) in self.mlp.named_parameters() {
            out.push((format!("mlp.{k}"), v));
        }
        for (k, v) in self.attn_norm.named_parameters() {
            out.push((format!("attn_norm.{k}"), v));
        }
        for (k, v) in self.ffn_norm.named_parameters() {
            out.push((format!("ffn_norm.{k}"), v));
        }
        out
    }
}

/// Complete LLaMA/GPT-style Decoder-only Language Model with RoPE, RMSNorm, SwiGLU,
/// and autoregressive token generation.
pub struct CausalLM {
    pub tok_emb: Embedding,
    pub rope: RotaryEmbedding,
    pub blocks: Vec<CausalLMBlock>,
    pub norm: RMSNorm,
    pub lm_head: Linear,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

impl CausalLM {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        ffn_hidden: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        check_heads(d_model, num_heads)?;
        let head_dim = d_model / num_heads;
        let mut blocks = crate::layout::try_with_capacity(num_layers)?;
        crate::limits::check_repeats("num_layers", num_layers)?;
        for _ in 0..num_layers {
            blocks.push(CausalLMBlock::new(d_model, num_heads, ffn_hidden)?);
        }
        Ok(Self {
            tok_emb: Embedding::new(vocab_size, d_model)?,
            rope: RotaryEmbedding::new(head_dim, max_seq_len, 10000.0)?,
            blocks,
            norm: RMSNorm::new(d_model, 1e-5)?,
            lm_head: Linear::with_bias(d_model, vocab_size, false)?,
            vocab_size,
            max_seq_len,
        })
    }

    /// Autoregressively generates `max_new_tokens` given a 1D prompt slice of token IDs,
    /// supporting greedy (`temperature <= 0.0`), temperature scaling, `top_k`, and `top_p`
    /// (nucleus) sampling.
    pub fn generate(
        &self,
        prompt: &[i64],
        max_new_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        top_p: Option<f64>,
    ) -> Result<Vec<i64>> {
        let cfg = crate::sampling::SamplingConfig {
            temperature,
            top_k,
            top_p,
            max_new_tokens,
            ..Default::default()
        };
        self.generate_with_config(prompt, &cfg)
    }

    /// Autoregressive text generation using a comprehensive [`crate::sampling::SamplingConfig`]
    /// (temperature, top_k, top_p, min_p, repetition/presence/frequency penalties, reproducible seed, stop tokens).
    pub fn generate_with_config(
        &self,
        prompt: &[i64],
        config: &crate::sampling::SamplingConfig,
    ) -> Result<Vec<i64>> {
        if prompt.is_empty() {
            return Err(Error::Msg("generate: prompt cannot be empty".into()));
        }
        if config.max_new_tokens == 0 {
            return Ok(prompt.to_vec());
        }
        // Each requested token costs a full forward pass; an uncapped budget
        // never terminates (1e8 tokens at 50 tok/s is 23 days).
        crate::limits::check_iterations("max_new_tokens", config.max_new_tokens)?;
        let mut tokens = prompt.to_vec();
        let mut cache = KVCache::new(self.blocks.len())?;
        no_grad(|| -> Result<()> {
            let mut rng_gen = config.seed.map(crate::rng::Generator::with_seed);
            let mut history: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();

            // Prefill phase
            let prefill_in = Tensor::from_slice(&tokens, &[1, tokens.len()])?;
            let mut logits = self.forward_cached(&prefill_in, &mut cache)?;
            let mut last_logits = logits.select(0, 0)?.select(0, -1)?;
            let first_tok =
                config.sample_token(&last_logits, Some(&history), rng_gen.as_mut())? as i64;
            tokens.push(first_tok);
            history.push(first_tok as usize);

            if config.stop_tokens.contains(&(first_tok as usize)) {
                return Ok(());
            }

            // Decode phase
            for _ in 1..config.max_new_tokens {
                if cache.seq_len() >= self.max_seq_len {
                    cache.clear();
                    let start = tokens.len().saturating_sub(self.max_seq_len);
                    let window = &tokens[start..];
                    let input = Tensor::from_slice(window, &[1, window.len()])?;
                    logits = self.forward_cached(&input, &mut cache)?;
                } else {
                    let cur_tok = *tokens.last().unwrap();
                    let step_in = Tensor::from_slice(&[cur_tok], &[1, 1])?;
                    logits = self.forward_cached(&step_in, &mut cache)?;
                }
                last_logits = logits.select(0, 0)?.select(0, -1)?;
                let next_tok =
                    config.sample_token(&last_logits, Some(&history), rng_gen.as_mut())? as i64;
                tokens.push(next_tok);
                history.push(next_tok as usize);

                if config.stop_tokens.contains(&(next_tok as usize)) {
                    break;
                }
            }
            Ok(())
        })?;
        Ok(tokens)
    }

    /// Forward pass using a stateful [`KVCache`] for incremental autoregressive decoding.
    pub fn forward_cached(&self, tokens: &Tensor, cache: &mut KVCache) -> Result<Tensor> {
        if tokens.numel() == 0 {
            return Err(Error::Msg(
                "forward_cached: input sequence cannot be empty".into(),
            ));
        }
        if cache.layers.len() != self.blocks.len() {
            return Err(Error::Msg(format!(
                "forward_cached: cache has {} layers but the model has {} (cache built for another model?)",
                cache.layers.len(),
                self.blocks.len()
            )));
        }
        let mut h = self.tok_emb.forward(tokens)?;
        for (i, blk) in self.blocks.iter().enumerate() {
            h = blk.forward_rope_cached(&h, &self.rope, Some(&mut cache.layers[i]))?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    /// Speculative Decoding (Leviathan et al. / Chen et al. 2023): uses a smaller `draft_model`
    /// to propose `gamma` candidate tokens per step and verifies them in a single parallel
    /// forward pass of `self`, returning `(generated_tokens, accepted_draft_tokens, total_draft_tokens)`.
    pub fn speculative_generate(
        &self,
        draft_model: &CausalLM,
        prompt: &[usize],
        max_new_tokens: usize,
        gamma: usize,
    ) -> Result<(Vec<usize>, usize, usize)> {
        if prompt.is_empty() {
            return Err(crate::error::Error::Msg(
                "speculative_generate: prompt must not be empty".into(),
            ));
        }
        if self.max_seq_len < 2 {
            return Err(crate::error::Error::Msg(
                "speculative_generate: max_seq_len must be at least 2".into(),
            ));
        }
        if draft_model.max_seq_len < 1 {
            return Err(crate::error::Error::Msg(
                "speculative_generate: draft max_seq_len must be at least 1".into(),
            ));
        }
        // Caps the verify loop below; the raw budget also overflowed
        // `seq.len() + max_new_tokens` at usize::MAX.
        crate::limits::check_iterations("max_new_tokens", max_new_tokens)?;
        // The verify window needs base_pos = full_len - 1 - k_steps >= 0;
        // clamping k_steps to the truncated window keeps it in range.
        let gamma = gamma.max(1).min(self.max_seq_len - 1);
        // Prompt ids past i64::MAX would wrap negative under `as i64` and the
        // embedding would then silently read them as from-the-end rows.
        if let Some(&bad) = prompt.iter().find(|&&t| t > i64::MAX as usize) {
            return Err(crate::error::Error::Msg(format!(
                "speculative_generate: prompt token id {bad} exceeds i64::MAX"
            )));
        }
        crate::tensor::no_grad(|| {
            let mut seq: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
            let target_len = seq.len() + max_new_tokens;
            let mut accepted_draft = 0usize;
            let mut total_draft = 0usize;

            while seq.len() < target_len {
                let remaining = target_len - seq.len();
                let k_steps = gamma.min(remaining);

                // 1. Draft model proposes k_steps tokens autoregressively
                let mut draft_seq = seq.clone();
                let mut draft_tokens = Vec::with_capacity(k_steps);
                let mut draft_probs_list = Vec::with_capacity(k_steps);

                for _ in 0..k_steps {
                    let ctx_len = draft_seq.len().min(draft_model.max_seq_len);
                    let start = draft_seq.len() - ctx_len;
                    let inp = Tensor::from_slice(&draft_seq[start..], &[1, ctx_len])?;
                    let logits = draft_model.forward(&inp)?;
                    let last_logits = logits.select(0, 0)?.select(0, (ctx_len - 1) as isize)?;
                    let probs = last_logits.softmax(-1)?;
                    let tok = probs.multinomial(1, true)?.item::<i64>()?;
                    draft_probs_list.push(probs.to_vec::<f64>()?);
                    draft_tokens.push(tok);
                    draft_seq.push(tok);
                }
                total_draft += k_steps;

                // 2. Target model evaluates prompt + draft_tokens in a single parallel forward pass!
                let full_len = draft_seq.len().min(self.max_seq_len);
                let full_start = draft_seq.len() - full_len;
                let target_inp = Tensor::from_slice(&draft_seq[full_start..], &[1, full_len])?;
                let target_logits = self.forward(&target_inp)?.select(0, 0)?;

                // Position in target_logits corresponding to the last token of `seq` (before draft tokens)
                let base_pos = full_len - 1 - k_steps;
                let mut all_accepted = true;

                for i in 0..k_steps {
                    if seq.len() >= target_len {
                        all_accepted = false;
                        break;
                    }
                    let cand = draft_tokens[i];
                    let cand_idx = cand as usize;
                    let p_vec = target_logits
                        .select(0, (base_pos + i) as isize)?
                        .softmax(-1)?
                        .to_vec::<f64>()?;
                    let q_vec = &draft_probs_list[i];

                    let p_x = p_vec.get(cand_idx).copied().unwrap_or(0.0);
                    let q_x = q_vec.get(cand_idx).copied().unwrap_or(1e-12).max(1e-12);
                    let u = crate::rng::with_rng(|r| r.next_f32() as f64);

                    if u < (p_x / q_x).min(1.0) {
                        seq.push(cand);
                        accepted_draft += 1;
                    } else {
                        // Sample replacement from residual distribution max(0, p - q)
                        let mut residual: Vec<f64> = p_vec
                            .iter()
                            .zip(q_vec.iter())
                            .map(|(&p, &q)| (p - q).max(0.0))
                            .collect();
                        let sum_r: f64 = residual.iter().sum();
                        let resampled = if sum_r > 1e-12 {
                            for r in &mut residual {
                                *r /= sum_r;
                            }
                            Tensor::from_slice(&residual, &[residual.len()])?
                                .multinomial(1, true)?
                                .item::<i64>()?
                        } else {
                            Tensor::from_slice(&p_vec, &[p_vec.len()])?
                                .multinomial(1, true)?
                                .item::<i64>()?
                        };
                        seq.push(resampled);
                        all_accepted = false;
                        break;
                    }
                }

                // 3. If all k_steps draft tokens were accepted and we still have budget, sample 1 bonus token!
                if all_accepted && seq.len() < target_len {
                    let bonus_probs = target_logits
                        .select(0, (base_pos + k_steps) as isize)?
                        .softmax(-1)?;
                    let bonus_tok = bonus_probs.multinomial(1, true)?.item::<i64>()?;
                    seq.push(bonus_tok);
                }
            }

            Ok((
                seq.into_iter().map(|t| t as usize).collect(),
                accepted_draft,
                total_draft,
            ))
        })
    }

    /// Beam Search decoding maintaining top `beam_width` hypotheses scored by
    /// length-normalized log-probability $\frac{\log P(Y)}{|Y|^{\text{length\_penalty}}}$.
    pub fn beam_search(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        beam_width: usize,
        length_penalty: f64,
    ) -> Result<Vec<(Vec<usize>, f64)>> {
        if prompt.is_empty() {
            return Err(crate::error::Error::Msg(
                "beam_search: prompt must not be empty".into(),
            ));
        }
        // `max_seq_len` is pub: a tampered 0 underflowed `ctx_len - 1` below.
        if self.max_seq_len < 1 {
            return Err(crate::error::Error::Msg(
                "beam_search: max_seq_len must be at least 1".into(),
            ));
        }
        // Each step costs `bw` full forwards; each requested token costs a step.
        crate::limits::check_iterations("max_new_tokens", max_new_tokens)?;
        let bw = beam_width.max(1);
        // Quadratic reserve (`beams * bw` candidates, each a cloned prefix):
        // 4096^2 already nears a gigabyte, and real beams are dozens wide.
        crate::limits::check_at_most("beam_width", bw, 4096)?;
        if let Some(&bad) = prompt.iter().find(|&&t| t > i64::MAX as usize) {
            return Err(crate::error::Error::Msg(format!(
                "beam_search: prompt token id {bad} exceeds i64::MAX"
            )));
        }
        crate::tensor::no_grad(|| {
            let init_seq: Vec<i64> = prompt.iter().map(|&t| t as i64).collect();
            let mut beams: Vec<(Vec<i64>, f64)> = vec![(init_seq, 0.0)];

            for _step in 0..max_new_tokens {
                // Quadratic in `bw`: reserve fallibly so vocab-scale widths
                // Err instead of aborting (the per-step cap above bounds time).
                let cap = beams.len().checked_mul(bw).ok_or_else(|| {
                    crate::error::Error::Msg("beam_search: candidate count overflows".into())
                })?;
                let mut candidates: Vec<(Vec<i64>, f64)> = Vec::new();
                candidates.try_reserve(cap).map_err(|_| {
                    crate::error::Error::Msg("beam_search: candidate reserve exceeds memory".into())
                })?;
                for (seq, log_prob) in &beams {
                    let ctx_len = seq.len().min(self.max_seq_len);
                    let start = seq.len() - ctx_len;
                    let inp = Tensor::from_slice(&seq[start..], &[1, ctx_len])?;
                    let logits = self.forward(&inp)?;
                    let last_log_probs = logits
                        .select(0, 0)?
                        .select(0, (ctx_len - 1) as isize)?
                        .log_softmax(-1)?;
                    let (top_vals, top_idx) = last_log_probs.topk(bw, -1, true, true)?;
                    let vals = top_vals.to_vec::<f64>()?;
                    let idxs = top_idx.to_vec::<i64>()?;

                    for (&v, &tok) in vals.iter().zip(&idxs) {
                        let mut next_seq = seq.clone();
                        next_seq.push(tok);
                        candidates.push((next_seq, log_prob + v));
                    }
                }

                candidates.sort_by(|(s1, lp1), (s2, lp2)| {
                    let score1 = lp1 / (s1.len() as f64).powf(length_penalty);
                    let score2 = lp2 / (s2.len() as f64).powf(length_penalty);
                    score2
                        .partial_cmp(&score1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(bw);
                beams = candidates;
            }

            Ok(beams
                .into_iter()
                .map(|(s, lp)| (s.into_iter().map(|t| t as usize).collect(), lp))
                .collect())
        })
    }
}

impl Module for CausalLM {
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        if tokens.numel() == 0 {
            return Err(Error::Msg(
                "CausalLM::forward: input sequence cannot be empty".into(),
            ));
        }
        let mut h = self.tok_emb.forward(tokens)?;
        for blk in &self.blocks {
            h = blk.forward_rope(&h, &self.rope)?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
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
        for (k, v) in self.norm.named_parameters() {
            out.push((format!("norm.{k}"), v));
        }
        for (k, v) in self.lm_head.named_parameters() {
            out.push((format!("lm_head.{k}"), v));
        }
        out
    }
}

pub(crate) fn causal_mask_with_offset(
    q_len: usize,
    k_len: usize,
    offset: usize,
    dev: &crate::backend::Device,
) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(q_len * k_len);
    for qi in 0..q_len {
        let abs_q = offset + qi;
        for ki in 0..k_len {
            mask.push(ki > abs_q);
        }
    }
    Tensor::from_slice_on(&mask, &[q_len, k_len], dev)
}

// ---------------------------------------------------------------------------
// Stateful Key-Value Cache (`LayerKVCache` & `KVCache`) and Grouped-Query Attention (`GQA`)
// ---------------------------------------------------------------------------

/// Per-layer Key-Value cache storing past `[batch, num_kv_heads, seq_len, head_dim]` tensors.
#[derive(Clone, Default)]
pub struct LayerKVCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn seq_len(&self) -> usize {
        // The cache slots are caller-visible (`pub k`); a malformed tensor must read as empty
        // here and fail closed in `append`'s cat, not panic on the rank indexing.
        self.k
            .as_ref()
            .and_then(|t| t.shape().get(2).copied())
            .unwrap_or(0)
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn append(&mut self, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor)> {
        let k_full = match self.k.take() {
            Some(prev_k) => Tensor::cat(&[prev_k, new_k], 2)?,
            None => new_k,
        };
        let v_full = match self.v.take() {
            Some(prev_v) => Tensor::cat(&[prev_v, new_v], 2)?,
            None => new_v,
        };
        self.k = Some(k_full.detach());
        self.v = Some(v_full.detach());
        Ok((k_full, v_full))
    }
}

/// Multi-layer Key-Value cache for [`CausalLM`] incremental decoding.
#[derive(Clone)]
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Result<Self> {
        // One slot per layer; an uncapped count aborted the collect below.
        crate::limits::check_repeats("num_layers", num_layers)?;
        Ok(Self {
            layers: (0..num_layers).map(|_| LayerKVCache::new()).collect(),
        })
    }

    pub fn seq_len(&self) -> usize {
        self.layers.first().map_or(0, LayerKVCache::seq_len)
    }

    pub fn clear(&mut self) {
        for l in &mut self.layers {
            l.clear();
        }
    }
}

/// Grouped-Query Attention (`GQA` / `MQA` used in LLaMA-3, Mistral, and Qwen)
/// where `num_kv_heads` divides `num_heads`.
pub struct GroupedQueryAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GroupedQueryAttention {
    pub fn new(d_model: usize, num_heads: usize, num_kv_heads: usize) -> Result<Self> {
        check_heads(d_model, num_heads)?;
        if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
            return Err(crate::error::Error::Msg(format!(
                "num_heads ({num_heads}) must be divisible by a positive num_kv_heads ({num_kv_heads})"
            )));
        }
        let head_dim = d_model / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        Ok(Self {
            q_proj: Linear::with_bias(d_model, d_model, false)?,
            k_proj: Linear::with_bias(d_model, kv_dim, false)?,
            v_proj: Linear::with_bias(d_model, kv_dim, false)?,
            o_proj: Linear::with_bias(d_model, d_model, false)?,
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn forward_gqa(
        &self,
        x: &Tensor,
        rope: Option<&RotaryEmbedding>,
        cache: Option<&mut LayerKVCache>,
        causal: bool,
    ) -> Result<Tensor> {
        check_bld("forward_gqa", x)?;
        let (b, l, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.num_heads;
        let kv_h = self.num_kv_heads;
        // Head counts are pub: a tampered 0 divided by zero below (`new()`
        // checks divisibility at construction only).
        if kv_h == 0 || h % kv_h != 0 {
            return Err(Error::Msg(format!(
                "forward_gqa: num_heads ({h}) must be divisible by a positive num_kv_heads ({kv_h})"
            )));
        }
        let d = self.head_dim;
        let group_size = h / kv_h;
        let offset = cache.as_ref().map_or(0, |c| c.seq_len());

        let mut q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let mut k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, l, kv_h, d])?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, l, kv_h, d])?
            .transpose(1, 2)?;

        if let Some(r) = rope {
            q = r.apply(&q, offset)?;
            k = r.apply(&k, offset)?;
        }

        let (k, v) = match cache {
            Some(c) => c.append(k, v)?,
            None => (k, v),
        };

        let total_k = k.shape()[2];
        // Zero-copy repeat of KV heads across `group_size` query heads!
        let (k_rep, v_rep) = if group_size > 1 {
            let kr = k
                .unsqueeze(2)?
                .expand(&[b, kv_h, group_size, total_k, d])?
                .reshape(&[b, h, total_k, d])?;
            let vr = v
                .unsqueeze(2)?
                .expand(&[b, kv_h, group_size, total_k, d])?
                .reshape(&[b, h, total_k, d])?;
            (kr, vr)
        } else {
            (k, v)
        };

        let scale = 1.0 / (d as f64).sqrt();
        let mut scores = q.matmul(&k_rep.t()?)?.mul_scalar(scale)?;
        if causal && l > 1 {
            let mask = causal_mask_with_offset(l, total_k, offset, &scores.device())?;
            scores = scores.masked_fill(&mask, -1e9)?;
        }
        let attn = scores.softmax(-1)?;
        let ctx = attn
            .matmul(&v_rep)?
            .transpose(1, 2)?
            .reshape(&[b, l, self.d_model])?;
        self.o_proj.forward(&ctx)
    }
}

impl Module for GroupedQueryAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_gqa(input, None, None, true)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("o_proj", &self.o_proj),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

/// Tiled online-softmax FlashAttention-2 (`[B, H, L_q, D]`).
///
/// Computes exact scaled dot-product attention by iterating over query blocks of size
/// `block_size_q` and key/value blocks of size `block_size_kv`, maintaining the running
/// row-maximum $m_i$ and running partition sum $\ell_i$ via the Milakov–Gimelshein online
/// softmax recurrence without materializing the full $O(L_q L_{kv})$ attention matrix.
/// When `causal = true`, future KV tiles are skipped entirely.
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    block_size_q: usize,
    block_size_kv: usize,
    causal: bool,
) -> Result<Tensor> {
    if q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 {
        return Err(Error::Msg(
            "flash_attention expects 4D [batch, heads, seq_len, head_dim] tensors".into(),
        ));
    }
    let (b, h, l_q, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let l_kv = k.shape()[2];
    if l_q == 0 {
        return Err(Error::Msg("flash_attention: empty query sequence".into()));
    }
    let bq = block_size_q.max(1);
    let bkv = block_size_kv.max(1);
    let scale = 1.0 / (d as f64).sqrt();
    let offset = l_kv.saturating_sub(l_q);

    let mut q_outputs = Vec::new();
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
            // In causal mode, if the entire KV block is strictly after the last query in this Q tile, skip!
            let max_valid_kv = offset + q_start + q_len - 1;
            if causal && kv_start > max_valid_kv {
                break;
            }

            let k_tile = k.narrow(2, kv_start, kv_len)?;
            let v_tile = v.narrow(2, kv_start, kv_len)?;
            let mut s_ij = q_tile.matmul(&k_tile.t()?)?.mul_scalar(scale)?;

            // Apply causal mask only on boundary tiles where kv_start + kv_len - 1 > offset + q_start
            if causal && (kv_start + kv_len - 1 > offset + q_start) {
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
            match (m_running.take(), l_running.take(), o_running.take()) {
                (None, None, None) => {
                    let p_tilde = s_ij.sub(&m_tile)?.exp()?;
                    let l_new = p_tilde.sum_axis(-1, true)?;
                    let o_new = p_tilde.matmul(&v_tile)?;
                    m_running = Some(m_tile);
                    l_running = Some(l_new);
                    o_running = Some(o_new);
                }
                (Some(m_prev), Some(l_prev), Some(o_prev)) => {
                    let m_new = Tensor::where_cond(&m_prev.gt(&m_tile)?, &m_prev, &m_tile)?;
                    let alpha = m_prev.sub(&m_new)?.exp()?;
                    let p_tilde = s_ij.sub(&m_new)?.exp()?;
                    let l_new = alpha.mul(&l_prev)?.add(&p_tilde.sum_axis(-1, true)?)?;
                    let o_new = alpha.mul(&o_prev)?.add(&p_tilde.matmul(&v_tile)?)?;
                    m_running = Some(m_new);
                    l_running = Some(l_new);
                    o_running = Some(o_new);
                }
                _ => {
                    return Err(Error::Msg(
                        "flash_attention: softmax running-stat invariant violated".into(),
                    ));
                }
            }

            kv_start += kv_len;
        }

        let o_tile = match (o_running, l_running) {
            (Some(o_acc), Some(l_acc)) => o_acc.div(&l_acc)?,
            _ => Tensor::zeros(&[b, h, q_len, d], q.dtype(), &q.device())?,
        };
        q_outputs.push(o_tile);
        q_start += q_len;
    }

    Tensor::cat(&q_outputs, 2)
}

// ---------------------------------------------------------------------------
// DeepSeek-V2 / V3 Multi-Head Latent Attention (`MLA`) & Differential Attention (`DiffAttention`)
// ---------------------------------------------------------------------------

/// DeepSeek-V2 / V3 **Multi-Head Latent Attention (`MLA`)**:
/// Jointly compresses Keys and Values into a low-rank latent vector $c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_{kv\_lora}}$
/// ($d_{kv\_lora} \ll H \cdot d_h$) and optionally compresses Queries via $c_t^Q = W^{DQ} h_t$,
/// dramatically shrinking the KV-cache memory footprint while preserving full multi-head capacity.
pub struct MultiHeadLatentAttention {
    pub q_down_proj: Linear,
    pub q_up_proj: Linear,
    pub kv_down_proj: Linear,
    pub k_up_proj: Linear,
    pub v_up_proj: Linear,
    pub o_proj: Linear,
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
}

impl MultiHeadLatentAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        kv_lora_rank: usize,
        q_lora_rank: usize,
    ) -> Result<Self> {
        check_heads(d_model, num_heads)?;
        let head_dim = d_model / num_heads;
        let kvr = kv_lora_rank.max(1);
        let qr = q_lora_rank.max(1);
        Ok(Self {
            q_down_proj: Linear::with_bias(d_model, qr, false)?,
            q_up_proj: Linear::with_bias(qr, d_model, false)?,
            kv_down_proj: Linear::with_bias(d_model, kvr, false)?,
            k_up_proj: Linear::with_bias(kvr, d_model, false)?,
            v_up_proj: Linear::with_bias(kvr, d_model, false)?,
            o_proj: Linear::with_bias(d_model, d_model, false)?,
            d_model,
            num_heads,
            head_dim,
            kv_lora_rank: kvr,
            q_lora_rank: qr,
        })
    }

    /// Ratio of uncompressed MHA KV-cache size ($2 \cdot d_{\text{model}}$) to MLA's compressed
    /// latent $c_t^{KV}$ cache size ($d_{\text{kv\_lora}}$).
    pub fn kv_cache_compression_ratio(&self) -> f64 {
        (2 * self.d_model) as f64 / (self.kv_lora_rank as f64)
    }

    pub fn forward_mla(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        check_bld("forward_mla", x)?;
        let (b, l, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.num_heads;
        let d = self.head_dim;

        let c_q = self.q_down_proj.forward(x)?;
        let q = self
            .q_up_proj
            .forward(&c_q)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;

        let c_kv = self.kv_down_proj.forward(x)?;
        let k = self
            .k_up_proj
            .forward(&c_kv)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let v = self
            .v_up_proj
            .forward(&c_kv)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;

        let ctx = flash_attention(&q, &k, &v, 32, 32, causal)?
            .transpose(1, 2)?
            .reshape(&[b, l, self.d_model])?;
        self.o_proj.forward(&ctx)
    }

    /// DeepSeek Latent KV Absorption for fast single-token auto-regressive decoding.
    ///
    /// Absorbs $W_{UK}$ into the Query vector and $W_{UV}$ into the attention output projection,
    /// enabling attention dot-products directly against compressed latents $c^{KV}$ without
    /// decompressing past tokens or expanding the KV cache into high-dimensional memory.
    ///
    /// - `x`: `[batch_size, 1, d_model]` single-step query input.
    /// - `latent_kv_cache`: in/out `[batch_size, seq_len, kv_lora_rank]` compressed latent cache.
    pub fn step_absorbed(
        &self,
        x: &Tensor,
        latent_kv_cache: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        check_bld("step_absorbed", x)?;
        let (b, l, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if l != 1 {
            return Err(Error::Msg(
                "step_absorbed is optimized for single-token decoding (l == 1)".into(),
            ));
        }
        let h = self.num_heads;
        let d = self.head_dim;
        let kvr = self.kv_lora_rank;

        // 1. Compute current compressed KV latent: [b, 1, kvr]
        let c_kv_new = self.kv_down_proj.forward(x)?;

        // Update latent KV cache: [b, seq_len, kvr]
        let c_kv_all = match latent_kv_cache {
            Some(prev) => Tensor::cat(&[prev.clone(), c_kv_new], 1)?,
            None => c_kv_new,
        };
        *latent_kv_cache = Some(c_kv_all.clone());

        // 2. Compute query: [b, 1, h, d] -> [b, h, 1, d]
        let c_q = self.q_down_proj.forward(x)?;
        let q = self
            .q_up_proj
            .forward(&c_q)?
            .reshape(&[b, 1, h, d])?
            .transpose(1, 2)?;

        // 3. Absorbed Query: q_tilde = q @ W_UK
        // W_UK weight is [h * d, kvr]. Reshaped to [h, d, kvr].
        let w_uk = self.k_up_proj.weight.reshape(&[h, d, kvr])?;
        // W_UV weight is [h * d, kvr]. Reshaped to [h, d, kvr].
        let w_uv = self.v_up_proj.weight.reshape(&[h, d, kvr])?;

        let scale = 1.0 / (d as f64).sqrt();
        let mut head_outputs = Vec::with_capacity(h);

        for head_idx in 0..h {
            // q_head: [b, 1, d]
            let q_head = q.narrow(1, head_idx, 1)?.squeeze_axis(1)?;
            // w_uk_head: [d, kvr]
            let w_uk_head = w_uk.narrow(0, head_idx, 1)?.squeeze_axis(0)?;
            // q_tilde: [b, 1, d] @ [d, kvr] -> [b, 1, kvr]
            let q_tilde = q_head.matmul(&w_uk_head)?;

            // Attention scores directly against compressed latents c_kv_all: [b, seq_len, kvr]
            // [b, 1, kvr] @ [b, kvr, seq_len] -> [b, 1, seq_len]
            let c_kv_t = c_kv_all.transpose(1, 2)?;
            let scores = q_tilde.matmul(&c_kv_t)?.mul_scalar(scale)?;
            let attn_weights = scores.softmax(2)?;

            // Absorbed value accumulation: [b, 1, seq_len] @ [b, seq_len, kvr] -> [b, 1, kvr]
            let u_head = attn_weights.matmul(&c_kv_all)?;

            // Project through W_UV^T: [b, 1, kvr] @ [kvr, d] -> [b, 1, d]
            // w_uv_head is [d, kvr], so transpose is [kvr, d]
            let w_uv_head_t = w_uv.narrow(0, head_idx, 1)?.squeeze_axis(0)?.t()?;
            let out_head = u_head.matmul(&w_uv_head_t)?;
            head_outputs.push(out_head);
        }

        // Concatenate across head dim: [b, 1, h * d]
        let cat_out = Tensor::cat(&head_outputs, 2)?;
        self.o_proj.forward(&cat_out)
    }
}

impl Module for MultiHeadLatentAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_mla(input, true)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("q_down_proj", &self.q_down_proj),
            ("q_up_proj", &self.q_up_proj),
            ("kv_down_proj", &self.kv_down_proj),
            ("k_up_proj", &self.k_up_proj),
            ("v_up_proj", &self.v_up_proj),
            ("o_proj", &self.o_proj),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

/// **Differential Attention** (`DiffAttention`, Ye et al. 2024 — Differential Transformer):
/// Computes two parallel softmax attention maps $A_1, A_2$ per head and subtracts them via a
/// learnable scalar $\lambda$ to cancel common-mode attention noise:
/// $$\operatorname{DiffAttn}(X) = \left(\operatorname{softmax}\!\left(\frac{Q_1 K_1^T}{\sqrt{d}}\right) - \lambda \operatorname{softmax}\!\left(\frac{Q_2 K_2^T}{\sqrt{d}}\right)\right) V$$
pub struct DiffAttention {
    pub q1_proj: Linear,
    pub q2_proj: Linear,
    pub k1_proj: Linear,
    pub k2_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub lambda_param: Tensor,
    pub lambda_init: f64,
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl DiffAttention {
    pub fn new(d_model: usize, num_heads: usize, layer_idx: usize) -> Result<Self> {
        check_heads(d_model, num_heads)?;
        let head_dim = d_model / num_heads;
        let lambda_init = 0.8 - 0.6 * (-(0.3 * (layer_idx as f64))).exp();
        let lambda_param = Tensor::from_slice(&[0.0f32], &[1])?.requires_grad_(true);
        Ok(Self {
            q1_proj: Linear::with_bias(d_model, d_model, false)?,
            q2_proj: Linear::with_bias(d_model, d_model, false)?,
            k1_proj: Linear::with_bias(d_model, d_model, false)?,
            k2_proj: Linear::with_bias(d_model, d_model, false)?,
            v_proj: Linear::with_bias(d_model, d_model, false)?,
            o_proj: Linear::with_bias(d_model, d_model, false)?,
            lambda_param,
            lambda_init,
            d_model,
            num_heads,
            head_dim,
        })
    }

    pub fn forward_diff(&self, x: &Tensor, causal: bool) -> Result<Tensor> {
        check_bld("forward_diff", x)?;
        let (b, l, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.num_heads;
        let d = self.head_dim;
        let scale = 1.0 / (d as f64).sqrt();

        let q1 = self
            .q1_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let q2 = self
            .q2_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let k1 = self
            .k1_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let k2 = self
            .k2_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;

        let mut s1 = q1.matmul(&k1.t()?)?.mul_scalar(scale)?;
        let mut s2 = q2.matmul(&k2.t()?)?.mul_scalar(scale)?;
        if causal && l > 1 {
            let mask = causal_mask_with_offset(l, l, 0, &s1.device())?;
            s1 = s1.masked_fill(&mask, -1e9)?;
            s2 = s2.masked_fill(&mask, -1e9)?;
        }

        let a1 = s1.softmax(-1)?;
        let a2 = s2.softmax(-1)?;
        let lambda = self
            .lambda_param
            .to_dtype(x.dtype())?
            .to_device(&x.device())?
            .tanh()?
            .mul_scalar(0.2)?
            .add_scalar(self.lambda_init)?
            .broadcast_to(a2.shape())?;

        let diff_attn = a1.sub(&a2.mul(&lambda)?)?;
        let ctx = diff_attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape(&[b, l, self.d_model])?;
        self.o_proj.forward(&ctx)
    }
}

impl Module for DiffAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_diff(input, true)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![("lambda_param".into(), self.lambda_param.clone())];
        for (p, m) in [
            ("q1_proj", &self.q1_proj),
            ("q2_proj", &self.q2_proj),
            ("k1_proj", &self.k1_proj),
            ("k2_proj", &self.k2_proj),
            ("v_proj", &self.v_proj),
            ("o_proj", &self.o_proj),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

/// **ALiBi (Attention with Linear Biases, Press et al., ICLR 2022)**:
/// Computes additive distance penalty tensor `[1, num_heads, q_len, k_len]` with geometric head slopes
/// $m_h = 2^{-8 h / H}$.
pub fn alibi_bias(
    num_heads: usize,
    q_len: usize,
    k_len: usize,
    device: &crate::backend::Device,
) -> Result<Tensor> {
    let step = 8.0 / (num_heads.max(1) as f64);
    let n = crate::limits::check_elements("alibi_bias", &[num_heads, q_len, k_len])?;
    let mut data = crate::layout::try_filled(n, 0.0f32)?;
    let offset = (k_len as isize) - (q_len as isize);
    for h in 0..num_heads {
        let slope = 2.0f64.powf(-((h + 1) as f64) * step) as f32;
        for qi in 0..q_len {
            let q_abs = (qi as isize) + offset;
            for ki in 0..k_len {
                let dist = (ki as isize) - q_abs;
                let val = if dist > 0 {
                    -1e9f32
                } else {
                    (dist as f32) * slope
                };
                data[(h * q_len + qi) * k_len + ki] = val;
            }
        }
    }
    Tensor::from_slice_on(&data, &[1, num_heads, q_len, k_len], device)
}

/// **Sliding Window Causal Attention with Attention Sinks** (Mistral-7B / Gemma-2 / StreamingLLM):
/// Each query at position $i$ attends to the first `sink_tokens` global prefix tokens $[0, \text{sink})$
/// plus the local sliding window $[\max(0, i - W + 1), i]$, bounding KV complexity to $O(W + S)$.
pub struct SlidingWindowAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub window_size: usize,
    pub sink_tokens: usize,
}

impl SlidingWindowAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        window_size: usize,
        sink_tokens: usize,
    ) -> Result<Self> {
        check_heads(d_model, num_heads)?;
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj: Linear::with_bias(d_model, d_model, false)?,
            k_proj: Linear::with_bias(d_model, d_model, false)?,
            v_proj: Linear::with_bias(d_model, d_model, false)?,
            o_proj: Linear::with_bias(d_model, d_model, false)?,
            d_model,
            num_heads,
            head_dim,
            window_size: window_size.max(1),
            sink_tokens,
        })
    }
}

impl Module for SlidingWindowAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        check_bld("SlidingWindowAttention", x)?;
        let (b, l, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.num_heads;
        let d = self.head_dim;
        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose(1, 2)?;

        let mut scores = q.matmul(&k.t()?)?.mul_scalar(1.0 / (d as f64).sqrt())?;
        let mut mask_bits = vec![0u8; l * l];
        // window_size is a pub field: clamp at use so a zeroed value degrades to 1, not underflow.
        let w = self.window_size.max(1);
        for qi in 0..l {
            for ki in 0..l {
                let is_future = ki > qi;
                let in_sink = ki < self.sink_tokens;
                let in_window = qi.saturating_sub(w - 1) <= ki && ki <= qi;
                if is_future || (!in_sink && !in_window) {
                    mask_bits[qi * l + ki] = 1;
                }
            }
        }
        let mask =
            Tensor::from_slice_on(&mask_bits, &[1, 1, l, l], &x.device())?.to_dtype(DType::Bool)?;
        scores = scores.masked_fill(&mask, -1e9)?;
        let attn = scores.softmax(-1)?;
        let ctx = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape(&[b, l, self.d_model])?;
        self.o_proj.forward(&ctx)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("q_proj", &self.q_proj),
            ("k_proj", &self.k_proj),
            ("v_proj", &self.v_proj),
            ("o_proj", &self.o_proj),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_window_rejects_zero_heads() {
        let err = match SlidingWindowAttention::new(8, 0, 4, 0) {
            Ok(_) => panic!("zero heads must be an error, not a div-by-zero panic"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("num_heads"), "{err}");
    }

    #[test]
    fn sliding_window_zeroed_window_size_does_not_panic() -> Result<()> {
        let mut attn = SlidingWindowAttention::new(8, 2, 4, 1)?;
        attn.window_size = 0; // pub field: must degrade to window 1, not underflow
        let x = Tensor::zeros_f32(&[1, 3, 8])?;
        assert_eq!(attn.forward(&x)?.shape(), &[1, 3, 8]);
        Ok(())
    }

    #[test]
    fn alibi_bias_rejects_overflowing_dims() -> Result<()> {
        let dev = crate::backend::Device::Cpu;
        let err = alibi_bias(1 << 22, 1 << 22, 1 << 22, &dev)
            .expect_err("overflowing dims must be an error, not a panic");
        assert!(err.to_string().contains("alibi"), "{err}");
        assert_eq!(alibi_bias(2, 3, 3, &dev)?.shape(), &[1, 2, 3, 3]);
        Ok(())
    }

    #[test]
    fn layer_kv_cache_malformed_slot_reads_empty_and_fails_closed() -> Result<()> {
        let mut c = LayerKVCache::new();
        c.k = Some(Tensor::from_slice(&[1.0f32, 2.0], &[1, 2])?);
        assert_eq!(c.seq_len(), 0); // must not panic on rank-2 indexing
        let err = c
            .append(
                Tensor::zeros_f32(&[1, 1, 1, 2])?,
                Tensor::zeros_f32(&[1, 1, 1, 2])?,
            )
            .expect_err("rank-mismatched append must fail closed");
        assert!(err.to_string().contains("cat"), "{err}");
        Ok(())
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= tol, "{what} element {i}: {g} vs {w}");
        }
    }

    /// Full-matrix SDPA reference through basic ops (independent path from the tiled kernel).
    fn naive_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
        let d = q.shape()[3] as f64;
        let mut s = q.matmul(&k.t()?)?.mul_scalar(1.0 / d.sqrt())?;
        if causal {
            let lq = q.shape()[2];
            let lkv = k.shape()[2];
            let off = lkv.saturating_sub(lq);
            let mut mask = vec![false; lq * lkv];
            for qi in 0..lq {
                for kj in 0..lkv {
                    if kj > off + qi {
                        mask[qi * lkv + kj] = true;
                    }
                }
            }
            let mt = Tensor::from_slice_on(&mask, &[lq, lkv], &s.device())?;
            s = s.masked_fill(&mt, f64::NEG_INFINITY)?;
        }
        s.softmax(-1)?.matmul(v)
    }

    fn arange4(b: usize, h: usize, l: usize, d: usize, start: f64) -> Result<Tensor> {
        let n = b * h * l * d;
        let v: Vec<f64> = (0..n).map(|i| start + i as f64 * 0.5).collect();
        Tensor::from_slice(&v, &[b, h, l, d])
    }

    #[test]
    fn flash_attention_matches_torch() -> Result<()> {
        // all values torch-verified (f64, F.scaled_dot_product_attention)
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
        let k = Tensor::from_slice(&[1.0, 0.0, 1.0, 1.0], &[1, 1, 2, 2])?;
        let v = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        for (bq, bkv) in [(1, 1), (2, 2), (8, 8)] {
            let y = flash_attention(&q, &k, &v, bq, bkv, false)?.to_vec::<f64>()?;
            assert_close(
                &y,
                &[2.0, 3.0, 2.3395230986533138, 3.3395230986533138],
                1e-9,
                "sdpa",
            );
            let y = flash_attention(&q, &k, &v, bq, bkv, true)?.to_vec::<f64>()?;
            assert_close(
                &y,
                &[1.0, 2.0, 2.3395230986533138, 3.3395230986533138],
                1e-9,
                "causal",
            );
        }
        // sharp softmax rows (e^26 ratios) across tile boundaries
        let q3 = Tensor::from_slice(
            &(1..10).map(|i| i as f64).collect::<Vec<_>>(),
            &[1, 1, 3, 3],
        )?;
        let k3 = Tensor::from_slice(
            &(10..19).map(|i| i as f64).collect::<Vec<_>>(),
            &[1, 1, 3, 3],
        )?;
        let v3 = Tensor::from_slice(
            &(19..28).map(|i| i as f64).collect::<Vec<_>>(),
            &[1, 1, 3, 3],
        )?;
        for (bq, bkv) in [(1, 1), (2, 1), (3, 3)] {
            let y = flash_attention(&q3, &k3, &v3, bq, bkv, true)?.to_vec::<f64>()?;
            assert_close(
                &y,
                &[
                    19.0,
                    20.0,
                    21.0,
                    21.999999999984375,
                    22.999999999984375,
                    23.999999999984375,
                    25.0,
                    26.0,
                    27.0,
                ],
                1e-9,
                "sharp",
            );
        }
        Ok(())
    }

    #[test]
    fn flash_attention_matches_naive_across_blocks_and_shapes() -> Result<()> {
        // tiled online recurrence == full-matrix softmax, including non-divisible
        // blocks, KV prefixes (decode), single queries, and single keys.
        for (lq, lkv) in [(5, 5), (3, 7), (1, 4), (4, 1), (6, 6)] {
            let q = arange4(2, 2, lq, 4, 0.0)?;
            let k = arange4(2, 2, lkv, 4, 1.0)?;
            let v = arange4(2, 2, lkv, 4, 2.0)?;
            for causal in [false, true] {
                let want = naive_sdpa(&q, &k, &v, causal)?.to_vec::<f64>()?;
                for (bq, bkv) in [(1, 1), (2, 3), (8, 8)] {
                    let got = flash_attention(&q, &k, &v, bq, bkv, causal)?.to_vec::<f64>()?;
                    assert_close(
                        &got,
                        &want,
                        1e-9,
                        &format!("lq={lq} lkv={lkv} causal={causal} b={bq},{bkv}"),
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn flash_attention_causal_never_peeks_future() -> Result<()> {
        // zeroing KV rows 1.. must leave query-0 output bit-identical (it only
        // attends row 0) while moving query 3 (sensitivity control).
        let q = arange4(1, 2, 4, 4, 0.0)?;
        let k = arange4(1, 2, 4, 4, 1.0)?;
        let v = arange4(1, 2, 4, 4, 2.0)?;
        let row_mask = Tensor::from_slice(&[false, true, true, true], &[4, 1])?;
        for (bq, bkv) in [(1, 1), (2, 2), (3, 1)] {
            let base = flash_attention(&q, &k, &v, bq, bkv, true)?;
            let kz = k.masked_fill(&row_mask, 0.0)?;
            let vz = v.masked_fill(&row_mask, 0.0)?;
            let got = flash_attention(&q, &kz, &vz, bq, bkv, true)?;
            let b0 = base.narrow(2, 0, 1)?.to_vec::<f64>()?;
            let g0 = got.narrow(2, 0, 1)?.to_vec::<f64>()?;
            assert_close(&g0, &b0, 1e-12, "no-peek q0");
            let b3 = base.narrow(2, 3, 1)?.to_vec::<f64>()?;
            let g3 = got.narrow(2, 3, 1)?.to_vec::<f64>()?;
            let moved: f64 = b3.iter().zip(&g3).map(|(a, b)| (a - b).abs()).sum();
            assert!(moved > 1e-6, "perturbation must move q3 (moved {moved})");
        }
        Ok(())
    }

    #[test]
    fn flash_attention_grouped_query_broadcasts() -> Result<()> {
        // H_q=2 against H_kv=1 broadcasts heads like manual repetition.
        let q = arange4(1, 2, 3, 4, 0.0)?;
        let k1 = arange4(1, 1, 3, 4, 1.0)?;
        let v1 = arange4(1, 1, 3, 4, 2.0)?;
        let k2 = Tensor::cat(&[k1.clone(), k1.clone()], 1)?;
        let v2 = Tensor::cat(&[v1.clone(), v1.clone()], 1)?;
        for causal in [false, true] {
            let got = flash_attention(&q, &k1, &v1, 2, 2, causal)?.to_vec::<f64>()?;
            let want = naive_sdpa(&q, &k2, &v2, causal)?.to_vec::<f64>()?;
            assert_close(&got, &want, 1e-9, "gqa");
        }
        Ok(())
    }

    #[test]
    fn flash_attention_geometry_edges() -> Result<()> {
        let q = arange4(1, 1, 2, 2, 0.0)?;
        let k = arange4(1, 1, 2, 2, 1.0)?;
        let v = arange4(1, 1, 2, 2, 2.0)?;
        // empty KV attends to nothing: zeros, matching torch (verified).
        let ke = Tensor::from_slice(&[] as &[f64], &[1, 1, 0, 2])?;
        let ve = Tensor::from_slice(&[] as &[f64], &[1, 1, 0, 2])?;
        let y = flash_attention(&q, &ke, &ve, 2, 2, false)?.to_vec::<f64>()?;
        assert_close(&y, &[0.0, 0.0, 0.0, 0.0], 0.0, "empty-kv");
        // empty queries, wrong rank, and head-dim mismatch fail closed.
        let qe = Tensor::from_slice(&[] as &[f64], &[1, 1, 0, 2])?;
        let err = flash_attention(&qe, &k, &v, 2, 2, false).unwrap_err();
        assert!(err.to_string().contains("query"), "{err}");
        let flat = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        assert!(flash_attention(&flat, &k, &v, 2, 2, false).is_err());
        let kd = arange4(1, 1, 2, 3, 1.0)?;
        assert!(flash_attention(&q, &kd, &v, 2, 2, false).is_err());
        Ok(())
    }
}
