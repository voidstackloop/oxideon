//! PyTorch-compatible attention and Transformer modules: [`MultiheadAttention`] with `key_padding_mask`, `attn_mask`,
//! `need_weights`, `kdim` / `vdim`, `batch_first` and attention dropout; [`TransformerEncoderLayerExt`] /
//! [`TransformerDecoderLayerExt`] with `norm_first` and selectable activation; the stacks
//! [`TransformerEncoderStack`] / [`TransformerDecoderStack`]; and the full encoder-decoder [`Transformer`] with
//! [`generate_square_subsequent_mask`].
//!
//! Mask conventions follow `torch.nn`: a boolean mask marks the positions that must **not** be attended to
//! (`true` = masked); a float mask is added to the attention scores.

use super::{Dropout, LayerNorm, Linear, Module};
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Additive causal mask `[sz, sz]`: `-inf` above the diagonal, `0` elsewhere
/// (`torch.nn.Transformer.generate_square_subsequent_mask`).
pub fn generate_square_subsequent_mask(sz: usize) -> Result<Tensor> {
    let v: Vec<f32> = (0..sz * sz)
        .map(|ix| if ix % sz > ix / sz { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    Tensor::from_slice(&v, &[sz, sz])
}

/// Adds `mask` (bool: true = masked, float: additive) to attention scores `[N, H, L, S]`.
/// `mask` may be `[L, S]`, `[N, H, L, S]`, or PyTorch's flattened `[N * H, L, S]`.
fn apply_attn_mask(scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (n, h, l, s) = (scores.shape()[0], scores.shape()[1], scores.shape()[2], scores.shape()[3]);
    let m = match mask.ndim() {
        2 => mask.reshape(&[1, 1, mask.shape()[0], mask.shape()[1]])?,
        3 if mask.shape()[0] == n * h => mask.reshape(&[n, h, l, s])?,
        4 => mask.clone(),
        _ => return Err(Error::Msg(format!("attn_mask: unsupported shape {:?}", mask.shape()))),
    };
    if m.dtype() == crate::dtype::DType::Bool {
        Tensor::where_cond(&m.broadcast_to(scores.shape())?, &scores.full_like(f64::NEG_INFINITY)?, scores)
    } else {
        scores.add(&m.to_dtype(scores.dtype())?)
    }
}

/// Multi-head attention with PyTorch's argument set.
pub struct MultiheadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub batch_first: bool,
    pub dropout: Dropout,
}

impl MultiheadAttention {
    /// `kdim` / `vdim` default to `embed_dim`.
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout_p: f32,
        bias: bool,
        kdim: Option<usize>,
        vdim: Option<usize>,
        batch_first: bool,
    ) -> Result<Self> {
        if num_heads == 0 || embed_dim % num_heads != 0 {
            return Err(Error::Msg(format!("embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")));
        }
        Ok(Self {
            q_proj: Linear::with_bias(embed_dim, embed_dim, bias)?,
            k_proj: Linear::with_bias(kdim.unwrap_or(embed_dim), embed_dim, bias)?,
            v_proj: Linear::with_bias(vdim.unwrap_or(embed_dim), embed_dim, bias)?,
            out_proj: Linear::with_bias(embed_dim, embed_dim, bias)?,
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
            batch_first,
            dropout: Dropout::new(dropout_p),
        })
    }

    /// Attention of `query` over `key` / `value`. Inputs are `[L, N, E]` (or `[N, L, E]` with `batch_first`, or
    /// unbatched `[L, E]`). `key_padding_mask` is `[N, S]`. Returns the output and, if `need_weights`, the attention
    /// weights `[N, L, S]` averaged over heads (or `[N, H, L, S]` when `average_attn_weights` is false).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_full(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_padding_mask: Option<&Tensor>,
        need_weights: bool,
        attn_mask: Option<&Tensor>,
        average_attn_weights: bool,
        is_causal: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let unbatched = query.ndim() == 2;
        let to_nle = |t: &Tensor| -> Result<Tensor> {
            if unbatched {
                t.unsqueeze(0)
            } else if self.batch_first {
                Ok(t.clone())
            } else {
                t.transpose(0, 1)
            }
        };
        let (q_in, k_in, v_in) = (to_nle(query)?, to_nle(key)?, to_nle(value)?);
        let (n, l, s) = (q_in.shape()[0], q_in.shape()[1], k_in.shape()[1]);
        let (h, hd) = (self.num_heads, self.head_dim);
        let heads = |t: Tensor, len: usize| -> Result<Tensor> { t.reshape(&[n, len, h, hd])?.transpose(1, 2) };
        let q = heads(self.q_proj.forward(&q_in)?, l)?;
        let k = heads(self.k_proj.forward(&k_in)?, s)?;
        let v = heads(self.v_proj.forward(&v_in)?, s)?;
        let mut scores = q.matmul(&k.transpose(-2, -1)?)?.mul_scalar(1.0 / (hd as f64).sqrt())?;
        if is_causal && attn_mask.is_none() {
            let m: Vec<f32> = (0..l * s).map(|ix| if ix % s > ix / s { f32::NEG_INFINITY } else { 0.0 }).collect();
            scores = apply_attn_mask(&scores, &Tensor::from_slice(&m, &[l, s])?)?;
        }
        if let Some(m) = attn_mask {
            scores = apply_attn_mask(&scores, m)?;
        }
        if let Some(kp) = key_padding_mask {
            let kp = if unbatched { kp.unsqueeze(0)? } else { kp.clone() };
            if kp.shape() != [n, s] {
                return Err(Error::Msg(format!("key_padding_mask must be [N, S] = [{n}, {s}], got {:?}", kp.shape())));
            }
            let kp4 = kp.reshape(&[n, 1, 1, s])?;
            scores = if kp4.dtype() == crate::dtype::DType::Bool {
                Tensor::where_cond(&kp4.broadcast_to(scores.shape())?, &scores.full_like(f64::NEG_INFINITY)?, &scores)?
            } else {
                scores.add(&kp4.to_dtype(scores.dtype())?)?
            };
        }
        let weights = scores.softmax(-1)?;
        let ctx = self.dropout.forward(&weights)?.matmul(&v)?; // [N, H, L, hd]
        let merged = ctx.transpose(1, 2)?.contiguous()?.reshape(&[n, l, self.embed_dim])?;
        let out = self.out_proj.forward(&merged)?;
        let out = if unbatched {
            out.squeeze_axis(0)?
        } else if self.batch_first {
            out
        } else {
            out.transpose(0, 1)?
        };
        let w = if need_weights {
            let w = if average_attn_weights { weights.mean_axis(1, false)? } else { weights };
            Some(if unbatched { w.squeeze_axis(0)? } else { w })
        } else {
            None
        };
        Ok((out, w))
    }

    /// Self-attention shortcut.
    pub fn forward_self(&self, x: &Tensor, key_padding_mask: Option<&Tensor>, attn_mask: Option<&Tensor>, is_causal: bool) -> Result<Tensor> {
        Ok(self.forward_full(x, x, x, key_padding_mask, false, attn_mask, true, is_causal)?.0)
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_self(input, None, None, false)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [("q_proj", &self.q_proj), ("k_proj", &self.k_proj), ("v_proj", &self.v_proj), ("out_proj", &self.out_proj)] {
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

/// Feed-forward activation of the Transformer layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnActivation {
    Relu,
    Gelu,
}

fn activate(a: FfnActivation, x: &Tensor) -> Result<Tensor> {
    match a {
        FfnActivation::Relu => x.relu(),
        FfnActivation::Gelu => x.gelu(),
    }
}

struct Ffn {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
    activation: FfnActivation,
}

impl Ffn {
    fn new(d_model: usize, d_ff: usize, dropout_p: f32, activation: FfnActivation) -> Result<Self> {
        Ok(Self { linear1: Linear::new(d_model, d_ff)?, linear2: Linear::new(d_ff, d_model)?, dropout: Dropout::new(dropout_p), activation })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear2.forward(&self.dropout.forward(&activate(self.activation, &self.linear1.forward(x)?)?)?)
    }
    fn params(&self, prefix: &str, out: &mut Vec<(String, Tensor)>) {
        for (p, m) in [("linear1", &self.linear1), ("linear2", &self.linear2)] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{prefix}{p}.{k}"), v));
            }
        }
    }
}

fn norm_params(prefix: &str, n: &LayerNorm, out: &mut Vec<(String, Tensor)>) {
    for (k, v) in n.named_parameters() {
        out.push((format!("{prefix}.{k}"), v));
    }
}

fn attn_params(prefix: &str, a: &MultiheadAttention, out: &mut Vec<(String, Tensor)>) {
    for (k, v) in a.named_parameters() {
        out.push((format!("{prefix}.{k}"), v));
    }
}

/// `torch.nn.TransformerEncoderLayer`: self-attention + feed-forward with residuals, post-norm by default or
/// pre-norm with `norm_first`.
pub struct TransformerEncoderLayerExt {
    pub self_attn: MultiheadAttention,
    ffn: Ffn,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    dropout1: Dropout,
    dropout2: Dropout,
    pub norm_first: bool,
}

impl TransformerEncoderLayerExt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout_p: f32,
        activation: FfnActivation,
        norm_first: bool,
        batch_first: bool,
        layer_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MultiheadAttention::new(d_model, nhead, dropout_p, true, None, None, batch_first)?,
            ffn: Ffn::new(d_model, dim_feedforward, dropout_p, activation)?,
            norm1: LayerNorm::new(d_model, layer_norm_eps)?,
            norm2: LayerNorm::new(d_model, layer_norm_eps)?,
            dropout1: Dropout::new(dropout_p),
            dropout2: Dropout::new(dropout_p),
            norm_first,
        })
    }

    pub fn forward_masked(
        &self,
        src: &Tensor,
        src_mask: Option<&Tensor>,
        src_key_padding_mask: Option<&Tensor>,
        is_causal: bool,
    ) -> Result<Tensor> {
        let sa = |x: &Tensor| -> Result<Tensor> {
            self.dropout1.forward(&self.self_attn.forward_self(x, src_key_padding_mask, src_mask, is_causal)?)
        };
        let ff = |x: &Tensor| -> Result<Tensor> { self.dropout2.forward(&self.ffn.forward(x)?) };
        if self.norm_first {
            let x = src.add(&sa(&self.norm1.forward(src)?)?)?;
            x.add(&ff(&self.norm2.forward(&x)?)?)
        } else {
            let x = self.norm1.forward(&src.add(&sa(src)?)?)?;
            self.norm2.forward(&x.add(&ff(&x)?)?)
        }
    }
}

impl Module for TransformerEncoderLayerExt {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_masked(x, None, None, false)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        attn_params("self_attn", &self.self_attn, &mut out);
        self.ffn.params("", &mut out);
        norm_params("norm1", &self.norm1, &mut out);
        norm_params("norm2", &self.norm2, &mut out);
        out
    }
    fn train(&self, mode: bool) {
        self.self_attn.train(mode);
        self.ffn.dropout.train(mode);
        self.dropout1.train(mode);
        self.dropout2.train(mode);
    }
}

/// `torch.nn.TransformerDecoderLayer`: masked self-attention, cross-attention over the memory, feed-forward.
pub struct TransformerDecoderLayerExt {
    pub self_attn: MultiheadAttention,
    pub cross_attn: MultiheadAttention,
    ffn: Ffn,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    dropout1: Dropout,
    dropout2: Dropout,
    dropout3: Dropout,
    pub norm_first: bool,
}

impl TransformerDecoderLayerExt {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        dropout_p: f32,
        activation: FfnActivation,
        norm_first: bool,
        batch_first: bool,
        layer_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: MultiheadAttention::new(d_model, nhead, dropout_p, true, None, None, batch_first)?,
            cross_attn: MultiheadAttention::new(d_model, nhead, dropout_p, true, None, None, batch_first)?,
            ffn: Ffn::new(d_model, dim_feedforward, dropout_p, activation)?,
            norm1: LayerNorm::new(d_model, layer_norm_eps)?,
            norm2: LayerNorm::new(d_model, layer_norm_eps)?,
            norm3: LayerNorm::new(d_model, layer_norm_eps)?,
            dropout1: Dropout::new(dropout_p),
            dropout2: Dropout::new(dropout_p),
            dropout3: Dropout::new(dropout_p),
            norm_first,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_masked(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
        tgt_key_padding_mask: Option<&Tensor>,
        memory_key_padding_mask: Option<&Tensor>,
        tgt_is_causal: bool,
    ) -> Result<Tensor> {
        let sa = |x: &Tensor| -> Result<Tensor> {
            self.dropout1.forward(&self.self_attn.forward_self(x, tgt_key_padding_mask, tgt_mask, tgt_is_causal)?)
        };
        let ca = |x: &Tensor| -> Result<Tensor> {
            self.dropout2.forward(&self.cross_attn.forward_full(x, memory, memory, memory_key_padding_mask, false, memory_mask, true, false)?.0)
        };
        let ff = |x: &Tensor| -> Result<Tensor> { self.dropout3.forward(&self.ffn.forward(x)?) };
        if self.norm_first {
            let x = tgt.add(&sa(&self.norm1.forward(tgt)?)?)?;
            let x = x.add(&ca(&self.norm2.forward(&x)?)?)?;
            x.add(&ff(&self.norm3.forward(&x)?)?)
        } else {
            let x = self.norm1.forward(&tgt.add(&sa(tgt)?)?)?;
            let x = self.norm2.forward(&x.add(&ca(&x)?)?)?;
            self.norm3.forward(&x.add(&ff(&x)?)?)
        }
    }

    fn params(&self, out: &mut Vec<(String, Tensor)>) {
        attn_params("self_attn", &self.self_attn, out);
        attn_params("cross_attn", &self.cross_attn, out);
        self.ffn.params("", out);
        norm_params("norm1", &self.norm1, out);
        norm_params("norm2", &self.norm2, out);
        norm_params("norm3", &self.norm3, out);
    }

    fn set_train(&self, mode: bool) {
        self.self_attn.train(mode);
        self.cross_attn.train(mode);
        self.ffn.dropout.train(mode);
        self.dropout1.train(mode);
        self.dropout2.train(mode);
        self.dropout3.train(mode);
    }
}

/// `torch.nn.TransformerEncoder`: a stack of encoder layers with an optional final norm.
pub struct TransformerEncoderStack {
    pub layers: Vec<TransformerEncoderLayerExt>,
    pub norm: Option<LayerNorm>,
}

impl TransformerEncoderStack {
    pub fn forward_masked(&self, src: &Tensor, mask: Option<&Tensor>, src_key_padding_mask: Option<&Tensor>, is_causal: bool) -> Result<Tensor> {
        let mut x = src.clone();
        for l in &self.layers {
            x = l.forward_masked(&x, mask, src_key_padding_mask, is_causal)?;
        }
        match &self.norm {
            Some(n) => n.forward(&x),
            None => Ok(x),
        }
    }
}

impl Module for TransformerEncoderStack {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_masked(x, None, None, false)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (k, v) in l.named_parameters() {
                out.push((format!("layers.{i}.{k}"), v));
            }
        }
        if let Some(n) = &self.norm {
            norm_params("norm", n, &mut out);
        }
        out
    }
    fn train(&self, mode: bool) {
        self.layers.iter().for_each(|l| l.train(mode));
    }
}

/// `torch.nn.TransformerDecoder`: a stack of decoder layers with an optional final norm.
pub struct TransformerDecoderStack {
    pub layers: Vec<TransformerDecoderLayerExt>,
    pub norm: Option<LayerNorm>,
}

impl TransformerDecoderStack {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_masked(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
        tgt_key_padding_mask: Option<&Tensor>,
        memory_key_padding_mask: Option<&Tensor>,
        tgt_is_causal: bool,
    ) -> Result<Tensor> {
        let mut x = tgt.clone();
        for l in &self.layers {
            x = l.forward_masked(&x, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_is_causal)?;
        }
        match &self.norm {
            Some(n) => n.forward(&x),
            None => Ok(x),
        }
    }

    fn params(&self, out: &mut Vec<(String, Tensor)>) {
        for (i, l) in self.layers.iter().enumerate() {
            let mut inner = Vec::new();
            l.params(&mut inner);
            for (k, v) in inner {
                out.push((format!("layers.{i}.{k}"), v));
            }
        }
        if let Some(n) = &self.norm {
            norm_params("norm", n, out);
        }
    }
}

/// Configuration of [`Transformer`].
#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub d_model: usize,
    pub nhead: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub dim_feedforward: usize,
    pub dropout: f32,
    pub activation: FfnActivation,
    pub norm_first: bool,
    pub batch_first: bool,
    pub layer_norm_eps: f64,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            nhead: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            dim_feedforward: 2048,
            dropout: 0.1,
            activation: FfnActivation::Relu,
            norm_first: false,
            batch_first: false,
            layer_norm_eps: 1e-5,
        }
    }
}

/// The full encoder-decoder Transformer (`torch.nn.Transformer`).
pub struct Transformer {
    pub encoder: TransformerEncoderStack,
    pub decoder: TransformerDecoderStack,
    pub config: TransformerConfig,
}

impl Transformer {
    pub fn new(cfg: TransformerConfig) -> Result<Self> {
        let enc = (0..cfg.num_encoder_layers)
            .map(|_| TransformerEncoderLayerExt::new(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation, cfg.norm_first, cfg.batch_first, cfg.layer_norm_eps))
            .collect::<Result<Vec<_>>>()?;
        let dec = (0..cfg.num_decoder_layers)
            .map(|_| TransformerDecoderLayerExt::new(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout, cfg.activation, cfg.norm_first, cfg.batch_first, cfg.layer_norm_eps))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            encoder: TransformerEncoderStack { layers: enc, norm: Some(LayerNorm::new(cfg.d_model, cfg.layer_norm_eps)?) },
            decoder: TransformerDecoderStack { layers: dec, norm: Some(LayerNorm::new(cfg.d_model, cfg.layer_norm_eps)?) },
            config: cfg,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_full(
        &self,
        src: &Tensor,
        tgt: &Tensor,
        src_mask: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
        src_key_padding_mask: Option<&Tensor>,
        tgt_key_padding_mask: Option<&Tensor>,
        memory_key_padding_mask: Option<&Tensor>,
        tgt_is_causal: bool,
    ) -> Result<Tensor> {
        let memory = self.encoder.forward_masked(src, src_mask, src_key_padding_mask, false)?;
        self.decoder.forward_masked(tgt, &memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_is_causal)
    }

    /// `forward_full` with no masks.
    pub fn forward_pair(&self, src: &Tensor, tgt: &Tensor) -> Result<Tensor> {
        self.forward_full(src, tgt, None, None, None, None, None, None, false)
    }
}

impl Module for Transformer {
    /// Encodes and decodes the same sequence (`src == tgt`); use [`Transformer::forward_pair`] for the two-input form.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_pair(input, input)
    }
    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.encoder.named_parameters() {
            out.push((format!("encoder.{k}"), v));
        }
        let mut dec = Vec::new();
        self.decoder.params(&mut dec);
        for (k, v) in dec {
            out.push((format!("decoder.{k}"), v));
        }
        out
    }
    fn train(&self, mode: bool) {
        self.encoder.train(mode);
        self.decoder.layers.iter().for_each(|l| l.set_train(mode));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(shape: &[usize]) -> Tensor {
        Tensor::randn(shape).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }
    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        vals(a).iter().zip(vals(b)).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn causal_mask_values() {
        let m = vals(&generate_square_subsequent_mask(3).unwrap());
        assert_eq!(m[0], 0.0);
        assert_eq!(m[1], f32::NEG_INFINITY);
        assert_eq!(m[3], 0.0);
        assert_eq!(m[5], f32::NEG_INFINITY);
        assert_eq!(m[8], 0.0);
    }

    #[test]
    fn padding_mask_equals_truncation() {
        let mha = MultiheadAttention::new(8, 2, 0.0, true, None, None, true).unwrap();
        let q = rand(&[2, 3, 8]);
        let kv = rand(&[2, 5, 8]);
        // mask out the last two keys of every sequence
        let mask = Tensor::from_slice(&[false, false, false, true, true, false, false, false, true, true], &[2, 5]).unwrap();
        let (masked, w) = mha.forward_full(&q, &kv, &kv, Some(&mask), true, None, true, false).unwrap();
        let short = kv.narrow(1, 0, 3).unwrap().contiguous().unwrap();
        let (trunc, _) = mha.forward_full(&q, &short, &short, None, false, None, true, false).unwrap();
        assert!(max_diff(&masked, &trunc) < 1e-5, "{}", max_diff(&masked, &trunc));
        let w = w.unwrap();
        assert_eq!(w.shape(), [2, 3, 5]);
        let wv = vals(&w);
        for row in 0..6 {
            assert!(wv[row * 5 + 3] < 1e-7 && wv[row * 5 + 4] < 1e-7);
            assert!((wv[row * 5..row * 5 + 5].iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        // per-head weights
        let (_, ph) = mha.forward_full(&q, &kv, &kv, None, true, None, false, false).unwrap();
        assert_eq!(ph.unwrap().shape(), [2, 2, 3, 5]);
    }

    #[test]
    fn causal_attention_does_not_look_ahead() {
        let mha = MultiheadAttention::new(8, 2, 0.0, true, None, None, true).unwrap();
        let x = rand(&[1, 4, 8]);
        let y = mha.forward_self(&x, None, None, true).unwrap();
        // perturb the last two positions; the first two outputs must not change
        let tail = rand(&[1, 2, 8]);
        let x2 = Tensor::cat(&[x.narrow(1, 0, 2).unwrap().contiguous().unwrap(), tail], 1).unwrap();
        let y2 = mha.forward_self(&x2, None, None, true).unwrap();
        assert!(max_diff(&y.narrow(1, 0, 2).unwrap().contiguous().unwrap(), &y2.narrow(1, 0, 2).unwrap().contiguous().unwrap()) < 1e-6);
        assert!(max_diff(&y.narrow(1, 2, 2).unwrap().contiguous().unwrap(), &y2.narrow(1, 2, 2).unwrap().contiguous().unwrap()) > 1e-4);
        // an explicit additive mask gives the same result as is_causal
        let m = generate_square_subsequent_mask(4).unwrap();
        assert!(max_diff(&y, &mha.forward_self(&x, None, Some(&m), false).unwrap()) < 1e-6);
        // a boolean mask of the same pattern too
        let bm = Tensor::from_slice(&(0..16).map(|ix| ix % 4 > ix / 4).collect::<Vec<bool>>(), &[4, 4]).unwrap();
        assert!(max_diff(&y, &mha.forward_self(&x, None, Some(&bm), false).unwrap()) < 1e-6);
    }

    #[test]
    fn layouts_agree() {
        let a = MultiheadAttention::new(8, 2, 0.0, true, None, None, true).unwrap();
        let b = MultiheadAttention::new(8, 2, 0.0, true, None, None, false).unwrap();
        for ((_, pa), (_, pb)) in a.named_parameters().iter().zip(b.named_parameters()) {
            pb.assign(pa).unwrap();
        }
        let x = rand(&[2, 3, 8]);
        let ya = a.forward_self(&x, None, None, false).unwrap();
        let yb = b.forward_self(&x.transpose(0, 1).unwrap().contiguous().unwrap(), None, None, false).unwrap();
        assert!(max_diff(&ya, &yb.transpose(0, 1).unwrap().contiguous().unwrap()) < 1e-5);
        // unbatched input
        let yu = a.forward_self(&x.narrow(0, 0, 1).unwrap().reshape(&[3, 8]).unwrap(), None, None, false).unwrap();
        assert!(max_diff(&yu, &ya.narrow(0, 0, 1).unwrap().reshape(&[3, 8]).unwrap()) < 1e-5);
        // kdim / vdim projections
        let cross = MultiheadAttention::new(8, 2, 0.0, false, Some(6), Some(5), true).unwrap();
        let out = cross.forward_full(&rand(&[2, 3, 8]), &rand(&[2, 4, 6]), &rand(&[2, 4, 5]), None, false, None, true, false).unwrap().0;
        assert_eq!(out.shape(), [2, 3, 8]);
    }

    #[test]
    fn encoder_layer_ignores_padded_positions_and_trains() {
        for norm_first in [false, true] {
            let layer = TransformerEncoderLayerExt::new(8, 2, 16, 0.0, FfnActivation::Gelu, norm_first, true, 1e-5).unwrap();
            let x = rand(&[1, 4, 8]);
            let pad = Tensor::from_slice(&[false, false, true, true], &[1, 4]).unwrap();
            let y = layer.forward_masked(&x, None, Some(&pad), false).unwrap();
            // changing the padded tokens does not change the real positions
            let x2 = Tensor::cat(&[x.narrow(1, 0, 2).unwrap().contiguous().unwrap(), rand(&[1, 2, 8])], 1).unwrap();
            let y2 = layer.forward_masked(&x2, None, Some(&pad), false).unwrap();
            assert!(max_diff(&y.narrow(1, 0, 2).unwrap().contiguous().unwrap(), &y2.narrow(1, 0, 2).unwrap().contiguous().unwrap()) < 1e-5, "norm_first={norm_first}");
            // every parameter gets a gradient
            y.sum().unwrap().backward().unwrap();
            for (name, p) in layer.named_parameters() {
                assert!(p.grad().is_some(), "{name} has no gradient");
            }
        }
    }

    #[test]
    fn full_transformer() {
        let cfg = TransformerConfig { d_model: 8, nhead: 2, num_encoder_layers: 2, num_decoder_layers: 2, dim_feedforward: 16, dropout: 0.0, batch_first: true, ..Default::default() };
        let t = Transformer::new(cfg).unwrap();
        let (src, tgt) = (rand(&[2, 5, 8]), rand(&[2, 3, 8]));
        let causal = generate_square_subsequent_mask(3).unwrap();
        let out = t.forward_full(&src, &tgt, None, Some(&causal), None, None, None, None, false).unwrap();
        assert_eq!(out.shape(), [2, 3, 8]);
        // the decoder is causal: changing the last target token leaves earlier outputs unchanged
        let tgt2 = Tensor::cat(&[tgt.narrow(1, 0, 2).unwrap().contiguous().unwrap(), rand(&[2, 1, 8])], 1).unwrap();
        let out2 = t.forward_full(&src, &tgt2, None, Some(&causal), None, None, None, None, false).unwrap();
        assert!(max_diff(&out.narrow(1, 0, 2).unwrap().contiguous().unwrap(), &out2.narrow(1, 0, 2).unwrap().contiguous().unwrap()) < 1e-5);
        // parameter count: encoder layer = attn (4*(64+8)) + ffn (8*16+16 + 16*8+8) + 2 norms (2*16); decoder adds a second attention + norm
        let attn = 4 * (64 + 8);
        let ffn = 8 * 16 + 16 + 16 * 8 + 8;
        let enc = attn + ffn + 2 * 16;
        let dec = 2 * attn + ffn + 3 * 16;
        assert_eq!(t.num_parameters(), 2 * enc + 2 * dec + 2 * 16);
        t.eval();
        let again = t.forward_full(&src, &tgt, None, Some(&causal), None, None, None, None, false).unwrap();
        let again2 = t.forward_full(&src, &tgt, None, Some(&causal), None, None, None, None, false).unwrap();
        assert!(max_diff(&again, &again2) < 1e-6);
        out.sum().unwrap().backward().unwrap();
        assert!(t.parameters().iter().all(|p| p.grad().is_some()));
    }
}
