//! Whisper Speech Foundation Model (Radford et al., OpenAI 2022).
//!
//! Complete audio-to-text sequence-to-sequence Transformer architecture:
//! - **Audio Encoder**: 2x 1D strided convolutional downsamplers + Transformer Encoder blocks
//! - **Text Decoder**: Causal text self-attention + cross-attention over audio encoder memory

use crate::error::{Error, Result};
use crate::nn::{
    Conv1d, Embedding, LayerNorm, Linear, Module, MultiHeadAttention, SinusoidalPositionalEncoding,
};
use crate::tensor::{Tensor, no_grad};

/// Whisper architecture configuration.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub n_mels: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_heads: usize,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub d_ff: usize,
    pub vocab_size: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
}

impl Default for WhisperConfig {
    /// Whisper Tiny configuration (39M parameters).
    fn default() -> Self {
        Self {
            n_mels: 80,
            d_model: 384,
            encoder_layers: 4,
            encoder_heads: 6,
            decoder_layers: 4,
            decoder_heads: 6,
            d_ff: 1536,
            vocab_size: 51865,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }
}

/// Whisper Audio Encoder.
pub struct AudioEncoder {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub pos_embed: SinusoidalPositionalEncoding,
    pub layers: Vec<AudioEncoderLayer>,
    pub ln_post: LayerNorm,
    pub d_model: usize,
}

pub struct AudioEncoderLayer {
    pub self_attn: MultiHeadAttention,
    pub ln1: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub ln2: LayerNorm,
}

impl AudioEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, num_heads)?,
            ln1: LayerNorm::new(d_model, 1e-5)?,
            fc1: Linear::new(d_model, d_ff)?,
            fc2: Linear::new(d_ff, d_model)?,
            ln2: LayerNorm::new(d_model, 1e-5)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-LN Transformer block
        let normed1 = self.ln1.forward(x)?;
        let attn_out = self
            .self_attn
            .forward_qkv(&normed1, &normed1, &normed1, false)?;
        let h = x.add(&attn_out)?;

        let normed2 = self.ln2.forward(&h)?;
        let ff_out = self.fc2.forward(&self.fc1.forward(&normed2)?.gelu()?)?;
        h.add(&ff_out)
    }
}

impl AudioEncoder {
    pub fn new(cfg: &WhisperConfig) -> Result<Self> {
        crate::limits::check_repeats("encoder_layers", cfg.encoder_layers)?;
        let conv1 = Conv1d::new(cfg.n_mels, cfg.d_model, 3, 1, 1)?;
        let conv2 = Conv1d::new(cfg.d_model, cfg.d_model, 3, 2, 1)?;
        let pos_embed = SinusoidalPositionalEncoding::new(cfg.max_source_positions, cfg.d_model)?;
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for _ in 0..cfg.encoder_layers {
            layers.push(AudioEncoderLayer::new(
                cfg.d_model,
                cfg.encoder_heads,
                cfg.d_ff,
            )?);
        }
        let ln_post = LayerNorm::new(cfg.d_model, 1e-5)?;

        Ok(Self {
            conv1,
            conv2,
            pos_embed,
            layers,
            ln_post,
            d_model: cfg.d_model,
        })
    }

    /// Encodes log-Mel spectrogram `[batch, n_mels, frames]` into speech features `[batch, frames/2, d_model]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // `conv1.weight` is pub: a rank-swapped weight panicked `shape()[1]`.
        let n_mels = self.conv1.weight.shape().get(1).copied().ok_or_else(|| {
            Error::Msg(format!(
                "AudioEncoder: conv1 weight has shape {:?}, need [out, in, k]",
                self.conv1.weight.shape()
            ))
        })?;
        if mel.ndim() != 3 || mel.shape()[1] != n_mels {
            return Err(Error::Msg(format!(
                "AudioEncoder expects [batch, {n_mels}, frames] mel input, got {:?}",
                mel.shape()
            )));
        }
        let c1 = self.conv1.forward(mel)?.gelu()?;
        let c2 = self.conv2.forward(&c1)?.gelu()?;
        // Permute to [batch, seq_len, d_model]
        let mut h = c2.permute(&[0, 2, 1])?;
        h = self.pos_embed.forward(&h)?;

        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        self.ln_post.forward(&h)
    }
}

/// Whisper Decoder Layer with cross-attention over audio representations.
pub struct AudioDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub ln1: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
    pub ln3: LayerNorm,
}

impl AudioDecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, num_heads)?,
            ln1: LayerNorm::new(d_model, 1e-5)?,
            cross_attn: MultiHeadAttention::new(d_model, num_heads)?,
            ln2: LayerNorm::new(d_model, 1e-5)?,
            fc1: Linear::new(d_model, d_ff)?,
            fc2: Linear::new(d_ff, d_model)?,
            ln3: LayerNorm::new(d_model, 1e-5)?,
        })
    }

    pub fn forward(&self, x: &Tensor, encoder_hidden: &Tensor, causal: bool) -> Result<Tensor> {
        // Causal self-attention
        let n1 = self.ln1.forward(x)?;
        let self_out = self.self_attn.forward_qkv(&n1, &n1, &n1, causal)?;
        let h1 = x.add(&self_out)?;

        // Cross-attention over audio encoder memory
        let n2 = self.ln2.forward(&h1)?;
        let cross_out = self
            .cross_attn
            .forward_qkv(&n2, encoder_hidden, encoder_hidden, false)?;
        let h2 = h1.add(&cross_out)?;

        // Feed-Forward
        let n3 = self.ln3.forward(&h2)?;
        let ff_out = self.fc2.forward(&self.fc1.forward(&n3)?.gelu()?)?;
        h2.add(&ff_out)
    }
}

/// Complete Whisper Speech-to-Text Foundation Model.
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub token_embedding: Embedding,
    pub pos_embed: SinusoidalPositionalEncoding,
    pub decoder_layers: Vec<AudioDecoderLayer>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub config: WhisperConfig,
}

impl Whisper {
    pub fn new(cfg: WhisperConfig) -> Result<Self> {
        crate::limits::check_repeats("decoder_layers", cfg.decoder_layers)?;
        let encoder = AudioEncoder::new(&cfg)?;
        let token_embedding = Embedding::new(cfg.vocab_size, cfg.d_model)?;
        let pos_embed = SinusoidalPositionalEncoding::new(cfg.max_target_positions, cfg.d_model)?;
        let mut decoder_layers = Vec::with_capacity(cfg.decoder_layers);
        for _ in 0..cfg.decoder_layers {
            decoder_layers.push(AudioDecoderLayer::new(
                cfg.d_model,
                cfg.decoder_heads,
                cfg.d_ff,
            )?);
        }
        let ln_post = LayerNorm::new(cfg.d_model, 1e-5)?;
        let lm_head = Linear::new(cfg.d_model, cfg.vocab_size)?;

        Ok(Self {
            encoder,
            token_embedding,
            pos_embed,
            decoder_layers,
            ln_post,
            lm_head,
            config: cfg,
        })
    }

    /// Forward pass: transforms audio Mel spectrogram and prompt text tokens into next-token logits.
    pub fn forward(&self, mel: &Tensor, text_tokens: &Tensor) -> Result<Tensor> {
        let enc_hidden = self.encoder.forward(mel)?;

        let mut h = self.token_embedding.forward(text_tokens)?;
        h = self.pos_embed.forward(&h)?;

        for layer in &self.decoder_layers {
            h = layer.forward(&h, &enc_hidden, true)?;
        }
        let normed = self.ln_post.forward(&h)?;
        self.lm_head.forward(&normed)
    }

    /// Autoregressive transcription loop from audio Mel spectrogram.
    pub fn transcribe(&self, mel: &Tensor, max_len: usize) -> Result<Vec<usize>> {
        no_grad(|| {
            let enc_hidden = self.encoder.forward(mel)?;
            let mut tokens = vec![1usize]; // Start of transcript token (SOT)

            for _ in 0..max_len {
                let seq_len = tokens.len();
                let tok_i64: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
                let tok_tensor = Tensor::from_slice_on(&tok_i64, &[1, seq_len], &mel.device())?;
                let mut h = self.token_embedding.forward(&tok_tensor)?;
                h = self.pos_embed.forward(&h)?;

                for layer in &self.decoder_layers {
                    h = layer.forward(&h, &enc_hidden, true)?;
                }
                let logits = self.lm_head.forward(&self.ln_post.forward(&h)?)?;
                // Greedy next token from last position
                let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze()?;
                let next_tok = last_logits.argmax_axis(-1, false)?.item::<i64>()? as usize;
                tokens.push(next_tok);
                if next_tok == 2 {
                    // EOS
                    break;
                }
            }
            Ok(tokens)
        })
    }
}

impl Module for Whisper {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        Err(crate::error::Error::Msg(
            "Whisper requires both audio and text inputs; use whisper.forward(mel, tokens)".into(),
        ))
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.encoder.conv1.named_parameters() {
            out.push((format!("encoder.conv1.{k}"), v));
        }
        for (k, v) in self.encoder.conv2.named_parameters() {
            out.push((format!("encoder.conv2.{k}"), v));
        }
        for (i, layer) in self.encoder.layers.iter().enumerate() {
            for (k, v) in layer.self_attn.named_parameters() {
                out.push((format!("encoder.layers.{i}.self_attn.{k}"), v));
            }
            for (k, v) in layer.ln1.named_parameters() {
                out.push((format!("encoder.layers.{i}.ln1.{k}"), v));
            }
            for (k, v) in layer.fc1.named_parameters() {
                out.push((format!("encoder.layers.{i}.fc1.{k}"), v));
            }
            for (k, v) in layer.fc2.named_parameters() {
                out.push((format!("encoder.layers.{i}.fc2.{k}"), v));
            }
            for (k, v) in layer.ln2.named_parameters() {
                out.push((format!("encoder.layers.{i}.ln2.{k}"), v));
            }
        }
        for (k, v) in self.encoder.ln_post.named_parameters() {
            out.push((format!("encoder.ln_post.{k}"), v));
        }
        for (k, v) in self.token_embedding.named_parameters() {
            out.push((format!("token_embedding.{k}"), v));
        }
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            for (k, v) in layer.self_attn.named_parameters() {
                out.push((format!("decoder_layers.{i}.self_attn.{k}"), v));
            }
            for (k, v) in layer.ln1.named_parameters() {
                out.push((format!("decoder_layers.{i}.ln1.{k}"), v));
            }
            for (k, v) in layer.cross_attn.named_parameters() {
                out.push((format!("decoder_layers.{i}.cross_attn.{k}"), v));
            }
            for (k, v) in layer.ln2.named_parameters() {
                out.push((format!("decoder_layers.{i}.ln2.{k}"), v));
            }
            for (k, v) in layer.fc1.named_parameters() {
                out.push((format!("decoder_layers.{i}.fc1.{k}"), v));
            }
            for (k, v) in layer.fc2.named_parameters() {
                out.push((format!("decoder_layers.{i}.fc2.{k}"), v));
            }
            for (k, v) in layer.ln3.named_parameters() {
                out.push((format!("decoder_layers.{i}.ln3.{k}"), v));
            }
        }
        for (k, v) in self.ln_post.named_parameters() {
            out.push((format!("ln_post.{k}"), v));
        }
        for (k, v) in self.lm_head.named_parameters() {
            out.push((format!("lm_head.{k}"), v));
        }
        out
    }
}
