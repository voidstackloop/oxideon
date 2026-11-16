//! # Oxideon
//!
//! A complete, multi-dtype, strided-view deep learning and tensor library in Rust.
//!
//! ## Subsystems
//! - **Zero-copy Strided Views** ([`layout`], [`indexing`]): Signed `isize` strides + offset over
//!   shared [`Storage`]. `permute`, `transpose`, `t`, `movedim`, `broadcast_to`/`expand` (`stride = 0`),
//!   `narrow`, stepped & negative-step slicing (`t.i((..))` & `s![..]`), `flip`, `select`, `squeeze`,
//!   `unsqueeze`, `diagonal`, and `as_strided` are all zero-copy views with invertible [`ViewSpec`]
//!   higher-order autograd.
//! - **8 DTypes** ([`dtype`]): `Bool`, `U8`, `I32`, `I64`, `F16`, `BF16`, `F32`, `F64` with pure-Rust
//!   IEEE-754 [`struct@f16`] and [`bf16`].
//! - **Blocked SIMD GEMM & Pluggable Backends** ([`gemm`], [`backend`]): Cache-blocked panel-packed
//!   strided GEMM with `8x8` AVX2+FMA microkernels and `rayon` multi-threading, plus an optional
//!   `wgpu` WGSL compute-shader GPU backend (`--features wgpu`).
//! - **Higher-Order Autograd** ([`tensor`]): Thread-safe `Arc` `Tensor` with `Tensor`-level backward
//!   rules (`create_graph = true` for 2nd/3rd derivatives, Hessians, JVP/VJP), hooks, `retain_grad`,
//!   [`CustomOp`], and activation [`checkpoint`]ing.
//! - **Linear Algebra & Tensor Ops** ([`ops`], [`linalg`]): 25+ unary/activations, broadcast binary &
//!   comparisons, multi-axis reductions, `matmul`, `einsum`, `cholesky`, `triangular_solve`, `solve`,
//!   `inv`, `det`, `slogdet`, `qr`, `norm_p`, `normalize`, `cdist`, `one_hot`, `multinomial`,
//!   `bernoulli`, `conv1d`/`conv2d`/`conv2d_grouped`, `conv_transpose1d`/`2d`, `upsample_nearest2d`,
//!   `upsample_bilinear2d`, `adaptive_avg_pool2d`/`adaptive_max_pool2d`.
//! - **Complete Neural Network Suite** ([`nn`]):
//!   - Core & Vision: `Linear`, `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`,
//!     `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`, `Upsample2d`,
//!     `ResidualBlock`, `Flatten`, `Dropout`, `Embedding`.
//!   - Normalization: `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm2d`.
//!   - Recurrent (`nn::rnn`): `RNNCell`, `LSTMCell`, `GRUCell`, multi-layer `RNN`, `LSTM`, `GRU`.
//!   - Transformers & LLM (`nn::transformer`): `MultiHeadAttention`, `SinusoidalPositionalEncoding`,
//!     `RotaryEmbedding` (RoPE), `SwiGLU`, `FeedForward`, `TransformerEncoder`, `TransformerDecoder`,
//!     and `CausalLM` (LLaMA/GPT-style decoder with greedy, temperature, top-k, and top-p generation).
//!   - Losses (`nn::loss`): `mse_loss`, `l1_loss`, `huber_loss`, `nll_loss`, `cross_entropy`,
//!     `cross_entropy_with_smoothing`, `binary_cross_entropy_with_logits`, `focal_loss`,
//!     `kl_div_loss`, `cosine_similarity`, `cosine_embedding_loss`, `triplet_margin_loss`, `info_nce_loss`.
//! - **Optimizers, Data Pipeline, Metrics & Safetensors** ([`optim`], [`data`], [`metrics`], [`serialize`]):
//!   `Sgd`, `Adam`, `AdamW`, `RmsProp`, `Adagrad`, `Lion`; `StepLr`, `CosineAnnealingLr`,
//!   `LinearWarmupCosineLr`, `ExponentialLr`, `ReduceLROnPlateau`; `Dataset`, `TensorDataset`,
//!   `DataLoader`, `Transform`, synthetic datasets (`make_moons`, `make_circles`, `make_blobs`);
//!   evaluation metrics (`accuracy`, `top_k_accuracy`, `precision_recall_f1`, `auroc`, `r2_score`,
//!   `perplexity`); and pure-Rust `.safetensors`.
//!
//! ```
//! use oxideon::prelude::*;
//!
//! fn main() -> Result<()> {
//!     let x = Tensor::from_slice(&[2.0f32], &[1])?.requires_grad_(true);
//!     let y = x.powf(3.0)?;
//!     let dy = grad(&[y], &[x.clone()], None, true)?.pop().unwrap().unwrap();
//!     assert_eq!(dy.to_vec::<f32>()?, vec![12.0]);
//!
//!     let d2y = grad(&[dy], &[x], None, false)?.pop().unwrap().unwrap();
//!     assert_eq!(d2y.to_vec::<f32>()?, vec![12.0]);
//!     Ok(())
//! }
//! ```

pub use oxideon_core::*;
pub use oxideon_engines::*;

/// Convenient glob import: `use oxideon::prelude::*;`
pub mod prelude {
    pub use oxideon_core::prelude::*;
    pub use oxideon_engines::prelude::*;
}

#[cfg(feature = "python")]
pub mod python;
#[cfg(feature = "python")]
pub mod python_flops;
#[cfg(feature = "python")]
pub mod python_hardware;
#[cfg(feature = "python")]
pub mod python_tokenizer;
#[cfg(feature = "python")]
pub mod python_interop;
#[cfg(feature = "python")]
pub mod python_millennium;
#[cfg(feature = "python")]
pub mod python_vision;
