//! Complete Neural Network Architectures Zoo:
//! - [`MLP`] (`MultiLayerPerceptron` with configurable [`ActivationKind`], `LayerNorm`, `Dropout`, and residual skip connections)
//! - [`SqueezeExcitation2d`] (`SEBlock`: channel attention recalibration from SENet / MobileNetV3 / EfficientNet)
//! - [`ConvNeXtBlock`] and [`ConvNeXt`] (Liu et al., CVPR 2022: depthwise $7\times 7$/$3\times 3$ conv + channel-last `LayerNorm` + $4\times$ inverted bottleneck + `GELU` + LayerScale + `DropPath`)
//! - [`BiLSTM`] (Bidirectional LSTM concatenating forward and time-reversed hidden states `[B, T, 2 * H]`)
//! - [`TCNBlock`] and [`TemporalConvNet`] (Causal 1D convolutional residual network for time-series and audio)
//! - [`MLPMixerBlock`] and [`MLPMixer`] (Tolstikhin et al., NeurIPS 2021: token-mixing + channel-mixing all-MLP architecture)
//! - [`VariationalAutoencoder`] (`VAE`: encoder $(\mu, \log\sigma^2)$ + reparameterization + decoder + ELBO loss)

use crate::backend::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::activations::DropPath;
use crate::nn::vision::{vae_kl_loss, vae_reparameterize};
use crate::nn::{
    AdaptiveAvgPool2d, Conv1d, Conv2d, Dropout, LSTM, LayerNorm, Linear, Module, PatchEmbedding,
    loss,
};
use crate::tensor::Tensor;

/// Selectable activation function for [`MLP`] and other configurable architectures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationKind {
    Identity,
    ReLU,
    GELU,
    SiLU,
    Mish,
    SELU,
    Tanh,
    Sigmoid,
    Hardswish,
}

impl ActivationKind {
    pub fn apply(self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Identity => Ok(x.clone()),
            Self::ReLU => x.relu(),
            Self::GELU => x.gelu(),
            Self::SiLU => x.silu(),
            Self::Mish => x.mish(),
            Self::SELU => x.selu(),
            Self::Tanh => x.tanh(),
            Self::Sigmoid => x.sigmoid(),
            Self::Hardswish => x.hardswish(),
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Configurable Multi-Layer Perceptron (`MLP` / `ResMLP`)
// ---------------------------------------------------------------------------

/// Configurable Multi-Layer Perceptron (`MLP`) supporting arbitrary hidden layer widths,
/// [`ActivationKind`], per-layer [`LayerNorm`], [`Dropout`], and residual skip connections
/// whenever consecutive layer dimensions match.
pub struct MLP {
    pub layers: Vec<Linear>,
    pub norms: Vec<Option<LayerNorm>>,
    pub dropouts: Vec<Dropout>,
    pub activation: ActivationKind,
    pub residual: bool,
}

impl MLP {
    pub fn new(
        dims: &[usize],
        activation: ActivationKind,
        use_layernorm: bool,
        dropout_p: f64,
        residual: bool,
    ) -> Result<Self> {
        if dims.len() < 2 {
            return Err(Error::Msg(
                "MLP requires at least 2 layer dimensions [in, ..., out]".into(),
            ));
        }
        // `Dropout::new` asserts on NaN; reject it here with an Err instead.
        if !dropout_p.is_finite() {
            return Err(Error::Msg("MLP dropout_p must be finite".into()));
        }
        let num_layers = dims.len() - 1;
        let mut layers = Vec::with_capacity(num_layers);
        let mut norms = Vec::with_capacity(num_layers);
        let mut dropouts = Vec::with_capacity(num_layers);

        for (i, w) in dims.windows(2).enumerate() {
            layers.push(Linear::new(w[0], w[1])?);
            if i + 1 < num_layers && use_layernorm {
                norms.push(Some(LayerNorm::new(w[1], 1e-5)?));
            } else {
                norms.push(None);
            }
            dropouts.push(Dropout::new(if i + 1 < num_layers {
                dropout_p as f32
            } else {
                0.0
            }));
        }
        Ok(Self {
            layers,
            norms,
            dropouts,
            activation,
            residual,
        })
    }
}

impl Module for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut h = input.clone();
        let n = self.layers.len();
        // Pub vecs: a truncated norms/dropouts list must Err, not panic on [i] below.
        if self.norms.len() != n || self.dropouts.len() != n {
            return Err(Error::Msg(format!(
                "MLP has {} layers but {} norms / {} dropouts",
                n,
                self.norms.len(),
                self.dropouts.len()
            )));
        }
        for i in 0..n {
            let prev = h.clone();
            let mut z = self.layers[i].forward(&h)?;
            if i + 1 < n {
                if let Some(norm) = &self.norms[i] {
                    z = norm.forward(&z)?;
                }
                z = self.activation.apply(&z)?;
                z = self.dropouts[i].forward(&z)?;
                if self.residual && prev.shape() == z.shape() {
                    z = z.add(&prev)?;
                }
            }
            h = z;
        }
        Ok(h)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (k, v) in l.named_parameters() {
                out.push((format!("layers.{i}.{k}"), v));
            }
            // Infallible listing: skip over truncated vecs instead of panicking.
            if let Some(Some(norm)) = self.norms.get(i) {
                for (k, v) in norm.named_parameters() {
                    out.push((format!("norms.{i}.{k}"), v));
                }
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        for d in &self.dropouts {
            d.train(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Squeeze-and-Excitation (`SqueezeExcitation2d`) & `ConvNeXt` (CVPR 2022)
// ---------------------------------------------------------------------------

/// Squeeze-and-Excitation 2D Channel Attention Block (Hu et al., CVPR 2018 / EfficientNet):
/// Pools `[N, C, H, W] -> [N, C]`, projects `C -> C / reduction -> C` with `SiLU` + `Sigmoid`,
/// and rescales each feature channel.
pub struct SqueezeExcitation2d {
    pub fc1: Linear,
    pub fc2: Linear,
    pub channels: usize,
}

impl SqueezeExcitation2d {
    pub fn new(channels: usize, reduction: usize) -> Result<Self> {
        let hidden = (channels / reduction.max(1)).max(1);
        Ok(Self {
            fc1: Linear::new(channels, hidden)?,
            fc2: Linear::new(hidden, channels)?,
            channels,
        })
    }
}

impl Module for SqueezeExcitation2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 4 {
            return Err(crate::error::Error::Msg(format!(
                "SqueezeExcitation2d expects 4D input [N, C, H, W], got {:?}",
                input.shape()
            )));
        }
        let (n, c, _, _) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        let pooled = input.mean_axes(&[2, 3], false)?;
        let scale = self
            .fc2
            .forward(&self.fc1.forward(&pooled)?.silu()?)?
            .sigmoid()?
            .reshape(&[n, c, 1, 1])?
            .broadcast_to(input.shape())?;
        input.mul(&scale)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.fc1.named_parameters() {
            out.push((format!("fc1.{k}"), v));
        }
        for (k, v) in self.fc2.named_parameters() {
            out.push((format!("fc2.{k}"), v));
        }
        out
    }
}

/// **ConvNeXt Block** (Liu et al., CVPR 2022):
/// 1. Depthwise $k\times k$ convolution (`groups = dim`)
/// 2. Permute `[N, C, H, W] -> [N, H, W, C]` + `LayerNorm(C)`
/// 3. Pointwise $4\times$ inverted bottleneck `Linear(C -> 4C)` + `GELU` + `Linear(4C -> C)`
/// 4. Learnable LayerScale $\gamma$ + `DropPath` + residual skip connection!
pub struct ConvNeXtBlock {
    pub dw_weight: Tensor,
    pub dw_bias: Tensor,
    pub norm: LayerNorm,
    pub pwconv1: Linear,
    pub pwconv2: Linear,
    pub gamma: Tensor,
    pub drop_path: DropPath,
    pub dim: usize,
    pub kernel_size: usize,
}

impl ConvNeXtBlock {
    pub fn new(
        dim: usize,
        kernel_size: usize,
        drop_path_prob: f64,
        layer_scale_init: f64,
    ) -> Result<Self> {
        let k = kernel_size;
        let k_sq = k
            .checked_mul(k)
            .ok_or_else(|| Error::Msg(format!("ConvNeXtBlock: kernel_size {k} * {k} overflows")))?;
        let dw_weight =
            crate::nn::init::kaiming_uniform(&[dim, 1, k, k], k_sq)?.requires_grad_(true);
        let dw_bias = Tensor::zeros(&[dim], DType::F32, &Device::Cpu)?.requires_grad_(true);
        let norm = LayerNorm::new(dim, 1e-6)?;
        let pwconv1 = Linear::new(dim, 4 * dim)?;
        let pwconv2 = Linear::new(4 * dim, dim)?;
        let gamma =
            Tensor::full(&[dim], layer_scale_init, DType::F32, &Device::Cpu)?.requires_grad_(true);
        Ok(Self {
            dw_weight,
            dw_bias,
            norm,
            pwconv1,
            pwconv2,
            gamma,
            drop_path: DropPath::new(drop_path_prob),
            dim,
            kernel_size: k,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 4 {
            return Err(crate::error::Error::Msg(format!(
                "ConvNeXtBlock expects 4D input [N, C, H, W], got {:?}",
                x.shape()
            )));
        }
        let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let pad = self.kernel_size / 2;
        let dw = x.conv2d_grouped(
            &self.dw_weight,
            Some(&self.dw_bias),
            (1, 1),
            (pad, pad),
            (1, 1),
            c,
        )?;
        // Channel-last [N, H, W, C]
        let ch_last = dw.permute(&[0, 2, 3, 1])?.contiguous()?;
        let normed = self.norm.forward(&ch_last)?;
        let hidden = self.pwconv1.forward(&normed)?.gelu()?;
        let mut out = self.pwconv2.forward(&hidden)?;
        let g = self
            .gamma
            .to_device(&x.device())?
            .to_dtype(x.dtype())?
            .reshape(&[1, 1, 1, c])?
            .broadcast_to(&[n, h, w, c])?;
        out = out.mul(&g)?;
        let ch_first = out.permute(&[0, 3, 1, 2])?.contiguous()?;
        x.add(&self.drop_path.forward(&ch_first)?)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = vec![
            ("dw_weight".into(), self.dw_weight.clone()),
            ("dw_bias".into(), self.dw_bias.clone()),
            ("gamma".into(), self.gamma.clone()),
        ];
        for (p, m_params) in [
            ("norm", self.norm.named_parameters()),
            ("pwconv1", self.pwconv1.named_parameters()),
            ("pwconv2", self.pwconv2.named_parameters()),
        ] {
            for (k, v) in m_params {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }

    fn train(&self, mode: bool) {
        self.drop_path.train(mode);
    }
}

/// Complete **ConvNeXt** Vision Classifier (patchify stem + `ConvNeXtBlock` stack + global pool + `LayerNorm` + `Linear` head).
pub struct ConvNeXt {
    pub stem: Conv2d,
    pub blocks: Vec<ConvNeXtBlock>,
    pub pool: AdaptiveAvgPool2d,
    pub head_norm: LayerNorm,
    pub head: Linear,
    pub dim: usize,
}

impl ConvNeXt {
    pub fn new(
        in_channels: usize,
        dim: usize,
        depth: usize,
        patch_size: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let stem = Conv2d::new(
            in_channels,
            dim,
            (patch_size, patch_size),
            (patch_size, patch_size),
            (0, 0),
        )?;
        crate::limits::check_repeats("depth", depth)?;
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(ConvNeXtBlock::new(dim, 3, 0.0, 1.0)?);
        }
        Ok(Self {
            stem,
            blocks,
            pool: AdaptiveAvgPool2d::new((1, 1)),
            head_norm: LayerNorm::new(dim, 1e-6)?,
            head: Linear::new(dim, num_classes)?,
            dim,
        })
    }
}

impl Module for ConvNeXt {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Batch from the stem output (same rows as the input): reading shape()[0]
        // first panicked on rank-0 input before the stem could reject it.
        let mut h = self.stem.forward(x)?;
        let b = h.shape()[0];
        for blk in &self.blocks {
            h = blk.forward(&h)?;
        }
        let pooled = self.pool.forward(&h)?.reshape(&[b, self.dim])?;
        let normed = self.head_norm.forward(&pooled)?;
        self.head.forward(&normed)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.stem.named_parameters() {
            out.push((format!("stem.{k}"), v));
        }
        for (i, blk) in self.blocks.iter().enumerate() {
            for (k, v) in blk.named_parameters() {
                out.push((format!("blocks.{i}.{k}"), v));
            }
        }
        for (k, v) in self.head_norm.named_parameters() {
            out.push((format!("head_norm.{k}"), v));
        }
        for (k, v) in self.head.named_parameters() {
            out.push((format!("head.{k}"), v));
        }
        out
    }

    fn train(&self, mode: bool) {
        for blk in &self.blocks {
            blk.train(mode);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Bidirectional LSTM (`BiLSTM`) & Causal Temporal Convolutional Net (`TCN`)
// ---------------------------------------------------------------------------

/// **Bidirectional LSTM (`BiLSTM`)** (`torch.nn.LSTM(bidirectional=True)`):
/// Runs a forward [`LSTM`] on $x_{1:T}$ and a backward [`LSTM`] on reversed $x_{T:1}$,
/// concatenating hidden states at each timestep into `[batch, seq_len, 2 * hidden_size]`.
pub struct BiLSTM {
    pub fwd_lstm: LSTM,
    pub bwd_lstm: LSTM,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl BiLSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        Ok(Self {
            fwd_lstm: LSTM::new(input_size, hidden_size, num_layers)?,
            bwd_lstm: LSTM::new(input_size, hidden_size, num_layers)?,
            input_size,
            hidden_size,
        })
    }
}

impl Module for BiLSTM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let fwd_out = self.fwd_lstm.forward(input)?;
        let rev_in = input.flip(&[1])?;
        let bwd_rev = self.bwd_lstm.forward(&rev_in)?;
        let bwd_out = bwd_rev.flip(&[1])?;
        Tensor::cat(&[fwd_out, bwd_out], 2)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.fwd_lstm.named_parameters() {
            out.push((format!("fwd.{k}"), v));
        }
        for (k, v) in self.bwd_lstm.named_parameters() {
            out.push((format!("bwd.{k}"), v));
        }
        out
    }
}

/// Causal 1D Temporal Convolution Residual Block (`TCNBlock`, Bai et al. 2018) over `[N, C, L]`.
pub struct TCNBlock {
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub downsample: Option<Conv1d>,
    pub causal_pad: usize,
}

impl TCNBlock {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Result<Self> {
        let causal_pad = kernel_size.saturating_sub(1);
        let conv1 = Conv1d::new(in_channels, out_channels, kernel_size, 1, causal_pad)?;
        let conv2 = Conv1d::new(out_channels, out_channels, kernel_size, 1, causal_pad)?;
        let downsample = if in_channels != out_channels {
            Some(Conv1d::new(in_channels, out_channels, 1, 1, 0)?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            downsample,
            causal_pad,
        })
    }
}

impl Module for TCNBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "TCNBlock expects 3D input [N, C, L], got {:?}",
                x.shape()
            )));
        }
        let l = x.shape()[2];
        let h1 = self.conv1.forward(x)?.narrow(2, 0, l)?.silu()?;
        let h2 = self.conv2.forward(&h1)?.narrow(2, 0, l)?.silu()?;
        let res = match &self.downsample {
            Some(ds) => ds.forward(x)?,
            None => x.clone(),
        };
        h2.add(&res)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.conv1.named_parameters() {
            out.push((format!("conv1.{k}"), v));
        }
        for (k, v) in self.conv2.named_parameters() {
            out.push((format!("conv2.{k}"), v));
        }
        if let Some(ds) = &self.downsample {
            for (k, v) in ds.named_parameters() {
                out.push((format!("downsample.{k}"), v));
            }
        }
        out
    }
}

/// Multi-layer Causal **Temporal Convolutional Network (`TemporalConvNet`)** over `[N, C_in, L]`.
pub struct TemporalConvNet {
    pub blocks: Vec<TCNBlock>,
}

impl TemporalConvNet {
    pub fn new(in_channels: usize, channel_dims: &[usize], kernel_size: usize) -> Result<Self> {
        let mut blocks = Vec::with_capacity(channel_dims.len());
        let mut prev_c = in_channels;
        for &out_c in channel_dims {
            blocks.push(TCNBlock::new(prev_c, out_c, kernel_size)?);
            prev_c = out_c;
        }
        Ok(Self { blocks })
    }
}

impl Module for TemporalConvNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for blk in &self.blocks {
            h = blk.forward(&h)?;
        }
        Ok(h)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, blk) in self.blocks.iter().enumerate() {
            for (k, v) in blk.named_parameters() {
                out.push((format!("blocks.{i}.{k}"), v));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 4. MLP-Mixer (Tolstikhin et al., NeurIPS 2021) & Variational Autoencoder (`VAE`)
// ---------------------------------------------------------------------------

/// Single **MLP-Mixer** block alternating Token-Mixing MLP (`[B, C, S]`) and Channel-Mixing MLP (`[B, S, C]`).
pub struct MLPMixerBlock {
    pub norm1: LayerNorm,
    pub token_fc1: Linear,
    pub token_fc2: Linear,
    pub norm2: LayerNorm,
    pub channel_fc1: Linear,
    pub channel_fc2: Linear,
}

impl MLPMixerBlock {
    pub fn new(
        num_tokens: usize,
        hidden_dim: usize,
        token_mlp_dim: usize,
        channel_mlp_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new(hidden_dim, 1e-6)?,
            token_fc1: Linear::new(num_tokens, token_mlp_dim)?,
            token_fc2: Linear::new(token_mlp_dim, num_tokens)?,
            norm2: LayerNorm::new(hidden_dim, 1e-6)?,
            channel_fc1: Linear::new(hidden_dim, channel_mlp_dim)?,
            channel_fc2: Linear::new(channel_mlp_dim, hidden_dim)?,
        })
    }
}

impl Module for MLPMixerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Token-mixing over sequence dimension S
        let n1 = self.norm1.forward(x)?.transpose(1, 2)?.contiguous()?;
        let t_mix = self
            .token_fc2
            .forward(&self.token_fc1.forward(&n1)?.gelu()?)?
            .transpose(1, 2)?
            .contiguous()?;
        let h = x.add(&t_mix)?;

        // 2. Channel-mixing over feature dimension C
        let n2 = self.norm2.forward(&h)?;
        let c_mix = self
            .channel_fc2
            .forward(&self.channel_fc1.forward(&n2)?.gelu()?)?;
        h.add(&c_mix)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m_params) in [
            ("norm1", self.norm1.named_parameters()),
            ("token_fc1", self.token_fc1.named_parameters()),
            ("token_fc2", self.token_fc2.named_parameters()),
            ("norm2", self.norm2.named_parameters()),
            ("channel_fc1", self.channel_fc1.named_parameters()),
            ("channel_fc2", self.channel_fc2.named_parameters()),
        ] {
            for (k, v) in m_params {
                out.push((format!("{p}.{k}"), v));
            }
        }
        out
    }
}

/// Complete **MLP-Mixer** Vision Architecture (Tolstikhin et al., NeurIPS 2021).
pub struct MLPMixer {
    pub patch_embed: PatchEmbedding,
    pub blocks: Vec<MLPMixerBlock>,
    pub norm: LayerNorm,
    pub head: Linear,
}

impl MLPMixer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        hidden_dim: usize,
        depth: usize,
        token_mlp_dim: usize,
        channel_mlp_dim: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let patch_embed = PatchEmbedding::new(img_size, patch_size, in_channels, hidden_dim)?;
        let num_tokens = 1 + patch_embed.num_patches;
        crate::limits::check_repeats("depth", depth)?;
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(MLPMixerBlock::new(
                num_tokens,
                hidden_dim,
                token_mlp_dim,
                channel_mlp_dim,
            )?);
        }
        Ok(Self {
            patch_embed,
            blocks,
            norm: LayerNorm::new(hidden_dim, 1e-6)?,
            head: Linear::new(hidden_dim, num_classes)?,
        })
    }
}

impl Module for MLPMixer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.patch_embed.forward(x)?;
        for blk in &self.blocks {
            h = blk.forward(&h)?;
        }
        let h = self.norm.forward(&h)?.mean_axis(1, false)?;
        self.head.forward(&h)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.patch_embed.named_parameters() {
            out.push((format!("patch_embed.{k}"), v));
        }
        for (i, blk) in self.blocks.iter().enumerate() {
            for (k, v) in blk.named_parameters() {
                out.push((format!("blocks.{i}.{k}"), v));
            }
        }
        for (k, v) in self.norm.named_parameters() {
            out.push((format!("norm.{k}"), v));
        }
        for (k, v) in self.head.named_parameters() {
            out.push((format!("head.{k}"), v));
        }
        out
    }
}

/// Complete **Variational Autoencoder (`VariationalAutoencoder` / `VAE`)** (Kingma & Welling, ICLR 2014):
/// Encodes input $x \in \mathbb{R}^{B \times D}$ into posterior parameters $(\mu(x), \log\sigma^2(x))$,
/// samples latent $z = \mu + \sigma \odot \epsilon$ via differentiable reparameterization,
/// and decodes $\hat{x} = \text{Decoder}(z)$.
pub struct VariationalAutoencoder {
    pub enc_fc1: Linear,
    pub enc_mu: Linear,
    pub enc_logvar: Linear,
    pub dec_fc1: Linear,
    pub dec_out: Linear,
    pub input_dim: usize,
    pub latent_dim: usize,
}

impl VariationalAutoencoder {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Result<Self> {
        Ok(Self {
            enc_fc1: Linear::new(input_dim, hidden_dim)?,
            enc_mu: Linear::new(hidden_dim, latent_dim)?,
            enc_logvar: Linear::new(hidden_dim, latent_dim)?,
            dec_fc1: Linear::new(latent_dim, hidden_dim)?,
            dec_out: Linear::new(hidden_dim, input_dim)?,
            input_dim,
            latent_dim,
        })
    }

    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = self.enc_fc1.forward(x)?.silu()?;
        Ok((self.enc_mu.forward(&h)?, self.enc_logvar.forward(&h)?))
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.dec_fc1.forward(z)?.silu()?;
        self.dec_out.forward(&h)
    }

    /// Runs the full VAE forward pass and computes the Evidence Lower Bound (`ELBO`) loss:
    /// $\mathcal{L}_{\text{ELBO}} = \operatorname{MSE}(\hat{x}, x) + \beta \cdot D_{\text{KL}}(q(z|x) \,\|\, \mathcal{N}(0, I))$.
    /// Returns `(recon, mu, logvar, total_elbo_loss)`.
    pub fn forward_elbo(
        &self,
        x: &Tensor,
        beta_kl: f64,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (mu, logvar) = self.encode(x)?;
        let z = vae_reparameterize(&mu, &logvar)?;
        let recon = self.decode(&z)?;
        let recon_loss = loss::mse_loss(&recon, x)?;
        let kl = vae_kl_loss(&mu, &logvar)?;
        let total = recon_loss.add(&kl.mul_scalar(beta_kl)?)?;
        Ok((recon, mu, logvar, total))
    }
}

impl Module for VariationalAutoencoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (recon, _, _, _) = self.forward_elbo(input, 1.0)?;
        Ok(recon)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (p, m) in [
            ("enc_fc1", &self.enc_fc1),
            ("enc_mu", &self.enc_mu),
            ("enc_logvar", &self.enc_logvar),
            ("dec_fc1", &self.dec_fc1),
            ("dec_out", &self.dec_out),
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
    fn mlp_truncated_aux_vecs_fail_closed() -> Result<()> {
        let mut mlp = MLP::new(&[4, 4, 2], ActivationKind::ReLU, true, 0.0, false)?;
        mlp.norms.truncate(1);
        mlp.dropouts.truncate(1);
        let x = Tensor::zeros_f32(&[2, 4])?;
        let err = mlp
            .forward(&x)
            .expect_err("truncated norms/dropouts must be an error, not a panic");
        assert!(err.to_string().contains("norms"), "{err}");
        // Infallible param listing degrades instead of panicking.
        let _ = mlp.named_parameters();
        Ok(())
    }

    #[test]
    fn convnext_block_rejects_overflowing_kernel() {
        let err = match ConvNeXtBlock::new(8, 1 << 32, 0.0, 1.0) {
            Ok(_) => panic!("overflowing kernel must be an error, not a panic"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("kernel"), "{err}");
    }

    #[test]
    fn convnext_rejects_rank0_input() -> Result<()> {
        let net = ConvNeXt::new(3, 8, 1, 2, 4)?;
        let x = Tensor::from_slice(&[1.0f32], &[])?;
        assert!(net.forward(&x).is_err());
        Ok(())
    }

    #[test]
    fn tcn_rejects_non_3d_input() -> Result<()> {
        let blk = TCNBlock::new(4, 4, 3)?;
        let x = Tensor::zeros_f32(&[2, 8])?;
        let err = blk
            .forward(&x)
            .expect_err("rank-2 input must be an error, not a panic");
        assert!(err.to_string().contains("[N, C, L]"), "{err}");
        Ok(())
    }
}
