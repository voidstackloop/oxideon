//! Vision architectures: Vision Transformer (`ViT`), `UNet2d`, and Variational Autoencoder (`VAE`) helpers.

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::nn::layers::{Conv2d, ConvTranspose2d, LayerNorm, Linear, MaxPool2d};
use crate::nn::transformer::TransformerEncoderLayer;
use crate::tensor::Tensor;

/// Image-to-patch embedding with a learnable `[CLS]` token and positional embeddings.
pub struct PatchEmbedding {
    pub proj: Conv2d,
    pub cls_token: Tensor,
    pub pos_embed: Tensor,
    pub patch_size: usize,
    pub num_patches: usize,
    pub embed_dim: usize,
}

impl PatchEmbedding {
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
    ) -> Result<Self> {
        if patch_size == 0 || img_size % patch_size != 0 {
            return Err(Error::Msg(format!(
                "PatchEmbedding: img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )));
        }
        let grid = img_size / patch_size;
        let num_patches = grid.checked_mul(grid).ok_or_else(|| {
            Error::Msg(format!(
                "PatchEmbedding: num_patches (grid {grid} * {grid}) overflows"
            ))
        })?;
        let proj = Conv2d::new(
            in_channels,
            embed_dim,
            (patch_size, patch_size),
            (patch_size, patch_size),
            (0, 0),
        )?;
        let cls_token = Tensor::randn(&[1, 1, embed_dim])?
            .mul_scalar(0.02)?
            .requires_grad_(true);
        let pos_embed = Tensor::randn(&[1, 1 + num_patches, embed_dim])?
            .mul_scalar(0.02)?
            .requires_grad_(true);
        Ok(Self {
            proj,
            cls_token,
            pos_embed,
            patch_size,
            num_patches,
            embed_dim,
        })
    }
}

impl Module for PatchEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 4 {
            return Err(Error::Msg(format!(
                "PatchEmbedding expects 4D [B, C, H, W], got {:?}",
                x.shape()
            )));
        }
        let b = x.shape()[0];
        // [B, D, H', W'] -> [B, D, N] -> [B, N, D]
        let patches = self.proj.forward(x)?;
        let n = patches.shape()[2] * patches.shape()[3];
        if n != self.num_patches {
            return Err(Error::Msg(format!(
                "PatchEmbedding expected {} patches, got {n}",
                self.num_patches
            )));
        }
        let patches = patches.reshape(&[b, self.embed_dim, n])?.transpose(1, 2)?;
        let cls = self.cls_token.broadcast_to(&[b, 1, self.embed_dim])?;
        let tokens = Tensor::cat(&[cls, patches], 1)?;
        tokens.add(&self.pos_embed)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p: Vec<(String, Tensor)> = self
            .proj
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("proj.{k}"), v))
            .collect();
        p.push(("cls_token".to_string(), self.cls_token.clone()));
        p.push(("pos_embed".to_string(), self.pos_embed.clone()));
        p
    }
}

/// Vision Transformer (`ViT`) for image classification.
pub struct VisionTransformer {
    pub patch_embed: PatchEmbedding,
    pub blocks: Vec<TransformerEncoderLayer>,
    pub norm: LayerNorm,
    pub head: Linear,
}

impl VisionTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_channels: usize,
        num_classes: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_dim: usize,
        dropout_p: f32,
    ) -> Result<Self> {
        let patch_embed = PatchEmbedding::new(img_size, patch_size, in_channels, embed_dim)?;
        crate::limits::check_repeats("depth", depth)?;
        let mut blocks = Vec::with_capacity(depth);
        for _ in 0..depth {
            blocks.push(TransformerEncoderLayer::new(
                embed_dim, num_heads, mlp_dim, dropout_p,
            )?);
        }
        let norm = LayerNorm::new(embed_dim, 1e-6)?;
        let head = Linear::new(embed_dim, num_classes)?;
        Ok(Self {
            patch_embed,
            blocks,
            norm,
            head,
        })
    }
}

impl Module for VisionTransformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.patch_embed.forward(x)?;
        for blk in &self.blocks {
            h = blk.forward(&h)?;
        }
        let h = self.norm.forward(&h)?;
        // Extract [CLS] token at index 0: [B, D]
        let cls = h.select(1, 0)?;
        self.head.forward(&cls)
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

    fn train(&self, mode: bool) {
        for blk in &self.blocks {
            blk.train(mode);
        }
    }
}

/// Two-layer 3x3 convolution block with ReLU activations used inside `UNet2d`.
pub struct DoubleConv2d {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl DoubleConv2d {
    pub fn new(in_ch: usize, out_ch: usize) -> Result<Self> {
        let conv1 = Conv2d::new(in_ch, out_ch, (3, 3), (1, 1), (1, 1))?;
        let conv2 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1))?;
        Ok(Self { conv1, conv2 })
    }
}

impl Module for DoubleConv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?.relu()?;
        self.conv2.forward(&h)?.relu()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut p: Vec<(String, Tensor)> = self
            .conv1
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("conv1.{k}"), v))
            .collect();
        for (k, v) in self.conv2.named_parameters() {
            p.push((format!("conv2.{k}"), v));
        }
        p
    }
}

/// 2D U-Net encoder-decoder with skip connections for dense prediction and diffusion.
pub struct UNet2d {
    pub enc1: DoubleConv2d,
    pub enc2: DoubleConv2d,
    pub bottleneck: DoubleConv2d,
    pub pool: MaxPool2d,
    pub up2: ConvTranspose2d,
    pub dec2: DoubleConv2d,
    pub up1: ConvTranspose2d,
    pub dec1: DoubleConv2d,
    pub head: Conv2d,
}

impl UNet2d {
    pub fn new(in_channels: usize, out_channels: usize, base_channels: usize) -> Result<Self> {
        let c1 = base_channels;
        let c2 = base_channels.checked_mul(2).ok_or_else(|| {
            Error::Msg(format!(
                "UNet2d: base_channels {base_channels} * 2 overflows"
            ))
        })?;
        let c3 = base_channels.checked_mul(4).ok_or_else(|| {
            Error::Msg(format!(
                "UNet2d: base_channels {base_channels} * 4 overflows"
            ))
        })?;
        Ok(Self {
            enc1: DoubleConv2d::new(in_channels, c1)?,
            enc2: DoubleConv2d::new(c1, c2)?,
            bottleneck: DoubleConv2d::new(c2, c3)?,
            pool: MaxPool2d::new((2, 2), (2, 2)),
            up2: ConvTranspose2d::new(c3, c2, (2, 2), (2, 2), (0, 0))?,
            dec2: DoubleConv2d::new(c2 + c2, c2)?,
            up1: ConvTranspose2d::new(c2, c1, (2, 2), (2, 2), (0, 0))?,
            dec1: DoubleConv2d::new(c1 + c1, c1)?,
            head: Conv2d::new(c1, out_channels, (1, 1), (1, 1), (0, 0))?,
        })
    }
}

impl Module for UNet2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s1 = self.enc1.forward(x)?;
        let p1 = self.pool.forward(&s1)?;

        let s2 = self.enc2.forward(&p1)?;
        let p2 = self.pool.forward(&s2)?;

        let b = self.bottleneck.forward(&p2)?;

        let u2 = self.up2.forward(&b)?;
        let cat2 = Tensor::cat(&[u2, s2], 1)?;
        let d2 = self.dec2.forward(&cat2)?;

        let u1 = self.up1.forward(&d2)?;
        let cat1 = Tensor::cat(&[u1, s1], 1)?;
        let d1 = self.dec1.forward(&cat1)?;

        self.head.forward(&d1)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (name, m) in [
            ("enc1", &self.enc1),
            ("enc2", &self.enc2),
            ("bottleneck", &self.bottleneck),
            ("dec2", &self.dec2),
            ("dec1", &self.dec1),
        ] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{name}.{k}"), v));
            }
        }
        for (name, m) in [("up2", &self.up2), ("up1", &self.up1)] {
            for (k, v) in m.named_parameters() {
                out.push((format!("{name}.{k}"), v));
            }
        }
        for (k, v) in self.head.named_parameters() {
            out.push((format!("head.{k}"), v));
        }
        out
    }
}

/// Variational Autoencoder (VAE) reparameterization trick:
/// $z = \mu + \epsilon \odot \exp(0.5 \cdot \text{logvar})$, where $\epsilon \sim \mathcal{N}(0, I)$.
pub fn vae_reparameterize(mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
    let std = logvar.mul_scalar(0.5)?.exp()?;
    let eps = Tensor::randn(mu.shape())?
        .to_dtype(mu.dtype())?
        .to_device(&mu.device())?;
    mu.add(&eps.mul(&std)?)
}

/// Analytical KL divergence between $\mathcal{N}(\mu, \exp(\text{logvar}))$ and $\mathcal{N}(0, I)$,
/// averaged over batch elements:
/// $\text{KL} = -\frac{1}{2 B} \sum (1 + \text{logvar} - \mu^2 - \exp(\text{logvar}))$.
pub fn vae_kl_loss(mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
    let batch = mu.shape().first().copied().unwrap_or(1).max(1) as f64;
    let term = logvar
        .add_scalar(1.0)?
        .sub(&mu.square()?)?
        .sub(&logvar.exp()?)?;
    term.sum()?.mul_scalar(-0.5 / batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unet_rejects_overflowing_channels() {
        let err = match UNet2d::new(3, 3, usize::MAX / 2) {
            Ok(_) => panic!("overflowing channels must be an error, not a panic"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("channel"), "{err}");
    }
}
