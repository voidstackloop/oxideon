//! Spatio-Temporal 3D Video Autoencoder (`VideoVAE3d`, Sora / CogVideoX / Wan 2.1 standard).
//!
//! Compresses high-dimensional video volumes `[B, C, T, H, W]` into compact latent representations
//! using causal 3D convolutions with temporal and spatial downsampling.

use crate::error::Result;
use crate::nn::{GroupNorm, Module};
use crate::tensor::Tensor;
use crate::video::causal_conv3d::CausalConv3d;

/// 3D Spatio-Temporal Residual Block with Causal Convolutions.
pub struct VideoResBlock3d {
    pub conv1: CausalConv3d,
    pub norm1: GroupNorm,
    pub conv2: CausalConv3d,
    pub norm2: GroupNorm,
    pub shortcut: Option<CausalConv3d>,
}

impl VideoResBlock3d {
    pub fn new(in_channels: usize, out_channels: usize) -> Result<Self> {
        // Largest divisor of C (at most 4): GroupNorm needs C % groups == 0,
        // which a flat min(4, C) violates for C in {6, 10, 14, ...}.
        fn groups_for(channels: usize) -> usize {
            (1..=4.min(channels))
                .rev()
                .find(|g| channels % g == 0)
                .unwrap_or(1)
        }
        let num_groups = groups_for(in_channels);
        let out_groups = groups_for(out_channels);
        let conv1 = CausalConv3d::new(in_channels, out_channels, (3, 3, 3), (1, 1, 1))?;
        let norm1 = GroupNorm::new(num_groups, in_channels, 1e-5)?;
        let conv2 = CausalConv3d::new(out_channels, out_channels, (3, 3, 3), (1, 1, 1))?;
        let norm2 = GroupNorm::new(out_groups, out_channels, 1e-5)?;

        let shortcut = if in_channels != out_channels {
            Some(CausalConv3d::new(
                in_channels,
                out_channels,
                (1, 1, 1),
                (1, 1, 1),
            )?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            norm1,
            conv2,
            norm2,
            shortcut,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let res = match &self.shortcut {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };

        let h = self.norm1.forward(x)?.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?.silu()?;
        let h = self.conv2.forward(&h)?;

        h.add(&res)
    }
}

impl VideoResBlock3d {
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.conv1.named_parameters() {
            out.push((format!("conv1.{k}"), v));
        }
        for (k, v) in self.norm1.named_parameters() {
            out.push((format!("norm1.{k}"), v));
        }
        for (k, v) in self.conv2.named_parameters() {
            out.push((format!("conv2.{k}"), v));
        }
        for (k, v) in self.norm2.named_parameters() {
            out.push((format!("norm2.{k}"), v));
        }
        if let Some(s) = &self.shortcut {
            for (k, v) in s.named_parameters() {
                out.push((format!("shortcut.{k}"), v));
            }
        }
        out
    }
}

impl VideoVAE3d {
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.enc_conv_in.named_parameters() {
            out.push((format!("enc_conv_in.{k}"), v));
        }
        for (k, v) in self.enc_block.named_parameters() {
            out.push((format!("enc_block.{k}"), v));
        }
        for (k, v) in self.enc_down.named_parameters() {
            out.push((format!("enc_down.{k}"), v));
        }
        for (k, v) in self.enc_out.named_parameters() {
            out.push((format!("enc_out.{k}"), v));
        }
        for (k, v) in self.dec_conv_in.named_parameters() {
            out.push((format!("dec_conv_in.{k}"), v));
        }
        for (k, v) in self.dec_block.named_parameters() {
            out.push((format!("dec_block.{k}"), v));
        }
        for (k, v) in self.dec_out.named_parameters() {
            out.push((format!("dec_out.{k}"), v));
        }
        out
    }
}

/// Spatio-Temporal 3D Video Variational Autoencoder.
pub struct VideoVAE3d {
    pub in_channels: usize,
    pub latent_channels: usize,
    pub enc_conv_in: CausalConv3d,
    pub enc_block: VideoResBlock3d,
    pub enc_down: CausalConv3d,
    pub enc_out: CausalConv3d,
    pub dec_conv_in: CausalConv3d,
    pub dec_block: VideoResBlock3d,
    pub dec_out: CausalConv3d,
}

impl VideoVAE3d {
    pub fn new(in_channels: usize, hidden_channels: usize, latent_channels: usize) -> Result<Self> {
        // Encoder: in -> hidden -> downsample (stride 2) -> latent
        let enc_conv_in = CausalConv3d::new(in_channels, hidden_channels, (3, 3, 3), (1, 1, 1))?;
        let enc_block = VideoResBlock3d::new(hidden_channels, hidden_channels)?;
        let enc_down = CausalConv3d::new(hidden_channels, hidden_channels, (3, 3, 3), (1, 2, 2))?;
        let enc_out = CausalConv3d::new(hidden_channels, latent_channels, (1, 1, 1), (1, 1, 1))?;

        // Decoder: latent -> hidden -> ResBlock -> out
        let dec_conv_in =
            CausalConv3d::new(latent_channels, hidden_channels, (1, 1, 1), (1, 1, 1))?;
        let dec_block = VideoResBlock3d::new(hidden_channels, hidden_channels)?;
        let dec_out = CausalConv3d::new(hidden_channels, in_channels, (3, 3, 3), (1, 1, 1))?;

        Ok(Self {
            in_channels,
            latent_channels,
            enc_conv_in,
            enc_block,
            enc_down,
            enc_out,
            dec_conv_in,
            dec_block,
            dec_out,
        })
    }

    /// Compresses video tensor `[B, C_in, T, H, W]` into latent space.
    pub fn encode(&self, video: &Tensor) -> Result<Tensor> {
        let h = self.enc_conv_in.forward(video)?;
        let h = self.enc_block.forward(&h)?;
        let h = self.enc_down.forward(&h)?;
        self.enc_out.forward(&h)
    }

    /// Reconstructs video from latents `[B, C_latent, T, H_lat, W_lat]`.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let h = self.dec_conv_in.forward(latents)?;
        let h = self.dec_block.forward(&h)?;
        // Spatial upsampling (nearest / repeat factor 2)
        let up = Self::upsample_spatial_2x(&h)?;
        self.dec_out.forward(&up)
    }

    /// Helper for 2x spatial upsampling over 5D video `[B, C, T, H, W]`.
    fn upsample_spatial_2x(x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        let (b, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let data = x.to_vec::<f32>()?;
        let mut out = Vec::with_capacity(b * c * t * (h * 2) * (w * 2));

        for batch in 0..b {
            for ch in 0..c {
                for frame in 0..t {
                    let plane_offset = ((batch * c + ch) * t + frame) * (h * w);
                    for r in 0..h {
                        for _rep_r in 0..2 {
                            for col in 0..w {
                                let val = data[plane_offset + r * w + col];
                                out.push(val);
                                out.push(val);
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_vec(out, &[b, c, t, h * 2, w * 2])?
            .to_device(&x.device())?
            .to_dtype(x.dtype())
    }

    /// Full forward autoencoder pass returning reconstructed video.
    pub fn forward(&self, video: &Tensor) -> Result<Tensor> {
        let z = self.encode(video)?;
        self.decode(&z)
    }
}
