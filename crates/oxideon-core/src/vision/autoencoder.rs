//! Mini Convolutional Autoencoder (Generative & Latent Compression).
//!
//! An encoder-decoder network that compresses $16 \times 16$ mini-images into a low-dimensional
//! latent bottleneck representation ($Z \in \mathbb{R}^{16}$) and reconstructs the image:
//! - Encoder: Conv2D ($1 \to 16$, stride 2) + ReLU + Conv2D ($16 \to 32$, stride 2) + Flatten + Linear(512 $\to$ 16)
//! - Decoder: Linear(16 $\to$ 512) + Reshape + ConvTranspose2D ($32 \to 16$, stride 2) + ReLU + ConvTranspose2D ($16 \to 1$, stride 2) + Sigmoid
//! - Features: Latent space interpolation, generative sampling, and pixel-level MSE reconstruction.

use crate::error::{Error, Result};
use crate::nn::{Conv2d, ConvTranspose2d, Linear, Module};
use crate::optim::{Adam, Optimizer};
use crate::tensor::Tensor;
use crate::vision::dataset::MiniImageDataset;

/// Metric summary for an individual training epoch of `MiniAutoEncoder`.
#[derive(Debug, Clone, Copy)]
pub struct AutoEncoderEpochResult {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
}

/// A compact Convolutional Autoencoder for image compression, reconstruction, and generation.
pub struct MiniAutoEncoder {
    pub enc_conv1: Conv2d,
    pub enc_conv2: Conv2d,
    pub enc_fc: Linear,
    pub dec_fc: Linear,
    pub dec_conv1: ConvTranspose2d,
    pub dec_conv2: ConvTranspose2d,
    pub in_channels: usize,
    pub latent_dim: usize,
}

impl MiniAutoEncoder {
    /// Constructs a `MiniAutoEncoder` with a specified bottleneck latent dimension (default 16).
    pub fn new(in_channels: usize, latent_dim: usize) -> Result<Self> {
        let enc_conv1 = Conv2d::new(in_channels, 16, (3, 3), (2, 2), (1, 1))?;
        let enc_conv2 = Conv2d::new(16, 32, (3, 3), (2, 2), (1, 1))?;
        let enc_fc = Linear::new(32 * 4 * 4, latent_dim)?;

        let dec_fc = Linear::new(latent_dim, 32 * 4 * 4)?;
        let dec_conv1 = ConvTranspose2d::new(32, 16, (4, 4), (2, 2), (1, 1))?;
        let dec_conv2 = ConvTranspose2d::new(16, in_channels, (4, 4), (2, 2), (1, 1))?;

        Ok(Self {
            enc_conv1,
            enc_conv2,
            enc_fc,
            dec_fc,
            dec_conv1,
            dec_conv2,
            in_channels,
            latent_dim,
        })
    }

    /// Convenience constructor for $1 \times 16 \times 16$ mini images with 16-dimensional bottleneck.
    pub fn standard() -> Result<Self> {
        Self::new(1, 16)
    }

    /// Encodes images `[B, C, H, W]` into latent vectors `[B, latent_dim]`.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 4 {
            return Err(Error::Msg(format!(
                "MiniAutoEncoder::encode expects 4D input [B, C, H, W], got {:?}",
                x.shape()
            )));
        }
        let b = x.shape()[0];
        let h = self.enc_conv1.forward(x)?.relu()?;
        let h = self.enc_conv2.forward(&h)?.relu()?;
        let flat = h.reshape(&[b, 32 * 4 * 4])?;
        self.enc_fc.forward(&flat)
    }

    /// Decodes latent vectors `[B, latent_dim]` back into reconstructed images `[B, C, H, W]`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        if z.ndim() != 2 {
            return Err(Error::Msg(format!(
                "MiniAutoEncoder::decode expects 2D latents [B, latent_dim], got {:?}",
                z.shape()
            )));
        }
        let b = z.shape()[0];
        let h = self.dec_fc.forward(z)?.relu()?;
        let h = h.reshape(&[b, 32, 4, 4])?;
        let h = self.dec_conv1.forward(&h)?.relu()?;
        self.dec_conv2.forward(&h)?.sigmoid()
    }

    /// Computes Mean Squared Error (MSE) reconstruction loss between input and reconstruction.
    pub fn reconstruction_loss(&self, x: &Tensor) -> Result<Tensor> {
        let recon = self.forward(x)?;
        x.sub(&recon)?.square()?.mean()
    }

    /// Interpolates smoothly between two images in latent space, returning `steps` intermediate reconstructed frames.
    pub fn interpolate(&self, x1: &Tensor, x2: &Tensor, steps: usize) -> Result<Tensor> {
        let z1 = self.encode(x1)?;
        let z2 = self.encode(x2)?;

        let steps = crate::limits::check_iterations("MiniAutoEncoder::interpolate steps", steps)?;
        let mut frames = Vec::with_capacity(steps.min(1024));
        for s in 0..steps {
            let alpha = if steps > 1 {
                (s as f64) / ((steps - 1) as f64)
            } else {
                0.0
            };
            // z_interp = (1 - alpha) * z1 + alpha * z2
            let z1_scaled = z1.mul_scalar(1.0 - alpha)?;
            let z2_scaled = z2.mul_scalar(alpha)?;
            let z_interp = z1_scaled.add(&z2_scaled)?;
            let frame = self.decode(&z_interp)?;
            frames.push(frame);
        }

        Tensor::cat(&frames, 0)
    }

    /// Generates new synthetic mini-images by decoding random standard Gaussian noise vectors.
    pub fn sample(&self, num_samples: usize) -> Result<Tensor> {
        let z = Tensor::randn(&[num_samples, self.latent_dim])?;
        self.decode(&z)
    }

    /// Trains the autoencoder on `train_dataset` using AdamW and MSE loss.
    pub fn fit(
        &mut self,
        train_set: &MiniImageDataset,
        val_set: Option<&MiniImageDataset>,
        epochs: usize,
        batch_size: usize,
        lr: f64,
    ) -> Result<Vec<AutoEncoderEpochResult>> {
        let params = self.parameters();
        let mut opt = Adam::adamw(params, lr, 1e-4);
        let num_batches = train_set.num_batches(batch_size);
        let epochs = crate::limits::check_iterations("MiniAutoEncoder::fit epochs", epochs)?;
        let mut history = Vec::with_capacity(epochs.min(1024));

        for epoch in 1..=epochs {
            for b in 0..num_batches {
                let (batch_images, _) = train_set.batch(b, batch_size)?;
                opt.zero_grad();

                let loss = self.reconstruction_loss(&batch_images)?;
                loss.backward()?;
                opt.step()?;
            }

            let train_loss = self.reconstruction_loss(&train_set.images)?.item::<f32>()? as f64;
            let val_loss = if let Some(val) = val_set {
                Some(self.reconstruction_loss(&val.images)?.item::<f32>()? as f64)
            } else {
                None
            };

            history.push(AutoEncoderEpochResult {
                epoch,
                train_loss,
                val_loss,
            });
        }

        Ok(history)
    }
}

impl Module for MiniAutoEncoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let z = self.encode(x)?;
        self.decode(&z)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.enc_conv1.named_parameters() {
            out.push((format!("enc_conv1.{k}"), v));
        }
        for (k, v) in self.enc_conv2.named_parameters() {
            out.push((format!("enc_conv2.{k}"), v));
        }
        for (k, v) in self.enc_fc.named_parameters() {
            out.push((format!("enc_fc.{k}"), v));
        }
        for (k, v) in self.dec_fc.named_parameters() {
            out.push((format!("dec_fc.{k}"), v));
        }
        for (k, v) in self.dec_conv1.named_parameters() {
            out.push((format!("dec_conv1.{k}"), v));
        }
        for (k, v) in self.dec_conv2.named_parameters() {
            out.push((format!("dec_conv2.{k}"), v));
        }
        out
    }
}
