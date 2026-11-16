//! Mini Residual Convolutional Network (ResNet).
//!
//! A compact Residual Convolutional Network architecture with skip connections:
//! - Stem: $3 \times 3$ Conv2D (in_channels $\to$ 16) + ReLU
//! - ResBlock 1: Two $3 \times 3$ Conv2D (16 $\to$ 16) with identity residual shortcut $x + \mathcal{F}(x)$
//! - Downsample: $3 \times 3$ Conv2D (16 $\to$ 32, stride 2) + ReLU
//! - ResBlock 2: Two $3 \times 3$ Conv2D (32 $\to$ 32) with identity residual shortcut $x + \mathcal{F}(x)$
//! - Head: AdaptiveAvgPool2d(1, 1) + Linear(32 $\to$ `num_classes`)

use crate::error::{Error, Result};
use crate::nn::loss::cross_entropy;
use crate::nn::{AdaptiveAvgPool2d, Conv2d, Linear, Module};
use crate::optim::{Adam, Optimizer};
use crate::tensor::Tensor;
use crate::vision::dataset::MiniImageDataset;

/// A residual convolutional block with identity skip connection: $y = \text{ReLU}(x + \text{Conv}_2(\text{ReLU}(\text{Conv}_1(x))))$.
pub struct ResidualBlock {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
}

impl ResidualBlock {
    pub fn new(channels: usize) -> Result<Self> {
        let conv1 = Conv2d::new(channels, channels, (3, 3), (1, 1), (1, 1))?;
        let conv2 = Conv2d::new(channels, channels, (3, 3), (1, 1), (1, 1))?;
        Ok(Self { conv1, conv2 })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self.conv1.forward(x)?.relu()?;
        let residual = self.conv2.forward(&residual)?;
        x.add(&residual)?.relu()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.conv1.named_parameters() {
            out.push((format!("conv1.{k}"), v));
        }
        for (k, v) in self.conv2.named_parameters() {
            out.push((format!("conv2.{k}"), v));
        }
        out
    }
}

/// Metric summary for an individual training epoch of `MiniResNet`.
#[derive(Debug, Clone, Copy)]
pub struct ResNetEpochResult {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: Option<f64>,
}

/// A compact Residual Convolutional Neural Network for image classification.
pub struct MiniResNet {
    pub stem: Conv2d,
    pub block1: ResidualBlock,
    pub downsample: Conv2d,
    pub block2: ResidualBlock,
    pub pool: AdaptiveAvgPool2d,
    pub fc: Linear,
    pub in_channels: usize,
    pub num_classes: usize,
}

impl MiniResNet {
    /// Constructs a `MiniResNet` with stem, 2 residual blocks, global pooling, and head.
    pub fn new(in_channels: usize, num_classes: usize) -> Result<Self> {
        let stem = Conv2d::new(in_channels, 16, (3, 3), (1, 1), (1, 1))?;
        let block1 = ResidualBlock::new(16)?;
        let downsample = Conv2d::new(16, 32, (3, 3), (2, 2), (1, 1))?;
        let block2 = ResidualBlock::new(32)?;
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let fc = Linear::new(32, num_classes)?;

        Ok(Self {
            stem,
            block1,
            downsample,
            block2,
            pool,
            fc,
            in_channels,
            num_classes,
        })
    }

    /// Convenience constructor for 4-class classification on $1 \times 16 \times 16$ mini images.
    pub fn standard() -> Result<Self> {
        Self::new(1, 4)
    }

    /// Evaluates the model on images and ground-truth targets, returning `(loss, accuracy)`.
    pub fn evaluate(&self, images: &Tensor, targets: &[usize]) -> Result<(f64, f64)> {
        if targets.is_empty() {
            return Err(Error::Msg("Cannot evaluate on empty targets".into()));
        }
        let logits = self.forward(images)?;
        let loss_tensor = cross_entropy(&logits, targets)?;
        let loss = loss_tensor.item::<f32>()? as f64;

        let preds = self.predict(images)?;
        let correct = preds
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| *p == *t)
            .count();
        let accuracy = (correct as f64) / (targets.len() as f64);

        Ok((loss, accuracy))
    }

    /// Predicts class indices using argmax of logits.
    pub fn predict(&self, images: &Tensor) -> Result<Vec<usize>> {
        let logits = self.forward(images)?;
        let n = logits.shape()[0];
        // Slice by the actual logit width (`num_classes` is public and may disagree).
        let k = logits.shape()[1];
        let data = logits.to_vec::<f32>()?;

        let mut predictions = Vec::with_capacity(n);
        for i in 0..n {
            let row = &data[i * k..(i + 1) * k];
            let mut best_cls = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (cls, &val) in row.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_cls = cls;
                }
            }
            predictions.push(best_cls);
        }
        Ok(predictions)
    }

    /// Trains the model using mini-batch Adam optimization over `epochs` epochs.
    pub fn fit(
        &mut self,
        train_set: &MiniImageDataset,
        val_set: Option<&MiniImageDataset>,
        epochs: usize,
        batch_size: usize,
        lr: f64,
    ) -> Result<Vec<ResNetEpochResult>> {
        let params = self.parameters();
        let mut opt = Adam::adamw(params, lr, 1e-4);
        let num_batches = train_set.num_batches(batch_size);
        let epochs = crate::limits::check_iterations("MiniResNet::fit epochs", epochs)?;
        let mut history = Vec::with_capacity(epochs.min(1024));

        for epoch in 1..=epochs {
            for b in 0..num_batches {
                let (batch_images, batch_targets) = train_set.batch(b, batch_size)?;
                opt.zero_grad();

                let logits = self.forward(&batch_images)?;
                let loss = cross_entropy(&logits, &batch_targets)?;
                loss.backward()?;
                opt.step()?;
            }

            let (train_loss, train_acc) = self.evaluate(&train_set.images, &train_set.labels)?;
            let val_acc = if let Some(val) = val_set {
                let (_, va) = self.evaluate(&val.images, &val.labels)?;
                Some(va)
            } else {
                None
            };

            history.push(ResNetEpochResult {
                epoch,
                train_loss,
                train_accuracy: train_acc,
                val_accuracy: val_acc,
            });
        }

        Ok(history)
    }
}

impl Module for MiniResNet {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 4 {
            return Err(Error::Msg(format!(
                "MiniResNet expects 4D input [B, C, H, W], got {:?}",
                x.shape()
            )));
        }
        let b = x.shape()[0];

        // Stem: [B, 1, 16, 16] -> [B, 16, 16, 16]
        let h = self.stem.forward(x)?.relu()?;

        // ResBlock 1: [B, 16, 16, 16] -> [B, 16, 16, 16]
        let h = self.block1.forward(&h)?;

        // Downsample: [B, 16, 16, 16] -> [B, 32, 8, 8]
        let h = self.downsample.forward(&h)?.relu()?;

        // ResBlock 2: [B, 32, 8, 8] -> [B, 32, 8, 8]
        let h = self.block2.forward(&h)?;

        // Global Average Pooling: [B, 32, 8, 8] -> [B, 32, 1, 1]
        let pooled = self.pool.forward(&h)?;
        let flat = pooled.reshape(&[b, 32])?;

        // Head: [B, 32] -> [B, 4]
        self.fc.forward(&flat)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.stem.named_parameters() {
            out.push((format!("stem.{k}"), v));
        }
        for (k, v) in self.block1.named_parameters() {
            out.push((format!("block1.{k}"), v));
        }
        for (k, v) in self.downsample.named_parameters() {
            out.push((format!("downsample.{k}"), v));
        }
        for (k, v) in self.block2.named_parameters() {
            out.push((format!("block2.{k}"), v));
        }
        for (k, v) in self.fc.named_parameters() {
            out.push((format!("fc.{k}"), v));
        }
        out
    }
}
