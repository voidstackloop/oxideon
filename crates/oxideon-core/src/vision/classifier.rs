//! Basic Mini Image Classifier (Convolutional Neural Network).
//!
//! A 4-stage convolutional neural network tailored for spatial feature extraction
//! and pattern recognition on mini-images:
//! - Stage 1: $3 \times 3$ Conv2D (1 $\to$ 16 channels) + ReLU + $2 \times 2$ MaxPool
//! - Stage 2: $3 \times 3$ Conv2D (16 $\to$ 32 channels) + ReLU + $2 \times 2$ MaxPool
//! - Stage 3: Dense Feature Linear projection (512 $\to$ 64 channels) + ReLU
//! - Stage 4: Logits Linear Classifier Head (64 $\to$ `num_classes`)

use crate::error::{Error, Result};
use crate::nn::loss::cross_entropy;
use crate::nn::{Conv2d, Linear, MaxPool2d, Module};
use crate::optim::{Adam, Optimizer};
use crate::tensor::Tensor;
use crate::vision::dataset::MiniImageDataset;

/// Summary metrics for each training epoch.
#[derive(Debug, Clone)]
pub struct EpochMetric {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy: f64,
    pub val_loss: Option<f64>,
    pub val_accuracy: Option<f64>,
}

/// Training history trajectory across all epochs.
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    pub history: Vec<EpochMetric>,
    pub final_train_accuracy: f64,
    pub final_val_accuracy: Option<f64>,
}

/// A compact 4-stage Convolutional Neural Network for mini image classification.
pub struct MiniImageClassifier {
    pub conv1: Conv2d,
    pub pool1: MaxPool2d,
    pub conv2: Conv2d,
    pub pool2: MaxPool2d,
    pub fc1: Linear,
    pub fc2: Linear,
    pub in_channels: usize,
    pub num_classes: usize,
    pub flat_dim: usize,
}

impl MiniImageClassifier {
    /// Constructs a `MiniImageClassifier` configured for images of shape `[channels, height, width]`.
    pub fn new(
        in_channels: usize,
        height: usize,
        width: usize,
        num_classes: usize,
    ) -> Result<Self> {
        let conv1 = Conv2d::new(in_channels, 16, (3, 3), (1, 1), (1, 1))?;
        let pool1 = MaxPool2d::new((2, 2), (2, 2));

        let conv2 = Conv2d::new(16, 32, (3, 3), (1, 1), (1, 1))?;
        let pool2 = MaxPool2d::new((2, 2), (2, 2));

        let h_down = height / 4;
        let w_down = width / 4;
        let flat_dim = 32usize
            .checked_mul(h_down)
            .and_then(|v| v.checked_mul(w_down))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "MiniImageClassifier: flat dim 32 * {h_down} * {w_down} overflows"
                ))
            })?;

        let fc1 = Linear::new(flat_dim, 64)?;
        let fc2 = Linear::new(64, num_classes)?;

        Ok(Self {
            conv1,
            pool1,
            conv2,
            pool2,
            fc1,
            fc2,
            in_channels,
            num_classes,
            flat_dim,
        })
    }

    /// Convenience constructor for $1 \times 16 \times 16$ 4-class mini images.
    pub fn standard() -> Result<Self> {
        Self::new(1, 16, 16, 4)
    }

    /// Evaluates the model on a set of images and targets, returning `(average_loss, accuracy)`.
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

    /// Predicts class indices for a batch of images using argmax of logits.
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
    ) -> Result<TrainingHistory> {
        if batch_size == 0 {
            return Err(Error::Msg(
                "MiniImageClassifier::fit requires batch_size > 0".into(),
            ));
        }
        let params = self.parameters();
        let mut opt = Adam::adamw(params, lr, 1e-4);
        let num_batches = train_set.num_batches(batch_size);
        let epochs = crate::limits::check_iterations("MiniImageClassifier::fit epochs", epochs)?;
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

            // Compute epoch training metrics
            let (train_loss, train_acc) = self.evaluate(&train_set.images, &train_set.labels)?;

            // Optional validation evaluation
            let (val_loss, val_acc) = if let Some(val) = val_set {
                let (vl, va) = self.evaluate(&val.images, &val.labels)?;
                (Some(vl), Some(va))
            } else {
                (None, None)
            };

            history.push(EpochMetric {
                epoch,
                train_loss,
                train_accuracy: train_acc,
                val_loss,
                val_accuracy: val_acc,
            });
        }

        let final_train_acc = history.last().map(|m| m.train_accuracy).unwrap_or(0.0);
        let final_val_acc = history.last().and_then(|m| m.val_accuracy);

        Ok(TrainingHistory {
            history,
            final_train_accuracy: final_train_acc,
            final_val_accuracy: final_val_acc,
        })
    }
}

impl Module for MiniImageClassifier {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 4 {
            return Err(Error::Msg(format!(
                "MiniImageClassifier expects 4D input [B, C, H, W], got {:?}",
                x.shape()
            )));
        }
        let b = x.shape()[0];

        // Stage 1: Conv1 + ReLU + MaxPool1 -> [B, 16, H/2, W/2]
        let c1 = self.conv1.forward(x)?.relu()?;
        let p1 = self.pool1.forward(&c1)?;

        // Stage 2: Conv2 + ReLU + MaxPool2 -> [B, 32, H/4, W/4]
        let c2 = self.conv2.forward(&p1)?.relu()?;
        let p2 = self.pool2.forward(&c2)?;

        // Flatten to [B, flat_dim]
        let flat = p2.reshape(&[b, self.flat_dim])?;

        // Stage 3 & 4: Dense FC layers
        let h = self.fc1.forward(&flat)?.relu()?;
        self.fc2.forward(&h)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.conv1.named_parameters() {
            out.push((format!("conv1.{k}"), v));
        }
        for (k, v) in self.conv2.named_parameters() {
            out.push((format!("conv2.{k}"), v));
        }
        for (k, v) in self.fc1.named_parameters() {
            out.push((format!("fc1.{k}"), v));
        }
        for (k, v) in self.fc2.named_parameters() {
            out.push((format!("fc2.{k}"), v));
        }
        out
    }
}
