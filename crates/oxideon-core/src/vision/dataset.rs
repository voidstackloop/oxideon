//! Procedural 1000 Mini Images Dataset Generator.
//!
//! Generates balanced, distinct, procedurally patterned $16 \times 16$ or $28 \times 28$
//! mini-images across 4 visual pattern classes:
//! - Class 0: Horizontal Harmonic Waveforms / Gratings
//! - Class 1: Vertical Harmonic Waveforms / Gratings
//! - Class 2: Concentric Radial Rings / Discs
//! - Class 3: Diagonal Waveforms / Checkerboards
//!
//! Includes deterministic Gaussian perturbation, random frequency jitter,
//! and phase shifts, ensuring all 1,000 images are distinct and challenging.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// A procedural collection of mini images with ground-truth class labels.
#[derive(Debug, Clone)]
pub struct MiniImageDataset {
    /// 4D image tensor with shape `[N, channels, height, width]`.
    pub images: Tensor,
    /// Integer ground-truth class labels for each image `[N]`.
    pub labels: Vec<usize>,
    pub num_classes: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}

impl MiniImageDataset {
    /// Generates `num_images` (e.g. 1000) mini images of dimension `channels x height x width`.
    pub fn generate(
        num_images: usize,
        channels: usize,
        height: usize,
        width: usize,
        seed: u64,
    ) -> Result<Self> {
        if num_images == 0 {
            return Err(Error::Msg("num_images must be greater than 0".into()));
        }
        crate::limits::check_elements("MiniImageDataset", &[num_images, channels, height, width])?;
        let num_classes = 4;
        let mut rng = SimpleRng::new(seed);
        let mut all_data = Vec::with_capacity(num_images * channels * height * width);
        let mut labels = Vec::with_capacity(num_images);

        for i in 0..num_images {
            let class_id = i % num_classes;
            labels.push(class_id);

            // Per-image parameter perturbations
            let freq = 6.0 + rng.next_f64() * 6.0;
            let phase = rng.next_f64() * std::f64::consts::PI * 2.0;
            let cx = (rng.next_f64() - 0.5) * 0.2;
            let cy = (rng.next_f64() - 0.5) * 0.2;

            for _c in 0..channels {
                for y_idx in 0..height {
                    let y = -1.0 + 2.0 * (y_idx as f64) / ((height - 1) as f64) - cy;
                    for x_idx in 0..width {
                        let x = -1.0 + 2.0 * (x_idx as f64) / ((width - 1) as f64) - cx;

                        let base_val = match class_id {
                            // Class 0: Horizontal Gratings
                            0 => (y * freq + phase).sin(),
                            // Class 1: Vertical Gratings
                            1 => (x * freq + phase).sin(),
                            // Class 2: Concentric Radial Discs
                            2 => {
                                let r = (x * x + y * y).sqrt();
                                (r * freq * 1.5 + phase).sin()
                            }
                            // Class 3: Diagonal Waveforms
                            _ => ((x + y) * freq * 0.707 + phase).sin(),
                        };

                        // Add small Gaussian-like noise perturbation
                        let noise = (rng.next_f64() - 0.5) * 0.15;
                        let pixel = ((base_val + noise + 1.0) * 0.5).clamp(0.0, 1.0) as f32;
                        all_data.push(pixel);
                    }
                }
            }
        }

        // Deterministic Fisher-Yates shuffle across the 1000 items
        let mut indices: Vec<usize> = (0..num_images).collect();
        for i in (1..num_images).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let item_len = channels * height * width;
        let mut shuffled_data = Vec::with_capacity(all_data.len());
        let mut shuffled_labels = Vec::with_capacity(labels.len());

        for &idx in &indices {
            let start = idx * item_len;
            shuffled_data.extend_from_slice(&all_data[start..start + item_len]);
            shuffled_labels.push(labels[idx]);
        }

        let images = Tensor::from_vec(shuffled_data, &[num_images, channels, height, width])?;

        Ok(Self {
            images,
            labels: shuffled_labels,
            num_classes,
            height,
            width,
            channels,
        })
    }

    /// Convenience constructor generating standard 1,000 mini images ($1 \times 16 \times 16$).
    pub fn standard_1000() -> Result<Self> {
        Self::generate(1000, 1, 16, 16, 42)
    }

    /// Returns the number of images in this dataset.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Self-consistency guard: layout fields are public and hand-tamperable,
    /// and `split`/`batch` slice storage by them. Returns the item length.
    fn check_layout(&self) -> Result<usize> {
        let s = self.images.shape();
        if s.len() != 4
            || s[0] != self.labels.len()
            || s[1] != self.channels
            || s[2] != self.height
            || s[3] != self.width
        {
            return Err(Error::Msg(format!(
                "MiniImageDataset: images shape {s:?} disagrees with layout [n={}, c={}, h={}, w={}]",
                self.labels.len(),
                self.channels,
                self.height,
                self.width
            )));
        }
        Ok(self.channels * self.height * self.width)
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Splits the dataset into train and validation sets, e.g. 80/20 split.
    pub fn split(&self, train_ratio: f64) -> Result<(Self, Self)> {
        let n = self.len();
        let train_n = ((n as f64) * train_ratio.clamp(0.0, 1.0)).round() as usize;
        let val_n = n.saturating_sub(train_n);

        if train_n == 0 || val_n == 0 {
            return Err(Error::Msg("Split resulted in an empty subset".into()));
        }

        let item_len = self.check_layout()?;
        let all_vec = self.images.to_vec::<f32>()?;

        let train_data = all_vec[..train_n * item_len].to_vec();
        let val_data = all_vec[train_n * item_len..].to_vec();

        let train_images = Tensor::from_vec(
            train_data,
            &[train_n, self.channels, self.height, self.width],
        )?;
        let val_images =
            Tensor::from_vec(val_data, &[val_n, self.channels, self.height, self.width])?;

        let train_labels = self.labels[..train_n].to_vec();
        let val_labels = self.labels[train_n..].to_vec();

        let train_set = Self {
            images: train_images,
            labels: train_labels,
            num_classes: self.num_classes,
            height: self.height,
            width: self.width,
            channels: self.channels,
        };

        let val_set = Self {
            images: val_images,
            labels: val_labels,
            num_classes: self.num_classes,
            height: self.height,
            width: self.width,
            channels: self.channels,
        };

        Ok((train_set, val_set))
    }

    /// Returns the number of mini-batches given a batch size.
    pub fn num_batches(&self, batch_size: usize) -> usize {
        if batch_size == 0 {
            return 0;
        }
        self.len().div_ceil(batch_size)
    }

    /// Retrieves the mini-batch at index `batch_idx`.
    pub fn batch(&self, batch_idx: usize, batch_size: usize) -> Result<(Tensor, Vec<usize>)> {
        let n = self.len();
        let start = batch_idx.saturating_mul(batch_size);
        if start >= n {
            return Err(Error::Msg(format!(
                "Batch index {batch_idx} out of range for dataset of size {n}"
            )));
        }
        let end = start.saturating_add(batch_size).min(n);
        let actual_batch = end - start;

        let item_len = self.check_layout()?;
        let all_vec = self.images.to_vec::<f32>()?;
        let batch_data = all_vec[start * item_len..end * item_len].to_vec();

        let batch_tensor = Tensor::from_vec(
            batch_data,
            &[actual_batch, self.channels, self.height, self.width],
        )?;
        let batch_labels = self.labels[start..end].to_vec();

        Ok((batch_tensor, batch_labels))
    }

    /// Returns the ground-truth labels as a 1D tensor.
    pub fn labels_tensor(&self) -> Result<Tensor> {
        let f_labels: Vec<f32> = self.labels.iter().map(|&l| l as f32).collect();
        Tensor::from_vec(f_labels, &[self.len()])
    }
}

/// Lightweight deterministic 64-bit XorShift pseudorandom generator.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}
