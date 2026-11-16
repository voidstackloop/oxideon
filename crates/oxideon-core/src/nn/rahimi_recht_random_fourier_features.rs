//! Bochner's Harmonic Theorem & Rahimi-Recht Random Fourier Features (RFF) for Shift-Invariant Kernels (Rahimi & Recht, 2007).
//!
//! ### 1. Bochner's Theorem & Unbiased Cosine Feature Embedding
//! By Bochner's Theorem, a continuous shift-invariant kernel $k(x, y) = k(x - y)$ on $\mathbb{R}^d$ with $k(0) = 1$ is positive-definite
//! iff $k(\delta) = \mathbb{E}_{\omega \sim p}[\cos(\omega^T \delta)]$. Rahimi & Recht map $x \in \mathbb{R}^d$ to $z(x) \in \mathbb{R}^{2D}$:
//! $$z(x) = \frac{1}{\sqrt{D}} \left[\cos(\omega_1^T x), \sin(\omega_1^T x), \dots, \cos(\omega_D^T x), \sin(\omega_D^T x)\right]^T$$
//! satisfying $\mathbb{E}[z(x)^T z(y)] = \frac{1}{D}\sum_{j=1}^D \cos(\omega_j^T (x - y)) = k(x - y)$.

use crate::error::{Error, Result};

/// Rahimi-Recht Random Fourier Features (RFF) & Bochner Kernel Engine.
#[derive(Debug, Clone)]
pub struct RandomFourierBochnerEngine {
    /// Flattened frequency matrix $\Omega$ of shape `num_features_d * input_dim`.
    pub frequencies: Vec<f64>,
    /// Input dimension $d$.
    pub input_dim: usize,
}

impl RandomFourierBochnerEngine {
    /// Creates a new Rahimi-Recht Random Fourier Feature engine.
    pub fn new(frequencies: Vec<f64>, input_dim: usize) -> Result<Self> {
        if frequencies.is_empty() || input_dim == 0 || frequencies.len() % input_dim != 0 {
            return Err(Error::Msg(
                "Invalid frequency matrix dimensions".to_string(),
            ));
        }
        Ok(Self {
            frequencies,
            input_dim,
        })
    }

    /// Evaluates the trigonometric RFF kernel approximation $\hat{k}(x, y) = z(x)^T z(y) = \frac{1}{D} \sum_{j=1}^D \cos(\omega_j^T (x - y))$.
    pub fn approximate_kernel(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != self.input_dim || y.len() != self.input_dim {
            return Err(Error::Msg("Input dimension mismatch".to_string()));
        }
        let d_feat = self.frequencies.len() / self.input_dim;
        let mut sum = 0.0;
        for j in 0..d_feat {
            let mut dot = 0.0;
            for k in 0..self.input_dim {
                dot += self.frequencies[j * self.input_dim + k] * (x[k] - y[k]);
            }
            sum += dot.cos();
        }
        Ok(sum / (d_feat as f64))
    }
}
