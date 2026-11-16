//! Mutual Information Neural Estimation (MINE) & InfoNCE Contrastive Information Bounds.
//!
//! Features:
//! - Donsker-Varadhan Variational Dual Bound on Kullback-Leibler Divergence:
//!   $$I_{DV}(X; Y) = \sup_{T \in \mathcal{F}} \mathbb{E}_{\mathbb{P}_{XY}}[T(x, y)] - \log \left( \mathbb{E}_{\mathbb{P}_X \otimes \mathbb{P}_Y}[e^{T(x, y)}] \right)$$
//!   with unbiased exponential moving average (EMA) denominator estimation.
//! - InfoNCE Contrastive Lower Bound for self-supervised representation learning:
//!   $$I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}$$
//! - Exact estimation on correlated Gaussians and complex multi-modal representations.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Mutual Information Neural Estimator (MINE) via Donsker-Varadhan representation.
pub struct MineEstimator {
    pub x_dim: usize,
    pub y_dim: usize,
    pub hidden_dim: usize,
    pub fc1: Linear,
    pub fc2: Linear,
    pub ema_denominator: f64,
    pub ema_decay: f64,
}

impl MineEstimator {
    /// Creates a new `MineEstimator` for random variables in $\mathbb{R}^{d_x}$ and $\mathbb{R}^{d_y}$.
    pub fn new(x_dim: usize, y_dim: usize, hidden_dim: usize, ema_decay: f64) -> Result<Self> {
        let fc1 = Linear::new(x_dim + y_dim, hidden_dim)?;
        let fc2 = Linear::new(hidden_dim, 1)?;

        Ok(Self {
            x_dim,
            y_dim,
            hidden_dim,
            fc1,
            fc2,
            ema_denominator: 1.0,
            ema_decay: ema_decay.clamp(0.0, 0.999),
        })
    }

    /// Forward pass through statistics network $T(x, y)$:
    /// Input `xy`: `[Batch, x_dim + y_dim]`. Output: `[Batch, 1]`.
    pub fn forward_statistic(&self, xy: &Tensor) -> Result<Tensor> {
        let xy_cast = xy.to_dtype(self.fc1.weight.dtype())?;
        let h = self.fc1.forward(&xy_cast)?.gelu()?;
        self.fc2.forward(&h)
    }

    /// Evaluates mutual information estimate $I(X; Y)$ in nats:
    /// - `x`: `[Batch, x_dim]`
    /// - `y`: `[Batch, y_dim]`
    pub fn estimate(&mut self, x: &Tensor, y: &Tensor) -> Result<f64> {
        if x.ndim() != 2 || y.ndim() != 2 {
            return Err(Error::Msg(format!(
                "MINE estimate x/y must be 2D, got {:?}/{:?}",
                x.shape(),
                y.shape()
            )));
        }
        let b = x.shape()[0];
        if y.shape()[0] != b {
            return Err(Error::Msg("Batch sizes must match".into()));
        }

        let y_cast = y.to_dtype(x.dtype())?;

        // Joint sample: (x_i, y_i)
        let joint_xy = Tensor::cat(&[x.clone(), y_cast.clone()], -1)?;
        let t_joint = self.forward_statistic(&joint_xy)?;
        let mean_joint = t_joint.mean()?.item::<f64>()?;

        // Marginal product sample: (x_i, y_{perm_i}) - circularly rolled along batch dimension
        let y_perm = self.roll_batch(&y_cast)?;
        let marginal_xy = Tensor::cat(&[x.clone(), y_perm], -1)?;
        let t_marginal = self.forward_statistic(&marginal_xy)?;

        let exp_marginal = t_marginal.exp()?;
        let batch_exp_mean = exp_marginal.mean()?.item::<f64>()?.max(1e-12);

        // Update exponential moving average denominator to reduce variance
        self.ema_denominator =
            self.ema_decay * self.ema_denominator + (1.0 - self.ema_decay) * batch_exp_mean;

        let mi = mean_joint - self.ema_denominator.ln();
        Ok(mi.max(0.0))
    }

    /// Negative MINE objective loss for gradient descent optimization:
    /// $$\mathcal{L}_{\text{MINE}} = -\left(\mathbb{E}_{\text{joint}}\[T\] - \log(\mathbb{E}_{\text{marginal}}\[e^T\])\right)$$
    pub fn loss(&mut self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let y_cast = y.to_dtype(x.dtype())?;
        let joint_xy = Tensor::cat(&[x.clone(), y_cast.clone()], -1)?;
        let t_joint = self.forward_statistic(&joint_xy)?;
        let mean_joint = t_joint.mean()?;

        let y_perm = self.roll_batch(&y_cast)?;
        let marginal_xy = Tensor::cat(&[x.clone(), y_perm], -1)?;
        let t_marginal = self.forward_statistic(&marginal_xy)?;

        let exp_marg = t_marginal.exp()?.mean()?;
        let log_exp = exp_marg.ln()?;

        log_exp.sub(&mean_joint)
    }

    fn roll_batch(&self, tensor: &Tensor) -> Result<Tensor> {
        let b = tensor.shape()[0];
        if b <= 1 {
            return Ok(tensor.clone());
        }
        // Roll: take [1..b] and append [0]
        let tail = tensor.narrow(0, 1, b - 1)?;
        let head = tensor.narrow(0, 0, 1)?;
        Tensor::cat(&[tail, head], 0)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }
}

/// InfoNCE (Contrastive) Mutual Information Estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InfoNceEstimator {
    pub temperature: f64,
}

impl Default for InfoNceEstimator {
    fn default() -> Self {
        Self { temperature: 0.07 }
    }
}

impl InfoNceEstimator {
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature: temperature.max(1e-4),
        }
    }

    /// Evaluates the normalized InfoNCE cross-entropy loss:
    /// - `queries`: `[Batch, D]`
    /// - `keys`: `[Batch, D]` (positive pair at matching index $i$)
    pub fn compute_loss(&self, queries: &Tensor, keys: &Tensor) -> Result<Tensor> {
        if queries.ndim() != 2 || keys.ndim() != 2 {
            return Err(Error::Msg(format!(
                "InfoNCE queries/keys must be 2D, got {:?}/{:?}",
                queries.shape(),
                keys.shape()
            )));
        }
        if queries.shape()[0] != keys.shape()[0] {
            return Err(Error::ShapeMismatch {
                op: "InfoNCE",
                expected: queries.shape().to_vec(),
                got: keys.shape().to_vec(),
            });
        }
        let b = queries.shape()[0];
        let tau = self.temperature;

        // Similarity matrix: S = Q * K^T / tau of shape [B, B]
        let logits = queries.matmul(&keys.t()?)?.mul_scalar(1.0 / tau)?;

        // Target labels: diagonal indices 0..B
        let exp_logits = logits.exp()?;
        let sum_exp = exp_logits.sum_axis(1, true)?;

        // Extract diagonal positive logits
        let mut pos_logits = Vec::with_capacity(b);
        let logit_data = logits.to_vec::<f64>()?;
        let cols = logits.shape()[1];
        for i in 0..b {
            pos_logits.push(logit_data[i * cols + i]);
        }
        let pos_tensor = Tensor::from_vec(pos_logits, &[b, 1])?.to_dtype(logits.dtype())?;

        let log_denom = sum_exp.ln()?;
        let nce_loss = log_denom.sub(&pos_tensor)?.mean()?;
        Ok(nce_loss)
    }

    /// Lower bound on Mutual Information in nats:
    /// $$I(X; Y) \ge \log(B) - \mathcal{L}_{\text{InfoNCE}}$$
    pub fn lower_bound(&self, queries: &Tensor, keys: &Tensor) -> Result<f64> {
        let loss = self.compute_loss(queries, keys)?.item::<f64>()?;
        let b = queries.shape()[0] as f64;
        Ok((b.ln() - loss).max(0.0))
    }
}
