//! Differential Privacy (DP-SGD) and Byzantine-Resilient Federated Aggregation.
//!
//! Features:
//! - DP-SGD: Per-sample gradient clipping by norm $C$ and calibrated Gaussian noise injection $\sigma C$.
//! - Byzantine-Tolerant Aggregators against adversarial or faulty nodes in decentralized training:
//!   - [`ByzantineAggregator::multi_krum`]: Distance-based outlier rejection with theoretical optimal breakdown point.
//!   - [`ByzantineAggregator::trimmed_mean`]: Coordinate-wise outlier elimination trimming extreme tails.
//!   - [`ByzantineAggregator::bulyan`]: Hybrid aggregation combining Multi-Krum candidate filtering with trimmed mean.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Differential Privacy hyperparameters for DP-SGD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpSgdConfig {
    /// Maximum gradient $L_2$ norm bound $C$.
    pub max_grad_norm: f64,
    /// Noise multiplier $\sigma$ determining standard deviation $\sigma \cdot C$.
    pub noise_multiplier: f64,
}

impl Default for DpSgdConfig {
    fn default() -> Self {
        Self {
            max_grad_norm: 1.0,
            noise_multiplier: 0.1,
        }
    }
}

impl DpSgdConfig {
    pub fn new(max_grad_norm: f64, noise_multiplier: f64) -> Self {
        Self {
            max_grad_norm,
            noise_multiplier,
        }
    }

    /// Clips gradient tensor to $L_2$ norm bound and injects calibrated Gaussian noise:
    /// $$\tilde{g} = g \cdot \min\left(1, \frac{C}{\|g\|_2}\right) + \mathcal{N}(0, (\sigma C)^2 I)$$
    pub fn clip_and_noise(&self, grad: &Tensor) -> Result<Tensor> {
        if !self.max_grad_norm.is_finite() || self.max_grad_norm < 0.0 {
            return Err(Error::Msg(format!(
                "DpSgdConfig: max_grad_norm must be finite and >= 0, got {}",
                self.max_grad_norm
            )));
        }
        if !self.noise_multiplier.is_finite() || self.noise_multiplier < 0.0 {
            return Err(Error::Msg(format!(
                "DpSgdConfig: noise_multiplier must be finite and >= 0, got {}",
                self.noise_multiplier
            )));
        }
        let l2_norm = grad.norm()?.item::<f64>()?;
        let scale = if l2_norm > self.max_grad_norm {
            self.max_grad_norm / (l2_norm + 1e-12)
        } else {
            1.0
        };

        let clipped = grad.mul_scalar(scale)?;
        let noise_std = self.noise_multiplier * self.max_grad_norm;
        if noise_std > 0.0 {
            let noise = Tensor::randn(grad.shape())?
                .to_dtype(grad.dtype())?
                .mul_scalar(noise_std)?;
            clipped.add(&noise)
        } else {
            Ok(clipped)
        }
    }
}

/// Byzantine-Resilient Distributed Aggregator suite.
pub struct ByzantineAggregator;

impl ByzantineAggregator {
    /// Multi-Krum aggregation rule:
    /// Selects $k$ worker gradients with minimum sum of squared Euclidean distances to their
    /// closest $m - f - 2$ neighbors, where $m$ is total workers and $f$ is maximum Byzantine adversaries.
    pub fn multi_krum(
        gradients: &[Tensor],
        num_byzantine: usize,
        num_selected: usize,
    ) -> Result<Tensor> {
        let m = gradients.len();
        if m == 0 {
            return Err(Error::Msg("multi_krum requires at least 1 gradient".into()));
        }
        for g in gradients.iter().skip(1) {
            if g.shape() != gradients[0].shape() {
                return Err(Error::ShapeMismatch {
                    op: "multi_krum",
                    expected: gradients[0].shape().to_vec(),
                    got: g.shape().to_vec(),
                });
            }
        }
        if num_byzantine.saturating_mul(2).saturating_add(2) >= m && m > 2 {
            return Err(Error::Msg(format!(
                "Multi-Krum requires 2*f + 2 < m, but f={}, m={}",
                num_byzantine, m
            )));
        }

        let k_neighbors = if m > num_byzantine.saturating_add(2) {
            m - num_byzantine - 2
        } else {
            1
        };

        // Flatten all gradients to 1D vectors for exact distance calculation
        let flat_grads: Vec<Vec<f64>> = gradients
            .iter()
            .map(|g| g.to_vec::<f64>())
            .collect::<Result<Vec<_>>>()?;

        // Rank indices by lowest Krum score (shared with Bulyan selection).
        let all: Vec<usize> = (0..m).collect();
        let ranking = Self::krum_ranking(&flat_grads, &all, k_neighbors);

        let num_to_average = num_selected.clamp(1, m);
        let selected_indices: Vec<usize> = ranking.into_iter().take(num_to_average).collect();

        // Average selected gradients
        let mut acc = gradients[selected_indices[0]].clone();
        for &idx in &selected_indices[1..] {
            acc = acc.add(&gradients[idx])?;
        }
        acc.div_scalar(num_to_average as f64)
    }

    /// Coordinate-wise Trimmed Mean aggregator:
    /// Trims the highest and lowest $\beta$-fraction of values per coordinate across workers.
    pub fn trimmed_mean(gradients: &[Tensor], trim_fraction: f64) -> Result<Tensor> {
        let m = gradients.len();
        if m == 0 {
            return Err(Error::Msg(
                "trimmed_mean requires at least 1 gradient".into(),
            ));
        }
        for g in gradients.iter().skip(1) {
            if g.shape() != gradients[0].shape() {
                return Err(Error::ShapeMismatch {
                    op: "trimmed_mean",
                    expected: gradients[0].shape().to_vec(),
                    got: g.shape().to_vec(),
                });
            }
        }
        if !trim_fraction.is_finite() {
            return Err(Error::Msg(format!(
                "trimmed_mean: trim_fraction must be finite, got {}",
                trim_fraction
            )));
        }
        let trim_count = ((m as f64) * trim_fraction.clamp(0.0, 0.49)).floor() as usize;

        let numel = gradients[0].numel();
        let shape = gradients[0].shape().to_vec();

        let flat_grads: Vec<Vec<f64>> = gradients
            .iter()
            .map(|g| g.to_vec::<f64>())
            .collect::<Result<Vec<_>>>()?;

        let mut result_vec = vec![0.0f64; numel];
        let remaining_count = (m - 2 * trim_count).max(1);

        for p in 0..numel {
            let mut col_vals: Vec<f64> = (0..m).map(|worker| flat_grads[worker][p]).collect();
            col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let trimmed = &col_vals[trim_count..(m - trim_count)];
            let mean: f64 = trimmed.iter().sum::<f64>() / (remaining_count as f64);
            result_vec[p] = mean;
        }

        Tensor::from_vec(result_vec, &shape)
    }

    /// Bulyan robust aggregator:
    /// Combines Multi-Krum selection with component-wise trimmed mean.
    pub fn bulyan(gradients: &[Tensor], num_byzantine: usize) -> Result<Tensor> {
        let m = gradients.len();
        if m > 0 {
            for g in gradients.iter().skip(1) {
                if g.shape() != gradients[0].shape() {
                    return Err(Error::ShapeMismatch {
                        op: "bulyan",
                        expected: gradients[0].shape().to_vec(),
                        got: g.shape().to_vec(),
                    });
                }
            }
        }
        if m < num_byzantine.saturating_mul(4).saturating_add(3) {
            // Fall back to trimmed mean if worker count insufficient for full Bulyan guarantee
            return Self::trimmed_mean(gradients, (num_byzantine as f64) / (m as f64));
        }

        // Iteratively select the lowest-Krum-score candidate and remove it
        // from the pool, so each of the theta rounds picks a new gradient.
        let theta = m - 2 * num_byzantine;
        let flat_grads: Vec<Vec<f64>> = gradients
            .iter()
            .map(|g| g.to_vec::<f64>())
            .collect::<Result<Vec<_>>>()?;
        let mut remaining: Vec<usize> = (0..m).collect();
        let mut candidates = Vec::with_capacity(theta);
        for _ in 0..theta {
            let k_neighbors = if remaining.len() > num_byzantine + 2 {
                remaining.len() - num_byzantine - 2
            } else {
                1
            };
            let ranking = Self::krum_ranking(&flat_grads, &remaining, k_neighbors);
            let best = ranking[0];
            candidates.push(gradients[best].clone());
            remaining.retain(|&i| i != best);
        }

        Self::trimmed_mean(
            &candidates,
            (num_byzantine as f64) / (candidates.len() as f64),
        )
    }

    /// Ranks candidate indices by Krum score (sum of squared distances to the
    /// `k_neighbors` nearest neighbors), lowest score first.
    fn krum_ranking(flat_grads: &[Vec<f64>], indices: &[usize], k_neighbors: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = indices
            .iter()
            .map(|&i| {
                let mut dists: Vec<f64> = indices
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| {
                        flat_grads[i]
                            .iter()
                            .zip(flat_grads[j].iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum()
                    })
                    .collect();
                dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                (i, dists.iter().take(k_neighbors).sum())
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().map(|(idx, _)| idx).collect()
    }
}
