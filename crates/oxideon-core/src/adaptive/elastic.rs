//! Elastic Subnet Pruning & Dynamic Architecture Morphing (Once-for-All / Matryoshka).
//!
//! Enables dynamic extraction and execution of arbitrary width subnets (e.g. 25%, 50%, 75%, 100%)
//! from a single full-capacity model at runtime without retraining or duplicating weights.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Dynamic Elastic Subnet Extractor for zero-copy runtime model slimming.
pub struct ElasticSubnetExtractor;

impl ElasticSubnetExtractor {
    /// Slices weight `[out_features, in_features]` and optional bias `[out_features]`
    /// down to fractional width ratios `in_ratio` and `out_ratio` $\in (0, 1]$.
    pub fn slice_linear_width(
        weight: &Tensor,
        bias: Option<&Tensor>,
        in_ratio: f64,
        out_ratio: f64,
    ) -> Result<(Tensor, Option<Tensor>)> {
        if weight.ndim() != 2 {
            return Err(Error::Msg(format!(
                "Elastic linear weight must be 2D, got shape {:?}",
                weight.shape()
            )));
        }
        let out_features = weight.shape()[0];
        let in_features = weight.shape()[1];
        // `.clamp(1, features)` below panics (min > max) on an empty dim.
        if in_features == 0 || out_features == 0 {
            return Err(Error::Msg(format!(
                "Elastic linear weight must have positive features, got [{out_features}, {in_features}]"
            )));
        }

        let target_in = ((in_features as f64) * in_ratio.clamp(0.01, 1.0)).round() as usize;
        let target_out = ((out_features as f64) * out_ratio.clamp(0.01, 1.0)).round() as usize;

        let target_in = target_in.clamp(1, in_features);
        let target_out = target_out.clamp(1, out_features);

        let sub_weight = weight.narrow(0, 0, target_out)?.narrow(1, 0, target_in)?;

        let sub_bias = match bias {
            Some(b) => Some(b.narrow(0, 0, target_out)?),
            None => None,
        };

        Ok((sub_weight, sub_bias))
    }

    /// Evaluates an elastic forward pass: $Y = X_{\text{sliced}} W_{\text{sliced}}^T + b_{\text{sliced}}$.
    pub fn evaluate_elastic_forward(
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        width_ratio: f64,
    ) -> Result<Tensor> {
        let (sub_w, sub_b) = Self::slice_linear_width(weight, bias, width_ratio, width_ratio)?;
        let sub_in = sub_w.shape()[1];
        let x_sliced = x.narrow(-1, 0, sub_in)?;
        let mut y = x_sliced.matmul(&sub_w.t()?)?;
        if let Some(b) = sub_b {
            y = y.add(&b)?;
        }
        Ok(y)
    }
}
