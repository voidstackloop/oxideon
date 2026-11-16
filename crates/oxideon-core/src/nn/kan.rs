//! Kolmogorov–Arnold Networks (`KANLinear` & `KAN`, Liu et al. 2024).
//!
//! Replaces fixed node-level activations with learnable univariate basis expansions on edges:
//! $$y_j = \sum_{i=1}^{d_{\text{in}}} \left( w_{j,i}^{\text{base}} \operatorname{SiLU}(x_i) + \sum_{g=1}^{G} c_{j,i,g}\, \exp\!\left(-\frac{(x_i - \mu_g)^2}{h^2}\right) \right)$$

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::{LayerNorm, Module, init};
use crate::tensor::Tensor;

/// A single Kolmogorov–Arnold Network layer (`KANLinear`) with `grid_size` learnable basis functions per edge.
pub struct KANLinear {
    pub base_weight: Tensor,
    pub spline_weight: Tensor,
    pub grid: Tensor,
    pub in_features: usize,
    pub out_features: usize,
    pub grid_size: usize,
    pub inv_bandwidth: f64,
}

impl KANLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        grid_range: (f64, f64),
    ) -> Result<Self> {
        let g = grid_size.max(2);
        let spline_in = in_features.checked_mul(g).ok_or_else(|| {
            Error::Msg(format!(
                "KANLinear sizes overflow: in_features={in_features}, grid_size={grid_size}"
            ))
        })?;
        let bound_base = 1.0 / (in_features as f32).sqrt();
        let bound_spline = 1.0 / (spline_in as f32).sqrt();

        let base_weight = init::uniform(&[out_features, in_features], -bound_base, bound_base)?
            .requires_grad_(true);
        let spline_weight = init::uniform(&[out_features, spline_in], -bound_spline, bound_spline)?
            .requires_grad_(true);

        let grid =
            Tensor::linspace(grid_range.0, grid_range.1, g, DType::F32)?.reshape(&[1, 1, g])?;
        let step = (grid_range.1 - grid_range.0).abs() / ((g - 1) as f64);
        let inv_bandwidth = 1.0 / step.max(1e-6);

        Ok(Self {
            base_weight,
            spline_weight,
            grid,
            in_features,
            out_features,
            grid_size: g,
            inv_bandwidth,
        })
    }
}

impl Module for KANLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let orig_shape = input.shape().to_vec();
        let d_in = *orig_shape
            .last()
            .ok_or_else(|| Error::Msg("KANLinear expects at least 2D input".into()))?;
        if d_in != self.in_features {
            return Err(Error::ShapeMismatch {
                op: "KANLinear",
                expected: vec![self.in_features],
                got: orig_shape,
            });
        }

        // Leading-dim product (not numel / d_in: d_in may be 0 for empty features).
        let rows = orig_shape[..orig_shape.len() - 1]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "KANLinear: batch shape {orig_shape:?} element count overflows"
                ))
            })?;
        let x_2d = input.reshape(&[rows, d_in])?;

        // 1. Base SiLU branch: [rows, d_in] @ [d_in, d_out]
        let base_out = x_2d.silu()?.matmul(&self.base_weight.t()?)?;

        // 2. RBF spline basis branch:
        // x_3d: [rows, d_in, 1], grid: [1, 1, G] -> diff: [rows, d_in, G]
        let grid = self
            .grid
            .to_device(&input.device())?
            .to_dtype(input.dtype())?;
        let diff = x_2d
            .unsqueeze(2)?
            .sub(&grid)?
            .mul_scalar(self.inv_bandwidth)?;
        let spline_cols = d_in.checked_mul(self.grid_size).ok_or_else(|| {
            Error::Msg(format!(
                "KANLinear: basis width {d_in} * grid_size {} overflows",
                self.grid_size
            ))
        })?;
        let basis = diff.square()?.neg()?.exp()?.reshape(&[rows, spline_cols])?;
        let spline_out = basis.matmul(&self.spline_weight.t()?)?;

        let out_2d = base_out.add(&spline_out)?;
        let mut out_shape = orig_shape[..orig_shape.len() - 1].to_vec();
        out_shape.push(self.out_features);
        out_2d.reshape(&out_shape)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("base_weight".to_string(), self.base_weight.clone()),
            ("spline_weight".to_string(), self.spline_weight.clone()),
        ]
    }
}

/// Multi-layer Kolmogorov–Arnold Network (`KAN`) stacking [`KANLinear`] and [`LayerNorm`] blocks.
pub struct KAN {
    pub layers: Vec<KANLinear>,
    pub norms: Vec<LayerNorm>,
}

impl KAN {
    pub fn new(layer_dims: &[usize], grid_size: usize) -> Result<Self> {
        if layer_dims.len() < 2 {
            return Err(Error::Msg(
                "KAN requires at least 2 layer dimensions".into(),
            ));
        }
        let num_layers = layer_dims.len() - 1;
        crate::limits::check_repeats("num_layers", num_layers)?;
        let mut layers = Vec::with_capacity(num_layers);
        let mut norms = Vec::with_capacity(num_layers.saturating_sub(1));

        for i in 0..num_layers {
            layers.push(KANLinear::new(
                layer_dims[i],
                layer_dims[i + 1],
                grid_size,
                (-2.0, 2.0),
            )?);
            if i + 1 < num_layers {
                norms.push(LayerNorm::new(layer_dims[i + 1], 1e-5)?);
            }
        }
        Ok(Self { layers, norms })
    }
}

impl Module for KAN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut h = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h)?;
            if let Some(norm) = self.norms.get(i) {
                h = norm.forward(&h)?;
            }
        }
        Ok(h)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (k, v) in l.named_parameters() {
                out.push((format!("layers.{i}.{k}"), v));
            }
        }
        for (i, n) in self.norms.iter().enumerate() {
            for (k, v) in n.named_parameters() {
                out.push((format!("norms.{i}.{k}"), v));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kan_rejects_tampered_grid_size() -> Result<()> {
        let mut layer = KANLinear::new(4, 4, 4, (-2.0, 2.0))?;
        layer.grid_size = usize::MAX; // pub field: d_in * grid_size would overflow
        let x = Tensor::zeros_f32(&[2, 4])?;
        let err = layer
            .forward(&x)
            .expect_err("tampered grid_size must be an error, not a panic");
        assert!(err.to_string().contains("grid"), "{err}");
        Ok(())
    }
}
