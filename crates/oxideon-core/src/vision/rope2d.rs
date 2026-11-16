//! 2D Rotary Position Embeddings (`2D RoPE`) for Vision Transformers & Vision DiT.
//!
//! Factorizes sequence coordinates across two spatial dimensions:
//! - Spatial vertical coordinate $y \in [0, H-1]$
//! - Spatial horizontal coordinate $x \in [0, W-1]$
//!
//! Provides mathematically exact analytical backward backpropagation via orthogonal
//! adjoint rotation $R(\theta)^{-1} = R(\theta)^T = R(-\theta)$.

use std::sync::Arc;

use crate::layout::Layout;
use crate::{CpuStorage, Error, Result, Storage, Tensor};

/// 2D Rotary Position Embedding decomposed across height ($y$) and width ($x$).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding2D {
    pub dim: usize,
    pub dim_h: usize,
    pub dim_w: usize,
    pub theta_base: f64,
}

impl RotaryEmbedding2D {
    /// Creates a 2D RoPE with balanced even partitions for height and width.
    pub fn new(dim: usize, theta_base: f64) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding2D dim {dim} must be even"
            )));
        }
        if dim < 4 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding2D dim {dim} is too small: height and width each need at least 2 (dim >= 4)"
            )));
        }

        // Allocate half to height and half to width (both even)
        let mut dh = dim / 2;
        if dh % 2 != 0 {
            dh += 1;
        }
        let Some(dw) = dim.checked_sub(dh) else {
            return Err(Error::Msg(format!(
                "invalid 2D RoPE partition ({dh}, ?) for dim {dim}"
            )));
        };
        if dw % 2 != 0 || dh + dw != dim {
            return Err(Error::Msg(format!(
                "invalid 2D RoPE partition ({dh}, {dw}) for dim {dim}"
            )));
        }

        Ok(Self {
            dim,
            dim_h: dh,
            dim_w: dw,
            theta_base,
        })
    }

    /// Creates a 2D RoPE with explicitly specified height and width partitions.
    pub fn with_partitions(dim_h: usize, dim_w: usize, theta_base: f64) -> Result<Self> {
        if dim_h % 2 != 0 || dim_w % 2 != 0 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding2D partitions must both be even, got ({dim_h}, {dim_w})"
            )));
        }
        if dim_h == 0 || dim_w == 0 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding2D partitions must be positive, got ({dim_h}, {dim_w})"
            )));
        }
        let dim = dim_h.saturating_add(dim_w);
        Ok(Self {
            dim,
            dim_h,
            dim_w,
            theta_base,
        })
    }

    /// Computes inverse frequencies for dimension `d`.
    pub fn inv_freq(d: usize, theta: f64) -> Vec<f64> {
        (0..d / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / (d as f64)))
            .collect()
    }

    /// Applies 2D RoPE to a 4D tensor `[batch, num_tokens, num_heads, head_dim]`
    /// or 3D tensor `[batch, num_tokens, head_dim]` where `num_tokens = H * W`.
    pub fn forward(&self, x: &Tensor, h_size: usize, w_size: usize) -> Result<Tensor> {
        let shape = x.shape();
        let (b, total_tokens, heads, d, _is_3d) = match shape.len() {
            4 => (shape[0], shape[1], shape[2], shape[3], false),
            3 => (shape[0], shape[1], 1, shape[2], true),
            _ => {
                return Err(Error::Msg(format!(
                    "RotaryEmbedding2D expected 3D or 4D tensor, got shape {:?}",
                    shape
                )));
            }
        };

        if d != self.dim {
            return Err(Error::Msg(format!(
                "RotaryEmbedding2D head_dim mismatch: expected {}, got {d}",
                self.dim
            )));
        }
        if h_size == 0 || w_size == 0 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding2D grid dims must be positive, got ({h_size}, {w_size})"
            )));
        }
        let grid = h_size.checked_mul(w_size).ok_or_else(|| {
            Error::Msg(format!(
                "RotaryEmbedding2D grid ({h_size}, {w_size}) overflows"
            ))
        })?;
        if total_tokens != grid {
            return Err(Error::Msg(format!(
                "tokens {total_tokens} != h({h_size}) * w({w_size})"
            )));
        }

        if self.dim_h % 2 != 0 || self.dim_w % 2 != 0 || self.dim_h + self.dim_w != self.dim {
            return Err(Error::Msg(format!(
                "RotaryEmbedding2D partitions ({}, {}) must be even and sum to {}",
                self.dim_h, self.dim_w, self.dim
            )));
        }

        let inv_freq_h = Arc::new(Self::inv_freq(self.dim_h, self.theta_base));
        let inv_freq_w = Arc::new(Self::inv_freq(self.dim_w, self.theta_base));

        let data = x.to_vec::<f32>()?;
        let mut out = data;

        // Apply 2D spatial rotation
        for token_idx in 0..total_tokens {
            let y = token_idx / w_size;
            let x_coord = token_idx % w_size;

            for batch_idx in 0..b {
                for head_idx in 0..heads {
                    let offset = ((batch_idx * total_tokens + token_idx) * heads + head_idx) * d;

                    // 1. Rotate spatial height partition [0 .. dim_h]
                    for (i, &freq) in inv_freq_h.iter().enumerate() {
                        let angle = (y as f64) * freq;
                        let cos = angle.cos() as f32;
                        let sin = angle.sin() as f32;
                        let idx0 = offset + 2 * i;
                        let idx1 = offset + 2 * i + 1;
                        let v0 = out[idx0];
                        let v1 = out[idx1];
                        out[idx0] = v0 * cos - v1 * sin;
                        out[idx1] = v0 * sin + v1 * cos;
                    }

                    // 2. Rotate spatial width partition [dim_h .. dim]
                    let w_offset = offset + self.dim_h;
                    for (i, &freq) in inv_freq_w.iter().enumerate() {
                        let angle = (x_coord as f64) * freq;
                        let cos = angle.cos() as f32;
                        let sin = angle.sin() as f32;
                        let idx0 = w_offset + 2 * i;
                        let idx1 = w_offset + 2 * i + 1;
                        let v0 = out[idx0];
                        let v1 = out[idx1];
                        out[idx0] = v0 * cos - v1 * sin;
                        out[idx1] = v0 * sin + v1 * cos;
                    }
                }
            }
        }

        let out_layout = Layout::contiguous(shape);
        let cpu_s = CpuStorage::F32(out);
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &x.device())?;

        let x_shape = x.shape().to_vec();
        let dev = x.device();
        let dt = x.dtype();
        let dh = self.dim_h;
        let _dw = self.dim_w;
        let dim_total = self.dim;
        let if_h = Arc::clone(&inv_freq_h);
        let if_w = Arc::clone(&inv_freq_w);

        Ok(Tensor::from_op(
            storage,
            out_layout,
            "rope2d",
            vec![x.clone()],
            move |grad_out| {
                let gv = grad_out.to_vec::<f32>()?;
                let mut dx = gv;

                // Orthogonal adjoint backward: R(theta)^T = R(-theta)
                for token_idx in 0..total_tokens {
                    let y = token_idx / w_size;
                    let x_coord = token_idx % w_size;

                    for batch_idx in 0..b {
                        for head_idx in 0..heads {
                            let offset = ((batch_idx * total_tokens + token_idx) * heads + head_idx) * dim_total;

                            // Height partition adjoint rotation
                            for (i, &freq) in if_h.iter().enumerate() {
                                let angle = (y as f64) * freq;
                                let cos = angle.cos() as f32;
                                let sin = angle.sin() as f32;
                                let idx0 = offset + 2 * i;
                                let idx1 = offset + 2 * i + 1;
                                let g0 = dx[idx0];
                                let g1 = dx[idx1];
                                dx[idx0] = g0 * cos + g1 * sin;
                                dx[idx1] = -g0 * sin + g1 * cos;
                            }

                            // Width partition adjoint rotation
                            let w_offset = offset + dh;
                            for (i, &freq) in if_w.iter().enumerate() {
                                let angle = (x_coord as f64) * freq;
                                let cos = angle.cos() as f32;
                                let sin = angle.sin() as f32;
                                let idx0 = w_offset + 2 * i;
                                let idx1 = w_offset + 2 * i + 1;
                                let g0 = dx[idx0];
                                let g1 = dx[idx1];
                                dx[idx0] = g0 * cos + g1 * sin;
                                dx[idx1] = -g0 * sin + g1 * cos;
                            }
                        }
                    }
                }

                let dx_t = Tensor::from_vec(dx, &x_shape)?
                    .to_device(&dev)?
                    .to_dtype(dt)?;
                Ok(vec![Some(dx_t)])
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope2d_partition_and_norm_conservation() {
        let rope = RotaryEmbedding2D::new(16, 10000.0).unwrap();
        assert_eq!(rope.dim, 16);
        assert_eq!(rope.dim_h, 8);
        assert_eq!(rope.dim_w, 8);

        // Grid 4x4 = 16 tokens, batch=1, heads=2, dim=16
        let x_data: Vec<f32> = (0..16 * 2 * 16)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let x = Tensor::from_vec(x_data, &[1, 16, 2, 16]).unwrap();

        let out = rope.forward(&x, 4, 4).unwrap();
        assert_eq!(out.shape(), &[1, 16, 2, 16]);

        let in_vec = x.to_vec::<f32>().unwrap();
        let out_vec = out.to_vec::<f32>().unwrap();

        // Check L2 norm conservation per head token: ||R x|| == ||x||
        for token in 0..16 {
            for h in 0..2 {
                let offset = (token * 2 + h) * 16;
                let in_norm: f32 = (0..16).map(|i| in_vec[offset + i].powi(2)).sum::<f32>().sqrt();
                let out_norm: f32 = (0..16).map(|i| out_vec[offset + i].powi(2)).sum::<f32>().sqrt();
                assert!((in_norm - out_norm).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_rope2d_autograd_adjoint_gradient() {
        let rope = RotaryEmbedding2D::new(8, 10000.0).unwrap();
        let x_data: Vec<f32> = (0..4 * 8).map(|i| i as f32 + 1.0).collect();
        let x = Tensor::from_vec(x_data, &[1, 4, 1, 8]).unwrap().requires_grad_(true);

        let out = rope.forward(&x, 2, 2).unwrap();
        let loss = out.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().expect("grad should exist");
        assert_eq!(grad.shape(), &[1, 4, 1, 8]);
    }
}
