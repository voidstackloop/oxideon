//! 3D Rotary Position Embeddings (`3D RoPE`) for Video & Spatio-Temporal Modeling.
//!
//! Factorizes sequence coordinates across three independent dimensions:
//! - Temporal frame index $t \in [0, T-1]$
//! - Spatial vertical coordinate $y \in [0, H-1]$
//! - Spatial horizontal coordinate $x \in [0, W-1]$

use std::sync::Arc;

use crate::layout::Layout;
use crate::{CpuStorage, Error, Result, Storage, Tensor};

/// 3D Rotary Position Embedding decomposed across time ($t$), height ($y$), and width ($x$).
#[derive(Debug, Clone)]
pub struct RotaryEmbedding3D {
    pub dim: usize,
    pub dim_t: usize,
    pub dim_h: usize,
    pub dim_w: usize,
    pub theta_base: f64,
}

impl RotaryEmbedding3D {
    pub fn new(dim: usize, theta_base: f64) -> Result<Self> {
        if dim % 2 != 0 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding3D dim {dim} must be even"
            )));
        }

        // Allocate head dimensions: ~1/4 to time, ~3/8 to height, ~3/8 to width
        let mut dt = (dim / 4).max(2);
        if dt % 2 != 0 {
            dt += 1;
        }
        let rem = dim.saturating_sub(dt);
        let mut dh = (rem / 2).max(2);
        if dh % 2 != 0 {
            dh += 1;
        }
        let Some(dw) = dim.checked_sub(dt + dh) else {
            return Err(Error::Msg(format!(
                "RotaryEmbedding3D dim {dim} is too small: time and height each need at least 2 (dim >= 4)"
            )));
        };
        if dw % 2 != 0 || dt + dh + dw != dim {
            return Err(Error::Msg(format!(
                "invalid 3D RoPE partition ({dt}, {dh}, {dw}) for dim {dim}"
            )));
        }

        Ok(Self {
            dim,
            dim_t: dt,
            dim_h: dh,
            dim_w: dw,
            theta_base,
        })
    }

    /// Computes inverse frequencies for dimension `d`.
    fn inv_freq(d: usize, theta: f64) -> Vec<f64> {
        (0..d / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / (d as f64)))
            .collect()
    }

    /// Applies 3D RoPE to a 4D tensor `[batch, num_tokens, num_heads, head_dim]`
    /// where `num_tokens = T * H * W`.
    pub fn forward(
        &self,
        x: &Tensor,
        t_size: usize,
        h_size: usize,
        w_size: usize,
    ) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding3D expected 4D tensor [B, tokens, heads, dim], got shape {:?}",
                shape
            )));
        }
        let (b, total_tokens, heads, d) = (shape[0], shape[1], shape[2], shape[3]);
        if d != self.dim {
            return Err(Error::Msg(format!(
                "RotaryEmbedding3D head_dim mismatch: expected {}, got {d}",
                self.dim
            )));
        }
        if t_size == 0 || h_size == 0 || w_size == 0 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding3D grid dims must be positive, got ({t_size}, {h_size}, {w_size})"
            )));
        }
        let grid = t_size
            .checked_mul(h_size)
            .and_then(|v| v.checked_mul(w_size))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "RotaryEmbedding3D grid ({t_size}, {h_size}, {w_size}) overflows"
                ))
            })?;
        if total_tokens != grid {
            return Err(Error::Msg(format!(
                "tokens {total_tokens} != t({t_size}) * h({h_size}) * w({w_size})"
            )));
        }
        // `dim_t/dim_h/dim_w` are public: re-check the constructor invariant
        // before indexing (saturating sum: tampered values may overflow, and a
        // wrong sum would rotate across head boundaries or panic).
        if self.dim_t % 2 != 0 || self.dim_h % 2 != 0 || self.dim_w % 2 != 0 {
            return Err(Error::Msg(format!(
                "RotaryEmbedding3D partitions must be even, got ({}, {}, {})",
                self.dim_t, self.dim_h, self.dim_w
            )));
        }
        let parts = self
            .dim_t
            .saturating_add(self.dim_h)
            .saturating_add(self.dim_w);
        if parts != self.dim {
            return Err(Error::Msg(format!(
                "RotaryEmbedding3D partitions ({}, {}, {}) sum to {parts}, expected dim {}",
                self.dim_t, self.dim_h, self.dim_w, self.dim
            )));
        }

        let inv_freq_t = Arc::new(Self::inv_freq(self.dim_t, self.theta_base));
        let inv_freq_h = Arc::new(Self::inv_freq(self.dim_h, self.theta_base));
        let inv_freq_w = Arc::new(Self::inv_freq(self.dim_w, self.theta_base));

        let mut data = x.to_vec::<f32>()?;

        // In-place rotation across each token coordinate (t, y, x)
        for token_idx in 0..total_tokens {
            let t = token_idx / (h_size * w_size);
            let rem = token_idx % (h_size * w_size);
            let y = rem / w_size;
            let x_coord = rem % w_size;

            for batch_idx in 0..b {
                for head_idx in 0..heads {
                    let offset = ((batch_idx * total_tokens + token_idx) * heads + head_idx) * d;

                    // 1. Rotate temporal partition [0 .. dim_t]
                    for (i, &freq) in inv_freq_t.iter().enumerate() {
                        let angle = (t as f64) * freq;
                        let cos = angle.cos() as f32;
                        let sin = angle.sin() as f32;
                        let idx0 = offset + 2 * i;
                        let idx1 = offset + 2 * i + 1;
                        let v0 = data[idx0];
                        let v1 = data[idx1];
                        data[idx0] = v0 * cos - v1 * sin;
                        data[idx1] = v0 * sin + v1 * cos;
                    }

                    // 2. Rotate spatial height partition [dim_t .. dim_t + dim_h]
                    let h_offset = offset + self.dim_t;
                    for (i, &freq) in inv_freq_h.iter().enumerate() {
                        let angle = (y as f64) * freq;
                        let cos = angle.cos() as f32;
                        let sin = angle.sin() as f32;
                        let idx0 = h_offset + 2 * i;
                        let idx1 = h_offset + 2 * i + 1;
                        let v0 = data[idx0];
                        let v1 = data[idx1];
                        data[idx0] = v0 * cos - v1 * sin;
                        data[idx1] = v0 * sin + v1 * cos;
                    }

                    // 3. Rotate spatial width partition [dim_t + dim_h .. dim]
                    let w_offset = offset + self.dim_t + self.dim_h;
                    for (i, &freq) in inv_freq_w.iter().enumerate() {
                        let angle = (x_coord as f64) * freq;
                        let cos = angle.cos() as f32;
                        let sin = angle.sin() as f32;
                        let idx0 = w_offset + 2 * i;
                        let idx1 = w_offset + 2 * i + 1;
                        let v0 = data[idx0];
                        let v1 = data[idx1];
                        data[idx0] = v0 * cos - v1 * sin;
                        data[idx1] = v0 * sin + v1 * cos;
                    }
                }
            }
        }

        let out_layout = Layout::contiguous(shape);
        let cpu_s = CpuStorage::F32(data);
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &x.device())?;

        let x_shape = x.shape().to_vec();
        let dev = x.device();
        let dt = x.dtype();
        let dt_part = self.dim_t;
        let dh_part = self.dim_h;
        let dim_total = self.dim;
        let if_t = Arc::clone(&inv_freq_t);
        let if_h = Arc::clone(&inv_freq_h);
        let if_w = Arc::clone(&inv_freq_w);

        Ok(Tensor::from_op(
            storage,
            out_layout,
            "rope3d",
            vec![x.clone()],
            move |grad_out| {
                let gv = grad_out.to_vec::<f32>()?;
                let mut dx = gv;

                // Orthogonal adjoint backward: R(theta)^T = R(-theta)
                for token_idx in 0..total_tokens {
                    let t = token_idx / (h_size * w_size);
                    let rem = token_idx % (h_size * w_size);
                    let y = rem / w_size;
                    let x_coord = rem % w_size;

                    for batch_idx in 0..b {
                        for head_idx in 0..heads {
                            let offset = ((batch_idx * total_tokens + token_idx) * heads + head_idx) * dim_total;

                            // 1. Temporal adjoint rotation
                            for (i, &freq) in if_t.iter().enumerate() {
                                let angle = (t as f64) * freq;
                                let cos = angle.cos() as f32;
                                let sin = angle.sin() as f32;
                                let idx0 = offset + 2 * i;
                                let idx1 = offset + 2 * i + 1;
                                let g0 = dx[idx0];
                                let g1 = dx[idx1];
                                dx[idx0] = g0 * cos + g1 * sin;
                                dx[idx1] = -g0 * sin + g1 * cos;
                            }

                            // 2. Spatial height adjoint rotation
                            let h_offset = offset + dt_part;
                            for (i, &freq) in if_h.iter().enumerate() {
                                let angle = (y as f64) * freq;
                                let cos = angle.cos() as f32;
                                let sin = angle.sin() as f32;
                                let idx0 = h_offset + 2 * i;
                                let idx1 = h_offset + 2 * i + 1;
                                let g0 = dx[idx0];
                                let g1 = dx[idx1];
                                dx[idx0] = g0 * cos + g1 * sin;
                                dx[idx1] = -g0 * sin + g1 * cos;
                            }

                            // 3. Spatial width adjoint rotation
                            let w_offset = offset + dt_part + dh_part;
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
    fn test_rope3d_norm_conservation_and_autograd() {
        // dim = 16: dt=4, dh=6, dw=6
        let rope = RotaryEmbedding3D::new(16, 10000.0).unwrap();
        assert_eq!(rope.dim, 16);
        assert_eq!(rope.dim_t + rope.dim_h + rope.dim_w, 16);

        // T=2, H=2, W=2 -> 8 tokens, batch=1, heads=1, dim=16
        let x_data: Vec<f32> = (0..8 * 16)
            .map(|i| (i as f32 * 0.2).cos())
            .collect();
        let x = Tensor::from_vec(x_data, &[1, 8, 1, 16]).unwrap().requires_grad_(true);

        let out = rope.forward(&x, 2, 2, 2).unwrap();
        assert_eq!(out.shape(), &[1, 8, 1, 16]);

        let in_vec = x.to_vec::<f32>().unwrap();
        let out_vec = out.to_vec::<f32>().unwrap();

        // Check L2 norm conservation per token: ||R x|| == ||x||
        for token in 0..8 {
            let offset = token * 16;
            let in_norm: f32 = (0..16).map(|i| in_vec[offset + i].powi(2)).sum::<f32>().sqrt();
            let out_norm: f32 = (0..16).map(|i| out_vec[offset + i].powi(2)).sum::<f32>().sqrt();
            assert!((in_norm - out_norm).abs() < 1e-5);
        }

        // Test backward autograd
        let loss = out.sum().unwrap();
        loss.backward().unwrap();

        let grad = x.grad().expect("rope3d grad should exist");
        assert_eq!(grad.shape(), &[1, 8, 1, 16]);
    }
}
