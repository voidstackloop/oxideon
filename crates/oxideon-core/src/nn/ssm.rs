//! Selective State Space Models (`SelectiveSSM` and `MambaBlock`, Gu & Dao 2023).
//!
//! Implements input-dependent continuous-to-discrete state space recurrence with linear-time
//! $O(L)$ sequence scaling:
//! $$\Delta_t = \operatorname{softplus}(W_\Delta u_t + b_\Delta), \quad \bar{A}_t = \exp(\Delta_t A), \quad h_t = \bar{A}_t \odot h_{t-1} + (\Delta_t \odot u_t) \otimes B_t, \quad y_t = (h_t \cdot C_t) + D \odot u_t$$

use crate::error::{Error, Result};
use crate::nn::{Linear, Module, RMSNorm};
use crate::tensor::Tensor;

/// Selective State Space layer operating on `[batch, seq_len, d_inner]`.
pub struct SelectiveSSM {
    pub x_proj: Linear,
    pub dt_proj: Linear,
    pub a_log: Tensor,
    pub d_skip: Tensor,
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
}

impl SelectiveSSM {
    pub fn new(d_inner: usize, d_state: usize, dt_rank: usize) -> Result<Self> {
        crate::limits::check_elements("SelectiveSSM A_log", &[d_inner, d_state])?;
        let x_width = d_state
            .checked_mul(2)
            .and_then(|w| w.checked_add(dt_rank))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "SelectiveSSM sizes overflow: d_state={d_state}, dt_rank={dt_rank}"
                ))
            })?;
        let x_proj = Linear::with_bias(d_inner, x_width, false)?;
        let dt_proj = Linear::new(dt_rank, d_inner)?;

        // Initialize A_log with log(1..=d_state) repeated across d_inner (S4D-Real initialization)
        let mut a_log_vec = crate::layout::try_with_capacity(d_inner * d_state)?;
        for _ in 0..d_inner {
            for s in 1..=d_state {
                a_log_vec.push((s as f32).ln());
            }
        }
        let a_log = Tensor::from_vec(a_log_vec, &[d_inner, d_state])?.requires_grad_(true);
        let d_skip = Tensor::ones_f32(&[d_inner])?.requires_grad_(true);

        Ok(Self {
            x_proj,
            dt_proj,
            a_log,
            d_skip,
            d_inner,
            d_state,
            dt_rank,
        })
    }
}

impl Module for SelectiveSSM {
    fn forward(&self, u: &Tensor) -> Result<Tensor> {
        if u.ndim() != 3 || u.shape()[2] != self.d_inner {
            return Err(Error::Msg(format!(
                "SelectiveSSM expects [batch, seq_len, {}], got {:?}",
                self.d_inner,
                u.shape()
            )));
        }
        let (b, l, d_in) = (u.shape()[0], u.shape()[1], self.d_inner);
        let n = self.d_state;

        // A = -exp(A_log) of shape [1, d_inner, d_state]
        let a = self.a_log.exp()?.neg()?.unsqueeze(0)?;
        let d_skip = self.d_skip.unsqueeze(0)?;

        // Project all timesteps at once: [B, L, dt_rank + 2 * d_state]
        let x_dbl = self.x_proj.forward(u)?;
        let dt_raw = x_dbl.narrow(2, 0, self.dt_rank)?;
        let b_all = x_dbl.narrow(2, self.dt_rank, n)?;
        let c_all = x_dbl.narrow(2, self.dt_rank + n, n)?;

        // dt = softplus(dt_proj(dt_raw)): [B, L, d_inner]
        let dt_all = self.dt_proj.forward(&dt_raw)?.softplus()?;

        let mut h = Tensor::zeros(&[b, d_in, n], u.dtype(), &u.device())?;
        let mut outputs = Vec::with_capacity(l);

        for t in 0..l {
            let u_t = u.select(1, t as isize)?; // [B, D]
            let dt_t = dt_all.select(1, t as isize)?; // [B, D]
            let b_t = b_all.select(1, t as isize)?; // [B, N]
            let c_t = c_all.select(1, t as isize)?; // [B, N]

            // dA = exp(dt_t.unsqueeze(-1) * A): [B, D, N]
            let dt_exp = dt_t.unsqueeze(2)?;
            let da = dt_exp.mul(&a)?.exp()?;

            // dB_u = (dt_t * u_t).unsqueeze(-1) * b_t.unsqueeze(1): [B, D, N]
            let du = dt_t.mul(&u_t)?.unsqueeze(2)?;
            let dbu = du.mul(&b_t.unsqueeze(1)?)?;

            h = da.mul(&h)?.add(&dbu)?;

            // y_t = sum_n(h * c_t.unsqueeze(1)) + d_skip * u_t: [B, D]
            let y_ssm = h.mul(&c_t.unsqueeze(1)?)?.sum_axis(2, false)?;
            let y_t = y_ssm.add(&u_t.mul(&d_skip)?)?;
            outputs.push(y_t);
        }

        Tensor::stack(&outputs, 1)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.x_proj.named_parameters() {
            out.push((format!("x_proj.{k}"), v));
        }
        for (k, v) in self.dt_proj.named_parameters() {
            out.push((format!("dt_proj.{k}"), v));
        }
        out.push(("a_log".to_string(), self.a_log.clone()));
        out.push(("d_skip".to_string(), self.d_skip.clone()));
        out
    }
}

/// Complete Mamba residual block (`RMSNorm` -> gated `in_proj` -> `SiLU` + `SelectiveSSM` -> `out_proj`).
pub struct MambaBlock {
    pub norm: RMSNorm,
    pub in_proj: Linear,
    pub ssm: SelectiveSSM,
    pub out_proj: Linear,
    pub d_model: usize,
    pub d_inner: usize,
}

impl MambaBlock {
    pub fn new(d_model: usize, expand: usize, d_state: usize) -> Result<Self> {
        let too_big = || {
            Error::Msg(format!(
                "MambaBlock sizes overflow: d_model={d_model}, expand={expand}"
            ))
        };
        let d_inner = d_model.checked_mul(expand.max(1)).ok_or_else(too_big)?;
        let dt_rank = d_model.div_ceil(16).max(4);
        Ok(Self {
            norm: RMSNorm::new(d_model, 1e-5)?,
            in_proj: Linear::with_bias(
                d_model,
                d_inner.checked_mul(2).ok_or_else(too_big)?,
                false,
            )?,
            ssm: SelectiveSSM::new(d_inner, d_state, dt_rank)?,
            out_proj: Linear::with_bias(d_inner, d_model, false)?,
            d_model,
            d_inner,
        })
    }
}

impl Module for MambaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(x)?;
        let xz = self.in_proj.forward(&normed)?;
        let x_branch = xz.narrow(2, 0, self.d_inner)?.silu()?;
        let z_branch = xz.narrow(2, self.d_inner, self.d_inner)?.silu()?;
        let y = self.ssm.forward(&x_branch)?.mul(&z_branch)?;
        let out = self.out_proj.forward(&y)?;
        x.add(&out)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.norm.named_parameters() {
            out.push((format!("norm.{k}"), v));
        }
        for (k, v) in self.in_proj.named_parameters() {
            out.push((format!("in_proj.{k}"), v));
        }
        for (k, v) in self.ssm.named_parameters() {
            out.push((format!("ssm.{k}"), v));
        }
        for (k, v) in self.out_proj.named_parameters() {
            out.push((format!("out_proj.{k}"), v));
        }
        out
    }
}

/// **Hyena Hierarchy Operator** (`HyenaOperator`, Poli et al. 2023):
/// Sub-quadratic sequence mixer combining `order` stages of data-controlled elementwise
/// multiplicative gating and causal depthwise 1D convolutions:
/// $$v, x_1, \dots, x_N = \operatorname{Proj}(u), \quad y^{(0)} = v, \quad y^{(m)} = x_m \odot (\mathcal{K}_m * y^{(m-1)})$$
pub struct HyenaOperator {
    pub norm: RMSNorm,
    pub in_proj: Linear,
    pub filter_weights: Vec<Tensor>,
    pub out_proj: Linear,
    pub d_model: usize,
    pub order: usize,
    pub kernel_size: usize,
}

impl HyenaOperator {
    pub fn new(d_model: usize, order: usize, kernel_size: usize) -> Result<Self> {
        let ord = order.max(1);
        let ks = kernel_size.max(1);
        let proj_width = ord
            .checked_add(1)
            .and_then(|o| o.checked_mul(d_model))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "HyenaOperator sizes overflow: order={order}, d_model={d_model}"
                ))
            })?;
        crate::limits::check_repeats("HyenaOperator order", ord)?;
        crate::limits::check_elements("HyenaOperator in_proj", &[d_model, proj_width])?;
        crate::limits::check_elements("HyenaOperator filter", &[d_model, ks])?;
        let in_proj = Linear::new(d_model, proj_width)?;
        let scale = (1.0 / (ks as f64)).sqrt();
        let mut filter_weights = Vec::with_capacity(ord);
        for _ in 0..ord {
            // Causal depthwise filter per channel: [d_model, 1, ks]
            let w = Tensor::randn(&[d_model, 1, ks])?
                .mul_scalar(scale)?
                .detach()
                .requires_grad_(true);
            filter_weights.push(w);
        }
        let out_proj = Linear::new(d_model, d_model)?;
        Ok(Self {
            norm: RMSNorm::new(d_model, 1e-5)?,
            in_proj,
            filter_weights,
            out_proj,
            d_model,
            order: ord,
            kernel_size: ks,
        })
    }
}

impl Module for HyenaOperator {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.ndim() != 3 {
            return Err(Error::Msg(format!(
                "HyenaOperator expects [B, L, D] input, got {:?}",
                x.shape()
            )));
        }
        let (_b, l, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let normed = self.norm.forward(x)?;
        let proj = self.in_proj.forward(&normed)?;

        // Transpose [B, L, (order+1)*D] -> [B, (order+1)*D, L] for causal 1D conv
        let proj_t = proj.transpose(1, 2)?;
        let mut y = proj_t.narrow(1, 0, d)?;

        for (stage, filt) in self.filter_weights.iter().enumerate() {
            let x_gate = proj_t.narrow(1, (stage + 1) * d, d)?.silu()?;
            // Left-pad by (kernel_size - 1) for strict causal convolution along sequence axis
            let y_padded = if self.kernel_size > 1 {
                y.pad(&[(0, 0), (0, 0), (self.kernel_size - 1, 0)], 0.0)?
            } else {
                y
            };
            // Depthwise 1D convolution via reshaped grouped 2D convolution `[B, D, 1, L_pad]`
            let b_sz = y_padded.shape()[0];
            let l_pad = y_padded.shape()[2];
            let y_4d = y_padded.reshape(&[b_sz, d, 1, l_pad])?;
            let w_4d = filt.reshape(&[d, 1, 1, self.kernel_size])?;
            let conv_4d = y_4d.conv2d_grouped(&w_4d, None, (1, 1), (0, 0), (1, 1), d)?;
            let conv_out = conv_4d.reshape(&[b_sz, d, l])?;
            y = conv_out.mul(&x_gate)?;
        }

        let y_seq = y.transpose(1, 2)?;
        let out = self.out_proj.forward(&y_seq)?;
        x.add(&out)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out = Vec::new();
        for (k, v) in self.norm.named_parameters() {
            out.push((format!("norm.{k}"), v));
        }
        for (k, v) in self.in_proj.named_parameters() {
            out.push((format!("in_proj.{k}"), v));
        }
        for (i, w) in self.filter_weights.iter().enumerate() {
            out.push((format!("filter_weights.{i}"), w.clone()));
        }
        for (k, v) in self.out_proj.named_parameters() {
            out.push((format!("out_proj.{k}"), v));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyena_rejects_non_3d_input() -> Result<()> {
        let op = HyenaOperator::new(8, 2, 3)?;
        let x = Tensor::zeros_f32(&[2, 8])?;
        let err = op
            .forward(&x)
            .expect_err("rank-2 input must be an error, not a panic");
        assert!(err.to_string().contains("[B, L, D]"), "{err}");
        Ok(())
    }

    #[test]
    fn hyena_rejects_overflowing_sizes() {
        let err = match HyenaOperator::new(usize::MAX / 2, usize::MAX / 2, 3) {
            Ok(_) => panic!("overflowing sizes must be an error, not a panic"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("overflow"), "{err}");
    }
}
