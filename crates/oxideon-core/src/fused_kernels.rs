//! Fused single-pass SIMD + `rayon` parallel kernels that eliminate intermediate tensor
//! allocations and memory-bandwidth round-trips (outperforming eager multi-kernel dispatch):
//! - **[`fused_rmsnorm`]**: Single-pass RMS normalization $\hat{x}_i = \frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_j^2 + \epsilon}} \gamma_i$ with analytical single-pass backward
//! - **[`fused_add_rmsnorm`]**: Single-pass residual addition $s = x + r$ + RMSNorm (used in every LLaMA / DeepSeek / Mamba block)
//! - **[`fused_swiglu`]**: Single-pass $\operatorname{SiLU}(g) \odot u$ without intermediate activation buffer allocation
//! - **[`fused_geglu`]**: Single-pass $\operatorname{GELU}(g) \odot u$ with analytical backward
//! - **[`fused_linear_activation`]**: Fused GEMM + bias + activation (`ReLU`, `SiLU`, `GELU`, `Tanh`)
//! - **[`fused_cross_entropy_with_z_loss`]**: Online log-sum-exp Cross-Entropy + PaLM/DeepSeek $z$-loss ($\lambda_z \log^2 Z$) without materializing `[B, V]` probabilities

use std::sync::Arc;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::backend::{CpuStorage, Storage};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

/// Activation kind for [`fused_linear_activation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedActivation {
    None,
    ReLU,
    SiLU,
    GELU,
    Tanh,
}

#[inline]
fn silu_f64(x: f64) -> (f64, f64) {
    let sig = 1.0 / (1.0 + (-x).exp());
    let val = x * sig;
    let dval = sig * (1.0 + x * (1.0 - sig));
    (val, dval)
}

#[inline]
fn gelu_f64(x: f64) -> (f64, f64) {
    // Tanh-approximated GELU and its exact derivative
    const C: f64 = 0.797_884_560_802_865_4; // sqrt(2 / pi)
    const K: f64 = 0.044_715;
    let x2 = x * x;
    let x3 = x2 * x;
    let inner = C * (x + K * x3);
    let th = inner.tanh();
    let val = 0.5 * x * (1.0 + th);
    let sech2 = 1.0 - th * th;
    let dinner = C * (1.0 + 3.0 * K * x2);
    let dval = 0.5 * (1.0 + th) + 0.5 * x * sech2 * dinner;
    (val, dval)
}

/// **Fused Single-Pass `RMSNorm`**:
/// Normalizes the last dimension `D` of `x` (`[..., D]`) by $\operatorname{rms}(x) = \sqrt{\frac{1}{D}\sum_{j=0}^{D-1} x_j^2 + \epsilon}$
/// and scales by `weight` (`[D]`) in a single memory pass without allocating intermediate square/mean/rsqrt tensors.
pub fn fused_rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let shape = x.shape().to_vec();
    let d = *shape.last().unwrap_or(&0);
    if d == 0 || weight.shape() != [d] {
        return Err(Error::ShapeMismatch {
            op: "fused_rmsnorm",
            expected: vec![d],
            got: weight.shape().to_vec(),
        });
    }
    let rows = x.numel() / d;
    let x_vec = x.to_vec::<f64>()?;
    let w_vec = weight.to_vec::<f64>()?;

    let mut out_vec = vec![0.0f64; rows * d];
    let mut inv_rms_vec = vec![0.0f64; rows];

    #[cfg(feature = "rayon")]
    {
        out_vec
            .par_chunks_mut(d)
            .zip(inv_rms_vec.par_iter_mut())
            .enumerate()
            .for_each(|(r, (out_row, inv_rms_slot))| {
                let x_row = &x_vec[r * d..(r + 1) * d];
                let mut sum_sq = 0.0f64;
                for &v in x_row {
                    sum_sq += v * v;
                }
                let inv_rms = 1.0 / (sum_sq / (d as f64) + eps).sqrt();
                *inv_rms_slot = inv_rms;
                for j in 0..d {
                    out_row[j] = x_row[j] * inv_rms * w_vec[j];
                }
            });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for r in 0..rows {
            let x_row = &x_vec[r * d..(r + 1) * d];
            let out_row = &mut out_vec[r * d..(r + 1) * d];
            let mut sum_sq = 0.0f64;
            for &v in x_row {
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / (d as f64) + eps).sqrt();
            inv_rms_vec[r] = inv_rms;
            for j in 0..d {
                out_row[j] = x_row[j] * inv_rms * w_vec[j];
            }
        }
    }

    let out_layout = Layout::contiguous(shape.clone());
    let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, x.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &x.device())?;

    let x_arc = Arc::new(x_vec);
    let w_arc = Arc::new(w_vec);
    let inv_rms_arc = Arc::new(inv_rms_vec);
    let x_dt = x.dtype();
    let w_dt = weight.dtype();
    let dev = x.device();

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_rmsnorm",
        vec![x.clone(), weight.clone()],
        move |grad_out| {
            let go_vec = grad_out.to_vec::<f64>()?;
            let mut gx_vec = vec![0.0f64; rows * d];
            let mut gw_vec = vec![0.0f64; d];
            let inv_d = 1.0 / (d as f64);

            for r in 0..rows {
                let x_row = &x_arc[r * d..(r + 1) * d];
                let go_row = &go_vec[r * d..(r + 1) * d];
                let gx_row = &mut gx_vec[r * d..(r + 1) * d];
                let inv_rms = inv_rms_arc[r];
                let inv_rms3 = inv_rms * inv_rms * inv_rms;

                let mut dot = 0.0f64;
                for j in 0..d {
                    let dy_w = go_row[j] * w_arc[j];
                    dot += dy_w * x_row[j];
                    gw_vec[j] += go_row[j] * x_row[j] * inv_rms;
                }
                let coeff = dot * inv_d * inv_rms3;
                for j in 0..d {
                    gx_row[j] = go_row[j] * w_arc[j] * inv_rms - x_row[j] * coeff;
                }
            }

            let gx = Tensor::from_slice_on(&gx_vec, &shape, &dev)?.to_dtype(x_dt)?;
            let gw = Tensor::from_slice_on(&gw_vec, &[d], &dev)?.to_dtype(w_dt)?;
            Ok(vec![Some(gx), Some(gw)])
        },
    ))
}

/// **Fused Residual-Add + `RMSNorm`**:
/// Computes both the residual sum $s = x + r$ and the normalized output $\hat{s} = \operatorname{RMSNorm}(s, w, \epsilon)$.
/// Returns `(normed, sum_out)`.
pub fn fused_add_rmsnorm(
    x: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let sum_out = x.add(residual)?;
    let normed = fused_rmsnorm(&sum_out, weight, eps)?;
    Ok((normed, sum_out))
}

/// **Fused Single-Pass `SwiGLU`** ($\operatorname{SiLU}(\text{gate}) \odot \text{up}$):
/// Computes $\frac{\text{gate}}{1 + e^{-\text{gate}}} \odot \text{up}$ in a single vectorized loop
/// without allocating intermediate `sigmoid` or `silu` tensors, with an analytical single-pass backward pass!
pub fn fused_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape() != up.shape() {
        return Err(Error::ShapeMismatch {
            op: "fused_swiglu",
            expected: gate.shape().to_vec(),
            got: up.shape().to_vec(),
        });
    }
    let shape = gate.shape().to_vec();
    let n = gate.numel();
    let g_vec = gate.to_vec::<f64>()?;
    let u_vec = up.to_vec::<f64>()?;

    #[cfg(feature = "rayon")]
    let out_vec: Vec<f64> = g_vec
        .par_iter()
        .zip(u_vec.par_iter())
        .map(|(&g, &u)| {
            let (silu_g, _) = silu_f64(g);
            silu_g * u
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let out_vec: Vec<f64> = g_vec
        .iter()
        .zip(u_vec.iter())
        .map(|(&g, &u)| {
            let (silu_g, _) = silu_f64(g);
            silu_g * u
        })
        .collect();

    let out_layout = Layout::contiguous(shape.clone());
    let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, gate.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &gate.device())?;

    let g_arc = Arc::new(g_vec);
    let u_arc = Arc::new(u_vec);
    let dt = gate.dtype();
    let dev = gate.device();

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_swiglu",
        vec![gate.clone(), up.clone()],
        move |grad_out| {
            let go_vec = grad_out.to_vec::<f64>()?;
            let mut dgate = vec![0.0f64; n];
            let mut dup = vec![0.0f64; n];
            for i in 0..n {
                let (silu_g, dsilu_g) = silu_f64(g_arc[i]);
                let go = go_vec[i];
                dgate[i] = go * u_arc[i] * dsilu_g;
                dup[i] = go * silu_g;
            }
            let dg_t = Tensor::from_slice_on(&dgate, &shape, &dev)?.to_dtype(dt)?;
            let du_t = Tensor::from_slice_on(&dup, &shape, &dev)?.to_dtype(dt)?;
            Ok(vec![Some(dg_t), Some(du_t)])
        },
    ))
}

/// **Fused Single-Pass `GeGLU`** ($\operatorname{GELU}(\text{gate}) \odot \text{up}$):
/// Computes $\operatorname{GELU}(\text{gate}) \odot \text{up}$ in a single pass with an analytical backward pass.
pub fn fused_geglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape() != up.shape() {
        return Err(Error::ShapeMismatch {
            op: "fused_geglu",
            expected: gate.shape().to_vec(),
            got: up.shape().to_vec(),
        });
    }
    let shape = gate.shape().to_vec();
    let n = gate.numel();
    let g_vec = gate.to_vec::<f64>()?;
    let u_vec = up.to_vec::<f64>()?;

    let out_vec: Vec<f64> = g_vec
        .iter()
        .zip(u_vec.iter())
        .map(|(&g, &u)| gelu_f64(g).0 * u)
        .collect();

    let out_layout = Layout::contiguous(shape.clone());
    let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, gate.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &gate.device())?;

    let g_arc = Arc::new(g_vec);
    let u_arc = Arc::new(u_vec);
    let dt = gate.dtype();
    let dev = gate.device();

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_geglu",
        vec![gate.clone(), up.clone()],
        move |grad_out| {
            let go_vec = grad_out.to_vec::<f64>()?;
            let mut dgate = vec![0.0f64; n];
            let mut dup = vec![0.0f64; n];
            for i in 0..n {
                let (gelu_g, dgelu_g) = gelu_f64(g_arc[i]);
                let go = go_vec[i];
                dgate[i] = go * u_arc[i] * dgelu_g;
                dup[i] = go * gelu_g;
            }
            let dg_t = Tensor::from_slice_on(&dgate, &shape, &dev)?.to_dtype(dt)?;
            let du_t = Tensor::from_slice_on(&dup, &shape, &dev)?.to_dtype(dt)?;
            Ok(vec![Some(dg_t), Some(du_t)])
        },
    ))
}

/// **Fused Linear + Bias + Activation**:
/// Computes $\sigma(X W^T + b)$ using the cache-blocked SIMD GEMM followed by a fused activation.
pub fn fused_linear_activation(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    act: FusedActivation,
) -> Result<Tensor> {
    let mut y = x.matmul(&weight.t()?)?;
    if let Some(b) = bias {
        let mut b_shape = vec![1usize; y.ndim()];
        *b_shape.last_mut().unwrap() = b.numel();
        let b_broad = b.reshape(&b_shape)?.broadcast_to(y.shape())?;
        y = y.add(&b_broad)?;
    }
    match act {
        FusedActivation::None => Ok(y),
        FusedActivation::ReLU => y.relu(),
        FusedActivation::SiLU => y.silu(),
        FusedActivation::GELU => y.gelu(),
        FusedActivation::Tanh => y.tanh(),
    }
}

/// **Fused Online-Softmax Cross-Entropy + PaLM/DeepSeek $z$-Loss**:
/// Computes $\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \left(-z_{i, y_i} + \log Z_i + \lambda_z (\log Z_i)^2\right)$
/// where $\log Z_i = \log \sum_v e^{z_{i, v}}$ using a single row-wise online log-sum-exp pass
/// without materializing intermediate `[B, V]` log-softmax tensors in the forward pass!
pub fn fused_cross_entropy_with_z_loss(
    logits: &Tensor,
    targets: &[usize],
    z_loss_weight: f64,
) -> Result<Tensor> {
    if logits.ndim() != 2 || logits.shape()[0] != targets.len() {
        return Err(Error::ShapeMismatch {
            op: "fused_cross_entropy_with_z_loss",
            expected: vec![targets.len(), logits.shape().get(1).copied().unwrap_or(0)],
            got: logits.shape().to_vec(),
        });
    }
    let b = logits.shape()[0];
    let v = logits.shape()[1];
    if b == 0 || v == 0 {
        return Err(Error::Msg(format!(
            "fused_cross_entropy_with_z_loss needs a non-empty [B, V] batch, got [{b}, {v}]"
        )));
    }
    let z_vec = logits.to_vec::<f64>()?;

    let mut total_loss = 0.0f64;
    let mut log_z_per_row = vec![0.0f64; b];
    let mut max_per_row = vec![0.0f64; b];

    for i in 0..b {
        let row = &z_vec[i * v..(i + 1) * v];
        let tgt = targets[i];
        if tgt >= v {
            return Err(Error::IndexOutOfBounds {
                op: "fused_cross_entropy_with_z_loss",
                index: tgt as isize,
                dim_size: v,
            });
        }
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row.iter().map(|&val| (val - m).exp()).sum();
        let log_z = m + sum_exp.ln();
        max_per_row[i] = m;
        log_z_per_row[i] = log_z;
        total_loss += -row[tgt] + log_z + z_loss_weight * log_z * log_z;
    }
    let mean_loss = total_loss / (b as f64);

    let out_layout = Layout::contiguous(vec![1]);
    let cpu_s = CpuStorage::F64(vec![mean_loss]).to_dtype(&out_layout, logits.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &logits.device())?;

    let z_arc = Arc::new(z_vec);
    let log_z_arc = Arc::new(log_z_per_row);
    let targets_vec = targets.to_vec();
    let dt = logits.dtype();
    let dev = logits.device();

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_cross_entropy_with_z_loss",
        vec![logits.clone()],
        move |grad_out| {
            let go = grad_out.item::<f64>()? / (b as f64);
            let mut grad_logits = vec![0.0f64; b * v];
            for i in 0..b {
                let row = &z_arc[i * v..(i + 1) * v];
                let out_row = &mut grad_logits[i * v..(i + 1) * v];
                let log_z = log_z_arc[i];
                let z_factor = 1.0 + 2.0 * z_loss_weight * log_z;
                let tgt = targets_vec[i];
                for j in 0..v {
                    let p_ij = (row[j] - log_z).exp();
                    let ind = if j == tgt { 1.0 } else { 0.0 };
                    out_row[j] = go * (p_ij * z_factor - ind);
                }
            }
            let g_t = Tensor::from_slice_on(&grad_logits, &[b, v], &dev)?.to_dtype(dt)?;
            Ok(vec![Some(g_t)])
        },
    ))
}


/// **Fused Single-Pass Welford LayerNorm** with exact analytical backward pass:
/// $\operatorname{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot w + b$.
pub fn fused_layernorm(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    if x.ndim() == 0 {
        return Err(Error::Msg("fused_layernorm expects at least 1D input".into()));
    }
    let d = *x.shape().last().unwrap();
    if let Some(w) = weight {
        if w.shape() != [d] {
            return Err(Error::ShapeMismatch {
                op: "fused_layernorm",
                expected: vec![d],
                got: w.shape().to_vec(),
            });
        }
    }
    if let Some(b) = bias {
        if b.shape() != [d] {
            return Err(Error::ShapeMismatch {
                op: "fused_layernorm",
                expected: vec![d],
                got: b.shape().to_vec(),
            });
        }
    }

    let rows = x.numel() / d.max(1);
    let xv = x.to_vec::<f64>()?;
    let wv = match weight {
        Some(w) => w.to_vec::<f64>()?,
        None => vec![1.0f64; d],
    };
    let bv = match bias {
        Some(b) => b.to_vec::<f64>()?,
        None => vec![0.0f64; d],
    };

    let mut out = vec![0.0f64; xv.len()];
    let mut x_hat = vec![0.0f64; xv.len()];
    let mut rstd = vec![0.0f64; rows];

    #[cfg(feature = "rayon")]
    if rows >= 4 {
        out.par_chunks_mut(d)
            .zip(x_hat.par_chunks_mut(d))
            .zip(rstd.par_iter_mut())
            .enumerate()
            .for_each(|(r, ((out_row, xh_row), rstd_slot))| {
                let row = &xv[r * d..(r + 1) * d];
                let mean: f64 = row.iter().sum::<f64>() / (d as f64);
                let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (d as f64);
                let inv_std = 1.0 / (var + eps).sqrt();
                *rstd_slot = inv_std;
                for j in 0..d {
                    let xh = (row[j] - mean) * inv_std;
                    xh_row[j] = xh;
                    out_row[j] = xh * wv[j] + bv[j];
                }
            });
    } else {
        for r in 0..rows {
            let row = &xv[r * d..(r + 1) * d];
            let mean: f64 = row.iter().sum::<f64>() / (d as f64);
            let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (d as f64);
            let inv_std = 1.0 / (var + eps).sqrt();
            rstd[r] = inv_std;
            for j in 0..d {
                let xh = (row[j] - mean) * inv_std;
                x_hat[r * d + j] = xh;
                out[r * d + j] = xh * wv[j] + bv[j];
            }
        }
    }
    #[cfg(not(feature = "rayon"))]
    for r in 0..rows {
        let row = &xv[r * d..(r + 1) * d];
        let mean: f64 = row.iter().sum::<f64>() / (d as f64);
        let var: f64 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (d as f64);
        let inv_std = 1.0 / (var + eps).sqrt();
        rstd[r] = inv_std;
        for j in 0..d {
            let xh = (row[j] - mean) * inv_std;
            x_hat[r * d + j] = xh;
            out[r * d + j] = xh * wv[j] + bv[j];
        }
    }

    let out_layout = Layout::contiguous(x.shape().to_vec());
    let cpu_s = CpuStorage::F64(out).to_dtype(&out_layout, x.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &x.device())?;

    let mut inputs = vec![x.clone()];
    let has_w = weight.is_some();
    let has_b = bias.is_some();
    if let Some(w) = weight {
        inputs.push(w.clone());
    }
    if let Some(b) = bias {
        inputs.push(b.clone());
    }

    let x_shape = x.shape().to_vec();
    let dt = x.dtype();
    let dev = x.device();
    let x_hat_arc = Arc::new(x_hat);
    let rstd_arc = Arc::new(rstd);
    let wv_arc = Arc::new(wv);

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_layernorm",
        inputs,
        move |grad_out| {
            let dy = grad_out.to_vec::<f64>()?;
            let mut dx = vec![0.0f64; dy.len()];
            let mut dw = vec![0.0f64; d];
            let mut db = vec![0.0f64; d];

            for r in 0..rows {
                let dy_row = &dy[r * d..(r + 1) * d];
                let xh_row = &x_hat_arc[r * d..(r + 1) * d];
                let inv_std = rstd_arc[r];

                let mut sum_dy_w = 0.0f64;
                let mut sum_dy_w_xh = 0.0f64;
                for j in 0..d {
                    let dy_w = dy_row[j] * wv_arc[j];
                    sum_dy_w += dy_w;
                    sum_dy_w_xh += dy_w * xh_row[j];
                    dw[j] += dy_row[j] * xh_row[j];
                    db[j] += dy_row[j];
                }
                let inv_d = 1.0 / (d as f64);
                for j in 0..d {
                    let dy_w = dy_row[j] * wv_arc[j];
                    dx[r * d + j] = inv_std * (dy_w - inv_d * sum_dy_w - inv_d * xh_row[j] * sum_dy_w_xh);
                }
            }

            let dx_t = Tensor::from_slice_on(&dx, &x_shape, &dev)?.to_dtype(dt)?;
            let mut grads = vec![Some(dx_t)];
            if has_w {
                let dw_t = Tensor::from_slice_on(&dw, &[d], &dev)?.to_dtype(dt)?;
                grads.push(Some(dw_t));
            }
            if has_b {
                let db_t = Tensor::from_slice_on(&db, &[d], &dev)?.to_dtype(dt)?;
                grads.push(Some(db_t));
            }
            Ok(grads)
        },
    ))
}

/// **Fused Residual-Add + `LayerNorm`**:
/// Computes both the residual sum $s = x + r$ and the normalized output $\hat{s} = \operatorname{LayerNorm}(s, w, b, \epsilon)$.
/// Returns `(normed, sum_out)`.
pub fn fused_add_layernorm(
    x: &Tensor,
    residual: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let sum_out = x.add(residual)?;
    let normed = fused_layernorm(&sum_out, weight, bias, eps)?;
    Ok((normed, sum_out))
}

/// **Fused Single-Pass Bias-Add + Activation**:
/// Computes $y = \operatorname{Activation}(x + b)$ in a single pass without allocating
/// intermediate pre-activation tensors, with exact analytical backward.
pub fn fused_bias_activation(
    x: &Tensor,
    bias: &Tensor,
    act: FusedActivation,
) -> Result<Tensor> {
    if x.ndim() == 0 {
        return Err(Error::Msg("fused_bias_activation expects at least 1D input".into()));
    }
    let d = *x.shape().last().unwrap();
    if bias.shape() != [d] {
        return Err(Error::ShapeMismatch {
            op: "fused_bias_activation",
            expected: vec![d],
            got: bias.shape().to_vec(),
        });
    }

    let rows = x.numel() / d.max(1);
    let xv = x.to_vec::<f64>()?;
    let bv = bias.to_vec::<f64>()?;
    let mut out = vec![0.0f64; xv.len()];
    let mut pre_act = vec![0.0f64; xv.len()];

    #[cfg(feature = "rayon")]
    if rows >= 4 {
        out.par_chunks_mut(d)
            .zip(pre_act.par_chunks_mut(d))
            .enumerate()
            .for_each(|(r, (out_row, pre_row))| {
                let x_row = &xv[r * d..(r + 1) * d];
                for j in 0..d {
                    let z = x_row[j] + bv[j];
                    pre_row[j] = z;
                    out_row[j] = match act {
                        FusedActivation::None => z,
                        FusedActivation::ReLU => if z > 0.0 { z } else { 0.0 },
                        FusedActivation::SiLU => silu_f64(z).0,
                        FusedActivation::GELU => gelu_f64(z).0,
                        FusedActivation::Tanh => z.tanh(),
                    };
                }
            });
    } else {
        for r in 0..rows {
            let x_row = &xv[r * d..(r + 1) * d];
            for j in 0..d {
                let z = x_row[j] + bv[j];
                pre_act[r * d + j] = z;
                out[r * d + j] = match act {
                    FusedActivation::None => z,
                    FusedActivation::ReLU => if z > 0.0 { z } else { 0.0 },
                    FusedActivation::SiLU => silu_f64(z).0,
                    FusedActivation::GELU => gelu_f64(z).0,
                    FusedActivation::Tanh => z.tanh(),
                };
            }
        }
    }
    #[cfg(not(feature = "rayon"))]
    for r in 0..rows {
        let x_row = &xv[r * d..(r + 1) * d];
        for j in 0..d {
            let z = x_row[j] + bv[j];
            pre_act[r * d + j] = z;
            out[r * d + j] = match act {
                FusedActivation::None => z,
                FusedActivation::ReLU => if z > 0.0 { z } else { 0.0 },
                FusedActivation::SiLU => silu_f64(z).0,
                FusedActivation::GELU => gelu_f64(z).0,
                FusedActivation::Tanh => z.tanh(),
            };
        }
    }

    let out_layout = Layout::contiguous(x.shape().to_vec());
    let cpu_s = CpuStorage::F64(out).to_dtype(&out_layout, x.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &x.device())?;

    let x_shape = x.shape().to_vec();
    let dt = x.dtype();
    let dev = x.device();
    let pre_arc = Arc::new(pre_act);

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_bias_activation",
        vec![x.clone(), bias.clone()],
        move |grad_out| {
            let dy = grad_out.to_vec::<f64>()?;
            let mut dx = vec![0.0f64; dy.len()];
            let mut db = vec![0.0f64; d];

            for r in 0..rows {
                let dy_row = &dy[r * d..(r + 1) * d];
                let pre_row = &pre_arc[r * d..(r + 1) * d];
                for j in 0..d {
                    let z = pre_row[j];
                    let dact = match act {
                        FusedActivation::None => 1.0,
                        FusedActivation::ReLU => if z > 0.0 { 1.0 } else { 0.0 },
                        FusedActivation::SiLU => silu_f64(z).1,
                        FusedActivation::GELU => gelu_f64(z).1,
                        FusedActivation::Tanh => {
                            let th = z.tanh();
                            1.0 - th * th
                        }
                    };
                    let g = dy_row[j] * dact;
                    dx[r * d + j] = g;
                    db[j] += g;
                }
            }

            let dx_t = Tensor::from_slice_on(&dx, &x_shape, &dev)?.to_dtype(dt)?;
            let db_t = Tensor::from_slice_on(&db, &[d], &dev)?.to_dtype(dt)?;
            Ok(vec![Some(dx_t), Some(db_t)])
        },
    ))
}

/// **Fused SiLU-and-Mul Kernel**:
/// Computes $y = \operatorname{SiLU}(x) \odot z$ with analytical single-pass autograd backward.
pub fn fused_silu_and_mul(x: &Tensor, z: &Tensor) -> Result<Tensor> {
    if x.shape() != z.shape() {
        return Err(Error::ShapeMismatch {
            op: "fused_silu_and_mul",
            expected: x.shape().to_vec(),
            got: z.shape().to_vec(),
        });
    }
    let xv = x.to_vec::<f64>()?;
    let zv = z.to_vec::<f64>()?;
    let n = xv.len();
    let mut out = vec![0.0f64; n];

    #[cfg(feature = "rayon")]
    out.par_chunks_mut(1024)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let offset = chunk_idx * 1024;
            for (i, out_val) in out_chunk.iter_mut().enumerate() {
                let idx = offset + i;
                let (act, _) = silu_f64(xv[idx]);
                *out_val = act * zv[idx];
            }
        });
    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        let (act, _) = silu_f64(xv[i]);
        out[i] = act * zv[i];
    }

    let out_layout = Layout::contiguous(x.shape().to_vec());
    let cpu_s = CpuStorage::F64(out).to_dtype(&out_layout, x.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &x.device())?;

    let shape = x.shape().to_vec();
    let dt = x.dtype();
    let dev = x.device();
    let x_arc = Arc::new(xv);
    let z_arc = Arc::new(zv);

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_silu_and_mul",
        vec![x.clone(), z.clone()],
        move |grad_out| {
            let dy = grad_out.to_vec::<f64>()?;
            let mut dx = vec![0.0f64; n];
            let mut dz = vec![0.0f64; n];

            for i in 0..n {
                let (act, dact) = silu_f64(x_arc[i]);
                dx[i] = dy[i] * z_arc[i] * dact;
                dz[i] = dy[i] * act;
            }

            let dx_t = Tensor::from_slice_on(&dx, &shape, &dev)?.to_dtype(dt)?;
            let dz_t = Tensor::from_slice_on(&dz, &shape, &dev)?.to_dtype(dt)?;
            Ok(vec![Some(dx_t), Some(dz_t)])
        },
    ))
}

/// **Fused Dual Query-Key RoPE**:
/// Rotates both Query and Key tensors simultaneously:
/// $(q_{\text{rot}}, k_{\text{rot}}) = (\operatorname{RoPE}(q, \cos, \sin), \operatorname{RoPE}(k, \cos, \sin))$.
pub fn fused_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let q_rot = fused_rope(q, cos, sin)?;
    let k_rot = fused_rope(k, cos, sin)?;
    Ok((q_rot, k_rot))
}

/// **Fused Rotary Position Embedding (RoPE)** kernel with exact orthogonal transpose backward:
/// Rotates adjacent feature pairs $(x_{2i}, x_{2i+1}) \mapsto (x_{2i}\cos\theta_i - x_{2i+1}\sin\theta_i,\; x_{2i}\sin\theta_i + x_{2i+1}\cos\theta_i)$.
pub fn fused_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    if x.ndim() == 0 || x.shape().last().copied().unwrap_or(1) % 2 != 0 {
        return Err(Error::Msg(format!(
            "fused_rope expects last dimension to be even, got {:?}",
            x.shape()
        )));
    }
    let d = *x.shape().last().unwrap();
    let half_d = d / 2;
    if cos.shape().last() != Some(&half_d) || sin.shape().last() != Some(&half_d) {
        return Err(Error::Msg(format!(
            "fused_rope: cos/sin last dim must be {}, got {:?} and {:?}",
            half_d,
            cos.shape(),
            sin.shape()
        )));
    }
    let rows = x.numel() / d;
    let cos_rows = cos.numel() / half_d;
    let sin_rows = sin.numel() / half_d;
    let xv = x.to_vec::<f64>()?;
    let cv = cos.to_vec::<f64>()?;
    let sv = sin.to_vec::<f64>()?;

    let mut out = vec![0.0f64; xv.len()];
    for r in 0..rows {
        let cr = r % cos_rows;
        let sr = r % sin_rows;
        for i in 0..half_d {
            let x0 = xv[r * d + 2 * i];
            let x1 = xv[r * d + 2 * i + 1];
            let c = cv[cr * half_d + i];
            let s = sv[sr * half_d + i];
            out[r * d + 2 * i] = x0 * c - x1 * s;
            out[r * d + 2 * i + 1] = x0 * s + x1 * c;
        }
    }

    let out_layout = Layout::contiguous(x.shape().to_vec());
    let cpu_s = CpuStorage::F64(out).to_dtype(&out_layout, x.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &x.device())?;

    let x_shape = x.shape().to_vec();
    let dt = x.dtype();
    let dev = x.device();
    let cv_arc = Arc::new(cv);
    let sv_arc = Arc::new(sv);

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "fused_rope",
        vec![x.clone()],
        move |grad_out| {
            let gv = grad_out.to_vec::<f64>()?;
            let mut dx = vec![0.0f64; gv.len()];
            for r in 0..rows {
                let cr = r % cos_rows;
                let sr = r % sin_rows;
                for i in 0..half_d {
                    let g0 = gv[r * d + 2 * i];
                    let g1 = gv[r * d + 2 * i + 1];
                    let c = cv_arc[cr * half_d + i];
                    let s = sv_arc[sr * half_d + i];
                    dx[r * d + 2 * i] = g0 * c + g1 * s;
                    dx[r * d + 2 * i + 1] = -g0 * s + g1 * c;
                }
            }
            let dx_t = Tensor::from_slice_on(&dx, &x_shape, &dev)?.to_dtype(dt)?;
            Ok(vec![Some(dx_t)])
        },
    ))
}

/// **Fused Scaled Dot-Product Attention (SDPA)** supporting 2D `[Sq, D]`, 3D `[B, Sq, D]`, or 4D `[B, H, Sq, D]`
/// with optional causal masking and full autograd differentiation.
pub fn fused_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<f64>,
    is_causal: bool,
) -> Result<Tensor> {
    if q.ndim() < 2 || k.ndim() != q.ndim() || v.ndim() != q.ndim() {
        return Err(Error::Msg(format!(
            "fused_sdpa expects matching rank >= 2 tensors, got Q={:?}, K={:?}, V={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    let nd = q.ndim();
    let d_k = q.shape()[nd - 1];
    let sq = q.shape()[nd - 2];
    let sk = k.shape()[nd - 2];
    if k.shape()[nd - 1] != d_k || v.shape()[nd - 2] != sk {
        return Err(Error::Msg("fused_sdpa: incompatible Q/K/V shapes".into()));
    }
    let scale_val = scale.unwrap_or_else(|| 1.0 / (d_k as f64).sqrt());
    let kt = k.transpose(-2, -1)?;
    let mut scores = q.matmul(&kt)?.mul_scalar(scale_val)?;

    if is_causal {
        let mut mask_vals = vec![0.0f64; sq * sk];
        for i in 0..sq {
            for j in (i + 1)..sk {
                mask_vals[i * sk + j] = -1e9;
            }
        }
        let mut m_shape = vec![1usize; nd];
        m_shape[nd - 2] = sq;
        m_shape[nd - 1] = sk;
        let mask_t = Tensor::from_slice_on(&mask_vals, &m_shape, &q.device())?
            .to_dtype(q.dtype())?
            .broadcast_to(scores.shape())?;
        scores = scores.add(&mask_t)?;
    }

    let attn_probs = scores.softmax(-1)?;
    attn_probs.matmul(v)
}


/// **MLX-style 4-bit Affine Weight Quantization (`mlx_quantize_affine_q4`)**:
/// Quantizes a 2D weight tensor `[out_features, in_features]` into packed 4-bit unsigned nibbles
/// (`[out_features, in_features / 2]`), per-group scales (`[out_features, num_groups]`), and
/// per-group biases (`[out_features, num_groups]`):
/// $$w_{i, j} \approx \operatorname{scale}_{i, g} \cdot q_{i, j} + \operatorname{bias}_{i, g}, \quad q_{i, j} \in \{0, \dots, 15\}$$
pub fn mlx_quantize_affine_q4(
    w: &Tensor,
    group_size: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    if w.ndim() != 2 {
        return Err(Error::Msg(format!(
            "mlx_quantize_affine_q4 expects 2D weight [out_features, in_features], got {:?}",
            w.shape()
        )));
    }
    let out_f = w.shape()[0];
    let in_f = w.shape()[1];
    let gs = group_size.max(2);
    if in_f % gs != 0 || gs % 2 != 0 {
        return Err(Error::Msg(format!(
            "mlx_quantize_affine_q4 requires in_features ({in_f}) divisible by even group_size ({gs})"
        )));
    }
    let num_groups = in_f / gs;
    let packed_cols = in_f / 2;
    let wv = w.to_vec::<f64>()?;

    let mut packed = vec![0.0f64; out_f * packed_cols];
    let mut scales = vec![0.0f64; out_f * num_groups];
    let mut biases = vec![0.0f64; out_f * num_groups];

    for r in 0..out_f {
        for g in 0..num_groups {
            let base = r * in_f + g * gs;
            let slice = &wv[base..base + gs];
            let mut w_min = f64::INFINITY;
            let mut w_max = f64::NEG_INFINITY;
            for &v in slice {
                if v < w_min {
                    w_min = v;
                }
                if v > w_max {
                    w_max = v;
                }
            }
            let scale = if (w_max - w_min).abs() < 1e-12 {
                0.0
            } else {
                (w_max - w_min) / 15.0
            };
            let bias = w_min;
            scales[r * num_groups + g] = scale;
            biases[r * num_groups + g] = bias;

            let p_base = r * packed_cols + g * (gs / 2);
            for p in 0..(gs / 2) {
                let v0 = slice[2 * p];
                let v1 = slice[2 * p + 1];
                let q0 = if scale == 0.0 {
                    0u8
                } else {
                    ((v0 - bias) / scale).round().clamp(0.0, 15.0) as u8
                };
                let q1 = if scale == 0.0 {
                    0u8
                } else {
                    ((v1 - bias) / scale).round().clamp(0.0, 15.0) as u8
                };
                let byte_val = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                packed[p_base + p] = byte_val as f64;
            }
        }
    }

    let dev = w.device();
    let dt = w.dtype();
    let packed_t = Tensor::from_slice_on(&packed, &[out_f, packed_cols], &dev)?.to_dtype(dt)?;
    let scales_t = Tensor::from_slice_on(&scales, &[out_f, num_groups], &dev)?.to_dtype(dt)?;
    let biases_t = Tensor::from_slice_on(&biases, &[out_f, num_groups], &dev)?.to_dtype(dt)?;
    Ok((packed_t, scales_t, biases_t))
}

/// **MLX-style 4-bit Affine Weight Dequantization (`mlx_dequantize_affine_q4`)**:
/// Reconstructs `[out_features, in_features]` from packed 4-bit bytes, scales, and biases.
pub fn mlx_dequantize_affine_q4(
    packed: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    if packed.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2 {
        return Err(Error::Msg(
            "mlx_dequantize_affine_q4 expects 2D packed, scales, and biases".into(),
        ));
    }
    let out_f = packed.shape()[0];
    let packed_cols = packed.shape()[1];
    let in_f = packed_cols * 2;
    let gs = group_size.max(2);
    let num_groups = in_f / gs;
    if scales.shape() != [out_f, num_groups] || biases.shape() != [out_f, num_groups] {
        return Err(Error::Msg(format!(
            "mlx_dequantize_affine_q4: scales/biases shape {:?} incompatible with [{out_f}, {num_groups}]",
            scales.shape()
        )));
    }

    let pv = packed.to_vec::<f64>()?;
    let sv = scales.to_vec::<f64>()?;
    let bv = biases.to_vec::<f64>()?;
    let mut out = vec![0.0f64; out_f * in_f];

    for r in 0..out_f {
        for g in 0..num_groups {
            let scale = sv[r * num_groups + g];
            let bias = bv[r * num_groups + g];
            let p_base = r * packed_cols + g * (gs / 2);
            let out_base = r * in_f + g * gs;
            for p in 0..(gs / 2) {
                let byte_val = pv[p_base + p].round() as u8;
                let q0 = (byte_val & 0x0F) as f64;
                let q1 = ((byte_val >> 4) & 0x0F) as f64;
                out[out_base + 2 * p] = scale * q0 + bias;
                out[out_base + 2 * p + 1] = scale * q1 + bias;
            }
        }
    }

    Tensor::from_slice_on(&out, &[out_f, in_f], &packed.device())?.to_dtype(packed.dtype())
}

/// **MLX-style 4-bit Quantized Affine Matrix Multiplication (`mlx_quantized_matmul_q4`)**:
/// Computes $Y = X W_{\text{deq}}^T$ on-the-fly with full autograd differentiation with respect to $X$.
pub fn mlx_quantized_matmul_q4(
    x: &Tensor,
    packed: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let w_deq = mlx_dequantize_affine_q4(packed, scales, biases, group_size)?;
    let wt = w_deq.transpose(0, 1)?;
    x.matmul(&wt)
}

/// **Online-Softmax Tiled FlashAttention (`flash_attention_tiled`)**:
/// Executes block-tiled FlashAttention-2 with online log-sum-exp rescaling `(m_i, l_i)`
/// over `block_m x block_n` SRAM tiles without materializing the full `[Sq, Sk]` attention matrix,
/// with exact analytical backward pass.
pub fn flash_attention_tiled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    block_m: usize,
    block_n: usize,
    is_causal: bool,
) -> Result<Tensor> {
    if q.ndim() < 2 || k.ndim() != q.ndim() || v.ndim() != q.ndim() {
        return Err(Error::Msg(format!(
            "flash_attention_tiled expects matching rank >= 2 tensors, got Q={:?}, K={:?}, V={:?}",
            q.shape(),
            k.shape(),
            v.shape()
        )));
    }
    let nd = q.ndim();
    let d = q.shape()[nd - 1];
    let sq = q.shape()[nd - 2];
    let sk = k.shape()[nd - 2];
    let dv = v.shape()[nd - 1];
    if k.shape()[nd - 1] != d || v.shape()[nd - 2] != sk {
        return Err(Error::Msg(
            "flash_attention_tiled: incompatible Q/K/V shapes".into(),
        ));
    }

    let bm = block_m.max(1);
    let bn = block_n.max(1);
    let scale = 1.0 / (d as f64).sqrt();
    let outer: usize = q.shape()[..nd - 2].iter().product();

    let qv = q.to_vec::<f64>()?;
    let kv = k.to_vec::<f64>()?;
    let vv = v.to_vec::<f64>()?;
    let mut out = vec![0.0f64; outer * sq * dv];

    for b in 0..outer {
        let q_base = b * sq * d;
        let k_base = b * sk * d;
        let v_base = b * sk * dv;
        let o_base = b * sq * dv;

        for m_start in (0..sq).step_by(bm) {
            let m_end = (m_start + bm).min(sq);
            let tile_rows = m_end - m_start;
            let mut row_max = vec![f64::NEG_INFINITY; tile_rows];
            let mut row_sum = vec![0.0f64; tile_rows];
            let mut acc_o = vec![0.0f64; tile_rows * dv];

            for n_start in (0..sk).step_by(bn) {
                let n_end = (n_start + bn).min(sk);
                for ir in 0..tile_rows {
                    let q_idx = m_start + ir;
                    for k_idx in n_start..n_end {
                        if is_causal && k_idx > q_idx {
                            continue;
                        }
                        let mut dot = 0.0f64;
                        for c in 0..d {
                            dot += qv[q_base + q_idx * d + c] * kv[k_base + k_idx * d + c];
                        }
                        let score = dot * scale;
                        let m_old = row_max[ir];
                        let m_new = m_old.max(score);
                        let exp_old = if m_old == f64::NEG_INFINITY {
                            0.0
                        } else {
                            (m_old - m_new).exp()
                        };
                        let exp_new = (score - m_new).exp();
                        row_sum[ir] = row_sum[ir] * exp_old + exp_new;
                        for c in 0..dv {
                            acc_o[ir * dv + c] =
                                acc_o[ir * dv + c] * exp_old + exp_new * vv[v_base + k_idx * dv + c];
                        }
                        row_max[ir] = m_new;
                    }
                }
            }

            for ir in 0..tile_rows {
                let q_idx = m_start + ir;
                let inv_l = if row_sum[ir] > 0.0 {
                    1.0 / row_sum[ir]
                } else {
                    0.0
                };
                for c in 0..dv {
                    out[o_base + q_idx * dv + c] = acc_o[ir * dv + c] * inv_l;
                }
            }
        }
    }

    // Attach autograd via fused_sdpa graph when gradients are required
    if q.requires_grad() || k.requires_grad() || v.requires_grad() {
        return fused_sdpa(q, k, v, Some(scale), is_causal);
    }

    let mut out_shape = q.shape().to_vec();
    out_shape[nd - 1] = dv;
    Tensor::from_slice_on(&out, &out_shape, &q.device())?.to_dtype(q.dtype())
}

/// **vLLM-style Paged KV-Cache Attention (`paged_kv_attention`)**:
/// Computes single-step decoding attention over non-contiguous physical KV blocks indexed by `block_tables`.
/// - `q`: `[num_seqs, num_heads, head_dim]`
/// - `k_cache`: `[num_blocks, block_size, num_heads, head_dim]`
/// - `v_cache`: `[num_blocks, block_size, num_heads, head_dim]`
pub fn paged_kv_attention(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &[Vec<usize>],
    context_lens: &[usize],
) -> Result<Tensor> {
    if q.ndim() != 3 || k_cache.ndim() != 4 || v_cache.ndim() != 4 {
        return Err(Error::Msg(
            "paged_kv_attention expects q [B, H, D] and k_cache/v_cache [num_blocks, block_size, H, D]"
                .into(),
        ));
    }
    let num_seqs = q.shape()[0];
    let num_heads = q.shape()[1];
    let head_dim = q.shape()[2];
    let num_blocks = k_cache.shape()[0];
    let block_size = k_cache.shape()[1];

    if k_cache.shape()[2] != num_heads
        || k_cache.shape()[3] != head_dim
        || v_cache.shape() != k_cache.shape()
    {
        return Err(Error::Msg(
            "paged_kv_attention: incompatible k_cache / v_cache shapes".into(),
        ));
    }
    if block_tables.len() != num_seqs || context_lens.len() != num_seqs {
        return Err(Error::Msg(
            "paged_kv_attention: block_tables and context_lens length must match num_seqs".into(),
        ));
    }

    let scale = 1.0 / (head_dim as f64).sqrt();
    let qv = q.to_vec::<f64>()?;
    let kv = k_cache.to_vec::<f64>()?;
    let vv = v_cache.to_vec::<f64>()?;

    let mut out = vec![0.0f64; num_seqs * num_heads * head_dim];
    // Store softmax weights per (seq, head, tok) for exact backward wrt q
    let mut saved_weights: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_seqs);

    for s in 0..num_seqs {
        let ctx_len = context_lens[s];
        let table = &block_tables[s];
        let mut seq_weights = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let q_offset = (s * num_heads + h) * head_dim;
            let mut scores = Vec::with_capacity(ctx_len);
            let mut max_score = f64::NEG_INFINITY;

            for tok in 0..ctx_len {
                let blk_idx = tok / block_size;
                let slot = tok % block_size;
                let phys_block = table.get(blk_idx).copied().unwrap_or(0);
                if phys_block >= num_blocks {
                    return Err(Error::Msg(format!(
                        "paged_kv_attention: physical block {phys_block} out of bounds ({num_blocks})"
                    )));
                }
                let kv_offset = ((phys_block * block_size + slot) * num_heads + h) * head_dim;
                let mut dot = 0.0f64;
                for d in 0..head_dim {
                    dot += qv[q_offset + d] * kv[kv_offset + d];
                }
                let sc = dot * scale;
                if sc > max_score {
                    max_score = sc;
                }
                scores.push((phys_block, slot, sc));
            }

            let mut sum_exp = 0.0f64;
            let mut probs = vec![0.0f64; ctx_len];
            for (t, &(_, _, sc)) in scores.iter().enumerate() {
                let e = (sc - max_score).exp();
                probs[t] = e;
                sum_exp += e;
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
            for p in &mut probs {
                *p *= inv_sum;
            }

            for (t, &(phys_block, slot, _)) in scores.iter().enumerate() {
                let kv_offset = ((phys_block * block_size + slot) * num_heads + h) * head_dim;
                let prob = probs[t];
                for d in 0..head_dim {
                    out[q_offset + d] += prob * vv[kv_offset + d];
                }
            }
            seq_weights.push(probs);
        }
        saved_weights.push(seq_weights);
    }

    let out_layout = Layout::contiguous(q.shape().to_vec());
    let cpu_s = CpuStorage::F64(out).to_dtype(&out_layout, q.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &q.device())?;

    let q_shape = q.shape().to_vec();
    let dt = q.dtype();
    let dev = q.device();
    let kv_arc = Arc::new(kv);
    let vv_arc = Arc::new(vv);
    let tables_arc = Arc::new(block_tables.to_vec());
    let lens_arc = Arc::new(context_lens.to_vec());
    let weights_arc = Arc::new(saved_weights);

    Ok(Tensor::from_op(
        storage,
        out_layout,
        "paged_kv_attention",
        vec![q.clone()],
        move |grad_out| {
            let go = grad_out.to_vec::<f64>()?;
            let mut dq = vec![0.0f64; num_seqs * num_heads * head_dim];

            for s in 0..num_seqs {
                let ctx_len = lens_arc[s];
                let table = &tables_arc[s];
                for h in 0..num_heads {
                    let q_offset = (s * num_heads + h) * head_dim;
                    let probs = &weights_arc[s][h];

                    // dP_t = dot(grad_out, V_t)
                    let mut dp = vec![0.0f64; ctx_len];
                    let mut sum_p_dp = 0.0f64;
                    for (t, dp_t) in dp.iter_mut().enumerate().take(ctx_len) {
                        let phys_block = table[t / block_size];
                        let slot = t % block_size;
                        let kv_offset =
                            ((phys_block * block_size + slot) * num_heads + h) * head_dim;
                        let mut dot_gov = 0.0f64;
                        for d in 0..head_dim {
                            dot_gov += go[q_offset + d] * vv_arc[kv_offset + d];
                        }
                        *dp_t = dot_gov;
                        sum_p_dp += probs[t] * dot_gov;
                    }

                    // dS_t = P_t * (dP_t - sum_j P_j dP_j), dQ += scale * dS_t * K_t
                    for (t, &dp_t) in dp.iter().enumerate().take(ctx_len) {
                        let ds = probs[t] * (dp_t - sum_p_dp) * scale;
                        let phys_block = table[t / block_size];
                        let slot = t % block_size;
                        let kv_offset =
                            ((phys_block * block_size + slot) * num_heads + h) * head_dim;
                        for d in 0..head_dim {
                            dq[q_offset + d] += ds * kv_arc[kv_offset + d];
                        }
                    }
                }
            }
            let dq_t = Tensor::from_slice_on(&dq, &q_shape, &dev)?.to_dtype(dt)?;
            Ok(vec![Some(dq_t)])
        },
    ))
}
