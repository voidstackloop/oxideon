//! Functional building blocks that `torch.nn.functional` offers and the layer set lacked:
//! [`scaled_dot_product_attention`], functional normalizations (`layer_norm`, `group_norm`, `instance_norm`,
//! `batch_norm`, `local_response_norm`), padding with reflect / replicate / circular modes, [`interpolate`] with
//! nearest / linear / bicubic / area resampling, `channel_shuffle` and `unflatten`, with module wrappers
//! ([`PadLayer`], [`Upsample`], [`ChannelShuffle`], [`Unflatten`], [`LocalResponseNorm`]).
//! Everything is composed from differentiable tensor ops.

use super::Module;
use crate::error::{Error, Result};
use crate::tensor::{Tensor, no_grad};

// ---------------------------------------------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------------------------------------------

/// `softmax(q k^T * scale + mask) v` over `[..., L, E]`, `[..., S, E]`, `[..., S, Ev]` tensors
/// (`torch.nn.functional.scaled_dot_product_attention` without dropout).
///
/// `attn_mask` is either a boolean tensor (`true` = may attend) or an additive float mask, broadcastable to
/// `[..., L, S]`. `is_causal` applies a lower-triangular mask aligned at the top-left. `scale` defaults to
/// `1 / sqrt(E)`.
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
    scale: Option<f64>,
) -> Result<Tensor> {
    if q.ndim() < 2 || k.ndim() < 2 || v.ndim() < 2 {
        return Err(Error::Msg("scaled_dot_product_attention: expects at least 2-D q, k and v".into()));
    }
    if is_causal && attn_mask.is_some() {
        return Err(Error::Msg("scaled_dot_product_attention: pass either attn_mask or is_causal".into()));
    }
    let (l, e) = (q.shape()[q.ndim() - 2], q.shape()[q.ndim() - 1]);
    let s = k.shape()[k.ndim() - 2];
    let scale = scale.unwrap_or_else(|| 1.0 / (e as f64).sqrt());
    if let Some(out) = attention_fused_cuda(q, k, v, attn_mask, is_causal, scale)? {
        return Ok(out);
    }
    let mut scores = q.matmul(&k.transpose(-2, -1)?)?.mul_scalar(scale)?;
    if is_causal {
        let keep: Vec<bool> = (0..l * s).map(|ix| ix % s <= ix / s).collect();
        let keep = Tensor::from_slice_on(&keep, &[l, s], &q.device())?;
        scores = Tensor::where_cond(&keep, &scores, &scores.full_like(f64::NEG_INFINITY)?)?;
    }
    if let Some(m) = attn_mask {
        scores = if m.dtype() == crate::dtype::DType::Bool {
            Tensor::where_cond(&m.broadcast_to(scores.shape())?, &scores, &scores.full_like(f64::NEG_INFINITY)?)?
        } else {
            scores.add(&m.to_dtype(scores.dtype())?)?
        };
    }
    scores.softmax(-1)?.matmul(v)
}

/// The fused CUDA attention kernel for inference (no gradient needed, 2-4 dimensional inputs with equal leading dimensions,
/// head dimension up to 128). `None` sends the call down the composed path, which also handles training.
fn attention_fused_cuda(q: &Tensor, k: &Tensor, v: &Tensor, attn_mask: Option<&Tensor>, is_causal: bool, scale: f64) -> Result<Option<Tensor>> {
    use crate::backend::cuda_ops::AttnDims;
    use crate::dtype::DType;
    if !matches!(q.device(), crate::device::Device::Cuda(_)) || k.device() != q.device() || v.device() != q.device() {
        return Ok(None);
    }
    let needs_grad = crate::tensor::is_grad_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad());
    let nd = q.ndim();
    if needs_grad || nd > 4 || k.ndim() != nd || v.ndim() != nd || k.dtype() != q.dtype() || v.dtype() != q.dtype() || !matches!(q.dtype(), DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    let lead = &q.shape()[..nd - 2];
    if k.shape()[..nd - 2] != *lead || v.shape()[..nd - 2] != *lead || k.shape()[nd - 2] != v.shape()[nd - 2] || k.shape()[nd - 1] != q.shape()[nd - 1] {
        return Ok(None);
    }
    let (l, d, s, dv) = (q.shape()[nd - 2], q.shape()[nd - 1], k.shape()[nd - 2], v.shape()[nd - 1]);
    let (bh, h) = match lead.len() {
        0 => (1, 1),
        1 => (lead[0], 1),
        _ => (lead[0] * lead[1], lead[1]),
    };
    // The kernel is a scalar one (no tensor cores): it wins when the queries are few (decoding, short prefill: one launch
    // instead of a matmul / softmax / matmul chain) or when the [L, S] scores per head would not fit in memory. A long
    // square attention is faster as the composed path, whose matrix products run on the matmul / tensor-core kernels.
    let score_bytes = bh.saturating_mul(l).saturating_mul(s).saturating_mul(q.dtype().size_in_bytes());
    if !(l <= 64 || score_bytes > (1 << 28) || crate::backend::cuda_ops::fused_attention_always()) {
        return Ok(None);
    }
    // an additive mask in the query dtype (a boolean one becomes 0 / -inf), broadcast to [lead.., L, S]
    let mut target = lead.to_vec();
    target.extend([l, s]);
    let mask = match attn_mask {
        None => None,
        Some(m) => {
            let additive = if m.dtype() == DType::Bool {
                let zeros = m.zeros_like()?.to_dtype(q.dtype())?;
                Tensor::where_cond(m, &zeros, &zeros.full_like(f64::NEG_INFINITY)?)?
            } else {
                m.to_dtype(q.dtype())?
            };
            match additive.broadcast_to(&target) {
                Ok(b) => Some(b),
                Err(_) => return Ok(None),
            }
        }
    };
    let (qc, kc, vc) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
    let dims = AttnDims { bh, h, l, s, d, dv };
    let storage = crate::backend::Storage::cuda_attention((qc.storage(), qc.layout()), (kc.storage(), kc.layout()), (vc.storage(), vc.layout()), mask.as_ref().map(|m| (m.storage(), m.layout())), is_causal, scale, dims)?;
    let mut out_shape = lead.to_vec();
    out_shape.extend([l, dv]);
    Ok(storage.map(|st| Tensor::from_storage(st, crate::layout::Layout::contiguous(out_shape))))
}

// ---------------------------------------------------------------------------------------------------------------
// Normalizations
// ---------------------------------------------------------------------------------------------------------------

fn affine(x: Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, shape: &[usize]) -> Result<Tensor> {
    let mut y = x;
    if let Some(w) = weight {
        y = y.mul(&w.reshape(shape)?)?;
    }
    if let Some(b) = bias {
        y = y.add(&b.reshape(shape)?)?;
    }
    Ok(y)
}

/// Layer normalization over the trailing `normalized_shape` dimensions (`torch.nn.functional.layer_norm`).
pub fn layer_norm(
    x: &Tensor,
    normalized_shape: &[usize],
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    let k = normalized_shape.len();
    if k == 0 || k > x.ndim() || x.shape()[x.ndim() - k..] != *normalized_shape {
        return Err(Error::Msg(format!(
            "layer_norm: normalized_shape {normalized_shape:?} does not match the trailing dims of {:?}",
            x.shape()
        )));
    }
    // autocast: normalization statistics are computed in float32
    if let Some(x32) = crate::amp::fp32_input(x)? {
        let w32 = weight.map(crate::amp::fp32_value).transpose()?;
        let b32 = bias.map(crate::amp::fp32_value).transpose()?;
        return layer_norm(&x32, normalized_shape, w32.as_ref(), b32.as_ref(), eps);
    }
    if k == 1
        && let Some(y) = layer_norm_last_f32(x, weight, bias, eps)?
    {
        return Ok(y);
    }
    layer_norm_composed(x, normalized_shape, weight, bias, eps)
}

/// Layer normalization written as a chain of differentiable tensor operations (any dtype, device and
/// `normalized_shape`); the reference the fused kernel is tested against.
fn layer_norm_composed(x: &Tensor, normalized_shape: &[usize], weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let k = normalized_shape.len();
    let axes: Vec<isize> = (x.ndim() - k..x.ndim()).map(|a| a as isize).collect();
    let (centered, _) = x.centered_by_mean(&axes)?;
    let var = centered.square()?.mean_axes(&axes, true)?;
    let y = centered.mul(&var.add_scalar(eps)?.rsqrt()?)?;
    affine(y, weight, bias, normalized_shape)
}

/// Fused layer normalization over the last dimension for contiguous `f32` host tensors: one pass per row computes the
/// statistics, the normalized values and the affine output (the composed form makes seven passes over the data and as
/// many temporaries). `None` when the input does not qualify.
///
/// The backward pass is the closed form `dx = rstd * (gy - mean(gy) - y_hat * mean(gy * y_hat))` with `gy = g * w`,
/// built from tensor operations; when the backward pass itself is being recorded (`create_graph`) the statistics are
/// recomputed differentiably so higher-order gradients stay exact.
fn layer_norm_last_f32(x: &Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f64) -> Result<Option<Tensor>> {
    use crate::backend::{CpuStorage, Storage};
    use crate::dtype::DType;
    use crate::layout::Layout;
    let nd = x.ndim();
    let d = x.shape()[nd - 1];
    let host_f32 = |t: &Tensor| t.dtype() == DType::F32 && t.device().is_cpu();
    if d == 0 || x.numel() == 0 || !host_f32(x) || weight.is_some_and(|w| !host_f32(w) || w.shape() != [d]) || bias.is_some_and(|b| !host_f32(b) || b.shape() != [d]) {
        return Ok(None);
    }
    let needs_grad = crate::tensor::is_grad_enabled() && (x.requires_grad() || weight.is_some_and(Tensor::requires_grad) || bias.is_some_and(Tensor::requires_grad));
    let w = weight.map(|w| w.to_vec::<f32>()).transpose()?;
    let b = bias.map(|b| b.to_vec::<f32>()).transpose()?;
    let (n, rows) = (x.numel(), x.numel() / d);
    let take = |len: usize| crate::pool::take_f32(len).unwrap_or_else(|| vec![0.0; len]);
    let mut out = take(n);
    let mut y_hat = if needs_grad { take(n) } else { Vec::new() };
    let mut rstd = vec![0.0f32; rows];
    {
        let guard = x.storage().cpu_ref()?;
        let CpuStorage::F32(v) = &*guard else { return Ok(None) };
        let Some(range) = x.layout().contiguous_range_in(v.len()) else { return Ok(None) };
        let src = &v[range];
        let eps = eps as f32;
        let row = |xr: &[f32], o: &mut [f32], h: &mut [f32], rs: &mut f32| {
            // the mean as hi + lo, accumulated in f64, so a large common offset in the row does not eat the signal
            let (hi, lo) = crate::backend::cpu::center_split(xr);
            let var = crate::backend::cpu::centered_sumsq(xr, hi, lo) / d as f32;
            let r = 1.0 / (var + eps).sqrt();
            *rs = r;
            match (&w, &b) {
                (Some(w), Some(b)) => o.iter_mut().zip(xr).zip(w.iter().zip(b)).for_each(|((o, &v), (&w, &b))| *o = ((v - hi) - lo) * r * w + b),
                (Some(w), None) => o.iter_mut().zip(xr).zip(w).for_each(|((o, &v), &w)| *o = ((v - hi) - lo) * r * w),
                (None, Some(b)) => o.iter_mut().zip(xr).zip(b).for_each(|((o, &v), &b)| *o = ((v - hi) - lo) * r + b),
                (None, None) => o.iter_mut().zip(xr).for_each(|(o, &v)| *o = ((v - hi) - lo) * r),
            }
            h.iter_mut().zip(xr).for_each(|(h, &v)| *h = ((v - hi) - lo) * r); // no-op when `h` is empty
        };
        // rows are grouped so each parallel task handles about 8K elements
        let group = (8192 / d).max(1);
        let task = |xs: &[f32], os: &mut [f32], rs: &mut [f32], hs: &mut [f32]| {
            let mut hrows = hs.chunks_mut(d);
            for ((xr, o), r) in xs.chunks(d).zip(os.chunks_mut(d)).zip(rs.iter_mut()) {
                row(xr, o, hrows.next().unwrap_or(&mut []), r);
            }
        };
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            let hs: Vec<&mut [f32]> = if needs_grad { y_hat.chunks_mut(group * d).collect() } else { (0..rows.div_ceil(group)).map(|_| -> &mut [f32] { &mut [] }).collect() };
            src.par_chunks(group * d).zip(out.par_chunks_mut(group * d)).zip(rstd.par_chunks_mut(group)).zip(hs.into_par_iter()).for_each(|(((xs, os), rs), h)| task(xs, os, rs, h));
        }
        #[cfg(not(feature = "rayon"))]
        {
            let mut hs = y_hat.chunks_mut(group * d);
            for ((xs, os), rs) in src.chunks(group * d).zip(out.chunks_mut(group * d)).zip(rstd.chunks_mut(group)) {
                task(xs, os, rs, hs.next().unwrap_or(&mut []));
            }
        }
    }
    let shape = x.shape().to_vec();
    let out_storage = Storage::new_cpu(CpuStorage::F32(out));
    if !needs_grad {
        return Ok(Some(Tensor::from_storage(out_storage, Layout::contiguous(shape))));
    }
    let y_hat = Tensor::from_storage(Storage::new_cpu(CpuStorage::F32(y_hat)), Layout::contiguous(shape.clone()));
    let mut stat_shape = shape.clone();
    stat_shape[nd - 1] = 1;
    let rstd = Tensor::from_storage(Storage::new_cpu(CpuStorage::F32(rstd)), Layout::contiguous(stat_shape));
    let (xi, wi, bi) = (x.clone(), weight.cloned(), bias.cloned());
    let mut inputs = vec![x.clone()];
    inputs.extend(weight.cloned());
    inputs.extend(bias.cloned());
    let lead: Vec<isize> = (0..nd as isize - 1).collect();
    Ok(Some(Tensor::from_op(out_storage, Layout::contiguous(shape), "layer_norm", inputs, move |g| {
        // create_graph (gradient recording is on during this backward pass): the statistics must be functions of x again
        let (yh, rs) = if crate::tensor::is_grad_enabled() {
            let c = xi.sub(&xi.mean_axis(-1, true)?)?;
            let rs = c.square()?.mean_axis(-1, true)?.add_scalar(eps)?.rsqrt()?;
            (c.mul(&rs)?, rs)
        } else {
            (y_hat.clone(), rstd.clone())
        };
        let gy = match &wi {
            Some(w) => g.mul(w)?,
            None => g.clone(),
        };
        let dx = gy.sub(&gy.mean_axis(-1, true)?)?.sub(&yh.mul(&gy.mul(&yh)?.mean_axis(-1, true)?)?)?.mul(&rs)?;
        let mut grads = vec![Some(dx)];
        if wi.is_some() {
            let gw = g.mul(&yh)?;
            grads.push(Some(if lead.is_empty() { gw } else { gw.sum_axes(&lead, false)? }));
        }
        if bi.is_some() {
            grads.push(Some(if lead.is_empty() { g.clone() } else { g.sum_axes(&lead, false)? }));
        }
        Ok(grads)
    })))
}

/// Group normalization of `[N, C, ...]` (`torch.nn.functional.group_norm`).
pub fn group_norm(
    x: &Tensor,
    groups: usize,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    if x.ndim() < 2 {
        return Err(Error::Msg("group_norm: expects [N, C, ...]".into()));
    }
    let (n, c) = (x.shape()[0], x.shape()[1]);
    if groups == 0 || c % groups != 0 {
        return Err(Error::Msg(format!("group_norm: {c} channels are not divisible by {groups} groups")));
    }
    let grouped = x.reshape(&[n, groups, x.numel() / (n * groups).max(1)])?;
    let (centered, _) = grouped.centered_by_mean(&[-1])?;
    let var = centered.square()?.mean_axis(-1, true)?;
    let y = centered.mul(&var.add_scalar(eps)?.rsqrt()?)?.reshape(x.shape())?;
    let mut shape = vec![1; x.ndim()];
    shape[1] = c;
    affine(y, weight, bias, &shape)
}

/// Instance normalization of `[N, C, ...]` (each channel of each sample normalized separately).
pub fn instance_norm(x: &Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    if x.ndim() < 3 {
        return Err(Error::Msg("instance_norm: expects [N, C, spatial...]".into()));
    }
    group_norm(x, x.shape()[1], weight, bias, eps)
}

/// Batch normalization of `[N, C, ...]`. In training mode the batch statistics are used and the running
/// statistics are updated in place with `momentum` (unbiased variance, like PyTorch); otherwise the running
/// statistics are used.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm(
    x: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> Result<Tensor> {
    if x.ndim() < 2 {
        return Err(Error::Msg("batch_norm: expects [N, C, ...]".into()));
    }
    let c = x.shape()[1];
    let mut shape = vec![1; x.ndim()];
    shape[1] = c;
    let axes: Vec<isize> = (0..x.ndim() as isize).filter(|&a| a != 1).collect();
    let stats = running_mean.zip(running_var);
    let mut batch_centered: Option<Tensor> = None;
    let (mean, var) = if let (false, Some((rm_, rv_))) = (training, stats) {
        (rm_.reshape(&shape)?, rv_.reshape(&shape)?)
    } else {
        let (centered, mean) = x.centered_by_mean(&axes)?;
        let var = centered.square()?.mean_axes(&axes, true)?;
        batch_centered = Some(centered);
        if training {
            let count = (x.numel() / c.max(1)) as f64;
            if count <= 1.0 {
                return Err(Error::Msg("batch_norm: expected more than one value per channel in training".into()));
            }
            no_grad(|| -> Result<()> {
                if let Some(rm) = running_mean {
                    let m = mean.detach().reshape(&[c])?;
                    rm.assign(&rm.mul_scalar(1.0 - momentum)?.add(&m.mul_scalar(momentum)?)?)?;
                }
                if let Some(rv) = running_var {
                    let unbiased = var.detach().reshape(&[c])?.mul_scalar(count / (count - 1.0))?;
                    rv.assign(&rv.mul_scalar(1.0 - momentum)?.add(&unbiased.mul_scalar(momentum)?)?)?;
                }
                Ok(())
            })?;
        }
        (mean, var)
    };
    let centered = match batch_centered {
        Some(c) => c,
        None => x.sub(&mean)?,
    };
    let y = centered.mul(&var.add_scalar(eps)?.rsqrt()?)?;
    affine(y, weight, bias, &shape)
}

/// Local response normalization across channels of `[N, C, ...]` (`torch.nn.functional.local_response_norm`).
pub fn local_response_norm(x: &Tensor, size: usize, alpha: f64, beta: f64, k: f64) -> Result<Tensor> {
    if x.ndim() < 3 || size == 0 {
        return Err(Error::Msg("local_response_norm: expects [N, C, spatial...] and size >= 1".into()));
    }
    let c = x.shape()[1];
    let sq = x.square()?;
    // zero-pad the channel axis by size/2 in front and (size - 1)/2 behind, then average windows of `size`
    let mut pads = vec![(0, 0); x.ndim()];
    pads[1] = (size / 2, (size - 1) / 2);
    let padded = sq.pad(&pads, 0.0)?;
    let mut acc = padded.narrow(1, 0, c)?;
    for i in 1..size {
        acc = acc.add(&padded.narrow(1, i, c)?)?;
    }
    let div = acc.mul_scalar(alpha / size as f64)?.add_scalar(k)?.powf(beta)?;
    x.div(&div)
}

// ---------------------------------------------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------------------------------------------

/// Padding mode of [`pad_mode`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PadMode {
    Constant(f64),
    /// Mirror without repeating the edge value.
    Reflect,
    /// Repeat the edge value.
    Replicate,
    /// Wrap around.
    Circular,
}

/// Pads `x` like `torch.nn.functional.pad`: `pads` lists `(before, after)` pairs starting from the **last**
/// dimension.
pub fn pad_mode(x: &Tensor, pads: &[(usize, usize)], mode: PadMode) -> Result<Tensor> {
    if pads.len() > x.ndim() {
        return Err(Error::Msg(format!("pad: {} pad pairs for a {}-d tensor", pads.len(), x.ndim())));
    }
    if let PadMode::Constant(v) = mode {
        let mut per_dim = vec![(0, 0); x.ndim()];
        for (i, &p) in pads.iter().enumerate() {
            per_dim[x.ndim() - 1 - i] = p;
        }
        return x.pad(&per_dim, v);
    }
    let mut out = x.clone();
    for (i, &(before, after)) in pads.iter().enumerate() {
        if before == 0 && after == 0 {
            continue;
        }
        let dim = x.ndim() - 1 - i;
        let n = x.shape()[dim] as i64;
        if n == 0 {
            return Err(Error::Msg("pad: cannot pad an empty dimension".into()));
        }
        if mode == PadMode::Reflect && (before as i64 >= n || after as i64 >= n) {
            return Err(Error::Msg(format!("pad: reflect padding ({before}, {after}) must be smaller than the size {n}")));
        }
        if mode == PadMode::Circular && (before as i64 > n || after as i64 > n) {
            return Err(Error::Msg(format!("pad: circular padding ({before}, {after}) must not exceed the size {n}")));
        }
        let idx: Vec<i64> = (-(before as i64)..n + after as i64)
            .map(|j| match mode {
                PadMode::Reflect => {
                    if j < 0 { -j } else if j >= n { 2 * (n - 1) - j } else { j }
                }
                PadMode::Replicate => j.clamp(0, n - 1),
                _ => j.rem_euclid(n),
            })
            .collect();
        let idx = Tensor::from_slice_on(&idx, &[idx.len()], &x.device())?;
        out = out.index_select(dim as isize, &idx)?;
    }
    Ok(out)
}

/// Padding layer for any rank; pairs start from the last dimension.
pub struct PadLayer {
    pub pads: Vec<(usize, usize)>,
    pub mode: PadMode,
}

impl PadLayer {
    /// Same amount on both sides of the last `dims` dimensions.
    pub fn uniform(amount: usize, dims: usize, mode: PadMode) -> Self {
        Self { pads: vec![(amount, amount); dims], mode }
    }
}

impl Module for PadLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        pad_mode(input, &self.pads, self.mode)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------------------------------------------

/// Resampling kernel of [`interpolate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpMode {
    Nearest,
    NearestExact,
    /// (Bi / tri)linear.
    Linear,
    Bicubic,
    /// Adaptive average pooling.
    Area,
}

fn cubic_weights(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let w = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
        } else {
            0.0
        }
    };
    [w(t + 1.0), w(t), w(1.0 - t), w(2.0 - t)]
}

/// Row-major `[out, inp]` resampling matrix of one dimension.
fn resample_matrix(inp: usize, out: usize, mode: InterpMode, align_corners: bool, scale_hint: Option<f64>) -> Vec<f64> {
    let mut m = vec![0.0; out * inp];
    let scale = if align_corners && out > 1 {
        (inp as f64 - 1.0) / (out as f64 - 1.0)
    } else if let Some(s) = scale_hint {
        1.0 / s
    } else {
        inp as f64 / out as f64
    };
    let last = inp as i64 - 1;
    for o in 0..out {
        match mode {
            InterpMode::Nearest => {
                let s = scale_hint.map_or(inp as f64 / out as f64, |s| 1.0 / s);
                let src = ((o as f64 * s).floor() as i64).min(last);
                m[o * inp + src.max(0) as usize] = 1.0;
            }
            InterpMode::NearestExact => {
                let s = scale_hint.map_or(inp as f64 / out as f64, |s| 1.0 / s);
                let src = (((o as f64 + 0.5) * s).floor() as i64).min(last);
                m[o * inp + src.max(0) as usize] = 1.0;
            }
            InterpMode::Linear => {
                let src = if align_corners { o as f64 * scale } else { ((o as f64 + 0.5) * scale - 0.5).max(0.0) };
                let i0 = (src.floor() as i64).min(last);
                let i1 = (i0 + 1).min(last);
                let frac = src - i0 as f64;
                m[o * inp + i0 as usize] += 1.0 - frac;
                m[o * inp + i1 as usize] += frac;
            }
            InterpMode::Bicubic => {
                let src = if align_corners { o as f64 * scale } else { (o as f64 + 0.5) * scale - 0.5 };
                let i0 = src.floor();
                let w = cubic_weights(src - i0);
                for (k, wk) in w.iter().enumerate() {
                    let idx = (i0 as i64 - 1 + k as i64).clamp(0, last) as usize;
                    m[o * inp + idx] += wk;
                }
            }
            InterpMode::Area => {
                let start = (o * inp) / out;
                let end = ((o + 1) * inp).div_ceil(out);
                let wgt = 1.0 / (end - start) as f64;
                for i in start..end {
                    m[o * inp + i] = wgt;
                }
            }
        }
    }
    m
}

/// Resizes the spatial dimensions of `[N, C, spatial...]` (`torch.nn.functional.interpolate`). Give either the
/// output `size` (one entry per spatial dim) or `scale_factor` (one entry per spatial dim). `align_corners` only
/// affects `Linear` and `Bicubic`.
pub fn interpolate(
    x: &Tensor,
    size: Option<&[usize]>,
    scale_factor: Option<&[f64]>,
    mode: InterpMode,
    align_corners: bool,
) -> Result<Tensor> {
    if x.ndim() < 3 {
        return Err(Error::Msg("interpolate: expects [N, C, spatial...]".into()));
    }
    let nsp = x.ndim() - 2;
    let in_sizes = &x.shape()[2..];
    let (out_sizes, hints): (Vec<usize>, Vec<Option<f64>>) = match (size, scale_factor) {
        (Some(s), None) if s.len() == nsp => (s.to_vec(), vec![None; nsp]),
        (None, Some(f)) if f.len() == nsp => (
            in_sizes.iter().zip(f).map(|(&i, &s)| (i as f64 * s).floor() as usize).collect(),
            f.iter().map(|&s| Some(s)).collect(),
        ),
        _ => return Err(Error::Msg(format!("interpolate: give exactly one of size / scale_factor with {nsp} entries"))),
    };
    if out_sizes.contains(&0) {
        return Err(Error::Msg("interpolate: output size must be positive".into()));
    }
    let mut y = x.clone();
    for d in 0..nsp {
        let axis = (d + 2) as isize;
        let (n_in, n_out) = (in_sizes[d], out_sizes[d]);
        if n_in == n_out && !matches!(mode, InterpMode::Bicubic | InterpMode::Linear) {
            continue;
        }
        let m = resample_matrix(n_in, n_out, mode, align_corners, hints[d]);
        // [in, out] so that (… , in) @ (in, out) resamples the last axis
        let wt = Tensor::from_slice_on(&m, &[n_out, n_in], &x.device())?.to_dtype(x.dtype())?.t()?;
        y = y.transpose(axis, -1)?.matmul(&wt)?.transpose(axis, -1)?;
    }
    Ok(y)
}

/// Resizing layer around [`interpolate`].
pub struct Upsample {
    pub size: Option<Vec<usize>>,
    pub scale_factor: Option<Vec<f64>>,
    pub mode: InterpMode,
    pub align_corners: bool,
}

impl Module for Upsample {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        interpolate(input, self.size.as_deref(), self.scale_factor.as_deref(), self.mode, self.align_corners)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Shape layers
// ---------------------------------------------------------------------------------------------------------------

/// Interleaves the channels of `[N, C, ...]` across `groups` (`torch.nn.functional.channel_shuffle`).
pub fn channel_shuffle(x: &Tensor, groups: usize) -> Result<Tensor> {
    if x.ndim() < 2 || groups == 0 || x.shape()[1] % groups != 0 {
        return Err(Error::Msg(format!("channel_shuffle: channels must be divisible by {groups}")));
    }
    let (n, c) = (x.shape()[0], x.shape()[1]);
    let mut split = vec![n, groups, c / groups];
    split.extend(&x.shape()[2..]);
    x.reshape(&split)?.transpose(1, 2)?.contiguous()?.reshape(x.shape())
}

/// Splits dimension `dim` into `sizes` (`torch.unflatten`); one entry may be `-1`.
pub fn unflatten(x: &Tensor, dim: isize, sizes: &[isize]) -> Result<Tensor> {
    let d = crate::layout::normalize_axis("unflatten", dim, x.ndim())?;
    let total = x.shape()[d];
    let known: usize = sizes.iter().filter(|&&s| s >= 0).map(|&s| s as usize).product();
    let infer = sizes.iter().filter(|&&s| s < 0).count();
    if infer > 1 || (infer == 0 && known != total) || (infer == 1 && (known == 0 || total % known != 0)) {
        return Err(Error::Msg(format!("unflatten: sizes {sizes:?} do not multiply to {total}")));
    }
    let mut shape: Vec<usize> = x.shape()[..d].to_vec();
    shape.extend(sizes.iter().map(|&s| if s < 0 { total / known } else { s as usize }));
    shape.extend(&x.shape()[d + 1..]);
    x.reshape(&shape)
}

pub struct ChannelShuffle(pub usize);

impl Module for ChannelShuffle {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        channel_shuffle(input, self.0)
    }
}

pub struct Unflatten {
    pub dim: isize,
    pub sizes: Vec<isize>,
}

impl Module for Unflatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        unflatten(input, self.dim, &self.sizes)
    }
}

pub struct LocalResponseNorm {
    pub size: usize,
    pub alpha: f64,
    pub beta: f64,
    pub k: f64,
}

impl Module for LocalResponseNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        local_response_norm(input, self.size, self.alpha, self.beta, self.k)
    }
}

/// Differentiable 2D col2im (`torch.nn.functional.fold`) over `[N, C * kH * kW, L]` -> `[N, C, H_out, W_out]`.
pub fn fold_2d(
    input: &Tensor,
    output_size: (usize, usize),
    kernel_size: (usize, usize),
    dilation: (usize, usize),
    padding: (usize, usize),
    stride: (usize, usize),
) -> Result<Tensor> {
    if input.ndim() != 3 {
        return Err(Error::Msg(format!(
            "fold: expected 3D input [N, C*kH*kW, L], got {:?}",
            input.shape()
        )));
    }
    let (n, c_block, l) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let (kh, kw) = kernel_size;
    let (dh, dw) = dilation;
    let (ph, pw) = padding;
    let (sh, sw) = stride;
    let (h_out, w_out) = output_size;
    if sh == 0 || sw == 0 || dh == 0 || dw == 0 || kh == 0 || kw == 0 || h_out == 0 || w_out == 0 {
        return Err(Error::Msg("fold: output_size, kernel_size, stride, and dilation must be positive".into()));
    }
    if c_block % (kh * kw) != 0 {
        return Err(Error::Msg(format!(
            "fold: channel dim {c_block} not divisible by kH*kW ({})",
            kh * kw
        )));
    }
    let c = c_block / (kh * kw);
    let h_blocks = (h_out + 2 * ph).saturating_sub(dh * (kh - 1) + 1) / sh + 1;
    let w_blocks = (w_out + 2 * pw).saturating_sub(dw * (kw - 1) + 1) / sw + 1;
    if h_blocks * w_blocks != l {
        return Err(Error::Msg(format!(
            "fold: expected L={} for output_size={output_size:?}, got {l}",
            h_blocks * w_blocks
        )));
    }

    let in_vec = input.to_vec::<f64>()?;
    let mut out_vec = vec![0.0f64; n * c * h_out * w_out];
    for bi in 0..n {
        for ci in 0..c {
            let out_base = (bi * c + ci) * h_out * w_out;
            for khi in 0..kh {
                for kwi in 0..kw {
                    let row = ((bi * c + ci) * kh + khi) * kw + kwi;
                    let row_base = row * l;
                    for hb in 0..h_blocks {
                        let y = hb * sh + khi * dh;
                        if y < ph || y >= h_out + ph {
                            continue;
                        }
                        let oy = y - ph;
                        for wb in 0..w_blocks {
                            let x = wb * sw + kwi * dw;
                            if x < pw || x >= w_out + pw {
                                continue;
                            }
                            let ox = x - pw;
                            out_vec[out_base + oy * w_out + ox] += in_vec[row_base + hb * w_blocks + wb];
                        }
                    }
                }
            }
        }
    }

    let out_shape = vec![n, c, h_out, w_out];
    let cpu = crate::backend::CpuStorage::F64(out_vec)
        .to_dtype(&crate::layout::Layout::contiguous(out_shape.clone()), input.dtype())?;
    let storage = crate::backend::Storage::new_cpu(cpu)
        .to_device(&crate::layout::Layout::contiguous(out_shape.clone()), &input.device())?;

    let in_dtype = input.dtype();
    let in_dev = input.device();
    let in_shape = input.shape().to_vec();
    Ok(Tensor::from_op(
        storage,
        crate::layout::Layout::contiguous(out_shape),
        "fold_2d",
        vec![input.clone()],
        move |g| {
            let g_vec = g.to_vec::<f64>()?;
            let mut grad_in = vec![0.0f64; n * c_block * l];
            for bi in 0..n {
                for ci in 0..c {
                    let g_base = (bi * c + ci) * h_out * w_out;
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let row = ((bi * c + ci) * kh + khi) * kw + kwi;
                            let row_base = row * l;
                            for hb in 0..h_blocks {
                                let y = hb * sh + khi * dh;
                                if y < ph || y >= h_out + ph {
                                    continue;
                                }
                                let oy = y - ph;
                                for wb in 0..w_blocks {
                                    let x = wb * sw + kwi * dw;
                                    if x < pw || x >= w_out + pw {
                                        continue;
                                    }
                                    let ox = x - pw;
                                    grad_in[row_base + hb * w_blocks + wb] = g_vec[g_base + oy * w_out + ox];
                                }
                            }
                        }
                    }
                }
            }
            let gi_cpu = crate::backend::CpuStorage::F64(grad_in)
                .to_dtype(&crate::layout::Layout::contiguous(in_shape.clone()), in_dtype)?;
            let gi_storage = crate::backend::Storage::new_cpu(gi_cpu)
                .to_device(&crate::layout::Layout::contiguous(in_shape.clone()), &in_dev)?;
            Ok(vec![Some(Tensor::from_storage(
                gi_storage,
                crate::layout::Layout::contiguous(in_shape.clone()),
            ))])
        },
    ))
}

#[cfg(test)]
mod fused_layer_norm_tests {
    use super::*;

    fn t(n: usize, shape: &[usize], seed: f32) -> Tensor {
        Tensor::from_slice(&(0..n).map(|i| ((i as f32 + seed) * 0.713).sin() * 2.0 + 0.3).collect::<Vec<f32>>(), shape).unwrap()
    }

    fn close(a: &Tensor, b: &Tensor, tol: f32) {
        assert_eq!(a.shape(), b.shape());
        let (a, b) = (a.to_vec::<f32>().unwrap(), b.to_vec::<f32>().unwrap());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).abs() <= tol * (1.0 + y.abs()), "element {i}: {x} vs {y}");
        }
    }

    #[test]
    fn fused_kernel_matches_the_composed_form_in_value_and_gradients() {
        // 3 x 700 x 40: large enough for several parallel tasks and a ragged last group
        for (with_w, with_b) in [(true, true), (true, false), (false, true), (false, false)] {
            let x = t(3 * 700 * 40, &[3, 700, 40], 0.0).requires_grad_(true);
            let w = t(40, &[40], 1.0).requires_grad_(true);
            let b = t(40, &[40], 2.0).requires_grad_(true);
            let (wo, bo) = (with_w.then_some(&w), with_b.then_some(&b));
            let probe = t(3 * 700 * 40, &[3, 700, 40], 5.0);
            let fused = layer_norm(&x, &[40], wo, bo, 1e-5).unwrap();
            let reference = layer_norm_composed(&x, &[40], wo, bo, 1e-5).unwrap();
            close(&fused, &reference, 2e-5);
            let grads = |y: &Tensor| {
                for p in [&x, &w, &b] {
                    p.zero_grad();
                }
                y.mul(&probe).unwrap().sum().unwrap().backward().unwrap();
                [x.grad(), w.grad(), b.grad()]
            };
            let (gf, gr) = (grads(&fused), grads(&reference));
            for (f, r) in gf.iter().zip(&gr) {
                match (f, r) {
                    (Some(f), Some(r)) => close(f, r, 2e-3),
                    (None, None) => {}
                    _ => panic!("gradient presence differs"),
                }
            }
        }
        // no gradient requested: the plain kernel, same values
        let x = t(64 * 16, &[64, 16], 3.0);
        close(&layer_norm(&x, &[16], None, None, 1e-5).unwrap(), &layer_norm_composed(&x, &[16], None, None, 1e-5).unwrap(), 2e-5);
        // a strided view and a 2-D normalized shape fall back to the composed form
        let v = x.transpose(0, 1).unwrap();
        close(&layer_norm(&v, &[64], None, None, 1e-5).unwrap(), &layer_norm_composed(&v, &[64], None, None, 1e-5).unwrap(), 1e-6);
    }

    #[test]
    fn double_backward_through_the_fused_kernel_matches_the_composed_form() {
        let second = |f: &dyn Fn(&Tensor) -> Tensor| {
            let x = t(4 * 6, &[4, 6], 0.5).requires_grad_(true);
            let y = f(&x).mul(&t(24, &[4, 6], 7.0)).unwrap().sum().unwrap();
            let g = crate::tensor::grad(&[y], &[x.clone()], None, true).unwrap()[0].clone().unwrap();
            let z = g.square().unwrap().sum().unwrap();
            crate::tensor::grad(&[z], &[x], None, false).unwrap()[0].clone().unwrap()
        };
        let fused = second(&|x| layer_norm(x, &[6], None, None, 1e-5).unwrap());
        let reference = second(&|x| layer_norm_composed(x, &[6], None, None, 1e-5).unwrap());
        close(&fused, &reference, 5e-3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn near(t: &Tensor, want: &[f64]) {
        let got = vals(t);
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9 * (1.0 + w.abs()), "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn attention() {
        let q = f(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let k = q.clone();
        let v = f(&[10.0, 0.0, 0.0, 20.0], &[1, 2, 2]);
        let out = scaled_dot_product_attention(&q, &k, &v, None, false, Some(1.0)).unwrap();
        let e = 1f64.exp();
        let p = e / (e + 1.0);
        near(&out, &[10.0 * p, 20.0 * (1.0 - p), 10.0 * (1.0 - p), 20.0 * p]);
        // causal: the first query only sees the first key
        let c = scaled_dot_product_attention(&q, &k, &v, None, true, Some(1.0)).unwrap();
        near(&c.narrow(1, 0, 1).unwrap(), &[10.0, 0.0]);
        // boolean mask equals the causal mask
        let mask = Tensor::from_slice(&[true, false, true, true], &[2, 2]).unwrap();
        let m = scaled_dot_product_attention(&q, &k, &v, Some(&mask), false, Some(1.0)).unwrap();
        near(&m, &vals(&c));
        // additive float mask
        let add = f(&[0.0, f64::NEG_INFINITY, 0.0, 0.0], &[2, 2]);
        near(&scaled_dot_product_attention(&q, &k, &v, Some(&add), false, Some(1.0)).unwrap(), &vals(&c));
    }

    #[test]
    fn normalizations() {
        let x = f(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let ln = layer_norm(&x, &[3], None, None, 0.0).unwrap();
        let s = (2.0f64 / 3.0).sqrt();
        near(&ln, &[-1.0 / s, 0.0, 1.0 / s, -1.0 / s, 0.0, 1.0 / s]);
        let w = f(&[2.0, 1.0, 1.0], &[3]);
        let b = f(&[0.0, 1.0, 0.0], &[3]);
        near(&layer_norm(&x, &[3], Some(&w), Some(&b), 0.0).unwrap().narrow(1, 0, 2).unwrap().reshape(&[4]).unwrap(), &[-2.0 / s, 1.0, -2.0 / s, 1.0]);
        // group norm with one group per channel == instance norm
        let img = f(&(0..8).map(|v| (v * v) as f64).collect::<Vec<_>>(), &[1, 2, 2, 2]);
        let gn = group_norm(&img, 2, None, None, 1e-5).unwrap();
        let ch0: Vec<f64> = vec![0.0, 1.0, 4.0, 9.0];
        let (mu, var) = (3.5, ch0.iter().map(|v| (v - 3.5f64).powi(2)).sum::<f64>() / 4.0);
        near(&gn.narrow(1, 0, 1).unwrap().reshape(&[4]).unwrap(), &ch0.iter().map(|v| (v - mu) / (var + 1e-5).sqrt()).collect::<Vec<_>>());
        near(&gn, &vals(&instance_norm(&img, None, None, 1e-5).unwrap()));
        // batch norm: training updates the running statistics
        let rm = f(&[0.0, 0.0], &[2]);
        let rv = f(&[1.0, 1.0], &[2]);
        let bx = f(&[1.0, 10.0, 3.0, 30.0], &[2, 2]);
        let y = batch_norm(&bx, Some(&rm), Some(&rv), None, None, true, 0.1, 0.0).unwrap();
        near(&y, &[-1.0, -1.0, 1.0, 1.0]);
        near(&rm, &[0.2, 2.0]);
        near(&rv, &[0.9 + 0.1 * 2.0, 0.9 + 0.1 * 200.0]);
        near(&batch_norm(&bx, Some(&rm), Some(&rv), None, None, false, 0.1, 0.0).unwrap().narrow(0, 0, 1).unwrap(), &[(1.0 - 0.2) / 1.1f64.sqrt(), (10.0 - 2.0) / 20.9f64.sqrt()]);
        // local response norm with size 3 on a single position
        let lrn = local_response_norm(&f(&[1.0, 2.0, 3.0], &[1, 3, 1]), 3, 1.0, 1.0, 1.0).unwrap();
        near(&lrn, &[1.0 / (1.0 + (1.0 + 4.0) / 3.0), 2.0 / (1.0 + (1.0 + 4.0 + 9.0) / 3.0), 3.0 / (1.0 + (4.0 + 9.0) / 3.0)]);
    }

    #[test]
    fn padding_modes() {
        let x = f(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        near(&pad_mode(&x, &[(2, 1)], PadMode::Reflect).unwrap(), &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0]);
        near(&pad_mode(&x, &[(2, 1)], PadMode::Replicate).unwrap(), &[1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0]);
        near(&pad_mode(&x, &[(2, 1)], PadMode::Circular).unwrap(), &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0]);
        near(&pad_mode(&x, &[(1, 1)], PadMode::Constant(9.0)).unwrap(), &[9.0, 1.0, 2.0, 3.0, 4.0, 9.0]);
        // two dims, pairs from the last dim
        let m = f(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        near(&pad_mode(&m, &[(1, 0), (0, 1)], PadMode::Replicate).unwrap(), &[1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 3.0, 3.0, 4.0]);
        assert!(pad_mode(&x, &[(4, 0)], PadMode::Reflect).is_err());
        // gradient of a reflect pad accumulates onto the mirrored source elements
        let p = f(&[1.0, 2.0, 3.0], &[3]).requires_grad_(true);
        pad_mode(&p, &[(2, 0)], PadMode::Reflect).unwrap().sum().unwrap().backward().unwrap();
        near(&p.grad().unwrap(), &[1.0, 2.0, 2.0]); // padded [3, 2, 1, 2, 3]
    }

    #[test]
    fn resampling() {
        let x = f(&[0.0, 1.0, 2.0, 3.0], &[1, 1, 4]);
        near(&interpolate(&x, Some(&[7]), None, InterpMode::Linear, true).unwrap(), &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
        near(&interpolate(&x, Some(&[8]), None, InterpMode::Linear, false).unwrap(), &[0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0]);
        near(&interpolate(&x, Some(&[8]), None, InterpMode::Nearest, false).unwrap(), &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        near(&interpolate(&x, None, Some(&[2.0]), InterpMode::Nearest, false).unwrap(), &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        near(&interpolate(&x, Some(&[2]), None, InterpMode::Area, false).unwrap(), &[0.5, 2.5]);
        // bicubic reproduces an affine ramp in the interior and keeps align_corners endpoints
        let bc = vals(&interpolate(&x, Some(&[7]), None, InterpMode::Bicubic, true).unwrap());
        assert!((bc[0] - 0.0).abs() < 1e-12 && (bc[6] - 3.0).abs() < 1e-12);
        assert!((bc[3] - 1.5).abs() < 1e-9, "{bc:?}");
        // 2-D bilinear 2x2 -> 3x3 (align_corners)
        let img = f(&[0.0, 1.0, 2.0, 3.0], &[1, 1, 2, 2]);
        near(&interpolate(&img, Some(&[3, 3]), None, InterpMode::Linear, true).unwrap(), &[0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0]);
        // gradients flow through the resampling matrices
        let g = f(&[0.0, 1.0], &[1, 1, 2]).requires_grad_(true);
        interpolate(&g, Some(&[4]), None, InterpMode::Linear, false).unwrap().sum().unwrap().backward().unwrap();
        near(&g.grad().unwrap(), &[2.0, 2.0]);
    }

    #[test]
    fn shape_layers() {
        let x = f(&(0..8).map(f64::from).collect::<Vec<_>>(), &[1, 4, 2]);
        near(&channel_shuffle(&x, 2).unwrap().reshape(&[8]).unwrap(), &[0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
        let u = unflatten(&f(&[0.0; 12], &[2, 6]), 1, &[3, -1]).unwrap();
        assert_eq!(u.shape(), [2, 3, 2]);
        assert!(unflatten(&f(&[0.0; 12], &[2, 6]), 1, &[4, -1]).is_err());
    }
}
