//! Loss functions with the full `torch.nn.functional` argument set: a [`Reduction`] (`none` / `mean` / `sum`),
//! per-class `weight`, `ignore_index`, probability targets, label smoothing, N-d inputs (`[N, C, d1, ...]`), plus
//! the losses the basic set lacks (`smooth_l1`, `binary_cross_entropy` on probabilities, `poisson_nll`,
//! `gaussian_nll`, `multilabel_soft_margin`, `multi_margin`). Everything is composed from differentiable ops.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// How per-element losses are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// Keep the per-element losses.
    None,
    Mean,
    Sum,
}

fn reduce(loss: Tensor, r: Reduction) -> Result<Tensor> {
    match r {
        Reduction::None => Ok(loss),
        Reduction::Mean => loss.mean(),
        Reduction::Sum => loss.sum(),
    }
}

/// `(pred - target)^2` reduced.
pub fn mse_loss_r(pred: &Tensor, target: &Tensor, r: Reduction) -> Result<Tensor> {
    reduce(pred.sub(target)?.square()?, r)
}

/// `|pred - target|` reduced.
pub fn l1_loss_r(pred: &Tensor, target: &Tensor, r: Reduction) -> Result<Tensor> {
    reduce(pred.sub(target)?.abs()?, r)
}

/// Smooth L1: `0.5 d^2 / beta` below `beta`, `|d| - 0.5 beta` above; `beta = 0` is the L1 loss.
pub fn smooth_l1_loss(pred: &Tensor, target: &Tensor, beta: f64, r: Reduction) -> Result<Tensor> {
    if beta < 0.0 || !beta.is_finite() {
        return Err(Error::Msg(format!("smooth_l1_loss: beta must be non-negative and finite, got {beta}")));
    }
    let diff = pred.sub(target)?.abs()?;
    let loss = if beta == 0.0 {
        diff
    } else {
        let quad = diff.square()?.mul_scalar(0.5 / beta)?;
        let lin = diff.sub_scalar(0.5 * beta)?;
        Tensor::where_cond(&diff.lt(&diff.full_like(beta)?)?, &quad, &lin)?
    };
    reduce(loss, r)
}

/// Huber loss with `delta` and a reduction (`delta * smooth_l1(beta = delta)`).
pub fn huber_loss_r(pred: &Tensor, target: &Tensor, delta: f64, r: Reduction) -> Result<Tensor> {
    if delta <= 0.0 || !delta.is_finite() {
        return Err(Error::Msg(format!("huber_loss_r: delta must be positive and finite, got {delta}")));
    }
    smooth_l1_loss(pred, target, delta, Reduction::None)?
        .mul_scalar(delta)
        .and_then(|l| reduce(l, r))
}

/// Binary cross-entropy on probabilities, with the log terms clamped at -100 like PyTorch.
pub fn binary_cross_entropy(
    probs: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    r: Reduction,
) -> Result<Tensor> {
    let log_p = probs.ln()?.clamp(-100.0, f64::INFINITY)?;
    let log_1mp = probs.neg()?.add_scalar(1.0)?.ln()?.clamp(-100.0, f64::INFINITY)?;
    let mut loss = target
        .mul(&log_p)?
        .add(&target.neg()?.add_scalar(1.0)?.mul(&log_1mp)?)?
        .neg()?;
    if let Some(w) = weight {
        loss = loss.mul(w)?;
    }
    reduce(loss, r)
}

/// Binary cross-entropy on logits with optional element `weight` and per-class `pos_weight`.
pub fn binary_cross_entropy_with_logits_r(
    logits: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    pos_weight: Option<&Tensor>,
    r: Reduction,
) -> Result<Tensor> {
    // loss = (1 - t) x + (1 + (pw - 1) t) * softplus(-x)
    let log_sig_neg = logits.neg()?.softplus()?; // -log(sigmoid(x))
    let loss = match pos_weight {
        Some(pw) => {
            let coef = pw.sub_scalar(1.0)?.mul(target)?.add_scalar(1.0)?;
            target.neg()?.add_scalar(1.0)?.mul(logits)?.add(&coef.mul(&log_sig_neg)?)?
        }
        None => target.neg()?.add_scalar(1.0)?.mul(logits)?.add(&log_sig_neg)?,
    };
    let loss = match weight {
        Some(w) => loss.mul(w)?,
        None => loss,
    };
    reduce(loss, r)
}

/// Class targets of a classification loss.
pub enum Target<'a> {
    /// Class indices, shape `[N, d1, ...]` (integer dtype).
    Indices(&'a Tensor),
    /// Class probabilities with the same shape as the logits.
    Probs(&'a Tensor),
}

fn class_axis_check(name: &str, input: &Tensor) -> Result<()> {
    if input.ndim() == 0 {
        return Err(Error::Msg(format!("{name}: input must have a class dimension")));
    }
    Ok(())
}

/// Negative log-likelihood of `target` classes under `log_probs` (`[C]`, `[N, C]` or `[N, C, d1, ...]`), with
/// optional class `weight` and `ignore_index`. `Mean` divides by the total weight of the counted targets.
pub fn nll_loss_nd(
    log_probs: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    ignore_index: Option<i64>,
    r: Reduction,
) -> Result<Tensor> {
    class_axis_check("nll_loss_nd", log_probs)?;
    // treat [C] as [1, C]
    let unbatched = log_probs.ndim() == 1;
    let (lp, tg) = if unbatched {
        (log_probs.unsqueeze(0)?, target.unsqueeze(0)?)
    } else {
        (log_probs.clone(), target.clone())
    };
    let c = lp.shape()[1] as i64;
    let tvals = tg.to_vec::<i64>()?;
    let mut safe = Vec::with_capacity(tvals.len());
    let mut valid = Vec::with_capacity(tvals.len());
    for &t in &tvals {
        if Some(t) == ignore_index {
            safe.push(0);
            valid.push(0.0);
        } else if t < 0 || t >= c {
            return Err(Error::Msg(format!("nll_loss_nd: target {t} is out of range for {c} classes")));
        } else {
            safe.push(t);
            valid.push(1.0);
        }
    }
    let dev = lp.device();
    let idx = Tensor::from_slice_on(&safe, tg.shape(), &dev)?;
    let picked = lp.gather(1, &idx.unsqueeze(1)?)?.squeeze_axis(1)?; // [N, d...]
    let mask = Tensor::from_slice_on(&valid, tg.shape(), &dev)?.to_dtype(lp.dtype())?;
    let w = match weight {
        Some(wt) => {
            if wt.shape() != [lp.shape()[1]] {
                return Err(Error::Msg("nll_loss_nd: weight must have one entry per class".into()));
            }
            wt.to_dtype(lp.dtype())?
                .index_select(0, &Tensor::from_slice_on(&safe, &[safe.len()], &dev)?)?
                .reshape(tg.shape())?
                .mul(&mask)?
        }
        None => mask,
    };
    let loss = picked.mul(&w)?.neg()?;
    let out = match r {
        Reduction::None => {
            if unbatched { loss.squeeze_axis(0)? } else { loss }
        }
        Reduction::Sum => loss.sum()?,
        Reduction::Mean => loss.sum()?.div(&w.sum()?)?,
    };
    Ok(out)
}

/// Cross-entropy of logits `[C]`, `[N, C]` or `[N, C, d1, ...]` against class indices or probabilities, with class
/// `weight`, `ignore_index` (indices only) and `label_smoothing`.
pub fn cross_entropy_nd(
    logits: &Tensor,
    target: Target<'_>,
    weight: Option<&Tensor>,
    ignore_index: Option<i64>,
    label_smoothing: f64,
    r: Reduction,
) -> Result<Tensor> {
    class_axis_check("cross_entropy_nd", logits)?;
    if !(0.0..=1.0).contains(&label_smoothing) {
        return Err(Error::Msg(format!("cross_entropy_nd: label_smoothing must lie in [0, 1], got {label_smoothing}")));
    }
    let cdim: isize = if logits.ndim() == 1 { 0 } else { 1 };
    let logp = logits.log_softmax(cdim)?;
    let c = logits.shape()[cdim as usize];
    let wshape = |w: &Tensor| -> Result<Tensor> {
        let mut s = vec![1; logits.ndim()];
        s[cdim as usize] = c;
        w.to_dtype(logp.dtype())?.reshape(&s)
    };
    match target {
        Target::Indices(idx) => {
            let nll = nll_loss_nd(&logp, idx, weight, ignore_index, r)?;
            if label_smoothing == 0.0 {
                return Ok(nll);
            }
            // smoothing term: -sum_c w_c logp_c / C per element, masked by ignore_index
            let weighted = match weight {
                Some(w) => logp.mul(&wshape(w)?)?,
                None => logp.clone(),
            };
            let smooth = weighted.sum_axis(cdim, false)?.neg()?.div_scalar(c as f64)?;
            let ivals = idx.to_vec::<i64>()?;
            let keep: Vec<f64> = ivals.iter().map(|&t| if Some(t) == ignore_index { 0.0 } else { 1.0 }).collect();
            let keep_t = Tensor::from_slice_on(&keep, idx.shape(), &logp.device())?.to_dtype(logp.dtype())?;
            let smooth = smooth.mul(&keep_t)?;
            let smooth = match r {
                Reduction::None => smooth,
                Reduction::Sum => smooth.sum()?,
                Reduction::Mean => {
                    let denom = match weight {
                        Some(w) => {
                            let safe: Vec<i64> = ivals.iter().map(|&t| if Some(t) == ignore_index { 0 } else { t }).collect();
                            let wi = w
                                .to_dtype(logp.dtype())?
                                .index_select(0, &Tensor::from_slice_on(&safe, &[safe.len()], &logp.device())?)?
                                .reshape(idx.shape())?
                                .mul(&keep_t)?;
                            wi.sum()?
                        }
                        None => keep_t.sum()?,
                    };
                    smooth.sum()?.div(&denom)?
                }
            };
            nll.mul_scalar(1.0 - label_smoothing)?.add(&smooth.mul_scalar(label_smoothing)?)
        }
        Target::Probs(p) => {
            if p.shape() != logits.shape() {
                return Err(Error::Msg("cross_entropy_nd: probability targets must match the logits' shape".into()));
            }
            let p = if label_smoothing > 0.0 {
                p.mul_scalar(1.0 - label_smoothing)?.add_scalar(label_smoothing / c as f64)?
            } else {
                p.clone()
            };
            let mut weighted = p.mul(&logp)?;
            if let Some(w) = weight {
                weighted = weighted.mul(&wshape(w)?)?;
            }
            let per = weighted.sum_axis(cdim, false)?.neg()?;
            match r {
                Reduction::None => Ok(per),
                Reduction::Sum => per.sum(),
                Reduction::Mean => per.sum()?.div_scalar((per.numel().max(1)) as f64),
            }
        }
    }
}

/// How [`kl_div_r`] combines the pointwise terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KlReduction {
    None,
    Sum,
    Mean,
    /// Sum divided by the batch size (the mathematically correct KL divergence).
    BatchMean,
}

/// KL divergence with `input` as log-probabilities (`torch.nn.functional.kl_div`).
pub fn kl_div_r(input: &Tensor, target: &Tensor, r: KlReduction, log_target: bool) -> Result<Tensor> {
    let pointwise = if log_target {
        target.exp()?.mul(&target.sub(input)?)?
    } else {
        target.xlogy(target)?.sub(&target.mul(input)?)?
    };
    match r {
        KlReduction::None => Ok(pointwise),
        KlReduction::Sum => pointwise.sum(),
        KlReduction::Mean => pointwise.mean(),
        KlReduction::BatchMean => {
            let n = input.shape().first().copied().unwrap_or(1).max(1);
            pointwise.sum()?.div_scalar(n as f64)
        }
    }
}

/// Poisson negative log-likelihood (`torch.nn.functional.poisson_nll_loss`).
pub fn poisson_nll_loss(
    input: &Tensor,
    target: &Tensor,
    log_input: bool,
    full: bool,
    eps: f64,
    r: Reduction,
) -> Result<Tensor> {
    let mut loss = if log_input {
        input.exp()?.sub(&target.mul(input)?)?
    } else {
        input.sub(&target.mul(&input.add_scalar(eps)?.ln()?)?)?
    };
    if full {
        // Stirling term for target > 1
        let t = target;
        let stirling = t
            .mul(&t.ln()?)?
            .sub(t)?
            .add(&t.mul_scalar(2.0 * std::f64::consts::PI)?.ln()?.mul_scalar(0.5)?)?;
        let big = t.gt(&t.ones_like()?)?;
        loss = loss.add(&Tensor::where_cond(&big, &stirling, &stirling.zeros_like()?)?)?;
    }
    reduce(loss, r)
}

/// Gaussian negative log-likelihood with predicted variance (`torch.nn.functional.gaussian_nll_loss`).
pub fn gaussian_nll_loss(
    input: &Tensor,
    target: &Tensor,
    var: &Tensor,
    full: bool,
    eps: f64,
    r: Reduction,
) -> Result<Tensor> {
    if var.numel() != 1 && var.shape() != input.shape() && var.ndim() + 1 != input.ndim() {
        return Err(Error::Msg("gaussian_nll_loss: var must match the input shape (or omit the last dim)".into()));
    }
    // clamp without letting the clamp's zero gradient through, like PyTorch
    let v = if var.ndim() + 1 == input.ndim() && var.numel() != 1 {
        var.unsqueeze(-1)?
    } else {
        var.clone()
    };
    let floor = v.full_like(eps)?;
    let below = v.lt(&floor)?;
    let v = Tensor::where_cond(&below, &v.detach().maximum(&floor)?.sub(&v.detach())?.add(&v)?, &v)?;
    let mut loss = v.ln()?.add(&input.sub(target)?.square()?.div(&v)?)?.mul_scalar(0.5)?;
    if full {
        loss = loss.add_scalar(0.5 * (2.0 * std::f64::consts::PI).ln())?;
    }
    reduce(loss, r)
}

/// Multi-label one-versus-all loss on logits `[N, C]` with 0/1 targets and optional class `weight`.
pub fn multilabel_soft_margin_loss(
    input: &Tensor,
    target: &Tensor,
    weight: Option<&Tensor>,
    r: Reduction,
) -> Result<Tensor> {
    let log_sig = input.neg()?.softplus()?.neg()?; // log(sigmoid(x))
    let log_sig_neg = input.softplus()?.neg()?; // log(sigmoid(-x))
    let mut loss = target
        .mul(&log_sig)?
        .add(&target.neg()?.add_scalar(1.0)?.mul(&log_sig_neg)?)?
        .neg()?;
    if let Some(w) = weight {
        loss = loss.mul(w)?;
    }
    let per_sample = loss.mean_axis(-1, false)?;
    reduce(per_sample, r)
}

/// Multi-class hinge loss on scores `[N, C]` (or `[C]`) with class indices (`torch.nn.functional.multi_margin_loss`).
pub fn multi_margin_loss(
    input: &Tensor,
    target: &Tensor,
    p: u32,
    margin: f64,
    weight: Option<&Tensor>,
    r: Reduction,
) -> Result<Tensor> {
    if p != 1 && p != 2 {
        return Err(Error::Msg("multi_margin_loss: p must be 1 or 2".into()));
    }
    let unbatched = input.ndim() == 1;
    let (x, t) = if unbatched { (input.unsqueeze(0)?, target.reshape(&[1])?) } else { (input.clone(), target.clone()) };
    let (n, c) = (x.shape()[0], x.shape()[1]);
    let tv = t.to_vec::<i64>()?;
    if tv.iter().any(|&v| v < 0 || v >= c as i64) {
        return Err(Error::Msg("multi_margin_loss: target out of range".into()));
    }
    let dev = x.device();
    let idx = Tensor::from_slice_on(&tv, &[n, 1], &dev)?;
    let correct = x.gather(1, &idx)?; // [N, 1]
    let mut hinge = x.sub(&correct)?.add_scalar(margin)?.relu()?; // zero gradient at an exactly-zero hinge, like torch
    if p == 2 {
        hinge = hinge.square()?;
    }
    // exclude the target class itself
    let mut mask = vec![1.0f64; n * c];
    for (i, &cls) in tv.iter().enumerate() {
        mask[i * c + cls as usize] = 0.0;
    }
    hinge = hinge.mul(&Tensor::from_slice_on(&mask, &[n, c], &dev)?.to_dtype(x.dtype())?)?;
    if let Some(w) = weight {
        let wt = w.to_dtype(x.dtype())?.index_select(0, &Tensor::from_slice_on(&tv, &[n], &dev)?)?.unsqueeze(1)?;
        hinge = hinge.mul(&wt)?;
    }
    let per = hinge.sum_axis(1, false)?.div_scalar(c as f64)?;
    let per = if unbatched && r == Reduction::None { per.squeeze_axis(0)? } else { per };
    reduce(per, r)
}

/// Label-smoothing helper for dtype-agnostic callers: one-hot encodes `labels` as probabilities `[.., C]`.
pub fn one_hot_probs(labels: &Tensor, classes: usize, dtype: DType) -> Result<Tensor> {
    let v = labels.to_vec::<i64>()?;
    let mut out = vec![0.0f64; v.len() * classes];
    for (i, &l) in v.iter().enumerate() {
        if l < 0 || l as usize >= classes {
            return Err(Error::Msg(format!("one_hot_probs: label {l} out of range for {classes} classes")));
        }
        out[i * classes + l as usize] = 1.0;
    }
    let mut shape = labels.shape().to_vec();
    shape.push(classes);
    Tensor::from_slice_on(&out, &shape, &labels.device())?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn i(v: &[i64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn near(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9 * (1.0 + w.abs()), "{got:?} vs {want:?}");
        }
    }
    fn lse(x: &[f64]) -> f64 {
        x.iter().map(|v| v.exp()).sum::<f64>().ln()
    }

    #[test]
    fn regression_losses() {
        let (p, t) = (f(&[0.0, 2.0, 5.0], &[3]), f(&[0.5, 2.0, 1.0], &[3]));
        near(&vals(&mse_loss_r(&p, &t, Reduction::None).unwrap()), &[0.25, 0.0, 16.0]);
        near(&vals(&l1_loss_r(&p, &t, Reduction::Sum).unwrap()), &[4.5]);
        // beta = 1: 0.5 * 0.25 = 0.125, 0, |4| - 0.5 = 3.5
        near(&vals(&smooth_l1_loss(&p, &t, 1.0, Reduction::None).unwrap()), &[0.125, 0.0, 3.5]);
        near(&vals(&smooth_l1_loss(&p, &t, 0.0, Reduction::Sum).unwrap()), &[4.5]);
        near(&vals(&huber_loss_r(&p, &t, 2.0, Reduction::None).unwrap()), &[0.125, 0.0, 6.0]);
    }

    #[test]
    fn bce_family() {
        let (p, t) = (f(&[0.8, 0.1], &[2]), f(&[1.0, 0.0], &[2]));
        near(&vals(&binary_cross_entropy(&p, &t, None, Reduction::None).unwrap()), &[-(0.8f64.ln()), -(0.9f64.ln())]);
        let x = f(&[2.0, -1.0], &[2]);
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        near(
            &vals(&binary_cross_entropy_with_logits_r(&x, &t, None, None, Reduction::None).unwrap()),
            &[-sig(2.0).ln(), -(1.0 - sig(-1.0)).ln()],
        );
        // pos_weight scales the positive term only
        near(
            &vals(&binary_cross_entropy_with_logits_r(&x, &t, None, Some(&f(&[3.0, 3.0], &[2])), Reduction::None).unwrap()),
            &[-3.0 * sig(2.0).ln(), -(1.0 - sig(-1.0)).ln()],
        );
        near(
            &vals(&binary_cross_entropy_with_logits_r(&x, &t, Some(&f(&[2.0, 0.5], &[2])), None, Reduction::Sum).unwrap()),
            &[-2.0 * sig(2.0).ln() - 0.5 * (1.0 - sig(-1.0)).ln()],
        );
    }

    #[test]
    fn cross_entropy_variants() {
        let x = f(&[1.0, 2.0, 3.0, 3.0, 1.0, 0.0], &[2, 3]);
        let t = i(&[2, 0], &[2]);
        let l0 = lse(&[1.0, 2.0, 3.0]) - 3.0;
        let l1 = lse(&[3.0, 1.0, 0.0]) - 3.0;
        near(&vals(&cross_entropy_nd(&x, Target::Indices(&t), None, None, 0.0, Reduction::None).unwrap()), &[l0, l1]);
        near(&vals(&cross_entropy_nd(&x, Target::Indices(&t), None, None, 0.0, Reduction::Mean).unwrap()), &[(l0 + l1) / 2.0]);
        // class weights: mean divides by the summed weights of the targets
        let w = f(&[2.0, 1.0, 0.5], &[3]);
        near(
            &vals(&cross_entropy_nd(&x, Target::Indices(&t), Some(&w), None, 0.0, Reduction::Mean).unwrap()),
            &[(0.5 * l0 + 2.0 * l1) / 2.5],
        );
        // ignore_index drops a sample from the numerator and the denominator
        near(&vals(&cross_entropy_nd(&x, Target::Indices(&t), None, Some(0), 0.0, Reduction::Mean).unwrap()), &[l0]);
        // label smoothing against the explicit formula
        let eps = 0.1;
        let smooth = |row: &[f64], y: usize| {
            let z = lse(row);
            let nll = z - row[y];
            let mean_nll: f64 = row.iter().map(|v| z - v).sum::<f64>() / 3.0;
            (1.0 - eps) * nll + eps * mean_nll
        };
        near(
            &vals(&cross_entropy_nd(&x, Target::Indices(&t), None, None, eps, Reduction::Mean).unwrap()),
            &[(smooth(&[1.0, 2.0, 3.0], 2) + smooth(&[3.0, 1.0, 0.0], 0)) / 2.0],
        );
        // probability targets
        let probs = f(&[0.0, 0.0, 1.0, 0.5, 0.5, 0.0], &[2, 3]);
        near(
            &vals(&cross_entropy_nd(&x, Target::Probs(&probs), None, None, 0.0, Reduction::Mean).unwrap()),
            &[(l0 + 0.5 * (lse(&[3.0, 1.0, 0.0]) - 3.0) + 0.5 * (lse(&[3.0, 1.0, 0.0]) - 1.0)) / 2.0],
        );
        // image-shaped [N, C, H]
        let img = f(&[0.0, 1.0, 1.0, 0.0], &[1, 2, 2]);
        let it = i(&[1, 0], &[1, 2]);
        let want = (1.0f64 + (-1.0f64).exp()).ln();
        near(&vals(&cross_entropy_nd(&img, Target::Indices(&it), None, None, 0.0, Reduction::Mean).unwrap()), &[want]);
        // unbatched [C]
        near(&vals(&cross_entropy_nd(&f(&[1.0, 2.0, 3.0], &[3]), Target::Indices(&i(&[2], &[])), None, None, 0.0, Reduction::Mean).unwrap()), &[l0]);
    }

    #[test]
    fn cross_entropy_gradient_is_softmax_minus_onehot() {
        let x = f(&[1.0, 2.0, 3.0], &[1, 3]).requires_grad_(true);
        cross_entropy_nd(&x, Target::Indices(&i(&[2], &[1])), None, None, 0.0, Reduction::Mean).unwrap().backward().unwrap();
        let z = lse(&[1.0, 2.0, 3.0]);
        let want: Vec<f64> = [1.0f64, 2.0, 3.0].iter().enumerate().map(|(k, v)| (v - z).exp() - if k == 2 { 1.0 } else { 0.0 }).collect();
        near(&vals(&x.grad().unwrap()), &want);
    }

    #[test]
    fn distribution_losses() {
        let logp = f(&[0.5f64.ln(), 0.5f64.ln()], &[1, 2]);
        let q = f(&[0.25, 0.75], &[1, 2]);
        let want = 0.25 * (0.25f64 / 0.5).ln() + 0.75 * (0.75f64 / 0.5).ln();
        near(&vals(&kl_div_r(&logp, &q, KlReduction::BatchMean, false).unwrap()), &[want]);
        near(&vals(&kl_div_r(&logp, &q.ln().unwrap(), KlReduction::Sum, true).unwrap()), &[want]);
        // a zero target contributes nothing
        near(&vals(&kl_div_r(&logp, &f(&[0.0, 1.0], &[1, 2]), KlReduction::Sum, false).unwrap()), &[-(0.5f64.ln())]);
        // poisson: exp(x) - t x
        near(&vals(&poisson_nll_loss(&f(&[1.0], &[1]), &f(&[2.0], &[1]), true, false, 1e-8, Reduction::Sum).unwrap()), &[1f64.exp() - 2.0]);
        near(&vals(&poisson_nll_loss(&f(&[3.0], &[1]), &f(&[2.0], &[1]), false, false, 1e-8, Reduction::Sum).unwrap()), &[3.0 - 2.0 * (3.0f64 + 1e-8).ln()]);
        // gaussian: 0.5 (ln var + (x - t)^2 / var)
        near(&vals(&gaussian_nll_loss(&f(&[1.0], &[1]), &f(&[3.0], &[1]), &f(&[2.0], &[1]), false, 1e-6, Reduction::Sum).unwrap()), &[0.5 * (2.0f64.ln() + 4.0 / 2.0)]);
    }

    #[test]
    fn margin_losses() {
        let x = f(&[0.1, 0.2, 0.4, 0.8], &[1, 4]);
        // multi_margin p=1, margin=1, target 3: sum_{i != 3} max(0, 1 - 0.8 + x_i) / 4
        near(&vals(&multi_margin_loss(&x, &i(&[3], &[1]), 1, 1.0, None, Reduction::Mean).unwrap()), &[(0.3 + 0.4 + 0.6) / 4.0]);
        near(&vals(&multi_margin_loss(&x, &i(&[3], &[1]), 2, 1.0, None, Reduction::Mean).unwrap()), &[(0.09 + 0.16 + 0.36) / 4.0]);
        let s = f(&[0.0, 2.0], &[1, 2]);
        let y = f(&[1.0, 0.0], &[1, 2]);
        let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
        near(
            &vals(&multilabel_soft_margin_loss(&s, &y, None, Reduction::Mean).unwrap()),
            &[(-(sig(0.0).ln()) - (1.0 - sig(2.0)).ln()) / 2.0],
        );
        let _ = one_hot_probs(&i(&[1], &[1]), 3, DType::F64).unwrap();
    }
}
