//! Differentiable loss functions and similarity metrics:
//! - Regression: [`mse_loss`], [`l1_loss`], [`huber_loss`]
//! - Classification: [`nll_loss`], [`cross_entropy`], [`cross_entropy_with_smoothing`],
//!   [`binary_cross_entropy_with_logits`], [`focal_loss`]
//! - Divergence & Metric Learning: [`kl_div_loss`], [`cosine_similarity`],
//!   [`cosine_embedding_loss`], [`triplet_margin_loss`], [`info_nce_loss`]

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Mean squared error: `mean((pred - target)^2)`.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    pred.sub(target)?.square()?.mean()
}

/// Mean absolute error (L1 loss): `mean(|pred - target|)`.
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    pred.sub(target)?.abs()?.mean()
}

/// Huber / Smooth L1 loss with transition threshold `delta`.
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f64) -> Result<Tensor> {
    if delta <= 0.0 || !delta.is_finite() {
        return Err(Error::Msg(format!(
            "huber_loss requires a positive finite delta, got {delta}"
        )));
    }
    let diff = pred.sub(target)?.abs()?;
    let thresh = diff.full_like(delta)?;
    let cond = diff.le(&thresh)?;
    let quad = diff.square()?.mul_scalar(0.5)?;
    let lin = diff.sub_scalar(0.5 * delta)?.mul_scalar(delta)?;
    Tensor::where_cond(&cond, &quad, &lin)?.mean()
}

/// Negative log-likelihood loss on `log_probs: [batch, classes]` with class targets.
pub fn nll_loss(log_probs: &Tensor, targets: &[usize]) -> Result<Tensor> {
    if log_probs.ndim() != 2 {
        return Err(Error::Msg(format!(
            "nll_loss expects 2D [batch, classes], got {:?}",
            log_probs.shape()
        )));
    }
    let n = log_probs.shape()[0];
    if n == 0 {
        return Err(Error::Msg(
            "nll_loss: empty batch (mean over zero rows would be NaN)".into(),
        ));
    }
    if targets.len() != n {
        return Err(Error::ShapeMismatch {
            op: "nll_loss",
            expected: vec![n],
            got: vec![targets.len()],
        });
    }
    // Targets past i64::MAX would wrap negative under `as i64` and gather
    // would then silently score them as from-the-end classes.
    if let Some(&bad) = targets.iter().find(|&&t| t > i64::MAX as usize) {
        return Err(Error::Msg(format!(
            "nll_loss: target class {bad} exceeds i64::MAX"
        )));
    }
    let idx_i64: Vec<i64> = targets.iter().map(|&t| t as i64).collect();
    let idx = Tensor::from_slice_on(&idx_i64, &[n, 1], &log_probs.device())?;
    let picked = log_probs.gather(1, &idx)?;
    picked.neg()?.mean()
}

/// Cross-entropy loss between raw `logits: [batch, classes]` and integer class `targets`.
pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> Result<Tensor> {
    nll_loss(&logits.log_softmax(1)?, targets)
}

/// Cross-entropy loss with label smoothing $\varepsilon \in [0, 1)$.
pub fn cross_entropy_with_smoothing(
    logits: &Tensor,
    targets: &[usize],
    label_smoothing: f64,
) -> Result<Tensor> {
    if !(0.0..1.0).contains(&label_smoothing) {
        return Err(Error::Msg(format!(
            "cross_entropy_with_smoothing requires label_smoothing in [0, 1), got {label_smoothing}"
        )));
    }
    let log_probs = logits.log_softmax(1)?;
    let hard_loss = nll_loss(&log_probs, targets)?;
    if label_smoothing <= 0.0 {
        return Ok(hard_loss);
    }
    let smooth_loss = log_probs.neg()?.mean()?;
    hard_loss
        .mul_scalar(1.0 - label_smoothing)?
        .add(&smooth_loss.mul_scalar(label_smoothing)?)
}

/// Numerically stable binary cross-entropy with logits:
/// `max(x, 0) - x * t + log(1 + exp(-|x|))`.
pub fn binary_cross_entropy_with_logits(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let pos = logits.relu()?;
    let xt = logits.mul(targets)?;
    let neg_abs = logits.abs()?.neg()?;
    let soft = neg_abs.softplus()?;
    pos.sub(&xt)?.add(&soft)?.mean()
}

/// Multi-class Focal Loss (Lin et al., 2017): $-\alpha (1 - p_t)^\gamma \log(p_t)$.
pub fn focal_loss(logits: &Tensor, targets: &[usize], alpha: f64, gamma: f64) -> Result<Tensor> {
    if logits.ndim() != 2 {
        return Err(Error::Msg(format!(
            "focal_loss expects 2D [batch, classes], got {:?}",
            logits.shape()
        )));
    }
    let n = logits.shape()[0];
    if n == 0 {
        return Err(Error::Msg(
            "focal_loss: empty batch (mean over zero rows would be NaN)".into(),
        ));
    }
    if targets.len() != n {
        return Err(Error::ShapeMismatch {
            op: "focal_loss",
            expected: vec![n],
            got: vec![targets.len()],
        });
    }
    let log_probs = logits.log_softmax(1)?;
    // Same wrap hazard as nll_loss: reject unrepresentable targets loudly.
    if let Some(&bad) = targets.iter().find(|&&t| t > i64::MAX as usize) {
        return Err(Error::Msg(format!(
            "focal_loss: target class {bad} exceeds i64::MAX"
        )));
    }
    let idx_i64: Vec<i64> = targets.iter().map(|&t| t as i64).collect();
    let idx = Tensor::from_slice_on(&idx_i64, &[n, 1], &logits.device())?;
    let log_pt = log_probs.gather(1, &idx)?;
    let pt = log_pt.exp()?;
    let one_minus_pt = pt.neg()?.add_scalar(1.0)?.clamp(1e-8, 1.0)?;
    let weight = one_minus_pt.powf(gamma)?.mul_scalar(-alpha)?;
    weight.mul(&log_pt)?.mean()
}

/// Kullback-Leibler divergence $D_{\text{KL}}(P \parallel Q) = \sum P (\ln P - \ln Q)$
/// where `log_probs` is $\ln Q$ and `target_probs` is $P$.
pub fn kl_div_loss(log_probs: &Tensor, target_probs: &Tensor) -> Result<Tensor> {
    let safe_t = target_probs.clamp(1e-12, f64::INFINITY)?;
    let log_t = safe_t.ln()?;
    target_probs
        .mul(&log_t.sub(log_probs)?)?
        .sum_axis(-1, false)?
        .mean()
}

/// [`cosine_similarity`] along the last axis of two contiguous host `f32` tensors of the same shape when no gradient is recorded:
/// the dot product and the norms are accumulated in `f64` and divided once, so the many cancelling terms of a nearly orthogonal pair
/// are not rounded to `f32` one by one (normalizing first and multiplying afterwards, as the differentiable path does, loses
/// about a thousand times more on such pairs). `None` when the tensors do not qualify.
fn cosine_similarity_host(a: &Tensor, b: &Tensor, axis: isize, eps: f64) -> Result<Option<Tensor>> {
    use crate::backend::CpuStorage;
    use crate::dtype::DType;
    let host_f32 = |t: &Tensor| t.dtype() == DType::F32 && t.device().is_cpu();
    let needs_grad = crate::tensor::is_grad_enabled() && (a.requires_grad() || b.requires_grad());
    let nd = a.ndim() as isize;
    if needs_grad || a.shape() != b.shape() || nd == 0 || !host_f32(a) || !host_f32(b) || (axis != -1 && axis != nd - 1) {
        return Ok(None);
    }
    let d = *a.shape().last().unwrap();
    if d == 0 {
        return Ok(None);
    }
    let (ga, gb) = (a.storage().cpu_ref()?, b.storage().cpu_ref()?);
    let (CpuStorage::F32(va), CpuStorage::F32(vb)) = (&*ga, &*gb) else { return Ok(None) };
    let (Some(ra), Some(rb)) = (a.layout().contiguous_range_in(va.len()), b.layout().contiguous_range_in(vb.len())) else { return Ok(None) };
    let (xa, xb) = (&va[ra], &vb[rb]);
    let dot = crate::backend::cpu::dot_f64_of_f32;
    let row = |(x, y): (&[f32], &[f32])| -> f32 {
        let (na, nb) = (dot(x, x).sqrt().max(eps), dot(y, y).sqrt().max(eps));
        (dot(x, y) / (na * nb)) as f32
    };
    let rows = xa.len() / d;
    let mut out = vec![0.0f32; rows];
    #[cfg(feature = "rayon")]
    if xa.len() >= 1 << 16 {
        use rayon::prelude::*;
        out.par_iter_mut().zip(xa.par_chunks(d).zip(xb.par_chunks(d))).for_each(|(o, p)| *o = row(p));
        return Ok(Some(Tensor::from_slice(&out, &a.shape()[..a.ndim() - 1])?));
    }
    for (o, p) in out.iter_mut().zip(xa.chunks(d).zip(xb.chunks(d))) {
        *o = row(p);
    }
    Ok(Some(Tensor::from_slice(&out, &a.shape()[..a.ndim() - 1])?))
}

/// Cosine similarity between `a` and `b` along `axis`: $\frac{a \cdot b}{\max(\|a\|_2 \|b\|_2, \varepsilon)}$.
pub fn cosine_similarity(a: &Tensor, b: &Tensor, axis: isize, eps: f64) -> Result<Tensor> {
    if let Some(r) = cosine_similarity_host(a, b, axis, eps)? {
        return Ok(r);
    }
    let na = a.normalize(2.0, axis, eps)?;
    let nb = b.normalize(2.0, axis, eps)?;
    na.mul(&nb)?.sum_axis(axis, false)
}

/// Cosine embedding loss for pairs `(a, b)` with `labels` in `{1.0, -1.0}` and `margin`.
pub fn cosine_embedding_loss(
    a: &Tensor,
    b: &Tensor,
    labels: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    let cos = cosine_similarity(a, b, -1, 1e-8)?;
    let pos_loss = cos.neg()?.add_scalar(1.0)?;
    let neg_loss = cos.sub_scalar(margin)?.relu()?;
    let is_pos = labels.gt(&labels.zeros_like()?)?;
    Tensor::where_cond(&is_pos, &pos_loss, &neg_loss)?.mean()
}

/// Triplet margin loss: $\max(0, \|a - p\|_2 - \|a - n\|_2 + \text{margin})$.
pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    // torch.nn.functional.triplet_margin_loss measures ||a - b + eps||_2 with eps = 1e-6
    let d_pos = anchor.pairwise_distance(positive, 2.0, 1e-6, false)?;
    let d_neg = anchor.pairwise_distance(negative, 2.0, 1e-6, false)?;
    d_pos.sub(&d_neg)?.add_scalar(margin)?.relu()?.mean()
}

/// InfoNCE / SimCLR contrastive loss between `[batch, dim]` query embeddings and
/// `[batch, dim]` positive key embeddings with temperature $\tau$.
pub fn info_nce_loss(queries: &Tensor, keys: &Tensor, temperature: f64) -> Result<Tensor> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(Error::Msg(format!(
            "info_nce_loss requires a positive finite temperature, got {temperature}"
        )));
    }
    if queries.ndim() != 2 || keys.ndim() != 2 {
        return Err(Error::Msg(format!(
            "info_nce_loss expects 2D [batch, dim] inputs, got {:?} and {:?}",
            queries.shape(),
            keys.shape()
        )));
    }
    if queries.shape()[0] != keys.shape()[0] {
        return Err(Error::ShapeMismatch {
            op: "info_nce_loss",
            expected: queries.shape().to_vec(),
            got: keys.shape().to_vec(),
        });
    }
    let q = queries.normalize(2.0, -1, 1e-8)?;
    let k = keys.normalize(2.0, -1, 1e-8)?;
    let logits = q.matmul(&k.t()?)?.div_scalar(temperature)?;
    let batch = queries.shape()[0];
    let targets: Vec<usize> = (0..batch).collect();
    cross_entropy(&logits, &targets)
}

/// Direct Preference Optimization (`DPO`, Rafailov et al. 2023) loss:
/// $$\mathcal{L}_{\text{DPO}} = \mathbb{E}\!\left[\operatorname{softplus}\!\left(-\beta \left((\log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w)) - (\log \pi_\theta(y_l) - \log \pi_{\text{ref}}(y_l))\right)\right)\right]$$
pub fn dpo_loss(
    policy_chosen_logps: &Tensor,
    policy_rejected_logps: &Tensor,
    ref_chosen_logps: &Tensor,
    ref_rejected_logps: &Tensor,
    beta: f64,
) -> Result<Tensor> {
    let chosen_logratios = policy_chosen_logps.sub(ref_chosen_logps)?;
    let rejected_logratios = policy_rejected_logps.sub(ref_rejected_logps)?;
    let margins = chosen_logratios
        .sub(&rejected_logratios)?
        .mul_scalar(-beta)?;
    margins.softplus()?.mean()
}

/// Proximal Policy Optimization (`PPO`, Schulman et al. 2017) clipped surrogate loss:
/// $$r(\theta) = \exp(\log \pi_\theta - \log \pi_{\text{old}}), \quad \mathcal{L}_{\text{PPO}} = -\mathbb{E}\!\left[\min\!\left(r(\theta) A, \operatorname{clamp}(r(\theta), 1 - \epsilon, 1 + \epsilon) A\right)\right]$$
pub fn ppo_clipped_loss(
    new_logps: &Tensor,
    old_logps: &Tensor,
    advantages: &Tensor,
    clip_eps: f64,
) -> Result<Tensor> {
    let ratio = new_logps.sub(&old_logps.detach())?.exp()?;
    let adv = advantages.detach();
    let surr1 = ratio.mul(&adv)?;
    let surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)?.mul(&adv)?;
    let use_surr1 = surr1.lt(&surr2)?;
    let min_surr = Tensor::where_cond(&use_surr1, &surr1, &surr2)?;
    min_surr.neg()?.mean()
}

/// Generalized Advantage Estimation (`GAE`, Schulman et al. 2015) over a trajectory of length $T$,
/// returning `(advantages, returns)`.
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    dones: &[bool],
    next_value: f64,
    gamma: f64,
    gae_lambda: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let t_len = rewards.len();
    if values.len() != t_len || dones.len() != t_len {
        return Err(Error::Msg(format!(
            "compute_gae: rewards/values/dones length mismatch ({} vs {} vs {})",
            t_len,
            values.len(),
            dones.len()
        )));
    }
    let mut advantages = vec![0.0f64; t_len];
    let mut returns = vec![0.0f64; t_len];
    let mut gae = 0.0f64;

    for t in (0..t_len).rev() {
        let non_terminal = if dones[t] { 0.0 } else { 1.0 };
        let v_next = if t + 1 == t_len {
            next_value
        } else {
            values[t + 1]
        };
        let delta = rewards[t] + gamma * v_next * non_terminal - values[t];
        gae = delta + gamma * gae_lambda * non_terminal * gae;
        advantages[t] = gae;
        returns[t] = gae + values[t];
    }
    Ok((advantages, returns))
}

/// VICReg self-supervised representation learning loss (Bardes et al. 2022) combining
/// invariance (MSE), variance hinge regularization, and off-diagonal covariance regularization
/// on `[batch, dim]` representations `z_a` and `z_b`.
pub fn vicreg_loss(
    z_a: &Tensor,
    z_b: &Tensor,
    sim_coeff: f64,
    std_coeff: f64,
    cov_coeff: f64,
) -> Result<Tensor> {
    if z_a.ndim() != 2 || z_b.ndim() != 2 {
        return Err(Error::Msg(format!(
            "vicreg_loss expects 2D [batch, dim] inputs, got {:?} and {:?}",
            z_a.shape(),
            z_b.shape()
        )));
    }
    let n = z_a.shape()[0] as f64;
    let d = z_a.shape()[1] as f64;

    // 1. Invariance term: MSE(z_a, z_b)
    let sim_loss = mse_loss(z_a, z_b)?;

    // 2. Variance hinge term: mean(relu(1 - std(z)))
    let std_a = z_a.var_axis(0, false, false)?.add_scalar(1e-4)?.sqrt()?;
    let std_b = z_b.var_axis(0, false, false)?.add_scalar(1e-4)?.sqrt()?;
    let var_loss = std_a
        .neg()?
        .add_scalar(1.0)?
        .relu()?
        .mean()?
        .add(&std_b.neg()?.add_scalar(1.0)?.relu()?.mean()?)?
        .mul_scalar(0.5)?;

    // 3. Covariance term: off-diagonal squared elements of centered covariance matrix / D
    let za_centered = z_a.sub(&z_a.mean_axis(0, true)?)?;
    let zb_centered = z_b.sub(&z_b.mean_axis(0, true)?)?;
    let cov_a = za_centered
        .t()?
        .matmul(&za_centered)?
        .div_scalar((n - 1.0).max(1.0))?;
    let cov_b = zb_centered
        .t()?
        .matmul(&zb_centered)?
        .div_scalar((n - 1.0).max(1.0))?;

    let off_diag_a = cov_a
        .square()?
        .sum()?
        .sub(&cov_a.diagonal(0, 0, 1)?.square()?.sum()?)?
        .div_scalar(d)?;
    let off_diag_b = cov_b
        .square()?
        .sum()?
        .sub(&cov_b.diagonal(0, 0, 1)?.square()?.sum()?)?
        .div_scalar(d)?;
    let cov_loss = off_diag_a.add(&off_diag_b)?;

    sim_loss
        .mul_scalar(sim_coeff)?
        .add(&var_loss.mul_scalar(std_coeff)?)?
        .add(&cov_loss.mul_scalar(cov_coeff)?)
}

/// **Group Relative Policy Optimization (`GRPO`)** loss (DeepSeekMath / DeepSeek-R1, Shao et al. 2024):
/// Computes group-normalized relative advantages $\hat{A}_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r) + 10^{-8}}$
/// across a group of sampled completions for a prompt, then minimizes the negative clipped
/// surrogate objective plus the unbiased Schulman KL divergence penalty against `ref_log_probs`:
/// $$\mathcal{L}_{\text{GRPO}} = -\operatorname{mean}\!\left(\min(\rho_i \hat{A}_i, \operatorname{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)$$
pub fn grpo_loss(
    new_log_probs: &Tensor,
    old_log_probs: &Tensor,
    ref_log_probs: &Tensor,
    rewards: &Tensor,
    clip_eps: f64,
    kl_beta: f64,
) -> Result<Tensor> {
    let r_det = rewards.detach();
    let r_flat = r_det.reshape(&[r_det.numel()])?;
    let r_mean = r_flat.mean()?;
    // Global (not axis-0) std so N-D reward tensors normalize correctly.
    let r_std = r_flat.var_axis(0, false, false)?.add_scalar(1e-8)?.sqrt()?;
    let advantages = r_det
        .sub(&r_mean.broadcast_to(r_det.shape())?)?
        .div(&r_std.broadcast_to(r_det.shape())?)?;

    let ratio = new_log_probs.sub(&old_log_probs.detach())?.exp()?;
    let surr1 = ratio.mul(&advantages)?;
    let surr2 = ratio
        .clamp(1.0 - clip_eps, 1.0 + clip_eps)?
        .mul(&advantages)?;
    let min_surr = Tensor::where_cond(&surr1.lt(&surr2)?, &surr1, &surr2)?;

    // Unbiased non-negative KL estimator: exp(log_ref - log_new) - (log_ref - log_new) - 1
    let log_diff = ref_log_probs.detach().sub(new_log_probs)?;
    let kl_pen = log_diff.exp()?.sub(&log_diff)?.sub_scalar(1.0)?;

    min_surr.sub(&kl_pen.mul_scalar(kl_beta)?)?.mean()?.neg()
}

/// **Simple Preference Optimization (`SimPO`)** loss (Meng et al. 2024):
/// Reference-free preference alignment using length-normalized log-probabilities
/// (`chosen_avg_logps`, `rejected_avg_logps`) and target reward margin `gamma`:
/// $$\mathcal{L}_{\text{SimPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta (\bar{\ell}_w - \bar{\ell}_l) - \gamma\right)\right]$$
pub fn simpo_loss(
    chosen_avg_logps: &Tensor,
    rejected_avg_logps: &Tensor,
    beta: f64,
    gamma: f64,
) -> Result<Tensor> {
    let logits = chosen_avg_logps
        .sub(rejected_avg_logps)?
        .mul_scalar(beta)?
        .sub_scalar(gamma)?;
    logits.neg()?.softplus()?.mean()
}

/// **Sigmoid Loss for Language-Image Pre-Training (`SigLIP`)** (Zhai et al. 2023):
/// Pairwise sigmoid binary classification over the `[N, N]` normalized cosine similarity matrix
/// between `z_img` (`[N, D]`) and `z_txt` (`[N, D]`):
/// $$\mathcal{L}_{\text{SigLIP}} = \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N \log\!\left(1 + \exp\!\left(-y_{ij} (t \cdot \hat{z}_i^{\text{img}} \cdot \hat{z}_j^{\text{txt}} + b)\right)\right)$$
/// where $y_{ii} = +1$ for matched pairs and $y_{ij} = -1$ for $i \ne j$.
pub fn siglip_loss(z_img: &Tensor, z_txt: &Tensor, temperature: f64, bias: f64) -> Result<Tensor> {
    if z_img.ndim() != 2 || z_txt.ndim() != 2 || z_img.shape() != z_txt.shape() {
        return Err(Error::ShapeMismatch {
            op: "siglip_loss",
            expected: z_img.shape().to_vec(),
            got: z_txt.shape().to_vec(),
        });
    }
    let n = z_img.shape()[0];
    let img_norm = z_img.normalize(2.0, -1, 1e-12)?;
    let txt_norm = z_txt.normalize(2.0, -1, 1e-12)?;
    let logits = img_norm
        .matmul(&txt_norm.t()?)?
        .mul_scalar(temperature)?
        .add_scalar(bias)?;

    let mut labels = vec![-1.0f64; n * n];
    for i in 0..n {
        labels[i * n + i] = 1.0;
    }
    let y = Tensor::from_slice_on(&labels, &[n, n], &logits.device())?.to_dtype(logits.dtype())?;
    logits.mul(&y)?.neg()?.softplus()?.mean()
}

/// **Barlow Twins** self-supervised redundancy-reduction loss (Zbontar et al. 2021):
/// Standardizes representations `z_a` and `z_b` (`[B, D]`) along the batch dimension, computes
/// the empirical cross-correlation matrix $C = \frac{1}{B} \hat{Z}_A^T \hat{Z}_B$ (`[D, D]`),
/// and minimizes:
/// $$\mathcal{L}_{\text{BT}} = \sum_{i=1}^D (1 - C_{ii})^2 + \lambda \sum_{i \ne j} C_{ij}^2$$
pub fn barlow_twins_loss(z_a: &Tensor, z_b: &Tensor, lambda: f64) -> Result<Tensor> {
    if z_a.ndim() != 2 || z_b.ndim() != 2 || z_a.shape() != z_b.shape() {
        return Err(Error::ShapeMismatch {
            op: "barlow_twins_loss",
            expected: z_a.shape().to_vec(),
            got: z_b.shape().to_vec(),
        });
    }
    let b = z_a.shape()[0] as f64;
    let za_norm = z_a
        .sub(&z_a.mean_axis(0, true)?.broadcast_to(z_a.shape())?)?
        .div(
            &z_a.var_axis(0, true, false)?
                .add_scalar(1e-5)?
                .sqrt()?
                .broadcast_to(z_a.shape())?,
        )?;
    let zb_norm = z_b
        .sub(&z_b.mean_axis(0, true)?.broadcast_to(z_b.shape())?)?
        .div(
            &z_b.var_axis(0, true, false)?
                .add_scalar(1e-5)?
                .sqrt()?
                .broadcast_to(z_b.shape())?,
        )?;

    let c_mat = za_norm.t()?.matmul(&zb_norm)?.div_scalar(b)?;
    let diag = c_mat.diagonal(0, 0, 1)?;
    let invariance = diag.sub_scalar(1.0)?.square()?.sum()?;
    let off_diag_sq = c_mat.square()?.sum()?.sub(&diag.square()?.sum()?)?;
    invariance.add(&off_diag_sq.mul_scalar(lambda)?)
}

/// Margin Ranking Loss (`torch.nn.MarginRankingLoss`):
/// $\text{loss}(x_1, x_2, y) = \max(0, -y(x_1 - x_2) + \text{margin})$
pub fn margin_ranking_loss(
    x1: &Tensor,
    x2: &Tensor,
    target: &Tensor,
    margin: f64,
) -> Result<Tensor> {
    let diff = x1.sub(x2)?;
    let scaled = diff.mul(target)?.neg()?.add_scalar(margin)?;
    scaled.relu()?.mean()
}

/// Margin Ranking Loss module (`torch.nn.MarginRankingLoss`).
#[derive(Clone, Copy, Debug)]
pub struct MarginRankingLoss {
    pub margin: f64,
}
impl Default for MarginRankingLoss {
    fn default() -> Self {
        Self { margin: 0.0 }
    }
}
impl MarginRankingLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
    pub fn forward(&self, x1: &Tensor, x2: &Tensor, target: &Tensor) -> Result<Tensor> {
        margin_ranking_loss(x1, x2, target, self.margin)
    }
}

/// Hinge Embedding Loss (`torch.nn.HingeEmbeddingLoss`):
/// $l_n = x_n$ if $y_n = 1$, and $\max(0, \text{margin} - x_n)$ if $y_n = -1$.
pub fn hinge_embedding_loss(input: &Tensor, target: &Tensor, margin: f64) -> Result<Tensor> {
    let pos = input.clone();
    let neg = input.neg()?.add_scalar(margin)?.relu()?;
    let cond = target.gt(&target.full_like(0.0)?)?;
    Tensor::where_cond(&cond, &pos, &neg)?.mean()
}

/// Hinge Embedding Loss module (`torch.nn.HingeEmbeddingLoss`).
#[derive(Clone, Copy, Debug)]
pub struct HingeEmbeddingLoss {
    pub margin: f64,
}
impl Default for HingeEmbeddingLoss {
    fn default() -> Self {
        Self { margin: 1.0 }
    }
}
impl HingeEmbeddingLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        hinge_embedding_loss(input, target, self.margin)
    }
}

/// Soft Margin Loss (`torch.nn.SoftMarginLoss`):
/// $\text{loss}(x, y) = \frac{1}{N} \sum_i \ln(1 + \exp(-y_i \cdot x_i))$
pub fn soft_margin_loss(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    input.mul(target)?.neg()?.softplus()?.mean()
}

/// Soft Margin Loss module (`torch.nn.SoftMarginLoss`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftMarginLoss;
impl SoftMarginLoss {
    pub fn new() -> Self {
        Self
    }
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        soft_margin_loss(input, target)
    }
}

/// Multi-Label Margin Loss (`torch.nn.MultiLabelMarginLoss`):
/// $\text{loss}(x, y) = \sum_{ij} \frac{\max(0, 1 - (x\[y\[j\]\] - x\[i\]))}{x.size(0)}$
pub fn multilabel_margin_loss(input: &Tensor, target: &Tensor) -> Result<Tensor> {
    if input.ndim() != 2 || target.ndim() != 2 {
        return Err(Error::Msg(
            "multilabel_margin_loss expects 2D tensors [N, C]".into(),
        ));
    }
    let (n, c) = (input.shape()[0], input.shape()[1]);
    if n == 0 || c == 0 {
        return Err(Error::Msg(format!(
            "multilabel_margin_loss requires non-empty [N, C], got [{n}, {c}]"
        )));
    }
    let y_vec = target.to_vec::<f64>()?;
    // Collect every (target class, non-target class) pair; the hinge terms are then built from differentiable gathers.
    let (mut idx_t, mut idx_i): (Vec<i64>, Vec<i64>) = (Vec::new(), Vec::new());
    for b in 0..n {
        let y_b = &y_vec[b * c..(b + 1) * c];
        // the target list ends at the first negative entry; classes in it are never used as the "negative" side
        let mut is_target = vec![false; c];
        for &yj in y_b {
            let target_idx = yj as isize;
            if target_idx < 0 {
                break;
            }
            if (target_idx as usize) < c {
                is_target[target_idx as usize] = true;
            }
        }
        for &yj in y_b {
            let target_idx = yj as isize;
            if target_idx < 0 {
                break;
            }
            let tj = target_idx as usize;
            if tj >= c {
                continue;
            }
            for (i, _) in is_target.iter().enumerate().filter(|(_, t)| !**t) {
                idx_t.push((b * c + tj) as i64);
                idx_i.push((b * c + i) as i64);
            }
        }
    }
    if idx_t.is_empty() {
        return input.sum()?.mul_scalar(0.0);
    }
    let flat = input.reshape(&[n * c])?;
    let it = Tensor::from_slice_on(&idx_t, &[idx_t.len()], &input.device())?;
    let ii = Tensor::from_slice_on(&idx_i, &[idx_i.len()], &input.device())?;
    // sum over pairs of max(0, 1 - (x[target] - x[other])), divided by C per sample and averaged over the batch
    let hinge = flat.gather(0, &it)?.sub(&flat.gather(0, &ii)?)?.neg()?.add_scalar(1.0)?.relu()?;
    hinge.sum()?.div_scalar((c * n) as f64)
}

/// Connectionist Temporal Classification loss (`torch.nn.CTCLoss`).
/// Performs dynamic-programming forward evaluation in log-space. Like `reduction="mean"` in PyTorch, every sample's
/// loss is divided by its target length before averaging over the batch.
/// `log_probs: [T, N, C]`, `targets: [N, S]`, `blank: usize`, `zero_infinity: bool`.
pub fn ctc_loss(
    log_probs: &Tensor,
    targets: &Tensor,
    blank: usize,
    zero_infinity: bool,
) -> Result<Tensor> {
    if log_probs.ndim() != 3 {
        return Err(Error::Msg("ctc_loss expects log_probs [T, N, C]".into()));
    }
    let (t_steps, n_batch, c_classes) = (
        log_probs.shape()[0],
        log_probs.shape()[1],
        log_probs.shape()[2],
    );
    if blank >= c_classes {
        return Err(Error::Msg(format!(
            "blank index {blank} >= classes {c_classes}"
        )));
    }
    let (target_n, s_len) = if targets.ndim() == 2 {
        (targets.shape()[0], targets.shape()[1])
    } else {
        return Err(Error::Msg("ctc_loss expects targets [N, S]".into()));
    };
    if target_n != n_batch {
        return Err(Error::Msg(format!(
            "batch size mismatch in ctc_loss: {n_batch} vs {target_n}"
        )));
    }
    if t_steps == 0 {
        return Err(Error::Msg(
            "ctc_loss needs at least one input timestep".into(),
        ));
    }
    if n_batch == 0 {
        return Err(Error::Msg(
            "ctc_loss needs a non-empty batch (N == 0 would average over nothing)".into(),
        ));
    }

    let lp_vec = log_probs.to_vec::<f64>()?;
    let tgt_vec = targets.to_vec::<f64>()?;
    let neg_inf = f64::NEG_INFINITY;

    let lse2 = |a: f64, b: f64| -> f64 {
        if a == neg_inf {
            b
        } else if b == neg_inf {
            a
        } else {
            let m = a.max(b);
            m + ((a - m).exp() + (b - m).exp()).ln()
        }
    };
    let lse3 = |a: f64, b: f64, c: f64| -> f64 { lse2(lse2(a, b), c) };

    let mut batch_losses = Vec::with_capacity(n_batch);
    // d loss / d log_probs, already scaled by the 'mean' reduction (1 / (N * target length)); PyTorch's convention
    // `exp(lp) - exp(log(alpha beta) - lp + nll)` is used so the result also matches when fed to log_softmax.
    let mut grad = vec![0.0f64; t_steps * n_batch * c_classes];

    for b in 0..n_batch {
        let mut sample_target = Vec::new();
        for s in 0..s_len {
            let val = tgt_vec[b * s_len + s] as isize;
            if val >= 0 && (val as usize) < c_classes {
                if val as usize == blank {
                    return Err(Error::Msg(format!(
                        "ctc_loss: target {b} contains the blank label {blank}"
                    )));
                }
                sample_target.push(val as usize);
            }
        }
        let u = sample_target.len();
        if t_steps < u {
            return Err(Error::Msg(format!(
                "ctc_loss: target {b} needs {u} frames but input has {t_steps}"
            )));
        }
        let l_prime = 2 * u + 1;
        let mut ext = Vec::with_capacity(l_prime);
        for &tok in &sample_target {
            ext.push(blank);
            ext.push(tok);
        }
        ext.push(blank);
        let lp = |t: usize, cls: usize| -> f64 { lp_vec[(t * n_batch + b) * c_classes + cls] };
        // skip transition s-2 -> s is allowed between two different non-blank labels
        let skip_ok = |s: usize| s >= 2 && ext[s] != blank && ext[s] != ext[s - 2];

        // forward variables (they include the emission at their own time step)
        let mut alpha = vec![vec![neg_inf; l_prime]; t_steps];
        alpha[0][0] = lp(0, blank);
        if l_prime > 1 {
            alpha[0][1] = lp(0, ext[1]);
        }
        for t in 1..t_steps {
            for s in 0..l_prime {
                let prev1 = if s > 0 { alpha[t - 1][s - 1] } else { neg_inf };
                let prev2 = if skip_ok(s) { alpha[t - 1][s - 2] } else { neg_inf };
                alpha[t][s] = lse3(alpha[t - 1][s], prev1, prev2) + lp(t, ext[s]);
            }
        }
        let total_log_p = if l_prime > 1 { lse2(alpha[t_steps - 1][l_prime - 1], alpha[t_steps - 1][l_prime - 2]) } else { alpha[t_steps - 1][0] };

        let loss_b = -total_log_p;
        if loss_b.is_nan() || loss_b.is_infinite() {
            batch_losses.push(if zero_infinity { 0.0 } else { 1e6 });
            continue; // an impossible alignment has no meaningful gradient
        }
        let scale = 1.0 / ((u.max(1) * n_batch) as f64);
        batch_losses.push(loss_b / (u.max(1) as f64));

        // backward variables (also including their own emission)
        let mut beta = vec![vec![neg_inf; l_prime]; t_steps];
        beta[t_steps - 1][l_prime - 1] = lp(t_steps - 1, ext[l_prime - 1]);
        if l_prime > 1 {
            beta[t_steps - 1][l_prime - 2] = lp(t_steps - 1, ext[l_prime - 2]);
        }
        for t in (0..t_steps - 1).rev() {
            for s in 0..l_prime {
                let next1 = if s + 1 < l_prime { beta[t + 1][s + 1] } else { neg_inf };
                let next2 = if s + 2 < l_prime && skip_ok(s + 2) { beta[t + 1][s + 2] } else { neg_inf };
                beta[t][s] = lse3(beta[t + 1][s], next1, next2) + lp(t, ext[s]);
            }
        }
        for t in 0..t_steps {
            // log sum over the extended positions carrying each class of alpha * beta
            let mut lab = vec![neg_inf; c_classes];
            for s in 0..l_prime {
                lab[ext[s]] = lse2(lab[ext[s]], alpha[t][s] + beta[t][s]);
            }
            for k in 0..c_classes {
                let l = lp(t, k);
                let dl = l.exp() - if lab[k] == neg_inf { 0.0 } else { (lab[k] - l + loss_b).exp() };
                grad[(t * n_batch + b) * c_classes + k] = dl * scale;
            }
        }
    }

    let mean_loss = batch_losses.iter().sum::<f64>() / (n_batch as f64);
    let out_layout = crate::layout::Layout::contiguous(vec![]);
    let out_storage = crate::backend::CpuStorage::F64(vec![mean_loss]).to_dtype(&out_layout, log_probs.dtype())?;
    let storage = crate::backend::Storage::new_cpu(out_storage).to_device(&out_layout, &log_probs.device())?;
    let (lp_shape, lp_dtype, lp_device) = (log_probs.shape().to_vec(), log_probs.dtype(), log_probs.device());
    Ok(Tensor::from_op(storage, out_layout, "ctc_loss", vec![log_probs.clone()], move |grad_out| {
        let go = grad_out.to_vec::<f64>()?[0];
        let g: Vec<f64> = grad.iter().map(|v| v * go).collect();
        Ok(vec![Some(Tensor::from_slice_on(&g, &lp_shape, &lp_device)?.to_dtype(lp_dtype)?)])
    }))
}

/// CTC Loss module (`torch.nn.CTCLoss`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CTCLoss {
    pub blank: usize,
    pub zero_infinity: bool,
}
impl CTCLoss {
    pub fn new(blank: usize, zero_infinity: bool) -> Self {
        Self {
            blank,
            zero_infinity,
        }
    }
    pub fn forward(&self, log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        ctc_loss(log_probs, targets, self.blank, self.zero_infinity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central finite differences of a scalar loss of one tensor.
    fn numeric_grad(f: &dyn Fn(&Tensor) -> Tensor, x: &[f64], shape: &[usize]) -> Vec<f64> {
        let at = |v: &[f64]| f(&Tensor::from_slice(v, shape).unwrap()).to_vec::<f64>().unwrap()[0];
        (0..x.len())
            .map(|i| {
                let (mut hi, mut lo) = (x.to_vec(), x.to_vec());
                hi[i] += 1e-6;
                lo[i] -= 1e-6;
                (at(&hi) - at(&lo)) / 2e-6
            })
            .collect()
    }

    #[test]
    fn ctc_and_multilabel_margin_are_differentiable() {
        // CTC: gradient w.r.t. logits (through log_softmax) equals finite differences
        let raw: Vec<f64> = (0..5 * 2 * 4).map(|i| ((i * 7 % 11) as f64 - 5.0) / 3.0).collect();
        let tg = Tensor::from_slice(&[1i64, 2, 3, 2, 2, 1], &[2, 3]).unwrap();
        let f = |x: &Tensor| ctc_loss(&x.log_softmax(2).unwrap(), &tg, 0, false).unwrap();
        let x = Tensor::from_slice(&raw, &[5, 2, 4]).unwrap().requires_grad_(true);
        f(&x).backward().unwrap();
        assert_close(&x.grad().unwrap().to_vec::<f64>().unwrap(), &numeric_grad(&f, &raw, &[5, 2, 4]), 1e-6, "ctc grad");
        // multilabel margin: d loss / d x for a case with active hinge terms
        let xs = [0.3, -1.2, 2.5, 0.9, -0.4, 1.8];
        let ml = Tensor::from_slice(&[1.0f64, 2.0, -1.0, 0.0, 2.0, -1.0], &[2, 3]).unwrap();
        let g = |x: &Tensor| multilabel_margin_loss(x, &ml).unwrap();
        let x = Tensor::from_slice(&xs, &[2, 3]).unwrap().requires_grad_(true);
        g(&x).backward().unwrap();
        assert_close(&x.grad().unwrap().to_vec::<f64>().unwrap(), &numeric_grad(&g, &xs, &[2, 3]), 1e-6, "multilabel grad");
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() <= tol, "{what} element {i}: {g} vs {w}");
        }
    }

    #[test]
    fn multilabel_margin_and_ctc_follow_torch_conventions() {
        // torch.nn.functional.multilabel_margin_loss: target classes are never the "negative" side
        let x = Tensor::from_slice(&[0.3f64, -1.2, 2.5, 0.9, -0.4, 1.8], &[2, 3]).unwrap();
        let y = Tensor::from_slice(&[1.0f64, 2.0, -1.0, 0.0, 2.0, -1.0], &[2, 3]).unwrap();
        let got = multilabel_margin_loss(&x, &y).unwrap().item::<f64>().unwrap();
        assert!((got - 0.416_666_666_666_666_7).abs() < 1e-12, "{got}");
        // torch ctc_loss(reduction="mean") divides each sample by its target length: 1.6509060634022887
        let raw: Vec<f64> = (0..48).map(|i| ((i * 7 % 11) as f64 - 5.0) / 3.0).collect();
        let lp = Tensor::from_slice(&raw, &[6, 2, 4]).unwrap().log_softmax(2).unwrap();
        let tg = Tensor::from_slice(&[1i64, 2, 3, 2, 2, 1], &[2, 3]).unwrap();
        let got = ctc_loss(&lp, &tg, 0, false).unwrap().item::<f64>().unwrap();
        assert!((got - 1.650_906_063_402_288_7).abs() < 1e-9, "{got}");
    }

    #[test]
    fn classification_losses_match_torch() {
        // all values and grads torch-verified (f64)
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3])
            .unwrap()
            .requires_grad_(true);
        let ce = cross_entropy(&logits, &[2, 0]).unwrap();
        assert!((ce.item::<f64>().unwrap() - 0.7531091265562451).abs() <= 1e-12);
        ce.backward().unwrap();
        assert_close(
            &logits.grad().unwrap().to_vec::<f64>().unwrap(),
            &[
                0.04501528658519022,
                0.12236423552739882,
                -0.1673795221125891,
                -0.33333333333333337,
                0.16666666666666666,
                0.16666666666666666,
            ],
            1e-12,
            "ce grad",
        );
        let sm = cross_entropy_with_smoothing(&logits, &[2, 0], 0.1).unwrap();
        assert!((sm.item::<f64>().unwrap() - 0.8031091265562451).abs() <= 1e-12);
        // no smoothing collapses exactly onto cross_entropy
        let hard = cross_entropy_with_smoothing(&logits, &[2, 0], 0.0).unwrap();
        assert!((hard.item::<f64>().unwrap() - 0.7531091265562451).abs() <= 1e-12);
        // extreme logits stay finite (stable formulation, torch-verified)
        let bl = Tensor::from_slice(&[50.0, -50.0, 0.0], &[3, 1]).unwrap();
        let bt = Tensor::from_slice(&[1.0, 0.0, 1.0], &[3, 1]).unwrap();
        let bce = binary_cross_entropy_with_logits(&bl, &bt).unwrap();
        assert!((bce.item::<f64>().unwrap() - 0.23104906018664842).abs() <= 1e-12);
        let fl = focal_loss(&logits, &[2, 0], 0.25, 2.0).unwrap();
        assert!((fl.item::<f64>().unwrap() - 0.06674374090810284).abs() <= 1e-12);
        // gamma = 0 reduces to alpha-weighted cross-entropy
        let fl0 = focal_loss(&logits, &[2, 0], 0.5, 0.0).unwrap();
        assert!((fl0.item::<f64>().unwrap() - 0.5 * 0.7531091265562451).abs() <= 1e-12);
        let lq = Tensor::from_slice(&[1.0, 2.0, 3.0, 1.0], &[2, 2])
            .unwrap()
            .log_softmax(1)
            .unwrap();
        let p = Tensor::from_slice(&[0.5, 0.5, 1.0, 0.0], &[2, 2]).unwrap();
        let kl = kl_div_loss(&lq, &p).unwrap();
        assert!((kl.item::<f64>().unwrap() - 0.12352125900062508).abs() <= 1e-12);
    }

    #[test]
    fn regression_and_metric_losses_match_hand_values() {
        let pred = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
        let tgt = Tensor::from_slice(&[0.0, 0.0], &[2]).unwrap();
        assert!((mse_loss(&pred, &tgt).unwrap().item::<f64>().unwrap() - 2.5).abs() <= 1e-12);
        assert!((l1_loss(&pred, &tgt).unwrap().item::<f64>().unwrap() - 1.5).abs() <= 1e-12);
        // huber: (0.125 + 1.0 + 2.5) / 3 (torch-verified)
        let ph = Tensor::from_slice(&[0.5, 1.5, 3.0], &[3]).unwrap();
        let th = Tensor::from_slice(&[0.0, 0.0, 0.0], &[3]).unwrap();
        let h = huber_loss(&ph, &th, 1.0).unwrap();
        assert!((h.item::<f64>().unwrap() - 1.2083333333333333).abs() <= 1e-12);
        assert!(huber_loss(&ph, &th, 0.0).is_err());
        assert!(huber_loss(&ph, &th, f64::INFINITY).is_err());
        // 1 - 1/sqrt(2) (torch-verified)
        let a = Tensor::from_slice(&[1.0, 0.0], &[1, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0, 1.0], &[1, 2]).unwrap();
        let lab = Tensor::from_slice(&[1.0], &[1]).unwrap();
        let c = cosine_embedding_loss(&a, &b, &lab, 0.5).unwrap();
        assert!((c.item::<f64>().unwrap() - 0.2928932188139829).abs() <= 1e-12);
    }

    #[test]
    fn loss_validation_fails_closed() {
        let lp = Tensor::from_slice(&[0.0, 0.0, 0.0], &[1, 3]).unwrap();
        // empty batch errors (torch returns NaN here: deliberate divergence)
        let empty = Tensor::from_slice(&[] as &[f64], &[0, 3]).unwrap();
        assert!(nll_loss(&empty, &[]).is_err());
        assert!(focal_loss(&empty, &[], 0.25, 2.0).is_err());
        assert!(nll_loss(&lp, &[0, 1]).is_err());
        assert!(nll_loss(&lp, &[usize::MAX]).is_err());
        let bad_rank = Tensor::from_slice(&[0.0; 6], &[1, 3, 2]).unwrap();
        assert!(nll_loss(&bad_rank, &[0]).is_err());
        assert!(focal_loss(&bad_rank, &[0], 0.25, 2.0).is_err());
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        assert!(cross_entropy_with_smoothing(&logits, &[0], 1.0).is_err());
        assert!(cross_entropy_with_smoothing(&logits, &[0], -0.5).is_err());
        assert!(cross_entropy_with_smoothing(&logits, &[0], f64::NAN).is_err());
        let q = Tensor::from_slice(&[1.0, 0.0], &[1, 2]).unwrap();
        assert!(info_nce_loss(&q, &q, 0.0).is_err());
        assert!(info_nce_loss(&q, &q, f64::INFINITY).is_err());
        let k3 = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        assert!(info_nce_loss(&q, &k3, 1.0).is_err());
        assert!(compute_gae(&[1.0], &[0.0, 0.0], &[false], 0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn gae_and_policy_losses_match_hand_values() {
        // gamma = lambda = 1: A = [2, 1], R = [2, 1]
        let (a, r) = compute_gae(&[1.0, 1.0], &[0.0, 0.0], &[false, false], 0.0, 1.0, 1.0).unwrap();
        assert_close(&a, &[2.0, 1.0], 0.0, "gae adv");
        assert_close(&r, &[2.0, 1.0], 0.0, "gae ret");
        // terminal mask stops bootstrapping: A = [1, 1], R = [1, 1]
        let (a, _) = compute_gae(&[1.0, 1.0], &[0.0, 0.0], &[true, true], 9.0, 1.0, 1.0).unwrap();
        assert_close(&a, &[1.0, 1.0], 0.0, "gae terminal");
        // dpo: softplus(-1) = ln(1 + 1/e)
        let pc = Tensor::from_slice(&[1.0], &[1]).unwrap();
        let pr = Tensor::from_slice(&[0.0], &[1]).unwrap();
        let rc = Tensor::from_slice(&[0.0], &[1]).unwrap();
        let rr = Tensor::from_slice(&[0.0], &[1]).unwrap();
        let d = dpo_loss(&pc, &pr, &rc, &rr, 1.0).unwrap();
        assert!((d.item::<f64>().unwrap() - 0.3132616875182228).abs() <= 1e-9);
        // ppo with unit ratio: -mean([2, -1]) = -0.5; grads reach new_logps only
        let new = Tensor::from_slice(&[0.0, 0.0], &[2])
            .unwrap()
            .requires_grad_(true);
        let old = Tensor::from_slice(&[0.0, 0.0], &[2])
            .unwrap()
            .requires_grad_(true);
        let adv = Tensor::from_slice(&[2.0, -1.0], &[2])
            .unwrap()
            .requires_grad_(true);
        let p = ppo_clipped_loss(&new, &old, &adv, 0.2).unwrap();
        assert!((p.item::<f64>().unwrap() + 0.5).abs() <= 1e-12);
        p.backward().unwrap();
        assert!(new.grad().is_some() && old.grad().is_none() && adv.grad().is_none());
    }
}
