//! Model compression: Unstructured & structured magnitude pruning, Lottery Ticket Hypothesis
//! [`PruningMask`], and Hinton Knowledge Distillation ([`distillation_loss`]).

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::nn::Module;
use crate::nn::loss::{cross_entropy, kl_div_loss};
use crate::tensor::{Tensor, no_grad};

/// Binary mask dictionary over a [`Module`]'s parameters for unstructured/structured pruning
/// and Lottery Ticket Hypothesis experiments.
#[derive(Clone, Default)]
pub struct PruningMask {
    pub masks: HashMap<String, Tensor>,
}

impl PruningMask {
    /// Re-applies the binary pruning masks in-place to `model`'s parameters (e.g. after an optimizer step).
    pub fn apply<M: Module>(&self, model: &M) -> Result<()> {
        no_grad(|| {
            for (name, param) in model.named_parameters() {
                if let Some(mask) = self.masks.get(&name) {
                    let masked = param.mul(mask)?;
                    param.assign(&masked)?;
                }
            }
            Ok(())
        })
    }

    /// Computes the overall fraction of zeroed elements across all masked parameters.
    pub fn overall_sparsity(&self) -> Result<f64> {
        let mut total = 0usize;
        let mut nonzero = 0.0f64;
        for m in self.masks.values() {
            total += m.numel();
            nonzero += m.to_vec::<f64>()?.into_iter().sum::<f64>();
        }
        if total == 0 {
            return Ok(0.0);
        }
        Ok((1.0 - nonzero / (total as f64)).clamp(0.0, 1.0))
    }
}

/// Global unstructured magnitude pruning: zeroes out the `sparsity` fraction ($s \in [0, 1)$)
/// of multi-dimensional weight parameters with the smallest $|w|$, applies the mask in-place,
/// and returns a [`PruningMask`].
pub fn prune_magnitude_unstructured<M: Module>(model: &M, sparsity: f64) -> Result<PruningMask> {
    if !(0.0..1.0).contains(&sparsity) {
        return Err(Error::Msg(format!(
            "prune_magnitude_unstructured requires sparsity in [0, 1), got {sparsity}"
        )));
    }
    let named = model.named_parameters();
    let mut ranked: Vec<(f64, usize, usize)> = Vec::new();
    let mut param_vecs: Vec<Vec<f64>> = Vec::with_capacity(named.len());
    for (p_idx, (_name, p)) in named.iter().enumerate() {
        if p.ndim() >= 2 {
            let vals = p.to_vec::<f64>()?;
            for (e_idx, &v) in vals.iter().enumerate() {
                ranked.push((v.abs(), p_idx, e_idx));
            }
            param_vecs.push(vals);
        } else {
            param_vecs.push(Vec::new());
        }
    }
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let num_zero = ((ranked.len() as f64) * sparsity).round() as usize;
    let num_zero = num_zero.min(ranked.len());

    let mut mask_vecs: Vec<Vec<f64>> = param_vecs.iter().map(|v| vec![1.0f64; v.len()]).collect();
    for &(_mag, p_idx, e_idx) in ranked.iter().take(num_zero) {
        mask_vecs[p_idx][e_idx] = 0.0;
    }

    let mut masks = HashMap::new();
    for ((name, p), mask_vec) in named.into_iter().zip(mask_vecs) {
        if p.ndim() >= 2 {
            let mask_t =
                Tensor::from_slice_on(&mask_vec, p.shape(), &p.device())?.to_dtype(p.dtype())?;
            masks.insert(name, mask_t);
        }
    }

    let pruning_mask = PruningMask { masks };
    pruning_mask.apply(model)?;
    Ok(pruning_mask)
}

/// Structured $\ell_2$ channel/row pruning on a 2D weight matrix `[out_features, in_features]`:
/// retains the top `keep_ratio` fraction of rows with largest $\ell_2$ norm, returning
/// `(pruned_weight, kept_row_indices)`.
pub fn prune_structured_l2_rows(weight: &Tensor, keep_ratio: f64) -> Result<(Tensor, Vec<usize>)> {
    if weight.ndim() != 2 {
        return Err(Error::Msg(format!(
            "prune_structured_l2_rows expects 2D weight tensor, got {:?}",
            weight.shape()
        )));
    }
    let rows = weight.shape()[0];
    let keep = ((rows as f64) * keep_ratio.clamp(0.0, 1.0))
        .round()
        .max(1.0) as usize;
    let keep = keep.min(rows);

    let row_norms = weight.norm_p(2.0, &[1], false)?.to_vec::<f64>()?;
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| {
        row_norms[b]
            .partial_cmp(&row_norms[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(keep);
    order.sort_unstable();

    let idx_i64: Vec<i64> = order.iter().map(|&i| i as i64).collect();
    let idx_t = Tensor::from_slice_on(&idx_i64, &[keep], &weight.device())?;
    let pruned = weight.index_select(0, &idx_t)?;
    Ok((pruned, order))
}

/// Knowledge Distillation loss (Hinton et al. 2015) combining temperature-scaled KL divergence
/// between `student_logits` and `teacher_logits` with hard-label `cross_entropy`:
/// $$\mathcal{L}_{\text{KD}} = \alpha T^2 \operatorname{KL}\!\left(\log \sigma(z_s / T) \,\|\, \sigma(z_t / T)\right) + (1 - \alpha) \mathcal{L}_{\text{CE}}(z_s, y)$$
pub fn distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    targets: &[usize],
    temperature: f64,
    alpha: f64,
) -> Result<Tensor> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(crate::error::Error::Msg(format!(
            "distillation_loss requires alpha in [0, 1], got {alpha}"
        )));
    }
    let t = temperature.max(1e-4);
    let log_p_student = student_logits.div_scalar(t)?.log_softmax(-1)?;
    let q_teacher = teacher_logits.detach().div_scalar(t)?.softmax(-1)?;
    let kd_part = kl_div_loss(&log_p_student, &q_teacher)?.mul_scalar(t * t * alpha)?;
    let ce_part = cross_entropy(student_logits, targets)?.mul_scalar(1.0 - alpha)?;
    kd_part.add(&ce_part)
}
