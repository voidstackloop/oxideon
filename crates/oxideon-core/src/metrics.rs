//! Evaluation metrics for classification, regression, and language modeling.

use crate::error::Result;
use crate::tensor::Tensor;

/// Top-1 classification accuracy given `logits: [batch, classes]` and integer `targets`.
///
/// Rejects empty `targets` and `targets.len() != batch` instead of silently
/// scoring a truncated zip (or `0/1` for empty input).
pub fn accuracy(logits: &Tensor, targets: &[usize]) -> Result<f64> {
    if logits.ndim() != 2 {
        return Err(crate::error::Error::Msg(format!(
            "accuracy expects 2D [batch, classes] logits, got {:?}",
            logits.shape()
        )));
    }
    let batch = logits.shape()[0];
    if targets.is_empty() {
        return Err(crate::error::Error::Msg(
            "accuracy: targets must be non-empty".into(),
        ));
    }
    if targets.len() != batch {
        return Err(crate::error::Error::Msg(format!(
            "accuracy: targets length {} != batch size {batch}",
            targets.len()
        )));
    }
    let preds = logits.argmax_axis(1, false)?.to_vec::<i64>()?;
    let correct = preds
        .iter()
        .zip(targets)
        .filter(|&(&p, &t)| p as usize == t)
        .count();
    Ok(correct as f64 / targets.len() as f64)
}

/// Top-$k$ classification accuracy given `logits: [batch, classes]` and integer `targets`.
///
/// Requires `1 <= k <= classes` (`k == 0` would otherwise score every row
/// wrong against an empty slice) and `targets.len() == batch` (extra targets
/// would otherwise slice out of bounds).
pub fn top_k_accuracy(logits: &Tensor, targets: &[usize], k: usize) -> Result<f64> {
    if logits.ndim() != 2 {
        return Err(crate::error::Error::Msg(format!(
            "top_k_accuracy expects 2D [batch, classes] logits, got {:?}",
            logits.shape()
        )));
    }
    let (batch, classes) = (logits.shape()[0], logits.shape()[1]);
    if k == 0 || k > classes {
        return Err(crate::error::Error::Msg(format!(
            "top_k_accuracy: k must satisfy 1 <= k <= classes ({classes}), got {k}"
        )));
    }
    if targets.is_empty() {
        return Err(crate::error::Error::Msg(
            "top_k_accuracy: targets must be non-empty".into(),
        ));
    }
    if targets.len() != batch {
        return Err(crate::error::Error::Msg(format!(
            "top_k_accuracy: targets length {} != batch size {batch}",
            targets.len()
        )));
    }
    let (_, top_idx) = logits.topk(k, 1, true, true)?;
    let idx_vec = top_idx.to_vec::<i64>()?;
    let mut correct = 0usize;
    for (row, &t) in targets.iter().enumerate() {
        let row_slice = &idx_vec[row * k..(row + 1) * k];
        if row_slice.contains(&(t as i64)) {
            correct += 1;
        }
    }
    Ok(correct as f64 / targets.len() as f64)
}

/// Computes `num_classes x num_classes` confusion matrix where entry `[true_class, pred_class]`
/// is the count of samples.
///
/// Lenient by design: entries are zipped, so length mismatches truncate to
/// the shorter side, and out-of-range labels are skipped rather than counted.
/// Validate label ranges up front when a miscount would hide a data bug.
/// Absurd `num_classes` is an `Err` (the `n^2` matrix cannot be built).
pub fn confusion_matrix(
    preds: &[usize],
    targets: &[usize],
    num_classes: usize,
) -> Result<Vec<Vec<usize>>> {
    crate::limits::check_elements("confusion_matrix", &[num_classes, num_classes])?;
    let mut mat = vec![vec![0usize; num_classes]; num_classes];
    for (&p, &t) in preds.iter().zip(targets) {
        if t < num_classes && p < num_classes {
            mat[t][p] += 1;
        }
    }
    Ok(mat)
}

/// Macro-averaged `(precision, recall, f1_score)` across `num_classes`.
pub fn precision_recall_f1(
    preds: &[usize],
    targets: &[usize],
    num_classes: usize,
) -> Result<(f64, f64, f64)> {
    let cm = confusion_matrix(preds, targets, num_classes)?;
    let mut sum_p = 0.0f64;
    let mut sum_r = 0.0f64;
    let mut sum_f1 = 0.0f64;

    for (c, row) in cm.iter().enumerate() {
        let tp = row[c] as f64;
        let fp: f64 = (0..num_classes)
            .filter(|&r| r != c)
            .map(|r| cm[r][c] as f64)
            .sum();
        let fn_cnt: f64 = (0..num_classes)
            .filter(|&k| k != c)
            .map(|k| row[k] as f64)
            .sum();

        let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let r = if tp + fn_cnt > 0.0 {
            tp / (tp + fn_cnt)
        } else {
            0.0
        };
        let f1 = if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        };
        sum_p += p;
        sum_r += r;
        sum_f1 += f1;
    }
    let nc = num_classes.max(1) as f64;
    Ok((sum_p / nc, sum_r / nc, sum_f1 / nc))
}

/// Area Under the Receiver Operating Characteristic curve (AUROC) for binary scores and `{0, 1}` targets.
///
/// Lenient by design (it returns a score, not a `Result`): inputs are zipped
/// (length mismatches truncate), non-`{0, 1}` targets count as negatives, NaN
/// scores sort as ties, and single-class input scores `0.5`.
pub fn auroc(scores: &[f64], targets: &[usize]) -> f64 {
    let mut pairs: Vec<(f64, usize)> = scores
        .iter()
        .copied()
        .zip(targets.iter().copied())
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Totals over the zipped pairs (not raw `targets`), so length mismatches
    // score the truncated data consistently instead of warping the rate.
    let total_pos = pairs.iter().filter(|&&(_, t)| t == 1).count() as f64;
    let total_neg = pairs.len() as f64 - total_pos;
    if total_pos == 0.0 || total_neg == 0.0 {
        return 0.5;
    }

    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    let mut prev_tp = 0.0f64;
    let mut prev_fp = 0.0f64;
    let mut area = 0.0f64;

    for (_, label) in pairs {
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        area += (fp - prev_fp) * (tp + prev_tp) * 0.5;
        prev_tp = tp;
        prev_fp = fp;
    }
    area / (total_pos * total_neg)
}

/// Coefficient of determination $R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$.
pub fn r2_score(pred: &Tensor, target: &Tensor) -> Result<f64> {
    if pred.numel() == 0 || target.numel() == 0 {
        return Err(crate::error::Error::Msg(
            "r2_score: inputs must be non-empty".into(),
        ));
    }
    let ss_res = pred.sub(target)?.square()?.sum()?.item::<f64>()?;
    let mean_y = target.mean()?;
    let ss_tot = target.sub(&mean_y)?.square()?.sum()?.item::<f64>()?;
    // sklearn parity: a perfect constant fit scores 1.0, an imperfect one 0.0.
    if ss_tot <= 1e-12 {
        Ok(if ss_res <= 1e-12 { 1.0 } else { 0.0 })
    } else {
        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Root mean squared error $\sqrt{\text{MSE}}$.
pub fn rmse(pred: &Tensor, target: &Tensor) -> Result<f64> {
    if pred.numel() == 0 || target.numel() == 0 {
        return Err(crate::error::Error::Msg(
            "rmse: inputs must be non-empty (mean over zero elements would be NaN)".into(),
        ));
    }
    Ok(pred.sub(target)?.square()?.mean()?.item::<f64>()?.sqrt())
}

/// Perplexity $\exp(\text{cross\_entropy\_loss})$.
pub fn perplexity(cross_entropy_loss: f64) -> f64 {
    cross_entropy_loss.exp()
}
