//! More evaluation metrics (after `sklearn.metrics` / `torchmetrics`): averaged precision / recall / F1
//! ([`prf_average`]), [`balanced_accuracy`], Matthews correlation ([`matthews_corrcoef`]), [`cohen_kappa`],
//! ROC and precision-recall curves with [`average_precision`], ranking metrics ([`ndcg_at_k`],
//! [`mean_reciprocal_rank`]), regression metrics ([`mae`], [`mape`], [`explained_variance`], [`median_absolute_error`])
//! and corpus [`bleu`] on token sequences.

use crate::error::{Error, Result};
use crate::metrics::confusion_matrix;
use std::collections::HashMap;

/// How per-class scores are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Average {
    /// Unweighted mean over classes.
    Macro,
    /// Pool all decisions first (precision = recall = F1 = accuracy for single-label problems).
    Micro,
    /// Mean weighted by class support.
    Weighted,
}

/// Precision, recall and F1 averaged as requested.
pub fn prf_average(preds: &[usize], targets: &[usize], num_classes: usize, avg: Average) -> Result<(f64, f64, f64)> {
    if preds.len() != targets.len() {
        return Err(Error::Msg("prf_average: preds and targets must have the same length".into()));
    }
    let cm = confusion_matrix(preds, targets, num_classes)?;
    let total: usize = cm.iter().flatten().sum();
    let tp_all: usize = (0..num_classes).map(|c| cm[c][c]).sum();
    if avg == Average::Micro {
        let v = if total > 0 { tp_all as f64 / total as f64 } else { 0.0 };
        return Ok((v, v, v));
    }
    let (mut p_sum, mut r_sum, mut f_sum, mut wsum) = (0.0, 0.0, 0.0, 0.0);
    for c in 0..num_classes {
        let tp = cm[c][c] as f64;
        let fp: f64 = (0..num_classes).filter(|&r| r != c).map(|r| cm[r][c] as f64).sum();
        let support: f64 = cm[c].iter().sum::<usize>() as f64;
        let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let r = if support > 0.0 { tp / support } else { 0.0 };
        let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        let w = if avg == Average::Weighted { support } else { 1.0 };
        p_sum += w * p;
        r_sum += w * r;
        f_sum += w * f;
        wsum += w;
    }
    if wsum == 0.0 {
        return Ok((0.0, 0.0, 0.0));
    }
    Ok((p_sum / wsum, r_sum / wsum, f_sum / wsum))
}

/// Mean per-class recall.
pub fn balanced_accuracy(preds: &[usize], targets: &[usize], num_classes: usize) -> Result<f64> {
    let cm = confusion_matrix(preds, targets, num_classes)?;
    let recalls: Vec<f64> = (0..num_classes)
        .filter_map(|c| {
            let s: usize = cm[c].iter().sum();
            (s > 0).then(|| cm[c][c] as f64 / s as f64)
        })
        .collect();
    Ok(if recalls.is_empty() { 0.0 } else { recalls.iter().sum::<f64>() / recalls.len() as f64 })
}

/// Matthews correlation coefficient (multi-class generalization), `0` when undefined.
pub fn matthews_corrcoef(preds: &[usize], targets: &[usize], num_classes: usize) -> Result<f64> {
    let cm = confusion_matrix(preds, targets, num_classes)?;
    let t: Vec<f64> = (0..num_classes).map(|c| cm[c].iter().sum::<usize>() as f64).collect(); // true counts
    let p: Vec<f64> = (0..num_classes).map(|c| (0..num_classes).map(|r| cm[r][c]).sum::<usize>() as f64).collect();
    let n: f64 = t.iter().sum();
    let correct: f64 = (0..num_classes).map(|c| cm[c][c] as f64).sum();
    let cov_tp = correct * n - t.iter().zip(&p).map(|(a, b)| a * b).sum::<f64>();
    let cov_pp = n * n - p.iter().map(|a| a * a).sum::<f64>();
    let cov_tt = n * n - t.iter().map(|a| a * a).sum::<f64>();
    let denom = (cov_pp * cov_tt).sqrt();
    Ok(if denom > 0.0 { cov_tp / denom } else { 0.0 })
}

/// Cohen's kappa agreement between predictions and targets.
pub fn cohen_kappa(preds: &[usize], targets: &[usize], num_classes: usize) -> Result<f64> {
    let cm = confusion_matrix(preds, targets, num_classes)?;
    let n: f64 = cm.iter().flatten().sum::<usize>() as f64;
    if n == 0.0 {
        return Ok(0.0);
    }
    let po: f64 = (0..num_classes).map(|c| cm[c][c] as f64).sum::<f64>() / n;
    let pe: f64 = (0..num_classes)
        .map(|c| {
            let row: f64 = cm[c].iter().sum::<usize>() as f64;
            let col: f64 = (0..num_classes).map(|r| cm[r][c]).sum::<usize>() as f64;
            row * col
        })
        .sum::<f64>()
        / (n * n);
    Ok(if (1.0 - pe).abs() < 1e-15 { 1.0 } else { (po - pe) / (1.0 - pe) })
}

/// ROC curve points `(fpr, tpr, thresholds)` for binary scores, thresholds in decreasing order starting above the
/// maximum score (first point `(0, 0)`).
pub fn roc_curve(scores: &[f64], targets: &[usize]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if scores.len() != targets.len() {
        return Err(Error::Msg("roc_curve: scores and targets must have the same length".into()));
    }
    let pos = targets.iter().filter(|&&t| t == 1).count() as f64;
    let neg = targets.len() as f64 - pos;
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let (mut fpr, mut tpr, mut thr) = (vec![0.0], vec![0.0], vec![f64::INFINITY]);
    let (mut tp, mut fp) = (0.0, 0.0);
    let mut i = 0;
    while i < idx.len() {
        let s = scores[idx[i]];
        while i < idx.len() && scores[idx[i]] == s {
            if targets[idx[i]] == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        fpr.push(if neg > 0.0 { fp / neg } else { 0.0 });
        tpr.push(if pos > 0.0 { tp / pos } else { 0.0 });
        thr.push(s);
    }
    Ok((fpr, tpr, thr))
}

/// Precision-recall curve `(precision, recall, thresholds)` with recall increasing; the final point is `(1, 0)`.
pub fn precision_recall_curve(scores: &[f64], targets: &[usize]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if scores.len() != targets.len() {
        return Err(Error::Msg("precision_recall_curve: scores and targets must have the same length".into()));
    }
    let pos = targets.iter().filter(|&&t| t == 1).count() as f64;
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let (mut prec, mut rec, mut thr) = (Vec::new(), Vec::new(), Vec::new());
    let (mut tp, mut seen) = (0.0, 0.0);
    let mut i = 0;
    while i < idx.len() {
        let s = scores[idx[i]];
        while i < idx.len() && scores[idx[i]] == s {
            if targets[idx[i]] == 1 {
                tp += 1.0;
            }
            seen += 1.0;
            i += 1;
        }
        prec.push(tp / seen);
        rec.push(if pos > 0.0 { tp / pos } else { 0.0 });
        thr.push(s);
    }
    prec.reverse();
    rec.reverse();
    thr.reverse();
    prec.push(1.0);
    rec.push(0.0);
    Ok((prec, rec, thr))
}

/// Average precision: the area under the precision-recall step function, `sum (R_n - R_{n-1}) P_n`.
pub fn average_precision(scores: &[f64], targets: &[usize]) -> Result<f64> {
    let (prec, rec, _) = precision_recall_curve(scores, targets)?;
    // recall decreases along the returned arrays
    Ok((0..prec.len() - 1).map(|i| (rec[i] - rec[i + 1]) * prec[i]).sum())
}

/// Normalized discounted cumulative gain at rank `k` for one query: `relevance` in the order the model ranked the
/// items.
pub fn ndcg_at_k(relevance: &[f64], k: usize) -> f64 {
    let dcg = |r: &[f64]| r.iter().take(k).enumerate().map(|(i, &g)| (2f64.powf(g) - 1.0) / ((i + 2) as f64).log2()).sum::<f64>();
    let mut ideal = relevance.to_vec();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let best = dcg(&ideal);
    if best == 0.0 { 0.0 } else { dcg(relevance) / best }
}

/// Mean reciprocal rank: for each query the 1-based rank of the first relevant item (`relevance[i] > 0`).
pub fn mean_reciprocal_rank(queries: &[Vec<f64>]) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }
    queries
        .iter()
        .map(|q| q.iter().position(|&r| r > 0.0).map_or(0.0, |p| 1.0 / (p + 1) as f64))
        .sum::<f64>()
        / queries.len() as f64
}

fn pair_check(pred: &[f64], target: &[f64], op: &str) -> Result<()> {
    if pred.len() != target.len() || pred.is_empty() {
        return Err(Error::Msg(format!("{op}: pred and target must be non-empty and equally long")));
    }
    Ok(())
}

pub fn mae(pred: &[f64], target: &[f64]) -> Result<f64> {
    pair_check(pred, target, "mae")?;
    Ok(pred.iter().zip(target).map(|(a, b)| (a - b).abs()).sum::<f64>() / pred.len() as f64)
}

/// Mean absolute percentage error (targets of zero are skipped).
pub fn mape(pred: &[f64], target: &[f64]) -> Result<f64> {
    pair_check(pred, target, "mape")?;
    let terms: Vec<f64> = pred.iter().zip(target).filter(|(_, t)| **t != 0.0).map(|(p, t)| ((p - t) / t).abs()).collect();
    Ok(if terms.is_empty() { 0.0 } else { terms.iter().sum::<f64>() / terms.len() as f64 })
}

pub fn median_absolute_error(pred: &[f64], target: &[f64]) -> Result<f64> {
    pair_check(pred, target, "median_absolute_error")?;
    let mut e: Vec<f64> = pred.iter().zip(target).map(|(a, b)| (a - b).abs()).collect();
    e.sort_by(f64::total_cmp);
    let n = e.len();
    Ok(if n % 2 == 1 { e[n / 2] } else { 0.5 * (e[n / 2 - 1] + e[n / 2]) })
}

/// `1 - Var(target - pred) / Var(target)`.
pub fn explained_variance(pred: &[f64], target: &[f64]) -> Result<f64> {
    pair_check(pred, target, "explained_variance")?;
    let var = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
    };
    let resid: Vec<f64> = target.iter().zip(pred).map(|(t, p)| t - p).collect();
    let vt = var(target);
    Ok(if vt == 0.0 { 0.0 } else { 1.0 - var(&resid) / vt })
}

/// Corpus BLEU with uniform weights up to `max_n`-grams and the standard brevity penalty. `candidates[i]` is scored
/// against all of `references[i]`. Smoothing adds one to the numerator and denominator of n-gram orders above one
/// when `smooth` is set (Lin & Och, method 1).
pub fn bleu(candidates: &[Vec<usize>], references: &[Vec<Vec<usize>>], max_n: usize, smooth: bool) -> Result<f64> {
    if candidates.len() != references.len() || candidates.is_empty() {
        return Err(Error::Msg("bleu: need one reference set per candidate".into()));
    }
    let ngrams = |s: &[usize], n: usize| -> HashMap<Vec<usize>, usize> {
        let mut m = HashMap::new();
        for w in s.windows(n) {
            *m.entry(w.to_vec()).or_insert(0) += 1;
        }
        m
    };
    let (mut clipped, mut total) = (vec![0usize; max_n], vec![0usize; max_n]);
    let (mut cand_len, mut ref_len) = (0usize, 0usize);
    for (cand, refs) in candidates.iter().zip(references) {
        cand_len += cand.len();
        // the reference closest in length (ties: the shorter one)
        ref_len += refs
            .iter()
            .map(Vec::len)
            .min_by_key(|&l| ((l as i64 - cand.len() as i64).abs(), l))
            .ok_or_else(|| Error::Msg("bleu: empty reference set".into()))?;
        for n in 1..=max_n {
            let c = ngrams(cand, n);
            let mut max_ref: HashMap<Vec<usize>, usize> = HashMap::new();
            for r in refs {
                for (g, cnt) in ngrams(r, n) {
                    let e = max_ref.entry(g).or_insert(0);
                    *e = (*e).max(cnt);
                }
            }
            for (g, cnt) in &c {
                clipped[n - 1] += (*cnt).min(*max_ref.get(g).unwrap_or(&0));
            }
            total[n - 1] += c.values().sum::<usize>();
        }
    }
    let mut log_p = 0.0;
    for n in 0..max_n {
        let (num, den) = if smooth && n > 0 { (clipped[n] as f64 + 1.0, total[n] as f64 + 1.0) } else { (clipped[n] as f64, total[n] as f64) };
        if num == 0.0 || den == 0.0 {
            return Ok(0.0);
        }
        log_p += (num / den).ln() / max_n as f64;
    }
    let bp = if cand_len >= ref_len || cand_len == 0 { 1.0 } else { (1.0 - ref_len as f64 / cand_len as f64).exp() };
    Ok(bp * log_p.exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn classification_summaries() {
        let preds = [0, 0, 1, 1, 1, 2, 2, 0];
        let targets = [0, 0, 0, 1, 1, 2, 1, 2];
        // per class: c0 tp2 fp0 fn1 (p 1, r 2/3); c1 tp2 fp1 fn1 (p 2/3, r 2/3); c2 tp1 fp1 fn1... compute directly
        let (p, r, f) = prf_average(&preds, &targets, 3, Average::Macro).unwrap();
        let cm = confusion_matrix(&preds, &targets, 3).unwrap();
        assert_eq!(cm[0], vec![2, 1, 0]);
        let prec = [2.0 / 3.0, 2.0 / 3.0, 1.0 / 2.0];
        let rec = [2.0 / 3.0, 2.0 / 3.0, 1.0 / 2.0];
        near(p, prec.iter().sum::<f64>() / 3.0);
        near(r, rec.iter().sum::<f64>() / 3.0);
        let fs: Vec<f64> = prec.iter().zip(&rec).map(|(p, r)| 2.0 * p * r / (p + r)).collect();
        near(f, fs.iter().sum::<f64>() / 3.0);
        let (mp, mr, mf) = prf_average(&preds, &targets, 3, Average::Micro).unwrap();
        assert!(mp == mr && mr == mf);
        near(mp, 5.0 / 8.0);
        let (_, wr, _) = prf_average(&preds, &targets, 3, Average::Weighted).unwrap();
        near(wr, 5.0 / 8.0); // support-weighted recall equals accuracy
        near(balanced_accuracy(&preds, &targets, 3).unwrap(), rec.iter().sum::<f64>() / 3.0);
        // perfect and chance-level agreement
        near(matthews_corrcoef(&[0, 1, 0, 1], &[0, 1, 0, 1], 2).unwrap(), 1.0);
        near(matthews_corrcoef(&[0, 0, 1, 1], &[0, 1, 0, 1], 2).unwrap(), 0.0);
        near(cohen_kappa(&[0, 1, 0, 1], &[0, 1, 0, 1], 2).unwrap(), 1.0);
        near(cohen_kappa(&[0, 0, 1, 1], &[0, 1, 0, 1], 2).unwrap(), 0.0);
        // binary MCC from the textbook formula: tp 3 tn 4 fp 1 fn 2
        let t: Vec<usize> = [vec![1; 5], vec![0; 5]].concat();
        let p_: Vec<usize> = [vec![1, 1, 1, 0, 0], vec![0, 0, 0, 0, 1]].concat();
        near(matthews_corrcoef(&p_, &t, 2).unwrap(), (3.0 * 4.0 - 1.0 * 2.0) / (4.0f64 * 5.0 * 5.0 * 6.0).sqrt());
    }

    #[test]
    fn curves() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.4];
        let targets = [1, 1, 0, 1, 0, 0];
        let (fpr, tpr, thr) = roc_curve(&scores, &targets).unwrap();
        assert_eq!(thr[0], f64::INFINITY);
        assert_eq!((fpr[0], tpr[0]), (0.0, 0.0));
        assert_eq!((*fpr.last().unwrap(), *tpr.last().unwrap()), (1.0, 1.0));
        // trapezoid area equals the Mann-Whitney AUROC: 8 of 9 positive-negative pairs are ordered correctly
        let auc: f64 = (1..fpr.len()).map(|i| (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0).sum();
        near(auc, 8.0 / 9.0);
        near(auc, crate::metrics::auroc(&scores, &targets));
        // average precision: hits at ranks 1, 2 and 4 -> (1 + 1 + 3/4) / 3
        near(average_precision(&scores, &targets).unwrap(), (1.0 + 1.0 + 0.75) / 3.0);
        let (p, r, _) = precision_recall_curve(&scores, &targets).unwrap();
        assert_eq!((*p.last().unwrap(), *r.last().unwrap()), (1.0, 0.0));
        near(average_precision(&[0.1, 0.2], &[1, 0]).unwrap(), 0.5);
    }

    #[test]
    fn ranking_and_regression() {
        // gains 3, 2, 3, 0, 1, 2 (SIGIR example, with 2^g - 1 gains): compare with the direct formula
        let rel = [3.0, 2.0, 3.0, 0.0, 1.0, 2.0];
        let dcg: f64 = rel.iter().enumerate().map(|(i, g)| (2f64.powf(*g) - 1.0) / ((i + 2) as f64).log2()).sum();
        let ideal = [3.0, 3.0, 2.0, 2.0, 1.0, 0.0];
        let idcg: f64 = ideal.iter().enumerate().map(|(i, g)| (2f64.powf(*g) - 1.0) / ((i + 2) as f64).log2()).sum();
        near(ndcg_at_k(&rel, 6), dcg / idcg);
        near(ndcg_at_k(&ideal, 3), 1.0);
        near(ndcg_at_k(&[0.0, 0.0], 2), 0.0);
        near(mean_reciprocal_rank(&[vec![0.0, 1.0], vec![1.0, 0.0], vec![0.0, 0.0]]), (0.5 + 1.0 + 0.0) / 3.0);
        let (p, t) = ([2.0, 4.0, 7.0, 1.0], [3.0, 4.0, 5.0, 2.0]);
        near(mae(&p, &t).unwrap(), 1.0);
        near(median_absolute_error(&p, &t).unwrap(), 1.0);
        near(mape(&p, &t).unwrap(), (1.0 / 3.0 + 0.0 + 0.4 + 0.5) / 4.0);
        near(explained_variance(&t, &t).unwrap(), 1.0);
        assert!(explained_variance(&p, &t).unwrap() < 1.0);
        assert!(mae(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn bleu_scores() {
        let cand = vec![vec![1, 2, 3, 4, 5, 6, 7, 8]];
        let same = vec![vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]];
        near(bleu(&cand, &same, 4, false).unwrap(), 1.0);
        // no 4-gram overlap -> 0 unless smoothed
        let other = vec![vec![vec![1, 2, 9, 4, 5, 9, 7, 8]]];
        assert_eq!(bleu(&cand, &other, 4, false).unwrap(), 0.0);
        assert!(bleu(&cand, &other, 4, true).unwrap() > 0.0);
        // brevity penalty: a candidate half the reference length
        let short = vec![vec![1, 2, 3, 4]];
        let v = bleu(&short, &same, 2, false).unwrap();
        near(v, (1.0f64 - 2.0).exp());
        // clipping: repeating a reference word does not help
        let rep = vec![vec![1, 1, 1, 1]];
        let r = bleu(&rep, &[vec![vec![1, 2, 3, 4]]], 1, false).unwrap();
        near(r, 0.25);
    }
}
