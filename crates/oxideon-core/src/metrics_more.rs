//! Further evaluation metrics: clustering (`adjusted_rand_index`, `normalized_mutual_info`, homogeneity /
//! completeness / V-measure, `silhouette_score`, `davies_bouldin`, `calinski_harabasz`), classification extras
//! (`roc_auc`, `log_loss`, `brier_score`, `expected_calibration_error`, `jaccard_score`, `hamming_loss`), and text
//! metrics (`edit_distance`, `wer`, `cer`, `rouge_n`, `rouge_l`, `token_f1`, `exact_match`).

use crate::error::{Error, Result};
use std::collections::HashMap;

fn same_len(a: usize, b: usize, op: &str) -> Result<()> {
    if a != b || a == 0 {
        return Err(Error::Msg(format!("{op}: inputs must be non-empty and of equal length ({a} vs {b})")));
    }
    Ok(())
}

// ---------------------------------------------------------------- clustering

type Counts = HashMap<usize, f64>;

fn contingency(a: &[usize], b: &[usize]) -> (HashMap<(usize, usize), f64>, Counts, Counts) {
    let (mut joint, mut ra, mut rb) = (HashMap::new(), HashMap::new(), HashMap::new());
    for (&x, &y) in a.iter().zip(b) {
        *joint.entry((x, y)).or_insert(0.0) += 1.0;
        *ra.entry(x).or_insert(0.0) += 1.0;
        *rb.entry(y).or_insert(0.0) += 1.0;
    }
    (joint, ra, rb)
}

fn comb2(n: f64) -> f64 {
    n * (n - 1.0) / 2.0
}

/// Adjusted Rand index between two labelings (1 = identical partitions, ~0 = random).
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> Result<f64> {
    same_len(a.len(), b.len(), "adjusted_rand_index")?;
    let n = a.len() as f64;
    let (joint, ra, rb) = contingency(a, b);
    let sum_joint: f64 = joint.values().map(|&v| comb2(v)).sum();
    let sum_a: f64 = ra.values().map(|&v| comb2(v)).sum();
    let sum_b: f64 = rb.values().map(|&v| comb2(v)).sum();
    let expected = sum_a * sum_b / comb2(n).max(1.0);
    let max = 0.5 * (sum_a + sum_b);
    Ok(if (max - expected).abs() < 1e-12 { 1.0 } else { (sum_joint - expected) / (max - expected) })
}

fn entropy(counts: &HashMap<usize, f64>, n: f64) -> f64 {
    -counts.values().map(|&c| c / n * (c / n).ln()).sum::<f64>()
}

fn mutual_info(a: &[usize], b: &[usize]) -> (f64, f64, f64) {
    let n = a.len() as f64;
    let (joint, ra, rb) = contingency(a, b);
    let mi = joint.iter().map(|(&(x, y), &c)| c / n * ((c * n) / (ra[&x] * rb[&y])).ln()).sum::<f64>();
    (mi, entropy(&ra, n), entropy(&rb, n))
}

/// Normalized mutual information (arithmetic-mean normalization).
pub fn normalized_mutual_info(a: &[usize], b: &[usize]) -> Result<f64> {
    same_len(a.len(), b.len(), "normalized_mutual_info")?;
    let (mi, ha, hb) = mutual_info(a, b);
    let denom = 0.5 * (ha + hb);
    Ok(if denom < 1e-12 { 1.0 } else { mi / denom })
}

/// `(homogeneity, completeness, v_measure)` of `pred` clusters against true `labels`.
pub fn homogeneity_completeness_v(labels: &[usize], pred: &[usize]) -> Result<(f64, f64, f64)> {
    same_len(labels.len(), pred.len(), "homogeneity_completeness_v")?;
    let (mi, h_labels, h_pred) = mutual_info(labels, pred);
    let h = if h_labels < 1e-12 { 1.0 } else { mi / h_labels };
    let c = if h_pred < 1e-12 { 1.0 } else { mi / h_pred };
    let v = if h + c < 1e-12 { 0.0 } else { 2.0 * h * c / (h + c) };
    Ok((h, c, v))
}

fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Mean silhouette coefficient of `points` (rows) under `labels`; needs at least two clusters.
pub fn silhouette_score(points: &[Vec<f64>], labels: &[usize]) -> Result<f64> {
    same_len(points.len(), labels.len(), "silhouette_score")?;
    let k = labels.iter().max().unwrap() + 1;
    if (0..k).filter(|c| labels.contains(c)).count() < 2 {
        return Err(Error::Msg("silhouette_score: needs at least two clusters".into()));
    }
    let mut total = 0.0;
    for i in 0..points.len() {
        let (mut sum, mut cnt) = (vec![0.0; k], vec![0usize; k]);
        for j in 0..points.len() {
            if i != j {
                sum[labels[j]] += dist(&points[i], &points[j]);
                cnt[labels[j]] += 1;
            }
        }
        if cnt[labels[i]] == 0 {
            continue; // singleton clusters score 0
        }
        let a = sum[labels[i]] / cnt[labels[i]] as f64;
        let b = (0..k).filter(|&c| c != labels[i] && cnt[c] > 0).map(|c| sum[c] / cnt[c] as f64).fold(f64::INFINITY, f64::min);
        total += (b - a) / a.max(b).max(1e-300);
    }
    Ok(total / points.len() as f64)
}

fn centroids(points: &[Vec<f64>], labels: &[usize]) -> (Vec<Vec<f64>>, Vec<usize>) {
    let k = labels.iter().max().unwrap() + 1;
    let d = points[0].len();
    let (mut c, mut n) = (vec![vec![0.0; d]; k], vec![0usize; k]);
    for (p, &l) in points.iter().zip(labels) {
        n[l] += 1;
        c[l].iter_mut().zip(p).for_each(|(s, v)| *s += v);
    }
    for (row, &cnt) in c.iter_mut().zip(&n) {
        row.iter_mut().for_each(|v| *v /= cnt.max(1) as f64);
    }
    (c, n)
}

/// Davies-Bouldin index (lower is better).
pub fn davies_bouldin(points: &[Vec<f64>], labels: &[usize]) -> Result<f64> {
    same_len(points.len(), labels.len(), "davies_bouldin")?;
    let (c, n) = centroids(points, labels);
    let live: Vec<usize> = (0..c.len()).filter(|&i| n[i] > 0).collect();
    if live.len() < 2 {
        return Err(Error::Msg("davies_bouldin: needs at least two clusters".into()));
    }
    let mut s = vec![0.0; c.len()];
    for (p, &l) in points.iter().zip(labels) {
        s[l] += dist(p, &c[l]) / n[l] as f64;
    }
    let total: f64 = live.iter().map(|&i| live.iter().filter(|&&j| j != i).map(|&j| (s[i] + s[j]) / dist(&c[i], &c[j]).max(1e-300)).fold(0.0, f64::max)).sum();
    Ok(total / live.len() as f64)
}

/// Calinski-Harabasz (variance ratio) index (higher is better).
pub fn calinski_harabasz(points: &[Vec<f64>], labels: &[usize]) -> Result<f64> {
    same_len(points.len(), labels.len(), "calinski_harabasz")?;
    let (c, n) = centroids(points, labels);
    let live = n.iter().filter(|&&x| x > 0).count();
    let total = points.len();
    if live < 2 || live >= total {
        return Err(Error::Msg("calinski_harabasz: needs 2..n-1 clusters".into()));
    }
    let mean: Vec<f64> = (0..points[0].len()).map(|d| points.iter().map(|p| p[d]).sum::<f64>() / total as f64).collect();
    let between: f64 = (0..c.len()).map(|i| n[i] as f64 * dist(&c[i], &mean).powi(2)).sum();
    let within: f64 = points.iter().zip(labels).map(|(p, &l)| dist(p, &c[l]).powi(2)).sum();
    Ok(between / (live - 1) as f64 / (within / (total - live) as f64).max(1e-300))
}

// ---------------------------------------------------------------- classification extras

/// Area under the ROC curve via the rank-sum (Mann-Whitney) statistic; ties get half credit.
pub fn roc_auc(scores: &[f64], targets: &[usize]) -> Result<f64> {
    same_len(scores.len(), targets.len(), "roc_auc")?;
    let (mut pos, mut neg) = (0.0, 0.0);
    let mut wins = 0.0;
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut i = 0;
    let mut neg_seen = 0.0;
    while i < order.len() {
        let mut j = i;
        while j < order.len() && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        let group_pos = order[i..j].iter().filter(|&&k| targets[k] == 1).count() as f64;
        let group_neg = (j - i) as f64 - group_pos;
        wins += group_pos * (neg_seen + 0.5 * group_neg);
        pos += group_pos;
        neg += group_neg;
        neg_seen += group_neg;
        i = j;
    }
    if pos == 0.0 || neg == 0.0 {
        return Err(Error::Msg("roc_auc: both classes must be present".into()));
    }
    Ok(wins / (pos * neg))
}

/// Binary cross-entropy of probabilities `p` (clipped to avoid infinities).
pub fn log_loss(p: &[f64], targets: &[usize]) -> Result<f64> {
    same_len(p.len(), targets.len(), "log_loss")?;
    Ok(-p.iter().zip(targets).map(|(&p, &t)| {
        let p = p.clamp(1e-15, 1.0 - 1e-15);
        if t == 1 { p.ln() } else { (1.0 - p).ln() }
    }).sum::<f64>() / p.len() as f64)
}

/// Mean squared error between probabilities and 0/1 outcomes.
pub fn brier_score(p: &[f64], targets: &[usize]) -> Result<f64> {
    same_len(p.len(), targets.len(), "brier_score")?;
    Ok(p.iter().zip(targets).map(|(&p, &t)| (p - t as f64).powi(2)).sum::<f64>() / p.len() as f64)
}

/// Expected calibration error with `bins` equal-width confidence bins (`p` = predicted probability of class 1).
pub fn expected_calibration_error(p: &[f64], targets: &[usize], bins: usize) -> Result<f64> {
    same_len(p.len(), targets.len(), "expected_calibration_error")?;
    let (mut conf, mut hit, mut cnt) = (vec![0.0; bins], vec![0.0; bins], vec![0.0; bins]);
    for (&pi, &t) in p.iter().zip(targets) {
        let b = ((pi * bins as f64) as usize).min(bins - 1);
        conf[b] += pi;
        hit[b] += t as f64;
        cnt[b] += 1.0;
    }
    let n = p.len() as f64;
    Ok((0..bins).filter(|&b| cnt[b] > 0.0).map(|b| cnt[b] / n * (conf[b] / cnt[b] - hit[b] / cnt[b]).abs()).sum())
}

/// Jaccard index of binary predictions for the positive class.
pub fn jaccard_score(preds: &[usize], targets: &[usize]) -> Result<f64> {
    same_len(preds.len(), targets.len(), "jaccard_score")?;
    let inter = preds.iter().zip(targets).filter(|(p, t)| **p == 1 && **t == 1).count() as f64;
    let union = preds.iter().zip(targets).filter(|(p, t)| **p == 1 || **t == 1).count() as f64;
    Ok(if union == 0.0 { 1.0 } else { inter / union })
}

/// Fraction of positions where the labels differ.
pub fn hamming_loss(preds: &[usize], targets: &[usize]) -> Result<f64> {
    same_len(preds.len(), targets.len(), "hamming_loss")?;
    Ok(preds.iter().zip(targets).filter(|(p, t)| p != t).count() as f64 / preds.len() as f64)
}

// ---------------------------------------------------------------- text

/// Levenshtein distance between two sequences.
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, y) in b.iter().enumerate() {
            cur.push((prev[j] + usize::from(x != y)).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Word error rate: word-level edit distance divided by the reference length.
pub fn wer(reference: &str, hypothesis: &str) -> Result<f64> {
    let (r, h): (Vec<&str>, Vec<&str>) = (reference.split_whitespace().collect(), hypothesis.split_whitespace().collect());
    if r.is_empty() {
        return Err(Error::Msg("wer: empty reference".into()));
    }
    Ok(edit_distance(&r, &h) as f64 / r.len() as f64)
}

/// Character error rate.
pub fn cer(reference: &str, hypothesis: &str) -> Result<f64> {
    let (r, h): (Vec<char>, Vec<char>) = (reference.chars().collect(), hypothesis.chars().collect());
    if r.is_empty() {
        return Err(Error::Msg("cer: empty reference".into()));
    }
    Ok(edit_distance(&r, &h) as f64 / r.len() as f64)
}

fn ngram_counts(t: &[&str], n: usize) -> HashMap<Vec<String>, usize> {
    let mut m = HashMap::new();
    for w in t.windows(n) {
        *m.entry(w.iter().map(|s| s.to_string()).collect()).or_insert(0) += 1;
    }
    m
}

fn f_measure(p: f64, r: f64) -> f64 {
    if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
}

/// ROUGE-N `(precision, recall, f1)` over whitespace tokens.
pub fn rouge_n(reference: &str, candidate: &str, n: usize) -> (f64, f64, f64) {
    let (r, c): (Vec<&str>, Vec<&str>) = (reference.split_whitespace().collect(), candidate.split_whitespace().collect());
    let (rc, cc) = (ngram_counts(&r, n), ngram_counts(&c, n));
    let overlap: usize = cc.iter().map(|(g, &k)| k.min(*rc.get(g).unwrap_or(&0))).sum();
    let (rt, ct): (usize, usize) = (rc.values().sum(), cc.values().sum());
    let p = if ct == 0 { 0.0 } else { overlap as f64 / ct as f64 };
    let rec = if rt == 0 { 0.0 } else { overlap as f64 / rt as f64 };
    (p, rec, f_measure(p, rec))
}

/// ROUGE-L `(precision, recall, f1)` from the longest common subsequence of whitespace tokens.
pub fn rouge_l(reference: &str, candidate: &str) -> (f64, f64, f64) {
    let (r, c): (Vec<&str>, Vec<&str>) = (reference.split_whitespace().collect(), candidate.split_whitespace().collect());
    let mut dp = vec![vec![0usize; c.len() + 1]; r.len() + 1];
    for i in 0..r.len() {
        for j in 0..c.len() {
            dp[i + 1][j + 1] = if r[i] == c[j] { dp[i][j] + 1 } else { dp[i][j + 1].max(dp[i + 1][j]) };
        }
    }
    let lcs = dp[r.len()][c.len()] as f64;
    let p = if c.is_empty() { 0.0 } else { lcs / c.len() as f64 };
    let rec = if r.is_empty() { 0.0 } else { lcs / r.len() as f64 };
    (p, rec, f_measure(p, rec))
}

/// SQuAD-style token F1 between an answer and a reference (bag of whitespace tokens).
pub fn token_f1(reference: &str, candidate: &str) -> f64 {
    rouge_n(reference, candidate, 1).2
}

/// Exact match after trimming and lower-casing.
pub fn exact_match(reference: &str, candidate: &str) -> bool {
    reference.trim().to_lowercase() == candidate.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }

    #[test]
    fn clustering_agreement() {
        let a = [0, 0, 1, 1, 2, 2];
        close(adjusted_rand_index(&a, &[5, 5, 7, 7, 9, 9]).unwrap(), 1.0); // relabelling does not matter
        close(normalized_mutual_info(&a, &[1, 1, 0, 0, 2, 2]).unwrap(), 1.0);
        // sklearn: adjusted_rand_score([0,0,1,1],[0,0,1,2]) = 0.5714285714
        close(adjusted_rand_index(&[0, 0, 1, 1], &[0, 0, 1, 2]).unwrap(), 4.0 / 7.0);
        let (h, c, v) = homogeneity_completeness_v(&[0, 0, 1, 1], &[0, 0, 1, 2]).unwrap();
        close(h, 1.0); // every predicted cluster is pure
        assert!(c < 1.0 && v > c && v < 1.0);
        assert!(adjusted_rand_index(&[], &[]).is_err());
    }

    #[test]
    fn cluster_quality_indices() {
        let pts: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![10.0, 10.0], vec![10.0, 11.0]];
        let good = [0, 0, 1, 1];
        let bad = [0, 1, 0, 1];
        let s_good = silhouette_score(&pts, &good).unwrap();
        assert!(s_good > 0.9 && s_good < silhouette_score(&pts, &good).unwrap() + 1e-12);
        assert!(silhouette_score(&pts, &bad).unwrap() < 0.0);
        assert!(davies_bouldin(&pts, &good).unwrap() < davies_bouldin(&pts, &bad).unwrap());
        assert!(calinski_harabasz(&pts, &good).unwrap() > calinski_harabasz(&pts, &bad).unwrap());
        assert!(silhouette_score(&pts, &[0, 0, 0, 0]).is_err());
    }

    #[test]
    fn classification_extras() {
        // sklearn roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8]) = 0.75
        close(roc_auc(&[0.1, 0.4, 0.35, 0.8], &[0, 0, 1, 1]).unwrap(), 0.75);
        close(roc_auc(&[0.5, 0.5], &[0, 1]).unwrap(), 0.5);
        assert!(roc_auc(&[0.1, 0.2], &[1, 1]).is_err());
        close(log_loss(&[0.9, 0.1], &[1, 0]).unwrap(), -(0.9f64).ln());
        close(brier_score(&[0.9, 0.2], &[1, 0]).unwrap(), (0.01 + 0.04) / 2.0);
        close(expected_calibration_error(&[1.0, 1.0, 0.0, 0.0], &[1, 1, 0, 0], 10).unwrap(), 0.0);
        close(expected_calibration_error(&[0.9, 0.9], &[0, 0], 10).unwrap(), 0.9);
        close(jaccard_score(&[1, 1, 0], &[1, 0, 1]).unwrap(), 1.0 / 3.0);
        close(hamming_loss(&[1, 0, 1, 1], &[1, 1, 1, 0]).unwrap(), 0.5);
    }

    #[test]
    fn text_metrics() {
        assert_eq!(edit_distance(&['k', 'i', 't', 't', 'e', 'n'], &['s', 'i', 't', 't', 'i', 'n', 'g']), 3);
        close(wer("the cat sat on the mat", "the cat sit on mat").unwrap(), 2.0 / 6.0);
        close(cer("abcd", "abxd").unwrap(), 0.25);
        assert!(wer("", "x").is_err());
        let (p, r, f) = rouge_n("the cat sat", "the cat ran", 1);
        close(p, 2.0 / 3.0);
        close(r, 2.0 / 3.0);
        close(f, 2.0 / 3.0);
        let (_, r2, _) = rouge_n("a b c d", "a b x d", 2);
        close(r2, 1.0 / 3.0); // only "a b" overlaps out of three reference bigrams
        let (p, r, _) = rouge_l("police killed the gunman", "police kill the gunman");
        close(p, 0.75);
        close(r, 0.75);
        close(token_f1("the quick fox", "quick brown fox"), 2.0 / 3.0);
        assert!(exact_match(" Paris ", "paris") && !exact_match("a", "b"));
    }
}
