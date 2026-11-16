//! Outlier / anomaly detection on `Vec<Vec<f64>>`: [`IsolationForest`], [`lof_scores`] (local outlier factor),
//! [`mahalanobis_scores`], [`zscore_outliers`] / [`iqr_outliers`], and a one-class [`Cusum`] change detector.

use crate::classical::Matrix;
use crate::error::{Error, Result};
use crate::rng::Rng;

fn rect(x: &Matrix, op: &str) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    if x.len() < 3 || d == 0 || x.iter().any(|r| r.len() != d || r.iter().any(|v| !v.is_finite())) {
        return Err(Error::Msg(format!("{op}: need at least 3 rows of finite, equal-length data")));
    }
    Ok(d)
}

fn c_factor(n: f64) -> f64 {
    if n <= 1.0 { 0.0 } else if n == 2.0 { 1.0 } else { 2.0 * ((n - 1.0).ln() + 0.577_215_664_9) - 2.0 * (n - 1.0) / n }
}

#[derive(Debug, Clone)]
enum Node {
    Leaf(usize),
    Split { feature: usize, threshold: f64, left: Box<Node>, right: Box<Node> },
}

fn build(x: &Matrix, idx: &[usize], depth: usize, limit: usize, rng: &mut Rng) -> Node {
    if idx.len() <= 1 || depth >= limit {
        return Node::Leaf(idx.len());
    }
    let d = x[0].len();
    // pick a feature with spread; give up after a few tries on constant ones
    for _ in 0..d.max(3) {
        let f = (rng.next_u64() % d as u64) as usize;
        let (lo, hi) = idx.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, &i| (a.0.min(x[i][f]), a.1.max(x[i][f])));
        if hi > lo {
            let th = lo + (hi - lo) * rng.next_f64();
            let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[i][f] < th);
            if l.is_empty() || r.is_empty() {
                continue;
            }
            return Node::Split { feature: f, threshold: th, left: Box::new(build(x, &l, depth + 1, limit, rng)), right: Box::new(build(x, &r, depth + 1, limit, rng)) };
        }
    }
    Node::Leaf(idx.len())
}

fn path_len(n: &Node, p: &[f64], depth: f64) -> f64 {
    match n {
        Node::Leaf(size) => depth + c_factor(*size as f64),
        Node::Split { feature, threshold, left, right } => path_len(if p[*feature] < *threshold { left } else { right }, p, depth + 1.0),
    }
}

/// Isolation forest (Liu et al. 2008): anomalies are isolated by few random splits.
#[derive(Debug, Clone)]
pub struct IsolationForest {
    trees: Vec<Node>,
    sample_size: f64,
}

impl IsolationForest {
    pub fn fit(x: &Matrix, n_trees: usize, sample_size: usize, seed: u64) -> Result<Self> {
        rect(x, "IsolationForest")?;
        if n_trees == 0 || sample_size < 2 {
            return Err(Error::Msg("IsolationForest: need n_trees >= 1 and sample_size >= 2".into()));
        }
        let ss = sample_size.min(x.len());
        let limit = (ss as f64).log2().ceil() as usize;
        let mut rng = Rng::new(seed);
        let trees = (0..n_trees)
            .map(|_| {
                let mut all: Vec<usize> = (0..x.len()).collect();
                rng.shuffle(&mut all);
                all.truncate(ss);
                build(x, &all, 0, limit, &mut rng)
            })
            .collect();
        Ok(Self { trees, sample_size: ss as f64 })
    }

    /// Anomaly score in `(0, 1)`: close to 1 for anomalies, around 0.5 or below for normal points.
    pub fn score(&self, x: &Matrix) -> Vec<f64> {
        x.iter().map(|p| { let avg = self.trees.iter().map(|t| path_len(t, p, 0.0)).sum::<f64>() / self.trees.len() as f64; 2f64.powf(-avg / c_factor(self.sample_size).max(1e-12)) }).collect()
    }
}

/// Local outlier factor with `k` neighbours (values well above 1 mark outliers).
pub fn lof_scores(x: &Matrix, k: usize) -> Result<Vec<f64>> {
    rect(x, "lof_scores")?;
    let n = x.len();
    if k == 0 || k >= n {
        return Err(Error::Msg("lof_scores: need 1 <= k < n".into()));
    }
    let dist = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt();
    let mut neigh: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut d: Vec<(usize, f64)> = (0..n).filter(|&j| j != i).map(|j| (j, dist(&x[i], &x[j]))).collect();
        d.sort_by(|a, b| a.1.total_cmp(&b.1));
        d.truncate(k);
        neigh.push(d);
    }
    let kdist: Vec<f64> = neigh.iter().map(|n| n.last().unwrap().1).collect();
    let lrd: Vec<f64> = neigh.iter().map(|nb| { let s: f64 = nb.iter().map(|&(j, d)| d.max(kdist[j])).sum::<f64>(); nb.len() as f64 / s.max(1e-300) }).collect();
    Ok((0..n).map(|i| neigh[i].iter().map(|&(j, _)| lrd[j]).sum::<f64>() / (neigh[i].len() as f64 * lrd[i])).collect())
}

fn solve(mut a: Matrix, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let w = a[r][c] / a[c][c];
            for k in c..n {
                a[r][k] -= w * a[c][k];
            }
            b[r] -= w * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|k| a[i][k] * x[k]).sum::<f64>()) / a[i][i];
    }
    Some(x)
}

/// Mahalanobis distance of every row from the sample mean (small ridge added to the covariance for stability).
pub fn mahalanobis_scores(x: &Matrix) -> Result<Vec<f64>> {
    let d = rect(x, "mahalanobis_scores")?;
    let n = x.len() as f64;
    let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n).collect();
    let mut cov = vec![vec![0.0; d]; d];
    for r in x {
        for i in 0..d {
            for j in 0..d {
                cov[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]) / (n - 1.0);
            }
        }
    }
    (0..d).for_each(|i| cov[i][i] += 1e-9);
    x.iter()
        .map(|r| {
            let diff: Vec<f64> = r.iter().zip(&mean).map(|(a, b)| a - b).collect();
            let sol = solve(cov.clone(), diff.clone()).ok_or_else(|| Error::Msg("mahalanobis_scores: singular covariance".into()))?;
            Ok(diff.iter().zip(&sol).map(|(a, b)| a * b).sum::<f64>().max(0.0).sqrt())
        })
        .collect()
}

/// Indices of values more than `threshold` standard deviations from the mean.
pub fn zscore_outliers(x: &[f64], threshold: f64) -> Vec<usize> {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n).sqrt();
    if sd == 0.0 {
        return Vec::new();
    }
    (0..x.len()).filter(|&i| ((x[i] - m) / sd).abs() > threshold).collect()
}

/// Indices outside `[q1 - k * iqr, q3 + k * iqr]` (Tukey fences; `k = 1.5` is the classic choice).
pub fn iqr_outliers(x: &[f64], k: f64) -> Vec<usize> {
    if x.len() < 4 {
        return Vec::new();
    }
    let mut s = x.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| { let pos = p * (s.len() - 1) as f64; let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize); s[lo] + (s[hi] - s[lo]) * (pos - lo as f64) };
    let (q1, q3) = (q(0.25), q(0.75));
    let iqr = q3 - q1;
    (0..x.len()).filter(|&i| x[i] < q1 - k * iqr || x[i] > q3 + k * iqr).collect()
}

/// Two-sided CUSUM change detector against a reference mean `target` with slack `drift` and alarm `threshold`
/// (all in the data's units). Feed values with [`Cusum::update`]; it returns `true` on an alarm and then resets.
#[derive(Debug, Clone)]
pub struct Cusum {
    pub target: f64,
    pub drift: f64,
    pub threshold: f64,
    hi: f64,
    lo: f64,
}

impl Cusum {
    pub fn new(target: f64, drift: f64, threshold: f64) -> Self {
        Self { target, drift, threshold, hi: 0.0, lo: 0.0 }
    }

    pub fn update(&mut self, x: f64) -> bool {
        self.hi = (self.hi + x - self.target - self.drift).max(0.0);
        self.lo = (self.lo + self.target - x - self.drift).max(0.0);
        if self.hi > self.threshold || self.lo > self.threshold {
            self.hi = 0.0;
            self.lo = 0.0;
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(rng: &mut Rng) -> f64 {
        let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn data_with_outliers() -> (Matrix, Vec<usize>) {
        let mut rng = Rng::new(3);
        let mut x: Matrix = (0..200).map(|_| vec![gauss(&mut rng), gauss(&mut rng)]).collect();
        let outliers = vec![200, 201, 202];
        x.extend([vec![9.0, 9.0], vec![-8.0, 7.0], vec![10.0, -9.0]]);
        (x, outliers)
    }

    fn top(scores: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        idx.truncate(k);
        idx.sort_unstable();
        idx
    }

    #[test]
    fn multivariate_detectors_find_planted_outliers() {
        let (x, out) = data_with_outliers();
        let forest = IsolationForest::fit(&x, 200, 128, 1).unwrap();
        let s = forest.score(&x);
        assert_eq!(top(&s, 3), out, "isolation forest");
        assert!(s[200] > 0.6 && s[..200].iter().sum::<f64>() / 200.0 < 0.55);
        let lof = lof_scores(&x, 10).unwrap();
        assert_eq!(top(&lof, 3), out, "lof");
        assert!(lof[200] > 2.0 && (lof[..200].iter().sum::<f64>() / 200.0 - 1.0).abs() < 0.2);
        let maha = mahalanobis_scores(&x).unwrap();
        assert_eq!(top(&maha, 3), out, "mahalanobis");
        assert!(IsolationForest::fit(&x, 0, 10, 0).is_err() && lof_scores(&x, 0).is_err() && lof_scores(&x, 500).is_err());
    }

    #[test]
    fn univariate_rules_and_cusum() {
        let mut v: Vec<f64> = (0..50).map(|i| 10.0 + (i % 5) as f64 * 0.1).collect();
        v.push(25.0);
        assert_eq!(zscore_outliers(&v, 3.0), vec![50]);
        assert_eq!(iqr_outliers(&v, 1.5), vec![50]);
        assert!(zscore_outliers(&[5.0; 10], 3.0).is_empty() && iqr_outliers(&[1.0, 2.0], 1.5).is_empty());
        // CUSUM stays quiet on in-control data and alarms soon after a +2 shift
        let mut rng = Rng::new(9);
        let mut cs = Cusum::new(0.0, 0.5, 5.0);
        let quiet = (0..300).filter(|_| cs.update(gauss(&mut rng))).count();
        assert!(quiet <= 2, "{quiet}");
        let mut cs = Cusum::new(0.0, 0.5, 5.0);
        let first = (0..100).position(|_| cs.update(2.0 + gauss(&mut rng) * 0.5)).unwrap();
        assert!(first < 8, "{first}");
    }
}
