//! Supporting machine-learning tools: [`AdaBoost`] (decision stumps), [`isotonic_regression`] (pool adjacent
//! violators), [`PlattScaler`] probability calibration, [`smote`] oversampling, optimal transport with
//! [`sinkhorn`], and filter-style feature selection ([`f_classif`], [`mutual_info_binned`], [`variance_threshold`]).

use crate::classical::Matrix;
use crate::error::{Error, Result};
use crate::rng::Rng;

// ------------------------------------------------------------------ AdaBoost

#[derive(Debug, Clone)]
struct Stump {
    feature: usize,
    threshold: f64,
    sign: f64, // prediction for x <= threshold
    alpha: f64,
}

/// Binary AdaBoost (SAMME) with threshold stumps. Labels are 0/1.
#[derive(Debug, Clone)]
pub struct AdaBoost {
    stumps: Vec<Stump>,
}

impl AdaBoost {
    pub fn fit(x: &Matrix, y: &[usize], n_rounds: usize) -> Result<Self> {
        let d = x.first().map_or(0, Vec::len);
        if x.is_empty() || d == 0 || x.len() != y.len() || y.iter().any(|&v| v > 1) || x.iter().any(|r| r.len() != d) {
            return Err(Error::Msg("AdaBoost: need rectangular data and 0/1 labels".into()));
        }
        let t: Vec<f64> = y.iter().map(|&v| if v == 1 { 1.0 } else { -1.0 }).collect();
        let n = x.len();
        let mut w = vec![1.0 / n as f64; n];
        let mut stumps = Vec::new();
        for _ in 0..n_rounds {
            let mut best: Option<(f64, Stump)> = None;
            for f in 0..d {
                let mut vals: Vec<f64> = x.iter().map(|r| r[f]).collect();
                vals.sort_by(f64::total_cmp);
                vals.dedup();
                for pair in vals.windows(2) {
                    let th = 0.5 * (pair[0] + pair[1]);
                    for sign in [1.0, -1.0] {
                        let err: f64 = (0..n).filter(|&i| (if x[i][f] <= th { sign } else { -sign }) != t[i]).map(|i| w[i]).sum();
                        if best.as_ref().is_none_or(|b| err < b.0) {
                            best = Some((err, Stump { feature: f, threshold: th, sign, alpha: 0.0 }));
                        }
                    }
                }
            }
            let Some((err, mut s)) = best else { break };
            let err = err.clamp(1e-12, 1.0 - 1e-12);
            if err >= 0.5 {
                break;
            }
            s.alpha = 0.5 * ((1.0 - err) / err).ln();
            for i in 0..n {
                let pred = if x[i][s.feature] <= s.threshold { s.sign } else { -s.sign };
                w[i] *= (-s.alpha * t[i] * pred).exp();
            }
            let z: f64 = w.iter().sum();
            w.iter_mut().for_each(|v| *v /= z);
            let perfect = err <= 1e-9;
            stumps.push(s);
            if perfect {
                break;
            }
        }
        Ok(Self { stumps })
    }

    /// Weighted vote margin (positive = class 1).
    pub fn decision_function(&self, x: &Matrix) -> Vec<f64> {
        x.iter().map(|r| self.stumps.iter().map(|s| s.alpha * if r[s.feature] <= s.threshold { s.sign } else { -s.sign }).sum()).collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.decision_function(x).iter().map(|&s| usize::from(s > 0.0)).collect()
    }

    pub fn n_estimators(&self) -> usize {
        self.stumps.len()
    }
}

// ------------------------------------------------------------------ calibration

/// Isotonic (monotone non-decreasing) least-squares fit of `y` ordered by `x`, via pool adjacent violators.
/// Returns fitted values aligned with the original order of `x`.
pub fn isotonic_regression(x: &[f64], y: &[f64]) -> Result<Vec<f64>> {
    if x.len() != y.len() || x.is_empty() {
        return Err(Error::Msg("isotonic_regression: need matching non-empty inputs".into()));
    }
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut blocks: Vec<(f64, f64)> = Vec::new(); // (sum, count)
    for &i in &order {
        blocks.push((y[i], 1.0));
        while blocks.len() > 1 {
            let (a, b) = (blocks[blocks.len() - 2], blocks[blocks.len() - 1]);
            if a.0 / a.1 <= b.0 / b.1 {
                break;
            }
            blocks.truncate(blocks.len() - 2);
            blocks.push((a.0 + b.0, a.1 + b.1));
        }
    }
    let mut fitted_sorted = Vec::with_capacity(x.len());
    for (s, c) in blocks {
        fitted_sorted.extend(std::iter::repeat_n(s / c, c as usize));
    }
    let mut out = vec![0.0; x.len()];
    for (pos, &i) in order.iter().enumerate() {
        out[i] = fitted_sorted[pos];
    }
    Ok(out)
}

/// Platt scaling: `P(y=1 | s) = sigmoid(a s + b)` fitted to classifier scores by Newton's method.
#[derive(Debug, Clone, Copy)]
pub struct PlattScaler {
    pub a: f64,
    pub b: f64,
}

impl PlattScaler {
    pub fn fit(scores: &[f64], y: &[usize]) -> Result<Self> {
        if scores.len() != y.len() || scores.is_empty() || !y.contains(&0) || !y.contains(&1) {
            return Err(Error::Msg("PlattScaler: need scores with 0/1 labels covering both classes".into()));
        }
        let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
        let (mut a, mut b) = (0.0, 0.0);
        for _ in 0..100 {
            let (mut ga, mut gb, mut haa, mut hab, mut hbb) = (0.0, 0.0, 1e-9, 0.0, 1e-9);
            for (&s, &t) in scores.iter().zip(y) {
                let p = sig(a * s + b);
                let e = p - t as f64;
                let w = p * (1.0 - p);
                ga += e * s;
                gb += e;
                haa += w * s * s;
                hab += w * s;
                hbb += w;
            }
            let det = haa * hbb - hab * hab;
            if det.abs() < 1e-18 {
                break;
            }
            let (da, db) = ((hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det);
            a -= da;
            b -= db;
            if da.abs() < 1e-10 && db.abs() < 1e-10 {
                break;
            }
        }
        Ok(Self { a, b })
    }

    pub fn predict_proba(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| 1.0 / (1.0 + (-(self.a * s + self.b)).exp())).collect()
    }
}

// ------------------------------------------------------------------ resampling

/// SMOTE: synthesizes `n_new` minority samples by interpolating between a minority point and one of its `k`
/// nearest minority neighbours.
pub fn smote(minority: &Matrix, n_new: usize, k: usize, seed: u64) -> Result<Matrix> {
    let d = minority.first().map_or(0, Vec::len);
    if minority.len() < 2 || d == 0 || k == 0 || minority.iter().any(|r| r.len() != d) {
        return Err(Error::Msg("smote: need at least two equal-width minority samples and k >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let dist = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>();
    let neigh: Vec<Vec<usize>> = (0..minority.len())
        .map(|i| {
            let mut o: Vec<usize> = (0..minority.len()).filter(|&j| j != i).collect();
            o.sort_by(|&a, &b| dist(&minority[i], &minority[a]).total_cmp(&dist(&minority[i], &minority[b])));
            o.truncate(k);
            o
        })
        .collect();
    Ok((0..n_new)
        .map(|_| {
            let i = (rng.next_u64() % minority.len() as u64) as usize;
            let j = neigh[i][(rng.next_u64() % neigh[i].len() as u64) as usize];
            let t = rng.next_f64();
            minority[i].iter().zip(&minority[j]).map(|(a, b)| a + t * (b - a)).collect()
        })
        .collect())
}

/// Entropic optimal transport (Sinkhorn): returns the transport plan between histograms `a` and `b` for cost matrix
/// `cost` with regularization `eps`, plus the transport cost `<plan, cost>`.
pub fn sinkhorn(a: &[f64], b: &[f64], cost: &Matrix, eps: f64, iters: usize) -> Result<(Matrix, f64)> {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 || cost.len() != n || cost.iter().any(|r| r.len() != m) || eps <= 0.0 || a.iter().chain(b).any(|&v| v < 0.0) {
        return Err(Error::Msg("sinkhorn: inconsistent shapes, negative mass or eps <= 0".into()));
    }
    let (sa, sb): (f64, f64) = (a.iter().sum(), b.iter().sum());
    if (sa - sb).abs() > 1e-9 * sa.max(1.0) {
        return Err(Error::Msg("sinkhorn: the two histograms must have equal total mass".into()));
    }
    let k: Matrix = cost.iter().map(|r| r.iter().map(|c| (-c / eps).exp()).collect()).collect();
    let (mut u, mut v) = (vec![1.0; n], vec![1.0; m]);
    for _ in 0..iters {
        for i in 0..n {
            u[i] = a[i] / (0..m).map(|j| k[i][j] * v[j]).sum::<f64>().max(1e-300);
        }
        for j in 0..m {
            v[j] = b[j] / (0..n).map(|i| k[i][j] * u[i]).sum::<f64>().max(1e-300);
        }
    }
    let plan: Matrix = (0..n).map(|i| (0..m).map(|j| u[i] * k[i][j] * v[j]).collect()).collect();
    let total = (0..n).flat_map(|i| (0..m).map(move |j| (i, j))).map(|(i, j)| plan[i][j] * cost[i][j]).sum();
    Ok((plan, total))
}

// ------------------------------------------------------------------ feature selection

/// ANOVA F-statistic of every column against class labels (larger = more discriminative).
pub fn f_classif(x: &Matrix, y: &[usize]) -> Result<Vec<f64>> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || x.len() != y.len() || d == 0 {
        return Err(Error::Msg("f_classif: need data with one label per row".into()));
    }
    let k = y.iter().max().unwrap() + 1;
    let n = x.len() as f64;
    Ok((0..d)
        .map(|j| {
            let grand = x.iter().map(|r| r[j]).sum::<f64>() / n;
            let (mut ssb, mut ssw) = (0.0, 0.0);
            for c in 0..k {
                let vals: Vec<f64> = x.iter().zip(y).filter(|(_, l)| **l == c).map(|(r, _)| r[j]).collect();
                if vals.is_empty() {
                    continue;
                }
                let m = vals.iter().sum::<f64>() / vals.len() as f64;
                ssb += vals.len() as f64 * (m - grand).powi(2);
                ssw += vals.iter().map(|v| (v - m).powi(2)).sum::<f64>();
            }
            (ssb / (k as f64 - 1.0).max(1.0)) / (ssw / (n - k as f64).max(1.0)).max(1e-300)
        })
        .collect())
}

/// Mutual information (nats) between each column (equal-width binned into `bins`) and the class labels.
pub fn mutual_info_binned(x: &Matrix, y: &[usize], bins: usize) -> Result<Vec<f64>> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || x.len() != y.len() || d == 0 || bins < 2 {
        return Err(Error::Msg("mutual_info_binned: need data with one label per row and bins >= 2".into()));
    }
    let n = x.len() as f64;
    let k = y.iter().max().unwrap() + 1;
    Ok((0..d)
        .map(|j| {
            let (lo, hi) = x.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, r| (a.0.min(r[j]), a.1.max(r[j])));
            let bin = |v: f64| if hi > lo { (((v - lo) / (hi - lo)) * bins as f64) as usize } else { 0 }.min(bins - 1);
            let mut joint = vec![vec![0.0; k]; bins];
            x.iter().zip(y).for_each(|(r, &c)| joint[bin(r[j])][c] += 1.0);
            let (pb, pc): (Vec<f64>, Vec<f64>) = ((0..bins).map(|b| joint[b].iter().sum::<f64>() / n).collect(), (0..k).map(|c| (0..bins).map(|b| joint[b][c]).sum::<f64>() / n).collect());
            (0..bins).flat_map(|b| (0..k).map(move |c| (b, c))).filter(|&(b, c)| joint[b][c] > 0.0).map(|(b, c)| joint[b][c] / n * (joint[b][c] / n / (pb[b] * pc[c])).ln()).sum()
        })
        .collect())
}

/// Indices of columns whose variance exceeds `threshold`.
pub fn variance_threshold(x: &Matrix, threshold: f64) -> Vec<usize> {
    let d = x.first().map_or(0, Vec::len);
    let n = x.len() as f64;
    (0..d)
        .filter(|&j| {
            let m = x.iter().map(|r| r[j]).sum::<f64>() / n;
            x.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / n > threshold
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn adaboost_beats_a_single_stump() {
        // diagonal boundary needs several axis-aligned stumps
        let mut rng = Rng::new(3);
        let x: Matrix = (0..300).map(|_| vec![rng.next_f64(), rng.next_f64()]).collect();
        let y: Vec<usize> = x.iter().map(|r| usize::from(r[0] + r[1] > 1.0)).collect();
        let acc = |p: Vec<usize>| p.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / y.len() as f64;
        let one = acc(AdaBoost::fit(&x, &y, 1).unwrap().predict(&x));
        let many = AdaBoost::fit(&x, &y, 60).unwrap();
        let boosted = acc(many.predict(&x));
        assert!(boosted > one + 0.1 && boosted > 0.92, "{one} -> {boosted}");
        assert!(many.n_estimators() > 5);
        assert!(AdaBoost::fit(&x, &[2; 300], 5).is_err());
    }

    #[test]
    fn isotonic_and_platt() {
        // scikit-learn style example: pooled violators
        let fit = isotonic_regression(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 3.0, 2.0, 4.0, 3.5]).unwrap();
        assert_eq!(fit, vec![1.0, 2.5, 2.5, 3.75, 3.75]);
        assert!(fit.windows(2).all(|w| w[1] >= w[0]));
        // unsorted input keeps its order in the output
        assert_eq!(isotonic_regression(&[3.0, 1.0, 2.0], &[3.0, 1.0, 2.0]).unwrap(), vec![3.0, 1.0, 2.0]);
        assert!(isotonic_regression(&[], &[]).is_err());
        // overconfident scores (separated, but wildly scaled) get calibrated to probabilities near the truth
        let mut rng = Rng::new(5);
        let scores: Vec<f64> = (0..500).map(|_| rng.next_f64() * 20.0 - 10.0).collect();
        let y: Vec<usize> = scores.iter().map(|&s| usize::from(rng.next_f64() < 1.0 / (1.0 + (-0.5 * s).exp()))).collect();
        let p = PlattScaler::fit(&scores, &y).unwrap();
        assert!((p.a - 0.5).abs() < 0.15 && p.b.abs() < 0.4, "{p:?}");
        let probs = p.predict_proba(&[-10.0, 0.0, 10.0]);
        assert!(probs[0] < 0.05 && probs[2] > 0.95 && (probs[1] - 0.5).abs() < 0.1);
        assert!(PlattScaler::fit(&[1.0, 2.0], &[1, 1]).is_err());
    }

    #[test]
    fn smote_and_transport() {
        let minority: Matrix = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let syn = smote(&minority, 50, 2, 1).unwrap();
        assert_eq!(syn.len(), 50);
        assert!(syn.iter().all(|p| (0.0..=1.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1])), "synthetic points stay in the convex hull");
        assert!(smote(&minority[..1].to_vec(), 5, 1, 0).is_err());
        // Sinkhorn on a 1-D problem with cost |i-j|: moving mass one step costs 1
        let cost: Matrix = (0..3).map(|i| (0..3).map(|j| (i as f64 - j as f64).abs()).collect()).collect();
        let (plan, total) = sinkhorn(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0], &cost, 0.05, 500).unwrap();
        close(total, 2.0, 1e-6);
        close(plan[0][2], 1.0, 1e-6);
        // marginals are respected for a non-trivial problem
        let (plan, _) = sinkhorn(&[0.5, 0.5, 0.0], &[0.0, 0.5, 0.5], &cost, 0.2, 2000).unwrap();
        close(plan[0].iter().sum::<f64>(), 0.5, 1e-6);
        close((0..3).map(|i| plan[i][2]).sum::<f64>(), 0.5, 1e-6);
        assert!(sinkhorn(&[1.0], &[0.5], &vec![vec![0.0]], 0.1, 10).is_err());
    }

    #[test]
    fn feature_scores_rank_informative_columns() {
        let mut rng = Rng::new(7);
        let y: Vec<usize> = (0..300).map(|i| i % 2).collect();
        let x: Matrix = y.iter().map(|&c| vec![c as f64 * 2.0 + rng.next_f64(), rng.next_f64(), 5.0]).collect();
        let f = f_classif(&x, &y).unwrap();
        assert!(f[0] > 100.0 * f[1].max(1e-9) && f[2] == 0.0, "{f:?}");
        let mi = mutual_info_binned(&x, &y, 8).unwrap();
        assert!(mi[0] > 0.3 && mi[1] < 0.05 && mi[2] == 0.0, "{mi:?}");
        assert_eq!(variance_threshold(&x, 1e-9), vec![0, 1]);
        assert!(f_classif(&x, &y[..5]).is_err() && mutual_info_binned(&x, &y, 1).is_err());
    }
}
