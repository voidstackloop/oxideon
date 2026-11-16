//! More classical models: [`RegressionTree`], [`GradientBoosting`] (squared-error regression or binary logistic loss),
//! [`GaussianMixture`] (diagonal covariances, EM), and a kernel [`Svm`] (binary, simplified SMO).

use crate::classical::{KMeans, Matrix};
use crate::error::{Error, Result};
use crate::rng::Rng;

fn check(x: &Matrix, n_y: usize, op: &str) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || d == 0 || x.len() != n_y || x.iter().any(|r| r.len() != d) {
        return Err(Error::Msg(format!("{op}: x must be a non-empty rectangular matrix with one target per row")));
    }
    Ok(d)
}

// ------------------------------------------------------------------ regression tree + boosting

#[derive(Debug, Clone)]
enum Node {
    Leaf(f64),
    Split { feature: usize, threshold: f64, left: Box<Node>, right: Box<Node> },
}

/// CART regression tree minimizing squared error.
#[derive(Debug, Clone)]
pub struct RegressionTree {
    root: Node,
}

impl RegressionTree {
    pub fn fit(x: &Matrix, y: &[f64], max_depth: usize, min_samples_leaf: usize) -> Result<Self> {
        let d = check(x, y.len(), "RegressionTree")?;
        let idx: Vec<usize> = (0..x.len()).collect();
        Ok(Self { root: Self::build(x, y, &idx, d, max_depth, min_samples_leaf.max(1)) })
    }

    fn build(x: &Matrix, y: &[f64], idx: &[usize], d: usize, depth: usize, min_leaf: usize) -> Node {
        let n = idx.len() as f64;
        let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / n;
        if depth == 0 || idx.len() < 2 * min_leaf {
            return Node::Leaf(mean);
        }
        let total_sum: f64 = idx.iter().map(|&i| y[i]).sum();
        let mut best: Option<(f64, usize, f64)> = None; // (gain proxy, feature, threshold)
        for f in 0..d {
            let mut order = idx.to_vec();
            order.sort_by(|&a, &b| x[a][f].total_cmp(&x[b][f]));
            let mut left_sum = 0.0;
            for s in 0..order.len() - 1 {
                left_sum += y[order[s]];
                let nl = (s + 1) as f64;
                if x[order[s]][f] == x[order[s + 1]][f] || s + 1 < min_leaf || order.len() - s - 1 < min_leaf {
                    continue;
                }
                // maximizing sum_l^2/n_l + sum_r^2/n_r minimizes the squared error
                let gain = left_sum * left_sum / nl + (total_sum - left_sum).powi(2) / (n - nl);
                if best.is_none_or(|b| gain > b.0) {
                    best = Some((gain, f, 0.5 * (x[order[s]][f] + x[order[s + 1]][f])));
                }
            }
        }
        match best {
            Some((gain, feature, threshold)) if gain > total_sum * total_sum / n + 1e-12 => {
                let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[i][feature] <= threshold);
                Node::Split { feature, threshold, left: Box::new(Self::build(x, y, &l, d, depth - 1, min_leaf)), right: Box::new(Self::build(x, y, &r, d, depth - 1, min_leaf)) }
            }
            _ => Node::Leaf(mean),
        }
    }

    pub fn predict_row(&self, r: &[f64]) -> f64 {
        let mut n = &self.root;
        loop {
            match n {
                Node::Leaf(v) => return *v,
                Node::Split { feature, threshold, left, right } => n = if r[*feature] <= *threshold { left } else { right },
            }
        }
    }

    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        x.iter().map(|r| self.predict_row(r)).collect()
    }
}

/// Gradient-boosted trees. `logistic = true` fits binary 0/1 labels with log loss; otherwise squared error.
pub struct GradientBoosting {
    base: f64,
    trees: Vec<RegressionTree>,
    lr: f64,
    logistic: bool,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

impl GradientBoosting {
    pub fn fit(x: &Matrix, y: &[f64], n_trees: usize, lr: f64, max_depth: usize, logistic: bool) -> Result<Self> {
        check(x, y.len(), "GradientBoosting")?;
        if logistic && y.iter().any(|&v| v != 0.0 && v != 1.0) {
            return Err(Error::Msg("GradientBoosting: logistic targets must be 0 or 1".into()));
        }
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let base = if logistic { (mean.clamp(1e-6, 1.0 - 1e-6) / (1.0 - mean.clamp(1e-6, 1.0 - 1e-6))).ln() } else { mean };
        let mut f = vec![base; y.len()];
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            // negative gradient: residual (squared error) or y - p (log loss)
            let resid: Vec<f64> = f.iter().zip(y).map(|(&fi, &yi)| if logistic { yi - sigmoid(fi) } else { yi - fi }).collect();
            let tree = RegressionTree::fit(x, &resid, max_depth, 1)?;
            for (fi, r) in f.iter_mut().zip(x) {
                *fi += lr * tree.predict_row(r);
            }
            trees.push(tree);
        }
        Ok(Self { base, trees, lr, logistic })
    }

    /// Raw scores (log-odds when logistic).
    pub fn decision_function(&self, x: &Matrix) -> Vec<f64> {
        x.iter().map(|r| self.base + self.lr * self.trees.iter().map(|t| t.predict_row(r)).sum::<f64>()).collect()
    }

    /// Regression values, or class-1 probabilities when logistic.
    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        let s = self.decision_function(x);
        if self.logistic { s.into_iter().map(sigmoid).collect() } else { s }
    }
}

// ------------------------------------------------------------------ Gaussian mixture

/// Gaussian mixture with diagonal covariances fitted by EM (k-means initialization).
pub struct GaussianMixture {
    pub weights: Vec<f64>,
    pub means: Matrix,
    pub vars: Matrix,
    pub log_likelihood: f64,
}

fn log_normal(x: &[f64], mean: &[f64], var: &[f64]) -> f64 {
    x.iter().enumerate().map(|(j, v)| -0.5 * ((2.0 * std::f64::consts::PI * var[j]).ln() + (v - mean[j]).powi(2) / var[j])).sum()
}

fn logsumexp(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

impl GaussianMixture {
    pub fn fit(x: &Matrix, k: usize, max_iter: usize, seed: u64) -> Result<Self> {
        let d = check(x, x.len(), "GaussianMixture")?;
        let km = KMeans::fit(x, k, 50, 3, seed)?;
        let n = x.len();
        let mut resp = vec![vec![0.0; k]; n];
        for (i, &l) in km.labels.iter().enumerate() {
            resp[i][l] = 1.0;
        }
        let (mut weights, mut means, mut vars) = (vec![0.0; k], vec![vec![0.0; d]; k], vec![vec![1.0; d]; k]);
        let mut prev = f64::NEG_INFINITY;
        let mut ll = prev;
        for _ in 0..max_iter.max(1) {
            // M step
            for c in 0..k {
                let nk: f64 = resp.iter().map(|r| r[c]).sum::<f64>().max(1e-10);
                weights[c] = nk / n as f64;
                for j in 0..d {
                    means[c][j] = x.iter().zip(&resp).map(|(r, g)| g[c] * r[j]).sum::<f64>() / nk;
                    vars[c][j] = x.iter().zip(&resp).map(|(r, g)| g[c] * (r[j] - means[c][j]).powi(2)).sum::<f64>() / nk + 1e-6;
                }
            }
            // E step
            ll = 0.0;
            for (i, r) in x.iter().enumerate() {
                let logp: Vec<f64> = (0..k).map(|c| weights[c].ln() + log_normal(r, &means[c], &vars[c])).collect();
                let z = logsumexp(&logp);
                ll += z;
                resp[i] = logp.iter().map(|l| (l - z).exp()).collect();
            }
            if (ll - prev).abs() < 1e-8 * ll.abs().max(1.0) {
                break;
            }
            prev = ll;
        }
        Ok(Self { weights, means, vars, log_likelihood: ll })
    }

    /// Posterior component probabilities per row.
    pub fn predict_proba(&self, x: &Matrix) -> Matrix {
        x.iter()
            .map(|r| {
                let logp: Vec<f64> = (0..self.weights.len()).map(|c| self.weights[c].ln() + log_normal(r, &self.means[c], &self.vars[c])).collect();
                let z = logsumexp(&logp);
                logp.iter().map(|l| (l - z).exp()).collect()
            })
            .collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.predict_proba(x).iter().map(|p| (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap()).collect()
    }

    /// Number of free parameters (for BIC / AIC model selection).
    pub fn n_parameters(&self) -> usize {
        let (k, d) = (self.weights.len(), self.means[0].len());
        k * 2 * d + k - 1
    }

    pub fn bic(&self, n_samples: usize) -> f64 {
        self.n_parameters() as f64 * (n_samples as f64).ln() - 2.0 * self.log_likelihood
    }
}

// ------------------------------------------------------------------ SVM

/// Kernel of an [`Svm`].
#[derive(Debug, Clone, Copy)]
pub enum Kernel {
    Linear,
    Rbf { gamma: f64 },
    Poly { degree: i32, coef0: f64 },
}

impl Kernel {
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        match *self {
            Kernel::Linear => dot,
            Kernel::Rbf { gamma } => (-gamma * a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>()).exp(),
            Kernel::Poly { degree, coef0 } => (dot + coef0).powi(degree),
        }
    }
}

/// Binary soft-margin SVM trained with the simplified SMO algorithm (Platt). Labels are 0/1.
pub struct Svm {
    kernel: Kernel,
    support: Matrix,
    coef: Vec<f64>, // alpha_i * y_i
    bias: f64,
}

impl Svm {
    pub fn fit(x: &Matrix, y: &[usize], c: f64, kernel: Kernel, max_passes: usize, seed: u64) -> Result<Self> {
        check(x, y.len(), "Svm")?;
        if y.iter().any(|&v| v > 1) || !y.contains(&0) || !y.contains(&1) {
            return Err(Error::Msg("Svm: needs binary labels 0/1 with both classes present".into()));
        }
        let n = x.len();
        let t: Vec<f64> = y.iter().map(|&v| if v == 1 { 1.0 } else { -1.0 }).collect();
        let k: Matrix = (0..n).map(|i| (0..n).map(|j| kernel.eval(&x[i], &x[j])).collect()).collect();
        let (mut alpha, mut b) = (vec![0.0; n], 0.0);
        let mut rng = Rng::new(seed);
        let f = |alpha: &[f64], b: f64, i: usize| -> f64 { (0..n).map(|j| alpha[j] * t[j] * k[j][i]).sum::<f64>() + b };
        let (tol, mut passes) = (1e-3, 0);
        let mut total_iters = 0;
        while passes < max_passes && total_iters < 200 * max_passes.max(1) {
            total_iters += 1;
            let mut changed = 0;
            for i in 0..n {
                let ei = f(&alpha, b, i) - t[i];
                if (t[i] * ei < -tol && alpha[i] < c) || (t[i] * ei > tol && alpha[i] > 0.0) {
                    let mut j = (rng.next_u64() % (n as u64 - 1).max(1)) as usize;
                    if j >= i {
                        j += 1;
                    }
                    if j >= n {
                        continue;
                    }
                    let ej = f(&alpha, b, j) - t[j];
                    let (ai, aj) = (alpha[i], alpha[j]);
                    let (lo, hi) = if t[i] != t[j] { ((aj - ai).max(0.0), (c + aj - ai).min(c)) } else { ((ai + aj - c).max(0.0), (ai + aj).min(c)) };
                    let eta = 2.0 * k[i][j] - k[i][i] - k[j][j];
                    if hi - lo < 1e-12 || eta >= 0.0 {
                        continue;
                    }
                    let mut ajn = aj - t[j] * (ei - ej) / eta;
                    ajn = ajn.clamp(lo, hi);
                    if (ajn - aj).abs() < 1e-5 {
                        continue;
                    }
                    let ain = ai + t[i] * t[j] * (aj - ajn);
                    let b1 = b - ei - t[i] * (ain - ai) * k[i][i] - t[j] * (ajn - aj) * k[i][j];
                    let b2 = b - ej - t[i] * (ain - ai) * k[i][j] - t[j] * (ajn - aj) * k[j][j];
                    b = if ain > 0.0 && ain < c { b1 } else if ajn > 0.0 && ajn < c { b2 } else { 0.5 * (b1 + b2) };
                    alpha[i] = ain;
                    alpha[j] = ajn;
                    changed += 1;
                }
            }
            passes = if changed == 0 { passes + 1 } else { 0 };
        }
        let keep: Vec<usize> = (0..n).filter(|&i| alpha[i] > 1e-8).collect();
        Ok(Self { kernel, support: keep.iter().map(|&i| x[i].clone()).collect(), coef: keep.iter().map(|&i| alpha[i] * t[i]).collect(), bias: b })
    }

    pub fn n_support(&self) -> usize {
        self.support.len()
    }

    /// Signed distance-like score (positive = class 1).
    pub fn decision_function(&self, x: &Matrix) -> Vec<f64> {
        x.iter().map(|r| self.bias + self.support.iter().zip(&self.coef).map(|(s, a)| a * self.kernel.eval(s, r)).sum::<f64>()).collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.decision_function(x).iter().map(|&s| usize::from(s > 0.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acc(p: &[usize], y: &[usize]) -> f64 {
        p.iter().zip(y).filter(|(a, b)| a == b).count() as f64 / y.len() as f64
    }

    #[test]
    fn trees_and_boosting_regress_and_classify() {
        let x: Matrix = (0..100).map(|i| vec![i as f64 / 10.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| r[0].sin()).collect();
        let tree = RegressionTree::fit(&x, &y, 8, 1).unwrap();
        let mse = |p: &[f64]| p.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / y.len() as f64;
        assert!(mse(&tree.predict(&x)) < 1e-3);
        let stump = RegressionTree::fit(&x, &y, 1, 1).unwrap();
        assert!(mse(&stump.predict(&x)) > 0.1);
        let gb = GradientBoosting::fit(&x, &y, 60, 0.3, 2, false).unwrap();
        assert!(mse(&gb.predict(&x)) < 0.01, "{}", mse(&gb.predict(&x)));
        // binary classification of a non-monotone rule
        let labels: Vec<f64> = x.iter().map(|r| f64::from(u8::from(r[0] > 3.0 && r[0] < 7.0))).collect();
        let clf = GradientBoosting::fit(&x, &labels, 40, 0.3, 2, true).unwrap();
        let pred: Vec<usize> = clf.predict(&x).iter().map(|&p| usize::from(p > 0.5)).collect();
        let truth: Vec<usize> = labels.iter().map(|&v| v as usize).collect();
        assert!(acc(&pred, &truth) > 0.97);
        assert!(GradientBoosting::fit(&x, &y, 5, 0.1, 2, true).is_err());
    }

    #[test]
    fn gmm_recovers_components_and_bic_picks_k() {
        let mut rng = Rng::new(9);
        let mut x: Matrix = Vec::new();
        let mut y = Vec::new();
        for i in 0..200 {
            let c = i % 2;
            x.push(vec![c as f64 * 6.0 + (rng.next_f64() - 0.5), (rng.next_f64() - 0.5) * 2.0]);
            y.push(c);
        }
        let g = GaussianMixture::fit(&x, 2, 100, 1).unwrap();
        assert!((crate::metrics_more::adjusted_rand_index(&g.predict(&x), &y).unwrap() - 1.0).abs() < 1e-9);
        assert!((g.weights[0] - 0.5).abs() < 0.01);
        let mut means: Vec<f64> = g.means.iter().map(|m| m[0]).collect();
        means.sort_by(f64::total_cmp);
        assert!(means[0].abs() < 0.2 && (means[1] - 6.0).abs() < 0.2);
        let g1 = GaussianMixture::fit(&x, 1, 100, 1).unwrap();
        assert!(g.bic(200) < g1.bic(200), "two components describe two clusters better");
    }

    #[test]
    fn svm_linear_and_rbf() {
        let mut rng = Rng::new(4);
        // linearly separable
        let x: Matrix = (0..80).map(|i| vec![(i % 2) as f64 * 4.0 - 2.0 + 0.5 * (rng.next_f64() - 0.5), rng.next_f64()]).collect();
        let y: Vec<usize> = (0..80).map(|i| i % 2).collect();
        let lin = Svm::fit(&x, &y, 1.0, Kernel::Linear, 5, 1).unwrap();
        assert_eq!(acc(&lin.predict(&x), &y), 1.0);
        assert!(lin.n_support() < 80);
        // concentric rings need the RBF kernel
        let ring: Matrix = (0..120).map(|i| { let a = i as f64 * 0.5; let r = if i % 2 == 0 { 1.0 } else { 3.0 }; vec![r * a.cos(), r * a.sin()] }).collect();
        let ry: Vec<usize> = (0..120).map(|i| i % 2).collect();
        let rbf = Svm::fit(&ring, &ry, 10.0, Kernel::Rbf { gamma: 0.5 }, 5, 1).unwrap();
        assert!(acc(&rbf.predict(&ring), &ry) > 0.95);
        let lin_ring = Svm::fit(&ring, &ry, 10.0, Kernel::Linear, 5, 1).unwrap();
        assert!(acc(&lin_ring.predict(&ring), &ry) < 0.8);
        assert!(Svm::fit(&x, &vec![0; 80], 1.0, Kernel::Linear, 3, 1).is_err());
    }
}
