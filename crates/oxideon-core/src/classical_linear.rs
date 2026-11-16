//! Classical supervised models on plain `Vec<Vec<f64>>` data: [`LinearRegression`] / ridge (normal equations with a
//! Cholesky solve), [`LogisticRegression`] (binary or softmax, Newton-free gradient descent with L2), and
//! [`DecisionTree`] / [`RandomForest`] classifiers (Gini impurity, exhaustive threshold search).

use crate::classical::Matrix;
use crate::error::{Error, Result};
use crate::rng::Rng;

/// Solves `A x = b` for symmetric positive definite `A` by Cholesky factorization.
pub fn cholesky_solve(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d <= 1e-14 {
                    return Err(Error::Msg("cholesky_solve: matrix is not positive definite (collinear features? add ridge)".into()));
                }
                l[i][j] = d.sqrt();
            } else {
                l[i][j] = (a[i][j] - s) / l[j][j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = (b[i] - (0..i).map(|k| l[i][k] * y[k]).sum::<f64>()) / l[i][i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (y[i] - (i + 1..n).map(|k| l[k][i] * x[k]).sum::<f64>()) / l[i][i];
    }
    Ok(x)
}

fn width(x: &Matrix, y_len: usize, op: &str) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || d == 0 || x.len() != y_len || x.iter().any(|r| r.len() != d) {
        return Err(Error::Msg(format!("{op}: x must be a non-empty rectangular matrix with one target per row")));
    }
    Ok(d)
}

/// Ordinary least squares / ridge regression with an unpenalized intercept.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub coef: Vec<f64>,
    pub intercept: f64,
}

impl LinearRegression {
    /// `alpha = 0` is plain OLS.
    pub fn fit(x: &Matrix, y: &[f64], alpha: f64) -> Result<Self> {
        let d = width(x, y.len(), "LinearRegression")?;
        let n = x.len() as f64;
        let xm: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n).collect();
        let ym = y.iter().sum::<f64>() / n;
        let mut gram = vec![vec![0.0; d]; d];
        let mut rhs = vec![0.0; d];
        for (r, &t) in x.iter().zip(y) {
            for i in 0..d {
                rhs[i] += (r[i] - xm[i]) * (t - ym);
                for j in 0..d {
                    gram[i][j] += (r[i] - xm[i]) * (r[j] - xm[j]);
                }
            }
        }
        (0..d).for_each(|i| gram[i][i] += alpha);
        let coef = cholesky_solve(&gram, &rhs)?;
        let intercept = ym - coef.iter().zip(&xm).map(|(c, m)| c * m).sum::<f64>();
        Ok(Self { coef, intercept })
    }

    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        x.iter().map(|r| self.intercept + r.iter().zip(&self.coef).map(|(v, c)| v * c).sum::<f64>()).collect()
    }
}

/// Multinomial logistic regression (a single sigmoid when there are two classes) with L2 penalty `l2`, trained by
/// full-batch gradient descent with a backtracking-free fixed step derived from the data scale.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    /// `[classes][features]` weights and per-class intercepts.
    pub weights: Matrix,
    pub bias: Vec<f64>,
}

fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

impl LogisticRegression {
    pub fn fit(x: &Matrix, y: &[usize], l2: f64, epochs: usize) -> Result<Self> {
        let d = width(x, y.len(), "LogisticRegression")?;
        let k = y.iter().max().unwrap() + 1;
        if k < 2 {
            return Err(Error::Msg("LogisticRegression: need at least two classes".into()));
        }
        let n = x.len() as f64;
        // Hessian of softmax cross-entropy is bounded by 0.5 * ||x||^2 (+ l2): a safe constant step is its inverse
        let scale = x.iter().map(|r| r.iter().map(|v| v * v).sum::<f64>() + 1.0).sum::<f64>() / n;
        let lr = 1.0 / (0.5 * scale + l2);
        let (mut w, mut b) = (vec![vec![0.0; d]; k], vec![0.0; k]);
        for _ in 0..epochs {
            let mut gw = vec![vec![0.0; d]; k];
            let mut gb = vec![0.0; k];
            for (r, &t) in x.iter().zip(y) {
                let z: Vec<f64> = (0..k).map(|c| b[c] + r.iter().zip(&w[c]).map(|(v, wi)| v * wi).sum::<f64>()).collect();
                let p = softmax(&z);
                for c in 0..k {
                    let e = p[c] - f64::from(u8::from(c == t));
                    gb[c] += e / n;
                    r.iter().enumerate().for_each(|(j, v)| gw[c][j] += e * v / n);
                }
            }
            for c in 0..k {
                b[c] -= lr * gb[c];
                for j in 0..d {
                    w[c][j] -= lr * (gw[c][j] + l2 * w[c][j] / n);
                }
            }
        }
        Ok(Self { weights: w, bias: b })
    }

    pub fn predict_proba(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| softmax(&self.weights.iter().zip(&self.bias).map(|(w, b)| b + r.iter().zip(w).map(|(v, wi)| v * wi).sum::<f64>()).collect::<Vec<_>>())).collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.predict_proba(x).iter().map(|p| (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap()).collect()
    }
}

#[derive(Debug, Clone)]
enum Node {
    Leaf(Vec<f64>),
    Split { feature: usize, threshold: f64, left: Box<Node>, right: Box<Node> },
}

/// CART classification tree (Gini impurity).
#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: Node,
    classes: usize,
}

fn gini(counts: &[f64], total: f64) -> f64 {
    if total == 0.0 { 0.0 } else { 1.0 - counts.iter().map(|c| (c / total).powi(2)).sum::<f64>() }
}

impl DecisionTree {
    /// `max_features` limits the features examined per split (random subset, as in random forests); `None` = all.
    pub fn fit(x: &Matrix, y: &[usize], max_depth: usize, min_samples_split: usize, max_features: Option<usize>, seed: u64) -> Result<Self> {
        let d = width(x, y.len(), "DecisionTree")?;
        let classes = y.iter().max().unwrap() + 1;
        let mut rng = Rng::new(seed);
        let idx: Vec<usize> = (0..x.len()).collect();
        let root = Self::build(x, y, &idx, classes, d, max_depth, min_samples_split.max(2), max_features, &mut rng);
        Ok(Self { root, classes })
    }

    #[allow(clippy::too_many_arguments)]
    fn build(x: &Matrix, y: &[usize], idx: &[usize], classes: usize, d: usize, depth: usize, min_split: usize, max_features: Option<usize>, rng: &mut Rng) -> Node {
        let mut counts = vec![0.0; classes];
        idx.iter().for_each(|&i| counts[y[i]] += 1.0);
        let total = idx.len() as f64;
        let leaf = || Node::Leaf(counts.iter().map(|c| c / total).collect());
        if depth == 0 || idx.len() < min_split || gini(&counts, total) == 0.0 {
            return leaf();
        }
        let mut features: Vec<usize> = (0..d).collect();
        if let Some(m) = max_features {
            rng.shuffle(&mut features);
            features.truncate(m.clamp(1, d));
        }
        let mut best: Option<(f64, usize, f64)> = None;
        for &f in &features {
            let mut order = idx.to_vec();
            order.sort_by(|&a, &b| x[a][f].total_cmp(&x[b][f]));
            let mut left = vec![0.0; classes];
            for s in 0..order.len() - 1 {
                left[y[order[s]]] += 1.0;
                if x[order[s]][f] == x[order[s + 1]][f] {
                    continue;
                }
                let right: Vec<f64> = counts.iter().zip(&left).map(|(c, l)| c - l).collect();
                let nl = (s + 1) as f64;
                let score = (nl * gini(&left, nl) + (total - nl) * gini(&right, total - nl)) / total;
                if best.is_none_or(|b| score < b.0) {
                    best = Some((score, f, 0.5 * (x[order[s]][f] + x[order[s + 1]][f])));
                }
            }
        }
        match best {
            Some((score, feature, threshold)) if score < gini(&counts, total) - 1e-12 => {
                let (l, r): (Vec<usize>, Vec<usize>) = idx.iter().partition(|&&i| x[i][feature] <= threshold);
                Node::Split {
                    feature,
                    threshold,
                    left: Box::new(Self::build(x, y, &l, classes, d, depth - 1, min_split, max_features, rng)),
                    right: Box::new(Self::build(x, y, &r, classes, d, depth - 1, min_split, max_features, rng)),
                }
            }
            _ => leaf(),
        }
    }

    fn proba_row(&self, r: &[f64]) -> &[f64] {
        let mut n = &self.root;
        loop {
            match n {
                Node::Leaf(p) => return p,
                Node::Split { feature, threshold, left, right } => n = if r[*feature] <= *threshold { left } else { right },
            }
        }
    }

    pub fn predict_proba(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| self.proba_row(r).to_vec()).collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.predict_proba(x).iter().map(|p| (0..self.classes).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap()).collect()
    }
}

/// Bagged decision trees with random feature subsets (`sqrt(d)` per split).
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    classes: usize,
}

impl RandomForest {
    pub fn fit(x: &Matrix, y: &[usize], n_trees: usize, max_depth: usize, seed: u64) -> Result<Self> {
        let d = width(x, y.len(), "RandomForest")?;
        let classes = y.iter().max().unwrap() + 1;
        let mut rng = Rng::new(seed);
        let mf = ((d as f64).sqrt().ceil() as usize).max(1);
        let mut trees = Vec::with_capacity(n_trees);
        for t in 0..n_trees.max(1) {
            let pick: Vec<usize> = (0..x.len()).map(|_| (rng.next_u64() % x.len() as u64) as usize).collect();
            let (bx, by): (Matrix, Vec<usize>) = pick.iter().map(|&i| (x[i].clone(), y[i])).unzip();
            // a bootstrap can miss the highest class; keep the class count aligned across trees
            let mut tree = DecisionTree::fit(&bx, &by, max_depth, 2, Some(mf), seed.wrapping_add(t as u64 + 1))?;
            tree.classes = classes;
            trees.push(tree);
        }
        Ok(Self { trees, classes })
    }

    pub fn predict_proba(&self, x: &Matrix) -> Matrix {
        x.iter()
            .map(|r| {
                let mut acc = vec![0.0; self.classes];
                for t in &self.trees {
                    t.proba_row(r).iter().enumerate().for_each(|(c, p)| acc[c] += p / self.trees.len() as f64);
                }
                acc
            })
            .collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.predict_proba(x).iter().map(|p| (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_and_ridge() {
        // y = 3 x0 - 2 x1 + 5 exactly
        let x: Matrix = (0..20).map(|i| vec![i as f64, ((i * 7) % 11) as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] - 2.0 * r[1] + 5.0).collect();
        let m = LinearRegression::fit(&x, &y, 0.0).unwrap();
        assert!((m.coef[0] - 3.0).abs() < 1e-8 && (m.coef[1] + 2.0).abs() < 1e-8 && (m.intercept - 5.0).abs() < 1e-7);
        assert!(m.predict(&vec![vec![1.0, 1.0]]).iter().all(|p| (p - 6.0).abs() < 1e-7));
        let ridge = LinearRegression::fit(&x, &y, 1e4).unwrap();
        assert!(ridge.coef[0].abs() < m.coef[0].abs(), "ridge shrinks");
        // duplicated column is collinear: OLS errors, ridge works
        let dup: Matrix = x.iter().map(|r| vec![r[0], r[0]]).collect();
        assert!(LinearRegression::fit(&dup, &y, 0.0).is_err());
        assert!(LinearRegression::fit(&dup, &y, 1.0).is_ok());
    }

    fn two_moons_like() -> (Matrix, Vec<usize>) {
        let mut rng = Rng::new(11);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..200 {
            let c = i % 2;
            x.push(vec![c as f64 * 3.0 + rng.next_f64() - 0.5, rng.next_f64() * 4.0 - 2.0]);
            y.push(c);
        }
        (x, y)
    }

    #[test]
    fn logistic_separates_and_gives_probabilities() {
        let (x, y) = two_moons_like();
        let m = LogisticRegression::fit(&x, &y, 0.0, 500).unwrap();
        let acc = m.predict(&x).iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / y.len() as f64;
        assert!(acc > 0.97, "{acc}");
        assert!(m.predict_proba(&x).iter().all(|p| (p.iter().sum::<f64>() - 1.0).abs() < 1e-9));
        // three classes
        let x3: Matrix = (0..90).map(|i| vec![(i % 3) as f64 * 5.0 + 0.1 * (i / 3) as f64 / 30.0, 0.0]).collect();
        let y3: Vec<usize> = (0..90).map(|i| i % 3).collect();
        assert_eq!(LogisticRegression::fit(&x3, &y3, 0.0, 800).unwrap().predict(&x3), y3);
        assert!(LogisticRegression::fit(&x, &vec![0; 200], 0.0, 10).is_err());
    }

    #[test]
    fn trees_learn_xor_that_linear_models_cannot() {
        let mut rng = Rng::new(5);
        let x: Matrix = (0..300).map(|_| vec![rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0]).collect();
        let y: Vec<usize> = x.iter().map(|r| usize::from((r[0] > 0.0) != (r[1] > 0.0))).collect();
        let acc = |p: Vec<usize>| p.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / y.len() as f64;
        let lin = acc(LogisticRegression::fit(&x, &y, 0.0, 300).unwrap().predict(&x));
        assert!(lin < 0.7, "XOR is not linearly separable: {lin}");
        let tree = DecisionTree::fit(&x, &y, 6, 2, None, 1).unwrap();
        assert!(acc(tree.predict(&x)) > 0.97);
        let shallow = DecisionTree::fit(&x, &y, 1, 2, None, 1).unwrap();
        assert!(acc(shallow.predict(&x)) < 0.8);
        let forest = RandomForest::fit(&x, &y, 25, 6, 2).unwrap();
        assert!(acc(forest.predict(&x)) > 0.95);
        assert!(forest.predict_proba(&x).iter().all(|p| (p.iter().sum::<f64>() - 1.0).abs() < 1e-9));
        assert!(DecisionTree::fit(&x, &y[..5], 3, 2, None, 0).is_err());
    }
}
