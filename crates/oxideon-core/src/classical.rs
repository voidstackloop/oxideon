//! Classical machine learning on plain `Vec<Vec<f64>>` data (rows = samples): [`StandardScaler`], [`MinMaxScaler`],
//! [`KMeans`] (k-means++), [`Dbscan`], [`Pca`] (Jacobi eigendecomposition of the covariance), [`KnnClassifier`] and
//! [`GaussianNb`]. Linear / logistic models live in `classical_linear`.

use crate::error::{Error, Result};
use crate::rng::Rng;

pub type Matrix = Vec<Vec<f64>>;

fn check(x: &Matrix, op: &str) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || d == 0 || x.iter().any(|r| r.len() != d || r.iter().any(|v| !v.is_finite())) {
        return Err(Error::Msg(format!("{op}: data must be a non-empty rectangular matrix of finite values")));
    }
    Ok(d)
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ------------------------------------------------------------------ scalers

/// Zero mean, unit variance per column (columns with zero variance are only centred).
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl StandardScaler {
    pub fn fit(x: &Matrix) -> Result<Self> {
        let d = check(x, "StandardScaler")?;
        let n = x.len() as f64;
        let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n).collect();
        let std = (0..d).map(|j| (x.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt()).map(|s| if s < 1e-12 { 1.0 } else { s }).collect();
        Ok(Self { mean, std })
    }
    pub fn transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| r.iter().enumerate().map(|(j, v)| (v - self.mean[j]) / self.std[j]).collect()).collect()
    }
    pub fn inverse_transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| r.iter().enumerate().map(|(j, v)| v * self.std[j] + self.mean[j]).collect()).collect()
    }
}

/// Rescales each column to `[0, 1]` (constant columns map to 0).
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    pub min: Vec<f64>,
    pub range: Vec<f64>,
}

impl MinMaxScaler {
    pub fn fit(x: &Matrix) -> Result<Self> {
        let d = check(x, "MinMaxScaler")?;
        let min: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min)).collect();
        let range = (0..d).map(|j| x.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max) - min[j]).map(|r| if r < 1e-12 { 1.0 } else { r }).collect();
        Ok(Self { min, range })
    }
    pub fn transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| r.iter().enumerate().map(|(j, v)| (v - self.min[j]) / self.range[j]).collect()).collect()
    }
}

// ------------------------------------------------------------------ clustering

/// Lloyd's k-means with k-means++ seeding and several restarts (best inertia wins).
#[derive(Debug, Clone)]
pub struct KMeans {
    pub centers: Matrix,
    pub labels: Vec<usize>,
    pub inertia: f64,
}

impl KMeans {
    pub fn fit(x: &Matrix, k: usize, max_iter: usize, restarts: usize, seed: u64) -> Result<Self> {
        check(x, "KMeans")?;
        if k == 0 || k > x.len() {
            return Err(Error::Msg(format!("KMeans: k = {k} must be in 1..={}", x.len())));
        }
        let mut rng = Rng::new(seed);
        let mut best: Option<Self> = None;
        for _ in 0..restarts.max(1) {
            let run = Self::run(x, k, max_iter, &mut rng);
            if best.as_ref().is_none_or(|b| run.inertia < b.inertia) {
                best = Some(run);
            }
        }
        Ok(best.expect("at least one restart"))
    }

    fn run(x: &Matrix, k: usize, max_iter: usize, rng: &mut Rng) -> Self {
        let d = x[0].len();
        let mut centers = vec![x[(rng.next_u64() % x.len() as u64) as usize].clone()];
        while centers.len() < k {
            let d2: Vec<f64> = x.iter().map(|p| centers.iter().map(|c| sq_dist(p, c)).fold(f64::INFINITY, f64::min)).collect();
            let total: f64 = d2.iter().sum();
            let mut u = rng.next_f64() * total;
            let mut pick = x.len() - 1;
            for (i, &w) in d2.iter().enumerate() {
                if u < w {
                    pick = i;
                    break;
                }
                u -= w;
            }
            centers.push(x[pick].clone());
        }
        let mut labels = vec![0; x.len()];
        for _ in 0..max_iter.max(1) {
            let mut changed = false;
            for (i, p) in x.iter().enumerate() {
                let l = (0..k).min_by(|&a, &b| sq_dist(p, &centers[a]).total_cmp(&sq_dist(p, &centers[b]))).unwrap();
                changed |= labels[i] != l;
                labels[i] = l;
            }
            let (mut sums, mut cnt) = (vec![vec![0.0; d]; k], vec![0usize; k]);
            for (p, &l) in x.iter().zip(&labels) {
                cnt[l] += 1;
                sums[l].iter_mut().zip(p).for_each(|(s, v)| *s += v);
            }
            for c in 0..k {
                if cnt[c] > 0 {
                    centers[c] = sums[c].iter().map(|s| s / cnt[c] as f64).collect();
                }
            }
            if !changed {
                break;
            }
        }
        let inertia = x.iter().zip(&labels).map(|(p, &l)| sq_dist(p, &centers[l])).sum();
        Self { centers, labels, inertia }
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        x.iter().map(|p| (0..self.centers.len()).min_by(|&a, &b| sq_dist(p, &self.centers[a]).total_cmp(&sq_dist(p, &self.centers[b]))).unwrap()).collect()
    }
}

/// Density-based clustering; noise points get `None`.
pub struct Dbscan {
    pub eps: f64,
    pub min_samples: usize,
}

impl Dbscan {
    pub fn fit_predict(&self, x: &Matrix) -> Result<Vec<Option<usize>>> {
        check(x, "Dbscan")?;
        let n = x.len();
        let eps2 = self.eps * self.eps;
        let neighbours = |i: usize| -> Vec<usize> { (0..n).filter(|&j| sq_dist(&x[i], &x[j]) <= eps2).collect() };
        let mut labels: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut cluster = 0;
        for i in 0..n {
            if visited[i] {
                continue;
            }
            visited[i] = true;
            let nb = neighbours(i);
            if nb.len() < self.min_samples {
                continue;
            }
            labels[i] = Some(cluster);
            let mut queue = nb;
            while let Some(j) = queue.pop() {
                if labels[j].is_none() {
                    labels[j] = Some(cluster);
                }
                if !visited[j] {
                    visited[j] = true;
                    let nj = neighbours(j);
                    if nj.len() >= self.min_samples {
                        queue.extend(nj);
                    }
                }
            }
            cluster += 1;
        }
        Ok(labels)
    }
}

// ------------------------------------------------------------------ PCA

/// Symmetric eigendecomposition by cyclic Jacobi rotations; returns eigenvalues (descending) and eigenvectors as rows.
pub fn jacobi_eigh(a: &Matrix) -> (Vec<f64>, Matrix) {
    let n = a.len();
    let mut a = a.clone();
    let mut v: Matrix = (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let vals = order.iter().map(|&i| a[i][i]).collect();
    let vecs = order.iter().map(|&i| (0..n).map(|k| v[k][i]).collect()).collect();
    (vals, vecs)
}

/// Principal component analysis.
#[derive(Debug, Clone)]
pub struct Pca {
    pub mean: Vec<f64>,
    /// Principal axes, one per row.
    pub components: Matrix,
    pub explained_variance: Vec<f64>,
    pub total_variance: f64,
}

impl Pca {
    pub fn fit(x: &Matrix, n_components: usize) -> Result<Self> {
        let d = check(x, "Pca")?;
        if n_components == 0 || n_components > d || x.len() < 2 {
            return Err(Error::Msg(format!("Pca: n_components {n_components} must be in 1..={d} and at least 2 samples are needed")));
        }
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
        let (vals, vecs) = jacobi_eigh(&cov);
        let total_variance = vals.iter().sum();
        Ok(Self { mean, components: vecs[..n_components].to_vec(), explained_variance: vals[..n_components].to_vec(), total_variance })
    }

    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        self.explained_variance.iter().map(|v| v / self.total_variance.max(1e-300)).collect()
    }

    pub fn transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| self.components.iter().map(|c| c.iter().zip(r.iter().zip(&self.mean)).map(|(w, (v, m))| w * (v - m)).sum()).collect()).collect()
    }

    pub fn inverse_transform(&self, z: &Matrix) -> Matrix {
        z.iter().map(|r| (0..self.mean.len()).map(|j| self.mean[j] + r.iter().zip(&self.components).map(|(zi, c)| zi * c[j]).sum::<f64>()).collect()).collect()
    }
}

// ------------------------------------------------------------------ classifiers

/// k-nearest-neighbours classifier (majority vote, ties go to the closest neighbour's class).
pub struct KnnClassifier {
    x: Matrix,
    y: Vec<usize>,
    pub k: usize,
}

impl KnnClassifier {
    pub fn fit(x: Matrix, y: Vec<usize>, k: usize) -> Result<Self> {
        check(&x, "KnnClassifier")?;
        if x.len() != y.len() || k == 0 {
            return Err(Error::Msg("KnnClassifier: need one label per row and k >= 1".into()));
        }
        Ok(Self { x, y, k })
    }

    pub fn predict(&self, q: &Matrix) -> Vec<usize> {
        q.iter()
            .map(|p| {
                let mut order: Vec<usize> = (0..self.x.len()).collect();
                order.sort_by(|&a, &b| sq_dist(p, &self.x[a]).total_cmp(&sq_dist(p, &self.x[b])));
                let top = &order[..self.k.min(order.len())];
                let classes = self.y.iter().max().unwrap() + 1;
                let mut votes = vec![0usize; classes];
                top.iter().for_each(|&i| votes[self.y[i]] += 1);
                let best = *votes.iter().max().unwrap();
                top.iter().map(|&i| self.y[i]).find(|&c| votes[c] == best).unwrap()
            })
            .collect()
    }
}

/// Gaussian naive Bayes with variance smoothing.
pub struct GaussianNb {
    prior: Vec<f64>,
    mean: Matrix,
    var: Matrix,
}

impl GaussianNb {
    pub fn fit(x: &Matrix, y: &[usize]) -> Result<Self> {
        let d = check(x, "GaussianNb")?;
        if x.len() != y.len() {
            return Err(Error::Msg("GaussianNb: need one label per row".into()));
        }
        let k = y.iter().max().unwrap() + 1;
        let global_var = (0..d).map(|j| {
            let m = x.iter().map(|r| r[j]).sum::<f64>() / x.len() as f64;
            x.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / x.len() as f64
        }).fold(0.0, f64::max);
        let (mut prior, mut mean, mut var) = (vec![0.0; k], vec![vec![0.0; d]; k], vec![vec![0.0; d]; k]);
        for c in 0..k {
            let rows: Vec<&Vec<f64>> = x.iter().zip(y).filter(|(_, l)| **l == c).map(|(r, _)| r).collect();
            if rows.is_empty() {
                prior[c] = 0.0;
                var[c] = vec![1.0; d];
                continue;
            }
            prior[c] = rows.len() as f64 / x.len() as f64;
            for j in 0..d {
                mean[c][j] = rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64;
                var[c][j] = rows.iter().map(|r| (r[j] - mean[c][j]).powi(2)).sum::<f64>() / rows.len() as f64 + 1e-9 * global_var.max(1e-12);
            }
        }
        Ok(Self { prior, mean, var })
    }

    /// Log joint likelihood of every class for every row.
    pub fn log_proba(&self, x: &Matrix) -> Matrix {
        x.iter()
            .map(|r| {
                (0..self.prior.len())
                    .map(|c| {
                        if self.prior[c] == 0.0 {
                            return f64::NEG_INFINITY;
                        }
                        self.prior[c].ln() - r.iter().enumerate().map(|(j, v)| 0.5 * ((2.0 * std::f64::consts::PI * self.var[c][j]).ln() + (v - self.mean[c][j]).powi(2) / self.var[c][j])).sum::<f64>()
                    })
                    .collect()
            })
            .collect()
    }

    pub fn predict(&self, x: &Matrix) -> Vec<usize> {
        self.log_proba(x).iter().map(|l| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs() -> (Matrix, Vec<usize>) {
        let mut rng = Rng::new(3);
        let centers = [[0.0, 0.0], [8.0, 8.0], [-8.0, 8.0]];
        let mut x = Vec::new();
        let mut y = Vec::new();
        for (c, ctr) in centers.iter().enumerate() {
            for _ in 0..30 {
                x.push(vec![ctr[0] + rng.next_f64() - 0.5, ctr[1] + rng.next_f64() - 0.5]);
                y.push(c);
            }
        }
        (x, y)
    }

    #[test]
    fn scalers_round_trip() {
        let x = vec![vec![1.0, 10.0], vec![3.0, 10.0], vec![5.0, 10.0]];
        let s = StandardScaler::fit(&x).unwrap();
        let z = s.transform(&x);
        assert!(z.iter().map(|r| r[0]).sum::<f64>().abs() < 1e-12 && z.iter().all(|r| r[1] == 0.0));
        let back = s.inverse_transform(&z);
        assert!(back.iter().flatten().zip(x.iter().flatten()).all(|(a, b)| (a - b).abs() < 1e-12));
        let m = MinMaxScaler::fit(&x).unwrap().transform(&x);
        assert_eq!((m[0][0], m[2][0], m[1][1]), (0.0, 1.0, 0.0));
        assert!(StandardScaler::fit(&vec![]).is_err());
    }

    #[test]
    fn kmeans_and_dbscan_recover_blobs() {
        let (x, y) = blobs();
        let km = KMeans::fit(&x, 3, 100, 5, 1).unwrap();
        assert!((crate::metrics_more::adjusted_rand_index(&km.labels, &y).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(km.predict(&vec![vec![7.5, 8.2]])[0], km.labels[30]);
        assert!(KMeans::fit(&x, 0, 10, 1, 1).is_err());
        let db = Dbscan { eps: 1.5, min_samples: 4 }.fit_predict(&x).unwrap();
        let labels: Vec<usize> = db.iter().map(|l| l.expect("no noise in clean blobs")).collect();
        assert!((crate::metrics_more::adjusted_rand_index(&labels, &y).unwrap() - 1.0).abs() < 1e-12);
        let mut with_outlier = x.clone();
        with_outlier.push(vec![100.0, 100.0]);
        assert_eq!(Dbscan { eps: 1.5, min_samples: 4 }.fit_predict(&with_outlier).unwrap().last().unwrap(), &None);
    }

    #[test]
    fn pca_finds_the_dominant_axis() {
        // points along y = 2x with tiny orthogonal noise
        let x: Matrix = (0..50).map(|i| { let t = i as f64 / 10.0 - 2.5; vec![t + 0.001 * ((i * 7) % 5) as f64, 2.0 * t] }).collect();
        let p = Pca::fit(&x, 1).unwrap();
        let (c0, c1) = (p.components[0][0].abs(), p.components[0][1].abs());
        assert!((c1 / c0 - 2.0).abs() < 0.01, "{:?}", p.components);
        assert!(p.explained_variance_ratio()[0] > 0.999);
        let rec = p.inverse_transform(&p.transform(&x));
        assert!(rec.iter().flatten().zip(x.iter().flatten()).all(|(a, b)| (a - b).abs() < 0.01));
        // eigen-solver on a known matrix: [[2,1],[1,2]] -> 3, 1
        let (vals, _) = jacobi_eigh(&vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        assert!((vals[0] - 3.0).abs() < 1e-12 && (vals[1] - 1.0).abs() < 1e-12);
        assert!(Pca::fit(&x, 3).is_err());
    }

    #[test]
    fn knn_and_naive_bayes_classify_blobs() {
        let (x, y) = blobs();
        let queries = vec![vec![0.2, -0.1], vec![7.7, 8.1], vec![-7.9, 7.6]];
        assert_eq!(KnnClassifier::fit(x.clone(), y.clone(), 5).unwrap().predict(&queries), vec![0, 1, 2]);
        let nb = GaussianNb::fit(&x, &y).unwrap();
        assert_eq!(nb.predict(&queries), vec![0, 1, 2]);
        assert_eq!(nb.predict(&x), y);
        assert!(KnnClassifier::fit(x, vec![0], 3).is_err());
    }
}
