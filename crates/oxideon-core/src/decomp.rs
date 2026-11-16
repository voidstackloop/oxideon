//! More decompositions and clustering on plain `Vec<Vec<f64>>` data: [`nmf`], [`fast_ica`], [`KernelPca`],
//! Fisher [`Lda`], [`agglomerative`] clustering (single / complete / average / Ward linkage), [`spectral_clustering`],
//! and [`mean_shift`].

use crate::classical::{jacobi_eigh, KMeans, Matrix};
use crate::error::{Error, Result};
use crate::rng::Rng;

fn rect(x: &Matrix, op: &str) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || d == 0 || x.iter().any(|r| r.len() != d || r.iter().any(|v| !v.is_finite())) {
        return Err(Error::Msg(format!("{op}: need a non-empty rectangular matrix of finite values")));
    }
    Ok(d)
}

fn gauss(rng: &mut Rng) -> f64 {
    let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    (0..a.len()).map(|i| (0..b[0].len()).map(|j| (0..b.len()).map(|k| a[i][k] * b[k][j]).sum()).collect()).collect()
}

fn transpose(a: &Matrix) -> Matrix {
    (0..a[0].len()).map(|j| a.iter().map(|r| r[j]).collect()).collect()
}

/// Non-negative matrix factorization `X ~ W H` by Lee-Seung multiplicative updates; returns `(W, H, error)`.
pub fn nmf(x: &Matrix, rank: usize, iters: usize, seed: u64) -> Result<(Matrix, Matrix, f64)> {
    let d = rect(x, "nmf")?;
    if x.iter().flatten().any(|&v| v < 0.0) || rank == 0 || rank > d.min(x.len()) {
        return Err(Error::Msg("nmf: data must be non-negative and 1 <= rank <= min(n, d)".into()));
    }
    let mut rng = Rng::new(seed);
    let scale = (x.iter().flatten().sum::<f64>() / (x.len() * d) as f64 / rank as f64).sqrt().max(1e-3);
    let mut w: Matrix = (0..x.len()).map(|_| (0..rank).map(|_| scale * (0.5 + rng.next_f64())).collect()).collect();
    let mut h: Matrix = (0..rank).map(|_| (0..d).map(|_| scale * (0.5 + rng.next_f64())).collect()).collect();
    let eps = 1e-12;
    for _ in 0..iters {
        let wt = transpose(&w);
        let (num, den) = (matmul(&wt, x), matmul(&matmul(&wt, &w), &h));
        for k in 0..rank {
            for j in 0..d {
                h[k][j] *= num[k][j] / (den[k][j] + eps);
            }
        }
        let ht = transpose(&h);
        let (num, den) = (matmul(x, &ht), matmul(&w, &matmul(&h, &ht)));
        for i in 0..x.len() {
            for k in 0..rank {
                w[i][k] *= num[i][k] / (den[i][k] + eps);
            }
        }
    }
    let wh = matmul(&w, &h);
    let err = x.iter().zip(&wh).map(|(a, b)| a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>()).sum::<f64>().sqrt();
    Ok((w, h, err))
}

/// FastICA (deflation, tanh non-linearity): returns `(sources, unmixing)` where `sources = whitened data * unmixing^T`.
pub fn fast_ica(x: &Matrix, n_components: usize, iters: usize, seed: u64) -> Result<Matrix> {
    let d = rect(x, "fast_ica")?;
    if n_components == 0 || n_components > d || x.len() < 2 {
        return Err(Error::Msg("fast_ica: need 1 <= n_components <= columns and at least 2 samples".into()));
    }
    let n = x.len() as f64;
    let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n).collect();
    let xc: Matrix = x.iter().map(|r| r.iter().zip(&mean).map(|(a, m)| a - m).collect()).collect();
    let cov: Matrix = (0..d).map(|i| (0..d).map(|j| xc.iter().map(|r| r[i] * r[j]).sum::<f64>() / n).collect()).collect();
    let (vals, vecs) = jacobi_eigh(&cov);
    // whitening: z = x * E * D^-1/2 (keep the top n_components directions)
    let z: Matrix = xc.iter().map(|r| (0..n_components).map(|k| vecs[k].iter().zip(r).map(|(a, b)| a * b).sum::<f64>() / vals[k].max(1e-12).sqrt()).collect()).collect();
    let mut rng = Rng::new(seed);
    let mut w: Vec<Vec<f64>> = Vec::new();
    for _ in 0..n_components {
        let mut v: Vec<f64> = (0..n_components).map(|_| gauss(&mut rng)).collect();
        for _ in 0..iters {
            // deflate against already found components, then fixed-point update
            for prev in &w {
                let dot: f64 = v.iter().zip(prev).map(|(a, b)| a * b).sum();
                v.iter_mut().zip(prev).for_each(|(a, b)| *a -= dot * b);
            }
            let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-300);
            v.iter_mut().for_each(|a| *a /= norm);
            let (mut e1, mut e2) = (vec![0.0; n_components], 0.0);
            for row in &z {
                let proj: f64 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
                let g = proj.tanh();
                e1.iter_mut().zip(row).for_each(|(acc, r)| *acc += r * g / n);
                e2 += (1.0 - g * g) / n;
            }
            let mut next: Vec<f64> = e1.iter().zip(&v).map(|(a, b)| a - e2 * b).collect();
            for prev in &w {
                let dot: f64 = next.iter().zip(prev).map(|(a, b)| a * b).sum();
                next.iter_mut().zip(prev).for_each(|(a, b)| *a -= dot * b);
            }
            let nn = next.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-300);
            next.iter_mut().for_each(|a| *a /= nn);
            let converged = next.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>().abs() > 1.0 - 1e-10;
            v = next;
            if converged {
                break;
            }
        }
        w.push(v);
    }
    Ok(z.iter().map(|row| w.iter().map(|wv| wv.iter().zip(row).map(|(a, b)| a * b).sum()).collect()).collect())
}

/// Kernel PCA with an RBF kernel.
#[derive(Debug, Clone)]
pub struct KernelPca {
    train: Matrix,
    alphas: Matrix, // [component][sample], scaled by 1/sqrt(lambda)
    row_mean: Vec<f64>,
    total_mean: f64,
    gamma: f64,
}

fn rbf(a: &[f64], b: &[f64], gamma: f64) -> f64 {
    (-gamma * a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>()).exp()
}

impl KernelPca {
    pub fn fit(x: &Matrix, n_components: usize, gamma: f64) -> Result<Self> {
        rect(x, "KernelPca")?;
        let n = x.len();
        if n_components == 0 || n_components > n || gamma <= 0.0 {
            return Err(Error::Msg("KernelPca: need 1 <= n_components <= n and gamma > 0".into()));
        }
        let k: Matrix = (0..n).map(|i| (0..n).map(|j| rbf(&x[i], &x[j], gamma)).collect()).collect();
        let row_mean: Vec<f64> = k.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
        let total_mean = row_mean.iter().sum::<f64>() / n as f64;
        let kc: Matrix = (0..n).map(|i| (0..n).map(|j| k[i][j] - row_mean[i] - row_mean[j] + total_mean).collect()).collect();
        let (vals, vecs) = jacobi_eigh(&kc);
        let alphas = (0..n_components).map(|c| vecs[c].iter().map(|v| v / vals[c].max(1e-12).sqrt()).collect()).collect();
        Ok(Self { train: x.clone(), alphas, row_mean, total_mean, gamma })
    }

    pub fn transform(&self, q: &Matrix) -> Matrix {
        let n = self.train.len();
        q.iter()
            .map(|p| {
                let kp: Vec<f64> = self.train.iter().map(|t| rbf(t, p, self.gamma)).collect();
                let m = kp.iter().sum::<f64>() / n as f64;
                let kc: Vec<f64> = (0..n).map(|j| kp[j] - m - self.row_mean[j] + self.total_mean).collect();
                self.alphas.iter().map(|a| a.iter().zip(&kc).map(|(x, y)| x * y).sum()).collect()
            })
            .collect()
    }
}

/// Fisher linear discriminant analysis: projects onto the directions maximizing between- over within-class scatter.
#[derive(Debug, Clone)]
pub struct Lda {
    pub components: Matrix,
    mean: Vec<f64>,
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

impl Lda {
    pub fn fit(x: &Matrix, y: &[usize], n_components: usize) -> Result<Self> {
        let d = rect(x, "Lda")?;
        let k = y.iter().max().map_or(0, |m| m + 1);
        if x.len() != y.len() || k < 2 || n_components == 0 || n_components > (k - 1).min(d) {
            return Err(Error::Msg("Lda: need labels per row, >= 2 classes and 1 <= n_components <= min(classes - 1, d)".into()));
        }
        let n = x.len() as f64;
        let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n).collect();
        let (mut sw, mut sb) = (vec![vec![0.0; d]; d], vec![vec![0.0; d]; d]);
        for c in 0..k {
            let rows: Vec<&Vec<f64>> = x.iter().zip(y).filter(|(_, l)| **l == c).map(|(r, _)| r).collect();
            if rows.is_empty() {
                continue;
            }
            let mc: Vec<f64> = (0..d).map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / rows.len() as f64).collect();
            for r in &rows {
                for i in 0..d {
                    for j in 0..d {
                        sw[i][j] += (r[i] - mc[i]) * (r[j] - mc[j]);
                    }
                }
            }
            for i in 0..d {
                for j in 0..d {
                    sb[i][j] += rows.len() as f64 * (mc[i] - mean[i]) * (mc[j] - mean[j]);
                }
            }
        }
        (0..d).for_each(|i| sw[i][i] += 1e-8);
        // M = Sw^-1 Sb, then top eigenvectors by power iteration with deflation (M is not symmetric)
        let mcols: Vec<Vec<f64>> = (0..d).map(|j| solve(sw.clone(), (0..d).map(|i| sb[i][j]).collect()).ok_or_else(|| Error::Msg("Lda: singular within-class scatter".into()))).collect::<Result<_>>()?;
        let m: Matrix = (0..d).map(|i| (0..d).map(|j| mcols[j][i]).collect()).collect();
        let mut comps: Matrix = Vec::new();
        let mut a = m.clone();
        for _ in 0..n_components {
            let mut v = vec![1.0; d];
            v[0] = 1.5;
            let mut lam = 0.0;
            for _ in 0..500 {
                let nv: Vec<f64> = (0..d).map(|i| a[i].iter().zip(&v).map(|(p, q)| p * q).sum()).collect();
                lam = nv.iter().map(|t| t * t).sum::<f64>().sqrt();
                if lam < 1e-300 {
                    break;
                }
                v = nv.iter().map(|t| t / lam).collect();
            }
            let sign = if v.iter().cloned().fold(0.0, |m: f64, t| if t.abs() > m.abs() { t } else { m }) < 0.0 { -1.0 } else { 1.0 };
            let v: Vec<f64> = v.iter().map(|t| t * sign).collect();
            // Hotelling deflation (valid here since the dominant eigenvalue is real and simple)
            let vtv: f64 = v.iter().map(|t| t * t).sum();
            for i in 0..d {
                for j in 0..d {
                    a[i][j] -= lam * v[i] * v[j] / vtv;
                }
            }
            comps.push(v);
        }
        Ok(Self { components: comps, mean })
    }

    pub fn transform(&self, x: &Matrix) -> Matrix {
        x.iter().map(|r| self.components.iter().map(|c| c.iter().zip(r.iter().zip(&self.mean)).map(|(w, (v, m))| w * (v - m)).sum()).collect()).collect()
    }
}

/// Linkage criterion of [`agglomerative`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    Single,
    Complete,
    Average,
    Ward,
}

/// Hierarchical clustering cut at `n_clusters`; returns a label per row.
pub fn agglomerative(x: &Matrix, n_clusters: usize, linkage: Linkage) -> Result<Vec<usize>> {
    rect(x, "agglomerative")?;
    let n = x.len();
    if n_clusters == 0 || n_clusters > n {
        return Err(Error::Msg("agglomerative: need 1 <= n_clusters <= n".into()));
    }
    let d = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>().sqrt();
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let dist = |a: &[usize], b: &[usize]| -> f64 {
        match linkage {
            Linkage::Single => a.iter().flat_map(|&i| b.iter().map(move |&j| (i, j))).map(|(i, j)| d(&x[i], &x[j])).fold(f64::INFINITY, f64::min),
            Linkage::Complete => a.iter().flat_map(|&i| b.iter().map(move |&j| (i, j))).map(|(i, j)| d(&x[i], &x[j])).fold(0.0, f64::max),
            Linkage::Average => a.iter().flat_map(|&i| b.iter().map(move |&j| (i, j))).map(|(i, j)| d(&x[i], &x[j])).sum::<f64>() / (a.len() * b.len()) as f64,
            Linkage::Ward => {
                let cen = |c: &[usize]| -> Vec<f64> { (0..x[0].len()).map(|j| c.iter().map(|&i| x[i][j]).sum::<f64>() / c.len() as f64).collect() };
                let (ca, cb) = (cen(a), cen(b));
                (2.0 * a.len() as f64 * b.len() as f64 / (a.len() + b.len()) as f64).sqrt() * d(&ca, &cb)
            }
        }
    };
    while clusters.len() > n_clusters {
        let (mut best, mut pair) = (f64::INFINITY, (0, 1));
        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let v = dist(&clusters[i], &clusters[j]);
                if v < best {
                    best = v;
                    pair = (i, j);
                }
            }
        }
        let merged = clusters.remove(pair.1);
        clusters[pair.0].extend(merged);
    }
    let mut labels = vec![0; n];
    for (c, members) in clusters.iter().enumerate() {
        members.iter().for_each(|&i| labels[i] = c);
    }
    Ok(labels)
}

/// Spectral clustering with an RBF affinity: k-means on the leading eigenvectors of the normalized affinity.
pub fn spectral_clustering(x: &Matrix, k: usize, gamma: f64, seed: u64) -> Result<Vec<usize>> {
    rect(x, "spectral_clustering")?;
    let n = x.len();
    if k < 2 || k > n || gamma <= 0.0 {
        return Err(Error::Msg("spectral_clustering: need 2 <= k <= n and gamma > 0".into()));
    }
    let a: Matrix = (0..n).map(|i| (0..n).map(|j| if i == j { 0.0 } else { rbf(&x[i], &x[j], gamma) }).collect()).collect();
    let deg: Vec<f64> = a.iter().map(|r| r.iter().sum::<f64>().max(1e-12)).collect();
    let l: Matrix = (0..n).map(|i| (0..n).map(|j| a[i][j] / (deg[i] * deg[j]).sqrt()).collect()).collect();
    let (_, vecs) = jacobi_eigh(&l);
    let mut emb: Matrix = (0..n).map(|i| (0..k).map(|c| vecs[c][i]).collect()).collect();
    for row in emb.iter_mut() {
        let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        row.iter_mut().for_each(|v| *v /= norm);
    }
    Ok(KMeans::fit(&emb, k, 100, 5, seed)?.labels)
}

/// Mean-shift clustering with a flat kernel of radius `bandwidth`; returns `(labels, modes)`.
pub fn mean_shift(x: &Matrix, bandwidth: f64, max_iter: usize) -> Result<(Vec<usize>, Matrix)> {
    let d = rect(x, "mean_shift")?;
    if bandwidth <= 0.0 {
        return Err(Error::Msg("mean_shift: bandwidth must be positive".into()));
    }
    let shifted: Matrix = x
        .iter()
        .map(|p| {
            let mut c = p.clone();
            for _ in 0..max_iter {
                let near: Vec<&Vec<f64>> = x.iter().filter(|q| q.iter().zip(&c).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() <= bandwidth).collect();
                if near.is_empty() {
                    break;
                }
                let next: Vec<f64> = (0..d).map(|j| near.iter().map(|q| q[j]).sum::<f64>() / near.len() as f64).collect();
                let moved = next.iter().zip(&c).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
                c = next;
                if moved < 1e-7 * bandwidth {
                    break;
                }
            }
            c
        })
        .collect();
    let mut modes: Matrix = Vec::new();
    let labels = shifted
        .iter()
        .map(|c| match modes.iter().position(|m| m.iter().zip(c).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() < bandwidth / 2.0) {
            Some(i) => i,
            None => {
                modes.push(c.clone());
                modes.len() - 1
            }
        })
        .collect();
    Ok((labels, modes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics_more::adjusted_rand_index;

    fn blobs(seed: u64) -> (Matrix, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let centers = [[0.0, 0.0], [6.0, 6.0], [-6.0, 6.0]];
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (c, ctr) in centers.iter().enumerate() {
            for _ in 0..25 {
                x.push(vec![ctr[0] + 0.5 * gauss(&mut rng), ctr[1] + 0.5 * gauss(&mut rng)]);
                y.push(c);
            }
        }
        (x, y)
    }

    #[test]
    fn nmf_reconstructs_low_rank_data() {
        let mut rng = Rng::new(2);
        let w: Matrix = (0..30).map(|_| (0..3).map(|_| rng.next_f64()).collect()).collect();
        let h: Matrix = (0..3).map(|_| (0..10).map(|_| rng.next_f64()).collect()).collect();
        let x = matmul(&w, &h);
        let (fw, fh, err) = nmf(&x, 3, 800, 1).unwrap();
        let norm = x.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err / norm < 0.02, "relative error {}", err / norm);
        assert!(fw.iter().chain(&fh).flatten().all(|&v| v >= 0.0));
        assert!(nmf(&vec![vec![-1.0, 1.0]], 1, 10, 0).is_err());
    }

    #[test]
    fn ica_separates_mixed_sources() {
        let n = 2000;
        let s1: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.05).sin().signum() * 1.0).collect(); // square wave
        let s2: Vec<f64> = (0..n).map(|t| ((t as f64 * 0.013) % 2.0) - 1.0).collect(); // sawtooth
        let x: Matrix = (0..n).map(|t| vec![0.8 * s1[t] + 0.4 * s2[t], 0.3 * s1[t] + 0.9 * s2[t]]).collect();
        let src = fast_ica(&x, 2, 500, 3).unwrap();
        // each recovered component correlates strongly (up to sign) with exactly one true source
        let corr = |a: &[f64], b: &[f64]| {
            let (ma, mb) = (a.iter().sum::<f64>() / a.len() as f64, b.iter().sum::<f64>() / b.len() as f64);
            let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
            for (p, q) in a.iter().zip(b) {
                sab += (p - ma) * (q - mb);
                saa += (p - ma).powi(2);
                sbb += (q - mb).powi(2);
            }
            (sab / (saa * sbb).sqrt()).abs()
        };
        let comp = |c: usize| -> Vec<f64> { src.iter().map(|r| r[c]).collect() };
        let best: Vec<f64> = (0..2).map(|c| corr(&comp(c), &s1).max(corr(&comp(c), &s2))).collect();
        assert!(best.iter().all(|&b| b > 0.95), "{best:?}");
        assert!((corr(&comp(0), &s1) > corr(&comp(0), &s2)) != (corr(&comp(1), &s1) > corr(&comp(1), &s2)), "components map to different sources");
    }

    #[test]
    fn kernel_pca_and_lda() {
        // concentric rings become linearly separable in the first kernel component
        let ring: Matrix = (0..80).map(|i| { let a = i as f64 * 0.7; let r = if i % 2 == 0 { 1.0 } else { 3.0 }; vec![r * a.cos(), r * a.sin()] }).collect();
        let kp = KernelPca::fit(&ring, 2, 0.5).unwrap();
        let z = kp.transform(&ring);
        let inner: Vec<f64> = (0..80).filter(|i| i % 2 == 0).map(|i| z[i][0]).collect();
        let outer: Vec<f64> = (0..80).filter(|i| i % 2 == 1).map(|i| z[i][0]).collect();
        let (imax, imin) = (inner.iter().cloned().fold(f64::NEG_INFINITY, f64::max), inner.iter().cloned().fold(f64::INFINITY, f64::min));
        let (omax, omin) = (outer.iter().cloned().fold(f64::NEG_INFINITY, f64::max), outer.iter().cloned().fold(f64::INFINITY, f64::min));
        assert!(imax < omin || omax < imin, "rings separated along component 1");
        let (x, y) = blobs(4);
        let lda = Lda::fit(&x, &y, 2).unwrap();
        let proj = lda.transform(&x);
        assert_eq!((proj.len(), proj[0].len()), (75, 2));
        let km = KMeans::fit(&proj, 3, 50, 3, 1).unwrap();
        assert!((adjusted_rand_index(&km.labels, &y).unwrap() - 1.0).abs() < 1e-9);
        assert!(Lda::fit(&x, &y, 3).is_err() && Lda::fit(&x, &vec![0; 75], 1).is_err());
    }

    #[test]
    fn clustering_methods_recover_blobs() {
        let (x, y) = blobs(5);
        for link in [Linkage::Single, Linkage::Complete, Linkage::Average, Linkage::Ward] {
            let l = agglomerative(&x, 3, link).unwrap();
            assert!((adjusted_rand_index(&l, &y).unwrap() - 1.0).abs() < 1e-9, "{link:?}");
        }
        let sc = spectral_clustering(&x, 3, 0.2, 1).unwrap();
        assert!((adjusted_rand_index(&sc, &y).unwrap() - 1.0).abs() < 1e-9);
        let (ms, modes) = mean_shift(&x, 3.0, 100).unwrap();
        assert_eq!(modes.len(), 3);
        assert!((adjusted_rand_index(&ms, &y).unwrap() - 1.0).abs() < 1e-9);
        // single linkage chains two crescents that k-means-style methods cannot split
        let moons: Matrix = (0..60).map(|i| { let t = (i / 2) as f64 * 0.1; if i % 2 == 0 { vec![t.cos(), t.sin()] } else { vec![1.0 - t.cos(), 0.5 - t.sin()] } }).collect();
        assert_eq!(agglomerative(&moons, 2, Linkage::Single).unwrap().iter().collect::<std::collections::HashSet<_>>().len(), 2);
        assert!(agglomerative(&x, 0, Linkage::Ward).is_err() && mean_shift(&x, 0.0, 5).is_err());
    }
}
