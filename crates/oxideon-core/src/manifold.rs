//! Dimensionality reduction on plain `Vec<Vec<f64>>` data: classical [`mds`], [`isomap`], exact [`tsne`] (for up to a
//! few thousand points), Gaussian [`random_projection`], and the [`trustworthiness`] score for judging embeddings.

use crate::classical::{jacobi_eigh, Matrix};
use crate::error::{Error, Result};
use crate::rng::Rng;

fn check(x: &Matrix, op: &str) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    if x.len() < 3 || d == 0 || x.iter().any(|r| r.len() != d || r.iter().any(|v| !v.is_finite())) {
        return Err(Error::Msg(format!("{op}: need at least 3 rows of finite, equal-length data")));
    }
    Ok(d)
}

fn pairwise(x: &Matrix) -> Matrix {
    let n = x.len();
    let mut d = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let v = x[i].iter().zip(&x[j]).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            d[i][j] = v;
            d[j][i] = v;
        }
    }
    d
}

/// Classical (Torgerson) multidimensional scaling of a distance matrix into `dims` coordinates.
pub fn mds_from_distances(dist: &Matrix, dims: usize) -> Result<Matrix> {
    let n = dist.len();
    if dims == 0 || dims > n || dist.iter().any(|r| r.len() != n) {
        return Err(Error::Msg("mds: need a square distance matrix and 1 <= dims <= n".into()));
    }
    let sq: Matrix = dist.iter().map(|r| r.iter().map(|v| v * v).collect()).collect();
    let row: Vec<f64> = sq.iter().map(|r| r.iter().sum::<f64>() / n as f64).collect();
    let total = row.iter().sum::<f64>() / n as f64;
    // double centering: B = -1/2 (D2 - row_i - row_j + total)
    let b: Matrix = (0..n).map(|i| (0..n).map(|j| -0.5 * (sq[i][j] - row[i] - row[j] + total)).collect()).collect();
    let (vals, vecs) = jacobi_eigh(&b);
    Ok((0..n).map(|i| (0..dims).map(|k| vecs[k][i] * vals[k].max(0.0).sqrt()).collect()).collect())
}

/// Classical MDS of the Euclidean distances between the rows of `x`.
pub fn mds(x: &Matrix, dims: usize) -> Result<Matrix> {
    check(x, "mds")?;
    mds_from_distances(&pairwise(x), dims)
}

/// Isomap: MDS of geodesic distances along the `k`-nearest-neighbour graph (errors if the graph is disconnected).
pub fn isomap(x: &Matrix, k: usize, dims: usize) -> Result<Matrix> {
    check(x, "isomap")?;
    let n = x.len();
    let d = pairwise(x);
    let mut g = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n {
        g[i][i] = 0.0;
        let mut order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        order.sort_by(|&a, &b| d[i][a].total_cmp(&d[i][b]));
        for &j in order.iter().take(k.max(1)) {
            g[i][j] = d[i][j];
            g[j][i] = d[i][j];
        }
    }
    for m in 0..n {
        for i in 0..n {
            for j in 0..n {
                let via = g[i][m] + g[m][j];
                if via < g[i][j] {
                    g[i][j] = via;
                }
            }
        }
    }
    if g.iter().flatten().any(|v| v.is_infinite()) {
        return Err(Error::Msg("isomap: the neighbour graph is disconnected; increase k".into()));
    }
    mds_from_distances(&g, dims)
}

/// Gaussian random projection to `dims` dimensions (Johnson-Lindenstrauss).
pub fn random_projection(x: &Matrix, dims: usize, seed: u64) -> Result<Matrix> {
    let d = check(x, "random_projection")?;
    if dims == 0 {
        return Err(Error::Msg("random_projection: dims must be positive".into()));
    }
    let mut rng = Rng::new(seed);
    let gauss = |rng: &mut Rng| {
        let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let r: Matrix = (0..dims).map(|_| (0..d).map(|_| gauss(&mut rng) / (dims as f64).sqrt()).collect()).collect();
    Ok(x.iter().map(|row| r.iter().map(|c| c.iter().zip(row).map(|(a, b)| a * b).sum()).collect()).collect())
}

/// t-SNE with exact affinities, early exaggeration and momentum gradient descent. Initialized from the leading MDS
/// coordinates scaled small, so results are deterministic given `seed`.
pub fn tsne(x: &Matrix, dims: usize, perplexity: f64, iters: usize, lr: f64, seed: u64) -> Result<Matrix> {
    check(x, "tsne")?;
    let n = x.len();
    if perplexity < 1.0 || perplexity >= n as f64 {
        return Err(Error::Msg(format!("tsne: perplexity {perplexity} must be in [1, n)")));
    }
    let d2: Matrix = pairwise(x).iter().map(|r| r.iter().map(|v| v * v).collect()).collect();
    // conditional affinities: binary-search each row's precision to hit the target perplexity
    let target = perplexity.ln();
    let mut p = vec![vec![0.0; n]; n];
    for i in 0..n {
        let (mut lo, mut hi, mut beta) = (0.0, f64::INFINITY, 1.0);
        for _ in 0..60 {
            let w: Vec<f64> = (0..n).map(|j| if j == i { 0.0 } else { (-beta * d2[i][j]).exp() }).collect();
            let s = w.iter().sum::<f64>().max(1e-300);
            let h = s.ln() + beta * (0..n).map(|j| d2[i][j] * w[j]).sum::<f64>() / s;
            if (h - target).abs() < 1e-5 {
                p[i] = w.iter().map(|v| v / s).collect();
                break;
            }
            if h > target {
                lo = beta;
                beta = if hi.is_infinite() { beta * 2.0 } else { 0.5 * (beta + hi) };
            } else {
                hi = beta;
                beta = 0.5 * (beta + lo);
            }
            p[i] = w.iter().map(|v| v / s).collect();
        }
    }
    let mut sym = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            sym[i][j] = ((p[i][j] + p[j][i]) / (2.0 * n as f64)).max(1e-12);
        }
    }
    let mut y = mds(x, dims.min(n))?;
    let scale = y.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
    let mut rng = Rng::new(seed);
    y.iter_mut().flatten().for_each(|v| *v = *v / scale * 1e-2 + 1e-4 * (rng.next_f64() - 0.5));
    let mut vel = vec![vec![0.0; dims]; n];
    for it in 0..iters {
        let exag = if it < iters / 4 { 12.0 } else { 1.0 };
        let mom = if it < iters / 4 { 0.5 } else { 0.8 };
        let mut q = vec![vec![0.0; n]; n];
        let mut z = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                let v = 1.0 / (1.0 + y[i].iter().zip(&y[j]).map(|(a, b)| (a - b).powi(2)).sum::<f64>());
                q[i][j] = v;
                q[j][i] = v;
                z += 2.0 * v;
            }
        }
        for i in 0..n {
            let mut grad = vec![0.0; dims];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let coef = (exag * sym[i][j] - q[i][j] / z) * q[i][j];
                for k in 0..dims {
                    grad[k] += 4.0 * coef * (y[i][k] - y[j][k]);
                }
            }
            for k in 0..dims {
                vel[i][k] = mom * vel[i][k] - lr * grad[k];
            }
        }
        for i in 0..n {
            for k in 0..dims {
                y[i][k] += vel[i][k];
            }
        }
        // recentre
        for k in 0..dims {
            let m = y.iter().map(|r| r[k]).sum::<f64>() / n as f64;
            y.iter_mut().for_each(|r| r[k] -= m);
        }
    }
    Ok(y)
}

/// Trustworthiness in `[0, 1]`: how many of each point's `k` nearest neighbours in the embedding were also close
/// in the original space (1 = perfect neighbourhood preservation).
pub fn trustworthiness(x: &Matrix, embedded: &Matrix, k: usize) -> Result<f64> {
    check(x, "trustworthiness")?;
    let n = x.len();
    if embedded.len() != n || k == 0 || 2 * k >= n {
        return Err(Error::Msg("trustworthiness: need matching row counts and 0 < k < n/2".into()));
    }
    let (dx, dy) = (pairwise(x), pairwise(embedded));
    let mut penalty = 0.0;
    for i in 0..n {
        let mut ox: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        ox.sort_by(|&a, &b| dx[i][a].total_cmp(&dx[i][b]));
        let rank: std::collections::HashMap<usize, usize> = ox.iter().enumerate().map(|(r, &j)| (j, r + 1)).collect();
        let mut oy: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        oy.sort_by(|&a, &b| dy[i][a].total_cmp(&dy[i][b]));
        penalty += oy.iter().take(k).map(|j| (rank[j].saturating_sub(k)) as f64).sum::<f64>();
    }
    let nf = n as f64;
    let kf = k as f64;
    Ok(1.0 - 2.0 / (nf * kf * (2.0 * nf - 3.0 * kf - 1.0)) * penalty)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    #[test]
    fn mds_preserves_euclidean_distances() {
        let mut rng = Rng::new(2);
        let x: Matrix = (0..20).map(|_| vec![rng.next_f64() * 5.0, rng.next_f64() * 2.0, 0.0]).collect(); // lies in a plane
        let y = mds(&x, 2).unwrap();
        for i in 0..20 {
            for j in 0..20 {
                assert!((d(&x[i], &x[j]) - d(&y[i], &y[j])).abs() < 1e-6);
            }
        }
        assert!(mds(&x, 0).is_err());
        assert!(mds(&x[..2].to_vec(), 1).is_err());
    }

    #[test]
    fn isomap_unrolls_a_swiss_roll_ribbon() {
        // a spiral strip: Euclidean MDS folds it, geodesic distances recover the arc length
        let x: Matrix = (0..60).map(|i| { let t = 0.2 + i as f64 * 0.1; vec![t * t.cos(), t * t.sin(), 0.0] }).collect();
        let y = isomap(&x, 4, 1).unwrap();
        // 1-D coordinate must be monotone in the curve parameter
        let rising = y.windows(2).all(|w| w[1][0] > w[0][0]);
        let falling = y.windows(2).all(|w| w[1][0] < w[0][0]);
        assert!(rising || falling, "{:?}", y.iter().take(5).collect::<Vec<_>>());
        assert!(isomap(&x, 1, 1).is_err() || isomap(&x, 1, 1).is_ok()); // k = 1 may or may not connect; must not panic
    }

    #[test]
    fn tsne_separates_clusters_and_scores_high_trustworthiness() {
        let mut rng = Rng::new(6);
        let mut x: Matrix = Vec::new();
        for c in 0..3 {
            for _ in 0..25 {
                x.push((0..6).map(|j| if j == c { 8.0 } else { 0.0 } + rng.next_f64()).collect());
            }
        }
        let y = tsne(&x, 2, 10.0, 300, 100.0, 1).unwrap();
        assert_eq!(y.len(), 75);
        let t = trustworthiness(&x, &y, 5).unwrap();
        assert!(t > 0.9, "trustworthiness {t}");
        // clusters occupy separate regions: within-cluster distances are smaller than between-cluster
        let within = d(&y[0], &y[1]);
        let between = d(&y[0], &y[30]);
        assert!(within < between);
        assert!(tsne(&x, 2, 500.0, 10, 100.0, 1).is_err());
        // a random embedding scores worse
        let noise: Matrix = (0..75).map(|_| vec![rng.next_f64(), rng.next_f64()]).collect();
        assert!(trustworthiness(&x, &noise, 5).unwrap() < t);
    }

    #[test]
    fn random_projection_roughly_preserves_distances() {
        let mut rng = Rng::new(8);
        let x: Matrix = (0..30).map(|_| (0..200).map(|_| rng.next_f64()).collect()).collect();
        let y = random_projection(&x, 100, 3).unwrap();
        let ratio = d(&y[0], &y[1]) / d(&x[0], &x[1]);
        assert!(ratio > 0.7 && ratio < 1.3, "{ratio}");
        assert!(random_projection(&x, 0, 3).is_err());
    }
}
