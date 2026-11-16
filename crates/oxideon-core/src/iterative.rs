//! Matrix-free iterative linear algebra on `&[f64]` vectors with a user-supplied matrix-vector product:
//! [`cg`] (conjugate gradient, optional Jacobi preconditioner), [`gmres`] (restarted), [`bicgstab`],
//! [`power_iteration`], [`lanczos_extreme`] (extreme eigenvalues of a symmetric operator), and a dense
//! [`randomized_svd`]. Works with [`SparseCoo`] through [`sparse_matvec`].

use crate::classical::{jacobi_eigh, Matrix};
use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::sparse::SparseCoo;

/// Convergence report of an iterative solve.
#[derive(Debug, Clone)]
pub struct Solve {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `y = A x` for a [`SparseCoo`] matrix.
pub fn sparse_matvec(a: &SparseCoo, x: &[f64]) -> Vec<f64> {
    a.spmv(x).expect("sparse_matvec: dimension mismatch")
}

/// Conjugate gradient for a symmetric positive definite operator. `precond_diag` is an optional diagonal
/// (Jacobi) preconditioner given as the diagonal of `A`.
pub fn cg<F: Fn(&[f64]) -> Vec<f64>>(matvec: F, b: &[f64], x0: Option<&[f64]>, precond_diag: Option<&[f64]>, tol: f64, max_iter: usize) -> Result<Solve> {
    let n = b.len();
    if n == 0 || x0.is_some_and(|x| x.len() != n) || precond_diag.is_some_and(|d| d.len() != n || d.iter().any(|&v| v <= 0.0)) {
        return Err(Error::Msg("cg: inconsistent sizes or a non-positive preconditioner diagonal".into()));
    }
    let mut x = x0.map_or_else(|| vec![0.0; n], <[f64]>::to_vec);
    let ax = matvec(&x);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
    let apply_m = |v: &[f64]| -> Vec<f64> { precond_diag.map_or_else(|| v.to_vec(), |d| v.iter().zip(d).map(|(a, b)| a / b).collect()) };
    let mut z = apply_m(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let target = tol * norm(b).max(1e-300);
    for it in 0..max_iter {
        if norm(&r) <= target {
            return Ok(Solve { x, iterations: it, residual_norm: norm(&r), converged: true });
        }
        let ap = matvec(&p);
        let alpha = rz / dot(&p, &ap);
        x.iter_mut().zip(&p).for_each(|(xi, pi)| *xi += alpha * pi);
        r.iter_mut().zip(&ap).for_each(|(ri, ai)| *ri -= alpha * ai);
        z = apply_m(&r);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        p.iter_mut().zip(&z).for_each(|(pi, zi)| *pi = zi + beta * *pi);
    }
    let res = norm(&r);
    Ok(Solve { x, iterations: max_iter, residual_norm: res, converged: res <= target })
}

/// Restarted GMRES(m) for general non-singular operators.
pub fn gmres<F: Fn(&[f64]) -> Vec<f64>>(matvec: F, b: &[f64], restart: usize, tol: f64, max_restarts: usize) -> Result<Solve> {
    let n = b.len();
    if n == 0 || restart == 0 {
        return Err(Error::Msg("gmres: need a non-empty system and restart >= 1".into()));
    }
    let m = restart.min(n);
    let target = tol * norm(b).max(1e-300);
    let mut x = vec![0.0; n];
    let mut total = 0;
    for _ in 0..max_restarts.max(1) {
        let ax = matvec(&x);
        let r: Vec<f64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
        let beta = norm(&r);
        if beta <= target {
            return Ok(Solve { x, iterations: total, residual_norm: beta, converged: true });
        }
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|t| t / beta).collect()];
        let mut h = vec![vec![0.0; m]; m + 1];
        let (mut cs, mut sn) = (vec![0.0; m], vec![0.0; m]);
        let mut g = vec![0.0; m + 1];
        g[0] = beta;
        let mut k_done = 0;
        for j in 0..m {
            total += 1;
            let mut w = matvec(&v[j]);
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                let hij = h[i][j];
                w.iter_mut().zip(&v[i]).for_each(|(wi, vi)| *wi -= hij * vi);
            }
            h[j + 1][j] = norm(&w);
            let breakdown = h[j + 1][j] < 1e-14;
            if !breakdown {
                v.push(w.iter().map(|t| t / h[j + 1][j]).collect());
            }
            // apply previous Givens rotations, then form a new one
            for i in 0..j {
                let t = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = t;
            }
            let d = (h[j][j] * h[j][j] + h[j + 1][j] * h[j + 1][j]).sqrt().max(1e-300);
            cs[j] = h[j][j] / d;
            sn[j] = h[j + 1][j] / d;
            h[j][j] = d;
            h[j + 1][j] = 0.0;
            g[j + 1] = -sn[j] * g[j];
            g[j] *= cs[j];
            k_done = j + 1;
            if g[j + 1].abs() <= target || breakdown {
                break;
            }
        }
        // back-substitute for y, then update x
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            y[i] = (g[i] - (i + 1..k_done).map(|c| h[i][c] * y[c]).sum::<f64>()) / h[i][i];
        }
        for (j, yj) in y.iter().enumerate() {
            x.iter_mut().zip(&v[j]).for_each(|(xi, vi)| *xi += yj * vi);
        }
    }
    let ax = matvec(&x);
    let res = norm(&b.iter().zip(&ax).map(|(p, q)| p - q).collect::<Vec<_>>());
    Ok(Solve { x, iterations: total, residual_norm: res, converged: res <= target })
}

/// BiCGSTAB for general (possibly non-symmetric) operators.
pub fn bicgstab<F: Fn(&[f64]) -> Vec<f64>>(matvec: F, b: &[f64], tol: f64, max_iter: usize) -> Result<Solve> {
    let n = b.len();
    if n == 0 {
        return Err(Error::Msg("bicgstab: empty system".into()));
    }
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let r_hat = r.clone();
    let (mut rho, mut alpha, mut omega) = (1.0, 1.0, 1.0);
    let (mut v, mut p) = (vec![0.0; n], vec![0.0; n]);
    let target = tol * norm(b).max(1e-300);
    for it in 0..max_iter {
        if norm(&r) <= target {
            return Ok(Solve { x, iterations: it, residual_norm: norm(&r), converged: true });
        }
        let rho_new = dot(&r_hat, &r);
        if rho_new.abs() < 1e-300 {
            break;
        }
        let beta = rho_new / rho * alpha / omega;
        rho = rho_new;
        p.iter_mut().zip(r.iter().zip(&v)).for_each(|(pi, (ri, vi))| *pi = ri + beta * (*pi - omega * vi));
        v = matvec(&p);
        alpha = rho / dot(&r_hat, &v);
        let s: Vec<f64> = r.iter().zip(&v).map(|(ri, vi)| ri - alpha * vi).collect();
        if norm(&s) <= target {
            x.iter_mut().zip(&p).for_each(|(xi, pi)| *xi += alpha * pi);
            return Ok(Solve { x, iterations: it + 1, residual_norm: norm(&s), converged: true });
        }
        let t = matvec(&s);
        omega = dot(&t, &s) / dot(&t, &t).max(1e-300);
        x.iter_mut().zip(p.iter().zip(&s)).for_each(|(xi, (pi, si))| *xi += alpha * pi + omega * si);
        r = s.iter().zip(&t).map(|(si, ti)| si - omega * ti).collect();
    }
    let res = norm(&r);
    Ok(Solve { x, iterations: max_iter, residual_norm: res, converged: res <= target })
}

/// Dominant eigenpair by power iteration: `(eigenvalue, unit eigenvector)`.
pub fn power_iteration<F: Fn(&[f64]) -> Vec<f64>>(matvec: F, n: usize, tol: f64, max_iter: usize, seed: u64) -> Result<(f64, Vec<f64>)> {
    if n == 0 {
        return Err(Error::Msg("power_iteration: n must be positive".into()));
    }
    let mut rng = Rng::new(seed);
    let mut v: Vec<f64> = (0..n).map(|_| rng.next_f64() - 0.5).collect();
    let nv = norm(&v);
    v.iter_mut().for_each(|t| *t /= nv);
    let mut lambda = 0.0;
    for _ in 0..max_iter {
        let w = matvec(&v);
        let new_lambda = dot(&v, &w);
        let nw = norm(&w);
        if nw < 1e-300 {
            return Ok((0.0, v));
        }
        let next: Vec<f64> = w.iter().map(|t| t / nw).collect();
        let converged = (new_lambda - lambda).abs() < tol * new_lambda.abs().max(1.0);
        v = next;
        lambda = new_lambda;
        if converged {
            break;
        }
    }
    Ok((lambda, v))
}

/// Largest and smallest eigenvalues of a symmetric operator from `steps` Lanczos iterations (full
/// reorthogonalization): returns `(min, max)` of the Ritz values.
pub fn lanczos_extreme<F: Fn(&[f64]) -> Vec<f64>>(matvec: F, n: usize, steps: usize, seed: u64) -> Result<(f64, f64)> {
    if n == 0 || steps == 0 {
        return Err(Error::Msg("lanczos_extreme: need n >= 1 and steps >= 1".into()));
    }
    let m = steps.min(n);
    let mut rng = Rng::new(seed);
    let mut q: Vec<Vec<f64>> = Vec::new();
    let mut v: Vec<f64> = (0..n).map(|_| rng.next_f64() - 0.5).collect();
    let nv = norm(&v);
    v.iter_mut().for_each(|t| *t /= nv);
    q.push(v);
    let (mut alpha, mut beta): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for j in 0..m {
        let mut w = matvec(&q[j]);
        alpha.push(dot(&w, &q[j]));
        for qi in &q {
            let c = dot(&w, qi);
            w.iter_mut().zip(qi).for_each(|(wi, qv)| *wi -= c * qv);
        }
        let b = norm(&w);
        if j + 1 == m || b < 1e-12 {
            break;
        }
        beta.push(b);
        q.push(w.iter().map(|t| t / b).collect());
    }
    let k = alpha.len();
    let mut t: Matrix = vec![vec![0.0; k]; k];
    for i in 0..k {
        t[i][i] = alpha[i];
        if i + 1 < k {
            t[i][i + 1] = beta[i];
            t[i + 1][i] = beta[i];
        }
    }
    let (vals, _) = jacobi_eigh(&t);
    Ok((*vals.last().unwrap(), vals[0]))
}

/// Randomized truncated SVD (Halko et al.) of a dense matrix: `(U, singular values, V^T)` with `rank` components.
pub fn randomized_svd(a: &Matrix, rank: usize, oversample: usize, power_iters: usize, seed: u64) -> Result<(Matrix, Vec<f64>, Matrix)> {
    let (m, n) = (a.len(), a.first().map_or(0, Vec::len));
    if m == 0 || n == 0 || rank == 0 || rank > m.min(n) || a.iter().any(|r| r.len() != n) {
        return Err(Error::Msg("randomized_svd: need a rectangular matrix and 1 <= rank <= min(m, n)".into()));
    }
    let l = (rank + oversample).min(m.min(n));
    let mut rng = Rng::new(seed);
    let gauss = |rng: &mut Rng| { let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() };
    let mul = |x: &Matrix, y: &Matrix| -> Matrix { (0..x.len()).map(|i| (0..y[0].len()).map(|j| (0..y.len()).map(|k| x[i][k] * y[k][j]).sum()).collect()).collect() };
    let tr = |x: &Matrix| -> Matrix { (0..x[0].len()).map(|j| x.iter().map(|r| r[j]).collect()).collect() };
    let orth = |y: &Matrix| -> Matrix {
        // modified Gram-Schmidt on the columns of y
        let mut cols: Vec<Vec<f64>> = (0..y[0].len()).map(|j| y.iter().map(|r| r[j]).collect()).collect();
        let scale = cols.iter().map(|c| norm(c)).fold(0.0, f64::max).max(1e-300);
        for j in 0..cols.len() {
            // two Gram-Schmidt passes keep the basis orthonormal; (numerically) dependent columns become zero
            for _ in 0..2 {
                for i in 0..j {
                    let c = dot(&cols[j], &cols[i]);
                    let ci = cols[i].clone();
                    cols[j].iter_mut().zip(&ci).for_each(|(a, b)| *a -= c * b);
                }
            }
            let nj = norm(&cols[j]);
            if nj < 1e-10 * scale {
                cols[j].iter_mut().for_each(|t| *t = 0.0);
            } else {
                cols[j].iter_mut().for_each(|t| *t /= nj);
            }
        }
        (0..y.len()).map(|i| cols.iter().map(|c| c[i]).collect()).collect()
    };
    let omega: Matrix = (0..n).map(|_| (0..l).map(|_| gauss(&mut rng)).collect()).collect();
    let mut q = orth(&mul(a, &omega));
    let at = tr(a);
    for _ in 0..power_iters {
        q = orth(&mul(a, &orth(&mul(&at, &q))));
    }
    let b = mul(&tr(&q), a); // l x n
    // eigen-decompose B B^T (l x l) to get the left factors of the small matrix
    let bbt = mul(&b, &tr(&b));
    let (vals, vecs) = jacobi_eigh(&bbt);
    let sing: Vec<f64> = vals.iter().take(rank).map(|v| v.max(0.0).sqrt()).collect();
    let ub: Matrix = (0..l).map(|i| (0..rank).map(|k| vecs[k][i]).collect()).collect(); // l x rank
    let u = mul(&q, &ub);
    let vt: Matrix = (0..rank).map(|k| { let col: Vec<f64> = (0..l).map(|i| ub[i][k]).collect(); (0..n).map(|j| (0..l).map(|i| col[i] * b[i][j]).sum::<f64>() / sing[k].max(1e-300)).collect() }).collect();
    Ok((u, sing, vt))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tridiagonal SPD matrix: 2 on the diagonal, -1 off it (1-D Laplacian).
    fn laplacian(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n).map(|i| 2.0 * x[i] - if i > 0 { x[i - 1] } else { 0.0 } - if i + 1 < n { x[i + 1] } else { 0.0 }).collect()
    }

    #[test]
    fn cg_solves_the_laplacian() {
        let n = 200;
        let truth: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let b = laplacian(&truth);
        let s = cg(laplacian, &b, None, None, 1e-10, 1000).unwrap();
        assert!(s.converged && s.iterations <= n + 5, "{} iterations", s.iterations);
        assert!(s.x.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-6));
        // Jacobi preconditioning is valid (diagonal of A is 2)
        let sp = cg(laplacian, &b, None, Some(&vec![2.0; n]), 1e-10, 1000).unwrap();
        assert!(sp.converged);
        // a good warm start needs fewer iterations
        let warm = cg(laplacian, &b, Some(&truth), None, 1e-10, 1000).unwrap();
        assert!(warm.iterations < 3);
        assert!(cg(laplacian, &b, Some(&[1.0]), None, 1e-6, 10).is_err());
    }

    #[test]
    fn general_solvers_handle_non_symmetric_systems() {
        // convection-diffusion style non-symmetric tridiagonal operator
        let op = |x: &[f64]| -> Vec<f64> { let n = x.len(); (0..n).map(|i| 3.0 * x[i] - 1.5 * if i > 0 { x[i - 1] } else { 0.0 } - 0.5 * if i + 1 < n { x[i + 1] } else { 0.0 }).collect() };
        let truth: Vec<f64> = (0..100).map(|i| ((i % 7) as f64) - 3.0).collect();
        let b = op(&truth);
        let g = gmres(op, &b, 30, 1e-10, 50).unwrap();
        assert!(g.converged && g.x.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-6), "gmres residual {}", g.residual_norm);
        let s = bicgstab(op, &b, 1e-10, 500).unwrap();
        assert!(s.converged && s.x.iter().zip(&truth).all(|(a, b)| (a - b).abs() < 1e-6), "bicgstab residual {}", s.residual_norm);
        assert!(gmres(op, &[], 5, 1e-6, 5).is_err() && bicgstab(op, &[], 1e-6, 5).is_err());
    }

    #[test]
    fn works_with_sparse_matrices() {
        let n = 50;
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for i in 0..n {
            idx.push((i, i));
            val.push(2.0);
            if i + 1 < n {
                idx.push((i, i + 1));
                val.push(-1.0);
                idx.push((i + 1, i));
                val.push(-1.0);
            }
        }
        let a = SparseCoo::new(idx, val, [n, n]).unwrap();
        let truth: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
        let b = sparse_matvec(&a, &truth);
        let s = cg(|v| sparse_matvec(&a, v), &b, None, None, 1e-12, 500).unwrap();
        assert!(s.x.iter().zip(&truth).all(|(p, q)| (p - q).abs() < 1e-6));
    }

    #[test]
    fn eigen_estimates_match_the_analytic_spectrum() {
        // 1-D Laplacian of size n has eigenvalues 2 - 2 cos(k pi / (n + 1)), k = 1..n
        let n = 60;
        let (lo, hi) = (2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos(), 2.0 - 2.0 * (n as f64 * std::f64::consts::PI / (n as f64 + 1.0)).cos());
        let (lmin, lmax) = lanczos_extreme(laplacian, n, 60, 1).unwrap();
        assert!((lmax - hi).abs() < 1e-8 && (lmin - lo).abs() < 1e-8, "{lmin} {lmax} vs {lo} {hi}");
        let (p, v) = power_iteration(laplacian, n, 1e-12, 20_000, 2).unwrap();
        assert!((p - hi).abs() < 1e-3, "{p} vs {hi}");
        assert!((norm(&v) - 1.0).abs() < 1e-9);
        assert!(lanczos_extreme(laplacian, 0, 5, 0).is_err() && power_iteration(laplacian, 0, 1e-6, 5, 0).is_err());
    }

    #[test]
    fn randomized_svd_recovers_low_rank_structure() {
        // exact rank-3 matrix: sigma = 5, 3, 1 with orthogonal-ish factors
        let mut rng = Rng::new(4);
        let u: Matrix = (0..40).map(|_| (0..3).map(|_| rng.next_f64() - 0.5).collect()).collect();
        let v: Matrix = (0..30).map(|_| (0..3).map(|_| rng.next_f64() - 0.5).collect()).collect();
        let a: Matrix = (0..40).map(|i| (0..30).map(|j| (0..3).map(|k| u[i][k] * v[j][k] * [5.0, 3.0, 1.0][k]).sum()).collect()).collect();
        let (uu, s, vt) = randomized_svd(&a, 3, 5, 2, 1).unwrap();
        // reconstruction is (near) exact
        let rec = |i: usize, j: usize| (0..3).map(|k| uu[i][k] * s[k] * vt[k][j]).sum::<f64>();
        let err = (0..40).flat_map(|i| (0..30).map(move |j| (i, j))).map(|(i, j)| (rec(i, j) - a[i][j]).powi(2)).sum::<f64>().sqrt();
        let nrm = a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err / nrm < 1e-6, "relative error {}", err / nrm);
        assert!(s[0] >= s[1] && s[1] >= s[2]);
        assert!(randomized_svd(&a, 0, 5, 1, 0).is_err() && randomized_svd(&a, 31, 5, 1, 0).is_err());
    }
}
