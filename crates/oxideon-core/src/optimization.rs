//! Mathematical programming: [`linprog`] (two-phase simplex with Bland's rule, `x >= 0`), non-negative least squares
//! ([`nnls`], Lawson-Hanson), and the [`assignment`] problem (Hungarian algorithm).

use crate::error::{Error, Result};

/// Outcome of [`linprog`].
#[derive(Debug, Clone, PartialEq)]
pub enum LpStatus {
    Optimal,
    Infeasible,
    Unbounded,
}

#[derive(Debug, Clone)]
pub struct LpResult {
    pub status: LpStatus,
    pub x: Vec<f64>,
    pub objective: f64,
}

const EPS: f64 = 1e-9;

/// Minimizes `c . x` subject to `a_ub x <= b_ub`, `a_eq x = b_eq`, `x >= 0`.
pub fn linprog(c: &[f64], a_ub: &[Vec<f64>], b_ub: &[f64], a_eq: &[Vec<f64>], b_eq: &[f64]) -> Result<LpResult> {
    let n = c.len();
    if a_ub.len() != b_ub.len() || a_eq.len() != b_eq.len() || a_ub.iter().chain(a_eq).any(|r| r.len() != n) || n == 0 {
        return Err(Error::Msg("linprog: constraint dimensions do not match c".into()));
    }
    let (m_ub, m_eq) = (a_ub.len(), a_eq.len());
    let m = m_ub + m_eq;
    // columns: x (n), slack (m_ub), artificial (m); rows normalized to rhs >= 0
    let width = n + m_ub + m;
    let mut t: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut basis = Vec::with_capacity(m);
    for i in 0..m {
        let (row, rhs, slack) = if i < m_ub { (&a_ub[i], b_ub[i], Some(i)) } else { (&a_eq[i - m_ub], b_eq[i - m_ub], None) };
        let mut r = vec![0.0; width + 1];
        r[..n].copy_from_slice(row);
        if let Some(s) = slack {
            r[n + s] = 1.0;
        }
        r[width] = rhs;
        if rhs < 0.0 {
            r.iter_mut().for_each(|v| *v = -*v);
        }
        r[n + m_ub + i] = 1.0;
        basis.push(n + m_ub + i);
        t.push(r);
    }
    // phase 1: minimize the sum of artificial variables
    let mut cost1 = vec![0.0; width + 1];
    (n + m_ub..width).for_each(|j| cost1[j] = 1.0);
    let mut obj = reduced_costs(&t, &basis, &cost1);
    if !simplex(&mut t, &mut basis, &mut obj, width)? {
        return Ok(LpResult { status: LpStatus::Unbounded, x: vec![0.0; n], objective: f64::NEG_INFINITY });
    }
    if -obj[width] > 1e-7 {
        return Ok(LpResult { status: LpStatus::Infeasible, x: vec![0.0; n], objective: f64::NAN });
    }
    // drive remaining artificial variables out of the basis (degenerate rows)
    for i in 0..m {
        if basis[i] >= n + m_ub {
            if let Some(j) = (0..n + m_ub).find(|&j| t[i][j].abs() > EPS) {
                pivot(&mut t, &mut obj, i, j);
                basis[i] = j;
            }
        }
    }
    // phase 2: real objective, artificial columns forbidden
    let mut cost2 = vec![0.0; width + 1];
    cost2[..n].copy_from_slice(c);
    let mut obj2 = reduced_costs(&t, &basis, &cost2);
    (n + m_ub..width).for_each(|j| obj2[j] = f64::INFINITY);
    if !simplex(&mut t, &mut basis, &mut obj2, width)? {
        return Ok(LpResult { status: LpStatus::Unbounded, x: vec![0.0; n], objective: f64::NEG_INFINITY });
    }
    let mut x = vec![0.0; n];
    for (i, &b) in basis.iter().enumerate() {
        if b < n {
            x[b] = t[i][width];
        }
    }
    Ok(LpResult { status: LpStatus::Optimal, objective: c.iter().zip(&x).map(|(a, b)| a * b).sum(), x })
}

fn reduced_costs(t: &[Vec<f64>], basis: &[usize], cost: &[f64]) -> Vec<f64> {
    let mut r = cost.to_vec();
    for (i, &b) in basis.iter().enumerate() {
        let cb = cost[b];
        if cb != 0.0 {
            r.iter_mut().zip(&t[i]).for_each(|(v, a)| *v -= cb * a);
        }
    }
    r
}

fn pivot(t: &mut [Vec<f64>], obj: &mut [f64], row: usize, col: usize) {
    let p = t[row][col];
    t[row].iter_mut().for_each(|v| *v /= p);
    let pr = t[row].clone();
    for (i, r) in t.iter_mut().enumerate() {
        if i != row && r[col].abs() > 0.0 {
            let w = r[col];
            r.iter_mut().zip(&pr).for_each(|(v, a)| *v -= w * a);
        }
    }
    let w = obj[col];
    if w.is_finite() {
        obj.iter_mut().zip(&pr).for_each(|(v, a)| if v.is_finite() { *v -= w * a });
    }
}

/// Runs simplex iterations (Bland's rule); `false` means unbounded.
fn simplex(t: &mut [Vec<f64>], basis: &mut [usize], obj: &mut [f64], width: usize) -> Result<bool> {
    for _ in 0..10_000 {
        let Some(col) = (0..width).find(|&j| obj[j] < -EPS) else { return Ok(true) };
        let mut best: Option<(f64, usize)> = None;
        for (i, r) in t.iter().enumerate() {
            if r[col] > EPS {
                let ratio = r[width] / r[col];
                if best.is_none_or(|(b, bi)| ratio < b - EPS || (ratio < b + EPS && basis[i] < basis[bi])) {
                    best = Some((ratio, i));
                }
            }
        }
        let Some((_, row)) = best else { return Ok(false) };
        pivot(t, obj, row, col);
        basis[row] = col;
    }
    Err(Error::Msg("linprog: simplex did not converge".into()))
}

fn gauss_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-14 {
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

/// Non-negative least squares `min ||A x - b||` s.t. `x >= 0` (Lawson-Hanson active-set method).
pub fn nnls(a: &[Vec<f64>], b: &[f64]) -> Result<(Vec<f64>, f64)> {
    let (m, n) = (a.len(), a.first().map_or(0, Vec::len));
    if m == 0 || n == 0 || b.len() != m || a.iter().any(|r| r.len() != n) {
        return Err(Error::Msg("nnls: dimension mismatch".into()));
    }
    let resid = |x: &[f64]| -> Vec<f64> { (0..m).map(|i| b[i] - a[i].iter().zip(x).map(|(p, q)| p * q).sum::<f64>()).collect() };
    let mut x = vec![0.0; n];
    let mut passive: Vec<usize> = Vec::new();
    for _ in 0..(3 * n + 10) {
        let r = resid(&x);
        let w: Vec<f64> = (0..n).map(|j| (0..m).map(|i| a[i][j] * r[i]).sum()).collect();
        let Some(t) = (0..n).filter(|j| !passive.contains(j) && w[*j] > 1e-10).max_by(|&i, &j| w[i].total_cmp(&w[j])) else { break };
        passive.push(t);
        loop {
            let k = passive.len();
            let ata: Vec<Vec<f64>> = (0..k).map(|p| (0..k).map(|q| (0..m).map(|i| a[i][passive[p]] * a[i][passive[q]]).sum()).collect()).collect();
            let atb: Vec<f64> = (0..k).map(|p| (0..m).map(|i| a[i][passive[p]] * b[i]).sum()).collect();
            let Some(s) = gauss_solve(ata, atb) else {
                passive.pop();
                break;
            };
            if s.iter().all(|&v| v > 1e-12) {
                x = vec![0.0; n];
                passive.iter().zip(&s).for_each(|(&j, &v)| x[j] = v);
                break;
            }
            // step toward s until the first passive variable hits zero, then drop the zeros
            let alpha = passive.iter().zip(&s).filter(|(_, v)| **v <= 1e-12).map(|(&j, &v)| x[j] / (x[j] - v)).fold(f64::INFINITY, f64::min);
            for (p, &j) in passive.iter().enumerate() {
                x[j] += alpha * (s[p] - x[j]);
            }
            passive.retain(|&j| x[j] > 1e-12);
            (0..n).filter(|j| !passive.contains(j)).for_each(|j| x[j] = 0.0);
            if passive.is_empty() {
                break;
            }
        }
    }
    let norm = resid(&x).iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok((x, norm))
}

/// Minimum-cost assignment of rows to columns of a square cost matrix (Hungarian algorithm).
/// Returns `(column for each row, total cost)`.
pub fn assignment(cost: &[Vec<f64>]) -> Result<(Vec<usize>, f64)> {
    let n = cost.len();
    if n == 0 || cost.iter().any(|r| r.len() != n) {
        return Err(Error::Msg("assignment: need a non-empty square cost matrix".into()));
    }
    let (mut u, mut v) = (vec![0.0; n + 1], vec![0.0; n + 1]);
    let (mut p, mut way) = (vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let (mut minv, mut used) = (vec![f64::INFINITY; n + 1], vec![false; n + 1]);
        loop {
            used[j0] = true;
            let (i0, mut delta, mut j1) = (p[j0], f64::INFINITY, 0);
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut cols = vec![0; n];
    for j in 1..=n {
        cols[p[j] - 1] = j - 1;
    }
    let total = cols.iter().enumerate().map(|(i, &j)| cost[i][j]).sum();
    Ok((cols, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-7, "{a} vs {b}");
    }

    #[test]
    fn linprog_textbook_problems() {
        // maximize 3x + 5y s.t. x <= 4, 2y <= 12, 3x + 2y <= 18  -> (2, 6), value 36
        let r = linprog(&[-3.0, -5.0], &[vec![1.0, 0.0], vec![0.0, 2.0], vec![3.0, 2.0]], &[4.0, 12.0, 18.0], &[], &[]).unwrap();
        assert_eq!(r.status, LpStatus::Optimal);
        close(r.x[0], 2.0);
        close(r.x[1], 6.0);
        close(r.objective, -36.0);
        // with an equality: minimize x + 2y s.t. x + y = 4, x <= 3 -> x = 3, y = 1, value 5
        let r = linprog(&[1.0, 2.0], &[vec![1.0, 0.0]], &[3.0], &[vec![1.0, 1.0]], &[4.0]).unwrap();
        assert_eq!(r.status, LpStatus::Optimal);
        close(r.objective, 5.0);
        // >= constraint via negation: minimize x s.t. -x <= -2 -> x = 2 (needs phase 1)
        close(linprog(&[1.0], &[vec![-1.0]], &[-2.0], &[], &[]).unwrap().x[0], 2.0);
        // infeasible: x <= 1 and x >= 2
        assert_eq!(linprog(&[1.0], &[vec![1.0], vec![-1.0]], &[1.0, -2.0], &[], &[]).unwrap().status, LpStatus::Infeasible);
        // unbounded: minimize -x with no upper limit
        assert_eq!(linprog(&[-1.0], &[vec![-1.0]], &[0.0], &[], &[]).unwrap().status, LpStatus::Unbounded);
        assert!(linprog(&[1.0], &[vec![1.0, 2.0]], &[1.0], &[], &[]).is_err());
    }

    #[test]
    fn nnls_matches_known_solutions() {
        // unconstrained solution is already non-negative: exact fit
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let (x, r) = nnls(&a, &[1.0, 2.0, 3.0]).unwrap();
        close(x[0], 1.0);
        close(x[1], 2.0);
        assert!(r < 1e-9);
        // the unconstrained fit would be negative in x1; nnls clamps it to 0
        let (x, _) = nnls(&[vec![1.0, 1.0], vec![1.0, 2.0]], &[2.0, 1.0]).unwrap();
        assert!(x.iter().all(|&v| v >= 0.0));
        close(x[1], 0.0);
        close(x[0], 1.5);
        assert!(nnls(&a, &[1.0]).is_err());
    }

    #[test]
    fn hungarian_finds_the_cheapest_assignment() {
        let cost = vec![vec![4.0, 1.0, 3.0], vec![2.0, 0.0, 5.0], vec![3.0, 2.0, 2.0]];
        let (cols, total) = assignment(&cost).unwrap();
        close(total, 5.0); // 1 + 2 + 2: rows -> (1, 0, 2)
        assert_eq!(cols, vec![1, 0, 2]);
        // brute force agrees on a random 5x5
        let mut s = 7u64;
        let rnd: Vec<Vec<f64>> = (0..5).map(|_| (0..5).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 40) as f64 % 20.0 }).collect()).collect();
        let (_, t) = assignment(&rnd).unwrap();
        fn brute(c: &[Vec<f64>], row: usize, used: &mut Vec<bool>) -> f64 {
            if row == c.len() {
                return 0.0;
            }
            let mut best = f64::INFINITY;
            for j in 0..c.len() {
                if !used[j] {
                    used[j] = true;
                    best = best.min(c[row][j] + brute(c, row + 1, used));
                    used[j] = false;
                }
            }
            best
        }
        close(t, brute(&rnd, 0, &mut vec![false; 5]));
        assert!(assignment(&[vec![1.0, 2.0]]).is_err());
    }
}
