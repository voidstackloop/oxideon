//! Numerical analysis helpers on `f64` functions: [`RegularGrid`] N-d multilinear interpolation, barycentric
//! [`lagrange_eval`] and Chebyshev interpolation ([`chebyshev_nodes`], [`ChebyshevInterp`]), finite-difference
//! derivatives ([`derivative`], [`gradient_fd`], [`jacobian_fd`], [`hessian_fd`]), and [`richardson_derivative`].

use crate::classical::Matrix;
use crate::error::{Error, Result};

/// Multilinear interpolation on a rectilinear grid: `axes[d]` are strictly increasing coordinates and `values` is
/// stored row-major (last axis fastest).
#[derive(Debug, Clone)]
pub struct RegularGrid {
    axes: Vec<Vec<f64>>,
    values: Vec<f64>,
    strides: Vec<usize>,
}

impl RegularGrid {
    pub fn new(axes: Vec<Vec<f64>>, values: Vec<f64>) -> Result<Self> {
        if axes.is_empty() || axes.iter().any(|a| a.len() < 2 || a.windows(2).any(|w| w[1] <= w[0])) {
            return Err(Error::Msg("RegularGrid: every axis needs >= 2 strictly increasing coordinates".into()));
        }
        let total: usize = axes.iter().map(Vec::len).product();
        if values.len() != total {
            return Err(Error::Msg(format!("RegularGrid: expected {total} values, got {}", values.len())));
        }
        let mut strides = vec![1; axes.len()];
        for d in (0..axes.len() - 1).rev() {
            strides[d] = strides[d + 1] * axes[d + 1].len();
        }
        Ok(Self { axes, values, strides })
    }

    /// Interpolated value at `point`; points outside the grid are clamped to its boundary.
    pub fn eval(&self, point: &[f64]) -> Result<f64> {
        if point.len() != self.axes.len() {
            return Err(Error::Msg("RegularGrid::eval: point dimension does not match the grid".into()));
        }
        let (mut lo, mut frac) = (Vec::new(), Vec::new());
        for (d, &p) in point.iter().enumerate() {
            let ax = &self.axes[d];
            let p = p.clamp(ax[0], ax[ax.len() - 1]);
            let i = ax.partition_point(|&v| v <= p).clamp(1, ax.len() - 1) - 1;
            lo.push(i);
            frac.push((p - ax[i]) / (ax[i + 1] - ax[i]));
        }
        let n = self.axes.len();
        let mut total = 0.0;
        for corner in 0..(1usize << n) {
            let (mut w, mut off) = (1.0, 0);
            for d in 0..n {
                let up = corner >> d & 1;
                w *= if up == 1 { frac[d] } else { 1.0 - frac[d] };
                off += (lo[d] + up) * self.strides[d];
            }
            if w != 0.0 {
                total += w * self.values[off];
            }
        }
        Ok(total)
    }
}

/// Lagrange interpolation through `(xs, ys)` evaluated with the barycentric formula.
pub fn lagrange_eval(xs: &[f64], ys: &[f64], x: f64) -> Result<f64> {
    if xs.len() != ys.len() || xs.is_empty() {
        return Err(Error::Msg("lagrange_eval: need matching non-empty xs and ys".into()));
    }
    let n = xs.len();
    let w: Vec<f64> = (0..n).map(|i| 1.0 / (0..n).filter(|&j| j != i).map(|j| xs[i] - xs[j]).product::<f64>()).collect();
    if let Some(i) = xs.iter().position(|&v| v == x) {
        return Ok(ys[i]);
    }
    let (num, den) = (0..n).fold((0.0, 0.0), |(a, b), i| (a + w[i] * ys[i] / (x - xs[i]), b + w[i] / (x - xs[i])));
    Ok(num / den)
}

/// Chebyshev points of the first kind on `[a, b]` (`n` nodes).
pub fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n).map(|k| 0.5 * (a + b) + 0.5 * (b - a) * ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * n) as f64).cos()).collect()
}

/// Chebyshev interpolant of a function on `[a, b]` (degree `n - 1`), evaluated with Clenshaw's recurrence.
#[derive(Debug, Clone)]
pub struct ChebyshevInterp {
    coef: Vec<f64>,
    a: f64,
    b: f64,
}

impl ChebyshevInterp {
    pub fn fit<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> Result<Self> {
        if n == 0 || a >= b {
            return Err(Error::Msg("ChebyshevInterp: need n >= 1 and a < b".into()));
        }
        let nodes = chebyshev_nodes(n, a, b);
        let fv: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();
        let coef = (0..n)
            .map(|j| {
                let s: f64 = (0..n).map(|k| fv[k] * (j as f64 * (2 * k + 1) as f64 * std::f64::consts::PI / (2 * n) as f64).cos()).sum();
                s * if j == 0 { 1.0 } else { 2.0 } / n as f64
            })
            .collect();
        Ok(Self { coef, a, b })
    }

    pub fn eval(&self, x: f64) -> f64 {
        let t = (2.0 * x - self.a - self.b) / (self.b - self.a);
        let (mut b1, mut b2) = (0.0, 0.0);
        for &c in self.coef.iter().skip(1).rev() {
            let b0 = 2.0 * t * b1 - b2 + c;
            b2 = b1;
            b1 = b0;
        }
        t * b1 - b2 + self.coef[0]
    }

    /// Derivative of the interpolant (as a new interpolant).
    pub fn derivative(&self) -> Self {
        let n = self.coef.len();
        let mut d = vec![0.0; n.max(2)];
        for k in (1..n).rev() {
            d[k - 1] = d.get(k + 1).copied().unwrap_or(0.0) + 2.0 * k as f64 * self.coef[k];
        }
        d[0] *= 0.5;
        // coefficients are with respect to t in [-1, 1]; chain rule 2 / (b - a)
        let scale = 2.0 / (self.b - self.a);
        d.truncate(n.saturating_sub(1).max(1));
        d.iter_mut().for_each(|c| *c *= scale);
        Self { coef: d, a: self.a, b: self.b }
    }
}

/// Central-difference first derivative with step `h` (error `O(h^2)`).
pub fn derivative<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Richardson-extrapolated derivative (the classic `D(h) -> D(h/2)` table), error `O(h^(2k))`.
pub fn richardson_derivative<F: Fn(f64) -> f64>(f: F, x: f64, h: f64, levels: usize) -> f64 {
    let levels = levels.max(1);
    let mut t = vec![vec![0.0; levels]; levels];
    for i in 0..levels {
        t[i][0] = derivative(&f, x, h / 2f64.powi(i as i32));
        for j in 1..=i {
            t[i][j] = t[i][j - 1] + (t[i][j - 1] - t[i - 1][j - 1]) / (4f64.powi(j as i32) - 1.0);
        }
    }
    t[levels - 1][levels - 1]
}

/// Gradient of a scalar function by central differences.
pub fn gradient_fd<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], h: f64) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let (mut p, mut m) = (x.to_vec(), x.to_vec());
            p[i] += h;
            m[i] -= h;
            (f(&p) - f(&m)) / (2.0 * h)
        })
        .collect()
}

/// Jacobian (`outputs x inputs`) of a vector function by central differences.
pub fn jacobian_fd<F: Fn(&[f64]) -> Vec<f64>>(f: F, x: &[f64], h: f64) -> Matrix {
    let cols: Vec<Vec<f64>> = (0..x.len())
        .map(|i| {
            let (mut p, mut m) = (x.to_vec(), x.to_vec());
            p[i] += h;
            m[i] -= h;
            f(&p).iter().zip(f(&m)).map(|(a, b)| (a - b) / (2.0 * h)).collect()
        })
        .collect();
    (0..cols.first().map_or(0, Vec::len)).map(|r| cols.iter().map(|c| c[r]).collect()).collect()
}

/// Hessian of a scalar function by central differences (symmetric by construction).
pub fn hessian_fd<F: Fn(&[f64]) -> f64>(f: F, x: &[f64], h: f64) -> Matrix {
    let n = x.len();
    let at = |di: &[(usize, f64)]| -> f64 {
        let mut p = x.to_vec();
        di.iter().for_each(|&(i, s)| p[i] += s * h);
        f(&p)
    };
    let mut hess = vec![vec![0.0; n]; n];
    for i in 0..n {
        hess[i][i] = (at(&[(i, 1.0)]) - 2.0 * f(x) + at(&[(i, -1.0)])) / (h * h);
        for j in i + 1..n {
            let v = (at(&[(i, 1.0), (j, 1.0)]) - at(&[(i, 1.0), (j, -1.0)]) - at(&[(i, -1.0), (j, 1.0)]) + at(&[(i, -1.0), (j, -1.0)])) / (4.0 * h * h);
            hess[i][j] = v;
            hess[j][i] = v;
        }
    }
    hess
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn grid_interpolation_is_exact_for_multilinear_functions() {
        // f(x, y) = 2x + 3y + xy + 1 is multilinear, so interpolation reproduces it everywhere
        let xs = vec![0.0, 1.0, 3.0];
        let ys = vec![-1.0, 0.5, 2.0, 4.0];
        let f = |x: f64, y: f64| 2.0 * x + 3.0 * y + x * y + 1.0;
        let values: Vec<f64> = xs.iter().flat_map(|&x| ys.iter().map(move |&y| f(x, y))).collect();
        let g = RegularGrid::new(vec![xs, ys], values).unwrap();
        for (x, y) in [(0.5, 1.0), (2.2, 3.3), (0.0, -1.0), (3.0, 4.0), (1.0, 0.5)] {
            close(g.eval(&[x, y]).unwrap(), f(x, y), 1e-12);
        }
        close(g.eval(&[10.0, 1.0]).unwrap(), f(3.0, 1.0), 1e-12); // clamped
        assert!(g.eval(&[1.0]).is_err());
        assert!(RegularGrid::new(vec![vec![0.0, 0.0]], vec![1.0, 2.0]).is_err() && RegularGrid::new(vec![vec![0.0, 1.0]], vec![1.0]).is_err());
        // 3-D
        let g3 = RegularGrid::new(vec![vec![0.0, 1.0]; 3], (0..8).map(|i| i as f64).collect()).unwrap();
        close(g3.eval(&[0.5, 0.5, 0.5]).unwrap(), 3.5, 1e-12);
    }

    #[test]
    fn polynomial_and_chebyshev_interpolation() {
        // Lagrange reproduces a cubic from four points
        let xs = [0.0, 1.0, 2.0, 5.0];
        let ys: Vec<f64> = xs.iter().map(|x: &f64| x.powi(3) - 2.0 * x + 1.0).collect();
        close(lagrange_eval(&xs, &ys, 3.3).unwrap(), 3.3f64.powi(3) - 6.6 + 1.0, 1e-9);
        close(lagrange_eval(&xs, &ys, 2.0).unwrap(), ys[2], 1e-12);
        // Runge's function: Chebyshev nodes converge where equispaced Lagrange does not
        let runge = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
        let ch = ChebyshevInterp::fit(runge, -1.0, 1.0, 40).unwrap();
        let worst = (0..=200).map(|i| (ch.eval(-1.0 + i as f64 * 0.01) - runge(-1.0 + i as f64 * 0.01)).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-3, "{worst}");
        let eq: Vec<f64> = (0..21).map(|i| -1.0 + i as f64 * 0.1).collect();
        let eq_y: Vec<f64> = eq.iter().map(|&x| runge(x)).collect();
        assert!(lagrange_eval(&eq, &eq_y, 0.95).unwrap().is_finite());
        assert!((lagrange_eval(&eq, &eq_y, 0.95).unwrap() - runge(0.95)).abs() > 10.0 * worst, "equispaced interpolation oscillates");
        // exactness on a polynomial and on a shifted interval
        let sh = ChebyshevInterp::fit(|x| 3.0 * x * x - x + 2.0, 1.0, 4.0, 5).unwrap();
        close(sh.eval(2.5), 3.0 * 6.25 - 2.5 + 2.0, 1e-9);
        close(sh.derivative().eval(2.5), 6.0 * 2.5 - 1.0, 1e-9);
        close(ChebyshevInterp::fit(f64::sin, 0.0, 3.0, 20).unwrap().derivative().eval(1.0), 1f64.cos(), 1e-9);
        assert!(ChebyshevInterp::fit(runge, 1.0, 1.0, 5).is_err() && lagrange_eval(&[], &[], 0.0).is_err());
    }

    #[test]
    fn finite_differences() {
        close(derivative(f64::sin, 1.0, 1e-5), 1f64.cos(), 1e-9);
        // Richardson reaches near machine precision with a coarse step
        close(richardson_derivative(f64::exp, 0.5, 0.5, 5), 0.5f64.exp(), 1e-9);
        assert!((richardson_derivative(f64::exp, 0.5, 0.5, 5) - 0.5f64.exp()).abs() < (derivative(f64::exp, 0.5, 0.5) - 0.5f64.exp()).abs() * 1e-3);
        let f = |p: &[f64]| p[0] * p[0] * p[1] + (p[1]).sin();
        let g = gradient_fd(f, &[2.0, 0.5], 1e-5);
        close(g[0], 2.0, 1e-8);
        close(g[1], 4.0 + 0.5f64.cos(), 1e-8);
        let h = hessian_fd(f, &[2.0, 0.5], 1e-4);
        close(h[0][0], 1.0, 1e-5);
        close(h[0][1], 4.0, 1e-5);
        close(h[1][0], h[0][1], 1e-12);
        close(h[1][1], -(0.5f64.sin()), 1e-5);
        let jac = jacobian_fd(|p| vec![p[0] * p[1], p[0] + p[1].powi(2)], &[2.0, 3.0], 1e-6);
        close(jac[0][0], 3.0, 1e-8);
        close(jac[0][1], 2.0, 1e-8);
        close(jac[1][0], 1.0, 1e-8);
        close(jac[1][1], 6.0, 1e-7);
    }
}
