//! SciPy-style scientific routines on plain `f64`: root finding ([`bisect`], [`brentq`], [`newton`]), minimization
//! ([`nelder_mead`], [`bfgs`], [`brent_min`]), integration ([`quad`] adaptive Gauss-Kronrod, [`simpson`],
//! [`romberg`]), natural [`CubicSpline`] interpolation, and nonlinear least squares ([`curve_fit`], Levenberg-Marquardt).

use crate::error::{Error, Result};

// ------------------------------------------------------------------ root finding

/// Bisection on a bracketing interval.
pub fn bisect<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> Result<f64> {
    let (mut fa, fb) = (f(a), f(b));
    if fa * fb > 0.0 {
        return Err(Error::Msg("bisect: f(a) and f(b) must have opposite signs".into()));
    }
    for _ in 0..200 {
        let m = 0.5 * (a + b);
        let fm = f(m);
        if fm == 0.0 || 0.5 * (b - a) < tol {
            return Ok(m);
        }
        if fa * fm < 0.0 {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    Ok(0.5 * (a + b))
}

/// Brent's method: bracketing root finder combining bisection, secant and inverse quadratic interpolation.
pub fn brentq<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64) -> Result<f64> {
    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    if fa * fb > 0.0 {
        return Err(Error::Msg("brentq: f(a) and f(b) must have opposite signs".into()));
    }
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let (mut c, mut fc, mut d) = (a, fa, a);
    let mut mflag = true;
    for _ in 0..200 {
        if fb == 0.0 || (b - a).abs() < tol {
            return Ok(b);
        }
        let mut s = if fa != fc && fb != fc {
            a * fb * fc / ((fa - fb) * (fa - fc)) + b * fa * fc / ((fb - fa) * (fb - fc)) + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            b - fb * (b - a) / (fb - fa)
        };
        let lo = (3.0 * a + b) / 4.0;
        let (l, h) = if lo < b { (lo, b) } else { (b, lo) };
        if !(s > l && s < h) || (mflag && (s - b).abs() >= (b - c).abs() / 2.0) || (!mflag && (s - b).abs() >= (c - d).abs() / 2.0) || (mflag && (b - c).abs() < tol) || (!mflag && (c - d).abs() < tol) {
            s = 0.5 * (a + b);
            mflag = true;
        } else {
            mflag = false;
        }
        let fs = f(s);
        d = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
    Ok(b)
}

/// Newton-Raphson with derivative `df`.
pub fn newton<F: Fn(f64) -> f64, D: Fn(f64) -> f64>(f: F, df: D, x0: f64, tol: f64) -> Result<f64> {
    let mut x = x0;
    for _ in 0..100 {
        let (fx, d) = (f(x), df(x));
        if d == 0.0 {
            return Err(Error::Msg("newton: zero derivative".into()));
        }
        let step = fx / d;
        x -= step;
        if step.abs() < tol {
            return Ok(x);
        }
    }
    Err(Error::Msg("newton: did not converge in 100 iterations".into()))
}

// ------------------------------------------------------------------ minimization

/// Golden-section / parabolic-free minimization of a unimodal function on `[a, b]`.
pub fn brent_min<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64) -> (f64, f64) {
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    let x = 0.5 * (a + b);
    (x, f(x))
}

/// Nelder-Mead simplex minimization; returns `(x, f(x))`.
pub fn nelder_mead<F: Fn(&[f64]) -> f64>(f: F, x0: &[f64], step: f64, tol: f64, max_iter: usize) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut pts: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] += if p[i] == 0.0 { step } else { step * p[i].abs().max(1.0) };
        pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();
    for _ in 0..max_iter {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        pts = idx.iter().map(|&i| pts[i].clone()).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();
        if (vals[n] - vals[0]).abs() < tol && pts.iter().all(|p| p.iter().zip(&pts[0]).all(|(a, b)| (a - b).abs() < tol.sqrt())) {
            break;
        }
        let centroid: Vec<f64> = (0..n).map(|j| pts[..n].iter().map(|p| p[j]).sum::<f64>() / n as f64).collect();
        let mix = |t: f64| -> Vec<f64> { centroid.iter().zip(&pts[n]).map(|(c, w)| c + t * (c - w)).collect() };
        let refl = mix(1.0);
        let fr = f(&refl);
        if fr < vals[0] {
            let exp = mix(2.0);
            let fe = f(&exp);
            if fe < fr { pts[n] = exp; vals[n] = fe; } else { pts[n] = refl; vals[n] = fr; }
        } else if fr < vals[n - 1] {
            pts[n] = refl;
            vals[n] = fr;
        } else {
            let con = if fr < vals[n] { mix(0.5) } else { mix(-0.5) };
            let fcn = f(&con);
            if fcn < vals[n].min(fr) {
                pts[n] = con;
                vals[n] = fcn;
            } else {
                for i in 1..=n {
                    pts[i] = pts[0].iter().zip(&pts[i]).map(|(b, p)| b + 0.5 * (p - b)).collect();
                    vals[i] = f(&pts[i]);
                }
            }
        }
    }
    let best = (0..=n).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    (pts[best].clone(), vals[best])
}

fn num_grad<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let h = 1e-6 * x[i].abs().max(1.0);
            let (mut p, mut m) = (x.to_vec(), x.to_vec());
            p[i] += h;
            m[i] -= h;
            (f(&p) - f(&m)) / (2.0 * h)
        })
        .collect()
}

/// BFGS with backtracking line search; the gradient is numerical unless `grad` is given.
pub fn bfgs<F, G>(f: F, grad: Option<G>, x0: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let g_of = |x: &[f64]| -> Vec<f64> { grad.as_ref().map_or_else(|| num_grad(&f, x), |g| g(x)) };
    let mut x = x0.to_vec();
    let mut h: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    let mut g = g_of(&x);
    for _ in 0..max_iter {
        if g.iter().map(|v| v * v).sum::<f64>().sqrt() < tol {
            break;
        }
        let dir: Vec<f64> = (0..n).map(|i| -(0..n).map(|j| h[i][j] * g[j]).sum::<f64>()).collect();
        let slope: f64 = dir.iter().zip(&g).map(|(a, b)| a * b).sum();
        let (mut t, fx) = (1.0, f(&x));
        let mut xn = x.clone();
        for _ in 0..50 {
            xn = x.iter().zip(&dir).map(|(a, d)| a + t * d).collect();
            if f(&xn) <= fx + 1e-4 * t * slope {
                break;
            }
            t *= 0.5;
        }
        let gn = g_of(&xn);
        let s: Vec<f64> = xn.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();
        if sy > 1e-12 {
            let hy: Vec<f64> = (0..n).map(|i| (0..n).map(|j| h[i][j] * y[j]).sum()).collect();
            let yhy: f64 = y.iter().zip(&hy).map(|(a, b)| a * b).sum();
            for i in 0..n {
                for j in 0..n {
                    h[i][j] += (1.0 + yhy / sy) * s[i] * s[j] / sy - (hy[i] * s[j] + s[i] * hy[j]) / sy;
                }
            }
        }
        x = xn;
        g = gn;
    }
    let fx = f(&x);
    (x, fx)
}

// ------------------------------------------------------------------ integration

const GK_X: [f64; 8] = [0.991_455_371_120_813, 0.949_107_912_342_759, 0.864_864_423_359_769, 0.741_531_185_599_394, 0.586_087_235_467_691, 0.405_845_151_377_397, 0.207_784_955_007_898, 0.0];
const GK_WK: [f64; 8] = [0.022_935_322_010_529, 0.063_092_092_629_979, 0.104_790_010_322_250, 0.140_653_259_715_525, 0.169_004_726_639_267, 0.190_350_578_064_785, 0.204_432_940_075_298, 0.209_482_141_084_728];
const GK_WG: [f64; 4] = [0.129_484_966_168_870, 0.279_705_391_489_277, 0.381_830_050_505_119, 0.417_959_183_673_469];

fn gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
    let mut k = GK_WK[7] * f(c);
    let mut g = GK_WG[3] * f(c);
    for i in 0..7 {
        let (p, m) = (f(c + h * GK_X[i]), f(c - h * GK_X[i]));
        k += GK_WK[i] * (p + m);
        if i % 2 == 1 {
            g += GK_WG[i / 2] * (p + m);
        }
    }
    (k * h, ((k - g) * h).abs())
}

/// Adaptive Gauss-Kronrod (G7, K15) quadrature; returns `(integral, error_estimate)`.
pub fn quad<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64) -> Result<(f64, f64)> {
    if !a.is_finite() || !b.is_finite() {
        return Err(Error::Msg("quad: limits must be finite".into()));
    }
    let mut parts = vec![(a, b, gk15(&f, a, b))];
    for _ in 0..500 {
        let err: f64 = parts.iter().map(|p| p.2 .1).sum();
        if err < tol {
            break;
        }
        let worst = (0..parts.len()).max_by(|&i, &j| parts[i].2 .1.total_cmp(&parts[j].2 .1)).unwrap();
        let (lo, hi, _) = parts.swap_remove(worst);
        let mid = 0.5 * (lo + hi);
        parts.push((lo, mid, gk15(&f, lo, mid)));
        parts.push((mid, hi, gk15(&f, mid, hi)));
    }
    Ok((parts.iter().map(|p| p.2 .0).sum(), parts.iter().map(|p| p.2 .1).sum()))
}

/// Composite Simpson rule over `n` (even) subintervals.
pub fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> Result<f64> {
    if n == 0 || n % 2 != 0 {
        return Err(Error::Msg("simpson: n must be a positive even number".into()));
    }
    let h = (b - a) / n as f64;
    let s: f64 = (1..n).map(|i| f(a + i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 }).sum();
    Ok((f(a) + f(b) + s) * h / 3.0)
}

/// Romberg integration (Richardson extrapolation of the trapezoid rule) with `levels` refinement levels.
pub fn romberg<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, levels: usize) -> f64 {
    let levels = levels.max(1);
    let mut r = vec![vec![0.0; levels]; levels];
    r[0][0] = 0.5 * (b - a) * (f(a) + f(b));
    for i in 1..levels {
        let n = 1usize << (i - 1);
        let h = (b - a) / (2 * n) as f64;
        let mid: f64 = (0..n).map(|k| f(a + (2 * k + 1) as f64 * h)).sum();
        r[i][0] = 0.5 * r[i - 1][0] + h * mid;
        for j in 1..=i {
            r[i][j] = r[i][j - 1] + (r[i][j - 1] - r[i - 1][j - 1]) / (4f64.powi(j as i32) - 1.0);
        }
    }
    r[levels - 1][levels - 1]
}

// ------------------------------------------------------------------ interpolation

/// Natural cubic spline through strictly increasing knots.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    m: Vec<f64>, // second derivatives at the knots
}

impl CubicSpline {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self> {
        let n = x.len();
        if n < 3 || y.len() != n || x.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::Msg("CubicSpline: need >= 3 strictly increasing knots and matching values".into()));
        }
        // tridiagonal system for the interior second derivatives (natural: m0 = m_{n-1} = 0)
        let (mut a, mut b, mut c, mut d) = (vec![0.0; n], vec![1.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 1..n - 1 {
            let (h0, h1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
            a[i] = h0;
            b[i] = 2.0 * (h0 + h1);
            c[i] = h1;
            d[i] = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0);
        }
        for i in 1..n {
            let w = a[i] / b[i - 1];
            b[i] -= w * c[i - 1];
            d[i] -= w * d[i - 1];
        }
        let mut m = vec![0.0; n];
        m[n - 1] = d[n - 1] / b[n - 1];
        for i in (0..n - 1).rev() {
            m[i] = (d[i] - c[i] * m[i + 1]) / b[i];
        }
        m[0] = 0.0;
        m[n - 1] = 0.0;
        Ok(Self { x: x.to_vec(), y: y.to_vec(), m })
    }

    /// Evaluates at `t` (linear extrapolation outside the knots follows the end polynomials).
    pub fn eval(&self, t: f64) -> f64 {
        let i = self.x.partition_point(|&k| k <= t).clamp(1, self.x.len() - 1) - 1;
        let h = self.x[i + 1] - self.x[i];
        let (a, b) = ((self.x[i + 1] - t) / h, (t - self.x[i]) / h);
        a * self.y[i] + b * self.y[i + 1] + ((a.powi(3) - a) * self.m[i] + (b.powi(3) - b) * self.m[i + 1]) * h * h / 6.0
    }

    pub fn derivative(&self, t: f64) -> f64 {
        let i = self.x.partition_point(|&k| k <= t).clamp(1, self.x.len() - 1) - 1;
        let h = self.x[i + 1] - self.x[i];
        let (a, b) = ((self.x[i + 1] - t) / h, (t - self.x[i]) / h);
        (self.y[i + 1] - self.y[i]) / h - (3.0 * a * a - 1.0) / 6.0 * h * self.m[i] + (3.0 * b * b - 1.0) / 6.0 * h * self.m[i + 1]
    }
}

// ------------------------------------------------------------------ least squares

fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 {
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

/// Levenberg-Marquardt fit of `model(x, params)` to `(xs, ys)` with a numerical Jacobian.
/// Returns `(params, sum_of_squared_residuals)`.
pub fn curve_fit<M: Fn(f64, &[f64]) -> f64>(model: M, xs: &[f64], ys: &[f64], p0: &[f64], max_iter: usize) -> Result<(Vec<f64>, f64)> {
    if xs.len() != ys.len() || xs.len() < p0.len() || p0.is_empty() {
        return Err(Error::Msg("curve_fit: need at least as many points as parameters".into()));
    }
    let n = p0.len();
    let sse = |p: &[f64]| -> f64 { xs.iter().zip(ys).map(|(&x, &y)| (model(x, p) - y).powi(2)).sum() };
    let mut p = p0.to_vec();
    let mut cost = sse(&p);
    let mut lambda = 1e-3;
    for _ in 0..max_iter {
        let r: Vec<f64> = xs.iter().zip(ys).map(|(&x, &y)| y - model(x, &p)).collect();
        let jac: Vec<Vec<f64>> = xs
            .iter()
            .map(|&x| {
                (0..n)
                    .map(|j| {
                        let h = 1e-7 * p[j].abs().max(1.0);
                        let (mut a, mut b) = (p.clone(), p.clone());
                        a[j] += h;
                        b[j] -= h;
                        (model(x, &a) - model(x, &b)) / (2.0 * h)
                    })
                    .collect()
            })
            .collect();
        let jtj: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| jac.iter().map(|row| row[i] * row[j]).sum()).collect()).collect();
        let jtr: Vec<f64> = (0..n).map(|i| jac.iter().zip(&r).map(|(row, ri)| row[i] * ri).sum()).collect();
        let mut improved = false;
        for _ in 0..20 {
            let mut a = jtj.clone();
            (0..n).for_each(|i| a[i][i] += lambda * (jtj[i][i].max(1e-12)));
            let Some(step) = solve(&mut a, &mut jtr.clone()) else {
                lambda *= 10.0;
                continue;
            };
            let cand: Vec<f64> = p.iter().zip(&step).map(|(a, b)| a + b).collect();
            let c = sse(&cand);
            if c < cost {
                let done = (cost - c).abs() < 1e-14 * cost.max(1e-300);
                p = cand;
                cost = c;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                if done {
                    return Ok((p, cost));
                }
                break;
            }
            lambda *= 4.0;
        }
        if !improved {
            break;
        }
    }
    Ok((p, cost))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn root_finders() {
        close(bisect(|x| x * x - 2.0, 0.0, 2.0, 1e-10).unwrap(), 2f64.sqrt(), 1e-9);
        close(brentq(|x| x.cos() - x, 0.0, 1.0, 1e-12).unwrap(), 0.739_085_133_215_160_6, 1e-10);
        close(brentq(|x| x * x * x - 2.0 * x - 5.0, 2.0, 3.0, 1e-12).unwrap(), 2.094_551_481_542_327, 1e-9);
        close(newton(|x| x * x - 9.0, |x| 2.0 * x, 1.0, 1e-12).unwrap(), 3.0, 1e-10);
        assert!(bisect(|x| x * x + 1.0, -1.0, 1.0, 1e-6).is_err());
        assert!(brentq(|x| x * x + 1.0, -1.0, 1.0, 1e-6).is_err());
        assert!(newton(|x| x * x + 1.0, |x| 2.0 * x, 0.0, 1e-9).is_err());
    }

    #[test]
    fn minimizers() {
        let (x, v) = brent_min(|x| (x - 2.0).powi(2) + 1.0, -10.0, 10.0, 1e-9);
        close(x, 2.0, 1e-6);
        close(v, 1.0, 1e-9);
        let rosen = |p: &[f64]| (1.0 - p[0]).powi(2) + 100.0 * (p[1] - p[0] * p[0]).powi(2);
        let (x, v) = nelder_mead(rosen, &[-1.2, 1.0], 0.1, 1e-12, 2000);
        close(x[0], 1.0, 1e-3);
        close(x[1], 1.0, 1e-3);
        assert!(v < 1e-6);
        let (x, v) = bfgs(rosen, None::<fn(&[f64]) -> Vec<f64>>, &[-1.2, 1.0], 1e-8, 200);
        close(x[0], 1.0, 1e-4);
        close(x[1], 1.0, 1e-4);
        assert!(v < 1e-8);
        let grad = |p: &[f64]| vec![-2.0 * (1.0 - p[0]) - 400.0 * p[0] * (p[1] - p[0] * p[0]), 200.0 * (p[1] - p[0] * p[0])];
        let (x, _) = bfgs(rosen, Some(grad), &[-1.2, 1.0], 1e-8, 200);
        close(x[0], 1.0, 1e-5);
    }

    #[test]
    fn integrators() {
        let (v, e) = quad(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-12).unwrap();
        close(v, 2.0, 1e-10);
        assert!(e < 1e-10);
        // sqrt singularity at 0 needs adaptivity: integral of sqrt(x) over [0,1] is 2/3
        close(quad(f64::sqrt, 0.0, 1.0, 1e-10).unwrap().0, 2.0 / 3.0, 1e-8);
        close(simpson(|x| x * x * x, 0.0, 2.0, 4).unwrap(), 4.0, 1e-12); // exact for cubics
        close(romberg(|x| x.exp(), 0.0, 1.0, 6), std::f64::consts::E - 1.0, 1e-10);
        assert!(simpson(|x| x, 0.0, 1.0, 3).is_err() && quad(|x| x, 0.0, f64::INFINITY, 1e-6).is_err());
    }

    #[test]
    fn cubic_spline_interpolates_and_differentiates() {
        let x: Vec<f64> = (0..11).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let s = CubicSpline::new(&x, &y).unwrap();
        for (a, b) in x.iter().zip(&y) {
            close(s.eval(*a), *b, 1e-12);
        }
        close(s.eval(2.25), 2.25f64.sin(), 2e-3);
        close(s.derivative(2.25), 2.25f64.cos(), 5e-3);
        assert!(CubicSpline::new(&[0.0, 1.0], &[0.0, 1.0]).is_err());
        assert!(CubicSpline::new(&[0.0, 1.0, 1.0], &[0.0, 1.0, 2.0]).is_err());
    }

    #[test]
    fn curve_fit_recovers_parameters() {
        let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.5 * (-1.3 * x).exp() + 0.4).collect();
        let (p, sse) = curve_fit(|x, p| p[0] * (-p[1] * x).exp() + p[2], &xs, &ys, &[1.0, 1.0, 0.0], 200).unwrap();
        close(p[0], 2.5, 1e-5);
        close(p[1], 1.3, 1e-5);
        close(p[2], 0.4, 1e-5);
        assert!(sse < 1e-10);
        assert!(curve_fit(|x, p| p[0] * x, &[1.0], &[1.0], &[1.0, 2.0], 5).is_err());
    }
}
