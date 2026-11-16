//! Generalized linear models and smoothing: [`Glm`] (Gaussian / Binomial / Poisson via IRLS, with standard errors and
//! deviance), [`huber_regression`] (robust), [`lowess`] (local regression), and Gaussian [`Kde`].

use crate::classical::Matrix;
use crate::error::{Error, Result};

/// Distribution family (canonical link) of a [`Glm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    Gaussian,
    Binomial,
    Poisson,
}

/// Fitted GLM; an intercept is added automatically (`coef[0]`).
#[derive(Debug, Clone)]
pub struct Glm {
    pub family: Family,
    pub coef: Vec<f64>,
    /// Standard errors from the inverse Fisher information (intercept first).
    pub se: Vec<f64>,
    pub deviance: f64,
    pub iterations: usize,
}

type Scalar = fn(f64) -> f64;

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

fn invert(a: &Matrix) -> Option<Matrix> {
    let n = a.len();
    let cols: Vec<Vec<f64>> = (0..n).map(|j| solve(a.clone(), (0..n).map(|i| f64::from(u8::from(i == j))).collect())).collect::<Option<_>>()?;
    Some((0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect())
}

fn design(x: &Matrix) -> Matrix {
    x.iter().map(|r| std::iter::once(1.0).chain(r.iter().copied()).collect()).collect()
}

impl Glm {
    pub fn fit(x: &Matrix, y: &[f64], family: Family, max_iter: usize) -> Result<Self> {
        let d = x.first().map_or(0, Vec::len);
        if x.is_empty() || x.len() != y.len() || x.iter().any(|r| r.len() != d || r.iter().any(|v| !v.is_finite())) || y.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg("Glm: need a rectangular finite design with one response per row".into()));
        }
        match family {
            Family::Binomial if y.iter().any(|&v| !(0.0..=1.0).contains(&v)) => return Err(Error::Msg("Glm: binomial responses must lie in [0, 1]".into())),
            Family::Poisson if y.iter().any(|&v| v < 0.0) => return Err(Error::Msg("Glm: Poisson responses must be non-negative".into())),
            _ => {}
        }
        let xd = design(x);
        let p = d + 1;
        let (inv_link, var): (Scalar, Scalar) = match family {
            Family::Gaussian => (|e| e, |_| 1.0),
            Family::Binomial => (|e| 1.0 / (1.0 + (-e).exp()), |m| (m * (1.0 - m)).max(1e-10)),
            Family::Poisson => (|e| e.exp().min(1e12), |m| m.max(1e-10)),
        };
        let mut beta = vec![0.0; p];
        // start the intercept at the link of the mean so Poisson/binomial do not begin at an extreme
        let ybar = y.iter().sum::<f64>() / y.len() as f64;
        beta[0] = match family {
            Family::Gaussian => ybar,
            Family::Binomial => (ybar.clamp(0.01, 0.99) / (1.0 - ybar.clamp(0.01, 0.99))).ln(),
            Family::Poisson => ybar.max(0.1).ln(),
        };
        let mut info = vec![vec![0.0; p]; p];
        let mut iterations = 0;
        for it in 0..max_iter.max(1) {
            iterations = it + 1;
            let eta: Vec<f64> = xd.iter().map(|r| r.iter().zip(&beta).map(|(a, b)| a * b).sum()).collect();
            let mu: Vec<f64> = eta.iter().map(|&e| inv_link(e)).collect();
            // working response and weights for the canonical link: w = var(mu), z = eta + (y - mu) / var(mu)
            let w: Vec<f64> = mu.iter().map(|&m| var(m)).collect();
            let z: Vec<f64> = (0..y.len()).map(|i| eta[i] + (y[i] - mu[i]) / w[i]).collect();
            info = (0..p).map(|a| (0..p).map(|b| xd.iter().zip(&w).map(|(r, wi)| wi * r[a] * r[b]).sum()).collect()).collect();
            let rhs: Vec<f64> = (0..p).map(|a| xd.iter().zip(&w).zip(&z).map(|((r, wi), zi)| wi * r[a] * zi).sum()).collect();
            let next = solve(info.clone(), rhs).ok_or_else(|| Error::Msg("Glm: singular information matrix (collinear columns, or perfect separation)".into()))?;
            let delta = next.iter().zip(&beta).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            beta = next;
            if delta < 1e-9 {
                break;
            }
        }
        let cov = invert(&info).ok_or_else(|| Error::Msg("Glm: could not invert the information matrix".into()))?;
        let mu: Vec<f64> = xd.iter().map(|r| inv_link(r.iter().zip(&beta).map(|(a, b)| a * b).sum())).collect();
        let deviance = (0..y.len())
            .map(|i| match family {
                Family::Gaussian => (y[i] - mu[i]).powi(2),
                Family::Binomial => 2.0 * (if y[i] > 0.0 { y[i] * (y[i] / mu[i].max(1e-12)).ln() } else { 0.0 } + if y[i] < 1.0 { (1.0 - y[i]) * ((1.0 - y[i]) / (1.0 - mu[i]).max(1e-12)).ln() } else { 0.0 }),
                Family::Poisson => 2.0 * (if y[i] > 0.0 { y[i] * (y[i] / mu[i].max(1e-12)).ln() } else { 0.0 } - (y[i] - mu[i])),
            })
            .sum();
        // Gaussian standard errors need the residual variance
        let scale = if family == Family::Gaussian { deviance / (y.len().saturating_sub(p)).max(1) as f64 } else { 1.0 };
        let se = (0..p).map(|i| (cov[i][i] * scale).max(0.0).sqrt()).collect();
        Ok(Self { family, coef: beta, se, deviance, iterations })
    }

    /// Mean response for each row.
    pub fn predict(&self, x: &Matrix) -> Vec<f64> {
        x.iter()
            .map(|r| {
                let eta = self.coef[0] + r.iter().zip(&self.coef[1..]).map(|(a, b)| a * b).sum::<f64>();
                match self.family {
                    Family::Gaussian => eta,
                    Family::Binomial => 1.0 / (1.0 + (-eta).exp()),
                    Family::Poisson => eta.exp(),
                }
            })
            .collect()
    }
}

/// Huber-loss robust linear regression by IRLS (`delta` in units of the residual scale); returns `[intercept, coefs]`.
pub fn huber_regression(x: &Matrix, y: &[f64], delta: f64, iters: usize) -> Result<Vec<f64>> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || x.len() != y.len() || delta <= 0.0 || x.iter().any(|r| r.len() != d) {
        return Err(Error::Msg("huber_regression: need matching data and delta > 0".into()));
    }
    let xd = design(x);
    let p = d + 1;
    let mut w = vec![1.0; y.len()];
    let mut beta = vec![0.0; p];
    for _ in 0..iters.max(1) {
        let a: Matrix = (0..p).map(|i| (0..p).map(|j| xd.iter().zip(&w).map(|(r, wi)| wi * r[i] * r[j]).sum()).collect()).collect();
        let b: Vec<f64> = (0..p).map(|i| xd.iter().zip(&w).zip(y).map(|((r, wi), yi)| wi * r[i] * yi).sum()).collect();
        beta = solve(a, b).ok_or_else(|| Error::Msg("huber_regression: singular design".into()))?;
        let res: Vec<f64> = xd.iter().zip(y).map(|(r, yi)| yi - r.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>()).collect();
        let mut abs: Vec<f64> = res.iter().map(|v| v.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let scale = (abs[abs.len() / 2] / 0.6745).max(1e-12); // MAD scale estimate
        w = res.iter().map(|r| { let u = (r / scale).abs(); if u <= delta { 1.0 } else { delta / u } }).collect();
    }
    Ok(beta)
}

/// LOWESS (locally weighted linear regression with tricube weights): smoothed `y` at each `x`.
pub fn lowess(x: &[f64], y: &[f64], frac: f64) -> Result<Vec<f64>> {
    let n = x.len();
    if n < 3 || y.len() != n || !(0.0 < frac && frac <= 1.0) {
        return Err(Error::Msg("lowess: need >= 3 matching points and 0 < frac <= 1".into()));
    }
    let k = ((frac * n as f64).ceil() as usize).clamp(2, n);
    Ok((0..n)
        .map(|i| {
            let mut dist: Vec<f64> = x.iter().map(|v| (v - x[i]).abs()).collect();
            let mut sorted = dist.clone();
            sorted.sort_by(f64::total_cmp);
            let h = sorted[k - 1].max(1e-12);
            dist.iter_mut().for_each(|d| *d /= h);
            let w: Vec<f64> = dist.iter().map(|&d| if d < 1.0 { (1.0 - d.powi(3)).powi(3) } else { 0.0 }).collect();
            let sw: f64 = w.iter().sum();
            let (mx, my) = (w.iter().zip(x).map(|(a, b)| a * b).sum::<f64>() / sw, w.iter().zip(y).map(|(a, b)| a * b).sum::<f64>() / sw);
            let (sxy, sxx) = (0..n).fold((0.0, 0.0), |(a, b), j| (a + w[j] * (x[j] - mx) * (y[j] - my), b + w[j] * (x[j] - mx).powi(2)));
            if sxx < 1e-12 { my } else { my + sxy / sxx * (x[i] - mx) }
        })
        .collect())
}

/// Gaussian kernel density estimate with Silverman's rule-of-thumb bandwidth (or a user bandwidth).
#[derive(Debug, Clone)]
pub struct Kde {
    data: Vec<f64>,
    pub bandwidth: f64,
}

impl Kde {
    pub fn fit(data: &[f64], bandwidth: Option<f64>) -> Result<Self> {
        if data.len() < 2 || data.iter().any(|v| !v.is_finite()) {
            return Err(Error::Msg("Kde: need at least 2 finite values".into()));
        }
        let n = data.len() as f64;
        let m = data.iter().sum::<f64>() / n;
        let sd = (data.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let mut s = data.to_vec();
        s.sort_by(f64::total_cmp);
        let iqr = s[(0.75 * (n - 1.0)).round() as usize] - s[(0.25 * (n - 1.0)).round() as usize];
        let spread = if iqr > 0.0 { sd.min(iqr / 1.34) } else { sd };
        let h = bandwidth.unwrap_or(0.9 * spread.max(1e-12) * n.powf(-0.2));
        if h <= 0.0 {
            return Err(Error::Msg("Kde: bandwidth must be positive".into()));
        }
        Ok(Self { data: data.to_vec(), bandwidth: h })
    }

    pub fn density(&self, x: f64) -> f64 {
        let h = self.bandwidth;
        self.data.iter().map(|d| (-0.5 * ((x - d) / h).powi(2)).exp()).sum::<f64>() / (self.data.len() as f64 * h * (2.0 * std::f64::consts::PI).sqrt())
    }

    pub fn log_likelihood(&self, points: &[f64]) -> f64 {
        points.iter().map(|&p| self.density(p).max(1e-300).ln()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn gaussian_glm_equals_ols_with_standard_errors() {
        let x: Matrix = (0..30).map(|i| vec![i as f64 / 3.0]).collect();
        let y: Vec<f64> = x.iter().enumerate().map(|(i, r)| 2.0 + 1.5 * r[0] + if i % 2 == 0 { 0.2 } else { -0.2 }).collect();
        let g = Glm::fit(&x, &y, Family::Gaussian, 10).unwrap();
        assert!((g.coef[0] - 2.0).abs() < 0.1 && (g.coef[1] - 1.5).abs() < 0.02, "{:?}", g.coef);
        assert!((g.se[1] - 0.0127).abs() < 0.003, "{}", g.se[1]); // sqrt(sigma^2 / Sxx) = sqrt(0.04 / 249.7)
        assert!(g.iterations <= 2, "linear problem solves in one IRLS step");
    }

    #[test]
    fn logistic_and_poisson_recover_coefficients() {
        let mut rng = Rng::new(4);
        let x: Matrix = (0..1500).map(|_| vec![rng.next_f64() * 4.0 - 2.0]).collect();
        let y: Vec<f64> = x.iter().map(|r| f64::from(u8::from(rng.next_f64() < 1.0 / (1.0 + (-(0.5 + 1.2 * r[0])).exp())))).collect();
        let g = Glm::fit(&x, &y, Family::Binomial, 50).unwrap();
        assert!((g.coef[0] - 0.5).abs() < 3.0 * g.se[0] && (g.coef[1] - 1.2).abs() < 3.0 * g.se[1], "{:?} +- {:?}", g.coef, g.se);
        assert!(g.predict(&vec![vec![2.0]])[0] > 0.9 && g.predict(&vec![vec![-2.0]])[0] < 0.3);
        // Poisson counts with log-rate 0.3 + 0.7 x
        let counts: Vec<f64> = x.iter().map(|r| { let lam = (0.3 + 0.7 * r[0] * 0.5).exp(); let (mut k, mut p, l) = (0.0, 1.0, (-lam).exp()); loop { p *= rng.next_f64(); if p <= l { break k; } k += 1.0; } }).collect();
        let pg = Glm::fit(&x, &counts, Family::Poisson, 50).unwrap();
        assert!((pg.coef[0] - 0.3).abs() < 3.0 * pg.se[0] && (pg.coef[1] - 0.35).abs() < 3.0 * pg.se[1], "{:?} +- {:?}", pg.coef, pg.se);
        assert!(Glm::fit(&x, &vec![2.0; 1500], Family::Binomial, 10).is_err());
    }

    #[test]
    fn robust_fit_ignores_outliers() {
        let x: Matrix = (0..40).map(|i| vec![i as f64]).collect();
        let mut y: Vec<f64> = x.iter().map(|r| 1.0 + 2.0 * r[0]).collect();
        y[5] += 200.0;
        y[30] -= 150.0;
        let robust = huber_regression(&x, &y, 1.345, 50).unwrap();
        let ols = Glm::fit(&x, &y, Family::Gaussian, 5).unwrap();
        assert!((robust[1] - 2.0).abs() < 0.05 && (robust[0] - 1.0).abs() < 1.0, "{robust:?}");
        assert!((ols.coef[1] - 2.0).abs() > (robust[1] - 2.0).abs() * 5.0, "OLS is dragged by the outliers");
        assert!(huber_regression(&x, &y, 0.0, 5).is_err());
    }

    #[test]
    fn lowess_and_kde() {
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
        let mut rng = Rng::new(2);
        let y: Vec<f64> = x.iter().map(|v| v.sin() + 0.3 * (rng.next_f64() - 0.5)).collect();
        let s = lowess(&x, &y, 0.2).unwrap();
        let err_s: f64 = s.iter().zip(&x).map(|(a, v)| (a - v.sin()).powi(2)).sum::<f64>() / 100.0;
        let err_raw: f64 = y.iter().zip(&x).map(|(a, v)| (a - v.sin()).powi(2)).sum::<f64>() / 100.0;
        assert!(err_s < err_raw, "{err_s} vs {err_raw}");
        assert!(lowess(&x[..2], &y[..2], 0.5).is_err());
        // KDE of a standard normal sample integrates to ~1 and peaks near 0
        let sample: Vec<f64> = { let mut r = Rng::new(6); (0..500).map(|_| { let (u1, u2) = (r.next_f64().max(1e-300), r.next_f64()); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() }).collect() };
        let kde = Kde::fit(&sample, None).unwrap();
        let area: f64 = (-800..800).map(|i| kde.density(i as f64 * 0.01) * 0.01).sum();
        assert!((area - 1.0).abs() < 0.01, "{area}");
        assert!(kde.density(0.0) > kde.density(2.5) * 5.0);
        assert!((kde.density(0.0) - 0.399).abs() < 0.07);
        assert!(kde.log_likelihood(&[0.0, 0.1]) > kde.log_likelihood(&[5.0, 6.0]));
        assert!(Kde::fit(&[1.0], None).is_err());
    }
}
