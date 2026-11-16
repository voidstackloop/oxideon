//! Classical time-series tools on `&[f64]`: [`acf`], [`pacf`], [`ljung_box`], differencing, rolling statistics and
//! [`ewma`], AR models by Yule-Walker ([`ArModel`], and [`Arima`] = AR on a differenced series),
//! exponential smoothing ([`holt_winters`]), and additive [`seasonal_decompose`].

use crate::error::{Error, Result};

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Sample autocorrelation for lags `0..=max_lag`.
pub fn acf(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    if x.len() < 2 || max_lag >= x.len() {
        return Err(Error::Msg(format!("acf: need max_lag < len (got {max_lag} for {} points)", x.len())));
    }
    let m = mean(x);
    let c0: f64 = x.iter().map(|v| (v - m).powi(2)).sum();
    if c0 == 0.0 {
        return Err(Error::Msg("acf: the series is constant".into()));
    }
    Ok((0..=max_lag).map(|k| (k..x.len()).map(|t| (x[t] - m) * (x[t - k] - m)).sum::<f64>() / c0).collect())
}

/// Partial autocorrelation for lags `0..=max_lag` via the Durbin-Levinson recursion.
pub fn pacf(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let r = acf(x, max_lag)?;
    let mut out = vec![1.0];
    let mut phi: Vec<f64> = Vec::new();
    for k in 1..=max_lag {
        let num = r[k] - phi.iter().enumerate().map(|(j, p)| p * r[k - 1 - j]).sum::<f64>();
        let den = 1.0 - phi.iter().enumerate().map(|(j, p)| p * r[j + 1]).sum::<f64>();
        let a = num / den;
        let next: Vec<f64> = phi.iter().enumerate().map(|(j, p)| p - a * phi[k - 2 - j]).collect();
        phi = next;
        phi.push(a);
        out.push(a);
    }
    Ok(out)
}

/// Ljung-Box statistic of the first `lags` autocorrelations (compare with chi-square, `lags` d.o.f.).
pub fn ljung_box(x: &[f64], lags: usize) -> Result<f64> {
    let r = acf(x, lags)?;
    let n = x.len() as f64;
    Ok(n * (n + 2.0) * (1..=lags).map(|k| r[k] * r[k] / (n - k as f64)).sum::<f64>())
}

/// `d`-th order difference.
pub fn diff(x: &[f64], d: usize) -> Vec<f64> {
    let mut v = x.to_vec();
    for _ in 0..d {
        v = v.windows(2).map(|w| w[1] - w[0]).collect();
    }
    v
}

/// Rolling mean over a trailing window (first `window - 1` outputs are `NaN`).
pub fn rolling_mean(x: &[f64], window: usize) -> Vec<f64> {
    (0..x.len()).map(|i| if window == 0 || i + 1 < window { f64::NAN } else { mean(&x[i + 1 - window..=i]) }).collect()
}

/// Rolling sample standard deviation over a trailing window.
pub fn rolling_std(x: &[f64], window: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if window < 2 || i + 1 < window {
                return f64::NAN;
            }
            let w = &x[i + 1 - window..=i];
            let m = mean(w);
            (w.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (window - 1) as f64).sqrt()
        })
        .collect()
}

/// Exponentially weighted moving average with smoothing factor `alpha` in `(0, 1]`.
pub fn ewma(x: &[f64], alpha: f64) -> Result<Vec<f64>> {
    if !(alpha > 0.0 && alpha <= 1.0) {
        return Err(Error::Msg("ewma: alpha must be in (0, 1]".into()));
    }
    let mut out = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        out.push(if i == 0 { v } else { alpha * v + (1.0 - alpha) * out[i - 1] });
    }
    Ok(out)
}

/// Autoregressive model `x_t - mu = sum_i phi_i (x_{t-i} - mu) + e_t` fitted by Yule-Walker.
#[derive(Debug, Clone)]
pub struct ArModel {
    pub mean: f64,
    pub phi: Vec<f64>,
    pub sigma2: f64,
}

impl ArModel {
    pub fn fit(x: &[f64], order: usize) -> Result<Self> {
        if order == 0 || x.len() <= order + 1 {
            return Err(Error::Msg(format!("ArModel: need order >= 1 and more than {} points", order + 1)));
        }
        let m = mean(x);
        let n = x.len() as f64;
        let gamma: Vec<f64> = (0..=order).map(|k| (k..x.len()).map(|t| (x[t] - m) * (x[t - k] - m)).sum::<f64>() / n).collect();
        if gamma[0] == 0.0 {
            return Err(Error::Msg("ArModel: the series is constant".into()));
        }
        // Levinson-Durbin on the autocovariances
        let mut phi: Vec<f64> = Vec::new();
        let mut v = gamma[0];
        for k in 1..=order {
            let a = (gamma[k] - phi.iter().enumerate().map(|(j, p)| p * gamma[k - 1 - j]).sum::<f64>()) / v;
            let next: Vec<f64> = phi.iter().enumerate().map(|(j, p)| p - a * phi[k - 2 - j]).collect();
            phi = next;
            phi.push(a);
            v *= 1.0 - a * a;
        }
        Ok(Self { mean: m, phi, sigma2: v })
    }

    /// Forecasts `steps` values after `history` (needs at least `order` points).
    pub fn forecast(&self, history: &[f64], steps: usize) -> Result<Vec<f64>> {
        let p = self.phi.len();
        if history.len() < p {
            return Err(Error::Msg(format!("ArModel::forecast: need at least {p} history points")));
        }
        let mut buf: Vec<f64> = history[history.len() - p..].iter().map(|v| v - self.mean).collect();
        let mut out = Vec::with_capacity(steps);
        for _ in 0..steps {
            let next: f64 = self.phi.iter().enumerate().map(|(i, c)| c * buf[buf.len() - 1 - i]).sum();
            buf.push(next);
            out.push(next + self.mean);
        }
        Ok(out)
    }
}

/// ARIMA(p, d, 0): an AR(p) fitted to the `d`-times differenced series, forecasts integrated back.
#[derive(Debug, Clone)]
pub struct Arima {
    pub ar: ArModel,
    pub d: usize,
}

impl Arima {
    pub fn fit(x: &[f64], p: usize, d: usize) -> Result<Self> {
        Ok(Self { ar: ArModel::fit(&diff(x, d), p)?, d })
    }

    pub fn forecast(&self, x: &[f64], steps: usize) -> Result<Vec<f64>> {
        // difference ladder: levels[k] = k-times differenced series
        let mut levels = vec![x.to_vec()];
        for _ in 0..self.d {
            levels.push(diff(levels.last().unwrap(), 1));
        }
        let mut cur = self.ar.forecast(levels.last().unwrap(), steps)?;
        for k in (0..self.d).rev() {
            let mut last = *levels[k].last().ok_or_else(|| Error::Msg("Arima::forecast: series too short".into()))?;
            cur = cur.iter().map(|inc| { last += inc; last }).collect();
        }
        Ok(cur)
    }
}

/// Additive Holt-Winters exponential smoothing. `period = 0` gives Holt's linear trend; returns the `steps`-ahead
/// forecast after smoothing all of `x` with fixed `alpha` (level), `beta` (trend), `gamma` (season).
pub fn holt_winters(x: &[f64], alpha: f64, beta: f64, gamma: f64, period: usize, steps: usize) -> Result<Vec<f64>> {
    let need = if period == 0 { 2 } else { 2 * period };
    if x.len() < need {
        return Err(Error::Msg(format!("holt_winters: need at least {need} points")));
    }
    let (mut level, mut trend, mut season);
    if period == 0 {
        level = x[0];
        trend = x[1] - x[0];
        season = Vec::new();
    } else {
        let m1 = mean(&x[..period]);
        let m2 = mean(&x[period..2 * period]);
        level = m1;
        trend = (m2 - m1) / period as f64;
        season = (0..period).map(|i| x[i] - m1).collect();
    }
    let start = if period == 0 { 1 } else { period };
    for t in start..x.len() {
        let s = if period == 0 { 0.0 } else { season[t % period] };
        let new_level = alpha * (x[t] - s) + (1.0 - alpha) * (level + trend);
        trend = beta * (new_level - level) + (1.0 - beta) * trend;
        if period > 0 {
            season[t % period] = gamma * (x[t] - new_level) + (1.0 - gamma) * s;
        }
        level = new_level;
    }
    let n = x.len();
    Ok((1..=steps).map(|h| level + h as f64 * trend + if period == 0 { 0.0 } else { season[(n + h - 1) % period] }).collect())
}

/// Additive decomposition `x = trend + seasonal + residual` using a centred moving average of one period.
pub struct Decomposition {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

pub fn seasonal_decompose(x: &[f64], period: usize) -> Result<Decomposition> {
    if period < 2 || x.len() < 2 * period {
        return Err(Error::Msg("seasonal_decompose: need period >= 2 and at least two full periods".into()));
    }
    let n = x.len();
    let half = period / 2;
    let trend: Vec<f64> = (0..n)
        .map(|t| {
            if t < half || t + half >= n + usize::from(period % 2 == 1) {
                return f64::NAN;
            }
            if period % 2 == 1 {
                mean(&x[t - half..=t + half])
            } else {
                // 2 x period moving average
                (0.5 * x[t - half] + x[t - half + 1..t + half].iter().sum::<f64>() + 0.5 * x[t + half]) / period as f64
            }
        })
        .collect();
    let mut sums = vec![0.0; period];
    let mut counts = vec![0usize; period];
    for t in 0..n {
        if !trend[t].is_nan() {
            sums[t % period] += x[t] - trend[t];
            counts[t % period] += 1;
        }
    }
    let raw: Vec<f64> = sums.iter().zip(&counts).map(|(s, &c)| s / c.max(1) as f64).collect();
    let centre = mean(&raw);
    let seasonal: Vec<f64> = (0..n).map(|t| raw[t % period] - centre).collect();
    let residual = (0..n).map(|t| x[t] - trend[t] - seasonal[t]).collect();
    Ok(Decomposition { trend, seasonal, residual })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn ar1(phi: f64, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let mut x = vec![0.0];
        for _ in 1..n {
            let e = rng.next_f64() - 0.5;
            x.push(phi * x.last().unwrap() + e);
        }
        x
    }

    #[test]
    fn correlation_functions() {
        let x = ar1(0.8, 2000, 1);
        let r = acf(&x, 3).unwrap();
        assert_eq!(r[0], 1.0);
        assert!((r[1] - 0.8).abs() < 0.05 && (r[2] - 0.64).abs() < 0.07, "{r:?}");
        let p = pacf(&x, 3).unwrap();
        assert!((p[1] - 0.8).abs() < 0.05 && p[2].abs() < 0.07 && p[3].abs() < 0.07, "AR(1): pacf cuts off after lag 1: {p:?}");
        // white noise has a small Ljung-Box statistic; a strongly autocorrelated series a huge one
        let noise: Vec<f64> = { let mut r = Rng::new(5); (0..500).map(|_| r.next_f64()).collect() };
        assert!(ljung_box(&noise, 10).unwrap() < 30.0);
        assert!(ljung_box(&x, 10).unwrap() > 1000.0);
        assert!(acf(&[1.0, 1.0, 1.0], 1).is_err() && acf(&[1.0, 2.0], 5).is_err());
    }

    #[test]
    fn helpers() {
        assert_eq!(diff(&[1.0, 4.0, 9.0, 16.0], 2), vec![2.0, 2.0]);
        let m = rolling_mean(&[1.0, 2.0, 3.0, 4.0], 2);
        assert!(m[0].is_nan() && m[1..] == [1.5, 2.5, 3.5]);
        assert!((rolling_std(&[1.0, 3.0], 2)[1] - 2f64.sqrt()).abs() < 1e-12);
        assert_eq!(ewma(&[1.0, 3.0], 0.5).unwrap(), vec![1.0, 2.0]);
        assert!(ewma(&[1.0], 0.0).is_err());
    }

    #[test]
    fn ar_and_arima_forecasts() {
        let x = ar1(0.7, 3000, 2);
        let m = ArModel::fit(&x, 1).unwrap();
        assert!((m.phi[0] - 0.7).abs() < 0.04, "{:?}", m.phi);
        assert!((m.sigma2 - 1.0 / 12.0).abs() < 0.01);
        // forecasts decay geometrically toward the mean
        let f = m.forecast(&[2.0], 3).unwrap();
        assert!((f[0] - (m.mean + m.phi[0] * (2.0 - m.mean))).abs() < 1e-12 && f[2].abs() < f[0].abs());
        // random walk with drift: ARIMA(1,1,0) forecast continues the upward line
        let walk: Vec<f64> = (0..200).map(|i| 0.5 * i as f64 + ar1(0.3, 200, 4)[i]).collect();
        let a = Arima::fit(&walk, 1, 1).unwrap();
        let f = a.forecast(&walk, 5).unwrap();
        assert!(f.windows(2).all(|w| w[1] > w[0]) && f[0] > walk[199] - 2.0, "{f:?}");
        assert!(ArModel::fit(&[1.0, 2.0], 3).is_err());
    }

    #[test]
    fn holt_winters_and_decomposition() {
        // linear trend: Holt forecasts continue the line
        let lin: Vec<f64> = (0..20).map(|i| 2.0 + 3.0 * i as f64).collect();
        let f = holt_winters(&lin, 0.5, 0.5, 0.0, 0, 3).unwrap();
        assert!((f[0] - 62.0).abs() < 1e-9 && (f[2] - 68.0).abs() < 1e-9, "{f:?}");
        // trend + period-4 season
        let pat = [5.0, -2.0, 1.0, -4.0];
        let x: Vec<f64> = (0..40).map(|t| 10.0 + 0.5 * t as f64 + pat[t % 4]).collect();
        let f = holt_winters(&x, 0.3, 0.1, 0.3, 4, 4).unwrap();
        for (h, v) in f.iter().enumerate() {
            let want = 10.0 + 0.5 * (40 + h) as f64 + pat[(40 + h) % 4];
            assert!((v - want).abs() < 1.0, "h={h}: {v} vs {want}");
        }
        let d = seasonal_decompose(&x, 4).unwrap();
        for k in 0..4 {
            assert!((d.seasonal[k] - (pat[k] - mean(&pat))).abs() < 1e-9, "{:?}", &d.seasonal[..4]);
        }
        assert!(d.residual.iter().filter(|v| !v.is_nan()).all(|v| v.abs() < 1e-9));
        assert!(seasonal_decompose(&x[..5], 4).is_err());
    }
}
