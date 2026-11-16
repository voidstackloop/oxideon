//! Experimentation and causal-inference helpers: multiple-testing corrections ([`bonferroni`], [`holm`],
//! [`benjamini_hochberg`]), [`two_proportion_ztest`], power / sample-size calculations, inverse-propensity-weighted
//! [`ipw_ate`], [`diff_in_diff`], and two-stage least squares ([`two_sls`]).

use crate::classical::Matrix;
use crate::error::{Error, Result};
use crate::glm::{Family, Glm};
use crate::stats_tests::{norm_cdf, TestResult};

fn check_p(p: &[f64], op: &str) -> Result<()> {
    if p.is_empty() || p.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(Error::Msg(format!("{op}: need non-empty p-values in [0, 1]")));
    }
    Ok(())
}

/// Bonferroni-adjusted p-values.
pub fn bonferroni(p: &[f64]) -> Result<Vec<f64>> {
    check_p(p, "bonferroni")?;
    Ok(p.iter().map(|v| (v * p.len() as f64).min(1.0)).collect())
}

fn sorted_order(p: &[f64]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..p.len()).collect();
    o.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    o
}

/// Holm step-down adjusted p-values (controls family-wise error; uniformly more powerful than Bonferroni).
pub fn holm(p: &[f64]) -> Result<Vec<f64>> {
    check_p(p, "holm")?;
    let (m, order) = (p.len(), sorted_order(p));
    let mut adj = vec![0.0; m];
    let mut running = 0.0f64;
    for (rank, &i) in order.iter().enumerate() {
        running = running.max(((m - rank) as f64 * p[i]).min(1.0));
        adj[i] = running;
    }
    Ok(adj)
}

/// Benjamini-Hochberg adjusted p-values (q-values; controls the false discovery rate).
pub fn benjamini_hochberg(p: &[f64]) -> Result<Vec<f64>> {
    check_p(p, "benjamini_hochberg")?;
    let (m, order) = (p.len(), sorted_order(p));
    let mut adj = vec![0.0; m];
    let mut running = 1.0f64;
    for (rank, &i) in order.iter().enumerate().rev() {
        running = running.min((p[i] * m as f64 / (rank + 1) as f64).min(1.0));
        adj[i] = running;
    }
    Ok(adj)
}

/// Inverse standard normal CDF (Acklam's rational approximation, relative error < 1.2e-9).
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969_683_028_665_376e1, 2.209_460_984_245_205e2, -2.759_285_104_469_687e2, 1.383_577_518_672_69e2, -3.066_479_806_614_716e1, 2.506_628_277_459_239];
    const B: [f64; 5] = [-5.447_609_879_822_406e1, 1.615_858_368_580_409e2, -1.556_989_798_598_866e2, 6.680_131_188_771_972e1, -1.328_068_155_288_572e1];
    const C: [f64; 6] = [-7.784_894_002_430_293e-3, -3.223_964_580_411_365e-1, -2.400_758_277_161_838, -2.549_732_539_343_734, 4.374_664_141_464_968, 2.938_163_982_698_783];
    const D: [f64; 4] = [7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996, 3.754_408_661_907_416];
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// Two-sided z-test comparing conversion rates `x1/n1` and `x2/n2` (pooled standard error).
pub fn two_proportion_ztest(x1: u64, n1: u64, x2: u64, n2: u64) -> Result<TestResult> {
    if n1 == 0 || n2 == 0 || x1 > n1 || x2 > n2 {
        return Err(Error::Msg("two_proportion_ztest: need 0 <= x <= n and n > 0 for both groups".into()));
    }
    let (p1, p2) = (x1 as f64 / n1 as f64, x2 as f64 / n2 as f64);
    let pooled = (x1 + x2) as f64 / (n1 + n2) as f64;
    let se = (pooled * (1.0 - pooled) * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
    if se == 0.0 {
        return Err(Error::Msg("two_proportion_ztest: both groups are all-success or all-failure".into()));
    }
    let z = (p1 - p2) / se;
    Ok(TestResult { statistic: z, p_value: (2.0 * (1.0 - norm_cdf(z.abs()))).clamp(0.0, 1.0) })
}

/// Per-group sample size to detect a difference `delta` in means with common standard deviation `sd`
/// (two-sided z approximation) at significance `alpha` and power `power`.
pub fn sample_size_means(delta: f64, sd: f64, alpha: f64, power: f64) -> Result<u64> {
    if delta == 0.0 || sd <= 0.0 || !(0.0 < alpha && alpha < 1.0) || !(0.0 < power && power < 1.0) {
        return Err(Error::Msg("sample_size_means: need delta != 0, sd > 0 and alpha, power in (0, 1)".into()));
    }
    let (za, zb) = (norm_ppf(1.0 - alpha / 2.0), norm_ppf(power));
    Ok((2.0 * ((za + zb) * sd / delta).powi(2)).ceil() as u64)
}

/// Per-group sample size to detect conversion rates `p1` vs `p2`.
pub fn sample_size_proportions(p1: f64, p2: f64, alpha: f64, power: f64) -> Result<u64> {
    if !(0.0 < p1 && p1 < 1.0 && 0.0 < p2 && p2 < 1.0 && 0.0 < alpha && alpha < 1.0 && 0.0 < power && power < 1.0) || p1 == p2 {
        return Err(Error::Msg("sample_size_proportions: need distinct p1, p2 in (0, 1) and alpha, power in (0, 1)".into()));
    }
    let (za, zb) = (norm_ppf(1.0 - alpha / 2.0), norm_ppf(power));
    let pbar = 0.5 * (p1 + p2);
    let num = za * (2.0 * pbar * (1.0 - pbar)).sqrt() + zb * (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt();
    Ok((num * num / (p1 - p2).powi(2)).ceil() as u64)
}

/// Power of a two-sided two-sample z-test for mean difference `delta` with `n` per group.
pub fn power_means(delta: f64, sd: f64, n: u64, alpha: f64) -> f64 {
    let se = sd * (2.0 / n as f64).sqrt();
    let za = norm_ppf(1.0 - alpha / 2.0);
    norm_cdf(delta.abs() / se - za) + norm_cdf(-delta.abs() / se - za)
}

/// Inverse-propensity-weighted average treatment effect (Horvitz-Thompson): propensities come from a logistic
/// regression of `treated` on the covariates, clipped to `[clip, 1 - clip]`.
pub fn ipw_ate(x: &Matrix, treated: &[bool], outcome: &[f64], clip: f64) -> Result<f64> {
    if x.len() != treated.len() || x.len() != outcome.len() || !(0.0..0.5).contains(&clip) || !treated.contains(&true) || !treated.contains(&false) {
        return Err(Error::Msg("ipw_ate: need matching inputs, both arms present and clip in [0, 0.5)".into()));
    }
    let t: Vec<f64> = treated.iter().map(|&b| f64::from(u8::from(b))).collect();
    let model = Glm::fit(x, &t, Family::Binomial, 50)?;
    let e = model.predict(x);
    let n = x.len() as f64;
    Ok((0..x.len())
        .map(|i| {
            let p = e[i].clamp(clip.max(1e-6), 1.0 - clip.max(1e-6));
            t[i] * outcome[i] / p - (1.0 - t[i]) * outcome[i] / (1.0 - p)
        })
        .sum::<f64>()
        / n)
}

/// Difference-in-differences estimate from the four group means: `(treat_post - treat_pre) - (ctrl_post - ctrl_pre)`.
pub fn diff_in_diff(treat_pre: &[f64], treat_post: &[f64], ctrl_pre: &[f64], ctrl_post: &[f64]) -> Result<f64> {
    let mean = |v: &[f64]| -> Result<f64> {
        if v.is_empty() { Err(Error::Msg("diff_in_diff: every group needs at least one observation".into())) } else { Ok(v.iter().sum::<f64>() / v.len() as f64) }
    };
    Ok((mean(treat_post)? - mean(treat_pre)?) - (mean(ctrl_post)? - mean(ctrl_pre)?))
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

fn ols(z: &Matrix, y: &[f64]) -> Option<Vec<f64>> {
    let p = z[0].len();
    let a: Matrix = (0..p).map(|i| (0..p).map(|j| z.iter().map(|r| r[i] * r[j]).sum()).collect()).collect();
    let b: Vec<f64> = (0..p).map(|i| z.iter().zip(y).map(|(r, v)| r[i] * v).sum()).collect();
    solve(a, b)
}

/// Two-stage least squares: coefficients `[intercept, endogenous regressors..., exogenous controls...]` of
/// `y ~ endog + exog`, instrumenting `endog` with `instruments` (plus the exogenous controls).
pub fn two_sls(y: &[f64], endog: &Matrix, exog: &Matrix, instruments: &Matrix) -> Result<Vec<f64>> {
    let n = y.len();
    if n == 0 || endog.len() != n || exog.len() != n || instruments.len() != n || endog[0].is_empty() || instruments[0].len() < endog[0].len() {
        return Err(Error::Msg("two_sls: need matching rows and at least as many instruments as endogenous regressors".into()));
    }
    let row = |e: &[f64], x: &[f64]| -> Vec<f64> { std::iter::once(1.0).chain(e.iter().copied()).chain(x.iter().copied()).collect() };
    let z: Matrix = (0..n).map(|i| std::iter::once(1.0).chain(instruments[i].iter().copied()).chain(exog[i].iter().copied()).collect()).collect();
    // first stage: regress each endogenous column on the instruments + controls
    let first: Vec<Vec<f64>> = (0..endog[0].len()).map(|k| ols(&z, &endog.iter().map(|r| r[k]).collect::<Vec<_>>()).ok_or_else(|| Error::Msg("two_sls: weak or collinear instruments".into()))).collect::<Result<_>>()?;
    let fitted: Matrix = (0..n).map(|i| first.iter().map(|b| z[i].iter().zip(b).map(|(a, c)| a * c).sum()).collect()).collect();
    let stage2: Matrix = (0..n).map(|i| row(&fitted[i], &exog[i])).collect();
    ols(&stage2, y).ok_or_else(|| Error::Msg("two_sls: singular second stage".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    fn gauss(rng: &mut Rng) -> f64 {
        let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn multiple_testing_corrections() {
        let p = [0.01, 0.04, 0.03, 0.005];
        assert_eq!(bonferroni(&p).unwrap(), vec![0.04, 0.16, 0.12, 0.02]);
        // Holm: sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 with monotone enforcement
        let h = holm(&p).unwrap();
        for (got, want) in h.iter().zip([0.03, 0.06, 0.06, 0.02]) {
            close(*got, want, 1e-12);
        }
        // BH on the classic example: p adjusted are non-decreasing in rank and never above Holm's
        let bh = benjamini_hochberg(&p).unwrap();
        for (got, want) in bh.iter().zip([0.02, 0.04, 0.04, 0.02]) {
            close(*got, want, 1e-12);
        }
        assert!(bh.iter().zip(&h).all(|(a, b)| a <= b));
        assert!(bonferroni(&[]).is_err() && holm(&[1.5]).is_err());
    }

    #[test]
    fn normal_quantiles_and_tests() {
        close(norm_ppf(0.975), 1.959_964, 1e-5);
        close(norm_ppf(0.5), 0.0, 1e-12);
        close(norm_ppf(0.001), -3.090_232, 1e-5);
        close(norm_cdf(norm_ppf(0.3)), 0.3, 1e-8);
        // 200/1000 vs 250/1000: pooled p = 0.225, se = sqrt(0.225 * 0.775 * 0.002) = 0.018675, z = -0.05 / se = -2.677
        let r = two_proportion_ztest(200, 1000, 250, 1000).unwrap();
        close(r.statistic, -2.677, 0.001);
        assert!(r.p_value < 0.02 && r.p_value > 0.005);
        assert!(two_proportion_ztest(5, 0, 1, 10).is_err() && two_proportion_ztest(0, 10, 0, 10).is_err());
    }

    #[test]
    fn sample_size_and_power_agree() {
        // classic: detect 0.5 sd shift at alpha 0.05, 80% power -> ~63 per group
        let n = sample_size_means(0.5, 1.0, 0.05, 0.8).unwrap();
        assert!((62..=65).contains(&n), "{n}");
        close(power_means(0.5, 1.0, n, 0.05), 0.8, 0.01);
        // 10% -> 12% conversion needs ~3,800 per group
        let np = sample_size_proportions(0.10, 0.12, 0.05, 0.8).unwrap();
        assert!((3600..=4000).contains(&np), "{np}");
        assert!(sample_size_means(0.2, 1.0, 0.05, 0.8).unwrap() > 4 * n - 40, "smaller effects need quadratically more data");
        assert!(sample_size_means(0.0, 1.0, 0.05, 0.8).is_err() && sample_size_proportions(0.1, 0.1, 0.05, 0.8).is_err());
    }

    #[test]
    fn causal_estimators_recover_known_effects() {
        // confounded observational data: x raises both treatment probability and the outcome; true effect = 2
        let mut rng = Rng::new(5);
        let n = 4000;
        let x: Matrix = (0..n).map(|_| vec![gauss(&mut rng)]).collect();
        let treated: Vec<bool> = x.iter().map(|r| rng.next_f64() < 1.0 / (1.0 + (-1.2 * r[0]).exp())).collect();
        let y: Vec<f64> = x.iter().zip(&treated).map(|(r, &t)| 2.0 * f64::from(u8::from(t)) + 3.0 * r[0] + 0.5 * gauss(&mut rng)).collect();
        let naive = y.iter().zip(&treated).filter(|(_, t)| **t).map(|(v, _)| v).sum::<f64>() / treated.iter().filter(|t| **t).count() as f64
            - y.iter().zip(&treated).filter(|(_, t)| !**t).map(|(v, _)| v).sum::<f64>() / treated.iter().filter(|t| !**t).count() as f64;
        let ipw = ipw_ate(&x, &treated, &y, 0.02).unwrap();
        assert!(naive > 3.5, "confounding inflates the naive estimate: {naive}");
        assert!((ipw - 2.0).abs() < 0.5, "IPW {ipw}");
        // diff-in-diff cancels a shared time trend
        close(diff_in_diff(&[10.0, 12.0], &[16.0, 18.0], &[5.0, 7.0], &[8.0, 10.0]).unwrap(), 3.0, 1e-12);
        assert!(diff_in_diff(&[], &[1.0], &[1.0], &[1.0]).is_err());
        // IV: unobserved u confounds x and y; z moves x only. OLS is biased, 2SLS recovers beta = 1.5
        let mut data = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..5000 {
            let (z, u, e) = (gauss(&mut rng), gauss(&mut rng), gauss(&mut rng));
            let xv = 0.8 * z + 1.0 * u + 0.3 * e;
            data.0.push(1.5 * xv + 2.0 * u + 0.3 * gauss(&mut rng));
            data.1.push(vec![xv]);
            data.2.push(vec![z]);
        }
        let exog: Matrix = vec![vec![]; 5000];
        let ols_beta = ols(&data.1.iter().map(|r| vec![1.0, r[0]]).collect(), &data.0).unwrap()[1];
        let iv = two_sls(&data.0, &data.1, &exog, &data.2).unwrap();
        assert!((ols_beta - 1.5).abs() > 0.5, "OLS is biased: {ols_beta}");
        assert!((iv[1] - 1.5).abs() < 0.15, "2SLS {iv:?}");
        assert!(two_sls(&data.0, &data.1, &exog, &vec![vec![]; 5000]).is_err());
    }
}
