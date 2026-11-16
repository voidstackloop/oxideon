//! Classical hypothesis tests and distribution functions on `&[f64]`: normal / Student-t / chi-square / F CDFs,
//! [`ttest_1samp`], [`ttest_ind`] (Welch), [`ttest_rel`], [`chi2_gof`], [`chi2_independence`], [`f_oneway`],
//! [`mannwhitneyu`], [`wilcoxon_signed_rank`], [`ks_2samp`], [`pearsonr`], [`spearmanr`], [`kendalltau`],
//! [`cohen_d`], and [`bootstrap_ci`]. p-values are two-sided unless stated.

use crate::error::{Error, Result};
use crate::rng::Rng;

// ------------------------------------------------------------------ special functions

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation (g = 7, n = 9)
    const C: [f64; 9] = [0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8, 771.323_428_777_653_1, -176.615_029_162_140_6, 12.507_343_278_686_905, -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6, 1.505_632_735_149_311_6e-7];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let s = C[0] + (1..9).map(|i| C[i] / (x + i as f64)).sum::<f64>();
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

/// Regularized lower incomplete gamma `P(a, x)`.
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        let (mut term, mut sum) = (1.0 / a, 1.0 / a);
        for n in 1..500 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Lentz continued fraction for Q
        let tiny = 1e-300;
        let (mut b, mut c, mut d) = (x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a));
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// Regularized incomplete beta `I_x(a, b)`.
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    let cf = |a: f64, b: f64, x: f64| -> f64 {
        let tiny = 1e-300;
        let (mut c, mut d) = (1.0, 1.0 - (a + b) * x / (a + 1.0));
        if d.abs() < tiny {
            d = tiny;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..500 {
            let m = m as f64;
            let aa = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
            d = 1.0 + aa * d;
            if d.abs() < tiny { d = tiny; }
            c = 1.0 + aa / c;
            if c.abs() < tiny { c = tiny; }
            d = 1.0 / d;
            h *= d * c;
            let aa = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
            d = 1.0 + aa * d;
            if d.abs() < tiny { d = tiny; }
            c = 1.0 + aa / c;
            if c.abs() < tiny { c = tiny; }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h
    };
    if x < (a + 1.0) / (a + b + 2.0) { front * cf(a, b, x) / a } else { 1.0 - front * cf(b, a, 1.0 - x) / b }
}

/// Standard normal CDF.
pub fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // erf(x) = P(1/2, x^2) for x >= 0
    gamma_p(0.5, x * x).copysign(x)
}

/// Student-t CDF with `df` degrees of freedom.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let tail = 0.5 * beta_inc(0.5 * df, 0.5, x);
    if t > 0.0 { 1.0 - tail } else { tail }
}

/// Chi-square CDF.
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    gamma_p(0.5 * df, 0.5 * x)
}

/// F-distribution CDF.
pub fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { beta_inc(0.5 * d1, 0.5 * d2, d1 * x / (d1 * x + d2)) }
}

/// Result of a test: the statistic and its p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn var(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)
}

fn need(x: &[f64], min: usize, op: &str) -> Result<()> {
    if x.len() < min || x.iter().any(|v| !v.is_finite()) {
        return Err(Error::Msg(format!("{op}: need at least {min} finite values")));
    }
    Ok(())
}

fn two_sided_t(t: f64, df: f64) -> TestResult {
    TestResult { statistic: t, p_value: (2.0 * (1.0 - t_cdf(t.abs(), df))).clamp(0.0, 1.0) }
}

/// One-sample t-test of the mean against `mu`.
pub fn ttest_1samp(x: &[f64], mu: f64) -> Result<TestResult> {
    need(x, 2, "ttest_1samp")?;
    let se = (var(x) / x.len() as f64).sqrt();
    Ok(two_sided_t((mean(x) - mu) / se, x.len() as f64 - 1.0))
}

/// Welch's two-sample t-test (unequal variances).
pub fn ttest_ind(a: &[f64], b: &[f64]) -> Result<TestResult> {
    need(a, 2, "ttest_ind")?;
    need(b, 2, "ttest_ind")?;
    let (va, vb) = (var(a) / a.len() as f64, var(b) / b.len() as f64);
    let df = (va + vb).powi(2) / (va * va / (a.len() as f64 - 1.0) + vb * vb / (b.len() as f64 - 1.0));
    Ok(two_sided_t((mean(a) - mean(b)) / (va + vb).sqrt(), df))
}

/// Paired t-test.
pub fn ttest_rel(a: &[f64], b: &[f64]) -> Result<TestResult> {
    if a.len() != b.len() {
        return Err(Error::Msg("ttest_rel: samples must have equal length".into()));
    }
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    ttest_1samp(&d, 0.0)
}

/// Chi-square goodness of fit of observed counts against expected counts (same total).
pub fn chi2_gof(observed: &[f64], expected: &[f64]) -> Result<TestResult> {
    if observed.len() != expected.len() || observed.len() < 2 || expected.iter().any(|&e| e <= 0.0) {
        return Err(Error::Msg("chi2_gof: need matching counts (>= 2 categories) and positive expectations".into()));
    }
    let stat: f64 = observed.iter().zip(expected).map(|(o, e)| (o - e).powi(2) / e).sum();
    Ok(TestResult { statistic: stat, p_value: 1.0 - chi2_cdf(stat, observed.len() as f64 - 1.0) })
}

/// Chi-square test of independence for a contingency table (no continuity correction).
pub fn chi2_independence(table: &[Vec<f64>]) -> Result<TestResult> {
    let (r, c) = (table.len(), table.first().map_or(0, Vec::len));
    if r < 2 || c < 2 || table.iter().any(|row| row.len() != c || row.iter().any(|&v| v < 0.0)) {
        return Err(Error::Msg("chi2_independence: need a rectangular table of non-negative counts, at least 2x2".into()));
    }
    let rows: Vec<f64> = table.iter().map(|row| row.iter().sum()).collect();
    let cols: Vec<f64> = (0..c).map(|j| table.iter().map(|row| row[j]).sum()).collect();
    let total: f64 = rows.iter().sum();
    if rows.iter().chain(&cols).any(|&s| s == 0.0) {
        return Err(Error::Msg("chi2_independence: a row or column is empty".into()));
    }
    let stat: f64 = (0..r).flat_map(|i| (0..c).map(move |j| (i, j))).map(|(i, j)| (table[i][j] - rows[i] * cols[j] / total).powi(2) / (rows[i] * cols[j] / total)).sum();
    Ok(TestResult { statistic: stat, p_value: 1.0 - chi2_cdf(stat, ((r - 1) * (c - 1)) as f64) })
}

/// One-way ANOVA F-test across groups.
pub fn f_oneway(groups: &[&[f64]]) -> Result<TestResult> {
    if groups.len() < 2 || groups.iter().any(|g| g.len() < 2) {
        return Err(Error::Msg("f_oneway: need >= 2 groups with >= 2 values each".into()));
    }
    let all: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let (n, k, grand) = (all.len() as f64, groups.len() as f64, mean(&all));
    let ssb: f64 = groups.iter().map(|g| g.len() as f64 * (mean(g) - grand).powi(2)).sum();
    let ssw: f64 = groups.iter().map(|g| g.iter().map(|v| (v - mean(g)).powi(2)).sum::<f64>()).sum();
    let f = (ssb / (k - 1.0)) / (ssw / (n - k)).max(1e-300);
    Ok(TestResult { statistic: f, p_value: 1.0 - f_cdf(f, k - 1.0, n - k) })
}

/// Average ranks (ties share the mean rank) and the tie-correction term `sum(t^3 - t)`.
fn ranks(x: &[f64]) -> (Vec<f64>, f64) {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let (mut r, mut ties, mut i) = (vec![0.0; x.len()], 0.0, 0);
    while i < idx.len() {
        let mut j = i;
        while j < idx.len() && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        idx[i..j].iter().for_each(|&k| r[k] = avg);
        let t = (j - i) as f64;
        ties += t * t * t - t;
        i = j;
    }
    (r, ties)
}

/// Mann-Whitney U test (normal approximation with tie and continuity correction).
pub fn mannwhitneyu(a: &[f64], b: &[f64]) -> Result<TestResult> {
    need(a, 2, "mannwhitneyu")?;
    need(b, 2, "mannwhitneyu")?;
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let all: Vec<f64> = a.iter().chain(b).copied().collect();
    let (r, ties) = ranks(&all);
    let u1 = r[..a.len()].iter().sum::<f64>() - n1 * (n1 + 1.0) / 2.0;
    let n = n1 + n2;
    let sd = (n1 * n2 / 12.0 * ((n + 1.0) - ties / (n * (n - 1.0)))).sqrt();
    if sd == 0.0 {
        return Err(Error::Msg("mannwhitneyu: all values are identical".into()));
    }
    let z = ((u1 - n1 * n2 / 2.0).abs() - 0.5).max(0.0) / sd;
    Ok(TestResult { statistic: u1, p_value: (2.0 * (1.0 - norm_cdf(z))).clamp(0.0, 1.0) })
}

/// Wilcoxon signed-rank test on paired differences (normal approximation; zeros dropped).
pub fn wilcoxon_signed_rank(a: &[f64], b: &[f64]) -> Result<TestResult> {
    if a.len() != b.len() {
        return Err(Error::Msg("wilcoxon_signed_rank: samples must have equal length".into()));
    }
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).filter(|v| *v != 0.0).collect();
    need(&d, 5, "wilcoxon_signed_rank")?;
    let n = d.len() as f64;
    let abs: Vec<f64> = d.iter().map(|v| v.abs()).collect();
    let (r, ties) = ranks(&abs);
    let w_plus: f64 = d.iter().zip(&r).filter(|(v, _)| **v > 0.0).map(|(_, r)| r).sum();
    let mu = n * (n + 1.0) / 4.0;
    let sd = (n * (n + 1.0) * (2.0 * n + 1.0) / 24.0 - ties / 48.0).sqrt();
    let z = ((w_plus - mu).abs() - 0.5).max(0.0) / sd;
    Ok(TestResult { statistic: w_plus, p_value: (2.0 * (1.0 - norm_cdf(z))).clamp(0.0, 1.0) })
}

/// Two-sample Kolmogorov-Smirnov test (asymptotic p-value).
pub fn ks_2samp(a: &[f64], b: &[f64]) -> Result<TestResult> {
    need(a, 1, "ks_2samp")?;
    need(b, 1, "ks_2samp")?;
    let (mut sa, mut sb) = (a.to_vec(), b.to_vec());
    sa.sort_by(f64::total_cmp);
    sb.sort_by(f64::total_cmp);
    let (mut i, mut j, mut d) = (0, 0, 0.0f64);
    while i < sa.len() && j < sb.len() {
        let x = sa[i].min(sb[j]);
        while i < sa.len() && sa[i] <= x { i += 1; }
        while j < sb.len() && sb[j] <= x { j += 1; }
        d = d.max((i as f64 / sa.len() as f64 - j as f64 / sb.len() as f64).abs());
    }
    let en = (sa.len() as f64 * sb.len() as f64 / (sa.len() + sb.len()) as f64).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    let p = if lambda < 1e-9 { 1.0 } else { 2.0 * (1..=100).map(|k| (-1f64).powi(k - 1) * (-2.0 * (k as f64 * lambda).powi(2)).exp()).sum::<f64>() };
    Ok(TestResult { statistic: d, p_value: p.clamp(0.0, 1.0) })
}

/// Pearson correlation with a t-test p-value.
pub fn pearsonr(x: &[f64], y: &[f64]) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::Msg("pearsonr: samples must have equal length".into()));
    }
    need(x, 3, "pearsonr")?;
    let (mx, my) = (mean(x), mean(y));
    let (sxy, sxx, syy) = x.iter().zip(y).fold((0.0, 0.0, 0.0), |(a, b, c), (&p, &q)| (a + (p - mx) * (q - my), b + (p - mx).powi(2), c + (q - my).powi(2)));
    if sxx == 0.0 || syy == 0.0 {
        return Err(Error::Msg("pearsonr: an input is constant".into()));
    }
    let r = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    let df = x.len() as f64 - 2.0;
    let p = if r.abs() == 1.0 { 0.0 } else { two_sided_t(r * (df / (1.0 - r * r)).sqrt(), df).p_value };
    Ok(TestResult { statistic: r, p_value: p })
}

/// Spearman rank correlation (Pearson on average ranks).
pub fn spearmanr(x: &[f64], y: &[f64]) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::Msg("spearmanr: samples must have equal length".into()));
    }
    pearsonr(&ranks(x).0, &ranks(y).0)
}

/// Kendall's tau-b with a normal-approximation p-value.
pub fn kendalltau(x: &[f64], y: &[f64]) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::Msg("kendalltau: samples must have equal length".into()));
    }
    need(x, 3, "kendalltau")?;
    let n = x.len();
    let (mut conc, mut disc, mut tx, mut ty) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        for j in i + 1..n {
            let (dx, dy) = (x[i] - x[j], y[i] - y[j]);
            if dx == 0.0 && dy == 0.0 { continue; }
            if dx == 0.0 { tx += 1.0; } else if dy == 0.0 { ty += 1.0; } else if dx * dy > 0.0 { conc += 1.0; } else { disc += 1.0; }
        }
    }
    let denom = ((conc + disc + tx) * (conc + disc + ty)).sqrt();
    if denom == 0.0 {
        return Err(Error::Msg("kendalltau: an input is constant".into()));
    }
    let tau = (conc - disc) / denom;
    let nf = n as f64;
    let z = 3.0 * (conc - disc) / (nf * (nf - 1.0) * (2.0 * nf + 5.0) / 2.0).sqrt();
    Ok(TestResult { statistic: tau, p_value: (2.0 * (1.0 - norm_cdf(z.abs()))).clamp(0.0, 1.0) })
}

/// Cohen's d effect size with the pooled standard deviation.
pub fn cohen_d(a: &[f64], b: &[f64]) -> Result<f64> {
    need(a, 2, "cohen_d")?;
    need(b, 2, "cohen_d")?;
    let pooled = (((a.len() - 1) as f64 * var(a) + (b.len() - 1) as f64 * var(b)) / (a.len() + b.len() - 2) as f64).sqrt();
    if pooled == 0.0 {
        return Err(Error::Msg("cohen_d: both samples are constant".into()));
    }
    Ok((mean(a) - mean(b)) / pooled)
}

/// Percentile bootstrap confidence interval of `statistic` at level `1 - alpha`.
pub fn bootstrap_ci<F: Fn(&[f64]) -> f64>(x: &[f64], statistic: F, n_resamples: usize, alpha: f64, seed: u64) -> Result<(f64, f64)> {
    need(x, 2, "bootstrap_ci")?;
    if n_resamples < 20 || !(0.0 < alpha && alpha < 1.0) {
        return Err(Error::Msg("bootstrap_ci: need n_resamples >= 20 and 0 < alpha < 1".into()));
    }
    let mut rng = Rng::new(seed);
    let mut stats: Vec<f64> = (0..n_resamples).map(|_| { let s: Vec<f64> = (0..x.len()).map(|_| x[(rng.next_u64() % x.len() as u64) as usize]).collect(); statistic(&s) }).collect();
    stats.sort_by(f64::total_cmp);
    let at = |q: f64| stats[((q * (n_resamples - 1) as f64).round() as usize).min(n_resamples - 1)];
    Ok((at(alpha / 2.0), at(1.0 - alpha / 2.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn distribution_functions_match_reference_values() {
        close(norm_cdf(1.96), 0.975_002_1, 1e-6);
        close(norm_cdf(0.0), 0.5, 1e-12);
        close(1.0 - t_cdf(2.0, 10.0), 0.036_694_0, 1e-6); // scipy t.sf(2, 10)
        close(t_cdf(-1.5, 5.0) + t_cdf(1.5, 5.0), 1.0, 1e-12);
        close(1.0 - chi2_cdf(3.841_458_8, 1.0), 0.05, 1e-6);
        close(1.0 - chi2_cdf(11.07, 5.0), 0.05, 1e-3); // critical value of chi2(5) at 5%
        close(1.0 - f_cdf(4.1, 2.0, 30.0), 0.0266, 5e-3);
        close(ln_gamma(5.0), 24f64.ln(), 1e-12);
        close(beta_inc(2.0, 3.0, 0.4), 0.5248, 1e-4);
    }

    #[test]
    fn t_tests() {
        // hand-computed: mean 5.442857, sd 0.395209, se 0.149369 -> t = 2.9647 on 6 d.o.f.
        let r = ttest_1samp(&[5.1, 4.9, 5.6, 5.8, 6.0, 5.2, 5.5], 5.0).unwrap();
        close(r.statistic, 2.9647, 1e-3);
        assert!(r.p_value > 0.02 && r.p_value < 0.03, "{}", r.p_value); // between the 5% (2.447) and 2% (3.143) critical values
        // clearly different means vs. identical-looking samples
        let (a, b) = ([10.0, 11.0, 9.5, 10.5, 10.2, 9.8], [12.0, 12.5, 11.8, 12.2, 12.9, 11.5]);
        assert!(ttest_ind(&a, &b).unwrap().p_value < 0.001);
        assert!(ttest_ind(&a, &[10.1, 10.3, 9.9, 10.4, 9.7, 10.0]).unwrap().p_value > 0.5);
        let paired = ttest_rel(&b, &a).unwrap();
        assert!(paired.statistic > 0.0 && paired.p_value < 0.001);
        assert!(ttest_1samp(&[1.0], 0.0).is_err() && ttest_rel(&a, &b[..3]).is_err());
    }

    #[test]
    fn categorical_and_anova() {
        // fair die: 60 rolls, perfectly uniform -> statistic 0, p = 1
        close(chi2_gof(&[10.0; 6], &[10.0; 6]).unwrap().p_value, 1.0, 1e-12);
        // scipy chisquare([16,18,16,14,12,12], [16,16,16,16,16,8]) -> statistic 3.5
        close(chi2_gof(&[16.0, 18.0, 16.0, 14.0, 12.0, 12.0], &[16.0, 16.0, 16.0, 16.0, 16.0, 8.0]).unwrap().statistic, 3.5, 1e-12);
        // strong association vs none
        assert!(chi2_independence(&[vec![50.0, 10.0], vec![10.0, 50.0]]).unwrap().p_value < 1e-6);
        assert!(chi2_independence(&[vec![25.0, 25.0], vec![25.0, 25.0]]).unwrap().p_value > 0.99);
        assert!(chi2_independence(&[vec![1.0, 0.0], vec![1.0, 0.0]]).is_err());
        // scipy f_oneway([6,8,4,5,3,4],[8,12,9,11,6,8],[13,9,11,8,7,12]) -> F = 9.2646, p = 0.00241
        let r = f_oneway(&[&[6.0, 8.0, 4.0, 5.0, 3.0, 4.0], &[8.0, 12.0, 9.0, 11.0, 6.0, 8.0], &[13.0, 9.0, 11.0, 8.0, 7.0, 12.0]]).unwrap();
        close(r.statistic, 9.2646, 1e-3);
        close(r.p_value, 0.00241, 2e-4);
    }

    #[test]
    fn rank_tests_and_correlations() {
        let (a, b) = ([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]);
        let mw = mannwhitneyu(&a, &b).unwrap();
        assert!(mw.p_value < 0.05 && mw.statistic < 16.0);
        let same: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!(mannwhitneyu(&same, &same).unwrap().p_value > 0.9);
        let after: Vec<f64> = a.iter().map(|v| v + 1.0 + (v * 3.7).sin() * 0.1).collect();
        assert!(wilcoxon_signed_rank(&after, &a).unwrap().p_value < 0.05);
        assert!(ks_2samp(&a, &b).unwrap().p_value < 0.05);
        close(ks_2samp(&same, &same).unwrap().p_value, 1.0, 1e-9);
        // perfectly monotone but non-linear: spearman = kendall = 1, pearson < 1
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| v.powi(3)).collect();
        close(spearmanr(&x, &y).unwrap().statistic, 1.0, 1e-12);
        close(kendalltau(&x, &y).unwrap().statistic, 1.0, 1e-12);
        let p = pearsonr(&x, &y).unwrap();
        assert!(p.statistic < 1.0 && p.statistic > 0.9 && p.p_value < 1e-3);
        close(pearsonr(&x, &x.iter().map(|v| -v).collect::<Vec<_>>()).unwrap().statistic, -1.0, 1e-12);
        assert!(pearsonr(&x, &[1.0; 10]).is_err());
        // scipy: kendalltau([1,2,3,4,5],[1,3,2,5,4]) tau = 0.6
        close(kendalltau(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 3.0, 2.0, 5.0, 4.0]).unwrap().statistic, 0.6, 1e-12);
    }

    #[test]
    fn effect_size_and_bootstrap() {
        let (a, b) = ([10.0, 11.0, 9.0, 10.0], [12.0, 13.0, 11.0, 12.0]);
        close(cohen_d(&b, &a).unwrap(), 2.0 / (2.0f64 / 3.0).sqrt(), 1e-9);
        let data: Vec<f64> = (0..200).map(|i| 5.0 + ((i * 37 % 101) as f64 - 50.0) / 25.0).collect();
        let (lo, hi) = bootstrap_ci(&data, |s| s.iter().sum::<f64>() / s.len() as f64, 500, 0.05, 1).unwrap();
        let m = data.iter().sum::<f64>() / 200.0;
        assert!(lo < m && m < hi && hi - lo < 0.5 && hi - lo > 0.05, "{lo} {hi} {m}");
        assert!(bootstrap_ci(&data, |s| s[0], 5, 0.05, 1).is_err());
    }
}
