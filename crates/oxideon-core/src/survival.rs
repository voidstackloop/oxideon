//! Survival analysis: [`kaplan_meier`] (with Greenwood variance), [`nelson_aalen`], the [`logrank_test`], Cox
//! proportional-hazards regression ([`CoxPh`], Breslow ties, Newton-Raphson), and the [`concordance_index`].
//! `time` is the follow-up time and `event` is `true` when the event was observed (`false` = censored).

use crate::error::{Error, Result};
use crate::stats_tests::chi2_cdf;

fn check(time: &[f64], event: &[bool], op: &str) -> Result<()> {
    if time.is_empty() || time.len() != event.len() || time.iter().any(|t| !t.is_finite() || *t < 0.0) {
        return Err(Error::Msg(format!("{op}: need matching non-empty finite non-negative times and event flags")));
    }
    Ok(())
}

/// Step function of a survival / cumulative-hazard estimate at its distinct event times.
#[derive(Debug, Clone)]
pub struct StepCurve {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    /// Pointwise variance (Greenwood for Kaplan-Meier; Aalen for Nelson-Aalen).
    pub variance: Vec<f64>,
    initial: f64,
}

impl StepCurve {
    /// Value of the curve at time `t` (right-continuous).
    pub fn at(&self, t: f64) -> f64 {
        match self.times.partition_point(|&x| x <= t) {
            0 => self.initial,
            i => self.values[i - 1],
        }
    }

    /// Smallest time at which the curve drops to `level` or below (`None` if it never does).
    pub fn quantile_time(&self, level: f64) -> Option<f64> {
        self.values.iter().position(|&v| v <= level).map(|i| self.times[i])
    }
}

/// Distinct event times with (deaths, number at risk).
fn risk_table(time: &[f64], event: &[bool]) -> Vec<(f64, f64, f64)> {
    let mut ts: Vec<f64> = time.iter().zip(event).filter(|(_, e)| **e).map(|(t, _)| *t).collect();
    ts.sort_by(f64::total_cmp);
    ts.dedup();
    ts.into_iter()
        .map(|t| {
            let deaths = time.iter().zip(event).filter(|(x, e)| **e && **x == t).count() as f64;
            let at_risk = time.iter().filter(|&&x| x >= t).count() as f64;
            (t, deaths, at_risk)
        })
        .collect()
}

/// Kaplan-Meier survival estimate.
pub fn kaplan_meier(time: &[f64], event: &[bool]) -> Result<StepCurve> {
    check(time, event, "kaplan_meier")?;
    let (mut s, mut green) = (1.0, 0.0);
    let (mut times, mut values, mut variance) = (Vec::new(), Vec::new(), Vec::new());
    for (t, d, n) in risk_table(time, event) {
        s *= 1.0 - d / n;
        if n > d {
            green += d / (n * (n - d));
        }
        times.push(t);
        values.push(s);
        variance.push(s * s * green);
    }
    Ok(StepCurve { times, values, variance, initial: 1.0 })
}

/// Nelson-Aalen cumulative hazard estimate.
pub fn nelson_aalen(time: &[f64], event: &[bool]) -> Result<StepCurve> {
    check(time, event, "nelson_aalen")?;
    let (mut h, mut v) = (0.0, 0.0);
    let (mut times, mut values, mut variance) = (Vec::new(), Vec::new(), Vec::new());
    for (t, d, n) in risk_table(time, event) {
        h += d / n;
        v += d / (n * n);
        times.push(t);
        values.push(h);
        variance.push(v);
    }
    Ok(StepCurve { times, values, variance, initial: 0.0 })
}

/// Log-rank test comparing the survival of two groups (`group[i]` is `false` or `true`). Returns the chi-square
/// statistic (1 d.o.f.) and its p-value.
pub fn logrank_test(time: &[f64], event: &[bool], group: &[bool]) -> Result<(f64, f64)> {
    check(time, event, "logrank_test")?;
    if group.len() != time.len() || !group.contains(&true) || !group.contains(&false) {
        return Err(Error::Msg("logrank_test: need a group label per row and both groups present".into()));
    }
    let mut ts: Vec<f64> = time.iter().zip(event).filter(|(_, e)| **e).map(|(t, _)| *t).collect();
    ts.sort_by(f64::total_cmp);
    ts.dedup();
    let (mut o_minus_e, mut v) = (0.0, 0.0);
    for t in ts {
        let n = time.iter().filter(|&&x| x >= t).count() as f64;
        let n1 = time.iter().zip(group).filter(|(x, g)| **x >= t && **g).count() as f64;
        let d = time.iter().zip(event).filter(|(x, e)| **e && **x == t).count() as f64;
        let d1 = time.iter().zip(event).zip(group).filter(|((x, e), g)| **e && **x == t && **g).count() as f64;
        o_minus_e += d1 - d * n1 / n;
        if n > 1.0 {
            v += d * (n1 / n) * (1.0 - n1 / n) * (n - d) / (n - 1.0);
        }
    }
    if v <= 0.0 {
        return Err(Error::Msg("logrank_test: no variance (no events?)".into()));
    }
    let stat = o_minus_e * o_minus_e / v;
    Ok((stat, 1.0 - chi2_cdf(stat, 1.0)))
}

/// Fitted Cox proportional-hazards model.
#[derive(Debug, Clone)]
pub struct CoxPh {
    pub coef: Vec<f64>,
    /// Standard errors from the observed information matrix.
    pub se: Vec<f64>,
    pub log_likelihood: f64,
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
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

fn invert(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let cols: Vec<Vec<f64>> = (0..n).map(|j| solve(a.to_vec(), (0..n).map(|i| f64::from(u8::from(i == j))).collect())).collect::<Option<_>>()?;
    Some((0..n).map(|i| (0..n).map(|j| cols[j][i]).collect()).collect())
}

impl CoxPh {
    /// Partial likelihood fit with Breslow's tie handling; `l2` adds a ridge penalty (0 for none).
    pub fn fit(x: &[Vec<f64>], time: &[f64], event: &[bool], l2: f64) -> Result<Self> {
        check(time, event, "CoxPh")?;
        let d = x.first().map_or(0, Vec::len);
        if x.len() != time.len() || d == 0 || x.iter().any(|r| r.len() != d) || !event.contains(&true) {
            return Err(Error::Msg("CoxPh: need covariates per row and at least one event".into()));
        }
        let table = risk_table(time, event);
        let mut beta = vec![0.0; d];
        let mut ll = f64::NEG_INFINITY;
        let mut info_final = vec![vec![0.0; d]; d];
        for _ in 0..50 {
            let eta: Vec<f64> = x.iter().map(|r| r.iter().zip(&beta).map(|(a, b)| a * b).sum::<f64>()).collect();
            let (mut grad, mut info, mut cur) = (vec![0.0; d], vec![vec![0.0; d]; d], 0.0);
            for &(t, deaths, _) in &table {
                let risk: Vec<usize> = (0..x.len()).filter(|&i| time[i] >= t).collect();
                let w: Vec<f64> = risk.iter().map(|&i| eta[i].exp()).collect();
                let s0: f64 = w.iter().sum();
                let s1: Vec<f64> = (0..d).map(|j| risk.iter().zip(&w).map(|(&i, wi)| wi * x[i][j]).sum()).collect();
                let s2: Vec<Vec<f64>> = (0..d).map(|j| (0..d).map(|k| risk.iter().zip(&w).map(|(&i, wi)| wi * x[i][j] * x[i][k]).sum()).collect()).collect();
                for i in (0..x.len()).filter(|&i| event[i] && time[i] == t) {
                    cur += eta[i];
                    for j in 0..d {
                        grad[j] += x[i][j];
                    }
                }
                cur -= deaths * s0.ln();
                for j in 0..d {
                    grad[j] -= deaths * s1[j] / s0;
                    for k in 0..d {
                        info[j][k] += deaths * (s2[j][k] / s0 - s1[j] * s1[k] / (s0 * s0));
                    }
                }
            }
            cur -= 0.5 * l2 * beta.iter().map(|b| b * b).sum::<f64>();
            for j in 0..d {
                grad[j] -= l2 * beta[j];
                info[j][j] += l2;
            }
            let step = solve(info.clone(), grad.clone()).ok_or_else(|| Error::Msg("CoxPh: singular information matrix (collinear covariates? add l2)".into()))?;
            info_final = info;
            let converged = step.iter().all(|s| s.abs() < 1e-9) || (cur - ll).abs() < 1e-12;
            // damp the step if it overshoots (keeps wild first iterations finite)
            let scale = step.iter().fold(0.0f64, |m, s| m.max(s.abs())).max(5.0) / 5.0;
            beta.iter_mut().zip(&step).for_each(|(b, s)| *b += s / scale);
            ll = cur;
            if converged {
                break;
            }
        }
        let cov = invert(&info_final).ok_or_else(|| Error::Msg("CoxPh: could not invert the information matrix".into()))?;
        Ok(Self { se: (0..d).map(|j| cov[j][j].max(0.0).sqrt()).collect(), coef: beta, log_likelihood: ll })
    }

    /// Relative hazard `exp(x . beta)` for each row.
    pub fn hazard_ratio(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter().map(|r| r.iter().zip(&self.coef).map(|(a, b)| a * b).sum::<f64>().exp()).collect()
    }
}

/// Harrell's concordance index: the fraction of comparable pairs whose predicted risk ordering matches the observed
/// survival ordering (higher `risk` = earlier event). 0.5 is chance, 1 perfect.
pub fn concordance_index(time: &[f64], event: &[bool], risk: &[f64]) -> Result<f64> {
    check(time, event, "concordance_index")?;
    if risk.len() != time.len() {
        return Err(Error::Msg("concordance_index: one risk score per row".into()));
    }
    let (mut conc, mut total) = (0.0, 0.0);
    for i in 0..time.len() {
        for j in 0..time.len() {
            if event[i] && time[i] < time[j] {
                total += 1.0;
                if risk[i] > risk[j] {
                    conc += 1.0;
                } else if risk[i] == risk[j] {
                    conc += 0.5;
                }
            }
        }
    }
    if total == 0.0 {
        return Err(Error::Msg("concordance_index: no comparable pairs".into()));
    }
    Ok(conc / total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn kaplan_meier_and_nelson_aalen_by_hand() {
        // times 1,2,3,4,5 with a censoring at 3: S = 4/5, 3/5, (skip at 3), then 3/5 * 1/2, 0
        let time = [1.0, 2.0, 3.0, 4.0, 5.0];
        let event = [true, true, false, true, true];
        let km = kaplan_meier(&time, &event).unwrap();
        assert_eq!(km.times, vec![1.0, 2.0, 4.0, 5.0]);
        close(km.at(0.5), 1.0, 1e-12);
        close(km.at(1.0), 0.8, 1e-12);
        close(km.at(2.5), 0.6, 1e-12);
        close(km.at(3.5), 0.6, 1e-12); // censoring does not change the estimate
        close(km.at(4.0), 0.3, 1e-12);
        close(km.at(9.0), 0.0, 1e-12);
        assert_eq!(km.quantile_time(0.5), Some(4.0));
        assert!(km.variance[0] > 0.0);
        let na = nelson_aalen(&time, &event).unwrap();
        close(na.at(2.0), 1.0 / 5.0 + 1.0 / 4.0, 1e-12);
        close(na.at(5.0), 1.0 / 5.0 + 1.0 / 4.0 + 1.0 / 2.0 + 1.0, 1e-12);
        assert!(kaplan_meier(&[], &[]).is_err() && kaplan_meier(&[1.0], &[true, false]).is_err());
    }

    fn cohort(n: usize, hr: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<bool>) {
        // exponential survival with baseline hazard 0.1, scaled by hr for the treated; censoring at 25
        let mut rng = Rng::new(seed);
        let (mut x, mut t, mut e) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let treated = (i % 2) as f64;
            let rate = 0.1 * hr.powf(treated);
            let u = rng.next_f64().max(1e-12);
            let life = -u.ln() / rate;
            x.push(vec![treated]);
            t.push(life.min(25.0));
            e.push(life <= 25.0);
        }
        (x, t, e)
    }

    #[test]
    fn logrank_and_cox_recover_a_hazard_ratio() {
        let (x, t, e) = cohort(600, 2.0, 5);
        let group: Vec<bool> = x.iter().map(|r| r[0] == 1.0).collect();
        let (stat, p) = logrank_test(&t, &e, &group).unwrap();
        assert!(stat > 10.0 && p < 0.001, "{stat} {p}");
        let (x0, t0, e0) = cohort(600, 1.0, 6);
        let g0: Vec<bool> = x0.iter().map(|r| r[0] == 1.0).collect();
        assert!(logrank_test(&t0, &e0, &g0).unwrap().1 > 0.01, "no effect -> not significant");
        let cox = CoxPh::fit(&x, &t, &e, 0.0).unwrap();
        let hr = cox.coef[0].exp();
        assert!(hr > 1.6 && hr < 2.5, "hazard ratio {hr} should be near 2");
        assert!((cox.coef[0] / cox.se[0]).abs() > 3.0);
        assert_eq!(cox.hazard_ratio(&[vec![0.0], vec![1.0]]).len(), 2);
        // discriminates: treated rows are riskier
        let risk: Vec<f64> = cox.hazard_ratio(&x);
        let c = concordance_index(&t, &e, &risk).unwrap();
        assert!(c > 0.55 && c < 0.75, "{c}");
        assert!(logrank_test(&t, &e, &vec![true; t.len()]).is_err());
    }

    #[test]
    fn concordance_extremes() {
        let (time, event) = ([1.0, 2.0, 3.0, 4.0], [true; 4]);
        close(concordance_index(&time, &event, &[4.0, 3.0, 2.0, 1.0]).unwrap(), 1.0, 1e-12);
        close(concordance_index(&time, &event, &[1.0, 2.0, 3.0, 4.0]).unwrap(), 0.0, 1e-12);
        close(concordance_index(&time, &event, &[1.0; 4]).unwrap(), 0.5, 1e-12);
        assert!(concordance_index(&time, &[false; 4], &[1.0; 4]).is_err());
        assert!(CoxPh::fit(&[vec![1.0]], &[1.0], &[false], 0.0).is_err());
    }
}
