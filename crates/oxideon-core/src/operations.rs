//! Operations-research and reliability formulas: queueing models ([`mm1`], [`mmc`], [`erlang_b`], [`mg1`]),
//! inventory ([`eoq`], [`newsvendor`]), reliability ([`weibull_reliability`], [`series_reliability`],
//! [`parallel_reliability`], [`k_out_of_n`], [`mtbf_availability`]), critical-path scheduling ([`critical_path`]),
//! and an event-driven single-server [`simulate_queue`] for checking the formulas.

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::stats_tests::norm_cdf;

/// Steady-state metrics of a queueing system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueueMetrics {
    pub utilization: f64,
    /// Mean number in the system.
    pub l: f64,
    /// Mean number waiting.
    pub lq: f64,
    /// Mean time in the system.
    pub w: f64,
    /// Mean waiting time.
    pub wq: f64,
}

/// M/M/1 queue with arrival rate `lambda` and service rate `mu`.
pub fn mm1(lambda: f64, mu: f64) -> Result<QueueMetrics> {
    if lambda <= 0.0 || mu <= 0.0 || lambda >= mu {
        return Err(Error::Msg("mm1: need 0 < lambda < mu for a stable queue".into()));
    }
    let rho = lambda / mu;
    Ok(QueueMetrics { utilization: rho, l: rho / (1.0 - rho), lq: rho * rho / (1.0 - rho), w: 1.0 / (mu - lambda), wq: rho / (mu - lambda) })
}

/// Erlang-B blocking probability for `servers` servers and offered load `a = lambda / mu` (no waiting room).
pub fn erlang_b(servers: u32, offered_load: f64) -> Result<f64> {
    if servers == 0 || offered_load < 0.0 {
        return Err(Error::Msg("erlang_b: need servers >= 1 and a non-negative load".into()));
    }
    Ok((1..=servers).fold(1.0, |b, k| offered_load * b / (k as f64 + offered_load * b)))
}

/// M/M/c queue (Erlang-C): returns the metrics and the probability that an arrival has to wait.
pub fn mmc(lambda: f64, mu: f64, servers: u32) -> Result<(QueueMetrics, f64)> {
    if lambda <= 0.0 || mu <= 0.0 || servers == 0 || lambda >= servers as f64 * mu {
        return Err(Error::Msg("mmc: need positive rates and lambda < servers * mu".into()));
    }
    let (a, c) = (lambda / mu, servers as f64);
    let rho = a / c;
    let b = erlang_b(servers, a)?;
    let p_wait = b / (1.0 - rho * (1.0 - b));
    let wq = p_wait / (c * mu - lambda);
    let lq = lambda * wq;
    Ok((QueueMetrics { utilization: rho, l: lq + a, lq, w: wq + 1.0 / mu, wq }, p_wait))
}

/// M/G/1 queue via Pollaczek-Khinchine: service time has mean `1 / mu` and squared coefficient of variation `cs2`.
pub fn mg1(lambda: f64, mu: f64, cs2: f64) -> Result<QueueMetrics> {
    if lambda <= 0.0 || mu <= 0.0 || lambda >= mu || cs2 < 0.0 {
        return Err(Error::Msg("mg1: need 0 < lambda < mu and cs2 >= 0".into()));
    }
    let rho = lambda / mu;
    let wq = rho * (1.0 + cs2) / (2.0 * mu * (1.0 - rho));
    Ok(QueueMetrics { utilization: rho, l: lambda * (wq + 1.0 / mu), lq: lambda * wq, w: wq + 1.0 / mu, wq })
}

/// Economic order quantity and the resulting annual cost for `demand` units/year, `order_cost` per order and
/// `holding_cost` per unit per year.
pub fn eoq(demand: f64, order_cost: f64, holding_cost: f64) -> Result<(f64, f64)> {
    if demand <= 0.0 || order_cost <= 0.0 || holding_cost <= 0.0 {
        return Err(Error::Msg("eoq: all inputs must be positive".into()));
    }
    let q = (2.0 * demand * order_cost / holding_cost).sqrt();
    Ok((q, demand / q * order_cost + q / 2.0 * holding_cost))
}

/// Newsvendor optimal stock for normally distributed demand `N(mean, sd^2)`: the critical-ratio quantile
/// `cu / (cu + co)` with underage cost `cu` and overage cost `co`.
pub fn newsvendor(mean: f64, sd: f64, underage: f64, overage: f64) -> Result<f64> {
    if sd < 0.0 || underage <= 0.0 || overage <= 0.0 {
        return Err(Error::Msg("newsvendor: need sd >= 0 and positive costs".into()));
    }
    let ratio = underage / (underage + overage);
    Ok(mean + sd * crate::causal::norm_ppf(ratio))
}

/// Weibull survival function `exp(-(t / scale)^shape)`.
pub fn weibull_reliability(t: f64, shape: f64, scale: f64) -> Result<f64> {
    if t < 0.0 || shape <= 0.0 || scale <= 0.0 {
        return Err(Error::Msg("weibull_reliability: need t >= 0 and positive shape and scale".into()));
    }
    Ok((-(t / scale).powf(shape)).exp())
}

fn check_rel(r: &[f64], op: &str) -> Result<()> {
    if r.is_empty() || r.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return Err(Error::Msg(format!("{op}: need non-empty component reliabilities in [0, 1]")));
    }
    Ok(())
}

/// Reliability of components in series (all must work).
pub fn series_reliability(r: &[f64]) -> Result<f64> {
    check_rel(r, "series_reliability")?;
    Ok(r.iter().product())
}

/// Reliability of components in parallel (at least one must work).
pub fn parallel_reliability(r: &[f64]) -> Result<f64> {
    check_rel(r, "parallel_reliability")?;
    Ok(1.0 - r.iter().map(|v| 1.0 - v).product::<f64>())
}

/// Reliability of a k-out-of-n system of identical components with reliability `r`.
pub fn k_out_of_n(k: u32, n: u32, r: f64) -> Result<f64> {
    if k == 0 || k > n || !(0.0..=1.0).contains(&r) {
        return Err(Error::Msg("k_out_of_n: need 1 <= k <= n and r in [0, 1]".into()));
    }
    Ok((k..=n).map(|j| crate::numtheory::binomial(n as u64, j as u64).unwrap_or(0) as f64 * r.powi(j as i32) * (1.0 - r).powi((n - j) as i32)).sum())
}

/// Steady-state availability `MTBF / (MTBF + MTTR)`.
pub fn mtbf_availability(mtbf: f64, mttr: f64) -> Result<f64> {
    if mtbf <= 0.0 || mttr < 0.0 {
        return Err(Error::Msg("mtbf_availability: need mtbf > 0 and mttr >= 0".into()));
    }
    Ok(mtbf / (mtbf + mttr))
}

/// Critical-path method over tasks given as `(duration, predecessor indices)`; predecessors must have smaller
/// indices. Returns `(project duration, earliest start per task, critical task indices)`.
pub fn critical_path(tasks: &[(f64, Vec<usize>)]) -> Result<(f64, Vec<f64>, Vec<usize>)> {
    let n = tasks.len();
    if n == 0 || tasks.iter().enumerate().any(|(i, t)| t.0 < 0.0 || t.1.iter().any(|&p| p >= i)) {
        return Err(Error::Msg("critical_path: durations must be non-negative and predecessors must precede their task".into()));
    }
    let mut es = vec![0.0; n];
    for i in 0..n {
        es[i] = tasks[i].1.iter().map(|&p| es[p] + tasks[p].0).fold(0.0, f64::max);
    }
    let total = (0..n).map(|i| es[i] + tasks[i].0).fold(0.0, f64::max);
    let mut lf = vec![total; n];
    for i in (0..n).rev() {
        for &p in &tasks[i].1 {
            lf[p] = lf[p].min(lf[i] - tasks[i].0);
        }
    }
    let critical = (0..n).filter(|&i| (lf[i] - tasks[i].0 - es[i]).abs() < 1e-9).collect();
    Ok((total, es, critical))
}

/// Discrete-event simulation of a single-server FIFO queue with exponential interarrival (rate `lambda`) and
/// service (rate `mu`) times; returns the mean waiting time in the queue over `customers` customers.
pub fn simulate_queue(lambda: f64, mu: f64, customers: usize, seed: u64) -> Result<f64> {
    if lambda <= 0.0 || mu <= 0.0 || customers == 0 {
        return Err(Error::Msg("simulate_queue: need positive rates and customers >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let exp = |rng: &mut Rng, rate: f64| -rng.next_f64().max(1e-300).ln() / rate;
    let (mut arrival, mut free_at, mut total_wait) = (0.0, 0.0f64, 0.0);
    for _ in 0..customers {
        arrival += exp(&mut rng, lambda);
        let start = arrival.max(free_at);
        total_wait += start - arrival;
        free_at = start + exp(&mut rng, mu);
    }
    Ok(total_wait / customers as f64)
}

/// Probability that a normal quantity `N(mean, sd^2)` falls below `limit` (e.g. on-time completion probability).
pub fn prob_below(mean: f64, sd: f64, limit: f64) -> f64 {
    if sd <= 0.0 { f64::from(u8::from(mean <= limit)) } else { norm_cdf((limit - mean) / sd) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn queueing_formulas() {
        let m = mm1(8.0, 10.0).unwrap();
        close(m.utilization, 0.8, 1e-12);
        close(m.l, 4.0, 1e-12);
        close(m.w, 0.5, 1e-12);
        close(m.l, 8.0 * m.w, 1e-12); // Little's law
        close(m.lq, 8.0 * m.wq, 1e-12);
        assert!(mm1(10.0, 10.0).is_err());
        // Erlang-B: 5 erlangs on 10 servers -> 1.84% blocking (classic table value)
        close(erlang_b(10, 5.0).unwrap(), 0.018_385, 1e-5);
        assert_eq!(erlang_b(1, 0.0).unwrap(), 0.0);
        // M/M/c with c = 1 reduces to M/M/1
        let (one, pw) = mmc(8.0, 10.0, 1).unwrap();
        close(one.wq, m.wq, 1e-12);
        close(pw, 0.8, 1e-12);
        // more servers cut waiting
        let (three, _) = mmc(8.0, 10.0 / 3.0, 3).unwrap();
        assert!(mmc(8.0, 2.0, 3).is_err() && three.wq > 0.0);
        let (many, p_many) = mmc(8.0, 10.0, 3).unwrap();
        assert!(many.wq < one.wq / 10.0 && p_many < pw);
        // M/G/1: exponential service (cs2 = 1) equals M/M/1; deterministic service halves the wait
        close(mg1(8.0, 10.0, 1.0).unwrap().wq, m.wq, 1e-12);
        close(mg1(8.0, 10.0, 0.0).unwrap().wq, m.wq / 2.0, 1e-12);
    }

    #[test]
    fn simulation_agrees_with_theory() {
        let sim = simulate_queue(8.0, 10.0, 400_000, 3).unwrap();
        let theory = mm1(8.0, 10.0).unwrap().wq;
        assert!((sim - theory).abs() / theory < 0.05, "simulated {sim} vs theoretical {theory}");
        assert!(simulate_queue(0.0, 1.0, 10, 0).is_err());
    }

    #[test]
    fn inventory_and_scheduling() {
        let (q, cost) = eoq(1000.0, 10.0, 0.5).unwrap();
        close(q, 200.0, 1e-9);
        close(cost, 100.0, 1e-9);
        // total cost is minimized at q: nearby order sizes cost more
        let tc = |x: f64| 1000.0 / x * 10.0 + x / 2.0 * 0.5;
        assert!(tc(150.0) > cost && tc(260.0) > cost);
        // newsvendor with critical ratio 0.75 -> mean + 0.6745 sd
        close(newsvendor(100.0, 20.0, 3.0, 1.0).unwrap(), 100.0 + 20.0 * 0.674_49, 1e-3);
        assert!(eoq(0.0, 1.0, 1.0).is_err());
        // tasks: A(3) -> C(2), B(5) -> C(2), D(1) after C: critical path B, C, D = 8
        let tasks = vec![(3.0, vec![]), (5.0, vec![]), (2.0, vec![0, 1]), (1.0, vec![2])];
        let (total, es, crit) = critical_path(&tasks).unwrap();
        assert_eq!((total, es), (8.0, vec![0.0, 0.0, 5.0, 7.0]));
        assert_eq!(crit, vec![1, 2, 3]);
        assert!(critical_path(&[(1.0, vec![0])]).is_err() && critical_path(&[]).is_err());
        close(prob_below(10.0, 2.0, 12.0), 0.841_345, 1e-5);
    }

    #[test]
    fn reliability_models() {
        // exponential special case: shape 1, scale = 1/lambda
        close(weibull_reliability(100.0, 1.0, 100.0).unwrap(), (-1.0f64).exp(), 1e-12);
        close(weibull_reliability(0.0, 2.0, 5.0).unwrap(), 1.0, 1e-12);
        close(series_reliability(&[0.9, 0.95, 0.99]).unwrap(), 0.9 * 0.95 * 0.99, 1e-12);
        close(parallel_reliability(&[0.9, 0.9]).unwrap(), 0.99, 1e-12);
        // 2-out-of-3 of 0.9 components = 3 * 0.81 * 0.1 + 0.729 = 0.972
        close(k_out_of_n(2, 3, 0.9).unwrap(), 0.972, 1e-12);
        close(k_out_of_n(1, 3, 0.9).unwrap(), parallel_reliability(&[0.9; 3]).unwrap(), 1e-12);
        close(k_out_of_n(3, 3, 0.9).unwrap(), series_reliability(&[0.9; 3]).unwrap(), 1e-12);
        close(mtbf_availability(990.0, 10.0).unwrap(), 0.99, 1e-12);
        assert!(series_reliability(&[1.2]).is_err() && k_out_of_n(4, 3, 0.5).is_err() && weibull_reliability(-1.0, 1.0, 1.0).is_err());
    }
}
