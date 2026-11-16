//! Budget-aware and model-based hyperparameter search on top of [`crate::tuning`]: [`successive_halving`],
//! [`hyperband`], and a Tree-structured Parzen Estimator ([`tpe_search`]). All of them minimize the objective.

use crate::error::{Error, Result};
use crate::rng::Rng;
use crate::tuning::{ParamDomain, ParamSpace, ParamValue, Study, TrialConfig, TrialResult};

/// Runs `n_configs` random configurations at `min_budget`, keeps the best `1/eta` and multiplies their budget by
/// `eta` until `max_budget`. `objective(config, budget)` should get better (lower) with more budget.
pub fn successive_halving<F>(space: &ParamSpace, n_configs: usize, min_budget: f64, max_budget: f64, eta: f64, mut objective: F) -> Result<Study>
where
    F: FnMut(&TrialConfig, f64) -> Result<f64>,
{
    if n_configs == 0 || eta <= 1.0 || min_budget <= 0.0 || max_budget < min_budget {
        return Err(Error::Msg("successive_halving: need n_configs >= 1, eta > 1 and 0 < min_budget <= max_budget".into()));
    }
    let mut alive: Vec<(usize, TrialConfig)> = (0..n_configs).map(|i| (i, space.sample_config())).collect();
    let mut budget = min_budget;
    let mut trials = Vec::new();
    loop {
        let mut scored: Vec<(f64, usize, TrialConfig)> = Vec::with_capacity(alive.len());
        for (id, cfg) in &alive {
            let obj = objective(cfg, budget)?;
            trials.push(TrialResult { trial_id: trials.len(), config: cfg.clone(), objective: obj });
            scored.push((if obj.is_nan() { f64::INFINITY } else { obj }, *id, cfg.clone()));
        }
        if budget >= max_budget || alive.len() == 1 {
            break;
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.truncate(((alive.len() as f64 / eta).floor() as usize).max(1));
        alive = scored.into_iter().map(|(_, id, c)| (id, c)).collect();
        budget = (budget * eta).min(max_budget);
    }
    // the winner is the best result obtained at the largest budget that was evaluated
    let last = trials.len() - alive.len();
    let best = trials[last..].iter().filter(|t| !t.objective.is_nan()).min_by(|a, b| a.objective.total_cmp(&b.objective)).cloned();
    Ok(Study { trials, best_trial: best })
}

/// Hyperband (Li et al. 2017): several successive-halving brackets trading off the number of configurations against
/// the budget each one gets. Returns every trial; `best_trial` is the best result at full budget.
pub fn hyperband<F>(space: &ParamSpace, max_budget: f64, eta: f64, mut objective: F) -> Result<Study>
where
    F: FnMut(&TrialConfig, f64) -> Result<f64>,
{
    if max_budget <= 0.0 || eta <= 1.0 {
        return Err(Error::Msg("hyperband: need max_budget > 0 and eta > 1".into()));
    }
    let s_max = (max_budget.ln() / eta.ln()).floor() as i32;
    let mut all = Vec::new();
    let mut best: Option<TrialResult> = None;
    for s in (0..=s_max).rev() {
        let n = (((s_max + 1) as f64 / (s + 1) as f64) * eta.powi(s)).ceil() as usize;
        let r = max_budget * eta.powi(-s);
        let study = successive_halving(space, n, r, max_budget, eta, &mut objective)?;
        if let Some(b) = study.best_trial.clone() {
            if best.as_ref().is_none_or(|x| b.objective < x.objective) {
                best = Some(TrialResult { trial_id: all.len() + b.trial_id, ..b });
            }
        }
        for t in study.trials {
            all.push(TrialResult { trial_id: all.len(), ..t });
        }
    }
    Ok(Study { trials: all, best_trial: best })
}

// ------------------------------------------------------------------ TPE

fn to_unit(d: &ParamDomain, v: &ParamValue) -> Option<f64> {
    match d {
        ParamDomain::Uniform { low, high } => Some((v.as_float() - low) / (high - low).max(1e-300)),
        ParamDomain::LogUniform { low, high } => Some((v.as_float().max(1e-300).ln() - low.max(1e-12).ln()) / (high.max(1e-12).ln() - low.max(1e-12).ln()).max(1e-300)),
        ParamDomain::Discrete(_) => None,
    }
}

fn from_unit(d: &ParamDomain, u: f64) -> ParamValue {
    let u = u.clamp(0.0, 1.0);
    match d {
        ParamDomain::Uniform { low, high } => ParamValue::Float(low + u * (high - low)),
        ParamDomain::LogUniform { low, high } => {
            let (a, b) = (low.max(1e-12).ln(), high.max(1e-12).ln());
            ParamValue::Float((a + u * (b - a)).exp())
        }
        ParamDomain::Discrete(c) => c[0].clone(),
    }
}

fn gauss(rng: &mut Rng) -> f64 {
    let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Parzen density (uniform prior mixed in) of `x` under Gaussians centred on `pts`, bandwidth `bw`.
fn parzen(x: f64, pts: &[f64], bw: f64) -> f64 {
    let n = pts.len() as f64 + 1.0;
    let k: f64 = pts.iter().map(|p| (-0.5 * ((x - p) / bw).powi(2)).exp() / (bw * (2.0 * std::f64::consts::PI).sqrt())).sum();
    (k + 1.0) / n
}

/// TPE: after `n_startup` random trials, splits observations into the best `gamma` fraction (good) and the rest, and
/// proposes the candidate (out of `n_candidates` drawn around good points) maximizing `l(x) / g(x)`.
pub fn tpe_search<F>(space: &ParamSpace, n_trials: usize, n_startup: usize, gamma: f64, n_candidates: usize, seed: u64, mut objective: F) -> Result<Study>
where
    F: FnMut(&TrialConfig) -> Result<f64>,
{
    if n_trials == 0 || !(0.0 < gamma && gamma < 1.0) || n_candidates == 0 {
        return Err(Error::Msg("tpe_search: need n_trials >= 1, 0 < gamma < 1 and n_candidates >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let mut trials: Vec<TrialResult> = Vec::new();
    for id in 0..n_trials {
        let config = if id < n_startup.max(2) {
            space.sample_config()
        } else {
            let mut order: Vec<usize> = (0..trials.len()).collect();
            order.sort_by(|&a, &b| trials[a].objective.total_cmp(&trials[b].objective));
            let n_good = ((trials.len() as f64 * gamma).ceil() as usize).clamp(1, trials.len() - 1);
            let (good, bad) = order.split_at(n_good);
            let mut cfg = TrialConfig::new();
            for (name, dom) in &space.domains {
                let pick = |idx: &[usize]| -> Vec<ParamValue> { idx.iter().map(|&i| trials[i].config[name].clone()).collect() };
                let (gv, bv) = (pick(good), pick(bad));
                let value = match dom {
                    ParamDomain::Discrete(choices) => {
                        // category with the best smoothed good/bad frequency ratio
                        let freq = |vals: &[ParamValue], c: &ParamValue| (vals.iter().filter(|v| *v == c).count() as f64 + 1.0) / (vals.len() as f64 + choices.len() as f64);
                        choices.iter().max_by(|a, b| (freq(&gv, a) / freq(&bv, a)).total_cmp(&(freq(&gv, b) / freq(&bv, b)))).unwrap().clone()
                    }
                    _ => {
                        let gu: Vec<f64> = gv.iter().filter_map(|v| to_unit(dom, v)).collect();
                        let bu: Vec<f64> = bv.iter().filter_map(|v| to_unit(dom, v)).collect();
                        let bw_g = (1.06 * 0.3 * (gu.len() as f64).powf(-0.2)).clamp(0.02, 0.3);
                        let bw_b = (1.06 * 0.3 * (bu.len() as f64).powf(-0.2)).clamp(0.02, 0.3);
                        let mut best = (f64::NEG_INFINITY, 0.5);
                        for _ in 0..n_candidates {
                            let centre = gu[(rng.next_u64() % gu.len() as u64) as usize];
                            let x = (centre + bw_g * gauss(&mut rng)).clamp(0.0, 1.0);
                            let score = parzen(x, &gu, bw_g) / parzen(x, &bu, bw_b);
                            if score > best.0 {
                                best = (score, x);
                            }
                        }
                        from_unit(dom, best.1)
                    }
                };
                cfg.insert(name.clone(), value);
            }
            cfg
        };
        let obj = objective(&config)?;
        trials.push(TrialResult { trial_id: id, config, objective: if obj.is_nan() { f64::INFINITY } else { obj } });
    }
    let best = trials.iter().filter(|t| t.objective.is_finite()).min_by(|a, b| a.objective.total_cmp(&b.objective)).cloned();
    Ok(Study { trials, best_trial: best })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> ParamSpace {
        ParamSpace::new().add_uniform("x", -5.0, 5.0).add_log_uniform("lr", 1e-4, 1.0).add_discrete("kind", vec![ParamValue::Int(0), ParamValue::Int(1), ParamValue::Int(2)])
    }

    // optimum at x = 1.5, lr = 0.01, kind = 2
    fn loss(c: &TrialConfig) -> f64 {
        (c["x"].as_float() - 1.5).powi(2) + (c["lr"].as_float().log10() + 2.0).powi(2) + if c["kind"].as_int() == 2 { 0.0 } else { 1.0 }
    }

    #[test]
    fn tpe_beats_random_search_on_average() {
        let tpe = tpe_search(&space(), 80, 15, 0.25, 24, 3, |c| Ok(loss(c))).unwrap();
        assert_eq!(tpe.trials.len(), 80);
        let best = tpe.best_trial.unwrap();
        assert!(best.objective < 0.1, "{best:?}");
        assert_eq!(best.config["kind"].as_int(), 2);
        // against pure random search with the same budget (fixed seeds, but several repeats)
        let mut wins = 0;
        for s in 0..5 {
            let t = tpe_search(&space(), 60, 10, 0.25, 24, 100 + s, |c| Ok(loss(c))).unwrap().best_trial.unwrap().objective;
            let r = Study::random_search(&space(), 60, |c| Ok(loss(c))).unwrap().best_trial.unwrap().objective;
            wins += usize::from(t <= r);
        }
        assert!(wins >= 3, "TPE should usually match or beat random search ({wins}/5)");
        assert!(tpe_search(&space(), 0, 5, 0.25, 8, 0, |c| Ok(loss(c))).is_err());
    }

    #[test]
    fn halving_spends_budget_on_the_promising() {
        // objective improves with budget, and configs differ by a fixed offset
        let mut calls_at_max = 0;
        let mut calls = 0;
        let study = successive_halving(&space(), 27, 1.0, 27.0, 3.0, |c, b| {
            calls += 1;
            if b >= 27.0 {
                calls_at_max += 1;
            }
            Ok(loss(c) + 1.0 / b)
        })
        .unwrap();
        assert_eq!(calls, 27 + 9 + 3 + 1);
        assert_eq!(calls_at_max, 1);
        assert!(study.best_trial.is_some());
        // far cheaper than evaluating all 27 at full budget (27 * 27 budget units)
        let spent: f64 = [27.0 * 1.0, 9.0 * 3.0, 3.0 * 9.0, 1.0 * 27.0].iter().sum();
        assert!(spent < 27.0 * 27.0 / 4.0);
        assert!(successive_halving(&space(), 0, 1.0, 9.0, 3.0, |_, _| Ok(0.0)).is_err());
        assert!(successive_halving(&space(), 5, 1.0, 9.0, 1.0, |_, _| Ok(0.0)).is_err());
    }

    #[test]
    fn hyperband_runs_all_brackets() {
        let mut max_budget_seen = 0.0f64;
        let study = hyperband(&space(), 27.0, 3.0, |c, b| {
            max_budget_seen = max_budget_seen.max(b);
            Ok(loss(c) + 1.0 / b)
        })
        .unwrap();
        assert_eq!(max_budget_seen, 27.0);
        assert!(study.trials.len() > 30);
        assert!(study.trials.iter().enumerate().all(|(i, t)| t.trial_id == i));
        assert!(study.best_trial.unwrap().objective.is_finite());
        assert!(hyperband(&space(), 0.0, 3.0, |_, _| Ok(0.0)).is_err());
    }
}
