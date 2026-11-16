//! Cross-validation batch 15: expected improvement, particle filter vs Kalman, TPE / Hyperband / Bayesian optimization
//! (compared by tools/crossval/check_batch15.py).
use oxideon_core::gp::*;
use oxideon_core::inference::*;
use oxideon_core::tuning::*;
use oxideon_core::tuning_ext::*;

fn u(i: u64) -> f64 {
    (((i.wrapping_mul(2_654_435_761)) % 4_294_967_296) as f64 + 0.5) / 4_294_967_296.0
}
fn gauss(i: u64) -> f64 {
    (-2.0 * u(2 * i + 1).ln()).sqrt() * (2.0 * std::f64::consts::PI * u(2 * i + 2)).cos()
}

#[test]
#[ignore]
fn emit() {
    // --- expected improvement over a grid of (mean, var, best)
    for (i, (m, v, b)) in [(0.0, 1.0, 0.5), (1.0, 0.25, 0.0), (-1.0, 4.0, -0.5), (5.0, 0.01, 0.0), (10.0, 1.0, 0.0), (2.0, 0.5, 8.0), (0.3, 1e-4, 0.35), (-3.0, 1.0, 0.0), (3.0, 1.0, 0.0), (4.5, 1.0, 0.0)].iter().enumerate() {
        println!("ei_{i} {:.15e}", expected_improvement(*m, *v, *b));
    }

    // --- particle filter vs exact Kalman posterior mean for a local-level model
    // x_t = 0.9 x_{t-1} + N(0, 0.3^2), y_t = x_t + N(0, 0.5^2)
    let (a, qs, rs) = (0.9, 0.3, 0.5);
    let mut x = 0.0;
    let ys: Vec<f64> = (0..60u64)
        .map(|t| {
            x = a * x + qs * gauss(t);
            x + rs * gauss(1000 + t)
        })
        .collect();
    let n = 20_000;
    let init: Vec<f64> = (0..n as u64).map(|i| gauss(5000 + i)).collect(); // prior N(0,1)
    let pf = particle_filter(&ys, n, &init, |x, w| a * x + qs * w, |x, y| (-0.5 * ((y - x) / rs).powi(2)).exp(), 42).unwrap();
    for (t, v) in pf.iter().enumerate() {
        println!("pf_{t} {v:.9}");
    }
    let kf = KalmanFilter { f: vec![vec![a]], h: vec![vec![1.0]], q: vec![vec![qs * qs]], r: vec![vec![rs * rs]] };
    let out = kf.filter(&ys.iter().map(|&y| vec![y]).collect::<Vec<_>>(), &[0.0], &vec![vec![1.0]]).unwrap();
    for (t, m) in out.means.iter().enumerate() {
        println!("kf_{t} {:.9}", m[0]);
    }
    for (t, y) in ys.iter().enumerate() {
        println!("y_{t} {y:.12}");
    }

    // --- TPE vs random search on a 3-parameter problem (30 seeds each; mean best value)
    let space = ParamSpace::new().add_uniform("x", -5.0, 5.0).add_log_uniform("lr", 1e-4, 1.0).add_discrete("kind", vec![ParamValue::Int(0), ParamValue::Int(1), ParamValue::Int(2)]);
    let loss = |c: &TrialConfig| (c["x"].as_float() - 1.5).powi(2) + (c["lr"].as_float().log10() + 2.0).powi(2) + if c["kind"].as_int() == 2 { 0.0 } else { 1.0 };
    let (mut tpe_sum, mut rnd_sum) = (0.0, 0.0);
    for seed in 1..=30u64 {
        let st = tpe_search(&space, 60, 10, 0.25, 24, seed, |c| Ok(loss(c))).unwrap();
        tpe_sum += st.best_trial.unwrap().objective;
        // plain random search with the same budget
        let mut best = f64::INFINITY;
        let mut sp_rng = space.clone();
        for _ in 0..60 {
            best = best.min(loss(&sp_rng.sample_config()));
        }
        rnd_sum += best;
        let _ = &mut sp_rng;
    }
    println!("tpe_mean_best {:.6}", tpe_sum / 30.0);
    println!("random_mean_best {:.6}", rnd_sum / 30.0);

    // --- Hyperband: budget accounting follows the paper; noise shrinks with budget
    let hs = space.clone();
    let mut calls: Vec<f64> = Vec::new();
    let st = hyperband(&hs, 81.0, 3.0, |c, budget| {
        calls.push(budget);
        Ok(loss(c) + 5.0 / budget)
    })
    .unwrap();
    let total: f64 = calls.iter().sum();
    println!("hb_ntrials {}", st.trials.len());
    println!("hb_total_budget {total:.6}");
    println!("hb_max_budget_evals {}", calls.iter().filter(|&&b| (b - 81.0).abs() < 1e-9).count());
    println!("hb_best {:.6}", st.best_trial.unwrap().objective);

    // --- Bayesian optimization on a smooth 2-D function (mean best of 10 seeds) vs random search
    let f = |x: &[f64]| (x[0] - 0.7).powi(2) * 3.0 + (x[1] + 1.2).powi(2) + 0.5 * (3.0 * x[0]).sin();
    let bounds = [(-3.0, 3.0), (-3.0, 3.0)];
    let (mut bo_sum, mut rs_sum) = (0.0, 0.0);
    for seed in 1..=10u64 {
        bo_sum += bayes_opt(f, &bounds, 5, 25, 400, seed).unwrap().value;
        let mut best = f64::INFINITY;
        for k in 0..30u64 {
            best = best.min(f(&[-3.0 + 6.0 * u(seed * 1000 + 2 * k), -3.0 + 6.0 * u(seed * 1000 + 2 * k + 1)]));
        }
        rs_sum += best;
    }
    println!("bo_mean_best {:.6}", bo_sum / 10.0);
    println!("rs30_mean_best {:.6}", rs_sum / 10.0);
}
