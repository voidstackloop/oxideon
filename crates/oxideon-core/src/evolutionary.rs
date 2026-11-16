//! Derivative-free global optimization (all minimize a black-box `f`): [`differential_evolution`], [`cma_es`]
//! (a compact separable-covariance CMA-ES), [`simulated_annealing`], [`particle_swarm`], and a real-coded
//! [`genetic_algorithm`]. Each returns an [`OptResult`] with the best point and a per-iteration best-value history.

use crate::error::{Error, Result};
use crate::rng::Rng;

/// Best solution found and the best objective value after each generation / iteration.
#[derive(Debug, Clone)]
pub struct OptResult {
    pub x: Vec<f64>,
    pub value: f64,
    pub history: Vec<f64>,
    pub evaluations: usize,
}

fn gauss(rng: &mut Rng) -> f64 {
    let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn check_bounds(bounds: &[(f64, f64)], op: &str) -> Result<()> {
    if bounds.is_empty() || bounds.iter().any(|&(lo, hi)| !(lo.is_finite() && hi.is_finite() && lo < hi)) {
        return Err(Error::Msg(format!("{op}: bounds must be non-empty finite (lo, hi) pairs with lo < hi")));
    }
    Ok(())
}

fn random_point(bounds: &[(f64, f64)], rng: &mut Rng) -> Vec<f64> {
    bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.next_f64()).collect()
}

fn clip(x: &mut [f64], bounds: &[(f64, f64)]) {
    x.iter_mut().zip(bounds).for_each(|(v, &(lo, hi))| *v = v.clamp(lo, hi));
}

/// Differential evolution (`rand/1/bin`) with mutation factor `f` and crossover rate `cr`.
pub fn differential_evolution<F: Fn(&[f64]) -> f64>(f: F, bounds: &[(f64, f64)], pop_size: usize, generations: usize, fmut: f64, cr: f64, seed: u64) -> Result<OptResult> {
    check_bounds(bounds, "differential_evolution")?;
    if pop_size < 4 || !(0.0..=2.0).contains(&fmut) || !(0.0..=1.0).contains(&cr) {
        return Err(Error::Msg("differential_evolution: need pop_size >= 4, F in [0, 2] and CR in [0, 1]".into()));
    }
    let (mut rng, d) = (Rng::new(seed), bounds.len());
    let mut pop: Vec<Vec<f64>> = (0..pop_size).map(|_| random_point(bounds, &mut rng)).collect();
    let mut fit: Vec<f64> = pop.iter().map(|p| f(p)).collect();
    let (mut evals, mut history) = (pop_size, Vec::new());
    for _ in 0..generations {
        for i in 0..pop_size {
            let pick = |rng: &mut Rng, excl: &[usize]| loop {
                let k = (rng.next_u64() % pop_size as u64) as usize;
                if !excl.contains(&k) {
                    return k;
                }
            };
            let a = pick(&mut rng, &[i]);
            let b = pick(&mut rng, &[i, a]);
            let c = pick(&mut rng, &[i, a, b]);
            let forced = (rng.next_u64() % d as u64) as usize;
            let mut trial = pop[i].clone();
            for j in 0..d {
                if j == forced || rng.next_f64() < cr {
                    trial[j] = pop[a][j] + fmut * (pop[b][j] - pop[c][j]);
                }
            }
            clip(&mut trial, bounds);
            let ft = f(&trial);
            evals += 1;
            if ft <= fit[i] {
                pop[i] = trial;
                fit[i] = ft;
            }
        }
        history.push(fit.iter().cloned().fold(f64::INFINITY, f64::min));
    }
    let best = (0..pop_size).min_by(|&a, &b| fit[a].total_cmp(&fit[b])).unwrap();
    Ok(OptResult { x: pop[best].clone(), value: fit[best], history, evaluations: evals })
}

/// CMA-ES with a diagonal covariance (sep-CMA-ES): step-size control by cumulative path length plus per-coordinate
/// variance adaptation. `sigma0` is the initial step size in the data's units.
pub fn cma_es<F: Fn(&[f64]) -> f64>(f: F, x0: &[f64], sigma0: f64, generations: usize, seed: u64) -> Result<OptResult> {
    let d = x0.len();
    if d == 0 || sigma0 <= 0.0 {
        return Err(Error::Msg("cma_es: need a start point and sigma0 > 0".into()));
    }
    let mut rng = Rng::new(seed);
    let lambda = 4 + (3.0 * (d as f64).ln()).floor() as usize;
    let mu = lambda / 2;
    let raw_w: Vec<f64> = (0..mu).map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln()).collect();
    let ws: f64 = raw_w.iter().sum();
    let w: Vec<f64> = raw_w.iter().map(|v| v / ws).collect();
    let mueff = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
    let df = d as f64;
    let (cs, ds) = ((mueff + 2.0) / (df + mueff + 5.0), 1.0 + 2.0 * (((mueff - 1.0) / (df + 1.0)).sqrt() - 1.0).max(0.0) + (mueff + 2.0) / (df + mueff + 5.0));
    let c_cov = (mueff / df + 2.0 / df.sqrt().max(1.0)).min(1.0) * 0.5 / (df + 2.0).max(1.0) * 4.0_f64.min(df + 2.0);
    let chi_n = df.sqrt() * (1.0 - 1.0 / (4.0 * df) + 1.0 / (21.0 * df * df));
    let (mut mean, mut sigma) = (x0.to_vec(), sigma0);
    let (mut var, mut ps): (Vec<f64>, Vec<f64>) = (vec![1.0; d], vec![0.0; d]);
    let (mut best_x, mut best_f, mut evals, mut history) = (x0.to_vec(), f(x0), 1usize, Vec::new());
    for _ in 0..generations {
        let mut pop: Vec<(f64, Vec<f64>, Vec<f64>)> = (0..lambda)
            .map(|_| {
                let z: Vec<f64> = (0..d).map(|_| gauss(&mut rng)).collect();
                let x: Vec<f64> = (0..d).map(|i| mean[i] + sigma * var[i].sqrt() * z[i]).collect();
                (f(&x), x, z)
            })
            .collect();
        evals += lambda;
        pop.sort_by(|a, b| a.0.total_cmp(&b.0));
        if pop[0].0 < best_f {
            best_f = pop[0].0;
            best_x = pop[0].1.clone();
        }
        history.push(best_f);
        let zmean: Vec<f64> = (0..d).map(|i| (0..mu).map(|k| w[k] * pop[k].2[i]).sum()).collect();
        mean = (0..d).map(|i| mean[i] + sigma * var[i].sqrt() * zmean[i]).collect();
        ps = (0..d).map(|i| (1.0 - cs) * ps[i] + (cs * (2.0 - cs) * mueff).sqrt() * zmean[i]).collect();
        let pn = ps.iter().map(|v| v * v).sum::<f64>().sqrt();
        sigma *= ((cs / ds) * (pn / chi_n - 1.0)).exp().clamp(0.5, 2.0);
        for i in 0..d {
            let rank_mu: f64 = (0..mu).map(|k| w[k] * pop[k].2[i] * pop[k].2[i]).sum();
            var[i] = ((1.0 - c_cov) * var[i] + c_cov * var[i] * rank_mu).max(1e-20);
        }
        if sigma * var.iter().cloned().fold(0.0, f64::max).sqrt() < 1e-14 {
            break;
        }
    }
    Ok(OptResult { x: best_x, value: best_f, history, evaluations: evals })
}

/// Simulated annealing with Gaussian neighbours and a geometric cooling schedule from `t0` to `t_end`.
pub fn simulated_annealing<F: Fn(&[f64]) -> f64>(f: F, x0: &[f64], step: f64, t0: f64, t_end: f64, iters: usize, seed: u64) -> Result<OptResult> {
    if x0.is_empty() || step <= 0.0 || t0 <= 0.0 || t_end <= 0.0 || t_end > t0 || iters == 0 {
        return Err(Error::Msg("simulated_annealing: need a start, step > 0, 0 < t_end <= t0 and iters >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let cool = (t_end / t0).powf(1.0 / iters as f64);
    let (mut x, mut fx) = (x0.to_vec(), f(x0));
    let (mut best_x, mut best_f, mut t, mut history) = (x.clone(), fx, t0, Vec::new());
    for _ in 0..iters {
        let cand: Vec<f64> = x.iter().map(|v| v + step * (t / t0).sqrt() * gauss(&mut rng)).collect();
        let fc = f(&cand);
        if fc < fx || rng.next_f64() < ((fx - fc) / t).exp() {
            x = cand;
            fx = fc;
            if fx < best_f {
                best_f = fx;
                best_x = x.clone();
            }
        }
        history.push(best_f);
        t *= cool;
    }
    Ok(OptResult { x: best_x, value: best_f, history, evaluations: iters + 1 })
}

/// Particle swarm optimization with inertia `w` and acceleration constants `c1` (cognitive) / `c2` (social).
pub fn particle_swarm<F: Fn(&[f64]) -> f64>(f: F, bounds: &[(f64, f64)], particles: usize, iters: usize, seed: u64) -> Result<OptResult> {
    check_bounds(bounds, "particle_swarm")?;
    if particles < 2 {
        return Err(Error::Msg("particle_swarm: need at least 2 particles".into()));
    }
    let (mut rng, d) = (Rng::new(seed), bounds.len());
    let (w, c1, c2) = (0.72, 1.49, 1.49);
    let mut pos: Vec<Vec<f64>> = (0..particles).map(|_| random_point(bounds, &mut rng)).collect();
    let mut vel: Vec<Vec<f64>> = (0..particles).map(|_| bounds.iter().map(|&(lo, hi)| (hi - lo) * 0.1 * (rng.next_f64() - 0.5)).collect()).collect();
    let mut pbest = pos.clone();
    let mut pf: Vec<f64> = pos.iter().map(|p| f(p)).collect();
    let g = (0..particles).min_by(|&a, &b| pf[a].total_cmp(&pf[b])).unwrap();
    let (mut gbest, mut gf, mut evals, mut history) = (pbest[g].clone(), pf[g], particles, Vec::new());
    for _ in 0..iters {
        for i in 0..particles {
            for j in 0..d {
                vel[i][j] = w * vel[i][j] + c1 * rng.next_f64() * (pbest[i][j] - pos[i][j]) + c2 * rng.next_f64() * (gbest[j] - pos[i][j]);
                pos[i][j] += vel[i][j];
            }
            clip(&mut pos[i], bounds);
            let fv = f(&pos[i]);
            evals += 1;
            if fv < pf[i] {
                pf[i] = fv;
                pbest[i] = pos[i].clone();
                if fv < gf {
                    gf = fv;
                    gbest = pos[i].clone();
                }
            }
        }
        history.push(gf);
    }
    Ok(OptResult { x: gbest, value: gf, history, evaluations: evals })
}

/// Real-coded genetic algorithm: tournament selection, blend crossover (BLX-0.5), Gaussian mutation, elitism.
pub fn genetic_algorithm<F: Fn(&[f64]) -> f64>(f: F, bounds: &[(f64, f64)], pop_size: usize, generations: usize, mutation: f64, seed: u64) -> Result<OptResult> {
    check_bounds(bounds, "genetic_algorithm")?;
    if pop_size < 4 || !(0.0..=1.0).contains(&mutation) {
        return Err(Error::Msg("genetic_algorithm: need pop_size >= 4 and mutation in [0, 1]".into()));
    }
    let mut rng = Rng::new(seed);
    let mut pop: Vec<Vec<f64>> = (0..pop_size).map(|_| random_point(bounds, &mut rng)).collect();
    let mut fit: Vec<f64> = pop.iter().map(|p| f(p)).collect();
    let (mut evals, mut history) = (pop_size, Vec::new());
    for _ in 0..generations {
        let elite = (0..pop_size).min_by(|&a, &b| fit[a].total_cmp(&fit[b])).unwrap();
        let mut next = vec![pop[elite].clone()];
        let tournament = |rng: &mut Rng, fit: &[f64]| -> usize {
            let (a, b) = ((rng.next_u64() % pop_size as u64) as usize, (rng.next_u64() % pop_size as u64) as usize);
            if fit[a] <= fit[b] { a } else { b }
        };
        while next.len() < pop_size {
            let (p, q) = (tournament(&mut rng, &fit), tournament(&mut rng, &fit));
            let mut child: Vec<f64> = (0..bounds.len())
                .map(|j| {
                    let (lo, hi) = (pop[p][j].min(pop[q][j]), pop[p][j].max(pop[q][j]));
                    let span = hi - lo;
                    lo - 0.5 * span + (1.0 + 2.0 * 0.5) * span * rng.next_f64()
                })
                .collect();
            for (j, &(lo, hi)) in bounds.iter().enumerate() {
                if rng.next_f64() < mutation {
                    child[j] += 0.1 * (hi - lo) * gauss(&mut rng);
                }
            }
            clip(&mut child, bounds);
            next.push(child);
        }
        fit = next.iter().enumerate().map(|(i, p)| if i == 0 { fit[elite] } else { f(p) }).collect();
        evals += pop_size - 1;
        pop = next;
        history.push(fit.iter().cloned().fold(f64::INFINITY, f64::min));
    }
    let best = (0..pop_size).min_by(|&a, &b| fit[a].total_cmp(&fit[b])).unwrap();
    Ok(OptResult { x: pop[best].clone(), value: fit[best], history, evaluations: evals })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rastrigin(x: &[f64]) -> f64 {
        10.0 * x.len() as f64 + x.iter().map(|v| v * v - 10.0 * (2.0 * std::f64::consts::PI * v).cos()).sum::<f64>()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        (0..x.len() - 1).map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2)).sum()
    }

    fn sphere_shifted(x: &[f64]) -> f64 {
        x.iter().enumerate().map(|(i, v)| (v - (i as f64 + 1.0)).powi(2)).sum()
    }

    #[test]
    fn differential_evolution_solves_multimodal_problems() {
        let r = differential_evolution(rastrigin, &[(-5.12, 5.12); 3], 40, 200, 0.7, 0.9, 1).unwrap();
        assert!(r.value < 1e-3, "{r:?}");
        assert!(r.history.windows(2).all(|w| w[1] <= w[0] + 1e-12), "elitist: the best value never gets worse");
        assert_eq!(r.evaluations, 40 + 40 * 200);
        assert!(differential_evolution(rastrigin, &[(1.0, 1.0)], 40, 10, 0.7, 0.9, 0).is_err() && differential_evolution(rastrigin, &[(0.0, 1.0)], 2, 10, 0.7, 0.9, 0).is_err());
    }

    #[test]
    fn cma_es_handles_ill_conditioned_valleys() {
        let r = cma_es(rosenbrock, &[-1.0, 1.5, -0.5], 0.5, 1500, 3).unwrap();
        assert!(r.value < 0.1, "{}", r.value); // diagonal covariance only: a rotated valley converges slowly, so a loose bound
        let s = cma_es(sphere_shifted, &[0.0; 5], 2.0, 400, 5).unwrap();
        assert!(s.value < 1e-8 && s.x.iter().enumerate().all(|(i, v)| (v - (i as f64 + 1.0)).abs() < 1e-3), "{s:?}");
        assert!(cma_es(rosenbrock, &[], 1.0, 10, 0).is_err() && cma_es(rosenbrock, &[0.0], 0.0, 10, 0).is_err());
    }

    #[test]
    fn annealing_swarm_and_genetic_find_good_minima() {
        let sa = simulated_annealing(sphere_shifted, &[10.0, -10.0], 3.0, 50.0, 1e-4, 20_000, 2).unwrap();
        assert!(sa.value < 0.05, "{}", sa.value);
        let pso = particle_swarm(rastrigin, &[(-5.12, 5.12); 2], 30, 200, 4).unwrap();
        assert!(pso.value < 0.1, "{}", pso.value);
        assert!(pso.history.windows(2).all(|w| w[1] <= w[0]));
        let ga = genetic_algorithm(sphere_shifted, &[(-10.0, 10.0); 3], 60, 300, 0.2, 6).unwrap();
        assert!(ga.value < 0.01, "{}", ga.value);
        assert!(ga.x.iter().all(|v| (-10.0..=10.0).contains(v)));
        assert!(simulated_annealing(sphere_shifted, &[0.0], 1.0, 1.0, 2.0, 10, 0).is_err() && particle_swarm(rastrigin, &[(0.0, 1.0)], 1, 5, 0).is_err() && genetic_algorithm(rastrigin, &[(0.0, 1.0)], 3, 5, 0.1, 0).is_err());
    }
}
