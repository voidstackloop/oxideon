//! Model-agnostic interpretability on any `Fn(&Matrix) -> Vec<f64>` predictor: [`permutation_importance`],
//! [`partial_dependence`] / [`ice_curves`], Monte-Carlo [`shapley_values`], and a [`lime_explain`] local linear surrogate.

use crate::classical::Matrix;
use crate::classical_linear::LinearRegression;
use crate::error::{Error, Result};
use crate::rng::Rng;

fn check(x: &Matrix, op: &str) -> Result<usize> {
    let d = x.first().map_or(0, Vec::len);
    if x.is_empty() || d == 0 || x.iter().any(|r| r.len() != d) {
        return Err(Error::Msg(format!("{op}: need a non-empty rectangular matrix")));
    }
    Ok(d)
}

/// Drop in `metric(y, predict(x))` (higher = better) when one feature is shuffled, averaged over `repeats`.
/// Returns `(mean_drop, std_drop)` per feature; a larger drop means a more important feature.
pub fn permutation_importance<P, M>(predict: P, metric: M, x: &Matrix, y: &[f64], repeats: usize, seed: u64) -> Result<Vec<(f64, f64)>>
where
    P: Fn(&Matrix) -> Vec<f64>,
    M: Fn(&[f64], &[f64]) -> f64,
{
    let d = check(x, "permutation_importance")?;
    if y.len() != x.len() || repeats == 0 {
        return Err(Error::Msg("permutation_importance: need one target per row and repeats >= 1".into()));
    }
    let base = metric(y, &predict(x));
    let mut rng = Rng::new(seed);
    Ok((0..d)
        .map(|j| {
            let drops: Vec<f64> = (0..repeats)
                .map(|_| {
                    let mut col: Vec<f64> = x.iter().map(|r| r[j]).collect();
                    rng.shuffle(&mut col);
                    let mut xp = x.clone();
                    xp.iter_mut().zip(col).for_each(|(r, v)| r[j] = v);
                    base - metric(y, &predict(&xp))
                })
                .collect();
            let m = drops.iter().sum::<f64>() / repeats as f64;
            (m, (drops.iter().map(|v| (v - m).powi(2)).sum::<f64>() / repeats as f64).sqrt())
        })
        .collect())
}

/// Individual conditional expectation: one prediction curve per row as `feature` sweeps `grid`.
pub fn ice_curves<P>(predict: P, x: &Matrix, feature: usize, grid: &[f64]) -> Result<Matrix>
where
    P: Fn(&Matrix) -> Vec<f64>,
{
    let d = check(x, "ice_curves")?;
    if feature >= d || grid.is_empty() {
        return Err(Error::Msg("ice_curves: feature out of range or empty grid".into()));
    }
    let mut curves = vec![Vec::with_capacity(grid.len()); x.len()];
    for &g in grid {
        let mut xp = x.clone();
        xp.iter_mut().for_each(|r| r[feature] = g);
        for (c, p) in curves.iter_mut().zip(predict(&xp)) {
            c.push(p);
        }
    }
    Ok(curves)
}

/// Partial dependence: the ICE curves averaged over rows.
pub fn partial_dependence<P>(predict: P, x: &Matrix, feature: usize, grid: &[f64]) -> Result<Vec<f64>>
where
    P: Fn(&Matrix) -> Vec<f64>,
{
    let c = ice_curves(predict, x, feature, grid)?;
    Ok((0..grid.len()).map(|g| c.iter().map(|r| r[g]).sum::<f64>() / c.len() as f64).collect())
}

/// Monte-Carlo Shapley values of `instance` against a `background` sample (permutation sampling, `samples` random
/// feature orders). The values sum to `predict(instance) - mean(predict(background))` up to sampling error; each
/// permutation is exactly efficient, so the sum is exact when averaged over complete permutations.
pub fn shapley_values<P>(predict: P, instance: &[f64], background: &Matrix, samples: usize, seed: u64) -> Result<Vec<f64>>
where
    P: Fn(&Matrix) -> Vec<f64>,
{
    let d = check(background, "shapley_values")?;
    if instance.len() != d || samples == 0 {
        return Err(Error::Msg("shapley_values: instance width must match the background and samples >= 1".into()));
    }
    let mut rng = Rng::new(seed);
    let mut phi = vec![0.0; d];
    for _ in 0..samples {
        let mut order: Vec<usize> = (0..d).collect();
        rng.shuffle(&mut order);
        let reference = &background[(rng.next_u64() % background.len() as u64) as usize];
        // walk from the reference row to the instance one feature at a time
        let mut cur = reference.clone();
        let mut prev = predict(&vec![cur.clone()])[0];
        for &j in &order {
            cur[j] = instance[j];
            let next = predict(&vec![cur.clone()])[0];
            phi[j] += (next - prev) / samples as f64;
            prev = next;
        }
    }
    Ok(phi)
}

/// LIME-style explanation: fit a proximity-weighted linear model to the predictor around `instance` by perturbing
/// each feature with Gaussian noise of scale `scale[j]`. Returns the local coefficients.
pub fn lime_explain<P>(predict: P, instance: &[f64], scale: &[f64], n_samples: usize, seed: u64) -> Result<Vec<f64>>
where
    P: Fn(&Matrix) -> Vec<f64>,
{
    if instance.len() != scale.len() || instance.is_empty() || n_samples < instance.len() + 2 {
        return Err(Error::Msg("lime_explain: scale must match the instance and n_samples must exceed the feature count + 1".into()));
    }
    let mut rng = Rng::new(seed);
    let gauss = |rng: &mut Rng| {
        let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let samples: Matrix = (0..n_samples).map(|_| instance.iter().zip(scale).map(|(v, s)| v + s * gauss(&mut rng)).collect()).collect();
    let y = predict(&samples);
    // weight by an RBF kernel on the standardized distance by duplicating nothing: scale rows and targets by sqrt(w)
    let w: Vec<f64> = samples.iter().map(|r| (-r.iter().zip(instance).zip(scale).map(|((a, b), s)| ((a - b) / s).powi(2)).sum::<f64>() / (2.0 * instance.len() as f64)).exp().sqrt()).collect();
    let xw: Matrix = samples.iter().zip(&w).map(|(r, wi)| r.iter().map(|v| v * wi).collect::<Vec<_>>().into_iter().chain(std::iter::once(*wi)).collect()).collect();
    let yw: Vec<f64> = y.iter().zip(&w).map(|(v, wi)| v * wi).collect();
    // the trailing column is the (weighted) intercept; ridge keeps the solve well posed
    let fit = LinearRegression::fit(&xw, &yw, 1e-8)?;
    Ok(fit.coef[..instance.len()].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Matrix {
        let mut rng = Rng::new(3);
        (0..200).map(|_| vec![rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0]).collect()
    }

    // f(x) = 3 x0 + x1^2 (x2 is irrelevant)
    fn f(x: &Matrix) -> Vec<f64> {
        x.iter().map(|r| 3.0 * r[0] + r[1] * r[1]).collect()
    }

    fn neg_mse(y: &[f64], p: &[f64]) -> f64 {
        -y.iter().zip(p).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / y.len() as f64
    }

    #[test]
    fn permutation_importance_ranks_features() {
        let x = data();
        let y = f(&x);
        let imp = permutation_importance(f, neg_mse, &x, &y, 5, 1).unwrap();
        assert!(imp[0].0 > imp[1].0 && imp[1].0 > 0.05, "{imp:?}");
        assert!(imp[2].0.abs() < 1e-12, "an unused feature has zero importance");
        assert!(permutation_importance(f, neg_mse, &x, &y[..3], 5, 1).is_err());
    }

    #[test]
    fn partial_dependence_traces_the_function() {
        let x = data();
        let grid = [-1.0, 0.0, 1.0];
        let pd = partial_dependence(f, &x, 1, &grid).unwrap();
        let base = f(&x).iter().sum::<f64>() / 200.0 - x.iter().map(|r| r[1] * r[1]).sum::<f64>() / 200.0;
        // PD of x1 is base + grid^2 (x1 enters only through its own square)
        for (g, p) in grid.iter().zip(&pd) {
            assert!((p - (base + g * g)).abs() < 1e-9);
        }
        let ice = ice_curves(f, &x, 0, &grid).unwrap();
        assert_eq!((ice.len(), ice[0].len()), (200, 3));
        assert!((ice[7][2] - ice[7][0] - 6.0).abs() < 1e-9); // slope 3 over a width of 2
        assert!(ice_curves(f, &x, 9, &grid).is_err());
    }

    #[test]
    fn shapley_values_are_efficient_and_attribute_correctly() {
        let bg = data();
        let inst = [0.5, -0.8, 0.9];
        let phi = shapley_values(f, &inst, &bg, 2000, 2).unwrap();
        let mean_pred = f(&bg).iter().sum::<f64>() / 200.0;
        assert!((phi.iter().sum::<f64>() - (f(&vec![inst.to_vec()])[0] - mean_pred)).abs() < 0.15, "{phi:?}");
        assert!(phi[2].abs() < 1e-12, "the unused feature gets zero credit");
        assert!(phi[0] > 0.8 && phi[0] < 2.2, "x0 contributes about 3 * 0.5: {phi:?}");
        assert!(shapley_values(f, &[1.0], &bg, 10, 0).is_err());
    }

    #[test]
    fn lime_recovers_the_local_gradient() {
        // gradient of f at (0.5, 1.0, 0) is (3, 2, 0)
        let c = lime_explain(f, &[0.5, 1.0, 0.0], &[0.1, 0.1, 0.1], 400, 3).unwrap();
        assert!((c[0] - 3.0).abs() < 0.3 && (c[1] - 2.0).abs() < 0.4 && c[2].abs() < 0.3, "{c:?}");
        assert!(lime_explain(f, &[1.0], &[1.0, 2.0], 50, 0).is_err());
    }
}
