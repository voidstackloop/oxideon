//! Index-based model selection: [`kfold`], [`stratified_kfold`], [`group_kfold`], [`time_series_split`],
//! [`cross_val_score`] for any `fit + predict` closure, and exhaustive [`grid_search`] with cross-validation.

use crate::classical::Matrix;
use crate::error::{Error, Result};
use crate::rng::Rng;

/// One cross-validation fold: indices to train on and to validate on.
#[derive(Debug, Clone, PartialEq)]
pub struct Fold {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

fn from_assignment(assign: &[usize], k: usize) -> Vec<Fold> {
    (0..k)
        .map(|f| Fold { train: (0..assign.len()).filter(|&i| assign[i] != f).collect(), test: (0..assign.len()).filter(|&i| assign[i] == f).collect() })
        .collect()
}

/// `k` folds of `n` samples (optionally shuffled with `seed`); fold sizes differ by at most one.
pub fn kfold(n: usize, k: usize, shuffle: bool, seed: u64) -> Result<Vec<Fold>> {
    if k < 2 || k > n {
        return Err(Error::Msg(format!("kfold: need 2 <= k <= n (k = {k}, n = {n})")));
    }
    let mut order: Vec<usize> = (0..n).collect();
    if shuffle {
        Rng::new(seed).shuffle(&mut order);
    }
    let mut assign = vec![0; n];
    for (pos, &i) in order.iter().enumerate() {
        assign[i] = pos % k;
    }
    Ok(from_assignment(&assign, k))
}

/// `k` folds that keep each class's proportion (every class needs at least `k` members).
pub fn stratified_kfold(labels: &[usize], k: usize, shuffle: bool, seed: u64) -> Result<Vec<Fold>> {
    let classes = labels.iter().max().map_or(0, |m| m + 1);
    let mut rng = Rng::new(seed);
    let mut assign = vec![0; labels.len()];
    let mut offset = 0;
    for c in 0..classes {
        let mut members: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == c).collect();
        if members.is_empty() {
            continue;
        }
        if members.len() < k {
            return Err(Error::Msg(format!("stratified_kfold: class {c} has {} members, fewer than k = {k}", members.len())));
        }
        if shuffle {
            rng.shuffle(&mut members);
        }
        // rotate the starting fold per class so remainders spread evenly
        for (pos, &i) in members.iter().enumerate() {
            assign[i] = (pos + offset) % k;
        }
        offset += members.len() % k;
    }
    if k < 2 {
        return Err(Error::Msg("stratified_kfold: k must be at least 2".into()));
    }
    Ok(from_assignment(&assign, k))
}

/// Folds that never split a group across train and test (greedy balancing by group size).
pub fn group_kfold(groups: &[usize], k: usize) -> Result<Vec<Fold>> {
    let ng = groups.iter().max().map_or(0, |m| m + 1);
    let mut sizes = vec![0usize; ng];
    groups.iter().for_each(|&g| sizes[g] += 1);
    let present: Vec<usize> = (0..ng).filter(|&g| sizes[g] > 0).collect();
    if k < 2 || k > present.len() {
        return Err(Error::Msg(format!("group_kfold: need 2 <= k <= number of groups ({})", present.len())));
    }
    let mut order = present;
    order.sort_by_key(|&g| std::cmp::Reverse(sizes[g]));
    let (mut load, mut fold_of) = (vec![0usize; k], vec![0usize; ng]);
    for g in order {
        let f = (0..k).min_by_key(|&f| load[f]).unwrap();
        fold_of[g] = f;
        load[f] += sizes[g];
    }
    let assign: Vec<usize> = groups.iter().map(|&g| fold_of[g]).collect();
    Ok(from_assignment(&assign, k))
}

/// Expanding-window splits for ordered data: each test block follows its training prefix.
pub fn time_series_split(n: usize, n_splits: usize) -> Result<Vec<Fold>> {
    if n_splits == 0 || n < n_splits + 1 {
        return Err(Error::Msg(format!("time_series_split: need n > n_splits (n = {n}, n_splits = {n_splits})")));
    }
    let block = n / (n_splits + 1);
    Ok((0..n_splits).map(|s| Fold { train: (0..block * (s + 1)).collect(), test: (block * (s + 1)..block * (s + 2)).collect() }).collect())
}

fn rows(x: &Matrix, idx: &[usize]) -> Matrix {
    idx.iter().map(|&i| x[i].clone()).collect()
}

/// Accuracy of `fit_predict(train_x, train_y, test_x)` on every fold; returns the per-fold scores.
pub fn cross_val_score<F>(x: &Matrix, y: &[usize], folds: &[Fold], fit_predict: F) -> Result<Vec<f64>>
where
    F: Fn(&Matrix, &[usize], &Matrix) -> Result<Vec<usize>>,
{
    if x.len() != y.len() {
        return Err(Error::Msg("cross_val_score: x and y must have the same number of rows".into()));
    }
    folds
        .iter()
        .map(|f| {
            let ty: Vec<usize> = f.train.iter().map(|&i| y[i]).collect();
            let pred = fit_predict(&rows(x, &f.train), &ty, &rows(x, &f.test))?;
            Ok(f.test.iter().zip(&pred).filter(|(i, p)| y[**i] == **p).count() as f64 / f.test.len().max(1) as f64)
        })
        .collect()
}

/// Result of [`grid_search`].
#[derive(Debug, Clone)]
pub struct GridResult<P> {
    pub best_params: P,
    pub best_score: f64,
    /// `(params, mean score)` for every candidate in input order.
    pub all: Vec<(P, f64)>,
}

/// Evaluates every candidate parameter set by mean cross-validated accuracy and returns the best (first wins ties).
pub fn grid_search<P, F>(x: &Matrix, y: &[usize], folds: &[Fold], candidates: &[P], fit_predict: F) -> Result<GridResult<P>>
where
    P: Clone,
    F: Fn(&P, &Matrix, &[usize], &Matrix) -> Result<Vec<usize>>,
{
    if candidates.is_empty() {
        return Err(Error::Msg("grid_search: no candidates".into()));
    }
    let mut all = Vec::with_capacity(candidates.len());
    for p in candidates {
        let scores = cross_val_score(x, y, folds, |a, b, c| fit_predict(p, a, b, c))?;
        all.push((p.clone(), scores.iter().sum::<f64>() / scores.len().max(1) as f64));
    }
    let best = (0..all.len()).fold(0, |b, i| if all[i].1 > all[b].1 { i } else { b });
    Ok(GridResult { best_params: all[best].0.clone(), best_score: all[best].1, all })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::KnnClassifier;

    #[test]
    fn splitters_partition_correctly() {
        let folds = kfold(10, 3, true, 1).unwrap();
        let mut seen: Vec<usize> = folds.iter().flat_map(|f| f.test.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
        assert!(folds.iter().all(|f| f.train.len() + f.test.len() == 10 && f.test.iter().all(|i| !f.train.contains(i))));
        assert!(folds.iter().all(|f| (3..=4).contains(&f.test.len())));
        assert!(kfold(3, 5, false, 0).is_err());
        // stratified: each fold's test set mirrors the 2:1 class ratio
        let labels: Vec<usize> = (0..30).map(|i| usize::from(i % 3 == 0)).collect();
        for f in stratified_kfold(&labels, 5, true, 2).unwrap() {
            let pos = f.test.iter().filter(|&&i| labels[i] == 1).count();
            assert_eq!((f.test.len(), pos), (6, 2));
        }
        assert!(stratified_kfold(&[0, 0, 0, 1], 3, false, 0).is_err());
        // groups stay together
        let groups = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3];
        for f in group_kfold(&groups, 2).unwrap() {
            let test_groups: std::collections::HashSet<_> = f.test.iter().map(|&i| groups[i]).collect();
            assert!(f.train.iter().all(|&i| !test_groups.contains(&groups[i])));
        }
        assert!(group_kfold(&groups, 5).is_err());
        // time series: train always precedes test
        let ts = time_series_split(12, 3).unwrap();
        assert_eq!(ts.len(), 3);
        for f in &ts {
            assert!(f.train.iter().max().unwrap() < f.test.iter().min().unwrap());
        }
        assert!(ts[0].train.len() < ts[2].train.len());
        assert!(time_series_split(3, 3).is_err());
    }

    #[test]
    fn cross_validation_and_grid_search_choose_a_good_k() {
        // two noisy-free clusters plus label noise on a few points: tiny k overfits it
        let mut rng = Rng::new(7);
        let mut x: Matrix = Vec::new();
        let mut y = Vec::new();
        for i in 0..120 {
            let c = i % 2;
            x.push(vec![c as f64 * 3.0 + rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0]);
            y.push(if i % 17 == 0 { 1 - c } else { c });
        }
        let folds = stratified_kfold(&y, 5, true, 3).unwrap();
        let knn = |k: &usize, a: &Matrix, b: &[usize], c: &Matrix| Ok(KnnClassifier::fit(a.clone(), b.to_vec(), *k)?.predict(c));
        let res = grid_search(&x, &y, &folds, &[1usize, 5, 15], knn).unwrap();
        assert_eq!(res.all.len(), 3);
        assert!(res.best_params > 1, "k = 1 memorizes the flipped labels: {:?}", res.all);
        assert!(res.best_score > 0.85);
        let scores = cross_val_score(&x, &y, &folds, |a, b, c| Ok(KnnClassifier::fit(a.clone(), b.to_vec(), 5)?.predict(c))).unwrap();
        assert_eq!(scores.len(), 5);
        assert!(cross_val_score(&x, &y[..5], &folds, |_, _, _| Ok(vec![])).is_err());
        assert!(grid_search::<usize, _>(&x, &y, &folds, &[], |_, _, _, _| Ok(vec![])).is_err());
    }
}
