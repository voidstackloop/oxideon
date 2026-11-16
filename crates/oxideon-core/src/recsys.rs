//! Recommender-system and pattern-mining tools: biased [`MatrixFactorization`] (SGD), [`ItemKnn`] collaborative
//! filtering, and frequent itemsets with association rules ([`apriori`], [`association_rules`]).

use crate::error::{Error, Result};
use crate::rng::Rng;
use std::collections::{BTreeSet, HashMap};

/// An observed rating.
#[derive(Debug, Clone, Copy)]
pub struct Rating {
    pub user: usize,
    pub item: usize,
    pub value: f64,
}

/// Biased matrix factorization `r ~ mu + b_u + b_i + p_u . q_i` trained by SGD with L2 regularization.
#[derive(Debug, Clone)]
pub struct MatrixFactorization {
    mu: f64,
    bu: Vec<f64>,
    bi: Vec<f64>,
    p: Vec<Vec<f64>>,
    q: Vec<Vec<f64>>,
}

impl MatrixFactorization {
    #[allow(clippy::too_many_arguments)]
    pub fn fit(ratings: &[Rating], n_users: usize, n_items: usize, factors: usize, epochs: usize, lr: f64, reg: f64, seed: u64) -> Result<Self> {
        if ratings.is_empty() || factors == 0 || ratings.iter().any(|r| r.user >= n_users || r.item >= n_items || !r.value.is_finite()) {
            return Err(Error::Msg("MatrixFactorization: need ratings with valid ids and factors >= 1".into()));
        }
        let mut rng = Rng::new(seed);
        let mu = ratings.iter().map(|r| r.value).sum::<f64>() / ratings.len() as f64;
        let init = |rng: &mut Rng| (rng.next_f64() - 0.5) * 0.1;
        let mut m = Self {
            mu,
            bu: vec![0.0; n_users],
            bi: vec![0.0; n_items],
            p: (0..n_users).map(|_| (0..factors).map(|_| init(&mut rng)).collect()).collect(),
            q: (0..n_items).map(|_| (0..factors).map(|_| init(&mut rng)).collect()).collect(),
        };
        let mut order: Vec<usize> = (0..ratings.len()).collect();
        for _ in 0..epochs {
            rng.shuffle(&mut order);
            for &k in &order {
                let r = ratings[k];
                let err = r.value - m.predict(r.user, r.item);
                m.bu[r.user] += lr * (err - reg * m.bu[r.user]);
                m.bi[r.item] += lr * (err - reg * m.bi[r.item]);
                for f in 0..factors {
                    let (pu, qi) = (m.p[r.user][f], m.q[r.item][f]);
                    m.p[r.user][f] += lr * (err * qi - reg * pu);
                    m.q[r.item][f] += lr * (err * pu - reg * qi);
                }
            }
        }
        Ok(m)
    }

    pub fn predict(&self, user: usize, item: usize) -> f64 {
        self.mu + self.bu[user] + self.bi[item] + self.p[user].iter().zip(&self.q[item]).map(|(a, b)| a * b).sum::<f64>()
    }

    pub fn rmse(&self, ratings: &[Rating]) -> f64 {
        (ratings.iter().map(|r| (r.value - self.predict(r.user, r.item)).powi(2)).sum::<f64>() / ratings.len().max(1) as f64).sqrt()
    }

    /// The `k` highest-scoring items for `user` that are not in `exclude`.
    pub fn recommend(&self, user: usize, k: usize, exclude: &[usize]) -> Vec<(usize, f64)> {
        let mut s: Vec<(usize, f64)> = (0..self.q.len()).filter(|i| !exclude.contains(i)).map(|i| (i, self.predict(user, i))).collect();
        s.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        s.truncate(k);
        s
    }
}

/// Item-based collaborative filtering with adjusted-cosine-style similarity on mean-centred user ratings.
#[derive(Debug, Clone)]
pub struct ItemKnn {
    by_user: Vec<Vec<(usize, f64)>>, // centred ratings
    user_mean: Vec<f64>,
    sim: HashMap<(usize, usize), f64>,
    n_items: usize,
    pub k: usize,
}

impl ItemKnn {
    pub fn fit(ratings: &[Rating], n_users: usize, n_items: usize, k: usize) -> Result<Self> {
        if ratings.is_empty() || k == 0 || ratings.iter().any(|r| r.user >= n_users || r.item >= n_items) {
            return Err(Error::Msg("ItemKnn: need ratings with valid ids and k >= 1".into()));
        }
        let mut sums = vec![(0.0, 0usize); n_users];
        ratings.iter().for_each(|r| { sums[r.user].0 += r.value; sums[r.user].1 += 1; });
        let user_mean: Vec<f64> = sums.iter().map(|s| if s.1 == 0 { 0.0 } else { s.0 / s.1 as f64 }).collect();
        let mut by_user = vec![Vec::new(); n_users];
        ratings.iter().for_each(|r| by_user[r.user].push((r.item, r.value - user_mean[r.user])));
        let (mut dot, mut norm) = (HashMap::<(usize, usize), f64>::new(), vec![0.0; n_items]);
        for u in &by_user {
            for (a, &(i, vi)) in u.iter().enumerate() {
                norm[i] += vi * vi;
                for &(j, vj) in &u[a + 1..] {
                    *dot.entry((i.min(j), i.max(j))).or_insert(0.0) += vi * vj;
                }
            }
        }
        let sim = dot.into_iter().filter_map(|((i, j), d)| { let den = (norm[i] * norm[j]).sqrt(); (den > 0.0).then_some(((i, j), d / den)) }).collect();
        Ok(Self { by_user, user_mean, sim, n_items, k })
    }

    fn similarity(&self, i: usize, j: usize) -> f64 {
        self.sim.get(&(i.min(j), i.max(j))).copied().unwrap_or(0.0)
    }

    /// Predicted rating of `item` for `user` from the user's `k` most similar rated items.
    pub fn predict(&self, user: usize, item: usize) -> f64 {
        let mut cands: Vec<(f64, f64)> = self.by_user[user].iter().filter(|&&(j, _)| j != item).map(|&(j, c)| (self.similarity(item, j), c)).filter(|(s, _)| *s != 0.0).collect();
        cands.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
        cands.truncate(self.k);
        let den: f64 = cands.iter().map(|c| c.0.abs()).sum();
        self.user_mean[user] + if den == 0.0 { 0.0 } else { cands.iter().map(|c| c.0 * c.1).sum::<f64>() / den }
    }

    pub fn recommend(&self, user: usize, k: usize) -> Vec<(usize, f64)> {
        let seen: BTreeSet<usize> = self.by_user[user].iter().map(|x| x.0).collect();
        let mut s: Vec<(usize, f64)> = (0..self.n_items).filter(|i| !seen.contains(i)).map(|i| (i, self.predict(user, i))).collect();
        s.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        s.truncate(k);
        s
    }
}

/// Frequent itemsets (Apriori): every itemset contained in at least `min_support` (fraction) of the transactions.
/// Returns `(itemset, support)` sorted by size then support.
pub fn apriori(transactions: &[Vec<usize>], min_support: f64) -> Result<Vec<(Vec<usize>, f64)>> {
    if transactions.is_empty() || !(0.0 < min_support && min_support <= 1.0) {
        return Err(Error::Msg("apriori: need transactions and 0 < min_support <= 1".into()));
    }
    let n = transactions.len() as f64;
    let sets: Vec<BTreeSet<usize>> = transactions.iter().map(|t| t.iter().copied().collect()).collect();
    let support = |items: &BTreeSet<usize>| sets.iter().filter(|t| items.is_subset(t)).count() as f64 / n;
    let mut level: Vec<BTreeSet<usize>> = sets.iter().flatten().copied().collect::<BTreeSet<_>>().into_iter().map(|i| BTreeSet::from([i])).filter(|s| support(s) >= min_support).collect();
    let mut all = Vec::new();
    while !level.is_empty() {
        let mut found: Vec<(Vec<usize>, f64)> = level.iter().map(|s| (s.iter().copied().collect(), support(s))).collect();
        found.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        all.extend(found);
        let mut next: BTreeSet<Vec<usize>> = BTreeSet::new();
        for a in &level {
            for b in &level {
                let u: BTreeSet<usize> = a.union(b).copied().collect();
                if u.len() == a.len() + 1 {
                    // prune: every subset of size k must be frequent
                    if u.iter().all(|&x| { let mut sub = u.clone(); sub.remove(&x); level.contains(&sub) }) {
                        next.insert(u.into_iter().collect());
                    }
                }
            }
        }
        level = next.into_iter().map(|v| v.into_iter().collect::<BTreeSet<_>>()).filter(|s| support(s) >= min_support).collect();
    }
    Ok(all)
}

/// An association rule `antecedent -> consequent`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    pub antecedent: Vec<usize>,
    pub consequent: Vec<usize>,
    pub support: f64,
    pub confidence: f64,
    pub lift: f64,
}

/// Rules with at least `min_confidence` derived from the output of [`apriori`].
pub fn association_rules(itemsets: &[(Vec<usize>, f64)], min_confidence: f64) -> Vec<Rule> {
    let sup: HashMap<&Vec<usize>, f64> = itemsets.iter().map(|(s, v)| (s, *v)).collect();
    let mut rules = Vec::new();
    for (items, s) in itemsets.iter().filter(|(i, _)| i.len() >= 2) {
        let m = items.len();
        for mask in 1..(1u32 << m) - 1 {
            let ant: Vec<usize> = (0..m).filter(|b| mask >> b & 1 == 1).map(|b| items[b]).collect();
            let con: Vec<usize> = (0..m).filter(|b| mask >> b & 1 == 0).map(|b| items[b]).collect();
            if let (Some(&sa), Some(&sc)) = (sup.get(&ant), sup.get(&con)) {
                let conf = s / sa;
                if conf >= min_confidence {
                    rules.push(Rule { antecedent: ant, consequent: con, support: *s, confidence: conf, lift: conf / sc });
                }
            }
        }
    }
    rules.sort_by(|a, b| b.confidence.total_cmp(&a.confidence).then(b.lift.total_cmp(&a.lift)).then(a.antecedent.cmp(&b.antecedent)));
    rules
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Users 0-9 like items 0-4 (rate high) and dislike 5-9; users 10-19 the opposite; 30% of cells observed.
    fn two_taste_ratings(seed: u64) -> (Vec<Rating>, Vec<Rating>) {
        let mut rng = Rng::new(seed);
        let (mut train, mut test) = (Vec::new(), Vec::new());
        for u in 0..20 {
            for i in 0..10 {
                let likes = (u < 10) == (i < 5);
                let value = if likes { 4.5 } else { 1.5 } + (rng.next_f64() - 0.5) * 0.4;
                let r = Rating { user: u, item: i, value };
                match rng.next_f64() {
                    x if x < 0.6 => train.push(r),
                    x if x < 0.8 => test.push(r),
                    _ => {}
                }
            }
        }
        (train, test)
    }

    #[test]
    fn factorization_generalizes_to_held_out_ratings() {
        let (train, test) = two_taste_ratings(1);
        let mf = MatrixFactorization::fit(&train, 20, 10, 4, 200, 0.05, 0.02, 7).unwrap();
        let (tr, te) = (mf.rmse(&train), mf.rmse(&test));
        assert!(tr < 0.5 && te < 0.8, "train {tr}, test {te}");
        let global = { let m = train.iter().map(|r| r.value).sum::<f64>() / train.len() as f64; (test.iter().map(|r| (r.value - m).powi(2)).sum::<f64>() / test.len() as f64).sqrt() };
        assert!(te < global * 0.5, "far better than predicting the mean: {te} vs {global}");
        // user 0 should be recommended items 0-4 first
        let seen: Vec<usize> = train.iter().filter(|r| r.user == 0).map(|r| r.item).collect();
        let rec = mf.recommend(0, 10, &seen);
        let mean_of = |pred: fn(usize) -> bool| { let v: Vec<f64> = rec.iter().filter(|(i, _)| pred(*i)).map(|x| x.1).collect(); v.iter().sum::<f64>() / v.len().max(1) as f64 };
        assert!(mean_of(|i| i < 5) > mean_of(|i| i >= 5) + 1.5, "{rec:?}");
        assert!(rec.iter().all(|(i, _)| !seen.contains(i)));
        assert!(MatrixFactorization::fit(&[], 2, 2, 2, 5, 0.1, 0.0, 0).is_err());
    }

    #[test]
    fn item_knn_predicts_from_similar_items() {
        let (train, test) = two_taste_ratings(2);
        let knn = ItemKnn::fit(&train, 20, 10, 5).unwrap();
        let rmse = (test.iter().map(|r| (r.value - knn.predict(r.user, r.item)).powi(2)).sum::<f64>() / test.len() as f64).sqrt();
        assert!(rmse < 1.2, "{rmse}");
        let rec = knn.recommend(0, 3);
        assert!(rec.iter().all(|(i, _)| !train.iter().any(|r| r.user == 0 && r.item == *i)));
        assert!(ItemKnn::fit(&train, 20, 10, 0).is_err());
    }

    #[test]
    fn apriori_and_rules() {
        // classic basket example: {bread=0, milk=1, diaper=2, beer=3, eggs=4}
        let t = vec![vec![0, 1], vec![0, 2, 3, 4], vec![1, 2, 3, 4], vec![0, 1, 2, 3], vec![0, 1, 2, 4]];
        let sets = apriori(&t, 0.6).unwrap();
        let get = |items: &[usize]| sets.iter().find(|(s, _)| s == items).map(|x| x.1);
        assert_eq!(get(&[0]), Some(0.8));
        assert_eq!(get(&[2]), Some(0.8));
        assert_eq!(get(&[0, 1]), Some(0.6));
        assert_eq!(get(&[2, 3]), Some(0.6));
        assert_eq!(get(&[3, 4]), None); // support 0.4 < 0.6
        assert!(sets.iter().all(|(s, v)| *v >= 0.6 && !s.is_empty()));
        let rules = association_rules(&sets, 0.75);
        assert!(rules.iter().all(|r| r.confidence >= 0.75));
        // {3} -> {2}: beer always comes with diapers (confidence 1, lift 1 / 0.8)
        let r = rules.iter().find(|r| r.antecedent == vec![3] && r.consequent == vec![2]).expect("rule 3 -> 2");
        assert!((r.confidence - 1.0).abs() < 1e-12 && (r.lift - 1.25).abs() < 1e-12 && (r.support - 0.6).abs() < 1e-12);
        assert!(apriori(&[], 0.5).is_err() && apriori(&t, 0.0).is_err());
    }
}
