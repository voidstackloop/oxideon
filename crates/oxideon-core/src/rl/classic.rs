//! Classic reinforcement learning: multi-armed bandits ([`EpsilonGreedy`], [`Ucb1`], [`ThompsonBernoulli`],
//! [`LinUcb`]), MDP planning ([`value_iteration`], [`policy_iteration`]), tabular control ([`QLearning`] with a
//! `sarsa` flag), and replay memory ([`ReplayBuffer`] uniform, [`PrioritizedReplay`] proportional).

use crate::error::{Error, Result};
use crate::rng::Rng;

// ------------------------------------------------------------------ bandits

/// A bandit policy: choose an arm, then learn from its reward.
pub trait Bandit {
    fn select(&mut self) -> usize;
    fn update(&mut self, arm: usize, reward: f64);
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b]).then(b.cmp(&a))).unwrap()
}

/// Epsilon-greedy over sample means.
pub struct EpsilonGreedy {
    pub epsilon: f64,
    counts: Vec<f64>,
    means: Vec<f64>,
    rng: Rng,
}

impl EpsilonGreedy {
    pub fn new(arms: usize, epsilon: f64, seed: u64) -> Result<Self> {
        if arms == 0 || !(0.0..=1.0).contains(&epsilon) {
            return Err(Error::Msg("EpsilonGreedy: need arms >= 1 and epsilon in [0, 1]".into()));
        }
        Ok(Self { epsilon, counts: vec![0.0; arms], means: vec![0.0; arms], rng: Rng::new(seed) })
    }
}

impl Bandit for EpsilonGreedy {
    fn select(&mut self) -> usize {
        if let Some(a) = self.counts.iter().position(|&c| c == 0.0) {
            return a;
        }
        if self.rng.next_f64() < self.epsilon { (self.rng.next_u64() % self.means.len() as u64) as usize } else { argmax(&self.means) }
    }
    fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1.0;
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm];
    }
}

/// UCB1: optimism in the face of uncertainty.
pub struct Ucb1 {
    counts: Vec<f64>,
    means: Vec<f64>,
    t: f64,
}

impl Ucb1 {
    pub fn new(arms: usize) -> Result<Self> {
        if arms == 0 {
            return Err(Error::Msg("Ucb1: need at least one arm".into()));
        }
        Ok(Self { counts: vec![0.0; arms], means: vec![0.0; arms], t: 0.0 })
    }
}

impl Bandit for Ucb1 {
    fn select(&mut self) -> usize {
        if let Some(a) = self.counts.iter().position(|&c| c == 0.0) {
            return a;
        }
        let scores: Vec<f64> = (0..self.means.len()).map(|a| self.means[a] + (2.0 * self.t.ln() / self.counts[a]).sqrt()).collect();
        argmax(&scores)
    }
    fn update(&mut self, arm: usize, reward: f64) {
        self.t += 1.0;
        self.counts[arm] += 1.0;
        self.means[arm] += (reward - self.means[arm]) / self.counts[arm];
    }
}

/// Thompson sampling for Bernoulli rewards with Beta(1, 1) priors.
pub struct ThompsonBernoulli {
    wins: Vec<f64>,
    losses: Vec<f64>,
    rng: Rng,
}

fn gamma_sample(shape: f64, rng: &mut Rng) -> f64 {
    // Marsaglia-Tsang (shape >= 1); boost smaller shapes
    if shape < 1.0 {
        return gamma_sample(shape + 1.0, rng) * rng.next_f64().max(1e-300).powf(1.0 / shape);
    }
    let (d, c) = (shape - 1.0 / 3.0, 1.0 / (9.0 * (shape - 1.0 / 3.0)).sqrt());
    loop {
        let (u1, u2) = (rng.next_f64().max(1e-300), rng.next_f64());
        let x = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 && rng.next_f64().max(1e-300).ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

impl ThompsonBernoulli {
    pub fn new(arms: usize, seed: u64) -> Result<Self> {
        if arms == 0 {
            return Err(Error::Msg("ThompsonBernoulli: need at least one arm".into()));
        }
        Ok(Self { wins: vec![1.0; arms], losses: vec![1.0; arms], rng: Rng::new(seed) })
    }
}

impl Bandit for ThompsonBernoulli {
    fn select(&mut self) -> usize {
        let draws: Vec<f64> = (0..self.wins.len())
            .map(|a| {
                let (g1, g2) = (gamma_sample(self.wins[a], &mut self.rng), gamma_sample(self.losses[a], &mut self.rng));
                g1 / (g1 + g2)
            })
            .collect();
        argmax(&draws)
    }
    fn update(&mut self, arm: usize, reward: f64) {
        if reward > 0.5 { self.wins[arm] += 1.0 } else { self.losses[arm] += 1.0 }
    }
}

/// Disjoint LinUCB for contextual bandits: per-arm ridge regression with an upper-confidence bonus.
pub struct LinUcb {
    alpha: f64,
    a_inv: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<f64>>,
}

impl LinUcb {
    pub fn new(arms: usize, dim: usize, alpha: f64) -> Result<Self> {
        if arms == 0 || dim == 0 || alpha < 0.0 {
            return Err(Error::Msg("LinUcb: need arms >= 1, dim >= 1 and alpha >= 0".into()));
        }
        let eye: Vec<Vec<f64>> = (0..dim).map(|i| (0..dim).map(|j| f64::from(u8::from(i == j))).collect()).collect();
        Ok(Self { alpha, a_inv: vec![eye; arms], b: vec![vec![0.0; dim]; arms] })
    }

    fn mat_vec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
        m.iter().map(|r| r.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
    }

    pub fn select(&self, context: &[f64]) -> usize {
        let scores: Vec<f64> = (0..self.b.len())
            .map(|a| {
                let theta = Self::mat_vec(&self.a_inv[a], &self.b[a]);
                let mean: f64 = theta.iter().zip(context).map(|(x, y)| x * y).sum();
                let ax = Self::mat_vec(&self.a_inv[a], context);
                mean + self.alpha * context.iter().zip(&ax).map(|(x, y)| x * y).sum::<f64>().max(0.0).sqrt()
            })
            .collect();
        argmax(&scores)
    }

    /// Sherman-Morrison update of the arm's inverse design matrix.
    pub fn update(&mut self, arm: usize, context: &[f64], reward: f64) {
        let ax = Self::mat_vec(&self.a_inv[arm], context);
        let denom = 1.0 + context.iter().zip(&ax).map(|(x, y)| x * y).sum::<f64>();
        for i in 0..ax.len() {
            for j in 0..ax.len() {
                self.a_inv[arm][i][j] -= ax[i] * ax[j] / denom;
            }
        }
        self.b[arm].iter_mut().zip(context).for_each(|(b, x)| *b += reward * x);
    }
}

// ------------------------------------------------------------------ MDP planning

/// Tabular MDP: `p[s][a]` is a list of `(next_state, probability, reward)` outcomes.
pub type Mdp = Vec<Vec<Vec<(usize, f64, f64)>>>;

fn check_mdp(mdp: &Mdp, gamma: f64) -> Result<()> {
    if mdp.is_empty() || !(0.0..1.0).contains(&gamma) {
        return Err(Error::Msg("MDP: need states and 0 <= gamma < 1".into()));
    }
    for (s, acts) in mdp.iter().enumerate() {
        if acts.is_empty() {
            return Err(Error::Msg(format!("MDP: state {s} has no actions")));
        }
        for out in acts {
            let total: f64 = out.iter().map(|o| o.1).sum();
            if (total - 1.0).abs() > 1e-9 || out.iter().any(|o| o.0 >= mdp.len() || o.1 < 0.0) {
                return Err(Error::Msg(format!("MDP: transition probabilities of state {s} must be valid and sum to 1")));
            }
        }
    }
    Ok(())
}

fn q_of(mdp: &Mdp, v: &[f64], s: usize, a: usize, gamma: f64) -> f64 {
    mdp[s][a].iter().map(|&(n, p, r)| p * (r + gamma * v[n])).sum()
}

/// Value iteration; returns `(values, greedy policy)`.
pub fn value_iteration(mdp: &Mdp, gamma: f64, tol: f64) -> Result<(Vec<f64>, Vec<usize>)> {
    check_mdp(mdp, gamma)?;
    let mut v = vec![0.0; mdp.len()];
    for _ in 0..100_000 {
        let next: Vec<f64> = (0..mdp.len()).map(|s| (0..mdp[s].len()).map(|a| q_of(mdp, &v, s, a, gamma)).fold(f64::NEG_INFINITY, f64::max)).collect();
        let delta = next.iter().zip(&v).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        v = next;
        if delta < tol {
            break;
        }
    }
    let policy = (0..mdp.len()).map(|s| argmax(&(0..mdp[s].len()).map(|a| q_of(mdp, &v, s, a, gamma)).collect::<Vec<_>>())).collect();
    Ok((v, policy))
}

/// Policy iteration (iterative policy evaluation); returns `(values, optimal policy)`.
pub fn policy_iteration(mdp: &Mdp, gamma: f64) -> Result<(Vec<f64>, Vec<usize>)> {
    check_mdp(mdp, gamma)?;
    let mut policy = vec![0usize; mdp.len()];
    let mut v = vec![0.0; mdp.len()];
    for _ in 0..1000 {
        for _ in 0..10_000 {
            let next: Vec<f64> = (0..mdp.len()).map(|s| q_of(mdp, &v, s, policy[s], gamma)).collect();
            let delta = next.iter().zip(&v).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            v = next;
            if delta < 1e-12 {
                break;
            }
        }
        let improved: Vec<usize> = (0..mdp.len())
            .map(|s| {
                let q: Vec<f64> = (0..mdp[s].len()).map(|a| q_of(mdp, &v, s, a, gamma)).collect();
                // keep the current action unless another is strictly better (avoids cycling on ties)
                let best = argmax(&q);
                if q[best] > q[policy[s]] + 1e-12 { best } else { policy[s] }
            })
            .collect();
        if improved == policy {
            break;
        }
        policy = improved;
    }
    Ok((v, policy))
}

// ------------------------------------------------------------------ tabular control

/// Tabular Q-learning (off-policy) or SARSA (on-policy) with epsilon-greedy exploration.
pub struct QLearning {
    pub q: Vec<Vec<f64>>,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub sarsa: bool,
    rng: Rng,
}

impl QLearning {
    pub fn new(states: usize, actions: usize, alpha: f64, gamma: f64, epsilon: f64, sarsa: bool, seed: u64) -> Result<Self> {
        if states == 0 || actions == 0 || !(0.0..=1.0).contains(&epsilon) || alpha <= 0.0 {
            return Err(Error::Msg("QLearning: need states, actions >= 1, alpha > 0 and epsilon in [0, 1]".into()));
        }
        Ok(Self { q: vec![vec![0.0; actions]; states], alpha, gamma, epsilon, sarsa, rng: Rng::new(seed) })
    }

    pub fn act(&mut self, state: usize) -> usize {
        if self.rng.next_f64() < self.epsilon { (self.rng.next_u64() % self.q[state].len() as u64) as usize } else { argmax(&self.q[state]) }
    }

    /// One TD update. `next_action` is the action actually chosen in `next_state` (used by SARSA; pass `None` at
    /// terminal states, which are bootstrapped as zero).
    pub fn learn(&mut self, s: usize, a: usize, reward: f64, next_state: usize, next_action: Option<usize>, terminal: bool) {
        let bootstrap = if terminal {
            0.0
        } else if self.sarsa {
            self.q[next_state][next_action.unwrap_or_else(|| argmax(&self.q[next_state]))]
        } else {
            self.q[next_state].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        };
        self.q[s][a] += self.alpha * (reward + self.gamma * bootstrap - self.q[s][a]);
    }

    pub fn greedy_policy(&self) -> Vec<usize> {
        self.q.iter().map(|r| argmax(r)).collect()
    }
}

// ------------------------------------------------------------------ replay

/// One stored transition.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Vec<f64>,
    pub done: bool,
}

/// Fixed-capacity ring buffer with uniform sampling.
pub struct ReplayBuffer {
    items: Vec<Transition>,
    capacity: usize,
    next: usize,
    rng: Rng,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, seed: u64) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Msg("ReplayBuffer: capacity must be positive".into()));
        }
        Ok(Self { items: Vec::new(), capacity, next: 0, rng: Rng::new(seed) })
    }

    pub fn push(&mut self, t: Transition) {
        if self.items.len() < self.capacity {
            self.items.push(t);
        } else {
            self.items[self.next] = t;
        }
        self.next = (self.next + 1) % self.capacity;
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// `batch` transitions drawn uniformly with replacement.
    pub fn sample(&mut self, batch: usize) -> Result<Vec<&Transition>> {
        if self.items.is_empty() {
            return Err(Error::Msg("ReplayBuffer::sample: the buffer is empty".into()));
        }
        let n = self.items.len() as u64;
        let idx: Vec<usize> = (0..batch).map(|_| (self.rng.next_u64() % n) as usize).collect();
        Ok(idx.into_iter().map(|i| &self.items[i]).collect())
    }
}

/// Proportional prioritized replay (Schaul et al. 2016): sampling probability `p_i^alpha`, importance weights
/// `(N * P(i))^-beta` normalized by their maximum.
pub struct PrioritizedReplay {
    items: Vec<Transition>,
    priority: Vec<f64>,
    capacity: usize,
    next: usize,
    pub alpha: f64,
    rng: Rng,
}

impl PrioritizedReplay {
    pub fn new(capacity: usize, alpha: f64, seed: u64) -> Result<Self> {
        if capacity == 0 || alpha < 0.0 {
            return Err(Error::Msg("PrioritizedReplay: need capacity >= 1 and alpha >= 0".into()));
        }
        Ok(Self { items: Vec::new(), priority: Vec::new(), capacity, next: 0, alpha, rng: Rng::new(seed) })
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// New transitions get the current maximum priority so they are replayed at least once.
    pub fn push(&mut self, t: Transition) {
        let p = self.priority.iter().cloned().fold(1.0, f64::max);
        if self.items.len() < self.capacity {
            self.items.push(t);
            self.priority.push(p);
        } else {
            self.items[self.next] = t;
            self.priority[self.next] = p;
        }
        self.next = (self.next + 1) % self.capacity;
    }

    /// Returns `(index, transition, importance weight)` triples.
    pub fn sample(&mut self, batch: usize, beta: f64) -> Result<Vec<(usize, &Transition, f64)>> {
        if self.items.is_empty() {
            return Err(Error::Msg("PrioritizedReplay::sample: the buffer is empty".into()));
        }
        let probs: Vec<f64> = self.priority.iter().map(|p| p.powf(self.alpha)).collect();
        let total: f64 = probs.iter().sum();
        let n = self.items.len() as f64;
        let mut picks = Vec::with_capacity(batch);
        for _ in 0..batch {
            let mut u = self.rng.next_f64() * total;
            let mut idx = probs.len() - 1;
            for (i, p) in probs.iter().enumerate() {
                if u < *p {
                    idx = i;
                    break;
                }
                u -= p;
            }
            picks.push(idx);
        }
        let weight = |i: usize| (n * probs[i] / total).powf(-beta);
        let max_w = (0..probs.len()).map(weight).fold(0.0, f64::max);
        Ok(picks.into_iter().map(|i| (i, &self.items[i], weight(i) / max_w)).collect())
    }

    /// Sets the priorities of sampled transitions from their new TD errors (`|delta| + eps`).
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        for (&i, e) in indices.iter().zip(td_errors) {
            self.priority[i] = e.abs() + 1e-6;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_bandit<B: Bandit>(b: &mut B, probs: &[f64], steps: usize, seed: u64) -> (usize, f64) {
        let mut rng = Rng::new(seed);
        let (mut best_pulls, mut total) = (0, 0.0);
        let best = probs.iter().cloned().fold(0.0, f64::max);
        for _ in 0..steps {
            let a = b.select();
            let r = f64::from(u8::from(rng.next_f64() < probs[a]));
            b.update(a, r);
            total += r;
            best_pulls += usize::from(probs[a] == best);
        }
        (best_pulls, total)
    }

    #[test]
    fn bandits_find_the_best_arm() {
        let probs = [0.2, 0.5, 0.8, 0.4];
        for (name, pulls) in [
            ("eps", run_bandit(&mut EpsilonGreedy::new(4, 0.1, 1).unwrap(), &probs, 3000, 2).0),
            ("ucb", run_bandit(&mut Ucb1::new(4).unwrap(), &probs, 3000, 3).0),
            ("ts", run_bandit(&mut ThompsonBernoulli::new(4, 4).unwrap(), &probs, 3000, 5).0),
        ] {
            assert!(pulls > 2000, "{name}: best arm pulled {pulls}/3000 times");
        }
        // Thompson sampling has lower regret than uniform random play
        let (_, ts_total) = run_bandit(&mut ThompsonBernoulli::new(4, 7).unwrap(), &probs, 3000, 8);
        assert!(ts_total > 3000.0 * 0.475 + 300.0, "{ts_total}");
        assert!(EpsilonGreedy::new(0, 0.1, 0).is_err() && Ucb1::new(0).is_err() && EpsilonGreedy::new(2, 2.0, 0).is_err());
    }

    #[test]
    fn linucb_uses_context() {
        // reward depends on which arm matches the context sign pattern
        let mut b = LinUcb::new(2, 2, 0.5).unwrap();
        let mut rng = Rng::new(11);
        let mut correct = 0;
        for t in 0..1500 {
            let ctx = if rng.next_f64() < 0.5 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
            let best = usize::from(ctx[1] == 1.0);
            let a = b.select(&ctx);
            b.update(a, &ctx, if a == best { 1.0 } else { 0.0 });
            if t >= 1000 && a == best {
                correct += 1;
            }
        }
        assert!(correct > 480, "{correct}/500");
    }

    /// Corridor of 4 cells; action 0 = left, 1 = right; reaching cell 3 pays 1 and ends; cell 3 loops to itself.
    fn corridor() -> Mdp {
        (0..4usize).map(|s| (0..2usize).map(|a| { let n = if s == 3 { 3 } else if a == 1 { s + 1 } else { s.saturating_sub(1) }; vec![(n, 1.0, if n == 3 && s != 3 { 1.0 } else { 0.0 })] }).collect()).collect()
    }

    #[test]
    fn planning_solves_the_corridor() {
        let (v, pi) = value_iteration(&corridor(), 0.9, 1e-10).unwrap();
        assert_eq!(&pi[..3], &[1, 1, 1]);
        assert!((v[2] - 1.0).abs() < 1e-6 && (v[1] - 0.9).abs() < 1e-6 && (v[0] - 0.81).abs() < 1e-6, "{v:?}");
        let (v2, pi2) = policy_iteration(&corridor(), 0.9).unwrap();
        assert_eq!(&pi2[..3], &[1, 1, 1]);
        assert!(v.iter().zip(&v2).all(|(a, b)| (a - b).abs() < 1e-6));
        let mut bad = corridor();
        bad[0][0] = vec![(0, 0.5, 0.0)];
        assert!(value_iteration(&bad, 0.9, 1e-6).is_err() && value_iteration(&corridor(), 1.0, 1e-6).is_err());
    }

    #[test]
    fn tabular_control_learns_the_corridor() {
        for sarsa in [false, true] {
            let mut agent = QLearning::new(4, 2, 0.2, 0.9, 0.2, sarsa, 3).unwrap();
            for _ in 0..600 {
                let mut s = 0;
                for _ in 0..30 {
                    let a = agent.act(s);
                    let n = if a == 1 { s + 1 } else { s.saturating_sub(1) };
                    let done = n == 3;
                    let na = if done { None } else { Some(agent.act(n)) };
                    agent.learn(s, a, if done { 1.0 } else { 0.0 }, n, na, done);
                    if done {
                        break;
                    }
                    s = n;
                }
            }
            assert_eq!(&agent.greedy_policy()[..3], &[1, 1, 1], "sarsa={sarsa}: {:?}", agent.q);
            assert!(agent.q[2][1] > 0.8 && agent.q[0][1] > agent.q[0][0]);
        }
        assert!(QLearning::new(0, 2, 0.1, 0.9, 0.1, false, 0).is_err());
    }

    fn t(i: usize) -> Transition {
        Transition { state: vec![i as f64], action: i % 2, reward: i as f64, next_state: vec![i as f64 + 1.0], done: false }
    }

    #[test]
    fn replay_buffers() {
        let mut b = ReplayBuffer::new(5, 1).unwrap();
        assert!(b.sample(2).is_err());
        (0..8).for_each(|i| b.push(t(i)));
        assert_eq!(b.len(), 5);
        let s = b.sample(200).unwrap();
        assert!(s.iter().all(|x| x.reward >= 3.0), "oldest three were overwritten");
        assert!(ReplayBuffer::new(0, 0).is_err());
        // prioritized: a high-priority transition is sampled far more often, with a smaller importance weight
        let mut p = PrioritizedReplay::new(10, 1.0, 2).unwrap();
        (0..10).for_each(|i| p.push(t(i)));
        p.update_priorities(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], &[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 5.0]);
        let batch = p.sample(2000, 0.6).unwrap();
        let hot = batch.iter().filter(|x| x.0 == 9).count();
        assert!(hot > 1800, "{hot}");
        let w_hot = batch.iter().find(|x| x.0 == 9).unwrap().2;
        let w_cold = batch.iter().find(|x| x.0 != 9).unwrap().2;
        assert!(w_hot < w_cold && (w_cold - 1.0).abs() < 1e-9);
        assert!(PrioritizedReplay::new(0, 0.6, 0).is_err());
    }
}
