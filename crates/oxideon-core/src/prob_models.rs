//! Discrete probabilistic models: [`MarkovChain`] (stationary distribution, hitting times, absorption, simulation,
//! maximum-likelihood estimation) and a discrete [`BayesNet`] with exact inference by variable elimination.

use crate::error::{Error, Result};
use crate::rng::Rng;

type Mat = Vec<Vec<f64>>;

fn solve(mut a: Mat, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
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

/// Finite Markov chain with a row-stochastic transition matrix.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    pub p: Mat,
}

impl MarkovChain {
    pub fn new(p: Mat) -> Result<Self> {
        let n = p.len();
        if n == 0 || p.iter().any(|r| r.len() != n || r.iter().any(|&v| v < 0.0 || !v.is_finite()) || (r.iter().sum::<f64>() - 1.0).abs() > 1e-9) {
            return Err(Error::Msg("MarkovChain: need a square row-stochastic matrix".into()));
        }
        Ok(Self { p })
    }

    /// Maximum-likelihood chain from an observed state sequence (rows with no observations become self-loops).
    pub fn estimate(seq: &[usize], n_states: usize, smoothing: f64) -> Result<Self> {
        if n_states == 0 || seq.iter().any(|&s| s >= n_states) || seq.len() < 2 {
            return Err(Error::Msg("MarkovChain::estimate: need >= 2 observations within 0..n_states".into()));
        }
        let mut c = vec![vec![smoothing; n_states]; n_states];
        seq.windows(2).for_each(|w| c[w[0]][w[1]] += 1.0);
        let p = c.into_iter().enumerate().map(|(i, r)| { let s: f64 = r.iter().sum(); if s == 0.0 { (0..n_states).map(|j| f64::from(u8::from(i == j))).collect() } else { r.iter().map(|v| v / s).collect() } }).collect();
        Self::new(p)
    }

    /// Stationary distribution `pi = pi P` (unique for an irreducible chain), found by solving the linear system.
    pub fn stationary(&self) -> Result<Vec<f64>> {
        let n = self.p.len();
        // (P^T - I) pi = 0 with the sum-to-one constraint replacing the last equation
        let mut a: Mat = (0..n).map(|i| (0..n).map(|j| self.p[j][i] - f64::from(u8::from(i == j))).collect()).collect();
        a[n - 1] = vec![1.0; n];
        let mut b = vec![0.0; n];
        b[n - 1] = 1.0;
        let pi = solve(a, b).ok_or_else(|| Error::Msg("MarkovChain::stationary: no unique stationary distribution (reducible chain?)".into()))?;
        if pi.iter().any(|&v| v < -1e-9) {
            return Err(Error::Msg("MarkovChain::stationary: the solution has negative entries (reducible chain)".into()));
        }
        Ok(pi.into_iter().map(|v| v.max(0.0)).collect())
    }

    /// Expected number of steps to first reach any state in `targets`, from every state.
    pub fn hitting_times(&self, targets: &[usize]) -> Result<Vec<f64>> {
        let n = self.p.len();
        if targets.is_empty() || targets.iter().any(|&t| t >= n) {
            return Err(Error::Msg("hitting_times: need valid target states".into()));
        }
        let free: Vec<usize> = (0..n).filter(|i| !targets.contains(i)).collect();
        let a: Mat = free.iter().map(|&i| free.iter().map(|&j| f64::from(u8::from(i == j)) - self.p[i][j]).collect()).collect();
        let sol = if free.is_empty() { Vec::new() } else { solve(a, vec![1.0; free.len()]).ok_or_else(|| Error::Msg("hitting_times: the targets are not reachable from every state".into()))? };
        let mut out = vec![0.0; n];
        free.iter().zip(sol).for_each(|(&i, v)| out[i] = v);
        Ok(out)
    }

    /// For a chain with absorbing states, the probability of ending in each absorbing state from every transient
    /// state: returns `(transient states, absorbing states, matrix[transient][absorbing])`.
    pub fn absorption_probabilities(&self) -> Result<(Vec<usize>, Vec<usize>, Mat)> {
        let n = self.p.len();
        let absorbing: Vec<usize> = (0..n).filter(|&i| (self.p[i][i] - 1.0).abs() < 1e-12).collect();
        let transient: Vec<usize> = (0..n).filter(|i| !absorbing.contains(i)).collect();
        if absorbing.is_empty() || transient.is_empty() {
            return Err(Error::Msg("absorption_probabilities: the chain needs both absorbing and transient states".into()));
        }
        // B = (I - Q)^-1 R, solved column by column
        let m = transient.len();
        let a: Mat = (0..m).map(|i| (0..m).map(|j| f64::from(u8::from(i == j)) - self.p[transient[i]][transient[j]]).collect()).collect();
        let cols: Vec<Vec<f64>> = absorbing.iter().map(|&s| solve(a.clone(), transient.iter().map(|&t| self.p[t][s]).collect()).ok_or_else(|| Error::Msg("absorption_probabilities: singular system".into()))).collect::<Result<_>>()?;
        let b = (0..m).map(|i| cols.iter().map(|c| c[i]).collect()).collect();
        Ok((transient, absorbing, b))
    }

    /// Simulates `steps` transitions from `start`; returns the visited states including the start.
    pub fn simulate(&self, start: usize, steps: usize, seed: u64) -> Vec<usize> {
        let mut rng = Rng::new(seed);
        let mut path = vec![start];
        for _ in 0..steps {
            let (mut u, cur) = (rng.next_f64(), *path.last().unwrap());
            let mut next = self.p.len() - 1;
            for (j, &pr) in self.p[cur].iter().enumerate() {
                if u < pr {
                    next = j;
                    break;
                }
                u -= pr;
            }
            path.push(next);
        }
        path
    }
}

/// A discrete factor over named-by-index variables (`vars`), with `card[i]` states each; `table` is row-major with
/// the last listed variable fastest.
#[derive(Debug, Clone)]
struct Factor {
    vars: Vec<usize>,
    card: Vec<usize>,
    table: Vec<f64>,
}

impl Factor {
    fn index(&self, assign: &[usize]) -> usize {
        assign.iter().zip(&self.card).fold(0, |acc, (&a, &c)| acc * c + a)
    }

    fn multiply(&self, o: &Factor) -> Factor {
        let mut vars = self.vars.clone();
        let mut card = self.card.clone();
        for (v, c) in o.vars.iter().zip(&o.card) {
            if !vars.contains(v) {
                vars.push(*v);
                card.push(*c);
            }
        }
        let total: usize = card.iter().product();
        let mut table = vec![0.0; total];
        for (flat, slot) in table.iter_mut().enumerate() {
            let mut rem = flat;
            let mut assign = vec![0; vars.len()];
            for i in (0..vars.len()).rev() {
                assign[i] = rem % card[i];
                rem /= card[i];
            }
            let pick = |f: &Factor| f.index(&f.vars.iter().map(|v| assign[vars.iter().position(|x| x == v).unwrap()]).collect::<Vec<_>>());
            *slot = self.table[pick(self)] * o.table[pick(o)];
        }
        Factor { vars, card, table }
    }

    fn sum_out(&self, var: usize) -> Factor {
        let pos = self.vars.iter().position(|&v| v == var).unwrap();
        let (vars, card): (Vec<usize>, Vec<usize>) = self.vars.iter().zip(&self.card).filter(|(v, _)| **v != var).map(|(v, c)| (*v, *c)).unzip();
        let mut table = vec![0.0; card.iter().product::<usize>().max(1)];
        for (flat, &val) in self.table.iter().enumerate() {
            let mut rem = flat;
            let mut assign = vec![0; self.vars.len()];
            for i in (0..self.vars.len()).rev() {
                assign[i] = rem % self.card[i];
                rem /= self.card[i];
            }
            assign.remove(pos);
            let idx = assign.iter().zip(&card).fold(0, |acc, (&a, &c)| acc * c + a);
            table[idx] += val;
        }
        Factor { vars, card, table }
    }

    fn reduce(&self, var: usize, value: usize) -> Factor {
        let Some(pos) = self.vars.iter().position(|&v| v == var) else { return self.clone() };
        let (vars, card): (Vec<usize>, Vec<usize>) = self.vars.iter().zip(&self.card).filter(|(v, _)| **v != var).map(|(v, c)| (*v, *c)).unzip();
        let mut table = Vec::new();
        for (flat, &val) in self.table.iter().enumerate() {
            let mut rem = flat;
            let mut assign = vec![0; self.vars.len()];
            for i in (0..self.vars.len()).rev() {
                assign[i] = rem % self.card[i];
                rem /= self.card[i];
            }
            if assign[pos] == value {
                table.push(val);
            }
        }
        Factor { vars, card, table }
    }
}

/// Discrete Bayesian network. Add variables with [`BayesNet::add_variable`] (parents must already exist), then query
/// marginals with optional evidence.
#[derive(Debug, Clone, Default)]
pub struct BayesNet {
    card: Vec<usize>,
    parents: Vec<Vec<usize>>,
    /// CPT per variable: rows indexed by the parent assignment (first parent slowest), each row a distribution.
    cpt: Vec<Vec<f64>>,
    names: Vec<String>,
}

impl BayesNet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a variable with `states` values, its parents (by index) and a CPT with `prod(parent states)` rows of
    /// `states` probabilities each. Returns the new variable's index.
    pub fn add_variable(&mut self, name: &str, states: usize, parents: &[usize], cpt: Vec<f64>) -> Result<usize> {
        if states < 2 || parents.iter().any(|&p| p >= self.card.len()) {
            return Err(Error::Msg("BayesNet: need >= 2 states and existing parents".into()));
        }
        let rows: usize = parents.iter().map(|&p| self.card[p]).product();
        if cpt.len() != rows * states || cpt.chunks(states).any(|r| (r.iter().sum::<f64>() - 1.0).abs() > 1e-9 || r.iter().any(|&v| v < 0.0)) {
            return Err(Error::Msg(format!("BayesNet: CPT of {name:?} must have {rows} rows of {states} probabilities summing to 1")));
        }
        self.card.push(states);
        self.parents.push(parents.to_vec());
        self.cpt.push(cpt);
        self.names.push(name.to_string());
        Ok(self.card.len() - 1)
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    fn factor(&self, v: usize) -> Factor {
        let mut vars = self.parents[v].clone();
        vars.push(v);
        let card = vars.iter().map(|&x| self.card[x]).collect();
        Factor { vars, card, table: self.cpt[v].clone() }
    }

    /// `P(query | evidence)` by variable elimination; `evidence` lists `(variable, observed state)`.
    pub fn query(&self, query: usize, evidence: &[(usize, usize)]) -> Result<Vec<f64>> {
        if query >= self.card.len() || evidence.iter().any(|&(v, s)| v >= self.card.len() || s >= self.card[v]) || evidence.iter().any(|&(v, _)| v == query) {
            return Err(Error::Msg("BayesNet::query: invalid query or evidence".into()));
        }
        let mut factors: Vec<Factor> = (0..self.card.len()).map(|v| evidence.iter().fold(self.factor(v), |f, &(ev, es)| f.reduce(ev, es))).collect();
        for v in (0..self.card.len()).filter(|&v| v != query && !evidence.iter().any(|e| e.0 == v)) {
            let (with, without): (Vec<Factor>, Vec<Factor>) = factors.into_iter().partition(|f| f.vars.contains(&v));
            factors = without;
            if let Some(first) = with.first().cloned() {
                let prod = with[1..].iter().fold(first, |acc, f| acc.multiply(f));
                factors.push(prod.sum_out(v));
            }
        }
        let joint = factors.iter().skip(1).fold(factors[0].clone(), |acc, f| acc.multiply(f));
        let mut marg = joint;
        for v in marg.vars.clone().into_iter().filter(|&v| v != query) {
            marg = marg.sum_out(v);
        }
        let z: f64 = marg.table.iter().sum();
        if z <= 0.0 {
            return Err(Error::Msg("BayesNet::query: the evidence has zero probability".into()));
        }
        Ok(marg.table.iter().map(|v| v / z).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    #[test]
    fn markov_chain_analysis() {
        // weather: sunny/rainy with P(s->s) = 0.9, P(r->s) = 0.5 -> stationary (5/6, 1/6)
        let mc = MarkovChain::new(vec![vec![0.9, 0.1], vec![0.5, 0.5]]).unwrap();
        let pi = mc.stationary().unwrap();
        close(pi[0], 5.0 / 6.0, 1e-12);
        close(pi[1], 1.0 / 6.0, 1e-12);
        // expected time to reach rain from sunny: 1 / 0.1 = 10
        let h = mc.hitting_times(&[1]).unwrap();
        close(h[0], 10.0, 1e-9);
        close(h[1], 0.0, 1e-12);
        // long simulation visits states in stationary proportion
        let path = mc.simulate(0, 50_000, 3);
        close(path.iter().filter(|&&s| s == 0).count() as f64 / path.len() as f64, 5.0 / 6.0, 0.01);
        // estimation recovers the matrix
        let est = MarkovChain::estimate(&path, 2, 0.0).unwrap();
        close(est.p[0][0], 0.9, 0.01);
        close(est.p[1][0], 0.5, 0.03);
        assert!(MarkovChain::new(vec![vec![0.5, 0.4]]).is_err() && MarkovChain::estimate(&[0], 2, 0.0).is_err());
        // reducible chain has no unique stationary distribution
        assert!(MarkovChain::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap().stationary().is_err());
    }

    #[test]
    fn gamblers_ruin_absorption() {
        // fair gambler's ruin on 0..=4 starting from 1, 2, 3: win probability i / 4
        let mut p = vec![vec![0.0; 5]; 5];
        p[0][0] = 1.0;
        p[4][4] = 1.0;
        for i in 1..4 {
            p[i][i - 1] = 0.5;
            p[i][i + 1] = 0.5;
        }
        let mc = MarkovChain::new(p).unwrap();
        let (transient, absorbing, b) = mc.absorption_probabilities().unwrap();
        assert_eq!((transient, absorbing), (vec![1, 2, 3], vec![0, 4]));
        for (i, row) in b.iter().enumerate() {
            close(row[1], (i + 1) as f64 / 4.0, 1e-12);
            close(row[0] + row[1], 1.0, 1e-12);
        }
        // expected duration from the middle: i * (4 - i) = 4
        close(mc.hitting_times(&[0, 4]).unwrap()[2], 4.0, 1e-9);
        assert!(MarkovChain::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]).unwrap().absorption_probabilities().is_err());
    }

    #[test]
    fn bayes_net_inference_matches_the_textbook_alarm_network() {
        // Russell & Norvig: Burglary (B), Earthquake (E), Alarm (A | B, E), JohnCalls (J | A), MaryCalls (M | A)
        let mut net = BayesNet::new();
        let b = net.add_variable("B", 2, &[], vec![0.999, 0.001]).unwrap();
        let e = net.add_variable("E", 2, &[], vec![0.998, 0.002]).unwrap();
        // rows (B, E): (0,0) (0,1) (1,0) (1,1); each row is [P(A=0), P(A=1)]
        let a = net.add_variable("A", 2, &[b, e], vec![0.999, 0.001, 0.71, 0.29, 0.06, 0.94, 0.05, 0.95]).unwrap();
        let j = net.add_variable("J", 2, &[a], vec![0.95, 0.05, 0.10, 0.90]).unwrap();
        let m = net.add_variable("M", 2, &[a], vec![0.99, 0.01, 0.30, 0.70]).unwrap();
        // P(B | J=1, M=1) = 0.284 (book value)
        let post = net.query(b, &[(j, 1), (m, 1)]).unwrap();
        close(post[1], 0.284, 0.001);
        close(post.iter().sum::<f64>(), 1.0, 1e-12);
        // prior marginals: P(A=1) = 0.002516...
        close(net.query(a, &[]).unwrap()[1], 0.002_516_442, 1e-8);
        // explaining away: learning of an earthquake lowers belief in a burglary given the alarm
        let with_alarm = net.query(b, &[(a, 1)]).unwrap()[1];
        let explained = net.query(b, &[(a, 1), (e, 1)]).unwrap()[1];
        assert!(explained < with_alarm / 10.0, "{explained} vs {with_alarm}");
        assert_eq!(net.index_of("M"), Some(m));
        assert!(net.query(9, &[]).is_err() && net.query(b, &[(b, 1)]).is_err());
        assert!(net.add_variable("X", 2, &[a], vec![0.5, 0.5]).is_err() && net.add_variable("Y", 2, &[], vec![0.7, 0.7]).is_err());
    }
}
