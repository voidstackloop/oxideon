//! Differentiable Neuro-Symbolic Inductive Logic Programming ($\partial$ILP) & Datalog Engine.
//!
//! Features:
//! - Continuous First-Order Logic Valuations over entities: facts $R(x, y) \in [0, 1]$.
//! - Continuous T-Norm fuzzy logic operators: Product, Gödel, and Łukasiewicz t-norms & s-norms.
//! - Continuous Immediate Consequence Operator ($T_{\mathcal{P}}$):
//!   - Base rules: $H(X, Y) \leftarrow B(X, Y)$.
//!   - Transitive / chain rules: $H(X, Y) \leftarrow B_1(X, Z) \wedge B_2(Z, Y)$.
//! - Multi-step differentiable forward chaining to deduce transitive closures, graph reachability,
//!   ancestor relationships, and complex relational reasoning without discrete combinatorial search.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Continuous T-norm logic system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TNorm {
    /// Product t-norm: $T(a, b) = a \cdot b$, $S(a, b) = a + b - a \cdot b$.
    Product,
    /// Gödel (min/max) t-norm: $T(a, b) = \min(a, b)$, $S(a, b) = \max(a, b)$.
    Godel,
    /// Łukasiewicz t-norm: $T(a, b) = \max(0, a + b - 1)$, $S(a, b) = \min(1, a + b)$.
    Lukasiewicz,
}

impl TNorm {
    /// Evaluates fuzzy conjunction $a \wedge b$.
    pub fn conjunction(&self, a: f64, b: f64) -> f64 {
        match self {
            Self::Product => a * b,
            Self::Godel => a.min(b),
            Self::Lukasiewicz => (a + b - 1.0).max(0.0),
        }
    }

    /// Evaluates fuzzy disjunction $a \vee b$.
    pub fn disjunction(&self, a: f64, b: f64) -> f64 {
        match self {
            Self::Product => a + b - a * b,
            Self::Godel => a.max(b),
            Self::Lukasiewicz => (a + b).min(1.0),
        }
    }
}

/// Datalog clause template type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleType {
    /// $H(X, Y) \leftarrow B(X, Y)$
    Base { head: usize, body: usize },
    /// $H(X, Y) \leftarrow B_1(X, Z) \wedge B_2(Z, Y)$
    Transitive {
        head: usize,
        body1: usize,
        body2: usize,
    },
}

/// Weighted Datalog candidate clause.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatalogRule {
    pub rule_type: RuleType,
    pub weight: f64,
}

/// Differentiable Inductive Logic Programming ($\partial$ILP) Engine.
pub struct DifferentiableILPEngine {
    pub num_entities: usize,
    pub num_predicates: usize,
    pub t_norm: TNorm,
    pub rules: Vec<DatalogRule>,
}

impl DifferentiableILPEngine {
    /// Creates a new $\partial$ILP engine with specified entity and predicate counts.
    pub fn new(num_entities: usize, num_predicates: usize, t_norm: TNorm) -> Self {
        Self {
            num_entities,
            num_predicates,
            t_norm,
            rules: Vec::new(),
        }
    }

    /// Adds a base clause $H(X, Y) \leftarrow B(X, Y)$ with rule confidence weight.
    pub fn add_base_rule(&mut self, head: usize, body: usize, weight: f64) -> Result<()> {
        if head >= self.num_predicates || body >= self.num_predicates {
            return Err(Error::Msg("Predicate index out of bounds".into()));
        }
        self.rules.push(DatalogRule {
            rule_type: RuleType::Base { head, body },
            weight,
        });
        Ok(())
    }

    /// Adds a transitive clause $H(X, Y) \leftarrow B_1(X, Z) \wedge B_2(Z, Y)$ with rule confidence weight.
    pub fn add_transitive_rule(
        &mut self,
        head: usize,
        body1: usize,
        body2: usize,
        weight: f64,
    ) -> Result<()> {
        if head >= self.num_predicates
            || body1 >= self.num_predicates
            || body2 >= self.num_predicates
        {
            return Err(Error::Msg("Predicate index out of bounds".into()));
        }
        self.rules.push(DatalogRule {
            rule_type: RuleType::Transitive { head, body1, body2 },
            weight,
        });
        Ok(())
    }

    /// Performs one step of the continuous Immediate Consequence Operator $T_{\mathcal{P}}(V)$:
    /// Input `facts` has shape `[num_predicates, num_entities, num_entities]`.
    pub fn infer_step(&self, facts: &Tensor) -> Result<Tensor> {
        let expected_shape = [self.num_predicates, self.num_entities, self.num_entities];
        if facts.shape() != expected_shape {
            return Err(Error::Msg(format!(
                "Facts shape {:?} does not match expected {:?}",
                facts.shape(),
                expected_shape
            )));
        }
        // `rules` is pub: a tampered predicate id overflowed the head/body
        // offsets below (`add_*_rule` checks construction only).
        let bad_id = self.rules.iter().find_map(|rule| match &rule.rule_type {
            RuleType::Base { head, body } => [*head, *body]
                .into_iter()
                .find(|id| *id >= self.num_predicates),
            RuleType::Transitive { head, body1, body2 } => [*head, *body1, *body2]
                .into_iter()
                .find(|id| *id >= self.num_predicates),
        });
        if let Some(id) = bad_id {
            return Err(Error::Msg(format!(
                "Rule predicate id {id} out of bounds (0..{})",
                self.num_predicates
            )));
        }

        let facts_data = facts.to_vec::<f64>()?;
        let n = self.num_entities;
        let mut new_facts = facts_data.clone();

        for rule in &self.rules {
            let conf = 1.0 / (1.0 + (-rule.weight).exp()); // sigmoid confidence

            match rule.rule_type {
                RuleType::Base { head, body } => {
                    let head_off = head * n * n;
                    let body_off = body * n * n;
                    for i in 0..n {
                        for j in 0..n {
                            let idx_h = head_off + i * n + j;
                            let idx_b = body_off + i * n + j;
                            let derived = self.t_norm.conjunction(facts_data[idx_b], conf);
                            new_facts[idx_h] = self.t_norm.disjunction(new_facts[idx_h], derived);
                        }
                    }
                }
                RuleType::Transitive { head, body1, body2 } => {
                    let head_off = head * n * n;
                    let b1_off = body1 * n * n;
                    let b2_off = body2 * n * n;

                    for i in 0..n {
                        for j in 0..n {
                            let mut path_disj = 0.0;
                            for k in 0..n {
                                let v1 = facts_data[b1_off + i * n + k];
                                let v2 = facts_data[b2_off + k * n + j];
                                let path_conj = self.t_norm.conjunction(v1, v2);
                                path_disj = self.t_norm.disjunction(path_disj, path_conj);
                            }
                            let derived = self.t_norm.conjunction(path_disj, conf);
                            let idx_h = head_off + i * n + j;
                            new_facts[idx_h] = self.t_norm.disjunction(new_facts[idx_h], derived);
                        }
                    }
                }
            }
        }

        Tensor::from_vec(new_facts, &expected_shape)
    }

    /// Multi-step forward deductive chaining until convergence or maximum `steps`.
    pub fn deductive_forward(&self, initial_facts: &Tensor, steps: usize) -> Result<Tensor> {
        // Each step runs a full consequence pass; closure needs at most
        // `num_entities` rounds, so this is generous.
        crate::limits::check_repeats("steps", steps)?;
        let mut current = initial_facts.clone();
        for _ in 0..steps {
            current = self.infer_step(&current)?;
        }
        Ok(current)
    }

    /// Evaluates binary cross-entropy loss over positive and negative ground-truth relations:
    /// Targets given as `(predicate, entity_x, entity_y)`.
    pub fn evaluate_loss(
        &self,
        predicted_facts: &Tensor,
        positive_targets: &[(usize, usize, usize)],
        negative_targets: &[(usize, usize, usize)],
    ) -> Result<f64> {
        let n = self.num_entities;
        let data = predicted_facts.to_vec::<f64>()?;
        // Counts are pub: the expected size overflowed on tampering.
        let need = self
            .num_predicates
            .checked_mul(n)
            .and_then(|v| v.checked_mul(n))
            .ok_or_else(|| Error::Msg("predicted facts size overflows".into()))?;
        if data.len() != need {
            return Err(Error::Msg(format!(
                "predicted facts have {} entries, need {need}",
                data.len()
            )));
        }
        for &(p, i, j) in positive_targets.iter().chain(negative_targets.iter()) {
            if p >= self.num_predicates || i >= n || j >= n {
                return Err(Error::Msg(format!(
                    "target ({p}, {i}, {j}) out of bounds for {}x{n}x{n} facts",
                    self.num_predicates
                )));
            }
        }
        let mut total_loss = 0.0;
        let count = (positive_targets.len() + negative_targets.len()).max(1) as f64;

        for &(p, i, j) in positive_targets {
            let val = data[p * n * n + i * n + j].clamp(1e-7, 1.0 - 1e-7);
            total_loss -= val.ln();
        }

        for &(p, i, j) in negative_targets {
            let val = data[p * n * n + i * n + j].clamp(1e-7, 1.0 - 1e-7);
            total_loss -= (1.0 - val).ln();
        }

        Ok(total_loss / count)
    }
}
