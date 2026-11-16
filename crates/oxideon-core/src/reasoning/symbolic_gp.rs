//! Symbolic Genetic Programming & Pareto-Optimal Formula Search.
//!
//! Autonomous scientific discovery of analytical mathematical laws from empirical observations:
//! - Expression tree evaluation with protected operators ($+$, $-$, $\times$, $\div$, $\sin$, $\cos$, $\exp$, $\ln$, $\operatorname{sqrt}$).
//! - Multiobjective Pareto optimization balancing prediction accuracy (MSE) against formula complexity (Occam's razor).
//! - Subtree crossover, point mutation, and non-dominated Pareto front extraction.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Unary and Binary Mathematical Operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Exp,
    Ln,
    Sqrt,
}

/// Abstract syntax tree node for symbolic expressions.
#[derive(Debug, Clone)]
pub enum SymbolicNode {
    Constant(f64),
    Variable(usize),
    Unary(OpType, Box<SymbolicNode>),
    Binary(OpType, Box<SymbolicNode>, Box<SymbolicNode>),
}

impl SymbolicNode {
    /// Evaluates the expression tree on an input vector $\mathbf{x}$.
    /// A mismatched operator arity (e.g. `Binary(Sin, ..)`, only buildable by
    /// hand since the engine never emits it) yields `NaN`; use
    /// [`SymbolicNode::try_evaluate`] for a strict error instead.
    pub fn evaluate(&self, x: &[f64]) -> f64 {
        self.try_evaluate(x).unwrap_or(f64::NAN)
    }

    /// Strict evaluation: an operator applied at the wrong arity is an error
    /// rather than a silent fallback value.
    pub fn try_evaluate(&self, x: &[f64]) -> Result<f64> {
        self.try_evaluate_limited(x, 0)
    }

    /// Depth-limited worker: evaluation recurses per tree level, and a
    /// hand-built 100k-deep tree overflowed the thread stack (process abort)
    /// instead of erroring. Engine-built trees stay dozens deep at most.
    fn try_evaluate_limited(&self, x: &[f64], depth: usize) -> Result<f64> {
        if depth > 4096 {
            return Err(Error::Msg(format!(
                "SymbolicNode: evaluation depth {depth} exceeds 4096"
            )));
        }
        match self {
            SymbolicNode::Constant(c) => Ok(*c),
            SymbolicNode::Variable(idx) => {
                if *idx < x.len() {
                    Ok(x[*idx])
                } else {
                    Ok(0.0)
                }
            }
            SymbolicNode::Unary(op, child) => {
                let v = child.try_evaluate_limited(x, depth + 1)?;
                match op {
                    OpType::Sin => Ok(v.sin()),
                    OpType::Cos => Ok(v.cos()),
                    OpType::Exp => {
                        let clamped = v.clamp(-20.0, 20.0);
                        Ok(clamped.exp())
                    }
                    OpType::Ln => {
                        let abs_v = v.abs();
                        if abs_v > 1e-12 {
                            Ok(abs_v.ln())
                        } else {
                            Ok(-27.6)
                        }
                    }
                    OpType::Sqrt => Ok(v.abs().sqrt()),
                    other => Err(Error::Msg(format!(
                        "SymbolicNode: {other:?} is not a unary operator"
                    ))),
                }
            }
            SymbolicNode::Binary(op, left, right) => {
                let l = left.try_evaluate_limited(x, depth + 1)?;
                let r = right.try_evaluate_limited(x, depth + 1)?;
                match op {
                    OpType::Add => Ok(l + r),
                    OpType::Sub => Ok(l - r),
                    OpType::Mul => {
                        let prod = l * r;
                        if prod.is_finite() { Ok(prod) } else { Ok(0.0) }
                    }
                    OpType::Div => {
                        if r.abs() > 1e-9 {
                            let div = l / r;
                            if div.is_finite() { Ok(div) } else { Ok(1.0) }
                        } else {
                            Ok(1.0)
                        }
                    }
                    other => Err(Error::Msg(format!(
                        "SymbolicNode: {other:?} is not a binary operator"
                    ))),
                }
            }
        }
    }

    /// Tree complexity: total number of nodes in the expression tree.
    pub fn complexity(&self) -> usize {
        // Iterative (explicit stack): like evaluation, this recursed per
        // level and a hand-built deep tree overflowed the thread stack.
        let mut count = 0usize;
        let mut stack: Vec<&SymbolicNode> = vec![self];
        while let Some(node) = stack.pop() {
            count = count.saturating_add(1);
            match node {
                SymbolicNode::Constant(_) | SymbolicNode::Variable(_) => {}
                SymbolicNode::Unary(_, child) => stack.push(child),
                SymbolicNode::Binary(_, left, right) => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        count
    }

    /// Maximum tree depth.
    pub fn depth(&self) -> usize {
        // Iterative, same reason as `complexity`.
        let mut max = 0usize;
        let mut stack: Vec<(&SymbolicNode, usize)> = vec![(self, 1)];
        while let Some((node, d)) = stack.pop() {
            max = max.max(d);
            match node {
                SymbolicNode::Constant(_) | SymbolicNode::Variable(_) => {}
                SymbolicNode::Unary(_, child) => stack.push((child, d.saturating_add(1))),
                SymbolicNode::Binary(_, left, right) => {
                    stack.push((left, d.saturating_add(1)));
                    stack.push((right, d.saturating_add(1)));
                }
            }
        }
        max
    }

    /// Formats the symbolic formula as a readable mathematical string.
    pub fn to_formula(&self) -> String {
        // Iterative (explicit action stack), same reason as `complexity`:
        // the recursive version overflowed on hand-built deep trees.
        enum Act<'a> {
            Visit(&'a SymbolicNode),
            Text(String),
        }
        let mut out = String::new();
        let mut stack: Vec<Act<'_>> = vec![Act::Visit(self)];
        while let Some(act) = stack.pop() {
            match act {
                Act::Text(s) => out.push_str(&s),
                Act::Visit(node) => match node {
                    SymbolicNode::Constant(c) => {
                        if (c.round() - c).abs() < 1e-6 {
                            out.push_str(&format!("{c:.0}"));
                        } else {
                            out.push_str(&format!("{c:.3}"));
                        }
                    }
                    SymbolicNode::Variable(idx) => out.push_str(&format!("x{idx}")),
                    SymbolicNode::Unary(op, child) => {
                        let name = match op {
                            OpType::Sin => "sin",
                            OpType::Cos => "cos",
                            OpType::Exp => "exp",
                            OpType::Ln => "ln",
                            OpType::Sqrt => "sqrt",
                            _ => "op",
                        };
                        stack.push(Act::Text(")".to_string()));
                        stack.push(Act::Visit(child));
                        stack.push(Act::Text(format!("{name}(")));
                    }
                    SymbolicNode::Binary(op, left, right) => {
                        let sym = match op {
                            OpType::Add => "+",
                            OpType::Sub => "-",
                            OpType::Mul => "*",
                            OpType::Div => "/",
                            _ => "+",
                        };
                        stack.push(Act::Text(")".to_string()));
                        stack.push(Act::Visit(right));
                        stack.push(Act::Text(format!(" {sym} ")));
                        stack.push(Act::Visit(left));
                        stack.push(Act::Text("(".to_string()));
                    }
                },
            }
        }
        out
    }
}

impl SymbolicNode {
    /// Moves the children out of `self` (leaving `Constant` leaves behind)
    /// so `Drop` can tear a deep tree down iteratively.
    fn detach_children(&mut self, pending: &mut Vec<SymbolicNode>) {
        match self {
            SymbolicNode::Unary(_, child) => {
                pending.push(std::mem::replace(&mut **child, SymbolicNode::Constant(0.0)));
            }
            SymbolicNode::Binary(_, left, right) => {
                pending.push(std::mem::replace(&mut **left, SymbolicNode::Constant(0.0)));
                pending.push(std::mem::replace(&mut **right, SymbolicNode::Constant(0.0)));
            }
            SymbolicNode::Constant(_) | SymbolicNode::Variable(_) => {}
        }
    }
}

impl Drop for SymbolicNode {
    fn drop(&mut self) {
        // Dropping a hand-built deep tree recursed per level (process
        // abort), e.g. when Python garbage-collects a deep SymbolicNode.
        let mut pending: Vec<SymbolicNode> = Vec::new();
        self.detach_children(&mut pending);
        while let Some(mut node) = pending.pop() {
            node.detach_children(&mut pending);
        }
    }
}

/// An individual in the Pareto-optimal population.
#[derive(Debug, Clone)]
pub struct ParetoIndividual {
    pub tree: SymbolicNode,
    pub mse: f64,
    pub complexity: usize,
    pub formula: String,
}

impl ParetoIndividual {
    pub fn new(tree: SymbolicNode, mse: f64) -> Self {
        let complexity = tree.complexity();
        let formula = tree.to_formula();
        Self {
            tree,
            mse,
            complexity,
            formula,
        }
    }

    /// True if `self` dominates `other` (strictly better in one metric and not worse in the other).
    pub fn dominates(&self, other: &ParetoIndividual) -> bool {
        let mse_better_or_equal = self.mse <= other.mse + 1e-9;
        let comp_better_or_equal = self.complexity <= other.complexity;
        let strictly_better = self.mse < other.mse - 1e-9 || self.complexity < other.complexity;

        mse_better_or_equal && comp_better_or_equal && strictly_better
    }
}

/// Configuration for Symbolic Genetic Search.
#[derive(Debug, Clone)]
pub struct SymbolicGpConfig {
    pub population_size: usize,
    pub num_generations: usize,
    pub max_depth: usize,
    pub crossover_prob: f64,
    pub mutation_prob: f64,
}

impl Default for SymbolicGpConfig {
    fn default() -> Self {
        Self {
            population_size: 60,
            num_generations: 30,
            max_depth: 5,
            crossover_prob: 0.70,
            mutation_prob: 0.25,
        }
    }
}

/// Symbolic Genetic Programming Engine for Scientific Discovery.
#[derive(Debug, Clone)]
pub struct SymbolicGpEngine {
    pub config: SymbolicGpConfig,
}

impl SymbolicGpEngine {
    pub fn new(config: SymbolicGpConfig) -> Self {
        Self { config }
    }

    /// Fits expression trees to empirical data points $\mathbf{X} \in \mathbb{R}^{N \times D}$ and $\mathbf{y} \in \mathbb{R}^N$.
    pub fn fit(&self, x: &Tensor, y: &Tensor, seed: Option<u64>) -> Result<Vec<ParetoIndividual>> {
        let x_shape = x.shape();
        let y_shape = y.shape();
        if x_shape.len() != 2 {
            return Err(Error::Msg(format!(
                "x must be 2D matrix [N, D], got {:?}",
                x_shape
            )));
        }
        let n_samples = x_shape[0];
        let n_vars = x_shape[1];
        if y_shape[0] != n_samples {
            return Err(Error::Msg(format!(
                "y length {} != x samples {n_samples}",
                y_shape[0]
            )));
        }
        // Population/rounds drive per-tree work each generation; real searches
        // use dozens (init is capped at depth 3, offspring at max_depth).
        crate::limits::check_repeats("population_size", self.config.population_size)?;
        crate::limits::check_repeats("num_generations", self.config.num_generations)?;

        let x_data = x.to_vec::<f64>()?;
        let y_data = y.to_vec::<f64>()?;
        // Only `y_shape[0]` was checked above: `[N, K]` targets silently
        // paired the first N of N*K values.
        if y_data.len() != n_samples {
            return Err(Error::Msg(format!(
                "y has {} entries, need one per sample ({n_samples})",
                y_data.len()
            )));
        }

        let mut rng = SimpleRng::new(seed.unwrap_or(2026));

        // Initialize population
        let mut population: Vec<SymbolicNode> = Vec::with_capacity(self.config.population_size);
        for _ in 0..self.config.population_size {
            let tree = Self::random_tree(n_vars, self.config.max_depth.min(3), &mut rng);
            population.push(tree);
        }

        // Add standard candidate templates (linear, quadratic, sin, cos)
        if n_vars > 0 {
            population.push(SymbolicNode::Variable(0));
            population.push(SymbolicNode::Binary(
                OpType::Mul,
                Box::new(SymbolicNode::Variable(0)),
                Box::new(SymbolicNode::Variable(0)),
            ));
            population.push(SymbolicNode::Unary(
                OpType::Sin,
                Box::new(SymbolicNode::Variable(0)),
            ));
        }

        // Evaluate MSE of a tree
        let eval_mse = |tree: &SymbolicNode| -> f64 {
            let mut sum_sq = 0.0;
            for i in 0..n_samples {
                let row = &x_data[i * n_vars..(i + 1) * n_vars];
                let pred = tree.evaluate(row);
                let diff = pred - y_data[i];
                sum_sq += diff * diff;
            }
            sum_sq / (n_samples as f64)
        };

        // Generations loop
        for _ in 0..self.config.num_generations {
            let mut evaluated: Vec<(SymbolicNode, f64)> = population
                .into_iter()
                .map(|tree| {
                    let mse = eval_mse(&tree);
                    (tree, mse)
                })
                .collect();

            // Sort by fitness (lower MSE is better)
            evaluated.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let elite_count = (self.config.population_size / 5).max(2);
            let mut next_gen: Vec<SymbolicNode> = evaluated
                .iter()
                .take(elite_count)
                .map(|(t, _)| t.clone())
                .collect();

            // Generate offspring
            while next_gen.len() < self.config.population_size {
                let p_choice = rng.next_f64();
                if p_choice < self.config.crossover_prob && evaluated.len() >= 2 {
                    // Tournament selection
                    let parent1 = Self::tournament_select(&evaluated, 3, &mut rng);
                    let parent2 = Self::tournament_select(&evaluated, 3, &mut rng);
                    let child = Self::crossover(parent1, parent2, &mut rng);
                    next_gen.push(child);
                } else if p_choice < self.config.crossover_prob + self.config.mutation_prob {
                    let parent = Self::tournament_select(&evaluated, 3, &mut rng);
                    let child = Self::mutate(parent, n_vars, self.config.max_depth, &mut rng);
                    next_gen.push(child);
                } else {
                    let random_tree = Self::random_tree(n_vars, self.config.max_depth, &mut rng);
                    next_gen.push(random_tree);
                }
            }

            population = next_gen;
        }

        // Final evaluation and Pareto Front extraction
        let individuals: Vec<ParetoIndividual> = population
            .into_iter()
            .map(|tree| {
                let mse = eval_mse(&tree);
                ParetoIndividual::new(tree, mse)
            })
            .collect();

        Ok(Self::extract_pareto_front(&individuals))
    }

    /// Extracts non-dominated individuals forming the Pareto optimal trade-off frontier.
    pub fn extract_pareto_front(population: &[ParetoIndividual]) -> Vec<ParetoIndividual> {
        let mut front = Vec::new();
        for ind in population {
            let is_dominated = population.iter().any(|other| other.dominates(ind));
            if !is_dominated
                && !front.iter().any(|existing: &ParetoIndividual| {
                    (existing.mse - ind.mse).abs() < 1e-9 && existing.complexity == ind.complexity
                })
            {
                front.push(ind.clone());
            }
        }
        front.sort_by_key(|a| a.complexity);
        front
    }

    fn tournament_select<'a>(
        evaluated: &'a [(SymbolicNode, f64)],
        k: usize,
        rng: &mut SimpleRng,
    ) -> &'a SymbolicNode {
        let n = evaluated.len();
        let mut best_idx = (rng.next_u32() as usize) % n;
        for _ in 1..k {
            let idx = (rng.next_u32() as usize) % n;
            if evaluated[idx].1 < evaluated[best_idx].1 {
                best_idx = idx;
            }
        }
        &evaluated[best_idx].0
    }

    fn random_tree(n_vars: usize, max_depth: usize, rng: &mut SimpleRng) -> SymbolicNode {
        if max_depth <= 1 || rng.next_f64() < 0.35 {
            if rng.next_f64() < 0.5 && n_vars > 0 {
                let var_idx = (rng.next_u32() as usize) % n_vars;
                SymbolicNode::Variable(var_idx)
            } else {
                let c = ((rng.next_u32() % 10) as f64) - 5.0;
                SymbolicNode::Constant(if c == 0.0 { 1.0 } else { c })
            }
        } else {
            let op_choice = rng.next_u32() % 6;
            match op_choice {
                0 => SymbolicNode::Unary(
                    OpType::Sin,
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                ),
                1 => SymbolicNode::Unary(
                    OpType::Cos,
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                ),
                2 => SymbolicNode::Binary(
                    OpType::Add,
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                ),
                3 => SymbolicNode::Binary(
                    OpType::Mul,
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                ),
                4 => SymbolicNode::Binary(
                    OpType::Sub,
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                ),
                _ => SymbolicNode::Binary(
                    OpType::Div,
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                    Box::new(Self::random_tree(n_vars, max_depth - 1, rng)),
                ),
            }
        }
    }

    fn crossover(
        parent1: &SymbolicNode,
        parent2: &SymbolicNode,
        rng: &mut SimpleRng,
    ) -> SymbolicNode {
        // Swap subtrees
        if rng.next_u32() % 2 == 0 {
            match parent1 {
                SymbolicNode::Binary(op, left, _) => {
                    SymbolicNode::Binary(*op, left.clone(), Box::new(parent2.clone()))
                }
                SymbolicNode::Unary(op, _) => SymbolicNode::Unary(*op, Box::new(parent2.clone())),
                _ => parent2.clone(),
            }
        } else {
            match parent2 {
                SymbolicNode::Binary(op, _, right) => {
                    SymbolicNode::Binary(*op, Box::new(parent1.clone()), right.clone())
                }
                _ => parent1.clone(),
            }
        }
    }

    fn mutate(
        parent: &SymbolicNode,
        n_vars: usize,
        _max_depth: usize,
        rng: &mut SimpleRng,
    ) -> SymbolicNode {
        match parent {
            SymbolicNode::Constant(c) => {
                let perturb = (rng.next_f64() * 0.4) - 0.2;
                SymbolicNode::Constant(c + perturb)
            }
            SymbolicNode::Variable(_) => {
                if rng.next_f64() < 0.5 {
                    Self::random_tree(n_vars, 2, rng)
                } else {
                    parent.clone()
                }
            }
            SymbolicNode::Unary(op, child) => {
                SymbolicNode::Unary(*op, Box::new(Self::mutate(child, n_vars, _max_depth, rng)))
            }
            SymbolicNode::Binary(op, left, right) => {
                if rng.next_f64() < 0.5 {
                    SymbolicNode::Binary(
                        *op,
                        Box::new(Self::mutate(left, n_vars, _max_depth, rng)),
                        right.clone(),
                    )
                } else {
                    SymbolicNode::Binary(
                        *op,
                        left.clone(),
                        Box::new(Self::mutate(right, n_vars, _max_depth, rng)),
                    )
                }
            }
        }
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64)
    }
}
