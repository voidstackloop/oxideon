//! Neural Algorithmic Reasoning: Pointer Graph Networks & Neural Bellman-Ford.
//!
//! Implements algorithmic message-passing architectures that align with classical graph algorithms:
//! - Pointer Graph Networks (PGN, Veličković et al. 2020) with attention-based predecessor pointer distributions.
//! - Neural Bellman-Ford Network (NBFNet, Zhu et al. 2021) generalized over $(\min, +)$, $(\max, \times)$, and $(\max, \min)$ semirings.
//! - Single-Source Shortest Paths (SSSP), bottleneck capacity, and path reconstruction.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Generalized Semiring for Algorithmic Message Passing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmicSemiring {
    /// Classical $(\min, +)$ tropical semiring for shortest path computation.
    MinPlus,
    /// $(\max, \times)$ semiring for maximum-reliability or probabilistic paths.
    MaxMul,
    /// $(\max, \min)$ bottleneck capacity semiring for widest paths.
    MaxMin,
}

/// Directed edge with weight and feature attributes.
#[derive(Debug, Clone)]
pub struct AlgorithmicEdge {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
    pub features: Vec<f64>,
}

/// Graph structure for neural algorithmic computation.
#[derive(Debug, Clone)]
pub struct AlgorithmicGraph {
    pub num_nodes: usize,
    pub edges: Vec<AlgorithmicEdge>,
}

impl AlgorithmicGraph {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            edges: Vec::new(),
        }
    }

    pub fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        weight: f64,
        features: Vec<f64>,
    ) -> Result<()> {
        if source >= self.num_nodes || target >= self.num_nodes {
            return Err(Error::Msg(format!(
                "Edge nodes ({source}, {target}) exceed graph num_nodes {}",
                self.num_nodes
            )));
        }
        self.edges.push(AlgorithmicEdge {
            source,
            target,
            weight,
            features,
        });
        Ok(())
    }

    pub fn incoming_edges(&self, target: usize) -> Vec<&AlgorithmicEdge> {
        self.edges.iter().filter(|e| e.target == target).collect()
    }

    pub fn outgoing_edges(&self, source: usize) -> Vec<&AlgorithmicEdge> {
        self.edges.iter().filter(|e| e.source == source).collect()
    }
}

/// Neural Bellman-Ford Network (NBFNet) execution engine.
#[derive(Debug, Clone)]
pub struct NeuralBellmanFord {
    pub semiring: AlgorithmicSemiring,
    pub hidden_dim: usize,
}

impl NeuralBellmanFord {
    pub fn new(semiring: AlgorithmicSemiring, hidden_dim: usize) -> Self {
        Self {
            semiring,
            hidden_dim,
        }
    }

    /// Solves Single-Source Shortest Paths (SSSP) or path reachability using generalized semiring message passing.
    pub fn solve(
        &self,
        graph: &AlgorithmicGraph,
        source: usize,
        max_steps: usize,
    ) -> Result<(Vec<f64>, Vec<Option<usize>>)> {
        let n = graph.num_nodes;
        if source >= n {
            return Err(Error::Msg(format!("Source node {source} >= num_nodes {n}")));
        }
        // `edges` is pub: tampered endpoints indexed the distance vecs out
        // of bounds below (`add_edge` checks construction only).
        for e in &graph.edges {
            if e.source >= n || e.target >= n {
                return Err(Error::Msg(format!(
                    "Edge nodes ({}, {}) exceed graph num_nodes {n}",
                    e.source, e.target
                )));
            }
        }

        let (fill, init_val, identity_val) = match self.semiring {
            AlgorithmicSemiring::MinPlus => (f64::INFINITY, 0.0, f64::INFINITY),
            AlgorithmicSemiring::MaxMul => (0.0, 1.0, 0.0),
            AlgorithmicSemiring::MaxMin => (0.0, f64::INFINITY, 0.0),
        };
        // Fallible: `num_nodes` is pub, and the per-node vecs aborted on
        // tampered counts (no arbitrary cap; big graphs on big iron work).
        let mut distances = crate::layout::try_with_capacity(n)?;
        distances.resize(n, fill);
        let mut predecessors: Vec<Option<usize>> = crate::layout::try_with_capacity(n)?;
        predecessors.resize(n, None);
        distances[source] = init_val;

        let steps = if max_steps == 0 {
            n.saturating_sub(1)
        } else {
            max_steps
        };
        // Rounds never converge on negative cycles; each round scans all edges.
        crate::limits::check_iterations("max_steps", steps)?;

        for _ in 0..steps {
            let mut updated = false;
            let current = distances.clone();

            for edge in &graph.edges {
                let u = edge.source;
                let v = edge.target;
                let w = edge.weight;

                if current[u] == identity_val && self.semiring == AlgorithmicSemiring::MinPlus {
                    continue;
                }

                match self.semiring {
                    AlgorithmicSemiring::MinPlus => {
                        let candidate = current[u] + w;
                        if candidate < distances[v] - 1e-12 {
                            distances[v] = candidate;
                            predecessors[v] = Some(u);
                            updated = true;
                        }
                    }
                    AlgorithmicSemiring::MaxMul => {
                        let candidate = current[u] * w;
                        if candidate > distances[v] + 1e-12 {
                            distances[v] = candidate;
                            predecessors[v] = Some(u);
                            updated = true;
                        }
                    }
                    AlgorithmicSemiring::MaxMin => {
                        let candidate = current[u].min(w);
                        if candidate > distances[v] + 1e-12 {
                            distances[v] = candidate;
                            predecessors[v] = Some(u);
                            updated = true;
                        }
                    }
                }
            }

            if !updated {
                break;
            }
        }

        Ok((distances, predecessors))
    }

    /// Reconstructs the shortest/optimal path from `source` to `target`.
    pub fn reconstruct_path(
        &self,
        source: usize,
        target: usize,
        predecessors: &[Option<usize>],
    ) -> Result<Vec<usize>> {
        // `target` and stored predecessors are plain usizes: both indexed
        // the slice out of bounds (panic) instead of erroring.
        if source >= predecessors.len() || target >= predecessors.len() {
            return Err(Error::Msg(format!(
                "Path nodes ({source}, {target}) exceed predecessor len {}",
                predecessors.len()
            )));
        }
        if source == target {
            return Ok(vec![source]);
        }
        let mut path = Vec::new();
        let mut curr = target;
        path.push(curr);

        while curr != source {
            match predecessors[curr] {
                Some(p) => {
                    if p >= predecessors.len() {
                        return Err(Error::Msg(format!(
                            "Predecessor {p} exceeds predecessor len {}",
                            predecessors.len()
                        )));
                    }
                    curr = p;
                    path.push(curr);
                    if path.len() > predecessors.len() {
                        return Err(Error::Msg(
                            "Cycle detected in predecessor path reconstruction".into(),
                        ));
                    }
                }
                None => {
                    return Err(Error::Msg(format!(
                        "Target node {target} is unreachable from source {source}"
                    )));
                }
            }
        }

        path.reverse();
        Ok(path)
    }
}

/// Pointer Graph Network (PGN) for neural algorithmic reasoning.
#[derive(Debug, Clone)]
pub struct PointerGraphNetwork {
    pub hidden_dim: usize,
    /// Projection weights for Query and Key pointer attention [hidden_dim, hidden_dim].
    w_q: Vec<f64>,
    w_k: Vec<f64>,
}

impl PointerGraphNetwork {
    pub fn new(hidden_dim: usize, seed: Option<u64>) -> Result<Self> {
        crate::limits::check_elements("PointerGraphNetwork weights", &[hidden_dim, hidden_dim])?;
        let mut s = seed.unwrap_or(2026);
        let mut next_val = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (u32::MAX as f64) * 0.2 - 0.1
        };

        let size = hidden_dim * hidden_dim;
        let mut w_q = crate::layout::try_with_capacity(size)?;
        let mut w_k = crate::layout::try_with_capacity(size)?;
        for _ in 0..size {
            w_q.push(next_val());
            w_k.push(next_val());
        }

        Ok(Self {
            hidden_dim,
            w_q,
            w_k,
        })
    }

    /// Forward message-passing step that updates node embeddings and calculates pointer attention distributions.
    ///
    /// For each node $v$, computes attention over incoming neighbors $u \in \mathcal{N}_{\text{in}}(v) \cup \{v\}$:
    /// $$A_{vu} = \text{softmax}\left(\frac{(W_q h_v)^T (W_k h_u)}{\sqrt{d}}\right)$$
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        node_states: &Tensor,
        graph: &AlgorithmicGraph,
    ) -> Result<(Tensor, Vec<Vec<(usize, f64)>>)> {
        let shape = node_states.shape();
        if shape.len() != 2 || shape[0] != graph.num_nodes || shape[1] != self.hidden_dim {
            return Err(Error::Msg(format!(
                "node_states shape {:?} does not match [{}, {}]",
                shape, graph.num_nodes, self.hidden_dim
            )));
        }

        let n = graph.num_nodes;
        let d = self.hidden_dim;
        // Same pub-edges tamper as NBF `solve`: bad endpoints sliced the
        // key/value rows out of bounds below.
        for e in &graph.edges {
            if e.source >= n || e.target >= n {
                return Err(Error::Msg(format!(
                    "Edge nodes ({}, {}) exceed graph num_nodes {n}",
                    e.source, e.target
                )));
            }
        }
        let h_data = node_states.to_vec::<f64>()?;

        // 1. Compute Projected Queries Q = H * W_q and Keys K = H * W_k
        let mut q_data = vec![0.0f64; n * d];
        let mut k_data = vec![0.0f64; n * d];
        for i in 0..n {
            let h_row = &h_data[i * d..(i + 1) * d];
            for j in 0..d {
                let mut q_sum = 0.0;
                let mut k_sum = 0.0;
                for (k, &h_val) in h_row.iter().enumerate().take(d) {
                    q_sum += h_val * self.w_q[k * d + j];
                    k_sum += h_val * self.w_k[k * d + j];
                }
                q_data[i * d + j] = q_sum;
                k_data[i * d + j] = k_sum;
            }
        }

        let scale = (d as f64).sqrt();
        let mut pointer_distributions = Vec::with_capacity(n);
        let mut updated_states = vec![0.0f64; n * d];

        for v in 0..n {
            let incoming = graph.incoming_edges(v);
            let mut candidates: Vec<usize> = incoming.iter().map(|e| e.source).collect();
            // Add self-loop candidate
            if !candidates.contains(&v) {
                candidates.push(v);
            }

            let q_v = &q_data[v * d..(v + 1) * d];
            let mut raw_scores = Vec::with_capacity(candidates.len());

            for &u in &candidates {
                let k_u = &k_data[u * d..(u + 1) * d];
                let dot: f64 = q_v.iter().zip(k_u.iter()).map(|(&a, &b)| a * b).sum();
                raw_scores.push(dot / scale);
            }

            // Softmax over candidate set
            let max_score = raw_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = raw_scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|&e| e / (sum_exp + 1e-12)).collect();

            let node_pointers: Vec<(usize, f64)> = candidates
                .iter()
                .cloned()
                .zip(probs.iter().cloned())
                .collect();
            pointer_distributions.push(node_pointers);

            // Aggregate neighbor embeddings weighted by pointer probabilities
            for j in 0..d {
                let mut agg = 0.0;
                for (idx, &u) in candidates.iter().enumerate() {
                    agg += probs[idx] * h_data[u * d + j];
                }
                // Residual update with tanh non-linearity
                updated_states[v * d + j] = (h_data[v * d + j] + agg).tanh();
            }
        }

        let updated_tensor = Tensor::from_vec(updated_states, &[n, d])?;
        Ok((updated_tensor, pointer_distributions))
    }

    /// Selects the highest probability predecessor pointer for each node.
    pub fn argmax_pointers(&self, pointer_distributions: &[Vec<(usize, f64)>]) -> Vec<usize> {
        pointer_distributions
            .iter()
            .map(|dist| {
                dist.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(u, _)| *u)
                    .unwrap_or(0)
            })
            .collect()
    }
}
