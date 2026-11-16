//! Monte Carlo Tree Search (`MCTS`) for Autonomous LLM Reasoning & Deep Thinking.
//!
//! Explores multi-step mathematical and algorithmic reasoning trajectories using
//! Upper Confidence bound applied to Trees (`UCT` / `PUCT`, Silver et al. / DeepSeek-R1):
//! $$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

use std::collections::HashMap;

/// A node in the reasoning search tree.
#[derive(Debug, Clone)]
pub struct MCTSNode {
    pub text: String,
    pub visit_count: usize,
    pub total_value: f64,
    pub prior_prob: f64,
    pub children: Vec<usize>,
    pub is_terminal: bool,
    pub reward: f64,
}

impl MCTSNode {
    pub fn new(text: String, prior_prob: f64) -> Self {
        Self {
            text,
            visit_count: 0,
            total_value: 0.0,
            prior_prob,
            children: Vec::new(),
            is_terminal: false,
            reward: 0.0,
        }
    }

    /// Average estimated action value $Q = W / N$.
    pub fn q_value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.total_value / (self.visit_count as f64)
        }
    }
}

/// Statistics and best trajectory found by MCTS search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_text: String,
    pub best_value: f64,
    pub total_nodes: usize,
    pub visit_counts: HashMap<String, usize>,
}

/// Monte Carlo Tree Search Engine for test-time compute scaling.
pub struct MCTSSearchEngine {
    pub c_puct: f64,
    pub num_simulations: usize,
    pub max_depth: usize,
}

impl MCTSSearchEngine {
    pub fn new(c_puct: f64, num_simulations: usize, max_depth: usize) -> Self {
        Self {
            c_puct,
            num_simulations,
            max_depth,
        }
    }

    /// Selects child node with highest PUCT score:
    /// $$\operatorname{PUCT}(c) = Q(c) + c_{\text{puct}} P(c) \frac{\sqrt{N_{\text{parent}}}}{1 + N_c}$$
    fn select_best_child(&self, nodes: &[MCTSNode], parent_idx: usize) -> Option<usize> {
        let parent = &nodes[parent_idx];
        if parent.children.is_empty() {
            return None;
        }

        let parent_n = (parent.visit_count.max(1) as f64).sqrt();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_child = None;

        for &child_idx in &parent.children {
            let child = &nodes[child_idx];
            let q = child.q_value();
            let u = self.c_puct * child.prior_prob * (parent_n / (1.0 + child.visit_count as f64));
            let score = q + u;
            if score > best_score {
                best_score = score;
                best_child = Some(child_idx);
            }
        }

        best_child
    }

    /// Executes MCTS search over candidate reasoning steps:
    /// - `expand_fn`: given node text, generates vector of candidate next reasoning steps with prior probabilities: `(step_text, prior)`
    /// - `eval_fn`: evaluates scalar value $V \in [0, 1]$ of a leaf node or terminal verification
    pub fn search<E, V>(
        &self,
        root_text: &str,
        mut expand_fn: E,
        mut eval_fn: V,
    ) -> crate::error::Result<SearchResult>
    where
        E: FnMut(&str) -> Vec<(String, f64)>,
        V: FnMut(&str) -> f64,
    {
        // Each simulation walks/expands the tree; an uncapped budget never
        // terminates (there is no early exit).
        crate::limits::check_iterations("num_simulations", self.num_simulations)?;
        let mut nodes = vec![MCTSNode::new(root_text.to_string(), 1.0)];

        for _sim in 0..self.num_simulations {
            let mut curr = 0usize;
            let mut path = vec![curr];
            let mut depth = 0;

            // 1. Selection: traverse down the tree using PUCT
            while !nodes[curr].children.is_empty() && depth < self.max_depth {
                if let Some(next_node) = self.select_best_child(&nodes, curr) {
                    curr = next_node;
                    path.push(curr);
                    depth += 1;
                } else {
                    break;
                }
            }

            // 2. Expansion: expand current leaf node if non-terminal
            if !nodes[curr].is_terminal && depth < self.max_depth {
                let candidates = expand_fn(&nodes[curr].text);
                if candidates.is_empty() {
                    nodes[curr].is_terminal = true;
                } else {
                    let mut child_indices = Vec::with_capacity(candidates.len());
                    for (step_text, prior) in candidates {
                        let child_idx = nodes.len();
                        nodes.push(MCTSNode::new(step_text, prior));
                        child_indices.push(child_idx);
                    }
                    nodes[curr].children = child_indices;
                    if let Some(&first_child) = nodes[curr].children.first() {
                        curr = first_child;
                        path.push(curr);
                    }
                }
            }

            // 3. Evaluation: evaluate leaf node
            let leaf_val = eval_fn(&nodes[curr].text);

            // 4. Backpropagation: backup visits and values along path
            for &node_idx in &path {
                nodes[node_idx].visit_count += 1;
                nodes[node_idx].total_value += leaf_val;
            }
        }

        // Determine best trajectory from root
        let root = &nodes[0];
        let mut best_child_text = root.text.clone();
        let mut best_v = f64::NEG_INFINITY;
        let mut visit_map = HashMap::new();

        for &c_idx in &root.children {
            let child = &nodes[c_idx];
            visit_map.insert(child.text.clone(), child.visit_count);
            let val = child.q_value();
            if val > best_v {
                best_v = val;
                best_child_text = child.text.clone();
            }
        }

        Ok(SearchResult {
            best_text: best_child_text,
            best_value: best_v.max(0.0),
            total_nodes: nodes.len(),
            visit_counts: visit_map,
        })
    }
}
