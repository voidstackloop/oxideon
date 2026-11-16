//! Medusa & EAGLE Tree-Attention Speculative Decoding Engine.
//!
//! Enables parallel multi-draft tree verification: multiple Medusa heads generate a tree
//! of speculative future token candidates (e.g. 1 -> 4 -> 16 paths). The base model verifies
//! all candidate paths simultaneously in a single forward pass using a 2D Tree-Attention Mask.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Constructs 2D causal Tree-Attention mask given parent indices of candidate tree nodes.
/// Node `i` can attend to node `j` if and only if `j` is an ancestor of `i` in the tree.
#[derive(Debug, Clone)]
pub struct TreeAttentionMask;

impl TreeAttentionMask {
    pub fn build(parents: &[usize]) -> Result<Tensor> {
        let n = parents.len();
        // Refuse the n*n mask up front: 50k nodes would otherwise transiently
        // allocate 10 GB (and larger inputs abort the process) before from_vec
        // gets a chance to reject the shape.
        crate::limits::check_elements("tree mask", &[n, n])?;
        let mut mask = vec![0.0f32; n * n];

        for i in 0..n {
            // Self-attention
            mask[i * n + i] = 1.0;
            // Trace ancestors back to root, rejecting out-of-range parents
            // and parent cycles (which would hang this loop forever).
            let mut curr = i;
            let mut seen = vec![false; n];
            seen[i] = true;
            while curr != parents[curr] {
                curr = parents[curr];
                if curr >= n {
                    return Err(crate::error::Error::Msg(format!(
                        "TreeAttentionMask: parent index {curr} out of range ({n} nodes)"
                    )));
                }
                if seen[curr] {
                    return Err(crate::error::Error::Msg(
                        "TreeAttentionMask: parent pointers contain a cycle".into(),
                    ));
                }
                seen[curr] = true;
                mask[i * n + curr] = 1.0;
            }
        }

        Tensor::from_vec(mask, &[1, 1, n, n])
    }
}

/// Medusa multi-head draft token generator.
pub struct MedusaHeads {
    pub heads: Vec<Linear>,
    pub num_heads: usize,
    pub topk: usize,
}

impl MedusaHeads {
    pub fn new(
        hidden_dim: usize,
        vocab_size: usize,
        num_heads: usize,
        topk: usize,
    ) -> Result<Self> {
        crate::limits::check_repeats("num_heads", num_heads)?;
        let mut heads = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            heads.push(Linear::new(hidden_dim, vocab_size)?);
        }
        Ok(Self {
            heads,
            num_heads,
            topk,
        })
    }

    /// Evaluates draft token logits across all Medusa heads.
    /// `h`: `[batch, 1, hidden_dim]`
    /// Returns vector of draft logits `[batch, vocab_size]`.
    pub fn forward(&self, h: &Tensor) -> Result<Vec<Tensor>> {
        let mut draft_logits = Vec::with_capacity(self.num_heads);
        for head in &self.heads {
            draft_logits.push(head.forward(h)?);
        }
        Ok(draft_logits)
    }
}

/// Tree-based Speculative Verifier.
#[derive(Debug, Clone)]
pub struct TreeSpeculativeDecoder;

impl TreeSpeculativeDecoder {
    /// Selects the longest accepted branch from candidate token tree.
    /// `verified_tokens`: ground truth next tokens accepted at each position.
    /// `parents`: parent pointers for tree topology.
    pub fn accept_longest_path(
        candidate_tokens: &[i64],
        ground_truth_tokens: &[i64],
        parents: &[usize],
    ) -> Result<(Vec<i64>, usize)> {
        let mut accepted_path = Vec::new();

        // Check if root candidate matches
        if candidate_tokens.is_empty() || ground_truth_tokens.is_empty() {
            return Ok((accepted_path, 0));
        }

        // Visited set: cyclic parent pointers (e.g. parents=[1,0]) would
        // otherwise loop forever, growing accepted_path without bound.
        let mut visited = vec![false; candidate_tokens.len().max(ground_truth_tokens.len())];
        let mut curr = 0;
        while curr < candidate_tokens.len() && curr < ground_truth_tokens.len() {
            if visited[curr] {
                return Err(Error::Msg(format!(
                    "accept_longest_path: parent pointers contain a cycle through node {curr}"
                )));
            }
            visited[curr] = true;
            if candidate_tokens[curr] == ground_truth_tokens[curr] {
                accepted_path.push(candidate_tokens[curr]);
                // Advance to child
                let child = parents
                    .iter()
                    .enumerate()
                    .position(|(idx, &p)| p == curr && idx != curr);
                if let Some(next_idx) = child {
                    curr = next_idx;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let count = accepted_path.len();
        Ok((accepted_path, count))
    }
}
