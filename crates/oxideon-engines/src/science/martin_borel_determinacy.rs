//! Martin Borel Determinacy Theorem & Gale-Stewart Infinite Zero-Sum Games on Baire Space $\omega^\omega$ (Gale & Stewart, 1953; Martin, 1975).
//!
//! ### 1. Backward Induction & Minimax Value on Finite-Horizon Unfoldings
//! By Martin's theorem, every Borel game $A \subseteq \omega^\omega$ is determined (Player I or Player II has a winning strategy).
//! For finite-horizon clopen approximations of depth $2k$ on binary trees, the game value $V \in \{-1, +1\}$ is computed via alternating $\max / \min$.

use crate::error::{Error, Result};

/// Martin Borel Determinacy & Gale-Stewart Tree Game Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct MartinBorelDeterminacyEngine;

impl MartinBorelDeterminacyEngine {
    /// Creates a new Martin Borel Determinacy engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the determined winner ($+1$ for Player I, $-1$ for Player II) of a binary Gale-Stewart clopen game with $2^d$ leaf payoffs.
    pub fn evaluate_binary_game_tree(leaf_payoffs: &[i32]) -> Result<i32> {
        let n = leaf_payoffs.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(Error::Msg(
                "leaf_payoffs length must be a power of 2".to_string(),
            ));
        }
        let depth = n.trailing_zeros() as usize;
        let mut layer = leaf_payoffs.to_vec();
        for lvl in (0..depth).rev() {
            let is_player_one = lvl % 2 == 0;
            let mut next = Vec::with_capacity(layer.len() / 2);
            for pair in layer.chunks_exact(2) {
                if is_player_one {
                    next.push(pair[0].max(pair[1]));
                } else {
                    next.push(pair[0].min(pair[1]));
                }
            }
            layer = next;
        }
        Ok(layer[0])
    }
}
