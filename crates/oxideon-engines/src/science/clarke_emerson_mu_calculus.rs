//! Clarke-Emerson-Sifakis CTL Model Checking & Knaster-Tarski $\mu$-Calculus Fixed Points (Clarke & Emerson, 1981; Queille & Sifakis, 1982).
//!
//! ### 1. Least Fixed Point $\mu Z. (P \lor \mathrm{EX}\, Z)$ for Reachability $\mathrm{EF}\, P$
//! On a finite Kripke structure $(S, R)$, the CTL reachability state set $\llbracket \mathrm{EF}\, P \rrbracket$ is the least fixed point
//! of the monotone predicate transformer $F(Z) = P \cup \{s \in S : \exists s' \in Z, (s, s') \in R\}$, converging in at most $|S|$ Tarski iterations.

use crate::error::{Error, Result};

/// Clarke-Emerson-Sifakis CTL Model Checker & $\mu$-Calculus Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClarkeEmersonModelChecker;

impl ClarkeEmersonModelChecker {
    /// Creates a new Clarke-Emerson Model Checker engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the CTL reachability least fixed point $\llbracket \mathrm{EF}\, P \rrbracket$ on an $n$-state bitmask transition matrix (`adj_masks[i]` has bit $j$ set iff $i \to j$).
    pub fn ctl_ef_least_fixed_point(
        num_states: usize,
        adj_masks: &[u64],
        target_mask_p: u64,
    ) -> Result<(u64, usize)> {
        if num_states == 0 || num_states > 64 || adj_masks.len() != num_states {
            return Err(Error::Msg(
                "1 <= num_states <= 64 and matching adj_masks required".to_string(),
            ));
        }
        let mut z = target_mask_p;
        let mut iters = 0usize;
        loop {
            iters += 1;
            let mut next_z = z;
            for (i, &edges) in adj_masks.iter().enumerate() {
                if (edges & z) != 0 {
                    next_z |= 1u64 << i;
                }
            }
            if next_z == z {
                break;
            }
            z = next_z;
        }
        Ok((z, iters))
    }
}
