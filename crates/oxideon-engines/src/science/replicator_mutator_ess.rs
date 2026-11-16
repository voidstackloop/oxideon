//! Evolutionary Game Theory, Maynard Smith ESS & Taylor-Jonker Replicator Dynamics (Maynard Smith & Price, 1973; Taylor & Jonker, 1978).
//!
//! ### 1. Replicator Equation & Hawk-Dove Mixed ESS
//! Under payoff matrix $A$, strategy frequencies $x_i$ evolve via $\dot{x}_i = x_i ((A x)_i - x^T A x)$.
//! For the Hawk-Dove game with resource value $V$ and conflict cost $C > V$, the unique mixed Evolutionarily Stable Strategy (ESS) Hawk frequency is $p^* = V / C$.

use crate::error::{Error, Result};

/// Replicator-Mutator Evolutionary Game & Maynard Smith ESS Engine.
#[derive(Debug, Clone, Copy)]
pub struct ReplicatorMutatorGameEngine {
    /// Resource value $V > 0$.
    pub resource_v: f64,
    /// Conflict injury cost $C > 0$.
    pub conflict_cost_c: f64,
}

impl Default for ReplicatorMutatorGameEngine {
    fn default() -> Self {
        Self {
            resource_v: 2.0,
            conflict_cost_c: 4.0,
        }
    }
}

impl ReplicatorMutatorGameEngine {
    /// Creates a new Replicator-Mutator Game engine.
    pub fn new(resource_v: f64, conflict_cost_c: f64) -> Result<Self> {
        if resource_v <= 0.0 || conflict_cost_c <= 0.0 {
            return Err(Error::Msg(
                "resource_v and conflict_cost_c must be > 0".to_string(),
            ));
        }
        Ok(Self {
            resource_v,
            conflict_cost_c,
        })
    }

    /// Evaluates the Hawk-Dove equilibrium ESS Hawk fraction $p^* = \min(1, V/C)$ and replicator velocity $\dot{p}$ at current Hawk fraction $p \in [0, 1]$.
    pub fn hawk_dove_ess_and_velocity(&self, hawk_fraction_p: f64) -> Result<(f64, f64)> {
        if !(0.0..=1.0).contains(&hawk_fraction_p) {
            return Err(Error::Msg("hawk_fraction_p must be in [0, 1]".to_string()));
        }
        let p_ess = (self.resource_v / self.conflict_cost_c).min(1.0);
        // Payoff difference W_H - W_D = 0.5 * (V - C * p)
        let dp_dt = hawk_fraction_p
            * (1.0 - hawk_fraction_p)
            * 0.5
            * (self.resource_v - self.conflict_cost_c * hawk_fraction_p);
        Ok((p_ess, dp_dt))
    }
}
