//! Eigen Quasispecies Error Catastrophe & Sequence Space Master Equation (Eigen, 1971).
//!
//! ### 1. Error Catastrophe & Master Sequence Equilibrium Fraction
//! For a genome of length $L$, per-base replication fidelity $q \in (0, 1]$ ($Q = q^L$), master-sequence fitness $A_0 > 1$, and mutant background fitness $A_m = 1$:
//! $$x_0^* = \frac{A_0 q^L - 1}{A_0 - 1}, \qquad \mu_{\mathrm{crit}} = 1 - q_{\mathrm{crit}} \approx \frac{\ln A_0}{L}$$

use crate::error::{Error, Result};

/// Eigen Quasispecies Error Threshold & Master Sequence Survival Engine.
#[derive(Debug, Clone, Copy)]
pub struct EigenQuasispeciesThreshold {
    /// Genome length $L \ge 1$.
    pub genome_length_l: usize,
    /// Relative fitness advantage $A_0 > 1$ of the master sequence over background mutants.
    pub master_fitness_a0: f64,
}

impl Default for EigenQuasispeciesThreshold {
    fn default() -> Self {
        Self {
            genome_length_l: 100,
            master_fitness_a0: std::f64::consts::E,
        }
    }
}

impl EigenQuasispeciesThreshold {
    /// Creates a new Eigen Quasispecies engine.
    pub fn new(genome_length_l: usize, master_fitness_a0: f64) -> Result<Self> {
        if genome_length_l == 0 || master_fitness_a0 <= 1.0 || !master_fitness_a0.is_finite() {
            return Err(Error::Msg(
                "genome_length_l >= 1 and master_fitness_a0 > 1 required".to_string(),
            ));
        }
        Ok(Self {
            genome_length_l,
            master_fitness_a0,
        })
    }

    /// Evaluates the critical per-base error threshold $\mu_{\mathrm{crit}} = 1 - A_0^{-1/L}$ and steady-state master sequence fraction $x_0^*$ for per-base error rate $\mu$.
    pub fn critical_mutation_rate_and_master_fraction(
        &self,
        per_base_error_mu: f64,
    ) -> Result<(f64, f64, bool)> {
        if !(0.0..1.0).contains(&per_base_error_mu) {
            return Err(Error::Msg(
                "per_base_error_mu must be in [0, 1)".to_string(),
            ));
        }
        let l = self.genome_length_l as f64;
        let q_crit = self.master_fitness_a0.powf(-1.0 / l);
        let mu_crit = 1.0 - q_crit;
        let q_big = (1.0 - per_base_error_mu).powf(l);
        let x0 = ((self.master_fitness_a0 * q_big - 1.0) / (self.master_fitness_a0 - 1.0)).max(0.0);
        Ok((mu_crit, x0, x0 > 0.0))
    }
}
