//! Vidal Multi-Scale Entanglement Renormalization Ansatz (MERA) & Causal Cone Conformal Dimensions (Vidal, 2007; Swingle, 2012).
//!
//! ### 1. MERA Holographic Causal Cone & Ryu-Takayanagi Area Law
//! In a 1D scale-invariant MERA tensor network of bond dimension $\chi \ge 2$ and branching ratio $b = 2$,
//! cutting a boundary interval of length $L = 2^k$ severs $2 k$ bonds along the minimal causal cone wedge,
//! reproducing the critical CFT entanglement entropy:
//! $$S(L) \le 2 \log_2(L) \ln(\chi)$$
//! Scaling operators $\mathcal{O}_\alpha$ are eigenstates of the ascending superoperator $\mathcal{A}(\mathcal{O}_\alpha) = \lambda_\alpha \mathcal{O}_\alpha$
//! with conformal scaling dimension $\Delta_\alpha = -\log_2 |\lambda_\alpha|$.

use crate::error::{Error, Result};

/// Vidal Multi-Scale Entanglement Renormalization Ansatz (MERA) Engine.
#[derive(Debug, Clone, Copy)]
pub struct MeraEntanglementEngine {
    /// Bond dimension $\chi \ge 2$.
    pub bond_dim_chi: usize,
}

impl MeraEntanglementEngine {
    /// Creates a new MERA tensor network engine with bond dimension $\chi \ge 2$.
    pub fn new(bond_dim_chi: usize) -> Result<Self> {
        if bond_dim_chi < 2 {
            return Err(Error::Msg("bond_dim_chi must be >= 2".to_string()));
        }
        Ok(Self { bond_dim_chi })
    }

    /// Computes the upper bound on entanglement entropy $S(L) = 2 \log_2(L) \ln \chi$ across a causal cone of width $L \ge 2$.
    pub fn causal_cone_entanglement_bound(&self, interval_l: f64) -> Result<f64> {
        if interval_l < 2.0 {
            return Err(Error::Msg("interval_l must be >= 2.0".to_string()));
        }
        Ok(2.0 * interval_l.log2() * (self.bond_dim_chi as f64).ln())
    }

    /// Extracts the CFT scaling dimension $\Delta = -\log_b |\lambda|$ from an ascending superoperator eigenvalue $\lambda \in (0, 1]$.
    pub fn scaling_dimension_from_eigenvalue(lambda: f64, branching_b: f64) -> Result<f64> {
        if lambda <= 0.0 || lambda > 1.0 || branching_b <= 1.0 {
            return Err(Error::Msg(
                "lambda must be in (0, 1] and branching_b > 1".to_string(),
            ));
        }
        Ok(-lambda.ln() / branching_b.ln())
    }
}
