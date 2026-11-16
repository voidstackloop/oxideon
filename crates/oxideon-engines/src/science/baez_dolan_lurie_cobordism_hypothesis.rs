//! Baez-Dolan-Lurie Cobordism Hypothesis & Fully Extended $n$-Dualizable TQFTs (Baez & Dolan, 1995; Lurie, 2009).
//!
//! ### 1. Classification of Framed Extended TQFTs $Z : \mathbf{Bord}_n^{\mathrm{fr}} \to \mathcal{C}$ by Fully Dualizable Objects
//! By Lurie's theorem, the $\infty$-groupoid of framed fully extended $n$-dimensional topological quantum field theories with values in a symmetric monoidal $(\infty, n)$-category $\mathcal{C}$ is equivalent to the core $\mathcal{C}^{\mathrm{fd}}_\sim$ of fully dualizable objects; for a finite gauge group $G$ (Dijkgraaf-Witten untwisted theory), $Z(T^n) = |\operatorname{Hom}(\pi_1(T^n), G) / G|$.

use crate::error::{Error, Result};

/// Baez-Dolan-Lurie Cobordism Hypothesis & Extended TQFT Engine.
#[derive(Debug, Clone, Copy)]
pub struct BaezDolanCobordismHypothesis {
    /// Spacetime cobordism dimension $n \ge 1$.
    pub cobordism_dim_n: u32,
    /// Abelian gauge group order $|A| \ge 1$.
    pub abelian_group_order: u64,
}

impl Default for BaezDolanCobordismHypothesis {
    fn default() -> Self {
        Self {
            cobordism_dim_n: 3,
            abelian_group_order: 2,
        }
    }
}

impl BaezDolanCobordismHypothesis {
    /// Creates a new Baez-Dolan-Lurie Cobordism Hypothesis engine.
    pub fn new(cobordism_dim_n: u32, abelian_group_order: u64) -> Result<Self> {
        if cobordism_dim_n == 0 || abelian_group_order == 0 {
            return Err(Error::Msg(
                "cobordism_dim_n and abelian_group_order must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            cobordism_dim_n,
            abelian_group_order,
        })
    }

    /// Evaluates the untwisted abelian Dijkgraaf-Witten partition function $Z(T^n) = |A|^{n-1}$ on the $n$-torus and $Z(S^n) = 1 / |A|$.
    pub fn torus_and_sphere_partition_invariants(&self) -> Result<(f64, f64)> {
        if self.cobordism_dim_n == 0 || self.abelian_group_order == 0 {
            return Err(Error::Msg(
                "cobordism_dim_n and abelian_group_order must be >= 1".to_string(),
            ));
        }
        let a = self.abelian_group_order as f64;
        let z_tn = a.powi((self.cobordism_dim_n - 1) as i32);
        let z_sn = 1.0 / a;
        Ok((z_tn, z_sn))
    }
}
