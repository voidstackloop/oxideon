//! Fomin-Zelevinsky Cluster Algebras, Quiver Mutations $\mu_k$ & the Laurent Phenomenon (Fomin & Zelevinsky, 2002).
//!
//! ### 1. Skew-Symmetric Exchange Matrix Mutation $\mu_k(B)$
//! Given a quiver (skew-symmetric $n \times n$ integer matrix $B_{ij} = -B_{ji}$), mutating at vertex $k$ produces
//! $B' = \mu_k(B)$ where:
//! $$B'_{ij} = \begin{cases} -B_{ij} & \text{if } i = k \text{ or } j = k \\ B_{ij} + \frac{|B_{ik}| B_{kj} + B_{ik} |B_{kj}|}{2} & \text{otherwise} \end{cases}$$
//! Mutation is an involution: $\mu_k(\mu_k(B)) = B$.
//!
//! ### 2. Cluster Variable Exchange & A2 Pentagon Periodicity
//! Cluster variables $x = (x_1, \dots, x_n)$ mutate via $x'_k = \frac{\prod_{B_{ik}>0} x_i^{B_{ik}} + \prod_{B_{ik}<0} x_i^{-B_{ik}}}{x_k}$.
//! For the $A_2$ quiver $B = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$, alternating mutations $\mu_1, \mu_2, \mu_1, \mu_2, \mu_1$
//! exhibit exact period-5 **Pentagon Periodicity** and the **Laurent Phenomenon** (every cluster variable is a Laurent polynomial!).

use crate::error::{Error, Result};

/// Fomin-Zelevinsky Cluster Algebra & Quiver Mutation Engine.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterAlgebraQuiverEngine {
    /// Rank $n$ of the cluster seed.
    pub rank: usize,
    /// Skew-symmetric $n \times n$ exchange matrix $B$ (flattened).
    pub exchange_matrix: Vec<i64>,
    /// Current cluster variables $(x_1, \dots, x_n)$.
    pub cluster_vars: Vec<f64>,
}

impl ClusterAlgebraQuiverEngine {
    /// Constructs a new cluster seed $(x, B)$ and verifies skew-symmetry $B_{ij} = -B_{ji}$.
    pub fn new(rank: usize, exchange_matrix: Vec<i64>, cluster_vars: Vec<f64>) -> Result<Self> {
        if rank == 0 {
            return Err(Error::Msg(
                "ClusterAlgebraQuiverEngine requires rank >= 1".to_string(),
            ));
        }
        let expect = rank
            .checked_mul(rank)
            .ok_or_else(|| Error::Msg(format!("rank^2 overflows usize for rank={rank}")))?;
        if exchange_matrix.len() != expect || cluster_vars.len() != rank {
            return Err(Error::Msg("Invalid cluster seed dimensions".to_string()));
        }
        for i in 0..rank {
            for j in 0..rank {
                if exchange_matrix[i * rank + j] != -exchange_matrix[j * rank + i] {
                    return Err(Error::Msg(
                        "Exchange matrix B must be skew-symmetric".to_string(),
                    ));
                }
            }
        }
        Ok(Self {
            rank,
            exchange_matrix,
            cluster_vars,
        })
    }

    /// Applies Fomin-Zelevinsky quiver & cluster mutation $\mu_k$ at index $k \in 0..n$.
    pub fn mutate(&mut self, k: usize) -> Result<()> {
        let n = self.rank;
        if k >= n {
            return Err(Error::Msg("Mutation vertex k out of bounds".to_string()));
        }
        let mut prod_pos = 1.0;
        let mut prod_neg = 1.0;
        for i in 0..n {
            let bik = self.exchange_matrix[i * n + k];
            if bik > 0 {
                prod_pos *= self.cluster_vars[i].powi(bik as i32);
            } else if bik < 0 {
                prod_neg *= self.cluster_vars[i].powi((-bik) as i32);
            }
        }
        if self.cluster_vars[k] == 0.0 {
            return Err(Error::Msg(format!(
                "Cluster mutation: x[{k}] is zero, exchange ratio undefined"
            )));
        }
        let exchanged = (prod_pos + prod_neg) / self.cluster_vars[k];
        if !exchanged.is_finite() {
            return Err(Error::Msg(format!(
                "Cluster mutation: x[{k}] exchange overflowed to {exchanged}"
            )));
        }
        self.cluster_vars[k] = exchanged;

        let mut next_b = self.exchange_matrix.clone();
        for i in 0..n {
            for j in 0..n {
                if i == k || j == k {
                    next_b[i * n + j] = -self.exchange_matrix[i * n + j];
                } else {
                    let bik = self.exchange_matrix[i * n + k];
                    let bkj = self.exchange_matrix[k * n + j];
                    let entry = (|| {
                        let term = bik
                            .checked_abs()?
                            .checked_mul(bkj)?
                            .checked_add(bik.checked_mul(bkj.checked_abs()?)?)?;
                        self.exchange_matrix[i * n + j].checked_add(term / 2)
                    })()
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "Cluster mutation: exchange entry ({i}, {j}) overflows i64"
                        ))
                    })?;
                    next_b[i * n + j] = entry;
                }
            }
        }
        self.exchange_matrix = next_b;
        Ok(())
    }
}
