//! Gromov-Wasserstein Optimal Transport & Relational Alignment.
//!
//! Solves quadratic optimal transport between incomparable metric measure spaces $(X, d_X, \mu)$
//! and $(Y, d_Y, \nu)$ (Mémoli 2011, Peyré et al. 2016):
//! - Entropic-regularized Gromov-Wasserstein discrepancy:
//!   $$\text{GW}(C_1, C_2, p, q) = \min_{T \in \Pi(p, q)} \sum_{i,j,k,l} |C_1(i, k) - C_2(j, l)|^2 T_{ij} T_{kl} - \epsilon H(T)$$
//! - Mirror descent optimization with inner Bregman/Sinkhorn-Knopp scaling iterations.
//! - Invariant relational graph matching and unsupervised cross-space node alignment.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Configuration for Entropic Gromov-Wasserstein Solver.
#[derive(Debug, Clone)]
pub struct GromovWassersteinConfig {
    pub epsilon: f64,
    pub max_iters: usize,
    pub sinkhorn_iters: usize,
    pub tolerance: f64,
}

impl Default for GromovWassersteinConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.05,
            max_iters: 40,
            sinkhorn_iters: 50,
            tolerance: 1e-6,
        }
    }
}

/// Output of Gromov-Wasserstein alignment.
#[derive(Debug, Clone)]
pub struct GromovWassersteinResult {
    /// Final quadratic Gromov-Wasserstein distance.
    pub gw_distance: f64,
    /// Optimal transport coupling matrix $T^* \in \mathbb{R}^{n \times m}$.
    pub coupling: Tensor,
    /// Whether outer mirror descent converged.
    pub converged: bool,
    /// Total outer iterations performed.
    pub iterations: usize,
}

/// Gromov-Wasserstein Engine.
#[derive(Debug, Clone)]
pub struct GromovWassersteinEngine {
    pub config: GromovWassersteinConfig,
}

impl GromovWassersteinEngine {
    pub fn new(config: GromovWassersteinConfig) -> Self {
        Self { config }
    }

    /// Computes the Gromov-Wasserstein distance and coupling between intra-metric matrices $C_1 \in \mathbb{R}^{n \times n}$ and $C_2 \in \mathbb{R}^{m \times m}$.
    pub fn compute_gw(
        &self,
        c1: &Tensor,
        c2: &Tensor,
        p_weights: Option<&[f64]>,
        q_weights: Option<&[f64]>,
    ) -> Result<GromovWassersteinResult> {
        let s1 = c1.shape();
        let s2 = c2.shape();
        if s1.len() != 2 || s1[0] != s1[1] {
            return Err(Error::Msg(format!("c1 must be square, got {:?}", s1)));
        }
        if s2.len() != 2 || s2[0] != s2[1] {
            return Err(Error::Msg(format!("c2 must be square, got {:?}", s2)));
        }

        let n = s1[0];
        let m = s2[0];

        let p = match p_weights {
            Some(w) => {
                if w.len() != n {
                    return Err(Error::Msg(format!("p_weights length {} != n {n}", w.len())));
                }
                w.to_vec()
            }
            None => vec![1.0 / n as f64; n],
        };

        let q = match q_weights {
            Some(w) => {
                if w.len() != m {
                    return Err(Error::Msg(format!("q_weights length {} != m {m}", w.len())));
                }
                w.to_vec()
            }
            None => vec![1.0 / m as f64; m],
        };

        let c1_data = c1.to_vec::<f64>()?;
        let c2_data = c2.to_vec::<f64>()?;

        // Constant term components for L2 loss:
        // C1_sq = C1.^2, C2_sq = C2.^2
        // const_c1 = C1_sq * p
        // const_c2 = C2_sq * q
        let mut const_c1 = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                let val = c1_data[i * n + k];
                sum += val * val * p[k];
            }
            const_c1[i] = sum;
        }

        let mut const_c2 = vec![0.0f64; m];
        for j in 0..m {
            let mut sum = 0.0;
            for l in 0..m {
                let val = c2_data[j * m + l];
                sum += val * val * q[l];
            }
            const_c2[j] = sum;
        }

        // Initialize coupling with marginal product plus small symmetry-breaking perturbation
        let mut t_data = vec![0.0f64; n * m];
        let mut s = 2026u64;
        let mut sum_t = 0.0;
        for i in 0..n {
            for j in 0..m {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let rand_jitter = ((s >> 33) as f64) / (u32::MAX as f64) * 0.005;
                let val = p[i] * q[j] + rand_jitter;
                t_data[i * m + j] = val;
                sum_t += val;
            }
        }
        for val in &mut t_data {
            *val /= sum_t;
        }

        let mut converged = false;
        let mut iters = 0;
        let eps = self.config.epsilon.max(1e-5);

        for iter in 0..self.config.max_iters {
            iters = iter + 1;

            // Compute gradient cost matrix:
            // G_ij = const_c1_i + const_c2_j - 2 * (C1 * T * C2^T)_ij
            // 1. Temp = T * C2^T: [n, m] * [m, m]
            let mut temp = vec![0.0f64; n * m];
            for i in 0..n {
                for l in 0..m {
                    let mut sum = 0.0;
                    for j in 0..m {
                        sum += t_data[i * m + j] * c2_data[l * m + j]; // C2^T[j, l] = C2[l, j]
                    }
                    temp[i * m + l] = sum;
                }
            }

            // 2. C1 * Temp: [n, n] * [n, m]
            let mut cost = vec![0.0f64; n * m];
            for i in 0..n {
                for j in 0..m {
                    let mut tens_val = 0.0;
                    for k in 0..n {
                        tens_val += c1_data[i * n + k] * temp[k * m + j];
                    }
                    cost[i * m + j] = const_c1[i] + const_c2[j] - 2.0 * tens_val;
                }
            }

            // Sinkhorn-Knopp scaling for K = exp(-cost / eps)
            let mut u = vec![1.0f64 / n as f64; n];
            let mut v = vec![1.0f64 / m as f64; m];

            // Kernel K
            let mut k_mat = vec![0.0f64; n * m];
            let min_cost = cost.iter().cloned().fold(f64::INFINITY, f64::min);
            for idx in 0..n * m {
                k_mat[idx] = (-(cost[idx] - min_cost) / eps).exp();
            }

            for _ in 0..self.config.sinkhorn_iters {
                // u = p / (K * v)
                for i in 0..n {
                    let mut kv = 0.0;
                    for j in 0..m {
                        kv += k_mat[i * m + j] * v[j];
                    }
                    u[i] = p[i] / (kv + 1e-15);
                }
                // v = q / (K^T * u)
                for j in 0..m {
                    let mut ktu = 0.0;
                    for i in 0..n {
                        ktu += k_mat[i * m + j] * u[i];
                    }
                    v[j] = q[j] / (ktu + 1e-15);
                }
            }

            // Updated coupling T_new = diag(u) * K * diag(v)
            let mut t_new = vec![0.0f64; n * m];
            let mut max_change = 0.0f64;
            for i in 0..n {
                for j in 0..m {
                    let val = u[i] * k_mat[i * m + j] * v[j];
                    let diff = (val - t_data[i * m + j]).abs();
                    if diff > max_change {
                        max_change = diff;
                    }
                    t_new[i * m + j] = val;
                }
            }

            t_data = t_new;

            if max_change < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Final GW distance: <cost, T>
        // Recalculate final cost
        let mut temp = vec![0.0f64; n * m];
        for i in 0..n {
            for l in 0..m {
                let mut sum = 0.0;
                for j in 0..m {
                    sum += t_data[i * m + j] * c2_data[l * m + j];
                }
                temp[i * m + l] = sum;
            }
        }

        let mut gw_dist = 0.0;
        for i in 0..n {
            for j in 0..m {
                let mut tens_val = 0.0;
                for k in 0..n {
                    tens_val += c1_data[i * n + k] * temp[k * m + j];
                }
                let c_ij = const_c1[i] + const_c2[j] - 2.0 * tens_val;
                gw_dist += c_ij * t_data[i * m + j];
            }
        }

        let coupling_tensor = Tensor::from_vec(t_data, &[n, m])?;
        Ok(GromovWassersteinResult {
            gw_distance: gw_dist.max(0.0),
            coupling: coupling_tensor,
            converged,
            iterations: iters,
        })
    }

    /// Aligns two graphs given their adjacency/distance matrices by finding max-probability node matches.
    pub fn align_graphs(&self, adj1: &Tensor, adj2: &Tensor) -> Result<Vec<(usize, usize)>> {
        let res = self.compute_gw(adj1, adj2, None, None)?;
        let shape = res.coupling.shape();
        let n = shape[0];
        let m = shape[1];
        let t_data = res.coupling.to_vec::<f64>()?;

        let mut matches = Vec::with_capacity(n);
        for i in 0..n {
            let mut best_j = 0;
            let mut best_val = -1.0;
            for j in 0..m {
                let val = t_data[i * m + j];
                if val > best_val {
                    best_val = val;
                    best_j = j;
                }
            }
            matches.push((i, best_j));
        }

        Ok(matches)
    }
}
