//! Optimal Transport on Riemannian Graphs, Wasserstein-2 Geodesic Flows & McCann Displacement Interpolation.
//!
//! ### 1. 1D & Chain-Graph Exact Wasserstein-2 Metric $W_2(\mu_0, \mu_1)$
//! For discrete probability measures $\mu_0, \mu_1$ supported on a metric path graph $x_0 < x_1 < \dots < x_{n-1}$
//! with cumulative distribution functions $F_0, F_1$, the quadratic Wasserstein distance and **Benamou-Brenier Kinetic Action**
//! coincide via the monotone optimal transport map:
//! $$W_2^2(\mu_0, \mu_1) = \inf_{\rho_t, v_t} \int_0^1 \int |v_t(x)|^2 d\rho_t(x) dt$$
//!
//! ### 2. McCann Geodesic Displacement Interpolation $\mu_t = ((1-t)\mathrm{Id} + t T_\#)\mu_0$
//! Along a constant-speed Wasserstein geodesic $t \in [0, 1]$, the metric satisfies exact linear scaling:
//! $$W_1(\mu_0, \mu_t) = t \, W_1(\mu_0, \mu_1)$$

use crate::error::{Error, Result};

/// Riemannian Graph Wasserstein Geodesic & Benamou-Brenier Optimal Transport Engine.
#[derive(Debug, Clone)]
pub struct GraphWassersteinGeodesicEngine {
    /// Number of graph vertices $N$.
    pub num_nodes: usize,
    /// Grid spacing $\Delta x > 0$.
    pub dx: f64,
}

impl GraphWassersteinGeodesicEngine {
    /// Creates a new graph Wasserstein geodesic engine on $N$ nodes with spacing `dx`.
    pub fn new(num_nodes: usize, dx: f64) -> Result<Self> {
        if num_nodes < 2 || dx <= 0.0 {
            return Err(Error::Msg("num_nodes must be >= 2 and dx > 0".to_string()));
        }
        Ok(Self { num_nodes, dx })
    }

    /// Computes the exact 1-Wasserstein (Earth Mover's / Kantorovich-Rubinstein) distance
    /// $W_1(\mu_0, \mu_1) = \Delta x \sum_{i=0}^{N-1} |F_0(i) - F_1(i)|$.
    pub fn wasserstein_1_distance(&self, mu0: &[f64], mu1: &[f64]) -> Result<f64> {
        if mu0.len() != self.num_nodes || mu1.len() != self.num_nodes {
            return Err(Error::Msg(
                "Distribution length mismatch in wasserstein_1_distance".to_string(),
            ));
        }
        let mut cdf0 = 0.0;
        let mut cdf1 = 0.0;
        let mut w1 = 0.0;
        for (&p0, &p1) in mu0.iter().zip(mu1.iter()) {
            cdf0 += p0;
            cdf1 += p1;
            w1 += (cdf0 - cdf1).abs() * self.dx;
        }
        Ok(w1)
    }

    /// Computes the quantile-interpolated Wasserstein displacement geodesic $\mu_t$ at $t \in [0, 1]$
    /// by inverting the convex combination of cumulative distribution functions $F_t^{-1}(u) = (1-t)F_0^{-1}(u) + t F_1^{-1}(u)$.
    pub fn displacement_interpolate(
        &self,
        mu0: &[f64],
        mu1: &[f64],
        t: f64,
    ) -> Result<(Vec<f64>, f64)> {
        let n = self.num_nodes;
        if n == 0 {
            return Err(Error::Msg(
                "GraphWassersteinGeodesicEngine needs num_nodes >= 1".into(),
            ));
        }
        if mu0.len() != n || mu1.len() != n || !(0.0..=1.0).contains(&t) {
            return Err(Error::Msg(
                "Invalid inputs to displacement_interpolate".to_string(),
            ));
        }
        let num_quantiles = 200;
        let mut cdf0 = vec![0.0; n];
        let mut cdf1 = vec![0.0; n];
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        for i in 0..n {
            s0 += mu0[i];
            s1 += mu1[i];
            cdf0[i] = s0;
            cdf1[i] = s1;
        }

        let mut mu_t = vec![0.0; n];
        let mut w2_sq = 0.0;
        let dq = 1.0 / (num_quantiles as f64);

        for q_idx in 0..num_quantiles {
            let u = (q_idx as f64 + 0.5) * dq;
            let i0 = cdf0.partition_point(|&c| c < u).min(n - 1);
            let i1 = cdf1.partition_point(|&c| c < u).min(n - 1);
            let x0 = (i0 as f64) * self.dx;
            let x1 = (i1 as f64) * self.dx;
            w2_sq += (x1 - x0).powi(2) * dq;

            let xt = (1.0 - t) * x0 + t * x1;
            let node_float = (xt / self.dx).clamp(0.0, (n - 1) as f64);
            let lo = node_float.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = node_float - (lo as f64);
            mu_t[lo] += (1.0 - frac) * dq;
            mu_t[hi] += frac * dq;
        }

        Ok((mu_t, w2_sq.sqrt()))
    }
}
