//! Hyperbolic Graph Attention Network (HGAT) over Lorentz Manifolds.
//!
//! Implements non-Euclidean graph attention mechanisms over negative curvature geometries:
//! - Tangent-space projection via logarithmic map $\log_0^c(h_i)$.
//! - Curvature-aware attention weights combining linear feature concatenation and hyperbolic geodesic distance:
//!   $$e_{ij} = \operatorname{LeakyReLU}\left( \mathbf{a}^T [W \log_0^c(h_i) \,\|\, W \log_0^c(h_j)] - \beta d_{\mathbb{L}}(h_i, h_j) \right)$$
//! - Riemannian Fréchet aggregation in the tangent space followed by exponential mapping $\exp_0^c(z_i)$
//!   back to the forward hyperboloid sheet $\mathbb{L}^{d_{out}}$.

use crate::error::{Error, Result};
use crate::geom_graphs::LorentzManifold;
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Hyperbolic Graph Attention Layer on the Lorentz manifold.
pub struct HyperbolicGATLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub curvature: f64,
    pub beta: f64,
    pub proj: Linear,
    pub attn_weight: Tensor,
}

impl HyperbolicGATLayer {
    /// Creates a new `HyperbolicGATLayer`.
    /// - `in_dim`: Spatial dimension in tangent space (manifold dimension is `in_dim + 1`).
    /// - `out_dim`: Output spatial dimension in tangent space (output manifold dimension is `out_dim + 1`).
    /// - `curvature`: Hyperbolic curvature parameter $c > 0$.
    /// - `beta`: Distance penalty weight in attention logits.
    pub fn new(in_dim: usize, out_dim: usize, curvature: f64, beta: f64) -> Result<Self> {
        let proj = Linear::new(in_dim, out_dim)?;
        let bound = 1.0 / (2.0 * out_dim as f64).sqrt();
        let attn_weight = Tensor::randn(&[2 * out_dim, 1])?.mul_scalar(bound)?;

        Ok(Self {
            in_dim,
            out_dim,
            curvature: curvature.max(1e-4),
            beta,
            proj,
            attn_weight,
        })
    }

    /// Forward pass of the Hyperbolic Graph Attention Layer.
    /// - `x`: Node representations on the Lorentz manifold of shape `[num_nodes, in_dim + 1]`.
    /// - `edges`: Directed edges as pairs of node indices `(source, target)`.
    ///
    /// Returns updated node representations on the Lorentz manifold of shape `[num_nodes, out_dim + 1]`.
    pub fn forward(&self, x: &Tensor, edges: &[(usize, usize)]) -> Result<Tensor> {
        let want_in = self.in_dim.checked_add(1).ok_or_else(|| {
            crate::error::Error::Msg(format!("GAT in_dim {} overflows", self.in_dim))
        })?;
        if x.ndim() != 2 || x.shape()[1] != want_in {
            return Err(crate::error::Error::Msg(format!(
                "HyperbolicGATLayer expects [num_nodes, {}], got {:?}",
                want_in,
                x.shape()
            )));
        }
        let num_nodes = x.shape()[0];
        let c = self.curvature;

        // 1. Map to Euclidean tangent space at origin: [num_nodes, in_dim]
        let x_tangent = LorentzManifold::log_map_zero(x, c)?;

        // 2. Linear projection in tangent space: [num_nodes, out_dim]
        let h = self.proj.forward(&x_tangent)?;

        // 3. Compute attention logits for each edge
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];
        for &(src, dst) in edges {
            if src >= num_nodes || dst >= num_nodes {
                return Err(Error::IndexOutOfBounds {
                    op: "HyperbolicGATLayer::forward",
                    index: src.max(dst) as isize,
                    dim_size: num_nodes,
                });
            }
            adj[dst].push(src); // dst attends to src
        }

        // Add self-loops if not present
        for (i, neighbors) in adj.iter_mut().enumerate() {
            if !neighbors.contains(&i) {
                neighbors.push(i);
            }
        }

        let mut aggregated_tangent = vec![0.0f64; num_nodes * self.out_dim];

        // Process node by node
        let h_data = h.to_vec::<f64>()?;
        let x_data = x.to_vec::<f64>()?;
        let a_data = self.attn_weight.to_vec::<f64>()?;
        let in_cols = want_in;
        let out_cols = self.out_dim;
        // Column agreement (projection/attention vs tamperable out_dim).
        let h_need = num_nodes.checked_mul(out_cols).ok_or_else(|| {
            Error::Msg(format!("GAT state size {num_nodes} * {out_cols} overflows"))
        })?;
        if h_data.len() != h_need {
            return Err(Error::ShapeMismatch {
                op: "HyperbolicGATLayer::forward",
                expected: vec![num_nodes, out_cols],
                got: vec![h_data.len()],
            });
        }
        let a_need = out_cols
            .checked_mul(2)
            .ok_or_else(|| Error::Msg(format!("GAT attention size {out_cols} * 2 overflows")))?;
        if a_data.len() != a_need {
            return Err(Error::ShapeMismatch {
                op: "HyperbolicGATLayer::forward",
                expected: vec![a_need],
                got: vec![a_data.len()],
            });
        }

        for dst in 0..num_nodes {
            let neighbors = &adj[dst];
            let mut logits = Vec::with_capacity(neighbors.len());

            for &src in neighbors {
                // Compute linear attention score a^T [h_src || h_dst]
                let mut score = 0.0;
                for k in 0..out_cols {
                    let val_src = h_data[src * out_cols + k];
                    let val_dst = h_data[dst * out_cols + k];
                    score += val_src * a_data[k];
                    score += val_dst * a_data[out_cols + k];
                }

                // Hyperbolic distance penalty
                if self.beta.abs() > 1e-12 {
                    let mut inner = -x_data[src * in_cols] * x_data[dst * in_cols];
                    for k in 1..in_cols {
                        inner += x_data[src * in_cols + k] * x_data[dst * in_cols + k];
                    }
                    let scaled = (-c * inner).max(1.0);
                    let dist = (scaled + (scaled * scaled - 1.0).max(0.0).sqrt()).ln() / c.sqrt();
                    score -= self.beta * dist;
                }

                // LeakyReLU (alpha = 0.2)
                let val = if score >= 0.0 { score } else { 0.2 * score };
                logits.push(val);
            }

            // Softmax over incoming neighbors
            let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;
            let mut exps = Vec::with_capacity(logits.len());
            for &l in &logits {
                let e = (l - max_logit).exp();
                exps.push(e);
                sum_exp += e;
            }

            for (idx, &src) in neighbors.iter().enumerate() {
                let alpha = exps[idx] / sum_exp.max(1e-12);
                for k in 0..out_cols {
                    aggregated_tangent[dst * out_cols + k] += alpha * h_data[src * out_cols + k];
                }
            }
        }

        let z = Tensor::from_vec(aggregated_tangent, &[num_nodes, self.out_dim])?;

        // 4. Exponential map back to Lorentz manifold: [num_nodes, out_dim + 1]
        LorentzManifold::exp_map_zero(&z, c)
    }
}
