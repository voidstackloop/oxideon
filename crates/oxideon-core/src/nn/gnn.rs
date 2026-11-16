//! Graph Neural Networks (`GCNConv` Kipf & Welling 2017, `GATConv` Veličković et al. 2018).

use crate::error::{Error, Result};
use crate::nn::{Linear, Module, init};
use crate::tensor::Tensor;

/// Computes symmetric normalized adjacency $\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$ for a 2D `[N, N]`
/// adjacency matrix `adj`, optionally adding self-loops ($\hat{A} = A + I$).
pub fn normalize_adjacency(adj: &Tensor, add_self_loops: bool) -> Result<Tensor> {
    if adj.ndim() != 2 || adj.shape()[0] != adj.shape()[1] {
        return Err(Error::Msg(format!(
            "normalize_adjacency expects square 2D [N, N] matrix, got {:?}",
            adj.shape()
        )));
    }
    let n = adj.shape()[0];
    let a_hat = if add_self_loops {
        let eye = Tensor::eye(n, adj.dtype(), &adj.device())?;
        adj.add(&eye)?
    } else {
        adj.clone()
    };
    let deg = a_hat.sum_axis(1, true)?.clamp(1e-12, f64::INFINITY)?;
    let deg_inv_sqrt = deg.rsqrt()?;
    // D^{-1/2} * A_hat * D^{-1/2}
    a_hat.mul(&deg_inv_sqrt)?.mul(&deg_inv_sqrt.t()?)
}

/// Spectral Graph Convolutional layer (`GCNConv`, Kipf & Welling 2017):
/// $$H^{(l+1)} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W + b$$
pub struct GCNConv {
    pub linear: Linear,
}

impl GCNConv {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        Ok(Self {
            linear: Linear::new(in_features, out_features)?,
        })
    }

    /// Applies the graph convolution using a raw or pre-normalized adjacency matrix `adj` (`[N, N]`).
    /// When `normalize = true`, `normalize_adjacency(adj, true)` is applied automatically.
    pub fn forward_graph(&self, x: &Tensor, adj: &Tensor, normalize: bool) -> Result<Tensor> {
        let a_norm = if normalize {
            normalize_adjacency(adj, true)?
        } else {
            adj.clone()
        };
        let xw = self.linear.forward(x)?;
        a_norm.matmul(&xw)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.linear.parameters()
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        self.linear.named_parameters()
    }
}

/// Graph Attention Network layer (`GATConv`, Veličković et al. 2018):
/// $$e_{ij} = \operatorname{LeakyReLU}\!\left(a_{\text{src}}^T W h_i + a_{\text{dst}}^T W h_j\right), \quad \alpha_{ij} = \operatorname{softmax}_{j \in \mathcal{N}(i)}(e_{ij}), \quad h'_i = \sum_{j} \alpha_{ij} W h_j + b$$
pub struct GATConv {
    pub proj: Linear,
    pub attn_src: Tensor,
    pub attn_dst: Tensor,
    pub bias: Tensor,
    pub out_features: usize,
    pub negative_slope: f64,
}

impl GATConv {
    pub fn new(in_features: usize, out_features: usize, negative_slope: f64) -> Result<Self> {
        // Sum in float: the usize add overflowed for absurd fans before Linear could refuse them.
        let bound = (6.0 / (in_features as f64 + out_features as f64) as f32).sqrt();
        let proj = Linear::with_bias(in_features, out_features, false)?;
        let attn_src = init::uniform(&[out_features, 1], -bound, bound)?.requires_grad_(true);
        let attn_dst = init::uniform(&[out_features, 1], -bound, bound)?.requires_grad_(true);
        let bias = Tensor::zeros_f32(&[1, out_features])?.requires_grad_(true);
        Ok(Self {
            proj,
            attn_src,
            attn_dst,
            bias,
            out_features,
            negative_slope,
        })
    }

    /// Computes masked graph attention over node features `x: [N, in_features]` and adjacency
    /// matrix `adj: [N, N]` (where `adj[i, j] > 0` indicates a directed edge $j \to i$,
    /// with self-loops automatically included).
    pub fn forward_graph(&self, x: &Tensor, adj: &Tensor) -> Result<Tensor> {
        if x.ndim() != 2 || adj.ndim() != 2 {
            return Err(Error::Msg(
                "GATConv::forward_graph expects 2D x [N, F] and 2D adj [N, N]".into(),
            ));
        }
        let n = x.shape()[0];
        let wh = self.proj.forward(x)?; // [N, F_out]

        // e_src: [N, 1], e_dst: [1, N] -> pairwise logits [N, N]
        let e_src = wh.matmul(&self.attn_src)?;
        let e_dst = wh.matmul(&self.attn_dst)?.t()?;
        let logits = e_src.add(&e_dst)?.leaky_relu(self.negative_slope)?;

        // Build non-edge boolean mask where (adj + I) <= 0
        let eye = Tensor::eye(n, adj.dtype(), &adj.device())?;
        let conn = adj.add(&eye)?;
        let zero = Tensor::zeros(&[n, n], adj.dtype(), &adj.device())?;
        let non_edge_mask = conn.le(&zero)?;

        let masked_logits = logits.masked_fill(&non_edge_mask, -1e9)?;
        let alpha = masked_logits.softmax(-1)?; // [N, N]

        alpha.matmul(&wh)?.add(&self.bias)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(_, t)| t)
            .collect()
    }

    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut out: Vec<(String, Tensor)> = self
            .proj
            .named_parameters()
            .into_iter()
            .map(|(k, v)| (format!("proj.{k}"), v))
            .collect();
        out.push(("attn_src".to_string(), self.attn_src.clone()));
        out.push(("attn_dst".to_string(), self.attn_dst.clone()));
        out.push(("bias".to_string(), self.bias.clone()));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gat_rejects_overflowing_fans() {
        let half = usize::MAX / 2 + 1;
        let err = match GATConv::new(half, half, 0.2) {
            Ok(_) => panic!("overflowing fans must be an error, not a panic"),
            Err(e) => e,
        };
        assert!(!err.to_string().is_empty());
    }
}
