//! Deep Operator Networks (DeepONet) for Learning Nonlinear Operators.
//!
//! Based on Chen & Chen (1995) and Lu et al. (Nature Machine Intelligence 2021):
//! An operator $G: \mathcal{U} \to \mathcal{V}$ mapping an infinite-dimensional function $u \in \mathcal{U}$
//! to a continuous output field $G(u)(y)$ evaluated at continuous coordinates $y \in \mathbb{R}^d$:
//!
//! $$G(u)(y) = \sum_{k=1}^p b_k(u) \cdot t_k(y) + b_0$$
//!
//! - Branch Network: takes discrete sensors $[u(x_1), \dots, u(x_m)] \in \mathbb{R}^m$ and maps to basis weights $[b_1, \dots, b_p] \in \mathbb{R}^p$.
//! - Trunk Network: takes continuous coordinate query $y \in \mathbb{R}^d$ and maps to basis functions $[t_1, \dots, t_p] \in \mathbb{R}^p$.
//! - Dot-product synthesis produces continuous operator evaluation at arbitrary query points.

use crate::error::{Error, Result};
use crate::nn::{Linear, Module};
use crate::tensor::Tensor;

/// Deep Operator Network (DeepONet).
pub struct DeepONet {
    pub branch_in_dim: usize,
    pub trunk_in_dim: usize,
    pub hidden_dim: usize,
    pub basis_dim: usize,
    pub branch_fc1: Linear,
    pub branch_fc2: Linear,
    pub trunk_fc1: Linear,
    pub trunk_fc2: Linear,
    pub bias: f64,
}

impl DeepONet {
    /// Creates a new `DeepONet` with given input dimensions and basis count.
    pub fn new(
        branch_in_dim: usize,
        trunk_in_dim: usize,
        hidden_dim: usize,
        basis_dim: usize,
    ) -> Result<Self> {
        let branch_fc1 = Linear::new(branch_in_dim, hidden_dim)?;
        let branch_fc2 = Linear::new(hidden_dim, basis_dim)?;
        let trunk_fc1 = Linear::new(trunk_in_dim, hidden_dim)?;
        let trunk_fc2 = Linear::new(hidden_dim, basis_dim)?;

        Ok(Self {
            branch_in_dim,
            trunk_in_dim,
            hidden_dim,
            basis_dim,
            branch_fc1,
            branch_fc2,
            trunk_fc1,
            trunk_fc2,
            bias: 0.0,
        })
    }

    /// Evaluates the branch network: $u \in \mathbb{R}^{B \times m} \to b(u) \in \mathbb{R}^{B \times p}$.
    pub fn forward_branch(&self, u: &Tensor) -> Result<Tensor> {
        let u_2d = if u.ndim() == 1 {
            u.unsqueeze(0)?
        } else if u.ndim() == 2 {
            u.clone()
        } else {
            return Err(Error::Msg(format!(
                "DeepONet branch input must be 1D or 2D [Batch, m], got {:?}",
                u.shape()
            )));
        };

        if u_2d.shape()[1] != self.branch_in_dim {
            return Err(Error::Msg(format!(
                "Branch input dimension {} != expected {}",
                u_2d.shape()[1],
                self.branch_in_dim
            )));
        }

        let h = self.branch_fc1.forward(&u_2d)?.gelu()?;
        self.branch_fc2.forward(&h)
    }

    /// Evaluates the trunk network: $y \in \mathbb{R}^{N \times d} \to t(y) \in \mathbb{R}^{N \times p}$.
    pub fn forward_trunk(&self, y: &Tensor) -> Result<Tensor> {
        let y_2d = if y.ndim() == 1 {
            y.unsqueeze(0)?
        } else if y.ndim() == 2 {
            y.clone()
        } else {
            return Err(Error::Msg(format!(
                "DeepONet trunk input must be 1D or 2D [N_coords, d], got {:?}",
                y.shape()
            )));
        };

        if y_2d.shape()[1] != self.trunk_in_dim {
            return Err(Error::Msg(format!(
                "Trunk input dimension {} != expected {}",
                y_2d.shape()[1],
                self.trunk_in_dim
            )));
        }

        let h = self.trunk_fc1.forward(&y_2d)?.gelu()?;
        self.trunk_fc2.forward(&h)
    }

    /// Evaluates the nonlinear operator $G(u)(y)$:
    /// - `u_sensors`: $\[B, m\]$ or $\[m\]$
    /// - `y_coords`: $\[N, d\]$ or $\[d\]$
    /// - Output shape: $[B, N]$
    pub fn forward(&self, u_sensors: &Tensor, y_coords: &Tensor) -> Result<Tensor> {
        let b = self.forward_branch(u_sensors)?;
        let t = self.forward_trunk(y_coords)?;

        let t_cast = t.to_dtype(b.dtype())?;
        let out = b.matmul(&t_cast.t()?)?;
        if self.bias.abs() > 1e-12 {
            out.add_scalar(self.bias)
        } else {
            Ok(out)
        }
    }

    /// Returns all trainable parameters of the DeepONet.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.branch_fc1.parameters());
        params.extend(self.branch_fc2.parameters());
        params.extend(self.trunk_fc1.parameters());
        params.extend(self.trunk_fc2.parameters());
        params
    }
}
