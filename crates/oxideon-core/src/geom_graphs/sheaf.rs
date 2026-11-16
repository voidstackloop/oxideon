//! Category-Theoretic Cellular Sheaves & Sheaf Laplacian Neural Networks.
//!
//! Implements cellular sheaves over graphs for topological and non-Euclidean representation learning:
//! - Cellular Sheaf $(G, \mathcal{F})$: Stalks $\mathcal{F}(v) = \mathbb{R}^{d_v}$ at vertices,
//!   $\mathcal{F}(e) = \mathbb{R}^{d_e}$ at edges, and linear restriction maps $\mathcal{F}_{v \unlhd e}: \mathcal{F}(v) \to \mathcal{F}(e)$.
//! - Sheaf Coboundary Operator $\delta: C^0(G; \mathcal{F}) \to C^1(G; \mathcal{F})$:
//!   $$(\delta x)_{e = (u, v)} = \mathcal{F}_{v \unlhd e} x_v - \mathcal{F}_{u \unlhd e} x_u$$
//! - Symmetric Positive Semi-Definite Sheaf Laplacian $L_{\mathcal{F}} = \delta^T \delta$:
//!   $$L_{uu} = \sum_{e \ni u} \mathcal{F}_{u \unlhd e}^T \mathcal{F}_{u \unlhd e}, \quad L_{uv} = -\mathcal{F}_{u \unlhd e}^T \mathcal{F}_{v \unlhd e}$$
//! - Continuous Sheaf Diffusion Flow $\dot{x} = -L_{\mathcal{F}} x$, provably overcoming
//!   graph neural network over-smoothing and topological bottlenecking.
//! - Sheaf Dirichlet Energy $E(x) = x^T L_{\mathcal{F}} x \ge 0$.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Cellular Sheaf defined over an undirected graph.
#[derive(Debug, Clone)]
pub struct CellularSheaf {
    pub num_nodes: usize,
    pub node_dim: usize,
    pub edge_dim: usize,
    pub edges: Vec<(usize, usize)>,
    /// Restriction maps $(F_u, F_v)$ for each edge $e = (u, v)$, each of shape `[edge_dim, node_dim]`.
    pub restriction_maps: Vec<(Tensor, Tensor)>,
}

impl CellularSheaf {
    /// Creates a new Cellular Sheaf with initialized orthogonal or random restriction maps.
    pub fn new(
        num_nodes: usize,
        node_dim: usize,
        edge_dim: usize,
        edges: Vec<(usize, usize)>,
    ) -> Result<Self> {
        let mut restriction_maps = Vec::with_capacity(edges.len());
        for &(u, v) in &edges {
            if u >= num_nodes || v >= num_nodes {
                return Err(Error::Msg(format!(
                    "Invalid edge ({u}, {v}) for num_nodes {num_nodes}"
                )));
            }
            let fu = Tensor::eye(
                edge_dim.min(node_dim),
                DType::F64,
                &crate::device::Device::Cpu,
            )?;
            let fv = Tensor::eye(
                edge_dim.min(node_dim),
                DType::F64,
                &crate::device::Device::Cpu,
            )?;

            let fu_full = if edge_dim == node_dim {
                fu
            } else {
                Tensor::randn(&[edge_dim, node_dim])?.mul_scalar(0.5)?
            };
            let fv_full = if edge_dim == node_dim {
                fv
            } else {
                Tensor::randn(&[edge_dim, node_dim])?.mul_scalar(0.5)?
            };

            restriction_maps.push((fu_full, fv_full));
        }

        Ok(Self {
            num_nodes,
            node_dim,
            edge_dim,
            edges,
            restriction_maps,
        })
    }

    /// Computes the complete block Sheaf Laplacian $L_{\mathcal{F}} \in \mathbb{R}^{(N \cdot d_v) \times (N \cdot d_v)}$.
    /// Verifies stalk/edge buffers agree with the (public, tamperable)
    /// dims before any indexing.
    fn check_layout(&self) -> Result<usize> {
        let total = self.num_nodes.checked_mul(self.node_dim).ok_or_else(|| {
            Error::Msg(format!(
                "Sheaf stalk size {} * {} overflows",
                self.num_nodes, self.node_dim
            ))
        })?;
        if self.restriction_maps.len() != self.edges.len() {
            return Err(Error::Msg(format!(
                "Sheaf maps length {} disagrees with {} edges",
                self.restriction_maps.len(),
                self.edges.len()
            )));
        }
        for &(u, v) in &self.edges {
            if u >= self.num_nodes || v >= self.num_nodes {
                return Err(Error::IndexOutOfBounds {
                    op: "CellularSheaf::check_layout",
                    index: u.max(v) as isize,
                    dim_size: self.num_nodes,
                });
            }
        }
        for (e_idx, (fu, fv)) in self.restriction_maps.iter().enumerate() {
            for (side, m) in [("u", fu), ("v", fv)] {
                let s = m.shape();
                if s.len() != 2 || s[0] != self.edge_dim || s[1] != self.node_dim {
                    return Err(Error::Msg(format!(
                        "Sheaf map {e_idx}{side} shape {:?} disagrees with [{}, {}]",
                        s, self.edge_dim, self.node_dim
                    )));
                }
            }
        }
        Ok(total)
    }

    pub fn compute_sheaf_laplacian(&self) -> Result<Tensor> {
        let total_dim = self.check_layout()?;
        let cells = total_dim.checked_mul(total_dim).ok_or_else(|| {
            Error::Msg(format!(
                "Sheaf laplacian size {total_dim} * {total_dim} overflows"
            ))
        })?;
        let mut laplacian_data = crate::layout::try_filled(cells, 0.0f64)?;

        for (e_idx, &(u, v)) in self.edges.iter().enumerate() {
            let (ref fu, ref fv) = self.restriction_maps[e_idx];

            // Fu^T * Fu, Fv^T * Fv, Fu^T * Fv, Fv^T * Fu
            let fu_t = fu.t()?;
            let fv_t = fv.t()?;

            let fu_t_fu = fu_t.matmul(fu)?.to_vec::<f64>()?;
            let fv_t_fv = fv_t.matmul(fv)?.to_vec::<f64>()?;
            let fu_t_fv = fu_t.matmul(fv)?.to_vec::<f64>()?;
            let fv_t_fu = fv_t.matmul(fu)?.to_vec::<f64>()?;

            let d = self.node_dim;

            // Add to block entries:
            // L_uu += Fu^T Fu
            for r in 0..d {
                for c in 0..d {
                    let global_idx = (u * d + r) * total_dim + (u * d + c);
                    laplacian_data[global_idx] += fu_t_fu[r * d + c];
                }
            }

            // L_vv += Fv^T Fv
            for r in 0..d {
                for c in 0..d {
                    let global_idx = (v * d + r) * total_dim + (v * d + c);
                    laplacian_data[global_idx] += fv_t_fv[r * d + c];
                }
            }

            // L_uv -= Fu^T Fv
            for r in 0..d {
                for c in 0..d {
                    let global_idx = (u * d + r) * total_dim + (v * d + c);
                    laplacian_data[global_idx] -= fu_t_fv[r * d + c];
                }
            }

            // L_vu -= Fv^T Fu
            for r in 0..d {
                for c in 0..d {
                    let global_idx = (v * d + r) * total_dim + (u * d + c);
                    laplacian_data[global_idx] -= fv_t_fu[r * d + c];
                }
            }
        }

        Tensor::from_vec(laplacian_data, &[total_dim, total_dim])
    }

    /// Evaluates the Sheaf Dirichlet Energy $E(x) = x^T L_{\mathcal{F}} x \ge 0$.
    pub fn dirichlet_energy(&self, x: &Tensor) -> Result<f64> {
        let total_dim = self.check_layout()?;
        let l = self.compute_sheaf_laplacian()?.to_dtype(x.dtype())?;
        let x_flat = if x.ndim() == 1 {
            x.clone()
        } else {
            x.reshape(&[total_dim])?
        };

        let x_col = x_flat.unsqueeze(1)?;
        let lx = l.matmul(&x_col)?;
        let energy = x_flat.unsqueeze(0)?.matmul(&lx)?.item::<f64>()?;
        Ok(energy)
    }

    /// Continuous sheaf diffusion step:
    /// $$x_{t + \Delta t} = x_t - \Delta t \cdot (L_{\mathcal{F}} x_t)$$
    pub fn sheaf_diffusion_step(&self, x: &Tensor, dt: f64) -> Result<Tensor> {
        let total_dim = self.check_layout()?;
        let l = self.compute_sheaf_laplacian()?.to_dtype(x.dtype())?;
        let x_flat = if x.ndim() == 1 {
            x.clone()
        } else {
            x.reshape(&[total_dim])?
        };

        let x_col = x_flat.unsqueeze(1)?;
        let lx = l.matmul(&x_col)?.squeeze()?;
        let dx = lx.mul_scalar(dt)?;
        let x_next = x_flat.sub(&dx)?;

        if x.ndim() > 1 {
            x_next.reshape(x.shape())
        } else {
            Ok(x_next)
        }
    }
}
