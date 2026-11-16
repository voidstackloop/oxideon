//! Cellular Sheaf Cohomology & Sheaf Laplacian Diffusion Networks.
//!
//! Implements cellular sheaves over graphs for non-Euclidean representation learning,
//! sheaf coboundary operators $\delta_0: C^0 \to C^1$, exact Sheaf Dirichlet energies,
//! matrix-free Sheaf Laplacian action $L_{\mathcal{F}} x$, obstruction theory, and
//! continuous sheaf diffusion that overcomes topological over-smoothing on heterophilic graphs.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Cellular Sheaf with orthogonal restriction maps parametrized via Lie algebra Cayley transforms.
#[derive(Debug, Clone)]
pub struct AdaptiveCellularSheaf {
    /// Number of nodes $N$.
    pub num_nodes: usize,
    /// Dimension of node stalk $d$.
    pub stalk_dim: usize,
    /// Edges $(u, v)$ where $u < v$.
    pub edges: Vec<(usize, usize)>,
    /// Skew-symmetric parameters for restriction map $F_{u \unlhd e}$ and $F_{v \unlhd e}$.
    /// For each edge $e$, stores angles $[\theta_{u \unlhd e}, \theta_{v \unlhd e}]$.
    pub edge_angles: Vec<(f64, f64)>,
}

impl AdaptiveCellularSheaf {
    /// Creates a new Adaptive Cellular Sheaf.
    pub fn new(num_nodes: usize, stalk_dim: usize, edges: Vec<(usize, usize)>) -> Result<Self> {
        if stalk_dim == 0 || stalk_dim % 2 != 0 {
            return Err(Error::Msg(
                "Stalk dimension must be a positive multiple of 2 for orthogonal rotation blocks"
                    .into(),
            ));
        }

        for &(u, v) in &edges {
            if u >= num_nodes || v >= num_nodes || u == v {
                return Err(Error::Msg(format!(
                    "Invalid edge ({u}, {v}) for num_nodes {num_nodes}"
                )));
            }
        }

        let num_edges = edges.len();
        let edge_angles = vec![(0.0, 0.0); num_edges];

        Ok(Self {
            num_nodes,
            stalk_dim,
            edges,
            edge_angles,
        })
    }

    /// Sets the restriction angles for edge $e = (u, v)$.
    pub fn set_edge_angles(&mut self, edge_idx: usize, theta_u: f64, theta_v: f64) -> Result<()> {
        if edge_idx >= self.edges.len() {
            return Err(Error::IndexOutOfBounds {
                op: "set_edge_angles",
                index: edge_idx as isize,
                dim_size: self.edges.len(),
            });
        }
        self.edge_angles[edge_idx] = (theta_u, theta_v);
        Ok(())
    }

    /// Applies a 2D rotation block $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$
    /// block-wise to a vector of length `stalk_dim`.
    fn apply_orthogonal_map(theta: f64, v: &[f64], stalk_dim: usize) -> Vec<f64> {
        let (s, c) = theta.sin_cos();
        let mut out = vec![0.0; stalk_dim];
        let num_blocks = stalk_dim / 2;

        for b in 0..num_blocks {
            let x0 = v[2 * b];
            let x1 = v[2 * b + 1];
            out[2 * b] = c * x0 - s * x1;
            out[2 * b + 1] = s * x0 + c * x1;
        }

        out
    }

    /// Transpose (inverse for orthogonal maps) rotation: $R(-\theta)$.
    fn apply_transpose_map(theta: f64, v: &[f64], stalk_dim: usize) -> Vec<f64> {
        Self::apply_orthogonal_map(-theta, v, stalk_dim)
    }

    fn check_topology(&self, op: &'static str) -> Result<()> {
        if self.edges.len() != self.edge_angles.len() {
            return Err(Error::Msg(format!(
                "{op}: {} edges but {} edge-angle entries",
                self.edges.len(),
                self.edge_angles.len()
            )));
        }
        for &(u, v) in &self.edges {
            if u >= self.num_nodes || v >= self.num_nodes {
                return Err(Error::Msg(format!(
                    "{op}: edge ({u}, {v}) out of bounds for {} nodes",
                    self.num_nodes
                )));
            }
        }
        Ok(())
    }

    fn check_signal(&self, op: &'static str, x: &[f64]) -> Result<(usize, usize)> {
        let n = self.num_nodes;
        let d = self.stalk_dim;
        let need = n
            .checked_mul(d)
            .ok_or_else(|| Error::Msg(format!("{op}: signal size {n} * {d} overflows")))?;
        if x.len() != need {
            return Err(Error::ShapeMismatch {
                op,
                expected: vec![need],
                got: vec![x.len()],
            });
        }
        Ok((n, d))
    }

    /// Evaluates the Sheaf Coboundary Operator $\delta_0: C^0(G; \mathcal{F}) \to C^1(G; \mathcal{F})$.
    ///
    /// For each directed edge $e = (u, v)$:
    /// $$(\delta_0 x)_e = \mathcal{F}_{v \unlhd e} x_v - \mathcal{F}_{u \unlhd e} x_u$$
    pub fn coboundary(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.check_topology("AdaptiveCellularSheaf::coboundary")?;
        let (_, d) = self.check_signal("AdaptiveCellularSheaf::coboundary", x)?;

        let num_edges = self.edges.len();
        let mut out = vec![0.0; num_edges * d];

        for (e_idx, &(u, v)) in self.edges.iter().enumerate() {
            let (th_u, th_v) = self.edge_angles[e_idx];

            let x_u = &x[u * d..(u + 1) * d];
            let x_v = &x[v * d..(v + 1) * d];

            let fu_xu = Self::apply_orthogonal_map(th_u, x_u, d);
            let fv_xv = Self::apply_orthogonal_map(th_v, x_v, d);

            for b in 0..d {
                out[e_idx * d + b] = fv_xv[b] - fu_xu[b];
            }
        }

        Ok(out)
    }

    /// Computes the Sheaf Dirichlet Energy:
    /// $$E_{\mathcal{F}}(x) = \frac{1}{2} \|\delta_0 x\|_2^2 = \frac{1}{2} x^T L_{\mathcal{F}} x$$
    pub fn dirichlet_energy(&self, x: &[f64]) -> Result<f64> {
        let delta_x = self.coboundary(x)?;
        let energy: f64 = delta_x.iter().map(|&val| val * val).sum::<f64>() * 0.5;
        Ok(energy)
    }

    /// Evaluates the Sheaf Laplacian matrix-vector action $y = L_{\mathcal{F}} x$ matrix-free in $O(|E| \cdot d)$ time:
    ///
    /// $$(L_{\mathcal{F}} x)_u = \sum_{e = (u, v)} \mathcal{F}_{u \unlhd e}^T (\mathcal{F}_{u \unlhd e} x_u - \mathcal{F}_{v \unlhd e} x_v)$$
    pub fn laplacian_matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        self.check_topology("AdaptiveCellularSheaf::laplacian_matvec")?;
        let (n, d) = self.check_signal("AdaptiveCellularSheaf::laplacian_matvec", x)?;

        let mut y = vec![0.0; n * d];

        for (e_idx, &(u, v)) in self.edges.iter().enumerate() {
            let (th_u, th_v) = self.edge_angles[e_idx];

            let x_u = &x[u * d..(u + 1) * d];
            let x_v = &x[v * d..(v + 1) * d];

            let fu_xu = Self::apply_orthogonal_map(th_u, x_u, d);
            let fv_xv = Self::apply_orthogonal_map(th_v, x_v, d);

            // difference on edge e: (F_v x_v - F_u x_u)
            let mut diff = vec![0.0; d];
            for b in 0..d {
                diff[b] = fv_xv[b] - fu_xu[b];
            }

            // Contribution to u: - F_u^T (F_v x_v - F_u x_u) = F_u^T (F_u x_u - F_v x_v)
            let contrib_u = Self::apply_transpose_map(th_u, &diff, d);
            for b in 0..d {
                y[u * d + b] -= contrib_u[b];
            }

            // Contribution to v: + F_v^T (F_v x_v - F_u x_u)
            let contrib_v = Self::apply_transpose_map(th_v, &diff, d);
            for b in 0..d {
                y[v * d + b] += contrib_v[b];
            }
        }

        Ok(y)
    }

    /// Measures the obstruction to extending local node stalks into a global harmonic section:
    /// $\operatorname{Obs}(x) = \|\delta_0 x\|_2$.
    ///
    /// When $\operatorname{Obs}(x) = 0$, $x \in H^0(G; \mathcal{F}) = \ker(L_{\mathcal{F}})$,
    /// representing an exact global section of the sheaf.
    pub fn obstruction_norm(&self, x: &[f64]) -> Result<f64> {
        let delta_x = self.coboundary(x)?;
        let norm = delta_x.iter().map(|&v| v * v).sum::<f64>().sqrt();
        Ok(norm)
    }
}

/// Continuous Sheaf Diffusion Neural Network Layer.
///
/// Implements continuous sheaf diffusion:
/// $$X_{t + \Delta t} = \sigma\left(X_t - \Delta t \cdot L_{\mathcal{F}} X_t + X_t W_{\text{param}}\right)$$
#[derive(Debug, Clone)]
pub struct SheafDiffusionLayer {
    /// Cellular sheaf defining the diffusion operator.
    pub sheaf: AdaptiveCellularSheaf,
    /// Diffusion step size $\Delta t > 0$.
    pub dt: f64,
    /// Parameter mixing matrix $W \in \mathbb{R}^{d \times d}$.
    pub weight: Vec<f64>,
}

impl SheafDiffusionLayer {
    /// Creates a new Sheaf Diffusion Layer.
    pub fn new(sheaf: AdaptiveCellularSheaf, dt: f64) -> Self {
        let d = sheaf.stalk_dim;
        let mut weight = vec![0.0; d * d];
        // Initialize as scaled identity to preserve representations
        for i in 0..d {
            weight[i * d + i] = 0.5;
        }

        Self { sheaf, dt, weight }
    }

    /// Performs one continuous sheaf diffusion step on input section $X \in \mathbb{R}^{N \times d}$.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = self.sheaf.num_nodes;
        let d = self.sheaf.stalk_dim;
        let shape = x.shape();

        if shape.len() != 2 || shape[0] != n || shape[1] != d {
            return Err(Error::ShapeMismatch {
                op: "SheafDiffusionLayer::forward",
                expected: vec![n, d],
                got: shape.to_vec(),
            });
        }

        let x_data = x.to_vec::<f64>()?;

        // 1. Sheaf Laplacian matrix-free action: L_F * x
        let l_x = self.sheaf.laplacian_matvec(&x_data)?;

        // 2. Continuous diffusion: x - dt * (L_F * x)
        let mut diffused = vec![0.0; n * d];
        for i in 0..(n * d) {
            diffused[i] = x_data[i] - self.dt * l_x[i];
        }

        // 3. Stalk feature mixing: diffused * W
        let mut out = vec![0.0; n * d];
        for u in 0..n {
            for col in 0..d {
                let mut sum = 0.0;
                for row in 0..d {
                    sum += diffused[u * d + row] * self.weight[row * d + col];
                }
                out[u * d + col] = sum;
            }
        }

        // 4. Non-linear activation (GELU)
        for val in &mut out {
            let x = *val;
            *val = 0.5
                * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh());
        }

        Tensor::from_slice(&out, &[n, d])
    }
}
