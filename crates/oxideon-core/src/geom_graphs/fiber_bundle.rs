//! Non-Euclidean Fiber Bundles & Principal Connection Gauge Networks.
//!
//! Implements gauge-equivariant neural architectures over discrete principal bundles $(P, B, \pi, G)$,
//! where base manifold $B$ is a graph, fiber $F \cong \mathbb{R}^d$ carries representation of structure group $G = SO(2)$,
//! parallel transport is governed by orthogonal connection matrices $R_{uv} \in SO(2)$,
//! and connection Laplacians measure field curvature and holonomy.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Discrete connection on a graph representing parallel transport in a principal $SO(2)$ bundle.
#[derive(Debug, Clone)]
pub struct PrincipalBundleConnection {
    /// Number of base graph nodes $N$.
    pub num_nodes: usize,
    /// Dimension of fiber space $d$ (must be a multiple of 2 for $SO(2)$ representation blocks).
    pub fiber_dim: usize,
    /// Connection angles $\theta_{uv} \in [-\pi, \pi]$ for directed edges $(u, v)$.
    /// Stored as a flat matrix of size $N \times N$.
    pub connection_angles: Vec<f64>,
    /// Adjacency existence mask of size $N \times N$.
    pub adj_mask: Vec<bool>,
}

impl PrincipalBundleConnection {
    /// Initializes an empty connection for $N$ nodes.
    pub fn new(num_nodes: usize, fiber_dim: usize) -> Result<Self> {
        if fiber_dim == 0 || fiber_dim % 2 != 0 {
            return Err(Error::Msg(
                "Fiber dimension must be a positive multiple of 2 for SO(2) representations".into(),
            ));
        }

        let cells = crate::layout::checked_numel(&[num_nodes, num_nodes])?;
        Ok(Self {
            num_nodes,
            fiber_dim,
            connection_angles: crate::layout::try_filled(cells, 0.0)?,
            adj_mask: crate::layout::try_filled(cells, false)?,
        })
    }

    /// Sets the parallel transport connection angle $\theta_{uv}$ for edge $(u, v)$.
    ///
    /// Preserves antisymmetry of connection forms: $\theta_{vu} = -\theta_{uv}$.
    pub fn set_connection(&mut self, u: usize, v: usize, angle: f64) -> Result<()> {
        if u >= self.num_nodes || v >= self.num_nodes {
            return Err(Error::IndexOutOfBounds {
                op: "PrincipalBundleConnection::set_connection",
                index: u.max(v) as isize,
                dim_size: self.num_nodes,
            });
        }

        let idx_uv = u * self.num_nodes + v;
        let idx_vu = v * self.num_nodes + u;

        self.connection_angles[idx_uv] = angle;
        self.adj_mask[idx_uv] = true;

        self.connection_angles[idx_vu] = -angle;
        self.adj_mask[idx_vu] = true;

        Ok(())
    }

    /// Computes the $2 \times 2$ rotation block for angle $\theta$:
    /// $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.
    #[inline]
    pub fn rotation_block_2d(theta: f64) -> [f64; 4] {
        let (s, c) = theta.sin_cos();
        [c, -s, s, c]
    }

    /// Parallel transports a fiber vector $x_v \in \mathbb{R}^d$ from node $v$ to node $u$ along $(u, v)$:
    /// $x_{u \leftarrow v} = R_{uv} x_v$.
    pub fn transport_vector(&self, u: usize, v: usize, x_v: &[f64]) -> Result<Vec<f64>> {
        if x_v.len() != self.fiber_dim {
            return Err(Error::ShapeMismatch {
                op: "transport_vector",
                expected: vec![self.fiber_dim],
                got: vec![x_v.len()],
            });
        }

        if u >= self.num_nodes || v >= self.num_nodes {
            return Err(Error::IndexOutOfBounds {
                op: "PrincipalBundleConnection::transport_vector",
                index: u.max(v) as isize,
                dim_size: self.num_nodes,
            });
        }
        let theta = self.connection_angles[u * self.num_nodes + v];
        let r = Self::rotation_block_2d(theta);

        let mut out = vec![0.0; self.fiber_dim];
        let num_blocks = self.fiber_dim / 2;

        for b in 0..num_blocks {
            let x0 = x_v[2 * b];
            let x1 = x_v[2 * b + 1];

            out[2 * b] = r[0] * x0 + r[1] * x1;
            out[2 * b + 1] = r[2] * x0 + r[3] * x1;
        }

        Ok(out)
    }

    /// Computes the holonomy (loop curvature) around a directed cycle of node indices $[v_0, v_1, \dots, v_k, v_0]$.
    ///
    /// For a flat connection, the cumulative angle is an integer multiple of $2\pi$.
    pub fn cycle_holonomy(&self, cycle: &[usize]) -> Result<f64> {
        if cycle.len() < 3 {
            return Err(Error::Msg("Cycle must have at least 3 nodes".into()));
        }

        let mut total_angle = 0.0;
        for i in 0..cycle.len() {
            let u = cycle[i];
            let v = cycle[(i + 1) % cycle.len()];
            if u >= self.num_nodes || v >= self.num_nodes {
                return Err(Error::IndexOutOfBounds {
                    op: "PrincipalBundleConnection::cycle_holonomy",
                    index: u.max(v) as isize,
                    dim_size: self.num_nodes,
                });
            }

            let idx = u * self.num_nodes + v;
            if !self.adj_mask[idx] {
                return Err(Error::Msg(format!(
                    "Edge ({u}, {v}) in cycle does not exist"
                )));
            }
            total_angle += self.connection_angles[idx];
        }

        // Wrap to [-pi, pi]
        let pi = std::f64::consts::PI;
        let mut wrapped = total_angle % (2.0 * pi);
        if wrapped > pi {
            wrapped -= 2.0 * pi;
        } else if wrapped < -pi {
            wrapped += 2.0 * pi;
        }

        Ok(wrapped)
    }
}

/// Gauge-Equivariant Principal Connection Convolution Layer.
///
/// Satisfies exact gauge equivariance:
/// If node reference frames undergo local gauge rotations $g_u \in SO(2)$,
/// the network outputs transform identically: $Y' = g \cdot Y$.
#[derive(Debug, Clone)]
pub struct GaugeEquivariantConvLayer {
    /// Principal bundle connection defining parallel transport.
    pub connection: PrincipalBundleConnection,
    /// Fiber feature dimension.
    pub fiber_dim: usize,
    /// Learnable self-connection scalar weights per 2D block.
    pub weight_self: Vec<f64>,
    /// Learnable neighbor-connection scalar weights per 2D block.
    pub weight_neighbor: Vec<f64>,
}

impl GaugeEquivariantConvLayer {
    /// Creates a new Gauge-Equivariant Convolution Layer.
    pub fn new(connection: PrincipalBundleConnection) -> Self {
        let fiber_dim = connection.fiber_dim;
        let num_blocks = fiber_dim / 2;

        let weight_self = vec![1.0; num_blocks];
        let weight_neighbor = vec![0.5; num_blocks];

        Self {
            connection,
            fiber_dim,
            weight_self,
            weight_neighbor,
        }
    }

    /// Performs a gauge-equivariant forward pass on input fiber sections $X \in \mathbb{R}^{N \times d}$.
    ///
    /// $$y_u = W_{\text{self}} x_u + \sum_{v \in \mathcal{N}(u)} W_{\text{neigh}} R_{uv} x_v$$
    /// Verifies connection masks and weight buffers agree with the
    /// (public, tamperable) layer dims before any indexing.
    fn check_layout(&self) -> Result<(usize, usize)> {
        let n = self.connection.num_nodes;
        let d = self.fiber_dim;
        let cells = n
            .checked_mul(n)
            .ok_or_else(|| Error::Msg(format!("gauge connection size {n} * {n} overflows")))?;
        if self.connection.adj_mask.len() != cells
            || self.connection.connection_angles.len() != cells
        {
            return Err(Error::Msg(format!(
                "gauge connection buffers ({}, {}) disagree with {n}*{n}",
                self.connection.adj_mask.len(),
                self.connection.connection_angles.len()
            )));
        }
        let blocks = d / 2;
        if self.weight_self.len() != blocks || self.weight_neighbor.len() != blocks {
            return Err(Error::Msg(format!(
                "gauge weight lengths ({}, {}) disagree with {d}/2 blocks",
                self.weight_self.len(),
                self.weight_neighbor.len()
            )));
        }
        Ok((n, d))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (n, d) = self.check_layout()?;
        let shape = x.shape();

        if shape.len() != 2 || shape[0] != n || shape[1] != d {
            return Err(Error::ShapeMismatch {
                op: "GaugeEquivariantConvLayer::forward",
                expected: vec![n, d],
                got: shape.to_vec(),
            });
        }

        let x_data = x.to_vec::<f64>()?;
        let mut y = vec![0.0; n * d];
        let num_blocks = d / 2;

        for u in 0..n {
            // 1. Self contribution: W_self * x_u
            for b in 0..num_blocks {
                let w = self.weight_self[b];
                y[u * d + 2 * b] += w * x_data[u * d + 2 * b];
                y[u * d + 2 * b + 1] += w * x_data[u * d + 2 * b + 1];
            }

            // 2. Neighbor contributions: W_neigh * (R_uv * x_v)
            for v in 0..n {
                if u == v || !self.connection.adj_mask[u * n + v] {
                    continue;
                }

                // Extract x_v
                let xv = &x_data[v * d..(v + 1) * d];
                let transported = self.connection.transport_vector(u, v, xv)?;

                for b in 0..num_blocks {
                    let w = self.weight_neighbor[b];
                    y[u * d + 2 * b] += w * transported[2 * b];
                    y[u * d + 2 * b + 1] += w * transported[2 * b + 1];
                }
            }
        }

        Tensor::from_slice(&y, &[n, d])
    }

    /// Verifies gauge equivariance numerically under arbitrary local gauge transformations $\{g_u\} \in SO(2)^N$.
    ///
    /// Computes $\Delta = \| Y' - g \cdot Y \|_{\infty}$, which must be zero within floating point error.
    pub fn verify_gauge_equivariance(&self, x: &Tensor, gauge_angles: &[f64]) -> Result<f64> {
        let n = self.connection.num_nodes;
        let d = self.fiber_dim;
        let num_blocks = d / 2;

        if gauge_angles.len() != n {
            return Err(Error::ShapeMismatch {
                op: "verify_gauge_equivariance",
                expected: vec![n],
                got: vec![gauge_angles.len()],
            });
        }

        // 1. Evaluate baseline output Y = f(X, R)
        let y_base = self.forward(x)?;
        let y_base_data = y_base.to_vec::<f64>()?;

        // 2. Rotate baseline output under gauge: g * Y
        let mut rotated_y_base = vec![0.0; n * d];
        for u in 0..n {
            let r_u = PrincipalBundleConnection::rotation_block_2d(gauge_angles[u]);
            for b in 0..num_blocks {
                let y0 = y_base_data[u * d + 2 * b];
                let y1 = y_base_data[u * d + 2 * b + 1];
                rotated_y_base[u * d + 2 * b] = r_u[0] * y0 + r_u[1] * y1;
                rotated_y_base[u * d + 2 * b + 1] = r_u[2] * y0 + r_u[3] * y1;
            }
        }

        // 3. Transform inputs and connection under gauge:
        // X'_u = g_u * X_u
        // R'_{uv} = g_u * R_{uv} * g_v^{-1} ==> theta'_{uv} = theta_{uv} + g_u - g_v
        let x_data = x.to_vec::<f64>()?;
        let mut x_transformed = vec![0.0; n * d];
        for u in 0..n {
            let r_u = PrincipalBundleConnection::rotation_block_2d(gauge_angles[u]);
            for b in 0..num_blocks {
                let x0 = x_data[u * d + 2 * b];
                let x1 = x_data[u * d + 2 * b + 1];
                x_transformed[u * d + 2 * b] = r_u[0] * x0 + r_u[1] * x1;
                x_transformed[u * d + 2 * b + 1] = r_u[2] * x0 + r_u[3] * x1;
            }
        }
        let x_gauge = Tensor::from_slice(&x_transformed, &[n, d])?;

        // Build transformed connection
        let mut gauge_conn = PrincipalBundleConnection::new(n, d)?;
        for u in 0..n {
            for v in (u + 1)..n {
                if self.connection.adj_mask[u * n + v] {
                    let old_theta = self.connection.connection_angles[u * n + v];
                    let new_theta = old_theta + gauge_angles[u] - gauge_angles[v];
                    gauge_conn.set_connection(u, v, new_theta)?;
                }
            }
        }

        let mut transformed_layer = self.clone();
        transformed_layer.connection = gauge_conn;

        // 4. Evaluate transformed output Y' = f(X', R')
        let y_transformed = transformed_layer.forward(&x_gauge)?;
        let y_transformed_data = y_transformed.to_vec::<f64>()?;

        // 5. Measure max absolute difference
        let mut max_diff = 0.0;
        for i in 0..(n * d) {
            let diff = (rotated_y_base[i] - y_transformed_data[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        Ok(max_diff)
    }
}
