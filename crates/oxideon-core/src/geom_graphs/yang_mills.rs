//! Non-Abelian Yang-Mills Gauge Fields & $SU(2)$ Wilson Loop Lattice Networks.
//!
//! Implements discrete non-Abelian gauge theory on graphs and lattices:
//! - Lie group $G = SU(2)$ represented via unit quaternions: $U = a_0 \mathbf{1} + i \sum_{k=1}^3 a_k \sigma_k$.
//! - Directed gauge links $U_{uv} \in SU(2)$ with anti-symmetry $U_{vu} = U_{uv}^{-1} = U_{uv}^\dagger$.
//! - Local $SU(2)$ gauge transformations: $U_{uv} \mapsto \Omega_u U_{uv} \Omega_v^\dagger$.
//! - Strictly gauge-invariant Wilson loop operators: $W(C) = \frac{1}{2} \operatorname{Re}\operatorname{Tr}\left(\prod_{e \in C} U_e\right) \in [-1, 1]$.
//! - Yang-Mills lattice plaquette action $S_{\text{YM}} = \beta \sum_p (1 - W(p))$.
//! - Non-Abelian gauge-equivariant spinor message passing layers.
//!
//! References:
//! - Wilson (1974): "Confinement of quarks", Phys. Rev. D 10, 2445
//! - Cohen et al. (2019): "Gauge Equivariant Convolutional Networks and the Icosahedral CNN"

use crate::error::{Error, Result};

/// An element of the non-Abelian Lie group $SU(2)$ parameterized as a unit quaternion:
///
/// $$q = a_0 + a_1 \mathbf{i} + a_2 \mathbf{j} + a_3 \mathbf{k}, \quad a_0^2 + a_1^2 + a_2^2 + a_3^2 = 1$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Su2Element {
    pub a0: f64,
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
}

impl Su2Element {
    /// Identity element $\mathbf{1}_{2 \times 2}$.
    pub fn identity() -> Self {
        Self {
            a0: 1.0,
            a1: 0.0,
            a2: 0.0,
            a3: 0.0,
        }
    }

    /// Creates an $SU(2)$ element from a normalized 4-vector.
    pub fn from_quaternion(a0: f64, a1: f64, a2: f64, a3: f64) -> Result<Self> {
        let norm_sq = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
        if norm_sq < 1e-12 {
            return Err(Error::Msg("Quaternion norm must be non-zero".to_string()));
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        Ok(Self {
            a0: a0 * inv_norm,
            a1: a1 * inv_norm,
            a2: a2 * inv_norm,
            a3: a3 * inv_norm,
        })
    }

    /// Creates an $SU(2)$ element from rotation axis $\hat{n} \in \mathbb{R}^3$ and angle $\theta$:
    ///
    /// $$U = \cos(\theta/2) \mathbf{1} - i \sin(\theta/2) (\hat{n} \cdot \vec{\sigma})$$
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Result<Self> {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < 1e-12 {
            return Ok(Self::identity());
        }
        let (s, c) = (angle * 0.5).sin_cos();
        let inv_norm = s / norm;
        Ok(Self {
            a0: c,
            a1: axis[0] * inv_norm,
            a2: axis[1] * inv_norm,
            a3: axis[2] * inv_norm,
        })
    }

    /// Non-Abelian group multiplication $U_1 \cdot U_2$ using Hamilton quaternion algebra.
    pub fn multiply(&self, other: &Self) -> Self {
        let a0 = self.a0 * other.a0 - self.a1 * other.a1 - self.a2 * other.a2 - self.a3 * other.a3;
        let a1 = self.a0 * other.a1 + self.a1 * other.a0 + self.a2 * other.a3 - self.a3 * other.a2;
        let a2 = self.a0 * other.a2 - self.a1 * other.a3 + self.a2 * other.a0 + self.a3 * other.a1;
        let a3 = self.a0 * other.a3 + self.a1 * other.a2 - self.a2 * other.a1 + self.a3 * other.a0;

        Self { a0, a1, a2, a3 }
    }

    /// Hermitian conjugate / inverse $U^{-1} = U^\dagger = a_0 - a_1 \mathbf{i} - a_2 \mathbf{j} - a_3 \mathbf{k}$.
    pub fn inverse(&self) -> Self {
        Self {
            a0: self.a0,
            a1: -self.a1,
            a2: -self.a2,
            a3: -self.a3,
        }
    }

    /// Real normalized trace $\frac{1}{2} \operatorname{Re}\operatorname{Tr}(U) = a_0 \in [-1, 1]$.
    pub fn normalized_trace(&self) -> f64 {
        self.a0
    }

    /// Acts on a complex 2-spinor $\psi = (\psi_1, \psi_2) \in \mathbb{C}^2$:
    ///
    /// $$U = \begin{pmatrix} a_0 + i a_3 & -a_2 + i a_1 \\ a_2 + i a_1 & a_0 - i a_3 \end{pmatrix}$$
    ///
    /// Each component represented as `(real, imag)`.
    pub fn act_spinor(&self, psi: [(f64, f64); 2]) -> [(f64, f64); 2] {
        let (u0_r, u0_i) = (self.a0, self.a3);
        let (u1_r, u1_i) = (-self.a2, self.a1);
        let (u2_r, u2_i) = (self.a2, self.a1);
        let (u3_r, u3_i) = (self.a0, -self.a3);

        let (p0_r, p0_i) = psi[0];
        let (p1_r, p1_i) = psi[1];

        // out[0] = U[0,0]*p0 + U[0,1]*p1
        let out0_r = (u0_r * p0_r - u0_i * p0_i) + (u1_r * p1_r - u1_i * p1_i);
        let out0_i = (u0_r * p0_i + u0_i * p0_r) + (u1_r * p1_i + u1_i * p1_r);

        // out[1] = U[1,0]*p0 + U[1,1]*p1
        let out1_r = (u2_r * p0_r - u2_i * p0_i) + (u3_r * p1_r - u3_i * p1_i);
        let out1_i = (u2_r * p0_i + u2_i * p0_r) + (u3_r * p1_i + u3_i * p1_r);

        [(out0_r, out0_i), (out1_r, out1_i)]
    }
}

/// A lattice / graph non-Abelian $SU(2)$ Yang-Mills gauge field configuration.
#[derive(Debug, Clone)]
pub struct YangMillsLatticeGauge {
    /// Number of lattice vertices $N$.
    pub num_nodes: usize,
    /// Directed edges $(u, v)$.
    pub edges: Vec<(usize, usize)>,
    /// Gauge link variables $U_{uv} \in SU(2)$ associated with each directed edge.
    pub links: Vec<Su2Element>,
}

impl YangMillsLatticeGauge {
    /// Constructs a new lattice gauge field initialized to identity links (pure gauge vacuum).
    pub fn new(num_nodes: usize, edges: Vec<(usize, usize)>) -> Result<Self> {
        for &(u, v) in &edges {
            if u >= num_nodes || v >= num_nodes || u == v {
                return Err(Error::Msg(format!(
                    "Invalid edge ({u}, {v}) for num_nodes {num_nodes}"
                )));
            }
        }
        let num_edges = edges.len();
        let links = vec![Su2Element::identity(); num_edges];
        Ok(Self {
            num_nodes,
            edges,
            links,
        })
    }

    /// Sets the gauge link element $U_e$ on edge index `edge_idx`.
    /// Verifies links/edges agree with the (public, tamperable) topology.
    fn check_topology(&self) -> Result<()> {
        if self.links.len() != self.edges.len() {
            return Err(Error::Msg(format!(
                "gauge links length {} disagrees with {} edges",
                self.links.len(),
                self.edges.len()
            )));
        }
        for &(u, v) in &self.edges {
            if u >= self.num_nodes || v >= self.num_nodes {
                return Err(Error::IndexOutOfBounds {
                    op: "YangMillsLatticeGauge::check_topology",
                    index: u.max(v) as isize,
                    dim_size: self.num_nodes,
                });
            }
        }
        Ok(())
    }

    pub fn set_link(&mut self, edge_idx: usize, element: Su2Element) -> Result<()> {
        if edge_idx >= self.links.len() {
            return Err(Error::IndexOutOfBounds {
                op: "YangMillsLatticeGauge::set_link",
                index: edge_idx as isize,
                dim_size: self.links.len(),
            });
        }
        self.links[edge_idx] = element;
        Ok(())
    }

    /// Computes the Wilson loop operator along a closed path of oriented edges:
    ///
    /// $$W(C) = \frac{1}{2} \operatorname{Re}\operatorname{Tr}\left(\prod_{k} U_{e_k}^{\sigma_k}\right)$$
    ///
    /// where `orientations[k] = true` for forward traversal and `false` for backward (inverse link $U^\dagger$).
    pub fn wilson_loop(&self, cycle_edges: &[usize], forward: &[bool]) -> Result<f64> {
        if cycle_edges.is_empty() || cycle_edges.len() != forward.len() {
            return Err(Error::Msg(
                "Cycle edges and forward flags length mismatch or empty".to_string(),
            ));
        }

        let mut product = Su2Element::identity();
        for (&e, &fwd) in cycle_edges.iter().zip(forward) {
            if e >= self.links.len() {
                return Err(Error::IndexOutOfBounds {
                    op: "wilson_loop",
                    index: e as isize,
                    dim_size: self.links.len(),
                });
            }
            let link = if fwd {
                self.links[e]
            } else {
                self.links[e].inverse()
            };
            product = product.multiply(&link);
        }

        Ok(product.normalized_trace())
    }

    /// Applies local $SU(2)$ gauge transformation parameterized by vertex elements $\Omega_u \in SU(2)$:
    ///
    /// $$U_{uv} \leftarrow \Omega_u U_{uv} \Omega_v^\dagger$$
    pub fn apply_gauge_transformation(&mut self, omegas: &[Su2Element]) -> Result<()> {
        if omegas.len() != self.num_nodes {
            return Err(Error::ShapeMismatch {
                op: "apply_gauge_transformation",
                expected: vec![self.num_nodes],
                got: vec![omegas.len()],
            });
        }
        self.check_topology()?;

        for (idx, &(u, v)) in self.edges.iter().enumerate() {
            let u_orig = self.links[idx];
            let omega_u = omegas[u];
            let omega_v_inv = omegas[v].inverse();

            // U'_{uv} = Omega_u * U_{uv} * Omega_v^\dagger
            self.links[idx] = omega_u.multiply(&u_orig).multiply(&omega_v_inv);
        }

        Ok(())
    }

    /// Computes the Yang-Mills lattice action:
    ///
    /// $$S_{\text{YM}} = \beta \sum_{p} \left(1 - W(p)\right)$$
    pub fn yang_mills_action(&self, plaquettes: &[Vec<(usize, bool)>], beta: f64) -> Result<f64> {
        let mut action = 0.0;
        for plaquette in plaquettes {
            let edges: Vec<usize> = plaquette.iter().map(|&(e, _)| e).collect();
            let fwds: Vec<bool> = plaquette.iter().map(|&(_, f)| f).collect();
            let w = self.wilson_loop(&edges, &fwds)?;
            action += 1.0 - w;
        }
        Ok(beta * action)
    }

    /// Executes non-Abelian gauge-equivariant convolution over vertex spinor fields:
    ///
    /// $$\psi_u^{(l+1)} = \sum_{v \in \mathcal{N}(u)} U_{uv} \psi_v^{(l)}$$
    pub fn gauge_conv_step(&self, spinors: &[[(f64, f64); 2]]) -> Result<Vec<[(f64, f64); 2]>> {
        if spinors.len() != self.num_nodes {
            return Err(Error::ShapeMismatch {
                op: "gauge_conv_step",
                expected: vec![self.num_nodes],
                got: vec![spinors.len()],
            });
        }
        self.check_topology()?;

        let mut out = vec![[(0.0, 0.0), (0.0, 0.0)]; self.num_nodes];

        for (idx, &(u, v)) in self.edges.iter().enumerate() {
            let link = self.links[idx];
            // Transport spinor from v to u along edge (u, v)
            let transported = link.act_spinor(spinors[v]);

            out[u][0].0 += transported[0].0;
            out[u][0].1 += transported[0].1;
            out[u][1].0 += transported[1].0;
            out[u][1].1 += transported[1].1;
        }

        Ok(out)
    }
}
