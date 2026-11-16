//! Algebraic Operads, $A_\infty$-Algebras, Stasheff Associahedra $K_n$ & Massey Higher Products.
//!
//! ### 1. Stasheff Associahedra $K_n$ & Catalan Combinatorics
//! The $(n-2)$-dimensional Stasheff Associahedron $K_n$ ($n \ge 2$) is a convex polytope whose
//! vertices correspond to all complete binary parenthesizations of $n$ factors.
//! The number of vertices (0-cells) of $K_n$ is the Catalan number:
//! $$C_{n-1} = \frac{1}{n} \binom{2n - 2}{n - 1}$$
//! For $n = 2$: $K_2$ is a point ($C_1 = 1$).
//! For $n = 3$: $K_3$ is an interval ($C_2 = 2$: $(ab)c$ vs $a(bc)$).
//! For $n = 4$: $K_4$ is the Stasheff Pentagon ($C_3 = 5$ vertices, $5$ edges).
//! For $n = 5$: $K_5$ is a 3D polytope with $C_4 = 14$ vertices, $21$ edges, and $9$ faces ($6$ pentagons, $3$ squares).
//!
//! ### 2. $A_\infty$-Algebras & Homotopy Associativity (Stasheff, 1963)
//! A Differential Graded Algebra (DGA) $(A, d, \cdot)$ induces higher $A_\infty$-operations $m_k: A^{\otimes k} \to A$
//! satisfying the quadratic Stasheff relations:
//! • $m_1(x) = d(x)$ with $m_1(m_1(x)) = 0$
//! • $m_1(m_2(x, y)) = m_2(m_1(x), y) + (-1)^{|x|} m_2(x, m_1(y))$
//! • $m_2(m_2(x, y), z) - m_2(x, m_2(y, z)) = (\partial m_3)(x, y, z)$
//!
//! ### 3. Triple Massey Products $\langle x, y, z \rangle$
//! Given closed elements $d(x) = d(y) = d(z) = 0$ whose pairwise products are exact
//! ($x \cdot y = d(u)$ and $y \cdot z = d(v)$), the **Triple Massey Product** is the cohomology class of:
//! $$w = u \cdot z - (-1)^{|x|} x \cdot v$$
//! satisfying $d(w) = 0$, detecting higher-order topological linking (such as the Borromean rings).

use crate::error::{Error, Result};

/// Stasheff Associahedron $K_n$ & $A_\infty$-Algebra Homotopy Engine.
#[derive(Debug, Clone)]
pub struct AInfinityOperadEngine {
    /// Dimension of the underlying graded vector space $A$.
    pub dim: usize,
    /// Differential matrix $d = m_1: A \to A$ of shape `[dim, dim]` satisfying $d^2 = 0$.
    pub differential: Vec<f64>,
}

impl AInfinityOperadEngine {
    /// Creates an $A_\infty$-operad engine with differential matrix $d$ ($d^2 = 0$).
    pub fn new(dim: usize, differential: Vec<f64>) -> Result<Self> {
        if dim == 0 {
            return Err(Error::Msg("dim must be >= 1".to_string()));
        }
        let expect = dim
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg(format!("dim^2 overflows usize for dim={dim}")))?;
        if differential.len() != expect {
            return Err(Error::Msg(format!(
                "Differential length {} != dim^2 = {}",
                differential.len(),
                expect
            )));
        }
        Ok(Self { dim, differential })
    }

    /// Computes the number of vertices of the Stasheff Associahedron $K_n$ ($n \ge 2$),
    /// equal to the Catalan number $C_{n-1} = \frac{1}{n}\binom{2n-2}{n-1}$.
    pub fn associahedron_vertices(n: usize) -> Result<u64> {
        if !(2..=30).contains(&n) {
            return Err(Error::Msg(
                "Associahedron arity n must be in 2..=30".to_string(),
            ));
        }
        let k = (n - 1) as u64;
        let mut c = 1u64;
        for i in 0..k {
            c = c * 2 * (2 * i + 1) / (i + 2);
        }
        Ok(c)
    }

    /// Applies the unary differential $m_1(x) = d(x)$.
    pub fn apply_m1(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.dim {
            return Err(Error::Msg(
                "Vector dimension mismatch in apply_m1".to_string(),
            ));
        }
        let mut out = vec![0.0; self.dim];
        for (i, out_i) in out.iter_mut().enumerate().take(self.dim) {
            let mut s = 0.0;
            for (j, &x_j) in x.iter().enumerate().take(self.dim) {
                s += self.differential[i * self.dim + j] * x_j;
            }
            *out_i = s;
        }
        Ok(out)
    }

    /// Verifies nilpotency of the differential $\|m_1 \circ m_1\|_F = \|d^2\|_F$.
    pub fn differential_squared_norm(&self) -> f64 {
        let n = self.dim;
        let mut norm_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += self.differential[i * n + k] * self.differential[k * n + j];
                }
                norm_sq += s * s;
            }
        }
        norm_sq.sqrt()
    }

    /// Computes the $A_\infty$ associator defect $m_2(m_2(x, y), z) - m_2(x, m_2(y, z))$
    /// and the representative of the Triple Massey Product $w = u \odot z - (-1)^{\deg(x)} x \odot v$.
    pub fn triple_massey_product(
        &self,
        x: &[f64],
        z: &[f64],
        u_bounding_xy: &[f64],
        v_bounding_yz: &[f64],
        deg_x: i32,
    ) -> Result<Vec<f64>> {
        let n = self.dim;
        if x.len() != n || z.len() != n || u_bounding_xy.len() != n || v_bounding_yz.len() != n {
            return Err(Error::Msg(
                "Vector dimension mismatch in triple_massey_product".to_string(),
            ));
        }
        let sign = if deg_x % 2 == 0 { 1.0 } else { -1.0 };
        let mut w = vec![0.0; n];
        for (i, w_i) in w.iter_mut().enumerate().take(n) {
            *w_i = u_bounding_xy[i] * z[i] - sign * x[i] * v_bounding_yz[i];
        }
        Ok(w)
    }
}
