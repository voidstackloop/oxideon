//! Higher Category Theory, $(\infty, 1)$-Quasi-Categories, Kan Complexes & Simplicial Horn Filling.
//!
//! # Mathematical Foundations (Joyal & Lurie, *Higher Topos Theory*)
//!
//! ### 1. **Simplicial Sets & Face Maps**:
//! An $n$-simplex $\sigma = [v_0, v_1, \dots, v_n]$ has $n + 1$ face maps $d_i: X_n \to X_{n-1}$ ($0 \le i \le n$)
//! obtained by omitting the $i$-th vertex:
//! $$d_i([v_0, \dots, v_n]) = [v_0, \dots, \hat{v}_i, \dots, v_n]$$
//! satisfying the fundamental simplicial identity:
//! $$d_i \circ d_j = d_{j-1} \circ d_i \quad \text{for all } i < j$$
//! which guarantees $\partial^2 = 0$ on simplicial chain complexes.
//!
//! ### 2. **Horns $\Lambda_k^n \subset \Delta^n$ & Quasi-Categories**:
//! The $k$-th horn $\Lambda_k^n$ is the union of all faces of $\Delta^n$ except the $k$-th face $d_k$.
//! • **$(\infty, 1)$-Quasi-Category**: Every *inner horn* $\Lambda_k^n$ ($0 < k < n$) admits a filler $\Delta^n$.
//! For $n = 2, k = 1$, the inner horn $\Lambda_1^2$ consists of 1-morphisms $f = d_2(\sigma): v_0 \to v_1$
//! and $g = d_0(\sigma): v_1 \to v_2$ with matching target/source $d_0(f) = v_1 = d_1(g)$.
//! Filling $\Lambda_1^2$ produces a 2-simplex $\sigma = [v_0, v_1, v_2]$ whose missing face $d_1(\sigma) = [v_0, v_2]$
//! is the homotopy composition $g \circ f$!
//! • **Kan Complex ($\infty$-Groupoid)**: Every horn (both inner $0 < k < n$ and outer $k = 0, n$) admits a filler,
//! meaning all 1-morphisms are weakly invertible up to higher homotopy.

use crate::error::{Error, Result};

/// A weighted $n$-simplex $\sigma = [v_0, v_1, \dots, v_n]$ equipped with a morphism feature payload.
#[derive(Debug, Clone, PartialEq)]
pub struct HigherSimplex {
    /// Ordered vertex sequence $[v_0, v_1, \dots, v_n]$ of length $n + 1$.
    pub vertices: Vec<usize>,
    /// Morphism / homotopy feature vector associated with this simplex.
    pub payload: Vec<f64>,
}

impl HigherSimplex {
    /// Creates an $n$-simplex from $n + 1$ vertices and a feature payload.
    pub fn new(vertices: Vec<usize>, payload: Vec<f64>) -> Result<Self> {
        if vertices.is_empty() {
            return Err(Error::Msg(
                "HigherSimplex requires at least 1 vertex (0-simplex)".to_string(),
            ));
        }
        Ok(Self { vertices, payload })
    }

    /// Simplicial dimension $n = |\text{vertices}| - 1$ (saturates to 0 for
    /// the invalid empty simplex instead of underflowing).
    #[inline]
    pub fn dimension(&self) -> usize {
        self.vertices.len().saturating_sub(1)
    }

    /// Applies the $i$-th simplicial face map $d_i: X_n \to X_{n-1}$ ($0 \le i \le n$).
    pub fn face(&self, i: usize) -> Result<Self> {
        if self.vertices.is_empty() {
            return Err(Error::Msg(
                "Cannot take face of an empty simplex".to_string(),
            ));
        }
        let n = self.dimension();
        if n == 0 {
            return Err(Error::Msg("Cannot take face of a 0-simplex".to_string()));
        }
        if i > n {
            return Err(Error::Msg(format!(
                "Face index {i} out of bounds for {n}-simplex"
            )));
        }
        let mut new_verts = Vec::with_capacity(n);
        for (idx, &v) in self.vertices.iter().enumerate() {
            if idx != i {
                new_verts.push(v);
            }
        }
        Ok(Self {
            vertices: new_verts,
            payload: self.payload.clone(),
        })
    }

    /// Applies the $i$-th simplicial degeneracy map $s_i: X_n \to X_{n+1}$ ($0 \le i \le n$),
    /// duplicating vertex $v_i$ to represent an identity $(n+1)$-morphism.
    pub fn degeneracy(&self, i: usize) -> Result<Self> {
        if self.vertices.is_empty() {
            return Err(Error::Msg(
                "Cannot take degeneracy of an empty simplex".to_string(),
            ));
        }
        let n = self.dimension();
        if i > n {
            return Err(Error::Msg(format!(
                "Degeneracy index {i} out of bounds for {n}-simplex"
            )));
        }
        let mut new_verts = Vec::with_capacity(n + 2);
        for (idx, &v) in self.vertices.iter().enumerate() {
            new_verts.push(v);
            if idx == i {
                new_verts.push(v);
            }
        }
        Ok(Self {
            vertices: new_verts,
            payload: self.payload.clone(),
        })
    }

    /// Verifies the fundamental simplicial identity $d_i(d_j(\sigma)) == d_{j-1}(d_i(\sigma))$ for $i < j$.
    pub fn verify_simplicial_identity(&self, i: usize, j: usize) -> Result<bool> {
        if i >= j {
            return Err(Error::Msg("Simplicial identity requires i < j".to_string()));
        }
        let lhs = self.face(j)?.face(i)?;
        let rhs = self.face(i)?.face(j - 1)?;
        Ok(lhs.vertices == rhs.vertices)
    }
}

/// $(\infty, 1)$-Quasi-Category & Kan Complex Horn-Filling Engine.
#[derive(Debug, Clone)]
pub struct QuasiCategoryEngine {
    /// Homotopy coherence decay factor $\gamma \in (0, 1]$ for higher composition.
    pub coherence_factor: f64,
}

impl QuasiCategoryEngine {
    /// Creates a new quasi-category engine.
    pub fn new(coherence_factor: f64) -> Result<Self> {
        if coherence_factor <= 0.0 || coherence_factor > 1.0 {
            return Err(Error::Msg("coherence_factor must be in (0, 1]".to_string()));
        }
        Ok(Self { coherence_factor })
    }

    /// Fills an inner 2-horn $\Lambda_1^2$ given two composable 1-simplices:
    /// • $f = [x, y]$ ($d_2$ face)
    /// • $g = [y, z]$ ($d_0$ face)
    ///
    /// Returns `(filler_2_simplex, composite_1_simplex)`:
    /// • `filler_2_simplex`: $\sigma = [x, y, z]$
    /// • `composite_1_simplex`: $d_1(\sigma) = [x, z] = g \circ f$
    pub fn fill_inner_horn_2(
        &self,
        f: &HigherSimplex,
        g: &HigherSimplex,
    ) -> Result<(HigherSimplex, HigherSimplex)> {
        if f.dimension() != 1 || g.dimension() != 1 {
            return Err(Error::Msg(
                "Inner 2-hornLambda_1^2 requires two 1-simplices".to_string(),
            ));
        }
        if f.vertices[1] != g.vertices[0] {
            return Err(Error::Msg(format!(
                "Non-composable 1-morphisms: target(f) = {} != source(g) = {}",
                f.vertices[1], g.vertices[0]
            )));
        }
        if f.payload.len() != g.payload.len() {
            return Err(Error::Msg(
                "Payload dimension mismatch in horn filler".to_string(),
            ));
        }

        let x = f.vertices[0];
        let y = f.vertices[1];
        let z = g.vertices[1];

        // Homotopy composition combines morphism payloads scaled by coherence factor
        let comp_payload: Vec<f64> = f
            .payload
            .iter()
            .zip(g.payload.iter())
            .map(|(&pf, &pg)| self.coherence_factor * (pf + pg))
            .collect();

        let filler = HigherSimplex::new(vec![x, y, z], comp_payload.clone())?;
        let composite = filler.face(1)?; // d_1([x, y, z]) = [x, z]
        Ok((filler, composite))
    }

    /// Fills an outer Kan horn $\Lambda_0^2$ (given $f = [x, y]$ and $h = [x, z]$, solves for $g = [y, z] \simeq h \circ f^{-1}$)
    /// demonstrating weak invertibility in a Kan complex ($\infty$-groupoid).
    pub fn fill_outer_kan_horn_0(
        &self,
        f: &HigherSimplex,
        h: &HigherSimplex,
    ) -> Result<(HigherSimplex, HigherSimplex)> {
        if f.dimension() != 1 || h.dimension() != 1 {
            return Err(Error::Msg(
                "Outer 2-horn Lambda_0^2 requires two 1-simplices".to_string(),
            ));
        }
        if f.vertices[0] != h.vertices[0] {
            return Err(Error::Msg(format!(
                "Outer horn Lambda_0^2 requires source(f) = {} == source(h) = {}",
                f.vertices[0], h.vertices[0]
            )));
        }
        if f.payload.len() != h.payload.len() {
            return Err(Error::Msg(
                "Payload dimension mismatch in Kan horn filler".to_string(),
            ));
        }

        let x = f.vertices[0];
        let y = f.vertices[1];
        let z = h.vertices[1];

        // Solve for missing face d_0([x, y, z]) = [y, z] such that coherence * (f + g) = h
        let g_payload: Vec<f64> = f
            .payload
            .iter()
            .zip(h.payload.iter())
            .map(|(&pf, &ph)| (ph / self.coherence_factor) - pf)
            .collect();

        let filler = HigherSimplex::new(vec![x, y, z], h.payload.clone())?;
        let retracted_face = HigherSimplex::new(vec![y, z], g_payload)?;
        Ok((filler, retracted_face))
    }

    /// Computes the homotopy associativity residual between $(h \circ g) \circ f$ and $h \circ (g \circ f)$
    /// across three composable 1-morphisms $f: w \to x$, $g: x \to y$, $h: y \to z$, constructing
    /// the witnessing 3-simplex tetrahedron $[w, x, y, z]$.
    pub fn associator_3_simplex(
        &self,
        f: &HigherSimplex,
        g: &HigherSimplex,
        h: &HigherSimplex,
    ) -> Result<(HigherSimplex, f64)> {
        let (_, gf) = self.fill_inner_horn_2(f, g)?;
        let (_, h_gf) = self.fill_inner_horn_2(&gf, h)?;

        let (_, hg) = self.fill_inner_horn_2(g, h)?;
        let (_, hg_f) = self.fill_inner_horn_2(f, &hg)?;

        let mut max_diff = 0.0_f64;
        for (&a, &b) in h_gf.payload.iter().zip(hg_f.payload.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }

        let w = f.vertices[0];
        let x = f.vertices[1];
        let y = g.vertices[1];
        let z = h.vertices[1];
        let tetra = HigherSimplex::new(vec![w, x, y, z], h_gf.payload)?;
        Ok((tetra, max_diff))
    }
}
