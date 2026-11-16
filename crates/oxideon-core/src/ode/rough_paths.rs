//! Rough Paths Theory, Iterated-Integral Path Signatures, Chen's Identity & Lévy Stochastic Area (Lyons, 1998).
//!
//! ### 1. Truncated Path Signature in the Tensor Algebra $T^{(2)}(\mathbb{R}^d)$
//! Given a continuous piecewise-linear path $X: [s, t] \to \mathbb{R}^d$, its step-2 signature is:
//! $$S_2(X)_{s, t} = \left(1, \; \mathbf{X}^1_{s, t} \in \mathbb{R}^d, \; \mathbf{X}^2_{s, t} \in \mathbb{R}^{d \times d}\right)$$
//! where $\mathbf{X}^{1, i}_{s, t} = X^i_t - X^i_s$ and $\mathbf{X}^{2, ij}_{s, t} = \int_s^t (X^i_u - X^i_s) \, dX^j_u$.
//!
//! ### 2. Chen's Multiplicative Identity (1954)
//! For the concatenation $X * Y$ of two paths $X: [s, u] \to \mathbb{R}^d$ and $Y: [u, t] \to \mathbb{R}^d$:
//! $$S_2(X * Y) = S_2(X) \otimes S_2(Y)$$
//! Explicitly:
//! • $\mathbf{X}^1(X * Y) = \mathbf{X}^1(X) + \mathbf{X}^1(Y)$
//! • $\mathbf{X}^2(X * Y)_{ij} = \mathbf{X}^2(X)_{ij} + \mathbf{X}^2(Y)_{ij} + \mathbf{X}^1(X)_i \mathbf{X}^1(Y)_j$
//!
//! ### 3. Lévy Stochastic Area
//! The antisymmetric part of the second-level signature is the signed **Lévy Area** enclosed by the path and its chord:
//! $$A_{ij} = \frac{1}{2}\left(\mathbf{X}^{2, ij}_{s, t} - \mathbf{X}^{2, ji}_{s, t}\right)$$

use crate::error::{Error, Result};

/// Truncated Step-2 Rough Path Signature $(1, \mathbf{X}^1, \mathbf{X}^2) \in T^{(2)}(\mathbb{R}^d)$.
#[derive(Debug, Clone, PartialEq)]
pub struct RoughPathSignature {
    /// Ambient path dimension $d$.
    pub dim: usize,
    /// Level-1 increment vector $\mathbf{X}^1 \in \mathbb{R}^d$.
    pub level1: Vec<f64>,
    /// Level-2 iterated integral matrix $\mathbf{X}^2 \in \mathbb{R}^{d \times d}$ (flattened).
    pub level2: Vec<f64>,
}

impl RoughPathSignature {
    /// Computes the exact step-2 signature of a discrete multidimensional path `points` of shape `[N, d]`.
    pub fn from_path(points: &[Vec<f64>]) -> Result<Self> {
        if points.len() < 2 {
            return Err(Error::Msg("Path requires at least 2 points".to_string()));
        }
        let d = points[0].len();
        if d == 0 {
            return Err(Error::Msg("Path dimension must be >= 1".to_string()));
        }
        for pt in points {
            if pt.len() != d {
                return Err(Error::Msg(
                    "Inconsistent point dimension in path".to_string(),
                ));
            }
        }

        let mut sig = Self {
            dim: d,
            level1: vec![0.0; d],
            level2: vec![0.0; d * d],
        };

        for w in points.windows(2) {
            let p0 = &w[0];
            let p1 = &w[1];
            let dx: Vec<f64> = p1.iter().zip(p0.iter()).map(|(a, b)| a - b).collect();
            let seg = Self::from_linear_segment(&dx);
            sig = sig.chen_multiply(&seg)?;
        }
        Ok(sig)
    }

    /// Signature of a single straight line segment $\Delta x \in \mathbb{R}^d$:
    /// $\mathbf{X}^1_i = \Delta x_i$, $\mathbf{X}^2_{ij} = \frac{1}{2} \Delta x_i \Delta x_j$.
    pub fn from_linear_segment(dx: &[f64]) -> Self {
        let d = dx.len();
        let mut l2 = vec![0.0; d * d];
        for (i, &dxi) in dx.iter().enumerate().take(d) {
            for (j, &dxj) in dx.iter().enumerate().take(d) {
                l2[i * d + j] = 0.5 * dxi * dxj;
            }
        }
        Self {
            dim: d,
            level1: dx.to_vec(),
            level2: l2,
        }
    }

    /// Tensor-algebra multiplication $S(X) \otimes S(Y)$ via Chen's Identity.
    pub fn chen_multiply(&self, other: &Self) -> Result<Self> {
        if self.dim != other.dim {
            return Err(Error::Msg(
                "Signature dimension mismatch in chen_multiply".to_string(),
            ));
        }
        let d = self.dim;
        // Pub level buffers can be truncated post-construction: verify
        // agreement before allocating or indexing.
        if self.level1.len() != d || other.level1.len() != d {
            return Err(Error::Msg(format!(
                "chen_multiply level1 lengths ({}, {}) disagree with dim {d}",
                self.level1.len(),
                other.level1.len()
            )));
        }
        let l2len = d
            .checked_mul(d)
            .ok_or_else(|| Error::Msg(format!("chen_multiply level2 size {d} * {d} overflows")))?;
        if self.level2.len() != l2len || other.level2.len() != l2len {
            return Err(Error::Msg(format!(
                "chen_multiply level2 lengths ({}, {}) disagree with {d}*{d}",
                self.level2.len(),
                other.level2.len()
            )));
        }
        let mut l1 = vec![0.0; d];
        let mut l2 = vec![0.0; l2len];

        for (i, l1_i) in l1.iter_mut().enumerate().take(d) {
            *l1_i = self.level1[i] + other.level1[i];
            for j in 0..d {
                l2[i * d + j] = self.level2[i * d + j]
                    + other.level2[i * d + j]
                    + self.level1[i] * other.level1[j];
            }
        }
        Ok(Self {
            dim: d,
            level1: l1,
            level2: l2,
        })
    }

    /// Computes the signed Lévy stochastic area $A_{ij} = \frac{1}{2}(\mathbf{X}^2_{ij} - \mathbf{X}^2_{ji})$.
    pub fn levy_area(&self, i: usize, j: usize) -> Result<f64> {
        if i >= self.dim || j >= self.dim {
            return Err(Error::Msg(
                "Coordinate index out of range in levy_area".to_string(),
            ));
        }
        let l2len = self.dim.checked_mul(self.dim).ok_or_else(|| {
            Error::Msg(format!(
                "levy_area level2 size {} * {} overflows",
                self.dim, self.dim
            ))
        })?;
        if self.level2.len() != l2len {
            return Err(Error::Msg(format!(
                "levy_area level2 length {} disagrees with {}*{}",
                self.level2.len(),
                self.dim,
                self.dim
            )));
        }
        Ok(0.5 * (self.level2[i * self.dim + j] - self.level2[j * self.dim + i]))
    }
}
