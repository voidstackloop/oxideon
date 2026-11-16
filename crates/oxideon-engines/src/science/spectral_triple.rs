//! Non-Commutative Geometry, Connes Spectral Triples $(\mathcal{A}, \mathcal{H}, D)$ & Connes Distance Formula (Connes, 1994).
//!
//! ### 1. Connes Spectral Triple $(\mathcal{A}, \mathcal{H}, D)$
//! In Alain Connes' Non-Commutative Geometry, a Riemannian spin manifold (or discrete quantum space) is encoded
//! algebraically by a **Spectral Triple** $(\mathcal{A}, \mathcal{H}, D)$ where:
//! • $\mathcal{A}$ is an involutive algebra of bounded operators acting on a Hilbert space $\mathcal{H}$,
//! • $D = D^*$ is an unbounded self-adjoint **Dirac Operator** with compact resolvent $(I + D^2)^{-1/2}$.
//!
//! ### 2. Connes Non-Commutative Metric Distance Formula
//! Without referencing points or paths, the geodesic distance between any two states (or points) $p, q$ is recovered
//! purely from the operator norm of the commutator $[D, a] = D a - a D$:
//! $$d_{\text{Connes}}(p, q) = \sup_{a \in \mathcal{A}} \left\{ |a(p) - a(q)| \; : \; \|[D, a]\|_{\text{op}} \le 1 \right\}$$

use crate::error::{Error, Result};

/// Connes Non-Commutative Spectral Triple $(\mathcal{A}, \mathcal{H}, D)$ Engine.
#[derive(Debug, Clone)]
pub struct ConnesSpectralTriple {
    /// Dimension $N$ of the finite Hilbert space $\mathcal{H} = \mathbb{C}^N$.
    pub dim: usize,
    /// Self-adjoint Dirac operator $D \in \mathbb{R}^{N \times N}$ (symmetric flattened matrix).
    pub dirac_operator: Vec<f64>,
}

impl ConnesSpectralTriple {
    /// Constructs a finite Connes spectral triple from a symmetric $N \times N$ Dirac operator $D$.
    pub fn new(dim: usize, dirac_operator: Vec<f64>) -> Result<Self> {
        if dim < 2 {
            return Err(Error::Msg(
                "ConnesSpectralTriple requires dim >= 2".to_string(),
            ));
        }
        let expect = dim
            .checked_mul(dim)
            .ok_or_else(|| Error::Msg(format!("dim^2 overflows usize for dim={dim}")))?;
        if dirac_operator.len() != expect {
            return Err(Error::Msg(format!(
                "Dirac operator length {} != dim^2 = {expect}",
                dirac_operator.len()
            )));
        }
        Ok(Self {
            dim,
            dirac_operator,
        })
    }

    /// Computes the Frobenius / Lipschitz seminar norm $\|[D, \pi(a)]\|_F$ of the commutator $[D, \pi(a)]$
    /// for a diagonal algebra element $\pi(a) = \operatorname{diag}(a_0, \dots, a_{N-1})$.
    /// Explicitly, $[D, \operatorname{diag}(a)]_{ij} = D_{ij} (a_j - a_i)$.
    pub fn commutator_frobenius_norm(&self, a: &[f64]) -> Result<f64> {
        let n = self.dim;
        if a.len() != n {
            return Err(Error::Msg("Algebra element dimension mismatch".to_string()));
        }
        let mut sum_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                let c_ij = self.dirac_operator[i * n + j] * (a[j] - a[i]);
                sum_sq += c_ij * c_ij;
            }
        }
        Ok(sum_sq.sqrt())
    }

    /// Computes the exact 1D chain Connes spectral distance $d_{\text{Connes}}(p, q) = \sum_{k=p}^{q-1} |D_{k, k+1}|^{-1}$
    /// along a tridiagonal Dirac operator with hopping amplitudes $D_{k, k+1} \neq 0$.
    pub fn chain_connes_distance(&self, p: usize, q: usize) -> Result<f64> {
        let n = self.dim;
        if p >= n || q >= n {
            return Err(Error::Msg(
                "Site index out of bounds in chain_connes_distance".to_string(),
            ));
        }
        if p == q {
            return Ok(0.0);
        }
        let (lo, hi) = if p < q { (p, q) } else { (q, p) };
        let mut dist = 0.0;
        for k in lo..hi {
            let hop = self.dirac_operator[k * n + (k + 1)].abs();
            if hop < 1e-15 {
                return Ok(f64::INFINITY);
            }
            dist += 1.0 / hop;
        }
        Ok(dist)
    }

    /// Evaluates the Zeta-regularized spectral action trace $\operatorname{Tr}(I + D^2 / \Lambda^2)^{-1}$
    /// (Chamseddine-Connes Spectral Action Principle) for diagonal entries of $D^2$.
    pub fn chamseddine_connes_spectral_trace(&self, cutoff_lambda: f64) -> Result<f64> {
        if cutoff_lambda <= 0.0 {
            return Err(Error::Msg("Cutoff scale Lambda must be > 0".to_string()));
        }
        let n = self.dim;
        let l2 = cutoff_lambda * cutoff_lambda;
        let mut trace = 0.0;
        for i in 0..n {
            let mut d2_ii = 0.0;
            for k in 0..n {
                let d_ik = self.dirac_operator[i * n + k];
                d2_ii += d_ik * d_ik;
            }
            trace += 1.0 / (1.0 + d2_ii / l2);
        }
        Ok(trace)
    }
}
