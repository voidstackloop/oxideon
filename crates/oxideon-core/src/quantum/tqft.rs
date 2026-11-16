//! Topological Quantum Field Theory (TQFT), Witten-Reshetikhin-Turaev $SU(2)_k$
//! Chern-Simons Invariants, Modular $S$-Matrices & Verlinde Fusion Rules.
//!
//! ### 1. $SU(2)_k$ Modular Tensor Category & Quantum Dimensions
//! At integer Chern-Simons level $k \ge 1$, the admissible topological anyon charges (integrable
//! representations of the affine Lie algebra $\widehat{\mathfrak{su}}(2)_k$) are indexed by
//! $a \in \{0, 1, \dots, k\}$ (corresponding to half-integer spins $j = a / 2$).
//! With quantum deformation parameter $q = \exp\left(\frac{2\pi i}{k + 2}\right)$, the **Quantum Dimension** of charge $a$ is:
//! $$d_a = [a + 1]_q = \frac{\sin\left(\frac{\pi (a + 1)}{k + 2}\right)}{\sin\left(\frac{\pi}{k + 2}\right)}$$
//! And the **Total Quantum Dimension** is:
//! $$\mathcal{D} = \sqrt{\sum_{a=0}^k d_a^2} = \sqrt{\frac{k + 2}{2}} \frac{1}{\sin\left(\frac{\pi}{k + 2}\right)}$$
//!
//! ### 2. Modular $S$-Matrix & Verlinde Formula (1988)
//! The $(k + 1) \times (k + 1)$ symmetric unitary modular $S$-matrix has entries:
//! $$S_{a, b} = \sqrt{\frac{2}{k + 2}} \sin\left(\frac{\pi (a + 1)(b + 1)}{k + 2}\right)$$
//! By **Verlinde's Theorem**, the modular $S$-matrix diagonalizes the non-Abelian anyon fusion rules
//! $a \times b = \sum_{c=0}^k N_{a b}^c \, c$:
//! $$N_{a b}^c = \sum_{x=0}^k \frac{S_{a, x} S_{b, x} S_{c, x}}{S_{0, x}}$$
//!
//! ### 3. Witten-Reshetikhin-Turaev 3-Manifold Partition Functions
//! • $Z(S^3) = S_{0, 0} = \frac{1}{\mathcal{D}}$
//! • $Z(S^2 \times S^1) = 1$
//! • $Z(T^3) = k + 1$

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// $SU(2)_k$ Chern-Simons Topological Quantum Field Theory (TQFT) Engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Su2ChernSimonsTqft {
    /// Integer Chern-Simons level $k \ge 1$ ($k = 2$ for Ising anyons, $k = 3$ for Fibonacci anyons).
    pub level_k: usize,
}

impl Su2ChernSimonsTqft {
    /// Constructs an $SU(2)_k$ TQFT engine at level $k \ge 1$.
    pub fn new(level_k: usize) -> Result<Self> {
        if !(1..=256).contains(&level_k) {
            return Err(Error::Msg(format!(
                "Chern-Simons level_k must be in 1..=256, got {level_k}"
            )));
        }
        Ok(Self { level_k })
    }

    /// Re-checks the construction bound: level_k is public and tamperable,
    /// and every entry below shifts, allocates, or loops on it.
    fn check_level(&self) -> Result<()> {
        if !(1..=256).contains(&self.level_k) {
            return Err(Error::Msg(format!(
                "Chern-Simons level_k must be in 1..=256, got {}",
                self.level_k
            )));
        }
        Ok(())
    }

    /// Number of simple anyon sectors (integrable representations) $k + 1$.
    #[inline]
    pub fn num_anyons(&self) -> Result<usize> {
        self.check_level()?;
        Ok(self.level_k + 1)
    }

    /// Computes the quantum dimension $d_a = [a + 1]_q$ of anyon charge $a \in \{0, \dots, k\}$.
    pub fn quantum_dimension(&self, a: usize) -> Result<f64> {
        if a > self.level_k {
            return Err(Error::Msg(format!(
                "Anyon index {a} exceeds level k = {}",
                self.level_k
            )));
        }
        self.check_level()?;
        let denom = (PI / ((self.level_k + 2) as f64)).sin();
        let numer = (PI * ((a + 1) as f64) / ((self.level_k + 2) as f64)).sin();
        Ok(numer / denom)
    }

    /// Computes the Total Quantum Dimension $\mathcal{D} = \sqrt{\sum_{a=0}^k d_a^2}$.
    pub fn total_quantum_dimension(&self) -> Result<f64> {
        self.check_level()?;
        let mut sum_sq = 0.0;
        for a in 0..=self.level_k {
            sum_sq += self.quantum_dimension(a)?.powi(2);
        }
        Ok(sum_sq.sqrt())
    }

    /// Computes the $(k+1) \times (k+1)$ modular $S$-matrix flattened row-major.
    pub fn modular_s_matrix(&self) -> Result<Vec<f64>> {
        self.check_level()?;
        let n = self.level_k + 1;
        let pref = (2.0 / ((self.level_k + 2) as f64)).sqrt();
        let mut s = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                let angle = PI * ((a + 1) * (b + 1)) as f64 / ((self.level_k + 2) as f64);
                s[a * n + b] = pref * angle.sin();
            }
        }
        Ok(s)
    }

    /// Computes the fusion multiplicity $N_{a b}^c$ via the exact Verlinde Formula:
    ///
    /// $$N_{a b}^c = \sum_{x=0}^k \frac{S_{a, x} S_{b, x} S_{c, x}}{S_{0, x}}$$
    pub fn verlinde_fusion_coefficient(&self, a: usize, b: usize, c: usize) -> Result<usize> {
        if a > self.level_k || b > self.level_k || c > self.level_k {
            return Err(Error::Msg(
                "Anyon index out of range in Verlinde formula".to_string(),
            ));
        }
        self.check_level()?;
        let n = self.level_k + 1;
        let s = self.modular_s_matrix()?;
        let mut sum = 0.0;
        for x in 0..n {
            let s0x = s[x];
            if s0x.abs() > 1e-14 {
                sum += (s[a * n + x] * s[b * n + x] * s[c * n + x]) / s0x;
            }
        }
        Ok(sum.round().max(0.0) as usize)
    }

    /// Computes Witten's 3-sphere partition function $Z(S^3) = S_{0, 0} = \frac{1}{\mathcal{D}}$.
    pub fn partition_function_s3(&self) -> Result<f64> {
        Ok(1.0 / self.total_quantum_dimension()?)
    }
}
