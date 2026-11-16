//! Donaldson-Thomas (DT) Curve Counting Invariants, MacMahon 3D Partition Function & MNOP GW/DT Correspondence.
//!
//! ### 1. Degree-0 Donaldson-Thomas Partition Function on Calabi-Yau 3-Folds
//! By the Maulik-Nekrasov-Okounkov-Pandharipande (MNOP) Theorem, the degree-$0$ Donaldson-Thomas partition function
//! counting ideal sheaves of points on a smooth threefold $X$ is governed by **MacMahon's 3D Plane Partition Function**:
//! $$M(q) = \prod_{n=1}^\infty \frac{1}{(1 - q^n)^n} = 1 + q + 3 q^2 + 6 q^3 + 13 q^4 + 24 q^5 + 48 q^6 + \cdots$$
//! And on a Calabi-Yau 3-fold $X$ with topological Euler characteristic $\chi(X)$, $Z_{\text{DT}, 0}(X, q) = M(-q)^{\chi(X)}$.

use crate::error::{Error, Result};

/// Donaldson-Thomas (DT) 3D Plane Partition & MNOP Calabi-Yau Engine.
#[derive(Debug, Clone)]
pub struct DonaldsonThomasEngine {
    /// Precomputed 3D plane partition counts $P_{3D}(0..=N)$.
    pub macmahon_coeffs: Vec<u64>,
}

impl DonaldsonThomasEngine {
    /// Computes MacMahon's 3D plane partition numbers $P_{3D}(0..=N)$ by expanding $\prod_{n=1}^N (1 - q^n)^{-n}$.
    pub fn new(max_n: usize) -> Result<Self> {
        if !(1..=30).contains(&max_n) {
            return Err(Error::Msg("max_n must be in 1..=30".to_string()));
        }
        let mut poly = vec![0_u64; max_n + 1];
        poly[0] = 1;
        for n in 1..=max_n {
            // Divide by (1 - q^n) n times
            for _ in 0..n {
                for k in n..=max_n {
                    poly[k] += poly[k - n];
                }
            }
        }
        Ok(Self {
            macmahon_coeffs: poly,
        })
    }

    /// Returns the exact 3D plane partition count $P_{3D}(n)$.
    pub fn plane_partition_count(&self, n: usize) -> Result<u64> {
        self.macmahon_coeffs
            .get(n)
            .copied()
            .ok_or_else(|| Error::Msg("Index n out of range".to_string()))
    }
}
