//! Arkani-Hamed–Trnka Amplituhedron, Positive Grassmannian $G_+(k, n)$ & Plücker Positivity (Arkani-Hamed & Trnka, 2014).
//!
//! ### 1. Positive Grassmannian $G_+(2, n)$ & Plücker Relations
//! A $2 \times n$ matrix $C$ defines a point in the **Positive Grassmannian** $G_+(2, n)$ if all ordered
//! $2 \times 2$ maximal minors (Plücker coordinates $p_{ij} = C_{0,i} C_{1,j} - C_{0,j} C_{1,i}$ for $1 \le i < j \le n$)
//! are strictly positive ($p_{ij} > 0$) and satisfy the three-term **Grassmann-Plücker Relation**:
//! $$p_{ik} p_{jl} = p_{ij} p_{kl} + p_{il} p_{jk} \qquad (i < j < k < l)$$

use crate::error::{Error, Result};

/// Positive Grassmannian $G_+(2, n)$ & Amplituhedron Plücker Engine.
#[derive(Debug, Clone)]
pub struct AmplituhedronGrassmannianEngine {
    /// Number of external particles $n \ge 4$.
    pub num_particles: usize,
    /// Row 0 of the $2 \times n$ Grassmannian matrix $C$.
    pub row0: Vec<f64>,
    /// Row 1 of the $2 \times n$ Grassmannian matrix $C$.
    pub row1: Vec<f64>,
}

impl AmplituhedronGrassmannianEngine {
    /// Constructs a $G(2, n)$ matrix configuration.
    pub fn new(row0: Vec<f64>, row1: Vec<f64>) -> Result<Self> {
        let n = row0.len();
        if n < 4 || row1.len() != n {
            return Err(Error::Msg(
                "G_+(2, n) requires n >= 4 and equal row lengths".to_string(),
            ));
        }
        Ok(Self {
            num_particles: n,
            row0,
            row1,
        })
    }

    /// Computes the $2 \times 2$ Plücker minor $\langle i \, j \rangle = C_{0,i} C_{1,j} - C_{0,j} C_{1,i}$;
    /// both indices must be below `num_particles`.
    pub fn plucker_minor(&self, i: usize, j: usize) -> Result<f64> {
        if i >= self.num_particles || j >= self.num_particles {
            return Err(Error::Msg(format!(
                "Plücker indices ({i}, {j}) must be below the number of particles {}",
                self.num_particles
            )));
        }
        Ok(self.minor(i, j))
    }

    fn minor(&self, i: usize, j: usize) -> f64 {
        self.row0[i] * self.row1[j] - self.row0[j] * self.row1[i]
    }

    /// Verifies total positivity ($\langle i \, j \rangle > 0$ for all $i < j$) and the 3-term Plücker identity.
    pub fn verify_positive_grassmannian(&self) -> (bool, f64) {
        let n = self.num_particles;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.minor(i, j) <= 0.0 {
                    return (false, f64::INFINITY);
                }
            }
        }
        let p02_13 = self.minor(0, 2) * self.minor(1, 3);
        let rhs = self.minor(0, 1) * self.minor(2, 3) + self.minor(0, 3) * self.minor(1, 2);
        (true, (p02_13 - rhs).abs())
    }
}
