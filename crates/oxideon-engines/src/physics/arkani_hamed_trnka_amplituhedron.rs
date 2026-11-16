//! Arkani-Hamed-Trnka Amplituhedron $\mathcal{A}_{n, k, m}$, Positive Grassmannian $G_+(k, n)$ & BCFW Cell Counts (Britto, Cachazo, Feng & Witten, 2005; Arkani-Hamed & Trnka, 2014).
//!
//! ### 1. Narayana Numbers $N(n-3, k) = \frac{1}{n-3}\binom{n-3}{k}\binom{n-3}{k-1}$ Counting Tree-Level $\mathrm{N}^k\mathrm{MHV}$ BCFW Cells
//! For $m = 4$ planar $\mathcal{N}=4$ SYM tree-level $n$-gluon $\mathrm{N}^k\mathrm{MHV}$ scattering, the Amplituhedron $\mathcal{A}_{n, k, 4}$ is triangulated by exactly the Narayana number $N(n-3, k+1)$ of positive Grassmannian BCFW cells of dimension $4k$.

use crate::error::{Error, Result};

/// Arkani-Hamed-Trnka Amplituhedron & BCFW Positive Grassmannian Cell Engine.
#[derive(Debug, Clone, Copy)]
pub struct AmplituhedronGrassmannianVolume {
    /// Number of external particles $n \ge 4$.
    pub num_particles_n: usize,
    /// NMHV degree $k$ ($0 \le k \le n - 4$).
    pub nmhv_degree_k: usize,
}

impl Default for AmplituhedronGrassmannianVolume {
    fn default() -> Self {
        Self {
            num_particles_n: 6,
            nmhv_degree_k: 1,
        }
    }
}

impl AmplituhedronGrassmannianVolume {
    /// Creates a new Amplituhedron Grassmannian engine.
    pub fn new(num_particles_n: usize, nmhv_degree_k: usize) -> Result<Self> {
        if num_particles_n < 4 || nmhv_degree_k > num_particles_n - 4 || num_particles_n > 22 {
            return Err(Error::Msg(
                "4 <= num_particles_n <= 22 and 0 <= nmhv_degree_k <= n - 4 required".to_string(),
            ));
        }
        Ok(Self {
            num_particles_n,
            nmhv_degree_k,
        })
    }

    /// Evaluates the number of BCFW triangulation cells (Narayana number $N(n-3, k+1)$) and the cell dimension $4k$.
    pub fn bcfw_cell_count_and_dimension(&self) -> Result<(u64, usize)> {
        // Re-validate: the fields are pub, and n < 3 underflows `n - 3`
        // while m == 0 divides by zero below.
        let n = self.num_particles_n;
        let k = self.nmhv_degree_k;
        if !(4..=22).contains(&n) || k > n - 4 {
            return Err(Error::Msg(
                "4 <= num_particles_n <= 22 and 0 <= nmhv_degree_k <= n - 4 required".to_string(),
            ));
        }
        let m = (n - 3) as u64;
        let r = (k + 1) as u64;
        fn binom(n: u64, k: u64) -> u64 {
            if k > n {
                return 0;
            }
            let mut res = 1_u64;
            for i in 0..k {
                res = res * (n - i) / (i + 1);
            }
            res
        }
        // n <= 22: binom(19,.) fits u64 with huge margin; m >= 1.
        let narayana = binom(m, r) * binom(m, r - 1) / m;
        Ok((narayana, 4 * k))
    }
}
