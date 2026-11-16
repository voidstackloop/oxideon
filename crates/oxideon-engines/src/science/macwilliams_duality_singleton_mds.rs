//! MacWilliams Weight Enumerator Duality & Singleton / MDS Bound $d \le n - k + 1$ (Singleton, 1964; MacWilliams, 1963).
//!
//! ### 1. MacWilliams Identity $W_{C^\perp}(x, y) = \frac{1}{|C|} W_C(x + (q-1)y, x - y)$ & MDS Codes
//! For an $[n, k, d]_q$ linear code $C$, the Singleton bound states $d \le n - k + 1$, achieved with equality by Maximum Distance Separable (MDS) codes (such as Reed-Solomon codes), where the dual $C^\perp$ is also MDS with parameters $[n, n - k, k + 1]_q$.

use crate::error::{Error, Result};

/// MacWilliams Duality & Singleton MDS Linear Code Engine.
#[derive(Debug, Clone, Copy)]
pub struct MacWilliamsSingletonMdsCode {
    /// Block length $n \ge 2$.
    pub length_n: usize,
    /// Dimension $1 \le k \le n$.
    pub dim_k: usize,
}

impl Default for MacWilliamsSingletonMdsCode {
    fn default() -> Self {
        Self {
            length_n: 15,
            dim_k: 9,
        }
    }
}

impl MacWilliamsSingletonMdsCode {
    /// Creates a new MacWilliams & Singleton MDS Code engine.
    pub fn new(length_n: usize, dim_k: usize) -> Result<Self> {
        if length_n < 2 || dim_k == 0 || dim_k > length_n {
            return Err(Error::Msg(
                "1 <= dim_k <= length_n and length_n >= 2 required".to_string(),
            ));
        }
        Ok(Self { length_n, dim_k })
    }

    /// Evaluates the Singleton upper bound $d_{\max} = n - k + 1$, guaranteed error-correction radius $t = \lfloor(d_{\max} - 1)/2\rfloor$, and dual MDS distance $k + 1$.
    pub fn singleton_distance_and_dual_mds(&self) -> (usize, usize, usize) {
        let d_mds = self.length_n - self.dim_k + 1;
        let t_corr = (d_mds - 1) / 2;
        let d_dual = self.dim_k + 1;
        (d_mds, t_corr, d_dual)
    }
}
