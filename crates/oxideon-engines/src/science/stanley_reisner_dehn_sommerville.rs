//! Stanley-Reisner Simplicial Face Ring, $f$-Vector to $h$-Vector Transform & Dehn-Sommerville Symmetry (Stanley, 1975).
//!
//! ### 1. $h$-Vector of a $(d-1)$-Dimensional Simplicial Complex
//! Given face numbers $(f_{-1}=1, f_0, f_1, \dots, f_{d-1})$, the $h$-polynomial is $\sum_{k=0}^d h_k x^{d-k} = \sum_{i=0}^d f_{i-1} (x - 1)^{d-i}$.
//! For a simplicial sphere (or Cohen-Macaulay Eulerian complex), Dehn-Sommerville symmetry holds: $h_k = h_{d - k}$ and $h_k \ge 0$.

use crate::error::{Error, Result};

/// Stanley-Reisner $h$-Vector & Dehn-Sommerville Symmetry Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct StanleyReisnerDehnSommerville;

impl StanleyReisnerDehnSommerville {
    /// Creates a new Stanley-Reisner Dehn-Sommerville engine.
    pub fn new() -> Self {
        Self
    }

    /// Computes the $h$-vector $(h_0, h_1, \dots, h_d)$ from $f = (f_{-1}=1, f_0, \dots, f_{d-1})$ and checks Dehn-Sommerville palindrome symmetry $h_k = h_{d-k}$.
    pub fn compute_h_vector_and_check_symmetry(
        f_vector_with_empty: &[i64],
    ) -> Result<(Vec<i64>, bool)> {
        if f_vector_with_empty.len() < 2 || f_vector_with_empty[0] != 1 {
            return Err(Error::Msg(
                "f_vector_with_empty must start with f_{-1} = 1 and have len >= 2".to_string(),
            ));
        }
        let d = f_vector_with_empty.len() - 1;
        let mut h = vec![0i64; d + 1];
        for k in 0..=d {
            let mut sum = 0i64;
            let mut binom = 1i64;
            for i in 0..=k {
                if i > 0 {
                    binom = binom * ((d - i + 1) as i64) / ((k - i + 1) as i64);
                } else {
                    // C(d, k)
                    let mut c = 1i64;
                    for j in 0..k {
                        c = c * ((d - j) as i64) / ((j + 1) as i64);
                    }
                    binom = c;
                }
                let sign = if (k - i) % 2 == 0 { 1 } else { -1 };
                // C(d - i, k - i)
                let mut c_di = 1i64;
                for j in 0..(k - i) {
                    c_di = c_di * ((d - i - j) as i64) / ((j + 1) as i64);
                }
                let _ = binom;
                sum += sign * c_di * f_vector_with_empty[i];
            }
            h[k] = sum;
        }
        let is_sym = (0..=d).all(|k| h[k] == h[d - k]);
        Ok((h, is_sym))
    }
}
