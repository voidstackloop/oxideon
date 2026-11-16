//! Arıkan Channel Polarization, Recursive Bhattacharyya Tree & Capacity-Achieving Polar Codes (Arıkan, 2009).
//!
//! ### 1. Arıkan's Synthetic Bad/Good Channel Splitting $(W^-, W^+)$
//! Combining two independent copies of a Binary Erasure Channel $\mathrm{BEC}(z)$ with erasure probability (Bhattacharyya parameter)
//! $z \in [0, 1]$ synthesizes a worse channel $W^-$ and a better channel $W^+$ with exact Bhattacharyya parameters:
//! $$Z(W^-) = 2 z - z^2, \qquad Z(W^+) = z^2$$
//! Notice that average capacity is conserved: $\frac{1}{2}\left[(1 - Z(W^-)) + (1 - Z(W^+))\right] = 1 - z = I(W)$,
//! while after $n$ levels all $N = 2^n$ bit-channels polarize to noiseless ($Z \to 0$) or pure-noise ($Z \to 1$)!

use crate::error::{Error, Result};

/// Arıkan Capacity-Achieving Polar Code & Bhattacharyya Polarization Engine.
#[derive(Debug, Clone, Copy)]
pub struct ArikanPolarCodeEngine {
    /// Initial channel erasure / Bhattacharyya parameter $z_0 \in [0, 1]$.
    pub initial_z0: f64,
}

impl ArikanPolarCodeEngine {
    /// Creates a new Arıkan Polar Code engine.
    pub fn new(initial_z0: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&initial_z0) {
            return Err(Error::Msg("initial_z0 must be in [0, 1]".to_string()));
        }
        Ok(Self { initial_z0 })
    }

    /// Computes all $2^n$ polarized Bhattacharyya parameters after `depth_n` recursive Arıkan steps.
    pub fn polarized_bhattacharyya_parameters(&self, depth_n: usize) -> Result<Vec<f64>> {
        if depth_n > 16 {
            return Err(Error::Msg("depth_n must be <= 16".to_string()));
        }
        let mut z_vec = vec![self.initial_z0];
        for _ in 0..depth_n {
            let mut next = Vec::with_capacity(z_vec.len() * 2);
            for &z in &z_vec {
                next.push(2.0 * z - z * z);
                next.push(z * z);
            }
            z_vec = next;
        }
        Ok(z_vec)
    }
}
