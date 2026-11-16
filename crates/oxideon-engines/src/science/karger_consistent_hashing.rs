//! Karger-Lehman-Leighton Consistent Hashing Ring & Virtual Node Load Balancing (Karger et al., 1997).
//!
//! ### 1. Minimal Key Remapping Fraction $1/N$ & Virtual Node Variance Reduction
//! Partitioning the unit circle among $N$ physical servers with $V$ virtual tokens each ensures that adding/removing a server
//! remaps only $1/N$ of keys in expectation, while maximum load imbalance scales as $1 + O(\sqrt{\frac{\ln N}{V}})$.

use crate::error::{Error, Result};

/// Karger-Lehman-Leighton Consistent Hashing Ring Engine.
#[derive(Debug, Clone, Copy)]
pub struct KargerConsistentHashingRing {
    /// Number of physical servers $N \ge 2$.
    pub num_servers_n: usize,
    /// Number of virtual nodes per physical server $V \ge 1$.
    pub vnodes_per_server_v: usize,
}

impl Default for KargerConsistentHashingRing {
    fn default() -> Self {
        Self {
            num_servers_n: 16,
            vnodes_per_server_v: 100,
        }
    }
}

impl KargerConsistentHashingRing {
    /// Creates a new Karger Consistent Hashing Ring engine.
    pub fn new(num_servers_n: usize, vnodes_per_server_v: usize) -> Result<Self> {
        if num_servers_n < 2 || vnodes_per_server_v == 0 {
            return Err(Error::Msg(
                "num_servers_n >= 2 and vnodes_per_server_v >= 1 required".to_string(),
            ));
        }
        Ok(Self {
            num_servers_n,
            vnodes_per_server_v,
        })
    }

    /// Evaluates the expected remapped key fraction $1 / (N + 1)$ upon adding a server and the high-probability peak-to-average load ratio $1 + \sqrt{2 \ln N / V}$.
    pub fn remapped_fraction_and_peak_load_ratio(&self) -> (f64, f64) {
        let n = self.num_servers_n as f64;
        let v = self.vnodes_per_server_v as f64;
        let remap_frac = 1.0 / (n + 1.0);
        let peak_ratio = 1.0 + (2.0 * n.ln() / v).sqrt();
        (remap_frac, peak_ratio)
    }
}
