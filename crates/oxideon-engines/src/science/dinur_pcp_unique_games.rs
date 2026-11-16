//! Dinur Combinatorial PCP Gap Amplification & Goemans-Williamson / Khot Unique Games Integrality Gap (Goemans & Williamson, 1995; Khot, 2002; Dinur, 2007).
//!
//! ### 1. Goemans-Williamson SDP Ratio & Khot UGC Threshold
//! The Goemans-Williamson semidefinite relaxation of Max-Cut achieves approximation ratio:
//! $$\alpha_{\mathrm{GW}} = \min_{0 < \theta \le \pi} \frac{2}{\pi} \frac{\theta}{1 - \cos \theta} \approx 0.8785672$$
//! Under Khot's Unique Games Conjecture (UGC), achieving $\alpha_{\mathrm{GW}} + \varepsilon$ is NP-hard.

use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Dinur PCP Gap Amplification & Khot Unique Games Max-Cut Engine.
#[derive(Debug, Clone, Copy)]
pub struct DinurPcpGapAmplification {
    /// Expander walk amplification step parameter $t \ge 1$.
    pub walk_length_t: usize,
}

impl Default for DinurPcpGapAmplification {
    fn default() -> Self {
        Self { walk_length_t: 4 }
    }
}

impl DinurPcpGapAmplification {
    /// Creates a new Dinur PCP & Unique Games engine.
    pub fn new(walk_length_t: usize) -> Result<Self> {
        if walk_length_t == 0 {
            return Err(Error::Msg("walk_length_t must be >= 1".to_string()));
        }
        Ok(Self { walk_length_t })
    }

    /// Evaluates the Goemans-Williamson hyperplane rounding ratio $\frac{2}{\pi} \frac{\theta}{1 - \cos \theta}$ at angle $\theta \in (0, \pi]$ and its global minimum $\alpha_{\mathrm{GW}} \approx 0.878567$.
    pub fn goemans_williamson_ratio_at_angle(theta_rad: f64) -> Result<(f64, f64)> {
        if theta_rad <= 0.0 || theta_rad > PI {
            return Err(Error::Msg("theta_rad must be in (0, pi]".to_string()));
        }
        let ratio = (2.0 / PI) * theta_rad / (1.0 - theta_rad.cos());
        let theta_star = 2.331122370414423;
        let alpha_gw = (2.0 / PI) * theta_star / (1.0 - theta_star.cos());
        Ok((ratio, alpha_gw))
    }
}
