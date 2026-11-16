//! Ahlswede-Cai-Li-Yeung Network Coding Max-Flow Min-Cut Theorem & Butterfly Multicast Gain (Ahlswede, Cai, Li & Yeung, 2000; Li, Yeung & Cai, 2003).
//!
//! ### 1. Multicast Capacity $C_{\mathrm{multicast}} = \min_{t \in T} \operatorname{MinCut}(s, t)$
//! For a directed acyclic communication network $G = (V, E, c)$ with source $s$ and sink set $T$, linear network coding over a sufficiently large finite field $\mathbb{F}_q$ ($q > |T|$) achieves multicast rate:
//! $$R^* = \min_{t \in T} \operatorname{MaxFlow}(s, t)$$

use crate::error::{Error, Result};

/// Ahlswede-Cai-Li-Yeung Linear Network Coding Engine.
#[derive(Debug, Clone, Copy)]
pub struct LinearNetworkCodingMaxFlow {
    /// Number of multicast receivers $|T| \ge 2$.
    pub num_sinks: usize,
}

impl Default for LinearNetworkCodingMaxFlow {
    fn default() -> Self {
        Self { num_sinks: 2 }
    }
}

impl LinearNetworkCodingMaxFlow {
    /// Creates a new Linear Network Coding engine.
    pub fn new(num_sinks: usize) -> Result<Self> {
        if num_sinks < 2 {
            return Err(Error::Msg("num_sinks must be >= 2".to_string()));
        }
        Ok(Self { num_sinks })
    }

    /// Evaluates the Ahlswede-Cai-Li-Yeung multicast capacity $\min_t C_t$ and the butterfly coding throughput ratio over routing.
    pub fn multicast_capacity_and_coding_gain(
        &self,
        sink_min_cuts: &[f64],
        routing_limit: f64,
    ) -> Result<(f64, f64)> {
        if sink_min_cuts.len() != self.num_sinks
            || routing_limit <= 0.0
            || sink_min_cuts.iter().any(|&c| c <= 0.0)
        {
            return Err(Error::Msg(
                "valid positive sink_min_cuts of length num_sinks and routing_limit > 0 required"
                    .to_string(),
            ));
        }
        let cap = sink_min_cuts.iter().copied().fold(f64::INFINITY, f64::min);
        Ok((cap, cap / routing_limit))
    }
}
