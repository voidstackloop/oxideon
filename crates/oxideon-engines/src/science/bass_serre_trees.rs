//! Bass-Serre Theory of Groups Acting on Trees, Amalgamated Products & Rational Euler Characteristic (Serre, 1977).
//!
//! ### 1. Wall-Serre Rational Euler Characteristic of a Graph of Groups
//! For a finite graph of finite groups $(G, Y)$, the rational Euler characteristic of the fundamental group $\pi_1(G, Y)$ is:
//! $$\chi(\pi_1(G, Y)) = \sum_{v \in V(Y)} \frac{1}{|G_v|} - \sum_{e \in E(Y)} \frac{1}{|G_e|}$$

use crate::error::{Error, Result};

/// Bass-Serre Graph of Groups & Rational Euler Characteristic Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct BassSerreGraphOfGroupsEngine;

impl BassSerreGraphOfGroupsEngine {
    /// Creates a new Bass-Serre Graph of Groups engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the rational Euler characteristic $\chi(\pi_1(G, Y)) = \sum |G_v|^{-1} - \sum |G_e|^{-1}$.
    pub fn rational_euler_characteristic(
        vertex_group_orders: &[usize],
        edge_group_orders: &[usize],
    ) -> Result<f64> {
        if vertex_group_orders.is_empty()
            || vertex_group_orders.contains(&0)
            || edge_group_orders.contains(&0)
        {
            return Err(Error::Msg("Group orders must be non-zero".to_string()));
        }
        let v_sum: f64 = vertex_group_orders.iter().map(|&n| 1.0 / (n as f64)).sum();
        let e_sum: f64 = edge_group_orders.iter().map(|&n| 1.0 / (n as f64)).sum();
        Ok(v_sum - e_sum)
    }
}
