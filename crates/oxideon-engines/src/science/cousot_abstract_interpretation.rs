//! Cousot-Cousot Abstract Interpretation, Interval Domain Galois Connection & Widening Operator $\nabla$ (Cousot & Cousot, 1977).
//!
//! ### 1. Interval Domain Widening $[a, b] \nabla [c, d]$
//! To accelerate fixed-point convergence across infinite ascending chains in the interval lattice, Cousot & Cousot define:
//! $$[a, b] \nabla [c, d] = \bigl[\text{if } c < a \text{ then } -\infty \text{ else } a, \; \text{if } d > b \text{ then } +\infty \text{ else } b\bigr]$$

use crate::error::{Error, Result};

/// Cousot-Cousot Interval Abstract Domain & Widening Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct CousotGaloisAbstractInterpreter;

impl CousotGaloisAbstractInterpreter {
    /// Creates a new Cousot Galois Abstract Interpreter engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the interval join $[a, b] \sqcup [c, d] = [\min(a, c), \max(b, d)]$ and Cousot threshold-widening $[a, b] \nabla_T [c, d]$ with bound $T > 0$.
    pub fn interval_join_and_widening(
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        bound_t: f64,
    ) -> Result<([f64; 2], [f64; 2])> {
        if a > b || c > d || bound_t <= 0.0 {
            return Err(Error::Msg(
                "Valid intervals a <= b, c <= d and bound_t > 0 required".to_string(),
            ));
        }
        let join = [a.min(c), b.max(d)];
        let wide_lo = if c < a { -bound_t } else { a };
        let wide_hi = if d > b { bound_t } else { b };
        Ok((join, [wide_lo, wide_hi]))
    }
}
