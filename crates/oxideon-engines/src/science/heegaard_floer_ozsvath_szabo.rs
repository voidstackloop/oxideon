//! Ozsváth-Szabó Heegaard Floer Homology $\widehat{HF}(Y)$, Knot Floer $\tau(K)$ & Legendrian Thurston-Bennequin Bound (Ozsváth & Szabó, 2004).
//!
//! ### 1. Knot Floer Homology Categorification of the Alexander Polynomial & $\tau$-Bound
//! Knot Floer homology $\widehat{HFK}_{m}(K, s)$ categorifies the symmetrized Alexander-Conway polynomial
//! $\Delta_K(t) = \sum_{m,s} (-1)^m t^s \dim \widehat{HFK}_m(K, s)$, and the Ozsváth-Szabó concordance invariant $\tau(K)$
//! bounds the Thurston-Bennequin number $\mathrm{tb}(L)$ and rotation number $\mathrm{rot}(L)$ of any Legendrian realization $L$ of $K$:
//! $$\mathrm{tb}(L) + |\mathrm{rot}(L)| \le 2 \tau(K) - 1$$

/// Ozsváth-Szabó Heegaard Floer & Knot Floer $\tau$-Invariant Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeegaardFloerKnotEngine;

impl HeegaardFloerKnotEngine {
    /// Creates a new Heegaard Floer Knot engine.
    pub fn new() -> Self {
        Self
    }

    /// Evaluates the graded Euler characteristic $\Delta_K(t) = \sum_s \chi_s t^s$ from symmetric Alexander-grading Euler characteristics.
    pub fn alexander_euler_characteristic(positive_gradings_chi: &[i64], t: f64) -> f64 {
        if positive_gradings_chi.is_empty() {
            return 0.0;
        }
        let mut val = positive_gradings_chi[0] as f64;
        for (s, &chi) in positive_gradings_chi.iter().enumerate().skip(1) {
            let ts = t.powi(s as i32);
            val += (chi as f64) * (ts + 1.0 / ts);
        }
        val
    }

    /// Verifies Plamenevskaya's Legendrian Thurston-Bennequin inequality $\mathrm{tb}(L) + |\mathrm{rot}(L)| < 2 \tau(K)$.
    pub fn satisfies_legendrian_tau_bound(tb: i64, rot: i64, tau_k: i64) -> bool {
        (tb as i128) + (rot as i128).abs() < 2 * (tau_k as i128)
    }
}
