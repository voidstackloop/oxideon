//! Malliavin Calculus on Wiener Space, Skorokhod Divergence $\delta(u)$ & Bismut-Elworthy-Li Integration by Parts (Malliavin, 1978).
//!
//! ### 1. Malliavin Integration-by-Parts Formula & Bismut Greek Weights
//! For a smooth Wiener functional $F = f(X_T)$ with Malliavin derivative $D_t F$ and an adapted weight process $u_t$,
//! the Malliavin-Bismut integration-by-parts identity eliminates derivatives of non-smooth payoffs $\Phi(X_T)$:
//! $$\mathbb{E}\left[\Phi'(X_T) G\right] = \mathbb{E}\left[\Phi(X_T) \, \delta\left(\frac{G \, u}{\langle D X_T, u \rangle_{L^2}}\right)\right]$$
//! For Black-Scholes geometric Brownian motion $dS_t = r S_t dt + \sigma S_t dW_t$, the Bismut Delta weight is $\pi_\Delta = \frac{W_T}{S_0 \sigma T}$.

use crate::error::{Error, Result};

/// Malliavin Calculus & Bismut-Elworthy-Li Sensitivity Weight Engine.
#[derive(Debug, Clone, Copy)]
pub struct MalliavinBismutEngine {
    /// Initial spot price $S_0 > 0$.
    pub spot_s0: f64,
    /// Volatility $\sigma > 0$.
    pub volatility_sigma: f64,
    /// Maturity $T > 0$.
    pub maturity_t: f64,
}

impl MalliavinBismutEngine {
    /// Creates a new Malliavin-Bismut sensitivity engine.
    pub fn new(spot_s0: f64, volatility_sigma: f64, maturity_t: f64) -> Result<Self> {
        if spot_s0 <= 0.0 || volatility_sigma <= 0.0 || maturity_t <= 0.0 {
            return Err(Error::Msg(
                "spot_s0, volatility_sigma, and maturity_t must be > 0".to_string(),
            ));
        }
        Ok(Self {
            spot_s0,
            volatility_sigma,
            maturity_t,
        })
    }

    /// Evaluates the Bismut Malliavin weight $\pi_\Delta(W_T) = \frac{W_T}{S_0 \sigma T}$ and $\pi_\Gamma(W_T) = \frac{W_T^2 - \sigma W_T T - T}{S_0^2 \sigma^2 T^2}$.
    pub fn bismut_delta_and_gamma_weights(&self, terminal_wiener_wt: f64) -> (f64, f64) {
        let denom = self.spot_s0 * self.volatility_sigma * self.maturity_t;
        let w_delta = terminal_wiener_wt / denom;
        let w_gamma = (terminal_wiener_wt * terminal_wiener_wt
            - self.volatility_sigma * terminal_wiener_wt * self.maturity_t
            - self.maturity_t)
            / (denom * denom);
        (w_delta, w_gamma)
    }
}
