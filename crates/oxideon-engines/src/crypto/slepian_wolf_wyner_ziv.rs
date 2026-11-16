//! Slepian-Wolf Distributed Lossless Source Coding & Wyner-Ziv Side-Information Rate-Distortion (Slepian & Wolf, 1973; Wyner & Ziv, 1976).
//!
//! ### 1. Slepian-Wolf Theorem & Wyner-Ziv Gaussian Side-Information Rate
//! Even without encoder cooperation, correlated sources $(X, Y)$ can be compressed at total rate $R_X + R_Y = H(X, Y)$
//! provided $R_X \ge H(X \mid Y)$ and $R_Y \ge H(Y \mid X)$. For a Gaussian source $X = Y + Z$ with side-information
//! noise $Z \sim \mathcal{N}(0, \sigma_{X|Y}^2)$ at the decoder, the Wyner-Ziv rate-distortion function equals the conditional rate:
//! $$R_{\mathrm{WZ}}(D) = \max\left(0, \frac{1}{2} \log_2\left(\frac{\sigma_{X|Y}^2}{D}\right)\right)$$

use crate::error::{Error, Result};

/// Slepian-Wolf Distributed Coding & Wyner-Ziv Side-Information Engine.
#[derive(Debug, Clone, Copy, Default)]
pub struct SlepianWolfDistributedCoding;

impl SlepianWolfDistributedCoding {
    /// Creates a new Slepian-Wolf / Wyner-Ziv engine.
    pub fn new() -> Self {
        Self
    }

    /// Checks whether a rate pair $(R_X, R_Y)$ lies inside the achievable Slepian-Wolf pentagon $(R_X \ge H(X|Y), R_Y \ge H(Y|X), R_X + R_Y \ge H(X,Y))$.
    pub fn is_slepian_wolf_achievable(
        rx: f64,
        ry: f64,
        h_x_given_y: f64,
        h_y_given_x: f64,
        h_joint_xy: f64,
    ) -> bool {
        rx >= h_x_given_y && ry >= h_y_given_x && (rx + ry) >= h_joint_xy
    }

    /// Evaluates the Gaussian Wyner-Ziv rate $R_{\mathrm{WZ}}(D) = \max(0, \frac{1}{2}\log_2(\sigma_X^2 (1 - \rho^2) / D))$ for correlation $\rho \in (-1, 1)$.
    pub fn gaussian_wyner_ziv_rate(
        var_x: f64,
        correlation_rho: f64,
        distortion_d: f64,
    ) -> Result<f64> {
        if var_x <= 0.0 || !(-1.0..1.0).contains(&correlation_rho) || distortion_d <= 0.0 {
            return Err(Error::Msg(
                "var_x > 0, correlation_rho in (-1, 1), and distortion_d > 0 required".to_string(),
            ));
        }
        let cond_var = var_x * (1.0 - correlation_rho * correlation_rho);
        if cond_var <= distortion_d {
            Ok(0.0)
        } else {
            Ok(0.5 * (cond_var / distortion_d).log2())
        }
    }
}
