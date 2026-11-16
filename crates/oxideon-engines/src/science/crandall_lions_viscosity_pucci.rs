//! Crandall-Ishii-Lions Viscosity Solutions & Pucci Extremal Operators $\mathcal{M}^{\pm}_{\lambda, \Lambda}$ (Crandall & Lions, 1983; Caffarelli & Cabré, 1995).
//!
//! ### 1. Pucci Maximal and Minimal Elliptic Operators
//! Given ellipticity constants $0 < \lambda \le \Lambda$ and Hessian eigenvalues $e_i(D^2 u)$, the Pucci extremal operators are:
//! $$\mathcal{M}^+_{\lambda, \Lambda}(D^2 u) = \Lambda \sum_{e_i > 0} e_i + \lambda \sum_{e_i < 0} e_i, \qquad \mathcal{M}^-_{\lambda, \Lambda}(D^2 u) = \lambda \sum_{e_i > 0} e_i + \Lambda \sum_{e_i < 0} e_i$$

use crate::error::{Error, Result};

/// Crandall-Ishii-Lions Viscosity Solution & Pucci Extremal Operator Engine.
#[derive(Debug, Clone, Copy)]
pub struct CrandallLionsViscosityPucci {
    /// Lower ellipticity bound $\lambda > 0$.
    pub lambda_min: f64,
    /// Upper ellipticity bound $\Lambda \ge \lambda$.
    pub lambda_max: f64,
}

impl Default for CrandallLionsViscosityPucci {
    fn default() -> Self {
        Self {
            lambda_min: 1.0,
            lambda_max: 3.0,
        }
    }
}

impl CrandallLionsViscosityPucci {
    /// Creates a new Crandall-Lions Viscosity Pucci engine.
    pub fn new(lambda_min: f64, lambda_max: f64) -> Result<Self> {
        if lambda_min <= 0.0 || lambda_max < lambda_min {
            return Err(Error::Msg(
                "0 < lambda_min <= lambda_max required".to_string(),
            ));
        }
        Ok(Self {
            lambda_min,
            lambda_max,
        })
    }

    /// Evaluates the Pucci extremal operators $(\mathcal{M}^+_{\lambda, \Lambda}, \mathcal{M}^-_{\lambda, \Lambda})$ given Hessian eigenvalues $e_1, \dots, e_n$.
    pub fn evaluate_pucci_extremal(&self, hessian_eigenvalues: &[f64]) -> (f64, f64) {
        let mut pos_sum = 0.0;
        let mut neg_sum = 0.0;
        for &e in hessian_eigenvalues {
            if e > 0.0 {
                pos_sum += e;
            } else {
                neg_sum += e;
            }
        }
        let m_plus = self.lambda_max * pos_sum + self.lambda_min * neg_sum;
        let m_minus = self.lambda_min * pos_sum + self.lambda_max * neg_sum;
        (m_plus, m_minus)
    }
}
