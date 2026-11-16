//! $p$-Adic Ultrametric Analysis, Ostrowski's Valuation $|\cdot|_p$ & Vladimirov Fractional Operators.
//!
//! ### 1. $p$-Adic Valuation & Ultrametric Norm $|x|_p$
//! For a prime $p \ge 2$ and any non-zero integer $n = p^v \cdot m$ with $\gcd(m, p) = 1$,
//! the $p$-adic valuation is $v_p(n) = v$, and the $p$-adic absolute value is:
//! $$|n|_p = p^{-v_p(n)}, \qquad |0|_p = 0$$
//! By **Ostrowski's Theorem**, every non-trivial absolute value on $\mathbb{Q}$ is equivalent either to
//! the real Euclidean norm or to $|\cdot|_p$ for some prime $p$.
//!
//! ### 2. Strong Ultrametric Triangle Inequality
//! Every $p$-adic metric $d_p(x, y) = |x - y|_p$ satisfies the **Strong Triangle Inequality**:
//! $$|x + y|_p \le \max\left(|x|_p, \; |y|_p\right)$$
//! with equality whenever $|x|_p \neq |y|_p$ (every triangle in $\mathbb{Q}_p$ is isosceles!).
//!
//! ### 3. Discrete Vladimirov Fractional Pseudo-Differential Operator $D^\alpha_p$
//! On a finite $p$-adic hierarchy of $p^k$ states, the Vladimirov operator $D^\alpha_p \psi(x)$ acts via the
//! ultrametric hypersingular integral kernel:
//! $$(D^\alpha_p \psi)(x) = \frac{1 - p^\alpha}{1 - p^{-\alpha - 1}} \sum_{y \neq x} \frac{\psi(x) - \psi(y)}{|x - y|_p^{\alpha + 1}}$$

use crate::error::{Error, Result};

/// $p$-Adic Ultrametric Valuation & Vladimirov Fractional Operator Engine.
#[derive(Debug, Clone, Copy)]
pub struct PAdicAnalysisEngine {
    /// Prime base $p \ge 2$.
    pub prime_p: i64,
}

impl PAdicAnalysisEngine {
    /// Constructs a $p$-adic analysis engine for a prime $p \ge 2$.
    pub fn new(prime_p: i64) -> Result<Self> {
        if prime_p < 2 || !Self::is_prime(prime_p) {
            return Err(Error::Msg(format!("Base p={prime_p} must be a prime >= 2")));
        }
        Ok(Self { prime_p })
    }

    fn is_prime(n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }
        let mut d = 3;
        while d <= n / d {
            if n % d == 0 {
                return false;
            }
            d += 2;
        }
        true
    }

    /// Computes the $p$-adic valuation $v_p(n)$ for $n \neq 0$
    /// (`None` for $n = 0$ or for an invalid base $p < 2$).
    pub fn valuation(&self, n: i64) -> Option<u32> {
        if n == 0 {
            return None;
        }
        let p = self.prime_p;
        if p < 2 {
            return None;
        }
        let mut val = n.unsigned_abs();
        let mut v = 0;
        while val % (p as u64) == 0 {
            val /= p as u64;
            v += 1;
        }
        Some(v)
    }

    /// Computes the $p$-adic norm $|n|_p = p^{-v_p(n)}$ (with $|0|_p = 0$,
    /// `NaN` for an invalid base $p < 2$).
    pub fn norm(&self, n: i64) -> f64 {
        if self.prime_p < 2 {
            return f64::NAN;
        }
        match self.valuation(n) {
            None => 0.0,
            Some(v) => (self.prime_p as f64).powi(-(v as i32)),
        }
    }

    /// Computes the ultrametric distance $d_p(x, y) = |x - y|_p$
    /// (`NaN` when the difference overflows or the base is invalid).
    pub fn distance(&self, x: i64, y: i64) -> f64 {
        match x.checked_sub(y) {
            Some(diff) => self.norm(diff),
            None => f64::NAN,
        }
    }

    /// Verifies the Strong Ultrametric Triangle Inequality $d_p(x, z) \le \max(d_p(x, y), d_p(y, z))$
    /// across all triples in `points`.
    pub fn verify_ultrametric_inequality(&self, points: &[i64]) -> bool {
        for &x in points {
            for &y in points {
                for &z in points {
                    let dxz = self.distance(x, z);
                    let dxy = self.distance(x, y);
                    let dyz = self.distance(y, z);
                    if dxz > dxy.max(dyz) + 1e-15 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Applies the discrete Vladimirov fractional derivative operator $D^\alpha_p$ of order $\alpha > 0$
    /// to a signal $\psi$ supported on integer nodes $0..N$.
    pub fn vladimirov_operator(&self, psi: &[f64], alpha: f64) -> Result<Vec<f64>> {
        if psi.len() < 2 || alpha <= 0.0 {
            return Err(Error::Msg(
                "vladimirov_operator requires len >= 2 and alpha > 0".to_string(),
            ));
        }
        let n = psi.len();
        let mut out = vec![0.0; n];
        for (x, out_x) in out.iter_mut().enumerate().take(n) {
            let mut sum = 0.0;
            for (y, &psi_y) in psi.iter().enumerate().take(n) {
                if x != y {
                    let dist = self.distance(x as i64, y as i64);
                    sum += (psi[x] - psi_y) / dist.powf(alpha + 1.0);
                }
            }
            *out_x = sum;
        }
        Ok(out)
    }
}
