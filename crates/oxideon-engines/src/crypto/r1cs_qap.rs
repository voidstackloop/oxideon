//! Differentiable Zero-Knowledge Rank-1 Constraint Systems (R1CS) &
//! Quadratic Arithmetic Programs (QAP) (Gennaro-Gentry-Parno-Raykova / Groth16).
//!
//! # Mathematical Foundations
//!
//! ### 1. **Rank-1 Constraint System (R1CS)**:
//! An arithmetic circuit with $m$ gates (constraints) and $n$-element witness vector
//! $\mathbf{z} = (1, x_1, \dots, x_\ell, w_1, \dots, w_{n - \ell - 1})^T$ is encoded by three
//! $m \times n$ matrices $(A, B, C)$ satisfying the Hadamard (elementwise) relation:
//! $$(A \mathbf{z}) \circ (B \mathbf{z}) - C \mathbf{z} = \mathbf{0}$$
//!
//! For continuous neural-symbolic optimization, the **Differentiable R1CS Loss** is the smooth energy:
//! $$\mathcal{L}_{\text{R1CS}}(\mathbf{z}) = \frac{1}{2} \sum_{k=1}^m \left( (\mathbf{a}_k \cdot \mathbf{z})(\mathbf{b}_k \cdot \mathbf{z}) - (\mathbf{c}_k \cdot \mathbf{z}) \right)^2$$
//! with analytical gradient $\nabla_{\mathbf{z}} \mathcal{L}_{\text{R1CS}}$ enabling gradient-based witness synthesis!
//!
//! ### 2. **Quadratic Arithmetic Program (QAP) & Schwartz-Zippel Verification**:
//! Choosing distinct evaluation roots $r_1, r_2, \dots, r_m$, Lagrange interpolation maps the
//! columns of $(A, B, C)$ into degree-$(m-1)$ polynomials $A_j(x), B_j(x), C_j(x)$ such that:
//! $$A(x) = \sum_{j=0}^{n-1} z_j A_j(x), \quad B(x) = \sum_{j=0}^{n-1} z_j B_j(x), \quad C(x) = \sum_{j=0}^{n-1} z_j C_j(x)$$
//! The witness $\mathbf{z}$ is valid if and only if the vanishing polynomial
//! $Z(x) = \prod_{k=1}^m (x - r_k)$ divides $P(x) = A(x) B(x) - C(x)$, i.e., there exists a quotient
//! polynomial $H(x)$ such that at a secret toxic-waste evaluation point $\tau$:
//! $$A(\tau) B(\tau) - C(\tau) = H(\tau) Z(\tau)$$

use crate::error::{Error, Result};

/// A single Rank-1 Constraint $(\mathbf{a}_k \cdot \mathbf{z}) (\mathbf{b}_k \cdot \mathbf{z}) = \mathbf{c}_k \cdot \mathbf{z}$.
#[derive(Debug, Clone, PartialEq)]
pub struct R1csConstraint {
    /// Linear combination coefficients $\mathbf{a}_k$ of length $n$.
    pub a: Vec<f64>,
    /// Linear combination coefficients $\mathbf{b}_k$ of length $n$.
    pub b: Vec<f64>,
    /// Linear combination coefficients $\mathbf{c}_k$ of length $n$.
    pub c: Vec<f64>,
}

/// Differentiable Rank-1 Constraint System (R1CS) & Quadratic Arithmetic Program (QAP) Verifier.
#[derive(Debug, Clone)]
pub struct DifferentiableR1csQap {
    /// Number of variables (length of witness vector $\mathbf{z}$, where $z_0 = 1$).
    pub num_vars: usize,
    /// List of $m$ rank-1 constraints.
    pub constraints: Vec<R1csConstraint>,
    /// Distinct QAP evaluation domain roots $r_1, \dots, r_m$.
    pub domain_roots: Vec<f64>,
}

impl DifferentiableR1csQap {
    /// Creates an empty R1CS system over `num_vars` witness variables.
    pub fn new(num_vars: usize) -> Result<Self> {
        if num_vars == 0 {
            return Err(Error::Msg("num_vars must be >= 1".to_string()));
        }
        Ok(Self {
            num_vars,
            constraints: Vec::new(),
            domain_roots: Vec::new(),
        })
    }

    /// Adds a Rank-1 constraint $(\mathbf{a} \cdot \mathbf{z})(\mathbf{b} \cdot \mathbf{z}) = \mathbf{c} \cdot \mathbf{z}$.
    pub fn add_constraint(&mut self, a: Vec<f64>, b: Vec<f64>, c: Vec<f64>) -> Result<()> {
        if a.len() != self.num_vars || b.len() != self.num_vars || c.len() != self.num_vars {
            return Err(Error::Msg(format!(
                "Constraint vectors must have length num_vars = {}",
                self.num_vars
            )));
        }
        let root = (self.constraints.len() + 1) as f64;
        self.constraints.push(R1csConstraint { a, b, c });
        self.domain_roots.push(root);
        Ok(())
    }

    /// Computes the gate-wise R1CS residual vector $r_k = (\mathbf{a}_k \cdot \mathbf{z})(\mathbf{b}_k \cdot \mathbf{z}) - \mathbf{c}_k \cdot \mathbf{z}$.
    pub fn residuals(&self, witness: &[f64]) -> Result<Vec<f64>> {
        if witness.len() != self.num_vars {
            return Err(Error::Msg(format!(
                "Witness length {} != num_vars {}",
                witness.len(),
                self.num_vars
            )));
        }
        let mut res = Vec::with_capacity(self.constraints.len());
        for con in &self.constraints {
            let az: f64 = con.a.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            let bz: f64 = con.b.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            let cz: f64 = con.c.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            res.push(az * bz - cz);
        }
        Ok(res)
    }

    /// Computes the differentiable R1CS loss $\mathcal{L}_{\text{R1CS}}(\mathbf{z}) = \frac{1}{2} \sum_k r_k^2$
    /// and its exact analytical gradient $\nabla_{\mathbf{z}} \mathcal{L}_{\text{R1CS}}$.
    pub fn loss_and_gradient(&self, witness: &[f64]) -> Result<(f64, Vec<f64>)> {
        if witness.len() != self.num_vars {
            return Err(Error::Msg("Witness length mismatch".to_string()));
        }
        let mut total_loss = 0.0;
        let mut grad = vec![0.0; self.num_vars];

        for con in &self.constraints {
            let az: f64 = con.a.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            let bz: f64 = con.b.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            let cz: f64 = con.c.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            let rk = az * bz - cz;
            total_loss += 0.5 * rk * rk;

            // d(r_k)/d(z_j) = a_{kj} * (b_k . z) + b_{kj} * (a_k . z) - c_{kj}
            for (j, g_j) in grad.iter_mut().enumerate().take(self.num_vars) {
                let drk_dzj = con.a[j] * bz + con.b[j] * az - con.c[j];
                *g_j += rk * drk_dzj;
            }
        }
        Ok((total_loss, grad))
    }

    /// Evaluates the QAP polynomials $(A(\tau), B(\tau), C(\tau), Z(\tau), H(\tau))$ at a challenge point $\tau$
    /// outside the interpolation domain $\{r_1, \dots, r_m\}$ using exact Lagrange basis polynomials $L_k(\tau)$.
    ///
    /// Returns `(p_tau, rhs, gap)` where:
    /// • `p_tau` $= A(\tau) B(\tau) - C(\tau)$
    /// • `rhs` $= H(\tau) Z(\tau)$
    /// • `gap` $= |P(\tau) - H(\tau) Z(\tau)|$, the QAP divisibility gap
    ///   (zero iff the witness satisfies every R1CS constraint).
    pub fn evaluate_qap_at_tau(&self, witness: &[f64], tau: f64) -> Result<(f64, f64, f64)> {
        if witness.len() != self.num_vars {
            return Err(Error::Msg("Witness length mismatch".to_string()));
        }
        let m = self.constraints.len();
        if m == 0 {
            return Err(Error::Msg("QAP requires at least 1 constraint".to_string()));
        }

        // Vanishing polynomial Z(tau) = prod_{k=0}^{m-1} (tau - r_k)
        let mut z_tau = 1.0;
        for &r_k in &self.domain_roots {
            if (tau - r_k).abs() < 1e-12 {
                return Err(Error::Msg(
                    "Challenge point tau must not coincide with a domain root".to_string(),
                ));
            }
            z_tau *= tau - r_k;
        }

        // Compute Lagrange basis polynomials L_k(tau) = prod_{j != k} (tau - r_j) / (r_k - r_j)
        let mut a_tau = 0.0;
        let mut b_tau = 0.0;
        let mut c_tau = 0.0;

        for (k, con) in self.constraints.iter().enumerate() {
            let r_k = self.domain_roots[k];
            let mut l_k = 1.0;
            for (j, &r_j) in self.domain_roots.iter().enumerate() {
                if j != k {
                    l_k *= (tau - r_j) / (r_k - r_j);
                }
            }
            let az_k: f64 = con.a.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            let bz_k: f64 = con.b.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();
            let cz_k: f64 = con.c.iter().zip(witness.iter()).map(|(&x, &z)| x * z).sum();

            a_tau += az_k * l_k;
            b_tau += bz_k * l_k;
            c_tau += cz_k * l_k;
        }

        let p_tau = a_tau * b_tau - c_tau;
        // Compute residual defect sum_k r_k * L_k(tau) (which is 0 iff witness satisfies all R1CS constraints)
        let mut defect_tau = 0.0;
        let res = self.residuals(witness)?;
        for (k, &rk) in res.iter().enumerate() {
            let r_root = self.domain_roots[k];
            let mut l_k = 1.0;
            for (j, &r_j) in self.domain_roots.iter().enumerate() {
                if j != k {
                    l_k *= (tau - r_j) / (r_root - r_j);
                }
            }
            defect_tau += rk * l_k;
        }

        // Exact quotient polynomial value H(tau) = (P(tau) - defect_tau) / Z(tau)
        let h_tau = (p_tau - defect_tau) / z_tau;
        let rhs = h_tau * z_tau;
        Ok((p_tau, rhs, (p_tau - rhs).abs()))
    }
}
