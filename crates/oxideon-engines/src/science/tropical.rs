//! Tropical Algebraic Geometry & Max-Plus Semiring Neural Computation.
//!
//! Provides max-plus tensor algebra, tropical matrices, tropical polynomials,
//! tropical hypersurfaces, and continuous piecewise linear (CPWL) ReLU representations:
//! - Semiring $(\mathbb{R} \cup \{-\infty\}, \oplus, \odot)$ where $a \oplus b = \max(a, b)$ and $a \odot b = a + b$.
//! - Tropical matrix multiplication $(A \odot B)_{ik} = \max_j (A_{ij} + B_{jk})$.
//! - Tropical Kleene star closure $A^* = I \oplus A \oplus A^{\odot 2} \oplus \dots \oplus A^{\odot n}$.
//! - Tropical Polynomials $P(x) = \max_k (c_k + \langle \alpha_k, x \rangle)$ and hypersurface singularity detection.
//! - Hilbert projective metric $d_{\mathbb{TP}}(x, y) = \max_i(x_i - y_i) - \min_i(x_i - y_i)$.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Tropical Matrix over the Max-Plus Semiring.
#[derive(Debug, Clone, PartialEq)]
pub struct TropicalMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl TropicalMatrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(Error::Msg(format!(
                "TropicalMatrix shape [{rows}, {cols}] does not match data length {}",
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn identity(n: usize) -> Result<Self> {
        let mut data =
            crate::layout::try_filled(crate::layout::checked_numel(&[n, n])?, f64::NEG_INFINITY)?;
        for i in 0..n {
            data[i * n + i] = 0.0;
        }
        Ok(Self {
            rows: n,
            cols: n,
            data,
        })
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    /// Tropical Matrix-Vector Multiplication: $(A \odot x)_i = \max_j (A_{ij} + x_j)$.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.cols {
            return Err(Error::Msg(format!(
                "Vector length {} does not match TropicalMatrix cols {}",
                x.len(),
                self.cols
            )));
        }
        let mut out = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut max_val = f64::NEG_INFINITY;
            for (j, &b) in x.iter().enumerate().take(self.cols) {
                let a = self.get(i, j);
                if a > f64::NEG_INFINITY && b > f64::NEG_INFINITY {
                    let sum = a + b;
                    if sum > max_val {
                        max_val = sum;
                    }
                }
            }
            out.push(max_val);
        }
        Ok(out)
    }

    /// Tropical Matrix-Matrix Multiplication: $(A \odot B)_{ik} = \max_j (A_{ij} + B_{jk})$.
    pub fn matmul(&self, other: &TropicalMatrix) -> Result<TropicalMatrix> {
        if self.cols != other.rows {
            return Err(Error::Msg(format!(
                "TropicalMatrix dimension mismatch: [{}, {}] x [{}, {}]",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut out_data = vec![f64::NEG_INFINITY; self.rows * other.cols];
        for i in 0..self.rows {
            for k in 0..other.cols {
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..self.cols {
                    let a = self.get(i, j);
                    let b = other.get(j, k);
                    if a > f64::NEG_INFINITY && b > f64::NEG_INFINITY {
                        let sum = a + b;
                        if sum > max_val {
                            max_val = sum;
                        }
                    }
                }
                out_data[i * other.cols + k] = max_val;
            }
        }
        TropicalMatrix::new(self.rows, other.cols, out_data)
    }

    /// Tropical Matrix Addition: $(A \oplus B)_{ij} = \max(A_{ij}, B_{ij})$.
    pub fn add(&self, other: &TropicalMatrix) -> Result<TropicalMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(Error::Msg(format!(
                "TropicalMatrix shape mismatch for addition: [{}, {}] vs [{}, {}]",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let out_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a.max(b))
            .collect();
        TropicalMatrix::new(self.rows, self.cols, out_data)
    }

    /// Computes the Kleene Star Closure $A^* = I \oplus A \oplus A^{\odot 2} \oplus \dots \oplus A^{\odot n}$.
    pub fn kleene_star(&self) -> Result<TropicalMatrix> {
        if self.rows != self.cols {
            return Err(Error::Msg("Kleene star requires square matrix".into()));
        }
        let n = self.rows;
        let mut current_star = TropicalMatrix::identity(n)?;
        let mut power = TropicalMatrix::identity(n)?;

        for _ in 1..=n {
            power = power.matmul(self)?;
            current_star = current_star.add(&power)?;
        }
        Ok(current_star)
    }
}

/// Tropical Monomial: $c \odot x^\alpha = c + \sum_i \alpha_i x_i$.
#[derive(Debug, Clone)]
pub struct TropicalMonomial {
    pub coefficient: f64,
    pub exponents: Vec<f64>,
}

/// Tropical Polynomial representing continuous piecewise linear (CPWL) functions.
///
/// $P(x) = \bigoplus_{k} c_k \odot x^{\alpha_k} = \max_k \left( c_k + \sum_i \alpha_{ki} x_i \right)$.
#[derive(Debug, Clone)]
pub struct TropicalPolynomial {
    pub monomials: Vec<TropicalMonomial>,
    pub num_variables: usize,
}

impl TropicalPolynomial {
    pub fn new(num_variables: usize) -> Self {
        Self {
            monomials: Vec::new(),
            num_variables,
        }
    }

    pub fn add_monomial(&mut self, coefficient: f64, exponents: Vec<f64>) -> Result<()> {
        if exponents.len() != self.num_variables {
            return Err(Error::Msg(format!(
                "Monomial exponents length {} != num_variables {}",
                exponents.len(),
                self.num_variables
            )));
        }
        self.monomials.push(TropicalMonomial {
            coefficient,
            exponents,
        });
        Ok(())
    }

    /// Evaluates the tropical polynomial at a point $x \in \mathbb{R}^n$.
    pub fn evaluate(&self, x: &[f64]) -> Result<f64> {
        if x.len() != self.num_variables {
            return Err(Error::Msg(format!(
                "Evaluation point dim {} != num_variables {}",
                x.len(),
                self.num_variables
            )));
        }
        if self.monomials.is_empty() {
            return Ok(f64::NEG_INFINITY);
        }

        let mut max_val = f64::NEG_INFINITY;
        for m in &self.monomials {
            let dot: f64 = m.exponents.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            let val = m.coefficient + dot;
            if val > max_val {
                max_val = val;
            }
        }
        Ok(max_val)
    }

    /// Determines if a point $x$ lies on the Tropical Hypersurface $\mathcal{T}(P)$.
    ///
    /// In tropical geometry, the hypersurface is the set of points where the maximum
    /// in $P(x)$ is achieved by at least two distinct monomials (non-differentiability locus).
    pub fn is_on_hypersurface(&self, x: &[f64], tolerance: f64) -> Result<bool> {
        if x.len() != self.num_variables {
            return Err(Error::Msg(format!(
                "Point dimension {} != num_variables {}",
                x.len(),
                self.num_variables
            )));
        }
        if self.monomials.len() < 2 {
            return Ok(false);
        }

        let mut values: Vec<f64> = self
            .monomials
            .iter()
            .map(|m| {
                let dot: f64 = m.exponents.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
                m.coefficient + dot
            })
            .collect();

        values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        // Check difference between top-1 and top-2 evaluated values
        Ok((values[0] - values[1]).abs() <= tolerance)
    }
}

/// Hilbert Projective Metric on Tropical Projective Space $\mathbb{TP}^{n-1}$.
///
/// $d_{\mathbb{TP}}(x, y) = \max_{1 \le i \le n} (x_i - y_i) - \min_{1 \le i \le n} (x_i - y_i)$.
///
/// Invariant under tropical scalar multiplication (translation by a constant vector $(c, c, \dots, c)$).
pub fn tropical_distance(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err(Error::Msg(format!(
            "Vectors must be non-empty and have matching dimensions (got {} vs {})",
            x.len(),
            y.len()
        )));
    }

    let mut max_diff = f64::NEG_INFINITY;
    let mut min_diff = f64::INFINITY;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let diff = xi - yi;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff < min_diff {
            min_diff = diff;
        }
    }

    Ok(max_diff - min_diff)
}

/// Converts a Tropical Matrix to a standard `Tensor`.
impl TryFrom<TropicalMatrix> for Tensor {
    type Error = Error;
    fn try_from(tm: TropicalMatrix) -> Result<Self> {
        Tensor::from_vec(tm.data, &[tm.rows, tm.cols])
    }
}
