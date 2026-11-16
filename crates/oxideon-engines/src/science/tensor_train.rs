//! Tensor Train (TT) Decompositions & High-Dimensional Multilinear Solvers (Oseledets, 2011).
//!
//! # Mathematical Foundations
//!
//! 1. **Tensor Train (TT) Format**:
//!    A $d$-dimensional tensor $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times \dots \times n_d}$ is factored into
//!    a chain of 3-way cores $G_k \in \mathbb{R}^{r_{k-1} \times n_k \times r_k}$ ($k = 1, \dots, d$) with $r_0 = r_d = 1$:
//!    $$\mathcal{A}(i_1, i_2, \dots, i_d) = \sum_{\alpha_1=1}^{r_1} \cdots \sum_{\alpha_{d-1}=1}^{r_{d-1}} G_1(1, i_1, \alpha_1) G_2(\alpha_1, i_2, \alpha_2) \cdots G_d(\alpha_{d-1}, i_d, 1)$$
//!
//!    Storage complexity collapses from exponential $O(n^d)$ to linear $O(d n r^2)$ in dimension $d$.
//!
//! 2. **Exponentially Faster Multilinear Contractions**:
//!    - **Entry Evaluation**: $O(d r^2)$ sequential matrix-vector products.
//!    - **Global Sum / Partition Function**: $\sum_{i_1, \dots, i_d} \mathcal{A}(i_1, \dots, i_d) = \prod_{k=1}^d \left(\sum_{i_k=1}^{n_k} G_k(:, i_k, :)\right)$ in $O(d n r^2)$.
//!    - **TT Inner Product & Frobenius Norm**: $\langle \mathcal{A}, \mathcal{B} \rangle$ computed in $O(d n r^3)$ via transfer-matrix sweep.

use crate::error::{Error, Result};

/// A single 3-way core $G_k \in \mathbb{R}^{r_{\text{left}} \times n_k \times r_{\text{right}}}$ of a Tensor Train.
#[derive(Debug, Clone, PartialEq)]
pub struct TtCore {
    /// Left TT-rank $r_{k-1}$.
    pub r_left: usize,
    /// Physical mode size $n_k$.
    pub mode_size: usize,
    /// Right TT-rank $r_k$.
    pub r_right: usize,
    /// Flattened data of shape `[r_left, mode_size, r_right]`.
    pub data: Vec<f64>,
}

impl TtCore {
    /// Constructs a validated TT-core.
    pub fn new(r_left: usize, mode_size: usize, r_right: usize, data: Vec<f64>) -> Result<Self> {
        if r_left == 0 || mode_size == 0 || r_right == 0 {
            return Err(Error::Msg("TtCore dimensions must all be > 0".to_string()));
        }
        let expect = r_left
            .checked_mul(mode_size)
            .and_then(|v| v.checked_mul(r_right))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "TtCore dimensions {r_left} * {mode_size} * {r_right} overflow"
                ))
            })?;
        if data.len() != expect {
            return Err(Error::Msg(format!(
                "TtCore data length {} != {} * {} * {}",
                data.len(),
                r_left,
                mode_size,
                r_right
            )));
        }
        Ok(Self {
            r_left,
            mode_size,
            r_right,
            data,
        })
    }

    /// Accesses element $G_k(\alpha, i, \beta)$.
    #[inline]
    pub fn get(&self, alpha: usize, i: usize, beta: usize) -> f64 {
        self.data[(alpha * self.mode_size + i) * self.r_right + beta]
    }
}

/// High-dimensional Tensor Train (TT) representation over $d$ modes.
#[derive(Debug, Clone)]
pub struct TensorTrain {
    /// Sequence of $d$ TT-cores $G_1, \dots, G_d$.
    pub cores: Vec<TtCore>,
}

impl TensorTrain {
    /// Constructs a Tensor Train from a sequence of cores, verifying rank compatibility
    /// ($r_0 = 1$, $r_d = 1$, and $G_k.\text{r\_right} == G_{k+1}.\text{r\_left}$).
    pub fn new(cores: Vec<TtCore>) -> Result<Self> {
        if cores.is_empty() {
            return Err(Error::Msg(
                "TensorTrain must have at least 1 core".to_string(),
            ));
        }
        if cores[0].r_left != 1 {
            return Err(Error::Msg(format!(
                "First TT-core must have r_left == 1, got {}",
                cores[0].r_left
            )));
        }
        let d = cores.len();
        if cores[d - 1].r_right != 1 {
            return Err(Error::Msg(format!(
                "Last TT-core must have r_right == 1, got {}",
                cores[d - 1].r_right
            )));
        }
        for k in 0..(d - 1) {
            if cores[k].r_right != cores[k + 1].r_left {
                return Err(Error::Msg(format!(
                    "TT-rank mismatch between core {k} (r_right={}) and core {} (r_left={})",
                    cores[k].r_right,
                    k + 1,
                    cores[k + 1].r_left
                )));
            }
        }
        Ok(Self { cores })
    }

    /// Constructs a rank-1 separable $d$-dimensional Tensor Train from 1D factor vectors:
    /// $\mathcal{A}(i_1, \dots, i_d) = \prod_{k=1}^d v_k\[i_k\]$.
    pub fn from_rank1_factors(factors: Vec<Vec<f64>>) -> Result<Self> {
        if factors.is_empty() {
            return Err(Error::Msg("Factors cannot be empty".to_string()));
        }
        let mut cores = Vec::with_capacity(factors.len());
        for f in factors {
            let n = f.len();
            cores.push(TtCore::new(1, n, 1, f)?);
        }
        Self::new(cores)
    }

    /// Constructs an exact Rank-2 Tensor Train for the $d$-dimensional additive function:
    /// $\mathcal{A}(i_1, \dots, i_d) = \sum_{k=1}^d f_k(i_k)$
    /// using the canonical $2 \times 2$ upper-triangular TT-core representation:
    ///
    /// $$G_1 = \begin{pmatrix} f_1(i_1) & 1 \end{pmatrix}, \quad G_k = \begin{pmatrix} 1 & 0 \\ f_k(i_k) & 1 \end{pmatrix}, \quad G_d = \begin{pmatrix} 1 \\ f_d(i_d) \end{pmatrix}$$
    pub fn from_additive_terms(terms: Vec<Vec<f64>>) -> Result<Self> {
        let d = terms.len();
        if d == 0 {
            return Err(Error::Msg("Additive terms cannot be empty".to_string()));
        }
        if d == 1 {
            return Self::from_rank1_factors(terms);
        }

        let mut cores = Vec::with_capacity(d);

        // First core: [1, n_1, 2] -> [f_1(i), 1.0]
        let n0 = terms[0].len();
        let mut d0 = Vec::with_capacity(n0 * 2);
        for &val in &terms[0] {
            d0.push(val);
            d0.push(1.0);
        }
        cores.push(TtCore::new(1, n0, 2, d0)?);

        // Interior cores: [2, n_k, 2]
        // Row 0: [1.0, 0.0], Row 1: [f_k(i), 1.0]
        for term in terms.iter().take(d - 1).skip(1) {
            let nk = term.len();
            let mut dk = vec![0.0; 2 * nk * 2];
            for (i, &val) in term.iter().enumerate() {
                // alpha = 0, mode = i: [1.0, 0.0]
                dk[(i) * 2] = 1.0;
                dk[(i) * 2 + 1] = 0.0;
                // alpha = 1, mode = i: [val, 1.0]
                dk[(nk + i) * 2] = val;
                dk[(nk + i) * 2 + 1] = 1.0;
            }
            cores.push(TtCore::new(2, nk, 2, dk)?);
        }

        // Last core: [2, n_d, 1] -> alpha=0: 1.0, alpha=1: f_d(i)
        let nd = terms[d - 1].len();
        let mut dd = vec![0.0; 2 * nd];
        for (i, &val) in terms[d - 1].iter().enumerate() {
            dd[i] = 1.0;
            dd[nd + i] = val;
        }
        cores.push(TtCore::new(2, nd, 1, dd)?);

        Self::new(cores)
    }

    /// Number of dimensions $d$.
    pub fn ndim(&self) -> usize {
        self.cores.len()
    }

    /// Returns the mode sizes $[n_1, n_2, \dots, n_d]$.
    pub fn shape(&self) -> Vec<usize> {
        self.cores.iter().map(|c| c.mode_size).collect()
    }

    /// Total number of floating-point parameters stored across all TT-cores.
    pub fn num_parameters(&self) -> usize {
        self.cores.iter().map(|c| c.data.len()).sum()
    }

    /// Evaluates a single entry $\mathcal{A}(i_1, \dots, i_d)$ in $O(d r^2)$ operations.
    pub fn evaluate(&self, indices: &[usize]) -> Result<f64> {
        if indices.len() != self.cores.len() {
            return Err(Error::Msg(format!(
                "Index dimension {} != TT ndim {}",
                indices.len(),
                self.cores.len()
            )));
        }

        let mut vec = vec![1.0]; // shape [1]
        for (k, &idx) in indices.iter().enumerate() {
            let core = &self.cores[k];
            if idx >= core.mode_size {
                return Err(Error::Msg(format!(
                    "Index {idx} out of bounds for mode {k} (size {})",
                    core.mode_size
                )));
            }
            let mut next_vec = vec![0.0; core.r_right];
            for (alpha, &v_alpha) in vec.iter().enumerate().take(core.r_left) {
                for (beta, next_b) in next_vec.iter_mut().enumerate().take(core.r_right) {
                    *next_b += v_alpha * core.get(alpha, idx, beta);
                }
            }
            vec = next_vec;
        }
        Ok(vec[0])
    }

    /// Computes the exact sum over all $n_1 \times \dots \times n_d$ entries in $O(d n r^2)$ time
    /// without materializing the full tensor.
    pub fn sum_all(&self) -> f64 {
        let mut vec = vec![1.0];
        for core in &self.cores {
            let mut next_vec = vec![0.0; core.r_right];
            for (alpha, &v_alpha) in vec.iter().enumerate().take(core.r_left) {
                for i in 0..core.mode_size {
                    for (beta, next_b) in next_vec.iter_mut().enumerate().take(core.r_right) {
                        *next_b += v_alpha * core.get(alpha, i, beta);
                    }
                }
            }
            vec = next_vec;
        }
        vec[0]
    }

    /// Computes the exact multilinear inner product $\langle \mathcal{A}, \mathcal{B} \rangle$ in $O(d n r^3)$ time.
    pub fn dot(&self, other: &Self) -> Result<f64> {
        if self.cores.len() != other.cores.len() {
            return Err(Error::Msg("TT dimension mismatch in dot".to_string()));
        }

        // Maintain transfer matrix M of shape [rA_k, rB_k], initially 1x1 = [[1.0]]
        let mut m = vec![1.0];
        let mut ra_prev = 1;
        let mut rb_prev = 1;

        for (ca, cb) in self.cores.iter().zip(other.cores.iter()) {
            if ca.mode_size != cb.mode_size {
                return Err(Error::Msg("TT mode size mismatch in dot".to_string()));
            }
            let n = ca.mode_size;
            let ra_next = ca.r_right;
            let rb_next = cb.r_right;

            // Contract M[alpha_a, alpha_b] with ca[alpha_a, i, beta_a] and cb[alpha_b, i, beta_b]
            let mut m_next = vec![0.0; ra_next * rb_next];
            for i in 0..n {
                // First intermediate: T[alpha_b, beta_a] = sum_{alpha_a} M[alpha_a, alpha_b] * ca[alpha_a, i, beta_a]
                let mut temp = vec![0.0; rb_prev * ra_next];
                for alpha_a in 0..ra_prev {
                    for alpha_b in 0..rb_prev {
                        let m_val = m[alpha_a * rb_prev + alpha_b];
                        for beta_a in 0..ra_next {
                            temp[alpha_b * ra_next + beta_a] += m_val * ca.get(alpha_a, i, beta_a);
                        }
                    }
                }
                // Second step: m_next[beta_a, beta_b] += sum_{alpha_b} temp[alpha_b, beta_a] * cb[alpha_b, i, beta_b]
                for alpha_b in 0..rb_prev {
                    for beta_a in 0..ra_next {
                        let t_val = temp[alpha_b * ra_next + beta_a];
                        for beta_b in 0..rb_next {
                            m_next[beta_a * rb_next + beta_b] += t_val * cb.get(alpha_b, i, beta_b);
                        }
                    }
                }
            }

            m = m_next;
            ra_prev = ra_next;
            rb_prev = rb_next;
        }

        Ok(m[0])
    }

    /// Computes the exact Frobenius norm $\|\mathcal{A}\|_F = \sqrt{\langle \mathcal{A}, \mathcal{A} \rangle}$.
    pub fn frobenius_norm(&self) -> Result<f64> {
        let sq = self.dot(self)?;
        Ok(sq.max(0.0).sqrt())
    }
}
