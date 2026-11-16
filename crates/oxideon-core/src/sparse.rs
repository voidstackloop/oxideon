//! Sparse Tensor formats (`SparseCoo` Coordinate & `SparseCsr` Compressed Sparse Row)
//! and differentiable Sparse-Dense Matrix Multiplication (`spmm`).

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::backend::{CpuStorage, Device, Storage};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

/// 2D Sparse Matrix in Coordinate (`COO`) format with coalesced `(row, col)` entries.
#[derive(Debug, Clone)]
pub struct SparseCoo {
    pub indices: Vec<(usize, usize)>,
    pub values: Vec<f64>,
    pub shape: [usize; 2],
}

impl SparseCoo {
    /// Constructs a coalesced [`SparseCoo`] matrix, summing any duplicate `(row, col)` entries.
    ///
    /// NOTE — deliberate `torch.sparse` divergence: exact zeros are dropped from
    /// storage while `coalesce()` keeps explicit zeros. Values (`to_dense`,
    /// `spmm`) are identical either way; only `nnz` differs.
    pub fn new(indices: Vec<(usize, usize)>, values: Vec<f64>, shape: [usize; 2]) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(Error::Msg(format!(
                "SparseCoo: indices.len() ({}) != values.len() ({})",
                indices.len(),
                values.len()
            )));
        }
        let [rows, cols] = shape;
        let mut map: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for (&(r, c), &v) in indices.iter().zip(&values) {
            if r >= rows || c >= cols {
                return Err(Error::Msg(format!(
                    "SparseCoo index ({r}, {c}) out of bounds for shape [{rows}, {cols}]"
                )));
            }
            *map.entry((r, c)).or_insert(0.0) += v;
        }
        let mut coal_indices = Vec::with_capacity(map.len());
        let mut coal_values = Vec::with_capacity(map.len());
        for ((r, c), v) in map {
            // Drop exact zeros but keep NaN (`NaN != 0.0`): sparse must
            // never silently swallow non-finite values.
            if v != 0.0 {
                coal_indices.push((r, c));
                coal_values.push(v);
            }
        }
        Ok(Self {
            indices: coal_indices,
            values: coal_values,
            shape,
        })
    }

    /// Extracts all entries with $|x_{ij}| > \text{threshold}$ from a 2D [`Tensor`].
    pub fn from_dense(tensor: &Tensor, threshold: f64) -> Result<Self> {
        if tensor.ndim() != 2 {
            return Err(Error::Msg(format!(
                "SparseCoo::from_dense expects 2D tensor, got {:?}",
                tensor.shape()
            )));
        }
        let (rows, cols) = (tensor.shape()[0], tensor.shape()[1]);
        let vals = tensor.to_vec::<f64>()?;
        let mut indices = Vec::new();
        let mut nz_vals = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = vals[r * cols + c];
                if v.abs() > threshold || v.is_nan() {
                    indices.push((r, c));
                    nz_vals.push(v);
                }
            }
        }
        Ok(Self {
            indices,
            values: nz_vals,
            shape: [rows, cols],
        })
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Validates a (possibly hand-built) COO struct: the public fields bypass
    /// [`SparseCoo::new`], so every consuming method calls this first and
    /// returns `Err` instead of panicking on out-of-bounds indices.
    pub fn check_valid(&self) -> Result<()> {
        if self.indices.len() != self.values.len() {
            return Err(Error::Msg(format!(
                "SparseCoo: indices.len() ({}) != values.len() ({})",
                self.indices.len(),
                self.values.len()
            )));
        }
        let [rows, cols] = self.shape;
        for &(r, c) in &self.indices {
            if r >= rows || c >= cols {
                return Err(Error::Msg(format!(
                    "SparseCoo index ({r}, {c}) out of bounds for shape [{rows}, {cols}]"
                )));
            }
        }
        Ok(())
    }

    /// Returns the transposed COO matrix of shape `[cols, rows]`.
    ///
    /// Validates the struct first: the fields are public, so a hand-built
    /// instance can carry out-of-bounds indices or mismatched lengths, which
    /// is an error here (like [`SparseCoo::to_csr`]) instead of silently
    /// transposing garbage.
    pub fn transpose(&self) -> Result<Self> {
        self.check_valid()?;
        let mut map: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for (&(r, c), &v) in self.indices.iter().zip(&self.values) {
            // Sum duplicates like `new`/`to_dense` (insert would keep one).
            *map.entry((c, r)).or_insert(0.0) += v;
        }
        let (indices, values): (Vec<_>, Vec<_>) = map.into_iter().unzip();
        Ok(Self {
            indices,
            values,
            shape: [self.shape[1], self.shape[0]],
        })
    }

    pub fn to_dense(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        self.check_valid()?;
        let [rows, cols] = self.shape;
        let n = crate::limits::check_elements("SparseCoo::to_dense", &[rows, cols])?;
        let mut dense = vec![0.0f64; n];
        for (&(r, c), &v) in self.indices.iter().zip(&self.values) {
            dense[r * cols + c] += v;
        }
        Tensor::from_slice_on(&dense, &[rows, cols], device)?.to_dtype(dtype)
    }

    /// Converts this `COO` matrix into Compressed Sparse Row ([`SparseCsr`]) format.
    pub fn to_csr(&self) -> Result<SparseCsr> {
        self.check_valid()?;
        let [rows, cols] = self.shape;
        rows.checked_add(1).ok_or_else(|| {
            Error::Msg(format!("SparseCoo::to_csr: rows {rows} overflows row_ptr"))
        })?;
        // Sort by (row, col): hand-built COO indices may be unsorted, which
        // would otherwise produce an invalid CSR (row_ptr counts must be grouped).
        let mut pairs: Vec<((usize, usize), f64)> = self
            .indices
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .collect();
        pairs.sort_by_key(|&(rc, _)| rc);
        // Merge duplicate (row, col) pairs so the CSR is canonical; without
        // this, `to_dense` (assign) and `spmm` (accumulate) disagree.
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_idx = Vec::with_capacity(pairs.len());
        let mut values: Vec<f64> = Vec::with_capacity(pairs.len());
        let mut last: Option<(usize, usize)> = None;
        for ((r, c), v) in pairs {
            if last == Some((r, c)) {
                let tail = values.len() - 1;
                values[tail] += v;
            } else {
                row_ptr[r + 1] += 1;
                col_idx.push(c);
                values.push(v);
                last = Some((r, c));
            }
        }
        for r in 0..rows {
            row_ptr[r + 1] += row_ptr[r];
        }
        Ok(SparseCsr {
            row_ptr,
            col_idx,
            values,
            shape: [rows, cols],
        })
    }

    /// Fraction of zero entries in $[0.0, 1.0]$.
    pub fn sparsity(&self) -> f64 {
        let total = ((self.shape[0] as f64) * (self.shape[1] as f64)).max(1.0);
        1.0 - (self.nnz() as f64) / total
    }

    /// Computes Frobenius norm: $\sqrt{\sum v_i^2}$.
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Re-coalesces COO entries and eliminates zero entries.
    pub fn coalesce(&self) -> Result<Self> {
        self.check_valid()?;
        Self::new(self.indices.clone(), self.values.clone(), self.shape)
    }

    /// Differentiable Sparse-Dense Matrix Multiplication via CSR.
    pub fn spmm(&self, dense: &Tensor) -> Result<Tensor> {
        self.to_csr()?.spmm(dense)
    }

    /// Sparse-Sparse Matrix Multiplication (`SpSpMM`) for COO via CSR.
    pub fn spspmm(&self, other: &Self) -> Result<Self> {
        let csr_a = self.to_csr()?;
        let csr_b = other.to_csr()?;
        csr_a.spspmm(&csr_b)?.to_coo()
    }

    /// Elementwise addition of two COO matrices of matching shape.
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.check_valid()?;
        other.check_valid()?;
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                op: "SparseCoo::add",
                expected: self.shape.to_vec(),
                got: other.shape.to_vec(),
            });
        }
        let mut combined_indices = Vec::with_capacity(self.indices.len() + other.indices.len());
        let mut combined_values = Vec::with_capacity(self.values.len() + other.values.len());
        combined_indices.extend_from_slice(&self.indices);
        combined_indices.extend_from_slice(&other.indices);
        combined_values.extend_from_slice(&self.values);
        combined_values.extend_from_slice(&other.values);
        Self::new(combined_indices, combined_values, self.shape)
    }

    /// Elementwise subtraction: `self - other`.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let neg_other = other.mul_scalar(-1.0)?;
        self.add(&neg_other)
    }

    /// Multiplies all entries by a scalar.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        self.check_valid()?;
        if scalar == 0.0 {
            return Ok(Self {
                indices: Vec::new(),
                values: Vec::new(),
                shape: self.shape,
            });
        }
        let mut new_vals = Vec::with_capacity(self.values.len());
        for &v in &self.values {
            new_vals.push(v * scalar);
        }
        Ok(Self {
            indices: self.indices.clone(),
            values: new_vals,
            shape: self.shape,
        })
    }
}

/// 2D Sparse Matrix in Compressed Sparse Row (`CSR`) format with differentiable `spmm`.
#[derive(Debug, Clone)]
pub struct SparseCsr {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
    pub shape: [usize; 2],
}

impl SparseCsr {
    /// Constructs a [`SparseCsr`] matrix from raw components, validating canonical invariants.
    pub fn new(
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
        shape: [usize; 2],
    ) -> Result<Self> {
        let csr = Self {
            row_ptr,
            col_idx,
            values,
            shape,
        };
        csr.check_valid()?;
        Ok(csr)
    }

    pub fn from_dense(tensor: &Tensor, threshold: f64) -> Result<Self> {
        SparseCoo::from_dense(tensor, threshold)?.to_csr()
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Fraction of zero entries in $[0.0, 1.0]$.
    pub fn sparsity(&self) -> f64 {
        // f64 product: cannot panic on huge hand-built shapes.
        let total = ((self.shape[0] as f64) * (self.shape[1] as f64)).max(1.0);
        1.0 - (self.nnz() as f64) / total
    }

    /// Validates a (possibly hand-built) CSR struct: canonical invariants are
    /// `row_ptr.len() == rows + 1`, monotonic `row_ptr`, `row_ptr[rows] ==
    /// col_idx.len() == values.len()`, and every column index in range.
    pub fn check_valid(&self) -> Result<()> {
        let [rows, cols] = self.shape;
        let need = rows
            .checked_add(1)
            .ok_or_else(|| Error::Msg(format!("SparseCsr: rows {rows} overflows row_ptr")))?;
        if self.row_ptr.len() != need {
            return Err(Error::Msg(format!(
                "SparseCsr: row_ptr.len() ({}) != rows + 1 ({need})",
                self.row_ptr.len()
            )));
        }
        if self.col_idx.len() != self.values.len() {
            return Err(Error::Msg(format!(
                "SparseCsr: col_idx.len() ({}) != values.len() ({})",
                self.col_idx.len(),
                self.values.len()
            )));
        }
        let mut prev = 0usize;
        for &p in &self.row_ptr {
            if p < prev {
                return Err(Error::Msg("SparseCsr: row_ptr is not monotonic".into()));
            }
            prev = p;
        }
        if prev != self.col_idx.len() {
            return Err(Error::Msg(format!(
                "SparseCsr: row_ptr[rows] ({prev}) != nnz ({})",
                self.col_idx.len()
            )));
        }
        for &c in &self.col_idx {
            if c >= cols {
                return Err(Error::Msg(format!(
                    "SparseCsr: col index {c} out of bounds for {cols} cols"
                )));
            }
        }
        Ok(())
    }

    /// Returns the transposed CSR matrix $S^T$ of shape `[cols, rows]`.
    pub fn transpose(&self) -> Result<Self> {
        self.check_valid()?;
        let [rows, cols] = self.shape;
        cols.checked_add(1).ok_or_else(|| {
            Error::Msg(format!(
                "SparseCsr::transpose: cols {cols} overflows row_ptr"
            ))
        })?;
        let mut entries = Vec::with_capacity(self.nnz());
        for r in 0..rows {
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                entries.push(((self.col_idx[idx], r), self.values[idx]));
            }
        }
        entries.sort_by_key(|&(rc, _)| rc);
        let mut row_ptr = vec![0usize; cols + 1];
        let mut col_idx = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());
        for ((tr, tc), v) in entries {
            row_ptr[tr + 1] += 1;
            col_idx.push(tc);
            values.push(v);
        }
        for r in 0..cols {
            row_ptr[r + 1] += row_ptr[r];
        }
        Ok(Self {
            row_ptr,
            col_idx,
            values,
            shape: [cols, rows],
        })
    }

    pub fn to_dense(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        self.check_valid()?;
        let [rows, cols] = self.shape;
        let n = crate::limits::check_elements("SparseCsr::to_dense", &[rows, cols])?;
        let mut dense = vec![0.0f64; n];
        for r in 0..rows {
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                let c = self.col_idx[idx];
                // Accumulate (not assign) so hand-built duplicate entries
                // agree with `spmm`, matching COO `to_dense`.
                dense[r * cols + c] += self.values[idx];
            }
        }
        Tensor::from_slice_on(&dense, &[rows, cols], device)?.to_dtype(dtype)
    }

    /// Differentiable Sparse-Dense Matrix Multiplication (`SpMM`):
    /// Computes $Y = S X$ where $S$ is `[M, K]` sparse CSR and `dense` $X$ is `[K, N]` dense [`Tensor`].
    /// Backward pass w.r.t. $X$: $\nabla_X \mathcal{L} = S^T \nabla_Y \mathcal{L}$ (also computed via CSR `spmm`)!
    pub fn spmm(&self, dense: &Tensor) -> Result<Tensor> {
        if dense.ndim() != 2 || dense.shape()[0] != self.shape[1] {
            return Err(Error::ShapeMismatch {
                op: "SparseCsr::spmm",
                expected: vec![self.shape[1], dense.shape().get(1).copied().unwrap_or(0)],
                got: dense.shape().to_vec(),
            });
        }
        self.check_valid()?;
        let [m, _k] = self.shape;
        let n = dense.shape()[1];
        let x_vec = dense.to_vec::<f64>()?;
        let y_len = crate::limits::check_elements("SparseCsr::spmm", &[m, n])?;
        let mut y_vec = vec![0.0f64; y_len];

        for r in 0..m {
            let y_row = &mut y_vec[r * n..(r + 1) * n];
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                let c = self.col_idx[idx];
                let val = self.values[idx];
                let x_row = &x_vec[c * n..(c + 1) * n];
                for j in 0..n {
                    y_row[j] += val * x_row[j];
                }
            }
        }

        let out_layout = Layout::contiguous(vec![m, n]);
        let cpu_s = CpuStorage::F64(y_vec).to_dtype(&out_layout, dense.dtype())?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &dense.device())?;

        let st = Arc::new(self.transpose()?);
        Ok(Tensor::from_op(
            storage,
            out_layout,
            "spmm",
            vec![dense.clone()],
            move |grad_out| Ok(vec![Some(st.spmm(grad_out)?)]),
        ))
    }

    /// Converts this `CSR` matrix into Coordinate ([`SparseCoo`]) format.
    pub fn to_coo(&self) -> Result<SparseCoo> {
        self.check_valid()?;
        let [rows, cols] = self.shape;
        let mut indices = Vec::with_capacity(self.nnz());
        let mut values = Vec::with_capacity(self.nnz());
        for r in 0..rows {
            for idx in self.row_ptr[r]..self.row_ptr[r + 1] {
                indices.push((r, self.col_idx[idx]));
                values.push(self.values[idx]);
            }
        }
        SparseCoo::new(indices, values, [rows, cols])
    }

    /// Computes Frobenius norm: $\sqrt{\sum v_i^2}$.
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Elementwise addition of two CSR matrices of matching shape.
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.check_valid()?;
        other.check_valid()?;
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                op: "SparseCsr::add",
                expected: self.shape.to_vec(),
                got: other.shape.to_vec(),
            });
        }
        let coo_a = self.to_coo()?;
        let coo_b = other.to_coo()?;
        coo_a.add(&coo_b)?.to_csr()
    }

    /// Elementwise subtraction: `self - other`.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let neg_other = other.mul_scalar(-1.0)?;
        self.add(&neg_other)
    }

    /// Multiplies all entries by a scalar.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        self.check_valid()?;
        if scalar == 0.0 {
            return Ok(Self {
                row_ptr: vec![0; self.shape[0] + 1],
                col_idx: Vec::new(),
                values: Vec::new(),
                shape: self.shape,
            });
        }
        let mut new_vals = Vec::with_capacity(self.values.len());
        for &v in &self.values {
            new_vals.push(v * scalar);
        }
        Ok(Self {
            row_ptr: self.row_ptr.clone(),
            col_idx: self.col_idx.clone(),
            values: new_vals,
            shape: self.shape,
        })
    }

    /// Sparse-Sparse Matrix Multiplication (`SpSpMM`):
    /// Computes $C = A B$ where $A$ is `[M, K]` CSR and $B$ is `[K, N]` CSR using Gustavson's algorithm.
    pub fn spspmm(&self, other: &Self) -> Result<Self> {
        self.check_valid()?;
        other.check_valid()?;
        let [m, k_a] = self.shape;
        let [k_b, n] = other.shape;
        if k_a != k_b {
            return Err(Error::ShapeMismatch {
                op: "SparseCsr::spspmm",
                expected: vec![k_a, other.shape[1]],
                got: other.shape.to_vec(),
            });
        }

        let mut row_ptr = Vec::with_capacity(m + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        let mut row_map: BTreeMap<usize, f64> = BTreeMap::new();
        for i in 0..m {
            row_map.clear();
            for idx_a in self.row_ptr[i]..self.row_ptr[i + 1] {
                let k = self.col_idx[idx_a];
                let val_a = self.values[idx_a];
                for idx_b in other.row_ptr[k]..other.row_ptr[k + 1] {
                    let j = other.col_idx[idx_b];
                    let val_b = other.values[idx_b];
                    *row_map.entry(j).or_insert(0.0) += val_a * val_b;
                }
            }
            for (col, val) in &row_map {
                if *val != 0.0 {
                    col_idx.push(*col);
                    values.push(*val);
                }
            }
            row_ptr.push(values.len());
        }

        Ok(Self {
            row_ptr,
            col_idx,
            values,
            shape: [m, n],
        })
    }
}

/// A CSR sparsity pattern whose values are a learnable [`Tensor`]: `Y = S X` is differentiable with respect to both the
/// non-zero values (the gradient of a value is the dot product of the matching row of `dY` and row of `X`, i.e. an SDDMM) and
/// the dense operand (`S^T dY`). Use it for sparse layers, pruned weights and graph attention coefficients.
#[derive(Clone)]
pub struct SparseCsrParam {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub shape: [usize; 2],
    /// The `nnz` values, in CSR order; mark it `requires_grad` to learn them.
    pub values: Tensor,
}

impl SparseCsrParam {
    /// Takes the pattern of `csr` and its values as a tensor of dtype `dtype` (F32 or F64).
    pub fn from_csr(csr: &SparseCsr, dtype: DType, requires_grad: bool) -> Result<Self> {
        csr.check_valid()?;
        let v = Tensor::from_slice(&csr.values, &[csr.values.len()])?.to_dtype(dtype)?;
        Ok(Self { row_ptr: csr.row_ptr.clone(), col_idx: csr.col_idx.clone(), shape: csr.shape, values: if requires_grad { v.requires_grad_(true) } else { v } })
    }

    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// The matrix with the current values (detached).
    pub fn to_csr(&self) -> Result<SparseCsr> {
        SparseCsr::new(self.row_ptr.clone(), self.col_idx.clone(), self.values.detach().to_dtype(DType::F64)?.to_vec::<f64>()?, self.shape)
    }

    pub fn to_dense(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        self.to_csr()?.to_dense(dtype, device)
    }

    /// `Y = S X` for a dense `[K, N]` operand.
    pub fn spmm(&self, dense: &Tensor) -> Result<Tensor> {
        let [m, k] = self.shape;
        if dense.ndim() != 2 || dense.shape()[0] != k || self.values.numel() != self.nnz() {
            return Err(Error::ShapeMismatch { op: "SparseCsrParam::spmm", expected: vec![k, dense.shape().get(1).copied().unwrap_or(0)], got: dense.shape().to_vec() });
        }
        if dense.dtype() != self.values.dtype() {
            return Err(Error::DTypeMismatch { op: "SparseCsrParam::spmm", expected: self.values.dtype(), got: dense.dtype() });
        }
        let n = dense.shape()[1];
        let vals = self.values.detach().to_dtype(DType::F64)?.to_vec::<f64>()?;
        let x = dense.detach().contiguous()?.to_dtype(DType::F64)?.to_vec::<f64>()?;
        let mut y = vec![0.0f64; crate::limits::check_elements("SparseCsrParam::spmm", &[m, n])?];
        for r in 0..m {
            for e in self.row_ptr[r]..self.row_ptr[r + 1] {
                let (c, v) = (self.col_idx[e], vals[e]);
                for j in 0..n {
                    y[r * n + j] += v * x[c * n + j];
                }
            }
        }
        let layout = Layout::contiguous(vec![m, n]);
        let storage = Storage::new_cpu(CpuStorage::F64(y).to_dtype(&layout, dense.dtype())?).to_device(&layout, &dense.device())?;
        let (row_ptr, col_idx) = (self.row_ptr.clone(), self.col_idx.clone());
        let (dense_c, vals_dtype, dense_dtype) = (dense.detach(), self.values.dtype(), dense.dtype());
        Ok(Tensor::from_op(storage, layout, "spmm_param", vec![self.values.clone(), dense.clone()], move |g| {
            let gy = g.contiguous()?.to_dtype(DType::F64)?.to_vec::<f64>()?;
            let xv = dense_c.contiguous()?.to_dtype(DType::F64)?.to_vec::<f64>()?;
            // d values[e] = sum_j dY[r, j] * X[c, j]
            let mut gv = vec![0.0f64; col_idx.len()];
            // d X[c, :] += values[e] * dY[r, :]
            let mut gx = vec![0.0f64; k * n];
            for r in 0..m {
                for e in row_ptr[r]..row_ptr[r + 1] {
                    let c = col_idx[e];
                    let mut acc = 0.0;
                    for j in 0..n {
                        acc += gy[r * n + j] * xv[c * n + j];
                        gx[c * n + j] += vals[e] * gy[r * n + j];
                    }
                    gv[e] = acc;
                }
            }
            let dv = Tensor::from_slice(&gv, &[gv.len()])?.to_dtype(vals_dtype)?;
            let dx = Tensor::from_slice(&gx, &[k, n])?.to_dtype(dense_dtype)?;
            Ok(vec![Some(dv), Some(dx)])
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{what} element {i}: {g} vs {w} (tol {tol})"
            );
        }
    }

    #[test]
    fn coo_new_coalesces_validates_and_keeps_nan() {
        // duplicates sum, exact zeros drop, NaN is kept (never silenced)
        let coo = SparseCoo::new(
            vec![(0, 0), (0, 0), (1, 1), (0, 1)],
            vec![1.0, 2.0, 0.0, f64::NAN],
            [2, 2],
        )
        .unwrap();
        assert_eq!(coo.nnz(), 2);
        assert_eq!(coo.indices, vec![(0, 0), (0, 1)]);
        assert_eq!(coo.values[0], 3.0);
        assert!(coo.values[1].is_nan());
        // length mismatch and out-of-bounds indices error
        assert!(SparseCoo::new(vec![(0, 0)], vec![1.0, 2.0], [1, 1]).is_err());
        assert!(SparseCoo::new(vec![(0, 1)], vec![1.0], [1, 1]).is_err());
        assert!(SparseCoo::new(vec![(1, 0)], vec![1.0], [1, 1]).is_err());
    }

    #[test]
    fn coo_from_dense_thresholds_and_rejects_rank() {
        let t = Tensor::from_slice(&[0.5, 0.0, -2.0, f64::NAN], &[2, 2]).unwrap();
        let coo = SparseCoo::from_dense(&t, 0.25).unwrap();
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.indices, vec![(0, 0), (1, 0), (1, 1)]);
        let bad = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(SparseCoo::from_dense(&bad, 0.0).is_err());
    }

    #[test]
    fn coo_transpose_roundtrip_and_validates() {
        let coo = SparseCoo::new(vec![(0, 1), (1, 0)], vec![5.0, 7.0], [2, 3]).unwrap();
        let tr = coo.transpose().unwrap();
        assert_eq!(tr.shape, [3, 2]);
        assert_eq!(tr.indices, vec![(0, 1), (1, 0)]);
        assert_eq!(tr.values, vec![7.0, 5.0]);
        assert_eq!(tr.transpose().unwrap().indices, coo.indices);
        // hand-built garbage errors instead of transposing silently
        let bad = SparseCoo {
            indices: vec![(9, 0)],
            values: vec![1.0],
            shape: [2, 2],
        };
        assert!(bad.transpose().is_err());
        let bad_len = SparseCoo {
            indices: vec![(0, 0)],
            values: vec![],
            shape: [2, 2],
        };
        assert!(bad_len.transpose().is_err());
    }

    #[test]
    fn csr_from_coo_merges_duplicates_and_stays_sorted() {
        // hand-built COO with duplicates -> canonical CSR (summed once)
        let coo = SparseCoo {
            indices: vec![(1, 0), (0, 0), (0, 0)],
            values: vec![4.0, 1.0, 2.0],
            shape: [2, 2],
        };
        let csr = coo.to_csr().unwrap();
        assert_eq!(csr.row_ptr, vec![0, 1, 2]);
        assert_eq!(csr.col_idx, vec![0, 0]);
        assert_eq!(csr.values, vec![3.0, 4.0]);
        // dense agrees with the coalesced values (not last-wins)
        let d = csr
            .to_dense(DType::F64, &Device::Cpu)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&d, &[3.0, 0.0, 4.0, 0.0], 0.0, "csr dense");
    }

    #[test]
    fn csr_check_valid_rejects_broken_structs() {
        let bad_ptr = SparseCsr {
            row_ptr: vec![0, 1],
            col_idx: vec![0],
            values: vec![1.0],
            shape: [2, 2],
        };
        assert!(bad_ptr.check_valid().is_err());
        let non_mono = SparseCsr {
            row_ptr: vec![0, 2, 1],
            col_idx: vec![0],
            values: vec![1.0],
            shape: [2, 2],
        };
        assert!(non_mono.check_valid().is_err());
        let bad_nnz = SparseCsr {
            row_ptr: vec![0, 1, 1],
            col_idx: vec![0, 0],
            values: vec![1.0, 2.0],
            shape: [2, 2],
        };
        assert!(bad_nnz.check_valid().is_err());
        let bad_col = SparseCsr {
            row_ptr: vec![0, 1, 1],
            col_idx: vec![5],
            values: vec![1.0],
            shape: [2, 2],
        };
        assert!(bad_col.check_valid().is_err());
        // ...and every consumer enforces it
        assert!(bad_col.to_dense(DType::F64, &Device::Cpu).is_err());
        assert!(bad_col.transpose().is_err());
    }

    #[test]
    fn spmm_values_shape_error_and_transpose_backward() {
        // S = [[1,2],[0,3]], X = ones(2,2) -> [[3,3],[3,3]]
        let csr = SparseCoo::new(vec![(0, 0), (0, 1), (1, 1)], vec![1.0, 2.0, 3.0], [2, 2])
            .unwrap()
            .to_csr()
            .unwrap();
        let x = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2]).unwrap();
        let y = csr.spmm(&x).unwrap().to_vec::<f64>().unwrap();
        assert_close(&y, &[3.0, 3.0, 3.0, 3.0], 1e-12, "spmm");
        // K mismatch errors
        let bad = Tensor::from_slice(&[1.0, 1.0, 1.0], &[3, 1]).unwrap();
        assert!(csr.spmm(&bad).is_err());
        // d sum(S X)/dX = S^T ones = [[1,1],[5,5]]
        let xg = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        csr.spmm(&xg).unwrap().sum().unwrap().backward().unwrap();
        let grad = xg.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[1.0, 1.0, 5.0, 5.0], 1e-12, "spmm grad");
    }

    #[test]
    fn sparsity_and_empty_shapes_behave() {
        let coo = SparseCoo::new(vec![(0, 0)], vec![2.0], [2, 2]).unwrap();
        assert!((coo.to_csr().unwrap().sparsity() - 0.75).abs() <= 1e-12);
        let empty = SparseCoo::new(vec![], vec![], [0, 0]).unwrap();
        assert_eq!(empty.nnz(), 0);
        assert_eq!(
            empty.to_dense(DType::F64, &Device::Cpu).unwrap().shape(),
            &[0, 0]
        );
        assert_eq!(empty.to_csr().unwrap().row_ptr, vec![0]);
    }

    #[test]
    fn coo_csr_roundtrip_and_norm() {
        let coo = SparseCoo::new(
            vec![(0, 1), (1, 2), (2, 0)],
            vec![3.0, 4.0, 12.0],
            [3, 3],
        )
        .unwrap();
        assert_eq!(coo.norm(), 13.0); // sqrt(9 + 16 + 144) = 13
        let csr = coo.to_csr().unwrap();
        assert_eq!(csr.norm(), 13.0);
        let coo_back = csr.to_coo().unwrap();
        assert_eq!(coo_back.indices, coo.indices);
        assert_eq!(coo_back.values, coo.values);
        assert_eq!(coo_back.shape, coo.shape);
    }

    #[test]
    fn sparse_arithmetic_add_sub_mul_scalar() {
        let a = SparseCoo::new(vec![(0, 0), (0, 1)], vec![2.0, 3.0], [2, 2]).unwrap();
        let b = SparseCoo::new(vec![(0, 1), (1, 1)], vec![4.0, 5.0], [2, 2]).unwrap();
        
        let c = a.add(&b).unwrap();
        assert_eq!(c.indices, vec![(0, 0), (0, 1), (1, 1)]);
        assert_eq!(c.values, vec![2.0, 7.0, 5.0]);

        let diff = a.sub(&b).unwrap();
        assert_eq!(diff.indices, vec![(0, 0), (0, 1), (1, 1)]);
        assert_eq!(diff.values, vec![2.0, -1.0, -5.0]);

        let scaled = a.mul_scalar(2.5).unwrap();
        assert_eq!(scaled.values, vec![5.0, 7.5]);

        let zero_scaled = a.mul_scalar(0.0).unwrap();
        assert_eq!(zero_scaled.nnz(), 0);

        // CSR arithmetic
        let csr_a = a.to_csr().unwrap();
        let csr_b = b.to_csr().unwrap();
        let csr_c = csr_a.add(&csr_b).unwrap();
        assert_eq!(csr_c.row_ptr, vec![0, 2, 3]);
        assert_eq!(csr_c.col_idx, vec![0, 1, 1]);
        assert_eq!(csr_c.values, vec![2.0, 7.0, 5.0]);
    }

    #[test]
    fn spspmm_gustavson_algorithm() {
        // A = [[1, 2], [0, 3]]
        // B = [[4, 0], [5, 6]]
        // C = A @ B = [[14, 12], [15, 18]]
        let a = SparseCoo::new(vec![(0, 0), (0, 1), (1, 1)], vec![1.0, 2.0, 3.0], [2, 2])
            .unwrap()
            .to_csr()
            .unwrap();
        let b = SparseCoo::new(vec![(0, 0), (1, 0), (1, 1)], vec![4.0, 5.0, 6.0], [2, 2])
            .unwrap()
            .to_csr()
            .unwrap();

        let c = a.spspmm(&b).unwrap();
        assert_eq!(c.shape, [2, 2]);
        let d = c
            .to_dense(DType::F64, &Device::Cpu)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&d, &[14.0, 12.0, 15.0, 18.0], 1e-12, "spspmm");

        // COO spspmm
        let coo_a = a.to_coo().unwrap();
        let coo_b = b.to_coo().unwrap();
        let coo_c = coo_a.spspmm(&coo_b).unwrap();
        let coo_d = coo_c
            .to_dense(DType::F64, &Device::Cpu)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&coo_d, &[14.0, 12.0, 15.0, 18.0], 1e-12, "coo spspmm");
    }
}

#[cfg(test)]
mod param_tests {
    use super::*;

    #[test]
    fn sparse_values_and_dense_operand_get_the_dense_gradients() {
        crate::rng::manual_seed(2);
        // a random 5 x 6 pattern with ~40% non-zeros
        let dense = Tensor::randn(&[5, 6]).unwrap();
        let mask = Tensor::rand_uniform(&[5, 6], 0.0, 1.0).unwrap().to_vec::<f32>().unwrap();
        let masked: Vec<f32> = dense.to_vec::<f32>().unwrap().iter().zip(&mask).map(|(v, m)| if *m < 0.4 { *v } else { 0.0 }).collect();
        let d = Tensor::from_slice(&masked, &[5, 6]).unwrap();
        let csr = SparseCsr::from_dense(&d, 0.0).unwrap();
        assert!(csr.nnz() > 3 && csr.nnz() < 30);
        for dt in [DType::F32, DType::F64] {
            let s = SparseCsrParam::from_csr(&csr, dt, true).unwrap();
            let x = Tensor::randn(&[6, 4]).unwrap().to_dtype(dt).unwrap().detach().requires_grad_(true);
            let w = Tensor::randn(&[5, 4]).unwrap().to_dtype(dt).unwrap();
            let y = s.spmm(&x).unwrap();
            y.mul(&w).unwrap().sum().unwrap().backward().unwrap();
            // reference: the same product with the dense matrix
            let dd = d.to_dtype(dt).unwrap().detach().requires_grad_(true);
            let xr = x.detach().requires_grad_(true);
            let yr = dd.matmul(&xr).unwrap();
            yr.mul(&w).unwrap().sum().unwrap().backward().unwrap();
            let tol = if dt == DType::F32 { 1e-4 } else { 1e-10 };
            assert!(y.allclose(&yr, tol, tol).unwrap());
            assert!(x.grad().unwrap().allclose(&xr.grad().unwrap(), tol, tol).unwrap());
            // gradient of each stored value = the dense gradient at its (row, col)
            let gd = dd.grad().unwrap().to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
            let gv = s.values.grad().unwrap().to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap();
            let mut e = 0;
            for r in 0..5 {
                for k in csr.row_ptr[r]..csr.row_ptr[r + 1] {
                    assert!((gv[e] - gd[r * 6 + csr.col_idx[k]]).abs() < tol.max(1e-9) * 10.0, "value {e}: {} vs {}", gv[e], gd[r * 6 + csr.col_idx[k]]);
                    e += 1;
                }
            }
            assert_eq!(e, s.nnz());
        }
        // dtype mismatch is rejected
        let s = SparseCsrParam::from_csr(&csr, DType::F32, false).unwrap();
        assert!(s.spmm(&Tensor::randn(&[6, 2]).unwrap().to_dtype(DType::F64).unwrap()).is_err());
    }
}
