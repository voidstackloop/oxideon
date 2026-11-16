//! Extra [`SparseCoo`] operations: `eye`/`diag`/`from_diagonal`, element-wise `hadamard`, `map_values`, row/column
//! `sum_axis`, `spmv`, row-wise `softmax` (over stored entries), `select_rows`, `triu`/`tril`, and `kron`.

use crate::error::{Error, Result};
use crate::sparse::SparseCoo;
use std::collections::HashMap;

impl SparseCoo {
    /// `n x n` identity.
    pub fn eye(n: usize) -> Result<Self> {
        Self::new((0..n).map(|i| (i, i)).collect(), vec![1.0; n], [n, n])
    }

    /// Square matrix with `d` on the diagonal.
    pub fn from_diagonal(d: &[f64]) -> Result<Self> {
        Self::new((0..d.len()).map(|i| (i, i)).collect(), d.to_vec(), [d.len(), d.len()])
    }

    /// The main diagonal as a dense vector.
    pub fn diag(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.shape[0].min(self.shape[1])];
        for (&(r, c), &v) in self.indices.iter().zip(&self.values) {
            if r == c {
                out[r] = v;
            }
        }
        out
    }

    /// Applies `f` to every stored value (zeros are dropped; `f(0)` is assumed to be 0).
    pub fn map_values(&self, f: impl Fn(f64) -> f64) -> Result<Self> {
        Self::new(self.indices.clone(), self.values.iter().map(|&v| f(v)).collect(), self.shape)
    }

    /// Element-wise product (intersection of the two patterns).
    pub fn hadamard(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            return Err(Error::Msg(format!("hadamard: shape mismatch {:?} vs {:?}", self.shape, other.shape)));
        }
        let lookup: HashMap<_, _> = other.indices.iter().copied().zip(other.values.iter().copied()).collect();
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        for (&p, &v) in self.indices.iter().zip(&self.values) {
            if let Some(&w) = lookup.get(&p) {
                idx.push(p);
                val.push(v * w);
            }
        }
        Self::new(idx, val, self.shape)
    }

    /// Sums over rows (`axis = 0`, result has one entry per column) or columns (`axis = 1`, one entry per row).
    pub fn sum_axis(&self, axis: usize) -> Result<Vec<f64>> {
        if axis > 1 {
            return Err(Error::Msg("sum_axis: axis must be 0 or 1".into()));
        }
        let mut out = vec![0.0; self.shape[1 - axis]];
        for (&(r, c), &v) in self.indices.iter().zip(&self.values) {
            out[if axis == 0 { c } else { r }] += v;
        }
        Ok(out)
    }

    /// Matrix-vector product.
    pub fn spmv(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.shape[1] {
            return Err(Error::Msg(format!("spmv: vector length {} != {} columns", x.len(), self.shape[1])));
        }
        let mut out = vec![0.0; self.shape[0]];
        for (&(r, c), &v) in self.indices.iter().zip(&self.values) {
            out[r] += v * x[c];
        }
        Ok(out)
    }

    /// Softmax over the stored entries of each row (absent entries are treated as `-inf`, like masked attention).
    pub fn softmax_rows(&self) -> Result<Self> {
        let mut max = vec![f64::NEG_INFINITY; self.shape[0]];
        for (&(r, _), &v) in self.indices.iter().zip(&self.values) {
            max[r] = max[r].max(v);
        }
        let e: Vec<f64> = self.indices.iter().zip(&self.values).map(|(&(r, _), &v)| (v - max[r]).exp()).collect();
        let mut sum = vec![0.0; self.shape[0]];
        for (&(r, _), &x) in self.indices.iter().zip(&e) {
            sum[r] += x;
        }
        Self::new(self.indices.clone(), self.indices.iter().zip(&e).map(|(&(r, _), &x)| x / sum[r]).collect(), self.shape)
    }

    /// The listed rows (in order, repeats allowed) as a new matrix.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self> {
        if let Some(&bad) = rows.iter().find(|&&r| r >= self.shape[0]) {
            return Err(Error::Msg(format!("select_rows: row {bad} out of bounds for {} rows", self.shape[0])));
        }
        let mut by_row: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        for (&(r, c), &v) in self.indices.iter().zip(&self.values) {
            by_row.entry(r).or_default().push((c, v));
        }
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        for (new_r, r) in rows.iter().enumerate() {
            for &(c, v) in by_row.get(r).into_iter().flatten() {
                idx.push((new_r, c));
                val.push(v);
            }
        }
        Self::new(idx, val, [rows.len(), self.shape[1]])
    }

    /// Upper (`k >= 0` above the diagonal) or lower triangle: keeps entries with `col - row >= k` (`upper`) or
    /// `col - row <= k` (lower).
    pub fn triangle(&self, upper: bool, k: i64) -> Result<Self> {
        let keep = |&(r, c): &(usize, usize)| {
            let d = c as i64 - r as i64;
            if upper { d >= k } else { d <= k }
        };
        let (idx, val): (Vec<_>, Vec<_>) = self.indices.iter().zip(&self.values).filter(|(p, _)| keep(p)).map(|(p, v)| (*p, *v)).unzip();
        Self::new(idx, val, self.shape)
    }

    /// Kronecker product.
    pub fn kron(&self, other: &Self) -> Result<Self> {
        let [br, bc] = other.shape;
        let mut idx = Vec::with_capacity(self.indices.len() * other.indices.len());
        let mut val = Vec::with_capacity(idx.capacity());
        for (&(r1, c1), &v1) in self.indices.iter().zip(&self.values) {
            for (&(r2, c2), &v2) in other.indices.iter().zip(&other.values) {
                idx.push((r1 * br + r2, c1 * bc + c2));
                val.push(v1 * v2);
            }
        }
        Self::new(idx, val, [self.shape[0] * br, self.shape[1] * bc])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device;
    use crate::dtype::DType;

    fn dense(s: &SparseCoo) -> Vec<f32> {
        s.to_dense(DType::F32, &Device::Cpu).unwrap().to_vec::<f32>().unwrap()
    }

    fn m() -> SparseCoo {
        // [[1, 0, 2], [0, 3, 0]]
        SparseCoo::new(vec![(0, 0), (0, 2), (1, 1)], vec![1.0, 2.0, 3.0], [2, 3]).unwrap()
    }

    #[test]
    fn construction_and_reductions() {
        assert_eq!(dense(&SparseCoo::eye(2).unwrap()), vec![1.0, 0.0, 0.0, 1.0]);
        let d = SparseCoo::from_diagonal(&[2.0, 5.0]).unwrap();
        assert_eq!(d.diag(), vec![2.0, 5.0]);
        assert_eq!(m().diag(), vec![1.0, 3.0]);
        assert_eq!(m().sum_axis(0).unwrap(), vec![1.0, 3.0, 2.0]);
        assert_eq!(m().sum_axis(1).unwrap(), vec![3.0, 3.0]);
        assert_eq!(m().spmv(&[1.0, 1.0, 1.0]).unwrap(), vec![3.0, 3.0]);
        assert!(m().spmv(&[1.0]).is_err());
    }

    #[test]
    fn elementwise_and_structure() {
        let a = m();
        let b = SparseCoo::new(vec![(0, 2), (1, 1), (1, 2)], vec![10.0, 2.0, 7.0], [2, 3]).unwrap();
        assert_eq!(dense(&a.hadamard(&b).unwrap()), vec![0.0, 0.0, 20.0, 0.0, 6.0, 0.0]);
        assert_eq!(dense(&a.map_values(|v| v * v).unwrap()), vec![1.0, 0.0, 4.0, 0.0, 9.0, 0.0]);
        assert_eq!(dense(&a.select_rows(&[1, 0, 1]).unwrap()), vec![0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        assert!(a.select_rows(&[2]).is_err());
        assert_eq!(dense(&a.triangle(true, 1).unwrap()), vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
        assert_eq!(dense(&a.triangle(false, 0).unwrap()), vec![1.0, 0.0, 0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn softmax_and_kron() {
        let s = dense(&m().softmax_rows().unwrap());
        assert!((s[0] + s[2] - 1.0).abs() < 1e-6 && (s[4] - 1.0).abs() < 1e-6);
        assert!(s[2] > s[0]);
        let k = SparseCoo::eye(2).unwrap().kron(&m()).unwrap();
        assert_eq!(k.shape, [4, 6]);
        let want = SparseCoo::eye(2).unwrap().to_dense(DType::F32, &Device::Cpu).unwrap();
        let _ = want;
        let kd = dense(&k);
        assert_eq!(&kd[..6], &[1.0, 0.0, 2.0, 0.0, 0.0, 0.0]); // top-left block row 0 = m row 0
        assert_eq!(&kd[18..24], &[0.0, 0.0, 0.0, 0.0, 3.0, 0.0]); // bottom-right block row 1
    }
}
