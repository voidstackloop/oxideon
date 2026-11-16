//! `trapezoid`, `isin` and `histogram` with explicit bin edges (`torch.trapezoid` / `isin` / `histogram`).

use crate::error::{Error, Result};
use crate::layout::normalize_axis;
use crate::tensor::Tensor;

impl Tensor {
    /// Trapezoid-rule integral along `axis` with uniform spacing `dx` (differentiable).
    pub fn trapezoid(&self, axis: isize, dx: f64) -> Result<Tensor> {
        let ax = normalize_axis("trapezoid", axis, self.ndim())?;
        let n = self.shape()[ax];
        if n < 2 {
            return self.sum_axis(axis, false)?.mul_scalar(0.0);
        }
        let left = self.narrow(axis, 0, n - 1)?;
        let right = self.narrow(axis, 1, n - 1)?;
        left.add(&right)?.sum_axis(axis, false)?.mul_scalar(0.5 * dx)
    }

    /// Trapezoid integral against sample points `x` (1-D, length of `axis`).
    pub fn trapezoid_x(&self, axis: isize, x: &Tensor) -> Result<Tensor> {
        let ax = normalize_axis("trapezoid", axis, self.ndim())?;
        let n = self.shape()[ax];
        if n < 2 {
            return self.sum_axis(axis, false)?.mul_scalar(0.0);
        }
        self.cumulative_trapezoid_x(axis, x)?.narrow(axis, n - 2, 1)?.squeeze_axis(axis)
    }

    /// Boolean mask: whether each element of `self` occurs in `test`.
    pub fn isin(&self, test: &Tensor, invert: bool) -> Result<Tensor> {
        let mut set: Vec<f64> = test.to_vec::<f64>()?;
        set.sort_by(|a, b| a.total_cmp(b));
        let out: Vec<u8> = self
            .to_vec::<f64>()?
            .iter()
            .map(|v| (set.binary_search_by(|p| p.total_cmp(v)).is_ok() != invert) as u8)
            .collect();
        Tensor::from_slice_on(&out, self.shape(), &self.device())?.to_dtype(crate::dtype::DType::Bool)
    }

    /// Histogram with explicit, increasing bin `edges` (`bins + 1` values). Returns `(counts, edges)`; the last bin is
    /// closed on the right and values outside the range are ignored. `weights` (same shape as `self`) are summed
    /// instead of counting.
    pub fn histogram(&self, edges: &Tensor, weights: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let e = edges.to_vec::<f64>()?;
        if e.len() < 2 || e.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::Msg("histogram: edges must be strictly increasing with at least 2 values".into()));
        }
        let w = weights.map(|w| w.to_vec::<f64>()).transpose()?;
        let vals = self.to_vec::<f64>()?;
        if w.as_ref().is_some_and(|w| w.len() != vals.len()) {
            return Err(Error::Msg("histogram: weights must match the input size".into()));
        }
        let bins = e.len() - 1;
        let mut out = vec![0.0f64; bins];
        for (i, &v) in vals.iter().enumerate() {
            if v.is_nan() || v < e[0] || v > e[bins] {
                continue;
            }
            // number of edges <= v, minus one; the right edge falls in the last bin
            let b = (e.partition_point(|&x| x <= v) - 1).min(bins - 1);
            out[b] += w.as_ref().map_or(1.0, |w| w[i]);
        }
        let counts = Tensor::from_slice_on(&out, &[bins], &self.device())?.to_dtype(if self.dtype() == crate::dtype::DType::F64 { crate::dtype::DType::F64 } else { crate::dtype::DType::F32 })?;
        Ok((counts, edges.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f32]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }

    #[test]
    fn trapezoid_matches_hand_values() {
        let y = t(&[1.0, 2.0, 4.0]);
        assert_eq!(y.trapezoid(0, 1.0).unwrap().to_vec::<f32>().unwrap(), vec![4.5]);
        assert_eq!(y.trapezoid(0, 2.0).unwrap().to_vec::<f32>().unwrap(), vec![9.0]);
        let x = t(&[0.0, 1.0, 3.0]);
        assert_eq!(y.trapezoid_x(0, &x).unwrap().to_vec::<f32>().unwrap(), vec![1.5 + 6.0]);
        let p = y.requires_grad_(true);
        p.trapezoid(0, 1.0).unwrap().backward().unwrap();
        assert_eq!(p.grad().unwrap().to_vec::<f32>().unwrap(), vec![0.5, 1.0, 0.5]);
    }

    #[test]
    fn isin_and_histogram() {
        let a = t(&[1.0, 2.0, 3.0, 4.0]);
        let set = t(&[2.0, 4.0, 9.0]);
        assert_eq!(a.isin(&set, false).unwrap().to_vec::<u8>().unwrap(), vec![0, 1, 0, 1]);
        assert_eq!(a.isin(&set, true).unwrap().to_vec::<u8>().unwrap(), vec![1, 0, 1, 0]);
        let x = t(&[0.0, 1.0, 1.5, 2.0, 3.0, 5.0]);
        let (c, _) = x.histogram(&t(&[0.0, 1.0, 2.0, 3.0]), None).unwrap();
        // [0,1): 0 ; [1,2): 1,1.5 ; [2,3]: 2,3 ; 5 ignored
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 2.0]);
        let (w, _) = x.histogram(&t(&[0.0, 2.0, 4.0]), Some(&t(&[1.0, 1.0, 1.0, 1.0, 10.0, 1.0]))).unwrap();
        assert_eq!(w.to_vec::<f32>().unwrap(), vec![3.0, 11.0]);
        assert!(x.histogram(&t(&[1.0, 1.0]), None).is_err());
    }
}
