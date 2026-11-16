//! Small tensor conveniences from PyTorch that were missing: global reductions (`max_all`, `min_all`, `std_all`,
//! `argmax_all`, `argmin_all`), random constructors (`randint`, `randperm`, `rand_like`, `randn_like`), `tri`,
//! `empty_like`, `new_zeros` / `new_ones` / `new_full`, `view_as` / `expand_as` / `reshape_as` / `type_as`,
//! multi-axis `squeeze_axes` / `unsqueeze_axes`, and the matrix transpose `mt`.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::rng;
use crate::tensor::Tensor;

impl Tensor {
    /// Largest element of the whole tensor as a scalar tensor (`torch.max(x)`).
    pub fn max_all(&self) -> Result<Tensor> {
        self.amax(&[], false)
    }

    /// Smallest element of the whole tensor as a scalar tensor (`torch.min(x)`).
    pub fn min_all(&self) -> Result<Tensor> {
        self.amin(&[], false)
    }

    /// Standard deviation of all elements (`torch.std(x)`).
    pub fn std_all(&self, unbiased: bool) -> Result<Tensor> {
        let n = self.numel();
        if n == 0 || (unbiased && n < 2) {
            return Err(Error::Msg("std_all: needs at least 2 elements (1 when biased)".into()));
        }
        let mean = self.mean()?;
        let ss = self.sub(&mean)?.square()?.sum()?;
        ss.div_scalar((if unbiased { n - 1 } else { n }) as f64)?.sqrt()
    }

    /// Flat index of the first maximum (`torch.argmax(x)`).
    pub fn argmax_all(&self) -> Result<usize> {
        let flat = self.reshape(&[self.numel()])?;
        Ok(flat.argmax_axis(0, false)?.item::<i64>()? as usize)
    }

    /// Flat index of the first minimum (`torch.argmin(x)`).
    pub fn argmin_all(&self) -> Result<usize> {
        let flat = self.reshape(&[self.numel()])?;
        Ok(flat.argmin_axis(0, false)?.item::<i64>()? as usize)
    }

    /// Integers uniformly drawn from `[low, high)` (`torch.randint`), dtype `I64`.
    pub fn randint(low: i64, high: i64, shape: &[usize]) -> Result<Tensor> {
        if low >= high {
            return Err(Error::Msg(format!("randint: need low < high, got [{low}, {high})")));
        }
        let n: usize = shape.iter().product();
        let span = (high - low) as usize;
        let v: Vec<i64> = rng::with_rng(|r| (0..n).map(|_| low + r.below(span).unwrap_or(0) as i64).collect());
        Tensor::from_slice(&v, shape)
    }

    /// A random permutation of `0..n` (`torch.randperm`), dtype `I64`.
    pub fn randperm(n: usize) -> Result<Tensor> {
        let mut v: Vec<i64> = (0..n as i64).collect();
        rng::with_rng(|r| r.shuffle(&mut v));
        Tensor::from_slice(&v, &[n])
    }

    /// Uniform `[0, 1)` samples shaped and typed like `self` (`torch.rand_like`).
    pub fn rand_like(&self) -> Result<Tensor> {
        Tensor::rand_uniform(self.shape(), 0.0, 1.0)?.to_dtype(self.dtype())?.to_device(&self.device())
    }

    /// Standard normal samples shaped and typed like `self` (`torch.randn_like`).
    pub fn randn_like(&self) -> Result<Tensor> {
        Tensor::randn(self.shape())?.to_dtype(self.dtype())?.to_device(&self.device())
    }

    /// A zero-initialized tensor shaped like `self` (`torch.empty_like`; the values are defined, unlike PyTorch).
    pub fn empty_like(&self) -> Result<Tensor> {
        self.zeros_like()
    }

    /// `rows x cols` matrix with ones on and below the `diagonal`-th diagonal (`torch.tril` of ones).
    pub fn tri(rows: usize, cols: usize, diagonal: isize, dtype: DType) -> Result<Tensor> {
        let v: Vec<f64> = (0..rows * cols)
            .map(|ix| if (ix % cols) as isize - (ix / cols) as isize <= diagonal { 1.0 } else { 0.0 })
            .collect();
        Tensor::from_slice(&v, &[rows, cols])?.to_dtype(dtype)
    }

    /// Zeros with the given shape and this tensor's dtype / device (`Tensor.new_zeros`).
    pub fn new_zeros(&self, shape: &[usize]) -> Result<Tensor> {
        Tensor::zeros(shape, self.dtype(), &self.device())
    }

    /// Ones with the given shape and this tensor's dtype / device (`Tensor.new_ones`).
    pub fn new_ones(&self, shape: &[usize]) -> Result<Tensor> {
        Tensor::ones(shape, self.dtype(), &self.device())
    }

    /// A constant with the given shape and this tensor's dtype / device (`Tensor.new_full`).
    pub fn new_full(&self, shape: &[usize], value: f64) -> Result<Tensor> {
        Tensor::full(shape, value, self.dtype(), &self.device())
    }

    /// Reshapes to the shape of `other` (`Tensor.view_as`).
    pub fn view_as(&self, other: &Tensor) -> Result<Tensor> {
        self.reshape(other.shape())
    }

    /// Reshapes to the shape of `other` (`Tensor.reshape_as`).
    pub fn reshape_as(&self, other: &Tensor) -> Result<Tensor> {
        self.reshape(other.shape())
    }

    /// Broadcasts to the shape of `other` (`Tensor.expand_as`).
    pub fn expand_as(&self, other: &Tensor) -> Result<Tensor> {
        self.broadcast_to(other.shape())
    }

    /// Casts to the dtype of `other` (`Tensor.type_as`).
    pub fn type_as(&self, other: &Tensor) -> Result<Tensor> {
        self.to_dtype(other.dtype())
    }

    /// Removes the listed size-1 axes (`torch.squeeze` with a tuple of dims).
    pub fn squeeze_axes(&self, axes: &[isize]) -> Result<Tensor> {
        let nd = self.ndim();
        let mut drop: Vec<usize> = axes
            .iter()
            .map(|&a| crate::layout::normalize_axis("squeeze_axes", a, nd))
            .collect::<Result<_>>()?;
        drop.sort_unstable();
        drop.dedup();
        let shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, s)| !(drop.contains(i) && **s == 1))
            .map(|(_, &s)| s)
            .collect();
        self.reshape(&shape)
    }

    /// Inserts size-1 axes at the listed positions of the **output** (`torch.unsqueeze` for several dims).
    pub fn unsqueeze_axes(&self, axes: &[isize]) -> Result<Tensor> {
        let out_nd = self.ndim() + axes.len();
        let mut pos: Vec<usize> = axes
            .iter()
            .map(|&a| crate::layout::normalize_axis("unsqueeze_axes", a, out_nd))
            .collect::<Result<_>>()?;
        pos.sort_unstable();
        if pos.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::Msg("unsqueeze_axes: repeated axis".into()));
        }
        let mut shape = Vec::with_capacity(out_nd);
        let mut src = self.shape().iter();
        for i in 0..out_nd {
            if pos.contains(&i) {
                shape.push(1);
            } else {
                shape.push(*src.next().expect("axis count matches"));
            }
        }
        self.reshape(&shape)
    }

    /// Transposes the last two dimensions (`Tensor.mT`).
    pub fn mt(&self) -> Result<Tensor> {
        self.transpose(-2, -1)
    }
}

/// Adaptive average pooling of `[N, C, spatial...]` to `output_size` for any number of spatial dimensions
/// (`torch.nn.functional.adaptive_avg_pool1d/2d/3d`); entries of `None` keep that dimension's size.
pub fn adaptive_avg_pool(x: &Tensor, output_size: &[Option<usize>]) -> Result<Tensor> {
    if x.ndim() < 3 || output_size.len() != x.ndim() - 2 {
        return Err(Error::Msg("adaptive_avg_pool: output_size needs one entry per spatial dimension".into()));
    }
    let size: Vec<usize> = output_size.iter().zip(&x.shape()[2..]).map(|(o, &i)| o.unwrap_or(i)).collect();
    crate::nn::functional_ext::interpolate(x, Some(&size), None, crate::nn::functional_ext::InterpMode::Area, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn reductions_and_creation() {
        let x = f(&[3.0, 9.0, 1.0, 9.0], &[2, 2]);
        assert_eq!(vals(&x.max_all().unwrap()), vec![9.0]);
        assert_eq!(vals(&x.min_all().unwrap()), vec![1.0]);
        assert_eq!(x.argmax_all().unwrap(), 1);
        assert_eq!(x.argmin_all().unwrap(), 2);
        let s = vals(&x.std_all(true).unwrap())[0];
        assert!((s - (((3.0f64 - 5.5).powi(2) + 2.0 * (9.0f64 - 5.5).powi(2) + (1.0f64 - 5.5).powi(2)) / 3.0).sqrt()).abs() < 1e-9);
        assert!(f(&[1.0], &[1]).std_all(true).is_err());
        rng::manual_seed(1);
        let r = Tensor::randint(-3, 4, &[500]).unwrap().to_vec::<i64>().unwrap();
        assert!(r.iter().all(|v| (-3..4).contains(v)) && (-3..4).all(|k| r.contains(&k)));
        let mut p = Tensor::randperm(10).unwrap().to_vec::<i64>().unwrap();
        p.sort_unstable();
        assert_eq!(p, (0..10).collect::<Vec<_>>());
        assert!(Tensor::randint(3, 3, &[1]).is_err());
        assert_eq!(x.rand_like().unwrap().shape(), [2, 2]);
        assert!(vals(&x.rand_like().unwrap()).iter().all(|v| (0.0..1.0).contains(v)));
        assert_eq!(x.randn_like().unwrap().dtype(), x.dtype());
        assert_eq!(vals(&Tensor::tri(3, 3, 0, DType::F64).unwrap()), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(vals(&Tensor::tri(2, 3, 1, DType::F64).unwrap()), vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(vals(&x.new_full(&[2], 7.0).unwrap()), vec![7.0, 7.0]);
        assert_eq!(x.new_zeros(&[1, 3]).unwrap().shape(), [1, 3]);
        assert_eq!(vals(&x.new_ones(&[2]).unwrap()), vec![1.0, 1.0]);
    }

    #[test]
    fn shape_helpers() {
        let a = f(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = f(&[0.0; 4], &[2, 2]);
        assert_eq!(a.view_as(&b).unwrap().shape(), [2, 2]);
        assert_eq!(f(&[1.0, 2.0], &[1, 2]).expand_as(&f(&[0.0; 6], &[3, 2])).unwrap().shape(), [3, 2]);
        assert_eq!(a.type_as(&Tensor::zeros(&[1], DType::F32, &crate::backend::Device::Cpu).unwrap()).unwrap().dtype(), DType::F32);
        let z = f(&[1.0, 2.0], &[1, 2, 1]);
        assert_eq!(z.squeeze_axes(&[0, 2]).unwrap().shape(), [2]);
        assert_eq!(z.squeeze_axes(&[1]).unwrap().shape(), [1, 2, 1]); // size-2 axis is kept
        assert_eq!(f(&[1.0, 2.0], &[2]).unsqueeze_axes(&[0, -1]).unwrap().shape(), [1, 2, 1]);
        assert!(f(&[1.0], &[1]).unsqueeze_axes(&[0, 0]).is_err());
        assert_eq!(f(&[0.0; 6], &[1, 2, 3]).mt().unwrap().shape(), [1, 3, 2]);
    }

    #[test]
    fn adaptive_pooling_any_rank() {
        let x = f(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6]);
        assert_eq!(vals(&adaptive_avg_pool(&x, &[Some(3)]).unwrap()), vec![1.5, 3.5, 5.5]);
        assert_eq!(vals(&adaptive_avg_pool(&x, &[Some(4)]).unwrap()).len(), 4);
        // uneven windows: 5 -> 2 uses [0, 3) and [2, 5)
        let y = f(&[0.0, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5]);
        assert_eq!(vals(&adaptive_avg_pool(&y, &[Some(2)]).unwrap()), vec![1.0, 3.0]);
        // 3-D volume to 1x1x1 is the global mean
        let v = Tensor::from_slice(&(0..8).map(f64::from).collect::<Vec<_>>(), &[1, 1, 2, 2, 2]).unwrap();
        assert_eq!(vals(&adaptive_avg_pool(&v, &[Some(1), Some(1), Some(1)]).unwrap()), vec![3.5]);
        assert!(adaptive_avg_pool(&v, &[Some(1)]).is_err());
    }
}
