//! In-place tensor updates in the style of `torch.Tensor.add_` / `mul_` / `zero_` / `fill_` / `copy_` /
//! `clamp_` / `masked_fill_` / `uniform_` / `normal_`.
//!
//! The update replaces the tensor's values, so every other handle to the same tensor (clones, entries of
//! `Module::parameters`) sees it. Like PyTorch, an in-place update is refused when it would corrupt autograd:
//! a leaf that requires gradients may only be modified under [`no_grad`](crate::tensor::no_grad), and a tensor that
//! was produced by an operation (and so may be needed by a backward pass) cannot be modified while gradients are
//! enabled. Views (non-contiguous tensors) are refused as well, because they would not update their base.

use crate::error::{Error, Result};
use crate::tensor::{Tensor, is_grad_enabled};

impl Tensor {
    fn check_inplace(&self, op: &str) -> Result<()> {
        if !self.is_contiguous() || self.storage_offset() != 0 {
            return Err(Error::Msg(format!("{op}: in-place updates need a contiguous, un-offset tensor (not a view)")));
        }
        if is_grad_enabled() && (self.requires_grad() || !self.is_leaf()) {
            return Err(Error::Msg(format!(
                "{op}: a tensor that requires grad (or was produced by an operation) cannot be updated in place while gradients are enabled; wrap the update in no_grad"
            )));
        }
        Ok(())
    }

    /// Replaces the values with `value`, which must already have this tensor's shape.
    fn write_inplace(&self, op: &str, value: &Tensor) -> Result<&Self> {
        self.check_inplace(op)?;
        self.assign(&value.detach())?;
        Ok(self)
    }

    /// `self += other` (broadcasting `other` to `self`'s shape).
    pub fn add_(&self, other: &Tensor) -> Result<&Self> {
        self.write_inplace("add_", &self.detach().add(&other.to_dtype(self.dtype())?)?)
    }

    /// `self -= other`.
    pub fn sub_(&self, other: &Tensor) -> Result<&Self> {
        self.write_inplace("sub_", &self.detach().sub(&other.to_dtype(self.dtype())?)?)
    }

    /// `self *= other`.
    pub fn mul_(&self, other: &Tensor) -> Result<&Self> {
        self.write_inplace("mul_", &self.detach().mul(&other.to_dtype(self.dtype())?)?)
    }

    /// `self /= other`.
    pub fn div_(&self, other: &Tensor) -> Result<&Self> {
        self.write_inplace("div_", &self.detach().div(&other.to_dtype(self.dtype())?)?)
    }

    /// `self += s`.
    pub fn add_scalar_(&self, s: f64) -> Result<&Self> {
        self.write_inplace("add_scalar_", &self.detach().add_scalar(s)?)
    }

    /// `self *= s`.
    pub fn mul_scalar_(&self, s: f64) -> Result<&Self> {
        self.write_inplace("mul_scalar_", &self.detach().mul_scalar(s)?)
    }

    /// `self = -self`.
    pub fn neg_(&self) -> Result<&Self> {
        self.write_inplace("neg_", &self.detach().neg()?)
    }

    /// Sets every element to `value`.
    pub fn fill_(&self, value: f64) -> Result<&Self> {
        self.write_inplace("fill_", &self.detach().full_like(value)?)
    }

    /// Sets every element to zero.
    pub fn zero_(&self) -> Result<&Self> {
        self.fill_(0.0)
    }

    /// Copies `src` (broadcast to this shape, cast to this dtype) into the tensor.
    pub fn copy_(&self, src: &Tensor) -> Result<&Self> {
        let v = src.to_dtype(self.dtype())?.broadcast_to(self.shape())?.contiguous()?;
        self.write_inplace("copy_", &v)
    }

    /// Clamps the values into `[min, max]`.
    pub fn clamp_(&self, min: f64, max: f64) -> Result<&Self> {
        self.write_inplace("clamp_", &self.detach().clamp(min, max)?)
    }

    /// Replaces the elements where `mask` is true with `value`.
    pub fn masked_fill_(&self, mask: &Tensor, value: f64) -> Result<&Self> {
        self.write_inplace("masked_fill_", &self.detach().masked_fill(mask, value)?)
    }

    /// Fills with samples of `U(low, high)`.
    pub fn uniform_(&self, low: f64, high: f64) -> Result<&Self> {
        let u = Tensor::rand_uniform(self.shape(), low as f32, high as f32)?.to_dtype(self.dtype())?.to_device(&self.device())?;
        self.write_inplace("uniform_", &u)
    }

    /// Fills with samples of `N(mean, std^2)`.
    pub fn normal_(&self, mean: f64, std: f64) -> Result<&Self> {
        let n = Tensor::randn(self.shape())?.mul_scalar(std)?.add_scalar(mean)?.to_dtype(self.dtype())?.to_device(&self.device())?;
        self.write_inplace("normal_", &n)
    }

    /// `self += alpha * other` (`torch.Tensor.add_(other, alpha=alpha)`), the building block of hand-written
    /// optimizers.
    pub fn add_scaled_(&self, other: &Tensor, alpha: f64) -> Result<&Self> {
        self.write_inplace("add_scaled_", &self.detach().add(&other.detach().mul_scalar(alpha)?.to_dtype(self.dtype())?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::no_grad;

    fn t(v: &[f64]) -> Tensor {
        Tensor::from_slice(v, &[v.len()]).unwrap()
    }
    fn vals(x: &Tensor) -> Vec<f64> {
        x.to_vec::<f64>().unwrap()
    }

    #[test]
    fn updates_are_visible_through_every_handle() {
        let a = t(&[1.0, 2.0, 3.0]);
        let alias = a.clone();
        a.add_(&t(&[10.0, 10.0, 10.0])).unwrap().mul_scalar_(2.0).unwrap();
        assert_eq!(vals(&alias), vec![22.0, 24.0, 26.0]);
        a.sub_(&t(&[2.0, 4.0, 6.0])).unwrap().div_(&t(&[4.0, 4.0, 4.0])).unwrap();
        assert_eq!(vals(&alias), vec![5.0, 5.0, 5.0]);
        a.neg_().unwrap().add_scalar_(1.0).unwrap().clamp_(-3.0, 0.0).unwrap();
        assert_eq!(vals(&a), vec![-3.0, -3.0, -3.0]);
        a.fill_(7.0).unwrap();
        assert_eq!(vals(&a), vec![7.0; 3]);
        a.zero_().unwrap();
        assert_eq!(vals(&a), vec![0.0; 3]);
        a.copy_(&t(&[1.0])).unwrap(); // broadcast
        assert_eq!(vals(&a), vec![1.0; 3]);
        a.add_scaled_(&t(&[1.0, 2.0, 3.0]), -0.5).unwrap();
        assert_eq!(vals(&a), vec![0.5, 0.0, -0.5]);
        let mask = Tensor::from_slice(&[true, false, true], &[3]).unwrap();
        a.masked_fill_(&mask, 9.0).unwrap();
        assert_eq!(vals(&a), vec![9.0, 0.0, 9.0]);
        a.uniform_(2.0, 3.0).unwrap();
        assert!(vals(&a).iter().all(|v| (2.0..3.0).contains(v)));
        a.normal_(100.0, 0.001).unwrap();
        assert!(vals(&a).iter().all(|v| (v - 100.0).abs() < 0.1));
    }

    #[test]
    fn autograd_safety_rules() {
        let w = t(&[1.0, 2.0]).requires_grad_(true);
        assert!(w.add_scalar_(1.0).is_err(), "a leaf that requires grad must be updated under no_grad");
        no_grad(|| w.add_scalar_(1.0)).unwrap();
        assert_eq!(vals(&w), vec![2.0, 3.0]);
        let y = w.mul_scalar(2.0).unwrap();
        assert!(y.zero_().is_err(), "an op result may be needed by backward");
        // views are refused
        let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(m.t().unwrap().fill_(0.0).is_err());
        // a plain tensor that needs no grad can be updated freely
        let plain = t(&[1.0]);
        plain.mul_scalar_(3.0).unwrap();
        assert_eq!(vals(&plain), vec![3.0]);
        // gradients still flow to the original after an update under no_grad
        let p = t(&[1.0]).requires_grad_(true);
        no_grad(|| p.fill_(4.0)).unwrap();
        p.square().unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(vals(&p.grad().unwrap()), vec![8.0]);
    }
}
