//! Tensors with structure beyond a rectangular grid, all built from differentiable primitives:
//! - [`RaggedTensor`]: a batch of sequences of different lengths (`torch.nested`), stored packed with offsets;
//! - [`MaskedTensor`]: data plus a validity mask whose reductions ignore masked-out elements (`torch.masked`);
//! - [`NamedTensor`]: dimension names, with name-based alignment, broadcasting and reductions (`torch` named tensors).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// ------------------------------------------------------------------------------------------------ ragged

/// Sequences `[L_i, ...]` of different lengths `L_i` (the trailing dimensions are shared), stored concatenated along the first
/// axis with the start offset of every sequence.
#[derive(Clone)]
pub struct RaggedTensor {
    values: Tensor,
    offsets: Vec<usize>,
}

impl RaggedTensor {
    /// Packs a list of tensors that agree on everything but their first dimension.
    pub fn from_list(seqs: &[Tensor]) -> Result<Self> {
        let first = seqs.first().ok_or_else(|| Error::Msg("RaggedTensor: need at least one sequence".into()))?;
        if first.ndim() == 0 {
            return Err(Error::Msg("RaggedTensor: sequences need at least one dimension".into()));
        }
        let mut offsets = vec![0];
        for s in seqs {
            if s.ndim() != first.ndim() || s.shape()[1..] != first.shape()[1..] || s.dtype() != first.dtype() {
                return Err(Error::Msg(format!("RaggedTensor: sequence of shape {:?} does not match {:?}", s.shape(), first.shape())));
            }
            offsets.push(offsets.last().unwrap() + s.shape()[0]);
        }
        Ok(Self { values: Tensor::cat(seqs, 0)?, offsets })
    }

    /// Packs a padded batch `[B, L_max, ...]` keeping the first `lengths[b]` rows of every row of the batch.
    pub fn from_padded(padded: &Tensor, lengths: &[usize]) -> Result<Self> {
        if padded.ndim() < 2 || padded.shape()[0] != lengths.len() || lengths.iter().any(|&l| l > padded.shape()[1]) {
            return Err(Error::Msg(format!("RaggedTensor::from_padded: lengths {lengths:?} do not fit {:?}", padded.shape())));
        }
        let seqs = lengths.iter().enumerate().map(|(b, &l)| padded.select(0, b as isize)?.narrow(0, 0, l)).collect::<Result<Vec<_>>>()?;
        Self::from_list(&seqs)
    }

    pub fn batch_size(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn lengths(&self) -> Vec<usize> {
        self.offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }
    /// The packed values `[sum L_i, ...]`.
    pub fn values(&self) -> &Tensor {
        &self.values
    }
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// The sequences as separate tensors (views of the packed values).
    pub fn unbind(&self) -> Result<Vec<Tensor>> {
        self.offsets.windows(2).map(|w| self.values.narrow(0, w[0], w[1] - w[0])).collect()
    }

    /// `[B, L_max, ...]` padded with `pad_value`, and the `[B, L_max]` validity mask.
    pub fn to_padded(&self, pad_value: f64) -> Result<(Tensor, Tensor)> {
        let lens = self.lengths();
        let max = lens.iter().copied().max().unwrap_or(0);
        let nd = self.values.ndim();
        let mut rows = Vec::with_capacity(lens.len());
        for s in self.unbind()? {
            let mut pads = vec![(0usize, 0usize); nd];
            pads[0] = (0, max - s.shape()[0]);
            rows.push(s.pad(&pads, pad_value)?);
        }
        let padded = Tensor::stack(&rows, 0)?;
        let mask: Vec<bool> = lens.iter().flat_map(|&l| (0..max).map(move |i| i < l)).collect();
        Ok((padded, Tensor::from_vec(mask, &[lens.len(), max])?))
    }

    /// Applies a function that keeps the number of rows (any elementwise operation, a `Linear` layer, ...) to the packed values.
    pub fn map(&self, f: impl FnOnce(&Tensor) -> Result<Tensor>) -> Result<Self> {
        let v = f(&self.values)?;
        if v.ndim() == 0 || v.shape()[0] != self.values.shape()[0] {
            return Err(Error::Msg("RaggedTensor::map: the function must keep the number of rows".into()));
        }
        Ok(Self { values: v, offsets: self.offsets.clone() })
    }

    /// Per-sequence sum over the first axis: `[B, ...]`.
    pub fn sum_seq(&self) -> Result<Tensor> {
        let rows = self.unbind()?.iter().map(|s| s.sum_axes(&[0], false)).collect::<Result<Vec<_>>>()?;
        Tensor::stack(&rows, 0)
    }

    /// Per-sequence mean over the first axis: `[B, ...]` (an empty sequence gives NaN).
    pub fn mean_seq(&self) -> Result<Tensor> {
        let rows = self.unbind()?.iter().map(|s| s.mean_axes(&[0], false)).collect::<Result<Vec<_>>>()?;
        Tensor::stack(&rows, 0)
    }

    /// Per-sequence maximum over the first axis: `[B, ...]`.
    pub fn max_seq(&self) -> Result<Tensor> {
        let rows = self.unbind()?.iter().map(|s| s.max_axis(0, false)).collect::<Result<Vec<_>>>()?;
        Tensor::stack(&rows, 0)
    }

    /// Softmax over the first axis within every sequence (padding never takes part).
    pub fn softmax_seq(&self) -> Result<Self> {
        let rows = self.unbind()?.iter().map(|s| s.softmax(0)).collect::<Result<Vec<_>>>()?;
        Self::from_list(&rows)
    }

    /// Boolean `[B, L_max]` mask that is true where a position holds a real element (an attention padding mask).
    pub fn mask(&self) -> Result<Tensor> {
        self.to_padded(0.0).map(|(_, m)| m)
    }
}

// ------------------------------------------------------------------------------------------------ masked

/// Data with a validity mask (`true` = valid). Reductions and softmax ignore the masked-out elements.
#[derive(Clone)]
pub struct MaskedTensor {
    data: Tensor,
    mask: Tensor,
}

impl MaskedTensor {
    pub fn new(data: &Tensor, mask: &Tensor) -> Result<Self> {
        if mask.dtype() != DType::Bool || mask.shape() != data.shape() {
            return Err(Error::Msg(format!("MaskedTensor: the mask must be a bool tensor of shape {:?}", data.shape())));
        }
        Ok(Self { data: data.clone(), mask: mask.clone() })
    }

    pub fn data(&self) -> &Tensor {
        &self.data
    }
    pub fn mask(&self) -> &Tensor {
        &self.mask
    }

    /// The data with masked-out positions replaced by `fill`.
    pub fn to_tensor(&self, fill: f64) -> Result<Tensor> {
        self.data.masked_fill(&self.mask.logical_not()?, fill)
    }

    /// Number of valid elements along `axes`.
    pub fn count(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.mask.to_dtype(self.data.dtype())?.sum_axes(axes, keepdim)
    }

    pub fn sum(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.to_tensor(0.0)?.sum_axes(axes, keepdim)
    }

    /// Mean of the valid elements (NaN where nothing is valid).
    pub fn mean(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.sum(axes, keepdim)?.div(&self.count(axes, keepdim)?)
    }

    /// Maximum of the valid elements (`-inf` where nothing is valid).
    pub fn amax(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.to_tensor(f64::NEG_INFINITY)?.amax(axes, keepdim)
    }

    pub fn amin(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.to_tensor(f64::INFINITY)?.amin(axes, keepdim)
    }

    /// Softmax over the valid elements along `axis`; masked-out positions come out as 0.
    pub fn softmax(&self, axis: isize) -> Result<Tensor> {
        let m = crate::tensor::no_grad(|| self.to_tensor(f64::NEG_INFINITY)?.amax(&[axis], true))?;
        // rows without a valid element have max -inf: use 0 there so that the shifted values stay finite
        let any = self.count(&[axis], true)?;
        let m = Tensor::where_cond(&any.gt(&any.zeros_like()?)?, &m, &m.zeros_like()?)?;
        let e = self.data.sub(&m)?.exp()?.mul(&self.mask.to_dtype(self.data.dtype())?)?;
        let denom = e.sum_axes(&[axis], true)?;
        // a fully masked row has denominator 0: report 0 instead of NaN
        let safe = Tensor::where_cond(&denom.gt(&denom.zeros_like()?)?, &denom, &denom.ones_like()?)?;
        e.div(&safe)
    }

    /// Elementwise sum of two masked tensors; an element is valid when both are.
    pub fn add(&self, other: &Self) -> Result<Self> {
        Ok(Self { data: self.data.add(&other.data)?, mask: self.mask.logical_and(&other.mask)? })
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        Ok(Self { data: self.data.mul(&other.data)?, mask: self.mask.logical_and(&other.mask)? })
    }
}

// ------------------------------------------------------------------------------------------------ named

/// A tensor whose dimensions have names. Binary operations align operands by name (adding size-1 dimensions for names one of
/// them lacks), reductions take names instead of positions.
#[derive(Clone)]
pub struct NamedTensor {
    tensor: Tensor,
    names: Vec<String>,
}

impl NamedTensor {
    pub fn new(tensor: &Tensor, names: &[&str]) -> Result<Self> {
        if tensor.ndim() != names.len() {
            return Err(Error::Msg(format!("NamedTensor: {} names for a tensor of rank {}", names.len(), tensor.ndim())));
        }
        for (i, n) in names.iter().enumerate() {
            if names[..i].contains(n) {
                return Err(Error::Msg(format!("NamedTensor: the name {n:?} is used twice")));
            }
        }
        Ok(Self { tensor: tensor.clone(), names: names.iter().map(|s| s.to_string()).collect() })
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
    fn axis(&self, name: &str) -> Result<usize> {
        self.names.iter().position(|n| n == name).ok_or_else(|| Error::Msg(format!("NamedTensor: no dimension named {name:?} (have {:?})", self.names)))
    }
    pub fn size(&self, name: &str) -> Result<usize> {
        Ok(self.tensor.shape()[self.axis(name)?])
    }

    pub fn rename(&self, from: &str, to: &str) -> Result<Self> {
        let i = self.axis(from)?;
        if self.names.iter().any(|n| n == to) && to != from {
            return Err(Error::Msg(format!("NamedTensor::rename: {to:?} is already a dimension")));
        }
        let mut names = self.names.clone();
        names[i] = to.to_string();
        Ok(Self { tensor: self.tensor.clone(), names })
    }

    /// Reorders the dimensions to `order`, inserting size-1 dimensions for names that are not present (all current names must
    /// appear in `order`).
    pub fn align_to(&self, order: &[&str]) -> Result<Self> {
        if let Some(missing) = self.names.iter().find(|n| !order.contains(&n.as_str())) {
            return Err(Error::Msg(format!("NamedTensor::align_to: {missing:?} is not in {order:?}")));
        }
        let present: Vec<usize> = order.iter().filter_map(|n| self.names.iter().position(|m| m == n)).collect();
        let mut t = self.tensor.permute(&present)?;
        // unsqueeze the absent names at their positions
        for (pos, n) in order.iter().enumerate() {
            if !self.names.iter().any(|m| m == n) {
                t = t.unsqueeze(pos as isize)?;
            }
        }
        Ok(Self { tensor: t, names: order.iter().map(|s| s.to_string()).collect() })
    }

    fn joint_names(&self, other: &Self) -> Vec<String> {
        let mut names = self.names.clone();
        for n in &other.names {
            if !names.contains(n) {
                names.push(n.clone());
            }
        }
        names
    }

    fn binary(&self, other: &Self, f: impl FnOnce(&Tensor, &Tensor) -> Result<Tensor>) -> Result<Self> {
        let names = self.joint_names(other);
        let order: Vec<&str> = names.iter().map(String::as_str).collect();
        let (a, b) = (self.align_to(&order)?, other.align_to(&order)?);
        Ok(Self { tensor: f(&a.tensor, &b.tensor)?, names })
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.binary(other, |a, b| a.add(b))
    }
    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.binary(other, |a, b| a.sub(b))
    }
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.binary(other, |a, b| a.mul(b))
    }
    pub fn div(&self, other: &Self) -> Result<Self> {
        self.binary(other, |a, b| a.div(b))
    }

    fn reduce(&self, names: &[&str], keepdim: bool, f: impl FnOnce(&Tensor, &[isize], bool) -> Result<Tensor>) -> Result<Self> {
        let axes = names.iter().map(|n| self.axis(n).map(|a| a as isize)).collect::<Result<Vec<_>>>()?;
        let t = f(&self.tensor, &axes, keepdim)?;
        let kept: Vec<String> = self.names.iter().filter(|n| keepdim || !names.contains(&n.as_str())).cloned().collect();
        Ok(Self { tensor: t, names: kept })
    }

    pub fn sum(&self, names: &[&str], keepdim: bool) -> Result<Self> {
        self.reduce(names, keepdim, |t, a, k| t.sum_axes(a, k))
    }
    pub fn mean(&self, names: &[&str], keepdim: bool) -> Result<Self> {
        self.reduce(names, keepdim, |t, a, k| t.mean_axes(a, k))
    }
    pub fn amax(&self, names: &[&str], keepdim: bool) -> Result<Self> {
        self.reduce(names, keepdim, |t, a, k| t.amax(a, k))
    }

    /// Softmax along a named dimension.
    pub fn softmax(&self, name: &str) -> Result<Self> {
        Ok(Self { tensor: self.tensor.softmax(self.axis(name)? as isize)?, names: self.names.clone() })
    }

    /// Contracts the shared dimension `name` of `self` and `other` (a named matrix product): the result carries the remaining
    /// dimensions of both, `self`'s first.
    pub fn contract(&self, other: &Self, name: &str) -> Result<Self> {
        let (ia, ib) = (self.axis(name)?, other.axis(name)?);
        if self.tensor.shape()[ia] != other.tensor.shape()[ib] {
            return Err(Error::Msg(format!("NamedTensor::contract: {name:?} has size {} vs {}", self.tensor.shape()[ia], other.tensor.shape()[ib])));
        }
        if let Some(dup) = self.names.iter().find(|n| *n != name && other.names.contains(n)) {
            return Err(Error::Msg(format!("NamedTensor::contract: the name {dup:?} would appear twice in the result")));
        }
        // a: [rest_a..., k] , b: [k, rest_b...] -> matmul on flattened sides
        let rest_a: Vec<usize> = (0..self.names.len()).filter(|&i| i != ia).collect();
        let rest_b: Vec<usize> = (0..other.names.len()).filter(|&i| i != ib).collect();
        let a_perm: Vec<usize> = rest_a.iter().copied().chain([ia]).collect();
        let b_perm: Vec<usize> = [ib].into_iter().chain(rest_b.iter().copied()).collect();
        let a = self.tensor.permute(&a_perm)?.contiguous()?;
        let b = other.tensor.permute(&b_perm)?.contiguous()?;
        let k = self.tensor.shape()[ia];
        let (ma, nb): (usize, usize) = (rest_a.iter().map(|&i| self.tensor.shape()[i]).product(), rest_b.iter().map(|&i| other.tensor.shape()[i]).product());
        let y = a.reshape(&[ma, k])?.matmul(&b.reshape(&[k, nb])?)?;
        let mut shape: Vec<usize> = rest_a.iter().map(|&i| self.tensor.shape()[i]).collect();
        shape.extend(rest_b.iter().map(|&i| other.tensor.shape()[i]));
        let mut names: Vec<String> = rest_a.iter().map(|&i| self.names[i].clone()).collect();
        names.extend(rest_b.iter().map(|&i| other.names[i].clone()));
        Ok(Self { tensor: y.reshape(&shape)?, names })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(x: &Tensor) -> Vec<f32> {
        x.to_dtype(DType::F32).unwrap().contiguous().unwrap().to_vec::<f32>().unwrap()
    }

    #[test]
    fn ragged_padding_round_trip_reductions_and_gradients() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let b = t(&[10.0, 20.0], &[1, 2]);
        let c = t(&[7.0, 8.0, 9.0, 1.0], &[2, 2]);
        let r = RaggedTensor::from_list(&[a.clone(), b.clone(), c.clone()]).unwrap();
        assert_eq!(r.lengths(), [3, 1, 2]);
        assert_eq!(r.batch_size(), 3);
        let (padded, mask) = r.to_padded(-1.0).unwrap();
        assert_eq!(padded.shape(), [3, 3, 2]);
        assert_eq!(vals(&padded.select(0, 1).unwrap()), vec![10.0, 20.0, -1.0, -1.0, -1.0, -1.0]);
        assert_eq!(mask.to_vec::<bool>().unwrap(), vec![true, true, true, true, false, false, true, true, false]);
        let back = RaggedTensor::from_padded(&padded, &[3, 1, 2]).unwrap();
        assert_eq!(vals(back.values()), vals(r.values()));
        assert_eq!(vals(&r.sum_seq().unwrap()), vec![9.0, 12.0, 10.0, 20.0, 16.0, 9.0]);
        assert_eq!(vals(&r.mean_seq().unwrap()), vec![3.0, 4.0, 10.0, 20.0, 8.0, 4.5]);
        assert_eq!(vals(&r.max_seq().unwrap()), vec![5.0, 6.0, 10.0, 20.0, 9.0, 8.0]);
        // softmax within each sequence equals the softmax of that sequence alone
        let s = r.softmax_seq().unwrap().unbind().unwrap();
        assert!(s[0].allclose(&a.softmax(0).unwrap(), 1e-6, 1e-6).unwrap() && s[2].allclose(&c.softmax(0).unwrap(), 1e-6, 1e-6).unwrap());
        // gradients flow through the packed values
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5, 1]).unwrap().requires_grad_(true);
        let rr = RaggedTensor { values: x.clone(), offsets: vec![0, 2, 5] };
        rr.map(|v| v.square()).unwrap().sum_seq().unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(vals(&x.grad().unwrap()), vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        assert!(RaggedTensor::from_list(&[a, t(&[1.0, 2.0, 3.0], &[1, 3])]).is_err());
    }

    #[test]
    fn masked_reductions_ignore_invalid_elements() {
        let data = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mask = Tensor::from_vec(vec![true, false, true, false, false, false], &[2, 3]).unwrap();
        let m = MaskedTensor::new(&data, &mask).unwrap();
        assert_eq!(vals(&m.sum(&[1], false).unwrap()), vec![4.0, 0.0]);
        let mean = vals(&m.mean(&[1], false).unwrap());
        assert_eq!(mean[0], 2.0);
        assert!(mean[1].is_nan());
        assert_eq!(vals(&m.amax(&[1], false).unwrap())[0], 3.0);
        assert_eq!(vals(&m.amax(&[1], false).unwrap())[1], f32::NEG_INFINITY);
        // softmax: valid entries normalize to 1, masked ones are 0, a fully masked row is all 0 (not NaN)
        let s = vals(&m.softmax(1).unwrap());
        let e1 = (1.0f32 - 3.0).exp();
        assert!((s[0] - e1 / (e1 + 1.0)).abs() < 1e-6 && s[1] == 0.0 && (s[2] - 1.0 / (e1 + 1.0)).abs() < 1e-6);
        assert_eq!(&s[3..], &[0.0, 0.0, 0.0]);
        assert!(MaskedTensor::new(&data, &data).is_err());
        // gradients of a masked mean reach only valid elements
        let x = t(&[1.0, 2.0, 3.0], &[1, 3]).requires_grad_(true);
        let mk = Tensor::from_vec(vec![true, false, true], &[1, 3]).unwrap();
        MaskedTensor::new(&x, &mk).unwrap().mean(&[1], false).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(vals(&x.grad().unwrap()), vec![0.5, 0.0, 0.5]);
    }

    #[test]
    fn named_dimensions_align_broadcast_and_reduce_by_name() {
        let x = NamedTensor::new(&t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]), &["batch", "feature"]).unwrap();
        let bias = NamedTensor::new(&t(&[10.0, 20.0, 30.0], &[3]), &["feature"]).unwrap();
        let scale = NamedTensor::new(&t(&[1.0, 100.0], &[2]), &["batch"]).unwrap();
        let y = x.add(&bias).unwrap().mul(&scale).unwrap();
        assert_eq!(y.names(), ["batch", "feature"]);
        assert_eq!(vals(y.tensor()), vec![11.0, 22.0, 33.0, 1400.0, 2500.0, 3600.0]);
        // operands with the dimensions in the other order still line up
        let xt = NamedTensor::new(&x.tensor().t().unwrap(), &["feature", "batch"]).unwrap();
        let z = xt.add(&x).unwrap();
        assert_eq!(z.names(), ["feature", "batch"]);
        assert_eq!(vals(z.tensor()), vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]);
        assert_eq!(vals(x.sum(&["feature"], false).unwrap().tensor()), vec![6.0, 15.0]);
        assert_eq!(x.mean(&["batch"], true).unwrap().names(), ["batch", "feature"]);
        assert_eq!(x.size("feature").unwrap(), 3);
        assert_eq!(x.rename("feature", "d").unwrap().names(), ["batch", "d"]);
        // a named contraction equals the plain matrix product
        let w = NamedTensor::new(&t(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]), &["feature", "out"]).unwrap();
        let p = x.contract(&w, "feature").unwrap();
        assert_eq!(p.names(), ["batch", "out"]);
        assert_eq!(vals(p.tensor()), vec![4.0, 5.0, 10.0, 11.0]);
        assert!(x.contract(&x, "feature").is_err());
        assert!(x.sum(&["nope"], false).is_err());
        assert!(NamedTensor::new(&t(&[1.0, 2.0], &[1, 2]), &["a", "a"]).is_err());
    }
}
