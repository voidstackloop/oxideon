//! Tensor operations that PyTorch has and the core op set was missing: log-add-exp and float power, `frexp` / `ldexp`,
//! integer helpers (`gcd`, `lcm`, shifts), index-based writes (`index_put`, `index_copy`, `scatter_reduce`, the
//! `*_scatter` family), `nonzero` / `unique` / `bucketize`, matrix-product helpers (`mm`, `bmm`, `addmm`, ...), and
//! small shape / reduction conveniences. Everything that has a derivative is composed from differentiable ops.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::{broadcast_shapes, normalize_axis};
use crate::tensor::Tensor;

fn int_binary(
    name: &'static str,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(i64, i64) -> i64,
) -> Result<Tensor> {
    for t in [a, b] {
        if !matches!(t.dtype(), DType::Bool | DType::U8 | DType::I32 | DType::I64) {
            return Err(Error::Msg(format!("{name}: expects integer tensors, got {:?}", t.dtype())));
        }
    }
    let shape = broadcast_shapes(name, a.shape(), b.shape())?;
    let av = a.broadcast_to(&shape)?.to_vec::<i64>()?;
    let bv = b.broadcast_to(&shape)?.to_vec::<i64>()?;
    let out: Vec<i64> = av.iter().zip(&bv).map(|(&x, &y)| f(x, y)).collect();
    let dt = if a.dtype() == DType::I64 || b.dtype() == DType::I64 { DType::I64 } else { a.dtype() };
    Tensor::from_slice_on(&out, &shape, &a.device())?.to_dtype(dt)
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.unsigned_abs(), b.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a as i64
}

/// Linear (row-major) strides of `shape`.
fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Replaces (or accumulates into) rows of a `[P, C]` matrix: the shared core of the index writers. With
/// `accumulate = false` and repeated rows the last value wins.
fn put_rows(base: &Tensor, rows: &[i64], vals: &Tensor, accumulate: bool) -> Result<Tensor> {
    let n_rows = base.shape()[0] as i64;
    for &r in rows {
        if r < 0 || r >= n_rows {
            return Err(Error::Msg(format!("index {r} is out of range for a dimension of size {n_rows}")));
        }
    }
    if rows.is_empty() {
        return Ok(base.clone());
    }
    let dev = base.device();
    if accumulate {
        let idx = Tensor::from_slice_on(rows, &[rows.len()], &dev)?;
        return base.index_add(0, &idx, &vals.to_dtype(base.dtype())?);
    }
    // keep the last occurrence of every row
    let mut last = std::collections::HashMap::new();
    for (pos, &r) in rows.iter().enumerate() {
        last.insert(r, pos);
    }
    let mut keep: Vec<(i64, usize)> = last.into_iter().collect();
    keep.sort_unstable();
    let ridx: Vec<i64> = keep.iter().map(|k| k.0).collect();
    let sel: Vec<i64> = keep.iter().map(|k| k.1 as i64).collect();
    let ridx_t = Tensor::from_slice_on(&ridx, &[ridx.len()], &dev)?;
    let sel_t = Tensor::from_slice_on(&sel, &[sel.len()], &dev)?;
    let picked = vals.to_dtype(base.dtype())?.index_select(0, &sel_t)?;
    base.index_fill(0, &ridx_t, 0.0)?.index_add(0, &ridx_t, &picked)
}

impl Tensor {
    // ------------------------------------------------------------------------------------------------------
    // Elementwise math
    // ------------------------------------------------------------------------------------------------------

    /// `ln(exp(self) + exp(other))` without overflow (`torch.logaddexp`).
    pub fn logaddexp(&self, other: &Tensor) -> Result<Tensor> {
        let shape = broadcast_shapes("logaddexp", self.shape(), other.shape())?;
        let a = self.broadcast_to(&shape)?;
        let b = other.broadcast_to(&shape)?;
        Tensor::stack(&[a, b], 0)?.logsumexp(0, false)
    }

    /// `log2(2^self + 2^other)` (`torch.logaddexp2`).
    pub fn logaddexp2(&self, other: &Tensor) -> Result<Tensor> {
        let ln2 = std::f64::consts::LN_2;
        self.mul_scalar(ln2)?
            .logaddexp(&other.mul_scalar(ln2)?)?
            .mul_scalar(1.0 / ln2)
    }

    /// `self ** exponent` computed in `f64` (`torch.float_power`).
    pub fn float_power(&self, exponent: &Tensor) -> Result<Tensor> {
        self.to_dtype(DType::F64)?.pow(&exponent.to_dtype(DType::F64)?)
    }

    /// `self * 2 ** other` (`torch.ldexp`).
    pub fn ldexp(&self, other: &Tensor) -> Result<Tensor> {
        let x = if matches!(self.dtype(), DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            self.clone()
        } else {
            self.to_dtype(DType::F32)?
        };
        x.mul(&other.to_dtype(x.dtype())?.exp2()?)
    }

    /// Decomposes into a mantissa in `[0.5, 1)` and an integer exponent with `self = mantissa * 2 ** exponent`
    /// (`torch.frexp`). The mantissa is differentiable.
    pub fn frexp(&self) -> Result<(Tensor, Tensor)> {
        let vals = self.to_vec::<f64>()?;
        let exps: Vec<i32> = vals.iter().map(|&v| libm::frexp(v).1).collect();
        let e = Tensor::from_slice_on(&exps, self.shape(), &self.device())?;
        let scale = e.to_dtype(DType::F64)?.neg()?.exp2()?;
        let x = if matches!(self.dtype(), DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            self.clone()
        } else {
            self.to_dtype(DType::F32)?
        };
        Ok((x.mul(&scale.to_dtype(x.dtype())?)?, e))
    }

    /// Step function: `0` for negative, `values` at zero, `1` for positive (`torch.heaviside`).
    pub fn heaviside(&self, values: &Tensor) -> Result<Tensor> {
        let pos = self.gt(&self.zeros_like()?)?.to_dtype(self.dtype())?;
        let at_zero = self.eq_tensor(&self.zeros_like()?)?;
        Tensor::where_cond(&at_zero, &values.to_dtype(self.dtype())?.broadcast_to(self.shape())?, &pos)
    }

    /// Whether the sign bit is set, so `-0.0` counts (`torch.signbit`).
    pub fn signbit(&self) -> Result<Tensor> {
        let v: Vec<bool> = self.to_vec::<f64>()?.iter().map(|x| x.is_sign_negative()).collect();
        Tensor::from_slice_on(&v, self.shape(), &self.device())
    }

    pub fn deg2rad(&self) -> Result<Tensor> {
        self.mul_scalar(std::f64::consts::PI / 180.0)
    }

    pub fn rad2deg(&self) -> Result<Tensor> {
        self.mul_scalar(180.0 / std::f64::consts::PI)
    }

    /// Elementwise maximum that ignores NaN when the other operand is a number (`torch.fmax`).
    pub fn fmax(&self, other: &Tensor) -> Result<Tensor> {
        let m = self.maximum(other)?;
        let m = Tensor::where_cond(&self.isnan()?, other, &m)?;
        Tensor::where_cond(&other.isnan()?, self, &m)
    }

    /// Elementwise minimum that ignores NaN when the other operand is a number (`torch.fmin`).
    pub fn fmin(&self, other: &Tensor) -> Result<Tensor> {
        let m = self.minimum(other)?;
        let m = Tensor::where_cond(&self.isnan()?, other, &m)?;
        Tensor::where_cond(&other.isnan()?, self, &m)
    }

    /// `floor(self / other)` (`torch.floor_divide`).
    pub fn floor_divide(&self, other: &Tensor) -> Result<Tensor> {
        self.division_family(other, crate::ops::DivKind::FloorDivide)
    }

    /// Alias of [`div`](Self::div) (`torch.true_divide`).
    pub fn true_divide(&self, other: &Tensor) -> Result<Tensor> {
        self.div(other)
    }

    pub fn clamp_min(&self, min: f64) -> Result<Tensor> {
        self.clamp(min, f64::INFINITY)
    }

    pub fn clamp_max(&self, max: f64) -> Result<Tensor> {
        self.clamp(f64::NEG_INFINITY, max)
    }

    /// `|self - other| <= atol + rtol * |other|` (`torch.isclose`).
    pub fn isclose(&self, other: &Tensor, rtol: f64, atol: f64, equal_nan: bool) -> Result<Tensor> {
        let tol = other.abs()?.mul_scalar(rtol)?.add_scalar(atol)?;
        let close = self.sub(other)?.abs()?.le(&tol)?;
        let same = self.eq_tensor(other)?; // infinities of the same sign
        let mut out = close.logical_or(&same)?;
        if equal_nan {
            out = out.logical_or(&self.isnan()?.logical_and(&other.isnan()?)?)?;
        }
        Ok(out)
    }

    /// Whether both tensors have the same shape and values (`torch.equal`).
    pub fn equal(&self, other: &Tensor) -> Result<bool> {
        if self.shape() != other.shape() {
            return Ok(false);
        }
        self.eq_tensor(other)?.all()?.item::<bool>()
    }

    /// Next representable float after `self` towards `other` (`torch.nextafter`; `f32` and `f64`).
    pub fn nextafter(&self, other: &Tensor) -> Result<Tensor> {
        let dt = self.dtype();
        if !matches!(dt, DType::F32 | DType::F64) {
            return Err(Error::Msg("nextafter: only f32 and f64 are supported".into()));
        }
        let shape = broadcast_shapes("nextafter", self.shape(), other.shape())?;
        let a = self.broadcast_to(&shape)?.to_vec::<f64>()?;
        let b = other.to_dtype(dt)?.broadcast_to(&shape)?.to_vec::<f64>()?;
        let out: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| {
                if dt == DType::F32 {
                    let (x, y) = (x as f32, y as f32);
                    if x.is_nan() || y.is_nan() {
                        f64::NAN
                    } else if x == y {
                        f64::from(y)
                    } else if x == 0.0 {
                        f64::from(f32::from_bits(1).copysign(y))
                    } else {
                        let bits = x.to_bits();
                        let up = (y > x) == (x > 0.0);
                        f64::from(f32::from_bits(if up { bits + 1 } else { bits - 1 }))
                    }
                } else if x.is_nan() || y.is_nan() {
                    f64::NAN
                } else if x == y {
                    y
                } else if x == 0.0 {
                    f64::from_bits(1).copysign(y)
                } else {
                    let bits = x.to_bits();
                    let up = (y > x) == (x > 0.0);
                    f64::from_bits(if up { bits + 1 } else { bits - 1 })
                }
            })
            .collect();
        Tensor::from_slice_on(&out, &shape, &self.device())?.to_dtype(dt)
    }

    /// Greatest common divisor of integer tensors (`torch.gcd`).
    pub fn gcd(&self, other: &Tensor) -> Result<Tensor> {
        int_binary("gcd", self, other, gcd_i64)
    }

    /// Least common multiple of integer tensors (`torch.lcm`).
    pub fn lcm(&self, other: &Tensor) -> Result<Tensor> {
        int_binary("lcm", self, other, |a, b| {
            let g = gcd_i64(a, b);
            if g == 0 { 0 } else { (a / g * b).abs() }
        })
    }

    /// `self << other` on integer tensors (`torch.bitwise_left_shift`).
    pub fn bitwise_left_shift(&self, other: &Tensor) -> Result<Tensor> {
        int_binary("bitwise_left_shift", self, other, |a, b| {
            if (0..64).contains(&b) { a.wrapping_shl(b as u32) } else { 0 }
        })
    }

    /// Arithmetic `self >> other` on integer tensors (`torch.bitwise_right_shift`).
    pub fn bitwise_right_shift(&self, other: &Tensor) -> Result<Tensor> {
        int_binary("bitwise_right_shift", self, other, |a, b| {
            if (0..64).contains(&b) { a >> b } else if a < 0 { -1 } else { 0 }
        })
    }

    // ------------------------------------------------------------------------------------------------------
    // Reductions
    // ------------------------------------------------------------------------------------------------------

    /// Maximum over several axes at once (`torch.amax`).
    pub fn amax(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.extreme_over(axes, keepdim, true)
    }

    /// Minimum over several axes at once (`torch.amin`).
    pub fn amin(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.extreme_over(axes, keepdim, false)
    }

    fn extreme_over(&self, axes: &[isize], keepdim: bool, max: bool) -> Result<Tensor> {
        let nd = self.ndim();
        let mut ax: Vec<usize> = if axes.is_empty() {
            (0..nd).collect()
        } else {
            axes.iter()
                .map(|&a| normalize_axis("amax/amin", a, nd))
                .collect::<Result<_>>()?
        };
        ax.sort_unstable();
        ax.dedup();
        // forward on detached data, keeping the reduced axes so the result broadcasts against the input
        let mut kept = self.detach();
        for &a in ax.iter().rev() {
            kept = if max { kept.max_axis(a as isize, true)? } else { kept.min_axis(a as isize, true)? };
        }
        let kept_shape = kept.shape().to_vec();
        let out_shape: Vec<usize> = if keepdim { kept_shape.clone() } else { (0..nd).filter(|d| !ax.contains(d)).map(|d| self.shape()[d]).collect() };
        let out = kept.reshape(&out_shape)?;
        let (inp, full_shape) = (self.clone(), self.shape().to_vec());
        Ok(Tensor::from_op(out.storage().clone(), out.layout().clone(), "amax_amin", vec![self.clone()], move |g| {
            // torch.amax / amin: the gradient is shared evenly among all tied extremal elements
            let mask = inp.detach().eq_tensor(&kept.broadcast_to(&full_shape)?)?.to_dtype(g.dtype())?;
            let mut count = mask.clone();
            for &a in ax.iter().rev() {
                count = count.sum_axis(a as isize, true)?;
            }
            Ok(vec![Some(mask.div(&count)?.mul(&g.reshape(&kept_shape)?)?)])
        }))
    }

    /// `(std, mean)` along `axis` (`torch.std_mean`).
    pub fn std_mean(&self, axis: isize, unbiased: bool, keepdim: bool) -> Result<(Tensor, Tensor)> {
        Ok((self.std_axis(axis, unbiased, keepdim)?, self.mean_axis(axis, keepdim)?))
    }

    /// `(var, mean)` along `axis` (`torch.var_mean`).
    pub fn var_mean(&self, axis: isize, unbiased: bool, keepdim: bool) -> Result<(Tensor, Tensor)> {
        Ok((self.var_axis(axis, unbiased, keepdim)?, self.mean_axis(axis, keepdim)?))
    }

    /// Product along `axis` treating NaN as one (`torch.nanprod`).
    pub fn nanprod(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        let clean = Tensor::where_cond(&self.isnan()?, &self.ones_like()?, self)?;
        clean.prod_axis(axis, keepdim)
    }

    /// Cumulative trapezoid integral along `axis` with uniform spacing `dx` (`torch.cumulative_trapezoid`).
    pub fn cumulative_trapezoid(&self, axis: isize, dx: f64) -> Result<Tensor> {
        let ax = normalize_axis("cumulative_trapezoid", axis, self.ndim())?;
        let n = self.shape()[ax];
        if n < 2 {
            return self.narrow(axis, 0, 0);
        }
        let left = self.narrow(axis, 0, n - 1)?;
        let right = self.narrow(axis, 1, n - 1)?;
        left.add(&right)?.mul_scalar(0.5 * dx)?.cumsum(axis)
    }

    /// Cumulative trapezoid integral against sample points `x` (1-D, same length as `axis`).
    pub fn cumulative_trapezoid_x(&self, axis: isize, x: &Tensor) -> Result<Tensor> {
        let ax = normalize_axis("cumulative_trapezoid", axis, self.ndim())?;
        let n = self.shape()[ax];
        if x.shape() != [n] {
            return Err(Error::Msg(format!("cumulative_trapezoid: x must have shape [{n}]")));
        }
        if n < 2 {
            return self.narrow(axis, 0, 0);
        }
        let dxs = x.narrow(0, 1, n - 1)?.sub(&x.narrow(0, 0, n - 1)?)?;
        let mut shape = vec![1; self.ndim()];
        shape[ax] = n - 1;
        let dxs = dxs.reshape(&shape)?.to_dtype(self.dtype())?;
        let left = self.narrow(axis, 0, n - 1)?;
        let right = self.narrow(axis, 1, n - 1)?;
        left.add(&right)?.mul(&dxs)?.mul_scalar(0.5)?.cumsum(axis)
    }

    // ------------------------------------------------------------------------------------------------------
    // Shape helpers
    // ------------------------------------------------------------------------------------------------------

    pub fn swapaxes(&self, a: isize, b: isize) -> Result<Tensor> {
        self.transpose(a, b)
    }

    pub fn swapdims(&self, a: isize, b: isize) -> Result<Tensor> {
        self.transpose(a, b)
    }

    /// Flattens to one dimension (`torch.ravel`).
    pub fn ravel(&self) -> Result<Tensor> {
        self.reshape(&[self.numel()])
    }

    /// Rotates by 90 degrees `k` times in the plane of `dims` (`torch.rot90`).
    pub fn rot90(&self, k: i64, dims: [isize; 2]) -> Result<Tensor> {
        let nd = self.ndim();
        let d0 = normalize_axis("rot90", dims[0], nd)?;
        let d1 = normalize_axis("rot90", dims[1], nd)?;
        if d0 == d1 {
            return Err(Error::Msg("rot90: dims must be different".into()));
        }
        match k.rem_euclid(4) {
            0 => Ok(self.clone()),
            1 => self.flip(&[dims[1]])?.transpose(dims[0], dims[1]),
            2 => self.flip(&[dims[0], dims[1]]),
            _ => self.transpose(dims[0], dims[1])?.flip(&[dims[1]]),
        }
    }

    /// The elements at flat `index` positions, shaped like `index` (`torch.take`).
    pub fn take(&self, index: &Tensor) -> Result<Tensor> {
        let n = self.numel() as i64;
        let flat = index.to_vec::<i64>()?;
        let fixed: Vec<i64> = flat
            .iter()
            .map(|&i| if i < 0 { i + n } else { i })
            .collect();
        if fixed.iter().any(|&i| i < 0 || i >= n) {
            return Err(Error::Msg(format!("take: index out of range for a tensor of {n} elements")));
        }
        let idx = Tensor::from_slice_on(&fixed, &[fixed.len()], &self.device())?;
        self.reshape(&[self.numel()])?.index_select(0, &idx)?.reshape(index.shape())
    }

    /// Writes (or accumulates) `source` at flat `index` positions, returning the new tensor (`torch.Tensor.put`).
    pub fn put(&self, index: &Tensor, source: &Tensor, accumulate: bool) -> Result<Tensor> {
        let n = self.numel() as i64;
        let rows: Vec<i64> = index
            .to_vec::<i64>()?
            .iter()
            .map(|&i| if i < 0 { i + n } else { i })
            .collect();
        if source.numel() != rows.len() {
            return Err(Error::Msg("put: index and source must have the same number of elements".into()));
        }
        let base = self.reshape(&[self.numel(), 1])?;
        let vals = source.reshape(&[rows.len(), 1])?;
        put_rows(&base, &rows, &vals, accumulate)?.reshape(self.shape())
    }

    /// `gather` along `axis` where `indices` and `self` broadcast against each other in the other dimensions
    /// (`torch.take_along_dim`).
    pub fn take_along_dim(&self, indices: &Tensor, axis: isize) -> Result<Tensor> {
        let nd = self.ndim();
        if indices.ndim() != nd {
            return Err(Error::Msg("take_along_dim: indices must have the same number of dimensions".into()));
        }
        let ax = normalize_axis("take_along_dim", axis, nd)?;
        let mut self_shape = self.shape().to_vec();
        let mut idx_shape = indices.shape().to_vec();
        let (s_ax, i_ax) = (self_shape[ax], idx_shape[ax]);
        self_shape[ax] = 1;
        idx_shape[ax] = 1;
        let common = broadcast_shapes("take_along_dim", &self_shape, &idx_shape)?;
        let mut s_t = common.clone();
        s_t[ax] = s_ax;
        let mut i_t = common;
        i_t[ax] = i_ax;
        self.broadcast_to(&s_t)?.gather(axis, &indices.broadcast_to(&i_t)?)
    }

    /// Advanced-indexing write: `out[indices[0], indices[1], ...] = values` (or `+=` with `accumulate`), the
    /// functional form of `torch.index_put`. `indices` index the leading dimensions; `values` broadcasts to
    /// `[broadcast(indices)..., remaining dims...]`.
    pub fn index_put(&self, indices: &[Tensor], values: &Tensor, accumulate: bool) -> Result<Tensor> {
        let k = indices.len();
        if k == 0 || k > self.ndim() {
            return Err(Error::Msg(format!("index_put: {k} index tensors for a {}-d tensor", self.ndim())));
        }
        let mut idx_shape = indices[0].shape().to_vec();
        for t in &indices[1..] {
            idx_shape = broadcast_shapes("index_put", &idx_shape, t.shape())?;
        }
        let lead: Vec<usize> = self.shape()[..k].to_vec();
        let lead_strides = strides_of(&lead);
        let n: usize = idx_shape.iter().product();
        let mut lin = vec![0i64; n];
        for (d, t) in indices.iter().enumerate() {
            let v = t.broadcast_to(&idx_shape)?.to_vec::<i64>()?;
            for (slot, &i) in lin.iter_mut().zip(&v) {
                let i = if i < 0 { i + lead[d] as i64 } else { i };
                if i < 0 || i >= lead[d] as i64 {
                    return Err(Error::Msg(format!(
                        "index_put: index {i} is out of range for dimension {d} of size {}",
                        lead[d]
                    )));
                }
                *slot += i * lead_strides[d] as i64;
            }
        }
        let rest: Vec<usize> = self.shape()[k..].to_vec();
        let c: usize = rest.iter().product();
        let p: usize = lead.iter().product();
        let mut vshape = idx_shape.clone();
        vshape.extend(&rest);
        let vals = values.to_dtype(self.dtype())?.broadcast_to(&vshape)?.reshape(&[n, c])?;
        let base = self.reshape(&[p, c])?;
        put_rows(&base, &lin, &vals, accumulate)?.reshape(self.shape())
    }

    /// Copies the slices of `source` into positions `index` along `axis` (`torch.index_copy`).
    pub fn index_copy(&self, axis: isize, index: &Tensor, source: &Tensor) -> Result<Tensor> {
        let ax = normalize_axis("index_copy", axis, self.ndim())?;
        let rows = index.to_vec::<i64>()?;
        let rows: Vec<i64> = rows
            .iter()
            .map(|&i| if i < 0 { i + self.shape()[ax] as i64 } else { i })
            .collect();
        let moved = self.transpose(0, ax as isize)?.contiguous()?;
        let src = source.transpose(0, ax as isize)?.contiguous()?;
        if src.shape()[0] != rows.len() {
            return Err(Error::Msg("index_copy: source must have one slice per index".into()));
        }
        let c: usize = moved.shape()[1..].iter().product();
        let base = moved.reshape(&[moved.shape()[0], c])?;
        let vals = src.reshape(&[rows.len(), c])?;
        put_rows(&base, &rows, &vals, false)?
            .reshape(moved.shape())?
            .transpose(0, ax as isize)
    }

    /// Embeds `src` into a copy of `self` at position `index` of dimension `dim` (`torch.select_scatter`).
    pub fn select_scatter(&self, src: &Tensor, dim: isize, index: usize) -> Result<Tensor> {
        let idx = Tensor::from_slice_on(&[index as i64], &[1], &self.device())?;
        self.index_copy(dim, &idx, &src.unsqueeze(dim)?)
    }

    /// Embeds `src` into the slice `[start, end)` with `step` of dimension `dim` (`torch.slice_scatter`).
    pub fn slice_scatter(&self, src: &Tensor, dim: isize, start: usize, end: usize, step: usize) -> Result<Tensor> {
        if step == 0 {
            return Err(Error::Msg("slice_scatter: step must be positive".into()));
        }
        let rows: Vec<i64> = (start..end.min(self.shape()[normalize_axis("slice_scatter", dim, self.ndim())?]))
            .step_by(step)
            .map(|i| i as i64)
            .collect();
        let idx = Tensor::from_slice_on(&rows, &[rows.len()], &self.device())?;
        self.index_copy(dim, &idx, src)
    }

    /// Embeds `src` into the diagonal of `self` (`torch.diagonal_scatter`); `src` has the diagonal as its last dim.
    pub fn diagonal_scatter(&self, src: &Tensor, offset: isize, dim1: isize, dim2: isize) -> Result<Tensor> {
        let nd = self.ndim();
        let (d1, d2) = (
            normalize_axis("diagonal_scatter", dim1, nd)?,
            normalize_axis("diagonal_scatter", dim2, nd)?,
        );
        if d1 == d2 {
            return Err(Error::Msg("diagonal_scatter: dim1 and dim2 must differ".into()));
        }
        let mut perm: Vec<usize> = (0..nd).filter(|&d| d != d1 && d != d2).collect();
        perm.extend([d1, d2]);
        let moved = self.permute(&perm)?.contiguous()?;
        let (r, c) = (self.shape()[d1], self.shape()[d2]);
        let batch: Vec<usize> = moved.shape()[..nd - 2].to_vec();
        let flat = moved.reshape(&[batch.iter().product::<usize>(), r * c])?;
        let diag: Vec<i64> = (0..r)
            .filter_map(|i| {
                let j = i as isize + offset;
                (j >= 0 && (j as usize) < c).then(|| (i * c) as i64 + j as i64)
            })
            .collect();
        let mut want = batch.clone();
        want.push(diag.len());
        if src.shape() != want {
            return Err(Error::Msg(format!("diagonal_scatter: src must have shape {want:?}")));
        }
        let idx = Tensor::from_slice_on(&diag, &[diag.len()], &self.device())?;
        let b = batch.iter().product::<usize>();
        let src2 = src.reshape(&[b, diag.len()])?;
        let out = flat.index_copy(1, &idx, &src2)?.reshape(moved.shape())?;
        let mut inv = vec![0; nd];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        out.permute(&inv)
    }

    /// Scatter with a reduction (`torch.scatter_reduce`): `"sum"`, `"mean"`, `"prod"`, `"amax"` or `"amin"`.
    /// `index` and `src` must have the same shape. `"sum"` and `"mean"` are differentiable; the others are
    /// forward-only.
    pub fn scatter_reduce(
        &self,
        axis: isize,
        index: &Tensor,
        src: &Tensor,
        reduce: &str,
        include_self: bool,
    ) -> Result<Tensor> {
        let ax = normalize_axis("scatter_reduce", axis, self.ndim())?;
        if index.shape() != src.shape() {
            return Err(Error::Msg("scatter_reduce: index and src must have the same shape".into()));
        }
        match reduce {
            "sum" | "mean" => {
                let base = if include_self { self.clone() } else { self.zeros_like()? };
                let summed = base.scatter_add(axis, index, src)?;
                let ones = src.ones_like()?;
                // number of contributions per slot (the original value counts once when include_self)
                let seed = if include_self { self.ones_like()? } else { self.zeros_like()? };
                let count = seed.scatter_add(axis, index, &ones)?;
                let result = if reduce == "sum" { summed } else { summed.div(&count.maximum(&count.ones_like()?)?)? };
                if include_self {
                    return Ok(result);
                }
                // without the original value, slots that nothing was scattered into keep it unchanged
                let touched = count.gt(&count.zeros_like()?)?;
                Tensor::where_cond(&touched, &result, self)
            }
            "prod" | "amax" | "amin" => {
                let mut out = self.to_vec::<f64>()?;
                let self_v = out.clone();
                let shape = self.shape().to_vec();
                let sstr = strides_of(&shape);
                let ishape = index.shape().to_vec();
                let idx = index.to_vec::<i64>()?;
                let sv = src.to_vec::<f64>()?;
                let mut touched = vec![false; out.len()];
                let mut dst_of_k = vec![0usize; sv.len()];
                let istr = strides_of(&ishape);
                for (flat, (&ix, &s)) in idx.iter().zip(&sv).enumerate() {
                    if ix < 0 {
                        return Err(Error::Msg("scatter_reduce: index out of range".into()));
                    }
                    let mut dst = 0usize;
                    for d in 0..ishape.len() {
                        let coord = (flat / istr[d]) % ishape[d];
                        let c = if d == ax { ix as usize } else { coord };
                        if c >= shape[d] {
                            return Err(Error::Msg("scatter_reduce: index out of range".into()));
                        }
                        dst += c * sstr[d];
                    }
                    dst_of_k[flat] = dst;
                    let cur = out[dst];
                    out[dst] = if !include_self && !touched[dst] {
                        s
                    } else {
                        match reduce {
                            "prod" => cur * s,
                            "amax" => cur.max(s),
                            _ => cur.min(s),
                        }
                    };
                    touched[dst] = true;
                }
                let res = Tensor::from_slice_on(&out, &shape, &self.device())?.to_dtype(self.dtype())?;
                let self_req = self.requires_grad();
                let src_req = src.requires_grad();
                if !crate::tensor::is_grad_enabled() || (!self_req && !src_req) {
                    return Ok(res);
                }

                let mut self_factor = vec![0.0f64; out.len()];
                let mut src_factor = vec![0.0f64; sv.len()];
                if reduce == "prod" {
                    let mut slots = vec![Vec::<usize>::new(); out.len()];
                    for (k, &dst) in dst_of_k.iter().enumerate() {
                        slots[dst].push(k);
                    }
                    for dst in 0..out.len() {
                        let ks = &slots[dst];
                        if ks.is_empty() {
                            self_factor[dst] = 1.0;
                            continue;
                        }
                        let mut vals = Vec::with_capacity(ks.len() + usize::from(include_self));
                        if include_self {
                            vals.push(self_v[dst]);
                        }
                        for &k in ks {
                            vals.push(sv[k]);
                        }
                        let mut pref = Vec::with_capacity(vals.len());
                        let mut acc = 1.0f64;
                        for &v in &vals {
                            pref.push(acc);
                            acc *= v;
                        }
                        let mut suf = 1.0f64;
                        let mut facs = vec![0.0f64; vals.len()];
                        for j in (0..vals.len()).rev() {
                            facs[j] = pref[j] * suf;
                            suf *= vals[j];
                        }
                        let offset = if include_self {
                            self_factor[dst] = facs[0];
                            1
                        } else {
                            0
                        };
                        for (pos, &k) in ks.iter().enumerate() {
                            src_factor[k] = facs[offset + pos];
                        }
                    }
                } else {
                    let mut win_count = vec![0usize; out.len()];
                    let mut self_wins = vec![false; out.len()];
                    let mut src_wins = vec![false; sv.len()];
                    for dst in 0..out.len() {
                        if (include_self || !touched[dst]) && self_v[dst] == out[dst] {
                            self_wins[dst] = true;
                            win_count[dst] += 1;
                        }
                    }
                    for (k, (&s, &dst)) in sv.iter().zip(&dst_of_k).enumerate() {
                        if s == out[dst] {
                            src_wins[k] = true;
                            win_count[dst] += 1;
                        }
                    }
                    for dst in 0..out.len() {
                        if self_wins[dst] && win_count[dst] > 0 {
                            self_factor[dst] = 1.0 / (win_count[dst] as f64);
                        }
                    }
                    for k in 0..sv.len() {
                        let dst = dst_of_k[k];
                        if src_wins[k] && win_count[dst] > 0 {
                            src_factor[k] = 1.0 / (win_count[dst] as f64);
                        }
                    }
                }

                let self_shape = shape;
                let src_shape = ishape;
                let self_dtype = self.dtype();
                let src_dtype = src.dtype();
                let dev = self.device();
                Ok(Tensor::from_op(
                    res.storage().clone(),
                    res.layout().clone(),
                    "scatter_reduce",
                    vec![self.clone(), src.clone()],
                    move |g| {
                        let gv = g.to_vec::<f64>()?;
                        let gx = if self_req {
                            let v: Vec<f64> = gv.iter().zip(&self_factor).map(|(&gi, &f)| gi * f).collect();
                            Some(Tensor::from_slice_on(&v, &self_shape, &dev)?.to_dtype(self_dtype)?)
                        } else {
                            None
                        };
                        let gsrc = if src_req {
                            let v: Vec<f64> = dst_of_k.iter().zip(&src_factor).map(|(&dst, &f)| gv[dst] * f).collect();
                            Some(Tensor::from_slice_on(&v, &src_shape, &dev)?.to_dtype(src_dtype)?)
                        } else {
                            None
                        };
                        Ok(vec![gx, gsrc])
                    },
                ))
            }
            other => Err(Error::Msg(format!("scatter_reduce: unknown reduction '{other}'"))),
        }
    }

    // ------------------------------------------------------------------------------------------------------
    // Searching and sets
    // ------------------------------------------------------------------------------------------------------

    /// Coordinates of the non-zero elements, `[count, ndim]` (`torch.nonzero`).
    pub fn nonzero(&self) -> Result<Tensor> {
        let vals = self.to_vec::<f64>()?;
        let shape = self.shape();
        let strides = strides_of(shape);
        let mut out: Vec<i64> = Vec::new();
        let mut count = 0usize;
        for (flat, &v) in vals.iter().enumerate() {
            if v != 0.0 {
                count += 1;
                for d in 0..shape.len() {
                    out.push(((flat / strides[d]) % shape[d]) as i64);
                }
            }
        }
        Tensor::from_slice_on(&out, &[count, shape.len()], &self.device())
    }

    /// Alias of [`nonzero`](Self::nonzero) (`torch.argwhere`).
    pub fn argwhere(&self) -> Result<Tensor> {
        self.nonzero()
    }

    /// Sorted unique values of the flattened tensor (`torch.unique`).
    pub fn unique(&self) -> Result<Tensor> {
        Ok(self.unique_with()?.0)
    }

    /// `(values, inverse, counts)`: the sorted unique values, the index of each input element in `values`
    /// (shaped like the input) and how often each value occurs.
    pub fn unique_with(&self) -> Result<(Tensor, Tensor, Tensor)> {
        let vals = self.to_vec::<f64>()?;
        let mut order: Vec<usize> = (0..vals.len()).collect();
        order.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        let mut uniq: Vec<f64> = Vec::new();
        let mut counts: Vec<i64> = Vec::new();
        let mut inverse = vec![0i64; vals.len()];
        for &i in &order {
            if uniq.last().is_some_and(|&u| u == vals[i]) {
                *counts.last_mut().unwrap() += 1;
            } else {
                uniq.push(vals[i]);
                counts.push(1);
            }
            inverse[i] = (uniq.len() - 1) as i64;
        }
        let dev = self.device();
        Ok((
            Tensor::from_slice_on(&uniq, &[uniq.len()], &dev)?.to_dtype(self.dtype())?,
            Tensor::from_slice_on(&inverse, self.shape(), &dev)?,
            Tensor::from_slice_on(&counts, &[counts.len()], &dev)?,
        ))
    }

    /// Collapses runs of equal values (`torch.unique_consecutive`); returns `(values, inverse, counts)`.
    pub fn unique_consecutive(&self) -> Result<(Tensor, Tensor, Tensor)> {
        let vals = self.to_vec::<f64>()?;
        let mut uniq: Vec<f64> = Vec::new();
        let mut counts: Vec<i64> = Vec::new();
        let mut inverse = Vec::with_capacity(vals.len());
        for &v in &vals {
            if uniq.last().is_some_and(|&u| u == v) {
                *counts.last_mut().unwrap() += 1;
            } else {
                uniq.push(v);
                counts.push(1);
            }
            inverse.push((uniq.len() - 1) as i64);
        }
        let dev = self.device();
        Ok((
            Tensor::from_slice_on(&uniq, &[uniq.len()], &dev)?.to_dtype(self.dtype())?,
            Tensor::from_slice_on(&inverse, self.shape(), &dev)?,
            Tensor::from_slice_on(&counts, &[counts.len()], &dev)?,
        ))
    }

    /// Index of the bucket of every element given sorted `boundaries` (`torch.bucketize`).
    pub fn bucketize(&self, boundaries: &Tensor, right: bool) -> Result<Tensor> {
        Tensor::searchsorted(boundaries, self, right)
    }

    /// `[2, N]` row / column indices of the lower triangle (`torch.tril_indices`).
    pub fn tril_indices(rows: usize, cols: usize, offset: isize) -> Result<Tensor> {
        let mut r = Vec::new();
        let mut c = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                if (j as isize) - (i as isize) <= offset {
                    r.push(i as i64);
                    c.push(j as i64);
                }
            }
        }
        let n = r.len();
        r.extend(c);
        Tensor::from_vec(r, &[2, n])
    }

    /// `[2, N]` row / column indices of the upper triangle (`torch.triu_indices`).
    pub fn triu_indices(rows: usize, cols: usize, offset: isize) -> Result<Tensor> {
        let mut r = Vec::new();
        let mut c = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                if (j as isize) - (i as isize) >= offset {
                    r.push(i as i64);
                    c.push(j as i64);
                }
            }
        }
        let n = r.len();
        r.extend(c);
        Tensor::from_vec(r, &[2, n])
    }

    /// All length-`r` combinations of the elements of a 1-D tensor, `[count, r]` (`torch.combinations`).
    pub fn combinations(&self, r: usize, with_replacement: bool) -> Result<Tensor> {
        if self.ndim() != 1 {
            return Err(Error::Msg("combinations: expects a 1-D tensor".into()));
        }
        let n = self.shape()[0];
        let mut idx: Vec<i64> = Vec::new();
        let mut count = 0usize;
        let mut cur = vec![0usize; r];
        #[allow(clippy::too_many_arguments)]
        fn rec(pos: usize, start: usize, n: usize, r: usize, rep: bool, cur: &mut Vec<usize>, out: &mut Vec<i64>, count: &mut usize) {
            if pos == r {
                out.extend(cur.iter().map(|&v| v as i64));
                *count += 1;
                return;
            }
            for v in start..n {
                cur[pos] = v;
                rec(pos + 1, if rep { v } else { v + 1 }, n, r, rep, cur, out, count);
            }
        }
        if r > 0 {
            rec(0, 0, n, r, with_replacement, &mut cur, &mut idx, &mut count);
        }
        if count == 0 {
            return Tensor::zeros(&[0, r], self.dtype(), &self.device());
        }
        let it = Tensor::from_slice_on(&idx, &[idx.len()], &self.device())?;
        self.index_select(0, &it)?.reshape(&[count, r])
    }

    /// `[steps]` values `base ** linspace(start, end)` (`torch.logspace`).
    pub fn logspace(start: f64, end: f64, steps: usize, base: f64, dtype: DType) -> Result<Tensor> {
        let exps = Tensor::linspace(start, end, steps, DType::F64)?;
        Tensor::full(&[steps], base, DType::F64, &crate::backend::Device::Cpu)?
            .pow(&exps)?
            .to_dtype(dtype)
    }

    // ------------------------------------------------------------------------------------------------------
    // Matrix products
    // ------------------------------------------------------------------------------------------------------

    /// Matrix product of two 2-D tensors (`torch.mm`).
    pub fn mm(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(Error::Msg("mm: both operands must be 2-D".into()));
        }
        self.matmul(other)
    }

    /// Batched matrix product of two 3-D tensors (`torch.bmm`).
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 3 || other.ndim() != 3 || self.shape()[0] != other.shape()[0] {
            return Err(Error::Msg("bmm: both operands must be 3-D with the same batch size".into()));
        }
        self.matmul(other)
    }

    /// Matrix-vector product (`torch.mv`).
    pub fn mv(&self, vec: &Tensor) -> Result<Tensor> {
        if self.ndim() != 2 || vec.ndim() != 1 {
            return Err(Error::Msg("mv: expects a matrix and a vector".into()));
        }
        self.matmul(vec)
    }

    /// Inner product over the last dimension (`torch.inner`).
    pub fn inner(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() == 0 || other.ndim() == 0 {
            return self.mul(other);
        }
        let k = self.shape()[self.ndim() - 1];
        if other.shape()[other.ndim() - 1] != k {
            return Err(Error::Msg("inner: the last dimensions must match".into()));
        }
        let a = self.reshape(&[self.numel() / k.max(1), k])?;
        let b = other.reshape(&[other.numel() / k.max(1), k])?;
        let mut shape = self.shape()[..self.ndim() - 1].to_vec();
        shape.extend(&other.shape()[..other.ndim() - 1]);
        a.matmul(&b.t()?)?.reshape(&shape)
    }

    /// Dot product of two 1-D tensors (`torch.vdot`; real inputs).
    pub fn vdot(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(Error::Msg("vdot: expects 1-D tensors".into()));
        }
        self.dot(other)
    }

    /// `beta * self + alpha * (m1 @ m2)` (`torch.addmm`).
    pub fn addmm(&self, m1: &Tensor, m2: &Tensor, beta: f64, alpha: f64) -> Result<Tensor> {
        self.mul_scalar(beta)?.add(&m1.mm(m2)?.mul_scalar(alpha)?)
    }

    /// `beta * self + alpha * (mat @ vec)` (`torch.addmv`).
    pub fn addmv(&self, mat: &Tensor, vec: &Tensor, beta: f64, alpha: f64) -> Result<Tensor> {
        self.mul_scalar(beta)?.add(&mat.mv(vec)?.mul_scalar(alpha)?)
    }

    /// `beta * self + alpha * outer(v1, v2)` (`torch.addr`).
    pub fn addr(&self, v1: &Tensor, v2: &Tensor, beta: f64, alpha: f64) -> Result<Tensor> {
        self.mul_scalar(beta)?.add(&v1.outer(v2)?.mul_scalar(alpha)?)
    }

    /// `beta * self + alpha * bmm(b1, b2)` (`torch.baddbmm`).
    pub fn baddbmm(&self, b1: &Tensor, b2: &Tensor, beta: f64, alpha: f64) -> Result<Tensor> {
        self.mul_scalar(beta)?.add(&b1.bmm(b2)?.mul_scalar(alpha)?)
    }

    /// Rescales every sub-tensor along `dim` whose `p`-norm exceeds `maxnorm` (`torch.renorm`).
    pub fn renorm(&self, p: f64, dim: isize, maxnorm: f64) -> Result<Tensor> {
        let d = normalize_axis("renorm", dim, self.ndim())?;
        let axes: Vec<isize> = (0..self.ndim()).filter(|&a| a != d).map(|a| a as isize).collect();
        let norms = self.norm_p(p, &axes, true)?;
        let too_big = norms.gt(&norms.full_like(maxnorm)?)?;
        let factor = Tensor::where_cond(
            &too_big,
            &norms.add_scalar(1e-7)?.recip()?.mul_scalar(maxnorm)?,
            &norms.ones_like()?,
        )?;
        self.mul(&factor)
    }

    /// Condensed pairwise `p`-norm distances between the rows of a 2-D tensor (`torch.pdist`).
    pub fn pdist(&self, p: f64) -> Result<Tensor> {
        if self.ndim() != 2 {
            return Err(Error::Msg("pdist: expects a 2-D tensor".into()));
        }
        let n = self.shape()[0];
        let (mut ii, mut jj) = (Vec::new(), Vec::new());
        for i in 0..n {
            for j in i + 1..n {
                ii.push(i as i64);
                jj.push(j as i64);
            }
        }
        if ii.is_empty() {
            return Tensor::zeros(&[0], self.dtype(), &self.device());
        }
        let dev = self.device();
        let a = self.index_select(0, &Tensor::from_slice_on(&ii, &[ii.len()], &dev)?)?;
        let b = self.index_select(0, &Tensor::from_slice_on(&jj, &[jj.len()], &dev)?)?;
        a.sub(&b)?.norm_p(p, &[1], false)
    }

    /// `||self - other + eps||_p` over the last dimension (`torch.nn.functional.pairwise_distance`).
    pub fn pairwise_distance(&self, other: &Tensor, p: f64, eps: f64, keepdim: bool) -> Result<Tensor> {
        self.sub(other)?.add_scalar(eps)?.norm_p(p, &[-1], keepdim)
    }

    /// `softmax(-self)` along `axis` (`torch.nn.functional.softmin`).
    pub fn softmin(&self, axis: isize) -> Result<Tensor> {
        self.neg()?.softmax(axis)
    }

    /// `x - tanh(x)` (`torch.nn.functional.tanhshrink`).
    pub fn tanhshrink(&self) -> Result<Tensor> {
        self.sub(&self.tanh()?)
    }
}

/// Product of a chain of matrices with the cheapest parenthesization (`torch.linalg.multi_dot`).
pub fn multi_dot(mats: &[Tensor]) -> Result<Tensor> {
    match mats.len() {
        0 => return Err(Error::Msg("multi_dot: expects at least one tensor".into())),
        1 => return Ok(mats[0].clone()),
        2 => return mats[0].matmul(&mats[1]),
        _ => {}
    }
    // 1-D first / last operands act as a row / column vector
    let first_vec = mats[0].ndim() == 1;
    let last_vec = mats[mats.len() - 1].ndim() == 1;
    let ms: Vec<Tensor> = mats
        .iter()
        .enumerate()
        .map(|(i, m)| {
            if i == 0 && first_vec {
                m.unsqueeze(0)
            } else if i == mats.len() - 1 && last_vec {
                m.unsqueeze(1)
            } else {
                Ok(m.clone())
            }
        })
        .collect::<Result<_>>()?;
    let n = ms.len();
    let mut dims = vec![ms[0].shape()[0]];
    for m in &ms {
        if m.ndim() != 2 || m.shape()[0] != *dims.last().unwrap() {
            return Err(Error::Msg("multi_dot: shapes do not chain".into()));
        }
        dims.push(m.shape()[1]);
    }
    // classic matrix-chain DP
    let mut cost = vec![vec![0usize; n]; n];
    let mut split = vec![vec![0usize; n]; n];
    for len in 2..=n {
        for i in 0..=n - len {
            let j = i + len - 1;
            cost[i][j] = usize::MAX;
            for k in i..j {
                let c = cost[i][k]
                    .saturating_add(cost[k + 1][j])
                    .saturating_add(dims[i] * dims[k + 1] * dims[j + 1]);
                if c < cost[i][j] {
                    cost[i][j] = c;
                    split[i][j] = k;
                }
            }
        }
    }
    fn eval(ms: &[Tensor], split: &[Vec<usize>], i: usize, j: usize) -> Result<Tensor> {
        if i == j {
            return Ok(ms[i].clone());
        }
        let k = split[i][j];
        eval(ms, split, i, k)?.matmul(&eval(ms, split, k + 1, j)?)
    }
    let out = eval(&ms, &split, 0, n - 1)?;
    match (first_vec, last_vec) {
        (true, true) => out.reshape(&[]),
        (true, false) => out.squeeze_axis(0),
        (false, true) => out.squeeze_axis(1),
        _ => Ok(out),
    }
}

/// Alias of [`multi_dot`] (`torch.chain_matmul`).
pub fn chain_matmul(mats: &[Tensor]) -> Result<Tensor> {
    multi_dot(mats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: &[f64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn i(v: &[i64], shape: &[usize]) -> Tensor {
        Tensor::from_slice(v, shape).unwrap()
    }
    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }
    fn ivals(t: &Tensor) -> Vec<i64> {
        t.to_vec::<i64>().unwrap()
    }
    fn close(t: &Tensor, want: &[f64]) {
        let got = vals(t);
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9 * (1.0 + w.abs()), "{got:?} vs {want:?}");
        }
    }

    #[test]
    fn elementwise() {
        close(&f(&[1000.0, 0.0, -1000.0], &[3]).logaddexp(&f(&[1000.0, 0.0, 1.0], &[3])).unwrap(),
              &[1000.0 + 2f64.ln(), 2f64.ln(), 1.0 + (-1001f64).exp().ln_1p()]);
        close(&f(&[1.0, 3.0], &[2]).logaddexp2(&f(&[1.0, 0.0], &[2])).unwrap(), &[2.0, (8f64 + 1.0).log2()]);
        close(&f(&[2.0], &[1]).float_power(&f(&[0.5], &[1])).unwrap(), &[2f64.sqrt()]);
        close(&f(&[3.0], &[1]).ldexp(&f(&[2.0], &[1])).unwrap(), &[12.0]);
        let (m, e) = f(&[8.0, 0.75, -3.0, 0.0], &[4]).frexp().unwrap();
        close(&m, &[0.5, 0.75, -0.75, 0.0]);
        assert_eq!(ivals(&e.to_dtype(DType::I64).unwrap()), vec![4, 0, 2, 0]);
        close(&f(&[-1.0, 0.0, 2.0], &[3]).heaviside(&f(&[0.5], &[1])).unwrap(), &[0.0, 0.5, 1.0]);
        assert_eq!(f(&[-0.0, 1.0, -2.0], &[3]).signbit().unwrap().to_vec::<bool>().unwrap(), vec![true, false, true]);
        close(&f(&[180.0], &[1]).deg2rad().unwrap(), &[std::f64::consts::PI]);
        close(&f(&[f64::NAN, 1.0, 5.0], &[3]).fmax(&f(&[2.0, f64::NAN, 3.0], &[3])).unwrap(), &[2.0, 1.0, 5.0]);
        close(&f(&[7.0, -7.0], &[2]).floor_divide(&f(&[2.0, 2.0], &[2])).unwrap(), &[3.0, -4.0]);
        // exact near multiples, integer semantics (floor, sign of the divisor), NaN-propagating relu, half-to-even round
        close(&f(&[-7.5], &[1]).floor_divide(&f(&[0.3], &[1])).unwrap(), &[-26.0]);
        let (ia, ib) = (Tensor::from_slice(&[-7i64, 7, -7], &[3]).unwrap(), Tensor::from_slice(&[2i64, -2, -2], &[3]).unwrap());
        assert_eq!(ia.floor_divide(&ib).unwrap().to_vec::<i64>().unwrap(), vec![-4, -4, 3]);
        assert_eq!(ia.remainder(&ib).unwrap().to_vec::<i64>().unwrap(), vec![1, -1, -1]);
        assert_eq!(ia.fmod(&ib).unwrap().to_vec::<i64>().unwrap(), vec![-1, 1, -1]);
        assert!(ia.floor_divide(&Tensor::from_slice(&[0i64, 1, 1], &[3]).unwrap()).is_err());
        assert!(f(&[f64::NAN], &[1]).relu().unwrap().to_vec::<f64>().unwrap()[0].is_nan());
        close(&f(&[0.5, 1.5, 2.5, -0.5], &[4]).round().unwrap(), &[0.0, 2.0, 2.0, -0.0]);
        assert_eq!(f(&[f64::INFINITY], &[1]).hypot(&f(&[f64::NAN], &[1])).unwrap().to_vec::<f64>().unwrap(), vec![f64::INFINITY]);
        // NaN propagates through max / min / amax and wins argmax, as in torch
        let nn = f(&[1.0, f64::NAN, 3.0], &[3]);
        assert!(nn.max_axis(0, false).unwrap().to_vec::<f64>().unwrap()[0].is_nan());
        assert!(nn.min_axis(0, false).unwrap().to_vec::<f64>().unwrap()[0].is_nan());
        assert_eq!(nn.argmax_axis(0, false).unwrap().to_vec::<i64>().unwrap(), vec![1]);
        // log_softmax: NaN-free zero gradient on a +inf row, floor_divide stays differentiable, hypot does not overflow
        let xi = Tensor::from_slice(&[f64::INFINITY, 1.0, 0.0, 1.0], &[2, 2]).unwrap().requires_grad_(true);
        xi.log_softmax(1).unwrap().sum().unwrap().backward().unwrap();
        assert!(xi.grad().unwrap().to_vec::<f64>().unwrap().iter().all(|v| v.is_finite()));
        let (p, q) = (f(&[5.0, -5.0], &[2]).requires_grad_(true), f(&[2.0, 2.0], &[2]));
        p.floor_divide(&q).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(p.grad().unwrap().to_vec::<f64>().unwrap(), vec![0.0, 0.0]);
        let hy = f(&[1e200, 3.0], &[2]).hypot(&f(&[1e200, 4.0], &[2])).unwrap().to_vec::<f64>().unwrap();
        assert!((hy[0] / 1.414_213_562_373_095e200 - 1.0).abs() < 1e-12 && (hy[1] - 5.0).abs() < 1e-12);
        // fmod / remainder gradients: d/da = 1, d/db = -trunc(a/b) = 3 (fmod) or -floor(a/b) = 4 (remainder) for a = -7, b = 2
        for (rem, want) in [(false, 3.0f32), (true, 4.0f32)] {
            let (p, q) = (Tensor::from_slice(&[-7.0f32], &[1]).unwrap().requires_grad_(true), Tensor::from_slice(&[2.0f32], &[1]).unwrap().requires_grad_(true));
            let y = if rem { p.remainder(&q) } else { p.fmod(&q) }.unwrap();
            y.sum().unwrap().backward().unwrap();
            assert_eq!(p.grad().unwrap().to_vec::<f32>().unwrap(), vec![1.0]);
            assert_eq!(q.grad().unwrap().to_vec::<f32>().unwrap(), vec![want]);
        }
        // log_softmax keeps precision for huge logits
        let big = f(&[1e300, 1e300], &[2]).log_softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert!((big[0] + std::f64::consts::LN_2).abs() < 1e-12);
        assert!(f(&[1.0, 2.0], &[2]).isclose(&f(&[1.0 + 1e-9, 2.1], &[2]), 1e-5, 1e-8, false).unwrap().to_vec::<bool>().unwrap() == vec![true, false]);
        assert!(f(&[1.0], &[1]).equal(&f(&[1.0], &[1])).unwrap());
        let one = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
        let next = one.nextafter(&Tensor::from_slice(&[2.0f32], &[1]).unwrap()).unwrap().to_vec::<f32>().unwrap()[0];
        assert_eq!(next, 1.0f32 + f32::EPSILON);
        assert_eq!(ivals(&i(&[12, -18, 0], &[3]).gcd(&i(&[18, 12, 5], &[3])).unwrap()), vec![6, 6, 5]);
        assert_eq!(ivals(&i(&[4, 6], &[2]).lcm(&i(&[6, 4], &[2])).unwrap()), vec![12, 12]);
        assert_eq!(ivals(&i(&[1, -8], &[2]).bitwise_left_shift(&i(&[3, 1], &[2])).unwrap()), vec![8, -16]);
        assert_eq!(ivals(&i(&[16, -16], &[2]).bitwise_right_shift(&i(&[2, 2], &[2])).unwrap()), vec![4, -4]);
    }

    #[test]
    fn logaddexp_gradient_is_softmax() {
        let a = f(&[0.3], &[1]).requires_grad_(true);
        let b = f(&[1.2], &[1]).requires_grad_(true);
        a.logaddexp(&b).unwrap().sum().unwrap().backward().unwrap();
        let (ga, gb) = (vals(&a.grad().unwrap())[0], vals(&b.grad().unwrap())[0]);
        let s = 1.0 / (1.0 + (1.2f64 - 0.3).exp());
        assert!((ga - s).abs() < 1e-9 && (gb - (1.0 - s)).abs() < 1e-9);
    }

    #[test]
    fn reductions_and_shapes() {
        let x = f(&[1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3]);
        close(&x.amax(&[0, 1], false).unwrap(), &[6.0]);
        close(&x.amin(&[1], true).unwrap(), &[1.0, 2.0]);
        let (s, m) = x.std_mean(1, true, false).unwrap();
        close(&m, &[3.0, 4.0]);
        close(&s, &[2.0, 2.0]);
        close(&f(&[2.0, f64::NAN, 3.0], &[3]).nanprod(0, false).unwrap(), &[6.0]);
        close(&f(&[1.0, 2.0, 3.0], &[3]).cumulative_trapezoid(0, 1.0).unwrap(), &[1.5, 4.0]);
        close(&f(&[1.0, 2.0, 3.0], &[3]).cumulative_trapezoid_x(0, &f(&[0.0, 1.0, 3.0], &[3])).unwrap(), &[1.5, 6.5]);
        // rot90 of [[1,2],[3,4]]
        let m = f(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        close(&m.rot90(1, [0, 1]).unwrap(), &[2.0, 4.0, 1.0, 3.0]);
        close(&m.rot90(2, [0, 1]).unwrap(), &[4.0, 3.0, 2.0, 1.0]);
        close(&m.rot90(-1, [0, 1]).unwrap(), &[3.0, 1.0, 4.0, 2.0]);
        close(&m.take(&i(&[3, -4], &[2])).unwrap(), &[4.0, 1.0]);
        close(&m.put(&i(&[0, 3], &[2]), &f(&[9.0, 8.0], &[2]), false).unwrap(), &[9.0, 2.0, 3.0, 8.0]);
        close(&m.put(&i(&[0, 0], &[2]), &f(&[1.0, 1.0], &[2]), true).unwrap(), &[3.0, 2.0, 3.0, 4.0]);
        let idx = i(&[1, 0], &[2, 1]);
        close(&m.take_along_dim(&idx, 1).unwrap(), &[2.0, 3.0]);
    }

    #[test]
    fn index_writers() {
        let base = f(&[0.0; 6], &[3, 2]);
        let out = base.index_put(&[i(&[0, 2], &[2])], &f(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), false).unwrap();
        close(&out, &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        let acc = base.index_put(&[i(&[1, 1], &[2]), i(&[0, 0], &[2])], &f(&[5.0], &[1]), true).unwrap();
        close(&acc, &[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]);
        let dup = base.index_put(&[i(&[1, 1], &[2])], &f(&[1.0, 2.0], &[2, 1]).broadcast_to(&[2, 2]).unwrap(), false).unwrap();
        close(&dup, &[0.0, 0.0, 2.0, 2.0, 0.0, 0.0]); // the last write wins
        close(&base.index_copy(0, &i(&[2], &[1]), &f(&[7.0, 8.0], &[1, 2])).unwrap(), &[0.0, 0.0, 0.0, 0.0, 7.0, 8.0]);
        close(&base.select_scatter(&f(&[1.0, 2.0], &[2]), 0, 1).unwrap(), &[0.0, 0.0, 1.0, 2.0, 0.0, 0.0]);
        close(&base.slice_scatter(&f(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), 0, 0, 3, 2).unwrap(), &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        let sq = f(&[0.0; 9], &[3, 3]);
        close(&sq.diagonal_scatter(&f(&[1.0, 2.0, 3.0], &[3]), 0, 0, 1).unwrap(), &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        close(&sq.diagonal_scatter(&f(&[5.0, 6.0], &[2]), 1, 0, 1).unwrap(), &[0.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn index_writer_gradients() {
        let base = f(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad_(true);
        let v = f(&[10.0, 20.0], &[1, 2]).requires_grad_(true);
        let out = base.index_copy(0, &i(&[0], &[1]), &v).unwrap();
        out.mul(&f(&[1.0, 2.0, 3.0, 4.0], &[2, 2])).unwrap().sum().unwrap().backward().unwrap();
        close(&base.grad().unwrap(), &[0.0, 0.0, 3.0, 4.0]); // overwritten row gets no gradient
        close(&v.grad().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn scatter_reduce_modes() {
        let base = f(&[1.0, 2.0, 3.0], &[3]);
        let idx = i(&[0, 0, 2], &[3]);
        let src = f(&[10.0, 20.0, 30.0], &[3]);
        close(&base.scatter_reduce(0, &idx, &src, "sum", true).unwrap(), &[31.0, 2.0, 33.0]);
        close(&base.scatter_reduce(0, &idx, &src, "sum", false).unwrap(), &[30.0, 2.0, 30.0]); // untouched slots keep self, as in torch
        close(&base.scatter_reduce(0, &idx, &src, "mean", false).unwrap(), &[15.0, 2.0, 30.0]);
        close(&base.scatter_reduce(0, &idx, &src, "mean", true).unwrap(), &[31.0 / 3.0, 2.0, 16.5]);
        close(&base.scatter_reduce(0, &idx, &src, "prod", true).unwrap(), &[200.0, 2.0, 90.0]);
        close(&base.scatter_reduce(0, &idx, &src, "amax", false).unwrap(), &[20.0, 2.0, 30.0]);
        close(&base.scatter_reduce(0, &idx, &src, "amin", true).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn searching_and_sets() {
        let x = f(&[0.0, 2.0, 0.0, 3.0, 4.0, 0.0], &[2, 3]);
        let nz = x.nonzero().unwrap();
        assert_eq!(nz.shape(), [3, 2]);
        assert_eq!(ivals(&nz), vec![0, 1, 1, 0, 1, 1]);
        let (u, inv, cnt) = f(&[3.0, 1.0, 3.0, 2.0, 1.0], &[5]).unique_with().unwrap();
        close(&u, &[1.0, 2.0, 3.0]);
        assert_eq!(ivals(&inv), vec![2, 0, 2, 1, 0]);
        assert_eq!(ivals(&cnt), vec![2, 1, 2]);
        let (u, inv, cnt) = f(&[1.0, 1.0, 2.0, 1.0], &[4]).unique_consecutive().unwrap();
        close(&u, &[1.0, 2.0, 1.0]);
        assert_eq!(ivals(&inv), vec![0, 0, 1, 2]);
        assert_eq!(ivals(&cnt), vec![2, 1, 1]);
        let b = f(&[1.0, 3.0, 5.0], &[3]);
        assert_eq!(ivals(&f(&[0.0, 1.0, 3.0, 6.0], &[4]).bucketize(&b, false).unwrap()), vec![0, 0, 1, 3]);
        assert_eq!(ivals(&f(&[0.0, 1.0, 3.0, 6.0], &[4]).bucketize(&b, true).unwrap()), vec![0, 1, 2, 3]);
        assert_eq!(ivals(&Tensor::tril_indices(3, 3, 0).unwrap()), vec![0, 1, 1, 2, 2, 2, 0, 0, 1, 0, 1, 2]);
        assert_eq!(ivals(&Tensor::triu_indices(3, 3, 1).unwrap()), vec![0, 0, 1, 1, 2, 2]);
        close(&f(&[1.0, 2.0, 3.0], &[3]).combinations(2, false).unwrap(), &[1.0, 2.0, 1.0, 3.0, 2.0, 3.0]);
        close(&f(&[1.0, 2.0], &[2]).combinations(2, true).unwrap(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        close(&Tensor::logspace(0.0, 2.0, 3, 10.0, DType::F64).unwrap(), &[1.0, 10.0, 100.0]);
    }

    #[test]
    fn matrix_products() {
        let a = f(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = f(&[0.0, 1.0, 1.0, 0.0], &[2, 2]);
        close(&a.mm(&b).unwrap(), &[2.0, 1.0, 4.0, 3.0]);
        close(&a.mv(&f(&[1.0, 1.0], &[2])).unwrap(), &[3.0, 7.0]);
        close(&f(&[1.0, 2.0], &[2]).vdot(&f(&[3.0, 4.0], &[2])).unwrap(), &[11.0]);
        close(&a.inner(&b).unwrap(), &[2.0, 1.0, 4.0, 3.0]);
        close(&a.addmm(&a, &b, 0.5, 2.0).unwrap(), &[4.5, 3.0, 9.5, 8.0]);
        close(&f(&[1.0, 1.0], &[2]).addmv(&a, &f(&[1.0, 0.0], &[2]), 1.0, 1.0).unwrap(), &[2.0, 4.0]);
        close(&a.addr(&f(&[1.0, 2.0], &[2]), &f(&[1.0, 1.0], &[2]), 1.0, 1.0).unwrap(), &[2.0, 3.0, 5.0, 6.0]);
        let batch = f(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[2, 2, 2]);
        close(&batch.bmm(&batch).unwrap(), &[1.0, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 4.0]);
        close(&multi_dot(&[a.clone(), b.clone(), a.clone()]).unwrap(), &vals(&a.mm(&b).unwrap().mm(&a).unwrap()));
        close(&multi_dot(&[f(&[1.0, 2.0], &[2]), a.clone(), f(&[1.0, 1.0], &[2])]).unwrap(), &[17.0]);
        let rn = f(&[3.0, 4.0, 0.3, 0.4], &[2, 2]).renorm(2.0, 0, 1.0).unwrap();
        for (g, w) in vals(&rn).iter().zip([0.6, 0.8, 0.3, 0.4]) {
            assert!((g - w).abs() < 1e-6);
        }
        close(&f(&[0.0, 0.0, 3.0, 4.0, 6.0, 8.0], &[3, 2]).pdist(2.0).unwrap(), &[5.0, 10.0, 5.0]);
        close(&f(&[1.0, 2.0], &[2]).pairwise_distance(&f(&[4.0, 6.0], &[2]), 2.0, 0.0, false).unwrap(), &[5.0]);
        close(&f(&[0.0, 0.0], &[2]).softmin(0).unwrap(), &[0.5, 0.5]);
    }
}
