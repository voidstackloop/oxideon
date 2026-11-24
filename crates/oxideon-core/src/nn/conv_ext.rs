//! Convolution and pooling pieces the layer set lacked: grouped / depthwise [`conv1d_grouped`],
//! [`max_pool2d_with_indices`] with its inverse [`max_unpool2d`], [`adaptive_max_pool`] for any number of spatial
//! dimensions, and the [`Fold`] / [`Unfold`] modules (`torch.nn.Fold`, `torch.nn.Unfold`).

use super::Module;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// 1-D convolution with `groups` (`groups == in_channels` is depthwise). `x` is `[N, C, L]`, `weight`
/// `[O, C / groups, K]`.
pub fn conv1d_grouped(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    if x.ndim() != 3 || weight.ndim() != 3 {
        return Err(Error::Msg("conv1d_grouped: expects x [N, C, L] and weight [O, C/groups, K]".into()));
    }
    let y = x.unsqueeze(2)?.conv2d_grouped(&weight.unsqueeze(2)?, bias, (1, stride), (0, padding), (1, dilation), groups)?;
    y.squeeze_axis(2)
}

/// Max pooling that also returns, for every output element, the flat index (`row * W + col`) of its maximum within
/// the input plane (`torch.nn.functional.max_pool2d(return_indices=True)`). The values are differentiable.
pub fn max_pool2d_with_indices(
    x: &Tensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<(Tensor, Tensor)> {
    if x.ndim() != 4 {
        return Err(Error::Msg("max_pool2d_with_indices: expects [N, C, H, W]".into()));
    }
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    if kernel.0 > h + 2 * padding.0 || kernel.1 > w + 2 * padding.1 || stride.0 == 0 || stride.1 == 0 {
        return Err(Error::Msg("max_pool2d_with_indices: the window does not fit the padded input".into()));
    }
    let oh = (h + 2 * padding.0 - kernel.0) / stride.0 + 1;
    let ow = (w + 2 * padding.1 - kernel.1) / stride.1 + 1;
    let data = x.to_vec::<f64>()?;
    let mut idx = Vec::with_capacity(n * c * oh * ow);
    for plane in 0..n * c {
        let base = plane * h * w;
        for i in 0..oh {
            for j in 0..ow {
                let mut best = (f64::NEG_INFINITY, usize::MAX);
                for ki in 0..kernel.0 {
                    for kj in 0..kernel.1 {
                        let (r, col) = ((i * stride.0 + ki) as isize - padding.0 as isize, (j * stride.1 + kj) as isize - padding.1 as isize);
                        if r < 0 || col < 0 || r >= h as isize || col >= w as isize {
                            continue;
                        }
                        let flat = r as usize * w + col as usize;
                        let v = data[base + flat];
                        if v > best.0 || best.1 == usize::MAX || v.is_nan() {
                            best = (v, flat);
                        }
                    }
                }
                idx.push(best.1 as i64);
            }
        }
    }
    let indices = Tensor::from_slice_on(&idx, &[n, c, oh, ow], &x.device())?;
    let values = x.reshape(&[n, c, h * w])?.gather(2, &indices.reshape(&[n, c, oh * ow])?)?.reshape(&[n, c, oh, ow])?;
    Ok((values, indices))
}

/// Puts the pooled `values` back at their `indices` in a zero tensor of `output_size = (H, W)`
/// (`torch.nn.functional.max_unpool2d`).
pub fn max_unpool2d(values: &Tensor, indices: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
    if values.ndim() != 4 || values.shape() != indices.shape() {
        return Err(Error::Msg("max_unpool2d: values and indices must be [N, C, H', W'] of the same shape".into()));
    }
    let (n, c) = (values.shape()[0], values.shape()[1]);
    let plane = output_size.0 * output_size.1;
    if indices.to_vec::<i64>()?.iter().any(|&i| i < 0 || i as usize >= plane) {
        return Err(Error::Msg("max_unpool2d: an index lies outside the output plane".into()));
    }
    let flat_v = values.reshape(&[n, c, values.numel() / (n * c).max(1)])?;
    let flat_i = indices.reshape(&[n, c, indices.numel() / (n * c).max(1)])?;
    // assign, not add: with overlapping windows one input element can be the maximum of several windows
    Tensor::zeros(&[n, c, plane], values.dtype(), &values.device())?
        .scatter(2, &flat_i, &flat_v)?
        .reshape(&[n, c, output_size.0, output_size.1])
}

/// Adaptive max pooling of `[N, C, spatial...]` to `output_size` (`None` keeps a dimension). Windows follow PyTorch:
/// `[floor(i * in / out), ceil((i + 1) * in / out))`.
pub fn adaptive_max_pool(x: &Tensor, output_size: &[Option<usize>]) -> Result<Tensor> {
    if x.ndim() < 3 || output_size.len() != x.ndim() - 2 {
        return Err(Error::Msg("adaptive_max_pool: output_size needs one entry per spatial dimension".into()));
    }
    let mut y = x.clone();
    for (d, want) in output_size.iter().enumerate() {
        let axis = (d + 2) as isize;
        let n_in = y.shape()[d + 2];
        let n_out = want.unwrap_or(n_in);
        if n_out == 0 {
            return Err(Error::Msg("adaptive_max_pool: output size must be positive".into()));
        }
        let parts: Vec<Tensor> = (0..n_out)
            .map(|o| {
                let (start, end) = ((o * n_in) / n_out, ((o + 1) * n_in).div_ceil(n_out));
                y.narrow(axis, start, end - start)?.max_axis(axis, true)
            })
            .collect::<Result<_>>()?;
        y = Tensor::cat(&parts, axis)?;
    }
    Ok(y)
}

/// Extracts sliding local blocks: `[N, C, H, W] -> [N, C * kh * kw, L]`.
pub struct Unfold {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Module for Unfold {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.unfold2d(self.kernel, self.stride, self.padding, self.dilation)
    }
}

/// Sums sliding blocks back into an image: `[N, C * kh * kw, L] -> [N, C, H, W]`.
pub struct Fold {
    pub output_size: (usize, usize),
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Module for Fold {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.fold2d(self.output_size, self.kernel, self.stride, self.padding, self.dilation)
    }
}

/// Output length of one pooled dimension (`torch.nn.functional.max_pool*d`), with `ceil_mode`'s rule that the last
/// window must start inside the input or its left padding.
fn pooled_len(len: usize, kernel: usize, stride: usize, pad: usize, dil: usize, ceil_mode: bool) -> Result<usize> {
    let span = (len + 2 * pad) as isize - (dil * (kernel - 1)) as isize - 1;
    if span < 0 {
        return Err(Error::Msg(format!(
            "max_pool: the window ({kernel} x dilation {dil}) does not fit the padded input ({len} + 2 x {pad})"
        )));
    }
    let span = span as usize;
    let mut out = if ceil_mode { span.div_ceil(stride) } else { span / stride } + 1;
    if ceil_mode && (out - 1) * stride >= len + pad {
        out -= 1;
    }
    Ok(out)
}

/// Finds, for every output position of every plane, the flat spatial index of its window's maximum (the first
/// maximum; a NaN is taken over any number, as the PyTorch CPU kernel does), then gathers the values from `x`, so they
/// stay differentiable. `windows[o]` lists the in-bounds flat input indices of output position `o`.
fn pool_by_windows(x: &Tensor, nd: usize, out_shape: &[usize], windows: Vec<Vec<usize>>) -> Result<(Tensor, Tensor)> {
    let shape = x.shape().to_vec();
    let lead = &shape[..shape.len() - nd];
    let planes: usize = lead.iter().product();
    let plane_len: usize = shape[shape.len() - nd..].iter().product();
    let n_out: usize = out_shape.iter().product();
    if windows.iter().any(Vec::is_empty) {
        return Err(Error::Msg("max_pool: a pooling window lies entirely in the padding".into()));
    }
    let data = x.to_vec::<f64>()?;
    let mut idx = vec![0i64; planes * n_out];
    let body = |plane: usize, out: &mut [i64]| {
        let base = plane * plane_len;
        for (o, slot) in out.iter_mut().enumerate() {
            let mut best = f64::NEG_INFINITY;
            let mut at = windows[o][0];
            for &flat in &windows[o] {
                let v = data[base + flat];
                if v > best || v.is_nan() {
                    best = v;
                    at = flat;
                }
            }
            *slot = at as i64;
        }
    };
    if n_out > 0 {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            idx.par_chunks_mut(n_out).enumerate().for_each(|(p, out)| body(p, out));
        }
        #[cfg(not(feature = "rayon"))]
        {
            for (p, out) in idx.chunks_mut(n_out).enumerate() {
                body(p, out);
            }
        }
    }
    let mut full = lead.to_vec();
    full.extend_from_slice(out_shape);
    let indices = Tensor::from_slice_on(&idx, &[planes, n_out], &x.device())?;
    let values = x.reshape(&[planes, plane_len])?.gather(1, &indices)?.reshape(&full)?;
    Ok((values, indices.reshape(&full)?))
}

/// Splits a flat output position into per-dimension coordinates (row-major).
fn unravel(mut o: usize, dims: &[usize]) -> Vec<usize> {
    let mut coord = vec![0usize; dims.len()];
    for d in (0..dims.len()).rev() {
        coord[d] = o % dims[d];
        o /= dims[d];
    }
    coord
}

/// Max pooling over the last `kernel.len()` (1 to 3) dimensions with stride, padding, dilation and `ceil_mode`,
/// returning the values (differentiable) and the indices of the maxima in the flattened spatial input, like
/// `torch.nn.functional.max_pool{1,2,3}d(..., return_indices=True)`.
pub fn max_pool_nd_with_indices(
    x: &Tensor,
    kernel: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    ceil_mode: bool,
) -> Result<(Tensor, Tensor)> {
    let nd = kernel.len();
    if !(1..=3).contains(&nd) || stride.len() != nd || padding.len() != nd || dilation.len() != nd {
        return Err(Error::Msg(
            "max_pool: kernel, stride, padding and dilation need one entry per spatial dimension (1 to 3)".into(),
        ));
    }
    if x.ndim() < nd + 1 {
        return Err(Error::Msg(format!("max_pool{nd}d: expected at least {} dimensions, got {:?}", nd + 1, x.shape())));
    }
    if kernel.iter().chain(stride).chain(dilation).any(|&v| v == 0) {
        return Err(Error::Msg("max_pool: kernel_size, stride and dilation must be positive".into()));
    }
    if padding.iter().zip(kernel).any(|(&p, &k)| p > k / 2) {
        return Err(Error::Msg(format!(
            "max_pool{nd}d: padding should be at most half of the kernel size, got padding {padding:?} and kernel {kernel:?}"
        )));
    }
    let spatial: Vec<usize> = x.shape()[x.ndim() - nd..].to_vec();
    let out: Vec<usize> = (0..nd)
        .map(|d| pooled_len(spatial[d], kernel[d], stride[d], padding[d], dilation[d], ceil_mode))
        .collect::<Result<_>>()?;
    let slots: usize = kernel.iter().product();
    let n_out: usize = out.iter().product();
    let windows = (0..n_out)
        .map(|o| {
            let coord = unravel(o, &out);
            (0..slots)
                .filter_map(|slot| {
                    let offs = unravel(slot, kernel);
                    let mut flat = 0usize;
                    for d in 0..nd {
                        let p = (coord[d] * stride[d] + offs[d] * dilation[d]) as isize - padding[d] as isize;
                        if p < 0 || p >= spatial[d] as isize {
                            return None;
                        }
                        flat = flat * spatial[d] + p as usize;
                    }
                    Some(flat)
                })
                .collect()
        })
        .collect();
    pool_by_windows(x, nd, &out, windows)
}

/// Adaptive max pooling over the last `output_size.len()` (1 to 3) dimensions with the indices of the maxima, like
/// `torch.nn.functional.adaptive_max_pool{1,2,3}d(..., return_indices=True)`: output position `o` of a dimension of
/// length `L` covers `[floor(o L / out), ceil((o + 1) L / out))`.
pub fn adaptive_max_pool_nd_with_indices(x: &Tensor, output_size: &[usize]) -> Result<(Tensor, Tensor)> {
    let nd = output_size.len();
    if !(1..=3).contains(&nd) || x.ndim() < nd + 1 {
        return Err(Error::Msg(
            "adaptive_max_pool: output_size needs 1 to 3 entries and the input one more dimension".into(),
        ));
    }
    if output_size.contains(&0) {
        return Err(Error::Msg("adaptive_max_pool: output_size must be positive".into()));
    }
    let spatial: Vec<usize> = x.shape()[x.ndim() - nd..].to_vec();
    let n_out: usize = output_size.iter().product();
    let windows = (0..n_out)
        .map(|o| {
            let coord = unravel(o, output_size);
            let mut list = vec![0usize];
            for d in 0..nd {
                let (s, e) = (
                    (coord[d] * spatial[d]) / output_size[d],
                    ((coord[d] + 1) * spatial[d]).div_ceil(output_size[d]),
                );
                let len = spatial[d];
                list = list.iter().flat_map(|&base| (s..e).map(move |p| base * len + p)).collect();
            }
            list
        })
        .collect();
    pool_by_windows(x, nd, output_size, windows)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn vals(t: &Tensor) -> Vec<f64> {
        t.to_vec::<f64>().unwrap()
    }

    #[test]
    fn unpool_assigns_when_windows_overlap() {
        // the element 5 is the maximum of both overlapping windows; PyTorch writes it once (not 10)
        let x = Tensor::from_slice(&[1.0f64, 5.0, 2.0], &[1, 1, 1, 3]).unwrap();
        let (v, i) = max_pool2d_with_indices(&x, (1, 2), (1, 1), (0, 0)).unwrap();
        assert_eq!(vals(&max_unpool2d(&v, &i, (1, 3)).unwrap()), vec![0.0, 5.0, 0.0]);
    }

    #[test]
    fn grouped_and_depthwise_1d_convolution() {
        let x = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[1, 2, 4]).unwrap();
        // depthwise: each channel convolved with its own [1, 1] kernel
        let w = Tensor::from_slice(&[1.0f64, 1.0, 0.5, 0.5], &[2, 1, 2]).unwrap();
        let y = conv1d_grouped(&x, &w, None, 1, 0, 1, 2).unwrap();
        assert_eq!(vals(&y), vec![3.0, 5.0, 7.0, 15.0, 25.0, 35.0]);
        // groups = 1 matches the ordinary convolution
        let w1 = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], &[2, 2, 2]).unwrap();
        let a = conv1d_grouped(&x, &w1, None, 1, 1, 1, 1).unwrap();
        let b = x.conv1d(&w1, None, 1, 1, 1).unwrap();
        assert_eq!(vals(&a), vals(&b));
        assert!(conv1d_grouped(&x, &w, None, 1, 0, 1, 3).is_err());
    }

    #[test]
    fn pooling_with_indices_round_trips() {
        let x = Tensor::from_slice(&[1.0f64, 3.0, 2.0, 0.0, 4.0, 2.0, 1.0, 5.0, 0.0, 1.0, 7.0, 2.0, 3.0, 2.0, 1.0, 6.0], &[1, 1, 4, 4]).unwrap().requires_grad_(true);
        let (v, i) = max_pool2d_with_indices(&x, (2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(vals(&v), vec![4.0, 5.0, 3.0, 7.0]);
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![4, 7, 12, 10]);
        let reference = x.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(vals(&reference), vals(&v));
        let up = max_unpool2d(&v, &i, (4, 4)).unwrap();
        let u = vals(&up);
        assert_eq!((u[4], u[7], u[12], u[10]), (4.0, 5.0, 3.0, 7.0));
        assert_eq!(u.iter().filter(|&&a| a != 0.0).count(), 4);
        // gradient flows only to the winning positions
        v.sum().unwrap().backward().unwrap();
        let g = vals(&x.grad().unwrap());
        assert_eq!(g.iter().sum::<f64>(), 4.0);
        assert_eq!((g[4], g[7], g[12], g[10]), (1.0, 1.0, 1.0, 1.0));
        // padded windows ignore the padding
        let (pv, _) = max_pool2d_with_indices(&Tensor::from_slice(&[-1.0f64, -2.0, -3.0, -4.0], &[1, 1, 2, 2]).unwrap(), (3, 3), (1, 1), (1, 1)).unwrap();
        assert!(vals(&pv).iter().all(|&a| a == -1.0));
        assert!(max_unpool2d(&v, &Tensor::from_slice(&[99i64, 0, 0, 0], &[1, 1, 2, 2]).unwrap(), (4, 4)).is_err());
    }

    #[test]
    fn adaptive_max_and_fold_unfold() {
        let x = Tensor::from_slice(&[1.0f64, 5.0, 2.0, 4.0, 3.0], &[1, 1, 5]).unwrap();
        assert_eq!(vals(&adaptive_max_pool(&x, &[Some(2)]).unwrap()), vec![5.0, 4.0]); // windows [0,3) and [2,5)
        assert_eq!(vals(&adaptive_max_pool(&x, &[Some(5)]).unwrap()), vals(&x));
        let v = Tensor::from_slice(&(0..8).map(f64::from).collect::<Vec<_>>(), &[1, 1, 2, 2, 2]).unwrap();
        assert_eq!(vals(&adaptive_max_pool(&v, &[Some(1), Some(1), Some(1)]).unwrap()), vec![7.0]);
        // unfold then fold with non-overlapping blocks restores the image
        let img = Tensor::from_slice(&(0..16).map(f64::from).collect::<Vec<_>>(), &[1, 1, 4, 4]).unwrap();
        let unfold = Unfold { kernel: (2, 2), stride: (2, 2), padding: (0, 0), dilation: (1, 1) };
        let cols = unfold.forward(&img).unwrap();
        assert_eq!(cols.shape(), [1, 4, 4]);
        let fold = Fold { output_size: (4, 4), kernel: (2, 2), stride: (2, 2), padding: (0, 0), dilation: (1, 1) };
        assert_eq!(vals(&fold.forward(&cols).unwrap()), vals(&img));
    }
}

#[cfg(test)]
mod max_pool_nd_tests {
    use super::*;

    #[test]
    fn agrees_with_the_2d_kernel() {
        let data: Vec<f32> = (0..2 * 3 * 7 * 6).map(|i| ((i * 37 % 101) as f32) * 0.1 - 5.0).collect();
        let x = Tensor::from_slice(&data, &[2, 3, 7, 6]).unwrap();
        let (v, i) = max_pool_nd_with_indices(&x, &[3, 2], &[2, 2], &[1, 0], &[1, 1], false).unwrap();
        let (v2, i2) = max_pool2d_with_indices(&x, (3, 2), (2, 2), (1, 0)).unwrap();
        assert_eq!(i.to_vec::<i64>().unwrap(), i2.to_vec::<i64>().unwrap());
        assert_eq!(v.to_vec::<f32>().unwrap(), v2.to_vec::<f32>().unwrap());
    }

    #[test]
    fn ceil_mode_dilation_and_adaptive() {
        let x = Tensor::from_slice(&(0..11).map(|v| v as f32).collect::<Vec<_>>(), &[1, 1, 11]).unwrap();
        // dilation 2, padding 1, stride 2, ceil_mode: windows {-1,1,3} {1,3,5} {3,5,7} {5,7,9} {7,9,11}; 11 is past the end
        let (v, i) = max_pool_nd_with_indices(&x, &[3], &[2], &[1], &[2], true).unwrap();
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![3, 5, 7, 9, 9]);
        assert_eq!(v.to_vec::<f32>().unwrap(), vec![3.0, 5.0, 7.0, 9.0, 9.0]);
        let (a, ai) = adaptive_max_pool_nd_with_indices(&x, &[4]).unwrap();
        assert_eq!(ai.to_vec::<i64>().unwrap(), vec![2, 5, 8, 10]);
        assert_eq!(a.shape(), &[1, 1, 4]);
        assert!(max_pool_nd_with_indices(&x, &[3], &[1], &[2], &[1], false).is_err()); // padding > kernel / 2
    }
}
