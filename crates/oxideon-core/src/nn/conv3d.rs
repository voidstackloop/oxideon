//! 3D Volumetric Convolutions (`Conv3d`), 3D Pooling (`MaxPool3d`, `AvgPool3d`),
//! `BatchNorm3d`, and Sub-Pixel Super-Resolution Layers (`PixelShuffle`, `PixelUnshuffle`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::backend::{CpuStorage, Device, Storage};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::nn::{Module, init};
use crate::tensor::Tensor;

impl Tensor {
    /// 3D volumetric convolution (`torch.nn.functional.conv3d`) over 5D input `[N, C_in, D, H, W]`
    /// with kernel `[C_out, C_in, kD, kH, kW]` and optional bias `[C_out]`.
    /// Equipped with exact analytical backward pass for `input`, `weight`, and `bias`.
    pub fn conv3d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 5 || weight.ndim() != 5 {
            return Err(Error::Msg(
                "conv3d requires 5D input [N, C_in, D, H, W] and 5D weight [C_out, C_in, kD, kH, kW]"
                    .into(),
            ));
        }
        let (n, c_in, d_in, h_in, w_in) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
            self.shape()[4],
        );
        let (c_out, wc_in, kd, kh, kw) = (
            weight.shape()[0],
            weight.shape()[1],
            weight.shape()[2],
            weight.shape()[3],
            weight.shape()[4],
        );
        if c_in != wc_in {
            return Err(Error::Msg(format!(
                "conv3d channel mismatch: input C_in={c_in} vs weight C_in={wc_in}"
            )));
        }
        let (sd, sh, sw) = stride;
        let (pd, ph, pw) = padding;
        if sd == 0 || sh == 0 || sw == 0 {
            return Err(Error::Msg("conv3d stride must be >= 1".into()));
        }
        // All caller-controlled size arithmetic is checked: padding/stride are bare usizes.
        let out_dim = |n_in: usize, p: usize, k: usize, s: usize, axis: &str| -> Result<usize> {
            let padded = p
                .checked_mul(2)
                .and_then(|t| n_in.checked_add(t))
                .ok_or_else(|| Error::Msg(format!("conv3d padding overflows {axis}")))?;
            let span = padded
                .checked_sub(k)
                .ok_or_else(|| Error::Msg(format!("conv3d kernel {axis} exceeds padded {axis}")))?;
            span.checked_div(s)
                .and_then(|q| q.checked_add(1))
                .ok_or_else(|| Error::Msg(format!("conv3d output {axis} overflows")))
        };
        let d_out = out_dim(d_in, pd, kd, sd, "D")?;
        let h_out = out_dim(h_in, ph, kh, sh, "H")?;
        let w_out = out_dim(w_in, pw, kw, sw, "W")?;

        let x_vec = self.to_vec::<f64>()?;
        let w_vec = weight.to_vec::<f64>()?;
        let b_vec = match bias {
            Some(b) => {
                if b.shape() != [c_out] {
                    return Err(Error::Msg("conv3d bias must have shape [C_out]".into()));
                }
                Some(b.to_vec::<f64>()?)
            }
            None => None,
        };

        let out_spatial = crate::layout::checked_numel(&[d_out, h_out, w_out])
            .map_err(|_| Error::Msg("conv3d output size overflows".into()))?;
        let out_numel = crate::limits::check_elements("conv3d output", &[n, c_out, out_spatial])?;
        let mut out = crate::layout::try_filled(out_numel, 0.0f64)?;

        let compute_channel = |(nc_idx, out_slice): (usize, &mut [f64])| {
            let b_idx = nc_idx / c_out;
            let co = nc_idx % c_out;
            let b_val = b_vec.as_ref().map_or(0.0, |bv| bv[co]);
            // Window positions in usize with saturation: the old `as isize` differences
            // wrapped or overflowed for huge stride/padding. Saturation can only push an
            // already-out-of-range position further out, so the bounds checks stay exact.
            for od in 0..d_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut acc = b_val;
                        for ci in 0..c_in {
                            let x_base = ((b_idx * c_in + ci) * d_in) * h_in * w_in;
                            let w_base = ((co * c_in + ci) * kd) * kh * kw;
                            for fd in 0..kd {
                                let raw_d = od.saturating_mul(sd).saturating_add(fd);
                                if raw_d < pd || raw_d - pd >= d_in {
                                    continue;
                                }
                                let id = raw_d - pd;
                                for fh in 0..kh {
                                    let raw_h = oh.saturating_mul(sh).saturating_add(fh);
                                    if raw_h < ph || raw_h - ph >= h_in {
                                        continue;
                                    }
                                    let ih = raw_h - ph;
                                    for fw in 0..kw {
                                        let raw_w = ow.saturating_mul(sw).saturating_add(fw);
                                        if raw_w < pw || raw_w - pw >= w_in {
                                            continue;
                                        }
                                        let iw = raw_w - pw;
                                        let xv = x_vec[x_base + (id * h_in + ih) * w_in + iw];
                                        let wv = w_vec[w_base + (fd * kh + fh) * kw + fw];
                                        acc += xv * wv;
                                    }
                                }
                            }
                        }
                        out_slice[(od * h_out + oh) * w_out + ow] = acc;
                    }
                }
            }
        };

        #[cfg(feature = "rayon")]
        out.par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(compute_channel);
        #[cfg(not(feature = "rayon"))]
        out.chunks_mut(out_spatial)
            .enumerate()
            .for_each(compute_channel);

        let out_shape = vec![n, c_out, d_out, h_out, w_out];
        let out_layout = Layout::contiguous(out_shape);
        let cpu = CpuStorage::F64(out).to_dtype(&out_layout, self.dtype())?;
        let storage = Storage::new_cpu(cpu).to_device(&out_layout, &self.device())?;

        let mut inputs = vec![self.clone(), weight.clone()];
        let has_bias = bias.is_some();
        if let Some(b) = bias {
            inputs.push(b.clone());
        }
        let dtype = self.dtype();
        let device = self.device();
        let x_shape = self.shape().to_vec();
        let w_shape = weight.shape().to_vec();

        Ok(Tensor::from_op(
            storage,
            out_layout,
            "conv3d",
            inputs,
            move |g| {
                let g_vec = g.to_vec::<f64>()?;
                let mut gx = crate::layout::try_filled(n * c_in * d_in * h_in * w_in, 0.0f64)?;
                let mut gw = crate::layout::try_filled(c_out * c_in * kd * kh * kw, 0.0f64)?;
                let mut gb = crate::layout::try_filled(c_out, 0.0f64)?;

                for b_idx in 0..n {
                    for (co, gb_co) in gb.iter_mut().enumerate().take(c_out) {
                        let g_nc_base = ((b_idx * c_out + co) * d_out) * h_out * w_out;
                        for od in 0..d_out {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let gy = g_vec[g_nc_base + (od * h_out + oh) * w_out + ow];
                                    *gb_co += gy;
                                    for ci in 0..c_in {
                                        let x_base = ((b_idx * c_in + ci) * d_in) * h_in * w_in;
                                        let w_base = ((co * c_in + ci) * kd) * kh * kw;
                                        for fd in 0..kd {
                                            let raw_d = od.saturating_mul(sd).saturating_add(fd);
                                            if raw_d < pd || raw_d - pd >= d_in {
                                                continue;
                                            }
                                            let id = raw_d - pd;
                                            for fh in 0..kh {
                                                let raw_h =
                                                    oh.saturating_mul(sh).saturating_add(fh);
                                                if raw_h < ph || raw_h - ph >= h_in {
                                                    continue;
                                                }
                                                let ih = raw_h - ph;
                                                for fw in 0..kw {
                                                    let raw_w =
                                                        ow.saturating_mul(sw).saturating_add(fw);
                                                    if raw_w < pw || raw_w - pw >= w_in {
                                                        continue;
                                                    }
                                                    let iw = raw_w - pw;
                                                    let x_idx =
                                                        x_base + (id * h_in + ih) * w_in + iw;
                                                    let w_idx = w_base + (fd * kh + fh) * kw + fw;
                                                    gx[x_idx] += gy * w_vec[w_idx];
                                                    gw[w_idx] += gy * x_vec[x_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let gx_t = Tensor::from_slice_on(&gx, &x_shape, &device)?.to_dtype(dtype)?;
                let gw_t = Tensor::from_slice_on(&gw, &w_shape, &device)?.to_dtype(dtype)?;
                if has_bias {
                    let gb_t = Tensor::from_slice_on(&gb, &[c_out], &device)?.to_dtype(dtype)?;
                    Ok(vec![Some(gx_t), Some(gw_t), Some(gb_t)])
                } else {
                    Ok(vec![Some(gx_t), Some(gw_t)])
                }
            },
        ))
    }

    /// 3D Transposed Convolution (`torch.nn.functional.conv_transpose3d`) over 5D input `[N, C_in, D_in, H_in, W_in]`
    /// with kernel `[C_in, C_out, kD, kH, kW]` and optional bias `[C_out]`.
    /// Analytical autograd backward pass for `input`, `weight`, and `bias`.
    pub fn conv_transpose3d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 5 || weight.ndim() != 5 {
            return Err(Error::Msg(
                "conv_transpose3d requires 5D input [N, C_in, D, H, W] and 5D weight [C_in, C_out, kD, kH, kW]"
                    .into(),
            ));
        }
        let (n, c_in, d_in, h_in, w_in) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
            self.shape()[4],
        );
        let (wc_in, c_out, kd, kh, kw) = (
            weight.shape()[0],
            weight.shape()[1],
            weight.shape()[2],
            weight.shape()[3],
            weight.shape()[4],
        );
        if c_in != wc_in {
            return Err(Error::Msg(format!(
                "conv_transpose3d channel mismatch: input C_in={c_in} vs weight C_in={wc_in}"
            )));
        }
        let (sd, sh, sw) = stride;
        let (pd, ph, pw) = padding;
        let (opd, oph, opw) = output_padding;
        let (dd, dh, dw) = dilation;
        if sd == 0 || sh == 0 || sw == 0 {
            return Err(Error::Msg("conv_transpose3d stride must be >= 1".into()));
        }
        // Effective kernel: dd * (kd - 1) + 1, fully checked (kd = 0 underflowed before).
        let eff = |k: usize, d: usize, axis: &str| -> Result<usize> {
            k.checked_sub(1)
                .and_then(|m| d.checked_mul(m))
                .and_then(|e| e.checked_add(1))
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "conv_transpose3d effective kernel {axis} overflows"
                    ))
                })
        };
        let eff_kd = eff(kd, dd, "D")?;
        let eff_kh = eff(kh, dh, "H")?;
        let eff_kw = eff(kw, dw, "W")?;

        // d_out = (d_in - 1) * sd + eff_kd + opd - 2 * pd, fully checked; a zero
        // output dim is invalid (there is nothing to convolve into).
        let out_dim = |n_in: usize,
                       s: usize,
                       eff_k: usize,
                       op: usize,
                       p: usize,
                       axis: &str|
         -> Result<usize> {
            let span = n_in
                .checked_sub(1)
                .and_then(|m| m.checked_mul(s))
                .and_then(|a| a.checked_add(eff_k))
                .and_then(|a| a.checked_add(op))
                .ok_or_else(|| Error::Msg(format!("conv_transpose3d output {axis} overflows")))?;
            let out = span
                .checked_sub(p.checked_mul(2).ok_or_else(|| {
                    Error::Msg(format!("conv_transpose3d padding overflows {axis}"))
                })?)
                .ok_or_else(|| {
                    Error::Msg(format!("conv_transpose3d invalid output {axis} dimension"))
                })?;
            if out == 0 {
                return Err(Error::Msg(format!(
                    "conv_transpose3d invalid output {axis} dimension"
                )));
            }
            Ok(out)
        };
        let d_out = out_dim(d_in, sd, eff_kd, opd, pd, "D")?;
        let h_out = out_dim(h_in, sh, eff_kh, oph, ph, "H")?;
        let w_out = out_dim(w_in, sw, eff_kw, opw, pw, "W")?;

        let x_vec = self.to_vec::<f64>()?;
        let w_vec = weight.to_vec::<f64>()?;
        let b_vec = match bias {
            Some(b) => {
                if b.shape() != [c_out] {
                    return Err(Error::Msg(
                        "conv_transpose3d bias must have shape [C_out]".into(),
                    ));
                }
                Some(b.to_vec::<f64>()?)
            }
            None => None,
        };

        let out_spatial = crate::layout::checked_numel(&[d_out, h_out, w_out])
            .map_err(|_| Error::Msg("conv_transpose3d output size overflows".into()))?;
        let out_numel =
            crate::limits::check_elements("conv_transpose3d output", &[n, c_out, out_spatial])?;
        let mut out = crate::layout::try_filled(out_numel, 0.0f64)?;

        let compute_channel = |(nc_idx, out_slice): (usize, &mut [f64])| {
            let b_idx = nc_idx / c_out;
            let co = nc_idx % c_out;
            let b_val = b_vec.as_ref().map_or(0.0, |bv| bv[co]);
            out_slice.fill(b_val);

            for ci in 0..c_in {
                let x_base = ((b_idx * c_in + ci) * d_in) * h_in * w_in;
                let w_base = ((ci * c_out + co) * kd) * kh * kw;
                for id in 0..d_in {
                    for ih in 0..h_in {
                        for iw in 0..w_in {
                            let xv = x_vec[x_base + (id * h_in + ih) * w_in + iw];
                            if xv == 0.0 {
                                continue;
                            }
                            for fd in 0..kd {
                                let raw =
                                    id.saturating_mul(sd).saturating_add(fd.saturating_mul(dd));
                                if raw < pd || raw - pd >= d_out {
                                    continue;
                                }
                                let od = raw - pd;
                                for fh in 0..kh {
                                    let raw =
                                        ih.saturating_mul(sh).saturating_add(fh.saturating_mul(dh));
                                    if raw < ph || raw - ph >= h_out {
                                        continue;
                                    }
                                    let oh = raw - ph;
                                    for fw in 0..kw {
                                        let raw = iw
                                            .saturating_mul(sw)
                                            .saturating_add(fw.saturating_mul(dw));
                                        if raw < pw || raw - pw >= w_out {
                                            continue;
                                        }
                                        let ow = raw - pw;
                                        let wv = w_vec[w_base + (fd * kh + fh) * kw + fw];
                                        out_slice[(od * h_out + oh) * w_out + ow] += xv * wv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        #[cfg(feature = "rayon")]
        out.par_chunks_mut(out_spatial)
            .enumerate()
            .for_each(compute_channel);
        #[cfg(not(feature = "rayon"))]
        out.chunks_mut(out_spatial)
            .enumerate()
            .for_each(compute_channel);

        let out_shape = vec![n, c_out, d_out, h_out, w_out];
        let out_layout = Layout::contiguous(out_shape);
        let cpu = CpuStorage::F64(out).to_dtype(&out_layout, self.dtype())?;
        let storage = Storage::new_cpu(cpu).to_device(&out_layout, &self.device())?;

        let mut inputs = vec![self.clone(), weight.clone()];
        let has_bias = bias.is_some();
        if let Some(b) = bias {
            inputs.push(b.clone());
        }
        let dtype = self.dtype();
        let device = self.device();
        let x_shape = self.shape().to_vec();
        let w_shape = weight.shape().to_vec();

        Ok(Tensor::from_op(
            storage,
            out_layout,
            "conv_transpose3d",
            inputs,
            move |g| {
                let g_vec = g.to_vec::<f64>()?;
                let mut gx = crate::layout::try_filled(n * c_in * d_in * h_in * w_in, 0.0f64)?;
                let mut gw = crate::layout::try_filled(c_in * c_out * kd * kh * kw, 0.0f64)?;
                let mut gb = crate::layout::try_filled(c_out, 0.0f64)?;

                for b_idx in 0..n {
                    for (co, gb_co) in gb.iter_mut().enumerate().take(c_out) {
                        let g_nc_base = ((b_idx * c_out + co) * d_out) * h_out * w_out;
                        for od in 0..d_out {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let gy = g_vec[g_nc_base + (od * h_out + oh) * w_out + ow];
                                    *gb_co += gy;
                                    for ci in 0..c_in {
                                        let x_base = ((b_idx * c_in + ci) * d_in) * h_in * w_in;
                                        let w_base = ((ci * c_out + co) * kd) * kh * kw;
                                        for fd in 0..kd {
                                            // usize form of `od + pd - fd*dd` with the stride-divisibility
                                            // test; saturation keeps huge args out of range, never in.
                                            let base_d = od.saturating_add(pd);
                                            let back_d = fd.saturating_mul(dd);
                                            if base_d < back_d {
                                                continue;
                                            }
                                            let shifted_d = base_d - back_d;
                                            if shifted_d % sd != 0 {
                                                continue;
                                            }
                                            let id = shifted_d / sd;
                                            if id >= d_in {
                                                continue;
                                            }
                                            for fh in 0..kh {
                                                let base_h = oh.saturating_add(ph);
                                                let back_h = fh.saturating_mul(dh);
                                                if base_h < back_h {
                                                    continue;
                                                }
                                                let shifted_h = base_h - back_h;
                                                if shifted_h % sh != 0 {
                                                    continue;
                                                }
                                                let ih = shifted_h / sh;
                                                if ih >= h_in {
                                                    continue;
                                                }
                                                for fw in 0..kw {
                                                    let base_w = ow.saturating_add(pw);
                                                    let back_w = fw.saturating_mul(dw);
                                                    if base_w < back_w {
                                                        continue;
                                                    }
                                                    let shifted_w = base_w - back_w;
                                                    if shifted_w % sw != 0 {
                                                        continue;
                                                    }
                                                    let iw = shifted_w / sw;
                                                    if iw >= w_in {
                                                        continue;
                                                    }
                                                    let x_idx =
                                                        x_base + (id * h_in + ih) * w_in + iw;
                                                    let w_idx = w_base + (fd * kh + fh) * kw + fw;
                                                    gx[x_idx] += gy * w_vec[w_idx];
                                                    gw[w_idx] += gy * x_vec[x_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let gx_t = Tensor::from_slice_on(&gx, &x_shape, &device)?.to_dtype(dtype)?;
                let gw_t = Tensor::from_slice_on(&gw, &w_shape, &device)?.to_dtype(dtype)?;
                if has_bias {
                    let gb_t = Tensor::from_slice_on(&gb, &[c_out], &device)?.to_dtype(dtype)?;
                    Ok(vec![Some(gx_t), Some(gw_t), Some(gb_t)])
                } else {
                    Ok(vec![Some(gx_t), Some(gw_t)])
                }
            },
        ))
    }
}

/// 3D Volumetric Convolution module (`torch.nn.Conv3d`).
pub struct Conv3d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
}

impl Conv3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        use_bias: bool,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let (kd, kh, kw) = kernel_size;
        let fan_in = in_channels
            .checked_mul(kd)
            .and_then(|v| v.checked_mul(kh))
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "Conv3d: fan_in (in_channels {in_channels} * kernel {kd}x{kh}x{kw}) overflows"
                ))
            })?;
        let weight = init::kaiming_uniform(&[out_channels, in_channels, kd, kh, kw], fan_in)?
            .to_dtype(dtype)?
            .to_device(&device)?
            .requires_grad_(true);
        let bias = if use_bias {
            let bound = 1.0 / (fan_in as f32).sqrt();
            Some(
                init::uniform(&[out_channels], -bound, bound)?
                    .to_dtype(dtype)?
                    .to_device(&device)?
                    .requires_grad_(true),
            )
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
        })
    }
}

impl Module for Conv3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv3d(&self.weight, self.bias.as_ref(), self.stride, self.padding)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }
}

/// 3D Transposed Volumetric Convolution module (`torch.nn.ConvTranspose3d`).
pub struct ConvTranspose3d {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
    pub output_padding: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
}

impl ConvTranspose3d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        use_bias: bool,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let (kd, kh, kw) = kernel_size;
        let fan_in = in_channels
            .checked_mul(kd)
            .and_then(|v| v.checked_mul(kh))
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "ConvTranspose3d: fan_in (in_channels {in_channels} * kernel {kd}x{kh}x{kw}) overflows"
                ))
            })?;
        if fan_in == 0 {
            return Err(Error::Msg(
                "ConvTranspose3d: fan_in (in_channels * kernel) must be positive".into(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();
        let weight = init::uniform(&[in_channels, out_channels, kd, kh, kw], -bound, bound)?
            .to_dtype(dtype)?
            .to_device(&device)?
            .requires_grad_(true);
        let bias = if use_bias {
            Some(
                init::uniform(&[out_channels], -bound, bound)?
                    .to_dtype(dtype)?
                    .to_device(&device)?
                    .requires_grad_(true),
            )
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation: (1, 1, 1),
        })
    }
}

impl Module for ConvTranspose3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.conv_transpose3d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b.clone()));
        }
        params
    }
}

/// 3D Average Pooling (`torch.nn.AvgPool3d`) implemented via zero-copy 3D `unfold` windows + `mean`.
pub struct AvgPool3d {
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
}

impl AvgPool3d {
    pub fn new(kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }
}

impl Module for AvgPool3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        // Unfold D (dim 2), H (dim 3), W (dim 4) -> [N, C, d_out, h_out, w_out, kd, kh, kw]
        let u = input
            .unfold(2, kd, sd)?
            .unfold(3, kh, sh)?
            .unfold(4, kw, sw)?;
        u.mean_axes(&[5, 6, 7], false)
    }
}

/// 3D Max Pooling (`torch.nn.MaxPool3d`) over 5D tensors `[N, C, D, H, W]`.
pub struct MaxPool3d {
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
}

impl MaxPool3d {
    pub fn new(kernel_size: (usize, usize, usize), stride: (usize, usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }
}

impl Module for MaxPool3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let u = input
            .unfold(2, kd, sd)?
            .unfold(3, kh, sh)?
            .unfold(4, kw, sw)?
            .contiguous()?;
        let s = u.shape().to_vec();
        let flat = u.reshape(&[s[0], s[1], s[2], s[3], s[4], kd * kh * kw])?;
        flat.max_axis(5, false)
    }
}

/// 3D Batch Normalization (`torch.nn.BatchNorm3d`) over `[N, C, D, H, W]`.
pub struct BatchNorm3d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Arc<RwLock<Tensor>>,
    pub running_var: Arc<RwLock<Tensor>>,
    pub momentum: f64,
    pub eps: f64,
    training: AtomicBool,
}

impl BatchNorm3d {
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let weight = Tensor::ones(&[num_features], dtype, &device)?.requires_grad_(true);
        let bias = Tensor::zeros(&[num_features], dtype, &device)?.requires_grad_(true);
        let running_mean = Tensor::zeros(&[num_features], dtype, &device)?;
        let running_var = Tensor::ones(&[num_features], dtype, &device)?;
        Ok(Self {
            weight,
            bias,
            running_mean: Arc::new(RwLock::new(running_mean)),
            running_var: Arc::new(RwLock::new(running_var)),
            momentum,
            eps,
            training: AtomicBool::new(true),
        })
    }
}

impl Module for BatchNorm3d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 5 {
            return Err(Error::Msg(
                "BatchNorm3d requires 5D input [N, C, D, H, W]".into(),
            ));
        }
        let c = input.shape()[1];
        let (mean, var) = if self.training.load(Ordering::Relaxed) {
            let m = input.mean_axes(&[0, 2, 3, 4], false)?;
            let m_view = m.reshape(&[1, c, 1, 1, 1])?;
            let diff = input.sub(&m_view)?;
            let v = diff.square()?.mean_axes(&[0, 2, 3, 4], false)?;
            // the running estimate uses the unbiased variance (n / (n - 1)), as PyTorch and BatchNorm1d/2d do
            let count = (input.numel() / c.max(1)) as f64;
            let unbiased = if count > 1.0 { count / (count - 1.0) } else { 1.0 };
            if let (Ok(mut rm), Ok(mut rv)) = (self.running_mean.write(), self.running_var.write())
            {
                let new_rm = rm
                    .mul_scalar(1.0 - self.momentum)?
                    .add(&m.detach().mul_scalar(self.momentum)?)?;
                let new_rv = rv
                    .mul_scalar(1.0 - self.momentum)?
                    .add(&v.detach().mul_scalar(self.momentum * unbiased)?)?;
                *rm = new_rm;
                *rv = new_rv;
            }
            (m, v)
        } else {
            let rm = self
                .running_mean
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let rv = self
                .running_var
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            (rm, rv)
        };
        let m_view = mean.reshape(&[1, c, 1, 1, 1])?;
        let v_view = var.reshape(&[1, c, 1, 1, 1])?;
        let w_view = self.weight.reshape(&[1, c, 1, 1, 1])?;
        let b_view = self.bias.reshape(&[1, c, 1, 1, 1])?;
        let x_hat = input
            .sub(&m_view)?
            .div(&v_view.add_scalar(self.eps)?.sqrt()?)?;
        x_hat.mul(&w_view)?.add(&b_view)
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.weight.clone()),
            ("bias".to_string(), self.bias.clone()),
        ]
    }

    fn named_buffers(&self) -> Vec<(String, Tensor)> {
        let (Ok(mean), Ok(var)) = (self.running_mean.read(), self.running_var.read()) else {
            return Vec::new();
        };
        vec![
            ("running_mean".to_string(), mean.clone()),
            ("running_var".to_string(), var.clone()),
        ]
    }

    fn train(&self, mode: bool) {
        self.training.store(mode, Ordering::Relaxed);
    }
}

/// Rearranges elements in a tensor of shape `[N, C * r^2, H, W]` to `[N, C, H * r, W * r]`
/// for sub-pixel super-resolution (`torch.nn.PixelShuffle`).
pub struct PixelShuffle {
    pub upscale_factor: usize,
}

impl PixelShuffle {
    pub fn new(upscale_factor: usize) -> Self {
        Self { upscale_factor }
    }
}

impl Module for PixelShuffle {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 4 {
            return Err(Error::Msg(
                "PixelShuffle requires 4D input [N, C*r^2, H, W]".into(),
            ));
        }
        let r = self.upscale_factor;
        if r == 0 {
            return Err(Error::Msg(
                "PixelShuffle upscale_factor must be >= 1".into(),
            ));
        }
        let r2 = r
            .checked_mul(r)
            .ok_or_else(|| Error::Msg(format!("PixelShuffle r^2 overflows for r={r}")))?;
        let (n, c_in, h, w) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        if c_in % r2 != 0 {
            return Err(Error::Msg(format!(
                "PixelShuffle input channels {c_in} must be divisible by r^2 = {r2}"
            )));
        }
        let c_out = c_in / r2;
        let (hr, wr) = (
            h.checked_mul(r)
                .ok_or_else(|| Error::Msg("PixelShuffle output height overflows".into()))?,
            w.checked_mul(r)
                .ok_or_else(|| Error::Msg("PixelShuffle output width overflows".into()))?,
        );
        input
            .contiguous()?
            .reshape(&[n, c_out, r, r, h, w])?
            .permute(&[0, 1, 4, 2, 5, 3])?
            .contiguous()?
            .reshape(&[n, c_out, hr, wr])
    }
}

/// Inverse of [`PixelShuffle`]: rearranges `[N, C, H * r, W * r]` into `[N, C * r^2, H, W]`
/// (`torch.nn.PixelUnshuffle`).
pub struct PixelUnshuffle {
    pub downscale_factor: usize,
}

impl PixelUnshuffle {
    pub fn new(downscale_factor: usize) -> Self {
        Self { downscale_factor }
    }
}

impl Module for PixelUnshuffle {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if input.ndim() != 4 {
            return Err(Error::Msg(
                "PixelUnshuffle requires 4D input [N, C, H*r, W*r]".into(),
            ));
        }
        let r = self.downscale_factor;
        if r == 0 {
            return Err(Error::Msg(
                "PixelUnshuffle downscale_factor must be >= 1".into(),
            ));
        }
        let (n, c, hr, wr) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
            input.shape()[3],
        );
        if hr % r != 0 || wr % r != 0 {
            return Err(Error::Msg(format!(
                "PixelUnshuffle spatial dims ({hr}, {wr}) must be divisible by r={r}"
            )));
        }
        let (h, w) = (hr / r, wr / r);
        let cc = c
            .checked_mul(r)
            .and_then(|v| v.checked_mul(r))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "PixelUnshuffle channels {c} * r^2 overflows for r={r}"
                ))
            })?;
        input
            .contiguous()?
            .reshape(&[n, c, h, r, w, r])?
            .permute(&[0, 1, 3, 5, 2, 4])?
            .contiguous()?
            .reshape(&[n, cc, h, w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv3d_huge_stride_and_padding_do_not_panic() -> Result<()> {
        // od=1 computes raw index 2^63: the old `as isize` difference underflowed (debug panic).
        let x = Tensor::zeros_f32(&[1, 1, 1, 1, 1])?;
        let w = Tensor::zeros_f32(&[1, 1, 1, 1, 1])?;
        let y = x.conv3d(&w, None, (1 << 63, 1, 1), (1 << 62, 0, 0))?;
        assert_eq!(y.shape(), &[1, 1, 2, 1, 1]);
        Ok(())
    }

    #[test]
    fn conv_transpose3d_rejects_zero_output_dim() -> Result<()> {
        // d_out = (2-1)*1+1+0-2*1 = 0: chunks_mut(0) panicked pre-fix.
        let x = Tensor::zeros_f32(&[1, 1, 2, 1, 1])?;
        let w = Tensor::zeros_f32(&[1, 1, 1, 1, 1])?;
        let err = x
            .conv_transpose3d(&w, None, (1, 1, 1), (1, 0, 0), (0, 0, 0), (1, 1, 1))
            .expect_err("zero output dim must be an error, not a panic");
        assert!(err.to_string().contains("output D"), "{err}");
        Ok(())
    }

    #[test]
    fn conv_transpose3d_rejects_empty_dims() -> Result<()> {
        // kd = 0: `kd - 1` underflowed pre-fix.
        let x = Tensor::zeros_f32(&[1, 1, 2, 1, 1])?;
        let w0 = Tensor::from_slice(&[] as &[f32], &[1, 1, 0, 1, 1])?;
        assert!(
            x.conv_transpose3d(&w0, None, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1))
                .is_err()
        );
        // d_in = 0: `d_in - 1` underflowed pre-fix.
        let x0 = Tensor::from_slice(&[] as &[f32], &[1, 1, 0, 1, 1])?;
        let w = Tensor::zeros_f32(&[1, 1, 1, 1, 1])?;
        assert!(
            x0.conv_transpose3d(&w, None, (1, 1, 1), (0, 0, 0), (0, 0, 0), (1, 1, 1))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn pixel_shuffle_rejects_degenerate_factors() -> Result<()> {
        let x = Tensor::zeros_f32(&[1, 4, 2, 2])?;
        // r = 0: `c_in % (r*r)` divided by zero pre-fix.
        assert!(PixelShuffle::new(0).forward(&x).is_err());
        // r = 2^40: `r * r` overflowed pre-fix.
        let mut sh = PixelShuffle::new(2);
        sh.upscale_factor = 1 << 40;
        assert!(sh.forward(&x).is_err());
        Ok(())
    }

    #[test]
    fn pixel_unshuffle_rejects_degenerate_factors() -> Result<()> {
        let x = Tensor::zeros_f32(&[1, 1, 4, 4])?;
        // r = 0: `hr % r` divided by zero pre-fix.
        assert!(PixelUnshuffle::new(0).forward(&x).is_err());
        // r = 2^40 with empty spatial dims (0 % r == 0 passes the check):
        // `c * r * r` overflowed pre-fix.
        let x0 = Tensor::from_slice(&[] as &[f32], &[1, 1, 0, 0])?;
        let mut sh = PixelUnshuffle::new(2);
        sh.downscale_factor = 1 << 40;
        assert!(sh.forward(&x0).is_err());
        Ok(())
    }
}
