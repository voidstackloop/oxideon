//! Differentiable 2D spatial vision & object detection primitives:
//! - **Spatial Transformer Networks**: [`affine_grid`] and [`grid_sample`] (bilinear interpolation with exact analytical backward pass w.r.t. both `input` and `grid`!)
//! - **Region-of-Interest Pooling**: [`roi_align`] (Mask R-CNN differentiable sub-pixel RoI Align)
//! - **Object Detection Geometry**: [`box_iou`] and [`nms`] (Non-Maximum Suppression)

use crate::backend::{CpuStorage, Storage};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

/// Generates a 2D flow-field sampling grid of shape `[N, H, W, 2]` in normalized $[-1, 1]^2$
/// coordinates from a batch of affine transformation matrices `theta` of shape `[N, 2, 3]`.
/// Fully differentiable with respect to `theta` via standard tensor matrix multiplication!
pub fn affine_grid(theta: &Tensor, size: [usize; 4], align_corners: bool) -> Result<Tensor> {
    let [n, _c, h, w] = size;
    if theta.shape() != [n, 2, 3] {
        return Err(Error::ShapeMismatch {
            op: "affine_grid",
            expected: vec![n, 2, 3],
            got: theta.shape().to_vec(),
        });
    }
    // `h`/`w` are plain params: refuse overflowing/giant grids up front.
    crate::limits::check_elements("affine_grid", &[n, h, w, 2])?;

    let mut base_coords = Vec::with_capacity(h * w * 3);
    for iy in 0..h {
        let y = if align_corners {
            if h > 1 {
                -1.0 + 2.0 * (iy as f64) / ((h - 1) as f64)
            } else {
                0.0
            }
        } else {
            -1.0 + (2.0 * (iy as f64) + 1.0) / (h as f64)
        };
        for ix in 0..w {
            let x = if align_corners {
                if w > 1 {
                    -1.0 + 2.0 * (ix as f64) / ((w - 1) as f64)
                } else {
                    0.0
                }
            } else {
                -1.0 + (2.0 * (ix as f64) + 1.0) / (w as f64)
            };
            base_coords.push(x);
            base_coords.push(y);
            base_coords.push(1.0);
        }
    }

    // base_grid: [1, H*W, 3] -> broadcast to [N, H*W, 3], multiply by theta^T: [N, 3, 2] -> [N, H*W, 2]
    let base_grid = Tensor::from_slice_on(&base_coords, &[1, h * w, 3], &theta.device())?
        .to_dtype(theta.dtype())?
        .broadcast_to(&[n, h * w, 3])?;
    let theta_t = theta.transpose(1, 2)?;
    let grid_flat = base_grid.matmul(&theta_t)?;
    grid_flat.reshape(&[n, h, w, 2])
}

/// Floors a sampling coordinate to a corner index. `inf.floor() as isize`
/// saturates to `MAX`, and the `+ 1` for the east/south corners then
/// overflows — clamp first so out-of-range corners stay out of bounds.
#[inline]
fn corner_floor(v: f64) -> isize {
    const HI: f64 = 9_223_372_036_854_775_808.0; // 2^63
    const LO: f64 = -9_223_372_036_854_775_808.0; // -(2^63)
    let f = v.floor();
    // Strict `< HI`: `f as isize` saturates at MAX, and the east/south
    // corners add 1, so MAX itself is still one too many.
    if f.is_finite() && (LO..HI).contains(&f) {
        f as isize
    } else if f.is_sign_negative() {
        isize::MIN
    } else {
        // +inf and NaN land here: out of bounds, sampling zeros. (NaN used
        // to convert to 0 and poison the pixel with NaN weights.)
        isize::MAX - 1
    }
}

#[inline]
fn unnormalize_coord(coord: f64, size: usize, align_corners: bool) -> (f64, f64) {
    if align_corners {
        if size <= 1 {
            (0.0, 0.0)
        } else {
            let s = (size - 1) as f64;
            (((coord + 1.0) * 0.5) * s, 0.5 * s)
        }
    } else {
        let s = size as f64;
        (((coord + 1.0) * s - 1.0) * 0.5, 0.5 * s)
    }
}

/// Differentiable 2D Bilinear Spatial Sampler (`grid_sample`):
/// Given `input` of shape `[N, C, H_in, W_in]` and normalized `grid` of shape `[N, H_out, W_out, 2]`
/// (where `grid[n, h, w] = [x, y]` in $[-1, 1]^2$ with `(-1, -1)` at top-left), samples `input`
/// using bilinear interpolation (zeros padding outside bounds).
/// Differentiable with respect to **both** `input` and `grid`!
pub fn grid_sample(input: &Tensor, grid: &Tensor, align_corners: bool) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(Error::Msg(format!(
            "grid_sample: input must be 4D [N, C, H, W], got {:?}",
            input.shape()
        )));
    }
    let [n, c, h_in, w_in] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];
    if grid.ndim() != 4 || grid.shape()[0] != n || grid.shape()[3] != 2 {
        return Err(Error::ShapeMismatch {
            op: "grid_sample",
            expected: vec![n, h_in, w_in, 2],
            got: grid.shape().to_vec(),
        });
    }
    let h_out = grid.shape()[1];
    let w_out = grid.shape()[2];

    let in_vec = input.to_vec::<f64>()?;
    let grid_vec = grid.to_vec::<f64>()?;
    // Input and grid sizes are independent real tensors: check the product.
    let out_len = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(h_out))
        .and_then(|v| v.checked_mul(w_out))
        .ok_or_else(|| Error::Msg("grid_sample output size overflows".into()))?;
    let mut out_vec = vec![0.0f64; out_len];

    let in_bounds = |y: isize, x: isize| -> bool {
        y >= 0 && (y as usize) < h_in && x >= 0 && (x as usize) < w_in
    };

    for b in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let g_base = ((b * h_out + oh) * w_out + ow) * 2;
                let gx = grid_vec[g_base];
                let gy = grid_vec[g_base + 1];

                let (ix, _) = unnormalize_coord(gx, w_in, align_corners);
                let (iy, _) = unnormalize_coord(gy, h_in, align_corners);

                let ix_nw = corner_floor(ix);
                let iy_nw = corner_floor(iy);
                let ix_ne = ix_nw + 1;
                let iy_ne = iy_nw;
                let ix_sw = ix_nw;
                let iy_sw = iy_nw + 1;
                let ix_se = ix_nw + 1;
                let iy_se = iy_nw + 1;

                let nw = ((ix_se as f64) - ix) * ((iy_se as f64) - iy);
                let ne = (ix - (ix_sw as f64)) * ((iy_sw as f64) - iy);
                let sw = ((ix_ne as f64) - ix) * (iy - (iy_ne as f64));
                let se = (ix - (ix_nw as f64)) * (iy - (iy_nw as f64));

                for ch in 0..c {
                    let chan_base = (b * c + ch) * (h_in * w_in);
                    let mut val = 0.0;
                    if in_bounds(iy_nw, ix_nw) {
                        val += in_vec[chan_base + (iy_nw as usize) * w_in + (ix_nw as usize)] * nw;
                    }
                    if in_bounds(iy_ne, ix_ne) {
                        val += in_vec[chan_base + (iy_ne as usize) * w_in + (ix_ne as usize)] * ne;
                    }
                    if in_bounds(iy_sw, ix_sw) {
                        val += in_vec[chan_base + (iy_sw as usize) * w_in + (ix_sw as usize)] * sw;
                    }
                    if in_bounds(iy_se, ix_se) {
                        val += in_vec[chan_base + (iy_se as usize) * w_in + (ix_se as usize)] * se;
                    }
                    out_vec[((b * c + ch) * h_out + oh) * w_out + ow] = val;
                }
            }
        }
    }

    let out_layout = Layout::contiguous(vec![n, c, h_out, w_out]);
    let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, input.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &input.device())?;

    let in_clone = input.clone();
    let grid_clone = grid.clone();
    Ok(Tensor::from_op(
        storage,
        out_layout,
        "grid_sample",
        vec![input.clone(), grid.clone()],
        move |grad_out| {
            let go_vec = grad_out.to_vec::<f64>()?;
            let in_v = in_clone.to_vec::<f64>()?;
            let gr_v = grid_clone.to_vec::<f64>()?;

            let mut grad_in = vec![0.0f64; n * c * h_in * w_in];
            let mut grad_grid = vec![0.0f64; n * h_out * w_out * 2];

            for b in 0..n {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let g_base = ((b * h_out + oh) * w_out + ow) * 2;
                        let gx = gr_v[g_base];
                        let gy = gr_v[g_base + 1];

                        let (ix, dix_dgx) = unnormalize_coord(gx, w_in, align_corners);
                        let (iy, diy_dgy) = unnormalize_coord(gy, h_in, align_corners);

                        let ix_nw = corner_floor(ix);
                        let iy_nw = corner_floor(iy);
                        let ix_ne = ix_nw + 1;
                        let iy_ne = iy_nw;
                        let ix_sw = ix_nw;
                        let iy_sw = iy_nw + 1;
                        let ix_se = ix_nw + 1;
                        let iy_se = iy_nw + 1;

                        let nw = ((ix_se as f64) - ix) * ((iy_se as f64) - iy);
                        let ne = (ix - (ix_sw as f64)) * ((iy_sw as f64) - iy);
                        let sw = ((ix_ne as f64) - ix) * (iy - (iy_ne as f64));
                        let se = (ix - (ix_nw as f64)) * (iy - (iy_nw as f64));

                        let mut d_ix = 0.0f64;
                        let mut d_iy = 0.0f64;

                        for ch in 0..c {
                            let go = go_vec[((b * c + ch) * h_out + oh) * w_out + ow];
                            let chan_base = (b * c + ch) * (h_in * w_in);

                            if iy_nw >= 0
                                && (iy_nw as usize) < h_in
                                && ix_nw >= 0
                                && (ix_nw as usize) < w_in
                            {
                                let idx = chan_base + (iy_nw as usize) * w_in + (ix_nw as usize);
                                let v_nw = in_v[idx];
                                grad_in[idx] += go * nw;
                                d_ix -= go * v_nw * ((iy_se as f64) - iy);
                                d_iy -= go * v_nw * ((ix_se as f64) - ix);
                            }
                            if iy_ne >= 0
                                && (iy_ne as usize) < h_in
                                && ix_ne >= 0
                                && (ix_ne as usize) < w_in
                            {
                                let idx = chan_base + (iy_ne as usize) * w_in + (ix_ne as usize);
                                let v_ne = in_v[idx];
                                grad_in[idx] += go * ne;
                                d_ix += go * v_ne * ((iy_sw as f64) - iy);
                                d_iy -= go * v_ne * (ix - (ix_sw as f64));
                            }
                            if iy_sw >= 0
                                && (iy_sw as usize) < h_in
                                && ix_sw >= 0
                                && (ix_sw as usize) < w_in
                            {
                                let idx = chan_base + (iy_sw as usize) * w_in + (ix_sw as usize);
                                let v_sw = in_v[idx];
                                grad_in[idx] += go * sw;
                                d_ix -= go * v_sw * (iy - (iy_ne as f64));
                                d_iy += go * v_sw * ((ix_ne as f64) - ix);
                            }
                            if iy_se >= 0
                                && (iy_se as usize) < h_in
                                && ix_se >= 0
                                && (ix_se as usize) < w_in
                            {
                                let idx = chan_base + (iy_se as usize) * w_in + (ix_se as usize);
                                let v_se = in_v[idx];
                                grad_in[idx] += go * se;
                                d_ix += go * v_se * (iy - (iy_nw as f64));
                                d_iy += go * v_se * (ix - (ix_nw as f64));
                            }
                        }

                        grad_grid[g_base] = d_ix * dix_dgx;
                        grad_grid[g_base + 1] = d_iy * diy_dgy;
                    }
                }
            }

            let gin_t = Tensor::from_slice_on(&grad_in, in_clone.shape(), &in_clone.device())?
                .to_dtype(in_clone.dtype())?;
            let ggrid_t =
                Tensor::from_slice_on(&grad_grid, grid_clone.shape(), &grid_clone.device())?
                    .to_dtype(grid_clone.dtype())?;
            Ok(vec![Some(gin_t), Some(ggrid_t)])
        },
    ))
}

/// Mask R-CNN differentiable **RoI Align** (`roi_align`, He et al. 2017):
/// Pools variable-sized regions `rois` (`[K, 5]` where each row is `[batch_idx, x1, y1, x2, y2]`)
/// from `input` (`[N, C, H, W]`) into fixed `[K, C, out_h, out_w]` feature maps using exact
/// sub-pixel bilinear interpolation without coordinate quantization.
pub fn roi_align(
    input: &Tensor,
    rois: &Tensor,
    output_size: [usize; 2],
    spatial_scale: f64,
    sampling_ratio: usize,
) -> Result<Tensor> {
    if input.ndim() != 4 {
        return Err(Error::Msg(format!(
            "roi_align expects 4D input [N, C, H, W], got {:?}",
            input.shape()
        )));
    }
    if rois.ndim() != 2 || rois.shape()[1] != 5 {
        return Err(Error::ShapeMismatch {
            op: "roi_align",
            expected: vec![rois.shape().first().copied().unwrap_or(0), 5],
            got: rois.shape().to_vec(),
        });
    }
    let [n, c, h_in, w_in] = [
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    ];
    let num_rois = rois.shape()[0];
    let [out_h, out_w] = output_size;
    // An empty pixel table still gets indexed below (corner 0 of nothing).
    if h_in == 0 || w_in == 0 {
        return Err(Error::Msg(format!(
            "roi_align needs positive spatial dims, got [{h_in}, {w_in}]"
        )));
    }
    if out_h == 0 || out_w == 0 {
        return Err(Error::Msg(format!(
            "roi_align output_size must be positive, got {output_size:?}"
        )));
    }
    crate::limits::check_elements("roi_align output", &[num_rois, c, out_h, out_w])?;
    let sr = crate::limits::check_repeats("roi_align sampling_ratio", sampling_ratio.max(1))?;

    let in_vec = input.to_vec::<f64>()?;
    let roi_vec = rois.to_vec::<f64>()?;
    // NaN batch indices would silently pool image 0 (`as usize` saturates)
    // and NaN coordinates would panic inside `clamp`.
    if roi_vec.iter().any(|v| v.is_nan()) {
        return Err(Error::Msg(
            "roi_align: roi coordinates must not be NaN".into(),
        ));
    }
    // Batch indices must name a real batch (torchvision errors; silently
    // clamping would pool (and backprop) the wrong image).
    for r in 0..num_rois {
        let b = roi_vec[r * 5].round();
        if b < 0.0 || b >= n as f64 {
            return Err(Error::IndexOutOfBounds {
                op: "roi_align",
                index: b as isize,
                dim_size: n,
            });
        }
    }
    let mut out_vec = vec![0.0f64; num_rois * c * out_h * out_w];

    for r in 0..num_rois {
        let b_idx = roi_vec[r * 5].round() as usize;
        let x1 = roi_vec[r * 5 + 1] * spatial_scale;
        let y1 = roi_vec[r * 5 + 2] * spatial_scale;
        let x2 = roi_vec[r * 5 + 3] * spatial_scale;
        let y2 = roi_vec[r * 5 + 4] * spatial_scale;

        let roi_w = (x2 - x1).max(1e-5);
        let roi_h = (y2 - y1).max(1e-5);
        let bin_w = roi_w / (out_w as f64);
        let bin_h = roi_h / (out_h as f64);

        for ph in 0..out_h {
            for pw in 0..out_w {
                for ch in 0..c {
                    let chan_base = (b_idx * c + ch) * (h_in * w_in);
                    let mut acc = 0.0f64;
                    for iy in 0..sr {
                        let y =
                            y1 + (ph as f64) * bin_h + ((iy as f64) + 0.5) * bin_h / (sr as f64);
                        for ix in 0..sr {
                            let x = x1
                                + (pw as f64) * bin_w
                                + ((ix as f64) + 0.5) * bin_w / (sr as f64);

                            let y_clamped = y.clamp(0.0, (h_in.saturating_sub(1)) as f64);
                            let x_clamped = x.clamp(0.0, (w_in.saturating_sub(1)) as f64);
                            let y_low = y_clamped.floor() as usize;
                            let x_low = x_clamped.floor() as usize;
                            let y_high = (y_low + 1).min(h_in.saturating_sub(1));
                            let x_high = (x_low + 1).min(w_in.saturating_sub(1));

                            let ly = y_clamped - (y_low as f64);
                            let lx = x_clamped - (x_low as f64);
                            let hy = 1.0 - ly;
                            let hx = 1.0 - lx;

                            let v1 = in_vec[chan_base + y_low * w_in + x_low];
                            let v2 = in_vec[chan_base + y_low * w_in + x_high];
                            let v3 = in_vec[chan_base + y_high * w_in + x_low];
                            let v4 = in_vec[chan_base + y_high * w_in + x_high];

                            acc += hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
                        }
                    }
                    out_vec[((r * c + ch) * out_h + ph) * out_w + pw] = acc / ((sr * sr) as f64);
                }
            }
        }
    }

    let out_layout = Layout::contiguous(vec![num_rois, c, out_h, out_w]);
    let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, input.dtype())?;
    let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &input.device())?;

    let in_clone = input.clone();
    Ok(Tensor::from_op(
        storage,
        out_layout,
        "roi_align",
        vec![input.clone()],
        move |grad_out| {
            let go_vec = grad_out.to_vec::<f64>()?;
            let mut grad_in = vec![0.0f64; n * c * h_in * w_in];
            let scale_count = 1.0 / ((sr * sr) as f64);

            for r in 0..num_rois {
                // Validated in the forward pass; `as usize` cannot wrap here.
                let b_idx = roi_vec[r * 5].round() as usize;
                let x1 = roi_vec[r * 5 + 1] * spatial_scale;
                let y1 = roi_vec[r * 5 + 2] * spatial_scale;
                let x2 = roi_vec[r * 5 + 3] * spatial_scale;
                let y2 = roi_vec[r * 5 + 4] * spatial_scale;

                let roi_w = (x2 - x1).max(1e-5);
                let roi_h = (y2 - y1).max(1e-5);
                let bin_w = roi_w / (out_w as f64);
                let bin_h = roi_h / (out_h as f64);

                for ph in 0..out_h {
                    for pw in 0..out_w {
                        for ch in 0..c {
                            let go = go_vec[((r * c + ch) * out_h + ph) * out_w + pw] * scale_count;
                            let chan_base = (b_idx * c + ch) * (h_in * w_in);
                            for iy in 0..sr {
                                let y = y1
                                    + (ph as f64) * bin_h
                                    + ((iy as f64) + 0.5) * bin_h / (sr as f64);
                                for ix in 0..sr {
                                    let x = x1
                                        + (pw as f64) * bin_w
                                        + ((ix as f64) + 0.5) * bin_w / (sr as f64);

                                    let y_clamped = y.clamp(0.0, (h_in.saturating_sub(1)) as f64);
                                    let x_clamped = x.clamp(0.0, (w_in.saturating_sub(1)) as f64);
                                    let y_low = y_clamped.floor() as usize;
                                    let x_low = x_clamped.floor() as usize;
                                    let y_high = (y_low + 1).min(h_in.saturating_sub(1));
                                    let x_high = (x_low + 1).min(w_in.saturating_sub(1));

                                    let ly = y_clamped - (y_low as f64);
                                    let lx = x_clamped - (x_low as f64);
                                    let hy = 1.0 - ly;
                                    let hx = 1.0 - lx;

                                    grad_in[chan_base + y_low * w_in + x_low] += go * hy * hx;
                                    grad_in[chan_base + y_low * w_in + x_high] += go * hy * lx;
                                    grad_in[chan_base + y_high * w_in + x_low] += go * ly * hx;
                                    grad_in[chan_base + y_high * w_in + x_high] += go * ly * lx;
                                }
                            }
                        }
                    }
                }
            }

            let gin = Tensor::from_slice_on(&grad_in, in_clone.shape(), &in_clone.device())?
                .to_dtype(in_clone.dtype())?;
            Ok(vec![Some(gin)])
        },
    ))
}

/// Computes the pairwise Intersection-over-Union (`IoU`) matrix of shape `[N, M]` between
/// `boxes1` (`[N, 4]`) and `boxes2` (`[M, 4]`) in `[x1, y1, x2, y2]` format.
pub fn box_iou(boxes1: &Tensor, boxes2: &Tensor) -> Result<Tensor> {
    if boxes1.ndim() != 2 || boxes1.shape()[1] != 4 || boxes2.ndim() != 2 || boxes2.shape()[1] != 4
    {
        return Err(Error::Msg(
            "box_iou expects 2D tensors of shape [N, 4] and [M, 4]".into(),
        ));
    }
    let n = boxes1.shape()[0];
    let m = boxes2.shape()[0];
    let b1 = boxes1.to_vec::<f64>()?;
    let b2 = boxes2.to_vec::<f64>()?;
    // Two independent real tensors: check the pairwise product.
    let nm = n
        .checked_mul(m)
        .ok_or_else(|| Error::Msg("box_iou output size overflows".into()))?;
    let mut ious = vec![0.0f64; nm];

    for i in 0..n {
        let (x1_a, y1_a, x2_a, y2_a) = (b1[i * 4], b1[i * 4 + 1], b1[i * 4 + 2], b1[i * 4 + 3]);
        let area_a = (x2_a - x1_a).max(0.0) * (y2_a - y1_a).max(0.0);
        for j in 0..m {
            let (x1_b, y1_b, x2_b, y2_b) = (b2[j * 4], b2[j * 4 + 1], b2[j * 4 + 2], b2[j * 4 + 3]);
            let area_b = (x2_b - x1_b).max(0.0) * (y2_b - y1_b).max(0.0);

            let inter_w = (x2_a.min(x2_b) - x1_a.max(x1_b)).max(0.0);
            let inter_h = (y2_a.min(y2_b) - y1_a.max(y1_b)).max(0.0);
            let inter = inter_w * inter_h;
            let union = area_a + area_b - inter;
            ious[i * m + j] = if union > 0.0 { inter / union } else { 0.0 };
        }
    }

    Tensor::from_slice_on(&ious, &[n, m], &boxes1.device())?.to_dtype(boxes1.dtype())
}

/// Non-Maximum Suppression (`nms`) for 2D bounding boxes `[N, 4]` (`[x1, y1, x2, y2]`)
/// with confidence `scores` (`[N]`), returning kept box indices in descending score order.
pub fn nms(boxes: &Tensor, scores: &Tensor, iou_threshold: f64) -> Result<Vec<usize>> {
    if boxes.ndim() != 2 || boxes.shape()[1] != 4 || scores.ndim() != 1 {
        return Err(Error::Msg("nms expects boxes [N, 4] and scores [N]".into()));
    }
    let n = boxes.shape()[0];
    if scores.shape()[0] != n {
        return Err(Error::ShapeMismatch {
            op: "nms",
            expected: vec![n],
            got: scores.shape().to_vec(),
        });
    }
    let b = boxes.to_vec::<f64>()?;
    let s = scores.to_vec::<f64>()?;

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &c| s[c].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut suppressed = vec![false; n];
    let mut keep = Vec::new();

    for idx_pos in 0..n {
        let i = order[idx_pos];
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        let (x1_a, y1_a, x2_a, y2_a) = (b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]);
        let area_a = (x2_a - x1_a).max(0.0) * (y2_a - y1_a).max(0.0);

        for &j in &order[(idx_pos + 1)..] {
            if suppressed[j] {
                continue;
            }
            let (x1_b, y1_b, x2_b, y2_b) = (b[j * 4], b[j * 4 + 1], b[j * 4 + 2], b[j * 4 + 3]);
            let area_b = (x2_b - x1_b).max(0.0) * (y2_b - y1_b).max(0.0);

            let inter_w = (x2_a.min(x2_b) - x1_a.max(x1_b)).max(0.0);
            let inter_h = (y2_a.min(y2_b) - y1_a.max(y1_b)).max(0.0);
            let inter = inter_w * inter_h;
            let union = area_a + area_b - inter;
            let iou = if union > 0.0 { inter / union } else { 0.0 };
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    Ok(keep)
}
