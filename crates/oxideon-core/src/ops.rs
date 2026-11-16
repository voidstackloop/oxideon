//! Differentiable tensor operations, zero-copy view transforms, reductions,
//! linear algebra, `einsum`, indexing/scatter/gather, and convolution/pooling.
//!
//! Because backward rules are written using [`Tensor`] methods, running
//! `backward_with_grad(..., true)` or `grad(..., true)` constructs a differentiable
//! backward graph for arbitrary higher-order derivatives.

use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

use crate::backend::{
    BinaryOp, BitOp, CmpOp, CpuStorage, Device, LogicOp, PredOp, ReduceOp, Storage, UnaryOp,
};
use crate::dtype::{DType, bf16, f16, f8e4m3, f8e5m2};
use crate::error::{Error, Result};
use crate::indexing::SliceElem;
use crate::layout::{
    DimMap, Layout, ViewSpec, broadcast_shapes, normalize_axis, normalize_insert_axis,
};
use crate::tensor::{Tensor, no_grad};

/// Interpolation method for `quantile`/`nanquantile` (torch's five spellings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantileMethod {
    Linear,
    Lower,
    Higher,
    Nearest,
    Midpoint,
}

impl QuantileMethod {
    fn parse(name: &str) -> Result<Self> {
        match name {
            "linear" => Ok(Self::Linear),
            "lower" => Ok(Self::Lower),
            "higher" => Ok(Self::Higher),
            "nearest" => Ok(Self::Nearest),
            "midpoint" => Ok(Self::Midpoint),
            _ => Err(Error::Msg(format!(
                "quantile() interpolation must be one of linear, lower, higher, nearest, midpoint (got {name:?})"
            ))),
        }
    }
}

/// Which member of the `fmod` / `remainder` / `floor_divide` family to evaluate.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DivKind {
    Fmod,
    Remainder,
    FloorDivide,
}

// Inherent `add`/`sub`/`mul`/`div`/`neg` mirror PyTorch's method API; std::ops traits
// are also implemented below (returning `Result<Tensor>`).
#[allow(clippy::should_implement_trait)]
/// Direct max pooling of a contiguous `f32` image batch `[planes, h, w]` into `[planes, out_h, out_w]`. Padding never
/// wins; a NaN in a window wins (it propagates, like the reduction this replaces); among equal values the first in
/// row-major window order is taken. `arg`, when not empty, receives the winner's position `ih * w + iw` in its plane.
fn max_pool_fill(src: &[f32], out: &mut [f32], arg: &mut [i64], g: &UnfoldGeom) {
    let (l, plane_len) = (g.out_h * g.out_w, g.h * g.w);
    if l == 0 || plane_len == 0 {
        return;
    }
    let body = |pi: usize, o: &mut [f32], a: &mut [i64]| {
        let plane = &src[pi * plane_len..(pi + 1) * plane_len];
        for oh in 0..g.out_h {
            let h0 = (oh * g.sh).saturating_sub(g.ph);
            let h1 = (oh * g.sh + g.kh).saturating_sub(g.ph).min(g.h);
            for ow in 0..g.out_w {
                let w0 = (ow * g.sw).saturating_sub(g.pw);
                let w1 = (ow * g.sw + g.kw).saturating_sub(g.pw).min(g.w);
                let (mut best, mut at) = (f32::NEG_INFINITY, h0 * g.w + w0);
                for ih in h0..h1 {
                    let row = &plane[ih * g.w + w0..ih * g.w + w1];
                    for (j, &v) in row.iter().enumerate() {
                        if v > best || (v.is_nan() && !best.is_nan()) {
                            best = v;
                            at = ih * g.w + w0 + j;
                        }
                    }
                }
                o[oh * g.out_w + ow] = best;
                if let Some(slot) = a.get_mut(oh * g.out_w + ow) {
                    *slot = at as i64;
                }
            }
        }
    };
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        if arg.is_empty() {
            out.par_chunks_mut(l).enumerate().for_each(|(pi, o)| body(pi, o, &mut []));
        } else {
            out.par_chunks_mut(l).zip(arg.par_chunks_mut(l)).enumerate().for_each(|(pi, (o, a))| body(pi, o, a));
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        let mut args = arg.chunks_mut(l);
        for (pi, o) in out.chunks_mut(l).enumerate() {
            body(pi, o, args.next().unwrap_or(&mut []));
        }
    }
}

/// Borrows the contiguous `f32` host data of `t`, or `None` when it is another dtype, a view, or on a device.
fn with_f32_host<R>(t: &Tensor, f: impl FnOnce(&[f32]) -> R) -> Result<Option<R>> {
    if t.dtype() != DType::F32 || !t.device().is_cpu() {
        return Ok(None);
    }
    let guard = t.storage().cpu_ref()?;
    let CpuStorage::F32(v) = &*guard else { return Ok(None) };
    Ok(t.layout().contiguous_range_in(v.len()).map(|r| f(&v[r])))
}

/// Window geometry of [`Tensor::unfold2d`].
struct UnfoldGeom {
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    out_h: usize,
    out_w: usize,
}

/// im2col: `cols` is `[N * C, kh * kw, out_h * out_w]` (zero-initialised, so padding stays zero) and `src` is
/// `[N * C, h, w]`. For each kernel tap the valid span of every output row is one slice copy (strided when `sw > 1`).
fn unfold_fill<T: Copy + Send + Sync>(src: &[T], cols: &mut [T], g: &UnfoldGeom) {
    let l = g.out_h * g.out_w;
    let per_ch = g.kh * g.kw * l;
    let plane_len = g.h * g.w;
    if per_ch == 0 || plane_len == 0 {
        return;
    }
    let body = |(bc, block): (usize, &mut [T])| {
        let plane = &src[bc * plane_len..(bc + 1) * plane_len];
        for r in 0..g.kh {
            for s in 0..g.kw {
                let dst = &mut block[(r * g.kw + s) * l..(r * g.kw + s + 1) * l];
                // output columns whose input column `ow * sw + s * dw - pw` lies inside the image
                let off = s * g.dw;
                let ow_lo = if off >= g.pw { 0 } else { (g.pw - off).div_ceil(g.sw) };
                let ow_hi = if g.w + g.pw > off { ((g.w - 1 + g.pw - off) / g.sw + 1).min(g.out_w) } else { 0 };
                if ow_lo >= ow_hi {
                    continue;
                }
                let first = ow_lo * g.sw + off - g.pw;
                for oh in 0..g.out_h {
                    let ih = (oh * g.sh + r * g.dh) as isize - g.ph as isize;
                    if ih < 0 || ih as usize >= g.h {
                        continue;
                    }
                    let srow = &plane[ih as usize * g.w..(ih as usize + 1) * g.w];
                    let drow = &mut dst[oh * g.out_w + ow_lo..oh * g.out_w + ow_hi];
                    if g.sw == 1 {
                        drow.copy_from_slice(&srow[first..first + drow.len()]);
                    } else {
                        for (d, x) in drow.iter_mut().zip(srow[first..].iter().step_by(g.sw)) {
                            *d = *x;
                        }
                    }
                }
            }
        }
    };
    #[cfg(feature = "rayon")]
    if cols.len() >= 16384 {
        use rayon::prelude::*;
        cols.par_chunks_mut(per_ch).enumerate().for_each(body);
        return;
    }
    cols.chunks_mut(per_ch).enumerate().for_each(body);
}

/// Single-threaded `unfold_fill` for use inside an outer `rayon` loop (avoids nested thread oversubscription).
fn unfold_fill_seq<T: Copy>(src: &[T], cols: &mut [T], g: &UnfoldGeom) {
    let l = g.out_h * g.out_w;
    let per_ch = g.kh * g.kw * l;
    let plane_len = g.h * g.w;
    if per_ch == 0 || plane_len == 0 {
        return;
    }
    for (bc, block) in cols.chunks_mut(per_ch).enumerate() {
        let plane = &src[bc * plane_len..(bc + 1) * plane_len];
        for r in 0..g.kh {
            for s in 0..g.kw {
                let dst = &mut block[(r * g.kw + s) * l..(r * g.kw + s + 1) * l];
                let off = s * g.dw;
                let ow_lo = if off >= g.pw { 0 } else { (g.pw - off).div_ceil(g.sw) };
                let ow_hi = if g.w + g.pw > off { ((g.w - 1 + g.pw - off) / g.sw + 1).min(g.out_w) } else { 0 };
                if ow_lo >= ow_hi {
                    continue;
                }
                let first = ow_lo * g.sw + off - g.pw;
                for oh in 0..g.out_h {
                    let ih = (oh * g.sh + r * g.dh) as isize - g.ph as isize;
                    if ih < 0 || ih as usize >= g.h {
                        continue;
                    }
                    let srow = &plane[ih as usize * g.w..(ih as usize + 1) * g.w];
                    let drow = &mut dst[oh * g.out_w + ow_lo..oh * g.out_w + ow_hi];
                    if g.sw == 1 {
                        drow.copy_from_slice(&srow[first..first + drow.len()]);
                    } else {
                        for (d, x) in drow.iter_mut().zip(srow[first..].iter().step_by(g.sw)) {
                            *d = *x;
                        }
                    }
                }
            }
        }
    }
}

/// Winograd F(2x2, 3x3) convolution for `F32` `[N, C_in, H, W]` * `[C_out, C_in, 3, 3]`,
/// reducing multiplications per output pixel by 2.25x and executing 16 parallel GEMMs.
#[cfg(feature = "rayon")]
#[allow(clippy::too_many_arguments)]
fn conv2d_winograd_3x3_f32(
    src: &[f32],
    wv: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
    n: usize,
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    ph: usize,
    pw: usize,
    out_h: usize,
    out_w: usize,
) {
    use rayon::prelude::*;

    let th = out_h.div_ceil(2);
    let tw = out_w.div_ceil(2);
    let tiles_per_img = th * tw;
    let total_tiles = n * tiles_per_img;
    let l = out_h * out_w;

    // 1. Filter transform U[16, c_out, c_in] = G g G^T
    let u_stride = c_out * c_in;
    let mut u = vec![0.0f32; 16 * u_stride];
    for oc in 0..c_out {
        for ic in 0..c_in {
            let g = &wv[(oc * c_in + ic) * 9..(oc * c_in + ic) * 9 + 9];
            let mut t = [[0.0f32; 4]; 3];
            for r in 0..3 {
                let (g0, g1, g2) = (g[r * 3], g[r * 3 + 1], g[r * 3 + 2]);
                t[r][0] = g0;
                t[r][1] = 0.5 * (g0 + g1 + g2);
                t[r][2] = 0.5 * (g0 - g1 + g2);
                t[r][3] = g2;
            }
            for c in 0..4 {
                let (t0, t1, t2) = (t[0][c], t[1][c], t[2][c]);
                let base = oc * c_in + ic;
                u[c * u_stride + base] = t0;
                u[(4 + c) * u_stride + base] = 0.5 * (t0 + t1 + t2);
                u[(8 + c) * u_stride + base] = 0.5 * (t0 - t1 + t2);
                u[(12 + c) * u_stride + base] = t2;
            }
        }
    }

    // 2. Input transform V[16, c_in, total_tiles] = B^T d B
    let v_stride = c_in * total_tiles;
    let mut v = crate::pool::take_f32(16 * v_stride).unwrap_or_else(|| vec![0.0; 16 * v_stride]);
    let hw = h * w;
    let img_elems = c_in * hw;
    let v_ptr = v.as_mut_ptr() as usize;
    (0..n * c_in).into_par_iter().for_each(|bi| {
        let b = bi / c_in;
        let ic = bi % c_in;
        let plane = &src[b * img_elems + ic * hw..b * img_elems + (ic + 1) * hw];
        let v_base_ptr = v_ptr as *mut f32;
        for ty in 0..th {
            let iy0 = (ty * 2) as isize - ph as isize;
            for tx in 0..tw {
                let ix0 = (tx * 2) as isize - pw as isize;
                let mut d = [[0.0f32; 4]; 4];
                if iy0 >= 0 && (iy0 as usize) + 3 < h && ix0 >= 0 && (ix0 as usize) + 3 < w {
                    let y0 = iy0 as usize;
                    let x0 = ix0 as usize;
                    for r in 0..4 {
                        let row = &plane[(y0 + r) * w + x0..(y0 + r) * w + x0 + 4];
                        d[r].copy_from_slice(row);
                    }
                } else {
                    for r in 0..4 {
                        let iy = iy0 + r as isize;
                        if iy >= 0 && (iy as usize) < h {
                            let row = &plane[iy as usize * w..iy as usize * w + w];
                            for c in 0..4 {
                                let ix = ix0 + c as isize;
                                if ix >= 0 && (ix as usize) < w {
                                    d[r][c] = row[ix as usize];
                                }
                            }
                        }
                    }
                }
                let mut s = [[0.0f32; 4]; 4];
                for c in 0..4 {
                    s[0][c] = d[0][c] - d[2][c];
                    s[1][c] = d[1][c] + d[2][c];
                    s[2][c] = -d[1][c] + d[2][c];
                    s[3][c] = d[1][c] - d[3][c];
                }
                let tile_idx = (b * th + ty) * tw + tx;
                let col_off = ic * total_tiles + tile_idx;
                for r in 0..4 {
                    let (a0, a1, a2, a3) = (s[r][0], s[r][1], s[r][2], s[r][3]);
                    unsafe {
                        *v_base_ptr.add((r * 4) * v_stride + col_off) = a0 - a2;
                        *v_base_ptr.add((r * 4 + 1) * v_stride + col_off) = a1 + a2;
                        *v_base_ptr.add((r * 4 + 2) * v_stride + col_off) = -a1 + a2;
                        *v_base_ptr.add((r * 4 + 3) * v_stride + col_off) = a1 - a3;
                    }
                }
            }
        }
    });

    // 3. 16 parallel GEMMs: M[p, c_out, total_tiles] = U[p, c_out, c_in] @ V[p, c_in, total_tiles]
    let m_stride = c_out * total_tiles;
    let mut m = crate::pool::take_f32(16 * m_stride).unwrap_or_else(|| vec![0.0; 16 * m_stride]);
    m.par_chunks_mut(m_stride)
        .enumerate()
        .for_each(|(p, m_slice)| {
            let up = &u[p * u_stride..(p + 1) * u_stride];
            let vp = &v[p * v_stride..(p + 1) * v_stride];
            crate::gemm::gemm_f32_strided_seq(
                up,
                0,
                c_in as isize,
                1,
                vp,
                0,
                total_tiles as isize,
                1,
                m_slice,
                c_out,
                c_in,
                total_tiles,
            );
        });
    crate::pool::give(v);

    // 4. Inverse transform Y = A^T M A + bias -> out[n, c_out, out_h, out_w]
    out.par_chunks_mut(l).enumerate().for_each(|(bo, o_plane)| {
        let b = bo / c_out;
        let oc = bo % c_out;
        let bb = bias.map_or(0.0, |bv| bv[oc]);
        for ty in 0..th {
            let y0 = ty * 2;
            let has_y1 = y0 + 1 < out_h;
            for tx in 0..tw {
                let x0 = tx * 2;
                let has_x1 = x0 + 1 < out_w;
                let tile_idx = (b * th + ty) * tw + tx;
                let col_off = oc * total_tiles + tile_idx;
                let mut mp = [0.0f32; 16];
                for p in 0..16 {
                    mp[p] = m[p * m_stride + col_off];
                }
                let mut t0 = [0.0f32; 4];
                let mut t1 = [0.0f32; 4];
                for c in 0..4 {
                    t0[c] = mp[c] + mp[4 + c] + mp[8 + c];
                    t1[c] = mp[4 + c] - mp[8 + c] - mp[12 + c];
                }
                let y00 = t0[0] + t0[1] + t0[2] + bb;
                let y01 = t0[1] - t0[2] - t0[3] + bb;
                let y10 = t1[0] + t1[1] + t1[2] + bb;
                let y11 = t1[1] - t1[2] - t1[3] + bb;
                o_plane[y0 * out_w + x0] = y00;
                if has_x1 {
                    o_plane[y0 * out_w + x0 + 1] = y01;
                }
                if has_y1 {
                    o_plane[(y0 + 1) * out_w + x0] = y10;
                    if has_x1 {
                        o_plane[(y0 + 1) * out_w + x0 + 1] = y11;
                    }
                }
            }
        }
    });
    crate::pool::give(m);
}

impl Tensor {
    // =========================================================================
    // 1. Zero-Copy Views, Strides, DType & Device Conversions
    // =========================================================================

    /// Applies an invertible [`ViewSpec`] as a zero-copy strided view.
    /// Its backward pass is [`Tensor::view_scatter`], whose own backward pass is `apply_view`!
    pub fn apply_view(&self, spec: ViewSpec) -> Result<Tensor> {
        let new_layout = spec.apply_layout(self.layout())?;
        // `ViewSpec` fields are pub: a hand-built spec can carry strides that
        // stay in-bounds at the offset but read past storage on first use
        // (panic) instead of erroring here.
        new_layout.check_storage_bounds(self.storage().len(), "view")?;
        Ok(Tensor::from_op(
            self.storage().clone(),
            new_layout,
            "view",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.view_scatter(spec.clone())?)]),
        ))
    }

    /// Adjoint of [`Tensor::apply_view`]: scatters `self` (of shape `spec.dst_shape`)
    /// into a zero tensor of `spec.src_shape`. Its backward pass is `apply_view(spec)`!
    pub fn view_scatter(&self, spec: ViewSpec) -> Result<Tensor> {
        let out_storage = self.storage().view_scatter(self.layout(), &spec)?;
        let out_layout = Layout::contiguous(spec.src_shape.clone());
        Ok(Tensor::from_op(
            out_storage,
            out_layout,
            "view_scatter",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.apply_view(spec.clone())?)]),
        ))
    }

    /// Returns a contiguous tensor (no-op clone if already contiguous with offset 0).
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() && self.storage_offset() == 0 {
            return Ok(self.clone());
        }
        let s = self.storage().to_contiguous(self.layout())?;
        let l = Layout::contiguous(self.shape().to_vec());
        Ok(Tensor::from_op(
            s,
            l,
            "contiguous",
            vec![self.clone()],
            |g| Ok(vec![Some(g.clone())]),
        ))
    }

    /// Casts tensor elements to `dst` dtype. Differentiable between floating-point dtypes!
    pub fn to_dtype(&self, dst: DType) -> Result<Tensor> {
        if self.dtype() == dst {
            return Ok(self.clone());
        }
        let src_dtype = self.dtype();
        let s = self.storage().to_dtype(self.layout(), dst)?;
        let l = Layout::contiguous(self.shape().to_vec());
        Ok(Tensor::from_op(
            s,
            l,
            "to_dtype",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.to_dtype(src_dtype)?)]),
        ))
    }

    /// Transfers tensor to `device`. Differentiable across devices!
    pub fn to_device(&self, device: &Device) -> Result<Tensor> {
        if &self.device() == device {
            return Ok(self.clone());
        }
        let src_dev = self.device();
        let s = self.storage().to_device(self.layout(), device)?;
        let l = Layout::contiguous(self.shape().to_vec());
        Ok(Tensor::from_op(
            s,
            l,
            "to_device",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.to_device(&src_dev)?)]),
        ))
    }

    /// Alias for `to_device`.
    #[inline]
    pub fn to(&self, device: &Device) -> Result<Tensor> {
        self.to_device(device)
    }

    /// Transfers this tensor to Host CPU.
    #[inline]
    pub fn cpu(&self) -> Result<Tensor> {
        self.to_device(&Device::Cpu)
    }

    /// Moves this tensor to the simulated CUDA device at `index`. No CUDA runtime is used: the data stays in
    /// host memory and kernels run on the CPU.
    #[inline]
    pub fn cuda(&self, index: usize) -> Result<Tensor> {
        self.to_device(&Device::Cuda(index))
    }

    /// Moves this tensor to a GPU. With the `wgpu` feature and a real adapter the data moves to that adapter;
    /// otherwise it is placed on a simulated CUDA device (host memory, CPU kernels).
    pub fn gpu(&self, index: usize) -> Result<Tensor> {
        #[cfg(feature = "wgpu")]
        {
            if let Ok(dev) = Device::wgpu() {
                return self.to_device(&dev);
            }
        }
        self.to_device(&Device::Cuda(index))
    }

    /// Moves this tensor to a simulated Google TPU chip and core (host memory, CPU kernels).
    #[inline]
    pub fn tpu(&self, chip: usize, core: usize) -> Result<Tensor> {
        self.to_device(&Device::Tpu { chip, core })
    }

    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.device().is_cpu()
    }

    #[inline]
    pub fn is_cuda(&self) -> bool {
        self.device().is_cuda()
    }

    #[inline]
    pub fn is_gpu(&self) -> bool {
        self.device().is_gpu()
    }

    #[inline]
    pub fn is_tpu(&self) -> bool {
        self.device().is_tpu()
    }

    /// Returns `true` if all elements are finite (not NaN, +inf, or -inf).
    pub fn all_finite(&self) -> Result<bool> {
        if self.is_cpu() || self.device().is_simulated() {
            Ok(self.storage().all_finite(self.layout()))
        } else {
            let v = self.to_vec::<f64>()?;
            Ok(v.iter().all(|x| x.is_finite()))
        }
    }

    /// Reshapes the tensor, returning a **zero-copy** view whenever current strides permit,
    /// or materializing a contiguous copy otherwise.
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let n = crate::layout::checked_numel(shape)?;
        if n != self.numel() {
            return Err(Error::ShapeMismatch {
                op: "reshape",
                expected: shape.to_vec(),
                got: self.shape().to_vec(),
            });
        }
        let orig_shape = self.shape().to_vec();
        if let Some(new_strides) = self.layout().reshape_strides(shape) {
            let new_layout = Layout::new(shape.to_vec(), new_strides, self.storage_offset());
            Ok(Tensor::from_op(
                self.storage().clone(),
                new_layout,
                "reshape_view",
                vec![self.clone()],
                move |g| Ok(vec![Some(g.reshape(&orig_shape)?)]),
            ))
        } else {
            let contig = no_grad(|| self.contiguous())?;
            let new_layout = Layout::contiguous(shape.to_vec());
            Ok(Tensor::from_op(
                contig.storage().clone(),
                new_layout,
                "reshape",
                vec![self.clone()],
                move |g| Ok(vec![Some(g.reshape(&orig_shape)?)]),
            ))
        }
    }

    /// Strict zero-copy view reshape; fails with `Error::NotContiguous` if strides are incompatible.
    pub fn view(&self, shape: &[usize]) -> Result<Tensor> {
        let Some(new_strides) = self.layout().reshape_strides(shape) else {
            return Err(Error::NotContiguous { op: "view" });
        };
        let orig_shape = self.shape().to_vec();
        let new_layout = Layout::new(shape.to_vec(), new_strides, self.storage_offset());
        Ok(Tensor::from_op(
            self.storage().clone(),
            new_layout,
            "view_reshape",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.reshape(&orig_shape)?)]),
        ))
    }

    /// Zero-copy dimension permutation.
    pub fn permute(&self, axes: &[usize]) -> Result<Tensor> {
        let new_layout = self.layout().permute(axes)?;
        let mut inv = vec![0usize; axes.len()];
        for (i, &a) in axes.iter().enumerate() {
            inv[a] = i;
        }
        Ok(Tensor::from_op(
            self.storage().clone(),
            new_layout,
            "permute",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.permute(&inv)?)]),
        ))
    }

    /// Zero-copy swap of two dimensions `d0` and `d1` (supports negative axes).
    pub fn transpose(&self, d0: isize, d1: isize) -> Result<Tensor> {
        let a0 = normalize_axis("transpose", d0, self.ndim())?;
        let a1 = normalize_axis("transpose", d1, self.ndim())?;
        let new_layout = self.layout().transpose(a0 as isize, a1 as isize)?;
        Ok(Tensor::from_op(
            self.storage().clone(),
            new_layout,
            "transpose",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.transpose(a0 as isize, a1 as isize)?)]),
        ))
    }

    /// Zero-copy transpose of the last two dimensions.
    pub fn t(&self) -> Result<Tensor> {
        if self.ndim() < 2 {
            return Err(Error::Msg(format!(
                "t() requires rank >= 2, got {:?}",
                self.shape()
            )));
        }
        self.transpose(-2, -1)
    }

    /// Zero-copy move of dimension `src` to `dst`.
    pub fn movedim(&self, src: isize, dst: isize) -> Result<Tensor> {
        let s = normalize_axis("movedim", src, self.ndim())?;
        let d = normalize_axis("movedim", dst, self.ndim())?;
        let mut axes: Vec<usize> = (0..self.ndim()).filter(|&i| i != s).collect();
        axes.insert(d, s);
        self.permute(&axes)
    }

    /// Zero-copy broadcast expansion (sets stride = 0 on expanded dimensions).
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Tensor> {
        if self.shape() == target {
            return Ok(self.clone());
        }
        let new_layout = self.layout().broadcast_to(target)?;
        let orig_shape = self.shape().to_vec();
        Ok(Tensor::from_op(
            self.storage().clone(),
            new_layout,
            "broadcast_to",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.sum_to_shape(&orig_shape)?)]),
        ))
    }

    /// Alias for [`Tensor::broadcast_to`].
    pub fn expand(&self, target: &[usize]) -> Result<Tensor> {
        self.broadcast_to(target)
    }

    /// Reduces `self` by summing over broadcast dimensions so the output has `target` shape.
    pub fn sum_to_shape(&self, target: &[usize]) -> Result<Tensor> {
        if self.shape() == target {
            return Ok(self.clone());
        }
        if target.len() > self.ndim() {
            return Err(Error::ShapeMismatch {
                op: "sum_to_shape",
                expected: target.to_vec(),
                got: self.shape().to_vec(),
            });
        }
        let pad = self.ndim() - target.len();
        let mut reduce_axes = Vec::new();
        for i in 0..self.ndim() {
            if i < pad || (target[i - pad] == 1 && self.shape()[i] != 1) {
                reduce_axes.push(i as isize);
            }
        }
        let mut out = if reduce_axes.is_empty() {
            self.clone()
        } else {
            self.sum_axes(&reduce_axes, true)?
        };
        if out.shape() != target {
            out = out.reshape(target)?;
        }
        Ok(out)
    }

    /// Zero-copy narrow along `axis`: `[start .. start + len]`.
    pub fn narrow(&self, axis: isize, start: usize, len: usize) -> Result<Tensor> {
        let ax = normalize_axis("narrow", axis, self.ndim())?;
        let dim = self.shape()[ax];
        if len > dim || start > dim - len {
            return Err(Error::IndexOutOfBounds {
                op: "narrow",
                index: start.saturating_add(len) as isize,
                dim_size: dim,
            });
        }
        let mut spec = ViewSpec::identity(self.shape());
        spec.dst_shape[ax] = len;
        spec.dim_maps[ax] = DimMap::Sliced {
            out_dim: ax,
            start: start as isize,
            step: 1,
        };
        self.apply_view(spec)
    }

    /// Zero-copy stepped slice along `axis`.
    pub fn slice(
        &self,
        axis: isize,
        start: Option<isize>,
        end: Option<isize>,
        step: isize,
    ) -> Result<Tensor> {
        let ax = normalize_axis("slice", axis, self.ndim())?;
        let mut elems = vec![SliceElem::FULL; self.ndim()];
        elems[ax] = SliceElem::Slice { start, end, step };
        self.i(elems)
    }

    /// Zero-copy reversal along `axes` using negative strides.
    pub fn flip(&self, axes: &[isize]) -> Result<Tensor> {
        let mut spec = ViewSpec::identity(self.shape());
        let mut seen = vec![false; self.ndim()];
        for &a in axes {
            let ax = normalize_axis("flip", a, self.ndim())?;
            if seen[ax] {
                return Err(Error::Msg(format!("flip: duplicate axis {a}")));
            }
            seen[ax] = true;
            let d = self.shape()[ax];
            spec.dim_maps[ax] = DimMap::Sliced {
                out_dim: ax,
                start: (d as isize).saturating_sub(1),
                step: -1,
            };
        }
        self.apply_view(spec)
    }

    /// Zero-copy selection of a single index along `axis`, removing that dimension.
    pub fn select(&self, axis: isize, index: isize) -> Result<Tensor> {
        let ax = normalize_axis("select", axis, self.ndim())?;
        let mut elems = vec![SliceElem::FULL; self.ndim()];
        elems[ax] = SliceElem::Index(index);
        self.i(elems)
    }

    /// Zero-copy insertion of a size-1 dimension at `axis`.
    pub fn unsqueeze(&self, axis: isize) -> Result<Tensor> {
        let ax = normalize_insert_axis("unsqueeze", axis, self.ndim())?;
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(ax, 1);
        self.reshape(&new_shape)
    }

    /// Zero-copy removal of a size-1 dimension at `axis`. Errors when the
    /// dimension is not 1, like PyTorch (a silent no-op would hide shape bugs).
    pub fn squeeze_axis(&self, axis: isize) -> Result<Tensor> {
        let ax = normalize_axis("squeeze_axis", axis, self.ndim())?;
        if self.shape()[ax] != 1 {
            return Err(Error::Msg(format!(
                "squeeze_axis: expected dimension {ax} to have size 1, got shape {:?}",
                self.shape()
            )));
        }
        let mut new_shape = self.shape().to_vec();
        new_shape.remove(ax);
        self.reshape(&new_shape)
    }

    /// Zero-copy removal of all size-1 dimensions.
    pub fn squeeze(&self) -> Result<Tensor> {
        let new_shape: Vec<usize> = self.shape().iter().copied().filter(|&d| d != 1).collect();
        self.reshape(&new_shape)
    }

    /// Flattens dimensions `[start_dim ..= end_dim]` into a single dimension.
    pub fn flatten_range(&self, start_dim: isize, end_dim: isize) -> Result<Tensor> {
        if self.ndim() == 0 {
            return self.reshape(&[1]);
        }
        let s = normalize_axis("flatten", start_dim, self.ndim())?;
        let e = normalize_axis("flatten", end_dim, self.ndim())?;
        if s > e {
            return Err(Error::Msg(format!("flatten: start_dim {s} > end_dim {e}")));
        }
        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&self.shape()[..s]);
        new_shape.push(self.shape()[s..=e].iter().product());
        new_shape.extend_from_slice(&self.shape()[e + 1..]);
        self.reshape(&new_shape)
    }

    /// Flattens all dimensions after dimension 0 (`[b, ...] -> [b, prod]`).
    pub fn flatten(&self) -> Result<Tensor> {
        if self.ndim() <= 1 {
            return self.reshape(&[self.numel()]);
        }
        self.flatten_range(1, -1)
    }

    /// Zero-copy diagonal view across `dim1` and `dim2` with offset `diag_offset`.
    pub fn diagonal(&self, diag_offset: isize, dim1: isize, dim2: isize) -> Result<Tensor> {
        let d1 = normalize_axis("diagonal", dim1, self.ndim())?;
        let d2 = normalize_axis("diagonal", dim2, self.ndim())?;
        let new_layout = self
            .layout()
            .diagonal(diag_offset, d1 as isize, d2 as isize)?;
        let src_shape = self.shape().to_vec();
        Ok(Tensor::from_op(
            self.storage().clone(),
            new_layout,
            "diagonal",
            vec![self.clone()],
            move |g| {
                // Scatter diagonal gradient back into a zero tensor of `src_shape`.
                let z = Tensor::zeros(&src_shape, g.dtype(), &g.device())?;
                let diag_layout = z.layout().diagonal(diag_offset, d1 as isize, d2 as isize)?;
                let g_vec = g.to_vec::<f64>()?;
                let mut z_vec = vec![0.0f64; z.numel()];
                for (dst_off, val) in diag_layout.storage_offsets().zip(g_vec) {
                    z_vec[dst_off] += val;
                }
                let s = CpuStorage::F64(z_vec)
                    .to_dtype(&Layout::contiguous(src_shape.clone()), g.dtype())?;
                let out = Tensor::from_storage(
                    Storage::new_cpu(s),
                    Layout::contiguous(src_shape.clone()),
                )
                .to_device(&g.device())?;
                Ok(vec![Some(out)])
            },
        ))
    }

    /// Low-level zero-copy view with arbitrary `shape`, `strides`, and `offset`.
    pub fn as_strided(&self, shape: &[usize], strides: &[isize], offset: usize) -> Result<Tensor> {
        let new_layout = Layout::try_new(shape.to_vec(), strides.to_vec(), offset)
            .map_err(|e| Error::Msg(format!("as_strided: {e}")))?;
        new_layout.check_storage_bounds(self.storage().len(), "as_strided")?;
        let src_layout = self.layout().clone();
        let src_shape = self.shape().to_vec();
        let view_layout = new_layout.clone();
        Ok(Tensor::from_op(
            self.storage().clone(),
            new_layout,
            "as_strided",
            vec![self.clone()],
            move |g| {
                // Adjoint of arbitrary strided view (including overlapping windows!):
                // view offsets are absolute storage positions, so map them back to
                // the source tensor's logical indices through the source layout
                // (the source itself may be a nonzero-offset/strided view).
                let mut off_to_logical: HashMap<usize, Vec<usize>> =
                    HashMap::with_capacity(src_layout.numel());
                for (logical, off) in src_layout.storage_offsets().enumerate() {
                    off_to_logical.entry(off).or_default().push(logical);
                }
                let g_vec = g.to_vec::<f64>()?;
                let mut dst = vec![0.0f64; src_shape.iter().product()];
                for (off, val) in view_layout.storage_offsets().zip(g_vec) {
                    let Some(logicals) = off_to_logical.get(&off) else {
                        return Err(Error::Autograd(
                            "as_strided backward: view reads storage outside the source tensor"
                                .into(),
                        ));
                    };
                    for &logical in logicals {
                        dst[logical] += val;
                    }
                }
                let s = CpuStorage::F64(dst)
                    .to_dtype(&Layout::contiguous(src_shape.clone()), g.dtype())?;
                let out = Tensor::from_storage(
                    Storage::new_cpu(s),
                    Layout::contiguous(src_shape.clone()),
                )
                .to_device(&g.device())?;
                Ok(vec![Some(out)])
            },
        ))
    }

    // =========================================================================
    // 2. Unary & Activation Operations (with higher-order Tensor backward!)
    // =========================================================================

    fn run_unary(
        &self,
        op_name: &'static str,
        op: UnaryOp,
        backward: impl Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + Send + Sync + 'static,
    ) -> Result<Tensor> {
        let s = self.storage().unary_op(self.layout(), op)?;
        let l = Layout::contiguous(self.shape().to_vec());
        let inp = self.clone();
        let out_for_bwd = Tensor::from_storage(s.clone(), l.clone());
        Ok(Tensor::from_op(
            s,
            l,
            op_name,
            vec![self.clone()],
            move |g| Ok(vec![Some(backward(g, &inp, &out_for_bwd)?)]),
        ))
    }

    pub fn neg(&self) -> Result<Tensor> {
        self.run_unary("neg", UnaryOp::Neg, |g, _, _| g.neg())
    }

    pub fn abs(&self) -> Result<Tensor> {
        self.run_unary("abs", UnaryOp::Abs, |g, x, _| g.mul(&x.sign()?))
    }

    pub fn sign(&self) -> Result<Tensor> {
        self.run_unary("sign", UnaryOp::Sign, |g, _, _| g.zeros_like())
    }

    pub fn exp(&self) -> Result<Tensor> {
        if let Some(x) = crate::amp::fp32_input(self)? {
            return x.exp();
        }
        // d/dx exp(x) = exp(x); using `x.exp()` keeps higher-order graph connected!
        self.run_unary("exp", UnaryOp::Exp, |g, x, _| g.mul(&x.exp()?))
    }

    pub fn ln(&self) -> Result<Tensor> {
        if let Some(x) = crate::amp::fp32_input(self)? {
            return x.ln();
        }
        self.run_unary("ln", UnaryOp::Ln, |g, x, _| g.div(x))
    }

    pub fn sqrt(&self) -> Result<Tensor> {
        self.run_unary("sqrt", UnaryOp::Sqrt, |g, x, _| {
            g.div(&x.sqrt()?.mul_scalar(2.0)?)
        })
    }

    pub fn rsqrt(&self) -> Result<Tensor> {
        self.run_unary("rsqrt", UnaryOp::Rsqrt, |g, x, _| {
            g.mul(&x.powf(-1.5)?.mul_scalar(-0.5)?)
        })
    }

    pub fn sin(&self) -> Result<Tensor> {
        self.run_unary("sin", UnaryOp::Sin, |g, x, _| g.mul(&x.cos()?))
    }

    pub fn cos(&self) -> Result<Tensor> {
        self.run_unary("cos", UnaryOp::Cos, |g, x, _| g.mul(&x.sin()?.neg()?))
    }

    pub fn tanh(&self) -> Result<Tensor> {
        self.run_unary("tanh", UnaryOp::Tanh, |g, x, _| {
            let y = x.tanh()?;
            let one_minus_y2 = y.square()?.neg()?.add_scalar(1.0)?;
            g.mul(&one_minus_y2)
        })
    }

    pub fn sigmoid(&self) -> Result<Tensor> {
        self.run_unary("sigmoid", UnaryOp::Sigmoid, |g, x, _| {
            let y = x.sigmoid()?;
            let one_minus_y = y.neg()?.add_scalar(1.0)?;
            g.mul(&y.mul(&one_minus_y)?)
        })
    }

    pub fn relu(&self) -> Result<Tensor> {
        self.run_unary("relu", UnaryOp::Relu, |g, x, _| {
            let zero = x.zeros_like()?;
            let mask = x.gt(&zero)?.to_dtype(g.dtype())?;
            g.mul(&mask)
        })
    }

    pub fn leaky_relu(&self, slope: f64) -> Result<Tensor> {
        self.run_unary("leaky_relu", UnaryOp::LeakyRelu(slope), move |g, x, _| {
            let zero = x.zeros_like()?;
            let cond = x.gt(&zero)?;
            let pos = g.clone();
            let neg = g.mul_scalar(slope)?;
            Tensor::where_cond(&cond, &pos, &neg)
        })
    }

    pub fn gelu(&self) -> Result<Tensor> {
        self.run_unary("gelu", UnaryOp::Gelu, |g, x, _| {
            if !crate::tensor::is_grad_enabled()
                && let Some(out_storage) = g.storage().try_gelu_backward(g.layout(), x.storage(), x.layout())
            {
                return Ok(Tensor::from_parts(
                    out_storage,
                    Layout::contiguous(g.shape()),
                    false,
                    None,
                ));
            }
            // d/dx [0.5 * x * (1 + erf(x / sqrt(2)))]
            // = 0.5 * (1 + erf(x / sqrt(2))) + x * (1 / sqrt(2*pi)) * exp(-0.5 * x^2)
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let inv_sqrt2pi = 0.398_942_280_401_432_7_f64;
            let cdf = x
                .mul_scalar(inv_sqrt2)?
                .erf()?
                .add_scalar(1.0)?
                .mul_scalar(0.5)?;
            let pdf = x
                .square()?
                .mul_scalar(-0.5)?
                .exp()?
                .mul_scalar(inv_sqrt2pi)?;
            let deriv = cdf.add(&x.mul(&pdf)?)?;
            g.mul(&deriv)
        })
    }

    pub fn silu(&self) -> Result<Tensor> {
        self.run_unary("silu", UnaryOp::Silu, |g, x, _| {
            // d/dx (x * sig(x)) = sig(x) * (1 + x * (1 - sig(x)))
            let sig = x.sigmoid()?;
            let one_minus_sig = sig.neg()?.add_scalar(1.0)?;
            let factor = x.mul(&one_minus_sig)?.add_scalar(1.0)?;
            g.mul(&sig.mul(&factor)?)
        })
    }

    pub fn elu(&self, alpha: f64) -> Result<Tensor> {
        self.run_unary("elu", UnaryOp::Elu(alpha), move |g, x, _| {
            let zero = x.zeros_like()?;
            let cond = x.gt(&zero)?;
            let neg_branch = x.exp()?.mul_scalar(alpha)?.mul(g)?;
            Tensor::where_cond(&cond, g, &neg_branch)
        })
    }

    pub fn softplus(&self) -> Result<Tensor> {
        // d/dx ln(1 + exp(x)) = sigmoid(x)
        self.run_unary("softplus", UnaryOp::Softplus, |g, x, _| {
            g.mul(&x.sigmoid()?)
        })
    }

    pub fn square(&self) -> Result<Tensor> {
        self.run_unary("square", UnaryOp::Square, |g, x, _| {
            g.mul(&x.mul_scalar(2.0)?)
        })
    }

    pub fn recip(&self) -> Result<Tensor> {
        self.run_unary("recip", UnaryOp::Recip, |g, x, _| {
            g.div(&x.square()?)?.neg()
        })
    }

    pub fn erf(&self) -> Result<Tensor> {
        self.run_unary("erf", UnaryOp::Erf, |g, x, _| {
            // d/dx erf(x) = (2 / sqrt(pi)) * exp(-x^2)
            let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
            let deriv = x.square()?.neg()?.exp()?.mul_scalar(two_over_sqrt_pi)?;
            g.mul(&deriv)
        })
    }

    pub fn floor(&self) -> Result<Tensor> {
        self.run_unary("floor", UnaryOp::Floor, |g, _, _| g.zeros_like())
    }

    pub fn ceil(&self) -> Result<Tensor> {
        self.run_unary("ceil", UnaryOp::Ceil, |g, _, _| g.zeros_like())
    }

    pub fn round(&self) -> Result<Tensor> {
        self.run_unary("round", UnaryOp::Round, |g, _, _| g.zeros_like())
    }

    pub fn powf(&self, p: f64) -> Result<Tensor> {
        self.run_unary("powf", UnaryOp::Powf(p), move |g, x, _| {
            let raw = g.mul(&x.powf(p - 1.0)?.mul_scalar(p)?)?;
            if p == 0.0 {
                // Torch parity: d(x^0)/dx = 0 everywhere including x = 0,
                // where the raw form gives 0 * inf = NaN.
                let x_zero = x.eq_tensor(&x.zeros_like()?)?;
                Tensor::where_cond(&x_zero, &g.zeros_like()?, &raw)
            } else {
                Ok(raw)
            }
        })
    }

    pub fn add_scalar(&self, s: f64) -> Result<Tensor> {
        self.run_unary("add_scalar", UnaryOp::AddScalar(s), |g, _, _| Ok(g.clone()))
    }

    pub fn sub_scalar(&self, s: f64) -> Result<Tensor> {
        self.add_scalar(-s)
    }

    pub fn mul_scalar(&self, s: f64) -> Result<Tensor> {
        self.run_unary("mul_scalar", UnaryOp::MulScalar(s), move |g, _, _| {
            g.mul_scalar(s)
        })
    }

    pub fn div_scalar(&self, s: f64) -> Result<Tensor> {
        // True division (x/s), not x*(1/s): the reciprocal overflows to inf
        // for tiny s, corrupting finite quotients like 1e-300/1e-300.
        self.div(&self.full_like(s)?)
    }

    pub fn clamp(&self, min: f64, max: f64) -> Result<Tensor> {
        if min.is_nan() || max.is_nan() || min > max {
            return Err(Error::Msg(format!(
                "clamp requires min <= max and no NaN bound, got min = {min}, max = {max}"
            )));
        }
        self.run_unary("clamp", UnaryOp::Clamp(min, max), move |g, x, _| {
            // torch.clamp: the gradient passes wherever min <= x <= max (bounds included)
            let lo = x.full_like(min)?;
            let hi = x.full_like(max)?;
            // (x >= min) & (x <= max): false for NaN, so NaN inputs get zero gradient as in PyTorch
            let ge_lo = x.ge(&lo)?.to_dtype(g.dtype())?;
            let le_hi = x.le(&hi)?.to_dtype(g.dtype())?;
            Tensor::mul(&g.mul(&ge_lo)?, &le_hi)
        })
    }

    /// Like [`clamp`](Self::clamp) but the gradient is zero at the bounds themselves, as for `hardtanh` / `relu6`
    /// (`torch.nn.functional.hardtanh` differs from `torch.clamp` exactly there).
    pub(crate) fn clamp_open(&self, min: f64, max: f64) -> Result<Tensor> {
        if min.is_nan() || max.is_nan() || min > max {
            return Err(Error::Msg(format!(
                "clamp requires min <= max and no NaN bound, got min = {min}, max = {max}"
            )));
        }
        self.run_unary("clamp_open", UnaryOp::Clamp(min, max), move |g, x, _| {
            let lo = x.full_like(min)?;
            let hi = x.full_like(max)?;
            let gt_lo = x.gt(&lo)?.to_dtype(g.dtype())?;
            let lt_hi = x.lt(&hi)?.to_dtype(g.dtype())?;
            Tensor::mul(&g.mul(&gt_lo)?, &lt_hi)
        })
    }

    pub fn logical_not(&self) -> Result<Tensor> {
        let s = self
            .storage()
            .unary_op(self.layout(), UnaryOp::LogicalNot)?;
        Ok(Tensor::from_storage(
            s,
            Layout::contiguous(self.shape().to_vec()),
        ))
    }

    // =========================================================================
    // 3. Binary & Comparison Operations (with NumPy Broadcasting)
    // =========================================================================

    fn broadcast_pair(
        &self,
        other: &Tensor,
        op: &'static str,
    ) -> Result<(Layout, Layout, Vec<usize>)> {
        let out_shape = broadcast_shapes(op, self.shape(), other.shape())?;
        let la = self.layout().broadcast_to(&out_shape)?;
        let lb = other.layout().broadcast_to(&out_shape)?;
        Ok((la, lb, out_shape))
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "add")?;
        let s = self
            .storage()
            .binary_op(&la, other.storage(), &lb, BinaryOp::Add)?;
        let sa = self.shape().to_vec();
        let sb = other.shape().to_vec();
        let (ra, rb) = (self.requires_grad(), other.requires_grad());
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "add",
            vec![self.clone(), other.clone()],
            move |g| {
                let ga = if ra { Some(g.sum_to_shape(&sa)?) } else { None };
                let gb = if rb { Some(g.sum_to_shape(&sb)?) } else { None };
                Ok(vec![ga, gb])
            },
        ))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "sub")?;
        let s = self
            .storage()
            .binary_op(&la, other.storage(), &lb, BinaryOp::Sub)?;
        let sa = self.shape().to_vec();
        let sb = other.shape().to_vec();
        let (ra, rb) = (self.requires_grad(), other.requires_grad());
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "sub",
            vec![self.clone(), other.clone()],
            move |g| {
                let ga = if ra { Some(g.sum_to_shape(&sa)?) } else { None };
                let gb = if rb {
                    Some(g.neg()?.sum_to_shape(&sb)?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "mul")?;
        let s = self
            .storage()
            .binary_op(&la, other.storage(), &lb, BinaryOp::Mul)?;
        let a = self.clone();
        let b = other.clone();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "mul",
            vec![self.clone(), other.clone()],
            move |g| {
                let ga = if a.requires_grad() {
                    Some(g.mul(&b)?.sum_to_shape(a.shape())?)
                } else {
                    None
                };
                let gb = if b.requires_grad() {
                    Some(g.mul(&a)?.sum_to_shape(b.shape())?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "div")?;
        let s = self
            .storage()
            .binary_op(&la, other.storage(), &lb, BinaryOp::Div)?;
        let a = self.clone();
        let b = other.clone();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "div",
            vec![self.clone(), other.clone()],
            move |g| {
                let ga = if a.requires_grad() {
                    Some(g.div(&b)?.sum_to_shape(a.shape())?)
                } else {
                    None
                };
                let gb = if b.requires_grad() {
                    Some(
                        g.mul(&a)?
                            .div(&b.square()?)?
                            .neg()?
                            .sum_to_shape(b.shape())?,
                    )
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    pub fn maximum(&self, other: &Tensor) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "maximum")?;
        let s = self
            .storage()
            .binary_op(&la, other.storage(), &lb, BinaryOp::Maximum)?;
        let a = self.clone();
        let b = other.clone();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "maximum",
            vec![self.clone(), other.clone()],
            move |g| {
                // Torch parity: ties split the gradient 0.5 / 0.5, and NaN
                // inputs (all three comparisons false) route full gradient
                // to both sides.
                let gt = a.gt(&b)?.to_dtype(g.dtype())?;
                let lt = a.lt(&b)?.to_dtype(g.dtype())?;
                let eq = a.eq_tensor(&b)?.to_dtype(g.dtype())?;
                let half_eq = eq.mul_scalar(0.5)?;
                let neither =
                    Tensor::neg(&Tensor::add(&Tensor::add(&gt, &lt)?, &eq)?)?.add_scalar(1.0)?;
                let ga_mask = Tensor::add(&Tensor::add(&gt, &half_eq)?, &neither)?;
                let gb_mask = Tensor::add(&Tensor::add(&lt, &half_eq)?, &neither)?;
                let ga = if a.requires_grad() {
                    Some(g.mul(&ga_mask)?.sum_to_shape(a.shape())?)
                } else {
                    None
                };
                let gb = if b.requires_grad() {
                    Some(g.mul(&gb_mask)?.sum_to_shape(b.shape())?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    pub fn minimum(&self, other: &Tensor) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "minimum")?;
        let s = self
            .storage()
            .binary_op(&la, other.storage(), &lb, BinaryOp::Minimum)?;
        let a = self.clone();
        let b = other.clone();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "minimum",
            vec![self.clone(), other.clone()],
            move |g| {
                // Torch parity: ties split the gradient 0.5 / 0.5, and NaN
                // inputs (all three comparisons false) route full gradient
                // to both sides.
                let gt = a.gt(&b)?.to_dtype(g.dtype())?;
                let lt = a.lt(&b)?.to_dtype(g.dtype())?;
                let eq = a.eq_tensor(&b)?.to_dtype(g.dtype())?;
                let half_eq = eq.mul_scalar(0.5)?;
                let neither =
                    Tensor::neg(&Tensor::add(&Tensor::add(&gt, &lt)?, &eq)?)?.add_scalar(1.0)?;
                let ga_mask = Tensor::add(&Tensor::add(&lt, &half_eq)?, &neither)?;
                let gb_mask = Tensor::add(&Tensor::add(&gt, &half_eq)?, &neither)?;
                let ga = if a.requires_grad() {
                    Some(g.mul(&ga_mask)?.sum_to_shape(a.shape())?)
                } else {
                    None
                };
                let gb = if b.requires_grad() {
                    Some(g.mul(&gb_mask)?.sum_to_shape(b.shape())?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    pub fn pow(&self, exponent: &Tensor) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(exponent, "pow")?;
        let s = self
            .storage()
            .binary_op(&la, exponent.storage(), &lb, BinaryOp::Pow)?;
        let a = self.clone();
        let b = exponent.clone();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "pow",
            vec![self.clone(), exponent.clone()],
            move |g| {
                let ga = if a.requires_grad() {
                    let b_minus_1 = b.sub_scalar(1.0)?;
                    let raw = Tensor::mul(&g.mul(&b)?, &a.pow(&b_minus_1)?)?;
                    // Torch parity: 0^0 has base-gradient 0 (raw form gives
                    // 0 * inf = NaN); every other point keeps the raw form.
                    let a_zero = a.eq_tensor(&a.zeros_like()?)?;
                    let b_zero = b.eq_tensor(&b.zeros_like()?)?;
                    let both_zero = Tensor::mul(&a_zero, &b_zero)?;
                    Some(
                        Tensor::where_cond(&both_zero, &g.zeros_like()?, &raw)?
                            .sum_to_shape(a.shape())?,
                    )
                } else {
                    None
                };
                let gb = if b.requires_grad() {
                    let raw = Tensor::mul(&g.mul(&a.pow(&b)?)?, &a.ln()?)?;
                    // Torch parity: 0^e has exp-gradient 0 for e >= 0; the
                    // raw result * log(0) form only survives for e < 0.
                    let a_zero = a.eq_tensor(&a.zeros_like()?)?;
                    let b_nonneg = b.ge(&b.zeros_like()?)?;
                    let corner = Tensor::mul(&a_zero, &b_nonneg)?;
                    Some(
                        Tensor::where_cond(&corner, &g.zeros_like()?, &raw)?
                            .sum_to_shape(b.shape())?,
                    )
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    fn cmp(&self, other: &Tensor, op: CmpOp) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "cmp")?;
        let s = self.storage().cmp_op(&la, other.storage(), &lb, op)?;
        Ok(Tensor::from_storage(s, Layout::contiguous(out_shape)))
    }

    pub fn eq_tensor(&self, other: &Tensor) -> Result<Tensor> {
        self.cmp(other, CmpOp::Eq)
    }

    pub fn ne_tensor(&self, other: &Tensor) -> Result<Tensor> {
        self.cmp(other, CmpOp::Ne)
    }

    pub fn lt(&self, other: &Tensor) -> Result<Tensor> {
        self.cmp(other, CmpOp::Lt)
    }

    pub fn le(&self, other: &Tensor) -> Result<Tensor> {
        self.cmp(other, CmpOp::Le)
    }

    pub fn gt(&self, other: &Tensor) -> Result<Tensor> {
        self.cmp(other, CmpOp::Gt)
    }

    pub fn ge(&self, other: &Tensor) -> Result<Tensor> {
        self.cmp(other, CmpOp::Ge)
    }

    fn bit(&self, other: &Tensor, op: BitOp) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "bitwise_op")?;
        let s = self.storage().bitwise_op(&la, other.storage(), &lb, op)?;
        Ok(Tensor::from_storage(s, Layout::contiguous(out_shape)))
    }

    /// Element-wise `&` on integer/bool tensors (same dtype, like torch).
    pub fn bitwise_and(&self, other: &Tensor) -> Result<Tensor> {
        self.bit(other, BitOp::And)
    }

    /// Element-wise `|` on integer/bool tensors (same dtype, like torch).
    pub fn bitwise_or(&self, other: &Tensor) -> Result<Tensor> {
        self.bit(other, BitOp::Or)
    }

    /// Element-wise `^` on integer/bool tensors (same dtype, like torch).
    pub fn bitwise_xor(&self, other: &Tensor) -> Result<Tensor> {
        self.bit(other, BitOp::Xor)
    }

    /// Element-wise `!` on integer/bool tensors (like torch).
    pub fn bitwise_not(&self) -> Result<Tensor> {
        let s = self.storage().bitwise_not(self.layout())?;
        Ok(Tensor::from_storage(
            s,
            Layout::contiguous(self.shape().to_vec()),
        ))
    }

    fn logic(&self, other: &Tensor, op: LogicOp) -> Result<Tensor> {
        let (la, lb, out_shape) = self.broadcast_pair(other, "logical_op")?;
        let s = self.storage().logical_op(&la, other.storage(), &lb, op)?;
        Ok(Tensor::from_storage(s, Layout::contiguous(out_shape)))
    }

    /// Element-wise logical `and` with `!= 0` truthiness on any dtype.
    pub fn logical_and(&self, other: &Tensor) -> Result<Tensor> {
        self.logic(other, LogicOp::And)
    }

    /// Element-wise logical `or` with `!= 0` truthiness on any dtype.
    pub fn logical_or(&self, other: &Tensor) -> Result<Tensor> {
        self.logic(other, LogicOp::Or)
    }

    /// Element-wise logical `xor` with `!= 0` truthiness on any dtype.
    pub fn logical_xor(&self, other: &Tensor) -> Result<Tensor> {
        self.logic(other, LogicOp::Xor)
    }

    fn pred(&self, op: PredOp) -> Result<Tensor> {
        let s = self.storage().pred_op(self.layout(), op)?;
        Ok(Tensor::from_storage(
            s,
            Layout::contiguous(self.shape().to_vec()),
        ))
    }

    /// Element-wise NaN check (integers/bools are never NaN, like torch).
    pub fn isnan(&self) -> Result<Tensor> {
        self.pred(PredOp::IsNan)
    }

    /// Element-wise infinity check (integers/bools never are, like torch).
    pub fn isinf(&self) -> Result<Tensor> {
        self.pred(PredOp::IsInf)
    }

    /// Element-wise finiteness check (integers/bools always are, like torch).
    pub fn isfinite(&self) -> Result<Tensor> {
        self.pred(PredOp::IsFinite)
    }

    /// Ternary conditional selection `where(cond, on_true, on_false)` with full 3-way broadcasting.
    pub fn where_cond(cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        let s1 = broadcast_shapes("where_cond", cond.shape(), on_true.shape())?;
        let out_shape = broadcast_shapes("where_cond", &s1, on_false.shape())?;
        let lc = cond.layout().broadcast_to(&out_shape)?;
        let lt = on_true.layout().broadcast_to(&out_shape)?;
        let lf = on_false.layout().broadcast_to(&out_shape)?;

        let s = Storage::where_cond(
            cond.storage(),
            &lc,
            on_true.storage(),
            &lt,
            on_false.storage(),
            &lf,
        )?;
        let c = cond.clone();
        let t = on_true.clone();
        let f = on_false.clone();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "where_cond",
            vec![on_true.clone(), on_false.clone()],
            move |g| {
                let z = g.zeros_like()?;
                let gt = if t.requires_grad() {
                    Some(Tensor::where_cond(&c, g, &z)?.sum_to_shape(t.shape())?)
                } else {
                    None
                };
                let gf = if f.requires_grad() {
                    Some(Tensor::where_cond(&c, &z, g)?.sum_to_shape(f.shape())?)
                } else {
                    None
                };
                Ok(vec![gt, gf])
            },
        ))
    }

    /// Replaces elements where `mask` is true (`1` / `true`) with `value`.
    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Result<Tensor> {
        let fill = self.full_like(value)?;
        Tensor::where_cond(mask, &fill, self)
    }

    /// Sorted flat positions where `mask` (broadcast against `self`) is true.
    /// Masks must be boolean on the input's device, like torch.
    fn mask_true_positions(&self, op: &'static str, mask: &Tensor) -> Result<Vec<i64>> {
        if mask.dtype() != DType::Bool {
            return Err(Error::UnsupportedDType {
                op,
                dtype: mask.dtype(),
            });
        }
        if mask.device() != self.device() {
            return Err(Error::DeviceMismatch {
                op,
                lhs: self.device().to_string(),
                rhs: mask.device().to_string(),
            });
        }
        let bits = mask.broadcast_to(self.shape())?.to_vec::<bool>()?;
        let mut pos = Vec::new();
        for (idx, &b) in bits.iter().enumerate() {
            if b {
                pos.push(idx as i64);
            }
        }
        Ok(pos)
    }

    /// Elements where `mask` is true, as a 1-D tensor (torch.masked_select
    /// parity). The mask broadcasts against the input; gradients scatter
    /// back to the selected positions (via `index_select`).
    pub fn masked_select(&self, mask: &Tensor) -> Result<Tensor> {
        let pos = self.mask_true_positions("masked_select", mask)?;
        let flat = self.reshape(&[self.numel()])?;
        let index = Tensor::from_slice_on(&pos, &[pos.len()], &self.device())?;
        flat.index_select(0, &index)
    }

    /// Copy of `self` with `source` (flattened) written at true mask positions
    /// (torch.masked_scatter parity). Extra source elements are ignored; too
    /// few fail closed. Differentiable in both operands (via `scatter`).
    pub fn masked_scatter(&self, mask: &Tensor, source: &Tensor) -> Result<Tensor> {
        let pos = self.mask_true_positions("masked_scatter", mask)?;
        if source.dtype() != self.dtype() {
            return Err(Error::DTypeMismatch {
                op: "masked_scatter",
                expected: self.dtype(),
                got: source.dtype(),
            });
        }
        if source.numel() < pos.len() {
            return Err(Error::Msg(format!(
                "masked_scatter: number of elements of source ({}) < number of ones in mask ({})",
                source.numel(),
                pos.len()
            )));
        }
        let flat = self.reshape(&[self.numel()])?;
        let index = Tensor::from_slice_on(&pos, &[pos.len()], &self.device())?;
        let src = source.reshape(&[source.numel()])?.narrow(0, 0, pos.len())?;
        flat.scatter(0, &index, &src)?.reshape(self.shape())
    }

    /// Fills slices along `dim` selected by int64 `index` with `value`
    /// (torch.index_fill parity). Negative indices wrap; out-of-range
    /// indices fail closed. Gradients flow around the filled slices
    /// (via `where_cond`).
    pub fn index_fill(&self, dim: isize, index: &Tensor, value: f64) -> Result<Tensor> {
        let (ax, n) = if self.ndim() == 0 {
            if dim != 0 && dim != -1 {
                return Err(Error::InvalidAxis {
                    op: "index_fill",
                    axis: dim,
                    ndim: 0,
                });
            }
            (0usize, 1usize)
        } else {
            let ax = normalize_axis("index_fill", dim, self.ndim())?;
            (ax, self.shape()[ax])
        };
        if index.dtype() != DType::I64 {
            return Err(Error::UnsupportedDType {
                op: "index_fill",
                dtype: index.dtype(),
            });
        }
        if index.ndim() > 1 {
            return Err(Error::Msg(format!(
                "index_fill: index has to be a vector/scalar (got rank {})",
                index.ndim()
            )));
        }
        if index.device() != self.device() {
            return Err(Error::DeviceMismatch {
                op: "index_fill",
                lhs: self.device().to_string(),
                rhs: index.device().to_string(),
            });
        }
        let mut bits = vec![false; n];
        for &i in index.to_vec::<i64>()?.iter() {
            if i < -(n as i64) || i >= n as i64 {
                return Err(Error::IndexOutOfBounds {
                    op: "index_fill",
                    index: i as isize,
                    dim_size: n,
                });
            }
            bits[(((i % n as i64) + n as i64) % n as i64) as usize] = true;
        }
        // Axis-marked mask shape (0-d stays scalar: [1] does not broadcast to []).
        let mut mshape = self.shape().to_vec();
        if !mshape.is_empty() {
            mshape.iter_mut().for_each(|d| *d = 1);
            mshape[ax] = n;
        }
        let m = Tensor::from_slice_on(&bits, &mshape, &self.device())?
            .broadcast_to(self.shape())?;
        Tensor::where_cond(&m, &self.full_like(value)?, self)
    }

    // =========================================================================
    // 4. Reductions & Softmax Family
    // =========================================================================

    /// Sums all elements into a 0-D scalar tensor.
    pub fn sum(&self) -> Result<Tensor> {
        if self.ndim() == 0 {
            return Ok(self.clone());
        }
        let axes: Vec<isize> = (0..self.ndim() as isize).collect();
        self.sum_axes(&axes, false)
    }

    /// Mean of all elements into a 0-D scalar tensor.
    pub fn mean(&self) -> Result<Tensor> {
        let n = self.numel() as f64;
        self.sum()?.div_scalar(n)
    }

    /// Sums along one or more axes.
    pub fn sum_axes(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        // torch parity: sums of bool and of integers narrower than 64 bits accumulate in int64
        if (self.dtype().is_int() || self.dtype() == DType::Bool) && self.dtype() != DType::I64 {
            return self.to_dtype(DType::I64)?.sum_axes(axes, keepdim);
        }
        let ndim = self.ndim();
        let mut norm_axes = Vec::with_capacity(axes.len());
        for &a in axes {
            let na = normalize_axis("sum_axes", a, ndim)?;
            if !norm_axes.contains(&na) {
                norm_axes.push(na);
            }
        }
        norm_axes.sort_unstable();

        let s = self
            .storage()
            .reduce_op(self.layout(), &norm_axes, ReduceOp::Sum)?;
        let mut kept_shape = self.shape().to_vec();
        for &a in &norm_axes {
            kept_shape[a] = 1;
        }
        let out_shape: Vec<usize> = if keepdim {
            kept_shape.clone()
        } else {
            self.shape()
                .iter()
                .enumerate()
                .filter(|(i, _)| !norm_axes.contains(i))
                .map(|(_, &d)| d)
                .collect()
        };

        let src_shape = self.shape().to_vec();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "sum_axes",
            vec![self.clone()],
            move |g| {
                let g_kept = g.reshape(&kept_shape)?;
                Ok(vec![Some(g_kept.broadcast_to(&src_shape)?)])
            },
        ))
    }

    pub fn sum_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.sum_axes(&[axis], keepdim)
    }

    pub fn mean_axes(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        // Deduplicate like sum_axes does: duplicate axes must not inflate the count.
        let mut seen = Vec::with_capacity(axes.len());
        let mut count = 1usize;
        for &a in axes {
            let na = normalize_axis("mean_axes", a, self.ndim())?;
            if !seen.contains(&na) {
                seen.push(na);
                count *= self.shape()[na];
            }
        }
        self.sum_axes(axes, keepdim)?.div_scalar(count as f64)
    }

    pub fn mean_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.mean_axes(&[axis], keepdim)
    }

    /// Count of non-NaN elements along `axes` (shared helper for nan-reductions).
    /// Casts to i64 BEFORE summing: `sum_axes` casts the result back to the
    /// input dtype, so a bool sum would collapse to true/false instead of a count.
    fn count_non_nan(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        self.isnan()?
            .logical_not()?
            .to_dtype(DType::I64)?
            .sum_axes(axes, keepdim)
    }

    /// Sum of non-NaN elements along `axes` (torch.nansum parity: NaN reads as zero).
    /// Empty `axes` is a no-op clone, matching `sum_axes` (and torch's `dim=()`).
    pub fn nansum_axes(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let filled = if self.dtype().is_floating_point() {
            self.masked_fill(&self.isnan()?, 0.0)?
        } else {
            self.clone()
        };
        filled.sum_axes(axes, keepdim)
    }

    /// Sum of all non-NaN elements (0 when every element is NaN, like torch).
    pub fn nansum(&self) -> Result<Tensor> {
        let filled = if self.dtype().is_floating_point() {
            self.masked_fill(&self.isnan()?, 0.0)?
        } else {
            self.clone()
        };
        filled.sum()
    }

    /// Mean of non-NaN elements along `axes` (torch.nanmean parity; all-NaN reads NaN).
    /// Float dtypes only: torch raises on integer input, so this fails closed too.
    pub fn nanmean_axes(&self, axes: &[isize], keepdim: bool) -> Result<Tensor> {
        if !self.dtype().is_floating_point() {
            return Err(Error::UnsupportedDType {
                op: "nanmean",
                dtype: self.dtype(),
            });
        }
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let total = self.masked_fill(&self.isnan()?, 0.0)?.sum_axes(axes, keepdim)?;
        let count = self.count_non_nan(axes, keepdim)?.to_dtype(self.dtype())?;
        total.div(&count)
    }

    /// Mean of all non-NaN elements (NaN when every element is NaN, like torch).
    pub fn nanmean(&self) -> Result<Tensor> {
        if !self.dtype().is_floating_point() {
            return Err(Error::UnsupportedDType {
                op: "nanmean",
                dtype: self.dtype(),
            });
        }
        if self.ndim() == 0 {
            return Ok(self.clone());
        }
        let axes: Vec<isize> = (0..self.ndim() as isize).collect();
        self.nanmean_axes(&axes, false)
    }

    /// Median of non-NaN elements along `axis` (torch.nanmedian parity).
    /// Even valid counts take the LOWER middle (torch does not average); `indices`
    /// point into the original tensor along `axis`. Differentiable wrt values
    /// (gradient flows to the chosen element only, like torch).
    pub fn nanmedian_axis(&self, axis: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let ax = normalize_axis("nanmedian", axis, self.ndim())?;
        self.require_nonempty("nanmedian", ax)?;
        let n = self.shape()[ax];
        let (sorted, order) = self.sort(axis, false)?;
        // Lower-middle position (count-1)/2 in f64 (exact to 2^53); all-NaN
        // slices clamp to 0, which reads NaN since NaN sorts last.
        let count = self.count_non_nan(&[axis], keepdim)?.to_dtype(DType::F64)?;
        let mid = count
            .sub_scalar(1.0)?
            .div_scalar(2.0)?
            .floor()?
            .clamp(0.0, (n - 1) as f64)?
            .to_dtype(DType::I64)?;
        let mid_full = if keepdim {
            mid
        } else {
            mid.unsqueeze(ax as isize)?
        };
        let values = sorted.gather(axis, &mid_full)?;
        let indices = order.gather(axis, &mid_full)?;
        if keepdim {
            return Ok((values, indices));
        }
        let mut out_shape = self.shape().to_vec();
        out_shape.remove(ax);
        Ok((values.reshape(&out_shape)?, indices.reshape(&out_shape)?))
    }

    /// Median of all non-NaN elements, values only (like torch.nanmedian(x)).
    pub fn nanmedian(&self) -> Result<Tensor> {
        // Full flatten (`flatten()` keeps dim 0, so reshape explicitly).
        let flat = self.reshape(&[self.numel()])?;
        Ok(flat.nanmedian_axis(0, false)?.0)
    }

    /// Median along `axis` (torch.median parity): the LOWER middle for odd
    /// AND even counts (torch never averages, unlike numpy). NaN propagates:
    /// any NaN along the dim wins, with values/indices/gradients addressing
    /// the FIRST NaN. Returns `(values, indices)`; indices point into the
    /// original tensor along `axis`.
    pub fn median_axis(&self, axis: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let ax = normalize_axis("median", axis, self.ndim())?;
        self.require_nonempty("median", ax)?;
        let n = self.shape()[ax];
        let (sorted, order) = self.sort(axis, false)?;
        let mut mid_shape = self.shape().to_vec();
        if keepdim {
            mid_shape[ax] = 1;
        } else {
            mid_shape.remove(ax);
        }
        let mid = Tensor::full(&mid_shape, ((n - 1) / 2) as f64, DType::I64, &self.device())?;
        let mid_full = if keepdim {
            mid
        } else {
            mid.unsqueeze(ax as isize)?
        };
        let lower_vals = sorted.gather(axis, &mid_full)?;
        let lower_idx = order.gather(axis, &mid_full)?;
        let done = |v: Tensor, i: Tensor| -> Result<(Tensor, Tensor)> {
            if keepdim {
                return Ok((v, i));
            }
            let mut out_shape = self.shape().to_vec();
            out_shape.remove(ax);
            Ok((v.reshape(&out_shape)?, i.reshape(&out_shape)?))
        };
        if !self.dtype().is_floating_point() {
            // Integers/bools hold no NaN: the lower middle is the answer
            // (bool sorts although torch rejects it: engine superset).
            return done(lower_vals, lower_idx);
        }
        // NaN branch: argmax keeps the FIRST maximum, i.e. the first NaN;
        // `where_cond` routes values AND gradients there (i64 cond reads
        // nonzero-truthy, matching `masked_fill` int masks).
        let is_nan = self.isnan()?.to_dtype(DType::I64)?;
        let nan_count = is_nan.sum_axes(&[axis], keepdim)?;
        let nan_full = if keepdim {
            nan_count
        } else {
            nan_count.unsqueeze(ax as isize)?
        };
        let first_nan = is_nan.argmax_axis(axis, keepdim)?;
        let first_full = if keepdim {
            first_nan
        } else {
            first_nan.unsqueeze(ax as isize)?
        };
        let nan_vals = self.gather(axis, &first_full)?;
        let values = Tensor::where_cond(&nan_full, &nan_vals, &lower_vals)?;
        let indices = Tensor::where_cond(&nan_full, &first_full, &lower_idx)?;
        done(values, indices)
    }

    /// Median of all elements, values only (like torch.median(x)).
    pub fn median(&self) -> Result<Tensor> {
        let flat = self.reshape(&[self.numel()])?;
        Ok(flat.median_axis(0, false)?.0)
    }

    /// Scalar rank position for one q over `n` elements: `(lo, hi, frac)`.
    /// q in the input width (torch casts scalar q to the input dtype), the
    /// position in f64 (torch-verified: near-integral f32 hits like
    /// 0.9*140 land just below the integer and bracket downwards).
    fn quantile_pos(q: f64, n: usize, f32: bool, m: QuantileMethod) -> (usize, usize, f64) {
        let qw = if f32 { (q as f32) as f64 } else { q };
        let pos = qw * ((n - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        match m {
            QuantileMethod::Linear | QuantileMethod::Midpoint => (lo, hi, frac),
            QuantileMethod::Lower => (lo, lo, 0.0),
            QuantileMethod::Higher => (hi, hi, 0.0),
            QuantileMethod::Nearest => {
                let nr = ((pos + 0.5).floor() as usize).min(n - 1);
                (nr, nr, 0.0)
            }
        }
    }

    /// Gathers one position per slice from a sorted tensor, returned at the
    /// reduced (`out_shape`) or kept shape.
    fn gather_sorted_at(
        sorted: &Tensor,
        axis: isize,
        ax: usize,
        idx: &Tensor,
        keepdim: bool,
        out_shape: &[usize],
    ) -> Result<Tensor> {
        let full_idx = if keepdim {
            idx.clone()
        } else {
            idx.unsqueeze(ax as isize)?
        };
        let g = sorted.gather(axis, &full_idx)?;
        if keepdim {
            Ok(g)
        } else {
            g.reshape(out_shape)
        }
    }

    /// Shared validation for the quantile family: f32/f64 input (torch
    /// rejects everything else, including f16) and q in [0, 1].
    fn check_quantile(&self, op: &'static str, qs: &[f64]) -> Result<()> {
        if !matches!(self.dtype(), DType::F32 | DType::F64) {
            return Err(Error::UnsupportedDType {
                op,
                dtype: self.dtype(),
            });
        }
        for &q in qs {
            if !(0.0..=1.0).contains(&q) {
                return Err(Error::Msg(format!(
                    "quantile() q must be in the range [0, 1] but got {q}"
                )));
            }
        }
        Ok(())
    }

    /// Quantiles along `axis` (torch.quantile parity): one output per q with
    /// the q dim prepended (`[qs.len(), *reduced]`). NaN propagates (any NaN
    /// wins; gradients address the first NaN). Differentiable: linear splits
    /// `(1-frac)/frac` across the bracketing values, midpoint splits evenly.
    pub fn quantile_axis(
        &self,
        axis: isize,
        qs: &[f64],
        keepdim: bool,
        method: &str,
    ) -> Result<Tensor> {
        self.check_quantile("quantile", qs)?;
        let ax = normalize_axis("quantile", axis, self.ndim())?;
        self.require_nonempty("quantile", ax)?;
        let m = QuantileMethod::parse(method)?;
        let n = self.shape()[ax];
        let f32 = self.dtype() == DType::F32;
        let mut out_shape = self.shape().to_vec();
        if keepdim {
            out_shape[ax] = 1;
        } else {
            out_shape.remove(ax);
        }
        if qs.is_empty() {
            let mut es = vec![0usize];
            es.extend_from_slice(&out_shape);
            return Tensor::zeros(&es, self.dtype(), &self.device());
        }
        let (sorted, _) = self.sort(axis, false)?;
        let is_nan = self.isnan()?.to_dtype(DType::I64)?;
        let nan_count = is_nan.sum_axes(&[axis], keepdim)?;
        let first_nan = is_nan.argmax_axis(axis, keepdim)?;
        let nan_v = Self::gather_sorted_at(self, axis, ax, &first_nan, keepdim, &out_shape)?;
        let mut parts = Vec::with_capacity(qs.len());
        for &q in qs {
            let (lo, hi, frac) = Self::quantile_pos(q, n, f32, m);
            let lo_t = Tensor::full(&out_shape, lo as f64, DType::I64, &self.device())?;
            let lo_v = Self::gather_sorted_at(&sorted, axis, ax, &lo_t, keepdim, &out_shape)?;
            let combined = match m {
                QuantileMethod::Lower | QuantileMethod::Nearest => lo_v,
                QuantileMethod::Higher => {
                    let hi_t = Tensor::full(&out_shape, hi as f64, DType::I64, &self.device())?;
                    Self::gather_sorted_at(&sorted, axis, ax, &hi_t, keepdim, &out_shape)?
                }
                QuantileMethod::Linear => {
                    let hi_t = Tensor::full(&out_shape, hi as f64, DType::I64, &self.device())?;
                    let hi_v =
                        Self::gather_sorted_at(&sorted, axis, ax, &hi_t, keepdim, &out_shape)?;
                    // UFCS: method-syntax add/sub resolve to the by-value
                    // std::ops traits, which would move `lo_v`.
                    Tensor::add(&lo_v, &Tensor::sub(&hi_v, &lo_v)?.mul_scalar(frac)?)?
                }
                QuantileMethod::Midpoint => {
                    let hi_t = Tensor::full(&out_shape, hi as f64, DType::I64, &self.device())?;
                    let hi_v =
                        Self::gather_sorted_at(&sorted, axis, ax, &hi_t, keepdim, &out_shape)?;
                    // Overflow-safe average (lo+hi overflows near f32::MAX).
                    Tensor::add(&lo_v, &Tensor::sub(&hi_v, &lo_v)?.mul_scalar(0.5)?)?
                }
            };
            parts.push(Tensor::where_cond(&nan_count, &nan_v, &combined)?);
        }
        Tensor::stack(&parts, 0)
    }

    /// Quantiles of all elements (like torch.quantile(x, q)).
    pub fn quantile(&self, qs: &[f64], method: &str) -> Result<Tensor> {
        let flat = self.reshape(&[self.numel()])?;
        flat.quantile_axis(0, qs, false, method)
    }

    /// NaN-skipping quantiles along `axis` (torch.nanquantile parity): the
    /// rank position uses each slice's own valid count; all-NaN reads NaN.
    pub fn nanquantile_axis(
        &self,
        axis: isize,
        qs: &[f64],
        keepdim: bool,
        method: &str,
    ) -> Result<Tensor> {
        self.check_quantile("nanquantile", qs)?;
        let ax = normalize_axis("nanquantile", axis, self.ndim())?;
        self.require_nonempty("nanquantile", ax)?;
        let m = QuantileMethod::parse(method)?;
        let n = self.shape()[ax];
        let mut out_shape = self.shape().to_vec();
        if keepdim {
            out_shape[ax] = 1;
        } else {
            out_shape.remove(ax);
        }
        if qs.is_empty() {
            let mut es = vec![0usize];
            es.extend_from_slice(&out_shape);
            return Tensor::zeros(&es, self.dtype(), &self.device());
        }
        let (sorted, _) = self.sort(axis, false)?;
        // Positions in f64 with q in the input width (see `quantile_pos`).
        let f32 = self.dtype() == DType::F32;
        let span = self
            .count_non_nan(&[axis], keepdim)?
            .to_dtype(DType::F64)?
            .sub_scalar(1.0)?;
        let mut parts = Vec::with_capacity(qs.len());
        for &q in qs {
            let qw = if f32 { (q as f32) as f64 } else { q };
            let pos = span.mul_scalar(qw)?;
            let lo_f = pos.floor()?;
            let frac = Tensor::sub(&pos, &lo_f)?.to_dtype(self.dtype())?;
            let lo_idx = lo_f.clamp(0.0, (n - 1) as f64)?.to_dtype(DType::I64)?;
            let lo_v = Self::gather_sorted_at(&sorted, axis, ax, &lo_idx, keepdim, &out_shape)?;
            let combined = match m {
                QuantileMethod::Lower => lo_v,
                QuantileMethod::Higher => {
                    let hi_idx = pos.ceil()?.clamp(0.0, (n - 1) as f64)?.to_dtype(DType::I64)?;
                    Self::gather_sorted_at(&sorted, axis, ax, &hi_idx, keepdim, &out_shape)?
                }
                QuantileMethod::Nearest => {
                    let nr_idx = pos
                        .add_scalar(0.5)?
                        .floor()?
                        .clamp(0.0, (n - 1) as f64)?
                        .to_dtype(DType::I64)?;
                    Self::gather_sorted_at(&sorted, axis, ax, &nr_idx, keepdim, &out_shape)?
                }
                QuantileMethod::Linear => {
                    let hi_idx = pos.ceil()?.clamp(0.0, (n - 1) as f64)?.to_dtype(DType::I64)?;
                    let hi_v =
                        Self::gather_sorted_at(&sorted, axis, ax, &hi_idx, keepdim, &out_shape)?;
                    Tensor::add(&lo_v, &Tensor::mul(&Tensor::sub(&hi_v, &lo_v)?, &frac)?)?
                }
                QuantileMethod::Midpoint => {
                    let hi_idx = pos.ceil()?.clamp(0.0, (n - 1) as f64)?.to_dtype(DType::I64)?;
                    let hi_v =
                        Self::gather_sorted_at(&sorted, axis, ax, &hi_idx, keepdim, &out_shape)?;
                    Tensor::add(&lo_v, &Tensor::sub(&hi_v, &lo_v)?.mul_scalar(0.5)?)?
                }
            };
            parts.push(combined);
        }
        Tensor::stack(&parts, 0)
    }

    /// NaN-skipping quantiles of all elements (like torch.nanquantile(x, q)).
    pub fn nanquantile(&self, qs: &[f64], method: &str) -> Result<Tensor> {
        let flat = self.reshape(&[self.numel()])?;
        flat.nanquantile_axis(0, qs, false, method)
    }

    /// max / min / argmax / argmin have no value over an empty axis (it used to index out of bounds and panic).
    fn require_nonempty(&self, op: &str, ax: usize) -> Result<()> {
        if self.shape()[ax] == 0 {
            return Err(Error::Msg(format!(
                "{op}: cannot reduce over an empty dimension (axis {ax} has size 0)"
            )));
        }
        Ok(())
    }

    pub fn max_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.extrema_axis(axis, keepdim, ReduceOp::Max)
    }

    pub fn min_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.extrema_axis(axis, keepdim, ReduceOp::Min)
    }

    fn extrema_axis(&self, axis: isize, keepdim: bool, op: ReduceOp) -> Result<Tensor> {
        let ax = normalize_axis("extrema_axis", axis, self.ndim())?;
        self.require_nonempty("max/min", ax)?;
        let s = self.storage().reduce_op(self.layout(), &[ax], op)?;
        let mut kept_shape = self.shape().to_vec();
        kept_shape[ax] = 1;
        let mut out_shape = self.shape().to_vec();
        if keepdim {
            out_shape[ax] = 1;
        } else {
            out_shape.remove(ax);
        }
        let inp = self.clone();
        let out_for_bwd = Tensor::from_storage(s.clone(), Layout::contiguous(kept_shape.clone()));
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "extrema_axis",
            vec![self.clone()],
            move |g| {
                let g_kept = g.reshape(&kept_shape)?;
                // torch.max(dim) / min(dim) send the whole gradient to the first extremal element
                // (amax / amin / global max split it evenly)
                let mask = inp.eq_tensor(&out_for_bwd)?.to_dtype(g.dtype())?;
                // first tied position, with elementwise / reduce ops only (stays on the tensor's device):
                // candidates carry their index along the axis, non-extremal elements carry `dim`
                let dim = inp.shape()[ax];
                let mut pos_shape = vec![1usize; inp.ndim()];
                pos_shape[ax] = dim;
                let pos = Tensor::from_slice_on(&(0..dim).map(|i| i as f64).collect::<Vec<_>>(), &pos_shape, &inp.device())?.to_dtype(g.dtype())?;
                let cand = pos.mul(&mask)?.add(&mask.neg()?.add_scalar(1.0)?.mul_scalar(dim as f64)?)?;
                let first_idx = cand.min_axis(ax as isize, true)?;
                let first = cand.eq_tensor(&first_idx)?.to_dtype(g.dtype())?.mul(&mask)?;
                Ok(vec![Some(g_kept.mul(&first)?)])
            },
        ))
    }

    pub fn argmax_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        let ax = normalize_axis("argmax_axis", axis, self.ndim())?;
        self.require_nonempty("argmax", ax)?;
        let s = self
            .storage()
            .reduce_op(self.layout(), &[ax], ReduceOp::ArgMax)?;
        let mut out_shape = self.shape().to_vec();
        if keepdim {
            out_shape[ax] = 1;
        } else {
            out_shape.remove(ax);
        }
        Ok(Tensor::from_storage(s, Layout::contiguous(out_shape)))
    }

    pub fn argmin_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        let ax = normalize_axis("argmin_axis", axis, self.ndim())?;
        self.require_nonempty("argmin", ax)?;
        let s = self
            .storage()
            .reduce_op(self.layout(), &[ax], ReduceOp::ArgMin)?;
        let mut out_shape = self.shape().to_vec();
        if keepdim {
            out_shape[ax] = 1;
        } else {
            out_shape.remove(ax);
        }
        Ok(Tensor::from_storage(s, Layout::contiguous(out_shape)))
    }

    /// Variance along `axis` (with Bessel's correction if `unbiased` is `true`).
    /// An unbiased estimate over 1 element is undefined and yields NaN, like PyTorch
    /// (the biased fallback it used to take silently claimed zero variance).
    pub fn var_axis(&self, axis: isize, unbiased: bool, keepdim: bool) -> Result<Tensor> {
        let ax = normalize_axis("var_axis", axis, self.ndim())?;
        let n = self.shape()[ax];
        let denom = if unbiased {
            n.saturating_sub(1) as f64
        } else {
            n as f64
        };
        let mu = self.mean_axis(ax as isize, true)?;
        let diff = self.sub(&mu)?;
        diff.square()?
            .sum_axis(ax as isize, keepdim)?
            .div_scalar(denom)
    }

    /// Standard deviation along `axis`: `sqrt(var + 1e-12)`. The epsilon keeps
    /// gradients finite at zero variance (so `std` of a constant is `1e-6`,
    /// where PyTorch returns 0).
    pub fn std_axis(&self, axis: isize, unbiased: bool, keepdim: bool) -> Result<Tensor> {
        self.var_axis(axis, unbiased, keepdim)?
            .add_scalar(1e-12)?
            .sqrt()
    }

    /// Numerically stable `log(sum(exp(x), axis))`. Slices whose max is `+inf`
    /// return `+inf` and all-`-inf` slices return `-inf` (PyTorch agrees on
    /// both values); the gradient there is zero instead of PyTorch's NaN,
    /// and NaN still propagates.
    pub fn logsumexp(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        let m = no_grad(|| self.max_axis(axis, true))?;
        let shifted = self.sub(&m)?;
        if !shifted.dtype().is_floating_point() {
            let lse = shifted.exp()?.sum_axis(axis, keepdim)?.ln()?;
            return if keepdim {
                lse.add(&m)
            } else {
                lse.add(&m.squeeze_axis(axis)?)
            };
        }
        // Fast path: a finite max means the plain shift trick is exact, with
        // no masking passes at all (NaN elements still propagate as NaN).
        if m.to_vec::<f64>()?.iter().all(|v| v.is_finite()) {
            let lse = shifted.exp()?.sum_axis(axis, keepdim)?.ln()?;
            return if keepdim {
                lse.add(&m)
            } else {
                lse.add(&m.squeeze_axis(axis)?)
            };
        }
        // A non-finite max makes the `x - max` shift NaN (`inf - inf`): stand
        // those elements down to 0 so `exp/sum/ln` stay finite (`ln(n) + m`
        // is still the right answer, with a zero gradient). NaN elements are
        // left alone so NaN propagates instead of being silenced.
        let nonfinite_max = m.abs()?.eq_tensor(&m.full_like(f64::INFINITY)?)?;
        let x_is_nan = self.ne_tensor(self)?;
        let safe = Tensor::where_cond(&x_is_nan, &shifted, &shifted.full_like(0.0)?)?;
        let safe_shifted = Tensor::where_cond(&nonfinite_max, &safe, &shifted)?;
        let lse = safe_shifted.exp()?.sum_axis(axis, true)?.ln()?.add(&m)?;
        if keepdim {
            Ok(lse)
        } else {
            lse.squeeze_axis(axis)
        }
    }

    /// Numerically stable log-softmax along `axis`. Rows whose max is `+inf`
    /// return `ln(1/k)` at the `k` `+inf` entries and `-inf` elsewhere
    /// (PyTorch returns NaN there); all-`-inf` rows stay NaN like PyTorch.
    pub fn log_softmax(&self, axis: isize) -> Result<Tensor> {
        if let Some(x) = crate::amp::fp32_input(self)? {
            return x.log_softmax(axis);
        }
        if !self.dtype().is_floating_point() {
            return self.sub(&self.logsumexp(axis, true)?);
        }
        // (x - max) - ln(sum(exp(x - max))): subtracting the max first keeps full precision for large magnitudes
        // (`x - (max + ln(sum))` would absorb the ln term into `max`). Only rows with a finite maximum use it; rows
        // with +inf / NaN / all -inf keep the original path below, whose gradient is zero instead of NaN.
        let axis_max = no_grad(|| self.max_axis(axis, true))?;
        let row_ok = axis_max.isfinite()?;
        let fast = |x: &Tensor, m: &Tensor| -> Result<Tensor> {
            let shifted = x.sub(m)?;
            let log_sum = shifted.exp()?.sum_axis(axis, true)?.ln()?;
            Tensor::sub(&shifted, &log_sum)
        };
        if row_ok.all()?.to_vec::<f64>()?[0] != 0.0 {
            return fast(self, &axis_max);
        }
        let lse = self.logsumexp(axis, true)?;
        let out = self.sub(&lse)?;
        if !lse.to_vec::<f64>()?.contains(&f64::INFINITY) {
            // no +inf rows: only NaN / all -inf rows, which the plain subtraction already handles
            let ok_b = row_ok.broadcast_to(self.shape())?;
            let (xs, ms) = (Tensor::where_cond(&ok_b, self, &self.zeros_like()?)?, Tensor::where_cond(&row_ok, &axis_max, &axis_max.zeros_like()?)?);
            return Tensor::where_cond(&ok_b, &fast(&xs, &ms)?, &out);
        }
        let has_pos_inf = lse.eq_tensor(&lse.full_like(f64::INFINITY)?)?;
        // Recompute +inf rows in log space (`ln(e_safe) - ln(denom)` splits
        // `ln(1/k)` over the `+inf` entries); the branch is constant, so the
        // gradient there is zero.
        let x_is_inf = self.eq_tensor(&self.full_like(f64::INFINITY)?)?;
        let x_is_nan = self.ne_tensor(self)?;
        let one = out.full_like(1.0)?;
        let zero = out.full_like(0.0)?;
        let nan = out.full_like(f64::NAN)?;
        let mask = Tensor::where_cond(&x_is_inf, &one, &zero)?;
        let e_safe = Tensor::where_cond(&x_is_nan, &nan, &mask)?;
        let denom = e_safe.sum_axis(axis, true)?;
        let corrected = e_safe.ln()?.sub(&denom.ln()?)?;
        Tensor::where_cond(&has_pos_inf, &corrected, &out)
    }

    /// Numerically stable softmax along `axis`. Rows whose max is `+inf` return
    /// a uniform one-hot over the `+inf` entries with a zero gradient
    /// (PyTorch returns NaN here); all-`-inf` rows stay NaN like PyTorch,
    /// and NaN still poisons.
    pub fn softmax(&self, axis: isize) -> Result<Tensor> {
        if let Some(x) = crate::amp::fp32_input(self)? {
            return x.softmax(axis);
        }
        let ax = normalize_axis("softmax", axis, self.ndim())?;
        if self.ndim() > 0
            && ax == self.ndim() - 1
            && let Some(out_storage) = self.storage().try_softmax_last_axis(self.layout())
        {
            let out_layout = Layout::contiguous(self.shape());
            if !crate::tensor::is_grad_enabled() || !self.requires_grad() {
                return Ok(Tensor::from_parts(out_storage, out_layout, false, None));
            }
            let x_saved = self.clone();
            let y_val = Tensor::from_parts(out_storage.clone(), out_layout.clone(), false, None);
            let backward: crate::tensor::BackwardFn = Box::new(move |g: &Tensor| {
                let gy = g.mul(&y_val)?;
                let sum_gy = gy.sum_axis(ax as isize, true)?;
                let gx = gy.sub(&Tensor::mul(&y_val, &sum_gy)?)?;
                Ok(vec![Some(gx)])
            });
            return Ok(Tensor::from_parts(
                out_storage,
                out_layout,
                true,
                Some(crate::tensor::Node {
                    op: "softmax",
                    inputs: vec![x_saved],
                    backward,
                }),
            ));
        }
        let m = no_grad(|| self.max_axis(axis, true))?;
        let shifted = self.sub(&m)?;
        if !shifted.dtype().is_floating_point() {
            let e = shifted.exp()?;
            let denom = e.sum_axis(axis, true)?;
            return e.div(&denom);
        }
        // Fast path: a finite max means the plain shift trick is exact, with
        // no masking passes at all (NaN elements still propagate as NaN).
        if m.all_finite()? {
            let e = shifted.exp()?;
            let denom = e.sum_axis(axis, true)?;
            return e.div(&denom);
        }
        // `inf - inf` shifts (and their `exp`) are NaN, which would poison
        // even a zeroed-out gradient via 0*NaN: stand +inf-max rows down to
        // a finite shift and a 0/1 mask so normalizing gives the one-hot.
        // NaN entries are left alone so mixed +inf/NaN rows still poison.
        let has_pos_inf = m.eq_tensor(&m.full_like(f64::INFINITY)?)?;
        let x_is_inf = self.eq_tensor(&self.full_like(f64::INFINITY)?)?;
        let x_is_nan = self.ne_tensor(self)?;
        let one = shifted.full_like(1.0)?;
        let zero = shifted.full_like(0.0)?;
        let nan = shifted.full_like(f64::NAN)?;
        let safe = Tensor::where_cond(&x_is_nan, &shifted, &zero)?;
        let safe_shifted = Tensor::where_cond(&has_pos_inf, &safe, &shifted)?;
        let e = safe_shifted.exp()?;
        let mask = Tensor::where_cond(&x_is_inf, &one, &zero)?;
        let onehot = Tensor::where_cond(&x_is_nan, &nan, &mask)?;
        let e_safe = Tensor::where_cond(&has_pos_inf, &onehot, &e)?;
        let denom = e_safe.sum_axis(axis, true)?;
        e_safe.div(&denom)
    }

    // =========================================================================
    // 5. Linear Algebra & Einsum
    // =========================================================================

    /// Batched matrix multiplication `[..., m, k] @ [..., k, n] -> [..., m, n]`.
    /// Consumes strided/transposed layouts directly without copy!
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // autocast: matrix products run in the low-precision dtype
        if let Some((a, b)) = crate::amp::lowp_pair(self, other)? {
            return a.matmul(&b);
        }
        // Torch parity: 1D operands act as row/column vectors (backward flows
        // through the views for free).
        if self.ndim() == 1 && other.ndim() == 1 {
            return self
                .unsqueeze(0)?
                .matmul(&other.unsqueeze(1)?)?
                .squeeze_axis(0)?
                .squeeze_axis(0);
        }
        if self.ndim() == 1 {
            let out = self.unsqueeze(0)?.matmul(other)?;
            return out.squeeze_axis(out.ndim() as isize - 2);
        }
        if other.ndim() == 1 {
            let out = self.matmul(&other.unsqueeze(1)?)?;
            return out.squeeze_axis(out.ndim() as isize - 1);
        }
        let (s, out_shape) =
            self.storage()
                .matmul(self.layout(), other.storage(), other.layout())?;
        let a = self.clone();
        let b = other.clone();
        Ok(Tensor::from_op(
            s,
            Layout::contiguous(out_shape),
            "matmul",
            vec![self.clone(), other.clone()],
            move |g| {
                let ga = if a.requires_grad() {
                    Some(g.matmul(&b.t()?)?.sum_to_shape(a.shape())?)
                } else {
                    None
                };
                let gb = if b.requires_grad() {
                    Some(a.t()?.matmul(g)?.sum_to_shape(b.shape())?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    /// 1D dot product of two vectors.
    pub fn dot(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(Error::Msg("dot requires 1D tensors".into()));
        }
        self.mul(other)?.sum()
    }

    /// Outer product of two 1D vectors `[m]` and `[n]` -> `[m, n]`.
    pub fn outer(&self, other: &Tensor) -> Result<Tensor> {
        if self.ndim() != 1 || other.ndim() != 1 {
            return Err(Error::Msg("outer requires 1D tensors".into()));
        }
        self.unsqueeze(1)?.matmul(&other.unsqueeze(0)?)
    }

    /// Lower-triangular part of a matrix (or batch of matrices).
    pub fn tril(&self, diagonal: isize) -> Result<Tensor> {
        self.tri_mask(diagonal, true)
    }

    /// Upper-triangular part of a matrix (or batch of matrices).
    pub fn triu(&self, diagonal: isize) -> Result<Tensor> {
        self.tri_mask(diagonal, false)
    }

    fn tri_mask(&self, diagonal: isize, lower: bool) -> Result<Tensor> {
        if self.ndim() < 2 {
            return Err(Error::Msg("tril/triu requires rank >= 2".into()));
        }
        let r = self.shape()[self.ndim() - 2];
        let c = self.shape()[self.ndim() - 1];
        let mut mask = Vec::with_capacity(r * c);
        for i in 0..r {
            for j in 0..c {
                let cond = if lower {
                    (j as isize) <= (i as isize) + diagonal
                } else {
                    (j as isize) >= (i as isize) + diagonal
                };
                mask.push(cond);
            }
        }
        let mask_t = Tensor::from_slice_on(&mask, &[r, c], &self.device())?;
        let zero = self.zeros_like()?;
        Tensor::where_cond(&mask_t, self, &zero)
    }

    /// Trace (sum of main diagonal) of a 2D matrix.
    pub fn trace(&self) -> Result<Tensor> {
        self.diagonal(0, 0, 1)?.sum()
    }

    // =========================================================================
    // 6. Concatenation, Stacking, Splitting, Padding & Repeat
    // =========================================================================

    /// Concatenates a slice of tensors along `axis`.
    /// Backward slices the upstream gradient along `axis` with zero-copy `narrow`!
    pub fn cat(tensors: &[Tensor], axis: isize) -> Result<Tensor> {
        let Some(first) = tensors.first() else {
            return Err(Error::Msg("cat: empty tensor list".into()));
        };
        let ndim = first.ndim();
        let ax = normalize_axis("cat", axis, ndim)?;
        let dtype = first.dtype();
        let device = first.device();

        let mut total_dim = 0usize;
        let mut sizes = Vec::with_capacity(tensors.len());
        for t in tensors {
            if t.dtype() != dtype {
                return Err(Error::Msg(format!(
                    "cat: all tensors must share one dtype, got {:?} and {:?}",
                    dtype,
                    t.dtype()
                )));
            }
            if t.device() != device {
                return Err(Error::DeviceMismatch {
                    op: "cat",
                    lhs: device.to_string(),
                    rhs: t.device().to_string(),
                });
            }
            if t.ndim() != ndim {
                return Err(Error::ShapeMismatch {
                    op: "cat",
                    expected: first.shape().to_vec(),
                    got: t.shape().to_vec(),
                });
            }
            for d in 0..ndim {
                if d != ax && t.shape()[d] != first.shape()[d] {
                    return Err(Error::ShapeMismatch {
                        op: "cat",
                        expected: first.shape().to_vec(),
                        got: t.shape().to_vec(),
                    });
                }
            }
            sizes.push(t.shape()[ax]);
            total_dim = total_dim
                .checked_add(t.shape()[ax])
                .ok_or_else(|| Error::Msg("cat: concatenated dim size overflows".into()))?;
        }

        let mut out_shape = first.shape().to_vec();
        out_shape[ax] = total_dim;
        // Refuse outputs that can never fit BEFORE the stitch macro reserves
        // outer*total_dim*inner below.
        crate::limits::check_elements("cat", &out_shape)?;

        let outer: usize = out_shape[..ax].iter().product();
        let inner: usize = out_shape[ax + 1..].iter().product();
        // Typed stitching: gather each input in the output dtype (exact per-element
        // casts when dtypes differ) and memcpy the chunks. The old code round-tripped
        // everything through f64, silently rounding large i64 values.
        macro_rules! stitch {
            ($variant:ident, $ty:ty) => {{
                // contiguous host inputs are read in place; views and device tensors are gathered first
                let guards = tensors.iter().map(|t| t.storage().cpu_ref()).collect::<Result<Vec<_>>>()?;
                let vecs: Vec<std::borrow::Cow<'_, [$ty]>> = tensors
                    .iter()
                    .zip(&guards)
                    .map(|(t, g)| {
                        if let CpuStorage::$variant(v) = &**g
                            && let Some(r) = t.layout().contiguous_range_in(v.len())
                        {
                            return Ok(std::borrow::Cow::Borrowed(&v[r]));
                        }
                        let t = if t.dtype() == dtype { t.clone() } else { t.to_dtype(dtype)? };
                        Ok(std::borrow::Cow::Owned(t.to_vec::<$ty>()?))
                    })
                    .collect::<Result<_>>()?;
                let n = outer * total_dim * inner;
                let mut out_data: Vec<$ty> = crate::pool::take(n).unwrap_or_default();
                if out_data.len() == n {
                    let mut at = 0;
                    for o in 0..outer {
                        for (t_idx, &sz) in sizes.iter().enumerate() {
                            let chunk = sz * inner;
                            out_data[at..at + chunk].copy_from_slice(&vecs[t_idx][o * chunk..(o + 1) * chunk]);
                            at += chunk;
                        }
                    }
                } else {
                    out_data.clear();
                    out_data.reserve(n);
                    for o in 0..outer {
                        for (t_idx, &sz) in sizes.iter().enumerate() {
                            let chunk = sz * inner;
                            out_data.extend_from_slice(&vecs[t_idx][o * chunk..(o + 1) * chunk]);
                        }
                    }
                }
                drop(vecs);
                drop(guards);
                CpuStorage::$variant(out_data)
            }};
        }
        let cpu = match dtype {
            DType::Bool => stitch!(Bool, bool),
            DType::U8 => stitch!(U8, u8),
            DType::I8 => stitch!(I8, i8),
            DType::I16 => stitch!(I16, i16),
            DType::U16 => stitch!(U16, u16),
            DType::U32 => stitch!(U32, u32),
            DType::U64 => stitch!(U64, u64),
            DType::I32 => stitch!(I32, i32),
            DType::I64 => stitch!(I64, i64),
            DType::F16 => stitch!(F16, f16),
            DType::BF16 => stitch!(BF16, bf16),
            DType::F8E4M3 => stitch!(F8E4M3, f8e4m3),
            DType::F8E5M2 => stitch!(F8E5M2, f8e5m2),
            DType::F32 => stitch!(F32, f32),
            DType::F64 => stitch!(F64, f64),
            DType::C64 => {
                let vecs: Vec<Vec<[f32; 2]>> = tensors
                    .iter()
                    .map(|t| t.to_c64_vec())
                    .collect::<Result<_>>()?;
                let n = outer * total_dim * inner;
                let mut out_data = Vec::with_capacity(n);
                for o in 0..outer {
                    for (t_idx, &sz) in sizes.iter().enumerate() {
                        let chunk = sz * inner;
                        out_data.extend_from_slice(&vecs[t_idx][o * chunk..(o + 1) * chunk]);
                    }
                }
                CpuStorage::C64(out_data)
            }
            DType::C128 => {
                let vecs: Vec<Vec<[f64; 2]>> = tensors
                    .iter()
                    .map(|t| t.to_c128_vec())
                    .collect::<Result<_>>()?;
                let n = outer * total_dim * inner;
                let mut out_data = Vec::with_capacity(n);
                for o in 0..outer {
                    for (t_idx, &sz) in sizes.iter().enumerate() {
                        let chunk = sz * inner;
                        out_data.extend_from_slice(&vecs[t_idx][o * chunk..(o + 1) * chunk]);
                    }
                }
                CpuStorage::C128(out_data)
            }
        };
        let storage =
            Storage::new_cpu(cpu).to_device(&Layout::contiguous(out_shape.clone()), &device)?;

        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(out_shape),
            "cat",
            tensors.to_vec(),
            move |g| {
                let mut grads = Vec::with_capacity(sizes.len());
                let mut offset = 0usize;
                for &sz in &sizes {
                    grads.push(Some(g.narrow(ax as isize, offset, sz)?));
                    offset += sz;
                }
                Ok(grads)
            },
        ))
    }

    /// Stacks tensors along a new dimension `axis`.
    pub fn stack(tensors: &[Tensor], axis: isize) -> Result<Tensor> {
        let Some(first) = tensors.first() else {
            return Err(Error::Msg("stack: empty tensor list".into()));
        };
        let ax = normalize_insert_axis("stack", axis, first.ndim())?;
        let unsqueezed: Vec<Tensor> = tensors
            .iter()
            .map(|t| t.unsqueeze(ax as isize))
            .collect::<Result<_>>()?;
        Self::cat(&unsqueezed, ax as isize)
    }

    /// Splits tensor into chunks of size `split_size` along `axis` (zero-copy views).
    pub fn split(&self, split_size: usize, axis: isize) -> Result<Vec<Tensor>> {
        if split_size == 0 {
            return Err(Error::Msg("split: split_size must be positive".into()));
        }
        let ax = normalize_axis("split", axis, self.ndim())?;
        let dim = self.shape()[ax];
        if dim == 0 {
            // Torch parity: splitting an empty dim yields one empty view.
            return Ok(vec![self.narrow(ax as isize, 0, 0)?]);
        }
        // Cap the fan-out up front (same rationale as unbind): a Vec of fat
        // Tensor handles over a huge dim would OOM-abort while growing.
        crate::limits::check_repeats("split", dim.div_ceil(split_size))?;
        let mut out = Vec::new();
        let mut start = 0;
        while start < dim {
            let len = (dim - start).min(split_size);
            out.push(self.narrow(ax as isize, start, len)?);
            start += len;
        }
        Ok(out)
    }

    /// Splits tensor into chunks along `axis` according to `split_sizes`.
    pub fn split_with_sizes(&self, split_sizes: &[usize], axis: isize) -> Result<Vec<Tensor>> {
        let ax = normalize_axis("split_with_sizes", axis, self.ndim())?;
        let dim = self.shape()[ax];
        let total: usize = split_sizes.iter().sum();
        if total != dim {
            return Err(Error::Msg(format!(
                "split_with_sizes: sum of split sizes ({total}) must match dim size ({dim})"
            )));
        }
        let mut out = Vec::with_capacity(split_sizes.len());
        let mut start = 0;
        for &sz in split_sizes {
            out.push(self.narrow(ax as isize, start, sz)?);
            start += sz;
        }
        Ok(out)
    }

    /// Splits tensor into `chunks` parts along `axis` (zero-copy views).
    pub fn chunk(&self, chunks: usize, axis: isize) -> Result<Vec<Tensor>> {
        if chunks == 0 {
            return Err(Error::Msg("chunk: chunks must be positive".into()));
        }
        let ax = normalize_axis("chunk", axis, self.ndim())?;
        let dim = self.shape()[ax];
        if dim == 0 {
            // Torch parity: chunking an empty dim yields `chunks` empty views.
            crate::limits::check_repeats("chunk", chunks)?;
            let mut out = Vec::with_capacity(chunks);
            for _ in 0..chunks {
                out.push(self.narrow(ax as isize, 0, 0)?);
            }
            return Ok(out);
        }
        let split_size = dim.div_ceil(chunks);
        self.split(split_size, ax as isize)
    }

    /// Repeats tensor along each dimension according to `repeats`.
    /// A zero repeat yields an empty (length-0) dimension like `torch.repeat`
    /// (differentiable: the backward scatters nothing, so the input gets zeros).
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        if repeats.len() != self.ndim() {
            return Err(Error::Msg(
                "repeat: repeats rank must match tensor ndim".into(),
            ));
        }
        // Refuse absurd work up front: cloning `rep` Arcs must not hang/abort.
        let mut out_shape = Vec::with_capacity(self.ndim());
        for (ax, (&d, &rep)) in self.shape().iter().zip(repeats.iter()).enumerate() {
            out_shape.push(d.checked_mul(rep).ok_or_else(|| {
                Error::Msg(format!("repeat: output dim {ax} ({d} x {rep}) overflows"))
            })?);
        }
        crate::limits::check_elements("repeat", &out_shape)?;
        let mut cur = self.clone();
        for (ax, &rep) in repeats.iter().enumerate() {
            if rep == 1 {
                continue;
            }
            if rep == 0 {
                cur = cur.narrow(ax as isize, 0, 0)?;
                continue;
            }
            // An empty axis stays empty under any repeat count: skip this axis
            // (rep can be astronomic while 0*rep still passes the check above).
            if cur.shape()[ax] == 0 {
                continue;
            }
            // Double-and-add: a Vec of `rep` handles would OOM-abort on large
            // but legal repeats (rep handles x ~100 B >> the output itself),
            // and spin forever when another axis is empty. Every intermediate
            // cat is bounded by the already-checked final `out_shape`.
            let mut acc: Option<Tensor> = None;
            let mut base = cur.clone();
            let mut n = rep;
            while n > 0 {
                if (n & 1) == 1 {
                    acc = Some(match acc {
                        None => base.clone(),
                        Some(a) => Tensor::cat(&[a, base.clone()], ax as isize)?,
                    });
                }
                n >>= 1;
                if n > 0 {
                    base = Tensor::cat(&[base.clone(), base], ax as isize)?;
                }
            }
            cur = acc.unwrap_or(cur);
        }
        Ok(cur)
    }

    /// Pads tensor with `constant_value` according to `pads: &[(before_0, after_0), ...]`.
    /// Uses [`Tensor::view_scatter`] so it is automatically differentiable to any order!
    pub fn pad(&self, pads: &[(usize, usize)], constant_value: f64) -> Result<Tensor> {
        if pads.len() != self.ndim() {
            return Err(Error::Msg("pad: pads length must equal tensor ndim".into()));
        }
        let mut padded_shape = Vec::with_capacity(self.ndim());
        let mut dim_maps = Vec::with_capacity(self.ndim());
        for (i, (&d, &(before, after))) in self.shape().iter().zip(pads).enumerate() {
            let total = before
                .checked_add(d)
                .and_then(|s| s.checked_add(after))
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "pad: padded dim {i} ({before} + {d} + {after}) overflows"
                    ))
                })?;
            padded_shape.push(total);
            dim_maps.push(DimMap::Sliced {
                out_dim: i,
                start: isize::try_from(before).map_err(|_| {
                    Error::Msg(format!("pad: pad-before {before} exceeds isize::MAX"))
                })?,
                step: 1,
            });
        }
        let spec = ViewSpec {
            src_shape: padded_shape.clone(),
            dst_shape: self.shape().to_vec(),
            dim_maps,
        };
        let scattered = self.view_scatter(spec.clone())?;
        if constant_value == 0.0 {
            Ok(scattered)
        } else {
            // Fill padded cells via an equality mask (works on every dtype,
            // including bool, where arithmetic negation is unsupported).
            let ones = self.ones_like()?.view_scatter(spec)?;
            let is_pad = ones.eq_tensor(&ones.zeros_like()?)?;
            let fill = scattered.full_like(constant_value)?;
            Tensor::where_cond(&is_pad, &fill, &scattered)
        }
    }

    // =========================================================================
    // 7. Indexing, Gather, Scatter, Cumsum, Sort & TopK
    // =========================================================================

    /// Selects slices along `axis` at 1D integer `indices`.
    /// Backward uses [`Tensor::index_add`] so higher-order derivatives work cleanly!
    pub fn index_select(&self, axis: isize, indices: &Tensor) -> Result<Tensor> {
        if indices.ndim() != 1 {
            return Err(Error::Msg("index_select requires a 1D index tensor".into()));
        }
        let ax = normalize_axis("index_select", axis, self.ndim())?;
        let idx_vec = indices.to_vec::<i64>()?;

        let mut out_shape = self.shape().to_vec();
        out_shape[ax] = idx_vec.len();
        // Refuse outputs that can never fit BEFORE reserving: outer*idx*inner
        // (each factor from a validated tensor) can exceed isize::MAX bytes.
        crate::limits::check_elements("index_select", &out_shape)?;
        let storage = self
            .storage()
            .index_select_vals(self.layout(), ax, &idx_vec)?;

        let src_shape = self.shape().to_vec();
        let idx_clone = indices.clone();
        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(out_shape),
            "index_select",
            vec![self.clone()],
            move |g| {
                let z = Tensor::zeros(&src_shape, g.dtype(), &g.device())?;
                Ok(vec![Some(z.index_add(ax as isize, &idx_clone, g)?)])
            },
        ))
    }

    /// Out-of-place `index_add`: adds `source` into `self` along `axis` at 1D `indices`.
    /// Negative indices wrap (torch rejects them here: deliberate); integer adds wrap.
    pub fn index_add(&self, axis: isize, indices: &Tensor, source: &Tensor) -> Result<Tensor> {
        if indices.ndim() != 1 {
            return Err(Error::Msg("index_add requires a 1D index tensor".into()));
        }
        let ax = normalize_axis("index_add", axis, self.ndim())?;
        let idx_vec = indices.to_vec::<i64>()?;

        // `source` must match `self` on every axis except `axis`, where it
        // must hold one slice per index (torch index_add_ semantics).
        let mut expected_src = self.shape().to_vec();
        expected_src[ax] = idx_vec.len();
        if source.shape() != expected_src.as_slice() {
            return Err(Error::ShapeMismatch {
                op: "index_add",
                expected: expected_src,
                got: source.shape().to_vec(),
            });
        }

        let shape = self.shape().to_vec();
        let cpu = self.storage().cpu_ref()?.index_add_vals(
            self.layout(),
            &*source.storage().cpu_ref()?,
            source.layout(),
            ax,
            &idx_vec,
        )?;
        let storage =
            Storage::new_cpu(cpu).to_device(&Layout::contiguous(shape.clone()), &self.device())?;

        let (ra, rb) = (self.requires_grad(), source.requires_grad());
        let idx_clone = indices.clone();
        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(shape),
            "index_add",
            vec![self.clone(), source.clone()],
            move |g| {
                let ga = if ra { Some(g.clone()) } else { None };
                let gb = if rb {
                    Some(g.index_select(ax as isize, &idx_clone)?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    /// Gathers values along `axis` using an index tensor of the same rank as `self`.
    /// Negative indices wrap (torch rejects them here: deliberate).
    pub fn gather(&self, axis: isize, index: &Tensor) -> Result<Tensor> {
        let ax = normalize_axis("gather", axis, self.ndim())?;
        if index.ndim() != self.ndim() {
            return Err(Error::Msg(
                "gather: index rank must equal input rank".into(),
            ));
        }
        for d in 0..self.ndim() {
            if d != ax && index.shape()[d] > self.shape()[d] {
                return Err(Error::ShapeMismatch {
                    op: "gather",
                    expected: self.shape().to_vec(),
                    got: index.shape().to_vec(),
                });
            }
        }
        let idx_shape = index.shape().to_vec();
        let idx = index.to_vec::<i64>()?;
        let cpu =
            self.storage()
                .cpu_ref()?
                .gather_vals(self.layout(), ax, &idx_shape, &idx)?;
        let storage = Storage::new_cpu(cpu)
            .to_device(&Layout::contiguous(idx_shape.clone()), &self.device())?;
        let src_shape = self.shape().to_vec();
        let idx_clone = index.clone();
        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(idx_shape),
            "gather",
            vec![self.clone()],
            move |g| {
                let z = Tensor::zeros(&src_shape, g.dtype(), &g.device())?;
                Ok(vec![Some(z.scatter_add(ax as isize, &idx_clone, g)?)])
            },
        ))
    }

    /// Out-of-place `scatter_add`: adds `src` into `self` along `axis` at `index`.
    /// Negative indices wrap (torch rejects them here: deliberate); integer adds wrap.
    pub fn scatter_add(&self, axis: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let ax = normalize_axis("scatter_add", axis, self.ndim())?;
        if index.ndim() != self.ndim() {
            return Err(Error::Msg(
                "scatter_add: index rank must equal input rank".into(),
            ));
        }
        if src.shape() != index.shape() {
            return Err(Error::ShapeMismatch {
                op: "scatter_add",
                expected: index.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }
        for d in 0..self.ndim() {
            if d != ax && index.shape()[d] > self.shape()[d] {
                return Err(Error::ShapeMismatch {
                    op: "scatter_add",
                    expected: self.shape().to_vec(),
                    got: index.shape().to_vec(),
                });
            }
        }
        let idx_shape = index.shape().to_vec();
        let idx = index.to_vec::<i64>()?;
        let shape = self.shape().to_vec();
        let cpu = self.storage().cpu_ref()?.scatter_add_vals(
            self.layout(),
            &*src.storage().cpu_ref()?,
            src.layout(),
            &idx_shape,
            ax,
            &idx,
        )?;
        let storage =
            Storage::new_cpu(cpu).to_device(&Layout::contiguous(shape.clone()), &self.device())?;
        let (ra, rb) = (self.requires_grad(), src.requires_grad());
        let idx_clone = index.clone();
        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(shape),
            "scatter_add",
            vec![self.clone(), src.clone()],
            move |g| {
                let ga = if ra { Some(g.clone()) } else { None };
                let gb = if rb {
                    Some(g.gather(ax as isize, &idx_clone)?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    /// Out-of-place `scatter` of `src` into `self` along `axis` (`torch.scatter`).
    /// Same-rank `index`/`src`; duplicates resolve last-write-wins; negative
    /// indices wrap (same pinned divergence as `gather`/`scatter_add`).
    /// Differentiable: base grads zero at scattered positions, src gathers grad.
    pub fn scatter(&self, axis: isize, index: &Tensor, src: &Tensor) -> Result<Tensor> {
        let ax = normalize_axis("scatter", axis, self.ndim())?;
        if index.ndim() != self.ndim() {
            return Err(Error::Msg(
                "scatter: index rank must equal input rank".into(),
            ));
        }
        if src.shape() != index.shape() {
            return Err(Error::ShapeMismatch {
                op: "scatter",
                expected: index.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }
        for d in 0..self.ndim() {
            if d != ax && index.shape()[d] > self.shape()[d] {
                return Err(Error::ShapeMismatch {
                    op: "scatter",
                    expected: self.shape().to_vec(),
                    got: index.shape().to_vec(),
                });
            }
        }
        let idx_shape = index.shape().to_vec();
        let idx = index.to_vec::<i64>()?;
        let shape = self.shape().to_vec();
        let cpu = self.storage().cpu_ref()?.scatter_vals(
            self.layout(),
            &*src.storage().cpu_ref()?,
            src.layout(),
            &idx_shape,
            ax,
            &idx,
        )?;
        let storage =
            Storage::new_cpu(cpu).to_device(&Layout::contiguous(shape.clone()), &self.device())?;
        let (ra, rb) = (self.requires_grad(), src.requires_grad());
        let idx_clone = index.clone();
        let idx_shape_clone = idx_shape.clone();
        let dev = self.device();
        let dt = self.dtype();
        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(shape),
            "scatter",
            vec![self.clone(), src.clone()],
            move |g| {
                let ga = if ra {
                    let zeros = Tensor::zeros(&idx_shape_clone, dt, &dev)?;
                    Some(g.scatter(ax as isize, &idx_clone, &zeros)?)
                } else {
                    None
                };
                let gb = if rb {
                    Some(g.gather(ax as isize, &idx_clone)?)
                } else {
                    None
                };
                Ok(vec![ga, gb])
            },
        ))
    }

    /// Cumulative sum along `axis`.
    /// Backward is the reverse cumulative sum: `g.flip(&[axis]).cumsum(axis).flip(&[axis])`!
    pub fn cumsum(&self, axis: isize) -> Result<Tensor> {
        if (self.dtype().is_int() || self.dtype() == DType::Bool) && self.dtype() != DType::I64 {
            return self.to_dtype(DType::I64)?.cumsum(axis);
        }
        let ax = normalize_axis("cumsum", axis, self.ndim())?;
        let dim = self.shape()[ax];
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();
        let mut data = self.to_vec::<f64>()?;

        for o in 0..outer {
            for k in 1..dim {
                let cur = (o * dim + k) * inner;
                let prev = (o * dim + k - 1) * inner;
                for i in 0..inner {
                    data[cur + i] += data[prev + i];
                }
            }
        }
        let shape = self.shape().to_vec();
        let cpu =
            CpuStorage::F64(data).to_dtype(&Layout::contiguous(shape.clone()), self.dtype())?;
        let storage =
            Storage::new_cpu(cpu).to_device(&Layout::contiguous(shape.clone()), &self.device())?;
        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(shape),
            "cumsum",
            vec![self.clone()],
            move |g| {
                let rev = g
                    .flip(&[ax as isize])?
                    .cumsum(ax as isize)?
                    .flip(&[ax as isize])?;
                Ok(vec![Some(rev)])
            },
        ))
    }

    /// Sorts elements along `axis`, returning `(sorted_values, indices_i64)`.
    /// Differentiable with respect to `values` via `gather`/`scatter_add`!
    pub fn sort(&self, axis: isize, descending: bool) -> Result<(Tensor, Tensor)> {
        let ax = normalize_axis("sort", axis, self.ndim())?;
        let k = self.shape()[ax];
        self.topk(k, ax as isize, descending, true)
    }

    /// Returns indices that sort the tensor along `axis`.
    pub fn argsort(&self, axis: isize, descending: bool) -> Result<Tensor> {
        Ok(self.sort(axis, descending)?.1)
    }

    /// Returns the `k` largest (or smallest if `largest == false`) elements and their
    /// `i64` indices along `axis`. NaN sorts as the largest value (PyTorch parity).
    /// If `sorted` is false, the winners arrive in unspecified order via selection
    /// (O(n) average) instead of a full sort. Differentiable wrt values!
    pub fn topk(
        &self,
        k: usize,
        axis: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor, Tensor)> {
        let ax = normalize_axis("topk", axis, self.ndim())?;
        let dim = self.shape()[ax];
        if k > dim {
            return Err(Error::IndexOutOfBounds {
                op: "topk",
                index: k as isize,
                dim_size: dim,
            });
        }
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();

        // I64 keys compare natively: past 2^53 an f64 key rounds (2^53+1
        // reads as 2^53) and misorders. Every other dtype widens to f64
        // exactly (bool/u8/i32/f16/bf16/f32) or IS float (f64).
        let mut indices = vec![0i64; outer * k * inner];
        if self.dtype() == DType::I64 {
            let data = self.to_vec::<i64>()?;
            for o in 0..outer {
                for i in 0..inner {
                    let mut pairs: Vec<(usize, i64)> = (0..dim)
                        .map(|d| (d, data[(o * dim + d) * inner + i]))
                        .collect();
                    let cmp = |a: &(usize, i64), b: &(usize, i64)| {
                        if largest {
                            b.1.cmp(&a.1)
                        } else {
                            a.1.cmp(&b.1)
                        }
                    };
                    if sorted {
                        pairs.sort_by(cmp);
                    } else if k > 0 && k < dim {
                        pairs.select_nth_unstable_by(k - 1, cmp);
                    }
                    for (rank, &(orig_idx, _)) in pairs.iter().take(k).enumerate() {
                        indices[(o * k + rank) * inner + i] = orig_idx as i64;
                    }
                }
            }
        } else {
            let data = self.to_vec::<f64>()?;
            for o in 0..outer {
                for i in 0..inner {
                    let mut pairs: Vec<(usize, f64)> = (0..dim)
                        .map(|d| (d, data[(o * dim + d) * inner + i]))
                        .collect();
                    // `total_cmp` is a valid total order with NaN sorting largest
                    // (PyTorch parity); `partial_cmp` left NaN stuck mid-slice.
                    let cmp = |a: &(usize, f64), b: &(usize, f64)| {
                        if largest {
                            b.1.total_cmp(&a.1)
                        } else {
                            a.1.total_cmp(&b.1)
                        }
                    };
                    if sorted {
                        pairs.sort_by(cmp);
                    } else if k > 0 && k < dim {
                        // Quickselect: the top-k land in front in unspecified
                        // order, O(n) average instead of O(n log n).
                        pairs.select_nth_unstable_by(k - 1, cmp);
                    }
                    for (rank, &(orig_idx, _)) in pairs.iter().take(k).enumerate() {
                        indices[(o * k + rank) * inner + i] = orig_idx as i64;
                    }
                }
            }
        }
        let mut out_shape = self.shape().to_vec();
        out_shape[ax] = k;
        let idx_tensor = Tensor::from_slice_on(&indices, &out_shape, &self.device())?;
        let values = self.gather(ax as isize, &idx_tensor)?;
        Ok((values, idx_tensor))
    }

    /// Returns the `k`-th smallest elements (1-based `k`) and their `i64`
    /// indices along `axis` (`torch.kthvalue`). O(n) average via quickselect;
    /// NaN sorts largest like [`Tensor::sort`]. Differentiable wrt values!
    pub fn kthvalue(&self, k: usize, axis: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
        let ax = normalize_axis("kthvalue", axis, self.ndim())?;
        let dim = self.shape()[ax];
        if k == 0 || k > dim {
            return Err(Error::IndexOutOfBounds {
                op: "kthvalue",
                index: k as isize,
                dim_size: dim,
            });
        }
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();
        // I64 keys compare natively (f64 rounds past 2^53); every other
        // dtype widens to f64 exactly or is float already.
        let mut indices = vec![0i64; outer * inner];
        if self.dtype() == DType::I64 {
            let data = self.to_vec::<i64>()?;
            for o in 0..outer {
                for i in 0..inner {
                    let mut pairs: Vec<(usize, i64)> = (0..dim)
                        .map(|d| (d, data[(o * dim + d) * inner + i]))
                        .collect();
                    pairs.select_nth_unstable_by(k - 1, |a, b| a.1.cmp(&b.1));
                    indices[o * inner + i] = pairs[k - 1].0 as i64;
                }
            }
        } else {
            let data = self.to_vec::<f64>()?;
            for o in 0..outer {
                for i in 0..inner {
                    let mut pairs: Vec<(usize, f64)> = (0..dim)
                        .map(|d| (d, data[(o * dim + d) * inner + i]))
                        .collect();
                    pairs.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));
                    indices[o * inner + i] = pairs[k - 1].0 as i64;
                }
            }
        }
        let mut kd_shape = self.shape().to_vec();
        kd_shape[ax] = 1;
        let idx = Tensor::from_slice_on(&indices, &kd_shape, &self.device())?;
        let values = self.gather(ax as isize, &idx)?;
        if keepdim {
            Ok((values, idx))
        } else {
            Ok((values.select(ax as isize, 0)?, idx.select(ax as isize, 0)?))
        }
    }

    /// Mode (most frequent value) and its index along `axis` (torch.mode
    /// parity on values). Ties resolve to the smallest value; NaN groups
    /// (NaN == NaN for counting) and sorts smallest; +-0.0 group. Indices
    /// deterministically address the FIRST occurrence (torch follows its
    /// internal unstable sort order instead: content-dependent, usually
    /// last). Differentiable wrt values (gradient flows to the reported
    /// index, like torch's self-consistent routing).
    pub fn mode_axis(&self, axis: isize, keepdim: bool) -> Result<(Tensor, Tensor)> {
        if self.ndim() == 0 {
            let idx = Tensor::from_slice_on(&[0i64], &[], &self.device())?;
            return Ok((self.clone(), idx));
        }
        let ax = normalize_axis("mode", axis, self.ndim())?;
        self.require_nonempty("mode", ax)?;
        let dim = self.shape()[ax];
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();
        let mut indices = vec![0i64; outer * inner];
        if matches!(
            self.dtype(),
            DType::I64 | DType::I32 | DType::U8 | DType::Bool
        ) {
            // I64 keys compare natively (f64 rounds past 2^53); smaller
            // integer widths and bool widen exactly.
            let data = self.to_vec::<i64>()?;
            for o in 0..outer {
                for i in 0..inner {
                    let mut counts: HashMap<i64, (usize, usize)> = HashMap::new();
                    for d in 0..dim {
                        let v = data[(o * dim + d) * inner + i];
                        counts.entry(v).and_modify(|e| e.0 += 1).or_insert((1, d));
                    }
                    let mut best: Option<(i64, usize, usize)> = None;
                    for (&v, &(c, f)) in counts.iter() {
                        let take = match best {
                            None => true,
                            Some((bv, bc, _)) => c > bc || (c == bc && v < bv),
                        };
                        if take {
                            best = Some((v, c, f));
                        }
                    }
                    indices[o * inner + i] = best.map(|(_, _, f)| f as i64).unwrap_or(0);
                }
            }
        } else {
            let data = self.to_vec::<f64>()?;
            for o in 0..outer {
                for i in 0..inner {
                    // Canonical keys: both zeros share 0, all NaNs share
                    // u64::MAX (every all-ones f64 is NaN: no collision).
                    let mut counts: HashMap<u64, (usize, usize, f64)> = HashMap::new();
                    for d in 0..dim {
                        let v = data[(o * dim + d) * inner + i];
                        let key = if v == 0.0 {
                            0
                        } else if v.is_nan() {
                            u64::MAX
                        } else {
                            v.to_bits()
                        };
                        counts
                            .entry(key)
                            .and_modify(|e| e.0 += 1)
                            .or_insert((1, d, v));
                    }
                    let mut best: Option<(usize, f64, usize)> = None;
                    for &(c, f, v) in counts.values() {
                        let take = match best {
                            None => true,
                            Some((bc, bv, _)) => {
                                if c != bc {
                                    c > bc
                                } else if v.is_nan() != bv.is_nan() {
                                    // NaN sorts smallest on ties.
                                    v.is_nan()
                                } else {
                                    v.total_cmp(&bv).is_lt()
                                }
                            }
                        };
                        if take {
                            best = Some((c, v, f));
                        }
                    }
                    indices[o * inner + i] = best.map(|(_, _, f)| f as i64).unwrap_or(0);
                }
            }
        }
        let mut kd_shape = self.shape().to_vec();
        kd_shape[ax] = 1;
        let idx = Tensor::from_slice_on(&indices, &kd_shape, &self.device())?;
        let values = self.gather(ax as isize, &idx)?;
        if keepdim {
            Ok((values, idx))
        } else {
            Ok((values.select(ax as isize, 0)?, idx.select(ax as isize, 0)?))
        }
    }

    // =========================================================================
    // 8. Einstein Summation (`einsum`)
    // =========================================================================

    /// Evaluates an arbitrary Einstein summation expression (e.g. `"ij,jk->ik"`,
    /// `"bhqd,bhkd->bhqk"`, `"ii->i"`, `"ijk->ik"`, `"bij,bjk->bik"`).
    ///
    /// Automatically differentiates through any 1- or 2-operand `einsum`!
    pub fn einsum(equation: &str, operands: &[&Tensor]) -> Result<Tensor> {
        let eq: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
        let (lhs_str, rhs_str) = match eq.split_once("->") {
            Some((l, r)) => (l.to_string(), r.to_string()),
            None => {
                // Implicit output: sorted unique indices that appear exactly once.
                let mut counts = HashMap::<char, usize>::new();
                for c in eq.chars().filter(|&c| c != ',') {
                    *counts.entry(c).or_insert(0) += 1;
                }
                let mut implicit: Vec<char> = counts
                    .into_iter()
                    .filter(|&(_, cnt)| cnt == 1)
                    .map(|(c, _)| c)
                    .collect();
                implicit.sort_unstable();
                (eq, implicit.into_iter().collect())
            }
        };

        let in_specs: Vec<&str> = lhs_str.split(',').collect();
        if in_specs.len() != operands.len() {
            return Err(Error::Msg(format!(
                "einsum: equation expects {} operands, got {}",
                in_specs.len(),
                operands.len()
            )));
        }

        let out_val = no_grad(|| eval_einsum_forward(&in_specs, &rhs_str, operands))?;
        let owned_ops: Vec<Tensor> = operands.iter().map(|&t| t.clone()).collect();
        let in_specs_owned: Vec<String> = in_specs.iter().map(|s| (*s).to_string()).collect();
        let out_spec = rhs_str.clone();

        Ok(Tensor::from_op(
            out_val.storage().clone(),
            out_val.layout().clone(),
            "einsum",
            owned_ops.clone(),
            move |g| {
                if owned_ops.len() == 1 {
                    let a = &owned_ops[0];
                    if !a.requires_grad() {
                        return Ok(vec![None]);
                    }
                    let ga = einsum_backward_single(&in_specs_owned[0], &out_spec, a.shape(), g)?;
                    Ok(vec![Some(ga)])
                } else if owned_ops.len() == 2 {
                    let (a, b) = (&owned_ops[0], &owned_ops[1]);
                    let ga = if a.requires_grad() {
                        Some(einsum_backward_pair(
                            &in_specs_owned[0],
                            &in_specs_owned[1],
                            &out_spec,
                            a.shape(),
                            g,
                            b,
                        )?)
                    } else {
                        None
                    };
                    let gb = if b.requires_grad() {
                        Some(einsum_backward_pair(
                            &in_specs_owned[1],
                            &in_specs_owned[0],
                            &out_spec,
                            b.shape(),
                            g,
                            a,
                        )?)
                    } else {
                        None
                    };
                    Ok(vec![ga, gb])
                } else {
                    Err(Error::Autograd(
                        "einsum backward currently supports 1 or 2 operands".into(),
                    ))
                }
            },
        ))
    }

    // =========================================================================
    // 9. Im2Col (`unfold` / `fold`), Conv1d, Conv2d, Pooling
    // =========================================================================

    /// Extracts sliding local blocks (`im2col`) from a 4D input `[N, C, H, W]`,
    /// returning `[N, C * kH * kW, L]` where `L = out_h * out_w`.
    /// Backward is [`Tensor::fold2d`] (`col2im`)!
    pub fn unfold2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Msg("unfold2d expects 4D tensor [N, C, H, W]".into()));
        }
        let (n, c, h, w) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        );
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        if kh == 0 || kw == 0 {
            return Err(Error::Msg("unfold2d: kernel_size must be positive".into()));
        }
        if sh == 0 || sw == 0 {
            return Err(Error::Msg("unfold2d: stride must be positive".into()));
        }
        if dh == 0 || dw == 0 {
            return Err(Error::Msg("unfold2d: dilation must be positive".into()));
        }

        let of = || Error::Msg("unfold2d: size arithmetic overflows".into());
        let eff_kh = dh
            .checked_mul(kh - 1)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(of)?;
        let eff_kw = dw
            .checked_mul(kw - 1)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(of)?;
        let pad_h = h
            .checked_add(ph.checked_mul(2).ok_or_else(of)?)
            .ok_or_else(of)?;
        let pad_w = w
            .checked_add(pw.checked_mul(2).ok_or_else(of)?)
            .ok_or_else(of)?;
        if pad_h < eff_kh || pad_w < eff_kw {
            return Err(Error::Msg(
                "unfold2d: kernel size exceeds padded input".into(),
            ));
        }
        let out_h = (pad_h - eff_kh) / sh + 1;
        let out_w = (pad_w - eff_kw) / sw + 1;
        let l = out_h.checked_mul(out_w).ok_or_else(of)?;
        let col_c = c
            .checked_mul(kh)
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(of)?;
        let cols_n = crate::limits::check_elements("unfold2d", &[n, col_c, l])?;

        let out_shape = vec![n, col_c, l];
        let geom = UnfoldGeom { h, w, kh, kw, sh, sw, ph, pw, dh, dw, out_h, out_w };
        let cpu = if self.dtype() == DType::F32 {
            // native f32: the source is read in place and whole row spans are copied, in parallel over channels
            let guard = self.storage().cpu_ref()?;
            let owned;
            let src: &[f32] = match &*guard {
                CpuStorage::F32(v) if self.layout().contiguous_range_in(v.len()).is_some() => &v[self.layout().contiguous_range_in(v.len()).unwrap()],
                other => {
                    owned = other.to_vec::<f32>(self.layout());
                    &owned
                }
            };
            let mut cols = match crate::pool::take_f32(cols_n) {
                Some(mut v) => {
                    v.fill(0.0);
                    v
                }
                None => crate::layout::try_filled(cols_n, 0.0f32)?,
            };
            unfold_fill(src, &mut cols, &geom);
            CpuStorage::F32(cols)
        } else {
            let src = self.to_vec::<f64>()?;
            let mut cols = crate::layout::try_filled(cols_n, 0.0f64)?;
            unfold_fill(&src, &mut cols, &geom);
            CpuStorage::F64(cols).to_dtype(&Layout::contiguous(out_shape.clone()), self.dtype())?
        };
        let storage = Storage::new_cpu(cpu)
            .to_device(&Layout::contiguous(out_shape.clone()), &self.device())?;

        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(out_shape),
            "unfold2d",
            vec![self.clone()],
            move |g| {
                Ok(vec![Some(g.fold2d(
                    (h, w),
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )?)])
            },
        ))
    }

    /// `col2im` scatter core shared by [`Tensor::fold2d`] and [`Tensor::conv_transpose2d`]:
    /// accumulates `cols` (`[N, col_c, gh * gw]` flattened) into a zero image `[N, C, H, W]`.
    /// Cells no window covers (the `output_padding` fringe in transposed conv) stay zero.
    fn fold2d_scatter(
        cols: &[f64],
        n: usize,
        c: usize,
        col_c: usize,
        g: Fold2dGeo,
    ) -> Result<Vec<f64>> {
        let l =
            g.gh.checked_mul(g.gw)
                .ok_or_else(|| Error::Msg("fold2d: col-grid size overflows".into()))?;
        let mut img = crate::layout::try_filled(
            crate::limits::check_elements("fold2d", &[n, c, g.h, g.w])?,
            0.0f64,
        )?;

        for b in 0..n {
            for ch in 0..c {
                for r in 0..g.kh {
                    for s in 0..g.kw {
                        let col_row = (ch * g.kh + r) * g.kw + s;
                        let src_base = (b * col_c + col_row) * l;
                        for oh in 0..g.gh {
                            let ih = (oh * g.sh + r * g.dh) as isize - (g.ph as isize);
                            if ih < 0 || ih as usize >= g.h {
                                continue;
                            }
                            let dst_row_base = ((b * c + ch) * g.h + ih as usize) * g.w;
                            for ow in 0..g.gw {
                                let iw = (ow * g.sw + s * g.dw) as isize - (g.pw as isize);
                                if iw >= 0 && (iw as usize) < g.w {
                                    img[dst_row_base + iw as usize] +=
                                        cols[src_base + oh * g.gw + ow];
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(img)
    }

    /// Combines an array of sliding local blocks (`col2im`) of shape `[N, C * kH * kW, L]`
    /// back into a 4D tensor `[N, C, H, W]`. Backward is [`Tensor::unfold2d`]!
    pub fn fold2d(
        &self,
        output_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 3 {
            return Err(Error::Msg(
                "fold2d expects 3D tensor [N, C*kH*kW, L]".into(),
            ));
        }
        let (n, col_c, l) = (self.shape()[0], self.shape()[1], self.shape()[2]);
        let (h, w) = output_size;
        let (kh, kw) = kernel_size;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;
        if kh == 0 || kw == 0 {
            return Err(Error::Msg("fold2d: kernel_size must be positive".into()));
        }
        if sh == 0 || sw == 0 {
            return Err(Error::Msg("fold2d: stride must be positive".into()));
        }
        if dh == 0 || dw == 0 {
            return Err(Error::Msg("fold2d: dilation must be positive".into()));
        }
        let karea = kh
            .checked_mul(kw)
            .ok_or_else(|| Error::Msg("fold2d: kernel size arithmetic overflows".into()))?;
        if col_c % karea != 0 {
            return Err(Error::Msg(format!(
                "fold2d: channel dim {col_c} is not divisible by kH*kW {karea}"
            )));
        }
        let c = col_c / karea;

        let of = || Error::Msg("fold2d: size arithmetic overflows".into());
        let eff_kh = dh
            .checked_mul(kh - 1)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(of)?;
        let eff_kw = dw
            .checked_mul(kw - 1)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(of)?;
        let pad_h = h
            .checked_add(ph.checked_mul(2).ok_or_else(of)?)
            .ok_or_else(of)?;
        let pad_w = w
            .checked_add(pw.checked_mul(2).ok_or_else(of)?)
            .ok_or_else(of)?;
        if pad_h < eff_kh || pad_w < eff_kw {
            return Err(Error::Msg(
                "fold2d: kernel size exceeds padded output".into(),
            ));
        }
        let out_h = (pad_h - eff_kh) / sh + 1;
        let out_w = (pad_w - eff_kw) / sw + 1;
        let expect_l = out_h.checked_mul(out_w).ok_or_else(of)?;
        if l != expect_l {
            return Err(Error::ShapeMismatch {
                op: "fold2d",
                expected: vec![n, col_c, expect_l],
                got: vec![n, col_c, l],
            });
        }
        let cols = self.to_vec::<f64>()?;
        let img = Self::fold2d_scatter(
            &cols,
            n,
            c,
            col_c,
            Fold2dGeo {
                h,
                w,
                gh: out_h,
                gw: out_w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                dh,
                dw,
            },
        )?;

        let out_shape = vec![n, c, h, w];
        let cpu =
            CpuStorage::F64(img).to_dtype(&Layout::contiguous(out_shape.clone()), self.dtype())?;
        let storage = Storage::new_cpu(cpu)
            .to_device(&Layout::contiguous(out_shape.clone()), &self.device())?;

        Ok(Tensor::from_op(
            storage,
            Layout::contiguous(out_shape),
            "fold2d",
            vec![self.clone()],
            move |g| {
                Ok(vec![Some(g.unfold2d(
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                )?)])
            },
        ))
    }

    /// 2D convolution `conv2d` on `input: [N, C_in, H, W]` with `weight: [C_out, C_in, kH, kW]`
    /// and optional `bias: [C_out]`, implemented via differentiable `unfold2d` + `matmul`!
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        if let Some((x, w)) = crate::amp::lowp_pair(self, weight)? {
            let b = bias.map(crate::amp::autocast_cast).transpose()?;
            return x.conv2d(&w, b.as_ref(), stride, padding, dilation);
        }
        if self.ndim() != 4 || weight.ndim() != 4 {
            return Err(Error::Msg("conv2d requires 4D input and 4D weight".into()));
        }
        let (n, c_in, h, w) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        );
        let (c_out, wc_in, kh, kw) = (
            weight.shape()[0],
            weight.shape()[1],
            weight.shape()[2],
            weight.shape()[3],
        );
        if c_in != wc_in {
            return Err(Error::ShapeMismatch {
                op: "conv2d",
                expected: vec![c_out, c_in, kh, kw],
                got: weight.shape().to_vec(),
            });
        }
        if kh == 0 || kw == 0 {
            return Err(Error::Msg("conv2d: kernel_size must be positive".into()));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(Error::Msg("conv2d: stride must be positive".into()));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(Error::Msg("conv2d: dilation must be positive".into()));
        }
        let eff_kh = dilation.0 * (kh - 1) + 1;
        let eff_kw = dilation.1 * (kw - 1) + 1;
        if h + 2 * padding.0 < eff_kh || w + 2 * padding.1 < eff_kw {
            return Err(Error::Msg(
                "conv2d: kernel size exceeds padded input".into(),
            ));
        }
        let out_h = (h + 2 * padding.0 - eff_kh) / stride.0 + 1;
        let out_w = (w + 2 * padding.1 - eff_kw) / stride.1 + 1;

        // inference on contiguous f32 host tensors: im2col and the matrix product run image by image, so the column
        // buffer (kh*kw times one image) stays in cache instead of being built for the whole batch first
        let needs_grad = crate::tensor::is_grad_enabled() && (self.requires_grad() || weight.requires_grad() || bias.is_some_and(Tensor::requires_grad));
        if !needs_grad && bias.is_none_or(|b| b.dtype() == DType::F32 && b.shape() == [c_out]) {
            let geom = UnfoldGeom { h, w, kh, kw, sh: stride.0, sw: stride.1, ph: padding.0, pw: padding.1, dh: dilation.0, dw: dilation.1, out_h, out_w };
            let (l, kdim) = (out_h * out_w, c_in * kh * kw);
            let bias_v = bias.map(|b| b.to_vec::<f32>()).transpose()?;
            let fused = with_f32_host(self, |src| {
                with_f32_host(weight, |wv| {
                    let total = n * c_out * l;
                    let mut out = crate::pool::take_f32(total).unwrap_or_else(|| vec![0.0; total]);
                    #[cfg(not(feature = "rayon"))]
                    let image = |(b, o): (usize, &mut [f32])| {
                        let mut cols = match crate::pool::take_f32(kdim * l) {
                            Some(mut v) => {
                                v.fill(0.0);
                                v
                            }
                            None => vec![0.0; kdim * l],
                        };
                        unfold_fill(&src[b * c_in * h * w..(b + 1) * c_in * h * w], &mut cols, &geom);
                        crate::gemm::gemm_f32_strided(wv, 0, kdim as isize, 1, &cols, 0, l as isize, 1, o, c_out, kdim, l);
                        if let Some(bv) = &bias_v {
                            o.chunks_mut(l).zip(bv).for_each(|(ch, &bb)| ch.iter_mut().for_each(|v| *v += bb));
                        }
                        crate::pool::give(cols);
                    };
                    if total > 0 && kdim * l > 0 {
                        #[cfg(feature = "rayon")]
                        if kh == 3
                            && kw == 3
                            && stride == (1, 1)
                            && dilation == (1, 1)
                            && c_in >= 8
                            && c_out >= 8
                            && out_h >= 4
                            && out_w >= 4
                        {
                            conv2d_winograd_3x3_f32(
                                src,
                                wv,
                                bias_v.as_deref(),
                                &mut out,
                                n,
                                c_in,
                                h,
                                w,
                                c_out,
                                padding.0,
                                padding.1,
                                out_h,
                                out_w,
                            );
                        } else if stride == (1, 1)
                            && dilation == (1, 1)
                            && out_w >= 8
                            && c_out % 4 == 0
                            && crate::backend::simd_x86::has_avx2_fma()
                        {
                            use rayon::prelude::*;
                            let bv_slice = bias_v.as_deref();
                            let blocks_per_img = c_out / 4;
                            let img_elems = c_in * h * w;
                            out.par_chunks_mut(4 * l).enumerate().for_each(|(task_id, o_blk)| {
                                let b = task_id / blocks_per_img;
                                let oc = (task_id % blocks_per_img) * 4;
                                let bb = bv_slice.map_or([0.0; 4], |bv| [bv[oc], bv[oc + 1], bv[oc + 2], bv[oc + 3]]);
                                crate::backend::simd_x86::conv2d_oc_block4_s1d1_f32(
                                    &src[b * img_elems..(b + 1) * img_elems],
                                    &wv[oc * kdim..(oc + 4) * kdim],
                                    bb,
                                    o_blk,
                                    c_in,
                                    h,
                                    w,
                                    kh,
                                    kw,
                                    padding.0,
                                    padding.1,
                                    out_h,
                                    out_w,
                                );
                            });
                        } else if stride == (1, 1)
                            && dilation == (1, 1)
                            && out_w >= 8
                            && crate::backend::simd_x86::has_avx2_fma()
                        {
                            use rayon::prelude::*;
                            let bv_slice = bias_v.as_deref();
                            out.par_chunks_mut(c_out * l).enumerate().for_each(|(b, o)| {
                                crate::backend::simd_x86::conv2d_image_s1d1_f32(
                                    &src[b * c_in * h * w..(b + 1) * c_in * h * w],
                                    wv,
                                    bv_slice,
                                    o,
                                    c_in,
                                    h,
                                    w,
                                    c_out,
                                    kh,
                                    kw,
                                    padding.0,
                                    padding.1,
                                    out_h,
                                    out_w,
                                );
                            });
                        } else if n >= 2 && kdim * n * l >= 65536 {
                            use rayon::prelude::*;
                            let nl = n * l;
                            let mut cols_all = crate::pool::take_f32(kdim * nl).unwrap_or_else(|| vec![0.0; kdim * nl]);
                            if padding.0 > 0 || padding.1 > 0 {
                                cols_all.par_chunks_mut(65536).for_each(|c| c.fill(0.0));
                            }
                            let img_plane = c_in * h * w;
                            // Fill cols_all in [kdim, n * l] layout: each kernel row k has [b * l .. (b+1) * l]
                            cols_all.par_chunks_mut(nl).enumerate().for_each(|(k_idx, k_row)| {
                                let c = k_idx / (kh * kw);
                                let rem = k_idx % (kh * kw);
                                let r = rem / kw;
                                let s = rem % kw;
                                let off = s * dilation.1;
                                let ow_lo = if off >= padding.1 { 0 } else { (padding.1 - off).div_ceil(stride.1) };
                                let ow_hi = if w + padding.1 > off { ((w - 1 + padding.1 - off) / stride.1 + 1).min(out_w) } else { 0 };
                                if ow_lo >= ow_hi {
                                    return;
                                }
                                let first = ow_lo * stride.1 + off - padding.1;
                                for b in 0..n {
                                    let plane = &src[b * img_plane + c * h * w..b * img_plane + (c + 1) * h * w];
                                    let dst = &mut k_row[b * l..(b + 1) * l];
                                    for oh in 0..out_h {
                                        let ih = (oh * stride.0 + r * dilation.0) as isize - padding.0 as isize;
                                        if ih < 0 || ih as usize >= h {
                                            continue;
                                        }
                                        let srow = &plane[ih as usize * w..(ih as usize + 1) * w];
                                        let drow = &mut dst[oh * out_w + ow_lo..oh * out_w + ow_hi];
                                        if stride.1 == 1 {
                                            drow.copy_from_slice(&srow[first..first + drow.len()]);
                                        } else {
                                            for (d, x) in drow.iter_mut().zip(srow[first..].iter().step_by(stride.1)) {
                                                *d = *x;
                                            }
                                        }
                                    }
                                }
                            });
                            let mut tmp = crate::pool::take_f32(total).unwrap_or_else(|| vec![0.0; total]);
                            crate::gemm::gemm_f32_strided(wv, 0, kdim as isize, 1, &cols_all, 0, nl as isize, 1, &mut tmp, c_out, kdim, nl);
                            crate::pool::give(cols_all);
                            // Transpose [c_out, n, l] -> out[n, c_out, l] + bias in parallel
                            out.par_chunks_mut(c_out * l).enumerate().for_each(|(b, ob)| {
                                for oc in 0..c_out {
                                    let src_sl = &tmp[oc * nl + b * l..oc * nl + (b + 1) * l];
                                    let dst_sl = &mut ob[oc * l..(oc + 1) * l];
                                    if let Some(bv) = &bias_v {
                                        let bb = bv[oc];
                                        for (d, &s) in dst_sl.iter_mut().zip(src_sl) {
                                            *d = s + bb;
                                        }
                                    } else {
                                        dst_sl.copy_from_slice(src_sl);
                                    }
                                }
                            });
                            crate::pool::give(tmp);
                        } else {
                            use rayon::prelude::*;
                            out.par_chunks_mut(c_out * l).enumerate().for_each_init(
                                || vec![0.0f32; kdim * l],
                                |cols, (b, o)| {
                                    if padding.0 > 0 || padding.1 > 0 {
                                        cols.fill(0.0);
                                    }
                                    unfold_fill_seq(&src[b * c_in * h * w..(b + 1) * c_in * h * w], cols, &geom);
                                    crate::gemm::gemm_f32_strided_seq(wv, 0, kdim as isize, 1, cols, 0, l as isize, 1, o, c_out, kdim, l);
                                    if let Some(bv) = &bias_v {
                                        o.chunks_mut(l).zip(bv).for_each(|(ch, &bb)| ch.iter_mut().for_each(|v| *v += bb));
                                    }
                                },
                            );
                        }
                        #[cfg(not(feature = "rayon"))]
                        out.chunks_mut(c_out * l).enumerate().for_each(image);
                    }
                    out
                })
            })?;
            if let Some(inner) = fused
                && let Some(out) = inner?
            {
                return Ok(Tensor::from_storage(Storage::new_cpu(CpuStorage::F32(out)), Layout::contiguous(vec![n, c_out, out_h, out_w])));
            }
        }
        let cols = self.unfold2d((kh, kw), stride, padding, dilation)?; // [N, C_in*kH*kW, L]
        let w_flat = weight.reshape(&[1, c_out, c_in * kh * kw])?;
        let out = w_flat.matmul(&cols)?.reshape(&[n, c_out, out_h, out_w])?;
        match bias {
            Some(b) => out.add(&b.reshape(&[1, c_out, 1, 1])?),
            None => Ok(out),
        }
    }

    /// 1D convolution on `input: [N, C_in, L]` with `weight: [C_out, C_in, K]`.
    pub fn conv1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        if self.ndim() != 3 || weight.ndim() != 3 {
            return Err(Error::Msg("conv1d requires 3D input and 3D weight".into()));
        }
        let x4 = self.unsqueeze(2)?; // [N, C_in, 1, L]
        let w4 = weight.unsqueeze(2)?; // [C_out, C_in, 1, K]
        let y4 = x4.conv2d(&w4, bias, (1, stride), (0, padding), (1, dilation))?;
        y4.squeeze_axis(2)
    }

    /// Grouped 1D convolution supporting `groups >= 1`.
    pub fn conv1d_grouped(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        if self.ndim() != 3 || weight.ndim() != 3 {
            return Err(Error::Msg("conv1d requires 3D input and 3D weight".into()));
        }
        let x4 = self.unsqueeze(2)?; // [N, C_in, 1, L]
        let w4 = weight.unsqueeze(2)?; // [C_out, C_in / groups, 1, K]
        let y4 = x4.conv2d_grouped(&w4, bias, (1, stride), (0, padding), (1, dilation), groups)?;
        y4.squeeze_axis(2)
    }

    /// 2D Max Pooling over `[N, C, H, W]`.
    pub fn max_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Msg(
                "max_pool2d expects 4D tensor [N, C, H, W]".into(),
            ));
        }
        let (n, c, h, w) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        );
        let (kh, kw) = kernel_size;
        if kh == 0 || kw == 0 {
            return Err(Error::Msg(
                "max_pool2d: kernel_size must be positive".into(),
            ));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(Error::Msg("max_pool2d: stride must be positive".into()));
        }
        if h + 2 * padding.0 < kh || w + 2 * padding.1 < kw {
            return Err(Error::Msg(
                "max_pool2d: kernel size exceeds padded input".into(),
            ));
        }
        // Torch parity: padding may be at most half the kernel (per side).
        if 2 * padding.0 > kh || 2 * padding.1 > kw {
            return Err(Error::Msg(format!(
                "max_pool2d: padding ({}, {}) must be at most half of kernel size ({kh}, {kw})",
                padding.0, padding.1
            )));
        }
        let out_h = (h + 2 * padding.0 - kh) / stride.0 + 1;
        let out_w = (w + 2 * padding.1 - kw) / stride.1 + 1;
        let l = out_h * out_w;

        // contiguous f32 host images: one direct pass per plane (the general path below builds an im2col copy kh*kw
        // times the size of the input and reduces it)
        let geom = UnfoldGeom { h, w, kh, kw, sh: stride.0, sw: stride.1, ph: padding.0, pw: padding.1, dh: 1, dw: 1, out_h, out_w };
        let needs_grad = crate::tensor::is_grad_enabled() && self.requires_grad();
        let direct = with_f32_host(self, |src| {
            let mut out = crate::pool::take_f32(n * c * l).unwrap_or_else(|| vec![0.0; n * c * l]);
            let mut arg = if needs_grad { vec![0i64; n * c * l] } else { Vec::new() };
            max_pool_fill(src, &mut out, &mut arg, &geom);
            (out, arg)
        })?;
        if let Some((out, arg)) = direct {
            let storage = Storage::new_cpu(CpuStorage::F32(out));
            let out_layout = Layout::contiguous(vec![n, c, out_h, out_w]);
            if !needs_grad {
                return Ok(Tensor::from_storage(storage, out_layout));
            }
            let arg = Tensor::from_slice(&arg, &[n, c, l])?;
            return Ok(Tensor::from_op(storage, out_layout, "max_pool2d", vec![self.clone()], move |g| {
                // each output sends its gradient to the input position that won its window
                let zeros = Tensor::zeros(&[n, c, h * w], g.dtype(), &g.device())?;
                Ok(vec![Some(zeros.scatter_add(2, &arg, &g.reshape(&[n, c, l])?)?.reshape(&[n, c, h, w])?)])
            }));
        }
        let cols = self.unfold2d(kernel_size, stride, padding, (1, 1))?; // [N, C*kh*kw, L]
        // unfold zero-fills padding, which would win over all-negative windows;
        // exclude padded entries (torch parity). Skipped for bool, where the
        // zero fill (false) already loses correctly, and when unpadded.
        // The -inf fill saturates to the dtype minimum for integer inputs.
        let cols = if (padding.0 != 0 || padding.1 != 0) && self.dtype() != DType::Bool {
            let validity = self
                .ones_like()?
                .unfold2d(kernel_size, stride, padding, (1, 1))?;
            let is_real = validity.gt(&validity.zeros_like()?)?;
            let fill = cols.full_like(f64::NEG_INFINITY)?;
            Tensor::where_cond(&is_real, &cols, &fill)?
        } else {
            cols
        };
        let cols_reshaped = cols.reshape(&[n, c, kh * kw, l])?;
        cols_reshaped
            .max_axis(2, false)?
            .reshape(&[n, c, out_h, out_w])
    }

    /// 2D Average Pooling over `[N, C, H, W]`.
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Msg(
                "avg_pool2d expects 4D tensor [N, C, H, W]".into(),
            ));
        }
        let (n, c, h, w) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        );
        let (kh, kw) = kernel_size;
        if kh == 0 || kw == 0 {
            return Err(Error::Msg(
                "avg_pool2d: kernel_size must be positive".into(),
            ));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(Error::Msg("avg_pool2d: stride must be positive".into()));
        }
        if h + 2 * padding.0 < kh || w + 2 * padding.1 < kw {
            return Err(Error::Msg(
                "avg_pool2d: kernel size exceeds padded input".into(),
            ));
        }
        // Torch parity: padding may be at most half the kernel (per side).
        if 2 * padding.0 > kh || 2 * padding.1 > kw {
            return Err(Error::Msg(format!(
                "avg_pool2d: padding ({}, {}) must be at most half of kernel size ({kh}, {kw})",
                padding.0, padding.1
            )));
        }
        let out_h = (h + 2 * padding.0 - kh) / stride.0 + 1;
        let out_w = (w + 2 * padding.1 - kw) / stride.1 + 1;
        let l = out_h * out_w;

        let cols = self.unfold2d(kernel_size, stride, padding, (1, 1))?;
        let cols_reshaped = cols.reshape(&[n, c, kh * kw, l])?;
        cols_reshaped
            .mean_axis(2, false)?
            .reshape(&[n, c, out_h, out_w])
    }

    /// Global average pooling `[N, C, H, W] -> [N, C, 1, 1]`.
    pub fn global_avg_pool2d(&self) -> Result<Tensor> {
        self.mean_axes(&[2, 3], true)
    }

    /// Grouped 2D convolution supporting `groups >= 1` (including depthwise `groups == C_in`).
    /// `input: [N, C_in, H, W]`, `weight: [C_out, C_in / groups, kH, kW]`.
    pub fn conv2d_grouped(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Tensor> {
        if let Some((x, w)) = crate::amp::lowp_pair(self, weight)? {
            let b = bias.map(crate::amp::autocast_cast).transpose()?;
            return x.conv2d_grouped(&w, b.as_ref(), stride, padding, dilation, groups);
        }
        if groups == 1 {
            return self.conv2d(weight, bias, stride, padding, dilation);
        }
        if groups == 0 {
            return Err(Error::Msg("conv2d_grouped: groups must be positive".into()));
        }
        if self.ndim() != 4 || weight.ndim() != 4 {
            return Err(Error::Msg(format!(
                "conv2d_grouped expects 4D input/weight, got {:?}/{:?}",
                self.shape(),
                weight.shape()
            )));
        }
        let c_in = self.shape()[1];
        let c_out = weight.shape()[0];
        if c_in % groups != 0 || c_out % groups != 0 {
            return Err(Error::Msg(
                "conv2d_grouped: in_channels and out_channels must be divisible by groups".into(),
            ));
        }
        // With zero channels the divisibility check passes for any `groups`
        // (0 % g == 0); bound it so the loop below cannot spin uselessly.
        if groups > c_in.max(c_out).max(1) {
            return Err(Error::Msg(format!(
                "conv2d_grouped: groups ({groups}) exceeds channels [{c_in}, {c_out}]"
            )));
        }
        let cin_g = c_in / groups;
        let cout_g = c_out / groups;
        let mut group_outs = Vec::with_capacity(groups);
        for g in 0..groups {
            let xg = self.narrow(1, g * cin_g, cin_g)?;
            let wg = weight.narrow(0, g * cout_g, cout_g)?;
            let bg = match bias {
                Some(b) => Some(b.narrow(0, g * cout_g, cout_g)?),
                None => None,
            };
            group_outs.push(xg.conv2d(&wg, bg.as_ref(), stride, padding, dilation)?);
        }
        Self::cat(&group_outs, 1)
    }

    /// 2D Transposed Convolution (`ConvTranspose2d` / deconvolution) on `input: [N, C_in, H, W]`
    /// with `weight: [C_in, C_out, kH, kW]` and optional `bias: [C_out]`.
    /// Implemented via differentiable `matmul` + [`Tensor::fold2d`] (`col2im`)!
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 4 || weight.ndim() != 4 {
            return Err(Error::Msg(
                "conv_transpose2d requires 4D input [N, C_in, H, W] and 4D weight [C_in, C_out, kH, kW]"
                    .into(),
            ));
        }
        let (n, c_in, h, w) = (
            self.shape()[0],
            self.shape()[1],
            self.shape()[2],
            self.shape()[3],
        );
        let (wc_in, c_out, kh, kw) = (
            weight.shape()[0],
            weight.shape()[1],
            weight.shape()[2],
            weight.shape()[3],
        );
        if c_in != wc_in {
            return Err(Error::ShapeMismatch {
                op: "conv_transpose2d",
                expected: vec![c_in, c_out, kh, kw],
                got: weight.shape().to_vec(),
            });
        }
        if kh == 0 || kw == 0 {
            return Err(Error::Msg(
                "conv_transpose2d: kernel size must be positive".into(),
            ));
        }
        if h == 0 || w == 0 {
            return Err(Error::Msg(
                "conv_transpose2d: input must have non-empty spatial dims".into(),
            ));
        }
        let conv_t_out =
            |x: usize, s: usize, p: usize, d: usize, k: usize, op: usize| -> Result<usize> {
                let eff = (d as u128) * ((k - 1) as u128) + 1;
                let base = ((x - 1) as u128) * (s as u128) + eff + (op as u128);
                let sub = 2 * (p as u128);
                if base < sub {
                    return Err(Error::Msg(
                        "conv_transpose2d: padding exceeds the upsampled output size".into(),
                    ));
                }
                usize::try_from(base - sub)
                    .map_err(|_| Error::Msg("conv_transpose2d: output size overflows usize".into()))
            };
        let out_h = conv_t_out(h, stride.0, padding.0, dilation.0, kh, output_padding.0)?;
        let out_w = conv_t_out(w, stride.1, padding.1, dilation.1, kw, output_padding.1)?;
        if stride.0 == 0 || stride.1 == 0 {
            return Err(Error::Msg(
                "conv_transpose2d: stride must be positive".into(),
            ));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(Error::Msg(
                "conv_transpose2d: dilation must be positive".into(),
            ));
        }
        // Torch parity: output_padding must be smaller than stride or dilation (per dim).
        for (o, s, d, name) in [
            (output_padding.0, stride.0, dilation.0, "height"),
            (output_padding.1, stride.1, dilation.1, "width"),
        ] {
            if o >= s && o >= d {
                return Err(Error::Msg(format!(
                    "conv_transpose2d: output_padding ({o}) must be smaller than stride ({s}) or dilation ({d}) ({name})"
                )));
            }
        }
        let l = h * w;

        // weight: [C_in, C_out * kH * kW] -> transpose -> [1, C_out * kH * kW, C_in]
        let col_c = c_out
            .checked_mul(kh)
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| Error::Msg("conv_transpose2d: channel arithmetic overflows".into()))?;
        let w_mat = weight.reshape(&[c_in, col_c])?.t()?.unsqueeze(0)?;
        let x_flat = self.reshape(&[n, c_in, l])?;
        let cols = w_mat.matmul(&x_flat)?; // [N, C_out * kH * kW, L]
        // Scatter over the input grid (h, w): with output_padding the output is
        // bigger than fold2d's forward-size check accepts, so bypass it; the
        // fringe cells no window covers stay zero, matching torch.
        let col_vec = cols.to_vec::<f64>()?;
        let img = Self::fold2d_scatter(
            &col_vec,
            n,
            c_out,
            col_c,
            Fold2dGeo {
                h: out_h,
                w: out_w,
                gh: h,
                gw: w,
                kh,
                kw,
                sh: stride.0,
                sw: stride.1,
                ph: padding.0,
                pw: padding.1,
                dh: dilation.0,
                dw: dilation.1,
            },
        )?;
        let out_shape = vec![n, c_out, out_h, out_w];
        let cpu =
            CpuStorage::F64(img).to_dtype(&Layout::contiguous(out_shape.clone()), self.dtype())?;
        let storage = Storage::new_cpu(cpu)
            .to_device(&Layout::contiguous(out_shape.clone()), &self.device())?;
        // The unfolded gradient spans the full (H', W') grid; keep the (h, w)
        // subgrid whose columns exist (a no-op view when output_padding = 0).
        let eff_kh = (dilation.0 as u128) * ((kh - 1) as u128) + 1;
        let eff_kw = (dilation.1 as u128) * ((kw - 1) as u128) + 1;
        let full_gh = usize::try_from(
            (out_h as u128 + 2 * (padding.0 as u128) - eff_kh) / (stride.0 as u128) + 1,
        )
        .map_err(|_| Error::Msg("conv_transpose2d: gradient grid overflows".into()))?;
        let full_gw = usize::try_from(
            (out_w as u128 + 2 * (padding.1 as u128) - eff_kw) / (stride.1 as u128) + 1,
        )
        .map_err(|_| Error::Msg("conv_transpose2d: gradient grid overflows".into()))?;
        let out = Tensor::from_op(
            storage,
            Layout::contiguous(out_shape),
            "conv_transpose2d",
            vec![cols],
            move |g| {
                let full = g.unfold2d((kh, kw), stride, padding, dilation)?;
                let sub = full
                    .reshape(&[n, col_c, full_gh, full_gw])?
                    .narrow(2, 0, h)?
                    .narrow(3, 0, w)?
                    .reshape(&[n, col_c, l])?;
                Ok(vec![Some(sub)])
            },
        );
        match bias {
            Some(b) => out.add(&b.reshape(&[1, c_out, 1, 1])?),
            None => Ok(out),
        }
    }

    /// 1D Transposed Convolution on `input: [N, C_in, L]` with `weight: [C_in, C_out, K]`.
    pub fn conv_transpose1d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> Result<Tensor> {
        let x4 = self.unsqueeze(2)?;
        let w4 = weight.unsqueeze(2)?;
        let y4 = x4.conv_transpose2d(
            &w4,
            bias,
            (1, stride),
            (0, padding),
            (0, output_padding),
            (1, dilation),
        )?;
        y4.squeeze_axis(2)
    }

    /// 1D Max Pooling over `[N, C, L]`.
    pub fn max_pool1d(&self, kernel_size: usize, stride: usize, padding: usize) -> Result<Tensor> {
        if self.ndim() != 3 {
            return Err(Error::Msg("max_pool1d expects 3D input [N, C, L]".into()));
        }
        let x4 = self.unsqueeze(2)?;
        let y4 = x4.max_pool2d((1, kernel_size), (1, stride), (0, padding))?;
        y4.squeeze_axis(2)
    }

    /// 1D Average Pooling over `[N, C, L]`.
    pub fn avg_pool1d(&self, kernel_size: usize, stride: usize, padding: usize) -> Result<Tensor> {
        if self.ndim() != 3 {
            return Err(Error::Msg("avg_pool1d expects 3D input [N, C, L]".into()));
        }
        let x4 = self.unsqueeze(2)?;
        let y4 = x4.avg_pool2d((1, kernel_size), (1, stride), (0, padding))?;
        y4.squeeze_axis(2)
    }

    /// Adaptive 1D average pooling to target length `output_size` (`[N, C, L] -> [N, C, output_size]`).
    pub fn adaptive_avg_pool1d(&self, output_size: usize) -> Result<Tensor> {
        if self.ndim() != 3 {
            return Err(Error::Msg(
                "adaptive_avg_pool1d expects 3D input [N, C, L]".into(),
            ));
        }
        let x4 = self.unsqueeze(2)?;
        let y4 = x4.adaptive_avg_pool2d((1, output_size))?;
        y4.squeeze_axis(2)
    }

    /// Adaptive 1D max pooling to target length `output_size` (`[N, C, L] -> [N, C, output_size]`).
    pub fn adaptive_max_pool1d(&self, output_size: usize) -> Result<Tensor> {
        if self.ndim() != 3 {
            return Err(Error::Msg(
                "adaptive_max_pool1d expects 3D input [N, C, L]".into(),
            ));
        }
        let x4 = self.unsqueeze(2)?;
        let y4 = x4.adaptive_max_pool2d((1, output_size))?;
        y4.squeeze_axis(2)
    }

    /// 1D Power-average ($L_p$) pooling: $(\sum x^p)^{1/p}$ over `[N, C, L]`.
    pub fn lp_pool1d(&self, norm_type: f64, kernel_size: usize, stride: usize) -> Result<Tensor> {
        if self.ndim() != 3 {
            return Err(Error::Msg("lp_pool1d expects 3D input [N, C, L]".into()));
        }
        if norm_type.is_infinite() {
            return self.abs()?.max_pool1d(kernel_size, stride, 0);
        }
        let p_powered = self.abs()?.powf(norm_type)?;
        let summed = p_powered
            .avg_pool1d(kernel_size, stride, 0)?
            .mul_scalar(kernel_size as f64)?;
        summed.powf(1.0 / norm_type)
    }

    /// 2D Power-average ($L_p$) pooling: $(\sum x^p)^{1/p}$ over `[N, C, H, W]`.
    pub fn lp_pool2d(
        &self,
        norm_type: f64,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Msg("lp_pool2d expects 4D input [N, C, H, W]".into()));
        }
        if norm_type.is_infinite() {
            return self.abs()?.max_pool2d(kernel_size, stride, (0, 0));
        }
        let p_powered = self.abs()?.powf(norm_type)?;
        let (kh, kw) = kernel_size;
        let summed = p_powered
            .avg_pool2d(kernel_size, stride, (0, 0))?
            .mul_scalar((kh * kw) as f64)?;
        summed.powf(1.0 / norm_type)
    }

    /// Adaptive 2D average pooling to any target spatial size `(out_h, out_w)`.
    pub fn adaptive_avg_pool2d(&self, output_size: (usize, usize)) -> Result<Tensor> {
        self.adaptive_pool2d(output_size, false)
    }

    /// Adaptive 2D max pooling to any target spatial size `(out_h, out_w)`.
    pub fn adaptive_max_pool2d(&self, output_size: (usize, usize)) -> Result<Tensor> {
        self.adaptive_pool2d(output_size, true)
    }

    fn adaptive_pool2d(&self, output_size: (usize, usize), is_max: bool) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Msg("adaptive_pool2d expects 4D [N, C, H, W]".into()));
        }
        let (h, w) = (self.shape()[2], self.shape()[3]);
        let (oh, ow) = output_size;
        if oh == 1 && ow == 1 && !is_max {
            return self.global_avg_pool2d();
        }
        // Cap the output BEFORE the oh*ow patch loop: huge sizes would spin
        // (and overflow (i+1)*h) long before any downstream check fires.
        crate::limits::check_elements(
            "adaptive_pool2d",
            &[self.shape()[0], self.shape()[1], oh, ow],
        )?;
        let mut row_tensors = Vec::with_capacity(oh);
        for i in 0..oh {
            let h_start = (i * h) / oh;
            let h_end = ((i + 1) * h).div_ceil(oh).max(h_start + 1);
            let h_slice = self.narrow(2, h_start, h_end - h_start)?;
            let mut col_tensors = Vec::with_capacity(ow);
            for j in 0..ow {
                let w_start = (j * w) / ow;
                let w_end = ((j + 1) * w).div_ceil(ow).max(w_start + 1);
                let patch = h_slice.narrow(3, w_start, w_end - w_start)?;
                let pooled = if is_max {
                    patch.max_axis(2, true)?.max_axis(3, true)?
                } else {
                    patch.mean_axes(&[2, 3], true)?
                };
                col_tensors.push(pooled);
            }
            row_tensors.push(Self::cat(&col_tensors, 3)?);
        }
        Self::cat(&row_tensors, 2)
    }

    /// Differentiable 2D nearest-neighbor upsampling to `output_size = (out_h, out_w)`.
    pub fn upsample_nearest2d(&self, output_size: (usize, usize)) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Msg(
                "upsample_nearest2d expects 4D [N, C, H, W]".into(),
            ));
        }
        let (h, w) = (self.shape()[2], self.shape()[3]);
        let (oh, ow) = output_size;
        if h == 0 || w == 0 {
            return Err(Error::Msg(
                "upsample_nearest2d: input spatial dims must be non-empty".into(),
            ));
        }
        let (n, c) = (self.shape()[0], self.shape()[1]);
        // Cap the output BEFORE building oh+ow index vectors: huge sizes would
        // hang here, and i*h would wrap and silently sample wrong pixels.
        // (With an empty batch/channels the product check cannot bound oh/ow,
        // so return the empty output directly instead of looping oh*ow times.)
        crate::limits::check_elements("upsample_nearest2d", &[n, c, oh, ow])?;
        // Bound oh*ow on its own too: with an empty batch/channels the product
        // above is 0 and cannot bound the spatial dims for stride computation.
        crate::limits::check_elements("upsample_nearest2d", &[oh, ow])?;
        if n == 0 || c == 0 {
            return Tensor::zeros(&[n, c, oh, ow], self.dtype(), &self.device());
        }
        let idx_h: Vec<i64> = (0..oh).map(|i| ((i * h) / oh).min(h - 1) as i64).collect();
        let idx_w: Vec<i64> = (0..ow).map(|j| ((j * w) / ow).min(w - 1) as i64).collect();
        let th = Tensor::from_slice_on(&idx_h, &[oh], &self.device())?;
        let tw = Tensor::from_slice_on(&idx_w, &[ow], &self.device())?;
        self.index_select(2, &th)?.index_select(3, &tw)
    }

    /// Differentiable 2D bilinear interpolation to `output_size = (out_h, out_w)`.
    pub fn upsample_bilinear2d(
        &self,
        output_size: (usize, usize),
        align_corners: bool,
    ) -> Result<Tensor> {
        if self.ndim() != 4 {
            return Err(Error::Msg(
                "upsample_bilinear2d expects 4D [N, C, H, W]".into(),
            ));
        }
        let (h, w) = (self.shape()[2], self.shape()[3]);
        let (oh, ow) = output_size;
        let dev = self.device();
        let dtype = self.dtype();
        if h == 0 || w == 0 {
            return Err(Error::Msg(
                "upsample_bilinear2d: input spatial dims must be non-empty".into(),
            ));
        }
        let (n, c) = (self.shape()[0], self.shape()[1]);
        // Cap the output BEFORE the oh+ow weight loops (same hang rationale as
        // upsample_nearest2d); empty batch/channels return the empty output.
        crate::limits::check_elements("upsample_bilinear2d", &[n, c, oh, ow])?;
        // Bound oh*ow on its own too: with an empty batch/channels the product
        // above is 0 and cannot bound the spatial dims for stride computation.
        crate::limits::check_elements("upsample_bilinear2d", &[oh, ow])?;
        if n == 0 || c == 0 {
            return Tensor::zeros(&[n, c, oh, ow], dtype, &dev);
        }

        let compute_axis_weights =
            |in_len: usize, out_len: usize| -> (Vec<i64>, Vec<i64>, Vec<f64>) {
                let mut i0 = Vec::with_capacity(out_len);
                let mut i1 = Vec::with_capacity(out_len);
                let mut lam = Vec::with_capacity(out_len);
                for o in 0..out_len {
                    let src = if in_len == 1 {
                        0.0
                    } else if align_corners && out_len > 1 {
                        (o as f64) * ((in_len - 1) as f64) / ((out_len - 1) as f64)
                    } else {
                        ((o as f64 + 0.5) * (in_len as f64) / (out_len as f64) - 0.5).max(0.0)
                    };
                    let lo = (src.floor() as usize).min(in_len - 1);
                    let hi = (lo + 1).min(in_len - 1);
                    i0.push(lo as i64);
                    i1.push(hi as i64);
                    lam.push(src - (lo as f64));
                }
                (i0, i1, lam)
            };

        let (h0, h1, wy_vec) = compute_axis_weights(h, oh);
        let (w0, w1, wx_vec) = compute_axis_weights(w, ow);

        let th0 = Tensor::from_slice_on(&h0, &[oh], &dev)?;
        let th1 = Tensor::from_slice_on(&h1, &[oh], &dev)?;
        let tw0 = Tensor::from_slice_on(&w0, &[ow], &dev)?;
        let tw1 = Tensor::from_slice_on(&w1, &[ow], &dev)?;

        let wy = Tensor::from_slice_on(&wy_vec, &[1, 1, oh, 1], &dev)?.to_dtype(dtype)?;
        let wy_inv = wy.neg()?.add_scalar(1.0)?;
        let wx = Tensor::from_slice_on(&wx_vec, &[1, 1, 1, ow], &dev)?.to_dtype(dtype)?;
        let wx_inv = wx.neg()?.add_scalar(1.0)?;

        let top = self.index_select(2, &th0)?;
        let bot = self.index_select(2, &th1)?;
        let interp_h = top.mul(&wy_inv)?.add(&bot.mul(&wy)?)?;

        let left = interp_h.index_select(3, &tw0)?;
        let right = interp_h.index_select(3, &tw1)?;
        left.mul(&wx_inv)?.add(&right.mul(&wx)?)
    }

    // =========================================================================
    // 10. One-Hot Encoding & Stochastic Sampling (`multinomial`, `bernoulli`)
    // =========================================================================

    /// One-hot encodes an integer tensor of indices `[...]` into `[..., num_classes]`.
    pub fn one_hot(&self, num_classes: usize, dtype: DType) -> Result<Tensor> {
        let idx = self.to_vec::<i64>()?;
        let mut early_shape = self.shape().to_vec();
        early_shape.push(num_classes);
        // Refuse outputs that can never fit BEFORE allocating: idx.len() (from a
        // validated tensor) times user-controlled num_classes can overflow or OOM.
        let out_n = crate::limits::check_elements("one_hot", &early_shape)?;
        let mut out = crate::layout::try_filled(out_n, 0.0f64)?;
        for (row, &c) in idx.iter().enumerate() {
            if c < 0 || c as usize >= num_classes {
                return Err(Error::IndexOutOfBounds {
                    op: "one_hot",
                    index: c as isize,
                    dim_size: num_classes,
                });
            }
            out[row * num_classes + c as usize] = 1.0;
        }
        let mut out_shape = self.shape().to_vec();
        out_shape.push(num_classes);
        let cpu = CpuStorage::F64(out).to_dtype(&Layout::contiguous(out_shape.clone()), dtype)?;
        let storage = Storage::new_cpu(cpu)
            .to_device(&Layout::contiguous(out_shape.clone()), &self.device())?;
        Ok(Tensor::from_storage(storage, Layout::contiguous(out_shape)))
    }

    /// Draws `num_samples` indices (`DType::I64`) from each row of non-negative unnormalized
    /// probabilities (`[C]` or `[N, C]`), using an optional [`crate::rng::Generator`].
    pub fn multinomial_with_generator(
        &self,
        num_samples: usize,
        replacement: bool,
        mut generator: Option<&mut crate::rng::Generator>,
    ) -> Result<Tensor> {
        if self.ndim() != 1 && self.ndim() != 2 {
            return Err(Error::Msg(
                "multinomial expects 1D [C] or 2D [N, C] probs".into(),
            ));
        }
        let is_1d = self.ndim() == 1;
        let (rows, cols) = if is_1d {
            (1, self.shape()[0])
        } else {
            (self.shape()[0], self.shape()[1])
        };
        if !replacement && num_samples > cols {
            return Err(Error::Msg(
                "multinomial without replacement: num_samples > num_categories".into(),
            ));
        }
        crate::limits::check_iterations("num_samples", num_samples)?;
        let probs = self.to_vec::<f64>()?;
        let mut sampled =
            crate::layout::try_with_capacity(crate::layout::checked_numel(&[rows, num_samples])?)?;

        for r in 0..rows {
            let mut row_w = probs[r * cols..(r + 1) * cols].to_vec();
            for &w in &row_w {
                if w < 0.0 || w.is_nan() {
                    return Err(Error::Msg("multinomial probabilities must be >= 0".into()));
                }
            }
            for _ in 0..num_samples {
                let sum: f64 = row_w.iter().sum();
                if sum <= 0.0 {
                    return Err(Error::Msg(
                        "multinomial row sum of probabilities must be > 0".into(),
                    ));
                }
                let rand_val = if let Some(ref mut g) = generator {
                    g.next_f64()
                } else {
                    crate::rng::with_rng(|rng| rng.next_f32() as f64)
                };
                let u = rand_val * sum;
                let mut acc = 0.0f64;
                let mut chosen = cols - 1;
                for (c, &w) in row_w.iter().enumerate() {
                    acc += w;
                    if u <= acc && w > 0.0 {
                        chosen = c;
                        break;
                    }
                }
                sampled.push(chosen as i64);
                if !replacement {
                    row_w[chosen] = 0.0;
                }
            }
        }

        let out_shape = if is_1d {
            vec![num_samples]
        } else {
            vec![rows, num_samples]
        };
        Tensor::from_slice_on(&sampled, &out_shape, &self.device())
    }

    /// Draws `num_samples` indices (`DType::I64`) from each row of non-negative unnormalized
    /// probabilities (`[C]` or `[N, C]`).
    pub fn multinomial(&self, num_samples: usize, replacement: bool) -> Result<Tensor> {
        self.multinomial_with_generator(num_samples, replacement, None)
    }

    /// Samples a single token index from 1D logits using [`crate::sampling::SamplingConfig`].
    pub fn sample_token(
        &self,
        config: &crate::sampling::SamplingConfig,
        history: Option<&[usize]>,
        generator: Option<&mut crate::rng::Generator>,
    ) -> Result<usize> {
        config.sample_token(self, history, generator)
    }

    /// Samples binary `0.0` / `1.0` values from Bernoulli distributions with probabilities `self`.
    pub fn bernoulli(&self) -> Result<Tensor> {
        let p = self.to_vec::<f32>()?;
        let samples: Vec<f32> = crate::rng::with_rng(|rng| {
            p.into_iter()
                .map(|prob| if rng.next_f32() < prob { 1.0 } else { 0.0 })
                .collect()
        });
        Tensor::from_slice_on(&samples, self.shape(), &self.device())?.to_dtype(self.dtype())
    }

    // =========================================================================
    // 11. PyTorch Parity Tensor Manipulation & Math (`unfold`, `roll`, `tile`,
    //     `repeat_interleave`, `unbind`, `diag_embed`, `cumprod`, `diff`,
    //     `lerp`, `atan2`, `hypot`, `nan_to_num`, `bincount`, `histc`,
    //     `searchsorted`, `meshgrid`)
    // =========================================================================

    /// **Zero-Copy Strided Sliding Window (`unfold`)**:
    /// Returns a zero-copy view of `self` containing all slices of size `size` along `dim`
    /// stepped by `step`. The new window dimension of length `size` is appended as the last axis.
    /// Differentiable to arbitrary order: [`Tensor::as_strided`]'s backward scatters each
    /// window's gradient back to its source elements (overlapping windows accumulate).
    pub fn unfold(&self, dim: isize, size: usize, step: usize) -> Result<Tensor> {
        let ax = normalize_axis("unfold", dim, self.ndim())?;
        let dim_len = self.shape()[ax];
        if size == 0 || step == 0 || size > dim_len {
            return Err(Error::Msg(format!(
                "unfold: invalid size={size}, step={step} for dim_len={dim_len}"
            )));
        }
        let num_windows = (dim_len - size) / step + 1;
        let mut new_shape = self.shape().to_vec();
        new_shape[ax] = num_windows;
        new_shape.push(size);

        let step_i = isize::try_from(step)
            .map_err(|_| Error::Msg(format!("unfold: step={step} exceeds isize::MAX")))?;
        let old_strides = self.strides();
        let mut new_strides = old_strides.to_vec();
        // A wrapped stride could still pass as_strided's bounds check with a
        // smaller value and silently return wrong windows: refuse overflow.
        new_strides[ax] = old_strides[ax].checked_mul(step_i).ok_or_else(|| {
            Error::Msg(format!(
                "unfold: stride {} x step {step} overflows",
                old_strides[ax]
            ))
        })?;
        new_strides.push(old_strides[ax]);

        self.as_strided(&new_shape, &new_strides, self.storage_offset())
    }

    /// Cyclically shifts elements along `dim` by `shift` positions (wrapping around).
    pub fn roll(&self, shift: isize, dim: isize) -> Result<Tensor> {
        let ax = normalize_axis("roll", dim, self.ndim())?;
        let len = self.shape()[ax] as isize;
        if len <= 1 {
            return Ok(self.clone());
        }
        let s = shift.rem_euclid(len) as usize;
        if s == 0 {
            return Ok(self.clone());
        }
        let split_pt = (len as usize) - s;
        let part1 = self.narrow(ax as isize, split_pt, s)?;
        let part2 = self.narrow(ax as isize, 0, split_pt)?;
        Tensor::cat(&[part1, part2], ax as isize)
    }

    /// Constructs a tensor by repeating `self` according to `reps` along each dimension (`torch.tile`).
    /// When `reps` is longer than `self.ndim()`, the input is promoted with
    /// leading size-1 dims; when shorter, `reps` is promoted with leading 1s.
    pub fn tile(&self, reps: &[usize]) -> Result<Tensor> {
        let ndim = self.ndim();
        if reps.len() >= ndim {
            let mut shape = vec![1usize; reps.len() - ndim];
            shape.extend_from_slice(self.shape());
            self.reshape(&shape)?.repeat(reps)
        } else {
            let mut full = vec![1usize; ndim - reps.len()];
            full.extend_from_slice(reps);
            self.repeat(&full)
        }
    }

    /// Repeats each element along `dim` `repeats` times (`[a, b] -> [a, a, b, b]`).
    pub fn repeat_interleave(&self, repeats: usize, dim: isize) -> Result<Tensor> {
        let ax = normalize_axis("repeat_interleave", dim, self.ndim())?;
        if repeats == 0 {
            // torch parity: zero repeats yields an empty dim, not an error.
            return self.narrow(ax as isize, 0, 0);
        }
        let mut exp_shape = self.shape().to_vec();
        exp_shape.insert(ax + 1, repeats);
        let mut final_shape = self.shape().to_vec();
        final_shape[ax] = final_shape[ax].checked_mul(repeats).ok_or_else(|| {
            Error::Msg(format!(
                "repeat_interleave: output dim {ax} ({} x {repeats}) overflows",
                self.shape()[ax]
            ))
        })?;
        crate::limits::check_elements("repeat_interleave", &final_shape)?;

        self.unsqueeze((ax + 1) as isize)?
            .broadcast_to(&exp_shape)?
            .contiguous()?
            .reshape(&final_shape)
    }

    /// Removes a tensor dimension by returning a `Vec<Tensor>` of all zero-copy slices along `dim` (`torch.unbind`).
    pub fn unbind(&self, dim: isize) -> Result<Vec<Tensor>> {
        let ax = normalize_axis("unbind", dim, self.ndim())?;
        let len = self.shape()[ax];
        // Each slice is a fat Tensor handle (~100 B): unbinding a huge dim
        // would OOM-abort on the Vec reserve, so cap the fan-out up front.
        crate::limits::check_repeats("unbind", len)?;
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            out.push(self.select(ax as isize, i as isize)?);
        }
        Ok(out)
    }

    /// Embeds a 1D vector `[N]` into a 2D diagonal matrix `[N, N]` (differentiable via `eye` masking).
    pub fn diag_embed(&self) -> Result<Tensor> {
        if self.ndim() == 0 {
            return Err(Error::Msg("diag_embed expects a tensor of rank >= 1 ([..., N])".into()));
        }
        // [..., N] -> [..., N, N] with the last axis on the diagonal (torch.diag_embed with default offset/dims)
        let n = self.shape()[self.ndim() - 1];
        let eye = Tensor::eye(n, self.dtype(), &self.device())?;
        let mut shape = self.shape().to_vec();
        shape.push(n);
        self.unsqueeze(-2)?.broadcast_to(&shape)?.mul(&eye)
    }

    /// Differentiable cumulative product $\prod_{k \le i} x_k$ along `axis` (`torch.cumprod`).
    pub fn cumprod(&self, axis: isize) -> Result<Tensor> {
        let ax = normalize_axis("cumprod", axis, self.ndim())?;
        let shape = self.shape().to_vec();
        let dim_len = shape[ax];
        let outer: usize = shape[..ax].iter().product();
        let inner: usize = shape[ax + 1..].iter().product();

        let x_vec = self.to_vec::<f64>()?;
        let mut out_vec = vec![0.0f64; x_vec.len()];

        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 1.0f64;
                for d in 0..dim_len {
                    let idx = (o * dim_len + d) * inner + i;
                    acc *= x_vec[idx];
                    out_vec[idx] = acc;
                }
            }
        }

        let out_layout = Layout::contiguous(shape.clone());
        let cpu_s = CpuStorage::F64(out_vec.clone()).to_dtype(&out_layout, self.dtype())?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &self.device())?;

        let dt = self.dtype();
        let dev = self.device();
        Ok(Tensor::from_op(
            storage,
            out_layout,
            "cumprod",
            vec![self.clone()],
            move |grad_out| {
                let go = grad_out.to_vec::<f64>()?;
                let mut gx = vec![0.0f64; x_vec.len()];
                for o in 0..outer {
                    for i in 0..inner {
                        for k in 0..dim_len {
                            let idx_k = (o * dim_len + k) * inner + i;
                            let prefix = if k == 0 {
                                1.0
                            } else {
                                out_vec[(o * dim_len + (k - 1)) * inner + i]
                            };
                            let mut suffix_prod = 1.0f64;
                            let mut sum_grad = 0.0f64;
                            for m in k..dim_len {
                                let idx_m = (o * dim_len + m) * inner + i;
                                if m > k {
                                    suffix_prod *= x_vec[idx_m];
                                }
                                sum_grad += go[idx_m] * prefix * suffix_prod;
                            }
                            gx[idx_k] = sum_grad;
                        }
                    }
                }
                let g_t = Tensor::from_slice_on(&gx, &shape, &dev)?.to_dtype(dt)?;
                Ok(vec![Some(g_t)])
            },
        ))
    }

    /// Computes the $n$-th forward discrete difference along `axis`:
    /// $\Delta x_i = x_{i+1} - x_i$ (`torch.diff`). Short dimensions read
    /// empty like torch (no error); `n = 0` returns the input unchanged.
    pub fn diff(&self, n: usize, axis: isize) -> Result<Tensor> {
        let ax = normalize_axis("diff", axis, self.ndim())?;
        let mut cur = self.clone();
        for _ in 0..n {
            let len = cur.shape()[ax];
            if len < 2 {
                cur = cur.narrow(ax as isize, 0, 0)?;
                continue;
            }
            let hi = cur.narrow(ax as isize, 1, len - 1)?;
            let lo = cur.narrow(ax as isize, 0, len - 1)?;
            cur = hi.sub(&lo)?;
        }
        Ok(cur)
    }

    /// Working dtype for gradient/trapezoid: float inputs keep their width,
    /// anything else computes in f32; any f64 input promotes to f64 (torch
    /// promotion; mixed f16/bf16 promotes to f32).
    fn grad_work_dtype(input: DType, others: &[DType]) -> DType {
        let mut kinds = vec![if input.is_floating_point() {
            input
        } else {
            DType::F32
        }];
        kinds.extend(others.iter().filter(|d| d.is_floating_point()).copied());
        if kinds.contains(&DType::F64) {
            DType::F64
        } else if kinds.contains(&DType::F32) {
            DType::F32
        } else if kinds.iter().all(|&k| k == kinds[0]) {
            kinds[0]
        } else {
            DType::F32
        }
    }

    /// Shared validation for gradient axes: non-bool input, edge_order 1/2,
    /// and at least edge_order+1 points (torch's rules and messages).
    fn check_gradient(&self, axis: isize, edge_order: isize) -> Result<(usize, usize)> {
        if self.dtype() == DType::Bool {
            return Err(Error::UnsupportedDType {
                op: "gradient",
                dtype: self.dtype(),
            });
        }
        if edge_order != 1 && edge_order != 2 {
            return Err(Error::Msg(format!(
                "torch.gradient only supports edge_order=1 and edge_order=2 (got {edge_order})"
            )));
        }
        let ax = normalize_axis("gradient", axis, self.ndim())?;
        let n = self.shape()[ax];
        if n < edge_order as usize + 1 {
            return Err(Error::Msg(format!(
                "torch.gradient expected each dimension size to be at least edge_order+1 (axis {ax} has size {n})"
            )));
        }
        Ok((ax, n))
    }

    /// First-order gradient along one axis with uniform spacing `h`
    /// (torch.gradient parity): central differences inside, one-sided
    /// (first- or second-order) edges. Differentiable (slicing composition).
    pub fn gradient_axis(&self, axis: isize, edge_order: isize, h: f64) -> Result<Tensor> {
        let (ax, n) = self.check_gradient(axis, edge_order)?;
        let a = ax as isize;
        let y = self.to_dtype(Self::grad_work_dtype(self.dtype(), &[]))?;
        let two_h = 2.0 * h;
        let (left, right) = if edge_order == 1 {
            let l = Tensor::sub(&y.narrow(a, 1, 1)?, &y.narrow(a, 0, 1)?)?.div_scalar(h)?;
            let r =
                Tensor::sub(&y.narrow(a, n - 1, 1)?, &y.narrow(a, n - 2, 1)?)?.div_scalar(h)?;
            (l, r)
        } else {
            let l = Tensor::sub(
                &Tensor::add(
                    &y.narrow(a, 0, 1)?.mul_scalar(-3.0)?,
                    &y.narrow(a, 1, 1)?.mul_scalar(4.0)?,
                )?,
                &y.narrow(a, 2, 1)?,
            )?
            .div_scalar(two_h)?;
            let r = Tensor::add(
                &Tensor::sub(
                    &y.narrow(a, n - 1, 1)?.mul_scalar(3.0)?,
                    &y.narrow(a, n - 2, 1)?.mul_scalar(4.0)?,
                )?,
                &y.narrow(a, n - 3, 1)?,
            )?
            .div_scalar(two_h)?;
            (l, r)
        };
        if n == 2 {
            debug_assert_eq!(edge_order, 1);
            return Tensor::cat(&[left, right], a);
        }
        let mid = Tensor::sub(&y.narrow(a, 2, n - 2)?, &y.narrow(a, 0, n - 2)?)?.div_scalar(two_h)?;
        Tensor::cat(&[left, mid, right], a)
    }

    /// First-order gradient along one axis at 1-D coordinates `coords`
    /// (torch.gradient parity): general second-order 3-point interior,
    /// one-sided edges (first-order for edge_order=1, general second-order
    /// for edge_order=2). Differentiable.
    pub fn gradient_axis_coords(
        &self,
        axis: isize,
        edge_order: isize,
        coords: &Tensor,
    ) -> Result<Tensor> {
        let (ax, n) = self.check_gradient(axis, edge_order)?;
        let a = ax as isize;
        if coords.ndim() != 1 {
            return Err(Error::Msg(format!(
                "torch.gradient expected each element of spacing to have one dimension (got rank {})",
                coords.ndim()
            )));
        }
        if coords.numel() != n {
            return Err(Error::Msg(format!(
                "gradient: coordinate spacing length {} does not match dimension size {n}",
                coords.numel()
            )));
        }
        if coords.device() != self.device() {
            return Err(Error::DeviceMismatch {
                op: "gradient",
                lhs: self.device().to_string(),
                rhs: coords.device().to_string(),
            });
        }
        let w = Self::grad_work_dtype(self.dtype(), &[coords.dtype()]);
        let y = self.to_dtype(w)?;
        let t = coords.to_dtype(w)?;
        // Forward gaps h[i] = t[i+1] - t[i] (1-D, length n-1).
        let h = Tensor::sub(&t.narrow(0, 1, n - 1)?, &t.narrow(0, 0, n - 1)?)?;
        // Reshape a 1-D length-L vector into an axis-marked broadcast frame.
        let mark = |v: &Tensor, len: usize| -> Result<Tensor> {
            let mut s = vec![1usize; self.ndim()];
            s[ax] = len;
            v.reshape(&s)
        };
        let (left, right) = if edge_order == 1 {
            let l = Tensor::div(
                &Tensor::sub(&y.narrow(a, 1, 1)?, &y.narrow(a, 0, 1)?)?,
                &mark(&h.narrow(0, 0, 1)?, 1)?,
            )?;
            let r = Tensor::div(
                &Tensor::sub(&y.narrow(a, n - 1, 1)?, &y.narrow(a, n - 2, 1)?)?,
                &mark(&h.narrow(0, n - 2, 1)?, 1)?,
            )?;
            (l, r)
        } else {
            let tv = t.to_vec::<f64>()?;
            let (h0, h1) = (tv[1] - tv[0], tv[2] - tv[1]);
            let dd = h0 * h1 * (h0 + h1);
            let l = Tensor::add(
                &Tensor::add(
                    &y.narrow(a, 0, 1)?.mul_scalar(-(2.0 * h0 * h1 + h1 * h1))?,
                    &y.narrow(a, 1, 1)?.mul_scalar((h0 + h1) * (h0 + h1))?,
                )?,
                &y.narrow(a, 2, 1)?.mul_scalar(-(h0 * h0))?,
            )?
            .div_scalar(dd)?;
            let (g0, g1) = (tv[n - 2] - tv[n - 3], tv[n - 1] - tv[n - 2]);
            let ee = g0 * g1 * (g0 + g1);
            let r = Tensor::add(
                &Tensor::add(
                    &y.narrow(a, n - 3, 1)?.mul_scalar(g1 * g1)?,
                    &y.narrow(a, n - 2, 1)?.mul_scalar(-((g0 + g1) * (g0 + g1)))?,
                )?,
                &y.narrow(a, n - 1, 1)?.mul_scalar(2.0 * g0 * g1 + g0 * g0)?,
            )?
            .div_scalar(ee)?;
            (l, r)
        };
        if n == 2 {
            debug_assert_eq!(edge_order, 1);
            return Tensor::cat(&[left, right], a);
        }
        // General central scheme: [h0^2 f+ + (h1^2-h0^2) f - h1^2 f-] / h0 h1 (h0+h1).
        let h0v = mark(&h.narrow(0, 0, n - 2)?, n - 2)?;
        let h1v = mark(&h.narrow(0, 1, n - 2)?, n - 2)?;
        let h0sq = Tensor::mul(&h0v, &h0v)?;
        let h1sq = Tensor::mul(&h1v, &h1v)?;
        let num = Tensor::sub(
            &Tensor::add(
                &Tensor::mul(&h0sq, &y.narrow(a, 2, n - 2)?)?,
                &Tensor::mul(&Tensor::sub(&h1sq, &h0sq)?, &y.narrow(a, 1, n - 2)?)?,
            )?,
            &Tensor::mul(&h1sq, &y.narrow(a, 0, n - 2)?)?,
        )?;
        let den = Tensor::mul(&Tensor::mul(&h0v, &h1v)?, &Tensor::add(&h0v, &h1v)?)?;
        let mid = Tensor::div(&num, &den)?;
        Tensor::cat(&[left, mid, right], a)
    }

    /// Trapezoidal integral along one axis with uniform spacing `dx`
    /// (torch.trapezoid parity). Fewer than 2 points integrate to zero.
    /// Differentiable (endpoints weigh half).
    pub fn trapezoid_axis(&self, axis: isize, dx: f64) -> Result<Tensor> {
        if self.dtype() == DType::Bool {
            return Err(Error::UnsupportedDType {
                op: "trapezoid",
                dtype: self.dtype(),
            });
        }
        let ax = normalize_axis("trapezoid", axis, self.ndim())?;
        let n = self.shape()[ax];
        let w = Self::grad_work_dtype(self.dtype(), &[]);
        if n <= 1 {
            let mut s = self.shape().to_vec();
            s.remove(ax);
            return Tensor::zeros(&s, w, &self.device());
        }
        let a = ax as isize;
        let y = self.to_dtype(w)?;
        let avg = Tensor::add(&y.narrow(a, 0, n - 1)?, &y.narrow(a, 1, n - 1)?)?
            .mul_scalar(0.5 * dx)?;
        avg.sum_axes(&[axis], false)
    }

    /// Trapezoidal integral along one axis at sample points `x`
    /// (torch.trapezoid parity): `x` is 1-D with one value per sample, or
    /// shaped like the input. Differentiable in both operands.
    pub fn trapezoid_axis_x(&self, axis: isize, x: &Tensor) -> Result<Tensor> {
        if self.dtype() == DType::Bool {
            return Err(Error::UnsupportedDType {
                op: "trapezoid",
                dtype: self.dtype(),
            });
        }
        let ax = normalize_axis("trapezoid", axis, self.ndim())?;
        let n = self.shape()[ax];
        if !(x.ndim() == 1 && x.numel() == n || x.shape() == self.shape()) {
            return Err(Error::Msg(
                "trapezoid: There must be one `x` value for each sample point".to_string(),
            ));
        }
        if x.device() != self.device() {
            return Err(Error::DeviceMismatch {
                op: "trapezoid",
                lhs: self.device().to_string(),
                rhs: x.device().to_string(),
            });
        }
        let w = Self::grad_work_dtype(self.dtype(), &[x.dtype()]);
        if n <= 1 {
            let mut s = self.shape().to_vec();
            s.remove(ax);
            return Tensor::zeros(&s, w, &self.device());
        }
        let a = ax as isize;
        let y = self.to_dtype(w)?;
        let xx = x.to_dtype(w)?;
        let d = Tensor::sub(&xx.narrow(a, 1, n - 1)?, &xx.narrow(a, 0, n - 1)?)?;
        let dm = if x.ndim() == 1 {
            let mut s = vec![1usize; self.ndim()];
            s[ax] = n - 1;
            d.reshape(&s)?
        } else {
            d
        };
        let avg = Tensor::add(&y.narrow(a, 0, n - 1)?, &y.narrow(a, 1, n - 1)?)?.mul_scalar(0.5)?;
        Tensor::mul(&avg, &dm)?.sum_axes(&[axis], false)
    }

    /// Linear interpolation between `self` and `end`: $\operatorname{lerp}(x, y, w) = x + w (y - x)$.
    pub fn lerp(&self, end: &Tensor, weight: f64) -> Result<Tensor> {
        self.add(&end.sub(self)?.mul_scalar(weight)?)
    }

    /// Differentiable elementwise 4-quadrant inverse tangent $\operatorname{atan2}(y, x)$ (`self` = $y$, `other` = $x$)
    /// with exact analytical gradients $\frac{\partial}{\partial y} = \frac{x}{x^2 + y^2}$, $\frac{\partial}{\partial x} = \frac{-y}{x^2 + y^2}$.
    pub fn atan2(&self, other: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_shapes("atan2", self.shape(), other.shape())?;
        let y_b = self.broadcast_to(&out_shape)?;
        let x_b = other.broadcast_to(&out_shape)?;
        let y_vec = y_b.to_vec::<f64>()?;
        let x_vec = x_b.to_vec::<f64>()?;
        let out_vec: Vec<f64> = y_vec
            .iter()
            .zip(&x_vec)
            .map(|(&y, &x)| y.atan2(x))
            .collect();

        let out_layout = Layout::contiguous(out_shape);
        let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, self.dtype())?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &self.device())?;

        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        let (ra, rb) = (self.requires_grad(), other.requires_grad());
        Ok(Tensor::from_op(
            storage,
            out_layout,
            "atan2",
            vec![self.clone(), other.clone()],
            move |grad_out| {
                let denom = x_b.square()?.add(&y_b.square()?)?.add_scalar(1e-12)?;
                let dy = grad_out.mul(&x_b)?.div(&denom)?.sum_to_shape(&sa)?;
                let dx = grad_out.mul(&y_b)?.neg()?.div(&denom)?.sum_to_shape(&sb)?;
                Ok(vec![
                    if ra { Some(dy) } else { None },
                    if rb { Some(dx) } else { None },
                ])
            },
        ))
    }

    /// Differentiable elementwise Euclidean hypotenuse $\sqrt{x^2 + y^2}$ (`torch.hypot`).
    pub fn hypot(&self, other: &Tensor) -> Result<Tensor> {
        // scale by the larger magnitude so large finite inputs do not overflow in the squares
        let (ax, ay) = (self.abs()?, other.abs()?);
        let m = ax.maximum(&ay)?;
        let one = m.full_like(1.0)?;
        let safe_m = Tensor::where_cond(&m.eq_tensor(&m.zeros_like()?)?, &one, &m)?;
        let (rx, ry) = (ax.div(&safe_m)?, ay.div(&safe_m)?);
        let h = rx.square()?.add(&ry.square()?)?.sqrt()?.mul(&safe_m)?;
        // IEEE 754 / C hypot: an infinite argument gives +inf even when the other one is NaN
        let any_inf = self.isinf()?.logical_or(&other.isinf()?)?;
        Tensor::where_cond(&any_inf, &h.full_like(f64::INFINITY)?, &h)
    }

    /// Replaces `NaN`, `+Inf`, and `-Inf` entries with `nan_val`, `posinf_val`, and `neginf_val` (`torch.nan_to_num`).
    pub fn nan_to_num(&self, nan_val: f64, posinf_val: f64, neginf_val: f64) -> Result<Tensor> {
        let vals = self.to_vec::<f64>()?;
        let cleaned: Vec<f64> = vals
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    nan_val
                } else if v.is_infinite() {
                    if v.is_sign_positive() {
                        posinf_val
                    } else {
                        neginf_val
                    }
                } else {
                    v
                }
            })
            .collect();
        let out = Tensor::from_slice_on(&cleaned, self.shape(), &self.device())?
            .to_dtype(self.dtype())?;
        if !self.requires_grad() {
            return Ok(out);
        }
        // Like torch, gradients flow through finite entries and are zeroed
        // where the input was NaN or infinite.
        let mask_vec: Vec<f64> = vals
            .iter()
            .map(|v| if v.is_finite() { 1.0 } else { 0.0 })
            .collect();
        let mask = Tensor::from_slice_on(&mask_vec, self.shape(), &self.device())?
            .to_dtype(self.dtype())?;
        Ok(Tensor::from_op(
            out.storage().clone(),
            out.layout().clone(),
            "nan_to_num",
            vec![self.clone()],
            move |g| Ok(vec![Some(g.mul(&mask)?)]),
        ))
    }

    /// Counts occurrences of each non-negative integer value in a 1D tensor
    /// (`torch.bincount`), optionally weighted. Unweighted output is `I64`;
    /// weighted output follows the weights dtype when floating, else `F64`
    /// (torch parity, probed). Non-differentiable like torch.
    pub fn bincount(&self, weights: Option<&Tensor>, minlength: usize) -> Result<Tensor> {
        if self.ndim() != 1 || !self.dtype().is_int() {
            return Err(Error::Msg(format!(
                "bincount expects a 1D integer tensor, got {:?} {:?}",
                self.shape(),
                self.dtype()
            )));
        }
        if let Some(w) = weights {
            if w.ndim() != 1 || w.numel() != self.numel() {
                return Err(Error::Msg(
                    "bincount weights should be 1-d and have the same length as input".into(),
                ));
            }
        }
        let vals = self.to_vec::<i64>()?;
        // Reject negatives BEFORE sizing the output: a huge max next to a
        // negative must not allocate gigabytes before failing.
        let mut max_v: i64 = -1;
        for &v in &vals {
            if v < 0 {
                return Err(Error::Msg("bincount requires non-negative integers".into()));
            }
            max_v = max_v.max(v);
        }
        let len = if max_v >= 0 {
            (max_v as usize)
                .checked_add(1)
                .ok_or_else(|| Error::Msg("bincount: value range overflows".into()))?
                .max(minlength)
        } else {
            minlength
        };
        crate::limits::check_elements("bincount", &[len])?;
        if let Some(w) = weights {
            let wvec = w.to_vec::<f64>()?;
            let mut counts = vec![0.0f64; len];
            for (v, w) in vals.iter().zip(wvec.iter()) {
                counts[*v as usize] += *w;
            }
            let out = Tensor::from_slice_on(&counts, &[len], &self.device())?;
            let dt = if w.dtype().is_floating_point() {
                w.dtype()
            } else {
                DType::F64
            };
            return out.to_dtype(dt);
        }
        let mut counts = vec![0i64; len];
        for v in vals {
            counts[v as usize] += 1;
        }
        Tensor::from_slice_on(&counts, &[len], &self.device())
    }

    /// Computes a 1D histogram with `bins` equal-width bins over `[min, max]` (`torch.histc`).
    /// `min == max` falls back to the data range (all-equal data expands by
    /// +/-0.5, empty input reads zeros, any NaN raises like torch); with an
    /// explicit range NaN values are skipped. Integer inputs are accepted
    /// (torch rejects them) and read back in-kind.
    pub fn histc(&self, bins: usize, min: f64, max: f64) -> Result<Tensor> {
        if bins == 0 {
            return Err(Error::Msg("histc requires bins >= 1".into()));
        }
        if max < min {
            return Err(Error::Msg("histc requires max >= min".into()));
        }
        crate::limits::check_elements("histc", &[bins])?;
        let vals = self.to_vec::<f64>()?;
        let (lo, hi) = if max > min {
            (min, max)
        } else {
            // min == max: data range like torch. Any NaN poisons the range
            // (torch raises); empty input reads zeros like torch.
            if vals.is_empty() {
                let h = vec![0.0f64; bins];
                return Tensor::from_slice_on(&h, &[bins], &self.device())?
                    .to_dtype(self.dtype());
            }
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &v in &vals {
                if v.is_nan() {
                    return Err(Error::Msg(
                        "histc: data range is not finite (NaN with min == max)".into(),
                    ));
                }
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if lo == hi {
                (lo - 0.5, hi + 0.5)
            } else {
                (lo, hi)
            }
        };
        let mut h = vec![0.0f64; bins];
        let width = (hi - lo) / (bins as f64);
        for v in vals {
            if v >= lo && v <= hi {
                let idx = (((v - lo) / width).floor() as usize).min(bins - 1);
                h[idx] += 1.0;
            }
        }
        Tensor::from_slice_on(&h, &[bins], &self.device())?.to_dtype(self.dtype())
    }

    /// Finds insertion indices of `values` in sorted `sorted_seq` (`torch.searchsorted`).
    ///
    /// The sequence's last axis holds the sorted rows. A 1D sequence accepts
    /// any `values` shape; an N-D sequence needs `values` with the same ndim
    /// and exactly matching leading dims (no broadcasting, like torch).
    /// NaN queries read the row length (torch parity: NaN sorts last). The
    /// sequence must be sorted (unchecked, like torch); the `sorter`
    /// variant is not supported.
    pub fn searchsorted(sorted_seq: &Tensor, values: &Tensor, right: bool) -> Result<Tensor> {
        if sorted_seq.ndim() == 0 {
            return Err(Error::Msg("searchsorted: sorted_seq must be at least 1D".into()));
        }
        let nd = sorted_seq.ndim();
        let n = sorted_seq.shape()[nd - 1];
        let seq_lead = &sorted_seq.shape()[..nd - 1];
        if nd > 1 && (values.ndim() != nd || values.shape()[..nd - 1] != seq_lead[..]) {
            return Err(Error::Msg(format!(
                "searchsorted: N-D sorted_seq needs values with the same ndim and matching leading dims, got seq {:?} values {:?}",
                sorted_seq.shape(),
                values.shape()
            )));
        }
        let out_shape = values.shape().to_vec();
        let out_numel: usize = out_shape.iter().product();
        let row_strides = |shape: &[usize]| -> Vec<usize> {
            let mut s = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                s[i] = s[i + 1] * shape[i + 1].max(1);
            }
            s
        };
        let ostr = row_strides(&out_shape);
        let lstr = row_strides(seq_lead);
        // Per-output (sequence-row, value) offsets over the logical vectors.
        // The output shape always equals the values shape, so values offsets
        // are the identity; rows come from the leading multi-index.
        let mut pairs = Vec::with_capacity(out_numel);
        for f in 0..out_numel {
            let mut rem = f;
            let mut srow = 0usize;
            for (i, &st) in ostr.iter().enumerate() {
                let k = rem / st;
                rem %= st;
                if i < seq_lead.len() {
                    srow += k * lstr[i];
                }
            }
            pairs.push((srow, f));
        }
        // I64 compares natively past 2^53 (f64 keys round and misplace);
        // every other dtype widens to f64 exactly or is float already.
        let indices: Vec<i64> = if sorted_seq.dtype() == DType::I64 && values.dtype() == DType::I64 {
            let seq = sorted_seq.to_vec::<i64>()?;
            let q = values.to_vec::<i64>()?;
            pairs
                .iter()
                .map(|&(r, v)| {
                    let row = &seq[r * n..(r + 1) * n];
                    let val = q[v];
                    let pos = if right {
                        row.partition_point(|&x| x <= val)
                    } else {
                        row.partition_point(|&x| x < val)
                    };
                    pos as i64
                })
                .collect()
        } else {
            let seq = sorted_seq.to_vec::<f64>()?;
            let q = values.to_vec::<f64>()?;
            pairs
                .iter()
                .map(|&(r, v)| {
                    let val = q[v];
                    if val.is_nan() {
                        return n as i64;
                    }
                    let row = &seq[r * n..(r + 1) * n];
                    let pos = if right {
                        row.partition_point(|&x| x <= val)
                    } else {
                        row.partition_point(|&x| x < val)
                    };
                    pos as i64
                })
                .collect()
        };
        Tensor::from_slice_on(&indices, &out_shape, &values.device())
    }

    /// Generates $N$-D coordinate grids from 1D coordinate tensors (`torch.meshgrid`).
    /// When `indexing_ij = true`, uses matrix (`"ij"`) indexing; when `false`, uses Cartesian (`"xy"`) indexing.
    pub fn meshgrid(tensors: &[&Tensor], indexing_ij: bool) -> Result<Vec<Tensor>> {
        if tensors.is_empty() {
            return Ok(Vec::new());
        }
        for t in tensors {
            if t.ndim() != 1 {
                return Err(Error::Msg("meshgrid expects 1D coordinate tensors".into()));
            }
        }
        let ndim = tensors.len();
        let mut grid_shape: Vec<usize> = tensors.iter().map(|t| t.shape()[0]).collect();
        if !indexing_ij && ndim >= 2 {
            grid_shape.swap(0, 1);
        }
        let mut grids = Vec::with_capacity(ndim);
        for (i, &t) in tensors.iter().enumerate() {
            let mut reshape_dims = vec![1usize; ndim];
            let target_axis = if !indexing_ij && ndim >= 2 {
                if i == 0 {
                    1
                } else if i == 1 {
                    0
                } else {
                    i
                }
            } else {
                i
            };
            reshape_dims[target_axis] = t.shape()[0];
            grids.push(t.reshape(&reshape_dims)?.broadcast_to(&grid_shape)?);
        }
        Ok(grids)
    }

    // =========================================================================
    // Core-1: Differentiable Trigonometric & Inverse Trigonometric Suite
    // =========================================================================

    fn run_unary_f64(
        &self,
        op_name: &'static str,
        fwd: impl Fn(f64) -> f64,
        backward: impl Fn(&Tensor, &Tensor, &Tensor) -> Result<Tensor> + Send + Sync + 'static,
    ) -> Result<Tensor> {
        let out_dt = if matches!(self.dtype(), DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            self.dtype()
        } else {
            DType::F32
        };
        let vals = self.to_vec::<f64>()?;
        let out_vec: Vec<f64> = vals.into_iter().map(fwd).collect();
        let out_layout = Layout::contiguous(self.shape().to_vec());
        let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, out_dt)?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &self.device())?;
        let inp = self.to_dtype(out_dt)?;
        let out_for_bwd = Tensor::from_storage(storage.clone(), out_layout.clone());
        Ok(Tensor::from_op(
            storage,
            out_layout,
            op_name,
            vec![self.clone()],
            move |g| Ok(vec![Some(backward(g, &inp, &out_for_bwd)?)]),
        ))
    }

    /// Differentiable tangent $\tan(x)$ with $\frac{d}{dx}\tan(x) = 1 + \tan^2(x)$.
    pub fn tan(&self) -> Result<Tensor> {
        self.run_unary_f64("tan", f64::tan, |g, _, y| {
            let sec2 = y.square()?.add_scalar(1.0)?;
            g.mul(&sec2)
        })
    }

    /// Differentiable inverse sine $\arcsin(x)$ with $\frac{d}{dx}\arcsin(x) = \frac{1}{\sqrt{1 - x^2}}$.
    pub fn asin(&self) -> Result<Tensor> {
        self.run_unary_f64("asin", f64::asin, |g, x, _| {
            let denom = x.square()?.neg()?.add_scalar(1.0)?.sqrt()?;
            g.div(&denom)
        })
    }

    /// Differentiable inverse cosine $\arccos(x)$ with $\frac{d}{dx}\arccos(x) = \frac{-1}{\sqrt{1 - x^2}}$.
    pub fn acos(&self) -> Result<Tensor> {
        self.run_unary_f64("acos", f64::acos, |g, x, _| {
            let denom = x.square()?.neg()?.add_scalar(1.0)?.sqrt()?;
            g.div(&denom)?.neg()
        })
    }

    /// Differentiable inverse tangent $\arctan(x)$ with $\frac{d}{dx}\arctan(x) = \frac{1}{1 + x^2}$.
    pub fn atan(&self) -> Result<Tensor> {
        self.run_unary_f64("atan", f64::atan, |g, x, _| {
            let denom = x.square()?.add_scalar(1.0)?;
            g.div(&denom)
        })
    }

    /// Differentiable normalized sinc function $\operatorname{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$
    /// with $\operatorname{sinc}(0) = 1$ and $\operatorname{sinc}'(0) = 0$.
    pub fn sinc(&self) -> Result<Tensor> {
        let pi = std::f64::consts::PI;
        self.run_unary_f64(
            "sinc",
            move |v| {
                let px = pi * v;
                if px.abs() < 1e-12 {
                    1.0 - (px * px) / 6.0
                } else {
                    px.sin() / px
                }
            },
            move |g, x, _| {
                let xv = x.to_vec::<f64>()?;
                let deriv_vec: Vec<f64> = xv
                    .into_iter()
                    .map(|v| {
                        let px = pi * v;
                        if px.abs() < 1e-8 {
                            -pi * px / 3.0
                        } else {
                            (px * px.cos() - px.sin()) / (pi * v * v)
                        }
                    })
                    .collect();
                let d_t = Tensor::from_slice_on(&deriv_vec, x.shape(), &x.device())?
                    .to_dtype(g.dtype())?;
                g.mul(&d_t)
            },
        )
    }

    // =========================================================================
    // Core-2: Differentiable Hyperbolic & Inverse Hyperbolic Suite
    // =========================================================================

    /// Differentiable hyperbolic sine $\sinh(x)$ with $\frac{d}{dx}\sinh(x) = \cosh(x)$.
    pub fn sinh(&self) -> Result<Tensor> {
        self.run_unary_f64("sinh", f64::sinh, |g, x, _| g.mul(&x.cosh()?))
    }

    /// Differentiable hyperbolic cosine $\cosh(x)$ with $\frac{d}{dx}\cosh(x) = \sinh(x)$.
    pub fn cosh(&self) -> Result<Tensor> {
        self.run_unary_f64("cosh", f64::cosh, |g, x, _| g.mul(&x.sinh()?))
    }

    /// Differentiable inverse hyperbolic sine $\operatorname{asinh}(x)$ with $\frac{d}{dx}\operatorname{asinh}(x) = \frac{1}{\sqrt{x^2 + 1}}$.
    pub fn asinh(&self) -> Result<Tensor> {
        self.run_unary_f64("asinh", f64::asinh, |g, x, _| {
            let denom = x.square()?.add_scalar(1.0)?.sqrt()?;
            g.div(&denom)
        })
    }

    /// Differentiable inverse hyperbolic cosine $\operatorname{acosh}(x)$ with $\frac{d}{dx}\operatorname{acosh}(x) = \frac{1}{\sqrt{x^2 - 1}}$.
    pub fn acosh(&self) -> Result<Tensor> {
        self.run_unary_f64("acosh", f64::acosh, |g, x, _| {
            let denom = x.square()?.sub_scalar(1.0)?.sqrt()?;
            g.div(&denom)
        })
    }

    /// Differentiable inverse hyperbolic tangent $\operatorname{atanh}(x)$ with $\frac{d}{dx}\operatorname{atanh}(x) = \frac{1}{1 - x^2}$.
    pub fn atanh(&self) -> Result<Tensor> {
        self.run_unary_f64("atanh", f64::atanh, |g, x, _| {
            let denom = x.square()?.neg()?.add_scalar(1.0)?;
            g.div(&denom)
        })
    }

    // =========================================================================
    // Core-3: Numerically Stable Logarithmic & Exponential Primitives
    // =========================================================================

    /// Differentiable $e^x - 1$ accurate for small $|x|$.
    pub fn expm1(&self) -> Result<Tensor> {
        self.run_unary_f64("expm1", f64::exp_m1, |g, x, _| g.mul(&x.exp()?))
    }

    /// Differentiable $\ln(1 + x)$ accurate for small $|x|$.
    pub fn log1p(&self) -> Result<Tensor> {
        self.run_unary_f64("log1p", f64::ln_1p, |g, x, _| {
            g.div(&x.add_scalar(1.0)?)
        })
    }

    /// Differentiable base-2 logarithm $\log_2(x)$.
    pub fn log2(&self) -> Result<Tensor> {
        let ln2 = std::f64::consts::LN_2;
        self.run_unary_f64("log2", f64::log2, move |g, x, _| {
            g.div(&x.mul_scalar(ln2)?)
        })
    }

    /// Differentiable base-10 logarithm $\log_{10}(x)$.
    pub fn log10(&self) -> Result<Tensor> {
        let ln10 = std::f64::consts::LN_10;
        self.run_unary_f64("log10", f64::log10, move |g, x, _| {
            g.div(&x.mul_scalar(ln10)?)
        })
    }

    /// Differentiable base-2 exponential $2^x$.
    pub fn exp2(&self) -> Result<Tensor> {
        let ln2 = std::f64::consts::LN_2;
        self.run_unary_f64("exp2", f64::exp2, move |g, _, y| {
            g.mul(&y.mul_scalar(ln2)?)
        })
    }

    /// Differentiable $x \ln(y)$ with exact $0 \cdot \ln(0) = 0$ convention (`torch.special.xlogy`).
    pub fn xlogy(&self, other: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_shapes("xlogy", self.shape(), other.shape())?;
        let x_b = self.broadcast_to(&out_shape)?;
        let y_b = other.broadcast_to(&out_shape)?;
        let xv = x_b.to_vec::<f64>()?;
        let yv = y_b.to_vec::<f64>()?;
        let out_vec: Vec<f64> = xv
            .iter()
            .zip(&yv)
            .map(|(&x, &y)| if x == 0.0 && !y.is_nan() { 0.0 } else { x * y.ln() })
            .collect();
        let out_dt = if matches!(self.dtype(), DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            self.dtype()
        } else {
            DType::F32
        };
        let out_layout = Layout::contiguous(out_shape.clone());
        let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, out_dt)?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &self.device())?;

        let (sa, sb) = (self.shape().to_vec(), other.shape().to_vec());
        let (ra, rb) = (self.requires_grad(), other.requires_grad());
        let dev = self.device();
        Ok(Tensor::from_op(
            storage,
            out_layout,
            "xlogy",
            vec![self.clone(), other.clone()],
            move |g| {
                let gv = g.to_vec::<f64>()?;
                let dx_vec: Vec<f64> = gv
                    .iter()
                    .zip( xv.iter().zip(&yv) )
                    .map(|(&gi, (&x, &y))| if x == 0.0 && y == 0.0 { 0.0 } else { gi * y.ln() })
                    .collect();
                let dy_vec: Vec<f64> = gv
                    .iter()
                    .zip( xv.iter().zip(&yv) )
                    .map(|(&gi, (&x, &y))| if x == 0.0 { 0.0 } else { gi * x / y })
                    .collect();
                let dx = Tensor::from_slice_on(&dx_vec, &out_shape, &dev)?
                    .to_dtype(out_dt)?
                    .sum_to_shape(&sa)?;
                let dy = Tensor::from_slice_on(&dy_vec, &out_shape, &dev)?
                    .to_dtype(out_dt)?
                    .sum_to_shape(&sb)?;
                Ok(vec![
                    if ra { Some(dx) } else { None },
                    if rb { Some(dy) } else { None },
                ])
            },
        ))
    }

    /// Differentiable logit function $\ln\left(\frac{p}{1 - p}\right)$ with optional clamping `eps`.
    pub fn logit(&self, eps: Option<f64>) -> Result<Tensor> {
        self.run_unary_f64(
            "logit",
            move |p| {
                let pc = match eps {
                    Some(e) => p.clamp(e, 1.0 - e),
                    None => p,
                };
                (pc / (1.0 - pc)).ln()
            },
            move |g, x, _| {
                let xv = x.to_vec::<f64>()?;
                let deriv: Vec<f64> = xv
                    .into_iter()
                    .map(|p| match eps {
                        Some(e) if p < e || p > 1.0 - e => 0.0,
                        _ => 1.0 / (p * (1.0 - p)),
                    })
                    .collect();
                let d_t = Tensor::from_slice_on(&deriv, x.shape(), &x.device())?
                    .to_dtype(g.dtype())?;
                g.mul(&d_t)
            },
        )
    }

    // =========================================================================
    // Core-4: Differentiable Cumulative Extremum & Log-Space Scans
    // =========================================================================

    /// Cumulative maximum along `axis`, returning `(values, argmax_indices_i64)` (`torch.cummax`).
    /// Differentiable with respect to `values`: upstream gradients route via `scatter_add` to winning indices.
    pub fn cummax(&self, axis: isize) -> Result<(Tensor, Tensor)> {
        self.cumext_impl(axis, true)
    }

    /// Cumulative minimum along `axis`, returning `(values, argmin_indices_i64)` (`torch.cummin`).
    /// Differentiable with respect to `values`: upstream gradients route via `scatter_add` to winning indices.
    pub fn cummin(&self, axis: isize) -> Result<(Tensor, Tensor)> {
        self.cumext_impl(axis, false)
    }

    fn cumext_impl(&self, axis: isize, is_max: bool) -> Result<(Tensor, Tensor)> {
        let ax = normalize_axis(if is_max { "cummax" } else { "cummin" }, axis, self.ndim())?;
        let dim = self.shape()[ax];
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();
        let mut vals = self.to_vec::<f64>()?;
        let mut idxs = vec![0i64; vals.len()];

        for o in 0..outer {
            for i in 0..inner {
                let first = (o * dim) * inner + i;
                let mut best_v = vals[first];
                let mut best_k = 0i64;
                idxs[first] = 0;
                for k in 1..dim {
                    let pos = (o * dim + k) * inner + i;
                    let v = vals[pos];
                    let replace = if is_max {
                        v >= best_v || v.is_nan()
                    } else {
                        v <= best_v || v.is_nan()
                    };
                    if replace && !best_v.is_nan() {
                        best_v = v;
                        best_k = k as i64;
                    }
                    vals[pos] = best_v;
                    idxs[pos] = best_k;
                }
            }
        }
        let shape = self.shape().to_vec();
        let layout = Layout::contiguous(shape.clone());
        let cpu_v = CpuStorage::F64(vals).to_dtype(&layout, self.dtype())?;
        let storage_v = Storage::new_cpu(cpu_v).to_device(&layout, &self.device())?;
        let idx_tensor = Tensor::from_slice_on(&idxs, &shape, &self.device())?;
        let idx_for_bwd = idx_tensor.clone();
        let v_tensor = Tensor::from_op(
            storage_v,
            layout,
            if is_max { "cummax" } else { "cummin" },
            vec![self.clone()],
            move |g| {
                let z = g.zeros_like()?;
                let scattered = z.scatter_add(ax as isize, &idx_for_bwd, g)?;
                Ok(vec![Some(scattered)])
            },
        );
        Ok((v_tensor, idx_tensor))
    }

    /// Numerically stable cumulative log-sum-exp along `axis`: $y_k = \ln \sum_{j=0}^{k} e^{x_j}$ (`torch.logcumsumexp`).
    pub fn logcumsumexp(&self, axis: isize) -> Result<Tensor> {
        let ax = normalize_axis("logcumsumexp", axis, self.ndim())?;
        let dim = self.shape()[ax];
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();
        let mut data = self.to_vec::<f64>()?;

        for o in 0..outer {
            for i in 0..inner {
                let mut acc = f64::NEG_INFINITY;
                for k in 0..dim {
                    let pos = (o * dim + k) * inner + i;
                    let x = data[pos];
                    if acc == f64::NEG_INFINITY {
                        acc = x;
                    } else {
                        let m = acc.max(x);
                        acc = m + ((acc - m).exp() + (x - m).exp()).ln();
                    }
                    data[pos] = acc;
                }
            }
        }
        let shape = self.shape().to_vec();
        let layout = Layout::contiguous(shape.clone());
        let out_dt = if matches!(self.dtype(), DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            self.dtype()
        } else {
            DType::F32
        };
        let cpu_s = CpuStorage::F64(data).to_dtype(&layout, out_dt)?;
        let storage = Storage::new_cpu(cpu_s).to_device(&layout, &self.device())?;
        let out_for_bwd = Tensor::from_storage(storage.clone(), layout.clone());
        let inp = self.to_dtype(out_dt)?;
        Ok(Tensor::from_op(
            storage,
            layout,
            "logcumsumexp",
            vec![self.clone()],
            move |g| {
                // d y_k / d x_j = exp(x_j - y_k) for j <= k
                // grad_x_j = exp(x_j) * sum_{k >= j} g_k * exp(-y_k)
                let weighted = g.mul(&out_for_bwd.neg()?.exp()?)?;
                let rev_sum = weighted
                    .flip(&[ax as isize])?
                    .cumsum(ax as isize)?
                    .flip(&[ax as isize])?;
                Ok(vec![Some(inp.exp()?.mul(&rev_sum)?)])
            },
        ))
    }

    // =========================================================================
    // Core-5: Multiplicative, Quantifier & Non-Zero Reductions
    // =========================================================================

    /// Differentiable product of all elements (`torch.prod`), using a zero-safe prefix-suffix backward pass.
    pub fn prod(&self) -> Result<Tensor> {
        let flat = self.reshape(&[self.numel()])?;
        flat.prod_axis(0, false)
    }

    /// Differentiable product along `axis` with zero-safe prefix-suffix product backward (`torch.prod`).
    pub fn prod_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        let ax = normalize_axis("prod_axis", axis, self.ndim())?;
        let dim = self.shape()[ax];
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();
        let x_vec = self.to_vec::<f64>()?;
        let mut out_vec = vec![1.0f64; outer * inner];

        for o in 0..outer {
            for k in 0..dim {
                let base = (o * dim + k) * inner;
                for i in 0..inner {
                    out_vec[o * inner + i] *= x_vec[base + i];
                }
            }
        }

        let mut out_shape = Vec::new();
        for (d_idx, &s) in self.shape().iter().enumerate() {
            if d_idx == ax {
                if keepdim {
                    out_shape.push(1);
                }
            } else {
                out_shape.push(s);
            }
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        let out_layout = Layout::contiguous(out_shape.clone());
        let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, self.dtype())?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &self.device())?;

        let in_shape = self.shape().to_vec();
        let dt = self.dtype();
        let dev = self.device();
        Ok(Tensor::from_op(
            storage,
            out_layout,
            "prod_axis",
            vec![self.clone()],
            move |g| {
                let g_vec = g.to_vec::<f64>()?;
                let mut gx = vec![0.0f64; outer * dim * inner];
                for o in 0..outer {
                    for i in 0..inner {
                        let go = g_vec[o * inner + i];
                        let mut prefix = vec![1.0f64; dim];
                        let mut suffix = vec![1.0f64; dim];
                        for k in 1..dim {
                            prefix[k] = prefix[k - 1] * x_vec[(o * dim + k - 1) * inner + i];
                        }
                        for k in (0..dim.saturating_sub(1)).rev() {
                            suffix[k] = suffix[k + 1] * x_vec[(o * dim + k + 1) * inner + i];
                        }
                        for k in 0..dim {
                            gx[(o * dim + k) * inner + i] = go * prefix[k] * suffix[k];
                        }
                    }
                }
                let gx_t = Tensor::from_slice_on(&gx, &in_shape, &dev)?.to_dtype(dt)?;
                Ok(vec![Some(gx_t)])
            },
        ))
    }

    /// Returns `true` if any element in the tensor is truthy (`!= 0`).
    pub fn any(&self) -> Result<Tensor> {
        let vals = self.to_vec::<f64>()?;
        let res = vals.iter().any(|&v| v != 0.0);
        Tensor::from_slice_on(&[res], &[1], &self.device())
    }

    /// Returns `true` along `axis` where any element is truthy (`!= 0`).
    pub fn any_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.quantifier_axis(axis, keepdim, true)
    }

    /// Returns `true` if all elements in the tensor are truthy (`!= 0`).
    pub fn all(&self) -> Result<Tensor> {
        let vals = self.to_vec::<f64>()?;
        let res = vals.iter().all(|&v| v != 0.0);
        Tensor::from_slice_on(&[res], &[1], &self.device())
    }

    /// Returns `true` along `axis` where all elements are truthy (`!= 0`).
    pub fn all_axis(&self, axis: isize, keepdim: bool) -> Result<Tensor> {
        self.quantifier_axis(axis, keepdim, false)
    }

    fn quantifier_axis(&self, axis: isize, keepdim: bool, is_any: bool) -> Result<Tensor> {
        let ax = normalize_axis(if is_any { "any_axis" } else { "all_axis" }, axis, self.ndim())?;
        let dim = self.shape()[ax];
        let outer: usize = self.shape()[..ax].iter().product();
        let inner: usize = self.shape()[ax + 1..].iter().product();
        let vals = self.to_vec::<f64>()?;
        let mut out = vec![!is_any; outer * inner];

        for o in 0..outer {
            for k in 0..dim {
                let base = (o * dim + k) * inner;
                for i in 0..inner {
                    let truthy = vals[base + i] != 0.0;
                    if is_any {
                        out[o * inner + i] |= truthy;
                    } else {
                        out[o * inner + i] &= truthy;
                    }
                }
            }
        }
        let mut out_shape = Vec::new();
        for (d_idx, &s) in self.shape().iter().enumerate() {
            if d_idx == ax {
                if keepdim {
                    out_shape.push(1);
                }
            } else {
                out_shape.push(s);
            }
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Tensor::from_slice_on(&out, &out_shape, &self.device())
    }

    /// Counts non-zero elements across the entire tensor or along `axis` (`torch.count_nonzero`).
    pub fn count_nonzero(&self, axis: Option<isize>, keepdim: bool) -> Result<Tensor> {
        match axis {
            None => {
                let vals = self.to_vec::<f64>()?;
                let cnt = vals.iter().filter(|&&v| v != 0.0).count() as i64;
                let shape = if keepdim && self.ndim() > 0 {
                    vec![1usize; self.ndim()]
                } else {
                    vec![1usize]
                };
                Tensor::from_slice_on(&[cnt], &shape, &self.device())
            }
            Some(a) => {
                let ax = normalize_axis("count_nonzero", a, self.ndim())?;
                let dim = self.shape()[ax];
                let outer: usize = self.shape()[..ax].iter().product();
                let inner: usize = self.shape()[ax + 1..].iter().product();
                let vals = self.to_vec::<f64>()?;
                let mut counts = vec![0i64; outer * inner];
                for o in 0..outer {
                    for k in 0..dim {
                        let base = (o * dim + k) * inner;
                        for i in 0..inner {
                            if vals[base + i] != 0.0 {
                                counts[o * inner + i] += 1;
                            }
                        }
                    }
                }
                let mut out_shape = Vec::new();
                for (d_idx, &s) in self.shape().iter().enumerate() {
                    if d_idx == ax {
                        if keepdim {
                            out_shape.push(1);
                        }
                    } else {
                        out_shape.push(s);
                    }
                }
                if out_shape.is_empty() {
                    out_shape.push(1);
                }
                Tensor::from_slice_on(&counts, &out_shape, &self.device())
            }
        }
    }

    ///Simultaneously computes `(min, max)` across all elements or along `axis` (`torch.aminmax`).
    pub fn aminmax(&self, axis: Option<isize>, keepdim: bool) -> Result<(Tensor, Tensor)> {
        match axis {
            None => {
                let flat = self.reshape(&[self.numel()])?;
                // torch.aminmax shares the gradient among ties (amin / amax semantics)
                let mn = flat.amin(&[0], false)?;
                let mx = flat.amax(&[0], false)?;
                if keepdim && self.ndim() > 0 {
                    let s = vec![1usize; self.ndim()];
                    Ok((mn.reshape(&s)?, mx.reshape(&s)?))
                } else {
                    Ok((mn, mx))
                }
            }
            Some(a) => Ok((self.amin(&[a], keepdim)?, self.amax(&[a], keepdim)?)),
        }
    }

    // =========================================================================
    // Core-6: Differentiable Fused Pointwise & Modern Activation Primitives
    // =========================================================================

    /// Differentiable fused elementwise multiply-accumulate: $\text{self} + \text{value} \cdot (\text{tensor1} \odot \text{tensor2})$ (`torch.addcmul`).
    pub fn addcmul(&self, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
        self.add(&tensor1.mul(tensor2)?.mul_scalar(value)?)
    }

    /// Differentiable fused elementwise divide-accumulate: $\text{self} + \text{value} \cdot (\text{tensor1} / \text{tensor2})$ (`torch.addcdiv`).
    pub fn addcdiv(&self, tensor1: &Tensor, tensor2: &Tensor, value: f64) -> Result<Tensor> {
        self.add(&tensor1.div(tensor2)?.mul_scalar(value)?)
    }


    /// Truncates fractional part toward zero (`torch.trunc`).
    pub fn trunc(&self) -> Result<Tensor> {
        self.run_unary_f64("trunc", f64::trunc, |g, _, _| g.zeros_like())
    }

    /// Differentiable fractional part $x - \operatorname{trunc}(x)$ (`torch.frac`).
    pub fn frac(&self) -> Result<Tensor> {
        self.sub(&self.trunc()?)
    }

    /// Differentiable elementwise magnitude of `self` with the sign of `other` (`torch.copysign`).
    pub fn copysign(&self, other: &Tensor) -> Result<Tensor> {
        let out_shape = broadcast_shapes("copysign", self.shape(), other.shape())?;
        let a_b = self.broadcast_to(&out_shape)?;
        let b_b = other.broadcast_to(&out_shape)?;
        let av = a_b.to_vec::<f64>()?;
        let bv = b_b.to_vec::<f64>()?;
        let out_vec: Vec<f64> = av.iter().zip(&bv).map(|(&a, &b)| a.copysign(b)).collect();
        let out_layout = Layout::contiguous(out_shape.clone());
        let cpu_s = CpuStorage::F64(out_vec).to_dtype(&out_layout, self.dtype())?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &self.device())?;
        let sa = self.shape().to_vec();
        let ra = self.requires_grad();
        let rb = other.requires_grad();
        let sb = other.shape().to_vec();
        let dev = self.device();
        let dt = self.dtype();
        Ok(Tensor::from_op(
            storage,
            out_layout,
            "copysign",
            vec![self.clone(), other.clone()],
            move |g| {
                let sign_vec: Vec<f64> = av
                    .iter()
                    .zip(&bv)
                    .map(|(&a, &b)| if a == 0.0 { 0.0 } else { (1.0f64).copysign(a) * (1.0f64).copysign(b) })
                    .collect();
                let st = Tensor::from_slice_on(&sign_vec, &out_shape, &dev)?.to_dtype(dt)?;
                let da = g.mul(&st)?.sum_to_shape(&sa)?;
                let db = Tensor::zeros(&sb, dt, &dev)?;
                Ok(vec![
                    if ra { Some(da) } else { None },
                    if rb { Some(db) } else { None },
                ])
            },
        ))
    }

    /// Elementwise C-style floating-point remainder $a - \operatorname{trunc}(a / b) \cdot b$ (`torch.fmod`).
    pub fn fmod(&self, other: &Tensor) -> Result<Tensor> {
        self.division_family(other, DivKind::Fmod)
    }

    /// Elementwise Python-style modulo remainder $a - \lfloor a / b \rfloor \cdot b$ (`torch.remainder`).
    pub fn remainder(&self, other: &Tensor) -> Result<Tensor> {
        self.division_family(other, DivKind::Remainder)
    }

    /// `fmod`, `remainder` and `floor_divide` evaluated exactly per element (not as `a - round(a / b) * b`, which is
    /// off by one near exact multiples and truncates for integers). Integer tensors follow PyTorch: `floor_divide`
    /// floors, `remainder` takes the sign of the divisor, `fmod` the sign of the dividend, and a zero divisor is an
    /// error. Gradients: `fmod` / `remainder` pass `g` to the dividend and `-quotient * g` to the divisor;
    /// `floor_divide` has none.
    pub(crate) fn division_family(&self, other: &Tensor, kind: DivKind) -> Result<Tensor> {
        let (op, bop) = match kind {
            DivKind::Fmod => ("fmod", BinaryOp::Fmod),
            DivKind::Remainder => ("remainder", BinaryOp::Remainder),
            DivKind::FloorDivide => ("floor_divide", BinaryOp::FloorDiv),
        };
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch { op, expected: self.dtype(), got: other.dtype() });
        }
        let (la, lb, out_shape) = self.broadcast_pair(other, op)?;
        // evaluated by the backend (on the tensors' device, parallel on CPU); integer division by zero is an error there
        let s = self.storage().binary_op(&la, other.storage(), &lb, bop)?;
        let (a, b) = (self.clone(), other.clone());
        let is_floor = matches!(kind, DivKind::FloorDivide);
        // The quotient that multiplies the divisor gradient is only needed when `b` requires grad.
        // fmod / remainder: d/da = 1, d/db = -quotient (trunc / floor); floor_divide is piecewise constant (zero gradient).
        Ok(Tensor::from_op(s, Layout::contiguous(out_shape), op, vec![self.clone(), other.clone()], move |g| {
            if is_floor {
                let zero = |t: &Tensor| if t.requires_grad() { g.zeros_like()?.sum_to_shape(t.shape()).map(Some) } else { Ok(None) };
                return Ok(vec![zero(&a)?, zero(&b)?]);
            }
            let ga = if a.requires_grad() { Some(g.sum_to_shape(a.shape())?) } else { None };
            let gb = if b.requires_grad() {
                let (ad, bd) = (a.detach(), b.detach());
                let q = match kind {
                    DivKind::Remainder => ad.division_family(&bd, DivKind::FloorDivide)?,
                    _ => {
                        let m = ad.division_family(&bd, DivKind::Fmod)?;
                        Tensor::sub(&ad, &m)?.div(&bd)?.trunc()? // (a - fmod(a, b)) / b
                    }
                };
                Some(g.mul(&q)?.neg()?.sum_to_shape(b.shape())?)
            } else {
                None
            };
            Ok(vec![ga, gb])
        }))
    }

    // =========================================================================
    // Core-7: Differentiable 3D Vector Cross Product & Multi-Axis Tensordot
    // =========================================================================

    /// Differentiable 3D vector cross product $a \times b$ along dimension `dim` of size 3 (`torch.cross`).
    /// Backward uses exact skew-symmetric adjoint: $\nabla_a = b \times g$, $\nabla_b = g \times a$.
    pub fn cross(&self, other: &Tensor, dim: isize) -> Result<Tensor> {
        let out_shape = broadcast_shapes("cross", self.shape(), other.shape())?;
        let ax = normalize_axis("cross", dim, out_shape.len())?;
        if out_shape[ax] != 3 {
            return Err(Error::Msg(format!(
                "cross: dimension {dim} must have size 3, got {}",
                out_shape[ax]
            )));
        }
        let a = self.broadcast_to(&out_shape)?;
        let b = other.broadcast_to(&out_shape)?;
        let a0 = a.select(ax as isize, 0)?;
        let a1 = a.select(ax as isize, 1)?;
        let a2 = a.select(ax as isize, 2)?;
        let b0 = b.select(ax as isize, 0)?;
        let b1 = b.select(ax as isize, 1)?;
        let b2 = b.select(ax as isize, 2)?;

        let c0 = Tensor::sub(&Tensor::mul(&a1, &b2)?, &Tensor::mul(&a2, &b1)?)?;
        let c1 = Tensor::sub(&Tensor::mul(&a2, &b0)?, &Tensor::mul(&a0, &b2)?)?;
        let c2 = Tensor::sub(&Tensor::mul(&a0, &b1)?, &Tensor::mul(&a1, &b0)?)?;
        Tensor::stack(&[c0, c1, c2], ax as isize)
    }

    /// Differentiable multi-axis tensor contraction (`torch.tensordot` / `np.tensordot`),
    /// reducing arbitrary axis contractions to a single high-throughput `matmul`.
    pub fn tensordot(&self, other: &Tensor, dims_a: &[usize], dims_b: &[usize]) -> Result<Tensor> {
        if dims_a.len() != dims_b.len() {
            return Err(Error::Msg(format!(
                "tensordot: dims_a ({:?}) and dims_b ({:?}) must have the same length",
                dims_a, dims_b
            )));
        }
        for (&da, &db) in dims_a.iter().zip(dims_b) {
            if da >= self.ndim() || db >= other.ndim() {
                return Err(Error::Msg("tensordot: contraction axis out of bounds".into()));
            }
            if self.shape()[da] != other.shape()[db] {
                return Err(Error::Msg(format!(
                    "tensordot: contracted dimension size mismatch ({} vs {})",
                    self.shape()[da],
                    other.shape()[db]
                )));
            }
        }
        let free_a: Vec<usize> = (0..self.ndim()).filter(|d| !dims_a.contains(d)).collect();
        let free_b: Vec<usize> = (0..other.ndim()).filter(|d| !dims_b.contains(d)).collect();

        let mut perm_a = free_a.clone();
        perm_a.extend_from_slice(dims_a);
        let mut perm_b = dims_b.to_vec();
        perm_b.extend_from_slice(&free_b);

        let a_perm = if perm_a.iter().enumerate().all(|(i, &p)| i == p) {
            self.clone()
        } else {
            self.permute(&perm_a)?.contiguous()?
        };
        let b_perm = if perm_b.iter().enumerate().all(|(i, &p)| i == p) {
            other.clone()
        } else {
            other.permute(&perm_b)?.contiguous()?
        };

        let m: usize = free_a.iter().map(|&d| self.shape()[d]).product::<usize>().max(1);
        let k: usize = dims_a.iter().map(|&d| self.shape()[d]).product::<usize>().max(1);
        let n: usize = free_b.iter().map(|&d| other.shape()[d]).product::<usize>().max(1);

        let a_2d = a_perm.reshape(&[m, k])?;
        let b_2d = b_perm.reshape(&[k, n])?;
        let prod = a_2d.matmul(&b_2d)?;

        let mut out_shape: Vec<usize> = free_a.iter().map(|&d| self.shape()[d]).collect();
        out_shape.extend(free_b.iter().map(|&d| other.shape()[d]));
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        prod.reshape(&out_shape)
    }

    // =========================================================================
    // Core-8: Differentiable Block-Diagonal, Vandermonde & Cartesian Product
    // =========================================================================

    /// Differentiable block-diagonal matrix assembly (`torch.block_diag`).
    /// Backward slices each input tensor's diagonal block out of `grad_output`.
    pub fn block_diag(tensors: &[&Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Tensor::zeros(&[0, 0], DType::F32, &Device::Cpu);
        }
        let dt = tensors[0].dtype();
        let dev = tensors[0].device();
        let mut shapes_2d = Vec::with_capacity(tensors.len());
        let mut orig_shapes = Vec::with_capacity(tensors.len());
        let mut total_r = 0usize;
        let mut total_c = 0usize;

        for &t in tensors {
            orig_shapes.push(t.shape().to_vec());
            let (r, c) = match t.ndim() {
                0 => (1, 1),
                1 => (1, t.shape()[0]),
                2 => (t.shape()[0], t.shape()[1]),
                _ => {
                    return Err(Error::Msg(
                        "block_diag: input tensors must be 0D, 1D, or 2D".into(),
                    ))
                }
            };
            shapes_2d.push((r, c));
            total_r += r;
            total_c += c;
        }

        let mut out_data = vec![0.0f64; total_r * total_c];
        let mut cur_r = 0usize;
        let mut cur_c = 0usize;
        for (&t, &(r, c)) in tensors.iter().zip(&shapes_2d) {
            let tv = t.to_vec::<f64>()?;
            for i in 0..r {
                for j in 0..c {
                    out_data[(cur_r + i) * total_c + (cur_c + j)] = tv[i * c + j];
                }
            }
            cur_r += r;
            cur_c += c;
        }

        let out_shape = vec![total_r, total_c];
        let out_layout = Layout::contiguous(out_shape);
        let cpu_s = CpuStorage::F64(out_data).to_dtype(&out_layout, dt)?;
        let storage = Storage::new_cpu(cpu_s).to_device(&out_layout, &dev)?;
        let inputs: Vec<Tensor> = tensors.iter().map(|&t| t.clone()).collect();
        let reqs: Vec<bool> = inputs.iter().map(Tensor::requires_grad).collect();

        Ok(Tensor::from_op(
            storage,
            out_layout,
            "block_diag",
            inputs,
            move |g| {
                let mut grads = Vec::with_capacity(shapes_2d.len());
                let mut r0 = 0usize;
                let mut c0 = 0usize;
                for ((&(r, c), orig_s), &req) in shapes_2d.iter().zip(&orig_shapes).zip(&reqs) {
                    if req {
                        let sub = g.narrow(0, r0, r)?.narrow(1, c0, c)?.contiguous()?.reshape(orig_s)?;
                        grads.push(Some(sub));
                    } else {
                        grads.push(None);
                    }
                    r0 += r;
                    c0 += c;
                }
                Ok(grads)
            },
        ))
    }

    /// Differentiable Vandermonde matrix generation from a 1D tensor (`torch.vander`).
    pub fn vander(&self, n: Option<usize>, increasing: bool) -> Result<Tensor> {
        if self.ndim() != 1 {
            return Err(Error::Msg(format!(
                "vander expects a 1D tensor, got {:?}",
                self.shape()
            )));
        }
        let cols = n.unwrap_or(self.shape()[0]);
        if cols == 0 {
            return Tensor::zeros(&[self.shape()[0], 0], self.dtype(), &self.device());
        }
        let mut col_tensors = Vec::with_capacity(cols);
        for j in 0..cols {
            let exp = if increasing { j } else { cols - 1 - j };
            let col = if exp == 0 {
                self.ones_like()?
            } else if exp == 1 {
                self.clone()
            } else {
                self.powf(exp as f64)?
            };
            col_tensors.push(col);
        }
        Tensor::stack(&col_tensors, 1)
    }

    /// Generates the Cartesian product of 1D input tensors (`torch.cartesian_prod`).
    pub fn cartesian_prod(tensors: &[&Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(Error::Msg("cartesian_prod expects at least one tensor".into()));
        }
        for &t in tensors {
            if t.ndim() != 1 {
                return Err(Error::Msg("cartesian_prod expects 1D tensors".into()));
            }
        }
        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }
        let grids = Tensor::meshgrid(tensors, true)?;
        let flat_grids: Result<Vec<Tensor>> = grids.iter().map(|g| g.reshape(&[g.numel()])).collect();
        Tensor::stack(&flat_grids?, 1)
    }
}

/// Geometry for [`Tensor::fold2d_scatter`]: image `(h, w)`, col grid `(gh, gw)`,
/// kernel, stride, padding, dilation.
#[derive(Clone, Copy)]
struct Fold2dGeo {
    h: usize,
    w: usize,
    gh: usize,
    gw: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
}

// ---------------------------------------------------------------------------
// Einsum helpers
// ---------------------------------------------------------------------------
fn eval_einsum_forward(in_specs: &[&str], out_spec: &str, ops: &[&Tensor]) -> Result<Tensor> {
    let mut dim_sizes = HashMap::<char, usize>::new();
    let mut all_labels = Vec::<char>::new();

    for (&spec, &t) in in_specs.iter().zip(ops) {
        if spec.len() != t.ndim() {
            return Err(Error::Msg(format!(
                "einsum: subscript '{spec}' has {} labels for tensor of rank {}",
                spec.len(),
                t.ndim()
            )));
        }
        for (ch, &d) in spec.chars().zip(t.shape()) {
            if let Some(&prev) = dim_sizes.get(&ch) {
                if prev != d {
                    return Err(Error::Msg(format!(
                        "einsum: dimension mismatch for label '{ch}' ({prev} vs {d})"
                    )));
                }
            } else {
                dim_sizes.insert(ch, d);
                all_labels.push(ch);
            }
        }
    }

    let out_labels: Vec<char> = out_spec.chars().collect();
    {
        let mut seen = Vec::with_capacity(out_labels.len());
        for &c in &out_labels {
            if !dim_sizes.contains_key(&c) {
                return Err(Error::Msg(format!(
                    "einsum: output subscript '{c}' does not appear in the inputs"
                )));
            }
            if seen.contains(&c) {
                return Err(Error::Msg(format!(
                    "einsum: output subscript '{c}' appears more than once"
                )));
            }
            seen.push(c);
        }
    }
    let sum_labels: Vec<char> = all_labels
        .iter()
        .copied()
        .filter(|c| !out_labels.contains(c))
        .collect();

    let out_shape: Vec<usize> = out_labels.iter().map(|c| dim_sizes[c]).collect();
    let sum_shape: Vec<usize> = sum_labels.iter().map(|c| dim_sizes[c]).collect();
    // Labels come from validated tensors, but an outer product over disjoint
    // labels multiplies INDEPENDENT sizes: the product can overflow usize
    // (debug panic / silently wrong contraction) or OOM-abort the vec below.
    let out_numel = crate::limits::check_elements("einsum", &out_shape)?;
    // The contraction-trip count needs overflow refusal (correctness), but no
    // size cap: like matmul's k, big contractions are legitimate slow work.
    let sum_numel = crate::layout::checked_numel(&sum_shape)?;

    let op_vecs: Vec<Vec<f64>> = ops
        .iter()
        .map(|t| t.to_vec::<f64>())
        .collect::<Result<_>>()?;
    let op_strides: Vec<Vec<isize>> = ops
        .iter()
        .map(|t| Layout::contiguous(t.shape().to_vec()).strides().to_vec())
        .collect();
    let op_chars: Vec<Vec<char>> = in_specs.iter().map(|s| s.chars().collect()).collect();

    let mut out_data = crate::layout::try_filled(out_numel, 0.0f64)?;
    let mut out_coord = vec![0usize; out_labels.len()];
    let mut label_val = HashMap::<char, usize>::new();

    for out_item in out_data.iter_mut().take(out_numel) {
        for (&ch, &v) in out_labels.iter().zip(&out_coord) {
            label_val.insert(ch, v);
        }
        let mut sum_coord = vec![0usize; sum_labels.len()];
        let mut acc = 0.0f64;
        for _ in 0..sum_numel {
            for (&ch, &v) in sum_labels.iter().zip(&sum_coord) {
                label_val.insert(ch, v);
            }
            let mut prod = 1.0f64;
            for op_idx in 0..ops.len() {
                let mut off = 0isize;
                for (ax, &ch) in op_chars[op_idx].iter().enumerate() {
                    off += (label_val[&ch] as isize) * op_strides[op_idx][ax];
                }
                prod *= op_vecs[op_idx][off as usize];
            }
            acc += prod;

            for d in (0..sum_shape.len()).rev() {
                sum_coord[d] += 1;
                if sum_coord[d] < sum_shape[d] {
                    break;
                }
                sum_coord[d] = 0;
            }
        }
        *out_item = acc;

        for d in (0..out_shape.len()).rev() {
            out_coord[d] += 1;
            if out_coord[d] < out_shape[d] {
                break;
            }
            out_coord[d] = 0;
        }
    }

    let dtype = ops[0].dtype();
    let dev = ops[0].device();
    let cpu = CpuStorage::F64(out_data).to_dtype(&Layout::contiguous(out_shape.clone()), dtype)?;
    let storage = Storage::new_cpu(cpu).to_device(&Layout::contiguous(out_shape.clone()), &dev)?;
    Ok(Tensor::from_storage(storage, Layout::contiguous(out_shape)))
}

fn einsum_backward_single(
    in_spec: &str,
    out_spec: &str,
    in_shape: &[usize],
    grad_out: &Tensor,
) -> Result<Tensor> {
    // General adjoint of 1-operand einsum: every output element `out[coord]` received a
    // contribution from `in[in_coord]`, so scatter/broadcast `grad_out` back into `in_shape`.
    let in_chars: Vec<char> = in_spec.chars().collect();
    let out_chars: Vec<char> = out_spec.chars().collect();
    let mut dim_sizes = HashMap::<char, usize>::new();
    for (&c, &d) in in_chars.iter().zip(in_shape) {
        dim_sizes.insert(c, d);
    }

    let g_vec = grad_out.to_vec::<f64>()?;
    let g_strides = Layout::contiguous(grad_out.shape().to_vec())
        .strides()
        .to_vec();
    let in_numel: usize = in_shape.iter().product();
    let mut grad_in = vec![0.0f64; in_numel];
    let mut coord = vec![0usize; in_shape.len()];

    for dst in grad_in.iter_mut().take(in_numel) {
        // Check if repeated indices in `in_spec` are equal at this coordinate (e.g. "ii->i").
        let mut valid = true;
        let mut vals = HashMap::<char, usize>::new();
        for (&ch, &c) in in_chars.iter().zip(&coord) {
            if let Some(&prev) = vals.get(&ch) {
                if prev != c {
                    valid = false;
                    break;
                }
            } else {
                vals.insert(ch, c);
            }
        }
        if valid {
            let mut g_off = 0isize;
            for (ax, &och) in out_chars.iter().enumerate() {
                g_off += (vals[&och] as isize) * g_strides[ax];
            }
            *dst = g_vec[g_off as usize];
        }
        for d in (0..in_shape.len()).rev() {
            coord[d] += 1;
            if coord[d] < in_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    let cpu = CpuStorage::F64(grad_in)
        .to_dtype(&Layout::contiguous(in_shape.to_vec()), grad_out.dtype())?;
    let storage = Storage::new_cpu(cpu)
        .to_device(&Layout::contiguous(in_shape.to_vec()), &grad_out.device())?;
    Ok(Tensor::from_storage(
        storage,
        Layout::contiguous(in_shape.to_vec()),
    ))
}

fn einsum_backward_pair(
    target_spec: &str,
    other_spec: &str,
    out_spec: &str,
    target_shape: &[usize],
    grad_out: &Tensor,
    other: &Tensor,
) -> Result<Tensor> {
    // When target_spec has no repeated labels and every label in target_spec appears in
    // either out_spec or other_spec, the gradient is simply `einsum("{out},{other}->{target}", [grad_out, other])`!
    let mut unique = Vec::new();
    let mut has_dup = false;
    for c in target_spec.chars() {
        if unique.contains(&c) {
            has_dup = true;
        } else {
            unique.push(c);
        }
    }
    let all_covered = target_spec
        .chars()
        .all(|c| out_spec.contains(c) || other_spec.contains(c));
    if !has_dup && all_covered {
        let eq = format!("{out_spec},{other_spec}->{target_spec}");
        return Tensor::einsum(&eq, &[grad_out, other]);
    }
    // General fallback when target_spec has uncontracted free dims not in out_spec or other_spec:
    let mut reduced_target = String::new();
    let mut kept_shape = Vec::new();
    for (c, &d) in target_spec.chars().zip(target_shape) {
        if (out_spec.contains(c) || other_spec.contains(c)) && !reduced_target.contains(c) {
            reduced_target.push(c);
            kept_shape.push(d);
        } else {
            kept_shape.push(1);
        }
    }
    let eq = format!("{out_spec},{other_spec}->{reduced_target}");
    let g_red = Tensor::einsum(&eq, &[grad_out, other])?;
    let g_full = g_red.reshape(&kept_shape)?.broadcast_to(target_shape)?;
    if !has_dup {
        return Ok(g_full);
    }
    // Repeated target labels only receive gradient where all their axes agree
    // (diagonal semantics); the broadcast above fills the off-diagonal too.
    let mask = einsum_dup_mask(target_spec, target_shape, &grad_out.device())?;
    Tensor::where_cond(&mask, &g_full, &g_full.zeros_like()?)
}

/// Boolean mask over `target_shape` that is true exactly where every repeated
/// label of `target_spec` has equal coordinates on all its axes.
fn einsum_dup_mask(target_spec: &str, target_shape: &[usize], dev: &Device) -> Result<Tensor> {
    let chars: Vec<char> = target_spec.chars().collect();
    let numel: usize = target_shape.iter().product();
    let mut mask = Tensor::from_slice_on(&vec![true; numel], target_shape, dev)?;
    let mut done = Vec::new();
    for (ax, &c) in chars.iter().enumerate() {
        if done.contains(&c) {
            continue;
        }
        done.push(c);
        let axes: Vec<usize> = chars
            .iter()
            .enumerate()
            .filter(|(_, x)| **x == c)
            .map(|(i, _)| i)
            .collect();
        if axes.len() < 2 {
            continue;
        }
        let d = target_shape[ax];
        let idx: Vec<i64> = (0..d as i64).collect();
        for pair in axes.windows(2) {
            let mut shape_a = vec![1usize; target_shape.len()];
            let mut shape_b = vec![1usize; target_shape.len()];
            shape_a[pair[0]] = d;
            shape_b[pair[1]] = d;
            let a = Tensor::from_slice_on(&idx, &shape_a, dev)?.broadcast_to(target_shape)?;
            let b = Tensor::from_slice_on(&idx, &shape_b, dev)?.broadcast_to(target_shape)?;
            mask = Tensor::mul(&mask, &a.eq_tensor(&b)?)?;
        }
    }
    Ok(mask)
}

// ---------------------------------------------------------------------------
// Operator Overloading (`Result<Tensor>` like Candle)
// ---------------------------------------------------------------------------
macro_rules! impl_op {
    ($trait:ident, $m:ident) => {
        impl $trait<&Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $m(self, rhs: &Tensor) -> Result<Tensor> {
                Tensor::$m(self, rhs)
            }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Result<Tensor>;
            fn $m(self, rhs: Tensor) -> Result<Tensor> {
                Tensor::$m(self, &rhs)
            }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $m(self, rhs: &Tensor) -> Result<Tensor> {
                Tensor::$m(&self, rhs)
            }
        }
        impl $trait<Tensor> for Tensor {
            type Output = Result<Tensor>;
            fn $m(self, rhs: Tensor) -> Result<Tensor> {
                Tensor::$m(&self, &rhs)
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

impl std::ops::Neg for &Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> {
        Tensor::neg(self)
    }
}

impl std::ops::Neg for Tensor {
    type Output = Result<Tensor>;
    fn neg(self) -> Result<Tensor> {
        Tensor::neg(&self)
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
    fn topk_sorts_i64_beyond_f64_range_exactly() {
        // 2^53+1 rounds to 2^53 in f64; an f64 comparison key misorders them.
        let vals = [9_007_199_254_740_993i64, 9_007_199_254_740_992, 0];
        let t = Tensor::from_slice(&vals, &[3]).unwrap();
        let (sorted, idx) = t.topk(3, 0, false, true).unwrap();
        assert_eq!(sorted.to_vec::<i64>().unwrap(), vec![0, 9_007_199_254_740_992, 9_007_199_254_740_993]);
        assert_eq!(idx.to_vec::<i64>().unwrap(), vec![2, 1, 0]);
        let (sorted_d, idx_d) = t.topk(3, 0, true, true).unwrap();
        assert_eq!(sorted_d.to_vec::<i64>().unwrap(), vec![9_007_199_254_740_993, 9_007_199_254_740_992, 0]);
        assert_eq!(idx_d.to_vec::<i64>().unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn searchsorted_compares_i64_beyond_f64_range_exactly() {
        let seq = Tensor::from_slice(
            &[9_007_199_254_740_992i64, 9_007_199_254_740_994, 9_007_199_254_740_996],
            &[3],
        )
        .unwrap();
        let q = Tensor::from_slice(&[9_007_199_254_740_993i64, 9_007_199_254_740_995], &[2]).unwrap();
        let pos = Tensor::searchsorted(&seq, &q, false).unwrap();
        assert_eq!(pos.to_vec::<i64>().unwrap(), vec![1, 2]);
        let pos_r = Tensor::searchsorted(&seq, &q, true).unwrap();
        assert_eq!(pos_r.to_vec::<i64>().unwrap(), vec![1, 2]);
    }

    #[test]
    fn searchsorted_nd_rows_nan_query_and_shape_rule() {
        let seq =
            Tensor::from_slice(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], &[2, 3]).unwrap();
        let q = Tensor::from_slice(&[0.0, 3.0, 9.0, 1.0, 4.0, f64::NAN], &[2, 3]).unwrap();
        let pos = Tensor::searchsorted(&seq, &q, false).unwrap();
        assert_eq!(pos.to_vec::<i64>().unwrap(), vec![0, 1, 3, 0, 1, 3]);
        let pos_r = Tensor::searchsorted(&seq, &q, true).unwrap();
        assert_eq!(pos_r.to_vec::<i64>().unwrap(), vec![0, 2, 3, 0, 2, 3]);
        // Leading dims must match exactly (torch parity, no broadcasting).
        let bad = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(Tensor::searchsorted(&seq, &bad, false).is_err());
        // 1D sequences accept anything, including scalars.
        let seq1 = Tensor::from_slice(&[1.0, 3.0, 5.0], &[3]).unwrap();
        let scalar = Tensor::from_slice(&[4.0], &[]).unwrap();
        let pos1 = Tensor::searchsorted(&seq1, &scalar, false).unwrap();
        assert_eq!(pos1.to_vec::<i64>().unwrap(), vec![2]);
        assert!(pos1.shape().is_empty());
    }

    #[test]
    fn kthvalue_selects_ranked_values_keepdim_and_rejects_bad_k() {
        let t = Tensor::from_slice(&[3.0, 1.0, 2.0, 0.0, -1.0, 4.0], &[2, 3]).unwrap();
        for (k, want_v, want_i) in [
            (1, vec![1.0, -1.0], vec![1, 1]),
            (2, vec![2.0, 0.0], vec![2, 0]),
            (3, vec![3.0, 4.0], vec![0, 2]),
        ] {
            let (v, i) = t.kthvalue(k, 1, false).unwrap();
            assert_eq!(v.shape(), &[2]);
            assert_eq!(v.to_vec::<f64>().unwrap(), want_v, "k={k} values");
            assert_eq!(i.to_vec::<i64>().unwrap(), want_i, "k={k} indices");
        }
        let (v, i) = t.kthvalue(2, 0, true).unwrap();
        assert_eq!(v.shape(), &[1, 3]);
        assert_eq!(v.to_vec::<f64>().unwrap(), vec![3.0, 1.0, 4.0]);
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![0, 0, 1]);
        assert!(t.kthvalue(0, 1, false).is_err());
        assert!(t.kthvalue(4, 1, false).is_err());
    }

    #[test]
    fn scatter_writes_last_wins_and_zeros_scattered_grads() {
        let base = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let idx = Tensor::from_slice(&[2i64, 0, 0, 2], &[2, 2]).unwrap();
        let src = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
        let out = base.scatter(1, &idx, &src).unwrap();
        assert_eq!(
            out.to_vec::<f64>().unwrap(),
            vec![20.0, 2.0, 10.0, 30.0, 5.0, 40.0]
        );
        // Duplicate index 0: last write (8.0) wins, deterministically.
        let dup = Tensor::from_slice(&[0i64, 0], &[1, 2]).unwrap();
        let dsrc = Tensor::from_slice(&[7.0, 8.0], &[1, 2]).unwrap();
        let row = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let out = row.scatter(1, &dup, &dsrc).unwrap();
        assert_eq!(out.to_vec::<f64>().unwrap(), vec![8.0, 2.0, 3.0]);
        // Gradients: scattered positions read 0 wrt the base, src gathers grad.
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        let s = Tensor::from_slice(&[10.0, 20.0], &[2])
            .unwrap()
            .requires_grad_(true);
        let ii = Tensor::from_slice(&[2i64, 0], &[2]).unwrap();
        b.scatter(0, &ii, &s).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(b.grad().unwrap().to_vec::<f64>().unwrap(), vec![0.0, 1.0, 0.0]);
        assert_eq!(s.grad().unwrap().to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn erf_matches_reference_values_at_f64_precision() {
        // Reference values: Abramowitz & Stegun / torch.special.erf (float64).
        let xs = [0.0, 0.5, 1.0, 2.0, -1.0, -0.25, 6.0];
        let want = [
            0.0,
            0.5204998778130465,
            0.8427007929497149,
            0.9953222650189527,
            -0.8427007929497149,
            -0.2763263901682369,
            1.0,
        ];
        let got = Tensor::from_slice(&xs, &[xs.len()])
            .unwrap()
            .erf()
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&got, &want, 1e-12, "erf");
    }

    #[test]
    fn gelu_exact_matches_normal_cdf_form_at_f64_precision() {
        // GELU(x) = x * Phi(x), Phi the standard normal CDF.
        // Anchor: gelu(1) = Phi(1) = 0.8413447460685429 (torch float64).
        let xs = [-2.0, -0.5, 0.0, 0.5, 1.0, 3.0];
        let got = Tensor::from_slice(&xs, &[xs.len()])
            .unwrap()
            .gelu()
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        let want: Vec<f64> = xs
            .iter()
            .map(|&x| x * 0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2)))
            .collect();
        assert_close(&got, &want, 1e-12, "gelu-vs-cdf-form");
        assert!(
            (got[4] - 0.8413447460685429).abs() <= 1e-12,
            "gelu(1) anchor: {}",
            got[4]
        );
    }

    #[test]
    fn logsumexp_all_negative_infinity_returns_negative_infinity() {
        for keepdim in [true, false] {
            let x = Tensor::from_slice(&[f64::NEG_INFINITY; 3], &[3]).unwrap();
            let got = x.logsumexp(0, keepdim).unwrap().to_vec::<f64>().unwrap();
            assert_eq!(got.len(), 1);
            assert_eq!(got[0], f64::NEG_INFINITY, "keepdim={keepdim}");
        }
        // a single finite entry keeps the row finite
        let x = Tensor::from_slice(&[f64::NEG_INFINITY, 2.0, f64::NEG_INFINITY], &[3]).unwrap();
        let got = x.logsumexp(-1, false).unwrap().item::<f64>().unwrap();
        assert!((got - 2.0).abs() <= 1e-12, "got {got}");
        // fully-masked rows must not poison gradients with NaN
        let x = Tensor::from_slice(&[f64::NEG_INFINITY; 3], &[3])
            .unwrap()
            .requires_grad_(true);
        x.logsumexp(0, false).unwrap().backward().unwrap();
        let grad = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[0.0, 0.0, 0.0], 0.0, "all-neg-inf grad");
    }

    #[test]
    fn logsumexp_matches_hand_computed_values_and_softmax_grads() {
        let x = Tensor::from_slice(&[0.0, 0.0], &[2]).unwrap();
        let got = x.logsumexp(0, false).unwrap().item::<f64>().unwrap();
        assert!((got - 2.0f64.ln()).abs() <= 1e-12, "got {got}");
        // d/dx logsumexp = softmax: grads are positive and sum to 1
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        x.logsumexp(0, false).unwrap().backward().unwrap();
        let grad = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert!(grad.iter().all(|&g| g > 0.0));
        assert!((grad.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn softmax_of_large_values_is_finite_and_normalized() {
        let x = Tensor::from_slice(&[1000.0, 1001.0, 999.0], &[3]).unwrap();
        let got = x.softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert!(got.iter().all(|v| v.is_finite()));
        assert!((got.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
        let max_i = got
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(max_i, 1);
    }

    #[test]
    fn topk_nan_sorts_as_largest() {
        let x = Tensor::from_slice(&[3.0, f64::NAN, 1.0, 2.0], &[4]).unwrap();
        let (values, indices) = x.topk(2, 0, true, true).unwrap();
        let v = values.to_vec::<f64>().unwrap();
        let idx = indices.to_vec::<i64>().unwrap();
        assert!(v[0].is_nan() && idx[0] == 1, "largest first: {v:?} {idx:?}");
        assert!(
            (v[1] - 3.0).abs() <= 0.0 && idx[1] == 0,
            "largest rest: {v:?} {idx:?}"
        );
        let (values, indices) = x.topk(2, 0, false, true).unwrap();
        let v = values.to_vec::<f64>().unwrap();
        let idx = indices.to_vec::<i64>().unwrap();
        assert_eq!((v, idx), (vec![1.0, 2.0], vec![2, 3]));
    }

    #[test]
    fn sort_and_argsort_inherit_nan_ordering() {
        let x = Tensor::from_slice(&[2.0, f64::NAN, 1.0], &[3]).unwrap();
        let (values, _) = x.sort(0, true).unwrap();
        let v = values.to_vec::<f64>().unwrap();
        assert!(v[0].is_nan(), "descending starts with NaN: {v:?}");
        assert_close(&v[1..], &[2.0, 1.0], 0.0, "descending rest");
        let idx = x.argsort(0, false).unwrap().to_vec::<i64>().unwrap();
        assert_eq!(idx, vec![2, 0, 1], "ascending puts NaN last");
    }

    #[test]
    fn logsumexp_positive_infinity_and_nan_slices() {
        // +inf max (no NaN): +inf with zero gradient
        let x = Tensor::from_slice(&[f64::INFINITY, 1.0, 2.0], &[3])
            .unwrap()
            .requires_grad_(true);
        let y = x.logsumexp(0, false).unwrap();
        assert_eq!(y.item::<f64>().unwrap(), f64::INFINITY);
        y.backward().unwrap();
        let grad = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[0.0, 0.0, 0.0], 0.0, "+inf grad");
        // NaN is never silenced: all-NaN and mixed rows stay NaN
        for data in [
            vec![f64::NAN, f64::NAN],
            vec![1.0, f64::NAN],
            vec![f64::INFINITY, f64::NAN],
        ] {
            let x = Tensor::from_slice(&data, &[2]).unwrap();
            let got = x.logsumexp(0, false).unwrap().item::<f64>().unwrap();
            assert!(got.is_nan(), "{data:?} -> {got}");
        }
    }

    #[test]
    fn softmax_positive_infinity_is_one_hot() {
        let x = Tensor::from_slice(&[f64::INFINITY, 1.0, 2.0], &[3]).unwrap();
        let got = x.softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert_close(&got, &[1.0, 0.0, 0.0], 0.0, "single +inf");
        let x = Tensor::from_slice(&[f64::INFINITY, f64::INFINITY, 1.0], &[3]).unwrap();
        let got = x.softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert_close(&got, &[0.5, 0.5, 0.0], 1e-12, "split +inf");
        // all -inf stays NaN (PyTorch parity)
        let x = Tensor::from_slice(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[2]).unwrap();
        let got = x.softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert!(got.iter().all(|v| v.is_nan()));
        // zero gradient at the one-hot corner (the Jacobian is flat there)
        let x = Tensor::from_slice(&[f64::INFINITY, 1.0], &[2])
            .unwrap()
            .requires_grad_(true);
        let y = x.softmax(0).unwrap().sum().unwrap();
        y.backward().unwrap();
        let grad = x.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[0.0, 0.0], 0.0, "softmax +inf grad");
        // NaN still poisons, even mixed with +inf
        for data in [vec![f64::INFINITY, f64::NAN], vec![f64::NAN, f64::NAN]] {
            let x = Tensor::from_slice(&data, &[2]).unwrap();
            let got = x.softmax(0).unwrap().to_vec::<f64>().unwrap();
            assert!(got.iter().all(|v| v.is_nan()), "{data:?} -> {got:?}");
        }
    }

    #[test]
    fn log_softmax_positive_infinity_and_matches_log_softmax() {
        let x = Tensor::from_slice(&[f64::INFINITY, 1.0], &[2]).unwrap();
        let got = x.log_softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert_eq!(got[0], 0.0);
        assert_eq!(got[1], f64::NEG_INFINITY);
        let x = Tensor::from_slice(&[f64::INFINITY, f64::INFINITY, 1.0], &[3]).unwrap();
        let got = x.log_softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert_close(&got[..2], &[0.5f64.ln(), 0.5f64.ln()], 1e-12, "split +inf");
        assert_eq!(got[2], f64::NEG_INFINITY);
        // all -inf stays NaN (PyTorch parity)
        let x = Tensor::from_slice(&[f64::NEG_INFINITY, f64::NEG_INFINITY], &[2]).unwrap();
        let got = x.log_softmax(0).unwrap().to_vec::<f64>().unwrap();
        assert!(got.iter().all(|v| v.is_nan()));
        // finite inputs agree with log(softmax)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let a = x.log_softmax(0).unwrap().to_vec::<f64>().unwrap();
        let b = x.softmax(0).unwrap().ln().unwrap().to_vec::<f64>().unwrap();
        assert_close(&a, &b, 1e-12, "log_softmax vs ln(softmax)");
        // NaN still poisons, even mixed with +inf
        for data in [vec![f64::INFINITY, f64::NAN], vec![f64::NAN, f64::NAN]] {
            let x = Tensor::from_slice(&data, &[2]).unwrap();
            let got = x.log_softmax(0).unwrap().to_vec::<f64>().unwrap();
            assert!(got.iter().all(|v| v.is_nan()), "{data:?} -> {got:?}");
        }
    }

    #[test]
    fn var_matches_hand_values_and_unbiased_single_is_nan() {
        // [1,2,3,4]: biased 5/4, unbiased 5/3
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let b = x.var_axis(0, false, false).unwrap().item::<f64>().unwrap();
        assert!((b - 1.25).abs() <= 1e-12, "biased {b}");
        let u = x.var_axis(0, true, false).unwrap().item::<f64>().unwrap();
        assert!((u - 5.0 / 3.0).abs() <= 1e-12, "unbiased {u}");
        // an unbiased single element is undefined -> NaN like torch
        let one = Tensor::from_slice(&[5.0], &[1]).unwrap();
        assert!(
            one.var_axis(0, true, false)
                .unwrap()
                .item::<f64>()
                .unwrap()
                .is_nan()
        );
        assert_eq!(
            one.var_axis(0, false, false)
                .unwrap()
                .item::<f64>()
                .unwrap(),
            0.0
        );
        // std of a constant sits on the documented 1e-6 stability floor
        let c = Tensor::from_slice(&[2.0, 2.0], &[2]).unwrap();
        let s = c.std_axis(0, false, false).unwrap().item::<f64>().unwrap();
        assert!((s - 1e-6).abs() <= 1e-12, "std floor {s}");
    }

    #[test]
    fn empty_reductions_match_torch_nan_and_zero() {
        let e = Tensor::from_slice(&[] as &[f64], &[0]).unwrap();
        assert!(e.mean().unwrap().item::<f64>().unwrap().is_nan());
        assert_eq!(e.sum().unwrap().item::<f64>().unwrap(), 0.0);
        // duplicate axes must not inflate the divisor
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let once = x.mean_axes(&[0], false).unwrap().to_vec::<f64>().unwrap();
        let twice = x
            .mean_axes(&[0, 0], false)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&once, &[2.0, 3.0], 1e-12, "mean axis 0");
        assert_close(&twice, &[2.0, 3.0], 1e-12, "dedup axes");
    }

    #[test]
    fn cumsum_values_and_backward_match_hand_computation() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let got = x.cumsum(0).unwrap().to_vec::<f64>().unwrap();
        assert_close(&got, &[1.0, 3.0, 6.0, 10.0], 1e-12, "cumsum");
        // d sum(cumsum(x))/dx[i] = n - i
        let xg = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4])
            .unwrap()
            .requires_grad_(true);
        xg.cumsum(0).unwrap().sum().unwrap().backward().unwrap();
        let grad = xg.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[4.0, 3.0, 2.0, 1.0], 1e-12, "cumsum grad");
    }

    #[test]
    fn cumprod_values_and_zero_safe_backward() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let got = x.cumprod(0).unwrap().to_vec::<f64>().unwrap();
        assert_close(&got, &[1.0, 2.0, 6.0, 24.0], 1e-12, "cumprod");
        // zeros flow through the backward without any division (torch-verified)
        let xz = Tensor::from_slice(&[1.0, 0.0, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        xz.cumprod(0).unwrap().sum().unwrap().backward().unwrap();
        let grad = xz.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[1.0, 4.0, 0.0], 1e-12, "cumprod zero grad");
    }

    #[test]
    fn nanmean_nansum_skip_nan_and_match_hand_computation() {
        let x = Tensor::from_slice(&[1.0, f64::NAN, 3.0, 5.0], &[2, 2]).unwrap();
        assert_close(
            &x.nanmean().unwrap().to_vec::<f64>().unwrap(),
            &[3.0],
            1e-12,
            "nanmean",
        );
        assert_close(
            &x.nansum().unwrap().to_vec::<f64>().unwrap(),
            &[9.0],
            1e-12,
            "nansum",
        );
        assert_close(
            &x.nanmean_axes(&[1], false)
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[1.0, 4.0],
            1e-12,
            "nanmean axis",
        );
        // all-NaN: mean NaN, sum 0
        let a = Tensor::from_slice(&[f64::NAN, f64::NAN], &[2]).unwrap();
        assert!(a.nanmean().unwrap().to_vec::<f64>().unwrap()[0].is_nan());
        assert_close(
            &a.nansum().unwrap().to_vec::<f64>().unwrap(),
            &[0.0],
            0.0,
            "nansum all-nan",
        );
        // integer input: nansum fine, nanmean fails closed like torch
        let xi = Tensor::from_slice(&[1i64, 2, 3], &[3]).unwrap();
        assert_eq!(xi.nansum().unwrap().to_vec::<i64>().unwrap(), vec![6]);
        assert!(xi.nanmean().is_err());
        // d nanmean/dx = 1/count on valid entries, 0 on NaN
        let xg = Tensor::from_slice(&[1.0, f64::NAN, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        xg.nanmean_axes(&[0], false)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &xg.grad().unwrap().to_vec::<f64>().unwrap(),
            &[0.5, 0.0, 0.5],
            1e-12,
            "nanmean grad",
        );
    }

    #[test]
    fn nanmedian_takes_lower_middle_with_original_indices() {
        // odd count
        let x = Tensor::from_slice(&[3.0, 1.0, 2.0], &[3]).unwrap();
        let (v, i) = x.nanmedian_axis(0, false).unwrap();
        assert_close(&v.to_vec::<f64>().unwrap(), &[2.0], 0.0, "nanmedian odd");
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![2]);
        // even count: LOWER middle, no averaging (torch parity)
        let e = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let (v, i) = e.nanmedian_axis(0, false).unwrap();
        assert_close(&v.to_vec::<f64>().unwrap(), &[2.0], 0.0, "nanmedian even");
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![1]);
        // NaN ignored for the count, indices still address the original tensor
        let n = Tensor::from_slice(&[3.0, f64::NAN, 1.0, 2.0], &[4]).unwrap();
        let (v, i) = n.nanmedian_axis(-1, true).unwrap();
        assert_eq!(v.shape(), &[1]);
        assert_close(&v.to_vec::<f64>().unwrap(), &[2.0], 0.0, "nanmedian nan");
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![3]);
        // full reduction (even valid count -> lower middle)
        let f = Tensor::from_slice(&[3.0, f64::NAN, 1.0], &[3]).unwrap();
        assert_close(
            &f.nanmedian().unwrap().to_vec::<f64>().unwrap(),
            &[1.0],
            0.0,
            "nanmedian full",
        );
        // all-NaN reads NaN, empty errors like torch
        let a = Tensor::from_slice(&[f64::NAN, f64::NAN], &[2]).unwrap();
        assert!(a.nanmedian().unwrap().to_vec::<f64>().unwrap()[0].is_nan());
        let empty = Tensor::from_slice(&[] as &[f64], &[0]).unwrap();
        assert!(empty.nanmedian_axis(0, false).is_err());
        assert!(empty.nanmedian().is_err());
        // gradient flows to the chosen element only
        let g = Tensor::from_slice(&[1.0, f64::NAN, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        g.nanmedian_axis(0, false)
            .unwrap()
            .0
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &g.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 0.0, 0.0],
            0.0,
            "nanmedian grad",
        );
    }

    #[test]
    fn median_takes_lower_middle_and_propagates_first_nan() {
        // odd count
        let x = Tensor::from_slice(&[3.0, 1.0, 2.0], &[3]).unwrap();
        let (v, i) = x.median_axis(0, false).unwrap();
        assert_close(&v.to_vec::<f64>().unwrap(), &[2.0], 0.0, "median odd");
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![2]);
        // even count: LOWER middle, never averaged (unlike numpy)
        let e = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let (v, i) = e.median_axis(-1, true).unwrap();
        assert_eq!(v.shape(), &[1]);
        assert_close(&v.to_vec::<f64>().unwrap(), &[2.0], 0.0, "median even");
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![1]);
        // integers: lower middle, no averaging
        let xi = Tensor::from_slice(&[1i64, 2, 3, 4], &[4]).unwrap();
        let (v, i) = xi.median_axis(0, false).unwrap();
        assert_eq!(v.to_vec::<i64>().unwrap(), vec![2]);
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![1]);
        // NaN propagates; values/indices address the FIRST NaN
        let n = Tensor::from_slice(&[1.0, f64::NAN, 3.0, f64::NAN, 5.0], &[5]).unwrap();
        let (v, i) = n.median_axis(0, false).unwrap();
        assert!(v.to_vec::<f64>().unwrap()[0].is_nan());
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![1]);
        // full reduction + empty errors like torch
        let f = Tensor::from_slice(&[3.0, 1.0, 2.0, 0.0], &[2, 2]).unwrap();
        assert_close(
            &f.median().unwrap().to_vec::<f64>().unwrap(),
            &[1.0],
            0.0,
            "median full",
        );
        let empty = Tensor::from_slice(&[] as &[f64], &[0]).unwrap();
        assert!(empty.median_axis(0, false).is_err());
        assert!(empty.median().is_err());
        // gradient flows to the winner only (first NaN when present)
        let g = Tensor::from_slice(&[1.0, f64::NAN, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        g.median_axis(0, false)
            .unwrap()
            .0
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &g.grad().unwrap().to_vec::<f64>().unwrap(),
            &[0.0, 1.0, 0.0],
            0.0,
            "median grad",
        );
    }

    #[test]
    fn quantile_methods_match_hand_computation_and_validate() {
        // sorted: [1,1,2,3,4,5,6,9]; q=0.5 -> pos 3.5, q=0.3 -> pos 2.1
        let x = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[8]).unwrap();
        for (m, half, third) in [
            ("linear", 3.5, 2.1),
            ("lower", 3.0, 2.0),
            ("higher", 4.0, 3.0),
            ("nearest", 4.0, 2.0),
            ("midpoint", 3.5, 2.5),
        ] {
            assert_close(
                &x.quantile(&[0.5], m).unwrap().to_vec::<f64>().unwrap(),
                &[half],
                1e-12,
                m,
            );
            assert_close(
                &x.quantile(&[0.3], m).unwrap().to_vec::<f64>().unwrap(),
                &[third],
                1e-12,
                m,
            );
        }
        // vector q prepends its dim
        let v = x.quantile_axis(0, &[0.25, 0.75], false, "linear").unwrap();
        assert_eq!(v.shape(), &[2]);
        assert_close(&v.to_vec::<f64>().unwrap(), &[1.75, 5.25], 1e-12, "vq");
        // midpoint at an exactly integral position reads the element (ceil rule)
        let f = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        assert_close(
            &f.quantile(&[0.5], "midpoint").unwrap().to_vec::<f64>().unwrap(),
            &[3.0],
            0.0,
            "midpoint exact",
        );
        // validation mirrors torch's RuntimeErrors
        let xi = Tensor::from_slice(&[1i64, 2, 3, 4], &[4]).unwrap();
        assert!(xi.quantile(&[0.5], "linear").is_err());
        assert!(x.quantile(&[1.5], "linear").is_err());
        assert!(x.quantile(&[f64::NAN], "linear").is_err());
        assert!(x.quantile(&[0.5], "foo").is_err());
        let empty = Tensor::from_slice(&[] as &[f64], &[0]).unwrap();
        assert!(empty.quantile(&[0.5], "linear").is_err());
    }

    #[test]
    fn nanquantile_skips_nan_and_quantile_propagates_first_nan() {
        // valid [1,3,4], q=0.5 -> pos 1.0 -> 3.0
        let x = Tensor::from_slice(&[1.0, f64::NAN, 3.0, 4.0], &[4]).unwrap();
        assert_close(
            &x.nanquantile(&[0.5], "linear")
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[3.0],
            1e-12,
            "nanquantile",
        );
        let a = Tensor::from_slice(&[f64::NAN, f64::NAN], &[2]).unwrap();
        assert!(
            a.nanquantile(&[0.5], "linear")
                .unwrap()
                .to_vec::<f64>()
                .unwrap()[0]
                .is_nan()
        );
        // quantile propagates; gradient addresses the first NaN
        assert!(
            x.quantile(&[0.5], "lower")
                .unwrap()
                .to_vec::<f64>()
                .unwrap()[0]
                .is_nan()
        );
        let xg = Tensor::from_slice(&[1.0, f64::NAN, 3.0, 4.0], &[4])
            .unwrap()
            .requires_grad_(true);
        xg.quantile(&[0.5], "linear")
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &xg.grad().unwrap().to_vec::<f64>().unwrap(),
            &[0.0, 1.0, 0.0, 0.0],
            0.0,
            "quantile nan grad",
        );
    }

    #[test]
    fn mode_counts_per_slice_with_smallest_tiebreak_and_first_index() {
        // ties resolve to the smallest value; indices address first occurrences
        let x = Tensor::from_slice(&[1.0, 3.0, 3.0, 1.0, 2.0], &[5]).unwrap();
        let (v, i) = x.mode_axis(-1, false).unwrap();
        assert_close(&v.to_vec::<f64>().unwrap(), &[1.0], 0.0, "mode");
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![0]);
        // NaN groups and sorts smallest
        let n = Tensor::from_slice(&[f64::NAN, 1.0, 2.0], &[3]).unwrap();
        let (v, i) = n.mode_axis(0, false).unwrap();
        assert!(v.to_vec::<f64>().unwrap()[0].is_nan());
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![0]);
        // i64 keys stay exact past 2^53
        let b = Tensor::from_slice(&[9_007_199_254_740_993i64, 9_007_199_254_740_993, 0], &[3])
            .unwrap();
        let (v, i) = b.mode_axis(0, true).unwrap();
        assert_eq!(v.shape(), &[1]);
        assert_eq!(v.to_vec::<i64>().unwrap(), vec![9_007_199_254_740_993]);
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![0]);
        // gradient flows to the reported index
        let g = Tensor::from_slice(&[1.0, 3.0, 3.0, 1.0, 2.0], &[5])
            .unwrap()
            .requires_grad_(true);
        g.mode_axis(0, false)
            .unwrap()
            .0
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &g.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 0.0, 0.0, 0.0, 0.0],
            0.0,
            "mode grad",
        );
    }

    #[test]
    fn masked_select_scatter_values_broadcast_and_grad() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let m = Tensor::from_slice(&[true, false, true, false, true, false], &[2, 3]).unwrap();
        assert_close(
            &x.masked_select(&m).unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 3.0, 5.0],
            0.0,
            "select",
        );
        // masks broadcast, like torch
        let mb = Tensor::from_slice(&[true, false, true], &[3]).unwrap();
        assert_close(
            &x.masked_select(&mb).unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 3.0, 4.0, 6.0],
            0.0,
            "select broadcast",
        );
        // non-boolean masks fail closed, like torch
        let mi = Tensor::from_slice(&[1i64, 0, 1, 0, 1, 0], &[2, 3]).unwrap();
        assert!(x.masked_select(&mi).is_err());
        // scatter writes in order, drops the tail, rejects short sources
        let s = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
        assert_close(
            &x.masked_scatter(&m, &s)
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[10.0, 2.0, 20.0, 4.0, 30.0, 6.0],
            0.0,
            "scatter",
        );
        let short = Tensor::from_slice(&[10.0, 20.0], &[2]).unwrap();
        assert!(x.masked_scatter(&m, &short).is_err());
        // gradients: select scatters back, scatter splits base/complement
        let xg = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap()
            .requires_grad_(true);
        xg.masked_select(&m).unwrap().sum().unwrap().backward().unwrap();
        assert_close(
            &xg.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            0.0,
            "select grad",
        );
        let xb = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap()
            .requires_grad_(true);
        let sb = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], &[4])
            .unwrap()
            .requires_grad_(true);
        xb.masked_scatter(&m, &sb).unwrap().sum().unwrap().backward().unwrap();
        assert_close(
            &xb.grad().unwrap().to_vec::<f64>().unwrap(),
            &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            0.0,
            "scatter base grad",
        );
        assert_close(
            &sb.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 1.0, 1.0, 0.0],
            0.0,
            "scatter src grad",
        );
    }

    #[test]
    fn index_fill_wraps_negatives_rejects_bad_index_and_flows_grad() {
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
        let idx = Tensor::from_slice(&[0i64, -1], &[2]).unwrap();
        assert_close(
            &x.index_fill(1, &idx, 9.0).unwrap().to_vec::<f64>().unwrap(),
            &[9.0, 1.0, 9.0, 9.0, 4.0, 9.0],
            0.0,
            "fill",
        );
        let oob = Tensor::from_slice(&[3i64], &[1]).unwrap();
        assert!(x.index_fill(1, &oob, 9.0).is_err());
        let bad = Tensor::from_slice(&[0i32], &[1]).unwrap();
        assert!(x.index_fill(1, &bad, 9.0).is_err());
        let mat = Tensor::from_slice(&[0i64], &[1, 1]).unwrap();
        assert!(x.index_fill(1, &mat, 9.0).is_err());
        let xg = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3])
            .unwrap()
            .requires_grad_(true);
        xg.index_fill(1, &idx, 9.0)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &xg.grad().unwrap().to_vec::<f64>().unwrap(),
            &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            0.0,
            "fill grad",
        );
    }

    #[test]
    fn gradient_uniform_and_coords_match_hand_computation() {
        let x = Tensor::from_slice(&[1.0, 2.0, 4.0, 7.0, 11.0], &[5]).unwrap();
        assert_close(
            &x.gradient_axis(0, 1, 1.0).unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 1.5, 2.5, 3.5, 4.0],
            1e-12,
            "grad e1",
        );
        assert_close(
            &x.gradient_axis(0, 2, 1.0).unwrap().to_vec::<f64>().unwrap(),
            &[0.5, 1.5, 2.5, 3.5, 4.5],
            1e-12,
            "grad e2",
        );
        // non-uniform coordinates: general second-order schemes
        let t = Tensor::from_slice(&[0.0, 1.0, 3.0, 6.0], &[4]).unwrap();
        let y = Tensor::from_slice(&[0.0, 1.0, 9.0, 36.0], &[4]).unwrap();
        assert_close(
            &y.gradient_axis_coords(0, 1, &t)
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[1.0, 2.0, 6.0, 9.0],
            1e-9,
            "grad coords e1",
        );
        assert_close(
            &y.gradient_axis_coords(0, 2, &t)
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[0.0, 2.0, 6.0, 12.0],
            1e-9,
            "grad coords e2",
        );
        assert!(x.gradient_axis(0, 3, 1.0).is_err());
        let one = Tensor::from_slice(&[5.0], &[1]).unwrap();
        assert!(one.gradient_axis(0, 1, 1.0).is_err());
    }

    #[test]
    fn trapezoid_uniform_and_coords_match_hand_computation() {
        let x = Tensor::from_slice(&[1.0, 2.0, 4.0, 7.0, 11.0], &[5]).unwrap();
        assert_close(
            &x.trapezoid_axis(0, 1.0).unwrap().to_vec::<f64>().unwrap(),
            &[19.0],
            1e-12,
            "trapz",
        );
        assert_close(
            &x.trapezoid_axis(0, 2.0).unwrap().to_vec::<f64>().unwrap(),
            &[38.0],
            1e-12,
            "trapz dx",
        );
        let t = Tensor::from_slice(&[0.0, 1.0, 3.0, 6.0, 10.0], &[5]).unwrap();
        assert_close(
            &x.trapezoid_axis_x(0, &t).unwrap().to_vec::<f64>().unwrap(),
            &[60.0],
            1e-9,
            "trapz x",
        );
        let one = Tensor::from_slice(&[5.0], &[1]).unwrap();
        assert_close(
            &one.trapezoid_axis(0, 1.0).unwrap().to_vec::<f64>().unwrap(),
            &[0.0],
            0.0,
            "trapz n=1",
        );
        let short = Tensor::from_slice(&[0.0, 1.0], &[2]).unwrap();
        assert!(x.trapezoid_axis_x(0, &short).is_err());
    }

    #[test]
    fn squeeze_axis_removes_one_and_rejects_other_sizes() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let got = x.squeeze_axis(0).unwrap();
        assert_eq!(got.shape(), &[3]);
        assert_close(
            &got.to_vec::<f64>().unwrap(),
            &[1.0, 2.0, 3.0],
            0.0,
            "squeeze",
        );
        // torch errors here; a silent no-op would hide shape bugs
        let y = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(y.squeeze_axis(0).is_err());
    }

    #[test]
    fn index_select_values_bounds_and_duplicate_grads() {
        // rows of [[0,1],[2,3],[4,5]] picked by [2,0,2]
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2]).unwrap();
        let idx = Tensor::from_slice(&[2i64, 0, 2], &[3]).unwrap();
        let got = x.index_select(0, &idx).unwrap();
        assert_eq!(got.shape(), &[3, 2]);
        assert_close(
            &got.to_vec::<f64>().unwrap(),
            &[4.0, 5.0, 0.0, 1.0, 4.0, 5.0],
            0.0,
            "rows",
        );
        // negative indices wrap
        let neg = Tensor::from_slice(&[-1i64], &[1]).unwrap();
        let gotn = x.index_select(0, &neg).unwrap().to_vec::<f64>().unwrap();
        assert_close(&gotn, &[4.0, 5.0], 0.0, "negative");
        // out of bounds (either side) errors
        for bad in [vec![3i64], vec![-4i64]] {
            let b = Tensor::from_slice(&bad, &[1]).unwrap();
            assert!(x.index_select(0, &b).is_err(), "{bad:?} OOB");
        }
        // empty selection is fine
        let e = Tensor::from_slice(&[] as &[i64], &[0]).unwrap();
        assert_eq!(x.index_select(0, &e).unwrap().shape(), &[0, 2]);
        // duplicate indices accumulate gradients
        let xg = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[3, 2])
            .unwrap()
            .requires_grad_(true);
        xg.index_select(0, &idx)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        let grad = xg.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[1.0, 1.0, 0.0, 0.0, 2.0, 2.0], 0.0, "dup grad");
        // index_add forward (also the engine behind the grad above)
        let base = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]).unwrap();
        let add_src = Tensor::from_slice(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[3, 2]).unwrap();
        let added = base
            .index_add(0, &idx, &add_src)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&added, &[2.0, 2.0, 0.0, 0.0, 4.0, 4.0], 0.0, "index_add");
        let wrong = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(base.index_add(0, &idx, &wrong).is_err());
    }

    #[test]
    fn gather_values_bounds_and_rank_checks() {
        // [[0,1,2],[3,4,5]] gather rows [2,0]/[1,1] along axis 1
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
        let idx = Tensor::from_slice(&[2i64, 0, 1, 1], &[2, 2]).unwrap();
        let got = x.gather(1, &idx).unwrap();
        assert_eq!(got.shape(), &[2, 2]);
        assert_close(
            &got.to_vec::<f64>().unwrap(),
            &[2.0, 0.0, 4.0, 4.0],
            0.0,
            "gather",
        );
        // negative indices wrap
        let neg = Tensor::from_slice(&[-1i64, 0, 0, -1], &[2, 2]).unwrap();
        let gotn = x.gather(1, &neg).unwrap().to_vec::<f64>().unwrap();
        assert_close(&gotn, &[2.0, 0.0, 3.0, 5.0], 0.0, "neg gather");
        // out of bounds errors; index rank must match input rank
        let oob = Tensor::from_slice(&[3i64], &[1, 1]).unwrap();
        assert!(x.narrow(0, 0, 1).unwrap().gather(1, &oob).is_err());
        let flat = Tensor::from_slice(&[0i64, 1], &[2]).unwrap();
        assert!(x.gather(1, &flat).is_err());
        // backward scatters ones back to gathered positions
        let xg = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3])
            .unwrap()
            .requires_grad_(true);
        xg.gather(1, &idx)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        let grad = xg.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&grad, &[1.0, 0.0, 1.0, 0.0, 2.0, 0.0], 0.0, "gather grad");
    }

    #[test]
    fn scatter_add_accumulates_duplicates_and_validates() {
        let z = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0], &[2, 2]).unwrap();
        let idx = Tensor::from_slice(&[1i64, 0, 1, 1], &[2, 2]).unwrap();
        let src = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        // row 0: dst[1]+=1, dst[0]+=2 -> [2,1]; row 1: dst[1]+=3+4 -> [0,7]
        let got = z
            .scatter_add(1, &idx, &src)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&got, &[2.0, 1.0, 0.0, 7.0], 0.0, "accumulate");
        // negative indices wrap
        let neg = Tensor::from_slice(&[-1i64], &[1, 1]).unwrap();
        let one = Tensor::from_slice(&[5.0], &[1, 1]).unwrap();
        let gotn = z
            .narrow(0, 0, 1)
            .unwrap()
            .scatter_add(1, &neg, &one)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&gotn, &[0.0, 5.0], 0.0, "neg scatter");
        // OOB index and src/index shape mismatch error
        let oob = Tensor::from_slice(&[2i64], &[1, 1]).unwrap();
        assert!(
            z.narrow(0, 0, 1)
                .unwrap()
                .scatter_add(1, &oob, &one)
                .is_err()
        );
        let wrong = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        assert!(
            z.narrow(0, 0, 1)
                .unwrap()
                .scatter_add(1, &neg, &wrong)
                .is_err()
        );
        // backward: grad flows to base unchanged, to src via gather
        let zb = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        let sb = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        zb.scatter_add(1, &idx, &sb)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        let ga = zb.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&ga, &[1.0, 1.0, 1.0, 1.0], 0.0, "base grad");
        let gb = sb.grad().unwrap().to_vec::<f64>().unwrap();
        assert_close(&gb, &[1.0, 1.0, 1.0, 1.0], 0.0, "src grad");
    }

    #[test]
    fn narrow_select_slice_flip_roll_behave() {
        let x = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]).unwrap();
        let n = x.narrow(1, 1, 2).unwrap().to_vec::<f64>().unwrap();
        assert_close(&n, &[1.0, 2.0, 4.0, 5.0], 0.0, "narrow");
        // overrun and past-the-end starts error; empty-at-end is fine
        assert!(x.narrow(1, 2, 2).is_err());
        assert!(x.narrow(1, 4, 0).is_err());
        assert_eq!(x.narrow(1, 3, 0).unwrap().shape(), &[2, 0]);
        let s = x.select(1, -1).unwrap().to_vec::<f64>().unwrap();
        assert_close(&s, &[2.0, 5.0], 0.0, "select");
        assert!(x.select(1, 3).is_err());
        assert!(x.select(1, -4).is_err());
        let f = x.flip(&[0]).unwrap().to_vec::<f64>().unwrap();
        assert_close(&f, &[3.0, 4.0, 5.0, 0.0, 1.0, 2.0], 0.0, "flip");
        assert!(x.flip(&[0, 0]).is_err());
        let r = x.roll(1, 1).unwrap().to_vec::<f64>().unwrap();
        assert_close(&r, &[2.0, 0.0, 1.0, 5.0, 3.0, 4.0], 0.0, "roll");
        let rn = x.roll(-4, 1).unwrap().to_vec::<f64>().unwrap();
        assert_close(&rn, &[1.0, 2.0, 0.0, 4.0, 5.0, 3.0], 0.0, "roll neg");
    }

    #[test]
    fn topk_unsorted_returns_the_winning_set() {
        let x = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], &[8]).unwrap();
        for largest in [true, false] {
            let (values, indices) = x.topk(3, 0, largest, false).unwrap();
            let mut v = values.to_vec::<f64>().unwrap();
            let idx = indices.to_vec::<i64>().unwrap();
            // indices point at the returned values
            let src = x.to_vec::<f64>().unwrap();
            for (j, &one) in v.iter().enumerate() {
                assert_eq!(one, src[idx[j] as usize], "index consistency");
            }
            v.sort_by(|a, b| a.total_cmp(b));
            let want = if largest {
                vec![5.0, 6.0, 9.0]
            } else {
                vec![1.0, 1.0, 2.0]
            };
            assert_eq!(v, want, "winning set largest={largest}");
        }
        // k = 0 selects nothing; k = dim selects everything
        let (v0, i0) = x.topk(0, 0, true, false).unwrap();
        assert_eq!(v0.numel(), 0);
        assert_eq!(i0.numel(), 0);
        let (vd, _) = x.topk(8, 0, true, false).unwrap();
        let mut all = vd.to_vec::<f64>().unwrap();
        all.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(all, vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]);
    }

    #[test]
    fn maximum_minimum_nan_propagates_and_tie_grads_split() {
        // torch.maximum/minimum propagate NaN from either side
        for (x, y) in [(f64::NAN, 1.0), (1.0, f64::NAN)] {
            let a = Tensor::from_slice(&[x], &[1]).unwrap();
            let b = Tensor::from_slice(&[y], &[1]).unwrap();
            assert!(a.maximum(&b).unwrap().item::<f64>().unwrap().is_nan());
            assert!(a.minimum(&b).unwrap().item::<f64>().unwrap().is_nan());
        }
        // ties split the gradient 0.5 / 0.5 (torch-verified)
        for is_max in [true, false] {
            let a = Tensor::from_slice(&[1.0], &[1])
                .unwrap()
                .requires_grad_(true);
            let b = Tensor::from_slice(&[1.0], &[1])
                .unwrap()
                .requires_grad_(true);
            let y = if is_max {
                a.maximum(&b).unwrap()
            } else {
                a.minimum(&b).unwrap()
            };
            y.sum().unwrap().backward().unwrap();
            assert_eq!(a.grad().unwrap().item::<f64>().unwrap(), 0.5);
            assert_eq!(b.grad().unwrap().item::<f64>().unwrap(), 0.5);
        }
        // strict winner still takes all the gradient
        let a = Tensor::from_slice(&[2.0], &[1])
            .unwrap()
            .requires_grad_(true);
        let b = Tensor::from_slice(&[1.0], &[1])
            .unwrap()
            .requires_grad_(true);
        a.maximum(&b).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(a.grad().unwrap().item::<f64>().unwrap(), 1.0);
        assert_eq!(b.grad().unwrap().item::<f64>().unwrap(), 0.0);
        let a = Tensor::from_slice(&[2.0], &[1])
            .unwrap()
            .requires_grad_(true);
        let b = Tensor::from_slice(&[1.0], &[1])
            .unwrap()
            .requires_grad_(true);
        a.minimum(&b).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(a.grad().unwrap().item::<f64>().unwrap(), 0.0);
        assert_eq!(b.grad().unwrap().item::<f64>().unwrap(), 1.0);
        // NaN inputs route full gradient to both sides (torch-verified)
        let a = Tensor::from_slice(&[f64::NAN], &[1])
            .unwrap()
            .requires_grad_(true);
        let b = Tensor::from_slice(&[1.0], &[1])
            .unwrap()
            .requires_grad_(true);
        a.maximum(&b).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(a.grad().unwrap().item::<f64>().unwrap(), 1.0);
        assert_eq!(b.grad().unwrap().item::<f64>().unwrap(), 1.0);
    }

    #[test]
    fn clamp_boundary_grads_are_zero() {
        // torch.clamp (checked against torch 2.13): the gradient passes on [min, max], bounds included
        for (x0, want) in [(1.0, 1.0), (2.0, 1.0), (1.5, 1.0), (0.0, 0.0), (3.0, 0.0)] {
            let x = Tensor::from_slice(&[x0], &[1])
                .unwrap()
                .requires_grad_(true);
            x.clamp(1.0, 2.0)
                .unwrap()
                .sum()
                .unwrap()
                .backward()
                .unwrap();
            assert_eq!(x.grad().unwrap().item::<f64>().unwrap(), want, "x={x0}");
        }
        let x = Tensor::from_slice(&[0.0, 1.5, 3.0], &[3]).unwrap();
        let got = x.clamp(1.0, 2.0).unwrap().to_vec::<f64>().unwrap();
        assert_close(&got, &[1.0, 1.5, 2.0], 0.0, "clamp fwd");
    }

    #[test]
    fn pow_zero_base_matches_torch() {
        // (exponent, fwd, base-grad, exp-grad), all torch-verified
        for (e, fwd, gb, ge) in [
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, f64::INFINITY, 0.0),
            (-1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        ] {
            let base = Tensor::from_slice(&[0.0], &[1])
                .unwrap()
                .requires_grad_(true);
            let exp = Tensor::from_slice(&[e], &[1]).unwrap();
            let y = base.pow(&exp).unwrap();
            assert_eq!(y.item::<f64>().unwrap(), fwd, "pow(0,{e}) fwd");
            y.sum().unwrap().backward().unwrap();
            assert_eq!(
                base.grad().unwrap().item::<f64>().unwrap(),
                gb,
                "pow(0,{e}) gbase"
            );
            let base = Tensor::from_slice(&[0.0], &[1]).unwrap();
            let exp = Tensor::from_slice(&[e], &[1]).unwrap().requires_grad_(true);
            base.pow(&exp).unwrap().sum().unwrap().backward().unwrap();
            assert_eq!(
                exp.grad().unwrap().item::<f64>().unwrap(),
                ge,
                "pow(0,{e}) gexp"
            );
        }
        // scalar-exponent twin: 0^0 = 1 with zero gradient (torch-verified)
        let x = Tensor::from_slice(&[0.0], &[1])
            .unwrap()
            .requires_grad_(true);
        let y = x.powf(0.0).unwrap();
        assert_eq!(y.item::<f64>().unwrap(), 1.0);
        y.sum().unwrap().backward().unwrap();
        assert_eq!(x.grad().unwrap().item::<f64>().unwrap(), 0.0);
        // ordinary point still exact: d(x^2)/dx at 3 = 6
        let x = Tensor::from_slice(&[3.0], &[1])
            .unwrap()
            .requires_grad_(true);
        x.powf(2.0).unwrap().sum().unwrap().backward().unwrap();
        assert_eq!(x.grad().unwrap().item::<f64>().unwrap(), 6.0);
    }

    #[test]
    fn singular_unary_grads_match_torch() {
        // (op, fwd(0), grad(0)), all torch-verified
        for (name, make) in [
            ("sqrt", Tensor::sqrt as fn(&Tensor) -> Result<Tensor>),
            ("ln", Tensor::ln as fn(&Tensor) -> Result<Tensor>),
            ("recip", Tensor::recip as fn(&Tensor) -> Result<Tensor>),
            ("rsqrt", Tensor::rsqrt as fn(&Tensor) -> Result<Tensor>),
        ] {
            let x = Tensor::from_slice(&[0.0], &[1])
                .unwrap()
                .requires_grad_(true);
            let y = make(&x).unwrap();
            let (want_fwd, want_grad) = match name {
                "sqrt" => (0.0, f64::INFINITY),
                "ln" => (f64::NEG_INFINITY, f64::INFINITY),
                "recip" => (f64::INFINITY, f64::NEG_INFINITY),
                _ => (f64::INFINITY, f64::NEG_INFINITY),
            };
            assert_eq!(y.item::<f64>().unwrap(), want_fwd, "{name} fwd");
            y.sum().unwrap().backward().unwrap();
            assert_eq!(
                x.grad().unwrap().item::<f64>().unwrap(),
                want_grad,
                "{name} grad"
            );
        }
    }

    #[test]
    fn div_by_zero_matches_torch() {
        // torch: 1/0 = inf, d/da = inf, d/db = -inf
        let a = Tensor::from_slice(&[1.0], &[1])
            .unwrap()
            .requires_grad_(true);
        let b = Tensor::from_slice(&[0.0], &[1])
            .unwrap()
            .requires_grad_(true);
        let y = Tensor::div(&a, &b).unwrap();
        assert_eq!(y.item::<f64>().unwrap(), f64::INFINITY);
        y.sum().unwrap().backward().unwrap();
        assert_eq!(a.grad().unwrap().item::<f64>().unwrap(), f64::INFINITY);
        assert_eq!(b.grad().unwrap().item::<f64>().unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn sign_nan_is_zero() {
        // torch.sign(nan) = 0, and the CPU/GPU kernels already agree
        let x = Tensor::from_slice(&[f64::NAN, -2.0, 0.0, 3.0], &[4]).unwrap();
        let got = x.sign().unwrap().to_vec::<f64>().unwrap();
        assert_close(&got, &[0.0, -1.0, 0.0, 1.0], 0.0, "sign");
    }

    fn arange_f64(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    #[test]
    fn conv2d_matches_torch_across_configs() {
        // multi-channel + bias, stride 1 (torch-verified)
        let x = Tensor::from_slice(&arange_f64(18), &[1, 2, 3, 3]).unwrap();
        let w = Tensor::from_slice(&arange_f64(16), &[2, 2, 2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0, -1.0], &[2]).unwrap();
        let y = x
            .conv2d(&w, Some(&b), (1, 1), (0, 0), (1, 1))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[269.0, 297.0, 353.0, 381.0, 683.0, 775.0, 959.0, 1051.0],
            1e-9,
            "c1",
        );
        // stride + padding + dilation (torch-verified)
        let x = Tensor::from_slice(&arange_f64(49), &[1, 1, 7, 7]).unwrap();
        let w = Tensor::from_slice(
            &(1..10).map(|i| i as f64).collect::<Vec<_>>(),
            &[1, 1, 3, 3],
        )
        .unwrap();
        let y = x
            .conv2d(&w, None, (2, 2), (1, 1), (2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                492.0, 734.0, 476.0, 930.0, 1344.0, 846.0, 524.0, 722.0, 428.0,
            ],
            1e-9,
            "c2",
        );
        // conv1d via the unsqueeze path (torch-verified)
        let x = Tensor::from_slice(&arange_f64(14), &[1, 2, 7]).unwrap();
        let w =
            Tensor::from_slice(&(1..13).map(|i| i as f64).collect::<Vec<_>>(), &[2, 2, 3]).unwrap();
        let b = Tensor::from_slice(&[0.5, -0.5], &[2]).unwrap();
        let y = x
            .conv1d(&w, Some(&b), 2, 1, 1)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[86.5, 151.5, 193.5, 130.5, 181.5, 348.5, 462.5, 345.5],
            1e-9,
            "c1d",
        );
    }

    #[test]
    fn conv2d_grouped_matches_torch_and_validates() {
        // groups = 2 (torch-verified)
        let x = Tensor::from_slice(&arange_f64(36), &[1, 4, 3, 3]).unwrap();
        let w = Tensor::from_slice(&arange_f64(32), &[4, 2, 2, 2]).unwrap();
        let y = x
            .conv2d_grouped(&w, None, (1, 1), (0, 0), (1, 1), 2)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                268.0, 296.0, 352.0, 380.0, 684.0, 776.0, 960.0, 1052.0, 3908.0, 4064.0, 4376.0,
                4532.0, 5476.0, 5696.0, 6136.0, 6356.0,
            ],
            1e-9,
            "grouped",
        );
        // depthwise (torch-verified)
        let x = Tensor::from_slice(&arange_f64(27), &[1, 3, 3, 3]).unwrap();
        let w = Tensor::from_slice(&arange_f64(12), &[3, 1, 2, 2]).unwrap();
        let y = x
            .conv2d_grouped(&w, None, (1, 1), (0, 0), (1, 1), 3)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                19.0, 25.0, 37.0, 43.0, 249.0, 271.0, 315.0, 337.0, 767.0, 805.0, 881.0, 919.0,
            ],
            1e-9,
            "depthwise",
        );
        // validation: zero groups, non-divisible channels, full-Cin weight under groups
        assert!(
            x.conv2d_grouped(&w, None, (1, 1), (0, 0), (1, 1), 0)
                .is_err()
        );
        let x4 = Tensor::from_slice(&arange_f64(36), &[1, 4, 3, 3]).unwrap();
        assert!(
            x4.conv2d_grouped(&w, None, (1, 1), (0, 0), (1, 1), 3)
                .is_err()
        );
        let w_full = Tensor::from_slice(&arange_f64(64), &[4, 4, 2, 2]).unwrap();
        assert!(
            x4.conv2d_grouped(&w_full, None, (1, 1), (0, 0), (1, 1), 2)
                .is_err()
        );
    }

    #[test]
    fn conv_transpose2d_matches_torch() {
        let x5 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let w5 = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]).unwrap();
        // stride 1 (torch-verified)
        let y = x5
            .conv_transpose2d(&w5, None, (1, 1), (0, 0), (0, 0), (1, 1))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[1.0, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0],
            1e-9,
            "t1",
        );
        // output_padding appends zero rows/cols (torch-verified)
        let y = x5
            .conv_transpose2d(&w5, None, (2, 2), (0, 0), (1, 1), (1, 1))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            1e-9,
            "t2",
        );
        // stride + padding (torch-verified)
        let x7 = Tensor::from_slice(&arange_f64(9), &[1, 1, 3, 3]).unwrap();
        let w7 = Tensor::from_slice(&arange_f64(9), &[1, 1, 3, 3]).unwrap();
        let y = x7
            .conv_transpose2d(&w7, None, (2, 2), (1, 1), (0, 0), (1, 1))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                0.0, 3.0, 4.0, 11.0, 8.0, 3.0, 12.0, 11.0, 28.0, 19.0, 12.0, 27.0, 16.0, 35.0,
                20.0, 27.0, 60.0, 35.0, 76.0, 43.0, 24.0, 51.0, 28.0, 59.0, 32.0,
            ],
            1e-9,
            "t3",
        );
        // output_padding with stride 1 needs dilation headroom in torch (torch-verified)
        let y = x5
            .conv_transpose2d(&w5, None, (1, 1), (0, 0), (1, 1), (2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0,
                0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            1e-9,
            "t4",
        );
        let y = x5
            .conv_transpose2d(&w5, None, (2, 2), (0, 0), (1, 1), (2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 5.0, 0.0,
                2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
            1e-9,
            "t5",
        );
        // torch: output_padding must be smaller than stride or dilation (per dim)
        let err = x5
            .conv_transpose2d(&w5, None, (2, 2), (0, 0), (2, 2), (1, 1))
            .unwrap_err();
        assert!(format!("{err:?}").contains("output_padding"), "{err:?}");
        assert!(
            x5.conv_transpose2d(&w5, None, (0, 1), (0, 0), (0, 0), (1, 1))
                .is_err()
        );
        assert!(
            x5.conv_transpose2d(&w5, None, (1, 1), (0, 0), (0, 0), (0, 1))
                .is_err()
        );
    }

    #[test]
    fn unfold2d_matches_torch_and_fold_roundtrips() {
        // im2col column layout equals torch.nn.functional.unfold (torch-verified)
        let x = Tensor::from_slice(&arange_f64(9), &[1, 1, 3, 3]).unwrap();
        let y = x.unfold2d((2, 2), (1, 1), (0, 0), (1, 1)).unwrap();
        assert_eq!(y.shape(), &[1, 4, 4]);
        assert_close(
            &y.to_vec::<f64>().unwrap(),
            &[
                0.0, 1.0, 3.0, 4.0, 1.0, 2.0, 4.0, 5.0, 3.0, 4.0, 6.0, 7.0, 4.0, 5.0, 7.0, 8.0,
            ],
            1e-9,
            "unfold",
        );
        // 1x1 fold∘unfold is the identity; overlapping folds accumulate
        let x = Tensor::from_slice(&arange_f64(6), &[1, 2, 1, 3]).unwrap();
        let rt = x
            .unfold2d((1, 1), (1, 1), (0, 0), (1, 1))
            .unwrap()
            .fold2d((1, 3), (1, 1), (1, 1), (0, 0), (1, 1))
            .unwrap();
        assert_close(
            &rt.to_vec::<f64>().unwrap(),
            &arange_f64(6),
            1e-9,
            "roundtrip",
        );
        let ones = Tensor::from_slice(&[1.0f64; 16], &[1, 4, 4]).unwrap();
        let acc = ones
            .fold2d((3, 3), (2, 2), (1, 1), (0, 0), (1, 1))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &acc,
            &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            1e-9,
            "overlap",
        );
    }

    #[test]
    fn max_pool2d_excludes_padding_and_validates() {
        // padding must not win over all-negative windows (torch-verified: -5s)
        let x = Tensor::from_slice(&[-5.0f64; 9], &[1, 1, 3, 3]).unwrap();
        let y = x
            .max_pool2d((2, 2), (2, 2), (1, 1))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[-5.0, -5.0, -5.0, -5.0], 1e-9, "negpad");
        // integer inputs exclude padding too (fill saturates to dtype minimum)
        let xi = Tensor::from_slice(&[-5i64; 9], &[1, 1, 3, 3]).unwrap();
        let yi = xi
            .max_pool2d((2, 2), (2, 2), (1, 1))
            .unwrap()
            .to_vec::<i64>()
            .unwrap();
        assert_eq!(yi, vec![-5, -5, -5, -5]);
        // torch rejects padding beyond half the kernel (both poolings)
        let z = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        assert!(z.max_pool2d((1, 1), (1, 1), (2, 2)).is_err());
        assert!(z.avg_pool2d((1, 1), (1, 1), (2, 2)).is_err());
        // ...while pad == kernel/2 stays legal
        assert!(z.max_pool2d((2, 2), (1, 1), (1, 1)).is_ok());
        assert!(z.avg_pool2d((2, 2), (1, 1), (1, 1)).is_ok());
    }

    #[test]
    fn avg_pool2d_matches_torch_padding() {
        // count_include_pad=True semantics (torch-verified)
        let x = Tensor::from_slice(&[1.0f64; 9], &[1, 1, 3, 3]).unwrap();
        let y = x
            .avg_pool2d((2, 2), (1, 1), (1, 1))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                0.25, 0.5, 0.5, 0.25, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.25, 0.5, 0.5, 0.25,
            ],
            1e-9,
            "avgpad",
        );
    }

    #[test]
    fn upsample_nearest2d_matches_torch_indices() {
        // floor(o*h/oh) index convention (all torch-verified)
        for (h, oh, want) in [
            (2, 3, vec![0.0, 0.0, 1.0]),
            (3, 2, vec![0.0, 1.0]),
            (2, 4, vec![0.0, 0.0, 1.0, 1.0]),
            (4, 2, vec![0.0, 2.0]),
            (3, 5, vec![0.0, 0.0, 1.0, 1.0, 2.0]),
        ] {
            let x =
                Tensor::from_slice(&(0..h).map(|i| i as f64).collect::<Vec<_>>(), &[1, 1, h, 1])
                    .unwrap();
            let y = x
                .upsample_nearest2d((oh, 1))
                .unwrap()
                .to_vec::<f64>()
                .unwrap();
            assert_close(&y, &want, 0.0, &format!("nearest {h}->{oh}"));
        }
    }

    #[test]
    fn upsample_bilinear2d_matches_torch() {
        // both align_corners modes (torch-verified)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let y = x
            .upsample_bilinear2d((4, 4), false)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                1.0, 1.25, 1.75, 2.0, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3.0, 3.25, 3.75,
                4.0,
            ],
            1e-9,
            "nocorner",
        );
        let y = x
            .upsample_bilinear2d((4, 4), true)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[
                1.0,
                1.3333333333333335,
                1.6666666666666665,
                2.0,
                1.6666666666666667,
                2.0,
                2.333333333333333,
                2.666666666666667,
                2.333333333333333,
                2.666666666666667,
                3.0,
                3.333333333333333,
                3.0,
                3.3333333333333335,
                3.6666666666666665,
                4.0,
            ],
            1e-9,
            "corner",
        );
    }

    #[test]
    fn adaptive_pool_matches_torch() {
        // even and uneven (overlapping) windows, avg and max (torch-verified)
        let x = Tensor::from_slice(&arange_f64(16), &[1, 1, 4, 4]).unwrap();
        let y = x
            .adaptive_avg_pool2d((2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[2.5, 4.5, 10.5, 12.5], 1e-9, "adapt44");
        let x3 = Tensor::from_slice(&arange_f64(9), &[1, 1, 3, 3]).unwrap();
        let y = x3
            .adaptive_avg_pool2d((2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[2.0, 3.0, 5.0, 6.0], 1e-9, "adapt32");
        let y = x3
            .adaptive_max_pool2d((2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[4.0, 5.0, 7.0, 8.0], 1e-9, "adaptmax32");
    }

    #[test]
    fn lp_pool_matches_hand_values() {
        // (3^2 + 4^2)^(1/2) = 5; p=inf reduces to max|x|
        let x = Tensor::from_slice(&[3.0, 4.0], &[1, 1, 2]).unwrap();
        let y = x.lp_pool1d(2.0, 2, 2).unwrap().to_vec::<f64>().unwrap();
        assert_close(&y, &[5.0], 1e-9, "lp1d");
        let x = Tensor::from_slice(&[3.0, -4.0, 1.0, 2.0], &[1, 1, 2, 2]).unwrap();
        let y = x
            .lp_pool2d(2.0, (2, 2), (2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[(9.0 + 16.0 + 1.0 + 4.0f64).sqrt()], 1e-9, "lp2d");
        let y = x
            .lp_pool2d(f64::INFINITY, (2, 2), (2, 2))
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[4.0], 1e-9, "lpinf");
    }

    #[test]
    fn matmul_1d_vectors_match_torch() {
        // vector-vector contracts to a scalar (torch-verified: 3.0)
        let a = Tensor::from_slice(&[1.0, 1.0, 1.0], &[3]).unwrap();
        let b = Tensor::from_slice(&[1.0, 1.0, 1.0], &[3]).unwrap();
        assert_eq!(a.matmul(&b).unwrap().item::<f64>().unwrap(), 3.0);
        // matrix-vector and vector-matrix (torch-verified)
        let m = Tensor::from_slice(&[1.0f64; 6], &[2, 3]).unwrap();
        let v = Tensor::from_slice(&[1.0, 1.0, 1.0], &[3]).unwrap();
        assert_close(
            &m.matmul(&v).unwrap().to_vec::<f64>().unwrap(),
            &[3.0, 3.0],
            0.0,
            "mv",
        );
        let w = Tensor::from_slice(&[1.0f64; 6], &[3, 2]).unwrap();
        assert_close(
            &v.matmul(&w).unwrap().to_vec::<f64>().unwrap(),
            &[3.0, 3.0],
            0.0,
            "vm",
        );
        // batched partners broadcast against the vector side
        let mb = Tensor::from_slice(&[1.0f64; 12], &[2, 2, 3]).unwrap();
        assert_close(
            &mb.matmul(&v).unwrap().to_vec::<f64>().unwrap(),
            &[3.0; 4],
            0.0,
            "bmv",
        );
        let wb = Tensor::from_slice(&[1.0f64; 12], &[2, 3, 2]).unwrap();
        assert_close(
            &v.matmul(&wb).unwrap().to_vec::<f64>().unwrap(),
            &[3.0; 4],
            0.0,
            "vbm",
        );
        // gradients flow through the vector views
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        let y = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3])
            .unwrap()
            .requires_grad_(true);
        x.matmul(&y).unwrap().backward().unwrap();
        assert_close(
            &x.grad().unwrap().to_vec::<f64>().unwrap(),
            &[4.0, 5.0, 6.0],
            0.0,
            "vv-ga",
        );
        assert_close(
            &y.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 2.0, 3.0],
            0.0,
            "vv-gb",
        );
        let mo = Tensor::from_slice(&[1.0f64; 6], &[2, 3]).unwrap();
        let vo = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])
            .unwrap()
            .requires_grad_(true);
        mo.matmul(&vo).unwrap().sum().unwrap().backward().unwrap();
        assert_close(
            &vo.grad().unwrap().to_vec::<f64>().unwrap(),
            &[2.0, 2.0, 2.0],
            0.0,
            "mv-g",
        );
    }

    #[test]
    fn matmul_broadcast_and_empty_edges() {
        // unbatched partners broadcast across batch dims (torch parity)
        let a = Tensor::from_slice(&[1.0f64; 24], &[2, 3, 4]).unwrap();
        let b = Tensor::from_slice(&[1.0f64; 20], &[4, 5]).unwrap();
        let y = a.matmul(&b).unwrap();
        assert_eq!(y.shape(), &[2, 3, 5]);
        assert_close(&y.to_vec::<f64>().unwrap(), &[4.0; 30], 0.0, "bcast-r");
        let c = Tensor::from_slice(&[1.0f64; 20], &[4, 5]).unwrap();
        let d = Tensor::from_slice(&[1.0f64; 60], &[2, 5, 6]).unwrap();
        let y = c.matmul(&d).unwrap();
        assert_eq!(y.shape(), &[2, 4, 6]);
        assert_close(&y.to_vec::<f64>().unwrap(), &[5.0; 48], 0.0, "bcast-l");
        // empty contractions are zeros, empty batches stay empty
        let e1 = Tensor::from_slice(&[] as &[f64], &[2, 0]).unwrap();
        let e2 = Tensor::from_slice(&[] as &[f64], &[0, 3]).unwrap();
        let y = e1.matmul(&e2).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
        assert_close(&y.to_vec::<f64>().unwrap(), &[0.0; 6], 0.0, "empty-k");
        let e3 = Tensor::from_slice(&[] as &[f64], &[0, 3]).unwrap();
        let e4 = Tensor::from_slice(&[1.0f64; 6], &[3, 2]).unwrap();
        assert_eq!(e3.matmul(&e4).unwrap().shape(), &[0, 2]);
    }

    #[test]
    fn einsum_values_match_torch() {
        // batched matmul, trace, diagonal, transpose, outer (hand-exact, f64)
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]).unwrap();
        let y = Tensor::einsum("bik,bkj->bij", &[&a, &b])
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[19.0, 22.0, 43.0, 50.0], 0.0, "bmm");
        let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(
            Tensor::einsum("ii->", &[&m])
                .unwrap()
                .item::<f64>()
                .unwrap(),
            5.0
        );
        assert_close(
            &Tensor::einsum("ii->i", &[&m])
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[1.0, 4.0],
            0.0,
            "diag",
        );
        assert_close(
            &Tensor::einsum("ij->ji", &[&m])
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[1.0, 3.0, 2.0, 4.0],
            0.0,
            "trans",
        );
        let u = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
        let v = Tensor::from_slice(&[3.0, 4.0], &[2]).unwrap();
        assert_close(
            &Tensor::einsum("a,b->ab", &[&u, &v])
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[3.0, 4.0, 6.0, 8.0],
            0.0,
            "outer",
        );
        // 3-operand forward (torch-verified)
        let x = Tensor::from_slice(&arange_f64(6), &[2, 3]).unwrap();
        let w = Tensor::from_slice(&arange_f64(12), &[3, 4]).unwrap();
        let z = Tensor::from_slice(&arange_f64(8), &[2, 4]).unwrap();
        let y = Tensor::einsum("bn,nm,bm->b", &[&x, &w, &z])
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(&y, &[162.0, 1688.0], 1e-9, "3op");
        // implicit outputs: 'ij' is the identity, 'ii' is the trace (torch parity)
        assert_close(
            &Tensor::einsum("ij", &[&m])
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
            0.0,
            "impl-id",
        );
        assert_eq!(
            Tensor::einsum("ii", &[&m]).unwrap().item::<f64>().unwrap(),
            5.0
        );
    }

    #[test]
    fn einsum_diagonal_pair_backward_matches_torch() {
        // 'ii,ij->ij': the diagonal operand gets gradient on the diagonal only
        // (torch-verified values and grads).
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        let y = Tensor::einsum("ii,ij->ij", &[&a, &b]).unwrap();
        assert_close(
            &y.to_vec::<f64>().unwrap(),
            &[5.0, 6.0, 28.0, 32.0],
            0.0,
            "fwd",
        );
        y.sum().unwrap().backward().unwrap();
        assert_close(
            &a.grad().unwrap().to_vec::<f64>().unwrap(),
            &[11.0, 0.0, 0.0, 15.0],
            1e-9,
            "ga",
        );
        assert_close(
            &b.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 1.0, 4.0, 4.0],
            1e-9,
            "gb",
        );
    }

    #[test]
    fn einsum_validation_and_3op_backward_limit() {
        let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(Tensor::einsum("ij,jk->ik", &[&m]).is_err());
        assert!(Tensor::einsum("ij->ik", &[&m]).is_err());
        assert!(Tensor::einsum("ij->ii", &[&m]).is_err());
        let v = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(Tensor::einsum("ij,jk->ik", &[&m, &v]).is_err());
        // 3+ operand backward is an explicit error (forward works)
        let x = Tensor::from_slice(&arange_f64(6), &[2, 3])
            .unwrap()
            .requires_grad_(true);
        let w = Tensor::from_slice(&arange_f64(12), &[3, 4])
            .unwrap()
            .requires_grad_(true);
        let z = Tensor::from_slice(&arange_f64(8), &[2, 4])
            .unwrap()
            .requires_grad_(true);
        let y = Tensor::einsum("bn,nm,bm->b", &[&x, &w, &z]).unwrap();
        assert!(y.sum().unwrap().backward().is_err());
    }

    #[test]
    fn split_chunk_unbind_empty_dims() {
        // torch: split of an empty dim yields ONE empty view ...
        let e = Tensor::from_slice(&[] as &[f64], &[0]).unwrap();
        let parts = e.split(2, 0).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].shape(), &[0]);
        // ... chunk yields `chunks` empty views ...
        let parts = e.chunk(3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert!(parts.iter().all(|t| t.shape() == [0]));
        // ... and unbind yields none at all.
        let e2 = Tensor::from_slice(&[] as &[f64], &[0, 3]).unwrap();
        assert!(e2.unbind(0).unwrap().is_empty());
        // uneven tails: split(7 by 3) -> [3,3,1], chunk(5 by 3) -> [2,2,1]
        let s = Tensor::from_slice(&arange_f64(7), &[7]).unwrap();
        let lens: Vec<usize> = s
            .split(3, 0)
            .unwrap()
            .iter()
            .map(|t| t.shape()[0])
            .collect();
        assert_eq!(lens, vec![3, 3, 1]);
        let c = Tensor::from_slice(&arange_f64(5), &[5]).unwrap();
        let lens: Vec<usize> = c
            .chunk(3, 0)
            .unwrap()
            .iter()
            .map(|t| t.shape()[0])
            .collect();
        assert_eq!(lens, vec![2, 2, 1]);
        assert!(s.split(0, 0).is_err());
        assert!(s.chunk(0, 0).is_err());
    }

    #[test]
    fn pad_modes_dtypes_and_backward() {
        // constant pad values incl. nonzero (hand-exact)
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let y = x
            .pad(&[(1, 0), (0, 1)], 9.0)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        assert_close(
            &y,
            &[9.0, 9.0, 9.0, 1.0, 2.0, 9.0, 3.0, 4.0, 9.0],
            0.0,
            "pad",
        );
        // bool pads with cast constants (torch-verified)
        let b = Tensor::from_slice(&[true, false], &[2]).unwrap();
        assert_eq!(
            b.pad(&[(1, 1)], 1.0).unwrap().to_vec::<bool>().unwrap(),
            vec![true, true, false, true]
        );
        assert_eq!(
            b.pad(&[(1, 1)], 0.0).unwrap().to_vec::<bool>().unwrap(),
            vec![false, true, false, false]
        );
        // backward reaches the data region only
        let xg = Tensor::from_slice(&[1.0, 2.0], &[2])
            .unwrap()
            .requires_grad_(true);
        xg.pad(&[(2, 1)], 5.0)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &xg.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 1.0],
            0.0,
            "pad-g",
        );
        assert!(x.pad(&[(1, 1)], 0.0).is_err());
    }

    #[test]
    fn diagonal_offsets_and_backward() {
        // main/super/sub diagonals (torch-verified)
        let m = Tensor::from_slice(&arange_f64(12), &[3, 4]).unwrap();
        assert_close(
            &m.diagonal(0, 0, 1).unwrap().to_vec::<f64>().unwrap(),
            &[0.0, 5.0, 10.0],
            0.0,
            "d0",
        );
        assert_close(
            &m.diagonal(1, 0, 1).unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 6.0, 11.0],
            0.0,
            "d1",
        );
        assert_close(
            &m.diagonal(-1, 0, 1).unwrap().to_vec::<f64>().unwrap(),
            &[4.0, 9.0],
            0.0,
            "dm1",
        );
        // gradient scatters back onto the diagonal only
        let mg = Tensor::from_slice(&arange_f64(12), &[3, 4])
            .unwrap()
            .requires_grad_(true);
        mg.diagonal(0, 0, 1)
            .unwrap()
            .sum()
            .unwrap()
            .backward()
            .unwrap();
        assert_close(
            &mg.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            0.0,
            "diag-g",
        );
    }

    #[test]
    fn repeat_interleave_and_diag_embed() {
        let v = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
        assert_close(
            &v.repeat_interleave(3, 0).unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            0.0,
            "ri",
        );
        assert_eq!(v.repeat_interleave(0, 0).unwrap().shape(), &[0]);
        let d = Tensor::from_slice(&[5.0, 7.0], &[2]).unwrap();
        assert_close(
            &d.diag_embed().unwrap().to_vec::<f64>().unwrap(),
            &[5.0, 0.0, 0.0, 7.0],
            0.0,
            "dembed",
        );
        let dg = Tensor::from_slice(&[5.0, 7.0], &[2])
            .unwrap()
            .requires_grad_(true);
        dg.diag_embed().unwrap().sum().unwrap().backward().unwrap();
        assert_close(
            &dg.grad().unwrap().to_vec::<f64>().unwrap(),
            &[1.0, 1.0],
            0.0,
            "dembed-g",
        );
        // batched like torch.diag_embed: [2, 2] -> [2, 2, 2]
        assert_eq!(d.diag_embed().unwrap().diag_embed().unwrap().shape(), [2, 2, 2]);
    }

    #[test]
    fn comparisons_nan_bool_broadcast_match_torch() {
        // NaN is unordered: only `ne` holds (torch-verified)
        let n = f64::NAN;
        let a = Tensor::from_slice(&[n, 1.0, f64::INFINITY], &[3]).unwrap();
        let b = Tensor::from_slice(&[n, 1.0, f64::INFINITY], &[3]).unwrap();
        let v = |t: Tensor| t.to_vec::<bool>().unwrap();
        assert_eq!(v(a.eq_tensor(&b).unwrap()), vec![false, true, true]);
        assert_eq!(v(a.ne_tensor(&b).unwrap()), vec![true, false, false]);
        assert_eq!(v(a.lt(&b).unwrap()), vec![false, false, false]);
        assert_eq!(v(a.le(&b).unwrap()), vec![false, true, true]);
        assert_eq!(v(a.gt(&b).unwrap()), vec![false, false, false]);
        assert_eq!(v(a.ge(&b).unwrap()), vec![false, true, true]);
        // bool operands + singleton broadcast (torch-verified)
        let t = Tensor::from_slice(&[true, false], &[2]).unwrap();
        let u = Tensor::from_slice(&[true, true], &[2]).unwrap();
        assert_eq!(v(t.eq_tensor(&u).unwrap()), vec![true, false]);
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let y = Tensor::from_slice(&[2.0], &[1]).unwrap();
        assert_eq!(v(x.lt(&y).unwrap()), vec![true, false, false]);
        // mixed dtypes fail closed (torch promotes: deliberate, matches binary_op)
        let i = Tensor::from_slice(&[1i64], &[1]).unwrap();
        let f = Tensor::from_slice(&[1.0], &[1]).unwrap();
        assert!(i.eq_tensor(&f).is_err());
    }

    #[test]
    fn comparisons_i64_exact_beyond_2pow53() {
        // f64 rounding must not leak into integer equality (torch-verified)
        let big = 2i64.pow(53);
        let a = Tensor::from_slice(&[big, big + 1], &[2]).unwrap();
        let b = Tensor::from_slice(&[big + 1, big + 1], &[2]).unwrap();
        let v = |t: Tensor| t.to_vec::<bool>().unwrap();
        assert_eq!(v(a.eq_tensor(&b).unwrap()), vec![false, true]);
        assert_eq!(v(a.ne_tensor(&b).unwrap()), vec![true, false]);
        assert_eq!(v(a.lt(&b).unwrap()), vec![true, false]);
        assert_eq!(v(a.ge(&b).unwrap()), vec![false, true]);
    }

    #[test]
    fn logical_not_matches_torch() {
        // nonzero (incl. NaN) -> false, zero (incl. -0.0) -> true (torch-verified)
        let f = Tensor::from_slice(&[0.0, 2.0, f64::NAN, -0.0], &[4]).unwrap();
        assert_eq!(
            f.logical_not().unwrap().to_vec::<bool>().unwrap(),
            vec![true, false, false, true]
        );
        let i = Tensor::from_slice(&[0i32, 3, -2], &[3]).unwrap();
        assert_eq!(
            i.logical_not().unwrap().to_vec::<bool>().unwrap(),
            vec![true, false, false]
        );
        let b = Tensor::from_slice(&[true, false], &[2]).unwrap();
        assert_eq!(
            b.logical_not().unwrap().to_vec::<bool>().unwrap(),
            vec![false, true]
        );
    }

    #[test]
    fn allclose_inf_broadcast_and_errors_match_torch() {
        // same-signed inf is close, NaN is not (equal_nan=false, torch-verified)
        let inf = Tensor::from_slice(&[f64::INFINITY], &[1]).unwrap();
        let ninf = Tensor::from_slice(&[f64::NEG_INFINITY], &[1]).unwrap();
        let nan = Tensor::from_slice(&[f64::NAN], &[1]).unwrap();
        assert!(inf.allclose(&inf, 1e-5, 1e-8).unwrap());
        assert!(ninf.allclose(&ninf, 1e-5, 1e-8).unwrap());
        assert!(!inf.allclose(&ninf, 1e-5, 1e-8).unwrap());
        assert!(!nan.allclose(&nan, 1e-5, 1e-8).unwrap());
        // singleton dims broadcast; incompatible shapes error like torch raises
        let z3 = Tensor::from_slice(&[0.0, 0.0, 0.0], &[3]).unwrap();
        let z1 = Tensor::from_slice(&[0.0], &[1]).unwrap();
        let z2 = Tensor::from_slice(&[0.0, 0.0], &[2]).unwrap();
        assert!(z3.allclose(&z1, 1e-5, 1e-8).unwrap());
        assert!(z3.allclose(&z2, 1e-5, 1e-8).is_err());
        // cross-dtype and empty keep working
        let f32t = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
        let f64t = Tensor::from_slice(&[1.0], &[1]).unwrap();
        assert!(f32t.allclose(&f64t, 1e-5, 1e-8).unwrap());
        let e = Tensor::from_slice(&[] as &[f64], &[0]).unwrap();
        assert!(e.allclose(&e, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn bitwise_int_bool_and_broadcast_match_torch() {
        // u8 nibble patterns (torch-verified)
        let a = Tensor::from_slice(&[12u8, 10, 0], &[3]).unwrap();
        let b = Tensor::from_slice(&[10u8, 6, 255], &[3]).unwrap();
        assert_eq!(
            a.bitwise_and(&b).unwrap().to_vec::<u8>().unwrap(),
            vec![8, 2, 0]
        );
        assert_eq!(
            a.bitwise_or(&b).unwrap().to_vec::<u8>().unwrap(),
            vec![14, 14, 255]
        );
        assert_eq!(
            a.bitwise_xor(&b).unwrap().to_vec::<u8>().unwrap(),
            vec![6, 12, 255]
        );
        assert_eq!(
            a.bitwise_not().unwrap().to_vec::<u8>().unwrap(),
            vec![243, 245, 255]
        );
        // i32 two's-complement negatives (torch-verified)
        let x = Tensor::from_slice(&[-1i32, -8, 5], &[3]).unwrap();
        let y = Tensor::from_slice(&[5i32, 3, -3], &[3]).unwrap();
        assert_eq!(
            x.bitwise_and(&y).unwrap().to_vec::<i32>().unwrap(),
            vec![5, 0, 5]
        );
        assert_eq!(
            x.bitwise_or(&y).unwrap().to_vec::<i32>().unwrap(),
            vec![-1, -5, -3]
        );
        assert_eq!(
            x.bitwise_not().unwrap().to_vec::<i32>().unwrap(),
            vec![0, 7, -6]
        );
        // i64 beyond 2^53 stays bit-exact (no f64 roundtrip)
        let big = 2i64.pow(53);
        let p = Tensor::from_slice(&[big + 1], &[1]).unwrap();
        let q = Tensor::from_slice(&[big + 3], &[1]).unwrap();
        assert_eq!(
            p.bitwise_and(&q).unwrap().to_vec::<i64>().unwrap(),
            vec![big + 1]
        );
        // bool operands + singleton broadcast (torch-verified)
        let t = Tensor::from_slice(&[true, false, true], &[3]).unwrap();
        let u = Tensor::from_slice(&[true], &[1]).unwrap();
        assert_eq!(
            t.bitwise_and(&u).unwrap().to_vec::<bool>().unwrap(),
            vec![true, false, true]
        );
        assert_eq!(
            t.bitwise_not().unwrap().to_vec::<bool>().unwrap(),
            vec![false, true, false]
        );
        // floats rejected like torch; mixed ints fail closed (torch promotes:
        // deliberate, matches binary_op/cmp_op)
        let f = Tensor::from_slice(&[1.0f32], &[1]).unwrap();
        assert!(f.bitwise_and(&f).is_err());
        assert!(f.bitwise_not().is_err());
        let i32t = Tensor::from_slice(&[1i32], &[1]).unwrap();
        let i64t = Tensor::from_slice(&[1i64], &[1]).unwrap();
        assert!(i32t.bitwise_and(&i64t).is_err());
    }

    #[test]
    fn index_ops_wrap_negatives_and_reject_mixed_dtypes() {
        // Deliberate divergence: negatives wrap (torch rejects them in these
        // four ops but wraps in take/advanced indexing; wrapping is pinned by
        // three sibling tests below). Still-OOB indices error either way.
        let v = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let neg = Tensor::from_slice(&[-1i64], &[1]).unwrap();
        let src = Tensor::from_slice(&[9.0], &[1]).unwrap();
        assert_eq!(
            v.index_select(0, &neg).unwrap().to_vec::<f64>().unwrap(),
            vec![3.0]
        );
        assert_eq!(
            v.index_add(0, &neg, &src).unwrap().to_vec::<f64>().unwrap(),
            vec![1.0, 2.0, 12.0]
        );
        let neg3 = Tensor::from_slice(&[-1i64, 0, 0], &[3]).unwrap();
        assert_eq!(
            v.gather(0, &neg3).unwrap().to_vec::<f64>().unwrap(),
            vec![3.0, 1.0, 1.0]
        );
        assert_eq!(
            v.scatter_add(0, &neg3, &v)
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            vec![6.0, 2.0, 4.0]
        );
        let oob = Tensor::from_slice(&[-4i64], &[1]).unwrap();
        assert!(v.index_select(0, &oob).is_err());
        // mixed value dtypes fail closed (torch raises)
        let f = Tensor::from_slice(&[1.0], &[1]).unwrap();
        let i = Tensor::from_slice(&[1i64], &[1]).unwrap();
        let z = Tensor::from_slice(&[0i64], &[1]).unwrap();
        assert!(f.index_add(0, &z, &i).is_err());
        assert!(f.scatter_add(0, &z, &i).is_err());
        // duplicates accumulate; empty selection yields empty (torch-verified)
        let d = Tensor::from_slice(&[0.0, 0.0], &[2]).unwrap();
        let dup = Tensor::from_slice(&[1i64, 1, 0], &[3]).unwrap();
        let s3 = Tensor::from_slice(&[1.0, 2.0, 4.0], &[3]).unwrap();
        assert_eq!(
            d.index_add(0, &dup, &s3).unwrap().to_vec::<f64>().unwrap(),
            vec![4.0, 3.0]
        );
        let e = Tensor::from_slice(&[] as &[i64], &[0]).unwrap();
        assert_eq!(v.index_select(0, &e).unwrap().shape(), &[0]);
    }

    #[test]
    fn index_ops_are_exact_for_integers_like_torch() {
        // values beyond 2^53 survive selection exactly (no f64 roundtrip)
        let big = 2i64.pow(53);
        let v = Tensor::from_slice(&[big + 1, 5], &[2]).unwrap();
        let z = Tensor::from_slice(&[0i64], &[1]).unwrap();
        assert_eq!(
            v.index_select(0, &z).unwrap().to_vec::<i64>().unwrap(),
            vec![big + 1]
        );
        let g = Tensor::from_slice(&[0i64, 1], &[2]).unwrap();
        assert_eq!(
            v.gather(0, &g).unwrap().to_vec::<i64>().unwrap(),
            vec![big + 1, 5]
        );
        // integer add wraps (torch: 2^30 + 2^30 -> -2^31)
        let a = Tensor::from_slice(&[1073741824i32], &[1]).unwrap();
        let s = Tensor::from_slice(&[1073741824i32], &[1]).unwrap();
        assert_eq!(
            a.index_add(0, &z, &s).unwrap().to_vec::<i32>().unwrap(),
            vec![-2147483648]
        );
        assert_eq!(
            a.scatter_add(0, &z, &s).unwrap().to_vec::<i32>().unwrap(),
            vec![-2147483648]
        );
        // bool add is OR (torch truth table)
        for (x, y, want) in [
            (false, false, false),
            (false, true, true),
            (true, false, true),
            (true, true, true),
        ] {
            let xb = Tensor::from_slice(&[x], &[1]).unwrap();
            let yb = Tensor::from_slice(&[y], &[1]).unwrap();
            assert_eq!(
                xb.index_add(0, &z, &yb).unwrap().to_vec::<bool>().unwrap(),
                vec![want]
            );
            assert_eq!(
                xb.scatter_add(0, &z, &yb)
                    .unwrap()
                    .to_vec::<bool>()
                    .unwrap(),
                vec![want]
            );
        }
    }

    #[test]
    fn masked_fill_and_where_coerce_nonbool_masks_by_design() {
        // Deliberate divergence: torch errors on non-bool masks, oxideon reads
        // them with != 0 truthiness (documented on masked_fill).
        let v = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let m = Tensor::from_slice(&[1i64, 0, 0], &[3]).unwrap();
        assert_eq!(
            v.masked_fill(&m, 5.0).unwrap().to_vec::<f64>().unwrap(),
            vec![5.0, 2.0, 3.0]
        );
        let t = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
        let f = Tensor::from_slice(&[3.0, 4.0], &[2]).unwrap();
        let c = Tensor::from_slice(&[1i64, 0], &[2]).unwrap();
        assert_eq!(
            Tensor::where_cond(&c, &t, &f)
                .unwrap()
                .to_vec::<f64>()
                .unwrap(),
            vec![1.0, 4.0]
        );
    }

    #[test]
    fn logical_truthiness_match_torch() {
        // NaN is truthy (proved by or/xor); -0.0 is falsy (torch-verified)
        let f = Tensor::from_slice(&[0.0, 2.0, f64::NAN, -0.0], &[4]).unwrap();
        let t = Tensor::from_slice(&[true, true, false, true], &[4]).unwrap();
        let v = |x: Tensor| x.to_vec::<bool>().unwrap();
        assert_eq!(
            v(f.logical_and(&t).unwrap()),
            vec![false, true, false, false]
        );
        assert_eq!(v(f.logical_or(&t).unwrap()), vec![true, true, true, true]);
        assert_eq!(v(f.logical_xor(&t).unwrap()), vec![true, false, true, true]);
        // int operands (torch-verified)
        let i = Tensor::from_slice(&[0i32, 2], &[2]).unwrap();
        let z = Tensor::from_slice(&[0i32, 0], &[2]).unwrap();
        assert_eq!(v(i.logical_and(&z).unwrap()), vec![false, false]);
        assert_eq!(v(i.logical_or(&z).unwrap()), vec![false, true]);
        // mixed dtypes work (per-side truthiness needs no promotion, like torch)
        let g = Tensor::from_slice(&[1.0, 0.0], &[2]).unwrap();
        let h = Tensor::from_slice(&[1i64, 0], &[2]).unwrap();
        assert_eq!(v(g.logical_and(&h).unwrap()), vec![true, false]);
        assert_eq!(v(g.logical_or(&h).unwrap()), vec![true, false]);
    }

    #[test]
    fn is_predicates_match_torch() {
        // float NaN/inf/finite (torch-verified, incl. -inf)
        let f = Tensor::from_slice(&[1.0, f64::NAN, f64::INFINITY], &[3]).unwrap();
        let v = |x: Tensor| x.to_vec::<bool>().unwrap();
        assert_eq!(v(f.isnan().unwrap()), vec![false, true, false]);
        assert_eq!(v(f.isinf().unwrap()), vec![false, false, true]);
        assert_eq!(v(f.isfinite().unwrap()), vec![true, false, false]);
        let n = Tensor::from_slice(&[f32::NEG_INFINITY], &[1]).unwrap();
        assert_eq!(v(n.isinf().unwrap()), vec![true]);
        assert_eq!(v(n.isfinite().unwrap()), vec![false]);
        // integers/bools: never NaN/inf, always finite (torch-verified)
        let i = Tensor::from_slice(&[1i32, 0], &[2]).unwrap();
        assert_eq!(v(i.isnan().unwrap()), vec![false, false]);
        assert_eq!(v(i.isinf().unwrap()), vec![false, false]);
        assert_eq!(v(i.isfinite().unwrap()), vec![true, true]);
        let b = Tensor::from_slice(&[true, false], &[2]).unwrap();
        assert_eq!(v(b.isfinite().unwrap()), vec![true, true]);
    }
}
