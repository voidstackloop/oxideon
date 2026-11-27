//! Pluggable compute backends (`Cpu`, accelerated devices, and optional `Wgpu`) and reference-counted [`Storage`].

pub mod cpu;
pub mod simd_arm;
#[cfg(target_arch = "x86_64")]
pub mod simd_x86;
#[cfg(not(target_arch = "x86_64"))]
pub mod simd_x86 {
    #[inline]
    pub fn has_avx2_fma() -> bool { false }
    #[inline]
    pub fn exp_f32_slice(_src: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn tanh_f32_slice(_src: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn gelu_f32_slice(_src: &[f32], _dst: &mut [f32], _fast: bool) -> bool { false }
    #[inline]
    pub fn gelu_backward_f32_slice(_g: &[f32], _x: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn sum_f32_slice(_src: &[f32]) -> Option<f32> { None }
    #[inline]
    pub fn softmax_row_f32(_src: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn layer_norm_row_f32(_src: &[f32], _w: &[f32], _b: &[f32], _dst: &mut [f32], _eps: f32) -> bool { false }
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_oc_block4_s1d1_f32(
        _src: &[f32], _w_block: &[f32], _b_block: [f32; 4], _out_block: &mut [f32],
        _c_in: usize, _h: usize, _w: usize,
        _kh: usize, _kw: usize, _ph: usize, _pw: usize, _out_h: usize, _out_w: usize,
    ) -> bool { false }
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_image_s1d1_f32(
        _src: &[f32], _wv: &[f32], _bias: Option<&[f32]>, _out: &mut [f32],
        _c_in: usize, _h: usize, _w: usize, _c_out: usize,
        _kh: usize, _kw: usize, _ph: usize, _pw: usize, _out_h: usize, _out_w: usize,
    ) -> bool { false }
    #[inline]
    pub fn add_into_f32_slice(_dst: &mut [f32], _src: &[f32]) -> bool { false }
    #[inline]
    pub fn add_f32_slice(_a: &[f32], _b: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn sub_f32_slice(_a: &[f32], _b: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn mul_f32_slice(_a: &[f32], _b: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn div_f32_slice(_a: &[f32], _b: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn sigmoid_f32_slice(_src: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn silu_f32_slice(_src: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn relu_f32_slice(_src: &[f32], _dst: &mut [f32]) -> bool { false }
    #[inline]
    pub fn add_bias_row_f32(_row: &mut [f32], _bias: &[f32]) -> bool { false }
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_f32_slice(
        _p: &mut [f32], _g: &[f32], _m: &mut [f32], _v: &mut [f32],
        _lr: f32, _beta1: f32, _beta2: f32, _eps: f32, _weight_decay: f32,
        _decoupled: bool, _inv_bc1: f32, _inv_bc2: f32,
    ) -> bool { false }
    #[inline]
    pub fn layer_norm_row_bwd_f32(
        _x: &[f32], _g: &[f32], _w: &[f32], _dx: &mut [f32], _dw: &mut [f32], _db: &mut [f32], _eps: f32,
    ) -> bool { false }
}
pub mod cublas;
pub mod cuda_driver;
pub mod cuda_kernels;
pub mod cuda_ops;
pub mod cuda_rt;
pub mod nvrtc;
pub mod stream;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

pub use crate::device::{
    CompositeDevice, Device, DeviceMesh, DeviceProperties, DeviceType, InterconnectMatrix,
    InterconnectType, get_default_device, set_default_device, with_device,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::{Layout, ViewSpec, broadcast_shapes};

pub use cpu::CpuStorage;
pub use stream::{ComputeGraph, ComputeStream, GraphOp, StreamEvent, StreamId};
#[cfg(feature = "wgpu")]
pub use wgpu_backend::{WgpuDevice, WgpuStorage};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sign,
    Exp,
    Ln,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Relu,
    LeakyRelu(f64),
    Gelu,
    Silu,
    Elu(f64),
    Softplus,
    Square,
    Recip,
    Floor,
    Ceil,
    Round,
    Erf,
    Powf(f64),
    AddScalar(f64),
    MulScalar(f64),
    Clamp(f64, f64),
    LogicalNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Pow,
    /// C-style remainder (sign of the dividend), `torch.fmod`.
    Fmod,
    /// Python-style remainder (sign of the divisor), `torch.remainder`.
    Remainder,
    /// `floor(a / b)` evaluated exactly, `torch.floor_divide`.
    FloorDiv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOp {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredOp {
    IsNan,
    IsInf,
    IsFinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    ArgMax,
    ArgMin,
}

/// Device-specific raw buffer enum.
#[derive(Clone, Debug)]
pub enum StorageInner {
    Cpu(CpuStorage),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuStorage),
    Cuda(cuda_kernels::CudaStorage),
    /// Placement label only: the data lives in host memory and every kernel runs on the CPU.
    /// Backs the simulated `Cuda`/`Metal`/`Vulkan`/`Tpu`/`Npu` devices (see [`Device::is_simulated`]).
    Accelerator {
        device: Device,
        storage: CpuStorage,
    },
    Meta {
        dtype: DType,
        len: usize,
    },
}

/// Shared, interior-mutable handle to device storage so views share the same allocation.
#[derive(Clone, Debug)]
pub struct Storage(Arc<Tracked>);

/// The lock around a buffer plus the byte count it contributed to [`crate::memory`] (taken back when the last handle drops).
#[derive(Debug)]
pub struct Tracked {
    lock: RwLock<StorageInner>,
    bytes: usize,
    dlpack_pins: std::sync::atomic::AtomicUsize,
    /// Set for storage that aliases memory owned by someone else (a zero-copy DLPack import): called once when the last
    /// handle drops, after the buffer has been detached from its `Vec` so Rust never frees it.
    foreign: Release,
}

/// A one-shot release callback (opaque for `Debug`).
#[derive(Default)]
struct Release(Option<Box<dyn FnOnce() + Send + Sync>>);

impl std::fmt::Debug for Release {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Release(foreign)" } else { "Release(none)" })
    }
}

impl std::ops::Deref for Tracked {
    type Target = RwLock<StorageInner>;
    fn deref(&self) -> &Self::Target {
        &self.lock
    }
}

impl Tracked {
    fn new(inner: StorageInner) -> Arc<Self> {
        let bytes = match &inner {
            StorageInner::Cpu(s) | StorageInner::Accelerator { storage: s, .. } => s.len() * s.dtype().size_in_bytes(),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => s.len() * s.dtype().size_in_bytes(),
            StorageInner::Cuda(s) => s.len * s.dtype.size_in_bytes(),
            StorageInner::Meta { .. } => 0,
        };
        crate::memory::on_alloc(bytes);
        Arc::new(Self {
            lock: RwLock::new(inner),
            bytes,
            dlpack_pins: std::sync::atomic::AtomicUsize::new(0),
            foreign: Release::default(),
        })
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
        if let Some(release) = self.foreign.0.take() {
            // the buffer belongs to the exporter: detach it so neither the allocator nor the pool ever sees it, then hand it
            // back through the exporter's own deleter
            if let StorageInner::Cpu(cpu) = self.lock.get_mut().unwrap_or_else(|p| p.into_inner()) {
                cpu.forget_buffer();
            }
            release();
            return;
        }
        crate::memory::on_free(self.bytes);
        // large f32 buffers go back to the pool instead of to the operating system
        if self.bytes >= crate::pool::MIN_LEN * 4
            && let Ok(StorageInner::Cpu(CpuStorage::F32(v))) = self.lock.get_mut()
        {
            crate::pool::give(std::mem::take(v));
        }
    }
}

/// Borrowed (or, for GPU storage, read-back) host data; dereferences to [`CpuStorage`].
pub enum CpuRef<'a> {
    Borrowed(RwLockReadGuard<'a, StorageInner>),
    Owned(CpuStorage),
}

impl std::ops::Deref for CpuRef<'_> {
    type Target = CpuStorage;
    fn deref(&self) -> &CpuStorage {
        match self {
            Self::Owned(s) => s,
            Self::Borrowed(g) => match &**g {
                StorageInner::Cpu(s) | StorageInner::Accelerator { storage: s, .. } => s,
                _ => unreachable!("cpu_ref only borrows host storage"),
            },
        }
    }
}

impl Storage {
    /// Wraps host memory owned by someone else (a DLPack producer) without copying it. `cpu` must be a [`CpuStorage`] whose
    /// `Vec` was built over that memory; it is detached again before `release` runs when the last handle drops, so it is
    /// never freed here. The storage counts as permanently pinned: it can be written in place but never reallocated or
    /// replaced, which is what keeps the aliasing sound.
    pub(crate) fn from_foreign_cpu(cpu: CpuStorage, release: Box<dyn FnOnce() + Send + Sync>) -> Self {
        Self(Arc::new(Tracked {
            lock: RwLock::new(StorageInner::Cpu(cpu)),
            bytes: 0,
            dlpack_pins: std::sync::atomic::AtomicUsize::new(1),
            foreign: Release(Some(release)),
        }))
    }

    /// Sum of squares of a CUDA tensor computed on the device (`None` for other storage).
    pub(crate) fn cuda_sum_squares(&self, layout: &Layout) -> Option<f64> {
        match &*self.read() {
            StorageInner::Cuda(s) => s.sum_squares_gpu(layout).ok().flatten(),
            _ => None,
        }
    }

    /// Fused single-pass last-axis softmax on contiguous CPU storage (returns `None` to fall back).
    pub(crate) fn try_softmax_last_axis(&self, layout: &Layout, needs_grad: bool) -> Option<Self> {
        // NVRTC-compiled kernel on CUDA tensors: rows with a non-finite maximum need the composed path when gradients are
        // recorded (its +inf rows have a zero gradient), so the kernel reports them and this returns `None` for those
        if let StorageInner::Cuda(s) = &*self.0.read().unwrap_or_else(|p| p.into_inner())
            && cuda_rt::available()
        {
            return match s.softmax_gpu(layout, false, needs_grad) {
                Ok(Some((out, false))) => Some(Self::new_cuda(out)),
                _ => None,
            };
        }
        #[cfg(feature = "wgpu")]
        if let StorageInner::Wgpu(s) = &*self.0.read().unwrap_or_else(|p| p.into_inner()) {
            return s.softmax_last_axis(layout, false).ok().flatten().map(Self::new_wgpu);
        }
        let out_inner = {
            let guard = self.0.read().unwrap_or_else(|p| p.into_inner());
            match &*guard {
                StorageInner::Cpu(s) => StorageInner::Cpu(s.softmax_last_axis(layout)?),
                StorageInner::Cuda(s) => StorageInner::Cuda(s.softmax_last_axis(layout)?),
                StorageInner::Accelerator { device, storage } => StorageInner::Accelerator {
                    device: device.clone(),
                    storage: storage.softmax_last_axis(layout)?,
                },
                _ => return None,
            }
        };
        Some(Self(Tracked::new(out_inner)))
    }

    /// Log-softmax over the last axis of a CUDA tensor with the NVRTC kernel (`None` for everything else, or when a row with
    /// a non-finite maximum must take the composed path because gradients are recorded).
    pub(crate) fn try_log_softmax_last_axis(&self, layout: &Layout, needs_grad: bool) -> Option<Self> {
        if let StorageInner::Cuda(s) = &*self.0.read().unwrap_or_else(|p| p.into_inner())
            && cuda_rt::available()
        {
            return match s.softmax_gpu(layout, true, needs_grad) {
                Ok(Some((out, false))) => Some(Self::new_cuda(out)),
                _ => None,
            };
        }
        #[cfg(feature = "wgpu")]
        if let StorageInner::Wgpu(s) = &*self.0.read().unwrap_or_else(|p| p.into_inner()) {
            return s.softmax_last_axis(layout, true).ok().flatten().map(Self::new_wgpu);
        }
        if let StorageInner::Cpu(s) = &*self.0.read().unwrap_or_else(|p| p.into_inner()) {
            return s.log_softmax_last_axis(layout).map(Self::new_cpu);
        }
        None
    }

    /// Fused single-pass last-axis `LayerNorm` on contiguous CPU `F32` storage.
    pub(crate) fn try_layer_norm_last_axis(
        &self,
        layout: &Layout,
        weight: &Self,
        w_layout: &Layout,
        bias: &Self,
        b_layout: &Layout,
        eps: f64,
    ) -> Option<Self> {
        let out_inner = {
            let gx = self.0.read().unwrap_or_else(|p| p.into_inner());
            let gw = weight.0.read().unwrap_or_else(|p| p.into_inner());
            let gb = bias.0.read().unwrap_or_else(|p| p.into_inner());
            match (&*gx, &*gw, &*gb) {
                (StorageInner::Cpu(xs), StorageInner::Cpu(ws), StorageInner::Cpu(bs)) => {
                    StorageInner::Cpu(xs.layer_norm_last_axis(layout, ws, w_layout, bs, b_layout, eps)?)
                }
                (StorageInner::Cuda(xs), StorageInner::Cuda(ws), StorageInner::Cuda(bs)) if w_layout.is_contiguous() && b_layout.is_contiguous() && w_layout.offset() == 0 && b_layout.offset() == 0 => {
                    match xs.layer_norm_gpu(layout, Some(ws), Some(bs), eps, false) {
                        Ok(Some((y, _, _))) => StorageInner::Cuda(y),
                        _ => return None,
                    }
                }
                _ => return None,
            }
        };
        Some(Self(Tracked::new(out_inner)))
    }

    /// Fused single-pass GELU backward on contiguous CPU `F32` storage.
    pub(crate) fn try_gelu_backward(&self, g_layout: &Layout, x: &Self, x_layout: &Layout) -> Option<Self> {
        let out_inner = {
            let gg = self.0.read().unwrap_or_else(|p| p.into_inner());
            let gx = x.0.read().unwrap_or_else(|p| p.into_inner());
            match (&*gg, &*gx) {
                (StorageInner::Cpu(gs), StorageInner::Cpu(xs)) => {
                    StorageInner::Cpu(gs.gelu_backward_f32(g_layout, xs, x_layout)?)
                }
                _ => return None,
            }
        };
        Some(Self(Tracked::new(out_inner)))
    }

    /// Fused single-pass backward `g * f(a)` of a unary op on CPU `F32` storage (see [`cpu::UnaryBwd`]).
    pub(crate) fn try_unary_backward(&self, g_layout: &Layout, a: &Self, a_layout: &Layout, kind: cpu::UnaryBwd) -> Option<Self> {
        let gg = self.0.read().unwrap_or_else(|p| p.into_inner());
        let ga = a.0.read().unwrap_or_else(|p| p.into_inner());
        match (&*gg, &*ga) {
            (StorageInner::Cpu(gs), StorageInner::Cpu(a_s)) => Some(Self(Tracked::new(StorageInner::Cpu(gs.unary_backward_f32(g_layout, a_s, a_layout, kind)?)))),
            _ => None,
        }
    }

    /// Adds a 1D `F32` `bias` (`[cols]`) in-place across every row of a freshly-allocated contiguous
    /// `[rows, cols]` CPU `F32` storage (returns `false` if not uniquely owned or incompatible).
    pub(crate) fn try_add_bias_inplace_f32(&self, bias: &Self, bias_layout: &Layout, cols: usize) -> bool {
        if cols == 0 || !bias_layout.is_contiguous() {
            return false;
        }
        let mut g_dst = self.0.write().unwrap_or_else(|p| p.into_inner());
        let g_bias = bias.0.read().unwrap_or_else(|p| p.into_inner());
        let (StorageInner::Cpu(CpuStorage::F32(dst)), StorageInner::Cpu(CpuStorage::F32(bv))) = (&mut *g_dst, &*g_bias) else {
            return false;
        };
        let Some(br) = bias_layout.contiguous_range_in(bv.len()) else {
            return false;
        };
        let bs = &bv[br];
        if bs.len() != cols || dst.len() % cols != 0 {
            return false;
        }
        #[cfg(feature = "rayon")]
        if dst.len() >= 16384 {
            use rayon::prelude::*;
            dst.par_chunks_mut(cols).for_each(|row| {
                if !simd_x86::add_bias_row_f32(row, bs) {
                    for (d, &b) in row.iter_mut().zip(bs) {
                        *d += b;
                    }
                }
            });
            return true;
        }
        for row in dst.chunks_mut(cols) {
            if !simd_x86::add_bias_row_f32(row, bs) {
                for (d, &b) in row.iter_mut().zip(bs) {
                    *d += b;
                }
            }
        }
        true
    }

    /// Single-pass fused Adam / AdamW parameter update on contiguous CPU `F32` storage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_adam_step_f32(
        &self,
        p_layout: &Layout,
        grad: &Self,
        g_layout: &Layout,
        m_stor: &Self,
        v_stor: &Self,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        decoupled: bool,
        bc1: f64,
        bc2: f64,
    ) -> bool {
        if self.ptr_eq(m_stor) || self.ptr_eq(v_stor) || m_stor.ptr_eq(v_stor) {
            return false;
        }
        let n = p_layout.numel();
        if n == 0 || !p_layout.is_contiguous() || !g_layout.is_contiguous() || g_layout.numel() != n {
            return false;
        }
        {
            // CUDA tensors: one fused kernel, updating the buffers in place
            let (rp, rg, rm, rv) = (self.read(), grad.read(), m_stor.read(), v_stor.read());
            if let (StorageInner::Cuda(p), StorageInner::Cuda(g), StorageInner::Cuda(m), StorageInner::Cuda(v)) = (&*rp, &*rg, &*rm, &*rv) {
                return p.adam_step_gpu(p_layout, g, g_layout, m, v, lr, beta1, beta2, eps, weight_decay, decoupled, bc1, bc2).unwrap_or(false);
            }
        }
        let mut gp = self.0.write().unwrap_or_else(|p| p.into_inner());
        let gg = grad.0.read().unwrap_or_else(|p| p.into_inner());
        let mut gm = m_stor.0.write().unwrap_or_else(|p| p.into_inner());
        let mut gv = v_stor.0.write().unwrap_or_else(|p| p.into_inner());
        let (
            StorageInner::Cpu(CpuStorage::F32(pv)),
            StorageInner::Cpu(CpuStorage::F32(gv_s)),
            StorageInner::Cpu(CpuStorage::F32(mv)),
            StorageInner::Cpu(CpuStorage::F32(vv)),
        ) = (&mut *gp, &*gg, &mut *gm, &mut *gv)
        else {
            return false;
        };
        let Some(pr) = p_layout.contiguous_range_in(pv.len()) else { return false };
        let Some(gr) = g_layout.contiguous_range_in(gv_s.len()) else { return false };
        if mv.len() != n || vv.len() != n {
            return false;
        }
        let ps = &mut pv[pr];
        let gs = &gv_s[gr];
        let (lr32, b1, b2, eps32, wd32, ibc1, ibc2) = (
            lr as f32,
            beta1 as f32,
            beta2 as f32,
            eps as f32,
            weight_decay as f32,
            (1.0 / bc1) as f32,
            (1.0 / bc2) as f32,
        );
        let run_chunk = |p_c: &mut [f32], g_c: &[f32], m_c: &mut [f32], v_c: &mut [f32]| {
            if !simd_x86::adam_step_f32_slice(p_c, g_c, m_c, v_c, lr32, b1, b2, eps32, wd32, decoupled, ibc1, ibc2) {
                for (((p, &g), m), v) in p_c.iter_mut().zip(g_c).zip(m_c.iter_mut()).zip(v_c.iter_mut()) {
                    let mut pv = *p;
                    let mut gv = g;
                    if wd32 != 0.0 {
                        if decoupled {
                            pv *= 1.0 - lr32 * wd32;
                        } else {
                            gv += pv * wd32;
                        }
                    }
                    *m = b1 * *m + (1.0 - b1) * gv;
                    *v = b2 * *v + (1.0 - b2) * gv * gv;
                    *p = pv - lr32 * (*m * ibc1) / ((*v * ibc2).sqrt() + eps32);
                }
            }
        };
        #[cfg(feature = "rayon")]
        if n >= 16384 {
            use rayon::prelude::*;
            ps.par_chunks_mut(8192)
                .zip(gs.par_chunks(8192))
                .zip(mv.par_chunks_mut(8192))
                .zip(vv.par_chunks_mut(8192))
                .for_each(|(((p_c, g_c), m_c), v_c)| run_chunk(p_c, g_c, m_c, v_c));
            return true;
        }
        run_chunk(ps, gs, mv, vv);
        true
    }

    /// Fused single-pass last-axis `LayerNorm` backward on contiguous CPU `F32` storage.
    pub(crate) fn try_layer_norm_bwd_f32(
        &self,
        x_layout: &Layout,
        grad: &Self,
        g_layout: &Layout,
        weight: &Self,
        w_layout: &Layout,
        eps: f64,
    ) -> Option<(Self, Self, Self)> {
        let d = w_layout.numel();
        let total = x_layout.numel();
        if d == 0 || total == 0 || total % d != 0 || x_layout.shape() != g_layout.shape() {
            return None;
        }
        let gx = self.0.read().unwrap_or_else(|p| p.into_inner());
        let gg = grad.0.read().unwrap_or_else(|p| p.into_inner());
        let gw = weight.0.read().unwrap_or_else(|p| p.into_inner());
        let (
            StorageInner::Cpu(CpuStorage::F32(xv)),
            StorageInner::Cpu(CpuStorage::F32(gv)),
            StorageInner::Cpu(CpuStorage::F32(wv)),
        ) = (&*gx, &*gg, &*gw)
        else {
            return None;
        };
        let xr = x_layout.contiguous_range_in(xv.len())?;
        let gr = g_layout.contiguous_range_in(gv.len())?;
        let wr = w_layout.contiguous_range_in(wv.len())?;
        if wr.len() != d {
            return None;
        }
        let xs = &xv[xr];
        let gs = &gv[gr];
        let ws = &wv[wr];
        let mut dx = crate::pool::take::<f32>(total).unwrap_or_else(|| vec![0.0; total]);
        let mut dw = vec![0.0f32; d];
        let mut db = vec![0.0f32; d];
        let eps32 = eps as f32;

        #[cfg(feature = "rayon")]
        if total >= 16384 {
            use rayon::prelude::*;
            let (dw_acc, db_acc) = xs
                .par_chunks(d)
                .zip(gs.par_chunks(d))
                .zip(dx.par_chunks_mut(d))
                .fold(
                    || (vec![0.0f32; d], vec![0.0f32; d]),
                    |(mut ldw, mut ldb), ((xr_s, gr_s), dx_s)| {
                        if !simd_x86::layer_norm_row_bwd_f32(xr_s, gr_s, ws, dx_s, &mut ldw, &mut ldb, eps32) {
                            let (hi, lo) = cpu::center_split(xr_s);
                            let var = xr_s.iter().map(|&v| ((v - hi) - lo) * ((v - hi) - lo)).sum::<f32>() / (d as f32);
                            let inv_std = 1.0 / (var + eps32).sqrt();
                            let mut s1 = 0.0f32;
                            let mut s2 = 0.0f32;
                            for i in 0..d {
                                let xhat = ((xr_s[i] - hi) - lo) * inv_std;
                                let gxh = gr_s[i] * ws[i];
                                s1 += gxh;
                                s2 += gxh * xhat;
                                ldw[i] += gr_s[i] * xhat;
                                ldb[i] += gr_s[i];
                            }
                            let inv_n = 1.0 / (d as f32);
                            for i in 0..d {
                                let xhat = ((xr_s[i] - hi) - lo) * inv_std;
                                dx_s[i] = (gr_s[i] * ws[i] - s1 * inv_n - xhat * (s2 * inv_n)) * inv_std;
                            }
                        }
                        (ldw, ldb)
                    },
                )
                .reduce(
                    || (vec![0.0f32; d], vec![0.0f32; d]),
                    |(mut a_w, mut a_b), (b_w, b_b)| {
                        for i in 0..d {
                            a_w[i] += b_w[i];
                            a_b[i] += b_b[i];
                        }
                        (a_w, a_b)
                    },
                );
            return Some((
                Self::new_cpu(CpuStorage::F32(dx)),
                Self::new_cpu(CpuStorage::F32(dw_acc)),
                Self::new_cpu(CpuStorage::F32(db_acc)),
            ));
        }

        for ((xr_s, gr_s), dx_s) in xs.chunks(d).zip(gs.chunks(d)).zip(dx.chunks_mut(d)) {
            if !simd_x86::layer_norm_row_bwd_f32(xr_s, gr_s, ws, dx_s, &mut dw, &mut db, eps32) {
                let (hi, lo) = cpu::center_split(xr_s);
                let var = xr_s.iter().map(|&v| ((v - hi) - lo) * ((v - hi) - lo)).sum::<f32>() / (d as f32);
                let inv_std = 1.0 / (var + eps32).sqrt();
                let mut s1 = 0.0f32;
                let mut s2 = 0.0f32;
                for i in 0..d {
                    let xhat = ((xr_s[i] - hi) - lo) * inv_std;
                    let gxh = gr_s[i] * ws[i];
                    s1 += gxh;
                    s2 += gxh * xhat;
                    dw[i] += gr_s[i] * xhat;
                    db[i] += gr_s[i];
                }
                let inv_n = 1.0 / (d as f32);
                for i in 0..d {
                    let xhat = ((xr_s[i] - hi) - lo) * inv_std;
                    dx_s[i] = (gr_s[i] * ws[i] - s1 * inv_n - xhat * (s2 * inv_n)) * inv_std;
                }
            }
        }
        Some((
            Self::new_cpu(CpuStorage::F32(dx)),
            Self::new_cpu(CpuStorage::F32(dw)),
            Self::new_cpu(CpuStorage::F32(db)),
        ))
    }

    pub fn new_cpu(s: CpuStorage) -> Self {
        Self(Tracked::new(StorageInner::Cpu(s)))
    }

    #[cfg(feature = "wgpu")]
    pub fn new_wgpu(s: WgpuStorage) -> Self {
        Self(Tracked::new(StorageInner::Wgpu(s)))
    }

    pub fn new_cuda(s: cuda_kernels::CudaStorage) -> Self {
        Self(Tracked::new(StorageInner::Cuda(s)))
    }

    fn from_cpu_on_device(cpu: CpuStorage, dst: &Device) -> Result<Self> {
        match dst {
            Device::Cpu => Ok(Self::new_cpu(cpu)),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(dev) => Ok(Self::new_wgpu(WgpuStorage::from_cpu(&cpu, dev)?)),
            Device::Cuda(idx)
                if cuda_ops::cpu_bytes(&cpu).is_some()
                    && cuda_driver::cuda_available()
                    && *idx < cuda_driver::cuda_device_count() =>
            {
                Ok(Self::new_cuda(cuda_kernels::CudaStorage::from_cpu(&cpu, *idx)?))
            }
            Device::Meta => Ok(Self::new_meta(cpu.dtype(), cpu.len())),
            dev => Ok(Self::new_accelerator(dev.clone(), cpu)),
        }
    }

    pub fn new_accelerator(device: Device, storage: CpuStorage) -> Self {
        Self(Tracked::new(StorageInner::Accelerator { device, storage }))
    }

    pub fn new_meta(dtype: DType, len: usize) -> Self {
        Self(Tracked::new(StorageInner::Meta { dtype, len }))
    }

    pub fn read(&self) -> RwLockReadGuard<'_, StorageInner> {
        self.0
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, StorageInner> {
        self.0
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Pins this storage against reallocation while a zero-copy DLPack consumer holds its pointer.
    pub fn pin_dlpack(&self) {
        self.0
            .dlpack_pins
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// Releases one DLPack export pin.
    pub fn unpin_dlpack(&self) {
        self.0
            .dlpack_pins
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// Returns true if at least one live DLPack capsule borrows this storage's buffer.
    pub fn is_dlpack_pinned(&self) -> bool {
        self.0
            .dlpack_pins
            .load(std::sync::atomic::Ordering::Acquire)
            > 0
    }

    pub fn dtype(&self) -> DType {
        match &*self.read() {
            StorageInner::Cpu(s) => s.dtype(),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => s.dtype(),
            StorageInner::Cuda(s) => s.dtype,
            StorageInner::Accelerator { storage, .. } => storage.dtype(),
            StorageInner::Meta { dtype, .. } => *dtype,
        }
    }

    pub fn device(&self) -> Device {
        match &*self.read() {
            StorageInner::Cpu(_) => Device::Cpu,
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Device::Wgpu(s.device().clone()),
            StorageInner::Cuda(s) => Device::Cuda(s.device_index),
            StorageInner::Accelerator { device, .. } => device.clone(),
            StorageInner::Meta { .. } => Device::Meta,
        }
    }

    pub fn len(&self) -> usize {
        match &*self.read() {
            StorageInner::Cpu(s) => s.len(),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => s.len(),
            StorageInner::Cuda(s) => s.len,
            StorageInner::Accelerator { storage, .. } => storage.len(),
            StorageInner::Meta { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn all_finite(&self, layout: &Layout) -> bool {
        match &*self.read() {
            StorageInner::Cpu(s) => s.all_finite(layout),
            StorageInner::Accelerator { storage, .. } => storage.all_finite(layout),
            _ => true,
        }
    }

    /// Downloads/copies storage to CPU (no-op clone if already on CPU).
    /// Read access to the host data without copying it (host and simulated-accelerator storage are borrowed under the
    /// read lock; GPU storage is read back once).
    pub fn cpu_ref(&self) -> Result<CpuRef<'_>> {
        let guard = self.read();
        match &*guard {
            StorageInner::Cpu(_) | StorageInner::Accelerator { .. } => Ok(CpuRef::Borrowed(guard)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Ok(CpuRef::Owned(s.to_cpu()?)),
            StorageInner::Cuda(s) => Ok(CpuRef::Owned(s.to_cpu()?)),
            StorageInner::Meta { .. } => Err(Error::Msg(
                "Cannot access raw memory of a Meta tensor (device='meta')".into(),
            )),
        }
    }

    pub fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(s.clone()),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => s.to_cpu(),
            StorageInner::Cuda(s) => s.to_cpu(),
            StorageInner::Accelerator { storage, .. } => Ok(storage.clone()),
            StorageInner::Meta { .. } => Err(Error::Msg(
                "Cannot access raw memory of a Meta tensor (device='meta')".into(),
            )),
        }
    }

    pub fn from_slice<T: Element>(data: &[T], device: &Device) -> Result<Self> {
        let cpu = CpuStorage::from_slice(data)?;
        Self::from_cpu_on_device(cpu, device)
    }

    pub fn full(numel: usize, val: f64, dtype: DType, device: &Device) -> Result<Self> {
        if matches!(device, Device::Meta) {
            return Ok(Self::new_meta(dtype, numel)); // no data: do not build a buffer just to drop it
        }
        if let Some(s) = Self::cuda_full(device, dtype, numel, val)? {
            return Ok(s); // constants are written by a kernel (or memset), not built on the host and uploaded
        }
        let cpu = CpuStorage::full(numel, val, dtype)?;
        Self::from_cpu_on_device(cpu, device)
    }

    pub fn unary_op(&self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        let _prof = crate::profiler::scope("unary_op", layout.shape());
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(s.unary_op(layout, op)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Ok(Self::new_wgpu(s.unary_op(layout, op)?)),
            StorageInner::Cuda(s) => Ok(Self::new_cuda(s.unary_op(layout, op)?)),
            StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                device.clone(),
                storage.unary_op(layout, op)?,
            )),
            StorageInner::Meta { dtype, .. } => {
                let n = layout.shape().iter().product();
                Ok(Self::new_meta(*dtype, n))
            }
        }
    }

    pub fn binary_op(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: BinaryOp,
    ) -> Result<Self> {
        let _prof = crate::profiler::scope("binary_op", lhs_layout.shape());
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch {
                op: "binary_op",
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            });
        }
        if self.ptr_eq(rhs) {
            let guard = self.read();
            return match &*guard {
                StorageInner::Cpu(s) => {
                    Ok(Self::new_cpu(s.binary_op(lhs_layout, s, rhs_layout, op)?))
                }
                #[cfg(feature = "wgpu")]
                StorageInner::Wgpu(s) => {
                    Ok(Self::new_wgpu(s.binary_op(lhs_layout, s, rhs_layout, op)?))
                }
                StorageInner::Cuda(s) => {
                    Ok(Self::new_cuda(s.binary_op(lhs_layout, s, rhs_layout, op)?))
                }
                StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                    device.clone(),
                    storage.binary_op(lhs_layout, storage, rhs_layout, op)?,
                )),
                StorageInner::Meta { dtype, .. } => {
                    let n = lhs_layout.shape().iter().product();
                    Ok(Self::new_meta(*dtype, n))
                }
            };
        }
        match (&*self.read(), &*rhs.read()) {
            (StorageInner::Cpu(a), StorageInner::Cpu(b)) => {
                Ok(Self::new_cpu(a.binary_op(lhs_layout, b, rhs_layout, op)?))
            }
            #[cfg(feature = "wgpu")]
            (StorageInner::Wgpu(a), StorageInner::Wgpu(b)) => {
                Ok(Self::new_wgpu(a.binary_op(lhs_layout, b, rhs_layout, op)?))
            }
            (StorageInner::Cuda(a), StorageInner::Cuda(b)) => {
                Ok(Self::new_cuda(a.binary_op(lhs_layout, b, rhs_layout, op)?))
            }
            (
                StorageInner::Accelerator { device, storage: a },
                StorageInner::Accelerator { storage: b, .. },
            ) => Ok(Self::new_accelerator(
                device.clone(),
                a.binary_op(lhs_layout, b, rhs_layout, op)?,
            )),
            (StorageInner::Meta { dtype, .. }, StorageInner::Meta { .. }) => {
                let n = lhs_layout.shape().iter().product();
                Ok(Self::new_meta(*dtype, n))
            }
            _ => Err(Error::DeviceMismatch {
                op: "binary_op",
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            }),
        }
    }

    pub fn cmp_op(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: CmpOp,
    ) -> Result<Self> {
        let _prof = crate::profiler::scope("cmp_op", lhs_layout.shape());
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch {
                op: "cmp_op",
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            });
        }
        if self.ptr_eq(rhs) {
            let guard = self.read();
            return match &*guard {
                StorageInner::Cpu(s) => Ok(Self::new_cpu(s.cmp_op(lhs_layout, s, rhs_layout, op)?)),
                #[cfg(feature = "wgpu")]
                StorageInner::Wgpu(s) => {
                    if let Some(out) = s.cmp_op(lhs_layout, s, rhs_layout, op)? {
                        return Ok(Self::new_wgpu(out));
                    }
                    let cpu = s.to_cpu()?;
                    let out = cpu.cmp_op(lhs_layout, &cpu, rhs_layout, op)?;
                    Ok(Self::new_wgpu(WgpuStorage::from_cpu(&out, s.device())?))
                }
                StorageInner::Cuda(s) => {
                    if let Some(out) = s.cmp_gpu(lhs_layout, s, rhs_layout, op)? {
                        return Ok(Self::new_cuda(out));
                    }
                    let cpu = s.to_cpu()?;
                    let out = cpu.cmp_op(lhs_layout, &cpu, rhs_layout, op)?;
                    Self::from_cpu_on_device(out, &Device::Cuda(s.device_index))
                }
                StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                    device.clone(),
                    storage.cmp_op(lhs_layout, storage, rhs_layout, op)?,
                )),
                StorageInner::Meta { .. } => {
                    let n = lhs_layout.shape().iter().product();
                    Ok(Self::new_meta(DType::Bool, n))
                }
            };
        }
        match (&*self.read(), &*rhs.read()) {
            (StorageInner::Cpu(a), StorageInner::Cpu(b)) => {
                Ok(Self::new_cpu(a.cmp_op(lhs_layout, b, rhs_layout, op)?))
            }
            #[cfg(feature = "wgpu")]
            (StorageInner::Wgpu(a), StorageInner::Wgpu(b)) => {
                if let Some(out) = a.cmp_op(lhs_layout, b, rhs_layout, op)? {
                    return Ok(Self::new_wgpu(out));
                }
                let ca = a.to_cpu()?;
                let cb = b.to_cpu()?;
                let out = ca.cmp_op(lhs_layout, &cb, rhs_layout, op)?;
                Ok(Self::new_wgpu(WgpuStorage::from_cpu(&out, a.device())?))
            }
            (StorageInner::Cuda(a), StorageInner::Cuda(b)) => {
                if let Some(out) = a.cmp_gpu(lhs_layout, b, rhs_layout, op)? {
                    return Ok(Self::new_cuda(out));
                }
                let ca = a.to_cpu()?;
                let cb = b.to_cpu()?;
                let out = ca.cmp_op(lhs_layout, &cb, rhs_layout, op)?;
                Self::from_cpu_on_device(out, &Device::Cuda(a.device_index))
            }
            (
                StorageInner::Accelerator { device, storage: a },
                StorageInner::Accelerator { storage: b, .. },
            ) => Ok(Self::new_accelerator(
                device.clone(),
                a.cmp_op(lhs_layout, b, rhs_layout, op)?,
            )),
            (StorageInner::Meta { .. }, StorageInner::Meta { .. }) => {
                let n = lhs_layout.shape().iter().product();
                Ok(Self::new_meta(DType::Bool, n))
            }
            _ => Err(Error::DeviceMismatch {
                op: "cmp_op",
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            }),
        }
    }

    /// Shared pair-dispatch (device check, `ptr_eq` fast path, WGPU CPU
    /// roundtrip, simulated-accelerator passthrough, Meta shape-only) so new
    /// elementwise kernels don't each copy the `cmp_op` match.
    fn pair_op(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op_name: &'static str,
        out_dtype: DType,
        f: impl Fn(&CpuStorage, &Layout, &CpuStorage, &Layout) -> Result<CpuStorage>,
    ) -> Result<Self> {
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch {
                op: op_name,
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            });
        }
        if self.ptr_eq(rhs) {
            let guard = self.read();
            return match &*guard {
                StorageInner::Cpu(s) => Ok(Self::new_cpu(f(s, lhs_layout, s, rhs_layout)?)),
                #[cfg(feature = "wgpu")]
                StorageInner::Wgpu(s) => {
                    let cpu = s.to_cpu()?;
                    let out = f(&cpu, lhs_layout, &cpu, rhs_layout)?;
                    Ok(Self::new_wgpu(WgpuStorage::from_cpu(&out, s.device())?))
                }
                StorageInner::Cuda(s) => {
                    let cpu = s.to_cpu()?;
                    let out = f(&cpu, lhs_layout, &cpu, rhs_layout)?;
                    Self::from_cpu_on_device(out, &Device::Cuda(s.device_index))
                }
                StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                    device.clone(),
                    f(storage, lhs_layout, storage, rhs_layout)?,
                )),
                StorageInner::Meta { .. } => {
                    let n = lhs_layout.shape().iter().product();
                    Ok(Self::new_meta(out_dtype, n))
                }
            };
        }
        match (&*self.read(), &*rhs.read()) {
            (StorageInner::Cpu(a), StorageInner::Cpu(b)) => {
                Ok(Self::new_cpu(f(a, lhs_layout, b, rhs_layout)?))
            }
            #[cfg(feature = "wgpu")]
            (StorageInner::Wgpu(a), StorageInner::Wgpu(b)) => {
                let ca = a.to_cpu()?;
                let cb = b.to_cpu()?;
                let out = f(&ca, lhs_layout, &cb, rhs_layout)?;
                Ok(Self::new_wgpu(WgpuStorage::from_cpu(&out, a.device())?))
            }
            (StorageInner::Cuda(a), StorageInner::Cuda(b)) => {
                let ca = a.to_cpu()?;
                let cb = b.to_cpu()?;
                let out = f(&ca, lhs_layout, &cb, rhs_layout)?;
                Self::from_cpu_on_device(out, &Device::Cuda(a.device_index))
            }
            (
                StorageInner::Accelerator { device, storage: a },
                StorageInner::Accelerator { storage: b, .. },
            ) => Ok(Self::new_accelerator(
                device.clone(),
                f(a, lhs_layout, b, rhs_layout)?,
            )),
            (StorageInner::Meta { .. }, StorageInner::Meta { .. }) => {
                let n = lhs_layout.shape().iter().product();
                Ok(Self::new_meta(out_dtype, n))
            }
            _ => Err(Error::DeviceMismatch {
                op: op_name,
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            }),
        }
    }

    /// Shared unary-dispatch with the same backend routing as [`Self::pair_op`].
    fn unary_op_cpu(
        &self,
        layout: &Layout,
        out_dtype: DType,
        f: impl Fn(&CpuStorage, &Layout) -> Result<CpuStorage>,
    ) -> Result<Self> {
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(f(s, layout)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => {
                let cpu = s.to_cpu()?;
                let out = f(&cpu, layout)?;
                Ok(Self::new_wgpu(WgpuStorage::from_cpu(&out, s.device())?))
            }
            StorageInner::Cuda(s) => {
                let cpu = s.to_cpu()?;
                let out = f(&cpu, layout)?;
                Self::from_cpu_on_device(out, &Device::Cuda(s.device_index))
            }
            StorageInner::Accelerator { device, storage } => {
                Ok(Self::new_accelerator(device.clone(), f(storage, layout)?))
            }
            StorageInner::Meta { .. } => {
                let n = layout.shape().iter().product();
                Ok(Self::new_meta(out_dtype, n))
            }
        }
    }

    pub fn bitwise_op(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: BitOp,
    ) -> Result<Self> {
        let dtype = self.dtype();
        self.pair_op(
            lhs_layout,
            rhs,
            rhs_layout,
            "bitwise_op",
            dtype,
            move |a, la, b, lb| a.bitwise_op(la, b, lb, op),
        )
    }

    pub fn logical_op(
        &self,
        lhs_layout: &Layout,
        rhs: &Self,
        rhs_layout: &Layout,
        op: LogicOp,
    ) -> Result<Self> {
        if let (StorageInner::Cuda(a), StorageInner::Cuda(b)) = (&*self.read(), &*rhs.read())
            && let Some(out) = a.logic_gpu(lhs_layout, b, rhs_layout, op)?
        {
            return Ok(Self::new_cuda(out));
        }
        self.pair_op(
            lhs_layout,
            rhs,
            rhs_layout,
            "logical_op",
            DType::Bool,
            move |a, la, b, lb| a.logical_op(la, b, lb, op),
        )
    }

    pub fn pred_op(&self, layout: &Layout, op: PredOp) -> Result<Self> {
        self.unary_op_cpu(layout, DType::Bool, move |s, l| s.pred_op(l, op))
    }

    pub fn bitwise_not(&self, layout: &Layout) -> Result<Self> {
        let dtype = self.dtype();
        self.unary_op_cpu(layout, dtype, |s, l| s.bitwise_not(l))
    }

    pub fn where_cond(
        cond: &Self,
        cond_l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let dev = t.device();
        if cond.device() != dev || f.device() != dev {
            return Err(Error::DeviceMismatch {
                op: "where_cond",
                lhs: dev.to_string(),
                rhs: format!("cond={}, f={}", cond.device(), f.device()),
            });
        }
        if dev.is_meta() {
            let n = t_l.shape().iter().product();
            return Ok(Self::new_meta(t.dtype(), n));
        }
        #[cfg(feature = "wgpu")]
        if let (StorageInner::Wgpu(c), StorageInner::Wgpu(a), StorageInner::Wgpu(b)) = (&*cond.read(), &*t.read(), &*f.read())
            && let Some(out) = WgpuStorage::where_cond(c, cond_l, a, t_l, b, f_l)?
        {
            return Ok(Self::new_wgpu(out));
        }
        if let (StorageInner::Cuda(c), StorageInner::Cuda(a), StorageInner::Cuda(b)) = (&*cond.read(), &*t.read(), &*f.read())
            && let Some(out) = cuda_kernels::CudaStorage::where_gpu(c, cond_l, a, t_l, b, f_l)?
        {
            return Ok(Self::new_cuda(out));
        }
        let cc = cond.cpu_ref()?;
        let ct = t.cpu_ref()?;
        let cf = f.cpu_ref()?;
        let out = CpuStorage::where_cond(&cc, cond_l, &ct, t_l, &cf, f_l)?;
        // a CUDA operand must come back as CUDA storage: an `Accelerator` one on the same device index fails the next matmul
        Self::from_cpu_on_device(out, &dev)
    }

    pub fn reduce_op(&self, layout: &Layout, axes: &[usize], op: ReduceOp) -> Result<Self> {
        let _prof = crate::profiler::scope("reduce_op", layout.shape());
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(s.reduce_op(layout, axes, op)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Ok(Self::new_wgpu(s.reduce_op(layout, axes, op)?)),
            StorageInner::Cuda(s) => {
                if let Some(out) = s.reduce_gpu(layout, axes, op)? {
                    return Ok(Self::new_cuda(out));
                }
                let cpu_out = s.to_cpu()?.reduce_op(layout, axes, op)?;
                Self::from_cpu_on_device(cpu_out, &Device::Cuda(s.device_index))
            }
            StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                device.clone(),
                storage.reduce_op(layout, axes, op)?,
            )),
            StorageInner::Meta { dtype, .. } => {
                let mut out_shape = layout.shape().to_vec();
                for &ax in axes {
                    if ax < out_shape.len() {
                        out_shape[ax] = 1;
                    }
                }
                let n = Self::meta_len("reduce", &out_shape)?;
                let out_dtype = match op {
                    ReduceOp::ArgMax | ReduceOp::ArgMin => DType::I64,
                    _ => *dtype,
                };
                Ok(Self::new_meta(out_dtype, n))
            }
        }
    }

    /// Checked element count for Meta (shape-only) storages. Meta shapes can be
    /// symbolic/huge, so the product must not wrap (debug panic) or truncate.
    fn meta_len(op: &'static str, shape: &[usize]) -> Result<usize> {
        shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                Error::Msg(format!(
                    "{op}: meta shape {shape:?} element count overflows"
                ))
            })
    }

    /// Meta-device matmul output shape with the same validation as the Cpu
    /// kernel (contracting-dim equality + batch broadcasting) so shape
    /// inference on `Device::Meta` never diverges from real execution.
    fn meta_matmul_shape(lhs_l: &Layout, rhs_l: &Layout) -> Result<Vec<usize>> {
        let (ar, br) = (lhs_l.ndim(), rhs_l.ndim());
        let (m, k) = (lhs_l.shape()[ar - 2], lhs_l.shape()[ar - 1]);
        let (k2, n) = (rhs_l.shape()[br - 2], rhs_l.shape()[br - 1]);
        if k != k2 {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: lhs_l.shape().to_vec(),
                got: rhs_l.shape().to_vec(),
            });
        }
        let mut shape =
            broadcast_shapes("matmul", &lhs_l.shape()[..ar - 2], &rhs_l.shape()[..br - 2])?;
        shape.push(m);
        shape.push(n);
        Ok(shape)
    }

    pub fn matmul(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<(Self, Vec<usize>)> {
        let _prof = crate::profiler::scope("matmul", lhs_l.shape());
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch {
                op: "matmul",
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            });
        }
        if lhs_l.ndim() < 2 || rhs_l.ndim() < 2 {
            return Err(Error::Msg(format!(
                "matmul: both operands need rank >= 2, got {:?} and {:?}",
                lhs_l.shape(),
                rhs_l.shape()
            )));
        }
        if self.ptr_eq(rhs) {
            let guard = self.read();
            return match &*guard {
                StorageInner::Cpu(s) => {
                    let (out, shape) = s.matmul(lhs_l, s, rhs_l)?;
                    Ok((Self::new_cpu(out), shape))
                }
                #[cfg(feature = "wgpu")]
                StorageInner::Wgpu(s) => {
                    let (out, shape) = s.matmul(lhs_l, s, rhs_l)?;
                    Ok((Self::new_wgpu(out), shape))
                }
                StorageInner::Cuda(s) => {
                    let (out, shape) = s.matmul(lhs_l, s, rhs_l)?;
                    Ok((Self::new_cuda(out), shape))
                }
                StorageInner::Accelerator { device, storage } => {
                    let (out, shape) = storage.matmul(lhs_l, storage, rhs_l)?;
                    Ok((Self::new_accelerator(device.clone(), out), shape))
                }
                StorageInner::Meta { dtype, .. } => {
                    let shape = Self::meta_matmul_shape(lhs_l, rhs_l)?;
                    let len = Self::meta_len("matmul", &shape)?;
                    Ok((Self::new_meta(*dtype, len), shape))
                }
            };
        }
        match (&*self.read(), &*rhs.read()) {
            (StorageInner::Cpu(a), StorageInner::Cpu(b)) => {
                let (out, shape) = a.matmul(lhs_l, b, rhs_l)?;
                Ok((Self::new_cpu(out), shape))
            }
            #[cfg(feature = "wgpu")]
            (StorageInner::Wgpu(a), StorageInner::Wgpu(b)) => {
                let (out, shape) = a.matmul(lhs_l, b, rhs_l)?;
                Ok((Self::new_wgpu(out), shape))
            }
            (StorageInner::Cuda(a), StorageInner::Cuda(b)) => {
                let (out, shape) = a.matmul(lhs_l, b, rhs_l)?;
                Ok((Self::new_cuda(out), shape))
            }
            (
                StorageInner::Accelerator { device, storage: a },
                StorageInner::Accelerator { storage: b, .. },
            ) => {
                let (out, shape) = a.matmul(lhs_l, b, rhs_l)?;
                Ok((Self::new_accelerator(device.clone(), out), shape))
            }
            (StorageInner::Meta { dtype: ldt, .. }, StorageInner::Meta { dtype: rdt, .. }) => {
                if ldt != rdt {
                    return Err(Error::DTypeMismatch {
                        op: "matmul",
                        expected: *ldt,
                        got: *rdt,
                    });
                }
                let shape = Self::meta_matmul_shape(lhs_l, rhs_l)?;
                let len = Self::meta_len("matmul", &shape)?;
                Ok((Self::new_meta(*ldt, len), shape))
            }
            _ => Err(Error::DeviceMismatch {
                op: "matmul",
                lhs: self.device().to_string(),
                rhs: rhs.device().to_string(),
            }),
        }
    }

    pub fn to_contiguous(&self, layout: &Layout) -> Result<Self> {
        let _prof = crate::profiler::scope("to_contiguous", layout.shape());
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(s.to_contiguous(layout)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Ok(Self::new_wgpu(s.to_contiguous(layout)?)),
            StorageInner::Cuda(s) => Ok(Self::new_cuda(s.to_contiguous(layout)?)),
            StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                device.clone(),
                storage.to_contiguous(layout)?,
            )),
            StorageInner::Meta { dtype, .. } => {
                let n = Self::meta_len("to_contiguous", layout.shape())?;
                Ok(Self::new_meta(*dtype, n))
            }
        }
    }

    pub fn to_dtype(&self, layout: &Layout, dst: DType) -> Result<Self> {
        let _prof = crate::profiler::scope("to_dtype", layout.shape());
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(s.to_dtype(layout, dst)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Ok(Self::new_wgpu(s.to_dtype(layout, dst)?)),
            StorageInner::Cuda(s) => {
                if let Some(out) = s.cast_gpu(layout, dst)? {
                    return Ok(Self::new_cuda(out));
                }
                let cpu_out = s.to_cpu()?.to_dtype(layout, dst)?;
                Self::from_cpu_on_device(cpu_out, &Device::Cuda(s.device_index))
            }
            StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                device.clone(),
                storage.to_dtype(layout, dst)?,
            )),
            StorageInner::Meta { .. } => {
                let n = Self::meta_len("to_dtype", layout.shape())?;
                Ok(Self::new_meta(dst, n))
            }
        }
    }

    pub fn to_device(&self, layout: &Layout, dst: &Device) -> Result<Self> {
        if &self.device() == dst && layout.is_contiguous() && layout.offset() == 0 {
            return Ok(self.clone());
        }
        if dst.is_meta() {
            let n = Self::meta_len("to_device", layout.shape())?;
            return Ok(Self::new_meta(self.dtype(), n));
        }
        if self.device().is_meta() {
            return Err(Error::Msg(
                "Cannot move Meta tensor to physical device without data materialization".into(),
            ));
        }
        let cpu = self.cpu_ref()?.to_contiguous(layout)?;
        Self::from_cpu_on_device(cpu, dst)
    }

    pub fn view_scatter(&self, src_layout: &Layout, spec: &ViewSpec) -> Result<Self> {
        let dev = self.device();
        if dev.is_meta() {
            let n = Self::meta_len("view_scatter", &spec.dst_shape)?;
            return Ok(Self::new_meta(self.dtype(), n));
        }
        if let Some(out) = self.cuda_view_scatter(src_layout, spec)? {
            return Ok(out);
        }
        let cpu = self.cpu_ref()?;
        let out = cpu.view_scatter(src_layout, spec)?;
        Self::from_cpu_on_device(out, &dev)
    }

    pub fn index_select_vals(&self, layout: &Layout, ax: usize, idx: &[i64]) -> Result<Self> {
        let _prof = crate::profiler::scope("index_select", layout.shape());
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(s.index_select_vals(layout, ax, idx)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => {
                if let Some(gpu_out) = s.index_select(layout, ax, idx)? {
                    return Ok(Self::new_wgpu(gpu_out));
                }
                let dev = s.device().clone();
                let cpu = s.to_cpu()?.index_select_vals(layout, ax, idx)?;
                Ok(Self::new_wgpu(WgpuStorage::from_cpu(&cpu, &dev)?))
            }
            StorageInner::Cuda(s) => Ok(Self::new_cuda(s.index_select_vals(layout, ax, idx)?)),
            StorageInner::Accelerator { device, storage } => Ok(Self::new_accelerator(
                device.clone(),
                storage.index_select_vals(layout, ax, idx)?,
            )),
            StorageInner::Meta { dtype, .. } => {
                let mut out_shape = layout.shape().to_vec();
                out_shape[ax] = idx.len();
                let n = Self::meta_len("index_select", &out_shape)?;
                Ok(Self::new_meta(*dtype, n))
            }
        }
    }
}
