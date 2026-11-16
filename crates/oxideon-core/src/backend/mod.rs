//! Pluggable compute backends (`Cpu`, accelerated devices, and optional `Wgpu`) and reference-counted [`Storage`].

pub mod cpu;
pub mod cuda_driver;
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
            StorageInner::Meta { .. } => 0,
        };
        crate::memory::on_alloc(bytes);
        Arc::new(Self {
            lock: RwLock::new(inner),
            bytes,
            dlpack_pins: std::sync::atomic::AtomicUsize::new(0),
        })
    }
}

impl Drop for Tracked {
    fn drop(&mut self) {
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
    /// Fused single-pass last-axis softmax on contiguous CPU storage (returns `None` to fall back).
    pub(crate) fn try_softmax_last_axis(&self, layout: &Layout) -> Option<Self> {
        let out_inner = {
            let guard = self.0.read().unwrap_or_else(|p| p.into_inner());
            match &*guard {
                StorageInner::Cpu(s) => StorageInner::Cpu(s.softmax_last_axis(layout)?),
                StorageInner::Accelerator { device, storage } => StorageInner::Accelerator {
                    device: device.clone(),
                    storage: storage.softmax_last_axis(layout)?,
                },
                _ => return None,
            }
        };
        Some(Self(Tracked::new(out_inner)))
    }

    pub fn new_cpu(s: CpuStorage) -> Self {
        Self(Tracked::new(StorageInner::Cpu(s)))
    }

    #[cfg(feature = "wgpu")]
    pub fn new_wgpu(s: WgpuStorage) -> Self {
        Self(Tracked::new(StorageInner::Wgpu(s)))
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
            StorageInner::Accelerator { storage, .. } => storage.dtype(),
            StorageInner::Meta { dtype, .. } => *dtype,
        }
    }

    pub fn device(&self) -> Device {
        match &*self.read() {
            StorageInner::Cpu(_) => Device::Cpu,
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Device::Wgpu(s.device().clone()),
            StorageInner::Accelerator { device, .. } => device.clone(),
            StorageInner::Meta { .. } => Device::Meta,
        }
    }

    pub fn len(&self) -> usize {
        match &*self.read() {
            StorageInner::Cpu(s) => s.len(),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => s.len(),
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
            StorageInner::Accelerator { storage, .. } => Ok(storage.clone()),
            StorageInner::Meta { .. } => Err(Error::Msg(
                "Cannot access raw memory of a Meta tensor (device='meta')".into(),
            )),
        }
    }

    pub fn from_slice<T: Element>(data: &[T], device: &Device) -> Result<Self> {
        let cpu = CpuStorage::from_slice(data)?;
        match device {
            Device::Cpu => Ok(Self::new_cpu(cpu)),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(dev) => Ok(Self::new_wgpu(WgpuStorage::from_cpu(&cpu, dev)?)),
            Device::Meta => Ok(Self::new_meta(cpu.dtype(), cpu.len())),
            dev => Ok(Self::new_accelerator(dev.clone(), cpu)),
        }
    }

    pub fn full(numel: usize, val: f64, dtype: DType, device: &Device) -> Result<Self> {
        if matches!(device, Device::Meta) {
            return Ok(Self::new_meta(dtype, numel)); // no data: do not build a buffer just to drop it
        }
        let cpu = CpuStorage::full(numel, val, dtype)?;
        match device {
            Device::Cpu => Ok(Self::new_cpu(cpu)),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(dev) => Ok(Self::new_wgpu(WgpuStorage::from_cpu(&cpu, dev)?)),
            dev => Ok(Self::new_accelerator(dev.clone(), cpu)),
        }
    }

    pub fn unary_op(&self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        let _prof = crate::profiler::scope("unary_op", layout.shape());
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(s.unary_op(layout, op)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Ok(Self::new_wgpu(s.unary_op(layout, op)?)),
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
        let cc = cond.cpu_ref()?;
        let ct = t.cpu_ref()?;
        let cf = f.cpu_ref()?;
        let out = CpuStorage::where_cond(&cc, cond_l, &ct, t_l, &cf, f_l)?;
        match &dev {
            Device::Cpu => Ok(Self::new_cpu(out)),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(d) => Ok(Self::new_wgpu(WgpuStorage::from_cpu(&out, d)?)),
            Device::Meta => Ok(Self::new_meta(out.dtype(), out.len())),
            d => Ok(Self::new_accelerator(d.clone(), out)),
        }
    }

    pub fn reduce_op(&self, layout: &Layout, axes: &[usize], op: ReduceOp) -> Result<Self> {
        let _prof = crate::profiler::scope("reduce_op", layout.shape());
        match &*self.read() {
            StorageInner::Cpu(s) => Ok(Self::new_cpu(s.reduce_op(layout, axes, op)?)),
            #[cfg(feature = "wgpu")]
            StorageInner::Wgpu(s) => Ok(Self::new_wgpu(s.reduce_op(layout, axes, op)?)),
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
        match dst {
            Device::Cpu => Ok(Self::new_cpu(cpu)),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(dev) => Ok(Self::new_wgpu(WgpuStorage::from_cpu(&cpu, dev)?)),
            Device::Meta => Ok(Self::new_meta(cpu.dtype(), cpu.len())),
            dev => Ok(Self::new_accelerator(dev.clone(), cpu)),
        }
    }

    pub fn view_scatter(&self, src_layout: &Layout, spec: &ViewSpec) -> Result<Self> {
        let dev = self.device();
        if dev.is_meta() {
            let n = Self::meta_len("view_scatter", &spec.dst_shape)?;
            return Ok(Self::new_meta(self.dtype(), n));
        }
        let cpu = self.cpu_ref()?;
        let out = cpu.view_scatter(src_layout, spec)?;
        match &dev {
            Device::Cpu => Ok(Self::new_cpu(out)),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(d) => Ok(Self::new_wgpu(WgpuStorage::from_cpu(&out, d)?)),
            Device::Meta => Ok(Self::new_meta(out.dtype(), out.len())),
            d => Ok(Self::new_accelerator(d.clone(), out)),
        }
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
