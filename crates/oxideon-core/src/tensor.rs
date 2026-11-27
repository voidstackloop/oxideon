//! Core [`Tensor`] handle (`Arc`-backed, `Send + Sync`) and the higher-order autograd engine.
//!
//! Because backward closures are expressed directly in terms of [`Tensor`] operations,
//! running backpropagation with `create_graph = true` records the backward pass itself
//! into the computation graph—giving exact second- and higher-order derivatives!

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::backend::{CpuStorage, Device, Storage};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::{Layout, broadcast_shapes};
use crate::rng;

/// A backward function that takes the upstream gradient `Tensor` and computes the
/// gradient `Tensor` for each input `Tensor` of the node.
pub type BackwardFn = Box<dyn Fn(&Tensor) -> Result<Vec<Option<Tensor>>> + Send + Sync + 'static>;

/// A gradient hook called whenever a gradient is computed for a tensor.
/// It may return `Some(modified_grad)` to replace the gradient or `None` to leave it unchanged.
pub type GradHook = Box<dyn Fn(&Tensor) -> Result<Option<Tensor>> + Send + Sync + 'static>;

/// Locks a grad/hook `Mutex`, recovering the inner value when a panicking holder
/// poisoned it instead of panicking again (which would brick every future
/// `grad()`/`backward()`/`register_hook()` call on the tensor).
fn mutex_guard<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) struct Node {
    pub(crate) op: &'static str,
    pub(crate) inputs: Vec<Tensor>,
    pub(crate) backward: BackwardFn,
}

struct TensorInner {
    id: usize,
    storage: Storage,
    layout: Layout,
    requires_grad: AtomicBool,
    retain_grad: AtomicBool,
    grad: Mutex<Option<Tensor>>,
    /// The op that produced this tensor (`None` = leaf). Behind a lock only so `backward(release)` can swap it for a stub
    /// that holds no inputs or saved tensors once it has been used; never held while a backward closure runs.
    node: Mutex<Option<Arc<Node>>>,
    hooks: Mutex<Vec<GradHook>>,
}

/// An n-dimensional strided tensor with multi-dtype storage and reverse-mode (and
/// higher-order) automatic differentiation.
#[derive(Clone)]
pub struct Tensor(Arc<TensorInner>);

/// Iterative deep-graph drop: a 20k chained graph overflows the call stack
/// when dropped recursively (each `Node.inputs` level adds drop frames), so
/// the last-`Arc` drop unwraps owned children onto an explicit heap worklist
/// instead. Shared children (live `Arc`s elsewhere) only decrement.
/// Tensors captured opaquely inside `backward`/hook closures still drop
/// inline when their `Box` drops; capture chains are bounded by construction
/// (closures capture forward inputs, not graph spines), so only the spine
/// needs the worklist.
impl Drop for TensorInner {
    fn drop(&mut self) {
        fn push_children(inner: &mut TensorInner, stack: &mut Vec<Tensor>) {
            let node = match inner.node.get_mut() {
                Ok(n) => n.take(),
                Err(p) => p.into_inner().take(),
            };
            if let Some(node) = node.and_then(|n| Arc::try_unwrap(n).ok()) {
                let Node {
                    inputs, backward, ..
                } = node;
                // `inputs` stay owned below while `backward` drops, so tensors
                // the closure shares with the spine only decrement here.
                stack.extend(inputs);
                drop(backward);
            }
            // Best-effort: if our own grad slot is locked (a tensor dropped
            // while its `backward()` guard is held), leave it for the normal
            // field drop rather than deadlocking on `lock()`.
            if let Ok(mut slot) = inner.grad.try_lock() {
                if let Some(g) = slot.take() {
                    stack.push(g);
                }
            }
        }

        let mut stack: Vec<Tensor> = Vec::new();
        push_children(self, &mut stack);
        while let Some(t) = stack.pop() {
            match Arc::try_unwrap(t.0) {
                Ok(mut owned) => {
                    push_children(&mut owned, &mut stack);
                    drop(owned);
                }
                Err(shared) => drop(shared),
            }
        }
    }
}

static NEXT_TENSOR_ID: AtomicUsize = AtomicUsize::new(1);

fn next_id() -> usize {
    NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed)
}

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns `true` if autograd graph recording is enabled on the current thread.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Sets autograd graph recording on the current thread and returns a guard that
/// restores the previous state when dropped.
pub fn set_grad_enabled(enabled: bool) -> GradModeGuard {
    let prev = GRAD_ENABLED.with(|g| g.replace(enabled));
    GradModeGuard { prev }
}

pub struct GradModeGuard {
    prev: bool,
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.prev));
    }
}

/// Runs `f` with gradient tracking disabled.
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = set_grad_enabled(false);
    f()
}

/// Runs `f` with gradient tracking explicitly enabled.
pub fn enable_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = set_grad_enabled(true);
    f()
}

/// User-defined differentiable operation.
pub trait CustomOp: Send + Sync + 'static {
    fn name(&self) -> &'static str {
        "custom_op"
    }
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor>;
    fn backward(&self, inputs: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
impl Tensor {
    pub(crate) fn from_parts(
        storage: Storage,
        layout: Layout,
        requires_grad: bool,
        node: Option<Node>,
    ) -> Self {
        Self(Arc::new(TensorInner {
            id: next_id(),
            storage,
            layout,
            requires_grad: AtomicBool::new(requires_grad),
            retain_grad: AtomicBool::new(false),
            grad: Mutex::new(None),
            node: Mutex::new(node.map(Arc::new)),
            hooks: Mutex::new(Vec::new()),
        }))
    }

    /// Creates a new leaf tensor from `storage` and `layout`.
    ///
    /// This is unchecked: the caller must guarantee the layout fits the
    /// storage. Prefer [`Tensor::try_from_storage`], which verifies the
    /// layout's touched range against the storage length and returns `Err`
    /// instead of building a tensor whose kernels would panic.
    pub fn from_storage(storage: Storage, layout: Layout) -> Self {
        Self::from_parts(storage, layout, false, None)
    }

    /// Fallible [`Tensor::from_storage`]: rejects layouts that read outside
    /// the storage (including negative-stride underflow and offset overflow).
    pub fn try_from_storage(storage: Storage, layout: Layout) -> Result<Self> {
        layout.check_storage_bounds(storage.len(), "from_storage")?;
        Ok(Self::from_parts(storage, layout, false, None))
    }

    /// Creates a tensor produced by a differentiable operation.
    pub(crate) fn from_op(
        storage: Storage,
        layout: Layout,
        op: &'static str,
        inputs: Vec<Tensor>,
        backward: impl Fn(&Tensor) -> Result<Vec<Option<Tensor>>> + Send + Sync + 'static,
    ) -> Self {
        if crate::profiler::anomaly_enabled() && storage.dtype().is_floating_point() && !storage.all_finite(&layout) {
            crate::profiler::note_non_finite(op, layout.shape());
        }
        let req = is_grad_enabled()
            && storage.dtype().is_floating_point()
            && inputs.iter().any(Tensor::requires_grad);
        let node = req.then(|| Node {
            op,
            inputs,
            backward: Box::new(backward),
        });
        Self::from_parts(storage, layout, req, node)
    }

    /// Creates a contiguous tensor on `Device::Cpu` from a slice and shape.
    pub fn from_slice<T: Element>(data: &[T], shape: &[usize]) -> Result<Self> {
        Self::from_slice_on(data, shape, &Device::Cpu)
    }

    pub fn from_slice_on<T: Element>(data: &[T], shape: &[usize], device: &Device) -> Result<Self> {
        let expected = crate::layout::checked_numel(shape)?;
        if data.len() != expected {
            return Err(Error::ShapeMismatch {
                op: "from_slice",
                expected: shape.to_vec(),
                got: vec![data.len()],
            });
        }
        let storage = Storage::from_slice(data, device)?;
        Ok(Self::from_storage(storage, Layout::contiguous(shape)))
    }

    pub fn from_vec<T: Element>(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        Self::from_slice(&data, shape)
    }

    /// Creates a contiguous `C64` tensor from interleaved `[re, im]` pairs on CPU.
    pub fn from_c64_vec(data: Vec<[f32; 2]>, shape: &[usize]) -> Result<Self> {
        let n = crate::layout::checked_numel(shape)?;
        if data.len() != n {
            return Err(Error::Msg(format!(
                "from_c64_vec: data length {} does not match shape {:?} ({n})",
                data.len(),
                shape
            )));
        }
        let storage = Storage::new_cpu(CpuStorage::C64(data));
        Ok(Self::from_storage(storage, Layout::contiguous(shape)))
    }

    /// Creates a contiguous `C128` tensor from interleaved `[re, im]` pairs on CPU.
    pub fn from_c128_vec(data: Vec<[f64; 2]>, shape: &[usize]) -> Result<Self> {
        let n = crate::layout::checked_numel(shape)?;
        if data.len() != n {
            return Err(Error::Msg(format!(
                "from_c128_vec: data length {} does not match shape {:?} ({n})",
                data.len(),
                shape
            )));
        }
        let storage = Storage::new_cpu(CpuStorage::C128(data));
        Ok(Self::from_storage(storage, Layout::contiguous(shape)))
    }

    /// Extracts interleaved `[re, im]` pairs for a `C64` tensor (converting `C128` or real dtypes).
    pub fn to_c64_vec(&self) -> Result<Vec<[f32; 2]>> {
        let t = if self.dtype() == DType::C64 {
            self.contiguous()?
        } else {
            self.to_dtype(DType::C64)?
        };
        let cpu = t.to_device(&Device::Cpu)?;
        let guard = cpu.storage().read();
        match &*guard {
            crate::backend::StorageInner::Cpu(CpuStorage::C64(v)) => {
                if let Some(r) = cpu.layout().contiguous_range_in(v.len()) {
                    Ok(v[r].to_vec())
                } else {
                    Ok(cpu.layout().storage_offsets().map(|i| v[i]).collect())
                }
            }
            _ => unreachable!(),
        }
    }

    /// Extracts interleaved `[re, im]` pairs for a `C128` tensor (converting `C64` or real dtypes).
    pub fn to_c128_vec(&self) -> Result<Vec<[f64; 2]>> {
        let t = if self.dtype() == DType::C128 {
            self.contiguous()?
        } else {
            self.to_dtype(DType::C128)?
        };
        let cpu = t.to_device(&Device::Cpu)?;
        let guard = cpu.storage().read();
        match &*guard {
            crate::backend::StorageInner::Cpu(CpuStorage::C128(v)) => {
                if let Some(r) = cpu.layout().contiguous_range_in(v.len()) {
                    Ok(v[r].to_vec())
                } else {
                    Ok(cpu.layout().storage_offsets().map(|i| v[i]).collect())
                }
            }
            _ => unreachable!(),
        }
    }

    /// Returns the real part of a tensor (`F32` for `C64`, `F64` for `C128`, or a clone for real tensors).
    pub fn real(&self) -> Result<Self> {
        match self.dtype() {
            DType::C64 => {
                let pairs = self.to_c64_vec()?;
                Self::from_vec(pairs.into_iter().map(|[r, _]| r).collect(), self.shape())
            }
            DType::C128 => {
                let pairs = self.to_c128_vec()?;
                Self::from_vec(pairs.into_iter().map(|[r, _]| r).collect(), self.shape())
            }
            _ => Ok(self.clone()),
        }
    }

    /// Returns the imaginary part of a tensor (`F32` for `C64`, `F64` for `C128`, or zeros for real tensors).
    pub fn imag(&self) -> Result<Self> {
        match self.dtype() {
            DType::C64 => {
                let pairs = self.to_c64_vec()?;
                Self::from_vec(pairs.into_iter().map(|[_, i]| i).collect(), self.shape())
            }
            DType::C128 => {
                let pairs = self.to_c128_vec()?;
                Self::from_vec(pairs.into_iter().map(|[_, i]| i).collect(), self.shape())
            }
            dt => Self::zeros(self.shape(), dt, &self.device()),
        }
    }

    /// Returns the complex conjugate of a tensor (`C64`/`C128` negate imag; real tensors clone).
    pub fn conj(&self) -> Result<Self> {
        match self.dtype() {
            DType::C64 => {
                let pairs = self.to_c64_vec()?;
                Self::from_c64_vec(pairs.into_iter().map(|[r, i]| [r, -i]).collect(), self.shape())
            }
            DType::C128 => {
                let pairs = self.to_c128_vec()?;
                Self::from_c128_vec(pairs.into_iter().map(|[r, i]| [r, -i]).collect(), self.shape())
            }
            _ => Ok(self.clone()),
        }
    }

    /// Creates a 0-dimensional scalar tensor on CPU.
    pub fn scalar<T: Element>(value: T) -> Result<Self> {
        Self::from_slice(&[value], &[])
    }

    pub fn full(shape: &[usize], value: f64, dtype: DType, device: &Device) -> Result<Self> {
        // A meta tensor holds no data, so only real storage is bounded. Whether the OS refuses a huge request is not
        // reliable (Linux with permissive overcommit hands out the pages and the zero-fill ends in the OOM killer).
        let n = if device.is_meta() {
            crate::layout::checked_numel(shape)?
        } else {
            crate::limits::check_elements("tensor element count", shape)?
        };
        let storage = Storage::full(n, value, dtype, device)?;
        Ok(Self::from_storage(storage, Layout::contiguous(shape)))
    }

    pub fn empty(shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        if device.is_meta() {
            let n = crate::layout::checked_numel(shape)?;
            Ok(Self::from_storage(
                Storage::new_meta(dtype, n),
                Layout::contiguous(shape),
            ))
        } else {
            Self::zeros(shape, dtype, device)
        }
    }

    pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        Self::full(shape, 0.0, dtype, device)
    }

    pub fn ones(shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        Self::full(shape, 1.0, dtype, device)
    }

    pub fn zeros_on(shape: &[usize], device: &Device) -> Result<Self> {
        Self::zeros(shape, DType::F32, device)
    }

    pub fn ones_on(shape: &[usize], device: &Device) -> Result<Self> {
        Self::ones(shape, DType::F32, device)
    }

    pub fn zeros_f32(shape: &[usize]) -> Result<Self> {
        Self::zeros(shape, DType::F32, &Device::Cpu)
    }

    pub fn ones_f32(shape: &[usize]) -> Result<Self> {
        Self::ones(shape, DType::F32, &Device::Cpu)
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Self::zeros(self.shape(), self.dtype(), &self.device())
    }

    pub fn ones_like(&self) -> Result<Self> {
        Self::ones(self.shape(), self.dtype(), &self.device())
    }

    pub fn full_like(&self, value: f64) -> Result<Self> {
        Self::full(self.shape(), value, self.dtype(), &self.device())
    }

    /// Standard normal random tensor (`f32` on CPU by default).
    pub fn randn(shape: &[usize]) -> Result<Self> {
        Self::randn_with(shape, DType::F32, &Device::Cpu)
    }

    pub fn randn_on(shape: &[usize], device: &Device) -> Result<Self> {
        Self::randn_with(shape, DType::F32, device)
    }

    pub fn randn_with(shape: &[usize], dtype: DType, device: &Device) -> Result<Self> {
        let n = crate::limits::check_elements("tensor element count", shape)?;
        // CUDA: sampled by a kernel from a key drawn from the global generator (so `manual_seed` stays deterministic)
        if matches!(device, Device::Cuda(_)) {
            let key = rng::with_rng(|r| r.next_u64());
            if let Some(storage) = Storage::cuda_rand(device, dtype, n, true, key, 0.0, 0.0)? {
                return Ok(Self::from_storage(storage, Layout::contiguous(shape)));
            }
        }
        let mut data: Vec<f32> = crate::layout::try_with_capacity(n)?;
        rng::with_rng(|r| data.extend((0..n).map(|_| r.normal())));
        debug_assert_eq!(data.len(), n);
        let t = if device.is_cpu() {
            // Move the filled buffer straight into storage (no second copy).
            Self::from_storage(
                Storage::new_cpu(CpuStorage::F32(data)),
                Layout::contiguous(shape),
            )
        } else {
            Self::from_slice_on(&data, shape, device)?
        };
        if dtype == DType::F32 {
            Ok(t)
        } else {
            t.to_dtype(dtype)
        }
    }

    /// Uniform samples in `[low, high)` created on `device` (a kernel on CUDA, the host generator elsewhere).
    pub fn rand_uniform_on(shape: &[usize], low: f32, high: f32, dtype: DType, device: &Device) -> Result<Self> {
        if matches!(device, Device::Cuda(_)) {
            let n = crate::limits::check_elements("tensor element count", shape)?;
            let key = rng::with_rng(|r| r.next_u64());
            if let Some(storage) = Storage::cuda_rand(device, dtype, n, false, key, f64::from(low), f64::from(high))? {
                return Ok(Self::from_storage(storage, Layout::contiguous(shape)));
            }
        }
        let t = Self::rand_uniform(shape, low, high)?.to_device(device)?;
        if dtype == DType::F32 { Ok(t) } else { t.to_dtype(dtype) }
    }

    pub fn rand_uniform(shape: &[usize], low: f32, high: f32) -> Result<Self> {
        let n = crate::limits::check_elements("tensor element count", shape)?;
        let mut data: Vec<f32> = crate::layout::try_with_capacity(n)?;
        rng::with_rng(|r| data.extend((0..n).map(|_| r.uniform(low, high))));
        debug_assert_eq!(data.len(), n);
        // Move the filled buffer straight into storage (no second copy).
        let storage = Storage::new_cpu(CpuStorage::F32(data));
        Ok(Self::from_storage(storage, Layout::contiguous(shape)))
    }

    /// 1D tensor `[start, start + step, ...)` in `[start, end)`.
    pub fn arange(start: f64, end: f64, step: f64, dtype: DType) -> Result<Self> {
        if step == 0.0 || (end > start && step < 0.0) || (end < start && step > 0.0) {
            return Err(Error::Msg(format!(
                "arange: invalid range ({start}, {end}, {step})"
            )));
        }
        let n = ((end - start) / step).ceil().max(0.0) as usize;
        crate::limits::check_at_most("arange length", n, crate::limits::MAX_ELEMENTS)?;
        let mut vals: Vec<f64> = crate::layout::try_with_capacity(n)?;
        vals.extend((0..n).map(|i| start + (i as f64) * step));
        let s = CpuStorage::F64(vals).to_dtype(&Layout::contiguous(vec![n]), dtype)?;
        Ok(Self::from_storage(
            Storage::new_cpu(s),
            Layout::contiguous(vec![n]),
        ))
    }

    /// `n` evenly spaced values from `start` to `end` inclusive.
    pub fn linspace(start: f64, end: f64, n: usize, dtype: DType) -> Result<Self> {
        let step = if n > 1 {
            (end - start) / (n - 1) as f64
        } else {
            0.0
        };
        crate::limits::check_at_most("linspace length", n, crate::limits::MAX_ELEMENTS)?;
        let mut vals: Vec<f64> = crate::layout::try_with_capacity(n)?;
        vals.extend((0..n).map(|i| start + (i as f64) * step));
        let s = CpuStorage::F64(vals).to_dtype(&Layout::contiguous(vec![n]), dtype)?;
        Ok(Self::from_storage(
            Storage::new_cpu(s),
            Layout::contiguous(vec![n]),
        ))
    }

    /// `n x n` identity matrix.
    pub fn eye(n: usize, dtype: DType, device: &Device) -> Result<Self> {
        let mut data = crate::layout::try_filled(
            crate::limits::check_elements("eye element count", &[n, n])?,
            0.0f32,
        )?;
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let t = Self::from_slice_on(&data, &[n, n], device)?;
        if dtype == DType::F32 {
            Ok(t)
        } else {
            t.to_dtype(dtype)
        }
    }

    /// Applies a user-defined [`CustomOp`].
    pub fn apply_custom<O: CustomOp>(op: O, inputs: &[Tensor]) -> Result<Self> {
        let op = Arc::new(op);
        let out = no_grad(|| op.forward(inputs))?;
        let saved: Vec<Tensor> = inputs.to_vec();
        let op_clone = Arc::clone(&op);
        Ok(Self::from_op(
            out.storage().clone(),
            out.layout().clone(),
            op.name(),
            inputs.to_vec(),
            move |grad_out| op_clone.backward(&saved, grad_out),
        ))
    }
}

// ---------------------------------------------------------------------------
// Metadata & Accessors
// ---------------------------------------------------------------------------
impl Tensor {
    #[inline]
    pub fn id(&self) -> usize {
        self.0.id
    }

    pub(crate) fn trace_node(&self) -> Option<(&'static str, Vec<Tensor>)> {
        self.node().map(|n| (n.op, n.inputs.clone()))
    }

    /// The producing op (a cheap `Arc` clone; the lock is released before returning).
    fn node(&self) -> Option<Arc<Node>> {
        mutex_guard(&self.0.node).clone()
    }

    /// Replaces a used node by a stub with no inputs and no saved tensors (`backward(release)`): the tensor stays a non-leaf,
    /// and a second backward through it fails clearly instead of silently reusing freed state.
    fn release_node(&self) {
        let mut slot = mutex_guard(&self.0.node);
        if let Some(op) = slot.as_ref().map(|n| n.op) {
            *slot = Some(Arc::new(Node {
                op,
                inputs: Vec::new(),
                backward: Box::new(move |_| {
                    Err(Error::Autograd(format!(
                        "backward through `{op}` a second time: its graph was freed by backward(retain_graph=False)"
                    )))
                }),
            }));
        }
    }

    #[inline]
    pub fn storage(&self) -> &Storage {
        &self.0.storage
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.0.layout
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.0.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.0.layout.strides()
    }

    #[inline]
    pub fn storage_offset(&self) -> usize {
        self.0.layout.offset()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.0.layout.ndim()
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.0.layout.numel()
    }

    #[inline]
    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.0.layout.is_contiguous()
    }

    /// Returns `true` if `self` and `other` share the exact same underlying storage buffer.
    #[inline]
    pub fn shares_storage(&self, other: &Self) -> bool {
        self.0.storage.ptr_eq(&other.0.storage)
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        mutex_guard(&self.0.node).is_none()
    }

    #[inline]
    pub fn op_name(&self) -> Option<&'static str> {
        mutex_guard(&self.0.node).as_ref().map(|n| n.op)
    }

    /// Returns the parent input tensors that produced this tensor in the autograd graph.
    #[inline]
    pub fn op_inputs(&self) -> Vec<Tensor> {
        self.node().map_or_else(Vec::new, |n| n.inputs.clone())
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn requires_grad_(self, flag: bool) -> Self {
        // Lenient builder form (matches `from_op`, which never enables grad on
        // non-float tensors): requesting grad on a non-float tensor is a no-op.
        // Use `set_requires_grad` for the strict, error-returning variant.
        let _ = self.set_requires_grad(flag);
        self
    }

    #[inline]
    pub fn set_requires_grad(&self, flag: bool) -> Result<()> {
        if flag && !self.dtype().is_floating_point() {
            return Err(Error::Msg(format!(
                "only floating-point tensors can require gradients (got {})",
                self.dtype()
            )));
        }
        self.0.requires_grad.store(flag, Ordering::Relaxed);
        Ok(())
    }

    /// Marks a non-leaf tensor so its `.grad()` is populated during `.backward()`.
    #[inline]
    pub fn retain_grad(&self) {
        self.0.retain_grad.store(true, Ordering::Relaxed);
    }

    #[inline]
    pub fn retains_grad(&self) -> bool {
        self.0.retain_grad.load(Ordering::Relaxed)
    }

    /// Registers a hook executed whenever a gradient is computed for this tensor.
    pub fn register_hook(
        &self,
        hook: impl Fn(&Tensor) -> Result<Option<Tensor>> + Send + Sync + 'static,
    ) {
        mutex_guard(&self.0.hooks).push(Box::new(hook));
    }

    /// Returns the accumulated gradient `Tensor` (if any).
    pub fn grad(&self) -> Option<Tensor> {
        mutex_guard(&self.0.grad).clone()
    }

    pub fn set_grad(&self, grad: Option<Tensor>) {
        *mutex_guard(&self.0.grad) = grad;
    }

    pub fn zero_grad(&self) {
        *mutex_guard(&self.0.grad) = None;
    }

    /// Returns a new leaf tensor sharing the same storage and layout (zero-copy!).
    pub fn detach(&self) -> Self {
        Self::from_parts(self.0.storage.clone(), self.0.layout.clone(), false, None)
    }

    /// Copies the logical elements of this tensor (respecting strides & offset) into a `Vec<T>`.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        let cpu = self.0.storage.cpu_ref()?;
        Ok(cpu.to_vec::<T>(&self.0.layout))
    }

    /// Extracts the single scalar value of a 1-element tensor.
    pub fn item<T: Element>(&self) -> Result<T> {
        if self.numel() != 1 {
            return Err(Error::ShapeMismatch {
                op: "item",
                expected: vec![1],
                got: self.shape().to_vec(),
            });
        }
        Ok(self.to_vec::<T>()?[0])
    }

    /// Overwrites this leaf tensor's values in-place from `src` (used by optimizers).
    /// Like `torch.Tensor.copy_`, the source is cast to this tensor's dtype and
    /// device; the destination is never retargeted.
    pub fn assign(&self, src: &Tensor) -> Result<()> {
        if self.shape() != src.shape() {
            return Err(Error::ShapeMismatch {
                op: "assign",
                expected: self.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }
        let contig = src
            .contiguous()?
            .to_dtype(self.dtype())?
            .to_device(&self.device())?;
        let new_inner = contig.0.storage.read().clone();
        if self.0.storage.is_dlpack_pinned() {
            let mut dst = self.0.storage.write();
            match (&mut *dst, &new_inner) {
                (
                    crate::backend::StorageInner::Cpu(d),
                    crate::backend::StorageInner::Cpu(s),
                ) if d.len() == s.len() => {
                    if d.copy_in_place_from(s) {
                        return Ok(());
                    }
                    return Err(Error::Msg(
                        "assign: cannot reallocate tensor storage while exported via DLPack".into(),
                    ));
                }
                _ => {
                    return Err(Error::Msg(
                        "assign: cannot reallocate tensor storage while exported via DLPack".into(),
                    ));
                }
            }
        }
        *self.0.storage.write() = new_inner;
        Ok(())
    }

    /// Retargets this leaf tensor's storage in-place from `src` (used by `Module.to()`).
    ///
    /// Unlike [`Tensor::assign`], the destination's dtype and device FOLLOW
    /// the source: the storage is swapped, not cast back. Requires identical
    /// shapes AND layouts (a `&self` method cannot swap the layout, which has
    /// no interior mutability); `to_dtype`/`to_device` preserve layouts, so
    /// tensors derived from `self` always qualify. `requires_grad` is
    /// preserved and any stored gradient is moved alongside.
    pub fn retarget(&self, src: &Tensor) -> Result<()> {
        if self.0.layout != src.0.layout {
            return Err(Error::Msg(format!(
                "retarget requires identical layouts, got {:?} vs {:?}",
                self.0.layout, src.0.layout
            )));
        }
        if self.0.storage.is_dlpack_pinned() {
            return Err(Error::Msg(
                "retarget: cannot replace tensor storage while exported via DLPack".into(),
            ));
        }
        let new_inner = src.0.storage.read().clone();
        *self.0.storage.write() = new_inner;
        let mut grad = mutex_guard(&self.0.grad);
        if let Some(g) = grad.take() {
            *grad = Some(g.to_dtype(src.dtype())?.to_device(&src.device())?);
        }
        Ok(())
    }

    /// Element-wise approximate equality (`|a - b| <= atol + rtol * |b|`).
    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> Result<bool> {
        // Torch parity: singleton dims broadcast (incompatible shapes error,
        // like torch raises), and exact equality short-circuits so same-signed
        // inf is close while NaN never is (`equal_nan=false`).
        let out_shape = broadcast_shapes("allclose", self.shape(), other.shape())?;
        let la = self.layout().broadcast_to(&out_shape)?;
        let lb = other.layout().broadcast_to(&out_shape)?;
        let a = self.0.storage.cpu_ref()?.to_vec::<f64>(&la);
        let b = other.0.storage.cpu_ref()?.to_vec::<f64>(&lb);
        Ok(a.into_iter().zip(b).all(|(x, y)| {
            if x == y {
                // Exact equality incl. same-signed inf (NaN != NaN falls through).
                true
            } else if !x.is_finite() || !y.is_finite() {
                // Unequal infinities are never close (`inf <= inf` would wrongly
                // pass below); NaN never is either (`equal_nan=false`).
                false
            } else {
                (x - y).abs() <= atol + rtol * y.abs()
            }
        }))
    }
}

// ---------------------------------------------------------------------------
// Reverse-mode & Higher-order Autograd Engine
// ---------------------------------------------------------------------------
impl Tensor {
    /// Backpropagates from a scalar tensor (`numel() == 1`), accumulating gradients
    /// into leaf tensors (and any intermediate tensors with `retain_grad()`).
    pub fn backward(&self) -> Result<()> {
        if self.numel() != 1 {
            return Err(Error::Autograd(format!(
                "backward() requires a scalar output, got shape {:?}",
                self.shape()
            )));
        }
        let seed = Tensor::ones(self.shape(), self.dtype(), &self.device())?;
        self.backward_with_grad(&seed, false)
    }

    /// Backpropagates with an explicit upstream gradient and optional `create_graph`
    /// for higher-order derivatives.
    pub fn backward_with_grad(&self, grad_output: &Tensor, create_graph: bool) -> Result<()> {
        self.backward_impl(grad_output, create_graph, false)
    }

    /// `backward()` that frees the graph as it goes (PyTorch's default, `retain_graph=False`): each op's inputs and saved
    /// tensors are dropped right after its gradient is computed, so activations do not outlive the step (or wait for every
    /// Python reference to the loss to go away). A second backward through the same graph then errors.
    pub fn backward_release(&self, grad_output: &Tensor) -> Result<()> {
        self.backward_impl(grad_output, false, true)
    }

    fn backward_impl(&self, grad_output: &Tensor, create_graph: bool, release: bool) -> Result<()> {
        if !self.requires_grad() {
            return Err(Error::Autograd(
                "called backward on a tensor that does not require grad".into(),
            ));
        }
        if grad_output.numel() != self.numel() {
            return Err(Error::Autograd(format!(
                "backward grad has {} elements for a tensor of shape {:?} ({} elements)",
                grad_output.numel(),
                self.shape(),
                self.numel()
            )));
        }
        let mut order = self.topological_order();
        let mut grads: HashMap<usize, Tensor> = HashMap::new();
        grads.insert(self.id(), grad_output.clone());

        let _grad_guard = set_grad_enabled(create_graph);

        // popped (not iterated) so that, with `release`, each processed tensor's last engine-held reference goes right away
        while let Some(t) = order.pop() {
            let t = &t;
            let Some(mut g) = grads.remove(&t.id()) else {
                if release {
                    t.release_node();
                }
                continue;
            };
            g = t.run_hooks(g)?;
            if g.numel() != t.numel() {
                return Err(Error::Autograd(format!(
                    "gradient for tensor of shape {:?} has {} elements (hook or producer mismatch)",
                    t.shape(),
                    g.numel()
                )));
            }

            if t.is_leaf() || t.retains_grad() {
                let mut slot = mutex_guard(&t.0.grad);
                *slot = Some(match slot.take() {
                    Some(prev) => prev.add(&g)?,
                    None => g.clone(),
                });
            }

            if let Some(node) = t.node() {
                let input_grads = (node.backward)(&g)?;
                if input_grads.len() != node.inputs.len() {
                    return Err(Error::Autograd(format!(
                        "{} backward returned {} grads for {} inputs",
                        node.op,
                        input_grads.len(),
                        node.inputs.len()
                    )));
                }
                for (inp, ig_opt) in node.inputs.iter().zip(input_grads) {
                    let Some(ig) = ig_opt else { continue };
                    if !inp.requires_grad() {
                        continue;
                    }
                    match grads.remove(&inp.id()) {
                        Some(prev) => {
                            grads.insert(inp.id(), prev.add(&ig)?);
                        }
                        None => {
                            grads.insert(inp.id(), ig);
                        }
                    }
                }
                drop(node);
                if release {
                    t.release_node();
                }
            }
        }
        Ok(())
    }

    fn run_hooks(&self, mut grad: Tensor) -> Result<Tensor> {
        let hooks = mutex_guard(&self.0.hooks);
        for h in hooks.iter() {
            if let Some(modified) = h(&grad)? {
                grad = modified;
            }
        }
        Ok(grad)
    }

    fn topological_order(&self) -> Vec<Tensor> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![(self.clone(), false)];
        while let Some((t, expanded)) = stack.pop() {
            if expanded {
                order.push(t);
                continue;
            }
            if !visited.insert(t.id()) {
                continue;
            }
            stack.push((t.clone(), true));
            if let Some(node) = t.node() {
                for inp in &node.inputs {
                    if inp.requires_grad() && !visited.contains(&inp.id()) {
                        stack.push((inp.clone(), false));
                    }
                }
            }
        }
        order
    }
}

/// Computes and returns the sum of gradients of `outputs` with respect to `inputs`.
///
/// If `create_graph` is `true`, the returned gradients are themselves connected to the
/// autograd graph so you can differentiate them again (e.g. for Hessians, JVP/VJP, or
/// gradient penalties).
pub fn grad(
    outputs: &[Tensor],
    inputs: &[Tensor],
    grad_outputs: Option<&[Tensor]>,
    create_graph: bool,
) -> Result<Vec<Option<Tensor>>> {
    // Build combined topological order across all outputs.
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    for out in outputs {
        for t in out.topological_order() {
            if visited.insert(t.id()) {
                order.push(t);
            }
        }
    }

    if let Some(gos) = grad_outputs {
        if gos.len() != outputs.len() {
            return Err(Error::Autograd(format!(
                "grad() got {} grad_outputs for {} outputs",
                gos.len(),
                outputs.len()
            )));
        }
    }
    for (i, out) in outputs.iter().enumerate() {
        if !out.requires_grad() {
            return Err(Error::Autograd(format!(
                "grad() output {i} does not require grad"
            )));
        }
    }

    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    for (i, out) in outputs.iter().enumerate() {
        let seed = match grad_outputs {
            Some(gos) => {
                if gos[i].numel() != out.numel() {
                    return Err(Error::Autograd(format!(
                        "grad() grad_output {i} has {} elements for output shape {:?} ({} elements)",
                        gos[i].numel(),
                        out.shape(),
                        out.numel()
                    )));
                }
                gos[i].clone()
            }
            None => {
                if out.numel() != 1 {
                    return Err(Error::Autograd(format!(
                        "grad() output {i} is non-scalar (shape {:?}) and no grad_outputs provided",
                        out.shape()
                    )));
                }
                Tensor::ones(out.shape(), out.dtype(), &out.device())?
            }
        };
        match grads.remove(&out.id()) {
            Some(prev) => {
                grads.insert(out.id(), prev.add(&seed)?);
            }
            None => {
                grads.insert(out.id(), seed);
            }
        }
    }

    let input_ids: HashSet<usize> = inputs.iter().map(Tensor::id).collect();
    let mut captured: HashMap<usize, Tensor> = HashMap::new();

    let _grad_guard = set_grad_enabled(create_graph);

    for t in order.iter().rev() {
        let Some(mut g) = grads.remove(&t.id()) else {
            continue;
        };
        g = t.run_hooks(g)?;
        if g.numel() != t.numel() {
            return Err(Error::Autograd(format!(
                "gradient for tensor of shape {:?} has {} elements (hook or producer mismatch)",
                t.shape(),
                g.numel()
            )));
        }
        if input_ids.contains(&t.id()) {
            captured.insert(t.id(), g.clone());
        }
        if let Some(node) = t.node() {
            let input_grads = (node.backward)(&g)?;
            if input_grads.len() != node.inputs.len() {
                return Err(Error::Autograd(format!(
                    "{} backward returned {} grads for {} inputs",
                    node.op,
                    input_grads.len(),
                    node.inputs.len()
                )));
            }
            for (inp, ig_opt) in node.inputs.iter().zip(input_grads) {
                let Some(ig) = ig_opt else { continue };
                if !inp.requires_grad() {
                    continue;
                }
                match grads.remove(&inp.id()) {
                    Some(prev) => {
                        grads.insert(inp.id(), prev.add(&ig)?);
                    }
                    None => {
                        grads.insert(inp.id(), ig);
                    }
                }
            }
        }
    }

    // Clone (not remove): the same input may be requested twice, and like torch
    // every entry must come back populated.
    Ok(inputs
        .iter()
        .map(|inp| captured.get(&inp.id()).cloned())
        .collect())
}

/// Gradient checkpointing: runs `f` without saving intermediate activations in the
/// forward pass, and recomputes `f` with gradients enabled during the backward pass.
pub fn checkpoint<F>(f: F, inputs: &[Tensor]) -> Result<Tensor>
where
    F: Fn(&[Tensor]) -> Result<Tensor> + Send + Sync + 'static,
{
    let out = no_grad(|| f(inputs))?;
    let saved_inputs: Vec<Tensor> = inputs.to_vec();
    let f = Arc::new(f);
    Ok(Tensor::from_op(
        out.storage().clone(),
        out.layout().clone(),
        "checkpoint",
        inputs.to_vec(),
        move |grad_out| {
            let create_graph = is_grad_enabled();
            let detached: Vec<Tensor> = saved_inputs
                .iter()
                .map(|x| x.detach().requires_grad_(x.requires_grad()))
                .collect();
            let recomputed = enable_grad(|| f(&detached))?;
            grad(
                &[recomputed],
                &detached,
                Some(std::slice::from_ref(grad_out)),
                create_graph,
            )
        },
    ))
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------
fn fmt_nested(
    f: &mut fmt::Formatter<'_>,
    data: &[f64],
    shape: &[usize],
    is_float: bool,
) -> fmt::Result {
    let Some((&n, rest)) = shape.split_first() else {
        // A 0-d tensor whose data cannot be read (e.g. Meta) prints NaN;
        // Display/Debug must never panic.
        let v = data.first().copied().unwrap_or(f64::NAN);
        return if is_float {
            write!(f, "{v:.4}")
        } else {
            write!(f, "{}", v as i64)
        };
    };
    if data.is_empty() {
        return write!(f, "[]");
    }
    let step = data.len() / n.max(1);
    let idxs: Vec<Option<usize>> = if n > 8 {
        (0..3)
            .map(Some)
            .chain([None])
            .chain((n - 3..n).map(Some))
            .collect()
    } else {
        (0..n).map(Some).collect()
    };
    write!(f, "[")?;
    for (j, i) in idxs.into_iter().enumerate() {
        if j > 0 {
            write!(f, ", ")?;
        }
        match i {
            Some(i) => fmt_nested(f, &data[i * step..(i + 1) * step], rest, is_float)?,
            None => write!(f, "...")?,
        }
    }
    write!(f, "]")
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vals = self.to_vec::<f64>().unwrap_or_default();
        fmt_nested(f, &vals, self.shape(), self.dtype().is_floating_point())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={}, device={}, data={self}",
            self.shape(),
            self.dtype(),
            self.device()
        )?;
        if !self.is_contiguous() {
            write!(f, ", strides={:?}", self.strides())?;
        }
        if self.requires_grad() {
            write!(f, ", requires_grad=true")?;
        }
        if let Some(op) = self.op_name() {
            write!(f, ", op={op}")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod release_graph_tests {
    use super::Tensor;
    use std::sync::Arc;

    // y = sum(exp(x) * x): exp saves its output, mul saves both inputs
    fn run(release: bool) -> (Vec<f32>, std::sync::Weak<super::TensorInner>, Tensor) {
        let x = Tensor::from_slice(&[0.5f32, -1.0, 2.0], &[3]).unwrap().requires_grad_(true);
        let e = x.exp().unwrap();
        let weak = Arc::downgrade(&e.0);
        let y = e.mul(&x).unwrap().sum().unwrap();
        drop(e);
        let seed = y.ones_like().unwrap();
        if release { y.backward_release(&seed).unwrap() } else { y.backward_with_grad(&seed, false).unwrap() }
        (x.grad().unwrap().to_vec::<f32>().unwrap(), weak, y)
    }

    #[test]
    fn release_frees_intermediates_and_keeps_grads() {
        let (kept, weak_kept, y_kept) = run(false);
        let (freed, weak_freed, y_freed) = run(true);
        assert_eq!(kept, freed);
        for (g, x) in freed.iter().zip([0.5f32, -1.0, 2.0]) {
            assert!((g - x.exp() * (1.0 + x)).abs() < 1e-5);
        }
        assert!(weak_kept.upgrade().is_some(), "default backward keeps the graph alive through the output");
        assert!(weak_freed.upgrade().is_none(), "released backward must drop the intermediate exp(x)");
        assert!(!y_freed.is_leaf() && !y_kept.is_leaf());
        let again = y_freed.backward_release(&y_freed.ones_like().unwrap());
        assert!(again.is_err(), "a second backward through a released graph must fail loudly");
        assert!(y_kept.backward_with_grad(&y_kept.ones_like().unwrap(), false).is_ok());
    }
}
