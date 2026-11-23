//! DLPack interchange (the `DLManagedTensor` C ABI, v0.x): hand tensors to and take tensors from NumPy, PyTorch, JAX,
//! CuPy and anything else that speaks the protocol.
//!
//! [`to_dlpack`] returns a heap `DLManagedTensor` whose `deleter` the consumer must call exactly once; [`from_dlpack`]
//! consumes one (strided, offset and non-contiguous inputs included) and calls its deleter.
//!
//! Both directions are zero-copy when they can be. Export pins the tensor's storage (see [`Storage::pin_dlpack`]) so the
//! consumer's pointer stays valid. Import of aligned, non-negatively strided CPU data builds a tensor that **aliases the
//! producer's memory** (writes through either side are visible to the other); the producer's `deleter` runs when the last
//! handle to that storage drops. Anything else (misaligned, negative strides, `bool`, empty) is copied and the deleter runs
//! immediately.

use crate::dtype::{DType, bf16, f16};
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::ffi::c_void;

pub const DL_CPU: i32 = 1;
const CODE_INT: u8 = 0;
const CODE_UINT: u8 = 1;
const CODE_FLOAT: u8 = 2;
const CODE_BFLOAT: u8 = 4;
const CODE_COMPLEX: u8 = 5;
const CODE_BOOL: u8 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

fn err(m: impl Into<String>) -> Error {
    Error::Msg(format!("dlpack: {}", m.into()))
}

fn dl_type(d: DType) -> Result<DLDataType> {
    let (code, bits) = match d {
        DType::Bool => (CODE_BOOL, 8),
        DType::U8 => (CODE_UINT, 8),
        DType::U16 => (CODE_UINT, 16),
        DType::U32 => (CODE_UINT, 32),
        DType::U64 => (CODE_UINT, 64),
        DType::I8 => (CODE_INT, 8),
        DType::I16 => (CODE_INT, 16),
        DType::I32 => (CODE_INT, 32),
        DType::I64 => (CODE_INT, 64),
        DType::F16 => (CODE_FLOAT, 16),
        DType::F32 => (CODE_FLOAT, 32),
        DType::F64 => (CODE_FLOAT, 64),
        DType::BF16 => (CODE_BFLOAT, 16),
        DType::C64 => (CODE_COMPLEX, 64),
        DType::C128 => (CODE_COMPLEX, 128),
        DType::F8E4M3 | DType::F8E5M2 => return Err(err("8-bit floats have no DLPack type code; cast to f32 first")),
    };
    Ok(DLDataType { code, bits, lanes: 1 })
}

fn from_dl_type(t: DLDataType) -> Result<DType> {
    if t.lanes != 1 {
        return Err(err("vector (multi-lane) types are not supported"));
    }
    Ok(match (t.code, t.bits) {
        (CODE_BOOL, 8) => DType::Bool,
        (CODE_UINT, 8) => DType::U8,
        (CODE_UINT, 16) => DType::U16,
        (CODE_UINT, 32) => DType::U32,
        (CODE_UINT, 64) => DType::U64,
        (CODE_INT, 8) => DType::I8,
        (CODE_INT, 16) => DType::I16,
        (CODE_INT, 32) => DType::I32,
        (CODE_INT, 64) => DType::I64,
        (CODE_FLOAT, 16) => DType::F16,
        (CODE_FLOAT, 32) => DType::F32,
        (CODE_FLOAT, 64) => DType::F64,
        (CODE_BFLOAT, 16) => DType::BF16,
        (CODE_COMPLEX, 64) => DType::C64,
        (CODE_COMPLEX, 128) => DType::C128,
        (c, b) => return Err(err(format!("unsupported type code {c} with {b} bits"))),
    })
}

/// The tensor's elements in logical (row-major) order as native-endian bytes.
fn to_bytes(t: &Tensor) -> Result<Vec<u8>> {
    macro_rules! raw {
        ($ty:ty) => {
            t.to_vec::<$ty>()?.iter().flat_map(|v| v.to_ne_bytes()).collect()
        };
    }
    Ok(match t.dtype() {
        DType::Bool => t.to_vec::<bool>()?.into_iter().map(u8::from).collect(),
        DType::U8 => t.to_vec::<u8>()?,
        DType::U16 => raw!(u16),
        DType::U32 => raw!(u32),
        DType::U64 => raw!(u64),
        DType::I8 => raw!(i8),
        DType::I16 => raw!(i16),
        DType::I32 => raw!(i32),
        DType::I64 => raw!(i64),
        DType::F32 => raw!(f32),
        DType::F64 => raw!(f64),
        DType::F16 => t.to_vec::<f16>()?.iter().flat_map(|v| v.to_bits().to_ne_bytes()).collect(),
        DType::BF16 => t.to_vec::<bf16>()?.iter().flat_map(|v| v.to_bits().to_ne_bytes()).collect(),
        DType::C64 => t
            .to_c64_vec()?
            .iter()
            .flat_map(|[r, i]| [r.to_ne_bytes(), i.to_ne_bytes()].into_iter().flatten())
            .collect(),
        DType::C128 => t
            .to_c128_vec()?
            .iter()
            .flat_map(|[r, i]| [r.to_ne_bytes(), i.to_ne_bytes()].into_iter().flatten())
            .collect(),
        DType::F8E4M3 | DType::F8E5M2 => return Err(err("8-bit floats have no DLPack type code; cast to f32 first")),
    })
}

fn from_bytes(bytes: &[u8], dtype: DType, shape: &[usize]) -> Result<Tensor> {
    macro_rules! typed {
        ($ty:ty, $n:expr) => {
            Tensor::from_vec(bytes.chunks_exact($n).map(|c| <$ty>::from_ne_bytes(c.try_into().unwrap())).collect::<Vec<$ty>>(), shape)
        };
    }
    match dtype {
        DType::Bool => Tensor::from_vec(bytes.iter().map(|&b| b != 0).collect::<Vec<bool>>(), shape),
        DType::U8 => Tensor::from_vec(bytes.to_vec(), shape),
        DType::U16 => typed!(u16, 2),
        DType::U32 => typed!(u32, 4),
        DType::U64 => typed!(u64, 8),
        DType::I8 => typed!(i8, 1),
        DType::I16 => typed!(i16, 2),
        DType::I32 => typed!(i32, 4),
        DType::I64 => typed!(i64, 8),
        DType::F32 => typed!(f32, 4),
        DType::F64 => typed!(f64, 8),
        DType::F16 => Tensor::from_vec(bytes.chunks_exact(2).map(|c| f16::from_bits(u16::from_ne_bytes([c[0], c[1]]))).collect::<Vec<f16>>(), shape),
        DType::BF16 => Tensor::from_vec(bytes.chunks_exact(2).map(|c| bf16::from_bits(u16::from_ne_bytes([c[0], c[1]]))).collect::<Vec<bf16>>(), shape),
        DType::C64 => Tensor::from_c64_vec(
            bytes
                .chunks_exact(8)
                .map(|c| [
                    f32::from_ne_bytes(c[0..4].try_into().unwrap()),
                    f32::from_ne_bytes(c[4..8].try_into().unwrap()),
                ])
                .collect(),
            shape,
        ),
        DType::C128 => Tensor::from_c128_vec(
            bytes
                .chunks_exact(16)
                .map(|c| [
                    f64::from_ne_bytes(c[0..8].try_into().unwrap()),
                    f64::from_ne_bytes(c[8..16].try_into().unwrap()),
                ])
                .collect(),
            shape,
        ),
        DType::F8E4M3 | DType::F8E5M2 => Err(err("8-bit floats have no DLPack type code")),
    }
}

struct Owner {
    _pinned: Option<crate::backend::Storage>,
    _owned: Vec<u8>,
    _shape: Vec<i64>,
    _strides: Vec<i64>,
}

impl Drop for Owner {
    fn drop(&mut self) {
        if let Some(s) = self._pinned.take() {
            s.unpin_dlpack();
        }
    }
}

unsafe extern "C" fn delete_owned(p: *mut DLManagedTensor) {
    if p.is_null() {
        return;
    }
    // SAFETY: `p` and its `manager_ctx` were produced by `Box::into_raw` in `to_dlpack`, and the DLPack contract has the
    // consumer call the deleter exactly once.
    unsafe {
        let m = Box::from_raw(p);
        drop(Box::from_raw(m.manager_ctx as *mut Owner));
    }
}

/// Exports `t` as a `DLManagedTensor`. CPU tensors are exported **zero-copy** with pinned storage
/// (preventing reallocation behind the storage lock until the consumer invokes `deleter`);
/// non-CPU tensors are copied to a host buffer. The caller owns the pointer and must hand it to a
/// consumer or call its `deleter`.
pub fn to_dlpack(t: &Tensor) -> Result<*mut DLManagedTensor> {
    let dtype = dl_type(t.dtype())?;
    let shape: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();
    if t.device().is_cpu() || matches!(t.device(), crate::device::Device::Cuda(_)) {
        let is_cuda = matches!(t.device(), crate::device::Device::Cuda(_));
        let strides: Vec<i64> = t
            .layout()
            .strides()
            .iter()
            .map(|&s| i64::try_from(s).map_err(|_| err("stride exceeds i64::MAX")))
            .collect::<Result<Vec<i64>>>()?;
        let byte_offset = (t.storage_offset() as u64)
            .checked_mul(t.dtype().size_in_bytes() as u64)
            .ok_or_else(|| err("byte_offset overflow"))?;
        let (data_ptr, dl_dev) = {
            let guard = t.storage().read();
            match &*guard {
                crate::backend::StorageInner::Cpu(cpu) => (
                    cpu.raw_cpu_ptr(),
                    DLDevice { device_type: DL_CPU, device_id: 0 },
                ),
                crate::backend::StorageInner::Cuda(cu) => (
                    cu.device_ptr() as *mut u8,
                    DLDevice { device_type: 2 /* kDLCUDA */, device_id: cu.device_index as i32 },
                ),
                _ => {
                    if is_cuda {
                        // Simulated Accelerator fallback: host pointer
                        return Err(err("CUDA tensor is in host-simulated storage"));
                    }
                    return Err(err("expected CPU storage"));
                }
            }
        };
        t.storage().pin_dlpack();
        let mut owner = Box::new(Owner {
            _pinned: Some(t.storage().clone()),
            _owned: Vec::new(),
            _shape: shape,
            _strides: strides,
        });
        let strides_ptr = if owner._strides.is_empty() {
            std::ptr::null_mut()
        } else {
            owner._strides.as_mut_ptr()
        };
        let dl_tensor = DLTensor {
            data: data_ptr as *mut c_void,
            device: dl_dev,
            ndim: owner._shape.len() as i32,
            dtype,
            shape: owner._shape.as_mut_ptr(),
            strides: strides_ptr,
            byte_offset,
        };
        return Ok(Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor,
            manager_ctx: Box::into_raw(owner) as *mut c_void,
            deleter: Some(delete_owned),
        })));
    }

    let mut owner = Box::new(Owner {
        _pinned: None,
        _owned: to_bytes(t)?,
        _shape: shape,
        _strides: Vec::new(),
    });
    let dl_tensor = DLTensor {
        data: owner._owned.as_mut_ptr() as *mut c_void,
        device: DLDevice { device_type: DL_CPU, device_id: 0 },
        ndim: owner._shape.len() as i32,
        dtype,
        shape: owner._shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };
    Ok(Box::into_raw(Box::new(DLManagedTensor { dl_tensor, manager_ctx: Box::into_raw(owner) as *mut c_void, deleter: Some(delete_owned) })))
}

/// Imports (copies) a `DLManagedTensor` and calls its deleter, whether or not the import succeeds.
///
/// # Safety
/// `managed` must be a valid, not yet consumed `DLManagedTensor` whose `data`, `shape` and (if non-null) `strides` describe
/// readable memory, as the DLPack protocol guarantees for a capsule received from a producer.
pub unsafe fn from_dlpack(managed: *mut DLManagedTensor) -> Result<Tensor> {
    if managed.is_null() {
        return Err(err("null DLManagedTensor"));
    }
    // SAFETY: validity of the pointer and of what it describes is the caller's contract (see above).
    if let Some(t) = unsafe { try_alias(managed) } {
        return Ok(t);
    }
    // SAFETY: validity of the pointer and of what it describes is the caller's contract (see above).
    let result = unsafe { read(&(*managed).dl_tensor) };
    // SAFETY: the tensor has not been consumed before (caller's contract); the deleter is called exactly once, here.
    unsafe {
        if let Some(d) = (*managed).deleter {
            d(managed);
        }
    }
    result
}

/// A `Send` wrapper for the producer's `DLManagedTensor` pointer (its deleter may run on any thread, as the protocol allows).
struct SendPtr(*mut DLManagedTensor);
// SAFETY: the DLPack contract lets the consumer call the deleter from any thread.
unsafe impl Send for SendPtr {}
// SAFETY: the pointer is only dereferenced once, to call the deleter.
unsafe impl Sync for SendPtr {}

/// Zero-copy import: a tensor aliasing the producer's buffer, or `None` when the data needs the copying path.
///
/// # Safety
/// As [`from_dlpack`].
unsafe fn try_alias(managed: *mut DLManagedTensor) -> Option<Tensor> {
    // SAFETY: the caller guarantees `managed` is valid.
    let (cpu, layout) = unsafe { alias_parts(&(*managed).dl_tensor) }?;
    let owner = SendPtr(managed);
    Some(alias_tensor(cpu, layout, move || {
        let owner = owner;
        // SAFETY: the producer's deleter is called exactly once, when the last handle to the aliased storage drops.
        unsafe {
            if let Some(d) = (*owner.0).deleter {
                d(owner.0);
            }
        }
    }))
}

fn alias_tensor(cpu: crate::backend::CpuStorage, layout: crate::layout::Layout, release: impl FnOnce() + Send + Sync + 'static) -> Tensor {
    Tensor::from_storage(crate::backend::Storage::from_foreign_cpu(cpu, Box::new(release)), layout)
}

/// The storage and layout for a zero-copy view of `t`, or `None` when the data needs the copying path.
///
/// # Safety
/// `t` must describe readable, writable memory that outlives the returned storage (see [`from_dlpack`]).
unsafe fn alias_parts(t: &DLTensor) -> Option<(crate::backend::CpuStorage, crate::layout::Layout)> {
    use crate::backend::CpuStorage;
    use crate::layout::Layout;
    if t.device.device_type != DL_CPU || t.data.is_null() || t.ndim < 0 {
        return None;
    }
    let dtype = from_dl_type(t.dtype).ok()?;
    let ndim = t.ndim as usize;
    if ndim > 0 && t.shape.is_null() {
        return None;
    }
    // SAFETY: `shape` / `strides` hold `ndim` entries per the DLPack contract.
    let shape: Vec<i64> = if ndim == 0 { vec![] } else { unsafe { std::slice::from_raw_parts(t.shape, ndim) }.to_vec() };
    if shape.iter().any(|&d| d <= 0) {
        return None; // empty tensors take the copying path
    }
    let strides: Vec<i64> = if t.strides.is_null() {
        // row-major strides in elements
        let mut s = vec![1i64; ndim];
        let mut acc = 1i64;
        for i in (0..ndim).rev() {
            s[i] = acc;
            acc = acc.checked_mul(shape[i])?;
        }
        s
    } else {
        unsafe { std::slice::from_raw_parts(t.strides, ndim) }.to_vec()
    };
    if strides.iter().any(|&s| s < 0) {
        return None;
    }
    // elements the buffer must span: one past the last addressable element
    let mut span: i64 = 1;
    for (d, s) in shape.iter().zip(&strides) {
        span = span.checked_add((d - 1).checked_mul(*s)?)?;
    }
    let len = usize::try_from(span).ok()?;
    let esize = dtype.size_in_bytes();
    len.checked_mul(esize)?;
    let ptr = (t.data as *mut u8).wrapping_add(t.byte_offset as usize);
    macro_rules! build {
        ($variant:ident, $ty:ty) => {{
            if (ptr as usize) % std::mem::align_of::<$ty>() != 0 {
                return None;
            }
            // SAFETY: the memory is `len` valid elements; this Vec is detached (`forget_buffer`) before the storage is
            // released, so it never reaches the allocator.
            CpuStorage::$variant(unsafe { Vec::from_raw_parts(ptr as *mut $ty, len, len) })
        }};
    }
    let cpu = match dtype {
        DType::U8 => build!(U8, u8),
        DType::I8 => build!(I8, i8),
        DType::U16 => build!(U16, u16),
        DType::I16 => build!(I16, i16),
        DType::U32 => build!(U32, u32),
        DType::I32 => build!(I32, i32),
        DType::U64 => build!(U64, u64),
        DType::I64 => build!(I64, i64),
        DType::F16 => build!(F16, f16),
        DType::BF16 => build!(BF16, bf16),
        DType::F32 => build!(F32, f32),
        DType::F64 => build!(F64, f64),
        DType::C64 => build!(C64, [f32; 2]),
        DType::C128 => build!(C128, [f64; 2]),
        _ => return None, // bool (needs 0/1 validation) and the 8-bit floats are copied
    };
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let layout = if t.strides.is_null() { Layout::contiguous(dims) } else { Layout::new(dims, strides.iter().map(|&s| s as isize).collect(), 0) };
    Some((cpu, layout))
}

// ---------------------------------------------------------------------------------------------------------------------
// DLPack 1.x (`DLManagedTensorVersioned`): carries a version and flags, so a consumer can tell a writable buffer from a
// read-only one (NumPy only hands out writable arrays for versioned capsules).
// ---------------------------------------------------------------------------------------------------------------------

/// The `flags` bit marking a read-only buffer.
pub const DLPACK_FLAG_READ_ONLY: u64 = 1;
/// The `flags` bit marking a producer that already copied the data.
pub const DLPACK_FLAG_IS_COPIED: u64 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

#[repr(C)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

unsafe extern "C" fn delete_versioned(p: *mut DLManagedTensorVersioned) {
    if p.is_null() {
        return;
    }
    // SAFETY: built by `to_dlpack_versioned` from two `Box::into_raw`s; the consumer calls the deleter exactly once.
    unsafe {
        let m = Box::from_raw(p);
        let inner = m.manager_ctx as *mut DLManagedTensor;
        if let Some(d) = (*inner).deleter {
            d(inner);
        }
    }
}

/// Exports `t` as a DLPack 1.0 `DLManagedTensorVersioned` (writable, zero-copy exactly like [`to_dlpack`]).
pub fn to_dlpack_versioned(t: &Tensor) -> Result<*mut DLManagedTensorVersioned> {
    let inner = to_dlpack(t)?;
    // SAFETY: `inner` was just created by `to_dlpack`; its shape / strides stay alive until its deleter runs, which only
    // `delete_versioned` does.
    let dl_tensor = unsafe { std::ptr::read(&(*inner).dl_tensor) };
    Ok(Box::into_raw(Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: inner as *mut c_void,
        deleter: Some(delete_versioned),
        flags: 0,
        dl_tensor,
    })))
}

struct SendPtrV(*mut DLManagedTensorVersioned);
// SAFETY: as `SendPtr`: only dereferenced once, to call the deleter.
unsafe impl Send for SendPtrV {}
unsafe impl Sync for SendPtrV {}

/// Imports a versioned capsule: zero-copy when it is writable and CPU-resident, otherwise a copy. Calls its deleter (now for
/// a copy, when the last handle drops for an alias).
///
/// # Safety
/// `managed` must be a valid, not yet consumed `DLManagedTensorVersioned` (see [`from_dlpack`]).
pub unsafe fn from_dlpack_versioned(managed: *mut DLManagedTensorVersioned) -> Result<Tensor> {
    if managed.is_null() {
        return Err(err("null DLManagedTensorVersioned"));
    }
    // SAFETY: validity is the caller's contract.
    let (version, flags) = unsafe { ((*managed).version, (*managed).flags) };
    let call_deleter = move || unsafe {
        if let Some(d) = (*managed).deleter {
            d(managed);
        }
    };
    if version.major != 1 {
        call_deleter();
        return Err(err(format!("unsupported DLPack major version {}", version.major)));
    }
    if flags & DLPACK_FLAG_READ_ONLY == 0
        // SAFETY: as above
        && let Some((cpu, layout)) = unsafe { alias_parts(&(*managed).dl_tensor) }
    {
        let owner = SendPtrV(managed);
        return Ok(alias_tensor(cpu, layout, move || {
            let owner = owner;
            // SAFETY: called once, when the last handle to the aliased storage drops
            unsafe {
                if let Some(d) = (*owner.0).deleter {
                    d(owner.0);
                }
            }
        }));
    }
    // read-only data must never be aliased by a writable tensor: copy it
    // SAFETY: the caller guarantees the tensor description is valid.
    let result = unsafe { read(&(*managed).dl_tensor) };
    call_deleter();
    result
}

/// # Safety
/// `t` must describe readable memory (see [`from_dlpack`]).
unsafe fn read(t: &DLTensor) -> Result<Tensor> {
    if t.device.device_type != DL_CPU {
        return Err(err(format!("only CPU tensors can be imported (device type {})", t.device.device_type)));
    }
    let dtype = from_dl_type(t.dtype)?;
    let ndim = usize::try_from(t.ndim).map_err(|_| err("negative ndim"))?;
    if ndim > 0 && t.shape.is_null() {
        return Err(err("null shape"));
    }
    // SAFETY: `shape` (and `strides` when non-null) hold `ndim` entries per the DLPack contract.
    let shape: Vec<i64> = if ndim == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(t.shape, ndim) }.to_vec() };
    if shape.iter().any(|&d| d < 0) {
        return Err(err("negative dimension"));
    }
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    crate::limits::check_elements("dlpack import", &dims)?;
    let count: usize = dims.iter().product();
    let esize = dtype.size_in_bytes();
    let strides: Vec<i64> = if t.strides.is_null() {
        let mut s = vec![1i64; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    } else {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts(t.strides, ndim) }.to_vec()
    };
    if count > 0 && t.data.is_null() {
        return Err(err("null data"));
    }
    let base = (t.data as *const u8).wrapping_add(t.byte_offset as usize);
    let mut bytes = Vec::with_capacity(count * esize);
    let mut idx = vec![0i64; ndim];
    for _ in 0..count {
        let off: i64 = idx.iter().zip(&strides).map(|(i, s)| i * s).sum();
        // SAFETY: every index is inside `shape`, so the element lies inside the producer's buffer.
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(base.offset(off as isize * esize as isize), esize) });
        for d in (0..ndim).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    from_bytes(&bytes, dtype, &dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_dtype_round_trips_including_views() {
        let base = Tensor::from_slice(&[1.5f32, -2.0, 3.25, 4.0, 5.5, -6.0], &[2, 3]).unwrap();
        for d in [DType::F32, DType::F64, DType::F16, DType::BF16, DType::I8, DType::I16, DType::I32, DType::I64, DType::U8, DType::U16, DType::U32, DType::U64, DType::Bool] {
            let t = base.abs().unwrap().to_dtype(d).unwrap().transpose(0, 1).unwrap(); // a non-contiguous view
            let p = to_dlpack(&t).unwrap();
            // SAFETY: `p` was just produced by `to_dlpack` and is consumed once.
            let back = unsafe { from_dlpack(p) }.unwrap();
            assert_eq!(back.dtype(), d);
            assert_eq!(back.shape(), [3, 2]);
            assert_eq!(back.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap(), t.to_dtype(DType::F64).unwrap().to_vec::<f64>().unwrap(), "{d:?}");
        }
        let big = Tensor::from_slice(&[u64::MAX, (1 << 63) + 7], &[2]).unwrap();
        // SAFETY: as above.
        let back = unsafe { from_dlpack(to_dlpack(&big).unwrap()) }.unwrap();
        assert_eq!(back.to_vec::<u64>().unwrap(), vec![u64::MAX, (1 << 63) + 7]);
        assert!(to_dlpack(&base.to_dtype(DType::F8E4M3).unwrap()).is_err());
    }

    #[test]
    fn cpu_export_is_zero_copy_and_pins_storage() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let raw_ptr = {
            let guard = t.storage().read();
            match &*guard {
                crate::backend::StorageInner::Cpu(cpu) => cpu.raw_cpu_ptr() as *mut c_void,
                _ => unreachable!(),
            }
        };
        let p = to_dlpack(&t).unwrap();
        assert!(t.storage().is_dlpack_pinned());
        unsafe {
            assert_eq!((*p).dl_tensor.data, raw_ptr);
        }
        // Reallocation/retarget is blocked while pinned; in-place assign succeeds and updates visible data.
        let other = Tensor::from_slice(&[9.0f32, 8.0, 7.0, 6.0], &[2, 2]).unwrap();
        assert!(t.retarget(&other).is_err());
        t.assign(&other).unwrap();
        let back = unsafe { from_dlpack(p) }.unwrap();
        // the import aliases the exported buffer, so the pin lasts as long as `back` lives
        assert!(t.storage().is_dlpack_pinned());
        assert_eq!(back.to_vec::<f32>().unwrap(), vec![9.0, 8.0, 7.0, 6.0]);
        drop(back);
        assert!(!t.storage().is_dlpack_pinned());
        assert!(t.retarget(&other).is_ok());
    }

    /// Host memory with a DLPack header around it, standing in for a producer such as NumPy.
    struct Foreign {
        data: Vec<f32>,
        shape: Vec<i64>,
        strides: Vec<i64>,
    }
    // the release counter below is global, so the tests that read it must not overlap
    static FOREIGN_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static FOREIGN_FREED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    unsafe extern "C" fn free_foreign(p: *mut DLManagedTensor) {
        // SAFETY: built by `foreign_capsule` from two `Box::into_raw`s
        unsafe {
            let m = Box::from_raw(p);
            drop(Box::from_raw(m.manager_ctx as *mut Foreign));
        }
        FOREIGN_FREED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    fn foreign_capsule(data: Vec<f32>, shape: Vec<i64>, strides: Option<Vec<i64>>, byte_offset: u64) -> *mut DLManagedTensor {
        let mut owner = Box::new(Foreign { data, shape, strides: strides.clone().unwrap_or_default() });
        let t = DLTensor {
            data: owner.data.as_mut_ptr() as *mut c_void,
            device: DLDevice { device_type: DL_CPU, device_id: 0 },
            ndim: owner.shape.len() as i32,
            dtype: DLDataType { code: CODE_FLOAT, bits: 32, lanes: 1 },
            shape: owner.shape.as_mut_ptr(),
            strides: if strides.is_some() { owner.strides.as_mut_ptr() } else { std::ptr::null_mut() },
            byte_offset,
        };
        Box::into_raw(Box::new(DLManagedTensor { dl_tensor: t, manager_ctx: Box::into_raw(owner) as *mut c_void, deleter: Some(free_foreign) }))
    }

    #[test]
    fn import_aliases_foreign_memory_and_releases_it_with_the_last_handle() {
        let _g = FOREIGN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let freed = || FOREIGN_FREED.load(std::sync::atomic::Ordering::SeqCst);
        let before = freed();
        let cap = foreign_capsule((0..6).map(|i| i as f32).collect(), vec![2, 3], None, 0);
        let data_ptr = unsafe { (*cap).dl_tensor.data };
        let t = unsafe { from_dlpack(cap) }.unwrap();
        assert_eq!(freed(), before, "the producer's buffer must stay alive while the tensor does");
        // the tensor reads the producer's memory itself, not a copy
        let ptr = match &*t.storage().read() {
            crate::backend::StorageInner::Cpu(c) => c.raw_cpu_ptr() as *mut c_void,
            _ => unreachable!(),
        };
        assert_eq!(ptr, data_ptr);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // computation works on the aliased data, views share it, and the storage cannot be reallocated or replaced
        assert_eq!(t.add_scalar(1.0).unwrap().to_vec::<f32>().unwrap()[5], 6.0);
        let view = t.transpose(0, 1).unwrap();
        assert!(t.storage().is_dlpack_pinned());
        assert!(t.retarget(&Tensor::from_slice(&[0.0f32; 6], &[2, 3]).unwrap()).is_err());
        // in-place writes go through to the producer's memory
        t.assign(&Tensor::from_slice(&[9.0f32; 6], &[2, 3]).unwrap()).unwrap();
        assert_eq!(unsafe { *(data_ptr as *const f32) }, 9.0);
        drop(t);
        assert_eq!(freed(), before, "a live view keeps the buffer");
        assert_eq!(view.to_vec::<f32>().unwrap()[0], 9.0);
        drop(view);
        assert_eq!(freed(), before + 1, "the deleter runs exactly once, when the last handle drops");
    }

    #[test]
    fn strided_and_offset_imports_alias_too_and_odd_cases_fall_back_to_copies() {
        let _g = FOREIGN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let freed = || FOREIGN_FREED.load(std::sync::atomic::Ordering::SeqCst);
        let before = freed();
        // a transposed 3x2 view of a 2x3 buffer, starting one element in
        let cap = foreign_capsule((0..8).map(|i| i as f32).collect(), vec![3, 2], Some(vec![1, 3]), 4);
        let t = unsafe { from_dlpack(cap) }.unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(freed(), before, "aliased: not released yet");
        drop(t);
        assert_eq!(freed(), before + 1);
        // negative strides, empty tensors and misaligned data are copied and released immediately
        let cap = foreign_capsule((0..4).map(|i| i as f32).collect(), vec![2, 2], Some(vec![-2, 1]), 8);
        let t = unsafe { from_dlpack(cap) }.unwrap();
        assert_eq!(freed(), before + 2, "copied: released at import");
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![2.0, 3.0, 0.0, 1.0]);
        let cap = foreign_capsule(vec![1.0; 4], vec![0, 2], None, 0);
        assert_eq!(unsafe { from_dlpack(cap) }.unwrap().numel(), 0);
        assert_eq!(freed(), before + 3);
        let cap = foreign_capsule(vec![1.0; 4], vec![3], None, 2); // 2 bytes in: not 4-aligned
        let _ = unsafe { from_dlpack(cap) };
        assert_eq!(freed(), before + 4);
    }

    #[test]
    fn an_imported_tensor_can_be_dropped_on_another_thread() {
        let _g = FOREIGN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let freed = || FOREIGN_FREED.load(std::sync::atomic::Ordering::SeqCst);
        let before = freed();
        let t = unsafe { from_dlpack(foreign_capsule(vec![2.0; 1000], vec![1000], None, 0)) }.unwrap();
        let sum = std::thread::spawn(move || t.sum().unwrap().item::<f32>().unwrap()).join().unwrap();
        assert_eq!(sum, 2000.0);
        assert_eq!(freed(), before + 1);
    }

    #[test]
    fn versioned_export_is_zero_copy_writable_and_round_trips() {
        let _g = FOREIGN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let p = to_dlpack_versioned(&t).unwrap();
        unsafe {
            assert_eq!(((*p).version.major, (*p).version.minor), (1, 0));
            assert_eq!((*p).flags & DLPACK_FLAG_READ_ONLY, 0, "exports are writable");
        }
        assert!(t.storage().is_dlpack_pinned());
        let back = unsafe { from_dlpack_versioned(p) }.unwrap();
        assert_eq!(back.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        back.assign(&Tensor::from_slice(&[9.0f32; 4], &[2, 2]).unwrap()).unwrap();
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![9.0; 4], "the round trip shares memory");
        drop(back);
        assert!(!t.storage().is_dlpack_pinned(), "the deleter chain ran and released the pin");
    }

    #[test]
    fn read_only_versioned_data_is_copied_never_aliased() {
        let _g = FOREIGN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let freed = || FOREIGN_FREED.load(std::sync::atomic::Ordering::SeqCst);
        let before = freed();
        let v0 = foreign_capsule(vec![1.0, 2.0, 3.0], vec![3], None, 0);
        // wrap the v0 capsule in a versioned header flagged read-only
        let mut hdr = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion { major: 1, minor: 0 },
            manager_ctx: v0 as *mut c_void,
            deleter: Some(delete_versioned),
            flags: DLPACK_FLAG_READ_ONLY,
            dl_tensor: unsafe { std::ptr::read(&(*v0).dl_tensor) },
        });
        let t = unsafe { from_dlpack_versioned(&mut *hdr as *mut _) }.unwrap();
        std::mem::forget(hdr); // the deleter already freed it
        assert_eq!(freed(), before + 1, "copied: the producer's buffer is released immediately");
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
        assert!(!t.storage().is_dlpack_pinned(), "a copy owns its memory and is not pinned");
    }

    #[test]
    fn re_exporting_an_import_shares_the_original_memory() {
        let _g = FOREIGN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let cap = foreign_capsule(vec![1.0, 2.0, 3.0, 4.0], vec![4], None, 0);
        let data_ptr = unsafe { (*cap).dl_tensor.data };
        let t = unsafe { from_dlpack(cap) }.unwrap();
        let again = to_dlpack(&t).unwrap();
        assert_eq!(unsafe { (*again).dl_tensor.data }, data_ptr, "numpy -> oxideon -> torch never copies");
        let back = unsafe { from_dlpack(again) }.unwrap();
        assert_eq!(back.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    static DELETED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    unsafe extern "C" fn count_delete(_p: *mut DLManagedTensor) {
        DELETED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn foreign_strided_tensors_are_read_through_strides_and_offset_and_always_deleted() {
        // a producer-owned 3x4 i32 buffer; expose rows reversed, every second column, starting past a 4-byte header
        let mut buf: Vec<i32> = vec![-1];
        buf.extend(0..12);
        let mut shape = [3i64, 2];
        let mut strides = [-4i64, 2];
        let mut m = DLManagedTensor {
            dl_tensor: DLTensor {
                data: buf.as_mut_ptr() as *mut c_void,
                device: DLDevice { device_type: DL_CPU, device_id: 0 },
                ndim: 2,
                dtype: DLDataType { code: CODE_INT, bits: 32, lanes: 1 },
                shape: shape.as_mut_ptr(),
                strides: strides.as_mut_ptr(),
                byte_offset: 4 + 8 * 4, // skip the header, start at row 2
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(count_delete),
        };
        // SAFETY: `m` describes `buf`, which outlives the call; the counting deleter frees nothing.
        let t = unsafe { from_dlpack(&mut m) }.unwrap();
        assert_eq!(t.to_vec::<i32>().unwrap(), vec![8, 10, 4, 6, 0, 2]);
        // an unsupported device is refused, and the deleter still runs
        m.dl_tensor.device.device_type = 2;
        // SAFETY: as above.
        assert!(unsafe { from_dlpack(&mut m) }.is_err());
        assert_eq!(DELETED.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
