//! DLPack interchange (the `DLManagedTensor` C ABI, v0.x): hand tensors to and take tensors from NumPy, PyTorch, JAX,
//! CuPy and anything else that speaks the protocol.
//!
//! [`to_dlpack`] returns a heap `DLManagedTensor` whose `deleter` the consumer must call exactly once; [`from_dlpack`]
//! consumes one (strided, offset and non-contiguous inputs included) and calls its deleter.
//!
//! ponytail: both directions copy the data (storage here can be reallocated behind its lock, so a borrowed pointer could
//! dangle). Zero-copy export needs pinned storage; add it if exchange of very large arrays becomes a bottleneck.

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
    if t.device().is_cpu() {
        let strides: Vec<i64> = t
            .layout()
            .strides()
            .iter()
            .map(|&s| i64::try_from(s).map_err(|_| err("stride exceeds i64::MAX")))
            .collect::<Result<Vec<i64>>>()?;
        let byte_offset = (t.storage_offset() as u64)
            .checked_mul(t.dtype().size_in_bytes() as u64)
            .ok_or_else(|| err("byte_offset overflow"))?;
        let data_ptr = {
            let guard = t.storage().read();
            match &*guard {
                crate::backend::StorageInner::Cpu(cpu) => cpu.raw_cpu_ptr(),
                _ => return Err(err("expected CPU storage")),
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
            device: DLDevice { device_type: DL_CPU, device_id: 0 },
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
    let result = unsafe { read(&(*managed).dl_tensor) };
    // SAFETY: the tensor has not been consumed before (caller's contract); the deleter is called exactly once, here.
    unsafe {
        if let Some(d) = (*managed).deleter {
            d(managed);
        }
    }
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
        assert!(!t.storage().is_dlpack_pinned());
        assert_eq!(back.to_vec::<f32>().unwrap(), vec![9.0, 8.0, 7.0, 6.0]);
        assert!(t.retarget(&other).is_ok());
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
