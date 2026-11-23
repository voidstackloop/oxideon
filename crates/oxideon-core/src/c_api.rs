//! Stable C ABI (`extern "C"`) for embedding Oxideon in C, C++, Swift, Go, or Zig.
//!
//! Companion header: `include/oxideon.h`.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use crate::tensor::Tensor;

/// Opaque C handle to an [`OxideonTensor`].
pub struct OxideonTensor {
    pub(crate) inner: Tensor,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Runs `f`; a panic inside becomes the thread's last error and `on_panic` is returned instead of unwinding into C.
fn guard<T>(on_panic: T, f: impl FnOnce() -> T) -> T {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => v,
        Err(_) => {
            set_err("internal panic in oxideon C API call");
            on_panic
        }
    }
}

fn boxed(r: crate::error::Result<Tensor>) -> *mut OxideonTensor {
    match r {
        Ok(inner) => Box::into_raw(Box::new(OxideonTensor { inner })),
        Err(e) => {
            set_err(e);
            ptr::null_mut()
        }
    }
}

fn set_err(e: impl std::fmt::Display) {
    let msg = CString::new(e.to_string()).unwrap_or_else(|_| CString::new("oxideon error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(msg));
}

/// Returns the last thread-local error string (or NULL if none).
#[unsafe(no_mangle)]
pub extern "C" fn oxideon_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(c) => c.as_ptr(),
        None => ptr::null(),
    })
}

/// Creates a contiguous `f32` CPU tensor from `data[len]` and `shape[ndim]`.
///
/// # Safety
/// `data` must point to `len` valid `f32` values and `shape` must point to `ndim` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_from_f32(
    data: *const f32,
    len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut OxideonTensor {
    if data.is_null() || (ndim > 0 && shape.is_null()) {
        set_err("null pointer passed to oxideon_tensor_from_f32");
        return ptr::null_mut();
    }
    let slice = unsafe { std::slice::from_raw_parts(data, len) };
    let shp = if ndim == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(shape, ndim) }
    };
    guard(ptr::null_mut(), || boxed(Tensor::from_slice(slice, shp)))
}

/// Frees an [`OxideonTensor`] created by the C API.
///
/// # Safety
/// `t` must be either NULL or a valid pointer returned by an `oxideon_tensor_*` function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_free(t: *mut OxideonTensor) {
    if !t.is_null() {
        unsafe {
            drop(Box::from_raw(t));
        }
    }
}

/// Returns the number of elements in `t` (or 0 if `t` is NULL).
///
/// # Safety
/// `t` must be NULL or a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_numel(t: *const OxideonTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    unsafe { (*t).inner.numel() }
}

/// Returns the rank (`ndim`) of `t`.
///
/// # Safety
/// `t` must be NULL or a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_ndim(t: *const OxideonTensor) -> usize {
    if t.is_null() {
        return 0;
    }
    unsafe { (*t).inner.ndim() }
}

/// Copies `t`'s `f32` values into `out[cap]`. Returns `0` on success, `-1` on error.
///
/// # Safety
/// `t` must be a valid `OxideonTensor` pointer and `out` must point to at least `cap` writable `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_copy_f32(
    t: *const OxideonTensor,
    out: *mut f32,
    cap: usize,
) -> i32 {
    if t.is_null() || out.is_null() {
        set_err("null pointer in oxideon_tensor_copy_f32");
        return -1;
    }
    let tensor = unsafe { &(*t).inner };
    guard(-1, || match tensor.to_vec::<f32>() {
        Ok(v) => {
            if cap < v.len() {
                set_err("buffer capacity too small in oxideon_tensor_copy_f32");
                return -1;
            }
            unsafe {
                ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
            }
            0
        }
        Err(e) => {
            set_err(e);
            -1
        }
    })
}

/// Writes `t`'s shape into `out[cap]` (a rank-`ndim` tensor needs `cap >= ndim`). Returns `0` on success, `-1` on error.
///
/// # Safety
/// `t` must be a valid `OxideonTensor` pointer and `out` must point to at least `cap` writable `usize`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_shape(t: *const OxideonTensor, out: *mut usize, cap: usize) -> i32 {
    if t.is_null() || (cap > 0 && out.is_null()) {
        set_err("null pointer in oxideon_tensor_shape");
        return -1;
    }
    let shape = unsafe { (*t).inner.shape() };
    if cap < shape.len() {
        set_err("buffer capacity too small in oxideon_tensor_shape");
        return -1;
    }
    unsafe { ptr::copy_nonoverlapping(shape.as_ptr(), out, shape.len()) };
    0
}

/// Elementwise addition `a + b`.
///
/// # Safety
/// `a` and `b` must be valid `OxideonTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_add(
    a: *const OxideonTensor,
    b: *const OxideonTensor,
) -> *mut OxideonTensor {
    if a.is_null() || b.is_null() {
        set_err("null operand in oxideon_tensor_add");
        return ptr::null_mut();
    }
    guard(ptr::null_mut(), || boxed(unsafe { (*a).inner.add(&(*b).inner) }))
}

/// Matrix multiplication `a @ b`.
///
/// # Safety
/// `a` and `b` must be valid `OxideonTensor` pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_matmul(
    a: *const OxideonTensor,
    b: *const OxideonTensor,
) -> *mut OxideonTensor {
    if a.is_null() || b.is_null() {
        set_err("null operand in oxideon_tensor_matmul");
        return ptr::null_mut();
    }
    guard(ptr::null_mut(), || boxed(unsafe { (*a).inner.matmul(&(*b).inner) }))
}

/// Elementwise ReLU `max(x, 0)`.
///
/// # Safety
/// `x` must be a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_relu(x: *const OxideonTensor) -> *mut OxideonTensor {
    if x.is_null() {
        set_err("null operand in oxideon_tensor_relu");
        return ptr::null_mut();
    }
    guard(ptr::null_mut(), || boxed(unsafe { (*x).inner.relu() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_api_roundtrip_matmul_and_relu() {
        let a_data = [1.0f32, -2.0, 3.0, 4.0];
        let b_data = [2.0f32, 0.0, 1.0, 1.0];
        let shape = [2usize, 2];
        unsafe {
            let ta = oxideon_tensor_from_f32(a_data.as_ptr(), 4, shape.as_ptr(), 2);
            let tb = oxideon_tensor_from_f32(b_data.as_ptr(), 4, shape.as_ptr(), 2);
            let tc = oxideon_tensor_matmul(ta, tb);
            let tr = oxideon_tensor_relu(tc);
            assert_eq!(oxideon_tensor_numel(tr), 4);
            let mut shp = [0usize; 2];
            assert_eq!(oxideon_tensor_shape(tr, shp.as_mut_ptr(), 2), 0);
            assert_eq!(shp, [2, 2]);
            assert_eq!(oxideon_tensor_shape(tr, shp.as_mut_ptr(), 1), -1, "too small a buffer is an error");
            let mut out = [0.0f32; 4];
            assert_eq!(oxideon_tensor_copy_f32(tr, out.as_mut_ptr(), 4), 0);
            // [1, -2; 3, 4] @ [2, 0; 1, 1] = [0, -2; 10, 4] -> relu -> [0, 0, 10, 4]
            assert_eq!(out, [0.0, 0.0, 10.0, 4.0]);
            oxideon_tensor_free(tr);
            oxideon_tensor_free(tc);
            oxideon_tensor_free(tb);
            oxideon_tensor_free(ta);
        }
    }

    #[test]
    fn c_api_reports_errors_instead_of_crashing() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        unsafe {
            // shape does not match the data length
            assert!(oxideon_tensor_from_f32(a.as_ptr(), 3, [2usize].as_ptr(), 1).is_null());
            assert!(!oxideon_last_error_message().is_null());
            assert!(oxideon_tensor_from_f32(std::ptr::null(), 3, [3usize].as_ptr(), 1).is_null());
            let ta = oxideon_tensor_from_f32(a.as_ptr(), 3, [3usize].as_ptr(), 1);
            let tb = oxideon_tensor_from_f32(b.as_ptr(), 4, [4usize].as_ptr(), 1);
            // incompatible shapes: an error, not a panic across the boundary
            assert!(oxideon_tensor_add(ta, tb).is_null());
            assert!(oxideon_tensor_matmul(ta, std::ptr::null()).is_null());
            let mut small = [0.0f32; 2];
            assert_eq!(oxideon_tensor_copy_f32(ta, small.as_mut_ptr(), 2), -1);
            oxideon_tensor_free(ta);
            oxideon_tensor_free(tb);
            oxideon_tensor_free(std::ptr::null_mut());
        }
    }
}
