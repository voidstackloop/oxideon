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

/// Elementwise GELU (exact, erf-based).
///
/// # Safety
/// `x` must be a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_gelu(x: *const OxideonTensor) -> *mut OxideonTensor {
    if x.is_null() {
        set_err("null operand in oxideon_tensor_gelu");
        return ptr::null_mut();
    }
    guard(ptr::null_mut(), || boxed(unsafe { (*x).inner.gelu() }))
}

/// Elementwise SiLU (swish) `x * sigmoid(x)`.
///
/// # Safety
/// `x` must be a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_silu(x: *const OxideonTensor) -> *mut OxideonTensor {
    if x.is_null() {
        set_err("null operand in oxideon_tensor_silu");
        return ptr::null_mut();
    }
    guard(ptr::null_mut(), || boxed(unsafe { (*x).inner.silu() }))
}

/// Softmax along `axis` (negative values count from the last dimension).
///
/// # Safety
/// `x` must be a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_softmax(x: *const OxideonTensor, axis: i64) -> *mut OxideonTensor {
    if x.is_null() {
        set_err("null operand in oxideon_tensor_softmax");
        return ptr::null_mut();
    }
    guard(ptr::null_mut(), || boxed(unsafe { (*x).inner.softmax(axis as isize) }))
}

/// Reshapes `x` to `shape[ndim]` (the element count must be unchanged).
///
/// # Safety
/// `x` must be a valid `OxideonTensor` pointer and `shape` must point to `ndim` valid `usize` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_reshape(
    x: *const OxideonTensor,
    shape: *const usize,
    ndim: usize,
) -> *mut OxideonTensor {
    if x.is_null() || (ndim > 0 && shape.is_null()) {
        set_err("null pointer passed to oxideon_tensor_reshape");
        return ptr::null_mut();
    }
    let shp = if ndim == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(shape, ndim) } };
    guard(ptr::null_mut(), || boxed(unsafe { (*x).inner.reshape(shp) }))
}

/// Swaps dimensions `dim0` and `dim1` (negative values count from the last dimension).
///
/// # Safety
/// `x` must be a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_transpose(x: *const OxideonTensor, dim0: i64, dim1: i64) -> *mut OxideonTensor {
    if x.is_null() {
        set_err("null operand in oxideon_tensor_transpose");
        return ptr::null_mut();
    }
    guard(ptr::null_mut(), || boxed(unsafe { (*x).inner.transpose(dim0 as isize, dim1 as isize) }))
}

/// Layer normalization over the trailing `normalized_ndim` dimensions, whose sizes are `normalized_shape[normalized_ndim]`.
/// `weight` and `bias` are optional (pass NULL to skip them).
///
/// # Safety
/// `x` must be a valid `OxideonTensor` pointer, `normalized_shape` must point to `normalized_ndim` valid `usize` values,
/// and `weight` / `bias` must each be NULL or a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_layer_norm(
    x: *const OxideonTensor,
    normalized_shape: *const usize,
    normalized_ndim: usize,
    weight: *const OxideonTensor,
    bias: *const OxideonTensor,
    eps: f64,
) -> *mut OxideonTensor {
    if x.is_null() || normalized_shape.is_null() || normalized_ndim == 0 {
        set_err("null pointer or empty normalized_shape passed to oxideon_tensor_layer_norm");
        return ptr::null_mut();
    }
    let shp = unsafe { std::slice::from_raw_parts(normalized_shape, normalized_ndim) };
    let (w, b) = unsafe { (weight.as_ref().map(|t| &t.inner), bias.as_ref().map(|t| &t.inner)) };
    guard(ptr::null_mut(), || boxed(crate::nn::functional_ext::layer_norm(unsafe { &(*x).inner }, shp, w, b, eps)))
}

/// 2D convolution of `input: [N, C_in, H, W]` with `weight: [C_out, C_in, kH, kW]` and an optional `bias: [C_out]` (NULL to skip).
///
/// # Safety
/// `input` and `weight` must be valid `OxideonTensor` pointers and `bias` must be NULL or a valid `OxideonTensor` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxideon_tensor_conv2d(
    input: *const OxideonTensor,
    weight: *const OxideonTensor,
    bias: *const OxideonTensor,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> *mut OxideonTensor {
    if input.is_null() || weight.is_null() {
        set_err("null operand in oxideon_tensor_conv2d");
        return ptr::null_mut();
    }
    let b = unsafe { bias.as_ref().map(|t| &t.inner) };
    guard(ptr::null_mut(), || {
        boxed(unsafe {
            (*input).inner.conv2d(&(*weight).inner, b, (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn make(data: &[f32], shape: &[usize]) -> *mut OxideonTensor {
        unsafe { oxideon_tensor_from_f32(data.as_ptr(), data.len(), shape.as_ptr(), shape.len()) }
    }

    unsafe fn read(t: *const OxideonTensor) -> Vec<f32> {
        let mut out = vec![0.0f32; unsafe { oxideon_tensor_numel(t) }];
        assert_eq!(unsafe { oxideon_tensor_copy_f32(t, out.as_mut_ptr(), out.len()) }, 0);
        out
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn c_api_activations_softmax_and_layer_norm() {
        unsafe {
            let x = make(&[-1.0, 0.0, 1.0, 2.0], &[2, 2]);
            let g = oxideon_tensor_gelu(x);
            assert_close(&read(g), &[-0.158_655_26, 0.0, 0.841_344_7, 1.954_499_7]);
            let s = oxideon_tensor_silu(x);
            assert_close(&read(s), &[-0.268_941_4, 0.0, 0.731_058_6, 1.761_594_2]);
            let sm = oxideon_tensor_softmax(x, -1);
            assert_close(&read(sm), &[0.268_941_4, 0.731_058_6, 0.268_941_4, 0.731_058_6]);
            assert!(oxideon_tensor_softmax(x, 5).is_null(), "out-of-range axis is an error");

            // layer_norm over the last dim: each row [a, b] -> [-1, 1] (eps = 0), then weight 2 / bias 1 -> [-1, 3]
            let w = make(&[2.0, 2.0], &[2]);
            let b = make(&[1.0, 1.0], &[2]);
            let ln = oxideon_tensor_layer_norm(x, [2usize].as_ptr(), 1, std::ptr::null(), std::ptr::null(), 0.0);
            assert_close(&read(ln), &[-1.0, 1.0, -1.0, 1.0]);
            let lnwb = oxideon_tensor_layer_norm(x, [2usize].as_ptr(), 1, w, b, 0.0);
            assert_close(&read(lnwb), &[-1.0, 3.0, -1.0, 3.0]);
            assert!(oxideon_tensor_layer_norm(x, [3usize].as_ptr(), 1, w, b, 1e-5).is_null(), "mismatched normalized_shape");
            assert!(oxideon_tensor_layer_norm(x, std::ptr::null(), 1, w, b, 1e-5).is_null());

            for t in [x, g, s, sm, w, b, ln, lnwb] {
                oxideon_tensor_free(t);
            }
        }
    }

    #[test]
    fn c_api_reshape_and_transpose() {
        unsafe {
            let x = make(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let r = oxideon_tensor_reshape(x, [3usize, 2].as_ptr(), 2);
            let mut shp = [0usize; 2];
            assert_eq!(oxideon_tensor_shape(r, shp.as_mut_ptr(), 2), 0);
            assert_eq!(shp, [3, 2]);
            assert_eq!(read(r), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            assert!(oxideon_tensor_reshape(x, [4usize].as_ptr(), 1).is_null(), "element count must match");
            assert!(oxideon_tensor_reshape(x, std::ptr::null(), 2).is_null());

            let t = oxideon_tensor_transpose(x, 0, 1);
            assert_eq!(oxideon_tensor_shape(t, shp.as_mut_ptr(), 2), 0);
            assert_eq!(shp, [3, 2]);
            assert_eq!(read(t), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            let t_neg = oxideon_tensor_transpose(x, -1, -2);
            assert_eq!(read(t_neg), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
            assert!(oxideon_tensor_transpose(x, 0, 7).is_null());

            for p in [x, r, t, t_neg] {
                oxideon_tensor_free(p);
            }
        }
    }

    #[test]
    fn c_api_conv2d() {
        unsafe {
            // 1x1x3x3 input, 1x1x2x2 all-ones kernel, bias 0.5: each output is a 2x2 window sum + 0.5
            let x = make(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 1, 3, 3]);
            let w = make(&[1.0; 4], &[1, 1, 2, 2]);
            let b = make(&[0.5], &[1]);
            let y = oxideon_tensor_conv2d(x, w, b, 1, 1, 0, 0, 1, 1);
            assert_close(&read(y), &[12.5, 16.5, 24.5, 28.5]);
            let y_nb = oxideon_tensor_conv2d(x, w, std::ptr::null(), 1, 1, 0, 0, 1, 1);
            assert_close(&read(y_nb), &[12.0, 16.0, 24.0, 28.0]);
            // padding 1 grows the output to 4x4
            let y_pad = oxideon_tensor_conv2d(x, w, std::ptr::null(), 1, 1, 1, 1, 1, 1);
            assert_eq!(oxideon_tensor_numel(y_pad), 16);
            // a 3D input is an error, not a crash
            let flat = make(&[1.0; 9], &[3, 3, 1]);
            assert!(oxideon_tensor_conv2d(flat, w, std::ptr::null(), 1, 1, 0, 0, 1, 1).is_null());
            assert!(oxideon_tensor_conv2d(std::ptr::null(), w, std::ptr::null(), 1, 1, 0, 0, 1, 1).is_null());
            for p in [x, w, b, y, y_nb, y_pad, flat] {
                oxideon_tensor_free(p);
            }
        }
    }

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
