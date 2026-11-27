//! f32 matrix products through NVIDIA cuBLAS, loaded dynamically like NVRTC ([`nvrtc`](super::nvrtc)): no toolkit at build
//! time; at run time `libcublas.so.13/12` (`cublas64_13/12.dll`) from the CUDA toolkit or the `nvidia-cublas-cu12` wheel.
//! Where it cannot be loaded, or `OXIDEON_CUDA_CUBLAS=0`, [`sgemm_strided_batched`] returns `Ok(false)` and the caller uses
//! the library's own `.cu` GEMM kernels. cuBLAS shares the device's primary context and the default stream with every other
//! kernel of this backend, so no extra synchronisation is needed.

use crate::error::{Error, Result};
use std::ffi::{CString, c_int, c_void};
use std::sync::{Mutex, OnceLock};

type Handle = *mut c_void;
type CreateFn = unsafe extern "C" fn(*mut Handle) -> c_int;
type SetMathModeFn = unsafe extern "C" fn(Handle, c_int) -> c_int;
type SgemmSbFn = unsafe extern "C" fn(
    Handle, c_int, c_int, c_int, c_int, c_int, *const f32, *const f32, c_int, i64, *const f32, c_int, i64,
    *const f32, *mut f32, c_int, i64, c_int,
) -> c_int;

struct Api {
    _lib: *mut c_void,
    create: CreateFn,
    set_math_mode: SetMathModeFn,
    sgemm_sb: SgemmSbFn,
}

// SAFETY: immutable after loading; handles are created per device and used under a mutex.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

#[cfg(unix)]
unsafe fn open(name: &CString) -> *mut c_void {
    unsafe extern "C" {
        fn dlopen(name: *const std::ffi::c_char, flags: c_int) -> *mut c_void;
    }
    unsafe { dlopen(name.as_ptr(), 2) }
}
#[cfg(unix)]
unsafe fn sym(lib: *mut c_void, name: &CString) -> *mut c_void {
    unsafe extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
    }
    unsafe { dlsym(lib, name.as_ptr()) }
}
#[cfg(windows)]
unsafe fn open(name: &CString) -> *mut c_void {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const std::ffi::c_char) -> *mut c_void;
    }
    unsafe { LoadLibraryA(name.as_ptr()) }
}
#[cfg(windows)]
unsafe fn sym(lib: *mut c_void, name: &CString) -> *mut c_void {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetProcAddress(module: *mut c_void, name: *const std::ffi::c_char) -> *mut c_void;
    }
    unsafe { GetProcAddress(lib, name.as_ptr()) }
}
#[cfg(not(any(unix, windows)))]
unsafe fn open(_: &CString) -> *mut c_void {
    std::ptr::null_mut()
}
#[cfg(not(any(unix, windows)))]
unsafe fn sym(_: *mut c_void, _: &CString) -> *mut c_void {
    std::ptr::null_mut()
}

fn load() -> std::result::Result<Api, String> {
    if std::env::var("OXIDEON_CUDA_CUBLAS").is_ok_and(|v| v == "0") {
        return Err("disabled by OXIDEON_CUDA_CUBLAS=0".into());
    }
    let mut names: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("OXIDEON_CUBLAS") {
        names.push(p);
    }
    let base: &[&str] = if cfg!(windows) { &["cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll"] } else { &["libcublas.so.13", "libcublas.so.12", "libcublas.so.11", "libcublas.so"] };
    if let Ok(c) = std::env::var("CUDA_PATH") {
        for n in base {
            names.push(format!("{c}/{}/{n}", if cfg!(windows) { "bin" } else { "lib64" }));
        }
    }
    names.extend(base.iter().map(|s| s.to_string()));
    for n in names {
        let Ok(c) = CString::new(n) else { continue };
        // SAFETY: NUL-terminated name.
        let lib = unsafe { open(&c) };
        if lib.is_null() {
            continue;
        }
        let get = |s: &str| -> std::result::Result<*mut c_void, String> {
            let cs = CString::new(s).map_err(|e| e.to_string())?;
            // SAFETY: live handle, NUL-terminated symbol.
            let p = unsafe { sym(lib, &cs) };
            if p.is_null() { Err(format!("missing cuBLAS symbol {s}")) } else { Ok(p) }
        };
        // SAFETY: each pointer is the documented cuBLAS v2 entry point with the matching signature.
        unsafe {
            return Ok(Api {
                _lib: lib,
                create: std::mem::transmute::<*mut c_void, CreateFn>(get("cublasCreate_v2")?),
                set_math_mode: std::mem::transmute::<*mut c_void, SetMathModeFn>(get("cublasSetMathMode")?),
                sgemm_sb: std::mem::transmute::<*mut c_void, SgemmSbFn>(get("cublasSgemmStridedBatched")?),
            });
        }
    }
    Err("cuBLAS library not found".into())
}

fn api() -> Option<&'static Api> {
    static A: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    A.get_or_init(load).as_ref().ok()
}

/// Whether cuBLAS was found (and is not disabled).
pub fn available() -> bool {
    api().is_some()
}

struct Handles(Vec<(usize, usize, c_int)>); // (device, handle, math mode)
// SAFETY: cuBLAS handles are only used while the mutex is held.
unsafe impl Send for Handles {}

fn handles() -> &'static Mutex<Handles> {
    static H: OnceLock<Mutex<Handles>> = OnceLock::new();
    H.get_or_init(|| Mutex::new(Handles(Vec::new())))
}

/// One matrix of a product: base pointer (bytes already offset), row / column element strides, batch element stride.
#[derive(Clone, Copy)]
pub struct Mat {
    pub ptr: u64,
    pub rs: i64,
    pub cs: i64,
    pub bs: i64,
}

/// Row-major `C[b] (m x n, contiguous) = A[b] (m x k) @ B[b] (k x n)` for `batch` products whose operands are each one strided
/// matrix per batch step. `Ok(false)` when cuBLAS is unavailable or an operand has no unit stride (the caller then uses its
/// own kernel). The current context must be the device's (the caller has just allocated `c` there).
#[allow(clippy::too_many_arguments)]
pub fn sgemm_strided_batched(device: usize, a: Mat, b: Mat, c: u64, m: usize, n: usize, k: usize, batch: usize, tf32: bool) -> Result<bool> {
    let Some(api) = api() else { return Ok(false) };
    // C row-major = C^T column-major (n x m, ld n) = B^T (n x k) @ A^T (k x m), each described in column-major terms.
    // X = B^T: unit column stride -> plain column-major with ld = rs; unit row stride -> transposed, ld = cs.
    let pick = |rs: i64, cs: i64, rows_x: usize, cols_x: usize| -> Option<(c_int, c_int)> {
        // returns (op, ld) for the column-major matrix X (rows_x x cols_x) where X[r][c] = M[c][r] of the row-major view
        if cs == 1 && rs >= rows_x.max(1) as i64 {
            Some((0, rs as c_int))
        } else if rs == 1 && cs >= cols_x.max(1) as i64 {
            Some((1, cs as c_int))
        } else {
            None
        }
    };
    let (Some((opb, ldb)), Some((opa, lda))) = (pick(b.rs, b.cs, n, k), pick(a.rs, a.cs, k, m)) else { return Ok(false) };
    if [m, n, k, batch].iter().any(|&v| v > c_int::MAX as usize) {
        return Ok(false);
    }
    let mode: c_int = if tf32 { 3 /* CUBLAS_TF32_TENSOR_OP_MATH */ } else { 0 /* CUBLAS_DEFAULT_MATH */ };
    let mut hs = handles().lock().unwrap_or_else(|e| e.into_inner());
    let h = match hs.0.iter_mut().find(|e| e.0 == device) {
        Some(e) => e,
        None => {
            let mut h: Handle = std::ptr::null_mut();
            // SAFETY: valid out-pointer; the device's primary context is current.
            let rc = unsafe { (api.create)(&mut h) };
            if rc != 0 {
                return Err(Error::Backend(format!("cublasCreate failed with status {rc}")));
            }
            hs.0.push((device, h as usize, -1));
            hs.0.last_mut().unwrap()
        }
    };
    if h.2 != mode {
        // SAFETY: live handle.
        let rc = unsafe { (api.set_math_mode)(h.1 as Handle, mode) };
        if rc != 0 {
            return Err(Error::Backend(format!("cublasSetMathMode failed with status {rc}")));
        }
        h.2 = mode;
    }
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: device pointers from live allocations of the stated extents; dimensions checked above.
    let rc = unsafe {
        (api.sgemm_sb)(h.1 as Handle, opb, opa, n as c_int, m as c_int, k as c_int, &alpha, b.ptr as *const f32, ldb, b.bs, a.ptr as *const f32, lda, a.bs,
                       &beta, c as *mut f32, n as c_int, (m * n) as i64, batch as c_int)
    };
    if rc != 0 {
        return Err(Error::Backend(format!("cublasSgemmStridedBatched failed with status {rc}")));
    }
    Ok(true)
}
