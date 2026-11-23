//! Runtime compilation of CUDA C++ (`.cu`) kernels with NVRTC, loaded dynamically like the driver API in
//! [`cuda_driver`](super::cuda_driver): no CUDA Toolkit is needed at build time, and none at run time beyond the small
//! NVRTC library (`nvrtc64_*_0.dll` / `libnvrtc.so.12`), which ships in the `nvidia-cuda-nvrtc-cu12` pip wheel and in every
//! CUDA toolkit.
//!
//! Kernels are compiled to a CUBIN for the exact architecture of the device (`sm_89` for an RTX 4060), so the driver
//! needs no PTX JIT and no newer PTX ISA than it supports. Where NVRTC cannot be found, [`available`] is false and the
//! CUDA backend keeps using its hand-written PTX kernels.
//!
//! Search order for the library: `$OXIDEON_NVRTC` (a full path), then `$CUDA_PATH/bin`, then the default loader path
//! (`PATH` / `LD_LIBRARY_PATH`), then `nvidia/cuda_nvrtc/bin` inside the directories of `$OXIDEON_NVRTC_SITE`
//! (`site-packages` of a Python environment, `;`/`:` separated).

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

type NvrtcProgram = *mut c_void;

struct Api {
    _lib: *mut c_void,
    create_program: unsafe extern "C" fn(*mut NvrtcProgram, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    destroy_program: unsafe extern "C" fn(*mut NvrtcProgram) -> c_int,
    compile_program: unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> c_int,
    get_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int,
    get_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int,
    get_cubin_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int,
    get_cubin: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int,
    get_ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int,
    get_ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
}

// SAFETY: the function pointers and the library handle are immutable after loading; NVRTC is thread-safe across programs.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

#[cfg(windows)]
mod dynlib {
    use std::ffi::{CStr, c_char, c_void};
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn SetDllDirectoryW(dir: *const u16) -> i32;
    }
    /// The wide (UTF-16) form of a path that was built from a Rust `String`: the ANSI loader entry points cannot name a
    /// directory such as `C:/Users/Jürgen/...` whose characters are outside the active code page.
    fn wide(s: &CStr) -> Option<Vec<u16>> {
        Some(s.to_str().ok()?.encode_utf16().chain(std::iter::once(0)).collect())
    }
    pub unsafe fn open(name: &CStr) -> *mut c_void {
        match wide(name) {
            Some(w) => unsafe { LoadLibraryW(w.as_ptr()) },
            None => std::ptr::null_mut(),
        }
    }
    pub unsafe fn sym(lib: *mut c_void, name: &CStr) -> *mut c_void {
        unsafe { GetProcAddress(lib, name.as_ptr()) }
    }
    /// NVRTC loads `nvrtc-builtins64_*.dll` from the loader path when it compiles: make its directory searchable.
    pub unsafe fn add_search_dir(dir: &CStr) {
        if let Some(w) = wide(dir) {
            unsafe { SetDllDirectoryW(w.as_ptr()) };
        }
    }
}

#[cfg(not(any(windows, unix)))]
mod dynlib {
    //! No dynamic loading on this target (for example WebAssembly): every library is reported as missing.
    use std::ffi::{CStr, c_void};
    pub unsafe fn open(_name: &CStr) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn sym(_lib: *mut c_void, _name: &CStr) -> *mut c_void {
        std::ptr::null_mut()
    }
    pub unsafe fn add_search_dir(_dir: &CStr) {}
}

#[cfg(unix)]
mod dynlib {
    use std::ffi::{CStr, c_char, c_int, c_void};
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    pub unsafe fn open(name: &CStr) -> *mut c_void {
        unsafe { dlopen(name.as_ptr(), 2 /* RTLD_NOW */) }
    }
    pub unsafe fn sym(lib: *mut c_void, name: &CStr) -> *mut c_void {
        unsafe { dlsym(lib, name.as_ptr()) }
    }
    pub unsafe fn add_search_dir(_dir: &CStr) {}
}

unsafe fn load_fn<T>(lib: *mut c_void, name: &[u8]) -> std::result::Result<T, String> {
    let c = CStr::from_bytes_with_nul(name).map_err(|e| e.to_string())?;
    // SAFETY: `lib` is a live library handle and `c` is NUL-terminated.
    let p = unsafe { dynlib::sym(lib, c) };
    if p.is_null() {
        return Err(format!("missing NVRTC symbol {}", c.to_string_lossy()));
    }
    // SAFETY: `T` is the matching `extern "C"` function pointer type; the pointer is non-null.
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

fn candidate_paths() -> Vec<(CString, Option<CString>)> {
    let mut out: Vec<(CString, Option<CString>)> = Vec::new();
    let mut push = |full: String, dir: Option<String>| {
        if let Ok(c) = CString::new(full) {
            out.push((c, dir.and_then(|d| CString::new(d).ok())));
        }
    };
    if let Ok(p) = std::env::var("OXIDEON_NVRTC") {
        let dir = std::path::Path::new(&p).parent().map(|d| d.display().to_string());
        push(p, dir);
    }
    let names: &[&str] = if cfg!(windows) { &["nvrtc64_130_0.dll", "nvrtc64_129_0.dll", "nvrtc64_128_0.dll", "nvrtc64_126_0.dll", "nvrtc64_125_0.dll", "nvrtc64_124_0.dll", "nvrtc64_123_0.dll", "nvrtc64_122_0.dll", "nvrtc64_121_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"] } else { &["libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2"] };
    let mut dirs: Vec<String> = Vec::new();
    if let Ok(c) = std::env::var("CUDA_PATH") {
        dirs.push(format!("{c}/bin"));
        dirs.push(format!("{c}/lib64"));
    }
    if let Ok(sites) = std::env::var("OXIDEON_NVRTC_SITE") {
        for s in sites.split(if cfg!(windows) { ';' } else { ':' }).filter(|s| !s.is_empty()) {
            dirs.push(format!("{s}/nvidia/cuda_nvrtc/bin"));
            dirs.push(format!("{s}/nvidia/cuda_nvrtc/lib"));
        }
    }
    for d in &dirs {
        for n in names {
            let full = format!("{d}/{n}");
            if std::path::Path::new(&full).exists() {
                push(full, Some(d.clone()));
            }
        }
    }
    for n in names {
        push((*n).to_string(), None);
    }
    out
}

fn load() -> std::result::Result<Api, String> {
    for (path, dir) in candidate_paths() {
        if let Some(d) = &dir {
            // SAFETY: plain FFI call with a valid NUL-terminated string.
            unsafe { dynlib::add_search_dir(d) };
        }
        // SAFETY: `path` is NUL-terminated.
        let lib = unsafe { dynlib::open(&path) };
        if lib.is_null() {
            continue;
        }
        // SAFETY: `lib` is a valid NVRTC handle; each symbol is requested with its documented signature.
        let api = unsafe {
            Api {
                _lib: lib,
                create_program: load_fn(lib, b"nvrtcCreateProgram\0")?,
                destroy_program: load_fn(lib, b"nvrtcDestroyProgram\0")?,
                compile_program: load_fn(lib, b"nvrtcCompileProgram\0")?,
                get_log_size: load_fn(lib, b"nvrtcGetProgramLogSize\0")?,
                get_log: load_fn(lib, b"nvrtcGetProgramLog\0")?,
                get_cubin_size: load_fn(lib, b"nvrtcGetCUBINSize\0")?,
                get_cubin: load_fn(lib, b"nvrtcGetCUBIN\0")?,
                get_ptx_size: load_fn(lib, b"nvrtcGetPTXSize\0")?,
                get_ptx: load_fn(lib, b"nvrtcGetPTX\0")?,
                version: load_fn(lib, b"nvrtcVersion\0")?,
            }
        };
        return Ok(api);
    }
    Err("NVRTC library not found (set OXIDEON_NVRTC to its full path, or install `nvidia-cuda-nvrtc-cu12`)".into())
}

fn api() -> Result<&'static Api> {
    static A: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    A.get_or_init(load).as_ref().map_err(|e| Error::Backend(e.clone()))
}

/// Whether NVRTC could be loaded.
pub fn available() -> bool {
    api().is_ok()
}

/// `(major, minor)` of the loaded NVRTC.
pub fn version() -> Result<(i32, i32)> {
    let a = api()?;
    let (mut ma, mut mi) = (0, 0);
    // SAFETY: both out-pointers are valid.
    let rc = unsafe { (a.version)(&mut ma, &mut mi) };
    if rc != 0 {
        return Err(Error::Backend(format!("nvrtcVersion failed ({rc})")));
    }
    Ok((ma, mi))
}

/// A compiled kernel image: a CUBIN when compiled for a real architecture, PTX text otherwise.
#[derive(Clone, Debug)]
pub struct Image {
    pub bytes: Vec<u8>,
    pub is_cubin: bool,
}

/// Compiles `source` (named `name` in diagnostics) for compute capability `major.minor`. `headers` are `(include name,
/// contents)` pairs resolvable with `#include "name"`.
pub fn compile(name: &str, source: &str, headers: &[(&str, &str)], arch: (i32, i32), extra_options: &[&str]) -> Result<Image> {
    let a = api()?;
    let src = CString::new(source).map_err(|e| Error::Backend(format!("kernel source contains NUL: {e}")))?;
    let cname = CString::new(name).map_err(|e| Error::Backend(e.to_string()))?;
    let hdr_names: Vec<CString> = headers.iter().map(|(n, _)| CString::new(*n).unwrap()).collect();
    let hdr_srcs: Vec<CString> = headers.iter().map(|(_, s)| CString::new(*s).unwrap()).collect();
    let hn: Vec<*const c_char> = hdr_names.iter().map(|c| c.as_ptr()).collect();
    let hs: Vec<*const c_char> = hdr_srcs.iter().map(|c| c.as_ptr()).collect();
    let mut prog: NvrtcProgram = std::ptr::null_mut();
    // SAFETY: all pointers are valid for the call; the header arrays have `headers.len()` entries.
    let rc = unsafe { (a.create_program)(&mut prog, src.as_ptr(), cname.as_ptr(), headers.len() as c_int, hs.as_ptr(), hn.as_ptr()) };
    if rc != 0 {
        return Err(Error::Backend(format!("nvrtcCreateProgram failed ({rc})")));
    }
    struct Guard(*mut c_void, unsafe extern "C" fn(*mut NvrtcProgram) -> c_int);
    impl Drop for Guard {
        fn drop(&mut self) {
            let mut p = self.0;
            // SAFETY: `p` is the program created above, destroyed exactly once.
            unsafe { (self.1)(&mut p) };
        }
    }
    let _guard = Guard(prog, a.destroy_program);

    let arch_opt = format!("--gpu-architecture=sm_{}{}", arch.0, arch.1);
    let mut opts: Vec<CString> = vec![
        CString::new(arch_opt).unwrap(),
        CString::new("--std=c++17").unwrap(),
        CString::new("--fmad=true").unwrap(),
        CString::new("-default-device").unwrap(),
    ];
    opts.extend(extra_options.iter().map(|o| CString::new(*o).unwrap()));
    let optp: Vec<*const c_char> = opts.iter().map(|c| c.as_ptr()).collect();
    // SAFETY: `prog` is valid and `optp` holds `opts.len()` valid option strings.
    let rc = unsafe { (a.compile_program)(prog, optp.len() as c_int, optp.as_ptr()) };
    let log = {
        let mut n = 0usize;
        // SAFETY: valid program and out-pointer, then a buffer of the reported size.
        unsafe { (a.get_log_size)(prog, &mut n) };
        let mut buf = vec![0u8; n.max(1)];
        // SAFETY: `buf` holds `n` bytes.
        unsafe { (a.get_log)(prog, buf.as_mut_ptr() as *mut c_char) };
        String::from_utf8_lossy(&buf).trim_end_matches('\0').trim().to_string()
    };
    if rc != 0 {
        return Err(Error::Backend(format!("NVRTC failed to compile {name} (code {rc}):\n{log}")));
    }
    let mut n = 0usize;
    // SAFETY: valid program; a successful compile for a real `sm_XX` target produces a CUBIN.
    let rc = unsafe { (a.get_cubin_size)(prog, &mut n) };
    if rc == 0 && n > 0 {
        let mut bytes = vec![0u8; n];
        // SAFETY: `bytes` holds `n` bytes.
        unsafe { (a.get_cubin)(prog, bytes.as_mut_ptr() as *mut c_char) };
        return Ok(Image { bytes, is_cubin: true });
    }
    let mut n = 0usize;
    // SAFETY: as above for PTX.
    let rc = unsafe { (a.get_ptx_size)(prog, &mut n) };
    if rc != 0 {
        return Err(Error::Backend(format!("nvrtcGetPTXSize failed ({rc})")));
    }
    let mut bytes = vec![0u8; n];
    // SAFETY: `bytes` holds `n` bytes.
    unsafe { (a.get_ptx)(prog, bytes.as_mut_ptr() as *mut c_char) };
    Ok(Image { bytes, is_cubin: false })
}

/// Names of the `extern "C"` kernels found in `source` (a `__global__ void name(` scan), for diagnostics and tests.
pub fn kernel_names(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen: HashMap<String, ()> = HashMap::new();
    for line in source.lines() {
        if let Some(i) = line.find("__global__") {
            let rest = &line[i + 10..];
            if let Some(j) = rest.find('(') {
                let name = rest[..j].split_whitespace().last().unwrap_or("").to_string();
                if !name.is_empty() && seen.insert(name.clone(), ()).is_none() {
                    out.push(name);
                }
            }
        }
    }
    out
}
