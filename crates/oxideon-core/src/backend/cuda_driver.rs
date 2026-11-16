//! Dynamic CUDA Driver API backend (`nvcuda.dll` on Windows / `libcuda.so.1` on Unix).
//!
//! Loads the NVIDIA CUDA driver at runtime without a build-time CUDA Toolkit dependency,
//! providing real device discovery, VRAM telemetry, device memory allocation (`cuMemAlloc`),
//! host<->device transfers (`cuMemcpyHtoD` / `cuMemcpyDtoH`), and PTX kernel execution.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::{Arc, Mutex, OnceLock};

use crate::error::{Error, Result};

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub index: usize,
    pub name: String,
    pub total_memory: usize,
    pub major: i32,
    pub minor: i32,
    pub multi_processor_count: i32,
}

struct CudaApi {
    _lib: *mut c_void,
    cu_init: unsafe extern "C" fn(c_uint) -> CUresult,
    cu_device_get_count: unsafe extern "C" fn(*mut c_int) -> CUresult,
    cu_device_get: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    cu_device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    cu_device_total_mem: unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult,
    cu_device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    cu_device_primary_ctx_retain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    cu_ctx_set_current: unsafe extern "C" fn(CUcontext) -> CUresult,
    cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,
    cu_mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    cu_mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    cu_mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    cu_memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    cu_module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    cu_module_unload: unsafe extern "C" fn(CUmodule) -> CUresult,
    cu_module_get_function:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cu_launch_kernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

#[cfg(windows)]
mod dynlib {
    use std::ffi::{CStr, c_char, c_void};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
        fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    }

    pub unsafe fn open(name: &CStr) -> *mut c_void {
        unsafe { LoadLibraryA(name.as_ptr()) }
    }

    pub unsafe fn sym(lib: *mut c_void, name: &CStr) -> *mut c_void {
        unsafe { GetProcAddress(lib, name.as_ptr()) }
    }
}

#[cfg(unix)]
mod dynlib {
    use std::ffi::{CStr, c_int, c_void};

    const RTLD_LAZY: c_int = 1;

    unsafe extern "C" {
        fn dlopen(filename: *const std::ffi::c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const std::ffi::c_char) -> *mut c_void;
    }

    pub unsafe fn open(name: &CStr) -> *mut c_void {
        unsafe { dlopen(name.as_ptr(), RTLD_LAZY) }
    }

    pub unsafe fn sym(lib: *mut c_void, name: &CStr) -> *mut c_void {
        unsafe { dlsym(lib, name.as_ptr()) }
    }
}

unsafe fn load_fn<T>(lib: *mut c_void, name: &[u8]) -> std::result::Result<T, String> {
    let cstr = CStr::from_bytes_with_nul(name).map_err(|e| e.to_string())?;
    let ptr = unsafe { dynlib::sym(lib, cstr) };
    if ptr.is_null() {
        return Err(format!("missing CUDA symbol {}", cstr.to_string_lossy()));
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

impl CudaApi {
    fn load() -> std::result::Result<Self, String> {
        #[cfg(windows)]
        let lib_name = c"nvcuda.dll";
        #[cfg(unix)]
        let lib_name = c"libcuda.so.1";

        let lib = unsafe { dynlib::open(lib_name) };
        if lib.is_null() {
            return Err("CUDA driver library not found".into());
        }
        let api = unsafe {
            Self {
                _lib: lib,
                cu_init: load_fn(lib, b"cuInit\0")?,
                cu_device_get_count: load_fn(lib, b"cuDeviceGetCount\0")?,
                cu_device_get: load_fn(lib, b"cuDeviceGet\0")?,
                cu_device_get_name: load_fn(lib, b"cuDeviceGetName\0")?,
                cu_device_total_mem: load_fn(lib, b"cuDeviceTotalMem_v2\0")?,
                cu_device_get_attribute: load_fn(lib, b"cuDeviceGetAttribute\0")?,
                cu_device_primary_ctx_retain: load_fn(lib, b"cuDevicePrimaryCtxRetain\0")?,
                cu_ctx_set_current: load_fn(lib, b"cuCtxSetCurrent\0")?,
                cu_ctx_synchronize: load_fn(lib, b"cuCtxSynchronize\0")?,
                cu_mem_get_info: load_fn(lib, b"cuMemGetInfo_v2\0")?,
                cu_mem_alloc: load_fn(lib, b"cuMemAlloc_v2\0")?,
                cu_mem_free: load_fn(lib, b"cuMemFree_v2\0")?,
                cu_memcpy_htod: load_fn(lib, b"cuMemcpyHtoD_v2\0")?,
                cu_memcpy_dtoh: load_fn(lib, b"cuMemcpyDtoH_v2\0")?,
                cu_module_load_data: load_fn(lib, b"cuModuleLoadData\0")?,
                cu_module_unload: load_fn(lib, b"cuModuleUnload\0")?,
                cu_module_get_function: load_fn(lib, b"cuModuleGetFunction\0")?,
                cu_launch_kernel: load_fn(lib, b"cuLaunchKernel\0")?,
            }
        };
        let rc = unsafe { (api.cu_init)(0) };
        if rc != CUDA_SUCCESS {
            return Err(format!("cuInit failed with CUDA error {rc}"));
        }
        Ok(api)
    }
}

static CUDA_API: OnceLock<std::result::Result<CudaApi, String>> = OnceLock::new();
static CUDA_CTXS: OnceLock<Mutex<Vec<Option<usize>>>> = OnceLock::new();

fn api() -> Result<&'static CudaApi> {
    CUDA_API
        .get_or_init(CudaApi::load)
        .as_ref()
        .map_err(|e| Error::Backend(e.clone()))
}

fn check(rc: CUresult, op: &str) -> Result<()> {
    if rc == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Backend(format!("CUDA {op} failed with error code {rc}")))
    }
}

fn ensure_context(device_index: usize) -> Result<()> {
    let a = api()?;
    let count = cuda_device_count();
    if device_index >= count {
        return Err(Error::Backend(format!(
            "CUDA device {device_index} out of range (count={count})"
        )));
    }
    let ctxs = CUDA_CTXS.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = ctxs.lock().map_err(|_| Error::Backend("CUDA ctx lock poisoned".into()))?;
    if guard.len() <= device_index {
        guard.resize(device_index + 1, None);
    }
    let ctx_ptr = if let Some(ptr) = guard[device_index] {
        ptr as CUcontext
    } else {
        let mut dev: CUdevice = 0;
        check(unsafe { (a.cu_device_get)(&mut dev, device_index as c_int) }, "cuDeviceGet")?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        check(
            unsafe { (a.cu_device_primary_ctx_retain)(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        guard[device_index] = Some(ctx as usize);
        ctx
    };
    check(unsafe { (a.cu_ctx_set_current)(ctx_ptr) }, "cuCtxSetCurrent")
}

/// Returns `true` if the CUDA driver is present and at least one CUDA GPU is available.
pub fn cuda_available() -> bool {
    cuda_device_count() > 0
}

/// Returns the number of CUDA-capable GPUs reported by the driver.
pub fn cuda_device_count() -> usize {
    let Ok(a) = api() else { return 0 };
    let mut count: c_int = 0;
    let rc = unsafe { (a.cu_device_get_count)(&mut count) };
    if rc == CUDA_SUCCESS && count > 0 {
        count as usize
    } else {
        0
    }
}

/// Queries hardware properties for CUDA device `index`.
pub fn cuda_device_info(index: usize) -> Result<CudaDeviceInfo> {
    let a = api()?;
    let count = cuda_device_count();
    if index >= count {
        return Err(Error::Backend(format!(
            "CUDA device {index} out of range (count={count})"
        )));
    }
    let mut dev: CUdevice = 0;
    check(unsafe { (a.cu_device_get)(&mut dev, index as c_int) }, "cuDeviceGet")?;
    let mut name_buf = [0i8; 256];
    check(
        unsafe { (a.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev) },
        "cuDeviceGetName",
    )?;
    let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    let mut total_memory: usize = 0;
    check(
        unsafe { (a.cu_device_total_mem)(&mut total_memory, dev) },
        "cuDeviceTotalMem_v2",
    )?;
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    let mut sm_count: c_int = 0;
    check(
        unsafe { (a.cu_device_get_attribute)(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) },
        "cuDeviceGetAttribute(major)",
    )?;
    check(
        unsafe { (a.cu_device_get_attribute)(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) },
        "cuDeviceGetAttribute(minor)",
    )?;
    check(
        unsafe { (a.cu_device_get_attribute)(&mut sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) },
        "cuDeviceGetAttribute(sm)",
    )?;
    Ok(CudaDeviceInfo {
        index,
        name,
        total_memory,
        major,
        minor,
        multi_processor_count: sm_count,
    })
}

/// Returns `(free_bytes, total_bytes)` on CUDA device `index`.
pub fn cuda_mem_info(index: usize) -> Result<(usize, usize)> {
    ensure_context(index)?;
    let a = api()?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    check(unsafe { (a.cu_mem_get_info)(&mut free, &mut total) }, "cuMemGetInfo_v2")?;
    Ok((free, total))
}

/// Synchronizes all work on CUDA device `index`.
pub fn cuda_synchronize(index: usize) -> Result<()> {
    ensure_context(index)?;
    let a = api()?;
    check(unsafe { (a.cu_ctx_synchronize)() }, "cuCtxSynchronize")
}

/// RAII GPU VRAM allocation on a real CUDA device.
#[derive(Debug)]
pub struct CudaBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
    device_index: usize,
}

impl CudaBuffer {
    pub fn alloc(device_index: usize, bytes: usize) -> Result<Self> {
        ensure_context(device_index)?;
        let a = api()?;
        let mut ptr: CUdeviceptr = 0;
        if bytes > 0 {
            check(unsafe { (a.cu_mem_alloc)(&mut ptr, bytes) }, "cuMemAlloc_v2")?;
        }
        Ok(Self {
            ptr,
            bytes,
            device_index,
        })
    }

    pub fn from_host_f32(device_index: usize, data: &[f32]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::alloc(device_index, bytes)?;
        if bytes > 0 {
            let a = api()?;
            check(
                unsafe { (a.cu_memcpy_htod)(buf.ptr, data.as_ptr() as *const c_void, bytes) },
                "cuMemcpyHtoD_v2",
            )?;
        }
        Ok(buf)
    }

    pub fn to_host_f32(&self, len: usize) -> Result<Vec<f32>> {
        let bytes = len * std::mem::size_of::<f32>();
        if bytes > self.bytes {
            return Err(Error::Backend("CUDA D2H read exceeds buffer size".into()));
        }
        ensure_context(self.device_index)?;
        let mut out = vec![0.0f32; len];
        if bytes > 0 {
            let a = api()?;
            check(
                unsafe { (a.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, bytes) },
                "cuMemcpyDtoH_v2",
            )?;
        }
        Ok(out)
    }

    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.ptr != 0
            && let Ok(a) = api()
            && ensure_context(self.device_index).is_ok()
        {
            unsafe {
                let _ = (a.cu_mem_free)(self.ptr);
            }
        }
    }
}

/// Loaded PTX module on a CUDA device.
pub struct CudaPtxModule {
    module: CUmodule,
    device_index: usize,
}

unsafe impl Send for CudaPtxModule {}
unsafe impl Sync for CudaPtxModule {}

impl CudaPtxModule {
    pub fn load(device_index: usize, ptx: &str) -> Result<Arc<Self>> {
        ensure_context(device_index)?;
        let a = api()?;
        let c_ptx = CString::new(ptx).map_err(|e| Error::Backend(e.to_string()))?;
        let mut module: CUmodule = std::ptr::null_mut();
        check(
            unsafe { (a.cu_module_load_data)(&mut module, c_ptx.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(Arc::new(Self {
            module,
            device_index,
        }))
    }

    pub fn launch_1d(
        &self,
        func_name: &str,
        numel: usize,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        if numel == 0 {
            return Ok(());
        }
        ensure_context(self.device_index)?;
        let a = api()?;
        let c_name = CString::new(func_name).map_err(|e| Error::Backend(e.to_string()))?;
        let mut func: CUfunction = std::ptr::null_mut();
        check(
            unsafe { (a.cu_module_get_function)(&mut func, self.module, c_name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        let block = 256u32;
        let grid = (numel as u32).div_ceil(block);
        check(
            unsafe {
                (a.cu_launch_kernel)(
                    func,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    kernel_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

impl Drop for CudaPtxModule {
    fn drop(&mut self) {
        if !self.module.is_null()
            && let Ok(a) = api()
            && ensure_context(self.device_index).is_ok()
        {
            unsafe {
                let _ = (a.cu_module_unload)(self.module);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VEC_ADD_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry vec_add_f32(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
"#;

    #[test]
    fn real_cuda_device_vram_and_ptx_kernel() {
        if !cuda_available() {
            return;
        }
        let info = cuda_device_info(0).unwrap();
        assert!(!info.name.is_empty());
        assert!(info.total_memory > 0);
        assert!(info.major >= 5);

        let (free, total) = cuda_mem_info(0).unwrap();
        assert!(total >= free && total > 0);

        let a = CudaBuffer::from_host_f32(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = CudaBuffer::from_host_f32(0, &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = CudaBuffer::alloc(0, 16).unwrap();

        let module = CudaPtxModule::load(0, VEC_ADD_PTX).unwrap();
        let mut pa = a.device_ptr();
        let mut pb = b.device_ptr();
        let mut pc = c.device_ptr();
        let mut n: u32 = 4;
        let mut params: [*mut c_void; 4] = [
            &mut pa as *mut u64 as *mut c_void,
            &mut pb as *mut u64 as *mut c_void,
            &mut pc as *mut u64 as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
        ];
        module.launch_1d("vec_add_f32", 4, &mut params).unwrap();
        cuda_synchronize(0).unwrap();
        let out = c.to_host_f32(4).unwrap();
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }
}
