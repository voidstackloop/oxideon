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
type CUgraph = *mut c_void;
type CUgraphExec = *mut c_void;
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
    cu_module_load_data_ex: unsafe extern "C" fn(
        *mut CUmodule,
        *const c_void,
        c_uint,
        *mut c_int,
        *mut *mut c_void,
    ) -> CUresult,
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
    cu_stream_create: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    cu_stream_destroy: unsafe extern "C" fn(CUstream) -> CUresult,
    cu_stream_synchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    cu_stream_begin_capture: unsafe extern "C" fn(CUstream, c_int) -> CUresult,
    cu_stream_end_capture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult,
    cu_graph_instantiate: unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    cu_graph_launch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
    cu_graph_destroy: unsafe extern "C" fn(CUgraph) -> CUresult,
    cu_graph_exec_destroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
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
        #[cfg(not(any(windows, unix)))]
        let lib_name = c"";

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
                cu_module_load_data_ex: load_fn(lib, b"cuModuleLoadDataEx\0")?,
                cu_module_unload: load_fn(lib, b"cuModuleUnload\0")?,
                cu_module_get_function: load_fn(lib, b"cuModuleGetFunction\0")?,
                cu_launch_kernel: load_fn(lib, b"cuLaunchKernel\0")?,
                cu_stream_create: load_fn(lib, b"cuStreamCreate\0")?,
                cu_stream_destroy: load_fn(lib, b"cuStreamDestroy_v2\0")?,
                cu_stream_synchronize: load_fn(lib, b"cuStreamSynchronize\0")?,
                cu_stream_begin_capture: load_fn(lib, b"cuStreamBeginCapture_v2\0")?,
                cu_stream_end_capture: load_fn(lib, b"cuStreamEndCapture\0")?,
                cu_graph_instantiate: load_fn(lib, b"cuGraphInstantiateWithFlags\0")?,
                cu_graph_launch: load_fn(lib, b"cuGraphLaunch\0")?,
                cu_graph_destroy: load_fn(lib, b"cuGraphDestroy\0")?,
                cu_graph_exec_destroy: load_fn(lib, b"cuGraphExecDestroy\0")?,
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
    let mut name_buf: [c_char; 256] = [0; 256]; // c_char is u8 on aarch64 Linux
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

/// Waits for everything queued on the default stream of CUDA device `index`: every kernel launch and copy of this library goes
/// there. (It deliberately is not `cuCtxSynchronize`, which the driver rejects while *any* stream of the context is being
/// captured into a graph, so a capture on one thread would break reads on every other thread. Work on a [`CudaStream`] is
/// waited for with [`CudaStream::synchronize`].)
pub fn cuda_synchronize(index: usize) -> Result<()> {
    ensure_context(index)?;
    let a = api()?;
    check(unsafe { (a.cu_stream_synchronize)(std::ptr::null_mut()) }, "cuStreamSynchronize(default)")
}

/// Freed device allocations kept for reuse, by `(device, size)`. `cuMemAlloc` / `cuMemFree` cost far more than a short kernel
/// (and `cuMemFree` waits for the device), so a buffer that is dropped goes back here and the next tensor of the same size
/// takes it. Bounded by `OXIDEON_CUDA_POOL_MB` (default 1024; 0 turns pooling off); an allocation failure empties the pool and
/// retries before giving up.
struct DevicePool {
    free: std::collections::HashMap<(usize, usize), Vec<CUdeviceptr>>,
    pooled_bytes: usize,
}

fn device_pool() -> &'static Mutex<DevicePool> {
    static P: OnceLock<Mutex<DevicePool>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(DevicePool { free: std::collections::HashMap::new(), pooled_bytes: 0 }))
}

fn pool_limit() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| std::env::var("OXIDEON_CUDA_POOL_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024) << 20)
}

/// Size class of an allocation: 512-byte multiples up to 32 KiB, then four classes per power of two (at most 25% slack), so
/// tensors whose shapes differ a little from batch to batch (variable sequence lengths) still reuse each other's memory instead
/// of each missing the pool and paying `cuMemAlloc` plus a device-stalling `cuMemFree`.
fn size_class(bytes: usize) -> usize {
    if bytes <= 32 << 10 {
        return bytes.next_multiple_of(512);
    }
    let top = 1usize << (usize::BITS - 1 - bytes.leading_zeros());
    bytes.next_multiple_of(top / 4)
}

#[cfg(test)]
mod size_class_tests {
    use super::size_class;
    #[test]
    fn classes_are_coarse_and_cover() {
        assert_eq!(size_class(1), 512);
        assert_eq!(size_class(32 << 10), 32 << 10);
        for b in [(32 << 10) + 1, 1_300_000, 38_400, 7_372_800, (1 << 30) + 5] {
            let c = size_class(b);
            assert!(c >= b && c - b <= b / 4, "{b} -> {c}");
        }
        assert_eq!(size_class(1_100_000), size_class(1_200_000)); // nearby lengths share a class
    }
}

/// Returns every pooled allocation of every device to the driver.
pub fn cuda_empty_pool() {
    let Ok(a) = api() else { return };
    let mut pool = device_pool().lock().unwrap_or_else(|p| p.into_inner());
    for ((dev, _), ptrs) in pool.free.drain() {
        if ensure_context(dev).is_ok() {
            for p in ptrs {
                // SAFETY: `p` was allocated by `cuMemAlloc` on this device and is freed once.
                unsafe {
                    let _ = (a.cu_mem_free)(p);
                }
            }
        }
    }
    pool.pooled_bytes = 0;
}

/// Bytes held by live device buffers (size classes, all devices) and their high-water mark since the last reset.
static LIVE_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static PEAK_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `(live, peak, pooled)` bytes: held by tensors now, the most held at once since the last `reset_peak`, and freed but kept
/// in the reuse pool. `reset_peak` restarts the high-water mark from the current live bytes.
pub fn cuda_memory_stats(reset_peak: bool) -> (usize, usize, usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let live = LIVE_BYTES.load(Relaxed);
    let peak = if reset_peak { PEAK_BYTES.swap(live, Relaxed) } else { PEAK_BYTES.load(Relaxed) };
    let pooled = device_pool().lock().unwrap_or_else(|p| p.into_inner()).pooled_bytes;
    (live, peak.max(live), pooled)
}

/// RAII GPU VRAM allocation on a real CUDA device.
#[derive(Debug)]
pub struct CudaBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
    /// Size of the underlying allocation (`bytes` rounded up to its size class), the key for returning it to the pool.
    cap: usize,
    device_index: usize,
}

impl CudaBuffer {
    pub fn alloc(device_index: usize, bytes: usize) -> Result<Self> {
        ensure_context(device_index)?;
        let a = api()?;
        let mut ptr: CUdeviceptr = 0;
        let mut cap = if bytes > 0 { size_class(bytes) } else { 0 };
        if bytes > 0 {
            let pooled = {
                // the exact class first, then free blocks up to about twice the size (a sequence a little longer than last
                // batch's reuses its buffer instead of allocating a new one)
                let mut pool = device_pool().lock().unwrap_or_else(|p| p.into_inner());
                let mut c = cap;
                let mut got = None;
                for _ in 0..5 {
                    if let Some(p) = pool.free.get_mut(&(device_index, c)).and_then(Vec::pop) {
                        got = Some((p, c));
                        break;
                    }
                    c = size_class(c + 1);
                }
                if let Some((_, c)) = got {
                    pool.pooled_bytes -= c;
                }
                got
            };
            match pooled {
                Some((p, c)) => (ptr, cap) = (p, c),
                None => {
                    let t0 = kernel_profiling().then(std::time::Instant::now);
                    let mut rc = unsafe { (a.cu_mem_alloc)(&mut ptr, cap) };
                    if rc != CUDA_SUCCESS {
                        // out of memory: give the pooled allocations back and try once more
                        cuda_empty_pool();
                        ensure_context(device_index)?;
                        rc = unsafe { (a.cu_mem_alloc)(&mut ptr, cap) };
                    }
                    check(rc, "cuMemAlloc_v2")?;
                    if let Some(t0) = t0 {
                        record_kernel_time("(cuMemAlloc)", t0.elapsed().as_secs_f64());
                    }
                }
            }
        }
        if cap > 0 {
            use std::sync::atomic::Ordering::Relaxed;
            let live = LIVE_BYTES.fetch_add(cap, Relaxed) + cap;
            PEAK_BYTES.fetch_max(live, Relaxed);
        }
        Ok(Self {
            ptr,
            bytes,
            cap,
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
        if self.ptr == 0 {
            return;
        }
        LIVE_BYTES.fetch_sub(self.cap, std::sync::atomic::Ordering::Relaxed);
        {
            let mut pool = device_pool().lock().unwrap_or_else(|p| p.into_inner());
            if pool.pooled_bytes + self.cap <= pool_limit() {
                pool.pooled_bytes += self.cap;
                pool.free.entry((self.device_index, self.cap)).or_default().push(self.ptr);
                return;
            }
        }
        if let Ok(a) = api()
            && ensure_context(self.device_index).is_ok()
        {
            let t0 = kernel_profiling().then(std::time::Instant::now);
            unsafe {
                let _ = (a.cu_mem_free)(self.ptr);
            }
            if let Some(t0) = t0 {
                record_kernel_time("(cuMemFree)", t0.elapsed().as_secs_f64());
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
        let mut err_buf = vec![0u8; 4096];
        let mut opts: [c_int; 2] = [5 /* CU_JIT_ERROR_LOG_BUFFER */, 6 /* CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES */];
        let mut vals: [*mut c_void; 2] = [
            err_buf.as_mut_ptr() as *mut c_void,
            err_buf.len() as *mut c_void,
        ];
        let rc = unsafe {
            (a.cu_module_load_data_ex)(
                &mut module,
                c_ptx.as_ptr() as *const c_void,
                2,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            )
        };
        if rc != CUDA_SUCCESS {
            let msg = CStr::from_bytes_until_nul(&err_buf)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            return Err(Error::Backend(format!(
                "CUDA cuModuleLoadDataEx failed with error code {rc}: {msg}"
            )));
        }
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
        let block = 256u32;
        let grid = (numel as u32).div_ceil(block);
        self.launch_grid(func_name, [grid, 1, 1], [block, 1, 1], 0, kernel_params)
    }

    pub fn launch_grid(
        &self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        self.launch_grid_on(func_name, grid, block, shared_mem_bytes, None, kernel_params)
    }

    pub fn launch_grid_on(
        &self,
        func_name: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem_bytes: u32,
        stream: Option<&CudaStream>,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        ensure_context(self.device_index)?;
        let a = api()?;
        let c_name = CString::new(func_name).map_err(|e| Error::Backend(e.to_string()))?;
        let mut func: CUfunction = std::ptr::null_mut();
        check(
            unsafe { (a.cu_module_get_function)(&mut func, self.module, c_name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        let raw_stream = stream.map_or(std::ptr::null_mut(), |s| s.stream);
        let timed = stream.is_none() && kernel_profiling();
        if timed {
            check(unsafe { (a.cu_stream_synchronize)(std::ptr::null_mut()) }, "cuStreamSynchronize(profile)")?;
        }
        let t0 = std::time::Instant::now();
        let rc = check(
            unsafe {
                (a.cu_launch_kernel)(
                    func,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_mem_bytes,
                    raw_stream,
                    kernel_params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        );
        if timed && rc.is_ok() {
            check(unsafe { (a.cu_stream_synchronize)(std::ptr::null_mut()) }, "cuStreamSynchronize(profile)")?;
            record_kernel_time(func_name, t0.elapsed().as_secs_f64());
        }
        rc
    }
}

/// `OXIDEON_CUDA_PROFILE=1`: every kernel on the default stream is timed (synchronising before and after the launch, so the
/// run gets slower but each kernel's GPU time is exact) and summed by kernel name. Read with [`cuda_kernel_profile_take`].
fn kernel_profiling() -> bool {
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| std::env::var("OXIDEON_CUDA_PROFILE").is_ok_and(|v| v == "1"))
}

fn record_kernel_time(name: &str, secs: f64) {
    let mut p = kernel_profile().lock().unwrap_or_else(|e| e.into_inner());
    let e = p.entry(name.to_string()).or_insert((0, 0.0));
    e.0 += 1;
    e.1 += secs;
}

fn kernel_profile() -> &'static Mutex<std::collections::HashMap<String, (u64, f64)>> {
    static P: OnceLock<Mutex<std::collections::HashMap<String, (u64, f64)>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// The kernel timings gathered since the last call, `(kernel, launches, seconds)` slowest first, and resets them.
pub fn cuda_kernel_profile_take() -> Vec<(String, u64, f64)> {
    let mut p = kernel_profile().lock().unwrap_or_else(|e| e.into_inner());
    let mut v: Vec<_> = p.drain().map(|(k, (n, s))| (k, n, s)).collect();
    v.sort_by(|a, b| b.2.total_cmp(&a.2));
    v
}

/// Non-blocking CUDA stream for asynchronous kernel launches and CUDA Graph capture.
pub struct CudaStream {
    stream: CUstream,
    device_index: usize,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new(device_index: usize) -> Result<Self> {
        ensure_context(device_index)?;
        let a = api()?;
        let mut stream: CUstream = std::ptr::null_mut();
        check(unsafe { (a.cu_stream_create)(&mut stream, 1 /* CU_STREAM_NON_BLOCKING: no implicit sync with the default stream */) }, "cuStreamCreate")?;
        Ok(Self { stream, device_index })
    }

    pub fn synchronize(&self) -> Result<()> {
        ensure_context(self.device_index)?;
        let a = api()?;
        check(unsafe { (a.cu_stream_synchronize)(self.stream) }, "cuStreamSynchronize")
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null()
            && let Ok(a) = api()
            && ensure_context(self.device_index).is_ok()
        {
            unsafe {
                let _ = (a.cu_stream_destroy)(self.stream);
            }
        }
    }
}

/// Captured and instantiated CUDA Graph for zero-CPU-launch-overhead replay.
pub struct CudaGraph {
    exec: CUgraphExec,
    device_index: usize,
}

unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Captures all CUDA kernels launched on `stream` during `f` and instantiates an executable graph.
    pub fn capture(stream: &CudaStream, f: impl FnOnce(&CudaStream) -> Result<()>) -> Result<Self> {
        ensure_context(stream.device_index)?;
        let a = api()?;
        check(
            unsafe { (a.cu_stream_begin_capture)(stream.stream, 2 /* CU_STREAM_CAPTURE_MODE_RELAXED: other threads keep working */) },
            "cuStreamBeginCapture",
        )?;
        let res = f(stream);
        let mut graph: CUgraph = std::ptr::null_mut();
        let end_rc = unsafe { (a.cu_stream_end_capture)(stream.stream, &mut graph) };
        res?;
        check(end_rc, "cuStreamEndCapture")?;
        let mut exec: CUgraphExec = std::ptr::null_mut();
        let inst_rc = unsafe { (a.cu_graph_instantiate)(&mut exec, graph, 0) };
        unsafe {
            let _ = (a.cu_graph_destroy)(graph);
        }
        check(inst_rc, "cuGraphInstantiateWithFlags")?;
        Ok(Self {
            exec,
            device_index: stream.device_index,
        })
    }

    /// Replays the entire captured kernel graph in a single driver call.
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        ensure_context(self.device_index)?;
        let a = api()?;
        check(unsafe { (a.cu_graph_launch)(self.exec, stream.stream) }, "cuGraphLaunch")
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.exec.is_null()
            && let Ok(a) = api()
            && ensure_context(self.device_index).is_ok()
        {
            unsafe {
                let _ = (a.cu_graph_exec_destroy)(self.exec);
            }
        }
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

// ---------------------------------------------------------------------------------------------------------------------
// Additions for the NVRTC-compiled kernel library (see `cuda_rt`): raw byte transfers, device-to-device copies, cubin
// loading and cached function handles. The extra driver symbols are resolved lazily, next to the ones above.
// ---------------------------------------------------------------------------------------------------------------------

struct Extra {
    memcpy_dtod: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    memset_d8: unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult,
    func_set_attribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    mem_alloc_host: unsafe extern "C" fn(*mut *mut c_void, usize) -> CUresult,
    mem_free_host: unsafe extern "C" fn(*mut c_void) -> CUresult,
}

fn extra() -> Result<&'static Extra> {
    static E: OnceLock<std::result::Result<Extra, String>> = OnceLock::new();
    E.get_or_init(|| {
        let a = api().map_err(|e| e.to_string())?;
        // SAFETY: `a._lib` is the live driver handle and each symbol is requested with its documented signature.
        unsafe {
            Ok(Extra {
                memcpy_dtod: load_fn(a._lib, b"cuMemcpyDtoD_v2\0")?,
                memset_d8: load_fn(a._lib, b"cuMemsetD8_v2\0")?,
                func_set_attribute: load_fn(a._lib, b"cuFuncSetAttribute\0")?,
                mem_alloc_host: load_fn(a._lib, b"cuMemAllocHost_v2\0")?,
                mem_free_host: load_fn(a._lib, b"cuMemFreeHost\0")?,
            })
        }
    })
    .as_ref()
    .map_err(|e| Error::Backend(e.clone()))
}

impl CudaBuffer {
    /// Uploads raw bytes (any element type).
    pub fn from_host_bytes(device_index: usize, data: &[u8]) -> Result<Self> {
        let buf = Self::alloc(device_index, data.len())?;
        if !data.is_empty() {
            let a = api()?;
            check(unsafe { (a.cu_memcpy_htod)(buf.ptr, data.as_ptr() as *const c_void, data.len()) }, "cuMemcpyHtoD_v2")?;
        }
        Ok(buf)
    }

    /// Downloads `len` raw bytes from the start of the buffer.
    pub fn to_host_bytes(&self, len: usize) -> Result<Vec<u8>> {
        if len > self.bytes {
            return Err(Error::Backend("CUDA D2H read exceeds buffer size".into()));
        }
        ensure_context(self.device_index)?;
        let mut out = vec![0u8; len];
        if len > 0 {
            let a = api()?;
            check(unsafe { (a.cu_memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, len) }, "cuMemcpyDtoH_v2")?;
        }
        Ok(out)
    }

    /// Copies `bytes` bytes from `src` (same device) into this buffer at `dst_offset`.
    pub fn copy_from_device(&self, dst_offset: usize, src: &CudaBuffer, src_offset: usize, bytes: usize) -> Result<()> {
        if dst_offset + bytes > self.bytes || src_offset + bytes > src.bytes {
            return Err(Error::Backend("CUDA D2D copy out of range".into()));
        }
        if bytes == 0 {
            return Ok(());
        }
        ensure_context(self.device_index)?;
        let e = extra()?;
        check(unsafe { (e.memcpy_dtod)(self.ptr + dst_offset as u64, src.ptr + src_offset as u64, bytes) }, "cuMemcpyDtoD_v2")
    }

    /// Sets every byte of the buffer to `value`.
    pub fn memset(&self, value: u8) -> Result<()> {
        if self.bytes == 0 {
            return Ok(());
        }
        ensure_context(self.device_index)?;
        let e = extra()?;
        check(unsafe { (e.memset_d8)(self.ptr, value, self.bytes) }, "cuMemsetD8_v2")
    }

    pub fn device_index(&self) -> usize {
        self.device_index
    }
}

/// A kernel entry point of a loaded module, resolved once.
#[derive(Clone)]
pub struct CudaFunction {
    raw: usize,
    device_index: usize,
    name: Arc<str>,
    // keeps the module (and with it the function) alive
    _module: Arc<CudaPtxModule>,
}

impl CudaFunction {
    /// Launches on the default stream. `args` holds one pointer per kernel parameter, each pointing at that parameter's value.
    pub fn launch(&self, grid: [u32; 3], block: [u32; 3], shared_mem_bytes: u32, args: &mut [*mut c_void]) -> Result<()> {
        ensure_context(self.device_index)?;
        let a = api()?;
        let timed = kernel_profiling();
        if timed {
            check(unsafe { (a.cu_stream_synchronize)(std::ptr::null_mut()) }, "cuStreamSynchronize(profile)")?;
        }
        let t0 = std::time::Instant::now();
        check(
            unsafe {
                (a.cu_launch_kernel)(
                    self.raw as CUfunction,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_mem_bytes,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        if sync_each_launch() {
            check(unsafe { (a.cu_ctx_synchronize)() }, "cuCtxSynchronize (OXIDEON_CUDA_SYNC)")?;
        }
        if timed {
            check(unsafe { (a.cu_stream_synchronize)(std::ptr::null_mut()) }, "cuStreamSynchronize(profile)")?;
            record_kernel_time(&self.name, t0.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Allows more than 48 KiB of dynamic shared memory for this kernel (`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`).
    pub fn set_max_dynamic_shared(&self, bytes: i32) -> Result<()> {
        ensure_context(self.device_index)?;
        let e = extra()?;
        check(unsafe { (e.func_set_attribute)(self.raw as CUfunction, 8, bytes) }, "cuFuncSetAttribute")
    }
}

/// With `OXIDEON_CUDA_SYNC=1` every launch is followed by a synchronize, so a faulting kernel is reported at its own launch.
fn sync_each_launch() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("OXIDEON_CUDA_SYNC").is_ok_and(|v| v != "0"))
}

impl CudaPtxModule {
    /// Loads a compiled image (a CUBIN from NVRTC, or PTX text with a trailing NUL) onto the device.
    pub fn load_image(device_index: usize, image: &[u8]) -> Result<Arc<Self>> {
        ensure_context(device_index)?;
        let a = api()?;
        let mut module: CUmodule = std::ptr::null_mut();
        let mut err_buf = vec![0u8; 4096];
        let mut opts: [c_int; 2] = [5, 6];
        let mut vals: [*mut c_void; 2] = [err_buf.as_mut_ptr() as *mut c_void, err_buf.len() as *mut c_void];
        let rc = unsafe { (a.cu_module_load_data_ex)(&mut module, image.as_ptr() as *const c_void, 2, opts.as_mut_ptr(), vals.as_mut_ptr()) };
        if rc != CUDA_SUCCESS {
            let msg = CStr::from_bytes_until_nul(&err_buf).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            return Err(Error::Backend(format!("cuModuleLoadDataEx failed with error code {rc}: {msg}")));
        }
        Ok(Arc::new(Self { module, device_index }))
    }

    /// Resolves the kernel `name` of this module.
    pub fn function(self: &Arc<Self>, name: &str) -> Result<CudaFunction> {
        ensure_context(self.device_index)?;
        let a = api()?;
        let c_name = CString::new(name).map_err(|e| Error::Backend(e.to_string()))?;
        let mut func: CUfunction = std::ptr::null_mut();
        check(unsafe { (a.cu_module_get_function)(&mut func, self.module, c_name.as_ptr()) }, &format!("cuModuleGetFunction({name})"))?;
        Ok(CudaFunction { raw: func as usize, device_index: self.device_index, name: Arc::from(name), _module: self.clone() })
    }
}

/// Page-locked host memory, for faster and asynchronous-capable transfers.
pub struct CudaHostBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: the allocation is exclusively owned, like a `Vec<u8>`.
unsafe impl Send for CudaHostBuffer {}
unsafe impl Sync for CudaHostBuffer {}

impl CudaHostBuffer {
    pub fn new(device_index: usize, bytes: usize) -> Result<Self> {
        ensure_context(device_index)?;
        let e = extra()?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        check(unsafe { (e.mem_alloc_host)(&mut ptr, bytes.max(1)) }, "cuMemAllocHost_v2")?;
        Ok(Self { ptr, bytes })
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `ptr` is valid for `bytes` bytes for the life of `self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.bytes) }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: as above.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.bytes) }
    }
}

impl Drop for CudaHostBuffer {
    fn drop(&mut self) {
        if let Ok(e) = extra() {
            // SAFETY: `ptr` came from `cuMemAllocHost` and is freed once.
            unsafe {
                let _ = (e.mem_free_host)(self.ptr);
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
