//! The `.cu` kernel library of the CUDA backend: the CUDA C++ sources (`crates/oxideon-core/cuda/`), compiled on first use with
//! NVRTC ([`nvrtc`](super::nvrtc)) into a CUBIN for the device's exact architecture, cached in memory per
//! `(device, file, element type)` and on disk (`<temp>/oxideon_cuda_cache`), then launched through the driver API.
//!
//! This module also holds the pieces every kernel launch needs: the argument packer, the `Strided` layout descriptor (shape
//! and element strides of up to three operands, coalesced so most tensors collapse to one dimension), and the dtype codes
//! shared with the kernels (`DT_*` in `common.cuh`).

use super::cuda_driver::{CudaFunction, CudaPtxModule, cuda_available, cuda_device_info};
use super::nvrtc;
use crate::dtype::DType;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

const COMMON: &str = include_str!("../../cuda/common.cuh");

/// Source of one library file.
fn source(file: &str) -> Option<&'static str> {
    Some(match file {
        "elementwise.cu" => include_str!("../../cuda/elementwise.cu"),
        "copy.cu" => include_str!("../../cuda/copy.cu"),
        "reduce.cu" => include_str!("../../cuda/reduce.cu"),
        "nn.cu" => include_str!("../../cuda/nn.cu"),
        "gemm.cu" => include_str!("../../cuda/gemm.cu"),
        "gemm_tc.cu" => include_str!("../../cuda/gemm_tc.cu"),
        "conv.cu" => include_str!("../../cuda/conv.cu"),
        "optim.cu" => include_str!("../../cuda/optim.cu"),
        "attention.cu" => include_str!("../../cuda/attention.cu"),
        "rng.cu" => include_str!("../../cuda/rng.cu"),
        _ => return None,
    })
}

static LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many `.cu` kernels this process has launched (for tests and profiling: proves work ran on the device).
pub fn launch_count() -> u64 {
    LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether `.cu` kernels can be compiled and run here (a CUDA device and NVRTC are both present).
pub fn available() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| std::env::var("OXIDEON_CUDA_NO_NVRTC").is_err() && cuda_available() && nvrtc::available())
}

/// The `DT_*` code of `dtype` in `common.cuh`, or `None` for types without a GPU kernel (8-bit floats, complex).
pub fn dtype_code(dtype: DType) -> Option<i32> {
    Some(match dtype {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::BF16 => 3,
        DType::I8 => 4,
        DType::I16 => 5,
        DType::I32 => 6,
        DType::I64 => 7,
        DType::U8 => 8,
        DType::U16 => 9,
        DType::U32 => 10,
        DType::U64 => 11,
        DType::Bool => 12,
        DType::F8E4M3 | DType::F8E5M2 | DType::C64 | DType::C128 => return None,
    })
}

/// The unsigned dtype code with the same element size (the type-agnostic `copy` kernels are built per size).
pub fn size_class_code(dtype: DType) -> i32 {
    match dtype.size_in_bytes() {
        1 => 8,
        2 => 9,
        4 => 10,
        _ => 11,
    }
}

fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn cache_dir() -> std::path::PathBuf {
    std::env::temp_dir().join("oxideon_cuda_cache")
}

type ModuleKey = (usize, &'static str, i32);
type FuncKey = (usize, &'static str, i32, String);

fn modules() -> &'static Mutex<HashMap<ModuleKey, Arc<CudaPtxModule>>> {
    static M: OnceLock<Mutex<HashMap<ModuleKey, Arc<CudaPtxModule>>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

fn functions() -> &'static Mutex<HashMap<FuncKey, CudaFunction>> {
    static F: OnceLock<Mutex<HashMap<FuncKey, CudaFunction>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The compiled and loaded module for `(device, file, dtype code)`.
pub fn module(device: usize, file: &'static str, dtype: i32) -> Result<Arc<CudaPtxModule>> {
    let key = (device, file, dtype);
    if let Some(m) = modules().lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(m.clone());
    }
    let src = source(file).ok_or_else(|| Error::Backend(format!("unknown CUDA source file {file}")))?;
    let info = cuda_device_info(device)?;
    let arch = (info.major, info.minor);
    let (nv_major, nv_minor) = nvrtc::version()?;
    let defines = format!("-DDTYPE={dtype}");
    let mut h = fnv(format!("{nv_major}.{nv_minor}|sm_{}{}|{file}|{defines}|", arch.0, arch.1).as_bytes(), 0xcbf2_9ce4_8422_2325);
    h = fnv(COMMON.as_bytes(), h);
    h = fnv(src.as_bytes(), h);
    let path = cache_dir().join(format!("{h:016x}.cubin"));
    let image: Vec<u8> = match std::fs::read(&path) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => {
            let img = nvrtc::compile(file, src, &[("common.cuh", COMMON)], arch, &[&defines])?;
            let mut bytes = img.bytes;
            if !img.is_cubin {
                bytes.push(0); // PTX text must be NUL-terminated for cuModuleLoadDataEx
            } else {
                let _ = std::fs::create_dir_all(cache_dir());
                let tmp = path.with_extension(format!("tmp{}", std::process::id()));
                if std::fs::write(&tmp, &bytes).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            }
            bytes
        }
    };
    let m = CudaPtxModule::load_image(device, &image)?;
    modules().lock().unwrap_or_else(|p| p.into_inner()).insert(key, m.clone());
    Ok(m)
}

/// The kernel `name` of `(file, dtype)` on `device`.
pub fn func(device: usize, file: &'static str, dtype: i32, name: &str) -> Result<CudaFunction> {
    let key = (device, file, dtype, name.to_string());
    if let Some(f) = functions().lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(f.clone());
    }
    let f = module(device, file, dtype)?.function(name)?;
    functions().lock().unwrap_or_else(|p| p.into_inner()).insert(key, f.clone());
    Ok(f)
}

// ---------------------------------------------------------------------------------------------------------------------
// Kernel arguments
// ---------------------------------------------------------------------------------------------------------------------

/// Packs kernel arguments by value. Each argument is stored in its own heap block, so the pointers handed to the driver stay
/// valid while more arguments are added.
#[derive(Default)]
pub struct Args {
    slots: Vec<Vec<u8>>,
}

impl Args {
    pub fn new() -> Self {
        Self::default()
    }
    fn push(mut self, b: &[u8]) -> Self {
        self.slots.push(b.to_vec());
        self
    }
    pub fn ptr(self, p: u64) -> Self {
        self.push(&p.to_ne_bytes())
    }
    pub fn i32(self, v: i32) -> Self {
        self.push(&v.to_ne_bytes())
    }
    pub fn u32(self, v: u32) -> Self {
        self.push(&v.to_ne_bytes())
    }
    pub fn i64(self, v: i64) -> Self {
        self.push(&v.to_ne_bytes())
    }
    pub fn u64(self, v: u64) -> Self {
        self.push(&v.to_ne_bytes())
    }
    pub fn f32(self, v: f32) -> Self {
        self.push(&v.to_ne_bytes())
    }
    pub fn f64(self, v: f64) -> Self {
        self.push(&v.to_ne_bytes())
    }
    /// A plain-data struct passed by value (it must match the kernel's `struct` layout, so use `repr(C)`).
    pub fn raw<T: Copy>(self, v: &T) -> Self {
        // SAFETY: `T: Copy` plain data viewed as bytes; the caller provides a `repr(C)` type without padding surprises.
        let bytes = unsafe { std::slice::from_raw_parts((v as *const T).cast::<u8>(), std::mem::size_of::<T>()) };
        self.push(bytes)
    }
    pub fn strided(self, s: &StridedRaw) -> Self {
        // SAFETY: `StridedRaw` is `repr(C)` plain data with no padding bytes left uninitialised (explicit `_pad`).
        let bytes = unsafe { std::slice::from_raw_parts((s as *const StridedRaw).cast::<u8>(), std::mem::size_of::<StridedRaw>()) };
        self.push(bytes)
    }

    pub fn launch(&self, f: &CudaFunction, grid: [u32; 3], block: [u32; 3], shared: u32) -> Result<()> {
        let mut ptrs: Vec<*mut c_void> = self.slots.iter().map(|s| s.as_ptr() as *mut c_void).collect();
        LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        f.launch(grid, block, shared, &mut ptrs)
    }

    /// One-dimensional launch with a grid-stride loop: 256 threads per block, at most 2^20 blocks.
    pub fn launch_1d(&self, f: &CudaFunction, n: u64) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let blocks = n.div_ceil(256).min(1 << 20) as u32;
        self.launch(f, [blocks, 1, 1], [256, 1, 1], 0)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Strided layouts
// ---------------------------------------------------------------------------------------------------------------------

pub const MAXD: usize = 8;

/// Mirror of `struct Strided` in `common.cuh`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StridedRaw {
    pub nd: i32,
    pub _pad: i32,
    pub shape: [i64; MAXD],
    pub stride: [[i64; MAXD]; 3],
}

/// Builds the descriptor for `shape` with up to three operands' element strides. Size-1 dimensions are dropped and
/// dimensions that are contiguous for every operand are merged; when what remains is one dimension with unit stride for
/// every operand (or nothing), `nd == 0` tells the kernels to use the linear index directly. `None` when more than
/// [`MAXD`] dimensions remain. Returns the element count along with it.
pub fn strided(shape: &[usize], ops: &[&[isize]]) -> Option<(StridedRaw, u64)> {
    let n: u64 = shape.iter().map(|&d| d as u64).product();
    let k = ops.len().min(3);
    let mut merged: Vec<(i64, [i64; 3])> = Vec::new();
    for (d, &sz) in shape.iter().enumerate() {
        if sz == 1 {
            continue;
        }
        let mut st = [0i64; 3];
        for (j, op) in ops.iter().take(3).enumerate() {
            st[j] = op[d] as i64;
        }
        if let Some(last) = merged.last_mut()
            && (0..k).all(|j| last.1[j] == sz as i64 * st[j])
        {
            last.0 *= sz as i64;
            last.1 = st;
            continue;
        }
        merged.push((sz as i64, st));
    }
    let linear = merged.is_empty() || (merged.len() == 1 && (0..k).all(|j| merged[0].1[j] == 1));
    if merged.len() > MAXD {
        return None;
    }
    let mut raw = StridedRaw { nd: if linear { 0 } else { merged.len() as i32 }, _pad: 0, shape: [1; MAXD], stride: [[0; MAXD]; 3] };
    if !linear {
        for (d, (sz, st)) in merged.iter().enumerate() {
            raw.shape[d] = *sz;
            for j in 0..3 {
                raw.stride[j][d] = st[j];
            }
        }
    }
    Some((raw, n))
}

/// Descriptor that keeps every dimension as given (no merging, no linear shortcut), for kernels that treat one axis
/// specially (gather / scatter). `None` beyond [`MAXD`] dimensions or for a 0-d shape.
pub fn strided_exact(shape: &[usize], ops: &[&[isize]]) -> Option<(StridedRaw, u64)> {
    if shape.is_empty() || shape.len() > MAXD {
        return None;
    }
    let n: u64 = shape.iter().map(|&d| d as u64).product();
    let mut raw = StridedRaw { nd: shape.len() as i32, _pad: 0, shape: [1; MAXD], stride: [[0; MAXD]; 3] };
    for (d, &sz) in shape.iter().enumerate() {
        raw.shape[d] = sz as i64;
        for (j, op) in ops.iter().take(3).enumerate() {
            raw.stride[j][d] = op[d] as i64;
        }
    }
    Some((raw, n))
}

/// [`strided`] for batch dimensions whose strides are in units of whole matrices: the "linear" shortcut (index used as
/// the element offset) is never taken, because a matrix stride is not 1. A batch of one (no dimensions left after dropping
/// size-1 ones) is the only case reported as `nd == 0`, where the offset is 0 anyway.
pub fn strided_batch(shape: &[usize], ops: &[&[isize]]) -> Option<(StridedRaw, u64)> {
    let n: u64 = shape.iter().map(|&d| d as u64).product();
    let mut merged: Vec<(i64, [i64; 3])> = Vec::new();
    let k = ops.len().min(3);
    for (d, &sz) in shape.iter().enumerate() {
        if sz == 1 {
            continue;
        }
        let mut st = [0i64; 3];
        for (j, op) in ops.iter().take(3).enumerate() {
            st[j] = op[d] as i64;
        }
        if let Some(last) = merged.last_mut()
            && (0..k).all(|j| last.1[j] == sz as i64 * st[j])
        {
            last.0 *= sz as i64;
            last.1 = st;
            continue;
        }
        merged.push((sz as i64, st));
    }
    if merged.len() > MAXD {
        return None;
    }
    let mut raw = StridedRaw { nd: merged.len() as i32, _pad: 0, shape: [1; MAXD], stride: [[0; MAXD]; 3] };
    for (d, (sz, st)) in merged.iter().enumerate() {
        raw.shape[d] = *sz;
        for j in 0..3 {
            raw.stride[j][d] = st[j];
        }
    }
    Some((raw, n))
}

/// Strides of a contiguous (row-major) tensor of `shape`.
pub fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut s = vec![0isize; shape.len()];
    let mut acc = 1isize;
    for d in (0..shape.len()).rev() {
        s[d] = acc;
        acc *= shape[d].max(1) as isize;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_descriptors_coalesce_contiguous_and_broadcast_dimensions() {
        // a contiguous 3-D tensor is one linear run
        let (s, n) = strided(&[4, 5, 6], &[&[30, 6, 1], &[30, 6, 1]]).unwrap();
        assert_eq!((s.nd, n), (0, 120));
        // a broadcast row [1, 6] against [5, 6]: two dimensions, stride 0 on the broadcast operand
        let (s, _) = strided(&[5, 6], &[&[6, 1], &[0, 1]]).unwrap();
        assert_eq!((s.nd, s.shape[0], s.shape[1], s.stride[1][0], s.stride[1][1]), (2, 5, 6, 0, 1));
        // a transposed view cannot merge
        let (s, _) = strided(&[6, 5], &[&[1, 6]]).unwrap();
        assert_eq!((s.nd, s.stride[0][0], s.stride[0][1]), (2, 1, 6));
        // size-1 dimensions vanish; a scalar is linear
        assert_eq!(strided(&[1, 7, 1], &[&[7, 1, 1]]).unwrap().0.nd, 0);
        assert_eq!(strided(&[], &[&[]]).unwrap().1, 1);
        // too many dimensions that cannot merge
        assert!(strided(&[2; 10], &[&[1, 3, 5, 7, 11, 13, 17, 19, 23, 29]]).is_none());
    }

    #[test]
    fn library_sources_compile_for_this_device() {
        if !available() {
            eprintln!("skipped: no CUDA device or NVRTC");
            return;
        }
        let info = cuda_device_info(0).unwrap();
        for (file, dt) in [("elementwise.cu", 0), ("elementwise.cu", 6), ("copy.cu", 10)] {
            let img = nvrtc::compile(file, source(file).unwrap(), &[("common.cuh", COMMON)], (info.major, info.minor), &[&format!("-DDTYPE={dt}")]);
            assert!(img.is_ok(), "{file} (dtype {dt}): {:?}", img.err());
        }
    }
}
