//! Tensor operations on [`CudaStorage`] implemented with the NVRTC-compiled `.cu` kernels of [`cuda_rt`](super::cuda_rt).
//!
//! Every function here returns `Ok(None)` when the combination is not covered (an 8-bit float, an integer division that must
//! report errors, a layout with too many dimensions, NVRTC missing, ...): the caller then takes the CPU path, which defines
//! the reference semantics. Layouts are passed to the kernels as strides, so views (transposes, slices, broadcasts, flips)
//! need no intermediate copy.

use super::cpu::CpuStorage;
use super::cuda_driver::{self, CudaBuffer};
use super::cuda_kernels::CudaStorage;
use super::cuda_rt::{self, Args, contiguous_strides, dtype_code, size_class_code, strided, strided_batch, strided_exact};
use super::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::dtype::{DType, bf16, f16};
use crate::error::{Error, Result};
use crate::layout::Layout;
use std::sync::Arc;

// ---------------------------------------------------------------------------------------------------------------------
// Host <-> device conversion for every dtype
// ---------------------------------------------------------------------------------------------------------------------

static DOWNLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many whole-tensor device -> host copies have happened (an operation without a kernel falls back to the CPU through
/// one). Useful to find ops that still bounce through the host: compare the value before and after.
pub fn download_count() -> u64 {
    DOWNLOADS.load(std::sync::atomic::Ordering::Relaxed)
}

macro_rules! raw_bytes {
    ($v:expr, $t:ty) => {
        // SAFETY: the element types are plain-old-data (integers, floats, `bool` as one byte, `repr(transparent)` half
        // floats), so viewing the slice as bytes is sound.
        unsafe { std::slice::from_raw_parts($v.as_ptr().cast::<u8>(), $v.len() * std::mem::size_of::<$t>()) }
    };
}

/// The elements of `cpu` as bytes, or `None` for dtypes without a device representation (complex, 8-bit floats).
pub(crate) fn cpu_bytes(cpu: &CpuStorage) -> Option<&[u8]> {
    Some(match cpu {
        CpuStorage::F32(v) => raw_bytes!(v, f32),
        CpuStorage::F64(v) => raw_bytes!(v, f64),
        CpuStorage::F16(v) => raw_bytes!(v, f16),
        CpuStorage::BF16(v) => raw_bytes!(v, bf16),
        CpuStorage::I8(v) => raw_bytes!(v, i8),
        CpuStorage::I16(v) => raw_bytes!(v, i16),
        CpuStorage::I32(v) => raw_bytes!(v, i32),
        CpuStorage::I64(v) => raw_bytes!(v, i64),
        CpuStorage::U8(v) => raw_bytes!(v, u8),
        CpuStorage::U16(v) => raw_bytes!(v, u16),
        CpuStorage::U32(v) => raw_bytes!(v, u32),
        CpuStorage::U64(v) => raw_bytes!(v, u64),
        CpuStorage::Bool(v) => raw_bytes!(v, bool),
        _ => return None,
    })
}

/// Rebuilds host storage of `dtype` from device bytes.
pub(crate) fn bytes_to_cpu(dtype: DType, bytes: &[u8]) -> Result<CpuStorage> {
    macro_rules! typed {
        ($variant:ident, $t:ty) => {{
            let n = bytes.len() / std::mem::size_of::<$t>();
            let mut v: Vec<$t> = vec![<$t>::default(); n];
            // SAFETY: `v` owns `n` elements = `bytes.len()` bytes, and any bit pattern is a valid value of these types.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr().cast::<u8>(), n * std::mem::size_of::<$t>()) };
            CpuStorage::$variant(v)
        }};
    }
    Ok(match dtype {
        DType::F32 => typed!(F32, f32),
        DType::F64 => typed!(F64, f64),
        DType::F16 => typed!(F16, f16),
        DType::BF16 => typed!(BF16, bf16),
        DType::I8 => typed!(I8, i8),
        DType::I16 => typed!(I16, i16),
        DType::I32 => typed!(I32, i32),
        DType::I64 => typed!(I64, i64),
        DType::U8 => typed!(U8, u8),
        DType::U16 => typed!(U16, u16),
        DType::U32 => typed!(U32, u32),
        DType::U64 => typed!(U64, u64),
        DType::Bool => CpuStorage::Bool(bytes.iter().map(|&b| b != 0).collect()),
        other => return Err(Error::Msg(format!("CUDA storage does not support {other:?}"))),
    })
}

pub type LayerNormOut = (CudaStorage, Option<CudaStorage>, Option<CudaStorage>);

impl CudaStorage {
    /// Uploads host storage of any supported dtype.
    pub fn from_cpu_any(cpu: &CpuStorage, device_index: usize) -> Result<Self> {
        let Some(bytes) = cpu_bytes(cpu) else {
            return Err(Error::Msg(format!("CudaStorage does not support {:?} tensors", cpu.dtype())));
        };
        Ok(Self { device_index, dtype: cpu.dtype(), len: cpu.len(), buf: Arc::new(CudaBuffer::from_host_bytes(device_index, bytes)?) })
    }

    /// Downloads the whole buffer as host storage of this tensor's dtype.
    pub fn to_cpu_any(&self) -> Result<CpuStorage> {
        DOWNLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        super::cuda_driver::cuda_synchronize(self.device_index)?;
        let bytes = self.buf.to_host_bytes(self.len * self.dtype.size_in_bytes())?;
        bytes_to_cpu(self.dtype, &bytes)
    }

    /// An uninitialised device tensor of `len` elements.
    pub fn alloc(device_index: usize, dtype: DType, len: usize) -> Result<Self> {
        Ok(Self { device_index, dtype, len, buf: Arc::new(CudaBuffer::alloc(device_index, len * dtype.size_in_bytes())?) })
    }

    /// Device address of the first element a layout selects.
    pub(crate) fn base(&self, layout: &Layout) -> u64 {
        self.buf.device_ptr() + (layout.offset() * self.dtype.size_in_bytes()) as u64
    }

    fn rt_ready(&self) -> bool {
        cuda_rt::available()
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------------------------------------------------

fn unary_code(op: UnaryOp) -> (i32, f64, f64) {
    match op {
        UnaryOp::Neg => (0, 0.0, 0.0),
        UnaryOp::Abs => (1, 0.0, 0.0),
        UnaryOp::Sign => (2, 0.0, 0.0),
        UnaryOp::Exp => (3, 0.0, 0.0),
        UnaryOp::Ln => (4, 0.0, 0.0),
        UnaryOp::Sqrt => (5, 0.0, 0.0),
        UnaryOp::Rsqrt => (6, 0.0, 0.0),
        UnaryOp::Sin => (7, 0.0, 0.0),
        UnaryOp::Cos => (8, 0.0, 0.0),
        UnaryOp::Tanh => (9, 0.0, 0.0),
        UnaryOp::Sigmoid => (10, 0.0, 0.0),
        UnaryOp::Relu => (11, 0.0, 0.0),
        UnaryOp::LeakyRelu(s) => (12, s, 0.0),
        UnaryOp::Gelu => (13, 0.0, 0.0),
        UnaryOp::Silu => (14, 0.0, 0.0),
        UnaryOp::Elu(a) => (15, a, 0.0),
        UnaryOp::Softplus => (16, 0.0, 0.0),
        UnaryOp::Square => (17, 0.0, 0.0),
        UnaryOp::Recip => (18, 0.0, 0.0),
        UnaryOp::Floor => (19, 0.0, 0.0),
        UnaryOp::Ceil => (20, 0.0, 0.0),
        UnaryOp::Round => (21, 0.0, 0.0),
        UnaryOp::Erf => (22, 0.0, 0.0),
        UnaryOp::Powf(p) => (23, p, 0.0),
        UnaryOp::AddScalar(s) => (24, s, 0.0),
        UnaryOp::MulScalar(s) => (25, s, 0.0),
        UnaryOp::Clamp(lo, hi) => (26, lo, hi),
        UnaryOp::LogicalNot => (27, 0.0, 0.0),
    }
}

impl CudaStorage {
    /// Unary operation on the device; `None` when the CPU path must run.
    pub fn unary_gpu(&self, layout: &Layout, op: UnaryOp) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() {
            return Ok(None);
        }
        let Some((sd, n)) = strided(layout.shape(), &[layout.strides()]) else { return Ok(None) };
        let (code, p0, p1) = unary_code(op);
        if op == UnaryOp::LogicalNot {
            let out = Self::alloc(self.device_index, DType::Bool, n as usize)?;
            let f = cuda_rt::func(self.device_index, "elementwise.cu", dt, "logical_not")?;
            Args::new().ptr(self.base(layout)).ptr(out.buf.device_ptr()).u64(n).strided(&sd).launch_1d(&f, n)?;
            return Ok(Some(out));
        }
        if self.dtype == DType::Bool {
            return Ok(None);
        }
        let out = Self::alloc(self.device_index, self.dtype, n as usize)?;
        let f = cuda_rt::func(self.device_index, "elementwise.cu", dt, "unary")?;
        Args::new().ptr(self.base(layout)).ptr(out.buf.device_ptr()).u64(n).i32(code).f64(p0).f64(p1).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// Binary operation on two same-dtype tensors of one (already broadcast) shape.
    pub fn binary_gpu(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout, op: BinaryOp) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || self.dtype != rhs.dtype || lhs_l.shape() != rhs_l.shape() || self.dtype == DType::Bool {
            return Ok(None);
        }
        let float = matches!(self.dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16);
        let code = match op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div if float => 3,
            BinaryOp::Maximum => 4,
            BinaryOp::Minimum => 5,
            BinaryOp::Pow if float => 6,
            BinaryOp::Fmod if float => 7,
            BinaryOp::Remainder if float => 8,
            BinaryOp::FloorDiv if float => 9,
            _ => return Ok(None), // integer division / remainder / pow report errors on the CPU path
        };
        let Some((sd, n)) = strided(lhs_l.shape(), &[lhs_l.strides(), rhs_l.strides()]) else { return Ok(None) };
        let out = Self::alloc(self.device_index, self.dtype, n as usize)?;
        let f = cuda_rt::func(self.device_index, "elementwise.cu", dt, "binary")?;
        Args::new().ptr(self.base(lhs_l)).ptr(rhs.base(rhs_l)).ptr(out.buf.device_ptr()).u64(n).i32(code).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// Comparison producing a Bool tensor.
    pub fn cmp_gpu(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout, op: CmpOp) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || self.dtype != rhs.dtype || lhs_l.shape() != rhs_l.shape() {
            return Ok(None);
        }
        let code = match op {
            CmpOp::Eq => 0,
            CmpOp::Ne => 1,
            CmpOp::Lt => 2,
            CmpOp::Le => 3,
            CmpOp::Gt => 4,
            CmpOp::Ge => 5,
        };
        let Some((sd, n)) = strided(lhs_l.shape(), &[lhs_l.strides(), rhs_l.strides()]) else { return Ok(None) };
        let out = Self::alloc(self.device_index, DType::Bool, n as usize)?;
        let f = cuda_rt::func(self.device_index, "elementwise.cu", dt, "cmp")?;
        Args::new().ptr(self.base(lhs_l)).ptr(rhs.base(rhs_l)).ptr(out.buf.device_ptr()).u64(n).i32(code).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// Element-wise logical and / or / xor (`!= 0` truthiness) of two same-dtype operands of one shape, as Bool.
    pub fn logic_gpu(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout, op: super::LogicOp) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || self.dtype != rhs.dtype || lhs_l.shape() != rhs_l.shape() {
            return Ok(None);
        }
        let code = match op {
            super::LogicOp::And => 0,
            super::LogicOp::Or => 1,
            super::LogicOp::Xor => 2,
        };
        let Some((sd, n)) = strided(lhs_l.shape(), &[lhs_l.strides(), rhs_l.strides()]) else { return Ok(None) };
        let out = Self::alloc(self.device_index, DType::Bool, n as usize)?;
        let f = cuda_rt::func(self.device_index, "elementwise.cu", dt, "logic")?;
        Args::new().ptr(self.base(lhs_l)).ptr(rhs.base(rhs_l)).ptr(out.buf.device_ptr()).u64(n).i32(code).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// `cond ? a : b` for a Bool condition and two same-dtype branches of one shape.
    pub fn where_gpu(cond: &Self, cond_l: &Layout, a: &Self, a_l: &Layout, b: &Self, b_l: &Layout) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(a.dtype) else { return Ok(None) };
        if !a.rt_ready() || cond.dtype != DType::Bool || a.dtype != b.dtype || cond_l.shape() != a_l.shape() || a_l.shape() != b_l.shape() {
            return Ok(None);
        }
        let Some((sd, n)) = strided(a_l.shape(), &[cond_l.strides(), a_l.strides(), b_l.strides()]) else { return Ok(None) };
        let out = Self::alloc(a.device_index, a.dtype, n as usize)?;
        let f = cuda_rt::func(a.device_index, "elementwise.cu", dt, "where_sel")?;
        Args::new().ptr(cond.base(cond_l)).ptr(a.base(a_l)).ptr(b.base(b_l)).ptr(out.buf.device_ptr()).u64(n).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// A tensor of `len` elements all equal to `value` (floats use `value`, integers its truncation).
    pub fn full_gpu(device_index: usize, dtype: DType, len: usize, value: f64) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(dtype) else { return Ok(None) };
        if !cuda_rt::available() {
            return Ok(None);
        }
        let out = Self::alloc(device_index, dtype, len)?;
        if value.to_bits() == 0 {
            out.buf.memset(0)?; // +0 is the all-zero bit pattern of every dtype
            return Ok(Some(out));
        }
        let f = cuda_rt::func(device_index, "elementwise.cu", dt, "fill")?;
        let iv = if value.is_finite() { value as i64 } else { i64::MIN };
        Args::new().ptr(out.buf.device_ptr()).u64(len as u64).f64(value).i64(iv).launch_1d(&f, len as u64)?;
        Ok(Some(out))
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Copies, casts, indexing
// ---------------------------------------------------------------------------------------------------------------------

impl CudaStorage {
    /// Contiguous copy of the elements a layout selects.
    pub fn contiguous_gpu(&self, layout: &Layout) -> Result<Option<Self>> {
        if !self.rt_ready() || dtype_code(self.dtype).is_none() {
            return Ok(None);
        }
        let n = layout.numel();
        if layout.is_contiguous() && layout.offset() == 0 && n == self.len {
            return Ok(Some(self.clone()));
        }
        let cs = contiguous_strides(layout.shape());
        let Some((sd, n)) = strided(layout.shape(), &[layout.strides(), &cs]) else { return Ok(None) };
        let out = Self::alloc(self.device_index, self.dtype, n as usize)?;
        let f = cuda_rt::func(self.device_index, "copy.cu", size_class_code(self.dtype), "copy")?;
        Args::new().ptr(self.base(layout)).ptr(out.buf.device_ptr()).u64(n).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// dtype conversion with the CPU backend's rounding / wrapping rules.
    pub fn cast_gpu(&self, layout: &Layout, dst: DType) -> Result<Option<Self>> {
        let (Some(sc), Some(dc)) = (dtype_code(self.dtype), dtype_code(dst)) else { return Ok(None) };
        if !self.rt_ready() {
            return Ok(None);
        }
        let Some((sd, n)) = strided(layout.shape(), &[layout.strides()]) else { return Ok(None) };
        let out = Self::alloc(self.device_index, dst, n as usize)?;
        let f = cuda_rt::func(self.device_index, "copy.cu", 8, "cast")?;
        Args::new().ptr(self.base(layout)).ptr(out.buf.device_ptr()).u64(n).i32(sc).i32(dc).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// `index_select` along `ax` with host-validated indices (negative indices wrap, like the CPU path).
    pub fn index_select_gpu(&self, layout: &Layout, ax: usize, idx: &[i64]) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() {
            return Ok(None);
        }
        let shape = layout.shape();
        let dim = shape[ax] as i64;
        let mut norm = Vec::with_capacity(idx.len());
        for &r in idx {
            let v = if r < 0 { r + dim } else { r };
            if v < 0 || v >= dim {
                return Err(Error::Msg(format!("index_select: index {r} is out of range for dimension of size {dim}")));
            }
            norm.push(v);
        }
        let src = match self.contiguous_gpu(layout)? {
            Some(s) => s,
            None => return Ok(None),
        };
        let outer: u64 = shape[..ax].iter().map(|&d| d as u64).product();
        let inner: u64 = shape[ax + 1..].iter().map(|&d| d as u64).product();
        let n = outer * norm.len() as u64 * inner;
        let out = Self::alloc(self.device_index, self.dtype, n as usize)?;
        if n == 0 {
            return Ok(Some(out));
        }
        let idx_dev = CudaBuffer::from_host_bytes(self.device_index, raw_bytes!(norm, i64))?;
        let f = cuda_rt::func(self.device_index, "copy.cu", dt, "index_select")?;
        Args::new().ptr(src.buf.device_ptr()).ptr(idx_dev.device_ptr()).ptr(out.buf.device_ptr()).u64(outer).u64(dim as u64).u64(inner).u64(norm.len() as u64).ptr(0).launch_1d(&f, n)?;
        Ok(Some(out))
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------------------------------------------------

impl CudaStorage {
    /// A device copy laid out contiguously for `layout`, and the address of its first element. Contiguous layouts are used
    /// in place.
    pub(crate) fn contiguous_ptr(&self, layout: &Layout) -> Result<Option<(Self, u64)>> {
        if layout.is_contiguous() {
            return Ok(Some((self.clone(), self.base(layout))));
        }
        Ok(self.contiguous_gpu(layout)?.map(|c| {
            let p = c.buf.device_ptr();
            (c, p)
        }))
    }

    /// Reduction over `axes` (sum, mean, max, min, prod, argmax, argmin). The result holds the kept dimensions in order.
    pub fn reduce_gpu(&self, layout: &Layout, axes: &[usize], op: ReduceOp) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || axes.is_empty() {
            return Ok(None);
        }
        let float = matches!(self.dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16);
        let arg = matches!(op, ReduceOp::ArgMax | ReduceOp::ArgMin);
        let supported = match self.dtype {
            DType::Bool => false,
            _ if float => true,
            DType::I64 => !matches!(op, ReduceOp::Mean),
            _ => matches!(op, ReduceOp::Max | ReduceOp::Min | ReduceOp::ArgMax | ReduceOp::ArgMin),
        };
        if !supported {
            return Ok(None);
        }
        let shape = layout.shape();
        let mut ax: Vec<usize> = axes.to_vec();
        ax.sort_unstable();
        ax.dedup();
        if ax.iter().any(|&a| a >= shape.len()) || (arg && ax.len() != 1) {
            return Ok(None);
        }
        let consecutive = ax.windows(2).all(|w| w[1] == w[0] + 1);
        let (lo, hi) = (ax[0], *ax.last().unwrap());
        // [outer, red, inner] view of a contiguous tensor
        let (src, ptr, outer, red, inner) = if consecutive && layout.is_contiguous() {
            let outer: u64 = shape[..lo].iter().map(|&d| d as u64).product();
            let red: u64 = shape[lo..=hi].iter().map(|&d| d as u64).product();
            let inner: u64 = shape[hi + 1..].iter().map(|&d| d as u64).product();
            let (src, ptr) = self.contiguous_ptr(layout)?.expect("contiguous layouts need no copy");
            (src, ptr, outer, red, inner)
        } else {
            // move the reduced axes last and copy: [kept..., reduced...] -> [outer, red]
            let kept: Vec<usize> = (0..shape.len()).filter(|d| !ax.contains(d)).collect();
            let perm: Vec<usize> = kept.iter().chain(&ax).copied().collect();
            let moved = layout.permute(&perm)?;
            let Some((src, ptr)) = self.contiguous_ptr(&moved)? else { return Ok(None) };
            let outer: u64 = kept.iter().map(|&d| shape[d] as u64).product();
            let red: u64 = ax.iter().map(|&d| shape[d] as u64).product();
            (src, ptr, outer, red, 1)
        };
        let out_len = (outer * inner) as usize;
        let out_dtype = if arg { DType::I64 } else { self.dtype };
        let out = Self::alloc(self.device_index, out_dtype, out_len)?;
        if out_len == 0 {
            return Ok(Some(out));
        }
        let csize: u32 = if matches!(self.dtype, DType::F64 | DType::I64 | DType::U64) || (!float && self.dtype.size_in_bytes() <= 8 && !matches!(self.dtype, DType::F32)) { 8 } else { 4 };
        let csize = if float && self.dtype != DType::F64 { 4 } else { csize };
        let dev = self.device_index;
        let op_code = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Mean => 1,
            ReduceOp::Max => 2,
            ReduceOp::Min => 3,
            ReduceOp::Prod => 4,
            ReduceOp::ArgMax | ReduceOp::ArgMin => 0,
        };
        let scale = 1.0 / red as f64;

        if arg {
            let is_max = i32::from(op == ReduceOp::ArgMax);
            if red == 0 {
                return Ok(None);
            }
            if inner == 1 {
                let bs = red.next_power_of_two().clamp(32, 1024) as u32;
                let f = cuda_rt::func(dev, "reduce.cu", dt, "argreduce_rows")?;
                Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(outer).u64(red).i32(is_max).launch(&f, [outer as u32, 1, 1], [bs, 1, 1], bs * 16)?;
            } else {
                if outer > 65535 {
                    return Ok(None);
                }
                let f = cuda_rt::func(dev, "reduce.cu", dt, "argreduce_cols")?;
                Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(outer).u64(red).u64(inner).i32(is_max).launch(&f, [inner.div_ceil(128) as u32, outer as u32, 1], [128, 1, 1], 0)?;
            }
            drop(src);
            return Ok(Some(out));
        }

        if inner == 1 {
            let bs = red.clamp(1, 1024).next_power_of_two().clamp(32, 1024) as u32;
            let splits: u64 = if outer < 64 && red > 8192 { 128u64.div_ceil(outer).min(red.div_ceil(4096)).clamp(1, 256) } else { 1 };
            let f = cuda_rt::func(dev, "reduce.cu", dt, "reduce_rows")?;
            if splits == 1 {
                Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(outer).u64(red).u64(red.max(1)).i32(op_code).f64(scale).launch(&f, [outer as u32, 1, 1], [bs, 1, 1], bs * csize)?;
            } else {
                let chunk = red.div_ceil(splits);
                let partial = Self::alloc(dev, self.dtype, (outer * splits) as usize)?;
                let pass1 = if op == ReduceOp::Mean { 0 } else { op_code };
                Args::new().ptr(ptr).ptr(partial.buf.device_ptr()).u64(outer).u64(red).u64(chunk).i32(pass1).f64(1.0).launch(&f, [outer as u32, splits as u32, 1], [bs, 1, 1], bs * csize)?;
                let bs2 = splits.next_power_of_two().clamp(32, 1024) as u32;
                Args::new().ptr(partial.buf.device_ptr()).ptr(out.buf.device_ptr()).u64(outer).u64(splits).u64(splits).i32(op_code).f64(scale).launch(&f, [outer as u32, 1, 1], [bs2, 1, 1], bs2 * csize)?;
            }
        } else {
            if outer > 65535 {
                return Ok(None);
            }
            let blocks = inner.div_ceil(32) * outer;
            let splits: u64 = if blocks < 128 && red > 2048 { 128u64.div_ceil(blocks).min(red.div_ceil(2048)).clamp(1, 64) } else { 1 };
            let f = cuda_rt::func(dev, "reduce.cu", dt, "reduce_cols")?;
            let grid = [inner.div_ceil(32) as u32, outer as u32, splits as u32];
            if splits == 1 {
                Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(outer).u64(red).u64(inner).u64(red.max(1)).i32(op_code).f64(scale).launch(&f, grid, [32, 8, 1], 0)?;
            } else {
                let chunk = red.div_ceil(splits);
                let partial = Self::alloc(dev, self.dtype, (outer * splits * inner) as usize)?;
                let pass1 = if op == ReduceOp::Mean { 0 } else { op_code };
                Args::new().ptr(ptr).ptr(partial.buf.device_ptr()).u64(outer).u64(red).u64(inner).u64(chunk).i32(pass1).f64(1.0).launch(&f, grid, [32, 8, 1], 0)?;
                Args::new().ptr(partial.buf.device_ptr()).ptr(out.buf.device_ptr()).u64(outer).u64(splits).u64(inner).u64(splits).i32(op_code).f64(scale).launch(&f, [grid[0], grid[1], 1], [32, 8, 1], 0)?;
            }
        }
        drop(src);
        Ok(Some(out))
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------------------------------------------------

static ALLOW_TF32: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether f32 matrix products may use TF32 tensor cores (10-bit mantissa: faster, less precise). Off unless enabled with
/// [`set_allow_tf32`] or `OXIDEON_CUDA_TF32=1`.
pub fn allow_tf32() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match ALLOW_TF32.load(Relaxed) {
        -1 => {
            let on = std::env::var("OXIDEON_CUDA_TF32").is_ok_and(|v| v != "0");
            ALLOW_TF32.store(i8::from(on), Relaxed);
            on
        }
        v => v == 1,
    }
}

pub fn set_allow_tf32(on: bool) {
    ALLOW_TF32.store(i8::from(on), std::sync::atomic::Ordering::Relaxed);
}

fn broadcast_batch(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![1usize; n];
    for i in 0..n {
        let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = match (x, y) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
    }
    Some(out)
}

impl CudaStorage {
    /// `A @ B` for rank >= 2 operands with broadcast batch dimensions; views are multiplied through their strides.
    pub fn matmul_gpu(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<Option<(Self, Vec<usize>)>> {
        if !matches!(self.dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16 | DType::I32 | DType::I64) || self.dtype != rhs.dtype || !self.rt_ready() {
            return Ok(None);
        }
        let (ar, br) = (lhs_l.ndim(), rhs_l.ndim());
        if ar < 2 || br < 2 {
            return Ok(None);
        }
        let (m, k) = (lhs_l.shape()[ar - 2], lhs_l.shape()[ar - 1]);
        let (k2, n) = (rhs_l.shape()[br - 2], rhs_l.shape()[br - 1]);
        if k != k2 || m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
            return Ok(None);
        }
        let Some(batch) = broadcast_batch(&lhs_l.shape()[..ar - 2], &rhs_l.shape()[..br - 2]) else { return Ok(None) };
        let nbatch: usize = batch.iter().product();
        let mut out_shape = batch.clone();
        out_shape.extend([m, n]);
        let out = Self::alloc(self.device_index, self.dtype, nbatch * m * n)?;
        if nbatch * m * n == 0 {
            return Ok(Some((out, out_shape)));
        }
        if nbatch > 65535 || m.div_ceil(32) > 65535 {
            return Ok(None);
        }
        let lb = Layout::new(lhs_l.shape()[..ar - 2].to_vec(), lhs_l.strides()[..ar - 2].to_vec(), 0).broadcast_to(&batch)?;
        let rb = Layout::new(rhs_l.shape()[..br - 2].to_vec(), rhs_l.strides()[..br - 2].to_vec(), 0).broadcast_to(&batch)?;
        let Some((bs, _)) = strided_batch(&batch, &[lb.strides(), rb.strides()]) else { return Ok(None) };
        let (rsa, csa) = (lhs_l.strides()[ar - 2] as i64, lhs_l.strides()[ar - 1] as i64);
        let (rsb, csb) = (rhs_l.strides()[br - 2] as i64, rhs_l.strides()[br - 1] as i64);
        let dt = dtype_code(self.dtype).unwrap();
        // f32 through cuBLAS when it is present (OXIDEON_CUDA_CUBLAS=0 turns it off): a batch that is one strided dimension
        // (or none) and operands with a unit row or column stride; anything else falls through to the .cu kernels below
        if self.dtype == DType::F32 && bs.nd <= 1 && super::cublas::available() {
            let (sa, sb) = if bs.nd == 1 { (bs.stride[0][0], bs.stride[1][0]) } else { (0, 0) };
            let a = super::cublas::Mat { ptr: self.base(lhs_l), rs: rsa, cs: csa, bs: sa };
            let b = super::cublas::Mat { ptr: rhs.base(rhs_l), rs: rsb, cs: csb, bs: sb };
            if super::cublas::sgemm_strided_batched(self.device_index, a, b, out.buf.device_ptr(), m, n, k, nbatch, allow_tf32())? {
                return Ok(Some((out, out_shape)));
            }
        }
        let large = m >= 64 && n >= 64;
        // tensor cores (sm_80+): always for f16 / bf16, for f32 only as TF32 when enabled (it lowers the precision)
        let tensor_cores = cuda_driver::cuda_device_info(self.device_index).is_ok_and(|i| i.major >= 8)
            && m >= 32
            && n >= 32
            && k >= 16
            && match self.dtype {
                DType::F16 | DType::BF16 => true,
                DType::F32 => allow_tf32(),
                _ => false,
            };
        let (file, name, bm, bn, threads) = if tensor_cores {
            ("gemm_tc.cu", if self.dtype == DType::F32 { "gemm_tf32" } else { "gemm_tc" }, 64usize, 64usize, 128u32)
        } else if large {
            ("gemm.cu", "gemm_large", 128, 128, 256)
        } else {
            ("gemm.cu", "gemm_small", 32, 32, 256)
        };
        let f = cuda_rt::func(self.device_index, file, dt, name)?;
        Args::new()
            .ptr(self.base(lhs_l))
            .ptr(rhs.base(rhs_l))
            .ptr(out.buf.device_ptr())
            .i32(m as i32)
            .i32(n as i32)
            .i32(k as i32)
            .i64(rsa)
            .i64(csa)
            .i64(rsb)
            .i64(csb)
            .strided(&bs)
            .launch(&f, [n.div_ceil(bn) as u32, m.div_ceil(bm) as u32, nbatch as u32], [threads, 1, 1], 0)?;
        Ok(Some((out, out_shape)))
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Softmax and normalization
// ---------------------------------------------------------------------------------------------------------------------

impl CudaStorage {
    fn float_module_ready(&self) -> bool {
        matches!(self.dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) && self.rt_ready()
    }

    /// Softmax or log-softmax over the last axis of a tensor. With `check_finite` the second value says whether some row had
    /// a non-finite maximum (which needs a host read, so ask for it only when gradients are being recorded).
    pub fn softmax_gpu(&self, layout: &Layout, log: bool, check_finite: bool) -> Result<Option<(Self, bool)>> {
        if !self.float_module_ready() {
            return Ok(None);
        }
        let Some(&cols) = layout.shape().last() else { return Ok(None) };
        if cols == 0 {
            return Ok(None);
        }
        let Some((src, ptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let n = layout.numel();
        let rows = (n / cols) as u64;
        let out = Self::alloc(self.device_index, self.dtype, n)?;
        let flag = if check_finite {
            let f = CudaBuffer::alloc(self.device_index, 4)?;
            f.memset(0)?;
            Some(f)
        } else {
            None
        };
        let f = cuda_rt::func(self.device_index, "nn.cu", dtype_code(self.dtype).unwrap(), "softmax_fwd")?;
        let bs = (cols as u64).next_power_of_two().clamp(32, 1024) as u32;
        Args::new()
            .ptr(ptr)
            .ptr(out.buf.device_ptr())
            .u64(rows)
            .u64(cols as u64)
            .i32(i32::from(log))
            .ptr(flag.as_ref().map_or(0, |f| f.device_ptr()))
            .launch(&f, [rows as u32, 1, 1], [bs, 1, 1], 0)?;
        let flagged = match &flag {
            Some(f) => {
                super::cuda_driver::cuda_synchronize(self.device_index)?;
                f.to_host_bytes(4)?.iter().any(|&b| b != 0)
            }
            None => false,
        };
        drop(src);
        Ok(Some((out, flagged)))
    }

    /// Layer norm over the last axis. Returns `(y, xhat, rstd)`; `xhat` and `rstd` only when `keep_stats` (needed for the
    /// backward pass). `w` / `b` are `[cols]` tensors of the same dtype.
    pub fn layer_norm_gpu(&self, layout: &Layout, w: Option<&Self>, b: Option<&Self>, eps: f64, keep_stats: bool) -> Result<Option<LayerNormOut>> {
        if !self.float_module_ready() {
            return Ok(None);
        }
        let Some(&cols) = layout.shape().last() else { return Ok(None) };
        if cols == 0 || layout.numel() == 0 {
            return Ok(None);
        }
        for t in [w, b].into_iter().flatten() {
            if t.dtype != self.dtype || t.len != cols {
                return Ok(None);
            }
        }
        let Some((src, ptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let n = layout.numel();
        let rows = (n / cols) as u64;
        let y = Self::alloc(self.device_index, self.dtype, n)?;
        let xhat = if keep_stats { Some(Self::alloc(self.device_index, self.dtype, n)?) } else { None };
        let stat_dtype = if self.dtype == DType::F64 { DType::F64 } else { DType::F32 };
        let rstd = if keep_stats { Some(Self::alloc(self.device_index, stat_dtype, rows as usize)?) } else { None };
        let f = cuda_rt::func(self.device_index, "nn.cu", dtype_code(self.dtype).unwrap(), "layernorm_fwd")?;
        let bs = (cols as u64).next_power_of_two().clamp(32, 1024) as u32;
        Args::new()
            .ptr(ptr)
            .ptr(w.map_or(0, |t| t.buf.device_ptr()))
            .ptr(b.map_or(0, |t| t.buf.device_ptr()))
            .ptr(y.buf.device_ptr())
            .ptr(xhat.as_ref().map_or(0, |t| t.buf.device_ptr()))
            .ptr(rstd.as_ref().map_or(0, |t| t.buf.device_ptr()))
            .u64(rows)
            .u64(cols as u64)
            .f64(eps)
            .launch(&f, [rows as u32, 1, 1], [bs, 1, 1], 0)?;
        drop(src);
        Ok(Some((y, xhat, rstd)))
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Convolution helpers: im2col / col2im and max pooling
// ---------------------------------------------------------------------------------------------------------------------

/// Mirror of `struct Geo` in `conv.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ConvGeo {
    pub n: i32,
    pub c: i32,
    pub h: i32,
    pub w: i32,
    pub kh: i32,
    pub kw: i32,
    pub sh: i32,
    pub sw: i32,
    pub ph: i32,
    pub pw: i32,
    pub dh: i32,
    pub dw: i32,
    pub oh: i32,
    pub ow: i32,
}

impl ConvGeo {
    #[allow(clippy::too_many_arguments)]
    pub fn new(n: usize, c: usize, h: usize, w: usize, k: (usize, usize), s: (usize, usize), p: (usize, usize), d: (usize, usize), oh: usize, ow: usize) -> Option<Self> {
        let f = |v: usize| i32::try_from(v).ok();
        Some(Self { n: f(n)?, c: f(c)?, h: f(h)?, w: f(w)?, kh: f(k.0)?, kw: f(k.1)?, sh: f(s.0)?, sw: f(s.1)?, ph: f(p.0)?, pw: f(p.1)?, dh: f(d.0)?, dw: f(d.1)?, oh: f(oh)?, ow: f(ow)? })
    }
}

impl CudaStorage {
    /// im2col of a `[N, C, H, W]` tensor: `[N, C*kh*kw, oh*ow]`.
    pub fn im2col_gpu(&self, layout: &Layout, g: &ConvGeo) -> Result<Option<Self>> {
        if !self.float_module_ready() {
            return Ok(None);
        }
        let Some((src, ptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let total = g.n as u64 * g.c as u64 * g.kh as u64 * g.kw as u64 * g.oh as u64 * g.ow as u64;
        let out = Self::alloc(self.device_index, self.dtype, total as usize)?;
        let f = cuda_rt::func(self.device_index, "conv.cu", dtype_code(self.dtype).unwrap(), "im2col")?;
        Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(total).raw(g).launch_1d(&f, total)?;
        drop(src);
        Ok(Some(out))
    }

    /// col2im: `[N, C*kh*kw, oh*ow]` back to `[N, C, H, W]`, summing overlapping windows.
    pub fn col2im_gpu(&self, layout: &Layout, g: &ConvGeo) -> Result<Option<Self>> {
        if !self.float_module_ready() {
            return Ok(None);
        }
        let Some((src, ptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let total = g.n as u64 * g.c as u64 * g.h as u64 * g.w as u64;
        let out = Self::alloc(self.device_index, self.dtype, total as usize)?;
        let f = cuda_rt::func(self.device_index, "conv.cu", dtype_code(self.dtype).unwrap(), "col2im")?;
        Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(total).raw(g).launch_1d(&f, total)?;
        drop(src);
        Ok(Some(out))
    }

    /// Max pooling. Returns the pooled tensor and, when `with_index`, the flat `ih * W + iw` position of each winner.
    pub fn max_pool_gpu(&self, layout: &Layout, g: &ConvGeo, with_index: bool) -> Result<Option<(Self, Option<Self>)>> {
        if !self.float_module_ready() {
            return Ok(None);
        }
        let Some((src, ptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let total = g.n as u64 * g.c as u64 * g.oh as u64 * g.ow as u64;
        let out = Self::alloc(self.device_index, self.dtype, total as usize)?;
        let arg = if with_index { Some(Self::alloc(self.device_index, DType::I64, total as usize)?) } else { None };
        let f = cuda_rt::func(self.device_index, "conv.cu", dtype_code(self.dtype).unwrap(), "maxpool_fwd")?;
        Args::new().ptr(ptr).ptr(out.buf.device_ptr()).ptr(arg.as_ref().map_or(0, |a| a.buf.device_ptr())).u64(total).raw(g).launch_1d(&f, total)?;
        drop(src);
        Ok(Some((out, arg)))
    }

    /// Gradient of max pooling: `self` is the gradient of the pooled output.
    pub fn max_pool_bwd_gpu(&self, layout: &Layout, arg: &Self, g: &ConvGeo) -> Result<Option<Self>> {
        if !self.float_module_ready() || arg.dtype != DType::I64 {
            return Ok(None);
        }
        let Some((src, ptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let total = g.n as u64 * g.c as u64 * g.h as u64 * g.w as u64;
        let out = Self::alloc(self.device_index, self.dtype, total as usize)?;
        let f = cuda_rt::func(self.device_index, "conv.cu", dtype_code(self.dtype).unwrap(), "maxpool_bwd")?;
        Args::new().ptr(ptr).ptr(arg.buf.device_ptr()).ptr(out.buf.device_ptr()).u64(total).raw(g).launch_1d(&f, total)?;
        drop(src);
        Ok(Some(out))
    }
}

/// `Storage`-level entry points for the tensor ops (they return `None` for host storage or unsupported cases).
impl super::Storage {
    pub(crate) fn cuda_im2col(&self, layout: &Layout, g: &ConvGeo) -> Result<Option<super::Storage>> {
        match &*self.read() {
            super::StorageInner::Cuda(s) => Ok(s.im2col_gpu(layout, g)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_col2im(&self, layout: &Layout, g: &ConvGeo) -> Result<Option<super::Storage>> {
        match &*self.read() {
            super::StorageInner::Cuda(s) => Ok(s.col2im_gpu(layout, g)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_max_pool(&self, layout: &Layout, g: &ConvGeo, with_index: bool) -> Result<Option<(super::Storage, Option<super::Storage>)>> {
        match &*self.read() {
            super::StorageInner::Cuda(s) => Ok(s.max_pool_gpu(layout, g, with_index)?.map(|(o, a)| (super::Storage::new_cuda(o), a.map(super::Storage::new_cuda)))),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_max_pool_bwd(&self, layout: &Layout, arg: &super::Storage, g: &ConvGeo) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*arg.read()) {
            (super::StorageInner::Cuda(s), super::StorageInner::Cuda(a)) => Ok(s.max_pool_bwd_gpu(layout, a, g)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Concatenation, view scatter, gather / scatter / index_add
// ---------------------------------------------------------------------------------------------------------------------

fn normalize_indices(op: &'static str, idx: &[i64], dim: usize) -> Result<Vec<i64>> {
    idx.iter()
        .map(|&r| {
            let v = if r < 0 { dim as i64 + r } else { r };
            if v < 0 || v as usize >= dim {
                Err(Error::IndexOutOfBounds { op, index: r as isize, dim_size: dim })
            } else {
                Ok(v)
            }
        })
        .collect()
}

fn has_atomic_add(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::U32 | DType::U64)
}

impl CudaStorage {
    /// Concatenation of same-dtype parts along `ax` into a new contiguous tensor of `out_shape`.
    pub fn cat_gpu(parts: &[(&Self, &Layout)], ax: usize, out_shape: &[usize]) -> Result<Option<Self>> {
        let Some((first, _)) = parts.first() else { return Ok(None) };
        let dtype = first.dtype;
        if dtype_code(dtype).is_none() || !first.rt_ready() || parts.iter().any(|(p, _)| p.dtype != dtype || p.device_index != first.device_index) {
            return Ok(None);
        }
        let numel: usize = out_shape.iter().product();
        let out = Self::alloc(first.device_index, dtype, numel)?;
        let ostr = contiguous_strides(out_shape);
        let esize = dtype.size_in_bytes() as u64;
        let f = cuda_rt::func(first.device_index, "copy.cu", size_class_code(dtype), "copy")?;
        let mut start = 0usize;
        for (p, l) in parts {
            let n = l.numel();
            let len_ax = l.shape()[ax];
            if n > 0 {
                let Some((sd, n)) = strided(l.shape(), &[l.strides(), &ostr]) else { return Ok(None) };
                let dst = out.buf.device_ptr() + (start as i64 * ostr[ax] as i64) as u64 * esize;
                Args::new().ptr(p.base(l)).ptr(dst).u64(n).strided(&sd).launch_1d(&f, n)?;
            }
            start += len_ax;
        }
        Ok(Some(out))
    }

    /// `view_scatter`: a zero tensor of `spec.src_shape` with `self` (shaped `spec.dst_shape`) written through the view.
    /// Only for views that do not overlap (narrow, slice, flip, select, pad), where scattering is a plain copy.
    pub fn view_scatter_gpu(&self, src_layout: &Layout, spec: &crate::layout::ViewSpec) -> Result<Option<Self>> {
        if dtype_code(self.dtype).is_none() || !self.rt_ready() || src_layout.shape() != spec.dst_shape.as_slice() {
            return Ok(None);
        }
        let dst_layout = Layout::contiguous(spec.src_shape.clone());
        let view = spec.apply_layout(&dst_layout)?;
        view.check_storage_bounds(dst_layout.numel(), "view_scatter")?;
        if view.shape().iter().zip(view.strides()).any(|(&d, &s)| d > 1 && s == 0) {
            return Ok(None); // overlapping view: the CPU path accumulates
        }
        let out = Self::alloc(self.device_index, self.dtype, dst_layout.numel())?;
        out.buf.memset(0)?;
        let n = src_layout.numel();
        if n > 0 {
            let Some((sd, n)) = strided(src_layout.shape(), &[src_layout.strides(), view.strides()]) else { return Ok(None) };
            let dst = out.buf.device_ptr() + (view.offset() * self.dtype.size_in_bytes()) as u64;
            let f = cuda_rt::func(self.device_index, "copy.cu", size_class_code(self.dtype), "copy")?;
            Args::new().ptr(self.base(src_layout)).ptr(dst).u64(n).strided(&sd).launch_1d(&f, n)?;
        }
        Ok(Some(out))
    }

    fn upload_i64(&self, v: &[i64]) -> Result<CudaBuffer> {
        CudaBuffer::from_host_bytes(self.device_index, raw_bytes!(v, i64))
    }

    /// `out[..] = self[.. idx ..]` along `ax`; `idx` (host values, shape `idx_shape`) is validated like the CPU path.
    pub fn gather_gpu(&self, layout: &Layout, ax: usize, idx_shape: &[usize], idx: &[i64]) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || layout.ndim() != idx_shape.len() {
            return Ok(None);
        }
        let norm = normalize_indices("gather", idx, layout.shape()[ax])?;
        let istr = contiguous_strides(idx_shape);
        let Some((sd, n)) = strided_exact(idx_shape, &[layout.strides(), &istr, &istr]) else { return Ok(None) };
        let out = Self::alloc(self.device_index, self.dtype, n as usize)?;
        if n == 0 {
            return Ok(Some(out));
        }
        let ibuf = self.upload_i64(&norm)?;
        let f = cuda_rt::func(self.device_index, "copy.cu", dt, "gather")?;
        Args::new().ptr(self.base(layout)).ptr(ibuf.device_ptr()).ptr(out.buf.device_ptr()).u64(n).i32(ax as i32).i64(layout.strides()[ax] as i64).i64(layout.shape()[ax] as i64).ptr(0).strided(&sd).launch_1d(&f, n)?;
        Ok(Some(out))
    }

    /// Out-of-place `scatter` (`add == false`, last write wins) or `scatter_add` of `src` into a copy of `self`.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_gpu(&self, layout: &Layout, src: &Self, src_layout: &Layout, idx_shape: &[usize], ax: usize, idx: &[i64], add: bool) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || self.dtype != src.dtype || layout.ndim() != idx_shape.len() || (add && !has_atomic_add(self.dtype)) {
            return Ok(None);
        }
        let op = if add { "scatter_add" } else { "scatter" };
        let norm = normalize_indices(op, idx, layout.shape()[ax])?;
        let Some(out) = self.contiguous_gpu(layout)? else { return Ok(None) };
        // never write into the caller's storage when `contiguous_gpu` handed it back unchanged
        let out = if Arc::ptr_eq(&out.buf, &self.buf) {
            let copy = Self::alloc(self.device_index, self.dtype, self.len)?;
            copy.buf.copy_from_device(0, &out.buf, 0, self.len * self.dtype.size_in_bytes())?;
            copy
        } else {
            out
        };
        let dstr = contiguous_strides(layout.shape());
        let istr = contiguous_strides(idx_shape);
        let Some((sd, n)) = strided_exact(idx_shape, &[&dstr, &istr, src_layout.strides()]) else { return Ok(None) };
        if n > 0 {
            let ibuf = self.upload_i64(&norm)?;
            let f = cuda_rt::func(self.device_index, "copy.cu", dt, if add { "scatter_add" } else { "scatter" })?;
            Args::new().ptr(out.buf.device_ptr()).ptr(ibuf.device_ptr()).ptr(src.base(src_layout)).u64(n).i32(ax as i32).i64(dstr[ax] as i64).i64(layout.shape()[ax] as i64).ptr(0).strided(&sd).launch_1d(&f, n)?;
        }
        Ok(Some(out))
    }

    /// Out-of-place `index_add` along `ax` (`src` is `[.., idx.len(), ..]`).
    pub fn index_add_gpu(&self, layout: &Layout, src: &Self, src_layout: &Layout, ax: usize, idx: &[i64]) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || self.dtype != src.dtype || !has_atomic_add(self.dtype) {
            return Ok(None);
        }
        let shape = layout.shape();
        let norm = normalize_indices("index_add", idx, shape[ax])?;
        let Some((srcc, sptr)) = src.contiguous_ptr(src_layout)? else { return Ok(None) };
        let out = Self::alloc(self.device_index, self.dtype, layout.numel())?;
        // copy `self` (contiguous) into `out`, then accumulate
        let cs = contiguous_strides(shape);
        if layout.numel() > 0 {
            let Some((sd, n)) = strided(shape, &[layout.strides(), &cs]) else { return Ok(None) };
            let f = cuda_rt::func(self.device_index, "copy.cu", size_class_code(self.dtype), "copy")?;
            Args::new().ptr(self.base(layout)).ptr(out.buf.device_ptr()).u64(n).strided(&sd).launch_1d(&f, n)?;
        }
        let outer: u64 = shape[..ax].iter().map(|&d| d as u64).product();
        let inner: u64 = shape[ax + 1..].iter().map(|&d| d as u64).product();
        let n = outer * norm.len() as u64 * inner;
        if n > 0 {
            let ibuf = self.upload_i64(&norm)?;
            let f = cuda_rt::func(self.device_index, "copy.cu", dt, "index_add")?;
            Args::new().ptr(out.buf.device_ptr()).ptr(ibuf.device_ptr()).ptr(sptr).u64(outer).u64(shape[ax] as u64).u64(inner).u64(norm.len() as u64).ptr(0).launch_1d(&f, n)?;
        }
        drop(srcc);
        Ok(Some(out))
    }
}

// ---- device-resident indices, scans and top-k --------------------------------------------------------------------------

impl CudaStorage {
    fn new_err_buf(&self) -> Result<CudaBuffer> {
        let b = CudaBuffer::alloc(self.device_index, 16)?;
        b.memset(0)?;
        Ok(b)
    }

    /// Reads the kernel's validation flag (this synchronizes with the device) and turns it into the CPU backend's error.
    fn check_err(&self, buf: &CudaBuffer, op: &'static str, dim: usize) -> Result<()> {
        super::cuda_driver::cuda_synchronize(self.device_index)?;
        let b = buf.to_host_bytes(16)?;
        if b[..4].iter().any(|&x| x != 0) {
            let raw = i64::from_ne_bytes(b[8..16].try_into().unwrap());
            return Err(Error::IndexOutOfBounds { op, index: raw as isize, dim_size: dim });
        }
        Ok(())
    }

    /// `gather` with an index tensor that lives on the device (validated there, no download).
    pub fn gather_dev(&self, layout: &Layout, ax: usize, idx: &Self, idx_layout: &Layout) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || idx.dtype != DType::I64 || layout.ndim() != idx_layout.ndim() || idx.device_index != self.device_index {
            return Ok(None);
        }
        let istr = contiguous_strides(idx_layout.shape());
        let Some((sd, n)) = strided_exact(idx_layout.shape(), &[layout.strides(), idx_layout.strides(), &istr]) else { return Ok(None) };
        let out = Self::alloc(self.device_index, self.dtype, n as usize)?;
        if n == 0 {
            return Ok(Some(out));
        }
        let err = self.new_err_buf()?;
        let f = cuda_rt::func(self.device_index, "copy.cu", dt, "gather")?;
        Args::new().ptr(self.base(layout)).ptr(idx.base(idx_layout)).ptr(out.buf.device_ptr()).u64(n).i32(ax as i32).i64(layout.strides()[ax] as i64).i64(layout.shape()[ax] as i64).ptr(err.device_ptr()).strided(&sd).launch_1d(&f, n)?;
        self.check_err(&err, "gather", layout.shape()[ax])?;
        Ok(Some(out))
    }

    /// `scatter` / `scatter_add` with a device index tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_dev(&self, layout: &Layout, src: &Self, src_layout: &Layout, idx: &Self, idx_layout: &Layout, ax: usize, add: bool) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || self.dtype != src.dtype || idx.dtype != DType::I64 || layout.ndim() != idx_layout.ndim() || (add && !has_atomic_add(self.dtype)) || idx.device_index != self.device_index {
            return Ok(None);
        }
        let Some(out) = self.contiguous_gpu(layout)? else { return Ok(None) };
        let out = if Arc::ptr_eq(&out.buf, &self.buf) {
            let copy = Self::alloc(self.device_index, self.dtype, self.len)?;
            copy.buf.copy_from_device(0, &out.buf, 0, self.len * self.dtype.size_in_bytes())?;
            copy
        } else {
            out
        };
        let dstr = contiguous_strides(layout.shape());
        let Some((sd, n)) = strided_exact(idx_layout.shape(), &[&dstr, idx_layout.strides(), src_layout.strides()]) else { return Ok(None) };
        if n > 0 {
            let err = self.new_err_buf()?;
            let f = cuda_rt::func(self.device_index, "copy.cu", dt, if add { "scatter_add" } else { "scatter" })?;
            Args::new().ptr(out.buf.device_ptr()).ptr(idx.base(idx_layout)).ptr(src.base(src_layout)).u64(n).i32(ax as i32).i64(dstr[ax] as i64).i64(layout.shape()[ax] as i64).ptr(err.device_ptr()).strided(&sd).launch_1d(&f, n)?;
            self.check_err(&err, if add { "scatter_add" } else { "scatter" }, layout.shape()[ax])?;
        }
        Ok(Some(out))
    }

    /// `index_select` with a device index vector.
    pub fn index_select_dev(&self, layout: &Layout, ax: usize, idx: &Self, idx_layout: &Layout) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || idx.dtype != DType::I64 || idx.device_index != self.device_index {
            return Ok(None);
        }
        let Some((srcc, sptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let Some((idxc, iptr)) = idx.contiguous_ptr(idx_layout)? else { return Ok(None) };
        let shape = layout.shape();
        let (outer, inner): (u64, u64) = (shape[..ax].iter().map(|&d| d as u64).product(), shape[ax + 1..].iter().map(|&d| d as u64).product());
        let nidx = idx_layout.numel() as u64;
        let n = outer * nidx * inner;
        let out = Self::alloc(self.device_index, self.dtype, n as usize)?;
        if n > 0 {
            let err = self.new_err_buf()?;
            let f = cuda_rt::func(self.device_index, "copy.cu", dt, "index_select")?;
            Args::new().ptr(sptr).ptr(iptr).ptr(out.buf.device_ptr()).u64(outer).u64(shape[ax] as u64).u64(inner).u64(nidx).ptr(err.device_ptr()).launch_1d(&f, n)?;
            self.check_err(&err, "index_select", shape[ax])?;
        }
        drop((srcc, idxc));
        Ok(Some(out))
    }

    /// `index_add` with a device index vector.
    pub fn index_add_dev(&self, layout: &Layout, src: &Self, src_layout: &Layout, idx: &Self, idx_layout: &Layout, ax: usize) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !self.rt_ready() || self.dtype != src.dtype || !has_atomic_add(self.dtype) || idx.dtype != DType::I64 || idx.device_index != self.device_index {
            return Ok(None);
        }
        let Some((srcc, sptr)) = src.contiguous_ptr(src_layout)? else { return Ok(None) };
        let Some((idxc, iptr)) = idx.contiguous_ptr(idx_layout)? else { return Ok(None) };
        let shape = layout.shape();
        let out = Self::alloc(self.device_index, self.dtype, layout.numel())?;
        let cs = contiguous_strides(shape);
        if layout.numel() > 0 {
            let Some((sd, n)) = strided(shape, &[layout.strides(), &cs]) else { return Ok(None) };
            let f = cuda_rt::func(self.device_index, "copy.cu", size_class_code(self.dtype), "copy")?;
            Args::new().ptr(self.base(layout)).ptr(out.buf.device_ptr()).u64(n).strided(&sd).launch_1d(&f, n)?;
        }
        let (outer, inner): (u64, u64) = (shape[..ax].iter().map(|&d| d as u64).product(), shape[ax + 1..].iter().map(|&d| d as u64).product());
        let nidx = idx_layout.numel() as u64;
        let n = outer * nidx * inner;
        if n > 0 {
            let err = self.new_err_buf()?;
            let f = cuda_rt::func(self.device_index, "copy.cu", dt, "index_add")?;
            Args::new().ptr(out.buf.device_ptr()).ptr(iptr).ptr(sptr).u64(outer).u64(shape[ax] as u64).u64(inner).u64(nidx).ptr(err.device_ptr()).launch_1d(&f, n)?;
            self.check_err(&err, "index_add", shape[ax])?;
        }
        drop((srcc, idxc));
        Ok(Some(out))
    }

    /// Cumulative sum along `ax` (accumulated in f64 for float types, like the CPU backend).
    pub fn cumsum_gpu(&self, layout: &Layout, ax: usize, prod: bool) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        let ok = matches!(self.dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16 | DType::I64);
        if !ok || !self.rt_ready() {
            return Ok(None);
        }
        let Some((src, ptr)) = self.contiguous_ptr(layout)? else { return Ok(None) };
        let shape = layout.shape();
        let (outer, n, inner): (u64, u64, u64) = (shape[..ax].iter().map(|&d| d as u64).product(), shape[ax] as u64, shape[ax + 1..].iter().map(|&d| d as u64).product());
        let out = Self::alloc(self.device_index, self.dtype, layout.numel())?;
        if layout.numel() == 0 {
            return Ok(Some(out));
        }
        if inner == 1 {
            let bs = n.clamp(1, 256).next_power_of_two().clamp(32, 256) as u32;
            let f = cuda_rt::func(self.device_index, "reduce.cu", dt, "scan_rows")?;
            Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(outer).u64(n).i32(prod as i32).launch(&f, [outer as u32, 1, 1], [bs, 1, 1], bs * 8)?;
        } else {
            if outer > 65535 {
                return Ok(None);
            }
            let f = cuda_rt::func(self.device_index, "reduce.cu", dt, "scan_cols")?;
            Args::new().ptr(ptr).ptr(out.buf.device_ptr()).u64(outer).u64(n).u64(inner).i32(prod as i32).launch(&f, [inner.div_ceil(128) as u32, outer as u32, 1], [128, 1, 1], 0)?;
        }
        drop(src);
        Ok(Some(out))
    }

    /// Gradient of an inclusive cumprod along `ax` (x, its cumprod `out` and the incoming gradient `go`), zero-safe.
    pub fn cumprod_bwd_gpu(&self, x_layout: &Layout, out: &Self, out_layout: &Layout, go: &Self, go_layout: &Layout, ax: usize) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        if !matches!(self.dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) || !self.rt_ready() || out.dtype != self.dtype || go.dtype != self.dtype {
            return Ok(None);
        }
        let (Some((_a, px)), Some((_b, po)), Some((_c, pg))) = (self.contiguous_ptr(x_layout)?, out.contiguous_ptr(out_layout)?, go.contiguous_ptr(go_layout)?) else { return Ok(None) };
        let shape = x_layout.shape();
        let (outer, n, inner): (u64, u64, u64) = (shape[..ax].iter().map(|&d| d as u64).product(), shape[ax] as u64, shape[ax + 1..].iter().map(|&d| d as u64).product());
        let gx = Self::alloc(self.device_index, self.dtype, x_layout.numel())?;
        if x_layout.numel() > 0 {
            let f = cuda_rt::func(self.device_index, "reduce.cu", dt, "cumprod_bwd")?;
            Args::new().ptr(px).ptr(po).ptr(pg).ptr(gx.buf.device_ptr()).u64(outer).u64(n).u64(inner).launch_1d(&f, outer * n * inner)?;
        }
        Ok(Some(gx))
    }

    /// Indices of the `k` largest / smallest elements along `ax` in the CPU backend's order (total order on values with NaN
    /// largest, ties by original index), as an i64 tensor of shape `[.., k, ..]`.
    pub fn topk_indices_gpu(&self, layout: &Layout, ax: usize, k: usize, largest: bool) -> Result<Option<Self>> {
        let Some(dt) = dtype_code(self.dtype) else { return Ok(None) };
        let shape = layout.shape();
        let n = shape[ax];
        // one block sorts 4096 candidates; reducing more than that in stages only makes progress while each chunk keeps at
        // most half of its candidates
        if self.dtype == DType::Bool || !self.rt_ready() {
            return Ok(None);
        }
        let global = n > 4096 && k > 2048;
        let (outer, inner): (usize, usize) = (shape[..ax].iter().product(), shape[ax + 1..].iter().product());
        let rows = outer * inner;
        let mut out_shape = shape.to_vec();
        out_shape[ax] = k;
        let result = Self::alloc(self.device_index, DType::I64, outer * k * inner)?;
        if rows == 0 || k == 0 || n == 0 {
            return Ok(Some(result));
        }
        if rows > 65535 || n > u32::MAX as usize {
            return Ok(None);
        }
        // the sorted axis last, contiguous: rows = outer * inner
        let mut perm: Vec<usize> = (0..shape.len()).filter(|&d| d != ax).collect();
        perm.push(ax);
        let moved = layout.permute(&perm)?;
        let Some((src, ptr)) = self.contiguous_ptr(&moved)? else { return Ok(None) };
        let sorted = if inner == 1 { None } else { Some(Self::alloc(self.device_index, DType::I64, rows * k)?) };
        let dest = sorted.as_ref().map_or(result.buf.device_ptr(), |s| s.buf.device_ptr());
        if global {
            // sort whole rows with a bitonic network in global memory (any k up to n)
            let npad = n.next_power_of_two();
            if (rows as u64) * (npad as u64) > (1u64 << 28) || npad > (1usize << 31) {
                return Ok(None);
            }
            let ids = CudaBuffer::alloc(self.device_index, rows * npad * 4)?;
            let init = cuda_rt::func(self.device_index, "reduce.cu", dt, "gsort_init")?;
            Args::new().ptr(ids.device_ptr()).u64(n as u64).u64(npad as u64).u64((rows * npad) as u64).launch_1d(&init, (rows * npad) as u64)?;
            let step = cuda_rt::func(self.device_index, "reduce.cu", dt, "gsort_step")?;
            let gx = (npad as u64 / 2).div_ceil(256).min(4096) as u32;
            let mut size = 2u32 as u64;
            while size <= npad as u64 {
                let mut stride = size >> 1;
                while stride > 0 {
                    Args::new().ptr(ptr).ptr(ids.device_ptr()).u64(n as u64).u64(npad as u64).u32(size as u32).u32(stride as u32).i32(i32::from(largest)).launch(&step, [gx, rows as u32, 1], [256, 1, 1], 0)?;
                    stride >>= 1;
                }
                size <<= 1;
            }
            let take = cuda_rt::func(self.device_index, "reduce.cu", dt, "gsort_take")?;
            Args::new().ptr(ids.device_ptr()).ptr(dest).u64(npad as u64).u64(k as u64).u64((rows * k) as u64).launch_1d(&take, (rows * k) as u64)?;
        }
        let f = cuda_rt::func(self.device_index, "reduce.cu", dt, "topk_stage")?;
        const SORT_N: u64 = 4096;
        let kk = (k as u64).min(SORT_N);
        let (mut count, mut cand_in, mut in_stride): (u64, Option<CudaBuffer>, u64) = (n as u64, None, 0);
        loop {
            if global {
                break;
            }
            let chunks = count.div_ceil(SORT_N);
            let last = chunks == 1;
            let cand_out = if last { None } else { Some(CudaBuffer::alloc(self.device_index, rows * (chunks * kk) as usize * 4)?) };
            Args::new()
                .ptr(ptr)
                .ptr(cand_in.as_ref().map_or(0, |b| b.device_ptr()))
                .ptr(cand_out.as_ref().map_or(0, |b| b.device_ptr()))
                .ptr(if last { dest } else { 0 })
                .u64(n as u64)
                .u64(count)
                .u64(in_stride)
                .u64(k as u64)
                .u64(if last { k as u64 } else { chunks * kk })
                .i32(i32::from(largest))
                .launch(&f, [chunks as u32, rows as u32, 1], [1024, 1, 1], (SORT_N * 12) as u32)?;
            if last {
                break;
            }
            let next = (chunks - 1) * kk + kk.min(count - (chunks - 1) * SORT_N);
            if next >= count {
                return Ok(None); // no progress (cannot happen with the limits above); let the CPU path answer
            }
            count = next;
            cand_in = cand_out;
            in_stride = chunks * kk;
        }
        drop(src);
        match sorted {
            None => Ok(Some(result)),
            Some(s) => {
                // [outer, inner, k] -> [outer, k, inner]
                let (o, i, kd) = (outer, inner, k);
                let l = Layout::new(vec![o, kd, i], vec![(i * kd) as isize, 1, kd as isize], 0);
                let cs = contiguous_strides(&[o, kd, i]);
                let Some((sd, total)) = strided(&[o, kd, i], &[l.strides(), &cs]) else { return Ok(None) };
                let cp = cuda_rt::func(self.device_index, "copy.cu", 11, "copy")?;
                Args::new().ptr(s.buf.device_ptr()).ptr(result.buf.device_ptr()).u64(total).strided(&sd).launch_1d(&cp, total)?;
                Ok(Some(result))
            }
        }
    }
}

impl super::Storage {
    pub(crate) fn cuda_gather_dev(&self, layout: &Layout, ax: usize, idx: &super::Storage, idx_layout: &Layout) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*idx.read()) {
            (super::StorageInner::Cuda(a), super::StorageInner::Cuda(i)) => Ok(a.gather_dev(layout, ax, i, idx_layout)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_scatter_dev(&self, layout: &Layout, src: &super::Storage, src_layout: &Layout, idx: &super::Storage, idx_layout: &Layout, ax: usize, add: bool) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*src.read(), &*idx.read()) {
            (super::StorageInner::Cuda(a), super::StorageInner::Cuda(b), super::StorageInner::Cuda(i)) => Ok(a.scatter_dev(layout, b, src_layout, i, idx_layout, ax, add)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_index_select_dev(&self, layout: &Layout, ax: usize, idx: &super::Storage, idx_layout: &Layout) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*idx.read()) {
            (super::StorageInner::Cuda(a), super::StorageInner::Cuda(i)) => Ok(a.index_select_dev(layout, ax, i, idx_layout)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_index_add_dev(&self, layout: &Layout, src: &super::Storage, src_layout: &Layout, idx: &super::Storage, idx_layout: &Layout, ax: usize) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*src.read(), &*idx.read()) {
            (super::StorageInner::Cuda(a), super::StorageInner::Cuda(b), super::StorageInner::Cuda(i)) => Ok(a.index_add_dev(layout, b, src_layout, i, idx_layout, ax)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_cumprod_bwd(&self, x_layout: &Layout, out: &super::Storage, out_layout: &Layout, go: &super::Storage, go_layout: &Layout, ax: usize) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*out.read(), &*go.read()) {
            (super::StorageInner::Cuda(a), super::StorageInner::Cuda(b), super::StorageInner::Cuda(c)) => Ok(a.cumprod_bwd_gpu(x_layout, b, out_layout, c, go_layout, ax)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_cumsum(&self, layout: &Layout, ax: usize, prod: bool) -> Result<Option<super::Storage>> {
        match &*self.read() {
            super::StorageInner::Cuda(a) => Ok(a.cumsum_gpu(layout, ax, prod)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_topk_indices(&self, layout: &Layout, ax: usize, k: usize, largest: bool) -> Result<Option<super::Storage>> {
        match &*self.read() {
            super::StorageInner::Cuda(a) => Ok(a.topk_indices_gpu(layout, ax, k, largest)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }
}

// ---- random numbers on the device ------------------------------------------------------------------------------------------

impl CudaStorage {
    /// `len` uniform samples in `[low, high)` (or standard normal samples when `normal`) drawn from the Philox stream `key`.
    pub fn rand_gpu(device_index: usize, dtype: DType, len: usize, normal: bool, key: u64, low: f64, high: f64) -> Result<Option<Self>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) || !cuda_rt::available() {
            return Ok(None);
        }
        let out = Self::alloc(device_index, dtype, len)?;
        let dt = dtype_code(dtype).unwrap();
        if normal {
            let f = cuda_rt::func(device_index, "rng.cu", dt, "rand_normal")?;
            Args::new().ptr(out.buf.device_ptr()).u64(len as u64).u64(key).launch_1d(&f, len as u64)?;
        } else {
            let f = cuda_rt::func(device_index, "rng.cu", dt, "rand_uniform")?;
            Args::new().ptr(out.buf.device_ptr()).u64(len as u64).u64(key).f64(low).f64(high).launch_1d(&f, len as u64)?;
        }
        Ok(Some(out))
    }

    /// A dropout mask: `scale` where an element is kept (probability `keep`), 0 elsewhere.
    pub fn bernoulli_mask_gpu(device_index: usize, dtype: DType, len: usize, key: u64, keep: f64, scale: f64) -> Result<Option<Self>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) || !cuda_rt::available() {
            return Ok(None);
        }
        let out = Self::alloc(device_index, dtype, len)?;
        let f = cuda_rt::func(device_index, "rng.cu", dtype_code(dtype).unwrap(), "bernoulli_mask")?;
        Args::new().ptr(out.buf.device_ptr()).u64(len as u64).u64(key).f64(keep).f64(scale).launch_1d(&f, len as u64)?;
        Ok(Some(out))
    }
}

impl super::Storage {
    /// Random tensor created directly on a CUDA device (`None` for other devices or unsupported dtypes).
    pub(crate) fn cuda_rand(device: &crate::device::Device, dtype: DType, len: usize, normal: bool, key: u64, low: f64, high: f64) -> Result<Option<super::Storage>> {
        match device {
            crate::device::Device::Cuda(idx) if super::cuda_driver::cuda_available() && *idx < super::cuda_driver::cuda_device_count() => Ok(CudaStorage::rand_gpu(*idx, dtype, len, normal, key, low, high)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_bernoulli_mask(device: &crate::device::Device, dtype: DType, len: usize, key: u64, keep: f64, scale: f64) -> Result<Option<super::Storage>> {
        match device {
            crate::device::Device::Cuda(idx) if super::cuda_driver::cuda_available() && *idx < super::cuda_driver::cuda_device_count() => Ok(CudaStorage::bernoulli_mask_gpu(*idx, dtype, len, key, keep, scale)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    /// A constant tensor created directly on a CUDA device.
    pub(crate) fn cuda_full(device: &crate::device::Device, dtype: DType, len: usize, value: f64) -> Result<Option<super::Storage>> {
        match device {
            crate::device::Device::Cuda(idx) if super::cuda_driver::cuda_available() && *idx < super::cuda_driver::cuda_device_count() => Ok(CudaStorage::full_gpu(*idx, dtype, len, value)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }
}

// ---- fused optimizer updates -----------------------------------------------------------------------------------------------

impl CudaStorage {
    /// In-place Adam / AdamW update of this (contiguous, f32 / f64) parameter buffer; `false` when it does not apply.
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_gpu(&self, p_layout: &Layout, g: &Self, g_layout: &Layout, m: &Self, v: &Self, lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64, decoupled: bool, bc1: f64, bc2: f64) -> Result<bool> {
        let n = p_layout.numel();
        if !matches!(self.dtype, DType::F32 | DType::F64) || g.dtype != self.dtype || m.dtype != self.dtype || v.dtype != self.dtype || !self.rt_ready() {
            return Ok(false);
        }
        if n == 0 || !p_layout.is_contiguous() || !g_layout.is_contiguous() || g_layout.numel() != n || m.len != n || v.len != n {
            return Ok(false);
        }
        let f = cuda_rt::func(self.device_index, "optim.cu", dtype_code(self.dtype).unwrap(), "adam_step")?;
        Args::new()
            .ptr(self.base(p_layout))
            .ptr(g.base(g_layout))
            .ptr(m.buf.device_ptr())
            .ptr(v.buf.device_ptr())
            .u64(n as u64)
            .f64(lr)
            .f64(beta1)
            .f64(beta2)
            .f64(eps)
            .f64(weight_decay)
            .i32(i32::from(decoupled))
            .f64(1.0 / bc1)
            .f64(1.0 / bc2)
            .launch_1d(&f, n as u64)?;
        Ok(true)
    }

    /// Sum of squares of all elements (accumulated in f64 on the device, one small download).
    pub fn sum_squares_gpu(&self, layout: &Layout) -> Result<Option<f64>> {
        if !matches!(self.dtype, DType::F32 | DType::F64) || !self.rt_ready() || !layout.is_contiguous() {
            return Ok(None);
        }
        let n = layout.numel() as u64;
        if n == 0 {
            return Ok(Some(0.0));
        }
        let blocks = n.div_ceil(1024).min(1024) as u32;
        let part = CudaBuffer::alloc(self.device_index, blocks as usize * 8)?;
        let f = cuda_rt::func(self.device_index, "optim.cu", dtype_code(self.dtype).unwrap(), "sumsq_partial")?;
        Args::new().ptr(self.base(layout)).ptr(part.device_ptr()).u64(n).launch(&f, [blocks, 1, 1], [1024, 1, 1], 0)?;
        super::cuda_driver::cuda_synchronize(self.device_index)?;
        let bytes = part.to_host_bytes(blocks as usize * 8)?;
        Ok(Some(bytes.chunks_exact(8).map(|c| f64::from_ne_bytes(c.try_into().unwrap())).sum()))
    }
}

// ---- fused attention -------------------------------------------------------------------------------------------------------

static FUSED_ATTENTION_ALWAYS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Forces the fused attention kernel for every eligible call (it is otherwise used only where it beats the composed path:
/// short queries, as in decoding, or score matrices too large to materialize). For tests and experiments.
pub fn set_fused_attention_always(on: bool) {
    FUSED_ATTENTION_ALWAYS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn fused_attention_always() -> bool {
    FUSED_ATTENTION_ALWAYS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Shape of a fused attention call: `bh` = batch * heads flattened, `h` = heads (for the mask's batch / head strides).
#[derive(Clone, Copy, Debug)]
pub struct AttnDims {
    pub bh: usize,
    pub h: usize,
    pub l: usize,
    pub s: usize,
    pub d: usize,
    pub dv: usize,
}

impl CudaStorage {
    /// `softmax(q k^T * scale + mask) v` without materializing the scores (inference: no gradient is recorded). `mask` is an
    /// additive tensor with its (batch, head, query, key) element strides. `q`, `k`, `v` must be contiguous.
    #[allow(clippy::too_many_arguments)]
    /// `lse` (a device pointer to `bh * l` compute-type values, or 0) receives each row's log-sum-exp for the backward pass.
    pub fn attention_gpu(&self, q_off: usize, k: &Self, k_off: usize, v: &Self, v_off: usize, mask: Option<(&Self, usize, [i64; 4])>, causal: bool, scale: f64, dims: AttnDims, lse: u64) -> Result<Option<Self>> {
        if !self.float_module_ready() || k.dtype != self.dtype || v.dtype != self.dtype || dims.d > 128 || dims.dv > 128 || dims.bh > 65535 {
            return Ok(None);
        }
        if let Some((m, _, _)) = mask
            && m.dtype != self.dtype
        {
            return Ok(None);
        }
        let esize = self.dtype.size_in_bytes();
        let out = Self::alloc(self.device_index, self.dtype, dims.bh * dims.l * dims.dv)?;
        if out.len == 0 {
            return Ok(Some(out));
        }
        let (mptr, strides) = match mask {
            Some((m, off, st)) => (m.buf.device_ptr() + (off * esize) as u64, st),
            None => (0, [0; 4]),
        };
        let f = cuda_rt::func(self.device_index, "attention.cu", dtype_code(self.dtype).unwrap(), "attn_fwd")?;
        Args::new()
            .ptr(self.buf.device_ptr() + (q_off * esize) as u64)
            .ptr(k.buf.device_ptr() + (k_off * esize) as u64)
            .ptr(v.buf.device_ptr() + (v_off * esize) as u64)
            .ptr(out.buf.device_ptr())
            .ptr(mptr)
            .u64(dims.l as u64)
            .u64(dims.s as u64)
            .i32(dims.d as i32)
            .i32(dims.dv as i32)
            .f64(scale)
            .i32(i32::from(causal))
            .u64(dims.h as u64)
            .i64(strides[0])
            .i64(strides[1])
            .i64(strides[2])
            .i64(strides[3])
            .ptr(lse)
            .launch(&f, [dims.l.div_ceil(4) as u32, dims.bh as u32, 1], [128, 1, 1], 0)?;
        Ok(Some(out))
    }

    /// Gradients `(dq, dk, dv)` of [`Self::attention_gpu`] for F32 from the saved output `o`, the output gradient `g` (all
    /// contiguous, offsets in elements) and the forward log-sum-exp `lse` (`bh * l` F32). No `[l, s]` buffer is allocated.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_bwd_gpu(q: (&Self, usize), k: (&Self, usize), v: (&Self, usize), o: (&Self, usize), g: (&Self, usize), lse: &Self, mask: Option<(&Self, usize, [i64; 4])>, causal: bool, scale: f64, dims: AttnDims) -> Result<Option<(Self, Self, Self)>> {
        let me = q.0;
        if me.dtype != DType::F32 || !me.float_module_ready() || dims.d > 128 || dims.dv > 128 || dims.bh > 65535 {
            return Ok(None);
        }
        let dev = me.device_index;
        let at = |(t, off): (&Self, usize)| t.buf.device_ptr() + (off * 4) as u64;
        let dq = Self::alloc(dev, DType::F32, dims.bh * dims.l * dims.d)?;
        let dk = Self::alloc(dev, DType::F32, dims.bh * dims.s * dims.d)?;
        let dv = Self::alloc(dev, DType::F32, dims.bh * dims.s * dims.dv)?;
        if dims.bh * dims.l == 0 || dims.s == 0 {
            for t in [&dq, &dk, &dv] {
                t.buf.memset(0)?;
            }
            return Ok(Some((dq, dk, dv)));
        }
        let drow = Self::alloc(dev, DType::F32, dims.bh * dims.l)?;
        let (mptr, st) = match mask {
            Some((m, off, st)) => (m.buf.device_ptr() + (off * 4) as u64, st),
            None => (0, [0; 4]),
        };
        let tail = |a: Args| {
            a.u64(dims.l as u64)
                .u64(dims.s as u64)
                .i32(dims.d as i32)
                .i32(dims.dv as i32)
                .f64(scale)
                .i32(i32::from(causal))
                .u64(dims.h as u64)
                .i64(st[0])
                .i64(st[1])
                .i64(st[2])
                .i64(st[3])
        };
        let f = cuda_rt::func(dev, "attention.cu", dtype_code(DType::F32).unwrap(), "attn_bwd_dq")?;
        tail(Args::new().ptr(at(q)).ptr(at(k)).ptr(at(v)).ptr(at(o)).ptr(at(g)).ptr(lse.buf.device_ptr()).ptr(mptr).ptr(dq.buf.device_ptr()).ptr(drow.buf.device_ptr()))
            .launch(&f, [dims.l.div_ceil(4) as u32, dims.bh as u32, 1], [128, 1, 1], 0)?;
        let f = cuda_rt::func(dev, "attention.cu", dtype_code(DType::F32).unwrap(), "attn_bwd_dkv")?;
        tail(Args::new().ptr(at(q)).ptr(at(k)).ptr(at(v)).ptr(at(g)).ptr(lse.buf.device_ptr()).ptr(drow.buf.device_ptr()).ptr(mptr).ptr(dk.buf.device_ptr()).ptr(dv.buf.device_ptr()))
            .launch(&f, [dims.s.div_ceil(4) as u32, dims.bh as u32, 1], [128, 1, 1], 0)?;
        Ok(Some((dq, dk, dv)))
    }
}

impl super::Storage {
    /// Fused attention over CUDA storages (see [`CudaStorage::attention_gpu`]); `None` when it does not apply.
    #[allow(clippy::too_many_arguments)]
    /// With `lse` (an F32 CUDA storage of `bh * l`), each row's log-sum-exp is written there too (training).
    pub(crate) fn cuda_attention(q: (&super::Storage, &Layout), k: (&super::Storage, &Layout), v: (&super::Storage, &Layout), mask: Option<(&super::Storage, &Layout)>, causal: bool, scale: f64, dims: AttnDims, lse: Option<&super::Storage>) -> Result<Option<super::Storage>> {
        let (gq, gk, gv) = (q.0.read(), k.0.read(), v.0.read());
        let (super::StorageInner::Cuda(cq), super::StorageInner::Cuda(ck), super::StorageInner::Cuda(cv)) = (&*gq, &*gk, &*gv) else { return Ok(None) };
        if !(q.1.is_contiguous() && k.1.is_contiguous() && v.1.is_contiguous()) {
            return Ok(None);
        }
        let gm = mask.map(|m| m.0.read());
        let mask_arg = match (&gm, mask) {
            (Some(g), Some((_, ml))) => match &**g {
                super::StorageInner::Cuda(cm) => {
                    let st = ml.strides();
                    // the mask has been broadcast to [lead.., L, S]: its trailing strides are the query / key strides
                    let n = st.len();
                    let (m_l, m_s) = (st[n - 2] as i64, st[n - 1] as i64);
                    let (m_b, m_h) = match n {
                        2 => (0, 0),
                        3 => (st[0] as i64, 0),
                        _ => (st[0] as i64, st[1] as i64),
                    };
                    Some((cm, ml.offset(), [m_b, m_h, m_l, m_s]))
                }
                _ => return Ok(None),
            },
            _ => None,
        };
        let gl = lse.map(|l| l.read());
        let lse_ptr = match gl.as_deref() {
            Some(super::StorageInner::Cuda(cl)) => cl.buf.device_ptr(),
            Some(_) => return Ok(None),
            None => 0,
        };
        Ok(cq.attention_gpu(q.1.offset(), ck, k.1.offset(), cv, v.1.offset(), mask_arg, causal, scale, dims, lse_ptr)?.map(super::Storage::new_cuda))
    }

    /// `(dq, dk, dv)` storages for [`CudaStorage::attention_bwd_gpu`]; every operand contiguous, the mask already broadcast to
    /// `[lead.., l, s]` as in the forward call. `None` when it does not apply.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_attention_bwd(q: (&super::Storage, &Layout), k: (&super::Storage, &Layout), v: (&super::Storage, &Layout), o: (&super::Storage, &Layout), g: (&super::Storage, &Layout), lse: &super::Storage, mask: Option<(&super::Storage, &Layout)>, causal: bool, scale: f64, dims: AttnDims) -> Result<Option<(super::Storage, super::Storage, super::Storage)>> {
        let guards = [q.0.read(), k.0.read(), v.0.read(), o.0.read(), g.0.read(), lse.read()];
        let mut cs = Vec::with_capacity(6);
        for gd in &guards {
            let super::StorageInner::Cuda(c) = &**gd else { return Ok(None) };
            cs.push(c);
        }
        if ![q.1, k.1, v.1, o.1, g.1].iter().all(|l| l.is_contiguous()) {
            return Ok(None);
        }
        let gm = mask.map(|m| m.0.read());
        let mask_arg = match (&gm, mask) {
            (Some(gd), Some((_, ml))) => match &**gd {
                super::StorageInner::Cuda(cm) => {
                    let st = ml.strides();
                    let n = st.len();
                    let (m_b, m_h) = match n {
                        2 => (0, 0),
                        3 => (st[0] as i64, 0),
                        _ => (st[0] as i64, st[1] as i64),
                    };
                    Some((cm, ml.offset(), [m_b, m_h, st[n - 2] as i64, st[n - 1] as i64]))
                }
                _ => return Ok(None),
            },
            _ => None,
        };
        let r = CudaStorage::attention_bwd_gpu((cs[0], q.1.offset()), (cs[1], k.1.offset()), (cs[2], v.1.offset()), (cs[3], o.1.offset()), (cs[4], g.1.offset()), cs[5], mask_arg, causal, scale, dims)?;
        Ok(r.map(|(a, b, c)| (super::Storage::new_cuda(a), super::Storage::new_cuda(b), super::Storage::new_cuda(c))))
    }
}

impl super::Storage {
    pub(crate) fn cuda_cat(parts: &[(&super::Storage, &Layout)], ax: usize, out_shape: &[usize]) -> Result<Option<super::Storage>> {
        let guards: Vec<_> = parts.iter().map(|(s, _)| s.read()).collect();
        let mut cuda = Vec::with_capacity(parts.len());
        for (g, (_, l)) in guards.iter().zip(parts) {
            match &**g {
                super::StorageInner::Cuda(c) => cuda.push((c, *l)),
                _ => return Ok(None),
            }
        }
        Ok(CudaStorage::cat_gpu(&cuda, ax, out_shape)?.map(super::Storage::new_cuda))
    }

    pub(crate) fn cuda_view_scatter(&self, src_layout: &Layout, spec: &crate::layout::ViewSpec) -> Result<Option<super::Storage>> {
        match &*self.read() {
            super::StorageInner::Cuda(s) => Ok(s.view_scatter_gpu(src_layout, spec)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_gather(&self, layout: &Layout, ax: usize, idx_shape: &[usize], idx: &[i64]) -> Result<Option<super::Storage>> {
        match &*self.read() {
            super::StorageInner::Cuda(s) => Ok(s.gather_gpu(layout, ax, idx_shape, idx)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cuda_scatter(&self, layout: &Layout, src: &super::Storage, src_layout: &Layout, idx_shape: &[usize], ax: usize, idx: &[i64], add: bool) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*src.read()) {
            (super::StorageInner::Cuda(a), super::StorageInner::Cuda(b)) => Ok(a.scatter_gpu(layout, b, src_layout, idx_shape, ax, idx, add)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }

    pub(crate) fn cuda_index_add(&self, layout: &Layout, src: &super::Storage, src_layout: &Layout, ax: usize, idx: &[i64]) -> Result<Option<super::Storage>> {
        match (&*self.read(), &*src.read()) {
            (super::StorageInner::Cuda(a), super::StorageInner::Cuda(b)) => Ok(a.index_add_gpu(layout, b, src_layout, ax, idx)?.map(super::Storage::new_cuda)),
            _ => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The TF32 switch is global, so the tests that depend on it run one at a time.
    static TF32_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn ready() -> bool {
        let ok = cuda_rt::available();
        if !ok {
            eprintln!("skipped: no CUDA device / NVRTC");
        }
        ok
    }

    /// Element values (as f64) of any host storage, for comparison.
    fn values(c: &CpuStorage) -> Vec<f64> {
        let l = Layout::contiguous(vec![c.len()]);
        c.to_vec::<f64>(&l)
    }

    fn close(what: &str, got: &CpuStorage, want: &CpuStorage, rel: f64) {
        assert_eq!(got.dtype(), want.dtype(), "{what}: dtype");
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in values(got).iter().zip(values(want)).enumerate() {
            let ok = (g.is_nan() && w.is_nan()) || g == &w || (g - w).abs() <= rel * (1.0 + w.abs());
            assert!(ok, "{what}: element {i}: gpu {g} vs cpu {w}");
        }
    }

    fn sample(dtype: DType, n: usize) -> CpuStorage {
        let mut v: Vec<f64> = (0..n).map(|i| (((i as f64) * 0.7311).sin() * 6.0) + if i % 11 == 0 { 0.5 } else { 0.0 }).collect();
        if matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            for (k, special) in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, 1e-30, 88.0, -88.0].into_iter().enumerate() {
                if k * 3 < v.len() {
                    v[k * 3] = special;
                }
            }
        }
        if dtype == DType::U64 {
            // negative floats would convert to values near 2^64, where the CPU narrows through f64 (to 0) while the kernel
            // wraps exactly, like C and torch do; keep the sample below 2^63 so both sides agree
            v.iter_mut().for_each(|x| *x = x.abs());
        }
        let l = Layout::contiguous(vec![n]);
        CpuStorage::from_slice(&v).unwrap().to_dtype(&l, dtype).unwrap()
    }

    const FLOATS: [DType; 4] = [DType::F32, DType::F64, DType::F16, DType::BF16];
    const INTS: [DType; 8] = [DType::I8, DType::I16, DType::I32, DType::I64, DType::U8, DType::U16, DType::U32, DType::U64];

    fn rel_for(dtype: DType) -> f64 {
        match dtype {
            DType::F64 => 1e-12,
            DType::F32 => 3e-6,
            DType::F16 => 2e-3,
            _ => 1.6e-2, // bf16: one unit in the last place is 2^-8
        }
    }

    #[test]
    fn unary_kernels_match_the_cpu_for_every_op_and_float_type() {
        if !ready() {
            return;
        }
        let ops = [
            UnaryOp::Neg,
            UnaryOp::Abs,
            UnaryOp::Sign,
            UnaryOp::Exp,
            UnaryOp::Ln,
            UnaryOp::Sqrt,
            UnaryOp::Rsqrt,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Tanh,
            UnaryOp::Sigmoid,
            UnaryOp::Relu,
            UnaryOp::LeakyRelu(0.1),
            UnaryOp::Gelu,
            UnaryOp::Silu,
            UnaryOp::Elu(1.3),
            UnaryOp::Softplus,
            UnaryOp::Square,
            UnaryOp::Recip,
            UnaryOp::Floor,
            UnaryOp::Ceil,
            UnaryOp::Round,
            UnaryOp::Erf,
            UnaryOp::Powf(2.5),
            UnaryOp::AddScalar(1.5),
            UnaryOp::MulScalar(-2.0),
            UnaryOp::Clamp(-1.0, 2.0),
            UnaryOp::LogicalNot,
        ];
        let layout = Layout::contiguous(vec![4099]);
        for dtype in FLOATS {
            let cpu = sample(dtype, 4099);
            let gpu = CudaStorage::from_cpu_any(&cpu, 0).unwrap();
            for op in ops {
                let want = cpu.unary_op(&layout, op).unwrap();
                let got = gpu.unary_gpu(&layout, op).unwrap().unwrap().to_cpu_any().unwrap();
                let slack = if matches!(op, UnaryOp::Gelu | UnaryOp::Erf | UnaryOp::Tanh | UnaryOp::Powf(_)) { 4.0 } else { 1.0 };
                close(&format!("{dtype:?} {op:?}"), &got, &want, rel_for(dtype) * slack);
            }
        }
    }

    #[test]
    fn integer_unary_kernels_follow_the_cpu_f64_round_trip() {
        if !ready() {
            return;
        }
        let layout = Layout::contiguous(vec![1000]);
        for dtype in INTS {
            let cpu = sample(dtype, 1000);
            let gpu = CudaStorage::from_cpu_any(&cpu, 0).unwrap();
            for op in [UnaryOp::Neg, UnaryOp::Abs, UnaryOp::Sign, UnaryOp::Square, UnaryOp::Relu, UnaryOp::AddScalar(3.0), UnaryOp::MulScalar(-2.0), UnaryOp::Clamp(-2.0, 5.0), UnaryOp::Sqrt, UnaryOp::LogicalNot] {
                let want = cpu.unary_op(&layout, op).unwrap();
                let got = gpu.unary_gpu(&layout, op).unwrap().unwrap().to_cpu_any().unwrap();
                close(&format!("{dtype:?} {op:?}"), &got, &want, 0.0);
            }
        }
    }

    #[test]
    fn binary_and_comparison_kernels_match_the_cpu_on_broadcast_and_strided_views() {
        if !ready() {
            return;
        }
        // lhs: a transposed [37, 53] view; rhs: a row broadcast to the same shape
        let (r, c) = (37usize, 53usize);
        let lhs_l = Layout::contiguous(vec![c, r]).transpose(0, 1).unwrap(); // shape [r, c], strides [1, r]
        let rhs_l = Layout::contiguous(vec![1, c]).broadcast_to(&[r, c]).unwrap(); // stride 0 on rows
        for dtype in FLOATS.into_iter().chain(INTS) {
            let a = sample(dtype, r * c);
            let b = sample(dtype, c);
            let (ga, gb) = (CudaStorage::from_cpu_any(&a, 0).unwrap(), CudaStorage::from_cpu_any(&b, 0).unwrap());
            let float = FLOATS.contains(&dtype);
            let ops: &[BinaryOp] = if float {
                &[BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Maximum, BinaryOp::Minimum, BinaryOp::Pow, BinaryOp::Fmod, BinaryOp::Remainder, BinaryOp::FloorDiv]
            } else {
                &[BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Maximum, BinaryOp::Minimum]
            };
            for &op in ops {
                let want = a.binary_op(&lhs_l, &b, &rhs_l, op).unwrap();
                let got = ga.binary_gpu(&lhs_l, &gb, &rhs_l, op).unwrap().unwrap().to_cpu_any().unwrap();
                close(&format!("{dtype:?} {op:?}"), &got, &want, rel_for(dtype) * if op == BinaryOp::Pow { 8.0 } else { 1.0 });
            }
            for op in [CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge] {
                let want = a.cmp_op(&lhs_l, &b, &rhs_l, op).unwrap();
                let got = ga.cmp_gpu(&lhs_l, &gb, &rhs_l, op).unwrap().unwrap().to_cpu_any().unwrap();
                close(&format!("{dtype:?} {op:?}"), &got, &want, 0.0);
            }
            // integer division must stay on the CPU path: it reports division by zero as an error
            if !float {
                assert!(ga.binary_gpu(&lhs_l, &gb, &rhs_l, BinaryOp::Div).unwrap().is_none());
            }
        }
    }

    #[test]
    fn select_cast_contiguous_and_index_select_match_the_cpu() {
        if !ready() {
            return;
        }
        let n = 997usize;
        let layout = Layout::contiguous(vec![n]);
        // where
        let cond = CpuStorage::Bool((0..n).map(|i| i % 3 != 0).collect());
        let a = sample(DType::F32, n);
        let b = a.unary_op(&layout, UnaryOp::Neg).unwrap();
        let want = CpuStorage::where_cond(&cond, &layout, &a, &layout, &b, &layout).unwrap();
        let (gc, ga, gb) = (CudaStorage::from_cpu_any(&cond, 0).unwrap(), CudaStorage::from_cpu_any(&a, 0).unwrap(), CudaStorage::from_cpu_any(&b, 0).unwrap());
        let got = CudaStorage::where_gpu(&gc, &layout, &ga, &layout, &gb, &layout).unwrap().unwrap().to_cpu_any().unwrap();
        close("where", &got, &want, 0.0);

        // every dtype pair of cast, edge values included (NaN, infinities, out-of-range)
        let all: Vec<DType> = FLOATS.into_iter().chain(INTS).chain([DType::Bool]).collect();
        for &src in &all {
            let cpu = sample(src, 311);
            let gpu = CudaStorage::from_cpu_any(&cpu, 0).unwrap();
            let l = Layout::contiguous(vec![311]);
            for &dst in &all {
                let want = cpu.to_dtype(&l, dst).unwrap();
                let got = gpu.cast_gpu(&l, dst).unwrap().unwrap().to_cpu_any().unwrap();
                close(&format!("cast {src:?} -> {dst:?}"), &got, &want, if dst == DType::BF16 || dst == DType::F16 { 2e-2 } else { 1e-6 });
            }
        }

        // contiguous copies of a transposed, a sliced and a flipped view
        let data = sample(DType::I32, 6 * 7 * 5);
        let gd = CudaStorage::from_cpu_any(&data, 0).unwrap();
        let base = Layout::contiguous(vec![6, 7, 5]);
        let views = [base.permute(&[2, 0, 1]).unwrap(), base.narrow(1, 2, 4).unwrap(), base.flip(&[0, 2]).unwrap()];
        for (i, v) in views.iter().enumerate() {
            let want = data.to_contiguous(v).unwrap();
            let got = gd.contiguous_gpu(v).unwrap().unwrap().to_cpu_any().unwrap();
            close(&format!("contiguous view {i}"), &got, &want, 0.0);
        }

        // index_select with repeated and negative indices, on every axis
        for (ax, dim) in [(0usize, 6i64), (1, 7), (2, 5)] {
            let idx: Vec<i64> = vec![3, -1, 0, 3, dim - 1];
            let want = data.index_select_vals(&base, ax, &idx).unwrap();
            let got = gd.index_select_gpu(&base, ax, &idx).unwrap().unwrap().to_cpu_any().unwrap();
            close(&format!("index_select axis {ax}"), &got, &want, 0.0);
        }
        assert!(gd.index_select_gpu(&base, 0, &[6]).is_err());
    }

    /// Finite values only (no NaN / infinity), so products stay meaningful.
    fn plain(dtype: DType, n: usize, seed: f64) -> CpuStorage {
        let v: Vec<f64> = (0..n).map(|i| ((i as f64 + seed) * 0.37).sin() * 1.7 + 0.05 * ((i % 7) as f64 - 3.0)).collect();
        CpuStorage::from_slice(&v).unwrap().to_dtype(&Layout::contiguous(vec![n]), dtype).unwrap()
    }

    #[test]
    fn device_random_numbers_are_reproducible_uniform_and_normal() {
        if !ready() {
            return;
        }
        // Philox4x32-10 known answer (Random123 kat_vectors): counter 0 and key 0 give 0x6627e8d5 as the first word
        let first = CudaStorage::rand_gpu(0, DType::F32, 8, false, 0, 0.0, 1.0).unwrap().unwrap().to_cpu_any().unwrap();
        assert_eq!(values(&first)[0], f64::from(0x6627e8u32) / 16_777_216.0);
        let n = 400_000;
        let a = CudaStorage::rand_gpu(0, DType::F32, n, false, 7, -1.0, 3.0).unwrap().unwrap().to_cpu_any().unwrap();
        let b = CudaStorage::rand_gpu(0, DType::F32, n, false, 7, -1.0, 3.0).unwrap().unwrap().to_cpu_any().unwrap();
        let c = CudaStorage::rand_gpu(0, DType::F32, n, false, 8, -1.0, 3.0).unwrap().unwrap().to_cpu_any().unwrap();
        assert_eq!(values(&a), values(&b), "the same key must reproduce the stream");
        assert_ne!(values(&a), values(&c));
        let v = values(&a);
        let mean = v.iter().sum::<f64>() / n as f64;
        assert!((mean - 1.0).abs() < 0.01, "uniform mean {mean}");
        assert!(v.iter().all(|&x| (-1.0..3.0).contains(&x)));
        let mut buckets = [0usize; 20];
        v.iter().for_each(|&x| buckets[(((x + 1.0) / 4.0) * 20.0) as usize] += 1);
        let expect = n as f64 / 20.0;
        let chi2: f64 = buckets.iter().map(|&o| (o as f64 - expect).powi(2) / expect).sum();
        assert!(chi2 < 43.8, "uniformity chi-square {chi2} (19 dof, p = 0.001)");
        let g = values(&CudaStorage::rand_gpu(0, DType::F32, n, true, 9, 0.0, 0.0).unwrap().unwrap().to_cpu_any().unwrap());
        let (m, var) = (g.iter().sum::<f64>() / n as f64, g.iter().map(|x| x * x).sum::<f64>() / n as f64);
        let kurt = g.iter().map(|x| x.powi(4)).sum::<f64>() / n as f64;
        assert!(m.abs() < 0.01 && (var - 1.0).abs() < 0.02 && (kurt - 3.0).abs() < 0.1, "normal moments: mean {m} var {var} kurtosis {kurt}");
        // dropout mask: kept fraction and scale
        let mask = values(&CudaStorage::bernoulli_mask_gpu(0, DType::F32, n, 11, 0.8, 1.25).unwrap().unwrap().to_cpu_any().unwrap());
        let kept = mask.iter().filter(|&&x| x != 0.0).count() as f64 / n as f64;
        assert!((kept - 0.8).abs() < 0.005 && mask.iter().all(|&x| x == 0.0 || x == 1.25));
        // constants made on the device
        let z = CudaStorage::full_gpu(0, DType::F32, 1000, 0.0).unwrap().unwrap().to_cpu_any().unwrap();
        let t = CudaStorage::full_gpu(0, DType::I32, 1000, 7.0).unwrap().unwrap().to_cpu_any().unwrap();
        assert!(values(&z).iter().all(|&x| x == 0.0) && values(&t).iter().all(|&x| x == 7.0));
    }

    #[test]
    fn tf32_matmul_is_opt_in_and_stays_within_tf32_precision() {
        if !ready() {
            return;
        }
        let _serial = TF32_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        let (m, k, n) = (96usize, 200usize, 70usize);
        let (a, b) = (plain(DType::F32, m * k, 1.0), plain(DType::F32, k * n, 2.0));
        let (ga, gb) = (CudaStorage::from_cpu_any(&a, 0).unwrap(), CudaStorage::from_cpu_any(&b, 0).unwrap());
        let (la, lb) = (Layout::contiguous(vec![m, k]), Layout::contiguous(vec![k, n]));
        let (want, _) = a.matmul(&la, &b, &lb).unwrap();
        set_allow_tf32(false);
        let (exact, _) = ga.matmul_gpu(&la, &gb, &lb).unwrap().unwrap();
        close("f32 (tf32 off)", &exact.to_cpu_any().unwrap(), &want, 1e-4);
        set_allow_tf32(true);
        let (fast, _) = ga.matmul_gpu(&la, &gb, &lb).unwrap().unwrap();
        set_allow_tf32(false);
        let (g, w) = (values(&fast.to_cpu_any().unwrap()), values(&want));
        let scale = w.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let worst = g.iter().zip(&w).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(worst <= 4e-3 * scale, "tf32 error {worst} on values up to {scale}");
        assert!(worst > 1e-6 * scale, "the TF32 kernel should be visibly less precise than f32 (it was not used?)");
    }

    #[test]
    fn softmax_and_layer_norm_kernels_match_the_cpu() {
        if !ready() {
            return;
        }
        for dtype in [DType::F32, DType::F64, DType::F16] {
            let (rows, cols) = (37usize, 301usize);
            let l = Layout::contiguous(vec![rows, cols]);
            let mut cpu = plain(dtype, rows * cols, 3.0);
            let gpu = CudaStorage::from_cpu_any(&cpu, 0).unwrap();
            let tol = match dtype {
                DType::F64 => 1e-12,
                DType::F32 => 3e-6,
                _ => 5e-3,
            };
            // softmax / log-softmax against a straightforward f64 reference
            let vals = values(&cpu);
            for log in [false, true] {
                let (got, flagged) = gpu.softmax_gpu(&l, log, true).unwrap().unwrap();
                assert!(!flagged);
                let mut want = vec![0.0f64; rows * cols];
                for r in 0..rows {
                    let row = &vals[r * cols..(r + 1) * cols];
                    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let s: f64 = row.iter().map(|x| (x - m).exp()).sum();
                    for c in 0..cols {
                        want[r * cols + c] = if log { (row[c] - m) - s.ln() } else { (row[c] - m).exp() / s };
                    }
                }
                let want = CpuStorage::from_slice(&want).unwrap().to_dtype(&Layout::contiguous(vec![rows * cols]), dtype).unwrap();
                close(&format!("{dtype:?} softmax log={log}"), &got.to_cpu_any().unwrap(), &want, tol);
            }
            // layer norm with and without affine parameters
            let w = plain(dtype, cols, 5.0);
            let b = plain(dtype, cols, 6.0);
            let (gw, gb) = (CudaStorage::from_cpu_any(&w, 0).unwrap(), CudaStorage::from_cpu_any(&b, 0).unwrap());
            for (use_w, use_b) in [(true, true), (true, false), (false, true), (false, false)] {
                let (y, xhat, rstd) = gpu.layer_norm_gpu(&l, use_w.then_some(&gw), use_b.then_some(&gb), 1e-5, true).unwrap().unwrap();
                let mut want = vec![0.0f64; rows * cols];
                let mut want_hat = vec![0.0f64; rows * cols];
                let (wv, bv) = (values(&w), values(&b));
                for r in 0..rows {
                    let row = &vals[r * cols..(r + 1) * cols];
                    let mean = row.iter().sum::<f64>() / cols as f64;
                    let var = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / cols as f64;
                    let rs = 1.0 / (var + 1e-5).sqrt();
                    for c in 0..cols {
                        let h = (row[c] - mean) * rs;
                        want_hat[r * cols + c] = h;
                        want[r * cols + c] = h * if use_w { wv[c] } else { 1.0 } + if use_b { bv[c] } else { 0.0 };
                    }
                }
                let to = |v: Vec<f64>| CpuStorage::from_slice(&v).unwrap().to_dtype(&Layout::contiguous(vec![rows * cols]), dtype).unwrap();
                close(&format!("{dtype:?} layer_norm w={use_w} b={use_b}"), &y.to_cpu_any().unwrap(), &to(want), tol * 4.0);
                close(&format!("{dtype:?} xhat"), &xhat.unwrap().to_cpu_any().unwrap(), &to(want_hat), tol * 4.0);
                assert_eq!(rstd.unwrap().len, rows);
            }
            // +inf rows become one-hot, a NaN row stays NaN, and the finite-maximum flag reports them
            if dtype == DType::F32 {
                let mut v = values(&cpu);
                v[5] = f64::INFINITY;
                v[cols + 7] = f64::NAN;
                cpu = CpuStorage::from_slice(&v).unwrap().to_dtype(&Layout::contiguous(vec![rows * cols]), dtype).unwrap();
                let g = CudaStorage::from_cpu_any(&cpu, 0).unwrap();
                let (out, flagged) = g.softmax_gpu(&l, false, true).unwrap().unwrap();
                assert!(flagged);
                let o = values(&out.to_cpu_any().unwrap());
                assert_eq!(o[5], 1.0);
                assert_eq!(o[6], 0.0);
                assert!(o[cols..2 * cols].iter().all(|x| x.is_nan()));
            }
        }
    }

    #[test]
    fn matmul_kernels_match_the_cpu_for_shapes_strides_batches_and_types() {
        if !ready() {
            return;
        }
        let _serial = TF32_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        // (lhs shape, rhs shape, transpose lhs?, transpose rhs?)
        let cases: [(&[usize], &[usize], bool, bool); 9] = [
            (&[37, 45], &[45, 29], false, false),
            (&[300, 257], &[257, 190], false, false),
            (&[130, 70], &[70, 65], false, false),
            (&[45, 37], &[45, 29], true, false),  // lhs is a transposed view
            (&[37, 45], &[29, 45], false, true),  // rhs is a transposed view
            (&[3, 1, 17, 20], &[1, 4, 20, 13], false, false), // broadcast batch dims
            (&[5, 64, 80], &[80, 72], false, false), // rhs without a batch dimension
            (&[1, 33], &[33, 1], false, false),
            (&[2, 3, 70, 66], &[2, 3, 66, 90], false, false),
        ];
        for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
            for (ls, rs, tl, tr) in cases {
                let (a, b) = (plain(dtype, ls.iter().product(), 1.0), plain(dtype, rs.iter().product(), 2.0));
                let (ga, gb) = (CudaStorage::from_cpu_any(&a, 0).unwrap(), CudaStorage::from_cpu_any(&b, 0).unwrap());
                let mk = |shape: &[usize], t: bool| {
                    let l = Layout::contiguous(shape.to_vec());
                    if t { l.transpose(-1, -2).unwrap() } else { l }
                };
                let (la, lb) = (mk(ls, tl), mk(rs, tr));
                let (want, wshape) = a.matmul(&la, &b, &lb).unwrap();
                let (got, gshape) = ga.matmul_gpu(&la, &gb, &lb).unwrap().unwrap();
                assert_eq!(gshape, wshape, "{dtype:?} {ls:?} x {rs:?}: shape");
                let tol = match dtype {
                    DType::F64 => 1e-10,
                    DType::F32 => 2e-4,
                    DType::F16 => 2e-2,
                    _ => 8e-2,
                };
                close(&format!("{dtype:?} {ls:?} x {rs:?} (t {tl} {tr})"), &got.to_cpu_any().unwrap(), &want, tol);
            }
        }
    }

    #[test]
    fn reductions_match_the_cpu_over_every_axis_set_and_op() {
        if !ready() {
            return;
        }
        use ReduceOp::*;
        // shapes chosen to hit: a single huge row (split reduction), a middle axis, a leading axis, a transposed view
        let cases: [(&[usize], &[&[usize]]); 4] = [
            (&[3, 5, 7], &[&[0], &[1], &[2], &[0, 1], &[1, 2], &[0, 1, 2], &[0, 2]]),
            (&[50_000], &[&[0]]),
            (&[200_000, 3], &[&[0], &[1], &[0, 1]]),
            (&[4, 33, 600], &[&[1], &[2], &[1, 2]]),
        ];
        for dtype in [DType::F32, DType::F64, DType::F16, DType::I64, DType::I32] {
            for (shape, axes_list) in cases {
                let n: usize = shape.iter().product();
                let cpu = sample(dtype, n);
                let gpu = CudaStorage::from_cpu_any(&cpu, 0).unwrap();
                let layout = Layout::contiguous(shape.to_vec());
                for axes in axes_list {
                    let ops: &[ReduceOp] = if matches!(dtype, DType::I64) {
                        &[Sum, Max, Min, Prod, ArgMax, ArgMin]
                    } else if matches!(dtype, DType::I32) {
                        &[Max, Min, ArgMax, ArgMin]
                    } else {
                        &[Sum, Mean, Max, Min, Prod, ArgMax, ArgMin]
                    };
                    for &op in ops {
                        if matches!(op, ArgMax | ArgMin) && axes.len() != 1 {
                            continue;
                        }
                        let want = cpu.reduce_op(&layout, axes, op).unwrap();
                        let got = gpu.reduce_gpu(&layout, axes, op).unwrap().unwrap().to_cpu_any().unwrap();
                        // sums of long rows differ in the last bits (tree vs sequential order); products overflow to inf
                        // identically only if the order matches, so they get a wider tolerance
                        let tol = match (op, dtype) {
                            (Sum | Mean, DType::F16) => 5e-2,
                            (Sum | Mean, DType::F32) => 2e-4,
                            (Prod, _) => 1e-3,
                            (_, DType::F64) => 1e-10,
                            _ => 0.0,
                        };
                        close(&format!("{dtype:?} {op:?} {shape:?} axes {axes:?}"), &got, &want, tol);
                    }
                }
            }
        }
        // a transposed view reduced over its first axis goes through the permute-and-copy path
        let cpu = sample(DType::F32, 40 * 30);
        let gpu = CudaStorage::from_cpu_any(&cpu, 0).unwrap();
        let view = Layout::contiguous(vec![40, 30]).transpose(0, 1).unwrap();
        for op in [Sum, Max, ArgMax] {
            let want = cpu.reduce_op(&view, &[0], op).unwrap();
            let got = gpu.reduce_gpu(&view, &[0], op).unwrap().unwrap().to_cpu_any().unwrap();
            close(&format!("view {op:?}"), &got, &want, 1e-4);
        }
    }
}
