//! Native NVIDIA CUDA PTX Compute Kernels (Cached JIT Modules + Shared-Memory Tiling).
//!
//! Runs directly on NVIDIA GPUs via [`super::cuda_driver`] (`nvcuda.dll` / `libcuda.so.1`)
//! with zero `nvcc` build-time dependency. PTX modules are compiled once per device and cached.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use super::cuda_driver::{CudaBuffer, CudaPtxModule, CudaStream, cuda_synchronize};
use super::{BinaryOp, CpuStorage, ReduceOp, UnaryOp};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::{Layout, broadcast_shapes};

type KernelMap = HashMap<(usize, &'static str), Arc<CudaPtxModule>>;

fn kernel_cache() -> &'static Mutex<KernelMap> {
    static CACHE: OnceLock<Mutex<KernelMap>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_load(device_index: usize, name: &'static str, ptx: &str) -> Result<Arc<CudaPtxModule>> {
    let mut guard = kernel_cache().lock().unwrap_or_else(|p| p.into_inner());
    if let Some(m) = guard.get(&(device_index, name)) {
        return Ok(m.clone());
    }
    let m = CudaPtxModule::load(device_index, ptx)?;
    guard.insert((device_index, name), m.clone());
    Ok(m)
}

/// 16x16 shared-memory tiled SGEMM (`C[M, N] = A[M, K] @ B[K, N]`) in PTX.
const MATMUL_TILED_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry matmul_tiled16_f32(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 m_dim,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .shared .align 4 .f32 tileA[256];
    .shared .align 4 .f32 tileB[256];

    .reg .pred %p<6>;
    .reg .b32 %r<28>;
    .reg .f32 %f<6>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];
    ld.param.u32 %r1, [m_dim];
    ld.param.u32 %r2, [k_dim];
    ld.param.u32 %r3, [n_dim];

    mov.u32 %r4, %tid.x;   // tx
    mov.u32 %r5, %tid.y;   // ty
    mov.u32 %r6, %ctaid.x; // bx
    mov.u32 %r7, %ctaid.y; // by

    shl.b32 %r8, %r6, 4;
    add.u32 %r8, %r8, %r4; // col = bx*16 + tx
    shl.b32 %r9, %r7, 4;
    add.u32 %r9, %r9, %r5; // row = by*16 + ty

    shl.b32 %r10, %r5, 4;
    add.u32 %r10, %r10, %r4; // s_idx = ty*16 + tx
    shl.b32 %r11, %r10, 2;   // s_byte = s_idx * 4
    mov.u32 %r21, tileA;
    mov.u32 %r22, tileB;

    mov.f32 %f1, 0f00000000; // acc = 0.0
    mov.u32 %r12, 0;         // t = 0

TILE_LOOP:
    setp.ge.u32 %p1, %r12, %r2;
    @%p1 bra TILE_DONE;

    // Load tileA[ty][tx] = A[row, t + tx]
    add.u32 %r13, %r12, %r4; // a_col = t + tx
    setp.lt.u32 %p2, %r9, %r1;
    setp.lt.u32 %p3, %r13, %r2;
    and.pred %p2, %p2, %p3;
    mov.f32 %f2, 0f00000000;
    @!%p2 bra SKIP_A;
    mad.lo.u32 %r14, %r9, %r2, %r13;
    mul.wide.u32 %rd4, %r14, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
SKIP_A:
    add.u32 %r23, %r21, %r11;
    st.shared.f32 [%r23], %f2;

    // Load tileB[ty][tx] = B[t + ty, col]
    add.u32 %r15, %r12, %r5; // b_row = t + ty
    setp.lt.u32 %p4, %r15, %r2;
    setp.lt.u32 %p5, %r8, %r3;
    and.pred %p4, %p4, %p5;
    mov.f32 %f3, 0f00000000;
    @!%p4 bra SKIP_B;
    mad.lo.u32 %r16, %r15, %r3, %r8;
    mul.wide.u32 %rd6, %r16, 4;
    add.s64 %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
SKIP_B:
    add.u32 %r24, %r22, %r11;
    st.shared.f32 [%r24], %f3;

    bar.sync 0;

    // Inner K-tile loop: k = 0..16
    mov.u32 %r17, 0;
K_LOOP:
    setp.ge.u32 %p1, %r17, 16;
    @%p1 bra K_DONE;
    shl.b32 %r18, %r5, 4;
    add.u32 %r18, %r18, %r17;
    shl.b32 %r18, %r18, 2;
    add.u32 %r23, %r21, %r18;
    ld.shared.f32 %f4, [%r23];

    shl.b32 %r19, %r17, 4;
    add.u32 %r19, %r19, %r4;
    shl.b32 %r19, %r19, 2;
    add.u32 %r24, %r22, %r19;
    ld.shared.f32 %f5, [%r24];

    fma.rn.f32 %f1, %f4, %f5, %f1;
    add.u32 %r17, %r17, 1;
    bra K_LOOP;
K_DONE:
    bar.sync 0;
    add.u32 %r12, %r12, 16;
    bra TILE_LOOP;

TILE_DONE:
    setp.lt.u32 %p2, %r9, %r1;
    setp.lt.u32 %p3, %r8, %r3;
    and.pred %p2, %p2, %p3;
    @!%p2 bra EXIT;
    mad.lo.u32 %r20, %r9, %r3, %r8;
    mul.wide.u32 %rd8, %r20, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;
EXIT:
    ret;
}
"#;

/// Block-per-row fused Softmax (`[rows, cols]`, 256 threads/row with shared-memory reduction).
const SOFTMAX_ROW_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry softmax_row_f32(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u32 cols
)
{
    .shared .align 4 .f32 smem[256];
    .reg .pred %p<4>;
    .reg .b32 %r<14>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [x_ptr];
    ld.param.u64 %rd2, [y_ptr];
    ld.param.u32 %r1, [cols];

    mov.u32 %r2, %ctaid.x; // row
    mov.u32 %r3, %tid.x;   // tid (0..255)
    mul.lo.u32 %r4, %r2, %r1; // row_off

    // 1. Thread-local max over strided elements
    mov.f32 %f1, 0fFF800000; // -inf
    mov.u32 %r5, %r3;
MAX_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra MAX_DONE;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    max.f32 %f1, %f1, %f2;
    add.u32 %r5, %r5, 256;
    bra MAX_LOOP;
MAX_DONE:
    mov.u32 %r11, smem;
    shl.b32 %r7, %r3, 2;
    add.u32 %r12, %r11, %r7;
    st.shared.f32 [%r12], %f1;
    bar.sync 0;

    // Tree reduction for max (stride 128 -> 1)
    mov.u32 %r8, 128;
RED_MAX:
    setp.eq.u32 %p1, %r8, 0;
    @%p1 bra RED_MAX_DONE;
    setp.lt.u32 %p2, %r3, %r8;
    @!%p2 bra RED_MAX_NEXT;
    add.u32 %r9, %r3, %r8;
    shl.b32 %r10, %r9, 2;
    add.u32 %r13, %r11, %r10;
    ld.shared.f32 %f2, [%r12];
    ld.shared.f32 %f3, [%r13];
    max.f32 %f2, %f2, %f3;
    st.shared.f32 [%r12], %f2;
RED_MAX_NEXT:
    bar.sync 0;
    shr.b32 %r8, %r8, 1;
    bra RED_MAX;
RED_MAX_DONE:
    ld.shared.f32 %f4, [smem]; // row_max

    // 2. Compute exp(x - row_max), store in y, accumulate local sum
    mov.f32 %f5, 0f00000000; // local_sum
    mov.f32 %f6, 0f3FB8AA3B; // log2(e)
    mov.u32 %r5, %r3;
EXP_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra EXP_DONE;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f2, [%rd4];
    sub.f32 %f2, %f2, %f4;
    mul.f32 %f2, %f2, %f6;
    ex2.approx.f32 %f2, %f2;
    st.global.f32 [%rd5], %f2;
    add.f32 %f5, %f5, %f2;
    add.u32 %r5, %r5, 256;
    bra EXP_LOOP;
EXP_DONE:
    st.shared.f32 [%r12], %f5;
    bar.sync 0;

    // Tree reduction for sum
    mov.u32 %r8, 128;
RED_SUM:
    setp.eq.u32 %p1, %r8, 0;
    @%p1 bra RED_SUM_DONE;
    setp.lt.u32 %p2, %r3, %r8;
    @!%p2 bra RED_SUM_NEXT;
    add.u32 %r9, %r3, %r8;
    shl.b32 %r10, %r9, 2;
    add.u32 %r13, %r11, %r10;
    ld.shared.f32 %f2, [%r12];
    ld.shared.f32 %f3, [%r13];
    add.f32 %f2, %f2, %f3;
    st.shared.f32 [%r12], %f2;
RED_SUM_NEXT:
    bar.sync 0;
    shr.b32 %r8, %r8, 1;
    bra RED_SUM;
RED_SUM_DONE:
    ld.shared.f32 %f6, [smem];
    mov.f32 %f5, 0f3F800000;
    div.rn.f32 %f7, %f5, %f6; // inv_sum = 1.0 / sum

    // 3. Normalize y *= inv_sum
    mov.u32 %r5, %r3;
NORM_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra EXIT;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f2, [%rd5];
    mul.f32 %f2, %f2, %f7;
    st.global.f32 [%rd5], %f2;
    add.u32 %r5, %r5, 256;
    bra NORM_LOOP;
EXIT:
    ret;
}
"#;

/// Elementwise binary kernel (`op`: 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Max, 5=Min).
const BINARY_F32_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry binary_f32(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n,
    .param .u32 op
)
{
    .reg .pred %p<7>;
    .reg .b32 %r<7>;
    .reg .f32 %f<4>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [op];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r6, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];

    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra DO_ADD;
    setp.eq.u32 %p2, %r2, 1;
    @%p2 bra DO_SUB;
    setp.eq.u32 %p2, %r2, 2;
    @%p2 bra DO_MUL;
    setp.eq.u32 %p2, %r2, 3;
    @%p2 bra DO_DIV;
    setp.eq.u32 %p2, %r2, 4;
    @%p2 bra DO_MAX;
    min.f32 %f3, %f1, %f2;
    bra STORE;
DO_ADD:
    add.f32 %f3, %f1, %f2;
    bra STORE;
DO_SUB:
    sub.f32 %f3, %f1, %f2;
    bra STORE;
DO_MUL:
    mul.f32 %f3, %f1, %f2;
    bra STORE;
DO_DIV:
    div.rn.f32 %f3, %f1, %f2;
    bra STORE;
DO_MAX:
    max.f32 %f3, %f1, %f2;
STORE:
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
"#;

/// Elementwise unary kernel (`op`: 0=Neg, 1=Abs, 2=Exp, 3=Ln, 4=Sqrt, 5=Rsqrt, 6=Relu, 7=Silu, 8=Sigmoid, 9=Square).
const UNARY_F32_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry unary_f32(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u32 n,
    .param .u32 op
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<7>;
    .reg .f32 %f<6>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [x_ptr];
    ld.param.u64 %rd2, [y_ptr];
    ld.param.u32 %r1, [n];
    ld.param.u32 %r2, [op];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r6, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r6, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd3, %r6, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];

    setp.eq.u32 %p2, %r2, 0;
    @%p2 bra DO_NEG;
    setp.eq.u32 %p2, %r2, 1;
    @%p2 bra DO_ABS;
    setp.eq.u32 %p2, %r2, 2;
    @%p2 bra DO_EXP;
    setp.eq.u32 %p2, %r2, 3;
    @%p2 bra DO_LN;
    setp.eq.u32 %p2, %r2, 4;
    @%p2 bra DO_SQRT;
    setp.eq.u32 %p2, %r2, 5;
    @%p2 bra DO_RSQRT;
    setp.eq.u32 %p2, %r2, 6;
    @%p2 bra DO_RELU;
    setp.eq.u32 %p2, %r2, 7;
    @%p2 bra DO_SILU;
    setp.eq.u32 %p2, %r2, 8;
    @%p2 bra DO_SIGMOID;
    mul.f32 %f2, %f1, %f1;
    bra STORE;
DO_NEG:
    neg.f32 %f2, %f1;
    bra STORE;
DO_ABS:
    abs.f32 %f2, %f1;
    bra STORE;
DO_EXP:
    mov.f32 %f3, 0f3FB8AA3B;
    mul.f32 %f4, %f1, %f3;
    ex2.approx.f32 %f2, %f4;
    bra STORE;
DO_LN:
    mov.f32 %f3, 0f3F317218; // ln(2)
    lg2.approx.f32 %f4, %f1;
    mul.f32 %f2, %f4, %f3;
    bra STORE;
DO_SQRT:
    sqrt.approx.f32 %f2, %f1;
    bra STORE;
DO_RSQRT:
    rsqrt.approx.f32 %f2, %f1;
    bra STORE;
DO_RELU:
    mov.f32 %f3, 0f00000000;
    max.f32 %f2, %f1, %f3;
    bra STORE;
DO_SILU:
    neg.f32 %f3, %f1;
    mov.f32 %f4, 0f3FB8AA3B;
    mul.f32 %f3, %f3, %f4;
    ex2.approx.f32 %f3, %f3;
    mov.f32 %f4, 0f3F800000;
    add.f32 %f3, %f3, %f4;
    div.rn.f32 %f2, %f1, %f3;
    bra STORE;
DO_SIGMOID:
    neg.f32 %f3, %f1;
    mov.f32 %f4, 0f3FB8AA3B;
    mul.f32 %f3, %f3, %f4;
    ex2.approx.f32 %f3, %f3;
    mov.f32 %f4, 0f3F800000;
    add.f32 %f3, %f3, %f4;
    div.rn.f32 %f2, %f4, %f3;
STORE:
    st.global.f32 [%rd5], %f2;
DONE:
    ret;
}
"#;

/// Device-resident CUDA tensor storage backed by [`CudaBuffer`].
#[derive(Clone, Debug)]
pub struct CudaStorage {
    pub device_index: usize,
    pub dtype: DType,
    pub len: usize,
    pub buf: Arc<CudaBuffer>,
}

impl CudaStorage {
    pub fn from_f32_slice(device_index: usize, data: &[f32]) -> Result<Self> {
        let buf = Arc::new(CudaBuffer::from_host_f32(device_index, data)?);
        Ok(Self {
            device_index,
            dtype: DType::F32,
            len: data.len(),
            buf,
        })
    }

    pub fn alloc_f32(device_index: usize, len: usize) -> Result<Self> {
        let buf = Arc::new(CudaBuffer::alloc(device_index, len * 4)?);
        Ok(Self {
            device_index,
            dtype: DType::F32,
            len,
            buf,
        })
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(Error::Msg(format!("CudaStorage::to_f32_vec: unsupported dtype {:?}", self.dtype)));
        }
        cuda_synchronize(self.device_index)?;
        self.buf.to_host_f32(self.len)
    }

    pub fn device_ptr(&self) -> u64 {
        self.buf.device_ptr()
    }

    pub fn from_cpu(cpu: &CpuStorage, device_index: usize) -> Result<Self> {
        match cpu {
            CpuStorage::F32(v) => Self::from_f32_slice(device_index, v),
            other => Err(Error::Msg(format!(
                "CudaStorage currently supports F32 tensors, got {:?}",
                other.dtype()
            ))),
        }
    }

    pub fn to_cpu(&self) -> Result<CpuStorage> {
        Ok(CpuStorage::F32(self.to_f32_vec()?))
    }

    pub fn unary_op(&self, layout: &Layout, op: UnaryOp) -> Result<Self> {
        let op_code = match op {
            UnaryOp::Neg => Some(0),
            UnaryOp::Abs => Some(1),
            UnaryOp::Exp => Some(2),
            UnaryOp::Ln => Some(3),
            UnaryOp::Sqrt => Some(4),
            UnaryOp::Rsqrt => Some(5),
            UnaryOp::Relu => Some(6),
            UnaryOp::Silu => Some(7),
            UnaryOp::Sigmoid => Some(8),
            UnaryOp::Square => Some(9),
            _ => None,
        };
        if let Some(code) = op_code
            && let Some(range) = layout.contiguous_range_in(self.len)
            && range.start == 0
            && range.len() == self.len
        {
            let out = Self::alloc_f32(self.device_index, self.len)?;
            launch_unary_f32(self.device_index, &self.buf, &out.buf, self.len, code, None)?;
            return Ok(out);
        }
        let cpu_out = self.to_cpu()?.unary_op(layout, op)?;
        if cpu_out.dtype() == DType::F32 {
            Self::from_cpu(&cpu_out, self.device_index)
        } else {
            Err(Error::Msg("CudaStorage::unary_op produced non-F32 output".into()))
        }
    }

    pub fn binary_op(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout, op: BinaryOp) -> Result<Self> {
        let op_code = match op {
            BinaryOp::Add => Some(0),
            BinaryOp::Sub => Some(1),
            BinaryOp::Mul => Some(2),
            BinaryOp::Div => Some(3),
            BinaryOp::Maximum => Some(4),
            BinaryOp::Minimum => Some(5),
            _ => None,
        };
        if let Some(code) = op_code
            && lhs_l.shape() == rhs_l.shape()
            && let Some(ra) = lhs_l.contiguous_range_in(self.len)
            && let Some(rb) = rhs_l.contiguous_range_in(rhs.len)
            && ra.start == 0
            && rb.start == 0
            && ra.len() == self.len
            && rb.len() == rhs.len
        {
            let out = Self::alloc_f32(self.device_index, self.len)?;
            launch_binary_f32(self.device_index, &self.buf, &rhs.buf, &out.buf, self.len, code, None)?;
            return Ok(out);
        }
        let cpu_out = self.to_cpu()?.binary_op(lhs_l, &rhs.to_cpu()?, rhs_l, op)?;
        Self::from_cpu(&cpu_out, self.device_index)
    }

    pub fn matmul(&self, lhs_l: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<(Self, Vec<usize>)> {
        let (ar, br) = (lhs_l.ndim(), rhs_l.ndim());
        let (m, k) = (lhs_l.shape()[ar - 2], lhs_l.shape()[ar - 1]);
        let (rk, n) = (rhs_l.shape()[br - 2], rhs_l.shape()[br - 1]);
        if k != rk {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                expected: vec![k],
                got: vec![rk],
            });
        }
        let mut out_shape = broadcast_shapes("matmul", &lhs_l.shape()[..ar - 2], &rhs_l.shape()[..br - 2])?;
        out_shape.push(m);
        out_shape.push(n);

        if ar == 2
            && br == 2
            && lhs_l.is_contiguous()
            && lhs_l.offset() == 0
            && rhs_l.is_contiguous()
            && rhs_l.offset() == 0
        {
            let out = Self::alloc_f32(self.device_index, m * n)?;
            launch_matmul_f32(self.device_index, &self.buf, &rhs.buf, &out.buf, m, k, n)?;
            return Ok((out, out_shape));
        }
        let (cpu_out, shape) = self.to_cpu()?.matmul(lhs_l, &rhs.to_cpu()?, rhs_l)?;
        Ok((Self::from_cpu(&cpu_out, self.device_index)?, shape))
    }

    pub fn softmax_last_axis(&self, layout: &Layout) -> Option<Self> {
        let shape = layout.shape();
        let cols = *shape.last()?;
        if cols == 0 {
            return None;
        }
        let range = layout.contiguous_range_in(self.len)?;
        if range.start != 0 || range.len() != self.len {
            return None;
        }
        let rows = self.len / cols;
        let out = Self::alloc_f32(self.device_index, self.len).ok()?;
        launch_softmax_row_f32(self.device_index, &self.buf, &out.buf, rows, cols).ok()?;
        Some(out)
    }

    pub fn reduce_op(&self, layout: &Layout, axes: &[usize], op: ReduceOp) -> Result<Self> {
        let cpu_out = self.to_cpu()?.reduce_op(layout, axes, op)?;
        Self::from_cpu(&cpu_out, self.device_index)
    }

    pub fn to_contiguous(&self, layout: &Layout) -> Result<Self> {
        let cpu_out = self.to_cpu()?.to_contiguous(layout)?;
        Self::from_cpu(&cpu_out, self.device_index)
    }

    pub fn index_select_vals(&self, layout: &Layout, ax: usize, idx: &[i64]) -> Result<Self> {
        let cpu_out = self.to_cpu()?.index_select_vals(layout, ax, idx)?;
        Self::from_cpu(&cpu_out, self.device_index)
    }
}

/// Launches elementwise binary `f32` op on `CudaBuffer`s (optionally on a `CudaStream` for CUDA Graph capture).
pub fn launch_binary_f32(
    device_index: usize,
    a: &CudaBuffer,
    b: &CudaBuffer,
    c: &CudaBuffer,
    n: usize,
    op_code: u32,
    stream: Option<&CudaStream>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let module = get_or_load(device_index, "binary_f32", BINARY_F32_PTX)?;
    let mut pa = a.device_ptr();
    let mut pb = b.device_ptr();
    let mut pc = c.device_ptr();
    let mut nu = n as u32;
    let mut opu = op_code;
    let mut params: [*mut c_void; 5] = [
        &mut pa as *mut u64 as *mut c_void,
        &mut pb as *mut u64 as *mut c_void,
        &mut pc as *mut u64 as *mut c_void,
        &mut nu as *mut u32 as *mut c_void,
        &mut opu as *mut u32 as *mut c_void,
    ];
    let grid = (n as u32).div_ceil(256);
    module.launch_grid_on("binary_f32", [grid, 1, 1], [256, 1, 1], 0, stream, &mut params)
}

/// Launches elementwise unary `f32` op on `CudaBuffer`s (optionally on a `CudaStream` for CUDA Graph capture).
pub fn launch_unary_f32(
    device_index: usize,
    x: &CudaBuffer,
    y: &CudaBuffer,
    n: usize,
    op_code: u32,
    stream: Option<&CudaStream>,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let module = get_or_load(device_index, "unary_f32", UNARY_F32_PTX)?;
    let mut px = x.device_ptr();
    let mut py = y.device_ptr();
    let mut nu = n as u32;
    let mut opu = op_code;
    let mut params: [*mut c_void; 4] = [
        &mut px as *mut u64 as *mut c_void,
        &mut py as *mut u64 as *mut c_void,
        &mut nu as *mut u32 as *mut c_void,
        &mut opu as *mut u32 as *mut c_void,
    ];
    let grid = (n as u32).div_ceil(256);
    module.launch_grid_on("unary_f32", [grid, 1, 1], [256, 1, 1], 0, stream, &mut params)
}

/// Launches 16x16 shared-memory tiled SGEMM on `CudaBuffer`s: `C[m, n] = A[m, k] @ B[k, n]`.
pub fn launch_matmul_f32(
    device_index: usize,
    a: &CudaBuffer,
    b: &CudaBuffer,
    c: &CudaBuffer,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    let module = get_or_load(device_index, "matmul_tiled16_f32", MATMUL_TILED_PTX)?;
    let mut pa = a.device_ptr();
    let mut pb = b.device_ptr();
    let mut pc = c.device_ptr();
    let mut mu = m as u32;
    let mut ku = k as u32;
    let mut nu = n as u32;
    let mut params: [*mut c_void; 6] = [
        &mut pa as *mut u64 as *mut c_void,
        &mut pb as *mut u64 as *mut c_void,
        &mut pc as *mut u64 as *mut c_void,
        &mut mu as *mut u32 as *mut c_void,
        &mut ku as *mut u32 as *mut c_void,
        &mut nu as *mut u32 as *mut c_void,
    ];
    let gx = (n as u32).div_ceil(16);
    let gy = (m as u32).div_ceil(16);
    module.launch_grid("matmul_tiled16_f32", [gx, gy, 1], [16, 16, 1], 0, &mut params)
}

/// Launches fused block-per-row CUDA Softmax on `[rows, cols]`.
pub fn launch_softmax_row_f32(
    device_index: usize,
    x: &CudaBuffer,
    y: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let module = get_or_load(device_index, "softmax_row_f32", SOFTMAX_ROW_PTX)?;
    let mut px = x.device_ptr();
    let mut py = y.device_ptr();
    let mut cu = cols as u32;
    let mut params: [*mut c_void; 3] = [
        &mut px as *mut u64 as *mut c_void,
        &mut py as *mut u64 as *mut c_void,
        &mut cu as *mut u32 as *mut c_void,
    ];
    module.launch_grid("softmax_row_f32", [rows as u32, 1, 1], [256, 1, 1], 0, &mut params)
}

/// Convenience wrapper: uploads `a` and `b`, runs tiled SGEMM on CUDA, and returns `C`.
pub fn cuda_matmul_f32(device_index: usize, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    let da = CudaBuffer::from_host_f32(device_index, a)?;
    let db = CudaBuffer::from_host_f32(device_index, b)?;
    let dc = CudaBuffer::alloc(device_index, m * n * 4)?;
    launch_matmul_f32(device_index, &da, &db, &dc, m, k, n)?;
    cuda_synchronize(device_index)?;
    dc.to_host_f32(m * n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda_driver::cuda_available;

    #[test]
    fn cuda_tiled_matmul_and_fused_softmax_on_gpu() {
        if !cuda_available() {
            return;
        }
        // 1. Tiled 16x16 shared-memory SGEMM (non-multiple of 16 dimensions to test boundary masks)
        let (m, k, n) = (37, 45, 29);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let gpu_c = cuda_matmul_f32(0, &a, &b, m, k, n).unwrap();
        let mut ref_c = vec![0.0f32; m * n];
        crate::gemm::gemm_f32_strided(&a, 0, k as isize, 1, &b, 0, n as isize, 1, &mut ref_c, m, k, n);
        for (i, (g, r)) in gpu_c.iter().zip(ref_c.iter()).enumerate() {
            assert!((g - r).abs() < 1e-4, "matmul mismatch at {i}: gpu={g} vs cpu={r}");
        }

        // 2. Fused block-per-row Softmax
        let (rows, cols) = (8, 300);
        let x: Vec<f32> = (0..rows * cols).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let dx = CudaBuffer::from_host_f32(0, &x).unwrap();
        let dy = CudaBuffer::alloc(device_index_zero(), rows * cols * 4).unwrap();
        launch_softmax_row_f32(0, &dx, &dy, rows, cols).unwrap();
        cuda_synchronize(0).unwrap();
        let gpu_sm = dy.to_host_f32(rows * cols).unwrap();
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - mx).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for c in 0..cols {
                let want = exps[c] / sum;
                let got = gpu_sm[r * cols + c];
                assert!((got - want).abs() < 1e-4, "softmax ({r},{c}): got={got} vs want={want}");
            }
        }

        // 3. CUDA Graph capture & replay (binary_f32 + unary_f32)
        let stream = CudaStream::new(0).unwrap();
        let ga = CudaBuffer::from_host_f32(0, &[1.0, 2.0, -3.0, 4.0]).unwrap();
        let gb = CudaBuffer::from_host_f32(0, &[2.0, -4.0, 1.0, 0.5]).unwrap();
        let gtmp = CudaBuffer::alloc(0, 16).unwrap();
        let gout = CudaBuffer::alloc(0, 16).unwrap();
        let graph = super::super::cuda_driver::CudaGraph::capture(&stream, |s| {
            launch_binary_f32(0, &ga, &gb, &gtmp, 4, 0, Some(s))?; // a + b = [3, -2, -2, 4.5]
            launch_unary_f32(0, &gtmp, &gout, 4, 6, Some(s)) // relu -> [3, 0, 0, 4.5]
        })
        .unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(gout.to_host_f32(4).unwrap(), vec![3.0, 0.0, 0.0, 4.5]);

        // 4. End-to-end Tensor on Device::Cuda(0) + zero-copy kDLCUDA DLPack export
        let dev = crate::device::Device::Cuda(0);
        let ta = crate::tensor::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let tb = crate::tensor::Tensor::from_slice(&[0.5f32, -1.0, 2.0, 1.0], &[2, 2])
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let tc = ta.add(&tb).unwrap().relu().unwrap().matmul(&tb).unwrap();
        assert_eq!(tc.device(), dev);
        let got = tc.to_vec::<f32>().unwrap();
        // ([1.5, 1.0; 5.0, 5.0]) @ ([0.5, -1.0; 2.0, 1.0]) = [2.75, -0.5; 12.5, 0.0]
        assert_eq!(got, vec![2.75, -0.5, 12.5, 0.0]);

        let dlp = crate::dlpack::to_dlpack(&tc).unwrap();
        unsafe {
            assert_eq!((*dlp).dl_tensor.device.device_type, 2); // kDLCUDA
            assert_eq!((*dlp).dl_tensor.device.device_id, 0);
            assert!(tc.storage().is_dlpack_pinned());
            if let Some(del) = (*dlp).deleter {
                del(dlp);
            }
        }
        assert!(!tc.storage().is_dlpack_pinned());
    }

    fn device_index_zero() -> usize {
        0
    }
}
