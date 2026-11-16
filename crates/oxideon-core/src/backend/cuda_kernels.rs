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
    // Store C[row, col]
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

/// 32x32 shared-memory + 2x2 register-blocked SGEMM (`C[M, N] = A[M, K] @ B[K, N]`) in PTX.
/// Block: [16, 16, 1] = 256 threads; each thread computes 4 outputs `(row0/1, col0/1)` with `BK=16`.
const MATMUL_TILED32_2X2_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry matmul_tiled32_2x2_f32(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 m_dim,
    .param .u32 k_dim,
    .param .u32 n_dim
)
{
    .shared .align 4 .f32 tileA[512]; // [32][16]
    .shared .align 4 .f32 tileB[512]; // [16][32]

    .reg .pred %p<6>;
    .reg .b32 %r<34>;
    .reg .f32 %f<14>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];
    ld.param.u32 %r1, [m_dim];
    ld.param.u32 %r2, [k_dim];
    ld.param.u32 %r3, [n_dim];

    mov.u32 %r4, %tid.x;   // tx (0..15)
    mov.u32 %r5, %tid.y;   // ty (0..15)
    mov.u32 %r6, %ctaid.x; // bx
    mov.u32 %r7, %ctaid.y; // by

    shl.b32 %r8, %r6, 5;
    add.u32 %r8, %r8, %r4;   // col0 = bx*32 + tx
    add.u32 %r9, %r8, 16;    // col1 = col0 + 16
    shl.b32 %r10, %r7, 5;
    add.u32 %r10, %r10, %r5; // row0 = by*32 + ty
    add.u32 %r11, %r10, 16;  // row1 = row0 + 16

    mov.u32 %r21, tileA;
    mov.u32 %r22, tileB;

    // Precompute shared store offsets
    // tileA[ty][tx] -> (ty*16 + tx)*4; tileA[ty+16][tx] -> ((ty+16)*16 + tx)*4
    shl.b32 %r12, %r5, 4;
    add.u32 %r12, %r12, %r4;
    shl.b32 %r13, %r12, 2;      // offA0
    add.u32 %r14, %r13, 1024;   // offA1 = offA0 + 256*4

    // tileB[ty][tx] -> (ty*32 + tx)*4; tileB[ty][tx+16] -> (ty*32 + tx + 16)*4
    shl.b32 %r15, %r5, 5;
    add.u32 %r15, %r15, %r4;
    shl.b32 %r16, %r15, 2;      // offB0
    add.u32 %r17, %r16, 64;     // offB1 = offB0 + 16*4

    mov.f32 %f1, 0f00000000; // c00
    mov.f32 %f2, 0f00000000; // c01
    mov.f32 %f3, 0f00000000; // c10
    mov.f32 %f4, 0f00000000; // c11

    mov.u32 %r18, 0; // t = 0

T32_LOOP:
    setp.ge.u32 %p1, %r18, %r2;
    @%p1 bra T32_DONE;

    add.u32 %r19, %r18, %r4; // a_col = t + tx
    setp.lt.u32 %p3, %r19, %r2;

    // Load A[row0, t + tx] -> tileA[ty][tx]
    setp.lt.u32 %p2, %r10, %r1;
    and.pred %p2, %p2, %p3;
    mov.f32 %f5, 0f00000000;
    @!%p2 bra T32_SA0;
    mad.lo.u32 %r20, %r10, %r2, %r19;
    mul.wide.u32 %rd4, %r20, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f5, [%rd5];
T32_SA0:
    add.u32 %r23, %r21, %r13;
    st.shared.f32 [%r23], %f5;

    // Load A[row1, t + tx] -> tileA[ty+16][tx]
    setp.lt.u32 %p2, %r11, %r1;
    and.pred %p2, %p2, %p3;
    mov.f32 %f6, 0f00000000;
    @!%p2 bra T32_SA1;
    mad.lo.u32 %r20, %r11, %r2, %r19;
    mul.wide.u32 %rd4, %r20, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f6, [%rd5];
T32_SA1:
    add.u32 %r23, %r21, %r14;
    st.shared.f32 [%r23], %f6;

    // Load B[t + ty, col0] -> tileB[ty][tx]
    add.u32 %r24, %r18, %r5; // b_row = t + ty
    setp.lt.u32 %p4, %r24, %r2;

    setp.lt.u32 %p2, %r8, %r3;
    and.pred %p2, %p2, %p4;
    mov.f32 %f7, 0f00000000;
    @!%p2 bra T32_SB0;
    mad.lo.u32 %r20, %r24, %r3, %r8;
    mul.wide.u32 %rd4, %r20, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f7, [%rd5];
T32_SB0:
    add.u32 %r23, %r22, %r16;
    st.shared.f32 [%r23], %f7;

    // Load B[t + ty, col1] -> tileB[ty][tx+16]
    setp.lt.u32 %p2, %r9, %r3;
    and.pred %p2, %p2, %p4;
    mov.f32 %f8, 0f00000000;
    @!%p2 bra T32_SB1;
    mad.lo.u32 %r20, %r24, %r3, %r9;
    mul.wide.u32 %rd4, %r20, 4;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f8, [%rd5];
T32_SB1:
    add.u32 %r23, %r22, %r17;
    st.shared.f32 [%r23], %f8;

    bar.sync 0;

    // Inner K-tile loop: k = 0..16
    mov.u32 %r25, 0;
T32_KLOOP:
    setp.ge.u32 %p1, %r25, 16;
    @%p1 bra T32_KDONE;

    // a0 = tileA[ty][k], a1 = tileA[ty+16][k]
    shl.b32 %r26, %r5, 4;
    add.u32 %r26, %r26, %r25;
    shl.b32 %r26, %r26, 2;
    add.u32 %r27, %r21, %r26;
    ld.shared.f32 %f9, [%r27];
    add.u32 %r27, %r27, 1024;
    ld.shared.f32 %f10, [%r27];

    // b0 = tileB[k][tx], b1 = tileB[k][tx+16]
    shl.b32 %r28, %r25, 5;
    add.u32 %r28, %r28, %r4;
    shl.b32 %r28, %r28, 2;
    add.u32 %r29, %r22, %r28;
    ld.shared.f32 %f11, [%r29];
    add.u32 %r29, %r29, 64;
    ld.shared.f32 %f12, [%r29];

    fma.rn.f32 %f1, %f9,  %f11, %f1;
    fma.rn.f32 %f2, %f9,  %f12, %f2;
    fma.rn.f32 %f3, %f10, %f11, %f3;
    fma.rn.f32 %f4, %f10, %f12, %f4;

    add.u32 %r25, %r25, 1;
    bra T32_KLOOP;

T32_KDONE:
    bar.sync 0;
    add.u32 %r18, %r18, 16;
    bra T32_LOOP;

T32_DONE:
    setp.lt.u32 %p2, %r10, %r1; // row0 < M
    setp.lt.u32 %p3, %r11, %r1; // row1 < M
    setp.lt.u32 %p4, %r8,  %r3; // col0 < N
    setp.lt.u32 %p5, %r9,  %r3; // col1 < N

    and.pred %p1, %p2, %p4;
    @!%p1 bra T32_ST01;
    mad.lo.u32 %r30, %r10, %r3, %r8;
    mul.wide.u32 %rd6, %r30, 4;
    add.s64 %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f1;
T32_ST01:
    and.pred %p1, %p2, %p5;
    @!%p1 bra T32_ST10;
    mad.lo.u32 %r30, %r10, %r3, %r9;
    mul.wide.u32 %rd6, %r30, 4;
    add.s64 %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f2;
T32_ST10:
    and.pred %p1, %p3, %p4;
    @!%p1 bra T32_ST11;
    mad.lo.u32 %r30, %r11, %r3, %r8;
    mul.wide.u32 %rd6, %r30, 4;
    add.s64 %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f3;
T32_ST11:
    and.pred %p1, %p3, %p5;
    @!%p1 bra T32_EXIT;
    mad.lo.u32 %r30, %r11, %r3, %r9;
    mul.wide.u32 %rd6, %r30, 4;
    add.s64 %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f4;
T32_EXIT:
    ret;
}
"#;

/// Block-per-row fused LayerNorm (`[rows, cols]`, 256 threads/row with shared-memory reduction).
const LAYER_NORM_ROW_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry layer_norm_row_f32(
    .param .u64 x_ptr,
    .param .u64 w_ptr,
    .param .u64 b_ptr,
    .param .u64 y_ptr,
    .param .u32 cols,
    .param .f32 eps
)
{
    .shared .align 4 .f32 smem[256];
    .reg .pred %p<5>;
    .reg .b32 %r<14>;
    .reg .f32 %f<12>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [x_ptr];
    ld.param.u64 %rd2, [w_ptr];
    ld.param.u64 %rd3, [b_ptr];
    ld.param.u64 %rd4, [y_ptr];
    ld.param.u32 %r1, [cols];
    ld.param.f32 %f1, [eps];

    mov.u32 %r2, %ctaid.x; // row
    mov.u32 %r3, %tid.x;   // tid (0..255)
    mul.lo.u32 %r4, %r2, %r1; // row_off
    mov.u32 %r10, smem;

    // 1. Mean
    mov.f32 %f2, 0f00000000;
    mov.u32 %r5, %r3;
LN_SUM_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra LN_SUM_DONE;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd5, %r6, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f3, [%rd6];
    add.f32 %f2, %f2, %f3;
    add.u32 %r5, %r5, 256;
    bra LN_SUM_LOOP;
LN_SUM_DONE:
    shl.b32 %r7, %r3, 2;
    add.u32 %r11, %r10, %r7;
    st.shared.f32 [%r11], %f2;
    bar.sync 0;

    mov.u32 %r8, 128;
LN_RED1:
    setp.eq.u32 %p1, %r8, 0;
    @%p1 bra LN_RED1_DONE;
    setp.lt.u32 %p2, %r3, %r8;
    @!%p2 bra LN_RED1_NEXT;
    add.u32 %r9, %r3, %r8;
    shl.b32 %r9, %r9, 2;
    add.u32 %r12, %r10, %r9;
    ld.shared.f32 %f4, [%r12];
    ld.shared.f32 %f5, [%r11];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%r11], %f5;
LN_RED1_NEXT:
    bar.sync 0;
    shr.b32 %r8, %r8, 1;
    bra LN_RED1;
LN_RED1_DONE:
    ld.shared.f32 %f6, [%r10]; // total sum
    cvt.rn.f32.u32 %f7, %r1;
    div.rn.f32 %f6, %f6, %f7;  // mean

    // 2. Variance
    mov.f32 %f2, 0f00000000;
    mov.u32 %r5, %r3;
LN_VAR_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra LN_VAR_DONE;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd5, %r6, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f3, [%rd6];
    sub.f32 %f3, %f3, %f6;
    fma.rn.f32 %f2, %f3, %f3, %f2;
    add.u32 %r5, %r5, 256;
    bra LN_VAR_LOOP;
LN_VAR_DONE:
    st.shared.f32 [%r11], %f2;
    bar.sync 0;

    mov.u32 %r8, 128;
LN_RED2:
    setp.eq.u32 %p1, %r8, 0;
    @%p1 bra LN_RED2_DONE;
    setp.lt.u32 %p2, %r3, %r8;
    @!%p2 bra LN_RED2_NEXT;
    add.u32 %r9, %r3, %r8;
    shl.b32 %r9, %r9, 2;
    add.u32 %r12, %r10, %r9;
    ld.shared.f32 %f4, [%r12];
    ld.shared.f32 %f5, [%r11];
    add.f32 %f5, %f5, %f4;
    st.shared.f32 [%r11], %f5;
LN_RED2_NEXT:
    bar.sync 0;
    shr.b32 %r8, %r8, 1;
    bra LN_RED2;
LN_RED2_DONE:
    ld.shared.f32 %f8, [%r10]; // sum_sq
    div.rn.f32 %f8, %f8, %f7;  // var
    add.f32 %f8, %f8, %f1;     // var + eps
    sqrt.rn.f32 %f8, %f8;
    mov.f32 %f9, 0f3F800000;   // 1.0
    div.rn.f32 %f8, %f9, %f8;  // inv_std

    // 3. Normalize + affine
    setp.ne.u64 %p3, %rd2, 0;  // has_w
    setp.ne.u64 %p4, %rd3, 0;  // has_b
    mov.u32 %r5, %r3;
LN_OUT_LOOP:
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra LN_DONE;
    add.u32 %r6, %r4, %r5;
    mul.wide.u32 %rd5, %r6, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.f32 %f3, [%rd6];
    sub.f32 %f3, %f3, %f6;
    mul.f32 %f3, %f3, %f8;

    mul.wide.u32 %rd7, %r5, 4;
    @!%p3 bra LN_NO_W;
    add.s64 %rd8, %rd2, %rd7;
    ld.global.f32 %f10, [%rd8];
    mul.f32 %f3, %f3, %f10;
LN_NO_W:
    @!%p4 bra LN_NO_B;
    add.s64 %rd9, %rd3, %rd7;
    ld.global.f32 %f11, [%rd9];
    add.f32 %f3, %f3, %f11;
LN_NO_B:
    add.s64 %rd8, %rd4, %rd5;
    st.global.f32 [%rd8], %f3;
    add.u32 %r5, %r5, 256;
    bra LN_OUT_LOOP;
LN_DONE:
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

/// Launches shared-memory tiled SGEMM on `CudaBuffer`s: `C[m, n] = A[m, k] @ B[k, n]`.
/// Uses 32x32 2x2 register-blocked tiling for `m >= 32 && n >= 32` and 16x16 tiling otherwise.
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
    if m >= 32 && n >= 32 {
        let module = get_or_load(device_index, "matmul_tiled32_2x2_f32", MATMUL_TILED32_2X2_PTX)?;
        let gx = (n as u32).div_ceil(32);
        let gy = (m as u32).div_ceil(32);
        module.launch_grid("matmul_tiled32_2x2_f32", [gx, gy, 1], [16, 16, 1], 0, &mut params)
    } else {
        let module = get_or_load(device_index, "matmul_tiled16_f32", MATMUL_TILED_PTX)?;
        let gx = (n as u32).div_ceil(16);
        let gy = (m as u32).div_ceil(16);
        module.launch_grid("matmul_tiled16_f32", [gx, gy, 1], [16, 16, 1], 0, &mut params)
    }
}

/// Launches fused block-per-row CUDA LayerNorm on `[rows, cols]`.
#[allow(clippy::too_many_arguments)]
pub fn launch_layer_norm_row_f32(
    device_index: usize,
    x: &CudaBuffer,
    w: Option<&CudaBuffer>,
    b: Option<&CudaBuffer>,
    y: &CudaBuffer,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let module = get_or_load(device_index, "layer_norm_row_f32", LAYER_NORM_ROW_PTX)?;
    let mut px = x.device_ptr();
    let mut pw = w.map_or(0u64, |buf| buf.device_ptr());
    let mut pb = b.map_or(0u64, |buf| buf.device_ptr());
    let mut py = y.device_ptr();
    let mut cu = cols as u32;
    let mut ef = eps;
    let mut params: [*mut c_void; 6] = [
        &mut px as *mut u64 as *mut c_void,
        &mut pw as *mut u64 as *mut c_void,
        &mut pb as *mut u64 as *mut c_void,
        &mut py as *mut u64 as *mut c_void,
        &mut cu as *mut u32 as *mut c_void,
        &mut ef as *mut f32 as *mut c_void,
    ];
    module.launch_grid("layer_norm_row_f32", [rows as u32, 1, 1], [256, 1, 1], 0, &mut params)
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
        // 1a. Tiled 16x16 shared-memory SGEMM (non-multiple of 16 dimensions to test boundary masks)
        let (m, k, n) = (37, 45, 29);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
        let gpu_c = cuda_matmul_f32(0, &a, &b, m, k, n).unwrap();
        let mut ref_c = vec![0.0f32; m * n];
        crate::gemm::gemm_f32_strided(&a, 0, k as isize, 1, &b, 0, n as isize, 1, &mut ref_c, m, k, n);
        for (i, (g, r)) in gpu_c.iter().zip(ref_c.iter()).enumerate() {
            assert!((g - r).abs() < 1e-4, "matmul16 mismatch at {i}: gpu={g} vs cpu={r}");
        }

        // 1b. Tiled 32x32 2x2 register-blocked SGEMM (non-multiple of 32 boundaries)
        let (m2, k2, n2) = (73, 51, 69);
        let a2: Vec<f32> = (0..m2 * k2).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let b2: Vec<f32> = (0..k2 * n2).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let gpu_c2 = cuda_matmul_f32(0, &a2, &b2, m2, k2, n2).unwrap();
        let mut ref_c2 = vec![0.0f32; m2 * n2];
        crate::gemm::gemm_f32_strided(&a2, 0, k2 as isize, 1, &b2, 0, n2 as isize, 1, &mut ref_c2, m2, k2, n2);
        for (i, (g, r)) in gpu_c2.iter().zip(ref_c2.iter()).enumerate() {
            assert!((g - r).abs() < 1e-4, "matmul32 mismatch at {i}: gpu={g} vs cpu={r}");
        }

        // 1c. Fused block-per-row LayerNorm
        let (ln_rows, ln_cols) = (4, 300);
        let lnx: Vec<f32> = (0..ln_rows * ln_cols).map(|i| ((i % 19) as f32 - 9.0) * 0.2).collect();
        let lnw: Vec<f32> = (0..ln_cols).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let lnb: Vec<f32> = (0..ln_cols).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
        let d_lnx = CudaBuffer::from_host_f32(0, &lnx).unwrap();
        let d_lnw = CudaBuffer::from_host_f32(0, &lnw).unwrap();
        let d_lnb = CudaBuffer::from_host_f32(0, &lnb).unwrap();
        let d_lny = CudaBuffer::alloc(0, ln_rows * ln_cols * 4).unwrap();
        launch_layer_norm_row_f32(0, &d_lnx, Some(&d_lnw), Some(&d_lnb), &d_lny, ln_rows, ln_cols, 1e-5).unwrap();
        cuda_synchronize(0).unwrap();
        let gpu_ln = d_lny.to_host_f32(ln_rows * ln_cols).unwrap();
        for r in 0..ln_rows {
            let row = &lnx[r * ln_cols..(r + 1) * ln_cols];
            let mean: f32 = row.iter().sum::<f32>() / ln_cols as f32;
            let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / ln_cols as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for c in 0..ln_cols {
                let want = (row[c] - mean) * inv * lnw[c] + lnb[c];
                let got = gpu_ln[r * ln_cols + c];
                assert!((got - want).abs() < 1e-4, "layernorm ({r},{c}): got={got} vs want={want}");
            }
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
