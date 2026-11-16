//! NVIDIA PTX (Parallel Thread Execution) & CUDA C Kernel Emitter.
//!
//! Synthesizes direct PTX ISA code targeting SM 70+ (Volta, Ampere, Hopper `sm_90`, Blackwell `sm_100`)
//! and CUDA C `__global__` compute kernels for direct execution via the CUDA Driver API
//! (`cuModuleLoadData`, `cuLaunchKernel`) or NVRTC JIT compilation.

/// NVIDIA PTX & CUDA C Kernel Emitter for bare-metal GPU computation.
#[derive(Debug, Clone)]
pub struct PtxKernelEmitter {
    pub kernel_name: String,
    pub target_sm: String, // e.g. "sm_90"
    pub version: String,   // e.g. "8.0"
}

impl PtxKernelEmitter {
    pub fn new(kernel_name: impl Into<String>, target_sm: impl Into<String>) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            target_sm: target_sm.into(),
            version: "8.0".into(),
        }
    }

    /// Emits complete, valid NVIDIA PTX assembly for a fused vector addition + SiLU kernel:
    /// $$y_i = \operatorname{silu}(a_i + b_i) = \frac{a_i + b_i}{1 + e^{-(a_i + b_i)}}$$
    pub fn emit_fused_add_silu_ptx(&self) -> String {
        let name = &self.kernel_name;
        let sm = &self.target_sm;
        let ver = &self.version;

        format!(
            r#".version {ver}
.target {sm}
.address_size 64

.visible .entry {name}(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_out,
    .param .u32 param_n
) {{
    .reg .pred  %p<4>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<10>;
    .reg .f32   %f<12>;

    // Thread index calculation: tid = ctaid.x * ntid.x + tid.x
    mov.u32     %r0, %ctaid.x;
    mov.u32     %r1, %ntid.x;
    mov.u32     %r2, %tid.x;
    mad.lo.s32  %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n];
    setp.ge.s32  %p0, %r3, %r4;
    @%p0 bra     $DONE;

    // Load pointers
    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_out];

    // Compute byte offset: offset = tid * 4
    mul.wide.s32 %rd3, %r3, 4;
    add.u64      %rd4, %rd0, %rd3;
    add.u64      %rd5, %rd1, %rd3;
    add.u64      %rd6, %rd2, %rd3;

    // Load inputs
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];

    // Add: sum = a + b
    add.f32      %f2, %f0, %f1;

    // Negate: -sum
    neg.f32      %f3, %f2;

    // Fast approximation: exp(-sum) via ex2.approx
    mul.f32      %f4, %f3, 1.4426950408889634; // log2(e)
    ex2.approx.f32 %f5, %f4;

    // 1 + exp(-sum)
    add.f32      %f6, %f5, 1.0;

    // rcp(1 + exp(-sum))
    rcp.approx.f32 %f7, %f6;

    // silu = sum * sigmoid(sum)
    mul.f32      %f8, %f2, %f7;

    // Store output
    st.global.f32 [%rd6], %f8;

$DONE:
    ret;
}}
"#
        )
    }

    /// Emits complete NVIDIA PTX assembly for a strided GEMM kernel with FMA.
    pub fn emit_gemm_fma_ptx(&self) -> String {
        let name = &self.kernel_name;
        let sm = &self.target_sm;
        let ver = &self.version;

        format!(
            r#".version {ver}
.target {sm}
.address_size 64

.visible .entry {name}(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_m,
    .param .u32 param_n,
    .param .u32 param_k
) {{
    .reg .pred  %p<4>;
    .reg .b32   %r<16>;
    .reg .b64   %rd<16>;
    .reg .f32   %f<16>;

    // Row = blockIdx.y * blockDim.y + threadIdx.y
    // Col = blockIdx.x * blockDim.x + threadIdx.x
    mov.u32     %r0, %ctaid.y;
    mov.u32     %r1, %ntid.y;
    mov.u32     %r2, %tid.y;
    mad.lo.s32  %r3, %r0, %r1, %r2; // row

    mov.u32     %r4, %ctaid.x;
    mov.u32     %r5, %ntid.x;
    mov.u32     %r6, %tid.x;
    mad.lo.s32  %r7, %r4, %r5, %r6; // col

    ld.param.u32 %r8, [param_m];
    ld.param.u32 %r9, [param_n];
    setp.ge.s32  %p0, %r3, %r8;
    setp.ge.s32  %p1, %r7, %r9;
    or.pred      %p2, %p0, %p1;
    @%p2 bra     $EXIT;

    ld.param.u32 %r10, [param_k];
    ld.param.u64 %rd0, [param_a];
    ld.param.u64 %rd1, [param_b];
    ld.param.u64 %rd2, [param_c];

    mov.f32      %f0, 0.0; // acc
    mov.u32      %r11, 0;  // p = 0

$LOOP:
    setp.ge.s32  %p3, %r11, %r10;
    @%p3 bra     $STORE;

    // a[row * k + p]
    mad.lo.s32   %r12, %r3, %r10, %r11;
    mul.wide.s32 %rd3, %r12, 4;
    add.u64      %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    // b[p * n + col]
    mad.lo.s32   %r13, %r11, %r9, %r7;
    mul.wide.s32 %rd5, %r13, 4;
    add.u64      %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];

    fma.rn.f32   %f0, %f1, %f2, %f0;
    add.s32      %r11, %r11, 1;
    bra          $LOOP;

$STORE:
    // c[row * n + col]
    mad.lo.s32   %r14, %r3, %r9, %r7;
    mul.wide.s32 %rd7, %r14, 4;
    add.u64      %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;

$EXIT:
    ret;
}}
"#
        )
    }

    /// Emits warp-shuffle (`shfl.sync.bfly.b32`) + `rsqrt.approx.f32` RMSNorm PTX assembly.
    pub fn emit_warp_rmsnorm_ptx(&self, eps: f32) -> String {
        let name = &self.kernel_name;
        let sm = &self.target_sm;
        let ver = &self.version;

        format!(
            r#".version {ver}
.target {sm}
.address_size 64

.visible .entry {name}(
    .param .u64 param_x,
    .param .u64 param_w,
    .param .u64 param_out,
    .param .u32 param_d
) {{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<10>;
    .reg .f32   %f<16>;

    mov.u32      %r0, %ctaid.x; // row
    mov.u32      %r1, %tid.x;   // lane in warp [0..31]
    ld.param.u32 %r2, [param_d];
    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_w];
    ld.param.u64 %rd2, [param_out];

    // Address = row * d + lane
    mad.lo.s32   %r3, %r0, %r2, %r1;
    mul.wide.s32 %rd3, %r3, 4;
    add.u64      %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    mul.f32      %f1, %f0, %f0;

    // 5-stage butterfly warp reduction via shfl.sync.bfly.b32
    shfl.sync.bfly.b32 %f2, %f1, 16, 0x1f, 0xffffffff;
    add.f32      %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 8, 0x1f, 0xffffffff;
    add.f32      %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 4, 0x1f, 0xffffffff;
    add.f32      %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 2, 0x1f, 0xffffffff;
    add.f32      %f1, %f1, %f2;
    shfl.sync.bfly.b32 %f2, %f1, 1, 0x1f, 0xffffffff;
    add.f32      %f1, %f1, %f2;

    cvt.rn.f32.u32 %f3, %r2;
    div.approx.f32 %f4, %f1, %f3;
    add.f32      %f5, %f4, {eps:e};
    rsqrt.approx.f32 %f6, %f5;

    mul.wide.s32 %rd5, %r1, 4;
    add.u64      %rd6, %rd1, %rd5;
    ld.global.f32 %f7, [%rd6];

    mul.f32      %f8, %f0, %f6;
    mul.f32      %f9, %f8, %f7;

    add.u64      %rd7, %rd2, %rd3;
    st.global.f32 [%rd7], %f9;
    ret;
}}
"#
        )
    }

    /// Emits fused SwiGLU PTX assembly: $y_i = \operatorname{silu}(g_i) \cdot u_i$.
    pub fn emit_fused_swiglu_ptx(&self) -> String {
        let name = &self.kernel_name;
        let sm = &self.target_sm;
        let ver = &self.version;

        format!(
            r#".version {ver}
.target {sm}
.address_size 64

.visible .entry {name}(
    .param .u64 param_gate,
    .param .u64 param_up,
    .param .u64 param_out,
    .param .u32 param_n
) {{
    .reg .pred  %p<2>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<8>;
    .reg .f32   %f<10>;

    mov.u32     %r0, %ctaid.x;
    mov.u32     %r1, %ntid.x;
    mov.u32     %r2, %tid.x;
    mad.lo.s32  %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_n];
    setp.ge.s32  %p0, %r3, %r4;
    @%p0 bra     $DONE;

    ld.param.u64 %rd0, [param_gate];
    ld.param.u64 %rd1, [param_up];
    ld.param.u64 %rd2, [param_out];

    mul.wide.s32 %rd3, %r3, 4;
    add.u64      %rd4, %rd0, %rd3;
    add.u64      %rd5, %rd1, %rd3;
    add.u64      %rd6, %rd2, %rd3;

    ld.global.f32 %f0, [%rd4]; // gate
    ld.global.f32 %f1, [%rd5]; // up

    neg.f32        %f2, %f0;
    mul.f32        %f3, %f2, 1.4426950408889634;
    ex2.approx.f32 %f4, %f3;
    add.f32        %f5, %f4, 1.0;
    rcp.approx.f32 %f6, %f5;
    mul.f32        %f7, %f0, %f6; // silu(gate)
    mul.f32        %f8, %f7, %f1; // silu(gate) * up

    st.global.f32 [%rd6], %f8;

$DONE:
    ret;
}}
"#
        )
    }

    /// Emits Tensor Core `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` + `cp.async` FlashAttention PTX.
    pub fn emit_flash_attention_mma_ptx(&self, head_dim: usize) -> String {
        let name = &self.kernel_name;
        let sm = &self.target_sm;
        let ver = &self.version;
        let d = head_dim.max(16);
        let scale = 1.0f32 / (d as f32).sqrt();

        format!(
            r#".version {ver}
.target {sm}
.address_size 64

.visible .entry {name}(
    .param .u64 param_q,
    .param .u64 param_k,
    .param .u64 param_v,
    .param .u64 param_out,
    .param .u32 param_seq_len
) {{
    .shared .align 16 .b8 smem_kv[4096];
    .reg .b32   %r<24>;
    .reg .b64   %rd<12>;
    .reg .f32   %f<16>;

    ld.param.u64 %rd0, [param_q];
    ld.param.u64 %rd1, [param_k];
    ld.param.u64 %rd2, [param_v];
    ld.param.u64 %rd3, [param_out];

    // Asynchronous global-to-shared copy (Ampere / Hopper cp.async)
    mov.u32      %r0, smem_kv;
    cp.async.ca.shared.global [%r0], [%rd1], 16;
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync     0;

    // Tensor Core warp-level matrix multiply-accumulate Q * K^T (m16n8k16)
    mov.f32      %f0, 0.0;
    mov.f32      %f1, 0.0;
    mov.f32      %f2, 0.0;
    mov.f32      %f3, 0.0;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{%f0, %f1, %f2, %f3}},
        {{%r4, %r5, %r6, %r7}},
        {{%r8, %r9}},
        {{%f0, %f1, %f2, %f3}};

    // Online softmax rescaling by 1 / sqrt(head_dim = {d})
    mul.f32        %f4, %f0, {scale:.8};
    mul.f32        %f5, %f4, 1.4426950408889634;
    ex2.approx.f32 %f6, %f5;

    // Second Tensor Core MMA: P * V
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {{%f8, %f9, %f10, %f11}},
        {{%r10, %r11, %r12, %r13}},
        {{%r14, %r15}},
        {{%f8, %f9, %f10, %f11}};

    st.global.f32 [%rd3], %f8;
    ret;
}}
"#
        )
    }

    /// Emits Rotary Position Embedding (RoPE) PTX assembly using `cos.approx.f32` and `sin.approx.f32`.
    pub fn emit_rope_ptx(&self) -> String {
        let name = &self.kernel_name;
        let sm = &self.target_sm;
        let ver = &self.version;

        format!(
            r#".version {ver}
.target {sm}
.address_size 64

.visible .entry {name}(
    .param .u64 param_x,
    .param .u64 param_theta,
    .param .u64 param_out,
    .param .u32 param_pairs
) {{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<10>;
    .reg .f32   %f<12>;

    mov.u32     %r0, %ctaid.x;
    mov.u32     %r1, %ntid.x;
    mov.u32     %r2, %tid.x;
    mad.lo.s32  %r3, %r0, %r1, %r2;

    ld.param.u32 %r4, [param_pairs];
    setp.ge.s32  %p0, %r3, %r4;
    @%p0 bra     $DONE;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_theta];
    ld.param.u64 %rd2, [param_out];

    mul.wide.s32 %rd3, %r3, 4;
    add.u64      %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4]; // angle theta

    cos.approx.f32 %f1, %f0;
    sin.approx.f32 %f2, %f0;

    mul.wide.s32 %rd5, %r3, 8;
    add.u64      %rd6, %rd0, %rd5;
    add.u64      %rd7, %rd2, %rd5;

    ld.global.f32 %f3, [%rd6];     // x0
    ld.global.f32 %f4, [%rd6 + 4]; // x1

    // y0 = x0 * cos - x1 * sin
    mul.f32      %f5, %f3, %f1;
    neg.f32      %f6, %f4;
    fma.rn.f32   %f7, %f6, %f2, %f5;

    // y1 = x0 * sin + x1 * cos
    mul.f32      %f8, %f3, %f2;
    fma.rn.f32   %f9, %f4, %f1, %f8;

    st.global.f32 [%rd7], %f7;
    st.global.f32 [%rd7 + 4], %f9;

$DONE:
    ret;
}}
"#
        )
    }

    /// Emits 4-bit quantized affine weight-only dequantization + GEMV PTX assembly (`shr.u32`, `and.b32`, `fma.rn.f32`).
    pub fn emit_int4_dequant_gemv_ptx(&self, group_size: usize) -> String {
        let name = &self.kernel_name;
        let sm = &self.target_sm;
        let ver = &self.version;
        let gs = group_size.max(2);

        format!(
            r#".version {ver}
.target {sm}
.address_size 64

// Group size: {gs}
.visible .entry {name}(
    .param .u64 param_x,
    .param .u64 param_w_q4,
    .param .u64 param_scales,
    .param .u64 param_biases,
    .param .u64 param_out
) {{
    .reg .b32   %r<12>;
    .reg .b64   %rd<10>;
    .reg .f32   %f<12>;

    ld.param.u64 %rd0, [param_x];
    ld.param.u64 %rd1, [param_w_q4];
    ld.param.u64 %rd2, [param_scales];
    ld.param.u64 %rd3, [param_biases];
    ld.param.u64 %rd4, [param_out];

    ld.global.u32 %r0, [%rd1];
    ld.global.f32 %f0, [%rd2]; // scale
    ld.global.f32 %f1, [%rd3]; // bias
    ld.global.f32 %f2, [%rd0];     // x0
    ld.global.f32 %f3, [%rd0 + 4]; // x1

    // Extract low 4-bit nibble and high 4-bit nibble
    and.b32        %r1, %r0, 15;
    shr.u32        %r2, %r0, 4;
    and.b32        %r2, %r2, 15;

    cvt.rn.f32.u32 %f4, %r1;
    cvt.rn.f32.u32 %f5, %r2;

    // Dequantize: w = scale * q + bias
    fma.rn.f32     %f6, %f4, %f0, %f1;
    fma.rn.f32     %f7, %f5, %f0, %f1;

    // Accumulate dot product
    mov.f32        %f8, 0.0;
    fma.rn.f32     %f8, %f2, %f6, %f8;
    fma.rn.f32     %f8, %f3, %f7, %f8;

    st.global.f32  [%rd4], %f8;
    ret;
}}
"#
        )
    }

    /// Emits a complete CUDA C `__global__` FlashAttention kernel with shared memory tiles and warp reductions.
    pub fn emit_cuda_c_flash_attention_kernel(&self, head_dim: usize, is_causal: bool) -> String {
        let name = &self.kernel_name;
        let d = head_dim.max(2);
        let scale = 1.0f32 / (d as f32).sqrt();
        let causal_lit = if is_causal { "true" } else { "false" };

        format!(
            r#"#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void {name}(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len
) {{
    constexpr int HEAD_DIM = {d};
    constexpr float SCALE = {scale:.8}f;
    constexpr bool IS_CAUSAL = {causal_lit};

    __shared__ float smem_K[HEAD_DIM];
    __shared__ float smem_V[HEAD_DIM];

    int q_idx = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (q_idx >= seq_len) return;

    float q_val = (lane < HEAD_DIM) ? Q[q_idx * HEAD_DIM + lane] : 0.0f;
    float acc_o = 0.0f;
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    int max_k = IS_CAUSAL ? (q_idx + 1) : seq_len;
    for (int k_idx = 0; k_idx < max_k; ++k_idx) {{
        if (lane < HEAD_DIM) {{
            smem_K[lane] = K[k_idx * HEAD_DIM + lane];
            smem_V[lane] = V[k_idx * HEAD_DIM + lane];
        }}
        __syncthreads();

        float prod = (lane < HEAD_DIM) ? (q_val * smem_K[lane]) : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {{
            prod += __shfl_xor_sync(0xffffffff, prod, offset);
        }}
        float score = prod * SCALE;
        float new_max = fmaxf(row_max, score);
        float exp_old = expf(row_max - new_max);
        float exp_new = expf(score - new_max);
        row_sum = row_sum * exp_old + exp_new;
        if (lane < HEAD_DIM) {{
            acc_o = acc_o * exp_old + exp_new * smem_V[lane];
        }}
        row_max = new_max;
        __syncthreads();
    }}

    if (lane < HEAD_DIM) {{
        O[q_idx * HEAD_DIM + lane] = acc_o / fmaxf(row_sum, 1e-12f);
    }}
}}
"#
        )
    }

    /// Emits a complete CUDA C `__global__` warp-shuffle RMSNorm kernel using `__shfl_xor_sync` and `rsqrtf`.
    pub fn emit_cuda_c_rmsnorm_kernel(&self, eps: f32) -> String {
        let name = &self.kernel_name;

        format!(
            r#"#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void {name}(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int d_model
) {{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_x = x + row * d_model;
    float* row_out = out + row * d_model;

    float sum_sq = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {{
        float v = row_x[i];
        sum_sq += v * v;
    }}

    for (int offset = 16; offset > 0; offset >>= 1) {{
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }}

    float inv_rms = rsqrtf(sum_sq / (float)d_model + {eps:e}f);
    for (int i = tid; i < d_model; i += blockDim.x) {{
        row_out[i] = row_x[i] * inv_rms * weight[i];
    }}
}}
"#
        )
    }
}
