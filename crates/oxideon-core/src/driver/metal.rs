//! Apple Metal Shading Language (`MSL`) Compute Kernel Emitter.
//!
//! Emits specialized compute shaders for Apple Silicon (M1/M2/M3/M4) unified memory architectures,
//! including MLX Steel-style `simdgroup_matrix` hardware matrix multiplication, `simd_sum`
//! reductions, online-softmax FlashAttention, Rotary Position Embeddings (RoPE), and 4-bit
//! quantized affine matrix-vector multiplication (`Q4 GEMV`).

/// Apple Metal Shading Language (`MSL`) Kernel Emitter.
#[derive(Debug, Clone)]
pub struct MslKernelEmitter {
    pub kernel_name: String,
    pub threadgroup_size: usize,
}

impl MslKernelEmitter {
    pub fn new(kernel_name: impl Into<String>) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            threadgroup_size: 256,
        }
    }

    /// Emits complete Apple Metal Shading Language (`MSL`) code for a fused vector addition + SiLU kernel.
    pub fn emit_fused_add_silu_msl(&self) -> String {
        let name = &self.kernel_name;

        format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
    device const float* in_a [[buffer(0)]],
    device const float* in_b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= count) {{
        return;
    }}
    float x = in_a[gid] + in_b[gid];
    float silu_x = x / (1.0f + exp(-x));
    out[gid] = silu_x;
}}
"#
        )
    }

    /// Emits an Apple Metal compute shader for tiled matrix multiplication using `threadgroup` shared memory.
    pub fn emit_matmul_tiled_msl(&self) -> String {
        let name = &self.kernel_name;

        format!(
            r#"#include <metal_stdlib>
using namespace metal;

constant uint TILE_DIM = 16;

kernel void {name}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 threadgroup_pos [[threadgroup_position_in_grid]],
    uint2 thread_pos [[thread_position_in_threadgroup]],
    uint2 global_pos [[thread_position_in_grid]]
) {{
    threadgroup float tileA[TILE_DIM][TILE_DIM];
    threadgroup float tileB[TILE_DIM][TILE_DIM];

    uint row = global_pos.y;
    uint col = global_pos.x;
    float acc = 0.0f;

    uint num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (uint t = 0; t < num_tiles; ++t) {{
        uint tiled_k_a = t * TILE_DIM + thread_pos.x;
        if (row < M && tiled_k_a < K) {{
            tileA[thread_pos.y][thread_pos.x] = A[row * K + tiled_k_a];
        }} else {{
            tileA[thread_pos.y][thread_pos.x] = 0.0f;
        }}

        uint tiled_k_b = t * TILE_DIM + thread_pos.y;
        if (col < N && tiled_k_b < K) {{
            tileB[thread_pos.y][thread_pos.x] = B[tiled_k_b * N + col];
        }} else {{
            tileB[thread_pos.y][thread_pos.x] = 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_DIM; ++k) {{
            acc += tileA[thread_pos.y][k] * tileB[k][thread_pos.x];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (row < M && col < N) {{
        C[row * N + col] = acc;
    }}
}}
"#
        )
    }

    /// Emits an Apple Silicon MLX Steel-style `simdgroup_matrix<bfloat, 8, 8>` hardware matrix core GEMM shader.
    pub fn emit_simdgroup_matmul_bf16_msl(&self) -> String {
        let name = &self.kernel_name;

        format!(
            r#"#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void {name}(
    device const bfloat* A [[buffer(0)]],
    device const bfloat* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {{
    uint row_tile = tgid.y * 8;
    uint col_tile = tgid.x * 8;
    if (row_tile >= M || col_tile >= N) {{
        return;
    }}

    simdgroup_matrix<float, 8, 8> acc(0.0f);
    simdgroup_matrix<bfloat, 8, 8> fragA;
    simdgroup_matrix<bfloat, 8, 8> fragB;

    for (uint k = 0; k < K; k += 8) {{
        simdgroup_load(fragA, A + row_tile * K + k, K);
        simdgroup_load(fragB, B + k * N + col_tile, N);
        simdgroup_multiply_accumulate(acc, fragA, fragB, acc);
    }}

    simdgroup_store(acc, C + row_tile * N + col_tile, N);
}}
"#
        )
    }

    /// Emits a fused single-pass RMSNorm Metal compute shader using `simd_sum` and `threadgroup` reduction.
    pub fn emit_fused_rmsnorm_msl(&self, d_model: usize, eps: f32) -> String {
        let name = &self.kernel_name;
        let tg = self.threadgroup_size.max(32);
        let num_simdgroups = tg.div_ceil(32);

        format!(
            r#"#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint D_MODEL = {d_model};
constant float EPS = {eps:e}f;

kernel void {name}(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {{
    threadgroup float shared_sums[{num_simdgroups}];
    uint row_offset = tgid * D_MODEL;

    float local_sq = 0.0f;
    for (uint i = tid; i < D_MODEL; i += {tg}) {{
        float v = x[row_offset + i];
        local_sq = fma(v, v, local_sq);
    }}

    float sg_sum = simd_sum(local_sq);
    if (simd_lane == 0) {{
        shared_sums[simd_id] = sg_sum;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sq = 0.0f;
    if (simd_id == 0) {{
        float v = (simd_lane < {num_simdgroups}) ? shared_sums[simd_lane] : 0.0f;
        total_sq = simd_sum(v);
        if (simd_lane == 0) {{
            shared_sums[0] = rsqrt(total_sq / float(D_MODEL) + EPS);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = shared_sums[0];
    for (uint i = tid; i < D_MODEL; i += {tg}) {{
        out[row_offset + i] = x[row_offset + i] * inv_rms * weight[i];
    }}
}}
"#
        )
    }

    /// Emits an Apple Silicon online-softmax FlashAttention Metal shader.
    pub fn emit_flash_attention_msl(
        &self,
        head_dim: usize,
        block_m: usize,
        block_n: usize,
        is_causal: bool,
    ) -> String {
        let name = &self.kernel_name;
        let d = head_dim.max(2);
        let bm = block_m.max(1);
        let bn = block_n.max(1);
        let scale = 1.0f32 / (d as f32).sqrt();
        let causal_flag = if is_causal { 1 } else { 0 };

        format!(
            r#"#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM = {d};
constant uint BLOCK_M = {bm};
constant uint BLOCK_N = {bn};
constant float ATTN_SCALE = {scale:.8}f;
constant bool IS_CAUSAL = ({causal_flag} != 0);

kernel void {name}(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    threadgroup float smem_K[BLOCK_N][HEAD_DIM];
    threadgroup float smem_V[BLOCK_N][HEAD_DIM];

    uint q_idx = tgid.x * BLOCK_M + tid;
    uint batch_head_offset = tgid.y * seq_len * HEAD_DIM;

    float q_reg[HEAD_DIM];
    float acc_o[HEAD_DIM];
    for (uint d = 0; d < HEAD_DIM; ++d) {{
        q_reg[d] = (q_idx < seq_len) ? Q[batch_head_offset + q_idx * HEAD_DIM + d] : 0.0f;
        acc_o[d] = 0.0f;
    }}

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    uint num_k_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    for (uint kt = 0; kt < num_k_tiles; ++kt) {{
        uint k_base = kt * BLOCK_N;
        if (tid < BLOCK_N) {{
            uint k_row = k_base + tid;
            for (uint d = 0; d < HEAD_DIM; ++d) {{
                smem_K[tid][d] = (k_row < seq_len) ? K[batch_head_offset + k_row * HEAD_DIM + d] : 0.0f;
                smem_V[tid][d] = (k_row < seq_len) ? V[batch_head_offset + k_row * HEAD_DIM + d] : 0.0f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < BLOCK_N; ++j) {{
            uint k_idx = k_base + j;
            if (k_idx >= seq_len) break;
            if (IS_CAUSAL && q_idx < k_idx) continue;

            float dot_qk = 0.0f;
            for (uint d = 0; d < HEAD_DIM; ++d) {{
                dot_qk = fma(q_reg[d], smem_K[j][d], dot_qk);
            }}
            float score = dot_qk * ATTN_SCALE;
            float new_max = max(row_max, score);
            float exp_old = exp(row_max - new_max);
            float exp_new = exp(score - new_max);
            row_sum = row_sum * exp_old + exp_new;
            for (uint d = 0; d < HEAD_DIM; ++d) {{
                acc_o[d] = acc_o[d] * exp_old + exp_new * smem_V[j][d];
            }}
            row_max = new_max;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (q_idx < seq_len) {{
        float inv_sum = 1.0f / max(row_sum, 1e-12f);
        for (uint d = 0; d < HEAD_DIM; ++d) {{
            O[batch_head_offset + q_idx * HEAD_DIM + d] = acc_o[d] * inv_sum;
        }}
    }}
}}
"#
        )
    }

    /// Emits a Rotary Position Embedding (RoPE) Metal compute shader.
    pub fn emit_rope_msl(&self, head_dim: usize, theta_base: f32) -> String {
        let name = &self.kernel_name;
        let d = head_dim.max(2);
        let half_d = d / 2;

        format!(
            r#"#include <metal_stdlib>
using namespace metal;

constant uint HEAD_DIM = {d};
constant uint HALF_DIM = {half_d};
constant float THETA_BASE = {theta_base:.1}f;

kernel void {name}(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& total_pairs [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= total_pairs) return;
    uint pair_in_head = gid % HALF_DIM;
    uint token_pos = gid / HALF_DIM;

    float freq_exp = float(2 * pair_in_head) / float(HEAD_DIM);
    float inv_freq = 1.0f / pow(THETA_BASE, freq_exp);
    float angle = float(token_pos) * inv_freq;
    float c = cos(angle);
    float s = sin(angle);

    uint idx0 = 2 * gid;
    uint idx1 = idx0 + 1;
    float x0 = x[idx0];
    float x1 = x[idx1];
    out[idx0] = x0 * c - x1 * s;
    out[idx1] = x0 * s + x1 * c;
}}
"#
        )
    }

    /// Emits an MLX-style 4-bit quantized affine weight-only matrix-vector multiply (`qmv`) Metal shader:
    /// $$w_{ij} = \operatorname{scale}_{i, g} \cdot q_{ij} + \operatorname{bias}_{i, g}, \quad q_{ij} \in \{0, \dots, 15\}$$
    pub fn emit_quantized_affine_q4_matvec_msl(&self, group_size: usize) -> String {
        let name = &self.kernel_name;
        let gs = group_size.max(2);
        let packed_per_group = gs / 2;

        format!(
            r#"#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint GROUP_SIZE = {gs};
constant uint PACKED_PER_GROUP = {packed_per_group};

kernel void {name}(
    device const float* x [[buffer(0)]],
    device const uchar* w_packed [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    if (row >= out_features) return;
    uint num_groups = in_features / GROUP_SIZE;
    float acc = 0.0f;

    for (uint g = lane; g < num_groups; g += 32) {{
        float scale = scales[row * num_groups + g];
        float bias = biases[row * num_groups + g];
        uint packed_base = (row * num_groups + g) * PACKED_PER_GROUP;
        uint x_base = g * GROUP_SIZE;

        for (uint p = 0; p < PACKED_PER_GROUP; ++p) {{
            uchar byte_val = w_packed[packed_base + p];
            float q_lo = float(byte_val & 0x0F);
            float q_hi = float((byte_val >> 4) & 0x0F);
            float w0 = fma(scale, q_lo, bias);
            float w1 = fma(scale, q_hi, bias);
            acc = fma(x[x_base + 2 * p], w0, acc);
            acc = fma(x[x_base + 2 * p + 1], w1, acc);
        }}
    }}

    float row_dot = simd_sum(acc);
    if (lane == 0) {{
        out[row] = row_dot;
    }}
}}
"#
        )
    }

    /// Emits a fused SwiGLU Metal compute shader: $y_i = \operatorname{silu}(g_i) \cdot u_i$.
    pub fn emit_fused_swiglu_msl(&self) -> String {
        let name = &self.kernel_name;

        format!(
            r#"#include <metal_stdlib>
using namespace metal;

kernel void {name}(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= count) return;
    float g = gate[gid];
    float u = up[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = silu_g * u;
}}
"#
        )
    }
}
