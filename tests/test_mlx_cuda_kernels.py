import math
import pytest
import oxideon as ox


def test_mlx_metal_kernel_emitter_enrichment():
    emitter = ox.MslKernelEmitter('mlx_test_kernel')

    # 1. Apple Silicon Steel-style simdgroup_matrix BF16/FP16 hardware GEMM
    steel_gemm = emitter.emit_simdgroup_matmul_bf16_msl()
    assert 'simdgroup_matrix<bfloat, 8, 8>' in steel_gemm or 'simdgroup_bfloat8x8' in steel_gemm
    assert 'simdgroup_multiply_accumulate' in steel_gemm
    assert 'simdgroup_load' in steel_gemm
    assert 'simdgroup_store' in steel_gemm

    # 2. Fused RMSNorm with simd_sum + threadgroup reduction
    rmsnorm_msl = emitter.emit_fused_rmsnorm_msl(d_model=128, eps=1e-5)
    assert 'simd_sum' in rmsnorm_msl
    assert 'rsqrt' in rmsnorm_msl
    assert '128' in rmsnorm_msl

    # 3. Online-softmax FlashAttention in Metal
    flash_msl = emitter.emit_flash_attention_msl(head_dim=64, block_m=16, block_n=16, is_causal=True)
    assert 'threadgroup_barrier' in flash_msl
    assert 'exp' in flash_msl
    assert 'is_causal' in flash_msl or 'q_idx < k_idx' in flash_msl or 'col > row' in flash_msl

    # 4. Rotary Position Embedding (RoPE) Metal shader
    rope_msl = emitter.emit_rope_msl(head_dim=64, theta_base=10000.0)
    assert 'cos' in rope_msl and 'sin' in rope_msl
    assert '10000' in rope_msl

    # 5. MLX-style 4-bit Quantized Affine Matvec (Q4 GEMV)
    q4_msl = emitter.emit_quantized_affine_q4_matvec_msl(group_size=32)
    assert '0x0F' in q4_msl or '0x0f' in q4_msl or '& 15' in q4_msl
    assert 'simd_sum' in q4_msl
    assert 'scales' in q4_msl and 'biases' in q4_msl

    # 6. Fused SwiGLU Metal shader
    swiglu_msl = emitter.emit_fused_swiglu_msl()
    assert 'exp' in swiglu_msl
    assert 'mlx_test_kernel' in swiglu_msl


def test_cuda_ptx_and_cuda_c_kernel_emitter_enrichment():
    emitter = ox.PtxKernelEmitter('cuda_test_kernel', target_sm='sm_90')

    # 1. Warp-shuffle RMSNorm PTX
    rms_ptx = emitter.emit_warp_rmsnorm_ptx(eps=1e-5)
    assert '.target sm_90' in rms_ptx
    assert 'shfl.sync.bfly.b32' in rms_ptx
    assert 'rsqrt.approx.f32' in rms_ptx

    # 2. Fused SwiGLU PTX
    swiglu_ptx = emitter.emit_fused_swiglu_ptx()
    assert 'ex2.approx.f32' in swiglu_ptx
    assert 'rcp.approx.f32' in swiglu_ptx

    # 3. Tensor Core MMA FlashAttention PTX
    flash_ptx = emitter.emit_flash_attention_mma_ptx(head_dim=64)
    assert 'mma.sync.aligned.m16n8k16' in flash_ptx
    assert 'cp.async' in flash_ptx

    # 4. RoPE PTX
    rope_ptx = emitter.emit_rope_ptx()
    assert 'cos.approx.f32' in rope_ptx
    assert 'sin.approx.f32' in rope_ptx

    # 5. INT4 Dequant GEMV PTX
    q4_ptx = emitter.emit_int4_dequant_gemv_ptx(group_size=32)
    assert 'shr.u32' in q4_ptx
    assert 'and.b32' in q4_ptx
    assert 'fma.rn.f32' in q4_ptx

    # 6. Full CUDA C kernels (FlashAttention + Warp RMSNorm)
    cuda_fa = emitter.emit_cuda_c_flash_attention_kernel(head_dim=64, is_causal=True)
    assert '__global__ void' in cuda_fa
    assert '__shared__' in cuda_fa
    assert '__shfl_xor_sync' in cuda_fa

    cuda_rms = emitter.emit_cuda_c_rmsnorm_kernel(eps=1e-5)
    assert '__global__ void' in cuda_rms
    assert 'rsqrtf' in cuda_rms
    assert '__shfl_xor_sync' in cuda_rms


def test_wgsl_kernel_builder_enrichment():
    builder = ox.WgslKernelBuilder('wgsl_test_entry', workgroup_x=64)

    # 1. 2D Tiled Matmul in WGSL
    matmul_wgsl = builder.emit_tiled_matmul_wgsl(tile_dim=16)
    assert 'var<workgroup> tileA' in matmul_wgsl
    assert 'var<workgroup> tileB' in matmul_wgsl
    assert 'workgroupBarrier()' in matmul_wgsl

    # 2. Online-softmax FlashAttention in WGSL
    fa_wgsl = builder.emit_flash_attention_wgsl(head_dim=64, block_size=16, is_causal=True)
    assert '@compute' in fa_wgsl
    assert 'exp(' in fa_wgsl
    assert 'workgroupBarrier()' in fa_wgsl

    # 3. RoPE in WGSL
    rope_wgsl = builder.emit_rope_wgsl(head_dim=64)
    assert 'cos(' in rope_wgsl and 'sin(' in rope_wgsl

    # 4. 4-bit Packed Quantized GEMV in WGSL
    q4_wgsl = builder.emit_q4_gemv_wgsl(group_size=32)
    assert '0xFu' in q4_wgsl or '0x0Fu' in q4_wgsl
    assert '>>' in q4_wgsl


def test_vulkan_spirv_compute_kernels_and_disassembler():
    vk = ox.VulkanSpirvKernel('vk_compute', num_bindings=3)
    raw = vk.emit_spirv_binary()
    assert raw[0] == 0x07230203  # SPIR-V magic

    # Emit real SPIR-V compute kernels with OpFAdd / OpFMul
    fadd_words = vk.emit_fadd_spirv_binary()
    fma_words = vk.emit_fma_spirv_binary()
    assert len(fadd_words) > len(raw)
    assert len(fma_words) > len(fadd_words)

    disasm_fadd = ox.VulkanSpirvKernel.disassemble_spirv(fadd_words)
    assert 'OpCapability Shader' in disasm_fadd
    assert 'OpEntryPoint GLCompute' in disasm_fadd
    assert 'OpFAdd' in disasm_fadd
    assert 'OpLoad' in disasm_fadd
    assert 'OpStore' in disasm_fadd

    disasm_fma = ox.VulkanSpirvKernel.disassemble_spirv(fma_words)
    assert 'OpFMul' in disasm_fma
    assert 'OpFAdd' in disasm_fma


def test_tile_ir_multi_backend_lowering_and_plans():
    # RMSNorm + SiLU fusion plan lowered to PTX, MSL, and WGSL
    rms_plan = ox.KernelFusionEngine.plan_fused_add_rmsnorm('fused_res_rms', dim=128, eps=1e-5)
    ptx = rms_plan.emit_ptx('sm_90')
    msl = rms_plan.emit_msl()
    wgsl = rms_plan.emit_wgsl()
    assert 'trap;' not in ptx
    assert '#error' not in msl
    assert 'rsqrt' in ptx and 'rsqrt' in msl and 'inverseSqrt' in wgsl

    # FlashAttention plan lowered to PTX, MSL, WGSL
    fa_plan = ox.KernelFusionEngine.plan_flash_attention('flash_attn', head_dim=64, is_causal=True)
    assert 'mma.sync' in fa_plan.emit_ptx('sm_90')
    assert 'threadgroup_barrier' in fa_plan.emit_msl()
    assert 'workgroupBarrier' in fa_plan.emit_wgsl()

    # Quantized Q4 GEMV plan
    q4_plan = ox.KernelFusionEngine.plan_quantized_q4_gemv('q4_gemv', group_size=32)
    assert 'shr.u32' in q4_plan.emit_ptx('sm_90')
    assert 'simd_sum' in q4_plan.emit_msl()
    assert '>>' in q4_plan.emit_wgsl()

    # RoPE plan
    rope_plan = ox.KernelFusionEngine.plan_fused_rope('rope_kern', head_dim=64)
    assert 'cos' in rope_plan.emit_ptx('sm_90')
    assert 'cos' in rope_plan.emit_msl()
    assert 'cos' in rope_plan.emit_wgsl()


def test_mlx_quantize_and_dequantize_affine_q4_roundtrip():
    # Create a [4, 32] weight matrix with smooth values
    vals = [math.sin(i * 0.1) * 2.5 + 0.5 for i in range(4 * 32)]
    w = ox.Tensor(vals, [4, 32])

    packed, scales, biases = ox.mlx_quantize_affine_q4(w, group_size=32)
    assert list(packed.shape) == [4, 16]  # 2 nibbles per byte
    assert list(scales.shape) == [4, 1]
    assert list(biases.shape) == [4, 1]

    w_rec = ox.mlx_dequantize_affine_q4(packed, scales, biases, group_size=32)
    assert list(w_rec.shape) == [4, 32]

    # Quantization error for 4-bit (15 levels over range ~5.0 -> step ~0.33, max err ~0.17)
    diff = (w - w_rec).abs().max().item()
    assert diff < 0.20


def test_mlx_quantized_matmul_q4_execution_and_autograd():
    x_vals = [0.1 * (i + 1) for i in range(2 * 32)]
    w_vals = [math.cos(i * 0.2) for i in range(6 * 32)]
    x = ox.Tensor(x_vals, [2, 32], requires_grad=True)
    w = ox.Tensor(w_vals, [6, 32])

    packed, scales, biases = ox.mlx_quantize_affine_q4(w, group_size=32)
    w_deq = ox.mlx_dequantize_affine_q4(packed, scales, biases, group_size=32)

    out = ox.mlx_quantized_matmul_q4(x, packed, scales, biases, group_size=32)
    ref = x.matmul(w_deq.transpose(0, 1))
    assert list(out.shape) == [2, 6]
    assert (out - ref).abs().max().item() < 1e-6

    # Verify autograd wrt input x
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    expected_gx = ox.Tensor([1.0] * 12, [2, 6]).matmul(w_deq)
    assert (x.grad - expected_gx).abs().max().item() < 1e-5


def test_flash_attention_tiled_execution_and_autograd():
    # Compare online-softmax tiled FlashAttention against standard SDPA
    q = ox.Tensor([math.sin(i * 0.3) for i in range(2 * 8 * 16)], [2, 8, 16], requires_grad=True)
    k = ox.Tensor([math.cos(i * 0.2) for i in range(2 * 8 * 16)], [2, 8, 16], requires_grad=True)
    v = ox.Tensor([math.sin(i * 0.1 + 0.5) for i in range(2 * 8 * 16)], [2, 8, 16], requires_grad=True)

    out_fa = ox.flash_attention_tiled(q, k, v, block_m=4, block_n=4, is_causal=True)
    out_ref = ox.fused_sdpa(q, k, v, is_causal=True)
    assert list(out_fa.shape) == [2, 8, 16]
    assert (out_fa - out_ref).abs().max().item() < 1e-6

    # Backward gradient equivalence
    q2 = ox.Tensor([math.sin(i * 0.3) for i in range(2 * 8 * 16)], [2, 8, 16], requires_grad=True)
    k2 = ox.Tensor([math.cos(i * 0.2) for i in range(2 * 8 * 16)], [2, 8, 16], requires_grad=True)
    v2 = ox.Tensor([math.sin(i * 0.1 + 0.5) for i in range(2 * 8 * 16)], [2, 8, 16], requires_grad=True)

    out_fa.sum().backward()
    ox.fused_sdpa(q2, k2, v2, is_causal=True).sum().backward()

    assert (q.grad - q2.grad).abs().max().item() < 1e-5
    assert (k.grad - k2.grad).abs().max().item() < 1e-5
    assert (v.grad - v2.grad).abs().max().item() < 1e-5


def test_paged_kv_attention_execution():
    # 2 sequences, 2 heads, head_dim=8, block_size=4
    # Seq 0 has context_len=6 (uses blocks [0, 2]), Seq 1 has context_len=3 (uses block [1])
    num_blocks, block_size, num_heads, head_dim = 4, 4, 2, 8
    q = ox.Tensor([0.1 * ((i % 7) - 3) for i in range(2 * num_heads * head_dim)], [2, num_heads, head_dim], requires_grad=True)
    k_cache = ox.Tensor([0.05 * ((i % 11) - 5) for i in range(num_blocks * block_size * num_heads * head_dim)], [num_blocks, block_size, num_heads, head_dim])
    v_cache = ox.Tensor([0.08 * ((i % 9) - 4) for i in range(num_blocks * block_size * num_heads * head_dim)], [num_blocks, block_size, num_heads, head_dim])

    block_tables = [[0, 2], [1, 0]]
    context_lens = [6, 3]

    out = ox.paged_kv_attention(q, k_cache, v_cache, block_tables, context_lens)
    assert list(out.shape) == [2, num_heads, head_dim]

    out.sum().backward()
    assert q.grad is not None
    assert list(q.grad.shape) == [2, num_heads, head_dim]


def test_unified_kernels_subpackage():
    assert hasattr(ox, 'kernels')
    # Inspect all 5 backends side by side for 'flash_attention', 'rmsnorm', 'swiglu', 'rope', 'q4_gemv', 'matmul'
    for op_name in ('flash_attention', 'rmsnorm', 'swiglu', 'rope', 'q4_gemv', 'matmul'):
        bundle = ox.kernels.inspect_kernel(op_name)
        assert set(bundle.keys()) == {'msl', 'ptx', 'cuda', 'wgsl', 'spirv_disasm'}
        for k, code in bundle.items():
            assert isinstance(code, str) and len(code) > 20
