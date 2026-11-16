"""Unified Multi-Backend Hardware Kernel Registry & Inspector (`oxideon.kernels`).

Provides side-by-side inspection and compilation plans for Apple Silicon Metal/MLX (`MSL`),
NVIDIA CUDA (`PTX` and CUDA C), WebGPU (`WGSL`), and Vulkan (`SPIR-V` 1.5), as well as
direct execution of fused hardware kernels (`mlx_quantize_affine_q4`, `mlx_dequantize_affine_q4`,
`mlx_quantized_matmul_q4`, `flash_attention_tiled`, `paged_kv_attention`, `fused_rmsnorm`,
`fused_swiglu`, `fused_rope`, `fused_sdpa`, `fused_layernorm`).
"""

from __future__ import annotations

from typing import Dict
from . import oxideon as _rust

MslKernelEmitter = _rust.MslKernelEmitter
PtxKernelEmitter = _rust.PtxKernelEmitter
WgslKernelBuilder = _rust.WgslKernelBuilder
VulkanSpirvKernel = _rust.VulkanSpirvKernel
TileKernelPlan = _rust.TileKernelPlan
KernelFusionEngine = _rust.KernelFusionEngine

mlx_quantize_affine_q4 = _rust.mlx_quantize_affine_q4
mlx_dequantize_affine_q4 = _rust.mlx_dequantize_affine_q4
mlx_quantized_matmul_q4 = _rust.mlx_quantized_matmul_q4
flash_attention_tiled = _rust.flash_attention_tiled
paged_kv_attention = _rust.paged_kv_attention


def _cuda_unimplemented(op: str) -> str:
    return (
        f"// oxideon codegen: no CUDA C emitter for '{op}' yet; refusing to emit mismatched code.\n"
        f"#error oxideon: CUDA C emitter for '{op}' is not implemented\n"
    )


def inspect_kernel(
    op_name: str,
    *,
    head_dim: int = 64,
    d_model: int = 128,
    group_size: int = 32,
    eps: float = 1e-5,
    is_causal: bool = True,
    target_sm: str = "sm_90",
) -> Dict[str, str]:
    """Returns a dictionary mapping `{'msl', 'ptx', 'cuda', 'wgsl', 'spirv_disasm'}` to the
    emitted hardware kernel source/assembly for `op_name`.

    Supported `op_name` values:
    - `'flash_attention'`
    - `'rmsnorm'`
    - `'swiglu'`
    - `'rope'`
    - `'q4_gemv'`
    - `'matmul'`
    """
    if not isinstance(op_name, str):
        raise ValueError(f"op_name must be a string, got {type(op_name).__name__}")
    key = op_name.strip().lower()
    msl = MslKernelEmitter(f"oxideon_{key}_msl")
    ptx = PtxKernelEmitter(f"oxideon_{key}_ptx", target_sm=target_sm)
    wgsl = WgslKernelBuilder(f"oxideon_{key}_wgsl", workgroup_x=64)
    vk = VulkanSpirvKernel(f"oxideon_{key}_spirv", num_bindings=4)

    if key in ("flash_attention", "flash_attn", "sdpa"):
        return {
            "msl": msl.emit_flash_attention_msl(head_dim=head_dim, block_m=16, block_n=16, is_causal=is_causal),
            "ptx": ptx.emit_flash_attention_mma_ptx(head_dim=head_dim),
            "cuda": ptx.emit_cuda_c_flash_attention_kernel(head_dim=head_dim, is_causal=is_causal),
            "wgsl": wgsl.emit_flash_attention_wgsl(head_dim=head_dim, block_size=16, is_causal=is_causal),
            "spirv_disasm": VulkanSpirvKernel.disassemble_spirv(vk.emit_fma_spirv_binary()),
        }
    if key in ("rmsnorm", "rms_norm"):
        return {
            "msl": msl.emit_fused_rmsnorm_msl(d_model=d_model, eps=eps),
            "ptx": ptx.emit_warp_rmsnorm_ptx(eps=eps),
            "cuda": ptx.emit_cuda_c_rmsnorm_kernel(eps=eps),
            "wgsl": wgsl.emit_fused_rmsnorm_wgsl(d_model=d_model, eps=eps),
            "spirv_disasm": VulkanSpirvKernel.disassemble_spirv(vk.emit_fma_spirv_binary()),
        }
    if key in ("swiglu", "fused_swiglu", "add_silu"):
        return {
            "msl": msl.emit_fused_swiglu_msl(),
            "ptx": ptx.emit_fused_swiglu_ptx(),
            "cuda": _cuda_unimplemented("swiglu"),
            "wgsl": wgsl.emit_fused_swiglu_wgsl(),
            "spirv_disasm": VulkanSpirvKernel.disassemble_spirv(vk.emit_fadd_spirv_binary()),
        }
    if key in ("rope", "fused_rope", "rotary"):
        return {
            "msl": msl.emit_rope_msl(head_dim=head_dim, theta_base=10000.0),
            "ptx": ptx.emit_rope_ptx(),
            "cuda": _cuda_unimplemented("rope"),
            "wgsl": wgsl.emit_rope_wgsl(head_dim=head_dim),
            "spirv_disasm": VulkanSpirvKernel.disassemble_spirv(vk.emit_fma_spirv_binary()),
        }
    if key in ("q4_gemv", "int4_gemv", "quantized_matmul"):
        return {
            "msl": msl.emit_quantized_affine_q4_matvec_msl(group_size=group_size),
            "ptx": ptx.emit_int4_dequant_gemv_ptx(group_size=group_size),
            "cuda": _cuda_unimplemented("q4_gemv"),
            "wgsl": wgsl.emit_q4_gemv_wgsl(group_size=group_size),
            "spirv_disasm": VulkanSpirvKernel.disassemble_spirv(vk.emit_fma_spirv_binary()),
        }
    if key in ("matmul", "gemm", "steel_gemm"):
        return {
            "msl": msl.emit_simdgroup_matmul_bf16_msl(),
            "ptx": ptx.emit_gemm_fma_ptx(),
            "cuda": _cuda_unimplemented("matmul"),
            "wgsl": wgsl.emit_tiled_matmul_wgsl(tile_dim=16),
            "spirv_disasm": VulkanSpirvKernel.disassemble_spirv(vk.emit_fma_spirv_binary()),
        }
    raise ValueError(
        f"Unknown kernel op_name={op_name!r}; expected one of: "
        "'flash_attention', 'rmsnorm', 'swiglu', 'rope', 'q4_gemv', 'matmul'"
    )


__all__ = [
    "MslKernelEmitter",
    "PtxKernelEmitter",
    "WgslKernelBuilder",
    "VulkanSpirvKernel",
    "TileKernelPlan",
    "KernelFusionEngine",
    "mlx_quantize_affine_q4",
    "mlx_dequantize_affine_q4",
    "mlx_quantized_matmul_q4",
    "flash_attention_tiled",
    "paged_kv_attention",
    "inspect_kernel",
]
