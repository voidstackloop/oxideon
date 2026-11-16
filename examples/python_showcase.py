#!/usr/bin/env python3
"""
Oxideon Python Bindings Showcase (`import oxideon as ox`)
All heavy CPU (SIMD AVX2+FMA + Rayon) and wgpu GPU operations release the Python GIL
and run 100% in native Rust.
"""

import oxideon as ox


def main() -> None:
    ox.manual_seed(2026)
    print("=== Oxideon Python Bindings (100% Rust CPU/GPU Offload) ===")

    # 1. Higher-Order Autograd & Differentiable Linear Algebra in Rust
    x = ox.Tensor([2.0, 3.0], [2], dtype="f64", requires_grad=True)
    y = x.powf(3.0).sum()
    dy = ox.grad([y], [x], create_graph=True)[0]
    d2y = ox.grad([dy.sum()], [x], create_graph=False)[0]
    print(f"[1] Higher-Order Autograd: x={x.to_list()} -> dy/dx={dy.to_list()}, d2y/dx2={d2y.to_list()}")

    a = ox.Tensor([2.0, 1.0, 0.5, 3.0, -1.0, 1.5], [3, 2], dtype="f64", requires_grad=True)
    a_pinv = a.pinv(1e-7)
    a_pinv.sum().backward()
    print(f"[2] Differentiable Moore-Penrose pinv shape={a_pinv.shape}, grad norm={a.grad().square().sum().sqrt().item():.4f}")

    # 2. Fused Single-Pass SIMD/Rayon Kernels & Compiled IR Graph
    h = ox.Tensor.randn([8, 64], requires_grad=True)
    r = ox.Tensor.randn([8, 64], requires_grad=True)
    w = ox.Tensor.ones([64], requires_grad=True)
    normed, _ = ox.fused_add_rmsnorm(h, r, w, 1e-5)
    swi = ox.fused_swiglu(normed, r)
    swi.sum().backward()

    cg = ox.compile_fused_linear_silu_with_grad()
    cg_outs = cg.execute([ox.Tensor.randn([8, 64]), ox.Tensor.randn([64, 32]), ox.Tensor.randn([8, 32])])
    print(
        f"[3] Fused SIMD Kernels & Compiled IR: fused_ops={cg.fused_ops_count}, "
        f"peak_slots={cg.peak_slots}, reuse={cg.reuse_ratio * 100:.1f}%, out={cg_outs[0].shape}"
    )

    # 3. Whole-Loop Native Rust Training (Zero Python Per-Step Overhead)
    mlp = ox.MLP([16, 32, 8])
    tx = ox.Tensor.randn([64, 16])
    ty = ox.Tensor.randn([64, 8])
    losses = mlp.train_epochs_in_rust(tx, ty, epochs=20, batch_size=16, lr=0.02, use_muon=True)
    print(f"[4] Whole-Loop Rust Training (Muon, 20 epochs): loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    # 4. Frontier Foundation Models (DeepSeek-V3 MLA+MoE+MTP, Jamba, MultimodalVLM, DiT)
    tokens = ox.Tensor.from_i64([1, 5, 9, 12, 3, 7, 2, 8, 4, 6, 11, 15], [2, 6])
    deepseek = ox.DeepSeekMoELM(64, 32, 2, 4, 8, 64, 64, 1, 4, 2, 2)
    main_logits, mtp_logits, aux_loss = deepseek.forward_with_mtp(tokens)

    jamba = ox.HybridJambaLM(64, 32, 2, 4, 2, 8, 64, 4, 2, 1, 1)
    jamba_out = jamba.forward(tokens)

    vlm = ox.MultimodalVLM(16, 4, 3, 32, 1, 4, 64, 32, 2, 4, 2, 64, 64)
    vlm_tokens = vlm.generate_multimodal(ox.Tensor.randn([1, 3, 16, 16]), [1, 5, 9], 4)

    dit = ox.DiT(16, 4, 3, 32, 2, 4, 10)
    dit_vel = dit.forward_dit(ox.Tensor.randn([2, 3, 16, 16]), [20, 80], None)
    print(
        f"[5] Frontier Models in Rust:\n"
        f"    DeepSeekMoELM ({deepseek.num_parameters()} params): main={main_logits.shape}, "
        f"MTP={[t.shape for t in mtp_logits]}, aux={aux_loss.item():.4f}\n"
        f"    HybridJambaLM: out={jamba_out.shape}, MultimodalVLM gen={vlm_tokens}, DiT vel={dit_vel.shape}"
    )


if __name__ == "__main__":
    main()
