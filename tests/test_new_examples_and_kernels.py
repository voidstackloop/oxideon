import math
import numpy as np
import pytest

import oxideon as ox
from oxideon import Tensor, from_numpy


# ---------------------------------------------------------------------------
# Test 1: Multimodal VLM & Fusion
# ---------------------------------------------------------------------------
def test_multimodal_vlm_and_fusion():
    from oxideon.multimodal_vlm import (
        VLMProjection,
        PerceiverResampler,
        VisionLanguageConnector,
    )
    from oxideon.multimodal_fusion import (
        CrossAttentionFusion,
        GatedMultimodalFusion,
    )

    b, num_patches, v_dim = 2, 16, 32
    text_len, t_dim = 8, 24
    num_latents = 4

    visual = ox.randn([b, num_patches, v_dim])
    text = ox.randn([b, text_len, t_dim])

    # 1. LLaVA MLP Connector
    llava = VisionLanguageConnector(visual_dim=v_dim, text_dim=t_dim, mode="mlp")
    prefix_llava = llava(visual, text)
    assert prefix_llava.shape == (b, num_patches + text_len, t_dim)

    # 2. Flamingo Perceiver Resampler
    flamingo = VisionLanguageConnector(
        visual_dim=v_dim, text_dim=t_dim, mode="resampler", num_latents=num_latents
    )
    prefix_flamingo = flamingo(visual, text)
    assert prefix_flamingo.shape == (b, num_latents + text_len, t_dim)

    # 3. Gated Multimodal Fusion
    gmu = GatedMultimodalFusion(dim_a=v_dim, dim_b=t_dim, out_dim=16)
    fused_gmu = gmu(visual.mean(dim=1), text.mean(dim=1))
    assert fused_gmu.shape == (b, 16)

    # 4. Cross-Attention Fusion
    cross_attn = CrossAttentionFusion(d_model=t_dim, num_heads=2)
    proj_v = llava.projector(visual)
    out_attn = cross_attn(text, proj_v)
    assert out_attn.shape == (b, text_len, t_dim)

    # 5. SigLIP Loss autograd
    z_i = ox.randn([b, 16]).requires_grad_(True)
    z_t = ox.randn([b, 16]).requires_grad_(True)
    loss = ox.siglip_loss(z_i, z_t)
    loss.backward()
    assert z_i.grad is not None
    assert z_t.grad is not None


# ---------------------------------------------------------------------------
# Test 2: DeepSeek & Switch Mixture of Experts (MoE)
# ---------------------------------------------------------------------------
def test_mixture_of_experts_deepseek():
    from oxideon.mixture_of_experts import MoELayer, TopKGating
    from oxideon.expert_parallel import CapacityConstrainedMoE, TopKRouterWithAuxLoss

    num_tokens = 16
    d_model = 16
    hidden_dim = 32
    num_experts = 4
    top_k = 2

    x = ox.randn([num_tokens, d_model]).requires_grad_(True)

    # 1. TopKGating
    router = TopKGating(d_model=d_model, num_experts=num_experts, top_k=top_k, noisy_gating=True)
    idx, weights, aux, z = router(x)
    assert idx.shape == (num_tokens, top_k)
    assert weights.shape == (num_tokens, top_k)
    assert aux > 0.0
    assert z >= 0.0

    # 2. Sparse MoELayer autograd
    moe = MoELayer(d_model=d_model, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    out, aux_l, z_l = moe(x)
    assert out.shape == (num_tokens, d_model)
    loss = out.sum() + 0.01 * aux_l
    loss.backward()
    assert x.grad is not None

    # 3. CapacityConstrainedMoE
    cc_moe = CapacityConstrainedMoE(
        hidden_dim=d_model,
        expert_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=1.5,
    )
    cc_out, dropped = cc_moe(x)
    assert cc_out.shape == (num_tokens, d_model)
    assert isinstance(dropped, (int, float))


# ---------------------------------------------------------------------------
# Test 3: Normalizing Flows & Diffusion Schedulers
# ---------------------------------------------------------------------------
def test_flow_matching_and_diffusion():
    from oxideon.flow import RealNVP
    from oxideon.diffusion import DDPMScheduler, SinusoidalPositionalEmbedding
    from oxideon.diffusion_sampler import (
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler,
        DDIMScheduler,
    )

    dim = 4
    num_samples = 8

    # 1. RealNVP
    flow = RealNVP(in_features=dim, num_layers=2, hidden_dim=16)
    x = ox.randn([num_samples, dim])
    z, log_det = flow(x)
    assert z.shape == (num_samples, dim)
    assert log_det.shape == (num_samples,)
    x_rec = flow.inverse(z)
    assert x_rec.shape == (num_samples, dim)

    # 2. DDPMScheduler
    ddpm = DDPMScheduler(num_train_timesteps=100)
    assert len(ddpm.alphas_cumprod) == 100
    pos_emb = SinusoidalPositionalEmbedding(dim=16)
    emb = pos_emb(ox.from_numpy(np.array([10, 50], dtype=np.int64)))
    assert emb.shape == (2, 16)

    # 3. High-Order ODE Schedulers
    schedulers = [
        EulerDiscreteScheduler(num_train_timesteps=100),
        DPMSolverMultistepScheduler(num_train_timesteps=100),
        DDIMScheduler(num_train_timesteps=100),
    ]

    for s in schedulers:
        s.set_timesteps(num_inference_steps=5)
        latents = ox.randn([2, dim])
        noise_pred = ox.randn([2, dim])
        t_val = int(s.timesteps[0])
        step_res = s.step(noise_pred, t_val, latents)
        out = step_res.prev_sample if hasattr(step_res, "prev_sample") else step_res
        assert out.shape == (2, dim)


# ---------------------------------------------------------------------------
# Test 4: Geometric Deep Learning & Molecular GNNs
# ---------------------------------------------------------------------------
def test_graph_neural_and_molecular():
    from oxideon.protein import (
        amino_acid_to_onehot,
        compute_contact_map,
        kabsch_rmsd,
        EGNNCoordinateUpdate,
    )
    from oxideon.graph_matching import gromov_wasserstein_distance
    from oxideon.gnn_ext import GraphSAGE

    # 1. Sequence one-hot
    seq = "ACDEFGHIKLMN"
    onehot = amino_acid_to_onehot(seq)
    assert onehot.shape == (len(seq), 20)

    # 2. Contact map & Kabsch RMSD
    coords = ox.from_numpy(np.random.randn(len(seq), 3).astype(np.float32) * 5.0)
    cmap = compute_contact_map(coords, threshold_angstrom=10.0)
    assert cmap.shape == (len(seq), len(seq))

    rmsd = kabsch_rmsd(coords, coords)
    assert abs(rmsd) < 1e-4

    # 3. EGNN update
    egnn = EGNNCoordinateUpdate(node_dim=8, hidden_dim=16)
    h = ox.randn([len(seq), 8])
    h_new, coords_new = egnn(h, coords)
    assert h_new.shape == h.shape
    assert coords_new.shape == coords.shape

    # 4. Gromov-Wasserstein
    c1 = from_numpy(np.eye(4, dtype=np.float32))
    c2 = from_numpy(np.eye(4, dtype=np.float32))
    gw, plan = gromov_wasserstein_distance(c1, c2, num_iters=5)
    assert float(gw.to_numpy().flat[0]) >= 0.0

    # 5. GraphSAGE
    sage = GraphSAGE(in_channels=20, out_channels=4, aggr="mean")
    logits = sage(onehot, cmap)
    assert logits.shape == (len(seq), 4)


# ---------------------------------------------------------------------------
# Test 5: Unified Hardware Kernels
# ---------------------------------------------------------------------------
def test_hardware_kernels_all():
    from oxideon.kernels import inspect_kernel

    # 1. Multi-backend kernel emission
    ops = ["flash_attention", "rmsnorm", "swiglu", "rope", "q4_gemv", "matmul"]
    for op in ops:
        em = inspect_kernel(op)
        assert set(em.keys()) == {"msl", "ptx", "cuda", "wgsl", "spirv_disasm"}

    # 2. Fused RoPE
    q = ox.randn([2, 2, 8, 16])
    cos = ox.ones([8, 8])
    sin = ox.zeros([8, 8])
    q_rot = ox.fused_rope(q, cos, sin)
    assert q_rot.shape == q.shape

    # 3. Fused Add + RMSNorm
    x = ox.randn([2, 16])
    r = ox.randn([2, 16])
    w = ox.ones([16])
    normed, added = ox.fused_add_rmsnorm(x, r, w)
    assert normed.shape == x.shape
    assert added.shape == r.shape

    # 4. Fused SwiGLU
    gate = ox.randn([2, 32])
    up = ox.randn([2, 32])
    sw = ox.fused_swiglu(gate, up)
    assert sw.shape == (2, 32)

    # 5. Fused SDPA & FlashAttention
    k = ox.randn([2, 2, 8, 16])
    v = ox.randn([2, 2, 8, 16])
    sdpa = ox.fused_sdpa(q, k, v, is_causal=True)
    fa = ox.flash_attention_tiled(q, k, v, is_causal=True)
    assert sdpa.shape == q.shape
    assert fa.shape == q.shape

    # 6. MLX 4-bit Quantized MatMul
    w_full = ox.randn([32, 16])
    w_q4, scales, biases = ox.mlx_quantize_affine_q4(w_full, group_size=16)
    out_q4 = ox.mlx_quantized_matmul_q4(x, w_q4, scales, biases, group_size=16)
    assert out_q4.shape == (2, 32)

    # 7. Paged KV Attention
    q_dec = ox.randn([2, 2, 16])
    k_tab = ox.randn([4, 8, 2, 16])
    v_tab = ox.randn([4, 8, 2, 16])
    tables = [[0, 1], [2, 3]]
    lens = [12, 14]
    paged = ox.paged_kv_attention(q_dec, k_tab, v_tab, tables, lens)
    assert paged.shape == (2, 2, 16)
