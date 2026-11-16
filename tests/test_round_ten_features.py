"""Comprehensive Test Suite for 52 New Frontier Features and 10 Flagship Systems."""

import math
import numpy as np
import pytest

import oxideon as ox
from oxideon.nn import Parameter, Linear, Module


# ==============================================================================
# Flagship 1: Radix Prefix Cache
# ==============================================================================
def test_flagship_radix_prefix_cache():
    cache = ox.RadixPrefixCache(block_size=4)
    prompt_a = [101, 102, 103, 104, 201, 202]
    blocks_a = [1, 2]
    cache.insert(prompt_a, blocks_a)

    matched_tokens, matched_blocks = cache.match_prefix([101, 102, 103, 104, 201, 202, 301, 302])
    assert matched_blocks == blocks_a
    assert matched_tokens == prompt_a

    prompt_b = [101, 102, 103, 104, 999]
    blocks_b = [1, 3]
    cache.insert(prompt_b, blocks_b)

    matched_tokens_b, matched_blocks_b = cache.match_prefix(prompt_b)
    assert matched_tokens_b == prompt_b
    assert cache.total_cached_tokens >= 6


# ==============================================================================
# Flagship 2: Monte Carlo Tree Search (MCTS)
# ==============================================================================
def test_flagship_mcts_reasoning():
    config = ox.MCTSConfig(num_simulations=15, c_puct=1.414)
    
    policy_fn = lambda s: [(10, 0.3), (42, 0.5), (99, 0.2)]
    value_fn = lambda s: 0.5 + (0.5 if 42 in s else 0.0)
    is_terminal_fn = lambda s: len(s) >= 6

    mcts = ox.MCTS(config)
    root = mcts.search([1, 2, 3], policy_fn, value_fn, is_terminal_fn)
    best_token = mcts.best_action(root)
    assert best_token in (10, 42, 99)
    assert root.visits == 15
    probs = mcts.action_probabilities(root)
    assert len(probs) >= 1


# ==============================================================================
# Flagship 3: Graph Transformer
# ==============================================================================
def test_flagship_graph_transformer():
    num_nodes = 5
    node_dim = 16
    edge_dim = 8
    
    model = ox.GraphTransformer(
        in_dim=node_dim,
        hidden_dim=node_dim,
        out_dim=2,
        edge_dim=edge_dim,
        num_layers=2,
        num_heads=2,
    )
    
    x = ox.from_numpy(np.random.randn(num_nodes, node_dim).astype(np.float32))
    edge_attr = ox.from_numpy(np.random.randn(num_nodes, num_nodes, edge_dim).astype(np.float32))
    
    node_out = model(x, edge_attr=edge_attr)
    assert node_out.shape == [num_nodes, 2]


# ==============================================================================
# Flagship 4: BitNet b1.58 Ternary Linear Layer
# ==============================================================================
def test_flagship_bitnet_ternary():
    in_features = 32
    out_features = 16
    bit_linear = ox.BitLinear158b(in_features, out_features)
    
    x = ox.from_numpy(np.random.randn(4, in_features).astype(np.float32) * 5.0)
    out = bit_linear(x)
    assert out.shape == [4, out_features]
    
    # Check weight quantizer
    w = np.random.randn(10, 10).astype(np.float32)
    w_q, gamma = ox.WeightQuantizerSTE.forward(w)
    unique_vals = set(np.unique(w_q))
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
    assert gamma > 0.0


# ==============================================================================
# Flagship 5: Diffusion Schedulers (Euler, DPMSolver, DDIM)
# ==============================================================================
def test_flagship_diffusion_schedulers():
    sample = ox.from_numpy(np.random.randn(2, 4).astype(np.float32))
    model_output = ox.from_numpy(np.random.randn(2, 4).astype(np.float32))

    # Euler discrete
    euler = ox.EulerDiscreteScheduler(num_train_timesteps=1000)
    euler.set_timesteps(num_inference_steps=20)
    assert len(euler.timesteps) == 20
    step_euler = euler.step(model_output, int(euler.timesteps[0]), sample)
    assert step_euler.shape == [2, 4]

    # DPM-Solver++
    dpm = ox.DPMSolverMultistepScheduler(num_train_timesteps=1000)
    dpm.set_timesteps(num_inference_steps=10)
    assert len(dpm.timesteps) == 10
    step_dpm = dpm.step(model_output, int(dpm.timesteps[0]), sample)
    assert step_dpm.shape == [2, 4]

    # DDIM
    ddim = ox.DDIMScheduler(num_train_timesteps=1000)
    ddim.set_timesteps(num_inference_steps=10)
    assert len(ddim.timesteps) == 10
    step_ddim = ddim.step(model_output, int(ddim.timesteps[0]), sample)
    assert step_ddim.shape == [2, 4]


# ==============================================================================
# Flagship 6: Audio Codec & Residual Vector Quantization
# ==============================================================================
def test_flagship_audio_codec_rvq():
    num_quantizers = 3
    num_embeddings = 32
    dim = 16
    rvq = ox.AudioResidualVQ(num_quantizers=num_quantizers, num_embeddings=num_embeddings, embedding_dim=dim)
    
    z = ox.from_numpy(np.random.randn(2, 10, dim).astype(np.float32))
    quantized, indices, loss = rvq(z)
    assert quantized.shape == [2, 10, dim]
    assert len(indices) == num_quantizers
    assert indices[0].shape == (2, 10)
    assert float(loss) >= 0.0

    enc = ox.AudioEncoder1D(hidden_dim=dim, downsample_rate=4)
    dec = ox.AudioDecoder1D(hidden_dim=dim, upsample_rate=4)
    audio = ox.from_numpy(np.random.randn(2, 64).astype(np.float32))
    z_enc = enc(audio)
    assert z_enc.shape == [2, 16, dim]
    audio_rec = dec(z_enc)
    assert audio_rec.shape == [2, 64]


# ==============================================================================
# Flagship 7: Adaptive Conformal Prediction Sets & CQR
# ==============================================================================
def test_flagship_adaptive_conformal():
    # APS
    aps = ox.AdaptivePredictionSets(alpha=0.1)
    cal_probs = np.array([[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.05, 0.25, 0.7]])
    cal_labels = np.array([0, 1, 2])
    aps.calibrate(cal_probs, cal_labels)
    assert aps.q_hat > 0.0
    
    test_probs = np.array([[0.85, 0.10, 0.05], [0.33, 0.33, 0.34]])
    sets = aps.predict(test_probs)
    assert 0 in sets[0]
    assert len(sets[1]) >= 1

    # CQR
    cqr = ox.ConformalizedQuantileRegression(alpha=0.1)
    y_cal = np.array([1.5, 2.8, 3.1, 4.2])
    q_low = np.array([1.0, 2.0, 3.0, 4.0])
    q_high = np.array([2.0, 3.0, 4.0, 5.0])
    cqr.calibrate(y_cal, q_low, q_high)
    low_adj, high_adj = cqr.predict(q_low, q_high)
    assert np.all(high_adj >= low_adj)

    # Mondrian
    mondrian = ox.MondrianConformalGroup(alpha=0.1)
    scores = np.array([0.1, 0.2, 0.5, 0.4])
    groups = np.array([0, 0, 1, 1])
    mondrian.calibrate(scores, groups)
    assert mondrian.get_threshold(0) > 0.0
    assert mondrian.get_threshold(1) > 0.0


# ==============================================================================
# Flagship 8: Synaptic Intelligence & Continual Replay
# ==============================================================================
def test_flagship_synaptic_intelligence():
    class DummyModel(Module):
        def __init__(self):
            super().__init__()
            self.linear = Linear(2, 2)
        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    si = ox.PathIntegralRegularizer(model=model, c=1.0)
    
    x = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2], dtype="f32", requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    si.on_step()
    si.consolidate_task()
    
    surr_loss = si.surrogate_loss()
    assert float(surr_loss.item()) >= 0.0

    replay = ox.MemoryReplayBuffer(capacity=5)
    for i in range(10):
        replay.add(np.array([i], dtype=np.float32), i)
    assert len(replay.samples) == 5
    xs, ys = replay.sample_batch(3)
    assert len(xs) == 3
    assert len(ys) == 3


# ==============================================================================
# Flagship 9: Multimodal VLM Connector & Perceiver Resampler
# ==============================================================================
def test_flagship_multimodal_vlm():
    vis_dim = 32
    llm_dim = 32
    
    # Perceiver resampler
    resampler = ox.PerceiverResampler(visual_dim=vis_dim, text_dim=llm_dim, num_latents=8, num_heads=2, num_layers=2)
    vis_tokens = ox.from_numpy(np.random.randn(2, 16, vis_dim).astype(np.float32))
    out_tokens = resampler(vis_tokens)
    assert out_tokens.shape == [2, 8, llm_dim]

    # Full connector
    connector = ox.VisionLanguageConnector(visual_dim=vis_dim, text_dim=llm_dim, mode="mlp")
    text_emb = ox.from_numpy(np.random.randn(2, 10, llm_dim).astype(np.float32))
    proj = connector(vis_tokens, text_emb)
    assert proj.shape == [2, 26, llm_dim]


# ==============================================================================
# Flagship 10: Expert Parallel MoE with Capacity & Aux Loss
# ==============================================================================
def test_flagship_expert_parallel_moe():
    in_dim = 16
    hidden_dim = 32
    num_experts = 4
    
    router = ox.TopKRouterWithAuxLoss(hidden_dim=in_dim, num_experts=num_experts, top_k=2)
    x = ox.from_numpy(np.random.randn(8, in_dim).astype(np.float32))
    weights, indices, aux_loss = router(x)
    assert indices.shape == (8, 2)
    assert weights.shape == (8, 2)
    assert float(aux_loss) >= 0.0

    moe = ox.CapacityConstrainedMoE(
        hidden_dim=in_dim,
        expert_dim=hidden_dim,
        num_experts=num_experts,
        top_k=2,
        capacity_factor=1.5,
    )
    y, total_aux = moe(x)
    assert y.shape == [8, in_dim]
    assert float(total_aux) >= 0.0


# ==============================================================================
# Features 11–16: Attention Extensions
# ==============================================================================
def test_attention_extensions():
    # 11. ALiBi
    alibi = ox.AlibiPositionalBias(num_heads=4)
    bias = alibi(8)
    assert bias.shape == [4, 8, 8]

    # 12. 3D RoPE
    rope3d = ox.SpatiotemporalRoPE(dim=12)
    (cos_t, cos_h, cos_w), (sin_t, sin_h, sin_w) = rope3d(2, 3, 3)
    assert cos_t.shape == (2, 2)
    assert cos_h.shape == (3, 2)
    assert cos_w.shape == (3, 2)

    # 13. MQA
    mqa = ox.MultiQueryAttention(hidden_dim=32, num_heads=4)
    q = ox.from_numpy(np.random.randn(2, 6, 32).astype(np.float32))
    out_mqa = mqa(q)
    assert out_mqa.shape == [2, 6, 32]

    # 14. Sliding window
    swa = ox.SlidingWindowAttention(hidden_dim=16, num_heads=2, window_size=3)
    out_swa = swa(ox.from_numpy(np.random.randn(2, 8, 16).astype(np.float32)))
    assert out_swa.shape == [2, 8, 16]

    # 15. Differential attention
    diff_attn = ox.DifferentialAttention(hidden_dim=32, num_heads=4)
    out_diff = diff_attn(ox.from_numpy(np.random.randn(2, 5, 32).astype(np.float32)))
    assert out_diff.shape == [2, 5, 32]

    # 16. LogitLens
    h_layer = ox.from_numpy(np.random.randn(2, 5, 16).astype(np.float32))
    lm_weight = ox.from_numpy(np.random.randn(50, 16).astype(np.float32))
    top_indices, top_logits = ox.LogitLens.inspect_layer(h_layer, lm_weight, top_k=3)
    assert top_indices.shape == (2, 5, 3)
    assert top_logits.shape == (2, 5, 3)


# ==============================================================================
# Features 17–23: Optim Extensions
# ==============================================================================
def test_optimizer_extensions():
    model = ox.nn.Linear(4, 4)
    x = ox.Tensor([1.0] * 8, shape=[2, 4], dtype="f32", requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # 17. Shampoo
    shampoo = ox.ShampooOptimizer(model.parameters(), lr=1e-3)
    shampoo.step()

    # 18. AdEMAMix
    ademamix = ox.AdEMAMix(model.parameters(), lr=1e-3)
    ademamix.step()

    # 19. Prodigy
    prodigy = ox.ProdigyOptimizer(model.parameters(), lr=1.0)
    prodigy.step()

    # 20. Sophia
    sophia = ox.SophiaOptimizer(model.parameters(), lr=1e-3)
    sophia.update_hessian({id(p): np.ones(p.shape, dtype=np.float32) for p in model.parameters()})
    sophia.step()

    # 21. ScheduleFreeSGD
    sf_sgd = ox.ScheduleFreeSGD(model.parameters(), lr=1e-2)
    sf_sgd.step()

    # 22. Lookahead
    lookahead = ox.LookaheadOptimizer(sf_sgd, k=2, alpha=0.5)
    lookahead.step()
    lookahead.step()

    # 23. SGDRScheduler
    class DummyOpt:
        def __init__(self):
            self.lr = 0.1
    d_opt = DummyOpt()
    scheduler = ox.SGDRScheduler(d_opt, t_0=5, t_mult=2)
    for _ in range(6):
        scheduler.step()
    assert d_opt.lr > 0.0


# ==============================================================================
# Features 24–29: Losses Extensions
# ==============================================================================
def test_loss_extensions():
    # 24. DiceLoss
    dice = ox.DiceLoss()
    p = ox.from_numpy(np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32))
    y = ox.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    loss_dice = dice(p, y)
    assert float(loss_dice.to_numpy()[0]) < 0.2

    # 25. AdaptiveFocalLoss
    focal = ox.AdaptiveFocalLoss(gamma=2.0)
    target_idx = ox.from_numpy(np.array([0, 1], dtype=np.int64))
    loss_focal = focal(p, target_idx)
    assert float(loss_focal.to_numpy()[0]) >= 0.0

    # 26. LabelSmoothingCrossEntropy
    smooth_ce = ox.LabelSmoothingCrossEntropy(smoothing=0.1)
    logits = ox.from_numpy(np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32))
    target = ox.from_numpy(np.array([0, 1], dtype=np.int64))
    loss_ce = smooth_ce(logits, target)
    assert float(loss_ce.to_numpy()[0]) > 0.0

    # 27. TripletMarginWithDistanceLoss
    triplet = ox.TripletMarginWithDistanceLoss(margin=1.0)
    a = ox.from_numpy(np.zeros((2, 4), dtype=np.float32))
    pos = ox.from_numpy(np.zeros((2, 4), dtype=np.float32))
    neg = ox.from_numpy(np.ones((2, 4), dtype=np.float32) * 2.0)
    loss_trip = triplet(a, pos, neg)
    assert float(loss_trip.to_numpy()[0]) == 0.0

    # 28. RankNetLoss
    ranknet = ox.RankNetLoss()
    s_i = ox.from_numpy(np.array([1.5, 2.0], dtype=np.float32))
    s_j = ox.from_numpy(np.array([0.5, 0.0], dtype=np.float32))
    targets = ox.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
    loss_rank = ranknet(s_i, s_j, targets)
    assert float(loss_rank.to_numpy()[0]) > 0.0

    # 29. InfoNCELoss
    infonce = ox.InfoNCELoss(temperature=0.1)
    q_rep = ox.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    k_rep = ox.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    loss_nce = infonce(q_rep, k_rep)
    assert float(loss_nce.to_numpy()[0]) < 0.1


# ==============================================================================
# Features 30–35: GNN Extensions
# ==============================================================================
def test_gnn_extensions():
    num_nodes = 4
    x = ox.from_numpy(np.random.randn(num_nodes, 8).astype(np.float32))
    adj = ox.from_numpy(np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32))

    # 30. GatedGraphConv
    ggc = ox.GatedGraphConv(out_channels=8, num_layers=2)
    out_ggc = ggc(x, adj)
    assert out_ggc.shape == [num_nodes, 8]

    # 31. GraphSAGE
    sage = ox.GraphSAGE(in_channels=8, out_channels=16, aggr="mean")
    out_sage = sage(x, adj)
    assert out_sage.shape == [num_nodes, 16]

    # 32. GINConv
    gin = ox.GINConv(in_channels=8, out_channels=16)
    out_gin = gin(x, adj)
    assert out_gin.shape == [num_nodes, 16]

    # 33. LaplacianEigenmaps
    emb = ox.LaplacianEigenmaps.fit_transform(adj, n_components=2)
    assert emb.shape == [num_nodes, 2]

    # 34. AdamicAdarIndex
    scores_aa = ox.AdamicAdarIndex.score_matrix(adj)
    assert scores_aa.shape == [num_nodes, num_nodes]

    # 35. JaccardSimilarityCoeff
    scores_jac = ox.JaccardSimilarityCoeff.score_matrix(adj)
    assert scores_jac.shape == [num_nodes, num_nodes]
    np.testing.assert_allclose(np.diag(scores_jac.to_numpy()), 1.0)


# ==============================================================================
# Features 36–41: ODE / PDE Solvers
# ==============================================================================
def test_gnn_ext_forward_values():
    # Golden values captured from the pre-rewrite NumPy implementation, which was
    # validated against equivalent PyTorch math to ~1e-7. The rewrite must preserve
    # forward values while restoring gradient flow (see below).
    adj = ox.from_numpy(np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32))
    GOLDEN_GATED = [[0.188211, 0.191584, 0.272743, 0.598173, 0.533767, -0.510378, 0.163661, 0.048734], [-0.361291, 0.253774, -0.054024, 0.103207, -0.093505, -0.010011, -0.270555, -0.159908], [0.171889, 0.031574, -0.180671, -0.1506, -1.157119, 0.324948, 0.464376, -0.035625], [0.878002, -0.335008, -0.076451, -0.270948, 0.445382, 0.213969, -0.108209, 0.211943]]
    GOLDEN_SAGE_MEAN = [[0.202064, -0.105344, 0.30328, -0.045358, -0.209079, -0.26425, 0.384804, 0.257699, -0.344912, 0.143734, -0.294555, -0.119373, -0.224389, -0.103273, 0.420801, -0.217454], [0.145067, -0.179766, 0.235017, -0.069667, -0.271922, -0.259045, 0.305754, 0.22845, -0.263258, 0.204885, -0.248225, 0.148359, -0.256118, -0.244444, 0.528728, -0.00449], [0.263763, -0.3231, -0.404688, 0.399856, -0.022972, -0.201112, 0.018963, -0.440256, 0.010531, 0.130027, 0.205256, -0.200698, -0.178935, 0.353021, -0.078074, 0.071094], [0.195911, -0.375634, -0.212697, 0.486153, -0.254698, 0.276817, -0.318438, -0.191423, -0.149595, -0.234156, 0.15772, -0.321568, -0.200917, -0.101871, -0.045896, -0.032374]]
    GOLDEN_GIN = [[-0.260564, -0.263882, 0.07102, 0.245075, 0.337843, -0.13057, -0.7747, -0.52828, -0.852794, -0.504273, -0.407173, 0.668916, -0.373437, -0.517177, 0.389954, -0.312839], [-0.402095, -0.275758, -0.106477, -0.004532, 0.254365, -0.475774, -0.629194, -0.889523, -0.806954, -0.255533, -0.232852, 0.753051, -0.452594, -0.391695, 0.341917, -0.373439], [-0.260564, -0.263882, 0.07102, 0.245075, 0.337843, -0.13057, -0.7747, -0.52828, -0.852794, -0.504273, -0.407173, 0.668916, -0.373437, -0.517177, 0.389954, -0.312839], [-0.252988, -0.276443, -0.325334, -0.46316, 0.067756, -0.328233, -0.039058, -0.573836, 0.009936, 0.375019, -0.108493, 0.342789, -0.457062, -0.160801, -0.058888, -0.039825]]

    ox.manual_seed(0)
    np.random.seed(0)
    x = ox.from_numpy(np.random.randn(4, 8).astype(np.float32))
    np.testing.assert_allclose(ox.GatedGraphConv(out_channels=8, num_layers=2)(x, adj).to_numpy(), GOLDEN_GATED, atol=1e-4)

    ox.manual_seed(0)
    np.random.seed(0)
    x = ox.from_numpy(np.random.randn(4, 8).astype(np.float32))
    np.testing.assert_allclose(ox.GraphSAGE(in_channels=8, out_channels=16, aggr="mean")(x, adj).to_numpy(), GOLDEN_SAGE_MEAN, atol=1e-4)

    ox.manual_seed(0)
    np.random.seed(0)
    x = ox.from_numpy(np.random.randn(4, 8).astype(np.float32))
    np.testing.assert_allclose(ox.GINConv(in_channels=8, out_channels=16)(x, adj).to_numpy(), GOLDEN_GIN, atol=1e-4)


def test_gnn_ext_layers_stay_differentiable():
    np.random.seed(0)
    num_nodes = 4
    x = ox.from_numpy(np.random.randn(num_nodes, 8).astype(np.float32))
    adj = ox.from_numpy(np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32))

    def assert_flowing(module, *lin_names):
        for name in lin_names:
            g = getattr(module, name).weight.grad()
            assert g is not None, name
            arr = g.to_numpy()
            assert np.isfinite(arr).all(), name
            assert np.abs(arr).sum() > 0, name

    # 30. GatedGraphConv: every gate gets a gradient
    ggc = ox.GatedGraphConv(out_channels=8, num_layers=2)
    ggc(x, adj).sum().backward()
    assert_flowing(ggc, "message_linear", "w_z", "w_r", "w_h")

    # 31. GraphSAGE: all three aggregations, normalized and raw
    for aggr in ("mean", "sum", "max"):
        sage = ox.GraphSAGE(in_channels=8, out_channels=16, aggr=aggr)
        sage(x, adj).sum().backward()
        assert_flowing(sage, "lin_self", "lin_neigh")
    sage_raw = ox.GraphSAGE(in_channels=8, out_channels=16, aggr="mean", normalize=False)
    sage_raw(x, adj).sum().backward()
    assert_flowing(sage_raw, "lin_self", "lin_neigh")

    # 32. GINConv: both MLP layers and the (trainable) epsilon
    gin = ox.GINConv(in_channels=8, out_channels=16)
    gin(x, adj).sum().backward()
    assert_flowing(gin, "mlp_1", "mlp_2")
    gin_eps = ox.GINConv(in_channels=8, out_channels=16, train_eps=True)
    gin_eps(x, adj).sum().backward()
    assert_flowing(gin_eps, "mlp_1", "mlp_2")
    ge = gin_eps.eps.grad()
    assert ge is not None and np.isfinite(ge.to_numpy()).all()


def test_ode_pde_solvers():
    # 36. RK4
    def f_linear(t, y):
        return -y
    t_hist, y_hist = ox.RungeKutta4.integrate(f_linear, np.array([1.0]), (0.0, 1.0), steps=10)
    assert len(t_hist) == 11
    assert math.isclose(y_hist.to_numpy()[-1, 0], math.exp(-1.0), rel_tol=1e-3)

    # 37. DOPRI5
    t_dp, y_dp = ox.DormandPrince54.integrate(f_linear, np.array([1.0]), (0.0, 1.0))
    assert math.isclose(y_dp.to_numpy()[-1, 0], math.exp(-1.0), rel_tol=1e-3)

    # 38. SymplecticEuler
    def grad_v(q):
        return q
    def grad_t(p):
        return p
    t_s, q_s, p_s = ox.SymplecticEuler.integrate(grad_v, grad_t, np.array([1.0]), np.array([0.0]), (0.0, 2.0 * math.pi), steps=100)
    # Energy E = 0.5 * p^2 + 0.5 * q^2 ~ 0.5
    energy_end = 0.5 * (p_s.to_numpy()[-1, 0] ** 2 + q_s.to_numpy()[-1, 0] ** 2)
    assert math.isclose(energy_end, 0.5, rel_tol=0.1)

    # 39. Burgers 1D
    u0 = np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False)).astype(np.float32)
    u_burgers = ox.Burgers1DEquation.solve(u0, dx=0.1, dt=0.001, nu=0.05, num_steps=20)
    assert u_burgers.shape == [20]

    # 40. Wave 1D
    u_wave = ox.Wave1DEquation.solve(u0, c=1.0, dx=0.1, dt=0.05, num_steps=10)
    assert u_wave.shape == [20]

    # 41. Diffusion 1D
    u_diff = ox.Diffusion1DEquation.solve(u0, alpha=0.01, dx=0.1, dt=0.01, num_steps=10)
    assert u_diff.shape == [20]


# ==============================================================================
# Features 42–46: Quantization, Sparsity & Compression
# ==============================================================================
def test_quant_compression():
    # 42. SmoothQuant
    w = np.random.randn(8, 16).astype(np.float32)
    x = np.random.randn(4, 16).astype(np.float32) * 5.0
    w_sm, x_sm, scale = ox.SmoothQuant.smooth_layer(w, x, alpha=0.5)
    # Check that product X * W^T is approximately preserved
    orig_prod = np.matmul(x, w.T)
    sm_prod = np.matmul(x_sm.to_numpy(), w_sm.to_numpy().T)
    np.testing.assert_allclose(orig_prod, sm_prod, rtol=1e-4)

    # 43. GPTQQuantizer
    w_gptq = ox.GPTQQuantizer.quantize(w, num_bits=4)
    assert w_gptq.shape == [8, 16]

    # 44. SparseMagnitudePruner
    pruned_unstr, mask_unstr = ox.SparseMagnitudePruner.prune_unstructured(w, sparsity=0.5)
    assert math.isclose(np.mean(mask_unstr.to_numpy() == 0.0), 0.5, abs_tol=0.05)
    
    pruned_nm, mask_nm = ox.SparseMagnitudePruner.prune_n_m_structured(w, n=2, m=4)
    assert math.isclose(np.mean(mask_nm.to_numpy() == 1.0), 0.5, abs_tol=0.05)

    # 45. VectorQuantizedVAE
    vqvae = ox.VectorQuantizedVAE(in_dim=16, hidden_dim=8, num_embeddings=16)
    x_in = ox.from_numpy(np.random.randn(4, 16).astype(np.float32))
    recon, total_loss, metrics = vqvae(x_in)
    assert recon.shape == [4, 16]
    assert float(total_loss.to_numpy().ravel()[0]) >= 0.0
    assert metrics["perplexity"] > 0.0

    # 46. HQQQuantizer
    q, s, zp = ox.HQQQuantizer.quantize(w, num_bits=4)
    w_rec = ox.HQQQuantizer.dequantize(q, s, zp)
    assert w_rec.shape == [8, 16]
    assert np.mean((w - w_rec.to_numpy()) ** 2) < 0.2


# ==============================================================================
# Features 47–52: Signal Metrics & Processing
# ==============================================================================
def test_signal_metrics():
    # 47. MelFilterbank
    fb = ox.MelFilterbank.create(sr=16000, n_fft=512, n_mels=40)
    assert fb.shape == [40, 257]

    # 48. FastDTW
    seq1 = np.sin(np.linspace(0, 2 * np.pi, 20))
    seq2 = np.sin(np.linspace(0, 2 * np.pi, 25) + 0.1)
    dist = ox.FastDTW.distance(seq1, seq2, radius=5)
    assert dist >= 0.0

    # 49. ExponentialMovingAverage
    p = Parameter(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    ema = ox.ExponentialMovingAverage([p], decay=0.9, warm_up=2)
    p.assign(ox.from_numpy(np.array([2.0, 3.0, 4.0], dtype=np.float32)))
    ema.update()
    ema.apply_shadow()
    assert float(p.to_numpy()[0]) < 2.0
    ema.restore()
    assert float(p.to_numpy()[0]) == 2.0

    # 50. Autocorrelation1D
    t = np.linspace(0, 1.0, 500, endpoint=False)
    # Sine wave with period 50 samples
    sig = np.sin(2.0 * np.pi * 10 * t).astype(np.float32)
    pitch = ox.Autocorrelation1D.detect_pitch_period(sig, min_lag=20, max_lag=80)
    assert abs(pitch - 50) <= 2

    # 51. PSNR
    img_true = np.ones((16, 16), dtype=np.float32)
    img_pred = img_true + 0.01 * np.random.randn(16, 16).astype(np.float32)
    psnr = ox.PeakSignalToNoiseRatio.compute(img_true, img_pred, data_range=1.0)
    assert psnr > 30.0

    # 52. SSIM
    ssim = ox.StructuralSimilarityIndex.compute(img_true, img_pred, data_range=1.0)
    assert ssim > 0.85
