"""
Tests for Round 7 advanced features in Oxideon:
1. ox.graph_gen (Erdos-Renyi, Barabasi-Albert, Watts-Strogatz, SBM)
2. ox.bayes (BayesLinear, ELBOLoss, BayesPredictor)
3. ox.energy (LangevinSampler, ContrastiveDivergenceLoss, FreeEnergyScorer)
4. ox.audio_synth (SineGenerator, ChirpGenerator, BiquadFilter, EnvelopeADSR)
5. ox.geometry (PointNetEncoder, Farthest Point Sampling, Chamfer Distance)
6. ox.causal_discovery (NOTEARS, correlation_skeleton, is_dag, topological_sort)
7. ox.federated (FedAvg, compute_proximal_loss, FederatedServer, FederatedClient)
8. ox.sparse_attn (BigBird sparse attention, sliding window, SparseMultiheadAttention)
9. ox.rl_continuous (GaussianActor, TwinCritic, SACLoss)
10. ox.interpret (TCAV, IntegratedHessians, difference_from_reference_multipliers)
"""
import numpy as np
import pytest
import oxideon as ox
from oxideon import (
    graph_gen,
    bayes,
    energy,
    audio_synth,
    geometry,
    causal_discovery,
    federated,
    sparse_attn,
    rl_continuous,
    interpret,
)

def test_graph_generation():
    # 1. Erdos-Renyi
    adj_er, g_er = graph_gen.erdos_renyi_graph(num_nodes=10, p=0.4, seed=42)
    assert adj_er.shape == [10, 10]
    assert g_er.x.shape == [10, 10]
    assert g_er.edge_index.shape[0] == 2

    # 2. Barabasi-Albert
    adj_ba, g_ba = graph_gen.barabasi_albert_graph(num_nodes=15, m=2, seed=42)
    assert adj_ba.shape == [15, 15]
    # Each new node (15 - 3 = 12) added 2 edges => total degree sum is non-zero
    assert np.sum(adj_ba.to_numpy()) > 0

    # 3. Watts-Strogatz
    adj_ws, g_ws = graph_gen.watts_strogatz_graph(num_nodes=12, k=4, p=0.1, seed=42)
    assert adj_ws.shape == [12, 12]

    # 4. Stochastic Block Model
    prob_mat = np.array([[0.8, 0.05], [0.05, 0.7]], dtype=np.float32)
    adj_sbm, g_sbm, labels = graph_gen.stochastic_block_model(block_sizes=[8, 8], prob_matrix=prob_mat, seed=42)
    assert adj_sbm.shape == [16, 16]
    assert labels.shape == [16]

def test_bayesian_neural_network():
    layer = bayes.BayesLinear(in_features=4, out_features=2, prior_std=1.0)
    x = ox.randn([5, 4])
    
    # Stochastic forward
    out1 = layer(x, sample=True)
    out2 = layer(x, sample=True)
    assert out1.shape == [5, 2]
    assert not np.allclose(out1.to_numpy(), out2.to_numpy())

    # Deterministic forward
    out_det1 = layer(x, sample=False)
    out_det2 = layer(x, sample=False)
    assert np.allclose(out_det1.to_numpy(), out_det2.to_numpy())

    # Analytical KL divergence
    kl = layer.kl_divergence()
    assert float(kl.to_numpy().flat[0]) > 0.0

    # ELBO loss
    target = ox.zeros([5, 2])
    elbo = bayes.ELBOLoss(beta=0.01, criterion="mse")(out1, target, kl)
    assert float(elbo.to_numpy().flat[0]) > 0.0

    # Monte Carlo predictor
    predictor = bayes.BayesPredictor(layer, num_samples=10)
    mean_p, var_p = predictor.predict(x)
    assert mean_p.shape == [5, 2]
    assert var_p.shape == [5, 2]
    assert np.all(var_p.to_numpy() >= 0.0)

def test_energy_based_models():
    # Define quadratic energy function E(x) = 0.5 * ||x||^2
    def quadratic_energy(x: ox.Tensor) -> ox.Tensor:
        arr = x.to_numpy()
        e = 0.5 * np.sum(arr ** 2)
        return ox.from_numpy(np.array(e, dtype=np.float32))

    sampler = energy.LangevinSampler(step_size=0.05, num_steps=30, noise_scale=0.01)
    init_x = ox.from_numpy(np.array([5.0, -5.0], dtype=np.float32))
    sampled_x = sampler.sample(quadratic_energy, init_x)
    
    # SGLD should move initial point closer to the energy minimum (0, 0)
    assert np.linalg.norm(sampled_x.to_numpy()) < np.linalg.norm(init_x.to_numpy())

    # Contrastive divergence
    cd_loss = energy.ContrastiveDivergenceLoss(reg_weight=0.01)
    e_pos = ox.from_numpy(np.array([1.0, 1.2], dtype=np.float32))
    e_neg = ox.from_numpy(np.array([3.0, 3.5], dtype=np.float32))
    l = cd_loss(e_pos, e_neg)
    assert float(l.to_numpy().flat[0]) < 0.0 # Pos has lower energy than neg

    # Free energy scoring
    scorer = energy.FreeEnergyScorer(temperature=1.0)
    logits = ox.from_numpy(np.array([[10.0, 2.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32))
    free_e = scorer.compute_energy(logits)
    # High confidence has lower free energy
    assert free_e.to_numpy()[0] < free_e.to_numpy()[1]

def test_audio_synth_and_dsp():
    # Sine generator
    sine_gen = audio_synth.SineGenerator(sample_rate=8000)
    tone = sine_gen.generate(freq_hz=440.0, duration_s=0.05)
    assert tone.shape == [400]

    # Chirp generator
    chirp_gen = audio_synth.ChirpGenerator(sample_rate=8000)
    chirp = chirp_gen.generate(f0=100.0, f1=1000.0, duration_s=0.05)
    assert chirp.shape == [400]

    # Biquad Filter (Lowpass)
    filt = audio_synth.BiquadFilter(filter_type="lowpass", cutoff_hz=500.0, sample_rate=8000)
    filtered = filt.apply(tone)
    assert filtered.shape == [400]

    # ADSR Envelope
    adsr = audio_synth.EnvelopeADSR(attack_s=0.01, decay_s=0.01, sustain_level=0.5, release_s=0.01, sample_rate=8000)
    env = adsr.generate(total_duration_s=0.05)
    assert env.shape == [400]
    assert np.max(env.to_numpy()) <= 1.0
    assert np.min(env.to_numpy()) >= 0.0

def test_geometry_3d_pointclouds():
    pts = ox.randn([4, 50, 3])
    encoder = geometry.PointNetEncoder(in_channels=3, hidden_dim=32, global_dim=64)
    global_feat = encoder(pts)
    assert global_feat.shape == [4, 64]

    # Single point cloud input
    single_pts = ox.randn([50, 3])
    single_feat = encoder(single_pts)
    assert single_feat.shape == [64]

    # Farthest point sampling (FPS)
    fps_idx = geometry.farthest_point_sampling(single_pts, num_samples=10)
    assert fps_idx.shape == [10]
    assert len(np.unique(fps_idx.to_numpy())) == 10

    # Chamfer distance
    p1 = ox.randn([20, 3])
    p2 = ox.randn([30, 3])
    cd = geometry.chamfer_distance(p1, p2)
    assert float(cd.to_numpy().flat[0]) > 0.0

    # 3D SO(3) random rotation
    rot_pts = geometry.random_rotation_3d(p1, seed=42)
    assert rot_pts.shape == [20, 3]

def test_causal_discovery():
    # Generate synthetic linear SEM data: X0 -> X1 -> X2
    rng = np.random.default_rng(42)
    n = 200
    x0 = rng.normal(0, 1, n)
    x1 = 0.8 * x0 + rng.normal(0, 0.2, n)
    x2 = 0.7 * x1 + rng.normal(0, 0.2, n)
    X = np.stack([x0, x1, x2], axis=1).astype(np.float32)
    X_t = ox.from_numpy(X)

    # Correlation skeleton
    skel = causal_discovery.correlation_skeleton(X_t, threshold=0.3)
    assert skel.shape == [3, 3]

    # NOTEARS continuous DAG learning
    w_est = causal_discovery.notears_linear(X_t, lambda1=0.01, max_iter=30, w_threshold=0.1)
    assert w_est.shape == [3, 3]

    # Acyclicity check
    dag_adj = ox.from_numpy(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32))
    assert causal_discovery.is_dag(dag_adj) is True

    cycle_adj = ox.from_numpy(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32))
    assert causal_discovery.is_dag(cycle_adj) is False

    # Topological sort
    topo = causal_discovery.topological_sort(dag_adj)
    assert topo == [0, 1, 2]

def test_federated_learning():
    model1 = ox.nn.Sequential(ox.nn.Linear(4, 2))
    model2 = ox.nn.Sequential(ox.nn.Linear(4, 2))

    client1 = federated.FederatedClient(client_id="c1", model=model1)
    client2 = federated.FederatedClient(client_id="c2", model=model2)

    server = federated.FederatedServer(initial_model=model1)

    # Simulate client updates
    updates = [client1.get_weights(), client2.get_weights()]
    counts = [100, 200]
    
    server.aggregate(updates, counts)
    assert server.round == 1

    # Synchronize client model with server
    server.sync_model(model1)

    # Compute proximal penalty
    prox = federated.compute_proximal_loss(client1.get_weights(), server.get_global_weights(), mu=0.1)
    assert prox >= 0.0

def test_sparse_attention():
    seq_len = 16
    mask_sw = sparse_attn.build_sliding_window_mask(seq_len, window_radius=2)
    assert mask_sw.shape == (16, 16)
    assert mask_sw[0, 2] == 1.0
    assert mask_sw[0, 5] == 0.0

    mask_bb = sparse_attn.build_bigbird_mask(seq_len, num_global=2, window_radius=2, num_random=2, seed=42)
    assert mask_bb.shape == (16, 16)
    # Global tokens attend to all
    assert np.all(mask_bb[:2, :] == 1.0)
    assert np.all(mask_bb[:, :2] == 1.0)

    # SparseMultiheadAttention
    attn = sparse_attn.SparseMultiheadAttention(embed_dim=16, num_heads=2, window_radius=2, num_global=2)
    x = ox.randn([2, 8, 16])
    out = attn(x)
    assert out.shape == [2, 8, 16]

def test_rl_continuous():
    actor = rl_continuous.GaussianActor(state_dim=6, action_dim=2, hidden_dim=16)
    state = ox.randn([4, 6])

    # Stochastic action & log prob
    act, log_prob = actor(state, deterministic=False)
    assert act.shape == [4, 2]
    assert log_prob.shape == [4]
    assert np.all(np.abs(act.to_numpy()) <= 1.0)

    # Deterministic action
    act_det, _ = actor(state, deterministic=True)
    assert act_det.shape == [4, 2]

    # Twin critic
    critic = rl_continuous.TwinCritic(state_dim=6, action_dim=2, hidden_dim=16)
    q1, q2 = critic(state, act)
    assert q1.shape == [4, 1]
    assert q2.shape == [4, 1]
    min_q = critic.min_q(state, act)
    assert min_q.shape == [4, 1]

    # SAC loss
    loss_calc = rl_continuous.SACLoss(alpha=0.2, gamma=0.99)
    target_q = ox.zeros([4, 1])
    c_loss = loss_calc.critic_loss(q1, q2, target_q)
    assert float(c_loss.to_numpy().flat[0]) >= 0.0

    a_loss = loss_calc.actor_loss(min_q, log_prob)
    assert isinstance(float(a_loss.to_numpy().flat[0]), float)

def test_model_interpretability():
    # Dummy classifier function
    def score_fn(x: ox.Tensor) -> ox.Tensor:
        arr = x.to_numpy()
        val = np.sum(arr ** 2) + 0.5 * arr[0] * arr[1]
        return ox.from_numpy(np.array(val, dtype=np.float32))

    # TCAV
    c_act = ox.randn([20, 8])
    r_act = ox.randn([20, 8])
    cav = interpret.TCAV.train_cav(c_act, r_act)
    assert cav.v.shape == (8,)

    grads = ox.randn([10, 8])
    deriv = cav.directional_derivative(grads)
    assert deriv.shape == [10]
    score = cav.tcav_score(grads)
    assert 0.0 <= score <= 1.0

    # Integrated Hessians (pairwise feature interaction)
    ih = interpret.IntegratedHessians(steps=5)
    x = ox.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    interactions = ih.attribute(score_fn, x)
    assert interactions.shape == [2, 2]

    # Difference from reference multipliers
    ref = ox.from_numpy(np.array([0.0, 0.0], dtype=np.float32))
    mults = interpret.difference_from_reference_multipliers(score_fn, x, ref)
    assert mults.shape == [2]
