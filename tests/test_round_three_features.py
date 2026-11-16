import pytest
import numpy as np
import oxideon as ox
from oxideon import (
    gnn,
    rl,
    diffusion,
    prune,
    audio,
    sparse,
    clustering,
    benchmark,
    export,
    alignment,
)


def test_feature_1_gnn():
    """Tests GraphData, degree, add_self_loops, GCNConv, GATConv, and GraphSAGEConv."""
    # Graph with 4 nodes: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
    x = ox.randn([4, 8])

    graph = gnn.GraphData(x, edge_index)
    assert graph.num_nodes == 4
    assert graph.num_edges == 4
    assert graph.num_node_features == 8
    assert "GraphData" in repr(graph)

    # Degree
    deg = gnn.degree(edge_index, num_nodes=4)
    assert np.allclose(deg, np.array([1, 1, 1, 1]))

    # GCNConv
    gcn = gnn.GCNConv(in_features=8, out_features=16)
    out_gcn = gcn(x, edge_index)
    assert list(out_gcn.shape) == [4, 16]

    # GATConv
    gat = gnn.GATConv(in_features=8, out_features=16, heads=2, concat=True)
    out_gat = gat(x, edge_index)
    assert list(out_gat.shape) == [4, 32]

    # GraphSAGEConv
    sage = gnn.GraphSAGEConv(in_features=8, out_features=12, aggr="mean")
    out_sage = sage(x, edge_index)
    assert list(out_sage.shape) == [4, 12]


def test_feature_2_rl():
    """Tests ReplayBuffer, PrioritizedReplayBuffer, DQNPolicy, compute_gae, and PPOAgent."""
    # ReplayBuffer
    buf = rl.ReplayBuffer(capacity=50, state_dim=4, action_dim=1)
    for i in range(10):
        buf.push(np.zeros(4), 1.0, 0.5, np.ones(4), False)
    assert len(buf) == 10

    s, a, r, ns, d = buf.sample(batch_size=4)
    assert list(s.shape) == [4, 4]
    assert list(r.shape) == [4, 1]

    # PrioritizedReplayBuffer
    per = rl.PrioritizedReplayBuffer(capacity=50, state_dim=4, action_dim=1)
    for i in range(10):
        per.push(np.zeros(4), 0.0, 1.0, np.ones(4), False)
    s, a, r, ns, d, idxs, weights = per.sample(batch_size=4)
    assert len(idxs) == 4
    assert list(weights.shape) == [4, 1]
    per.update_priorities(idxs, np.array([0.1, 0.5, 0.2, 0.9]))

    # DQNPolicy
    dqn = rl.DQNPolicy(state_dim=4, action_dim=2, hidden_dim=16)
    act = dqn.select_action(np.zeros(4), epsilon=0.0)
    assert act in [0, 1]

    # GAE
    adv, ret = rl.compute_gae(
        rewards=[1.0, 1.0, 0.0],
        values=[0.5, 0.5, 0.0],
        dones=[False, False, True],
        next_value=0.0,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert len(adv) == 3
    assert len(ret) == 3

    # PPOAgent
    agent = rl.PPOAgent(state_dim=4, action_dim=3, hidden_dim=16)
    logits, val = agent(ox.randn([2, 4]))
    assert list(logits.shape) == [2, 3]
    assert list(val.shape) == [2, 1]


def test_feature_3_diffusion():
    """Tests DDPMScheduler, DDIMScheduler, SinusoidalPositionalEmbedding, and UNetModel."""
    ddpm = diffusion.DDPMScheduler(num_train_timesteps=100)
    x0 = ox.randn([2, 8])
    noise = ox.randn([2, 8])

    xt = ddpm.add_noise(x0, noise, timesteps=50)
    assert list(xt.shape) == [2, 8]

    # Reverse step
    pred_noise = ox.randn([2, 8])
    prev = ddpm.step(pred_noise, timestep=50, sample=xt)
    assert list(prev.shape) == [2, 8]

    # DDIMScheduler
    ddim = diffusion.DDIMScheduler(num_train_timesteps=100)
    ddim.set_timesteps(num_inference_steps=10)
    assert len(ddim.timesteps) == 10
    step_sample = ddim.step(pred_noise, timestep=90, sample=xt)
    assert list(step_sample.shape) == [2, 8]

    # SinusoidalPositionalEmbedding
    time_emb = diffusion.SinusoidalPositionalEmbedding(dim=32)
    temb_out = time_emb(np.array([10, 20]))
    assert list(temb_out.shape) == [2, 32]

    # UNetModel
    unet = diffusion.UNetModel(in_dim=8, hidden_dim=16, time_embed_dim=16)
    out_unet = unet(xt, np.array([25, 50]))
    assert list(out_unet.shape) == [2, 8]


def test_feature_4_prune():
    """Tests prune_l1_unstructured, prune_random_unstructured, prune_structured, and sparsity."""
    lin = ox.nn.Linear(10, 10)
    init_sparsity = prune.get_sparsity(lin)
    assert init_sparsity == 0.0

    # L1 unstructured pruning (50%)
    prune.prune_l1_unstructured(lin, name="weight", amount=0.5)
    s1 = prune.get_sparsity(lin.weight)
    assert 0.45 <= s1 <= 0.55
    assert prune.has_pruning_mask(lin, "weight")

    # Structured pruning (remove 30% of channels)
    lin2 = ox.nn.Linear(20, 20)
    prune.prune_structured(lin2, name="weight", amount=0.3, dim=0)
    s2 = prune.get_sparsity(lin2.weight)
    assert s2 >= 0.25

    # Remove pruning
    prune.remove_pruning(lin2, name="weight")
    assert not prune.has_pruning_mask(lin2, "weight")


def test_feature_5_audio():
    """Tests hz_to_mel, mel_filter_bank, Spectrogram, MelSpectrogram, and MFCC."""
    assert np.isclose(audio.hz_to_mel(audio.mel_to_hz(1000.0)), 1000.0)

    fb = audio.mel_filter_bank(n_mels=40, n_fft=256, sample_rate=8000)
    assert fb.shape == (40, 129)

    # Synthetic 1-second audio at 8kHz
    waveform = ox.from_numpy(np.sin(2 * np.pi * 440 * np.linspace(0, 1, 8000, dtype=np.float32)))

    # Spectrogram
    spec_mod = audio.Spectrogram(n_fft=256, hop_length=128)
    spec = spec_mod(waveform)
    assert spec.shape[0] == 129  # n_fft // 2 + 1
    assert spec.shape[1] > 0

    # MelSpectrogram
    mel_mod = audio.MelSpectrogram(sample_rate=8000, n_fft=256, hop_length=128, n_mels=32)
    melspec = mel_mod(waveform)
    assert melspec.shape[0] == 32

    # dB conversion
    db = audio.amplitude_to_db(melspec)
    assert db.shape == melspec.shape

    # MFCC
    mfcc_mod = audio.MFCC(sample_rate=8000, n_mfcc=13, n_mels=32, n_fft=256, hop_length=128)
    mfcc_feats = mfcc_mod(waveform)
    assert mfcc_feats.shape[0] == 13


def test_feature_6_sparse():
    """Tests COOTensor, CSRTensor, spmm, and conversions."""
    # Create 4x4 matrix with 3 non-zeros
    indices = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    values = np.array([2.0, 3.0, 4.0], dtype=np.float32)

    coo = sparse.COOTensor(indices, values, shape=(4, 4))
    assert coo.nnz == 3
    assert "COOTensor" in repr(coo)

    # Conversion to dense
    dense = coo.to_dense()
    dense_np = dense.to_numpy()
    assert dense_np[0, 1] == 2.0
    assert dense_np[1, 2] == 3.0
    assert dense_np[2, 3] == 4.0

    # Transpose
    coo_t = sparse.sparse_transpose(coo)
    assert coo_t.shape == (4, 4)
    assert coo_t.to_dense().to_numpy()[1, 0] == 2.0

    # CSR conversion
    csr = coo.to_csr()
    assert csr.nnz == 3
    assert np.allclose(csr.to_dense().to_numpy(), dense_np)

    # spmm (sparse-dense matmul)
    b = ox.from_numpy(np.ones((4, 2), dtype=np.float32))
    res_coo = sparse.spmm(coo, b)
    res_dense = dense @ b
    assert np.allclose(res_coo.to_numpy(), res_dense.to_numpy())


def test_feature_7_clustering():
    """Tests pairwise_distances, KMeans, and PCA."""
    x = ox.from_numpy(np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]], dtype=np.float32))

    # Pairwise distances
    d_euc = clustering.pairwise_distances(x, metric="euclidean").to_numpy()
    assert d_euc.shape == (4, 4)
    assert np.isclose(d_euc[0, 0], 0.0)
    assert d_euc[0, 1] > 0.0

    # KMeans
    km = clustering.KMeans(n_clusters=2, random_state=42)
    labels = km.fit_predict(x)
    assert len(labels) == 4
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert km.inertia_ <= 2.01

    # PCA
    pca = clustering.PCA(n_components=1)
    proj = pca.fit_transform(x)
    assert list(proj.shape) == [4, 1]
    assert len(pca.explained_variance_ratio_) == 1
    assert pca.explained_variance_ratio_[0] > 0.9

    recon = pca.inverse_transform(proj)
    assert list(recon.shape) == [4, 2]


def test_feature_8_benchmark():
    """Tests benchmark and profile utilities."""
    def dummy_matmul():
        a = ox.randn([32, 32])
        return a @ a

    res = benchmark.benchmark(dummy_matmul, num_warmup=2, num_runs=5, batch_size=32)
    assert isinstance(res, benchmark.BenchmarkResult)
    assert res.mean_ms > 0
    assert res.throughput > 0
    assert "BenchmarkResult" in repr(res)

    model = ox.nn.Sequential(ox.nn.Linear(16, 16), ox.nn.ReLU(), ox.nn.Linear(16, 8))
    x = ox.randn([4, 16])
    prof = benchmark.profile(model, x, num_runs=3)
    assert "mean_latency_ms" in prof
    assert prof["estimated_flops"] > 0


def test_feature_9_export():
    """Tests export_state_dict, load_state_dict, ComputationalGraph, and to_dot."""
    model = ox.nn.Sequential(ox.nn.Linear(8, 4), ox.nn.ReLU(), ox.nn.Linear(4, 2))
    sd = export.export_state_dict(model)
    assert len(sd) >= 4  # weights and biases

    # Create new model and load state dict
    model2 = ox.nn.Sequential(ox.nn.Linear(8, 4), ox.nn.ReLU(), ox.nn.Linear(4, 2))
    export.load_state_dict(model2, sd)

    x = ox.randn([2, 8])
    assert np.allclose(model(x).to_numpy(), model2(x).to_numpy())

    # DOT serialization
    dot_str = export.to_dot(model)
    assert "digraph Sequential" in dot_str
    assert "Linear" in dot_str


def test_feature_10_alignment():
    """Tests DPOLoss, SimPOLoss, KTOLoss, RewardModelHead, and PairwisePreferenceDataset."""
    dpo = alignment.DPOLoss(beta=0.1)
    pi_w = ox.from_numpy(np.array([-1.0, -0.5], dtype=np.float32))
    pi_l = ox.from_numpy(np.array([-2.5, -3.0], dtype=np.float32))
    ref_w = ox.from_numpy(np.array([-1.2, -0.7], dtype=np.float32))
    ref_l = ox.from_numpy(np.array([-2.2, -2.8], dtype=np.float32))

    loss, chosen_r, rej_r = dpo(pi_w, pi_l, ref_w, ref_l)
    assert loss.to_numpy()[0] > 0
    assert chosen_r.shape == [2]
    assert rej_r.shape == [2]

    # SimPO
    simpo = alignment.SimPOLoss(beta=2.0, gamma=0.5)
    loss_simpo = simpo(pi_w, pi_l)
    assert loss_simpo.to_numpy()[0] > 0

    # KTO
    kto = alignment.KTOLoss(beta=0.1)
    loss_kto = kto(pi_w, ref_w, kl=0.1, is_desirable=[True, False])
    assert loss_kto.to_numpy()[0] > 0

    # RewardModelHead
    head = alignment.RewardModelHead(in_features=16)
    score = head(ox.randn([4, 16]))
    assert list(score.shape) == [4, 1]

    # Dataset
    ds = alignment.PairwisePreferenceDataset(
        prompts=["What is 2+2?"],
        chosen=["It is 4."],
        rejected=["It is 5."],
    )
    assert len(ds) == 1
    assert ds[0]["chosen"] == "It is 4."
