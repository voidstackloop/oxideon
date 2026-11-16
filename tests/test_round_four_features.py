import pytest
import numpy as np
import oxideon as ox
from oxideon import (
    timeseries,
    spatial,
    flow,
    calibration,
    adversarial,
    distill,
    active_learning,
    multimodal,
    meta,
    fairness,
)


def test_feature_1_timeseries():
    """Tests seasonal_decompose, lag_features, rolling_window, exponential_smoothing, and PatchTST."""
    # Synthetic series: linear trend + sine wave
    t = np.linspace(0, 10, 100, dtype=np.float32)
    s = 2.0 * t + np.sin(2 * np.pi * t)
    s_tensor = ox.from_numpy(s)

    # Seasonal decomposition
    decomp = timeseries.seasonal_decompose(s_tensor, period=10, model="additive")
    assert len(decomp.trend) == 100
    assert len(decomp.seasonal) == 100
    assert len(decomp.resid) == 100
    assert "DecomposeResult" in repr(decomp)

    # Lag features
    lags = timeseries.lag_features(s_tensor, lags=[1, 2, 5])
    assert lags.shape == [95, 3]

    # Rolling window
    X, y = timeseries.rolling_window(s_tensor, window_size=16, horizon=4, step=2)
    assert X.shape[1] == 16
    assert y.shape[1] == 4
    assert X.shape[0] == y.shape[0]

    # Exponential smoothing
    smoothed = timeseries.exponential_smoothing(s_tensor, alpha=0.3)
    assert len(smoothed) == 100

    # PatchTST
    patch_model = timeseries.PatchTST(seq_len=32, pred_len=8, patch_len=8, stride=4, d_model=16)
    dummy_input = ox.randn([4, 32])
    forecast = patch_model(dummy_input)
    assert list(forecast.shape) == [4, 8]


def test_feature_2_spatial():
    """Tests affine_grid, grid_sample, random_affine_matrix, and SpatialTransformerNetwork."""
    # Identity affine transform
    theta = np.zeros((2, 2, 3), dtype=np.float32)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta_t = ox.from_numpy(theta)

    grid = spatial.affine_grid(theta_t, (2, 1, 8, 8))
    assert list(grid.shape) == [2, 8, 8, 2]

    # Image tensor
    img = ox.randn([2, 1, 8, 8])
    sampled = spatial.grid_sample(img, grid)
    assert list(sampled.shape) == [2, 1, 8, 8]
    # Under identity transform, output equals input
    assert np.allclose(img.to_numpy(), sampled.to_numpy(), atol=1e-4)

    # Random affine matrix
    rand_theta = spatial.random_affine_matrix(batch_size=4)
    assert list(rand_theta.shape) == [4, 2, 3]

    # STN
    stn = spatial.SpatialTransformerNetwork(in_features=64, hidden_dim=16)
    out_stn = stn(img)
    assert list(out_stn.shape) == [2, 1, 8, 8]


def test_feature_3_flow():
    """Tests AffineCouplingLayer and RealNVP normalizing flow."""
    x = ox.randn([4, 6])
    coupling = flow.AffineCouplingLayer(dim=6, hidden_dim=16, mask_type="even")
    y, log_det = coupling(x)
    assert list(y.shape) == [4, 6]
    assert list(log_det.shape) == [4]

    # Exact invertibility
    x_rec = coupling.inverse(y)
    assert np.allclose(x.to_numpy(), x_rec.to_numpy(), atol=1e-5)

    # Full RealNVP model
    realnvp = flow.RealNVP(in_features=6, num_layers=4, hidden_dim=16)
    z, total_log_det = realnvp(x)
    assert list(z.shape) == [4, 6]
    assert np.allclose(x.to_numpy(), realnvp.inverse(z).to_numpy(), atol=1e-5)

    # Log-probability and sampling
    logp = realnvp.log_prob(x)
    assert list(logp.shape) == [4]
    samples = realnvp.sample(num_samples=8)
    assert list(samples.shape) == [8, 6]


def test_feature_4_calibration():
    """Tests expected_calibration_error, reliability_diagram_data, TemperatureScaling, and MC Dropout."""
    # Uncalibrated overconfident logits
    logits = np.array([
        [10.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [10.0, 0.0],
    ], dtype=np.float32)
    targets = np.array([0, 1, 1, 0], dtype=np.int64)

    ece = calibration.expected_calibration_error(logits, targets, num_bins=5)
    assert ece > 0.0

    diag = calibration.reliability_diagram_data(logits, targets, num_bins=5)
    assert "bin_accuracies" in diag
    assert "bin_confidences" in diag
    assert len(diag["bin_counts"]) == 5

    # Temperature Scaling
    ts = calibration.TemperatureScaling(init_temperature=1.0)
    ts.fit(logits, targets, max_iter=20)
    scaled_logits = ts(ox.from_numpy(logits))
    assert list(scaled_logits.shape) == [4, 2]

    # MC Dropout
    net = ox.nn.Sequential(ox.nn.Linear(8, 16), ox.nn.Dropout(0.5), ox.nn.Linear(16, 2))
    mc = calibration.MonteCarloDropout(net, num_samples=5)
    mean_pred, var_pred = mc.predict(ox.randn([4, 8]))
    assert list(mean_pred.shape) == [4, 2]
    assert list(var_pred.shape) == [4, 2]


def test_feature_5_adversarial():
    """Tests fgsm_attack, pgd_attack, and adversarial_training_step."""
    net = ox.nn.Sequential(ox.nn.Linear(4, 8), ox.nn.ReLU(), ox.nn.Linear(8, 2))
    x = ox.randn([2, 4])
    y = ox.from_numpy(np.array([0, 1], dtype=np.int64))

    # FGSM
    x_adv_fgsm = adversarial.fgsm_attack(net, x, y, epsilon=0.05)
    assert list(x_adv_fgsm.shape) == [2, 4]
    assert not np.allclose(x.to_numpy(), x_adv_fgsm.to_numpy())

    # PGD
    x_adv_pgd = adversarial.pgd_attack(net, x, y, epsilon=0.05, alpha=0.01, num_steps=5)
    assert list(x_adv_pgd.shape) == [2, 4]
    # Check L_inf bound
    diff = np.abs(x.to_numpy() - x_adv_pgd.to_numpy())
    assert np.max(diff) <= 0.05001

    # Adversarial Training Step
    opt = ox.optim.Adam(net.parameters(), lr=0.01)
    loss_val = adversarial.adversarial_training_step(net, opt, x, y, epsilon=0.05)
    assert loss_val > 0.0


def test_feature_6_distill():
    """Tests DistillationLoss, FeatureMatchingLoss, and DistillationTrainer."""
    dist_loss = distill.DistillationLoss(temperature=2.0, alpha=0.5)
    s_logits = ox.randn([4, 3])
    t_logits = ox.randn([4, 3])
    targets = ox.from_numpy(np.array([0, 1, 2, 0], dtype=np.int64))

    loss = dist_loss(s_logits, t_logits, targets)
    assert float(loss.to_numpy()) > 0

    # Feature matching
    feat_loss = distill.FeatureMatchingLoss(loss_type="mse")
    s_feat = ox.randn([4, 16])
    t_feat = ox.randn([4, 16])
    fl = feat_loss(s_feat, t_feat)
    assert float(fl.to_numpy()) > 0

    # DistillationTrainer
    student = ox.nn.Sequential(ox.nn.Linear(8, 4), ox.nn.Linear(4, 2))
    teacher = ox.nn.Sequential(ox.nn.Linear(8, 16), ox.nn.Linear(16, 2))
    opt = ox.optim.Adam(student.parameters(), lr=0.01)

    trainer = distill.DistillationTrainer(student, teacher, opt)
    x = ox.randn([4, 8])
    y = ox.from_numpy(np.array([0, 1, 0, 1], dtype=np.int64))
    step_loss = trainer.train_step(x, y)
    assert step_loss > 0


def test_feature_7_active_learning():
    """Tests entropy_sampling, margin_sampling, least_confidence_sampling, and bald_sampling."""
    probs = np.array([
        [0.9, 0.1, 0.0],  # very certain
        [0.34, 0.33, 0.33],  # highest entropy, smallest margin
        [0.6, 0.4, 0.0],  # intermediate
    ], dtype=np.float32)

    # Entropy
    idx_ent = active_learning.entropy_sampling(probs, n_samples=1)
    assert idx_ent[0] == 1

    # Margin
    idx_mar = active_learning.margin_sampling(probs, n_samples=1)
    assert idx_mar[0] == 1

    # Least confidence
    idx_lc = active_learning.least_confidence_sampling(probs, n_samples=1)
    assert idx_lc[0] == 1

    # BALD
    # [3 samples, 2 data points, 2 classes]
    mc_runs = np.array([
        [[0.9, 0.1], [0.5, 0.5]],
        [[0.1, 0.9], [0.5, 0.5]],
        [[0.8, 0.2], [0.5, 0.5]],
    ], dtype=np.float32)
    # Point 0 has huge disagreement across runs, point 1 has constant predictions
    idx_bald = active_learning.bald_sampling(mc_runs, n_samples=1)
    assert idx_bald[0] == 0


def test_feature_8_multimodal():
    """Tests CLIPLoss, CrossAttentionBlock, and MultimodalProjection."""
    clip = multimodal.CLIPLoss(temperature=0.07)
    img_emb = ox.randn([4, 16])
    txt_emb = ox.randn([4, 16])

    loss = clip(img_emb, txt_emb)
    assert float(loss.to_numpy()) > 0

    # Cross-Attention
    cross_attn = multimodal.CrossAttentionBlock(query_dim=16, context_dim=32, hidden_dim=32, num_heads=2)
    q = ox.randn([2, 5, 16])
    c = ox.randn([2, 10, 32])
    out_attn = cross_attn(q, c)
    assert list(out_attn.shape) == [2, 5, 16]

    # MultimodalProjection
    proj = multimodal.MultimodalProjection(in_dim=16, proj_dim=32, normalize=True)
    out_p = proj(img_emb)
    assert list(out_p.shape) == [4, 32]
    # Unit norm
    norms = np.linalg.norm(out_p.to_numpy(), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_feature_9_meta():
    """Tests compute_prototypes, PrototypicalNetworksLoss, and MAMLLinear."""
    # 2 classes, 2 support samples each
    support_emb = np.array([
        [1.0, 1.0],
        [1.0, 3.0],
        [10.0, 10.0],
        [10.0, 12.0],
    ], dtype=np.float32)
    support_lbl = np.array([0, 0, 1, 1], dtype=np.int64)

    protos = meta.compute_prototypes(support_emb, support_lbl, num_classes=2)
    assert list(protos.shape) == [2, 2]
    assert np.allclose(protos.to_numpy()[0], [1.0, 2.0])
    assert np.allclose(protos.to_numpy()[1], [10.0, 11.0])

    # Query evaluation
    proto_loss = meta.PrototypicalNetworksLoss()
    query_emb = ox.from_numpy(np.array([[1.0, 2.0], [10.0, 11.0]], dtype=np.float32))
    query_lbl = ox.from_numpy(np.array([0, 1], dtype=np.int64))

    loss, acc = proto_loss(query_emb, protos, query_lbl)
    assert loss.to_numpy()[0] >= 0
    assert acc == 1.0

    # MAMLLinear
    maml = meta.MAMLLinear(in_features=4, out_features=2)
    x = ox.randn([2, 4])
    base_out = maml(x)
    assert list(base_out.shape) == [2, 2]

    # Functional forward with adapted weights
    w_adapted = ox.zeros([4, 2])
    adapted_out = maml(x, weight=w_adapted)
    assert np.allclose(adapted_out.to_numpy(), 0.0)


def test_feature_10_fairness():
    """Tests demographic_parity_difference, disparate_impact_ratio, equalized_odds_difference, and reweighting."""
    # Group 0: 80% positive; Group 1: 40% positive
    y_pred = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0], dtype=np.float32)
    group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    y_true = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64)

    # Demographic parity
    dp = fairness.demographic_parity_difference(y_pred, group)
    assert np.isclose(dp, 0.4)  # 0.8 - 0.4 = 0.4

    # Disparate impact
    di = fairness.disparate_impact_ratio(y_pred, group)
    assert np.isclose(di, 0.5)  # 0.4 / 0.8 = 0.5

    # Equalized odds
    eo = fairness.equalized_odds_difference(y_true, y_pred, group)
    assert eo >= 0.0

    # Reweighting weights
    weights = fairness.reweighting_weights(y_true, group)
    assert len(weights) == 10
    assert np.all(weights.to_numpy() > 0)
