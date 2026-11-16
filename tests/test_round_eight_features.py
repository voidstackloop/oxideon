"""
Tests for Round 8 advanced features in Oxideon:
1. ox.quantization_advanced (FP8 E4M3/E5M2, WeightOnlyInt4Quantizer, QATLinear)
2. ox.state_space (SelectiveScan, MambaBlock)
3. ox.contrastive_learning (BarlowTwinsLoss, BYOLLoss, SwAVLoss)
4. ox.mixture_of_experts (TopKGating, MoELayer)
5. ox.protein (amino_acid_to_onehot, compute_contact_map, EGNNCoordinateUpdate)
6. ox.graph_matching (gromov_wasserstein_distance, graduated_assignment)
7. ox.information_theory (MINE, differential_entropy_gaussian)
8. ox.survival (CoxProportionalHazardsLoss, concordance_index, KaplanMeierEstimator)
9. ox.uncertainty_quant (DirichletLoss, decompose_uncertainty, brier_score)
10. ox.meta_rl (RL2ContextPolicy, MAMLPolicyAdapter)
"""
import numpy as np
import pytest
import oxideon as ox
from oxideon import (
    quantization_advanced,
    state_space,
    contrastive_learning,
    mixture_of_experts,
    protein,
    graph_matching,
    information_theory,
    survival,
    uncertainty_quant,
    meta_rl,
)

def test_quantization_advanced():
    x = ox.from_numpy(np.array([-5.0, 0.0, 3.2, 120.0], dtype=np.float32))

    # FP8 E4M3
    x_e4m3, scale_e4 = quantization_advanced.quantize_fp8_e4m3(x)
    deq_e4 = quantization_advanced.dequantize_fp8_e4m3(x_e4m3, scale_e4)
    assert np.allclose(x.to_numpy(), deq_e4.to_numpy(), rtol=0.1, atol=0.5)

    # FP8 E5M2
    x_e5m2, scale_e5 = quantization_advanced.quantize_fp8_e5m2(x)
    deq_e5 = quantization_advanced.dequantize_fp8_e5m2(x_e5m2, scale_e5)
    assert np.allclose(x.to_numpy(), deq_e5.to_numpy(), rtol=0.2, atol=1.0)

    # Weight-only 4-bit INT4
    w = ox.randn([16, 16])
    q_w, scales, zeros = quantization_advanced.WeightOnlyInt4Quantizer.quantize(w, group_size=32)
    assert q_w.shape == [16, 16]
    assert np.all(q_w.to_numpy() >= 0) and np.all(q_w.to_numpy() <= 15)

    deq_w = quantization_advanced.WeightOnlyInt4Quantizer.dequantize(q_w, scales, zeros, group_size=32)
    assert deq_w.shape == [16, 16]
    # Reconstructed weights correlate with original
    assert np.corrcoef(w.to_numpy().flatten(), deq_w.to_numpy().flatten())[0, 1] > 0.95

    # QATLinear
    layer = quantization_advanced.QATLinear(in_features=8, out_features=4, num_bits=8)
    out = layer(ox.randn([2, 8]))
    assert out.shape == [2, 4]

def test_state_space_mamba():
    # Direct SelectiveScan test
    B, L, D, N = 2, 4, 8, 4
    x = np.random.randn(B, L, D).astype(np.float32)
    delta = np.full((B, L, D), 0.1, dtype=np.float32)
    A = -np.ones((D, N), dtype=np.float32)
    B_mat = np.random.randn(B, L, N).astype(np.float32)
    C_mat = np.random.randn(B, L, N).astype(np.float32)
    D_param = np.ones(D, dtype=np.float32)

    scan_out = state_space.SelectiveScan.forward(x, delta, A, B_mat, C_mat, D_param)
    assert scan_out.shape == (B, L, D)

    # MambaBlock test
    mamba = state_space.MambaBlock(d_model=8, d_state=4, d_conv=3, expand=2)
    tokens = ox.randn([2, 6, 8])
    out = mamba(tokens)
    assert out.shape == [2, 6, 8]

def test_contrastive_learning():
    # Barlow Twins
    z1 = ox.randn([32, 16])
    z2 = ox.randn([32, 16])
    bt_loss = contrastive_learning.BarlowTwinsLoss(lambda_param=0.01)
    l_bt = bt_loss(z1, z2)
    assert float(l_bt.to_numpy().flat[0]) > 0.0

    # Self-correlation on identical views should be significantly lower than random
    l_identical = bt_loss(z1, z1)
    assert float(l_identical.to_numpy().flat[0]) < float(l_bt.to_numpy().flat[0])

    # BYOL Loss
    byol_loss = contrastive_learning.BYOLLoss()
    l_byol = byol_loss(z1, z2)
    assert float(l_byol.to_numpy().flat[0]) >= 0.0
    # Zero loss for identical vectors
    assert np.isclose(float(byol_loss(z1, z1).to_numpy().flat[0]), 0.0, atol=1e-5)

    # SwAV Loss
    swav_loss = contrastive_learning.SwAVLoss(temperature=0.1, sinkhorn_iters=3)
    s1 = ox.randn([16, 5])  # 16 samples, 5 prototypes
    s2 = ox.randn([16, 5])
    l_swav = swav_loss(s1, s2)
    assert float(l_swav.to_numpy().flat[0]) > 0.0

def test_mixture_of_experts():
    router = mixture_of_experts.TopKGating(d_model=16, num_experts=4, top_k=2, noisy_gating=False)
    x = ox.randn([8, 16])
    indices, weights, aux_loss, z_loss = router(x)

    assert indices.shape == (8, 2)
    assert weights.shape == (8, 2)
    assert np.allclose(np.sum(weights, axis=-1), 1.0)
    assert aux_loss >= 0.0
    assert z_loss >= 0.0

    # MoELayer
    moe = mixture_of_experts.MoELayer(d_model=16, hidden_dim=32, num_experts=4, top_k=2)
    out, a_loss, z_l = moe(x)
    assert out.shape == [8, 16]
    assert a_loss >= 0.0

def test_protein_structural_modeling():
    # Sequence one-hot
    seq = "ACDEF"
    onehot = protein.amino_acid_to_onehot(seq)
    assert onehot.shape == [5, 20]
    assert np.all(np.sum(onehot.to_numpy(), axis=-1) == 1.0)

    # Residue contact map
    # 3 points along a line: 0, 5, 12 Angstroms
    coords = ox.from_numpy(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [12.0, 0.0, 0.0]], dtype=np.float32))
    cmap = protein.compute_contact_map(coords, threshold_angstrom=8.0)
    assert cmap.shape == [3, 3]
    # Points 0 and 1 are 5A apart (< 8A) -> contact = 1
    assert cmap.to_numpy()[0, 1] == 1.0
    # Points 0 and 2 are 12A apart (> 8A) -> contact = 0
    assert cmap.to_numpy()[0, 2] == 0.0

    # Equivariant EGNN update
    egnn = protein.EGNNCoordinateUpdate(node_dim=8, hidden_dim=16)
    h = ox.randn([3, 8])
    h_out, coords_out = egnn(h, coords)
    assert coords_out.shape == [3, 3]

def test_graph_matching():
    # Two simple 3-node graphs
    c1 = ox.from_numpy(np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=np.float32))
    c2 = ox.from_numpy(np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], dtype=np.float32))

    # Gromov-Wasserstein
    plan, gw_dist = graph_matching.gromov_wasserstein_distance(c1, c2, num_iters=15)
    assert plan.shape == [3, 3]
    assert float(gw_dist.to_numpy().flat[0]) >= 0.0
    # Transport plan sum should be close to 1
    assert np.isclose(np.sum(plan.to_numpy()), 1.0, atol=0.1)

    # Graduated assignment
    m_soft = graph_matching.graduated_assignment(c1, c2, max_iter=10)
    assert m_soft.shape == [3, 3]
    assert np.all(m_soft.to_numpy() >= 0.0)

def test_information_theory():
    mine = information_theory.MINE(x_dim=4, y_dim=4, hidden_dim=16)
    x = ox.randn([20, 4])
    # Perfectly correlated y = x + small noise
    y = ox.from_numpy(x.to_numpy() + 0.1 * np.random.randn(20, 4).astype(np.float32))

    mi, loss = mine(x, y)
    assert float(mi.to_numpy().flat[0]) >= 0.0
    assert isinstance(float(loss.to_numpy().flat[0]), float)

    # Differential entropy of Gaussian
    cov = ox.from_numpy(np.eye(3, dtype=np.float32) * 2.0)
    ent = information_theory.differential_entropy_gaussian(cov)
    assert ent > 0.0

def test_survival_analysis():
    # Log hazards, durations, binary event indicators
    hazards = ox.from_numpy(np.array([1.5, 0.2, -0.8, 2.1], dtype=np.float32))
    durations = ox.from_numpy(np.array([10.0, 35.0, 50.0, 5.0], dtype=np.float32))
    events = ox.from_numpy(np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32))

    loss_fn = survival.CoxProportionalHazardsLoss()
    loss = loss_fn(hazards, durations, events)
    assert float(loss.to_numpy().flat[0]) > 0.0

    # Concordance index: higher hazard corresponds to shorter duration
    c_idx = survival.concordance_index(hazards, durations, events)
    assert 0.0 <= c_idx <= 1.0
    # Patient with highest hazard (2.1) had shortest duration (5.0) -> high concordance
    assert c_idx > 0.7

    # Kaplan-Meier
    km = survival.KaplanMeierEstimator()
    times, probs = km.fit(durations, events)
    assert len(times) == 4
    # Survival probability is non-increasing
    assert np.all(np.diff(probs) <= 1e-6)

def test_uncertainty_quantification():
    # Dirichlet evidential loss
    dir_loss = uncertainty_quant.DirichletLoss(kl_weight=0.01)
    logits = ox.from_numpy(np.array([[2.0, -1.0, 0.0], [0.5, 1.5, -0.5]], dtype=np.float32))
    targets = ox.from_numpy(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    l = dir_loss(logits, targets)
    assert float(l.to_numpy().flat[0]) > 0.0

    # Decomposition into aleatoric and epistemic uncertainty
    # 3 ensemble members, 2 samples, 3 classes
    m1 = np.array([[0.9, 0.05, 0.05], [0.33, 0.33, 0.34]])
    m2 = np.array([[0.85, 0.1, 0.05], [0.33, 0.33, 0.34]])
    m3 = np.array([[0.88, 0.08, 0.04], [0.34, 0.33, 0.33]])
    ensemble = ox.from_numpy(np.stack([m1, m2, m3], axis=0).astype(np.float32))

    tot_ent, aleatoric, epistemic = uncertainty_quant.decompose_uncertainty(ensemble)
    assert tot_ent.shape == [2]
    assert aleatoric.shape == [2]
    assert epistemic.shape == [2]
    # Sample 0 has high agreement/confidence -> low entropy
    assert tot_ent.to_numpy()[0] < tot_ent.to_numpy()[1]

    # Brier score
    bs = uncertainty_quant.brier_score(ox.from_numpy(m1.astype(np.float32)), targets)
    assert 0.0 <= bs <= 2.0

def test_meta_rl():
    policy = meta_rl.RL2ContextPolicy(state_dim=4, action_dim=2, hidden_dim=16)

    s0 = ox.randn([4])
    a_prev = ox.zeros([2])
    logits, val, h_next = policy.step(s0, a_prev, prev_reward=0.0, prev_done=0.0)
    assert logits.shape == [1, 2]
    assert val.shape == [1, 1]
    assert h_next.shape == (1, 16)

    # Next step with carry-over hidden state
    s1 = ox.randn([4])
    a0 = ox.from_numpy(np.array([1.0, 0.0], dtype=np.float32))
    logits2, val2, h_next2 = policy.step(s1, a0, prev_reward=1.0, prev_done=0.0, hidden_state=h_next)
    assert not np.allclose(h_next, h_next2)

    # MAML Policy Adapter
    w = {"fc.weight": np.array([[1.0, 0.5], [-0.5, 1.0]], dtype=np.float32)}
    grads = {"fc.weight": np.array([[0.1, 0.1], [-0.1, -0.1]], dtype=np.float32)}
    advantages = np.array([2.0, 2.0], dtype=np.float32)

    w_adapted = meta_rl.MAMLPolicyAdapter.adapt(w, advantages, grads, inner_lr=0.1)
    assert not np.allclose(w["fc.weight"], w_adapted["fc.weight"])
