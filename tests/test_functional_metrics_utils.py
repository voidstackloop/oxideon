"""Tests for Oxideon functional API (ox.F / ox.functional), evaluation metrics (ox.metrics),
high-level utilities (ox.utils), and additional nn loss modules.
"""

import math
import numpy as np
import pytest

import oxideon as ox
from oxideon import nn, F, functional, metrics, utils


# ---------------------------------------------------------------------------
# Functional API Tests
# ---------------------------------------------------------------------------

def test_functional_aliases_and_activations():
    assert F is functional
    assert ox.F is ox.functional
    assert nn.functional is F
    assert nn.F is F

    x = ox.Tensor.from_numpy(np.array([-2.0, -0.5, 0.0, 1.0, 3.0], dtype=np.float32))

    # Test activations
    assert np.allclose(F.relu(x).to_numpy(), np.maximum(0, x.to_numpy()))
    assert np.allclose(F.sigmoid(x).to_numpy(), 1.0 / (1.0 + np.exp(-x.to_numpy())), atol=1e-4)
    assert np.allclose(F.tanh(x).to_numpy(), np.tanh(x.to_numpy()), atol=1e-4)
    assert F.silu(x).shape == [5]
    assert F.gelu(x).shape == [5]
    assert F.leaky_relu(x, negative_slope=0.1).shape == [5]
    assert F.softplus(x).shape == [5]
    assert F.softsign(x).shape == [5]

    # Test softmax & log_softmax
    mat = ox.Tensor.from_numpy(np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32))
    sm = F.softmax(mat, dim=-1).to_numpy()
    assert np.allclose(sm.sum(axis=-1), [1.0, 1.0], atol=1e-5)
    log_sm = F.log_softmax(mat, dim=-1).to_numpy()
    assert np.allclose(np.exp(log_sm), sm, atol=1e-4)


def test_functional_projections_and_transforms():
    x = ox.randn([4, 8])
    w = ox.randn([16, 8])
    b = ox.randn([16])

    # F.linear
    out = F.linear(x, w, b)
    assert out.shape == [4, 16]

    out_nobias = F.linear(x, w)
    assert out_nobias.shape == [4, 16]

    # F.dropout
    dr_eval = F.dropout(x, p=0.5, training=False)
    assert np.allclose(dr_eval.to_numpy(), x.to_numpy())

    dr_train = F.dropout(x, p=0.5, training=True)
    assert dr_train.shape == [4, 8]

    # F.cosine_similarity
    v1 = ox.Tensor.from_numpy(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    v2 = ox.Tensor.from_numpy(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32))
    cos_sim = F.cosine_similarity(v1, v2, dim=-1).to_numpy()
    assert np.isclose(cos_sim[0], 1.0, atol=1e-4)
    assert np.isclose(cos_sim[1], -1.0, atol=1e-4)


def test_functional_losses_and_autograd():
    pred = ox.Tensor.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    target = ox.Tensor.from_numpy(np.array([1.0, 2.0, 4.0], dtype=np.float32))

    mse = F.mse_loss(pred, target)
    assert np.isclose(mse.item(), 1.0 / 3.0, atol=1e-4)

    l1 = F.l1_loss(pred, target)
    assert np.isclose(l1.item(), 1.0 / 3.0, atol=1e-4)

    # Binary Cross Entropy
    probs = ox.Tensor.from_numpy(np.array([0.9, 0.1], dtype=np.float32))
    bce_target = ox.Tensor.from_numpy(np.array([1.0, 0.0], dtype=np.float32))
    bce = F.binary_cross_entropy(probs, bce_target)
    assert bce.item() < 0.2

    # Higher-order Functional Autograd: vjp and jvp
    def square_fn(t):
        return t * t

    inp = ox.Tensor.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    y, pullback = F.vjp(square_fn, inp)
    assert np.allclose(y.to_numpy(), [9.0, 16.0])

    grad_res = pullback()
    # d/dx (x^2) = 2x -> [6.0, 8.0]
    assert np.allclose(grad_res.to_numpy(), [6.0, 8.0], atol=1e-4)

    # jvp
    v = ox.Tensor.from_numpy(np.array([1.0, 1.0], dtype=np.float32))
    y_out, jvp_val = F.jvp(square_fn, inp, v)
    assert np.allclose(y_out.to_numpy(), [9.0, 16.0])
    assert np.allclose(jvp_val.to_numpy(), [6.0, 8.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------

def test_metrics_classification():
    preds = np.array([0, 1, 2, 0, 1, 2])
    targets = np.array([0, 1, 1, 0, 1, 2])

    acc = metrics.accuracy(preds, targets)
    assert np.isclose(acc, 5.0 / 6.0)

    # Logits input support
    logits = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 5.0, 10.0],
    ])
    tgt = np.array([0, 1, 2])
    assert metrics.accuracy(logits, tgt) == 1.0

    # Top-K Accuracy
    topk_acc = metrics.topk_accuracy(logits, tgt, k=2)
    assert topk_acc == 1.0

    # Confusion matrix
    cm = metrics.confusion_matrix(preds, targets, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[1, 2] == 1  # target 1 predicted as 2

    # Precision, Recall, F1
    prf = metrics.precision_recall_f1(preds, targets, num_classes=3, average="macro")
    assert "precision" in prf and "recall" in prf and "f1" in prf
    assert 0.0 <= prf["f1"] <= 1.0

    # Binary PRF
    bin_prf = metrics.precision_recall_f1(np.array([1, 0, 1]), np.array([1, 1, 1]), num_classes=2, average="binary")
    assert bin_prf["precision"] == 1.0
    assert np.isclose(bin_prf["recall"], 2.0 / 3.0)

    # ROC-AUC
    scores = np.array([0.9, 0.8, 0.3, 0.1])
    bin_targets = np.array([1, 1, 0, 0])
    auc = metrics.roc_auc_score(scores, bin_targets)
    assert auc == 1.0


def test_metrics_regression():
    p = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([1.5, 2.5, 2.5, 4.5])

    mse = metrics.mean_squared_error(p, t)
    assert np.isclose(mse, 0.25)

    mae = metrics.mean_absolute_error(p, t)
    assert np.isclose(mae, 0.5)

    r2 = metrics.r2_score(p, t)
    assert r2 > 0.75


def test_metric_tracker():
    tracker = metrics.MetricTracker("train_loss")
    tracker.update(0.5, n=2)
    tracker.update(0.3, n=2)

    assert np.isclose(tracker.avg, 0.4)
    assert np.isclose(tracker.min, 0.3)
    assert np.isclose(tracker.max, 0.5)
    assert tracker.count == 4

    summary = tracker.summary()
    assert summary["name"] == "train_loss"
    assert summary["count"] == 4

    tracker.reset()
    assert tracker.count == 0


# ---------------------------------------------------------------------------
# Utils Tests
# ---------------------------------------------------------------------------

def test_utils_set_seed_and_benchmarking():
    utils.set_seed(12345)
    r1 = np.random.rand(5)

    utils.set_seed(12345)
    r2 = np.random.rand(5)

    assert np.allclose(r1, r2)

    with utils.BenchmarkTimer("tensor_matmul") as bench:
        a = ox.randn([50, 50])
        b = ox.randn([50, 50])
        c = a @ b
    assert bench.ms > 0.0
    assert "tensor_matmul" in repr(bench)


def test_utils_gradient_clipping():
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = DummyNet()
    x = ox.randn([3, 4])
    out = model(x).sum()
    out.backward()

    # Pre-clip norm
    total_norm = utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    assert total_norm > 0.0

    # Ensure grads were scaled if norm exceeded max_norm
    post_norm = math.sqrt(sum(p.grad().norm().item() ** 2 for p in model.parameters() if p.grad() is not None))
    assert post_norm <= 0.5001


def test_utils_model_summary():
    class DeepVisionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.norm = nn.LayerNorm(16)
            self.head = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            return self.head(x)

    model = DeepVisionNet()
    summary = utils.model_summary(model)
    assert "Layer (type)" in summary
    assert "conv (Conv2d)" in summary
    assert "head (Sequential)" in summary
    assert "Total params:" in summary
    assert "Trainable params:" in summary
    assert "Estimated Model Size:" in summary


# ---------------------------------------------------------------------------
# New Loss Modules in nn Tests
# ---------------------------------------------------------------------------

def test_nn_loss_modules():
    cos_loss = nn.CosineEmbeddingLoss(margin=0.5)
    x1 = ox.randn([4, 8])
    x2 = ox.randn([4, 8])
    target = ox.Tensor.from_numpy(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))
    l_cos = cos_loss(x1, x2, target)
    assert l_cos.numel() == 1

    trip_loss = nn.TripletMarginLoss(margin=1.0)
    anc = ox.randn([4, 8])
    pos = ox.randn([4, 8])
    neg = ox.randn([4, 8])
    l_trip = trip_loss(anc, pos, neg)
    assert l_trip.numel() == 1

    rank_loss = nn.MarginRankingLoss(margin=0.1)
    l_rank = rank_loss(x1, x2, target)
    assert l_rank.numel() == 1

    hinge_loss = nn.HingeEmbeddingLoss(margin=1.0)
    l_hinge = hinge_loss(x1, target)
    assert l_hinge.numel() == 1

    soft_loss = nn.SoftMarginLoss()
    l_soft = soft_loss(x1.mean(), ox.Tensor.from_numpy(np.array(1.0, dtype=np.float32)))
    assert l_soft.numel() == 1
