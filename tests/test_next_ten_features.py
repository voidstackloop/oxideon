"""Tests for 10 Advanced Deep Learning & AI Features in Oxideon:
1. ox.nn.init (Xavier, Kaiming, Orthogonal, Constant initializations)
2. ox.peft (LoRA parameter-efficient fine-tuning, adapter merging)
3. ox.models (ResNet18, VisionTransformer ViT, CausalLM with generation)
4. ox.amp (Automatic Mixed Precision: autocast & GradScaler)
5. ox.losses (DiceLoss, ContrastiveLoss, LabelSmoothingCrossEntropy, ArcFaceLoss)
6. ox.explain (IntegratedGradients, SaliencyMap, FeatureAblation)
7. ox.search (VectorIndex vector database & semantic similarity search)
8. ox.optim.lr_scheduler (WarmupCosineScheduler, PolynomialLR, CyclicLR, ChainedScheduler)
9. ox.root (Differentiable Newton-Raphson & Bisection root finding)
10. ox.distributed (DataParallel multi-worker container & collective operations)
"""

import math
import os
import shutil
import tempfile
import numpy as np
import pytest

import oxideon as ox
from oxideon import nn, optim, init, peft, models, amp, losses, explain, search, root, distributed


# ---------------------------------------------------------------------------
# Feature 1: ox.nn.init
# ---------------------------------------------------------------------------

def test_feature1_nn_init():
    w = ox.Tensor.randn([16, 32])

    w_xu = init.xavier_uniform_(w)
    assert w_xu.shape == [16, 32]

    w_xn = init.xavier_normal_(w)
    assert w_xn.shape == [16, 32]

    w_ku = init.kaiming_uniform_(w, nonlinearity="relu")
    assert w_ku.shape == [16, 32]

    w_kn = init.kaiming_normal_(w, nonlinearity="leaky_relu")
    assert w_kn.shape == [16, 32]

    w_ort = init.orthogonal_(w)
    assert w_ort.shape == [16, 32]

    w_c = init.constant_(w, 3.14)
    assert np.allclose(w_c.to_numpy(), 3.14, atol=1e-5)

    w_z = init.zeros_(w)
    assert np.allclose(w_z.to_numpy(), 0.0)

    w_o = init.ones_(w)
    assert np.allclose(w_o.to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# Feature 2: ox.peft (LoRA)
# ---------------------------------------------------------------------------

def test_feature2_peft_lora():
    class BigMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = BigMLP()
    x = ox.randn([4, 32])
    base_out = model(x).to_numpy()

    # Apply LoRA with rank 4
    stats = peft.apply_lora(model, r=4, lora_alpha=8.0)
    assert stats["trainable_pct"] < 50.0  # Vast majority of parameters frozen
    assert isinstance(model.fc1, peft.LoRALinear)
    assert isinstance(model.fc2, peft.LoRALinear)

    # Initial forward pass with LoRA (B is initialized to 0, so output matches base model initially)
    lora_out = model(x).to_numpy()
    assert np.allclose(base_out, lora_out, atol=1e-4)

    # Merge LoRA into base weights
    peft.merge_lora(model)
    merged_out = model(x).to_numpy()
    assert np.allclose(base_out, merged_out, atol=1e-4)


# ---------------------------------------------------------------------------
# Feature 3: ox.models (ResNet18, ViT, CausalLM)
# ---------------------------------------------------------------------------

def test_feature3_canonical_models():
    # 1. ResNet18
    resnet = models.ResNet18(in_channels=3, num_classes=5, hidden_dim=16)
    img = ox.randn([2, 3, 16, 16])
    res_out = resnet(img)
    assert res_out.shape == [2, 5]

    # 2. VisionTransformer (ViT)
    vit = models.VisionTransformer(
        img_size=16,
        patch_size=4,
        in_channels=3,
        num_classes=4,
        embed_dim=32,
        depth=1,
        num_heads=2,
    )
    vit_out = vit(img)
    assert vit_out.shape == [2, 4]

    # 3. MiniGPT (Autoregressive language model with text generation)
    gpt = models.MiniGPT(vocab_size=50, max_seq_len=16, embed_dim=32, num_layers=1, num_heads=2)
    seq = ox.Tensor.from_numpy(np.array([[1, 5, 12, 8]], dtype=np.float32))
    logits = gpt(seq)
    assert logits.shape == [1, 4, 50]

    gen_tokens = gpt.generate(prompt_ids=[1, 5], max_new_tokens=4, temperature=1.0)
    assert len(gen_tokens) == 6
    assert gen_tokens[:2] == [1, 5]


# ---------------------------------------------------------------------------
# Feature 4: ox.amp (Automatic Mixed Precision)
# ---------------------------------------------------------------------------

def test_feature4_amp_autocast_and_scaler():
    assert not amp.is_autocast_enabled()
    with amp.autocast(enabled=True, dtype="f16"):
        assert amp.is_autocast_enabled()
        assert amp.get_autocast_dtype() == "f16"
    assert not amp.is_autocast_enabled()

    scaler = amp.GradScaler(init_scale=1024.0, growth_factor=2.0)
    loss = ox.Tensor.from_numpy(np.array(2.0, dtype=np.float32))
    scaled_loss = scaler.scale(loss)
    assert np.isclose(scaled_loss.item(), 2048.0)

    model = nn.Linear(4, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    x = ox.randn([2, 4])
    l = model(x).sum()
    l.backward()

    success = scaler.step(optimizer)
    assert success
    scaler.update()


# ---------------------------------------------------------------------------
# Feature 5: ox.losses (Dice, Contrastive, LabelSmoothing, ArcFace)
# ---------------------------------------------------------------------------

def test_feature5_specialized_losses():
    # DiceLoss
    dice = losses.DiceLoss(smooth=1.0)
    p = ox.Tensor.from_numpy(np.array([0.9, 0.8, 0.1], dtype=np.float32))
    t = ox.Tensor.from_numpy(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    l_dice = dice(p, t)
    assert 0.0 <= l_dice.item() <= 1.0

    # ContrastiveLoss
    contrast = losses.ContrastiveLoss(margin=2.0)
    v1 = ox.Tensor.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32))
    v2 = ox.Tensor.from_numpy(np.array([[1.0, 0.0]], dtype=np.float32))
    sim_target = ox.Tensor.from_numpy(np.array([0.0], dtype=np.float32))  # similar
    l_contrast = contrast(v1, v2, sim_target)
    assert np.isclose(l_contrast.item(), 0.0)

    # LabelSmoothingCrossEntropy
    ls_ce = losses.LabelSmoothingCrossEntropy(smoothing=0.1)
    logits = ox.Tensor.from_numpy(np.array([[5.0, 1.0]], dtype=np.float32))
    lbl = ox.Tensor.from_numpy(np.array([0], dtype=np.float32))
    l_smooth = ls_ce(logits, lbl)
    assert l_smooth.item() > 0.0

    # ArcFaceLoss
    arcface = losses.ArcFaceLoss(in_features=8, out_features=4, s=10.0, m=0.3)
    emb = ox.randn([2, 8])
    labels = ox.Tensor.from_numpy(np.array([1, 3], dtype=np.float32))
    l_arc = arcface(emb, labels)
    assert l_arc.numel() == 1


# ---------------------------------------------------------------------------
# Feature 6: ox.explain (Integrated Gradients, Saliency, Ablation)
# ---------------------------------------------------------------------------

def test_feature6_interpretability_and_attribution():
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = SimpleClassifier()
    x = ox.randn([1, 4])

    # Integrated Gradients
    ig = explain.IntegratedGradients(model)
    attrs = ig.attribute(x, target_class=0, steps=5)
    assert attrs.shape == [1, 4]

    # Saliency Map
    saliency = explain.SaliencyMap(model)
    sal_map = saliency.attribute(x, target_class=0)
    assert sal_map.shape == [1, 4]
    assert np.all(sal_map.to_numpy() >= 0.0)

    # Feature Ablation
    ablation = explain.FeatureAblation(model)
    abl_attrs = ablation.attribute(x, target_class=0)
    assert abl_attrs.shape == [1, 4]


# ---------------------------------------------------------------------------
# Feature 7: ox.search (Vector Database & Semantic Search)
# ---------------------------------------------------------------------------

def test_feature7_vector_database():
    tmp_dir = tempfile.mkdtemp()
    try:
        index = search.VectorIndex(dim=8, metric="cosine")

        v1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        index.add([v1, v2, v3], ids=["doc1", "doc2", "doc3"], metadatas=[{"category": "A"}, {"category": "B"}, {"category": "A"}])
        assert len(index) == 3

        query = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, k=2)

        assert len(results) == 2
        assert results[0].id == "doc1"
        assert np.isclose(results[0].score, 1.0)
        assert results[1].id == "doc3"

        # Save and reload index
        idx_path = os.path.join(tmp_dir, "vector_index.json")
        index.save(idx_path)
        loaded_index = search.VectorIndex.load(idx_path)
        assert len(loaded_index) == 3
        loaded_res = loaded_index.search(query, k=1)
        assert loaded_res[0].id == "doc1"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Feature 8: Learning Rate Schedulers Suite Extensions
# ---------------------------------------------------------------------------

def test_feature8_lr_schedulers_extensions():
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # WarmupCosineScheduler
    warmup_cos = optim.lr_scheduler.WarmupCosineScheduler(
        optimizer, warmup_steps=5, total_steps=20, min_lr=0.01
    )
    for _ in range(10):
        warmup_cos.step()
    assert 0.01 <= warmup_cos.get_last_lr()[0] <= 0.1

    # PolynomialLR
    poly = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=2.0)
    poly.step()
    assert poly.get_last_lr()[0] < 0.1

    # CyclicLR
    cyclic = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.05, step_size_up=5)
    cyclic.step()
    assert 0.01 <= cyclic.get_last_lr()[0] <= 0.05

    # ChainedScheduler
    s1 = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    s2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    chained = optim.lr_scheduler.ChainedScheduler([s1, s2])
    chained.step()
    assert len(chained.get_last_lr()) > 0


# ---------------------------------------------------------------------------
# Feature 9: ox.root (Differentiable Root Finding)
# ---------------------------------------------------------------------------

def test_feature9_differentiable_root_finding():
    # Find root of f(x) = x^2 - 4 = 0 near x0 = 3
    def quadratic(t):
        return (t * t) - 4.0

    res_nr = root.newton_raphson(quadratic, x0=3.0, tol=1e-5)
    assert res_nr.converged
    assert np.isclose(res_nr.root, 2.0, atol=1e-4)

    # Bisection method on [0.0, 5.0]
    res_bis = root.bisection(lambda x: x**3 - 8.0, a=0.0, b=4.0, tol=1e-5)
    assert res_bis.converged
    assert np.isclose(res_bis.root, 2.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Feature 10: ox.distributed (DataParallel & Collectives)
# ---------------------------------------------------------------------------

def test_feature10_distributed():
    class TwoLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    base_net = TwoLayerNet()
    dp_net = distributed.DataParallel(base_net, num_replicas=2)

    x_batch = ox.randn([8, 4])
    out_dp = dp_net(x_batch)
    assert out_dp.shape == [8, 2]

    # Collectives: all_reduce, broadcast, all_gather
    t1 = ox.Tensor.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    t2 = ox.Tensor.from_numpy(np.array([3.0, 4.0], dtype=np.float32))

    reduced = distributed.all_reduce([t1, t2], op="sum")
    assert np.allclose(reduced[0].to_numpy(), [4.0, 6.0])

    bcast = distributed.broadcast(t1, num_replicas=3)
    assert len(bcast) == 3
    assert np.allclose(bcast[2].to_numpy(), [1.0, 2.0])

    gathered = distributed.all_gather([t1, t2], dim=0)
    assert gathered.shape == [4]
    assert np.allclose(gathered.to_numpy(), [1.0, 2.0, 3.0, 4.0])
