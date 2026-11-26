"""Comprehensive verification suite for Oxideon utilities:
- `treemap` hierarchical container operations
- `nn.utils` norms, weight/spectral norm, BatchNorm fusion, reparameterization, pruning, and RNN helpers
- `utils.*` hooks, runtime diagnostics, FLOP profiling, DLPack, weight caching, microtiming, and event logging
- Workflow utilities: `one_hot_encode`, `axis_normalize`, `pad_sequences`, `sliding_window_dataset`,
  `split_dataset`, `image_folder_dataset`, `text_folder_dataset`, `ProgressBar`, class/sample weights,
  `fetch_artifact`, `SerializationScope`, `CheckpointVault`, `ExponentialMovingAverage`
- `vision.utils`: `image_grid`, `save_image`, `draw_bounding_boxes`, `draw_segmentation_masks`, `draw_keypoints`
"""

from collections import OrderedDict, defaultdict, deque, namedtuple
import hashlib
import io
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest

import oxideon as ox
import oxideon.nn as nn


# ---------------------------------------------------------------------------
# 1. `treemap` Hierarchical Container Utilities
# ---------------------------------------------------------------------------

def test_treemap_flatten_unflatten_and_custom_nodes():
    Pair = namedtuple("Pair", ["left", "right"])
    dd = defaultdict(int, {"k": 5})
    dq = deque([7, 8], maxlen=4)
    tree = {
        "pair": Pair(ox.tensor([1.0, 2.0]), [ox.tensor([3.0]), OrderedDict([("z", 4)])]),
        "dd": dd,
        "dq": dq,
        "text": "atomic_string",
    }

    assert ox.is_nested(tree)
    assert not ox.is_nested("atomic_string")
    assert not ox.is_nested(b"atomic_bytes")

    leaves, schema = ox.tree_flatten(tree)
    assert len(leaves) == 7
    assert schema.num_leaves == 7
    rebuilt = ox.tree_unflatten(leaves, schema)
    assert isinstance(rebuilt["pair"], Pair)
    assert isinstance(rebuilt["pair"].right[1], OrderedDict)
    assert isinstance(rebuilt["dd"], defaultdict) and rebuilt["dd"].default_factory is int
    assert isinstance(rebuilt["dq"], deque) and rebuilt["dq"].maxlen == 4
    assert rebuilt["text"] == "atomic_string"

    doubled = ox.tree_map_only(int, lambda v: v * 2, tree)
    assert doubled["pair"].right[1]["z"] == 8
    assert doubled["dd"]["k"] == 10
    assert doubled["text"] == "atomic_string"

    s1 = {"b": [1, 2], "a": (3, 4)}
    s2 = {"a": (10, 20), "b": [30, 40]}
    ox.assert_same_structure(s1, s2, check_types=True)
    combined = ox.map_structure(lambda x, y: x + y, s1, s2)
    assert combined == {"b": [31, 42], "a": (13, 24)}

    with pytest.raises(TypeError):
        ox.assert_same_structure([1, 2], (1, 2), check_types=True)
    ox.assert_same_structure([1, 2], (1, 2), check_types=False)

    bcast = ox.broadcast_tree_prefix({"x": 10, "y": 20}, {"x": [1, 2, 3], "y": (4, 5)})
    assert bcast == [10, 10, 10, 20, 20]

    class Box:
        def __init__(self, tag: str, val: int):
            self.tag = tag
            self.val = val

    ox.register_tree_node(Box, lambda b: ([b.val], b.tag), lambda vs, tag: Box(tag, list(vs)[0]))
    try:
        mapped_box = ox.tree_map(lambda x: x + 100, Box("demo", 7))
        assert isinstance(mapped_box, Box) and mapped_box.tag == "demo" and mapped_box.val == 107
    finally:
        ox.unregister_tree_node(Box)


# ---------------------------------------------------------------------------
# 2. `nn.utils`: Norms, Weight/Spectral Norm, Fusion, Reparam, Prune, RNN
# ---------------------------------------------------------------------------

def test_parameters_vector_and_total_norm():
    m = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    vec = ox.parameters_to_vector(m.parameters())
    assert list(vec.shape) == [4 * 3 + 3 + 3 * 2 + 2]

    zeros_vec = ox.zeros(list(vec.shape))
    ox.vector_to_parameters(zeros_vec, m.parameters())
    assert all(np.allclose(p.to_numpy(), 0.0) for p in m.parameters())

    t1 = ox.tensor([3.0, 4.0])
    t2 = ox.tensor([12.0])
    n2 = float(ox.total_norm([t1, t2], norm_type=2.0).item())
    ninf = float(ox.total_norm([t1, t2], norm_type=float("inf")).item())
    assert abs(n2 - 13.0) < 1e-5
    assert abs(ninf - 12.0) < 1e-5

    bad = ox.tensor([float("nan"), 1.0])
    with pytest.raises(RuntimeError):
        ox.total_norm([bad], error_if_nonfinite=True)


def test_weight_norm_and_spectral_norm_on_conv_and_linear():
    x_img = ox.randn([2, 3, 6, 6])
    conv = nn.Conv2d(3, 4, kernel_size=3)
    y_before = conv(x_img).to_numpy()

    conv_wn = ox.weight_norm(conv, name="weight", dim=0)
    assert hasattr(conv_wn, "weight_g") and hasattr(conv_wn, "weight_v")
    y_wn = conv_wn(x_img)
    assert np.allclose(y_before, y_wn.to_numpy(), atol=1e-5)
    y_wn.sum().backward()
    assert conv_wn.weight_g.grad is not None and conv_wn.weight_v.grad is not None

    ox.remove_weight_norm(conv_wn)
    assert not hasattr(conv_wn, "weight_g")
    assert np.allclose(y_before, conv_wn(x_img).to_numpy(), atol=1e-5)

    # Spectral norm on native `nn.Linear` (scaled large so singular value > 1 initially)
    lin = nn.Linear(6, 4)
    lin.weight = ox.from_numpy((np.random.default_rng(1).standard_normal((4, 6)) * 8.0).astype(np.float32))
    sn_lin = ox.spectral_norm(lin, n_power_iterations=15)
    x_vec = ox.randn([3, 6])
    for _ in range(5):
        out = sn_lin(x_vec)
    out.sum().backward()
    assert sn_lin.weight_orig.grad is not None

    w_normed = sn_lin.weight.detach().to_numpy()
    top_sv = float(np.linalg.svd(w_normed, compute_uv=False)[0])
    assert abs(top_sv - 1.0) < 0.05

    sn_lin.eval()
    y_eval = sn_lin(x_vec).to_numpy()
    ox.remove_spectral_norm(sn_lin)
    assert np.allclose(y_eval, sn_lin(x_vec).to_numpy(), atol=1e-5)


def test_conv_and_linear_batchnorm_fusion():
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    bn2d = nn.BatchNorm2d(5)
    for _ in range(4):
        _ = bn2d(conv(ox.randn([4, 3, 8, 8])))
    bn2d.weight = nn.Parameter(ox.tensor([1.2, 0.7, 1.5, 0.9, 2.0]))
    bn2d.bias = nn.Parameter(ox.tensor([0.1, -0.3, 0.5, 0.0, -0.2]))
    conv.eval()
    bn2d.eval()

    fused_conv = ox.fuse_conv_batchnorm(conv, bn2d)
    x_test = ox.randn([2, 3, 8, 8])
    ref = bn2d(conv(x_test)).to_numpy()
    got = fused_conv(x_test).to_numpy()
    assert np.allclose(ref, got, atol=1e-5)

    lin = nn.Linear(6, 4)
    bn1d = nn.BatchNorm1d(4)
    for _ in range(4):
        _ = bn1d(lin(ox.randn([5, 6])))
    lin.eval()
    bn1d.eval()
    fused_lin = ox.fuse_linear_batchnorm(lin, bn1d)
    x_lin = ox.randn([3, 6])
    assert np.allclose(bn1d(lin(x_lin)).to_numpy(), fused_lin(x_lin).to_numpy(), atol=1e-5)


def test_reparam_apply_with_weights_prune_and_rnn_unpad():
    calls = {"n": 0}

    class Symmetric(nn.Module):
        def forward(self, w):
            calls["n"] += 1
            return (w + w.transpose(0, 1)) * 0.5

    class ToyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(ox.randn([4, 4]))

        def forward(self, x):
            return x.matmul(self.weight) + x.matmul(self.weight)

    m = ToyLinear()
    ox.bind_reparam(m, "weight", Symmetric())
    assert ox.has_reparam(m, "weight")
    x = ox.randn([2, 4])
    with ox.cache_reparam():
        y = m(x)
    assert np.allclose(m.weight.to_numpy(), m.weight.to_numpy().T, atol=1e-6)
    y.sum().backward()
    ox.unbind_reparam(m, "weight", leave_transformed=True)
    assert not ox.has_reparam(m)
    assert np.allclose(m.weight.to_numpy(), m.weight.to_numpy().T, atol=1e-6)

    # `apply_with_weights` does not mutate original weights
    orig_w = m.weight.to_numpy().copy()
    alt_w = ox.eye(4) * 3.0
    out_alt = ox.apply_with_weights(m, {"weight": alt_w}, x)
    assert np.allclose(out_alt.to_numpy(), x.to_numpy() * 6.0, atol=1e-5)
    assert np.allclose(m.weight.to_numpy(), orig_w, atol=1e-6)

    # Structured & mask pruning
    nn.utils.prune.ln_structured_prune(m, "weight", amount=2, n=2, dim=0)
    assert nn.utils.prune.is_pruned(m)
    row_norms = np.linalg.norm(m.weight.to_numpy(), axis=1)
    assert int(np.sum(row_norms == 0.0)) == 2
    nn.utils.prune.remove_pruning(m, "weight")
    assert not nn.utils.prune.is_pruned(m)

    # RNN left/right padding + unpad + unpack
    s1 = ox.tensor([1.0, 2.0, 3.0])
    s2 = ox.tensor([9.0])
    left_padded = nn.utils.rnn.pad_sequence([s1, s2], batch_first=True, padding_value=-1.0, padding_side="left")
    assert np.allclose(left_padded.to_numpy(), [[1.0, 2.0, 3.0], [-1.0, -1.0, 9.0]])
    right_padded = nn.utils.rnn.pad_sequence([s1, s2], batch_first=True, padding_value=0.0)
    unpadded = nn.utils.rnn.unpad_sequence(right_padded, [3, 1], batch_first=True)
    assert np.allclose(unpadded[0].to_numpy(), [1.0, 2.0, 3.0])
    assert np.allclose(unpadded[1].to_numpy(), [9.0])
    packed = nn.utils.rnn.pack_sequence([s1, s2])
    unpacked = nn.utils.rnn.unpack_sequence(packed)
    assert len(unpacked) == 2 and np.allclose(unpacked[1].to_numpy(), [9.0])


# ---------------------------------------------------------------------------
# 3. `utils.*` Submodules: Hooks, SysInfo, FLOPs, DLPack, Timing, Events, Data
# ---------------------------------------------------------------------------

def test_runtime_flops_hooks_dlpack_timing_and_events():
    reg = {}
    with ox.HookHandle(reg) as h:
        reg[h.id] = lambda: None
        assert h.id in reg
    assert h.id not in reg

    info = ox.collect_runtime_info()
    assert info.cpu_count >= 1
    assert "Oxideon Runtime Environment" in ox.format_runtime_info()

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 4, bias=True)
            self.fc2 = nn.Linear(4, 2, bias=False)

        def forward(self, z):
            return self.fc2(self.fc1(z).relu())

    mlp = TinyMLP()
    inp = ox.randn([3, 8])
    total_fl = ox.count_flops(mlp, inp)
    expected_fl = (2 * 3 * 8 * 4 + 3 * 4) + (2 * 3 * 4 * 2)
    assert total_fl == expected_fl

    # DLPack round-trip via `ox.utils.dlpack`
    orig = ox.tensor([[1.5, -2.5], [3.0, 4.0]])
    cap = ox.utils.dlpack.to_dlpack(orig)
    back = ox.utils.dlpack.from_dlpack(cap)
    assert np.allclose(orig.to_numpy(), back.to_numpy())

    # `data.to_tensors` preserves shapes without batching
    converted = ox.data.to_tensors({"arr": np.array([1.0, 2.0], dtype=np.float32), "label": np.int64(3), "note": "keep"})
    assert isinstance(converted["arr"], ox.Tensor) and list(converted["arr"].shape) == [2]
    assert isinstance(converted["label"], ox.Tensor) and int(converted["label"].item()) == 3
    assert converted["note"] == "keep"

    # MicroTimer & TimingComparison
    timer = ox.MicroTimer(stmt="x + x", globals={"x": np.ones(16)})
    rep = timer.autorange(min_run_time=0.01)
    assert rep.median > 0.0
    table = ox.TimingComparison([rep]).format()
    assert "Benchmark" in table

    # SummaryWriter extended event types
    with tempfile.TemporaryDirectory() as tmp:
        with ox.utils.events.SummaryWriter(log_dir=tmp) as writer:
            writer.add_histogram("weights", ox.randn([32]), global_step=1)
            writer.add_image("sample", ox.zeros([3, 8, 8]), global_step=1)
            writer.add_images("batch", ox.zeros([2, 3, 8, 8]), global_step=1)
            writer.add_hparams({"lr": 1e-3}, {"val_acc": 0.95}, global_step=1)
            writer.add_graph(mlp)
            json_path = writer.export_json()
            assert os.path.exists(json_path)
            assert len(writer.events) >= 6


# ---------------------------------------------------------------------------
# 4. Workflow, Encoding, Datasets, Caching, Registry, Vault & EMA
# ---------------------------------------------------------------------------

def test_workflow_encodings_datasets_vault_ema_and_registry():
    # `one_hot_encode`
    oh = ox.one_hot_encode([[0], [2], [1]], num_classes=4)
    assert list(oh.shape) == [3, 4]
    assert np.allclose(oh.to_numpy(), [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])

    # `axis_normalize` (zero-row safe + differentiable)
    v = ox.tensor([[3.0, 4.0], [0.0, 0.0]], requires_grad=True)
    vn = ox.axis_normalize(v, axis=-1, order=2)
    assert np.allclose(vn.to_numpy(), [[0.6, 0.8], [0.0, 0.0]], atol=1e-5)
    vn.sum().backward()
    assert v.grad is not None and np.all(np.isfinite(v.grad.to_numpy()))

    # `pad_sequences` pre/post pad & truncate
    seqs = [[1, 2, 3, 4], [5, 6], []]
    ps = ox.pad_sequences(seqs, maxlen=3, padding="post", truncating="pre", value=-1)
    assert np.array_equal(ps.to_numpy(), np.array([[2, 3, 4], [5, 6, -1], [-1, -1, -1]], dtype=np.int32))

    # `pack_batch` / `unpack_batch`
    packed = ox.pack_batch("x_in", "y_in")
    assert packed == ("x_in", "y_in")
    assert ox.unpack_batch(packed) == ("x_in", "y_in", None)

    # `compute_class_weight` & `compute_sample_weight`
    y_imb = [0, 0, 0, 1]
    cw = ox.compute_class_weight("balanced", classes=[0, 1], y=y_imb)
    assert np.allclose(cw, [4.0 / 6.0, 4.0 / 2.0])
    sw = ox.compute_sample_weight("balanced", y_imb)
    assert np.allclose(sw.to_numpy(), [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0], atol=1e-5)

    # `sliding_window_dataset` & `split_dataset`
    series = np.arange(10, dtype=np.float32)
    targets = np.arange(100, 110, dtype=np.float32)
    win_ds = ox.sliding_window_dataset(series, targets, sequence_length=3, sequence_stride=2, batch_size=None)
    assert len(win_ds) == 4
    w0, t0 = win_ds[0]
    assert np.allclose(w0.to_numpy(), [0.0, 1.0, 2.0]) and float(t0.item()) == 100.0

    left_ds, right_ds = ox.split_dataset(ox.arange(0, 10, 1), left_size=0.7, shuffle=False)
    assert len(left_ds) == 7 and len(right_ds) == 3

    # `ProgressBar`
    buf = io.StringIO()
    bar = ox.ProgressBar(target=4, width=10, interval=0.0, stateful_metrics=["lr"], stream=buf)
    bar.update(2, [("loss", 1.0), ("lr", 0.01)])
    bar.update(4, [("loss", 0.5), ("lr", 0.005)])
    rendered = buf.getvalue()
    assert "4/4" in rendered and "loss: 0.7500" in rendered and "lr: 0.0050" in rendered

    # `SerializationScope` & `register_serializable`
    @ox.register_serializable(package="TestPkg")
    class ScaledUnit:
        def __init__(self, factor: float = 2.0):
            self.factor = factor

        def get_config(self):
            return {"factor": self.factor}

    cfg = ox.serialize_config(ScaledUnit(factor=3.5))
    restored = ox.deserialize_config(cfg)
    assert isinstance(restored, ScaledUnit) and restored.factor == 3.5

    # `CheckpointVault`, `ExponentialMovingAverage`, `fetch_artifact`, folder datasets
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Folder datasets (image + text)
        for cls_idx, cls_name in enumerate(("cats", "dogs")):
            img_dir = root / "imgs" / cls_name
            txt_dir = root / "txts" / cls_name
            img_dir.mkdir(parents=True)
            txt_dir.mkdir(parents=True)
            for k in range(3):
                img = ox.from_numpy(np.full((3, 6, 6), 40 * (cls_idx + 1) + k, dtype=np.uint8))
                ox.vision.utils.save_image(img, img_dir / f"s_{k}.png")
                (txt_dir / f"doc_{k}.txt").write_text(f"{cls_name} sample {k}", encoding="utf-8")

        train_loader, val_loader = ox.image_folder_dataset(
            root / "imgs",
            label_mode="categorical",
            image_size=(8, 8),
            batch_size=2,
            validation_split=0.34,
            subset="both",
            seed=7,
        )
        assert train_loader.class_names == ["cats", "dogs"]
        xb, yb = next(iter(train_loader))
        assert list(xb.shape)[1:] == [3, 8, 8] and list(yb.shape)[1] == 2

        txt_ds = ox.text_folder_dataset(root / "txts", label_mode="binary", batch_size=None, shuffle=False)
        sample_txt, sample_lbl = txt_ds[0]
        assert "cats" in sample_txt and list(sample_lbl.shape) == [1]

        # `fetch_artifact` + zip extraction + zip-slip rejection
        good_zip = root / "payload.zip"
        with zipfile.ZipFile(good_zip, "w") as zf:
            zf.writestr("hello.txt", "verified payload")
        digest = hashlib.sha256(good_zip.read_bytes()).hexdigest()
        extracted_dir = ox.fetch_artifact(
            fname="payload.zip",
            origin=good_zip.as_uri(),
            cache_dir=root / "cache",
            file_hash=digest,
            extract=True,
        )
        assert (Path(extracted_dir) / "hello.txt").read_text(encoding="utf-8") == "verified payload"

        evil_zip = root / "evil.zip"
        with zipfile.ZipFile(evil_zip, "w") as zf:
            zf.writestr("../escaped.txt", "nope")
        with pytest.raises(RuntimeError):
            ox.fetch_artifact(fname="evil.zip", origin=evil_zip.as_uri(), cache_dir=root / "cache2", extract=True)

        # `CheckpointVault` rotation + `fetch_weights_url`
        vault = ox.CheckpointVault(root / "ckpts", max_to_keep=2)
        p = ox.tensor([1.0, 2.0])
        for s in range(1, 5):
            vault.save({"w": p * float(s)}, step=s)
        assert len(vault.checkpoints) == 2
        latest_sd = vault.restore_latest()
        assert np.allclose(latest_sd["w"].to_numpy(), [4.0, 8.0])

        loaded_url_sd = ox.fetch_weights_url(Path(vault.latest_checkpoint).as_uri(), model_dir=str(root / "url_cache"))
        assert np.allclose(loaded_url_sd["w"].to_numpy(), [4.0, 8.0])

        # `ShadowWeightEMA`
        param = ox.tensor([10.0, 20.0], requires_grad=True)
        ema = ox.ShadowWeightEMA([param], decay=0.5, use_num_updates=False)
        with ox.no_grad():
            param.copy_(ox.tensor([2.0, 4.0]))
        ema.update()
        assert np.allclose(ema.shadow_params[0].to_numpy(), [6.0, 12.0])
        with ema.average_parameters():
            assert np.allclose(param.to_numpy(), [6.0, 12.0])
        assert np.allclose(param.to_numpy(), [2.0, 4.0])


# ---------------------------------------------------------------------------
# 5. `vision.utils`: Grid, PNG Saving, Boxes, Masks & Keypoints
# ---------------------------------------------------------------------------

def test_vision_utils_grid_save_and_drawing():
    batch = ox.rand([5, 1, 8, 10])
    grid = ox.vision.utils.image_grid(batch, nrow=3, padding=2, normalize=True, scale_each=True)
    # 2 rows x 3 cols: H = (8+2)*2 + 2 = 22, W = (10+2)*3 + 2 = 38, single-channel expanded to 3
    assert list(grid.shape) == [3, 22, 38]

    bio = io.BytesIO()
    ox.vision.utils.save_image(batch, bio, nrow=3, padding=1)
    png_bytes = bio.getvalue()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    canvas = ox.from_numpy(np.zeros((3, 24, 24), dtype=np.uint8))
    boxes = ox.tensor([[2.0, 8.0, 18.0, 20.0]])
    boxed = ox.vision.utils.draw_bounding_boxes(canvas, boxes, labels=["OBJ1"], colors="red", fill=True, width=2)
    assert boxed.dtype == "u8" and list(boxed.shape) == [3, 24, 24]
    assert int(boxed.to_numpy().sum()) > 0

    mask = ox.from_numpy(np.eye(24, dtype=bool))
    masked = ox.vision.utils.draw_segmentation_masks(canvas, mask, alpha=0.75, colors=[(0, 255, 0)])
    assert int(masked.to_numpy()[1, 5, 5]) == 191

    kpts = ox.tensor([[[4.0, 4.0, 1.0], [18.0, 18.0, 1.0], [10.0, 10.0, 0.0]]])
    drawn_kp = ox.vision.utils.draw_keypoints(canvas, kpts, connectivity=[(0, 1), (1, 2)], colors="#00FFFF", radius=2)
    assert int(drawn_kp.to_numpy()[1, 4, 4]) == 255
    assert int(drawn_kp.to_numpy()[2, 11, 11]) == 255
