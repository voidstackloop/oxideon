"""Tests for model and tensor serialization: save_file, load_file, and running loaded models."""

import os
import tempfile
import pickle
import numpy as np
import pytest
import oxideon as ox
import oxideon.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: ox.Tensor) -> ox.Tensor:
        h = self.relu(self.fc1(x))
        return self.fc2(h)


class DeepContainerNet(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.stem = nn.Linear(in_features, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.heads = nn.ModuleDict({
            "primary": nn.Linear(hidden_dim, num_classes),
            "aux": nn.Linear(hidden_dim, 1),
        })

    def forward(self, x: ox.Tensor, head: str = "primary") -> ox.Tensor:
        h = self.stem(x).relu()
        for block in self.blocks:
            h = block(h).relu()
        return self.heads[head](h)


def test_save_file_and_load_file_dictionary():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tensors.safetensors")
        w = ox.randn([4, 8])
        b = ox.zeros([8])
        tensors = {"weight": w, "bias": b}

        # Save
        ox.save_file(tensors, path)
        assert os.path.exists(path)

        # Load
        loaded = ox.load_file(path)
        assert "weight" in loaded and "bias" in loaded
        assert loaded["weight"].shape == [4, 8]
        assert loaded["bias"].shape == [8]

        np.testing.assert_allclose(w.to_numpy(), loaded["weight"].to_numpy(), atol=1e-7)
        np.testing.assert_allclose(b.to_numpy(), loaded["bias"].to_numpy(), atol=1e-7)


def test_save_load_roundtrip_all_dtypes_exact():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dtypes.safetensors")
        f32 = ox.from_numpy(((np.arange(24).reshape(4, 6) % 251) / 4).astype(np.float32))
        tensors = {
            "f32": f32,
            "f64": ox.from_numpy(((np.arange(24).reshape(4, 6) % 251) / 7).astype(np.float64)),
            "i32": ox.from_numpy(((np.arange(24).reshape(4, 6) % 251) - 100).astype(np.int32)),
            "i64": ox.from_numpy(np.array([2**60 + 1, -(2**60 + 1), 5, -7, 0, 2**62], dtype=np.int64)),
            "u8": ox.from_numpy((np.arange(24).reshape(4, 6) % 251).astype(np.uint8)),
            "bool": ox.from_numpy(np.array([True, False, True, False, True, False])),
            "f16": ox.from_numpy(np.array([1.0, 2.5, -3.0, 0.5], dtype=np.float32)).to_dtype("f16"),
            "bf16": ox.from_numpy(np.array([1.0, 2.5, -3.0, 0.5], dtype=np.float32)).to_dtype("bf16"),
            "strided": f32.transpose(0, 1),
        }
        ox.save_file(tensors, path)
        loaded = ox.load_file(path)
        assert set(loaded) == set(tensors)
        for name, t in tensors.items():
            lt = loaded[name]
            assert lt.shape == t.shape, name
            assert lt.dtype == t.dtype, name
            assert np.array_equal(lt.numpy(), t.numpy()), name


def test_save_file_inverted_arguments():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "inverted.safetensors")
        t = ox.ones([3, 3])
        # Calling save_file with filename first
        ox.save_file(path, {"w": t})
        loaded = ox.load_file(path)
        assert loaded["w"].shape == [3, 3]


def test_save_file_with_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "meta.safetensors")
        w = ox.ones([2, 2])
        meta = {"architecture": "SimpleClassifier", "framework": "Oxideon", "version": "1.0"}

        ox.save_file({"w": w}, path, metadata=meta)
        loaded_meta = ox.load_file_metadata(path)

        assert loaded_meta["architecture"] == "SimpleClassifier"
        assert loaded_meta["framework"] == "Oxideon"
        assert loaded_meta["version"] == "1.0"


def test_save_and_run_neural_network_module():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "classifier.safetensors")

        # 1. Instantiate and run original model
        model = SimpleClassifier(4, 16, 3)
        model.eval()
        x = ox.randn([5, 4])
        orig_out = model(x)

        # 2. Save model directly using ox.save_file and model.save_file
        ox.save_file(model, path)
        assert os.path.exists(path)

        # 3. Instantiate fresh clone with random weights
        clone = SimpleClassifier(4, 16, 3)
        clone.eval()
        clone_out_before = clone(x)
        assert not np.allclose(orig_out.to_numpy(), clone_out_before.to_numpy())

        # 4. Load saved weights into clone
        clone.load_file(path)

        # 5. Run inference on loaded model
        clone_out_after = clone(x)
        diff = np.max(np.abs(orig_out.to_numpy() - clone_out_after.to_numpy()))
        assert diff == 0.0, f"Loaded model output diverges from original: max diff = {diff}"


def test_load_file_direct_into_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "direct_load.safetensors")

        model = SimpleClassifier(6, 12, 2)
        model.eval()
        x = ox.randn([4, 6])
        expected = model(x)

        model.save_file(path)

        # Load directly via load_file(..., model=target)
        target = SimpleClassifier(6, 12, 2)
        target.eval()
        ret = ox.load_file(path, model=target)
        assert ret is target

        actual = target(x)
        np.testing.assert_allclose(expected.to_numpy(), actual.to_numpy(), atol=1e-7)


def test_deep_nested_model_save_and_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "deep_container.safetensors")

        model = DeepContainerNet(8, 16, 4)
        model.eval()
        x = ox.randn([3, 8])

        out_primary = model(x, head="primary")
        out_aux = model(x, head="aux")

        model.save_file(path)

        # Reconstruct into a new instance
        new_model = DeepContainerNet(8, 16, 4)
        new_model.eval()
        new_model.load_file(path)

        new_out_primary = new_model(x, head="primary")
        new_out_aux = new_model(x, head="aux")

        np.testing.assert_allclose(out_primary.to_numpy(), new_out_primary.to_numpy(), atol=1e-7)
        np.testing.assert_allclose(out_aux.to_numpy(), new_out_aux.to_numpy(), atol=1e-7)


def test_save_file_train_save_load_continue_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "training_ckpt.safetensors")

        # 1. Train model for 5 steps
        model = SimpleClassifier(4, 8, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = ox.optim.Adam(model.parameters(), lr=0.05)

        x = ox.randn([8, 4])
        target = ox.zeros([8])

        initial_loss = float(criterion(model(x), target).to_numpy().item())

        for _ in range(5):
            model.zero_grad()
            logits = model(x)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        loss_at_save = float(criterion(model(x), target).to_numpy().item())
        assert loss_at_save < initial_loss

        # 2. Save model
        model.save_file(path)

        # 3. Load model in a new instance and continue training
        resumed_model = SimpleClassifier(4, 8, 2)
        resumed_model.load_file(path)
        resumed_optimizer = ox.optim.Adam(resumed_model.parameters(), lr=0.05)

        # Confirm output is identical before further training
        resumed_loss_start = float(criterion(resumed_model(x), target).to_numpy().item())
        np.testing.assert_allclose(loss_at_save, resumed_loss_start, atol=1e-5)

        # Train 5 more steps
        for _ in range(5):
            resumed_model.zero_grad()
            logits = resumed_model(x)
            loss = criterion(logits, target)
            loss.backward()
            resumed_optimizer.step()

        final_loss = float(criterion(resumed_model(x), target).to_numpy().item())
        assert final_loss < loss_at_save, f"Expected final loss {final_loss} < {loss_at_save}"


def test_ox_save_and_load_full_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")

        model = SimpleClassifier(4, 8, 2)
        checkpoint = {
            "epoch": 10,
            "best_acc": 0.985,
            "state_dict": model.state_dict(),
        }

        ox.save(checkpoint, path)
        loaded_ckpt = ox.load(path)

        assert loaded_ckpt["epoch"] == 10
        assert loaded_ckpt["best_acc"] == 0.985
        assert len(loaded_ckpt["state_dict"]) == len(model.state_dict())

        clone = SimpleClassifier(4, 8, 2)
        clone.load_state_dict(loaded_ckpt["state_dict"])

        x = ox.randn([2, 4])
        np.testing.assert_allclose(model(x).to_numpy(), clone(x).to_numpy(), atol=1e-7)


def test_tensor_pickle_support():
    t = ox.randn([3, 5])
    pickled = pickle.dumps(t)
    unpickled = pickle.loads(pickled)

    assert unpickled.shape == [3, 5]
    np.testing.assert_allclose(t.to_numpy(), unpickled.to_numpy(), atol=1e-7)

    t_grad = ox.ones([2, 2]).requires_grad_(True)
    unpickled_grad = pickle.loads(pickle.dumps(t_grad))
    assert unpickled_grad.requires_grad is True


def test_tensor_to_numpy_and_numpy_methods():
    t = ox.ones([2, 3])
    arr1 = t.to_numpy()
    arr2 = t.numpy()
    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)
    assert arr1.shape == (2, 3)
    assert np.all(arr1 == 1.0)
    assert np.all(arr2 == 1.0)
