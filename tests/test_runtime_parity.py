"""
Oxideon Comprehensive Runtime & Engine API Test Suite.
Validates multi-dimensional tensor factories, in-place mutations,
dynamic shapes, autograd graph toggling, hardware placement, and end-to-end model training.
"""
import pytest
import numpy as np

import oxideon as ox
import oxideon.nn as nn
import oxideon.optim as optim
import oxideon.nn.functional as F
from oxideon.utils.data import DataLoader, TensorDataset


def test_top_level_tensor_factories():
    z1 = ox.zeros(2, 3)
    z2 = ox.zeros([2, 3])
    z3 = ox.zeros((2, 3))
    assert z1.shape == (2, 3)
    assert z2.shape == (2, 3)
    assert z3.shape == (2, 3)
    assert z1.shape == [2, 3]

    o1 = ox.ones(3, 4)
    assert o1.shape == (3, 4)
    assert float(o1.sum().item()) == 12.0

    rn = ox.randn(4, 5)
    assert rn.shape == (4, 5)

    u = ox.rand(2, 2)
    assert u.shape == (2, 2)

    e = ox.empty(3, 3)
    assert e.shape == (3, 3)


def test_tensor_constructor_from_data():
    t1 = ox.tensor([1, 2, 3, 4])
    assert t1.shape == (4,)
    assert t1.dtype == "i64"

    t2 = ox.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=ox.float32)
    assert t2.shape == (2, 2)
    assert t2.dtype == "f32"

    t3 = ox.tensor([True, False, True])
    assert t3.shape == (3,)
    assert t3.dtype == "bool"

    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    t4 = ox.from_numpy(arr)
    assert t4.shape == (3,)
    assert t4.tolist() == [10.0, 20.0, 30.0]


def test_arange_linspace_eye():
    a1 = ox.arange(10)
    assert a1.shape == (10,)
    assert a1.tolist() == list(range(10))

    a2 = ox.arange(2, 10, 2)
    assert a2.shape == (4,)
    assert a2.tolist() == [2.0, 4.0, 6.0, 8.0]

    lin = ox.linspace(0.0, 1.0, 5)
    assert lin.shape == (5,)
    assert np.allclose(lin.tolist(), [0.0, 0.25, 0.5, 0.75, 1.0])

    ey = ox.eye(3)
    assert ey.shape == (3, 3)
    assert float(ey.sum().item()) == 3.0


def test_full_and_like_methods():
    f = ox.full((2, 3), 7.0)
    assert f.shape == (2, 3)
    assert float(f.sum().item()) == 42.0

    z_like = ox.zeros_like(f)
    assert z_like.shape == f.shape
    assert float(z_like.sum().item()) == 0.0

    o_like = ox.ones_like(f)
    assert o_like.shape == f.shape
    assert float(o_like.sum().item()) == 6.0

    f_like = ox.full_like(f, 3.0)
    assert f_like.shape == f.shape
    assert float(f_like.sum().item()) == 18.0


def test_randint_and_randperm():
    ri = ox.randint(0, 10, size=(5, 5))
    assert ri.shape == (5, 5)
    arr = ri.to_numpy()
    assert np.all(arr >= 0) and np.all(arr < 10)

    rp = ox.randperm(8)
    assert rp.shape == (8,)
    assert sorted(rp.tolist()) == list(range(8))


def test_tensor_metadata_and_inspection():
    t = ox.randn(3, 4, 5)
    assert t.size() == (3, 4, 5)
    assert t.size(0) == 3
    assert t.size(1) == 4
    assert t.size(2) == 5
    assert t.dim() == 3
    assert t.ndimension() == 3
    assert t.numel() == 60
    assert t.nelement() == 60
    assert t.is_contiguous() is True


def test_tensor_inplace_mutations():
    x = ox.zeros(2, 2)
    x.add_(5.0)
    assert x.tolist() == [[5.0, 5.0], [5.0, 5.0]]

    x.sub_(2.0)
    assert x.tolist() == [[3.0, 3.0], [3.0, 3.0]]

    x.mul_(4.0)
    assert x.tolist() == [[12.0, 12.0], [12.0, 12.0]]

    x.div_(3.0)
    assert x.tolist() == [[4.0, 4.0], [4.0, 4.0]]

    x.fill_(9.5)
    assert x.tolist() == [[9.5, 9.5], [9.5, 9.5]]

    x.zero_()
    assert x.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_autograd_context_managers():
    assert ox.is_grad_enabled() is True

    with ox.no_grad():
        assert ox.is_grad_enabled() is False

    assert ox.is_grad_enabled() is True

    with ox.set_grad_enabled(False):
        assert ox.is_grad_enabled() is False
        with ox.enable_grad():
            assert ox.is_grad_enabled() is True
        assert ox.is_grad_enabled() is False

    assert ox.is_grad_enabled() is True


def test_device_and_hardware_backends():
    d_cpu = ox.device("cpu")
    assert d_cpu.type == "cpu"
    assert str(d_cpu) == "cpu"

    d_cuda = ox.device("cuda:0")
    assert d_cuda.type == "cuda"
    assert d_cuda.index == 0

    assert isinstance(ox.cuda.is_available(), bool)
    assert isinstance(ox.cuda.device_count(), int)
    assert ox.backends.cudnn.enabled is True


def test_utils_data_and_checkpoints():
    x = ox.randn(20, 4)
    y = ox.randint(0, 2, size=(20,))
    ds = TensorDataset(x, y)
    assert len(ds) == 20

    loader = DataLoader(ds, batch_size=5, shuffle=True)
    batches = list(loader)
    assert len(batches) == 4
    b_x, b_y = batches[0]
    assert b_x.shape == (5, 4)
    assert b_y.shape == (5,)

    res = ox.utils.checkpoint.checkpoint(lambda t: t * 2.0, b_x)
    assert res.shape == b_x.shape


def test_compile_decorator():
    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    model = ToyNet()
    compiled_model = ox.compile(model)
    x = ox.randn(3, 4)
    out = compiled_model(x)
    assert out.shape == (3, 2)
    assert getattr(compiled_model, "_compiled", False) is True


def test_end_to_end_neural_training_pipeline():
    ox.manual_seed(42)

    class ClassificationNetwork(nn.Module):
        def __init__(self, in_features: int, hidden: int, num_classes: int):
            super().__init__()
            self.layer1 = nn.Linear(in_features, hidden)
            self.act = nn.GELU()
            self.layer2 = nn.Linear(hidden, num_classes)

        def forward(self, x):
            return self.layer2(self.act(self.layer1(x)))

    net = ClassificationNetwork(8, 16, 3)
    optimizer = optim.AdamW(net.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    x = ox.randn(32, 8)
    y = ox.randint(0, 3, size=(32,))

    optimizer.zero_grad()
    logits = net(x)
    loss = criterion(logits, y)
    loss_val = float(loss.item())
    assert not np.isnan(loss_val)
    loss.backward()
    optimizer.step()

    sd = net.state_dict()
    assert "layer1.weight" in sd
    assert "layer2.weight" in sd

    net2 = ClassificationNetwork(8, 16, 3)
    net2.load_state_dict(sd)

    out1 = net(x).to_numpy()
    out2 = net2(x).to_numpy()
    assert np.allclose(out1, out2, atol=1e-5)
