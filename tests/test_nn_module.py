"""Comprehensive test suite for Oxideon OOP neural network module architecture (`ox.nn.Module`)."""

import pytest
import numpy as np
import oxideon as ox
import oxideon.nn as nn


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


class CustomParameterLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Direct Parameter definition
        self.w = nn.Parameter(ox.ones([in_features, out_features], dtype="f32"))
        self.b = nn.Parameter(ox.zeros([out_features], dtype="f32"))

    def forward(self, x):
        return x.matmul(self.w).add(self.b)


def test_module_subclassing_and_parameters():
    model = SimpleClassifier(4, 8, 2)
    assert isinstance(model, nn.Module)
    assert isinstance(model, ox.Module)

    params = model.parameters()
    # fc1.weight, fc1.bias, fc2.weight, fc2.bias -> 4 parameters
    assert len(params) == 4

    named_params = dict(model.named_parameters())
    assert "fc1.weight" in named_params
    assert "fc1.bias" in named_params
    assert "fc2.weight" in named_params
    assert "fc2.bias" in named_params


def test_forward_pass_and_autograd():
    model = SimpleClassifier(4, 8, 2)
    x = ox.Tensor([1.0, 2.0, 3.0, 4.0], shape=[1, 4], dtype="f32")
    out = model(x)
    assert out.shape == [1, 2]

    # Compute loss and backward
    loss = out.powf(2.0).sum()
    loss.backward()

    # Check that every parameter in the model received gradients
    for p in model.parameters():
        grad = p.grad() if callable(p.grad) else p.grad
        assert grad is not None
        assert grad.shape == p.shape


def test_custom_parameters_in_module():
    layer = CustomParameterLayer(3, 2)
    assert len(layer.parameters()) == 2
    x = ox.Tensor([1.0, 1.0, 1.0], shape=[1, 3], dtype="f32")
    out = layer(x)
    assert out.shape == [1, 2]
    # out = 1*1 + 1*1 + 1*1 = 3 for both outputs
    assert np.allclose(out.numpy(), [[3.0, 3.0]])


def test_end_to_end_training_loop():
    # Synthetic problem: learn f(x) = x * 2.0
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )
    optimizer = ox.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    x = ox.Tensor([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0], shape=[4, 2], dtype="f32")
    target = ox.Tensor([6.0, 10.0, 14.0, 18.0], shape=[4, 1], dtype="f32")

    initial_loss = None
    final_loss = None

    for epoch in range(25):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_state_dict_save_and_load():
    model1 = SimpleClassifier(4, 8, 2)
    sd = model1.state_dict()
    assert len(sd) == 4

    model2 = SimpleClassifier(4, 8, 2)
    # Forward before load
    x = ox.Tensor([0.5, -0.5, 1.0, -1.0], shape=[1, 4], dtype="f32")
    out1 = model1(x)

    model2.load_state_dict(sd)
    out2 = model2(x)
    assert np.allclose(out1.numpy(), out2.numpy(), atol=1e-5)


def test_sequential_container():
    seq = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    assert len(seq) == 3
    assert len(seq.parameters()) == 4

    # Indexing
    first_layer = seq[0]
    assert isinstance(first_layer, nn.Linear)

    # Append
    seq.append(nn.Linear(5, 1))
    assert len(seq) == 4
    assert len(seq.parameters()) == 6

    x = ox.ones([2, 10], dtype="f32")
    out = seq(x)
    assert out.shape == [2, 1]


def test_module_list():
    class MultiBranchNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Linear(4, 2),
                nn.Linear(4, 2),
                nn.Linear(4, 2),
            ])

        def forward(self, x):
            outs = [b(x) for b in self.branches]
            return outs[0] + outs[1] + outs[2]

    net = MultiBranchNet()
    assert len(net.parameters()) == 6
    x = ox.ones([1, 4], dtype="f32")
    out = net(x)
    assert out.shape == [1, 2]


def test_module_dict():
    class TaskRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = nn.ModuleDict({
                "clf": nn.Linear(4, 2),
                "reg": nn.Linear(4, 1),
            })

        def forward(self, x, task="clf"):
            return self.heads[task](x)

    router = TaskRouter()
    assert len(router.parameters()) == 4
    x = ox.ones([1, 4], dtype="f32")
    assert router(x, "clf").shape == [1, 2]
    assert router(x, "reg").shape == [1, 1]


def test_freeze_and_unfreeze():
    model = SimpleClassifier(4, 8, 2)
    model.freeze()
    for p in model.parameters():
        assert not p.requires_grad

    model.unfreeze()
    for p in model.parameters():
        assert p.requires_grad


def test_train_and_eval_modes():
    model = SimpleClassifier(4, 8, 2)
    assert model.training is True
    model.eval()
    assert model.training is False
    model.train()
    assert model.training is True
