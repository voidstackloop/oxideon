"""Comprehensive tests for Oxideon Neural Network Modules, Containers, and Optimizers.

Validates ParameterList, ParameterDict, Unflatten, Full Transformer architectures,
1D/2D Convolutions, Normalizations, Spatial modules, Loss modules,
first- and second-order optimizers, and custom Autograd Functions.
"""

import math
import numpy as np
import pytest

import oxideon as ox
import oxideon.nn as nn
import oxideon.optim as optim


def test_parameter_list_and_dict():
    p1 = ox.randn(2, 3)
    p2 = ox.randn(4)
    plist = nn.ParameterList([p1, p2])
    assert len(plist) == 2
    assert plist[0].shape == (2, 3)
    plist.append(ox.randn(5))
    assert len(plist) == 3

    pdict = nn.ParameterDict({"w1": p1, "w2": p2})
    assert len(pdict) == 2
    assert "w1" in pdict
    assert pdict["w1"].shape == (2, 3)
    pdict["w3"] = ox.randn(6)
    assert len(pdict) == 3


def test_unflatten_module():
    uf = nn.Unflatten(dim=1, unflattened_size=(2, 3))
    x = ox.randn(4, 6)
    out = uf(x)
    assert out.shape == (4, 2, 3)

    uf_neg = nn.Unflatten(dim=-1, unflattened_size=(2, 3))
    out_neg = uf_neg(x)
    assert out_neg.shape == (4, 2, 3)


def test_transformer_full_architecture():
    # Test TransformerEncoderLayer
    enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64)
    src = ox.randn(10, 2, 32)  # (seq_len, batch, d_model)
    enc_out = enc_layer(src)
    assert enc_out.shape == (10, 2, 32)

    # Test TransformerEncoder
    encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
    enc_stack = encoder(src)
    assert enc_stack.shape == (10, 2, 32)

    # Test TransformerDecoderLayer
    dec_layer = nn.TransformerDecoderLayer(d_model=32, nhead=4, dim_feedforward=64)
    tgt = ox.randn(8, 2, 32)
    dec_out = dec_layer(tgt, enc_stack)
    assert dec_out.shape == (8, 2, 32)

    # Full Transformer
    tf = nn.Transformer(d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64)
    tf_out = tf(src, tgt)
    assert tf_out.shape == (8, 2, 32)


def test_conv1d_and_conv_transpose2d():
    c1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
    x1 = ox.randn(2, 3, 16)
    y1 = c1(x1)
    assert y1.shape == (2, 6, 16)

    ct2 = nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=3, padding=1)
    x2 = ox.randn(2, 6, 8, 8)
    y2 = ct2(x2)
    assert y2.shape == (2, 3, 8, 8)


def test_pixel_shuffle_and_fold_unfold():
    ps = nn.PixelShuffle(upscale_factor=2)
    x = ox.randn(1, 16, 4, 4)
    y = ps(x)
    assert y.shape == (1, 4, 8, 8)

    pus = nn.PixelUnshuffle(downscale_factor=2)
    rec = pus(y)
    assert rec.shape == (1, 16, 4, 4)

    unfold = nn.Unfold(kernel_size=3, padding=1)
    uf = unfold(x)
    assert uf.shape == (1, 144, 16)


def test_dropouts_and_normalizations():
    d1 = nn.Dropout1d(p=0.2)
    d2 = nn.Dropout2d(p=0.2)
    ad = nn.AlphaDropout(p=0.2)
    x = ox.randn(2, 4, 8, 8)
    assert d2(x).shape == x.shape
    assert ad(x).shape == x.shape

    # InstanceNorm1d and 2d
    in1 = nn.InstanceNorm1d(num_features=4)
    assert in1(ox.randn(2, 4, 16)).shape == (2, 4, 16)

    in2 = nn.InstanceNorm2d(num_features=4)
    assert in2(x).shape == (2, 4, 8, 8)


def test_class_based_losses():
    bce = nn.BCELoss()
    p = ox.tensor([[0.2, 0.8], [0.9, 0.1]])
    t = ox.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss_bce = bce(p, t)
    assert loss_bce.item() < 0.5

    sl1 = nn.SmoothL1Loss(beta=1.0)
    loss_sl1 = sl1(p, t)
    assert loss_sl1.item() >= 0.0

    pnll = nn.PoissonNLLLoss()
    loss_pnll = pnll(p, t)
    assert not math.isnan(loss_pnll.item())


def test_optimizer_suite():
    # Test RMSprop, Adagrad, Adadelta, Adamax, NAdam, RAdam, ASGD
    p = ox.randn(2, 2)
    p.grad = ox.ones(2, 2)

    for opt_cls in [
        optim.RMSprop,
        optim.Adagrad,
        optim.Adadelta,
        optim.Adamax,
        optim.NAdam,
        optim.RAdam,
        optim.ASGD,
    ]:
        param = ox.randn(2, 2)
        param.grad = ox.ones(2, 2)
        opt = opt_cls([param], lr=0.01)
        opt.step()
        assert param.shape == (2, 2)
        opt.zero_grad()
        assert param.grad is None or float(param.grad.sum().item()) == 0.0

    # Test LBFGS with quadratic closure
    p_opt = ox.tensor([2.0, -3.0])
    lbfgs = optim.LBFGS([p_opt], lr=0.5, max_iter=10)

    def closure():
        lbfgs.zero_grad()
        loss = (p_opt ** 2).sum()
        p_opt.grad = p_opt * 2.0
        return float(loss.item())

    init_loss = closure()
    final_loss = lbfgs.step(closure)
    assert final_loss < init_loss


def test_custom_autograd_function():
    class SquareFunction(ox.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            return grad_output * x * 2.0

    t = ox.tensor([3.0, 4.0])
    out = SquareFunction.apply(t)
    assert np.allclose(out.to_numpy(), [9.0, 16.0])


def test_tensor_manipulation_ops():
    a = ox.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = ox.tensor([[5.0, 6.0], [7.0, 8.0]])

    # mm, dot, outer
    assert ox.mm(a, b).shape == (2, 2)
    assert ox.dot(ox.tensor([1.0, 2.0]), ox.tensor([3.0, 4.0])).item() == 11.0
    assert ox.outer(ox.tensor([1.0, 2.0]), ox.tensor([3.0, 4.0])).shape == (2, 2)

    # stack & cat
    stk = ox.stack([a, b], dim=0)
    assert stk.shape == (2, 2, 2)
    cat_0 = ox.cat([a, b], dim=0)
    assert cat_0.shape == (4, 2)

    # where & masked_select
    w = ox.where(a > 2.0, a, 0.0)
    assert np.allclose(w.to_numpy(), [[0.0, 0.0], [3.0, 4.0]])
    ms = ox.masked_select(a, a > 2.0)
    assert np.allclose(ms.to_numpy(), [3.0, 4.0])

    # sort & topk
    vals, idx = ox.sort(ox.tensor([3.0, 1.0, 2.0]))
    assert np.allclose(vals.to_numpy(), [1.0, 2.0, 3.0])
    top_vals, top_idx = ox.topk(ox.tensor([3.0, 1.0, 2.0]), k=2)
    assert np.allclose(top_vals.to_numpy(), [3.0, 2.0])

    # roll & flip
    rl = ox.roll(ox.tensor([1.0, 2.0, 3.0]), shifts=1)
    assert np.allclose(rl.to_numpy(), [3.0, 1.0, 2.0])
    flp = ox.flip(ox.tensor([1.0, 2.0, 3.0]), dims=(0,))
    assert np.allclose(flp.to_numpy(), [3.0, 2.0, 1.0])
