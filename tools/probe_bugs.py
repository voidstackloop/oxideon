import math
import numpy as np
import oxideon as ox
import oxideon.nn as nn
import oxideon.nn.functional as F

def probe():
    print("=== PROBING MODULES & FUNCTIONS ===")

    # 1. Linear with bias=False
    try:
        lin = nn.Linear(10, 5, bias=False)
        inp = ox.Tensor.randn([2, 10])
        out = lin(inp)
        assert out.shape == [2, 5]
        print("Linear(bias=False): OK")
    except Exception as e:
        print("Linear(bias=False) ERROR:", e)

    # 2. Embedding
    try:
        emb = nn.Embedding(100, 16)
        idx = ox.Tensor.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
        e_out = emb(idx)
        print("Embedding forward:", e_out.shape)
    except Exception as e:
        print("Embedding ERROR:", e)

    # 3. LayerNorm with normalized_shape as int vs list
    try:
        ln1 = nn.LayerNorm(16)
        x = ox.Tensor.randn([2, 8, 16])
        out1 = ln1(x)
        print("LayerNorm(int): OK, shape", out1.shape)
    except Exception as e:
        print("LayerNorm(int) ERROR:", e)

    try:
        ln2 = nn.LayerNorm([8, 16])
        out2 = ln2(x)
        print("LayerNorm([8, 16]): OK, shape", out2.shape)
    except Exception as e:
        print("LayerNorm([8, 16]) ERROR:", e)

    # 4. BatchNorm1d with 2D and 3D
    try:
        bn = nn.BatchNorm1d(16)
        x2d = ox.Tensor.randn([4, 16])
        out_bn2d = bn(x2d)
        print("BatchNorm1d(2D): OK, shape", out_bn2d.shape)
    except Exception as e:
        print("BatchNorm1d(2D) ERROR:", e)

    try:
        x3d = ox.Tensor.randn([4, 16, 10])
        out_bn3d = bn(x3d)
        print("BatchNorm1d(3D): OK, shape", out_bn3d.shape)
    except Exception as e:
        print("BatchNorm1d(3D) ERROR:", e)

    # 5. Conv2d with stride, padding, dilation
    try:
        conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        img = ox.Tensor.randn([2, 3, 32, 32])
        c_out = conv(img)
        print("Conv2d forward: OK, shape", c_out.shape)
    except Exception as e:
        print("Conv2d ERROR:", e)

    # 6. MaxPool2d
    try:
        pool = nn.MaxPool2d(2, stride=2)
        p_out = pool(c_out)
        print("MaxPool2d forward: OK, shape", p_out.shape)
    except Exception as e:
        print("MaxPool2d ERROR:", e)

    # 7. AdaptiveAvgPool2d
    try:
        adap = nn.AdaptiveAvgPool2d((1, 1))
        a_out = adap(c_out)
        print("AdaptiveAvgPool2d: OK, shape", a_out.shape)
    except Exception as e:
        print("AdaptiveAvgPool2d ERROR:", e)

    # 8. Dropout in train vs eval
    try:
        drop = nn.Dropout(p=0.5)
        d_tr = drop(ox.Tensor.ones([10, 10]))
        drop.eval()
        d_ev = drop(ox.Tensor.ones([10, 10]))
        assert np.allclose(d_ev.to_numpy(), 1.0)
        print("Dropout train/eval: OK")
    except Exception as e:
        print("Dropout ERROR:", e)

    # 9. Optimizer step & zero_grad
    try:
        param = ox.Tensor.randn([4, 4], requires_grad=True)
        opt = ox.optim.AdamW([param], lr=1e-3, weight_decay=1e-2)
        loss = (param ** 2).sum()
        loss.backward()
        assert param.grad is not None
        opt.step()
        opt.zero_grad()
        assert param.grad is None or np.allclose(param.grad.to_numpy(), 0.0)
        print("AdamW step & zero_grad: OK")
    except Exception as e:
        print("AdamW ERROR:", e)

    # 10. Special math functions
    for fn_name in ["erf", "sigmoid", "silu", "tanh", "gelu"]:
        try:
            fn = getattr(F, fn_name)
            res = fn(ox.Tensor.randn([5]))
            print(f"F.{fn_name}: OK")
        except Exception as e:
            print(f"F.{fn_name} ERROR:", e)

if __name__ == "__main__":
    probe()
