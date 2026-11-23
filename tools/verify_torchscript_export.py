"""Loads the TorchScript archive written by tests/torchscript_export.rs with torch.jit.load and compares it with eager PyTorch.

    OXIDEON_TS_OUT=model.pt cargo test -p oxideon-core --test torchscript_export -- --ignored
    python tools/verify_torchscript_export.py model.pt
"""
import math, sys
import torch

m = torch.jit.load(sys.argv[1])
w1 = torch.tensor([math.sin(i * 0.37) for i in range(12)]).reshape(4, 3)
w2 = torch.tensor([math.cos(i * 0.91) for i in range(8)]).reshape(2, 4)
b1 = torch.tensor([0.1, -0.2, 0.3, 0.0])
x = torch.randn(5, 3)
F = torch.nn.functional
ref = torch.softmax(F.gelu(F.linear(torch.relu(F.linear(x, w1, b1)), w2)), -1)
diff = (m(x) - ref).abs().max().item()
print("max difference vs eager PyTorch:", diff)
assert diff < 1e-5, "TorchScript export disagrees with eager PyTorch"
