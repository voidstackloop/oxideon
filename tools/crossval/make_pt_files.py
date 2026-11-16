"""Writes PyTorch checkpoints of every flavour into target/pt_files for crossval batch 48 (needs torch)."""
import os, sys, torch, torch.nn as nn
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "target", "pt_files"); os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
def save(name, obj): torch.save(obj, os.path.join(out, name + ".pt"))
save("sd_linear", nn.Linear(8, 4).state_dict())
save("conv_bn", nn.Sequential(nn.Conv2d(3, 5, 3), nn.BatchNorm2d(5)).state_dict())
save("dtypes", {"f64": torch.randn(3, 2, dtype=torch.float64), "f16": torch.randn(5).half(), "bf16": torch.randn(4, 3).bfloat16(), "i64": torch.arange(-3, 4),
                "i32": torch.arange(6, dtype=torch.int32).reshape(2, 3) - 2, "u8": torch.arange(10, dtype=torch.uint8), "bool": torch.tensor([True, False, True]),
                "scalar": torch.tensor(3.5), "empty": torch.zeros(0, 3), "inf_nan": torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0])})
base = torch.randn(4, 6)
save("views", {"transposed": base.t(), "slice": base[1:3, ::2], "shared_a": base, "shared_b": base.view(-1)[2:10], "expanded": torch.randn(3, 1).expand(3, 4)})
sd = nn.Linear(3, 2).state_dict()
save("ckpt", {"epoch": 3, "model_state_dict": sd, "optimizer": {"state": {}, "param_groups": [{"lr": 0.1, "params": [0, 1]}]}, "loss": 0.25})
save("big", {"emb": nn.Embedding(1000, 64).state_dict()["weight"], "x": torch.randn(1_000_000)})
save("list", [torch.randn(2, 2), torch.randn(3)])
save("single", torch.randn(4, 4))
save("param", {"p": nn.Parameter(torch.randn(3, 3)), "buf": torch.ones(2)})
class Net(nn.Module):
    def __init__(s): super().__init__(); s.a = nn.Linear(4, 4); s.ln = nn.LayerNorm(4); s.e = nn.Embedding(10, 4)
save("net_sd", Net().state_dict())
save("compressed_free", {"w": torch.randn(16, 16)})
print(sorted(os.listdir(out)))
