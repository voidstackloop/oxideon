"""Eleventh cross-validation batch: conv / pooling / norm / recurrent layers vs PyTorch.

    cargo test -p oxideon-core --test crossval_emit11 -- --nocapture --ignored > rust_values11.txt
    py -3.11 tools/crossval/check_batch11.py rust_values11.txt
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

rust = {}
for line in open(sys.argv[1], encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2:
        try:
            rust[parts[0]] = float(parts[1])
        except ValueError:
            pass


def genv(n, salt):
    return [math.sin(0.37 * i + salt) * 0.8 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)]


def t(shape, salt):
    n = 1
    for s in shape:
        n *= s
    return torch.tensor(genv(n, salt), dtype=torch.float32).reshape(shape)


ref = {}


def put(name, x):
    for i, v in enumerate(x.detach().reshape(-1).double().tolist()):
        ref[f"{name}_{i}"] = v


x = t((2, 4, 7, 6), 0.3)
put("conv2d", F.conv2d(x, t((6, 4, 3, 3), 1.0), t((6,), 1.5)))
put("conv2d_sp", F.conv2d(x, t((6, 4, 3, 2), 2.0), t((6,), 2.5), stride=(2, 1), padding=(1, 2)))
put("conv2d_groups", F.conv2d(x, t((6, 2, 3, 3), 3.0), t((6,), 3.5), padding=1, groups=2))
put("conv2d_dil", F.conv2d(x, t((3, 4, 3, 3), 4.0), t((3,), 4.5), padding=2, dilation=2))
put("conv1d", F.conv1d(t((2, 3, 11), 5.0), t((4, 3, 3), 6.0), t((4,), 6.5), stride=2, padding=1))
put("convt2d", F.conv_transpose2d(t((2, 3, 4, 5), 7.0), t((3, 2, 3, 3), 8.0), t((2,), 8.5), stride=2, padding=1))
put("maxpool", F.max_pool2d(x, 2, 2))
put("maxpool_k3s1", F.max_pool2d(x, 3, 1))
put("avgpool", F.avg_pool2d(x, (3, 2), (2, 2)))
put("adaptive_avg", F.adaptive_avg_pool2d(x, (3, 4)))
put("layernorm", F.layer_norm(t((3, 5, 6), 10.0), (6,), t((6,), 9.0), t((6,), 9.5), 1e-5))
put("rmsnorm", F.rms_norm(t((3, 5, 6), 12.0), (6,), t((6,), 11.0), 1e-6))
put("groupnorm", F.group_norm(x, 2, t((4,), 13.0), t((4,), 13.5), 1e-5))

seq = t((2, 5, 3), 14.0)


def load_rnn(mod, base):
    sd = {}
    for l in range(2 if mod.num_layers == 2 else 1):
        sd[f"weight_ih_l{l}"] = t(((mod.gate_mult) * 4, 3 if l == 0 else 4), base + 4.0 * l + 0.0 * 0) if False else None
    return sd


def fill(mod, ih_salt0, hh_salt0, gates):
    with torch.no_grad():
        for l in range(mod.num_layers):
            in_dim = 3 if l == 0 else 4
            getattr(mod, f"weight_ih_l{l}").copy_(t((gates * 4, in_dim), ih_salt0 + 4.0 * l))
            getattr(mod, f"bias_ih_l{l}").copy_(t((gates * 4,), ih_salt0 + 4.0 * l + 0.5))
            getattr(mod, f"weight_hh_l{l}").copy_(t((gates * 4, 4), hh_salt0 + 4.0 * l))
            getattr(mod, f"bias_hh_l{l}").copy_(t((gates * 4,), hh_salt0 + 4.0 * l + 0.5))


lstm = nn.LSTM(3, 4, 2, batch_first=True)
fill(lstm, 20.0, 22.0, 4)
o, (h, c) = lstm(seq)
put("lstm_out", o); put("lstm_h", h); put("lstm_c", c)
gru = nn.GRU(3, 4, 2, batch_first=True)
fill(gru, 30.0, 32.0, 3)
o, h = gru(seq)
put("gru_out", o); put("gru_h", h)
rnn = nn.RNN(3, 4, 1, batch_first=True, nonlinearity="tanh")
fill(rnn, 40.0, 42.0, 1)
o, h = rnn(seq)
put("rnn_out", o); put("rnn_h", h)

worst, bad = 0.0, []
for k, want in ref.items():
    if k not in rust:
        bad.append((k, "missing", want)); continue
    err = abs(rust[k] - want) / max(1.0, abs(want))
    worst = max(worst, err)
    if err > 2e-5:
        bad.append((k, rust[k], want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
groups = {}
for k, got, want in bad:
    groups.setdefault(k.rsplit("_", 1)[0], []).append((k, got, want))
for g, items in groups.items():
    k, got, want = items[0]
    print(f"MISMATCH {g}: {len(items)} values differ, e.g. {k}: rust={got} torch={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
