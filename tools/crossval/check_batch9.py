"""Ninth cross-validation batch: nn functional ops and MultiheadAttention vs PyTorch.

    cargo test -p oxideon-core --test crossval_emit9 -- --nocapture --ignored > rust_values9.txt
    py -3.11 tools/crossval/check_batch9.py rust_values9.txt
"""
import math
import sys

import torch
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
    return torch.tensor(genv(n, salt), dtype=torch.float64).reshape(shape)


ref = {}


def put(name, x):
    for i, v in enumerate(x.detach().reshape(-1).tolist()):
        ref[f"{name}_{i}"] = v


q, k, v = t((2, 3, 5, 4), 0.3), t((2, 3, 6, 4), 1.1), t((2, 3, 6, 4), 2.2)
put("sdpa", F.scaled_dot_product_attention(q, k, v))
put("sdpa_scale", F.scaled_dot_product_attention(q, k, v, scale=0.37))
put("sdpa_causal", F.scaled_dot_product_attention(q, t((2, 3, 5, 4), 1.4), t((2, 3, 5, 4), 2.6), is_causal=True))
put("sdpa_mask", F.scaled_dot_product_attention(q, k, v, attn_mask=t((5, 6), 3.3)))
x = t((2, 6, 3, 3), 0.9)
w6, b6 = t((6,), 4.0), t((6,), 5.0)
put("group_norm", F.group_norm(x, 3, w6, b6, 1e-5))
put("layer_norm", F.layer_norm(t((4, 6), 1.7), (6,), w6, b6, 1e-5))
rm = t((6,), 6.0)
rv = t((6,), 7.0).abs() + 0.3
put("batch_norm_eval", F.batch_norm(x, rm.clone(), rv.clone(), w6, b6, False, 0.1, 1e-5))
rm2, rv2 = rm.clone(), rv.clone()
put("batch_norm_train", F.batch_norm(x, rm2, rv2, w6, b6, True, 0.1, 1e-5))
put("bn_running_mean", rm2)
put("bn_running_var", rv2)
put("lrn", F.local_response_norm(x, 3, alpha=1e-3, beta=0.75, k=1.5))
img = t((1, 2, 4, 5), 0.2)
for name, kw in [("nearest", dict(mode="nearest")), ("linear", dict(mode="bilinear", align_corners=False)), ("linear_ac", dict(mode="bilinear", align_corners=True)), ("bicubic", dict(mode="bicubic", align_corners=False)), ("bicubic_ac", dict(mode="bicubic", align_corners=True)), ("area", dict(mode="area"))]:
    put(f"interp_{name}_up", F.interpolate(img, size=(7, 8), **kw))
    put(f"interp_{name}_down", F.interpolate(img, size=(3, 2), **kw))
put("interp_scale", F.interpolate(img, scale_factor=(2.0, 1.5), mode="bilinear", align_corners=False))
# pads: Rust lists (before, after) pairs starting from the LAST dimension, like torch.nn.functional.pad
for name, mode, kw in [("reflect", "reflect", {}), ("replicate", "replicate", {}), ("circular", "circular", {}), ("constant", "constant", dict(value=0.7))]:
    put(f"pad_{name}", F.pad(img, (2, 1, 1, 2), mode=mode, **kw))
cs = t((1, 6, 2, 2), 0.6)
put("channel_shuffle", F.channel_shuffle(cs, 3))

E, H = 8, 2
mha = torch.nn.MultiheadAttention(E, H, dropout=0.0, bias=True, batch_first=True).double()
with torch.no_grad():
    mha.in_proj_weight.copy_(torch.cat([t((8, 8), 10.0), t((8, 8), 11.0), t((8, 8), 12.0)], 0))
    mha.in_proj_bias.copy_(torch.cat([t((8,), 10.5), t((8,), 11.5), t((8,), 12.5)], 0))
    mha.out_proj.weight.copy_(t((8, 8), 13.0))
    mha.out_proj.bias.copy_(t((8,), 13.5))
qin, kin, vin = t((2, 5, 8), 20.0), t((2, 7, 8), 21.0), t((2, 7, 8), 22.0)
o, w = mha(qin, kin, vin, need_weights=True, average_attn_weights=True)
put("mha_out", o)
put("mha_w", w)
kpm = torch.tensor([[0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0]], dtype=torch.bool)
# Rust's second case builds the mask from the flat list [0,0,0,0,0,1,1, 0,0,0,0,1,0,0]
o2, w2 = mha(qin, kin, vin, key_padding_mask=kpm, need_weights=True, average_attn_weights=False)
put("mha_kpm_out", o2)
put("mha_kpm_w", w2)
sq = t((2, 5, 8), 23.0)
causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)
o3, _ = mha(sq, sq, sq, attn_mask=causal, need_weights=False)
put("mha_causal", o3)

worst, bad = 0.0, []
for k_, want in ref.items():
    if k_ not in rust:
        bad.append((k_, "missing", want)); continue
    err = abs(rust[k_] - want) / max(1.0, abs(want))
    tol = 5e-5 if k_.startswith("mha_") else 1e-6  # the attention module computes in float32
    worst = max(worst, err)
    if err > tol:
        bad.append((k_, rust[k_], want))
print(f"compared {len(ref)} values, worst relative error {worst:.2e}")
groups = {}
for k_, got, want in bad:
    groups.setdefault(k_.rsplit("_", 1)[0], []).append((k_, got, want))
for g, items in groups.items():
    k_, got, want = items[0]
    print(f"MISMATCH {g}: {len(items)} values differ, e.g. {k_}: rust={got} torch={want}")
print(f"{len(bad)} mismatches")
sys.exit(1 if bad else 0)
