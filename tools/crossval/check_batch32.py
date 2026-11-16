"""Batch 32: full encoder-decoder Transformer vs torch.nn.Transformer (float32), outputs and gradients."""
import math, re, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

arrs = {}; params = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.strip())
    m = re.match(r"^param_(\w+) (\d+) (\S+) \[([\d, ]+)\]$", line)
    if m:
        params.setdefault(m.group(1), []).append((int(m.group(2)), m.group(3), [int(v) for v in m.group(4).split(",")]))
        continue
    k, _, rest = line.partition(" ")
    try: arrs[k] = np.array([float(x) for x in rest.split()])
    except ValueError: pass

def genv(n, salt): return [math.sin(0.37 * i + salt) * 0.8 + 0.1 * math.cos(0.11 * i * salt) for i in range(n)]
def t(shape, salt): return torch.tensor(genv(int(np.prod(shape)), salt), dtype=torch.float32).reshape(tuple(shape))
bad = n = 0; worst = 0.0
def check(name, tensor, tol=2e-4):
    global bad, n, worst
    n += 1
    if name not in arrs: bad += 1; print("MISSING", name); return
    want = tensor.detach().double().flatten().numpy(); got = arrs[name]
    if got.shape != want.shape: bad += 1; print("SHAPE", name, got.shape, want.shape); return
    err = np.max(np.abs(got - want)) / max(1.0, np.abs(want).max()); worst = max(worst, err)
    if err > tol: bad += 1; print("MISMATCH", name, f"err {err:.2e}", got[:4], want[:4])

E = 8
def locate(model, name):
    """(torch tensor, slice) holding the parameter the Rust side calls `name`."""
    name = name.replace("cross_attn", "multihead_attn")
    m = re.match(r"^(.*)\.([qkv])_proj\.(weight|bias)$", name)
    sd = dict(model.named_parameters())
    if m:
        i = "qkv".index(m.group(2))
        return sd[f"{m.group(1)}.in_proj_{m.group(3)}"], slice(E * i, E * (i + 1))
    return sd[name], slice(None)

for tag, act, norm_first in [("post_relu", "relu", False), ("pre_gelu", "gelu", True)]:
    model = torch.nn.Transformer(d_model=E, nhead=2, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=16, dropout=0.0,
                                 activation=act, norm_first=norm_first, batch_first=True, layer_norm_eps=1e-5)
    plist = sorted(params[tag])
    with torch.no_grad():
        for k, name, shape in plist:
            p, sl = locate(model, name)
            p[sl].copy_(t(shape, 1.0 + 0.61 * k) * 0.5)
    src, tgt = t((2, 6, E), 50.0).requires_grad_(True), t((2, 5, E), 51.0).requires_grad_(True)
    out = model(src, tgt)
    check(f"{tag}_plain_out", out)
    (out * t(list(out.shape), 99.0)).sum().backward()
    check(f"{tag}_plain_dsrc", src.grad); check(f"{tag}_plain_dtgt", tgt.grad)
    for k in (0, len(plist) // 2, len(plist) - 1):
        p, sl = locate(model, plist[k][1])
        check(f"{tag}_plain_dparam{k}", p.grad[sl])
    kpm = torch.tensor([[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.bool)
    src2, tgt2 = t((2, 6, E), 52.0).requires_grad_(True), t((2, 5, E), 53.0).requires_grad_(True)
    causal = torch.nn.Transformer.generate_square_subsequent_mask(5)
    out = model(src2, tgt2, tgt_mask=causal, src_key_padding_mask=kpm, memory_key_padding_mask=kpm, tgt_is_causal=True)
    check(f"{tag}_masked_out", out)
    (out * t(list(out.shape), 99.0)).sum().backward()
    check(f"{tag}_masked_dsrc", src2.grad); check(f"{tag}_masked_dtgt", tgt2.grad)
print(f"{n} tensors compared, worst scaled error {worst:.2e}, {bad} mismatches")
sys.exit(1 if bad else 0)
