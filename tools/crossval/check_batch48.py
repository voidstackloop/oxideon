"""Batch 48: PyTorch checkpoints read by the pure-Rust loader vs torch.load. Run make_pt_files.py first."""
import json, os, re, sys
import numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__)); d = os.path.join(here, "..", "..", "target", "pt_files")
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
def flat(prefix, o, out):
    join = lambda k: f"{prefix}.{k}" if prefix else str(k)
    if isinstance(o, torch.Tensor): out.append((prefix, o.detach()))
    elif isinstance(o, dict):
        for k, v in o.items(): flat(join(k), v, out)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o): flat(join(i), v, out)
def fnv(a):
    h = 0xcbf29ce484222325
    for b in a.tobytes(): h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h
DT = {torch.float32: "F32", torch.float64: "F64", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32", torch.uint8: "U8", torch.bool: "Bool"}
bad = n = 0
for fn in sorted(os.listdir(d)):
    name = fn[:-3]; items = []
    flat("", torch.load(os.path.join(d, fn), weights_only=True), items)
    for k, t in items:
        n += 1; key = f"{name}::{k}"; g = got.get(key)
        a = np.ascontiguousarray(t.to(torch.float64).numpy())
        want = {"shape": list(t.shape), "dtype": DT[t.dtype], "hash": fnv(a)}
        if g != want: bad += 1; print("MISMATCH", key, "want", want, "got", g)
    extra = [k for k in got if k.startswith(name + "::") and k.split("::", 1)[1] not in {k2 for k2, _ in items}]
    if extra: bad += 1; print("EXTRA tensors from the Rust loader", name, extra)
print(f"{n} tensors compared, {bad} mismatches"); sys.exit(1 if bad else 0)
