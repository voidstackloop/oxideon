#!/usr/bin/env python3
"""Builds the DeepSeek-V4 variants the long-prompt parity tests use from llama.cpp's tiny `deepseek4-moe.gguf`:

  deepseek4-long   the model itself (a 200-token prompt reaches the ratio-128 layer's first compressed block)
  deepseek4-topk   indexer.top_k = 3, so the compressed-block indexer really selects among ~50 blocks
  deepseek4-hash   layer 0 routes by token id (a random `ffn_gate_tid2eid` table) and every layer uses one expert
  deepseek4-yarn   YaRN rope scaling, factor 4

usage: deepseek4_variants.py <deepseek4-moe.gguf> <out-dir> [path-to-llama.cpp/gguf-py]
Then make the reference logits with `llama_logits <variant>.gguf <tokens>` where the tokens are 0..199 mapped to i % 127 + 1,
and run the tests with OXIDEON_LLAMA_EXTRA_MODELS=<out-dir>.
"""
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/projects/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402

src, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
reader = GGUFReader(src)
arch = reader.fields["general.architecture"].contents()
vocab = int(reader.fields[f"{arch}.vocab_size"].contents())


def write(name, overrides=None, drop=(), extra=()):
    w = GGUFWriter(os.path.join(out_dir, f"{name}.gguf"), arch)
    overrides = {f"{arch}.{k}": v for k, v in (overrides or {}).items()}
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = overrides.get(key, field.contents())
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue  # the writer rejects empty arrays
        vt = field.types[0]
        w.add_key_value(key, value, vt, sub_type=field.types[-1] if vt == GGUFValueType.ARRAY else None)
    for t in reader.tensors:
        if t.name not in drop:
            w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
    for tname, data in extra:
        w.add_tensor(tname, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


rng = np.random.default_rng(7)
write("deepseek4-long")
write("deepseek4-topk", {"attention.indexer.top_k": 3})
write(
    "deepseek4-hash",
    {"hash_layer_count": 1, "expert_used_count": 1},
    drop={"blk.0.exp_probs_b.bias", "blk.0.exp_probs_b_vl.bias"},
    extra=[("blk.0.ffn_gate_tid2eid.weight", rng.integers(0, 2, size=(vocab, 1)).astype(np.int32))],
)
write("deepseek4-yarn", {"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0})
print("wrote 4 variants to", out_dir)
