#!/usr/bin/env python3
"""Copies llama.cpp's tiny sparse-attention test models with `attention.indexer.top_k` lowered, so the lightning indexer
really selects (a 100-token prompt then has far more tokens / blocks than the budget):

  <arch>-topk.gguf for deepseek32, glm-dsa, dots3note, hy_v4 (top_k = 4 tokens)

and, named on the command line, the pooled / block indexers (the tests use these values):
  glm5-next-moe=8     (kpool 4: two pools + tail)
  minimax-m3-moe=3    (blocks of 4 tokens, one forced local block)   OXIDEON_TENSOR_GAIN=attn_v.scale=300,attn_q.scale=60,attn_k.scale=60,attn_output.weight=3000
  qwen4exp-moe=8      (kpool 4: two pools + tail)                    OXIDEON_TENSOR_GAIN=attn_output.weight=3000
The tiny models' attention barely reaches the logits, so a wrong selection would pass unnoticed; the gains on the
attention tensors (applied to the copy, so llama.cpp's reference sees them too) make the selection decisive.

usage: topk_variants.py <test-models-dir> <out-dir> [path-to-llama.cpp/gguf-py] [name=top_k ...]
Make the reference logits with `llama_logits <variant>.gguf <tokens>` (tokens 0..99 as i % 127 + 1) and run the tests
with OXIDEON_LLAMA_EXTRA_MODELS=<out-dir>.
"""
import os
import sys

import numpy as np

models, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/projects/llama.cpp/gguf-py"))
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402

os.makedirs(out_dir, exist_ok=True)
# OXIDEON_TENSOR_GAIN="attn_v.scale=200,attn_output.weight=3000": multiplies every tensor whose name ends so.
gains = {s: float(g) for s, g in (p.split("=") for p in os.environ.get("OXIDEON_TENSOR_GAIN", "").split(",") if p)}
wanted = {"deepseek32-moe": 4, "glm-dsa-moe": 4, "dots3note-moe": 4, "hy_v4-moe": 4}
for arg in sys.argv[4:]:
    name, k = arg.split("=")
    wanted[name] = int(k)
for name, top_k in wanted.items():
    reader = GGUFReader(os.path.join(models, f"{name}.gguf"))
    arch = reader.fields["general.architecture"].contents()
    w = GGUFWriter(os.path.join(out_dir, f"{name.replace('-moe', '')}-topk.gguf"), arch)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = top_k if key == f"{arch}.attention.indexer.top_k" else field.contents()
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue  # the writer rejects empty arrays
        vt = field.types[0]
        w.add_key_value(key, value, vt, sub_type=field.types[-1] if vt == GGUFValueType.ARRAY else None)
    for t in reader.tensors:
        data = np.array(t.data)
        # The tiny models' attention barely reaches the logits (tiny projection scales, near-constant values); gains on
        # chosen tensors make the selection matter.
        for suffix, g in gains.items():
            if t.name.endswith(suffix) and data.dtype in (np.float32, np.float16):
                data = (data.astype(np.float32) * g).astype(data.dtype)
        w.add_tensor(t.name, data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", name, "top_k", top_k)
