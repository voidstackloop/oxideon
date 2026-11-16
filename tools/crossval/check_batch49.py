"""Batch 49: ONNX models run by the pure-Rust importer vs onnxruntime. Run make_onnx_models.py first (venv with onnx + onnxruntime)."""
import json, re, sys
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
bad = n = 0
for k, v in sorted(got.items()):
    n += 1
    if "err" in v: bad += 1; print("ERROR", k, v["err"][:300]); continue
    ok = v["same_shape"] and v["maxdiff"] <= 2e-4
    if not ok: bad += 1
    print("ok  " if ok else "FAIL", k, "maxdiff", v["maxdiff"], v["got"], v["want"])
print(f"{n} outputs compared, {bad} mismatches"); sys.exit(1 if bad else 0)
