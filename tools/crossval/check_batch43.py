"""Batch 43: real model tokenizers vs the `tokenizers` library. Needs target/real_tokenizers (fetch_real_tokenizers.py)."""
import json, os, re, sys, warnings
warnings.filterwarnings("ignore")
from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, "..", "..")
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
texts = json.load(open(f"{here}/tokenizer_fixtures/real_texts.json", encoding="utf-8"))
d = os.path.join(root, "target", "real_tokenizers")
only = sys.argv[2:] or None
bad = n = 0
per = {}
def cmp(name, key, field, want, have):
    global bad, n
    n += 1
    if want != have:
        bad += 1; per[name] = per.get(name, 0) + 1
        if per[name] <= 3:
            print("MISMATCH", key, field)
            if isinstance(want, list) and isinstance(have, list):
                i = next((k for k in range(min(len(want), len(have))) if want[k] != have[k]), min(len(want), len(have)))
                print("   first diff at", i, "want", want[max(0, i-2):i+4], "got", have[max(0, i-2):i+4], "lens", len(want), len(have))
            else: print("   want", str(want)[:200], "\n   got ", str(have)[:200])

for fn in sorted(os.listdir(d)):
    if not fn.endswith(".json"): continue
    name = fn[:-5]
    if only and name not in only: continue
    g = got.get(f"{name}.load")
    if g is not None:
        bad += 1; n += 1; print("LOAD FAILED", name, g["err"][:300]); continue
    tk = Tokenizer.from_file(os.path.join(d, fn))
    # the reference's skip_special_tokens ignores special tokens declared with normalized=true (e.g. Llama-2 <s>); skipping every
    # special token is the intended behaviour, so the expected text is built by removing the special ids explicitly
    special_ids = {i for i, a in tk.get_added_tokens_decoder().items() if a.special}
    for i, tx in enumerate(texts["texts"]):
        for sp in (1, 0):
            key = f"{name}.t{i}.s{sp}"; r = got.get(key)
            if r is None or "err" in r: bad += 1; n += 1; per[name] = per.get(name, 0) + 1; print("ERR", key, r); continue
            try: e = tk.encode(tx, add_special_tokens=bool(sp))
            except Exception as ex: print("REF ERR", key, ex); continue
            cmp(name, key, "ids", e.ids, r["ids"])
            cmp(name, key, "offsets", [list(o) for o in e.offsets], r["offsets"])
            cmp(name, key, "type_ids", e.type_ids, r["type_ids"]); cmp(name, key, "special", e.special_tokens_mask, r["special"])
            cmp(name, key, "dec", tk.decode(e.ids, skip_special_tokens=False), r["dec"]); cmp(name, key, "dec_skip", tk.decode([i for i in e.ids if i not in special_ids], skip_special_tokens=False), r["dec_skip"])
    for i, (a, b) in enumerate(texts["pairs"]):
        key = f"{name}.p{i}"; r = got.get(key)
        if r is None or "err" in r: bad += 1; n += 1; per[name] = per.get(name, 0) + 1; print("ERR", key, r); continue
        e = tk.encode(a, b, add_special_tokens=True)
        cmp(name, key, "ids", e.ids, r["ids"]); cmp(name, key, "type_ids", e.type_ids, r["type_ids"]); cmp(name, key, "special", e.special_tokens_mask, r["special"])
    print(f"{name}: {per.get(name, 0)} mismatches")
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
