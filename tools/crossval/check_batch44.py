"""Batch 44: tiktoken-format encodings vs the `tiktoken` package. Run with the venv that has tiktoken:
   target/tokvenv/Scripts/python.exe tools/crossval/check_batch44.py --fetch      (downloads the rank files)
   target/tokvenv/Scripts/python.exe tools/crossval/check_batch44.py <emit output>"""
import json, os, re, sys, urllib.request

here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, "..", "..")
names = ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]
if sys.argv[1] == "--fetch":
    os.makedirs(os.path.join(root, "target", "tiktoken"), exist_ok=True)
    for n in names:
        d = os.path.join(root, "target", "tiktoken", n + ".tiktoken")
        if not os.path.exists(d):
            open(d, "wb").write(urllib.request.urlopen(f"https://openaipublic.blob.core.windows.net/encodings/{n}.tiktoken", timeout=120).read()); print("fetched", n)
    sys.exit(0)
import tiktoken
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
load = lambda n: json.load(open(f"{here}/tokenizer_fixtures/{n}.json", encoding="utf-8"))
texts = load("real_texts")["texts"] + load("fuzz_texts")
bad = n = 0
for name in names:
    if f"{name}.missing" in got: print("missing rank file for", name); bad += 1; continue
    enc = tiktoken.get_encoding(name); mism = 0
    for i, tx in enumerate(texts):
        g = got.get(f"{name}.{i}")
        if g is None: bad += 1; n += 1; mism += 1; continue
        want_o = enc.encode_ordinary(tx); want_a = enc.encode(tx, allowed_special="all")
        for fld, want in (("ord", want_o), ("all", want_a)):
            n += 1
            if g[fld] != want:
                bad += 1; mism += 1
                if mism <= 3:
                    k = next((j for j in range(min(len(want), len(g[fld]))) if want[j] != g[fld][j]), min(len(want), len(g[fld])))
                    print("MISMATCH", name, i, fld, repr(tx[:60]), "first diff", k, "want", want[max(0, k-2):k+3], "got", g[fld][max(0, k-2):k+3])
    n += 1
    if got.get(f"{name}.n_vocab", {}).get("n") != enc.n_vocab: bad += 1; print("n_vocab", name, got.get(f"{name}.n_vocab"), enc.n_vocab)
    print(f"{name}: {mism} mismatches")
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
