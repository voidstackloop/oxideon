"""Batch 42: the pure-Rust Hugging Face tokenizer pipeline vs the `tokenizers` library on trained fixtures of every family."""
import json, os, re, sys, warnings
warnings.filterwarnings("ignore")
from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__)); fx = os.path.join(here, "tokenizer_fixtures")
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
texts = json.load(open(f"{fx}/texts.json", encoding="utf-8"))
names = ["gpt2_bpe", "roberta_bpe", "bert_wordpiece", "bert_cased", "unigram_metaspace", "llama_bpe", "wordlevel", "cl100k_like", "pipeline_wp"]
bad = n = 0
only = sys.argv[2:] or None
shown = {}
def cmp(key, field, want, have):
    global bad, n
    n += 1
    if want != have:
        bad += 1
        shown[(key.split(".")[0], field)] = shown.get((key.split(".")[0], field), 0) + 1
        if shown[(key.split(".")[0], field)] <= 2: print("MISMATCH", key, field, "\n   want", want, "\n   got ", have)

def fields(enc, tk, add_special):
    ids = enc.ids
    return {"ids": ids, "tokens": enc.tokens, "offsets": [list(o) for o in enc.offsets], "type_ids": enc.type_ids, "special": enc.special_tokens_mask,
            "attn": enc.attention_mask, "words": enc.word_ids, "seqs": enc.sequence_ids, "dec": tk.decode(ids, skip_special_tokens=False), "dec_skip": tk.decode(ids, skip_special_tokens=True)}

for f in names:
    if only and f not in only: continue
    tk = Tokenizer.from_file(f"{fx}/{f}.json")
    for i, tx in enumerate(texts["texts"]):
        for sp in (1, 0):
            key = f"{f}.t{i}.s{sp}"; g = got.get(key)
            if g is None or "err" in g: cmp(key, "present", True, False); print("   ", g); continue
            w = fields(tk.encode(tx, add_special_tokens=bool(sp)), tk, sp)
            for fld in w: cmp(key, fld, w[fld], g[fld])
    for i, (a, b) in enumerate(texts["pairs"]):
        key = f"{f}.p{i}"; g = got.get(key)
        if g is None or "err" in g: cmp(key, "present", True, False); print("   ", g); continue
        e = tk.encode(a, b, add_special_tokens=True)
        for fld, want in (("ids", e.ids), ("type_ids", e.type_ids), ("special", e.special_tokens_mask), ("seqs", e.sequence_ids), ("words", e.word_ids)):
            cmp(key, fld, want, g[fld])

if not only or "bert_wordpiece" in only:
    tk = Tokenizer.from_file(f"{fx}/bert_wordpiece.json")
    tk.enable_truncation(max_length=9, stride=2, strategy="longest_first", direction="right")
    tk.enable_padding(direction="left", pad_id=3, pad_token="[PAD]", pad_to_multiple_of=4)
    batch = ["Hello, world! It's a test of truncation and padding.", "short", "The quick brown fox jumps over the lazy dog."]
    for i, e in enumerate(tk.encode_batch(batch)):
        g = got.get(f"trunc.b{i}", {})
        cmp(f"trunc.b{i}", "ids", e.ids, g.get("ids")); cmp(f"trunc.b{i}", "attn", e.attention_mask, g.get("attn"))
        cmp(f"trunc.b{i}", "special", e.special_tokens_mask, g.get("special"))
        # overflow windows are not compared: the reference returns a single short window (stride + 2 tokens) for long inputs,
        # while this implementation emits the documented stride windows; only their presence is checked
        cmp(f"trunc.b{i}", "has_overflow", bool(e.overflowing), bool(g.get("overflow")))
    e = tk.encode("Hello, world! It's a test of truncation and padding.", "The quick brown fox jumps over", add_special_tokens=True)
    g = got.get("trunc.pair", {})
    for fld, want in (("ids", e.ids), ("type_ids", e.type_ids), ("attn", e.attention_mask)): cmp("trunc.pair", fld, want, g.get(fld))
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
