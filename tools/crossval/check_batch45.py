"""Batch 45: SentencePiece .model files vs the `sentencepiece` package. Run with the venv that has sentencepiece:
   target/tokvenv/Scripts/python.exe tools/crossval/check_batch45.py --fetch
   target/tokvenv/Scripts/python.exe tools/crossval/check_batch45.py <emit output> [model names]"""
import json, os, re, sys, urllib.request

here = os.path.dirname(os.path.abspath(__file__)); root = os.path.join(here, "..", "..")
models = {  # local name -> (repo, file)
    "t5_small": ("google-t5/t5-small", "spiece.model"), "xlnet_base": ("xlnet/xlnet-base-cased", "spiece.model"), "albert_base": ("albert/albert-base-v2", "spiece.model"),
    "mt5_small": ("google/mt5-small", "spiece.model"), "xlmr_base": ("FacebookAI/xlm-roberta-base", "sentencepiece.bpe.model"), "llama2": ("NousResearch/Llama-2-7b-hf", "tokenizer.model"),
    "mistral_7b": ("mistralai/Mistral-7B-v0.1", "tokenizer.model"), "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tokenizer.model"), "deberta_v3": ("microsoft/deberta-v3-base", "spm.model"),
    "pegasus": ("google/pegasus-large", "spiece.model"), "opus_en_de": ("Helsinki-NLP/opus-mt-en-de", "source.spm"), "camembert": ("almanach/camembert-base", "sentencepiece.bpe.model"),
    "bart_pho": ("vinai/bartpho-syllable", "sentencepiece.bpe.model"), "marian_en_fr": ("Helsinki-NLP/opus-mt-en-fr", "source.spm"), "reformer": ("google/reformer-crime-and-punishment", "spiece.model"),
}
if sys.argv[1] == "--fetch":
    os.makedirs(os.path.join(root, "target", "spm"), exist_ok=True)
    for k, (repo, f) in models.items():
        d = os.path.join(root, "target", "spm", k + ".model")
        if os.path.exists(d): continue
        try: open(d, "wb").write(urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{f}", timeout=120).read()); print("fetched", k)
        except Exception as e: print("skip", k, e)
    sys.exit(0)
import sentencepiece as spm
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
load = lambda n: json.load(open(f"{here}/tokenizer_fixtures/{n}.json", encoding="utf-8"))
texts = load("real_texts")["texts"] + load("fuzz_texts")
only = sys.argv[2:] or None
bad = n = 0
for name in models:
    path = os.path.join(root, "target", "spm", name + ".model")
    if not os.path.exists(path) or (only and name not in only): continue
    if f"{name}.load" in got: bad += 1; n += 1; print("LOAD FAILED", name, got[f"{name}.load"]["err"][:200]); continue
    sp = spm.SentencePieceProcessor(model_file=path); mism = 0; shown = 0
    if got.get(f"{name}.info", {}).get("n") != sp.get_piece_size(): bad += 1; print("vocab size", name)
    for i, tx in enumerate(texts):
        g = got.get(f"{name}.{i}")
        if g is None or "err" in g: bad += 1; n += 1; mism += 1; continue
        want = sp.encode(tx); n += 1
        if g["ids"] != want:
            bad += 1; mism += 1
            if shown < 3:
                shown += 1; k = next((j for j in range(min(len(want), len(g["ids"]))) if want[j] != g["ids"][j]), min(len(want), len(g["ids"])))
                print("MISMATCH", name, i, repr(tx[:70]), "first diff", k, "want", want[max(0, k-2):k+4], [sp.id_to_piece(x) for x in want[max(0, k-2):k+4]], "got", g["ids"][max(0, k-2):k+4])
        n += 1
        wd = sp.decode(want)
        if g["dec"] != wd:
            bad += 1; mism += 1
            if shown < 3: shown += 1; print("DEC MISMATCH", name, i, repr(tx[:60]), "\n   want", repr(wd[:90]), "\n   got ", repr(g["dec"][:90]))
    print(f"{name}: {mism} mismatches")
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
