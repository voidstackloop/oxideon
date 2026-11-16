"""Batch 46: BPE / WordPiece / WordLevel training vs the `tokenizers` trainers.
BPE and WordLevel are deterministic in the reference and compared exactly (vocabulary order and merges); the reference's
WordPiece trainer (and BPE with an end-of-word suffix) visit words in hash order, so those are compared as sets."""
import json, os, re, sys, warnings
warnings.filterwarnings("ignore")
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

here = os.path.dirname(os.path.abspath(__file__))
got = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = re.sub(r"^test \S+ \.\.\. ", "", line.rstrip("\n"))
    k, _, rest = line.partition(" ")
    try: got[k] = json.loads(rest)
    except ValueError: pass
corpus = json.load(open(f"{here}/tokenizer_fixtures/train_corpus.json", encoding="utf-8"))
bad = n = 0
def merges_of(m): return [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in m]
def cmp(name, exact, t):
    global bad, n
    ref = json.loads(t.to_str())["model"]; g = got.get(name)
    if g is None: bad += 1; n += 1; print("MISSING", name); return
    rv = [k for k, _ in sorted(ref["vocab"].items(), key=lambda kv: kv[1])]; gv = [k for k, _ in sorted(g["vocab"].items(), key=lambda kv: kv[1])]
    n += 1
    if exact:
        if rv != gv: bad += 1; k = next((i for i in range(min(len(rv), len(gv))) if rv[i] != gv[i]), min(len(rv), len(gv))); print("VOCAB MISMATCH", name, "first diff", k, rv[max(0,k-2):k+3], gv[max(0,k-2):k+3], len(rv), len(gv))
        if "merges" in ref:
            n += 1; rm, gm = merges_of(ref["merges"]), merges_of(g["merges"])
            if rm != gm: bad += 1; k = next((i for i in range(min(len(rm), len(gm))) if rm[i] != gm[i]), min(len(rm), len(gm))); print("MERGES MISMATCH", name, "first diff", k, rm[max(0,k-1):k+3], gm[max(0,k-1):k+3], len(rm), len(gm))
    else:
        common = len(set(rv) & set(gv)); frac = common / max(len(rv), len(gv))
        print(f"{name}: vocab overlap {common}/{max(len(rv), len(gv))} ({frac:.1%})")
        if frac < 0.9: bad += 1

def bl():
    return pre_tokenizers.ByteLevel(add_prefix_space=False)
t = Tokenizer(models.BPE()); t.pre_tokenizer = bl(); t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=500, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())); cmp("bpe_bytelevel", True, t)
t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace(); t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, min_frequency=2, special_tokens=["[UNK]"])); cmp("bpe_ws", True, t)
t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace(); t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"], end_of_word_suffix="</w>")); cmp("bpe_suffix", False, t)
t = Tokenizer(models.BPE(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace(); t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=250, special_tokens=["[UNK]"], limit_alphabet=30)); cmp("bpe_limit", True, t)
t = Tokenizer(models.WordLevel(unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace(); t.train_from_iterator(corpus, trainers.WordLevelTrainer(vocab_size=80, min_frequency=2, special_tokens=["[UNK]", "[PAD]"])); cmp("wordlevel", True, t)
t = Tokenizer(models.BPE(unk_token="[UNK]")); t.normalizer = normalizers.BertNormalizer(lowercase=True); t.pre_tokenizer = pre_tokenizers.BertPreTokenizer(); t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"])); cmp("bpe_bert", True, t)
t = Tokenizer(models.WordPiece(unk_token="[UNK]")); t.normalizer = normalizers.BertNormalizer(lowercase=True); t.pre_tokenizer = pre_tokenizers.BertPreTokenizer(); t.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=300, special_tokens=["[UNK]", "[CLS]", "[SEP]"])); cmp("wordpiece", False, t)
for k in ("bpe_bytelevel", "wordlevel"):  # specials registered as added tokens, in vocabulary order
    n += 1
    a = got.get(k + ".added")
    if not a or not all(x["special"] and not x["normalized"] for x in a): bad += 1; print("ADDED TOKENS", k, a)
print(f"{n} comparisons, {bad} mismatches"); sys.exit(1 if bad else 0)
