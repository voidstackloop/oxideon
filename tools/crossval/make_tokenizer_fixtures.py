"""Trains small Hugging Face tokenizers of every family and saves them as tokenizer.json fixtures
(tools/crossval/tokenizer_fixtures/*.json), plus the shared test texts. Used by crossval batch 42."""
import json, os
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, decoders, trainers, Regex

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_fixtures")
os.makedirs(out, exist_ok=True)

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, world! It's a beautiful day, isn't it? I'll say 42 times: don't stop.",
    "Tokenization splits text into subword units, e.g. un-believ-able and token ##ization.",
    "naïve café résumé Zürich São Paulo — déjà vu, façade, crème brûlée.",
    "日本語のテキストと中文文本 mixed with English words and 123456 numbers.",
    "def foo(x, y):\n    return x + y  # add numbers\n\nprint(foo(1, 2))",
    "Emoji test 😀 👍🏽 and symbols: $100, 50%, a+b=c, #hashtag @user http://example.com/path?q=1",
    "   leading spaces, trailing spaces   \n\ttabs and\nnewlines\n\n",
    "ALL CAPS WORDS and MiXeD cAsE and ﬁnance ① ㎏ Å",
] * 8

specials = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

# 1. byte-level BPE (GPT-2 style)
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
t.decoder = decoders.ByteLevel()
t.post_processor = processors.ByteLevel(trim_offsets=True)
t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=500, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
t.save(f"{out}/gpt2_bpe.json")

# 2. RoBERTa style (add_prefix_space + RobertaProcessing)
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
t.decoder = decoders.ByteLevel()
t.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 1), trim_offsets=True, add_prefix_space=True)
t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=500, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
t.save(f"{out}/roberta_bpe.json")

# 3. BERT WordPiece (uncased)
t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
t.normalizer = normalizers.BertNormalizer(lowercase=True)
t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
t.decoder = decoders.WordPiece(prefix="##")
t.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=400, special_tokens=specials))
t.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                 special_tokens=[("[CLS]", t.token_to_id("[CLS]")), ("[SEP]", t.token_to_id("[SEP]"))])
t.save(f"{out}/bert_wordpiece.json")

# 4. BERT cased, no accent stripping
t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
t.normalizer = normalizers.BertNormalizer(lowercase=False, strip_accents=False)
t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
t.decoder = decoders.WordPiece(prefix="##")
t.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=400, special_tokens=specials))
t.post_processor = processors.BertProcessing(("[SEP]", t.token_to_id("[SEP]")), ("[CLS]", t.token_to_id("[CLS]")))
t.save(f"{out}/bert_cased.json")

# 5. Unigram with Metaspace (T5 / XLNet / ALBERT-like)
t = Tokenizer(models.Unigram())
t.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " ")])
t.pre_tokenizer = pre_tokenizers.Metaspace()
t.decoder = decoders.Metaspace()
t.train_from_iterator(corpus, trainers.UnigramTrainer(vocab_size=300, special_tokens=["<pad>", "</s>", "<unk>"], unk_token="<unk>"))
t.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>", special_tokens=[("</s>", t.token_to_id("</s>"))])
t.save(f"{out}/unigram_metaspace.json")

# 6. Llama-style BPE with Metaspace, byte fallback
t = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True, fuse_unk=True))
t.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
t.pre_tokenizer = None
t.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)])
tr = trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<s>", "</s>"], initial_alphabet=[f"<0x{b:02X}>" for b in range(256)])
t.train_from_iterator(corpus, tr)
t.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", t.token_to_id("<s>"))])
t.save(f"{out}/llama_bpe.json")

# 7. WordLevel with Whitespace
t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
t.normalizer = normalizers.Lowercase()
t.pre_tokenizer = pre_tokenizers.Whitespace()
t.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"]))
t.save(f"{out}/wordlevel.json")

# 8. GPT-4 style: Split with regex + ByteLevel(use_regex=False)
cl100k = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(cl100k), "isolated", False), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
t.decoder = decoders.ByteLevel()
t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=500, special_tokens=["<|endoftext|>", "<|fim_prefix|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
t.save(f"{out}/cl100k_like.json")

# 9. Digits + Punctuation + whitespace-split pipeline, NFD + StripAccents + Lowercase normalizer
t = Tokenizer(models.WordPiece(unk_token="[UNK]", continuing_subword_prefix="##"))
t.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(), normalizers.Strip()])
t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(), pre_tokenizers.Digits(individual_digits=True)])
t.decoder = decoders.WordPiece()
t.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=300, special_tokens=specials))
t.save(f"{out}/pipeline_wp.json")

texts = [
    "Hello, world! It's a test.",
    "The quick brown fox jumps over the lazy dog.",
    "naïve café résumé Zürich — déjà vu",
    "日本語のテキストと中文文本 mixed 123456",
    "def foo(x, y):\n    return x + y  # add\n\nprint(foo(1, 2))",
    "Emoji 😀 👍🏽 $100, 50%, a+b=c #tag http://x.com/p?q=1",
    "   leading and trailing   \n\t",
    "ALL CAPS and MiXeD cAsE ﬁnance ① ㎏ Å",
    "",
    "a",
    "don't stop; won't stop. I'll go, they've gone, we'd see, she's here, I'm there, you're ok",
    "<|endoftext|> special [CLS] token [SEP] and </s> text <s>",
    "1234567890 and 3.14159, 0xFF, 1e-5",
    "x" * 300,
]
pairs = [("Hello, world!", "How are you?"), ("a b c", "d e f g"), ("naïve", "")]
json.dump({"texts": texts, "pairs": pairs}, open(f"{out}/texts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", sorted(os.listdir(out)))
