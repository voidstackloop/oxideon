"""Tests for the pure-Rust tokenizer stack exposed to Python (Tokenizer, SentencePieceProcessor, TiktokenEncoding,
AutoTokenizer). Self-contained: tokenizers are built from inline JSON / trained on a tiny corpus."""
import json

import pytest

import oxideon as ox

pytestmark = pytest.mark.skipif(not hasattr(ox, "Tokenizer"), reason="built without the tokenizer bindings")

BERT = json.dumps({
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
                     for i, t in enumerate(["[UNK]", "[CLS]", "[SEP]", "[PAD]"])],
    "normalizer": {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True, "strip_accents": None, "lowercase": True},
    "pre_tokenizer": {"type": "BertPreTokenizer"},
    "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", 2], "cls": ["[CLS]", 1]},
    "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": True},
    "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
              "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "hello": 4, ",": 5, "world": 6, "un": 7, "##believ": 8, "##able": 9, "cafe": 10}},
})


def test_encode_decode_and_offsets():
    t = ox.Tokenizer.from_str(BERT)
    e = t.encode("Hello, wörld! Unbelievable café")
    assert e.tokens == ["[CLS]", "hello", ",", "world", "[UNK]", "un", "##believ", "##able", "cafe", "[SEP]"]
    assert e.special_tokens_mask[0] == 1 and e.special_tokens_mask[1] == 0
    assert e.offsets[1] == (0, 5)
    assert t.decode(e.ids) == "hello, world unbelievable cafe"
    assert t.token_to_id("world") == 6 and t.id_to_token(6) == "world"


def test_pairs_truncation_padding_batch():
    t = ox.Tokenizer.from_str(BERT)
    p = t.encode("hello", "world")
    assert p.type_ids == [0, 0, 0, 1, 1]
    t.enable_truncation(max_length=4)
    t.enable_padding(length=6, pad_id=3)
    a, b = t.encode_batch(["hello , world hello world", "hello"])
    assert len(a.ids) == len(b.ids) == 6
    assert b.attention_mask == [1, 1, 1, 0, 0, 0]
    assert len(a.overflowing) >= 1


def test_save_round_trip(tmp_path):
    t = ox.Tokenizer.from_str(BERT)
    t.add_special_tokens(["<extra>"])
    path = str(tmp_path / "tokenizer.json")
    t.save(path)
    back = ox.Tokenizer.from_file(path)
    assert back.encode("hello <extra> world").ids == t.encode("hello <extra> world").ids
    assert back.token_to_id("<extra>") == t.token_to_id("<extra>")


def test_train_bpe_and_word_level():
    corpus = ["the cat sat on the mat", "the dog sat on the log"] * 6
    t = ox.Tokenizer.untrained("BPE", "[UNK]")
    t.set_pre_tokenizer(json.dumps({"type": "Whitespace"}))
    t.train_bpe(corpus, vocab_size=60, special_tokens=["[UNK]"])
    assert t.token_to_id("[UNK]") == 0
    e = t.encode("the cat sat")
    assert "".join(e.tokens) == "thecatsat" and len(e.ids) <= 6
    w = ox.Tokenizer.untrained("WordLevel", "[UNK]")
    w.set_pre_tokenizer(json.dumps({"type": "Whitespace"}))
    w.train_word_level(corpus, vocab_size=10, special_tokens=["[UNK]"])
    assert w.encode("the zebra").tokens == ["the", "[UNK]"]


def test_tiktoken_with_a_tiny_rank_file(tmp_path):
    import base64
    ranks = {bytes([b]): b for b in range(256)}
    for i, tok in enumerate([b"ab", b"abc", b" a"]):
        ranks[tok] = 256 + i
    path = tmp_path / "tiny.tiktoken"
    path.write_text("\n".join(f"{base64.b64encode(k).decode()} {v}" for k, v in ranks.items()))
    enc = ox.TiktokenEncoding(str(path), ox.TIKTOKEN_R50K_PATTERN, [("<|endoftext|>", 1000)])
    assert enc.encode_ordinary("abc") == [257]
    assert enc.encode("a<|endoftext|>", allowed_special="all") == [97, 1000]
    with pytest.raises(ValueError):
        enc.encode("a<|endoftext|>")
    assert enc.decode([257, 1000]) == "abc<|endoftext|>"


def test_auto_tokenizer_and_chat_template(tmp_path):
    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "do_lower_case": True, "unk_token": "[UNK]", "pad_token": "[PAD]", "bos_token": "[CLS]", "eos_token": "[SEP]",
        "chat_template": "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
    }))
    t = ox.AutoTokenizer.from_pretrained(str(tmp_path))
    assert t.encode("Hello world") == [2, 4, 5, 3]
    assert t.pad_token_id == 0 and t.vocab_size == 6
    text = t.apply_chat_template([{"role": "user", "content": "hello"}], add_generation_prompt=True)
    assert text == "user: hello\nassistant:"
    ids = t.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=True)
    assert isinstance(ids, list) and 4 in ids
