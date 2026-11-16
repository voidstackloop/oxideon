"""Writes tokenizer_fixtures/fuzz_texts.json: deterministic pseudo-random strings mixing scripts, whitespace, digits,
punctuation, contractions, emoji, combining marks and special-token look-alikes (for tiktoken / SentencePiece parity)."""
import json, os, random

rnd = random.Random(20261113)
words = ["the", "The", "THE", "hello", "World", "don't", "I'll", "we've", "DON'T", "naïve", "café", "Zürich", "日本語", "中文", "한국어", "Привет", "мир", "مرحبا", "שלום", "नमस्ते",
         "x", "ab", "abc", "foo_bar", "fooBar", "FooBarBaz", "HTTPServer", "é", "😀", "👍🏽", "🇹🇷", "<|endoftext|>", "<|fim_prefix|>", "<s>", "</s>", "[CLS]"]
puncts = [",", ".", "!", "?", ";", ":", "-", "--", "—", "...", "(", ")", "[", "]", "{", "}", "'", '"', "/", "\\", "@", "#", "$", "%", "^", "&", "*", "_", "=", "+", "~", "`", "|", "<", ">"]
spaces = [" ", " ", " ", "  ", "   ", "\n", "\n\n", "\r\n", "\t", " \n", "\n ", " ", "　", "​"]
digits = ["0", "1", "42", "123", "1234", "12345678", "3.14", "1,000", "007"]
out = []
for _ in range(400):
    parts = []
    for _ in range(rnd.randint(1, 30)):
        r = rnd.random()
        parts.append(rnd.choice(words) if r < 0.4 else rnd.choice(puncts) if r < 0.55 else rnd.choice(spaces) if r < 0.8 else rnd.choice(digits) if r < 0.9 else
                     "".join(chr(rnd.choice([rnd.randint(0x20, 0x7e), rnd.randint(0xa0, 0x24f), rnd.randint(0x400, 0x4ff), rnd.randint(0x4e00, 0x4fff), rnd.randint(0x300, 0x36f), rnd.randint(0x1f300, 0x1f5ff)])) for _ in range(rnd.randint(1, 5))))
    out.append("".join(parts))
out += ["", " ", "  ", "\n", "a" * 5000, " " * 300 + "x", "\n" * 50, "ab" * 2000, "日" * 1500, "x y " * 800]
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_fixtures", "fuzz_texts.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(len(out), "strings")
