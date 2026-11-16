"""Writes tokenizer_fixtures/real_texts.json: a diverse text set (languages, scripts, code, whitespace, emoji, specials)."""
import json, os
texts = [
    "Hello, world! It's a test.",
    "The quick brown fox jumps over the lazy dog. It's 5 o'clock; don't panic — we've got 3.14159 reasons.",
    "naïve café résumé Zürich — déjà vu, façade, crème brûlée, São Paulo, Ångström, İstanbul, ß ẞ",
    "日本語のテキストと中文文本 mixed with English 한국어 텍스트 and 123456 numbers.",
    "Привет, мир! Это тест токенизатора. Ελληνικά κείμενο. مرحبا بالعالم. שלום עולם. नमस्ते दुनिया। สวัสดีชาวโลก",
    "def foo(x, y):\n    return x + y  # add numbers\n\nclass A:\n\tdef __init__(self): pass\n\nprint(foo(1, 2))",
    "fn main() { let v: Vec<u32> = (0..10).map(|x| x * 2).collect(); println!(\"{:?}\", v); }",
    "Emoji 😀 👍🏽 👨‍👩‍👧‍👦 🇹🇷 and symbols: $100, 50%, a+b=c, #hashtag @user http://example.com/path?q=1&r=2 user@example.com",
    "   leading spaces, trailing spaces   \n\ttabs and\nnewlines\n\n\n  indented  line  ",
    "ALL CAPS WORDS and MiXeD cAsE and ﬁnance ① ㎏ Å ǅ ŉ",
    "",
    " ",
    "\n",
    "a",
    "don't stop; won't stop. I'll go, they've gone, we'd see, she's here, I'm there, you're ok, ISN'T, CAN'T",
    "<s> </s> <unk> <pad> <mask> [CLS] [SEP] [MASK] [PAD] <|endoftext|> <|im_start|>user\nhi<|im_end|> <|begin_of_text|>",
    "1234567890 and 3.14159, 0xFF, 1e-5, 1,000,000, 2024-11-13T10:20:30Z, 555-1234",
    "x" * 100 + " " + "é" * 50 + " " + "日" * 30,
    "Tab\tseparated\tvalues\there and  double  spaces and trailing tab\t",
    "Ünïcödé combining: e\u0301 a\u0300 o\u0308 n\u0303 and zero\u200bwidth and soft\u00adhyphen and nbsp\u00a0space",
    "The quick brown fox. " * 20,
    "\u0000 control \u0007 chars \u001b[0m and \ufffd replacement and \ud7ff boundary",
    "Mr. Smith's co-op costs $4.99/lb (approx.) — isn't it \"fun\"? 'Quoted' text; semi-colon: colon, comma.",
    "SELECT * FROM users WHERE id = 42 AND name LIKE '%john%' ORDER BY created_at DESC LIMIT 10;",
    "{\"key\": [1, 2, {\"nested\": null}], \"s\": \"a\\nb\", \"t\": true}",
    "Ceci n'est pas une pipe. Das ist ein Test mit Umlauten: ä ö ü Ä Ö Ü ß. Esto es una prueba: ¿cómo estás? ¡Hola!",
]
pairs = [("Hello, world!", "How are you?"), ("a b c", "d e f g"), ("naïve café", ""), ("", "only second"), ("What is the capital of France?", "Paris is the capital and most populous city of France.")]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer_fixtures", "real_texts.json")
json.dump({"texts": texts, "pairs": pairs}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", len(texts), "texts")
