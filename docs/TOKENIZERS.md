# Tokenizers

`oxideon_core::tokenizer` is a pure-Rust (no extra dependency) implementation of the tokenizer formats people actually
ship. Everything below is checked against the reference implementation in `tools/crossval/` (see its README, batches 41-47).

| You have | Load it with | Reference it matches |
|---|---|---|
| `tokenizer.json` (GPT-2, Llama, Mistral, Qwen, Phi, BERT, RoBERTa, T5, XLM-R, ...) | `hf::Tokenizer::from_file` | Hugging Face `tokenizers` (26 real models, 8 800 comparisons) |
| `vocab.json` + `merges.txt` (GPT-2 family) | `hf::Tokenizer::from_vocab_and_merges` | the matching `tokenizer.json` |
| `vocab.txt` (BERT family) | `hf::Tokenizer::from_vocab_txt` | the matching `tokenizer.json` |
| SentencePiece `.model` (Unigram or BPE) | `sentencepiece::SentencePiece::from_file` | the `sentencepiece` package (15 models, 13 000 comparisons) |
| `.tiktoken` rank file (`r50k_base`, `p50k_base`, `cl100k_base`, `o200k_base`, custom) | `tiktoken::Tiktoken::preset` / `from_tiktoken_file` | the `tiktoken` package (3 500 comparisons) |
| a model directory | `auto::AutoTokenizer::from_dir` | picks the format, reads `tokenizer_config.json` |
| a chat template | `AutoTokenizer::apply_chat_template` / `chat_template::ChatTemplate` | `transformers` + `jinja2` (28 real templates) |

```rust
use oxideon_core::tokenizer::auto::AutoTokenizer;
let tok = AutoTokenizer::from_dir("path/to/model")?;
let ids = tok.encode("Hello, world!", true)?;
let text = tok.decode(&ids, true)?;
```

```python
import oxideon as ox
tok = ox.AutoTokenizer.from_pretrained("path/to/model")      # or ox.Tokenizer.from_file("tokenizer.json")
ids = tok.encode("Hello, world!")
prompt = tok.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True)
```

## What the Hugging Face pipeline covers

* **Normalizers**: BERT (clean text, CJK spacing, accent stripping, lowercase), NFC / NFD / NFKC / NFKD (table driven, identical
  to `unicodedata` for every code point), Lowercase, StripAccents, Strip, Replace (string or regex), Prepend, Append, Sequence,
  SentencePiece's precompiled character map.
* **Pre-tokenizers**: BERT, ByteLevel (GPT-2 regex or none), Whitespace, WhitespaceSplit, Metaspace (all prepend schemes),
  Punctuation, Digits, Split (regex, every split behaviour, invert), CharDelimiterSplit, Sequence.
* **Models**: BPE (prefix / suffix, byte fallback, fuse-unk, ignore-merges, merges as strings or pairs), WordPiece, Unigram
  (Viterbi, byte fallback, unknown fusion), WordLevel. Old files without a model `type` are recognised by shape.
* **Post-processors**: BERT, RoBERTa, ByteLevel (offset trimming), Template (single / pair), Sequence.
* **Decoders**: ByteLevel, WordPiece, Metaspace, BPE, ByteFallback, Fuse, Strip, Replace, CTC, Sequence.
* **Added tokens** (`special`, `normalized`, `lstrip`, `rstrip`, `single_word`), truncation (strategies, stride, direction),
  padding (batch-longest / fixed, multiple-of, left / right), batches, pairs, character offsets, word / sequence ids,
  `save()` back to `tokenizer.json`.
* **Training**: BPE (identical vocabulary order and merges to the reference), WordPiece, WordLevel, Unigram.

The regular expressions the splitters need are handled by a small built-in engine (`tokenizer::regex`: classes, `\p{..}`
properties from generated Unicode tables, look-around, lazy / possessive quantifiers, inline flags).

## Differences worth knowing

* `decode(skip_special_tokens=true)` skips every special token. The reference keeps those declared `normalized: true`
  (Llama-2's `<s>`), which is a quirk, not a feature.
* Overflow windows (`stride`) follow the documented sliding-window rule; the reference returns a single short window for long
  inputs.
* WordPiece and Unigram training in the reference visit words in hash order, so no two of its runs agree; here the order is
  sorted and reproducible. The Unigram trainer (seed substrings, EM, loss-based pruning) is deterministic and produces working
  tokenizers, but it is not bit-identical to the reference's.
* Not supported: the `UnicodeScripts` and `FixedLength` pre-tokenizers, SentencePiece `WORD` models, tiktoken's
  unstable-token APIs, and Jinja features outside the subset described in `chat_template.rs` (templates using other
  statements or filters fail with an explicit error rather than rendering wrongly).
* Unicode data are those of Python 3.11 (Unicode 14); newer code points are treated as unassigned.
* `\p{Script}` supports Han, Hiragana, Katakana, Hangul, Latin, Greek, Cyrillic, Arabic, Hebrew, Thai and Devanagari
  (range based); other scripts raise an error.
