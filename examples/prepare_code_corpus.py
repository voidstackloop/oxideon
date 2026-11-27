"""Builds a small byte-level BPE tokenizer and a token file from a folder of source code (default: the Python standard library),
for `train_gpt.py`.

    python examples/prepare_code_corpus.py --out runs/data --vocab-size 4096
    python examples/prepare_code_corpus.py --hf codeparrot/codeparrot-clean --max-chars 400e6 --out runs/data  # deduplicated GitHub Python
"""
import argparse
import glob
import json
import os
import sys
import sysconfig
import time

import numpy as np

import oxideon as ox


def bytes_to_unicode():
    """GPT-2's byte -> printable character map: the 256 symbols a byte-level BPE needs in its initial alphabet."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", help="folder searched for *.py (repeatable; default: the standard library and this environment's packages)")
    ap.add_argument("--hf", default=None, help="stream this Hugging Face dataset instead of local folders, e.g. codeparrot/codeparrot-clean")
    ap.add_argument("--hf-field", default="content")
    ap.add_argument("--max-chars", type=float, default=120e6, help="stop collecting after this many characters")
    ap.add_argument("--out", default="runs/data")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.hf:  # a Hugging Face dataset, streamed (nothing is stored except the kept texts)
        from datasets import load_dataset
        stream = (r[a.hf_field] for r in load_dataset(a.hf, split="train", streaming=True))
    else:
        srcs = a.src or [os.path.join(sys.base_prefix, "Lib"), sysconfig.get_paths()["purelib"]]
        files = sorted({f for src in srcs for f in glob.glob(os.path.join(src, "**", "*.py"), recursive=True)})

        def read_all():
            for f in files:
                try:
                    if os.path.getsize(f) <= 1_000_000:  # generated tables and data dumps
                        with open(f, encoding="utf8") as fh:
                            yield fh.read()
                except (UnicodeDecodeError, OSError):
                    continue
        stream = read_all()
    texts, total, seen = [], 0, set()
    for t in stream:
        if len(t) > 1_000_000 or max(map(len, t.split(chr(10)))) > 2000 or hash(t) in seen:  # minified / generated / duplicated files
            continue
        seen.add(hash(t))
        texts.append(t)
        total += len(t)
        if total >= a.max_chars:
            break
    print(f"{len(texts)} files, {sum(map(len, texts)) / 1e6:.1f} M characters")
    tok = ox.Tokenizer.untrained("BPE", None)
    tok.set_pre_tokenizer(json.dumps({"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}))
    tok.set_decoder(json.dumps({"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}))
    t0 = time.time()
    tok.train_bpe(texts, vocab_size=a.vocab_size, special_tokens=["<|endoftext|>"], initial_alphabet=sorted(bytes_to_unicode().values()))
    print(f"trained a {tok.get_vocab_size()}-token BPE in {time.time() - t0:.0f}s")
    sample = "def f(x):\n\treturn x  # café ü\n"
    assert tok.decode(tok.encode(sample).ids) == sample, "tokenizer is not lossless"
    tok.save(os.path.join(a.out, "tokenizer.json"))
    eos = tok.token_to_id("<|endoftext|>")
    ids = []
    t0 = time.time()
    for i in range(0, len(texts), 64):
        for enc in tok.encode_batch(texts[i:i + 64]):
            ids.extend(enc.ids)
            ids.append(eos)
    arr = np.asarray(ids, dtype=np.uint16)
    n_val = int(len(arr) * a.val_fraction)
    np.save(os.path.join(a.out, "train.npy"), arr[:-n_val])
    np.save(os.path.join(a.out, "val.npy"), arr[-n_val:])
    print(f"{len(arr) / 1e6:.1f} M tokens ({len(arr) - n_val} train / {n_val} val) in {time.time() - t0:.0f}s; {sum(map(len, texts)) / len(arr):.2f} chars per token")


if __name__ == "__main__":
    main()
