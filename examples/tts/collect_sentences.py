"""Collects clean English sentences (docstrings, comments and Markdown from a Python installation and this repository) as the
text for a text-to-speech corpus. Only letters and basic punctuation survive, so the synthesizer reads exactly what is written.

    python examples/tts/collect_sentences.py --out runs/tts/sentences.txt --count 3000
"""
import argparse
import ast
import glob
import os
import random
import re
import sys
import sysconfig

OK = re.compile(r"^[A-Za-z][A-Za-z ,.;:?!'\-]*[.?!]$")
SPLIT = re.compile(r"(?<=[.?!])\s+")


def sentences_from(text):
    text = re.sub(r"\s+", " ", text.replace("`", "").replace("*", "").replace("_", " "))
    for s in SPLIT.split(text):
        s = s.strip()
        words = s.split()
        if 5 <= len(words) <= 28 and 30 <= len(s) <= 170 and OK.match(s) and not any(len(w) > 18 for w in words):
            yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tts/sentences.txt")
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", action="append")
    a = ap.parse_args()
    srcs = a.src or [os.path.join(sys.base_prefix, "Lib"), sysconfig.get_paths()["purelib"]]
    seen, out = set(), []
    files = sorted({f for s in srcs for f in glob.glob(os.path.join(s, "**", "*.py"), recursive=True)})
    random.Random(a.seed).shuffle(files)
    for f in files:
        try:
            if os.path.getsize(f) > 400_000:
                continue
            tree = ast.parse(open(f, encoding="utf8").read())
        except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                for s in sentences_from(doc or ""):
                    key = s.lower()
                    if key not in seen:
                        seen.add(key)
                        out.append(s)
        if len(out) >= a.count * 3:
            break
    random.Random(a.seed).shuffle(out)
    out = out[:a.count]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"{len(out)} sentences, mean {sum(map(len, out)) / len(out):.0f} characters; e.g.\n  " + "\n  ".join(out[:5]))


if __name__ == "__main__":
    main()
