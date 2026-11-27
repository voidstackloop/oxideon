"""Learns the CMUdict -> tiny-voca-label mapping (see frontend_cmu.py), saves it, and reports its accuracy on held-out words.

    python examples/tts/learn_frontend.py --annotations runs/tinyvoca/annotations/v4 --cmudict cmudict.dict \
        --lexicon runs/voca/data/lexicon.json --out runs/voca/data/frontend.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frontend_cmu import CmuFrontEnd, load_cmudict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--cmudict", required=True)
    ap.add_argument("--lexicon", required=True, help="lexicon.json written by synthesize_voca.build_lexicon (word -> [tokens, stress])")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cmu = load_cmudict(a.cmudict)
    lex = {w: (v[0], v[1]) for w, v in json.load(open(a.lexicon)).items()}
    fe = CmuFrontEnd.learn(a.annotations, cmu, lex)
    fe.save(a.out)
    print("CMUdict words:", len(cmu), "| merges:", fe.rules["merge"], "| function words:", len(fe.rules["function_words"]))
    print("held-out:", json.dumps(fe.rules["heldout"]))
    for t in ("Unfortunately, all the promises in the world wouldn't give him a square meal.", "Dr. Reynolds measured 4.5 centimeters in 1999."):
        toks, st, unk = fe.phonemize(t)
        print(t, "->", len(toks), "tokens; unknown:", unk)


if __name__ == "__main__":
    main()
