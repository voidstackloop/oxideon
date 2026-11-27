"""Samples face attributes and writes (caption, teacher prompt) pairs.

The caption is what the 10M model sees; the teacher prompt is what the big model (FLUX.1-schnell) renders. Both come from the
same attribute dict, so the image always matches its caption by construction.

    python examples/faces/make_prompts.py --n 40000 --out faces_prompts.jsonl
"""
import argparse
import json
import random

ATTRS = {
    "age": ["a child", "a teenager", "a young adult", "a middle-aged", "an elderly"],
    "gender": ["woman", "man"],
    "hair_color": ["black", "brown", "blonde", "red", "gray", "white"],
    "hair_style": ["short", "long", "curly", "wavy", "straight", "bald", "a ponytail", "braided"],
    "expression": ["smiling", "neutral", "laughing", "serious", "surprised", "sad"],
    "glasses": ["", "", "wearing glasses", "wearing sunglasses"],
    "beard": ["", "", "", "with a beard", "with a mustache"],
    "background": ["a white", "a gray", "a blue", "a green", "a warm", "an outdoor"],
    "light": ["soft", "bright", "dramatic", "warm"],
}


def sample(rng):
    a = {k: rng.choice(v) for k, v in ATTRS.items()}
    if a["hair_style"] == "bald":
        a["hair_color"] = ""
    if a["gender"] == "woman" or a["age"] in ("a child", "a teenager"):
        a["beard"] = ""
    return a


def who(a):
    """'a child' + woman -> 'a girl' (children and teenagers get their own nouns)."""
    girl = a["gender"] == "woman"
    if a["age"] == "a child":
        return "a girl" if girl else "a boy"
    if a["age"] == "a teenager":
        return "a teenage girl" if girl else "a teenage boy"
    return f"{a['age']} {a['gender']}"


def hair(a):
    if a["hair_style"] == "bald":
        return "bald"
    if a["hair_style"] in ("a ponytail", "braided"):
        return f"{a['hair_color']} hair in {a['hair_style']}" if a["hair_style"] == "a ponytail" else f"braided {a['hair_color']} hair"
    return f"{a['hair_style']} {a['hair_color']} hair"


def caption(a):
    extras = ", ".join(x for x in (a["glasses"], a["beard"]) if x)
    s = f"{who(a)} with {hair(a)}, {a['expression']}"
    if extras:
        s += f", {extras}"
    return s + f", {a['background']} background"


def prompt(a):
    extras = ", ".join(x for x in (a["glasses"], a["beard"]) if x)
    return (f"studio portrait photo of {who(a)} with {hair(a)}, {a['expression']} expression"
            + (f", {extras}" if extras else "") + f", {a['background']} background, {a['light']} lighting, face centered, looking at camera, sharp focus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="faces_prompts.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    seen = set()
    with open(a.out, "w", encoding="utf8") as f:
        while len(seen) < a.n:
            at = sample(rng)
            c = caption(at)
            if c in seen:  # distinct captions only (the grid has ~10^6 combinations)
                continue
            seen.add(c)
            f.write(json.dumps({"id": len(seen) - 1, "caption": c, "prompt": prompt(at), "attrs": at}) + "\n")
    print(f"wrote {len(seen)} prompts to {a.out}")


if __name__ == "__main__":
    main()
