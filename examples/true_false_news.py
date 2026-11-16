#!/usr/bin/env python3
"""TRUE / FALSE news verdicts with a pretrained Oxideon fake-news detector.

Loads the `fake_news_detector.safetensors` weights (+ vocabulary) produced by
`examples/fake_news_detector.py` - training them quickly when they are missing -
then prints a TRUE (credible / real) or FALSE (wrong / fake) verdict for each
sample headline and checks every verdict against its expected label.

Usage:
    python3 examples/true_false_news.py [--weights PATH] [--epochs N]
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from examples.fake_news_detector import (
    FakeNewsClassifier,
    NewsTextPipeline,
    eval_input,
    train_and_evaluate,
)

# (headline, expected verdict): TRUE = credible / real news, FALSE = wrong / fake news.
SAMPLE_HEADLINES = [
    (
        "NASA astronomers published peer-reviewed findings confirming water ice "
        "plumes erupting from Jupiter's moon Europa.",
        "TRUE",
    ),
    (
        "The Federal Reserve Board announced a 25 basis point benchmark interest "
        "rate adjustment following monthly labor statistics.",
        "TRUE",
    ),
    (
        "Clinical researchers at Mayo Clinic concluded a five-year double-blind "
        "trial on pediatric immunotherapy tolerance rates.",
        "TRUE",
    ),
    (
        "Department of Energy researchers demonstrated a 14 percent efficiency "
        "increase in perovskite photovoltaic solar cells.",
        "TRUE",
    ),
    (
        "SHOCKING BOMBSHELL: Rogue insider reveals drinking boiled lemon garlic "
        "kerosene permanently cures all viral cancer within 36 hours! SHARE BEFORE CENSORED!!!",
        "FALSE",
    ),
    (
        "WHISTLEBLOWER LEAKS: Secret government treaty proves 5G cell towers emit "
        "psychoactive mind-control frequencies causing obedience!",
        "FALSE",
    ),
    (
        "THEY DON'T WANT YOU TO KNOW: Undercover general confirms elite globalists "
        "secretly replaced world leaders with android clones!",
        "FALSE",
    ),
    (
        "URGENT ALERT: Mysterious Antarctic ice wall breach exposes ancient reptilian "
        "stargate guarded by nuclear submarines!",
        "FALSE",
    ),
]


def default_paths():
    """Repo-root weights + vocabulary paths shared with `fake_news_detector.py`."""
    weights = os.path.join(str(REPO_ROOT), "fake_news_detector.safetensors")
    vocab = os.path.splitext(weights)[0] + "_vocab.json"
    return weights, vocab


def load_or_train_detector(weights_path, vocab_path, epochs=1):
    """Loads a trained detector, training one quickly when no weights exist."""
    if os.path.exists(weights_path) and os.path.exists(vocab_path):
        print(f"Loading pre-trained model from '{weights_path}'...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            pipeline = NewsTextPipeline.from_json(f.read())
        model = FakeNewsClassifier(vocab_size=len(pipeline.vocab), embed_dim=64, hidden_dim=128)
        model.load_file(weights_path)
        print("Model loaded successfully!")
        return model, pipeline
    print(f"No weights at '{weights_path}'. Training a detector ({epochs} epoch(s))...")
    return train_and_evaluate(save_path=weights_path, epochs=epochs)


def verdict_for(text, model, pipeline):
    """Returns the TRUE/FALSE verdict and confidence for one headline."""
    result = eval_input(text, model, pipeline, verbose=False)
    verdict = "FALSE" if result["is_fake"] else "TRUE"
    return verdict, result["confidence"]


def main():
    parser = argparse.ArgumentParser(
        description="TRUE/FALSE news verdicts with a pretrained Oxideon detector"
    )
    parser.add_argument("--weights", type=str, default=None, help="Weights path (default: repo-root detector)")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs when no weights exist")
    args = parser.parse_args()

    weights_path, vocab_path = default_paths()
    if args.weights:
        weights_path = args.weights
        vocab_path = os.path.splitext(weights_path)[0] + "_vocab.json"

    model, pipeline = load_or_train_detector(weights_path, vocab_path, epochs=args.epochs)

    print("\nTRUE / FALSE news verdicts:")
    correct = 0
    for text, expected in SAMPLE_HEADLINES:
        verdict, confidence = verdict_for(text, model, pipeline)
        mark = "ok" if verdict == expected else "MISMATCH"
        if verdict == expected:
            correct += 1
        print(f"  [{verdict:5s}] (confidence {confidence * 100:5.2f}%) {mark}  {text[:70]}...")

    print(f"\n{correct}/{len(SAMPLE_HEADLINES)} verdicts correct.")
    assert correct == len(SAMPLE_HEADLINES), "Some verdicts did not match their expected labels"
    print("TRUE/FALSE news demo passed!")


if __name__ == "__main__":
    main()
