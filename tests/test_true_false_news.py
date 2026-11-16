"""Focused tests for the TRUE/FALSE news verdict demo."""

import os
import sys
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.true_false_news import (
    SAMPLE_HEADLINES,
    default_paths,
    load_or_train_detector,
    verdict_for,
)


def test_sample_headlines_cover_both_verdicts():
    labels = {expected for _, expected in SAMPLE_HEADLINES}
    assert labels == {"TRUE", "FALSE"}
    assert len(SAMPLE_HEADLINES) >= 4


def test_true_false_verdicts_match_expected_labels():
    weights_path, vocab_path = default_paths()
    if os.path.exists(weights_path) and os.path.exists(vocab_path):
        model, pipeline = load_or_train_detector(weights_path, vocab_path)
    else:
        # No pretrained weights (e.g. a fresh checkout): train to a temp dir.
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "detector.safetensors")
            vocab_path = os.path.splitext(weights_path)[0] + "_vocab.json"
            model, pipeline = load_or_train_detector(weights_path, vocab_path, epochs=1)
            assert os.path.exists(weights_path)
            assert os.path.exists(vocab_path)
            _assert_verdicts(model, pipeline)
            return

    _assert_verdicts(model, pipeline)


def _assert_verdicts(model, pipeline):
    for text, expected in SAMPLE_HEADLINES:
        verdict, confidence = verdict_for(text, model, pipeline)
        assert verdict == expected, f"{verdict} != {expected} for {text!r}"
        assert 0.0 <= confidence <= 1.0
