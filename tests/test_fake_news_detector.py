"""Unit tests for the AI Wrong/Fake News Detector trained on 10,000 dataset."""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import oxideon as ox
from examples.fake_news_detector import (
    generate_10k_news_dataset,
    NewsTextPipeline,
    FakeNewsClassifier,
    eval_input,
)


def test_generate_10k_news_dataset_balance_and_size():
    dataset = generate_10k_news_dataset(seed=123)
    assert len(dataset) == 10000, f"Expected 10,000 samples, got {len(dataset)}"

    real_count = sum(1 for _, label in dataset if label == 0)
    fake_count = sum(1 for _, label in dataset if label == 1)

    assert real_count == 5000, f"Expected 5,000 real samples, got {real_count}"
    assert fake_count == 5000, f"Expected 5,000 fake samples, got {fake_count}"


def test_pipeline_tokenization_and_stylometrics():
    pipeline = NewsTextPipeline(max_vocab=100, max_len=16)
    sample_texts = [
        "NASA astronomers published peer-reviewed findings in Nature.",
        "SHOCKING BOMBSHELL: Underground alien stargate discovered! SHARE NOW!!!",
    ]
    pipeline.fit(sample_texts)

    ids, stylom = pipeline.transform_batch(sample_texts)
    assert ids.shape == (2, 16)
    assert stylom.shape == (2, 6)

    # Stylometric assertions
    # Real news should have low caps ratio and no exclamation marks
    assert stylom[0, 1] == 0.0  # no '!'
    # Fake news should have high exclamation marks
    assert stylom[1, 1] > 0.0  # has '!'


def test_fake_news_classifier_forward_pass():
    vocab_size = 50
    model = FakeNewsClassifier(vocab_size=vocab_size, embed_dim=16, hidden_dim=32)
    model.eval()

    batch_size = 4
    token_ids = ox.from_numpy(np.random.randint(0, vocab_size, size=(batch_size, 16)).astype(np.int64))
    stylom = ox.from_numpy(np.random.randn(batch_size, 6).astype(np.float32))

    logits = model(token_ids, stylom)
    assert logits.shape == [batch_size, 2]

    probs = logits.softmax(1).to_numpy()
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)


def test_eval_input_predictions():
    # Build quick pipeline and model
    dataset = generate_10k_news_dataset(seed=42)
    texts, labels = zip(*dataset[:500])  # fast subset for test
    pipeline = NewsTextPipeline(max_vocab=500, max_len=24)
    pipeline.fit(list(texts))

    model = FakeNewsClassifier(vocab_size=len(pipeline.vocab), embed_dim=32, hidden_dim=64)

    # Check eval_input structure
    real_sample = "Researchers at Stanford University published official peer-reviewed clinical findings."
    fake_sample = "SHOCKING BOMBSHELL: Whistleblower admits secret 5G frequency mind control! SPREAD THIS!"

    res_real = eval_input(real_sample, model, pipeline, verbose=False)
    assert "verdict" in res_real
    assert "confidence" in res_real
    assert "prob_real" in res_real
    assert "prob_fake" in res_real
    assert 0.0 <= res_real["confidence"] <= 1.0

    res_fake = eval_input(fake_sample, model, pipeline, verbose=False)
    assert "signals" in res_fake
    assert len(res_fake["signals"]) > 0


def test_save_and_load_trained_detector():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_detector.safetensors")

        vocab_size = 120
        model = FakeNewsClassifier(vocab_size=vocab_size, embed_dim=16, hidden_dim=32)
        model.eval()

        # Save
        ox.save_file(model, save_path)
        assert os.path.exists(save_path)

        # Clone and Load
        loaded_model = FakeNewsClassifier(vocab_size=vocab_size, embed_dim=16, hidden_dim=32)
        loaded_model.eval()
        loaded_model.load_file(save_path)

        token_ids = ox.from_numpy(np.random.randint(0, vocab_size, size=(2, 16)).astype(np.int64))
        stylom = ox.from_numpy(np.random.randn(2, 6).astype(np.float32))

        orig_out = model(token_ids, stylom).to_numpy()
        loaded_out = loaded_model(token_ids, stylom).to_numpy()

        np.testing.assert_allclose(orig_out, loaded_out, atol=1e-7)
