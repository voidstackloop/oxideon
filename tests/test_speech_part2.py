"""Focused tests for Group 9: verification, ASR-LM, forced alignment, TTS frontend."""

import math

import numpy as np
import pytest

import oxideon as ox


def test_verification_scores():
    e = np.eye(3, dtype=np.float32)
    s = ox.verification_scores(e, e).to_numpy()
    assert s.tolist() == pytest.approx([1.0, 1.0, 1.0])
    s = ox.verification_scores([[1.0, 0]], [[0.0, 1.0]]).to_numpy()
    assert s.tolist() == [0.0]
    with pytest.raises(ValueError):
        ox.verification_scores([[0.0, 0]], [[1.0, 0]])
    with pytest.raises(ValueError):
        ox.verification_scores(np.zeros((2, 2)), np.zeros((3, 2)))


def test_equal_error_rate():
    eer, _ = ox.equal_error_rate([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert eer == 0.0
    eer, thr = ox.equal_error_rate([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 0])
    assert eer == pytest.approx(0.5)
    assert thr == pytest.approx(0.6)
    eer, _ = ox.equal_error_rate([0.7, 0.3], [0, 1])
    assert eer == 1.0
    with pytest.raises(ValueError):
        ox.equal_error_rate([0.5, 0.6], [1, 1])


def test_min_detection_cost_and_det():
    assert ox.min_detection_cost([0.9, 0.1], [1, 0]) == 0.0
    assert ox.min_detection_cost([0.7, 0.3], [0, 1]) == pytest.approx(1.0)
    d = ox.det_curve([0.9, 0.6, 0.4, 0.1], [1, 0, 1, 0])
    assert d["fpr"][0] == 0.0 and d["fpr"][-1] == 1.0
    assert d["fnr"][0] == 1.0 and d["fnr"][-1] == 0.0
    assert len(d["fpr"]) == len(d["thresholds"]) == 5
    with pytest.raises(ValueError):
        ox.min_detection_cost([0.5], [1], p_target=0.0)


def test_ngram_lm_exact_unigram():
    lm = ox.NGramLM(n=1).fit([["a", "b", "a"]])
    assert lm.prob("a") == pytest.approx(0.5)
    assert lm.prob("b") == pytest.approx(1 / 3)
    assert lm.prob("zzz") == pytest.approx(1 / 6)
    assert lm.sentence_logprob(["a", "b"]) == pytest.approx(math.log(0.5) + math.log(1 / 3))
    assert lm.perplexity(["a", "b"]) == pytest.approx(math.sqrt(6))
    with pytest.raises(ValueError):
        ox.NGramLM(n=0)
    with pytest.raises(ValueError):
        ox.NGramLM().fit([])
    with pytest.raises(ValueError):
        ox.NGramLM().prob("a")


def test_ngram_lm_bigram_backoff_and_rescore():
    lm = ox.NGramLM(n=2).fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
    assert lm.prob("cat", ["the"]) > lm.prob("zebra", ["the"]) > 0
    assert lm.perplexity(["the", "cat"]) >= 1.0
    beams = [(["zebra"], -0.5), (["the", "cat"], -0.6)]
    out = ox.beam_rescore(beams, lm, lm_weight=1.0)
    assert out[0][0] == ("the", "cat")
    assert out[0][1] == pytest.approx(out[0][2] + out[0][3])
    with pytest.raises(ValueError):
        ox.beam_rescore([], lm)
    with pytest.raises(ValueError):
        ox.beam_rescore(beams, object())


def test_ctc_forced_align():
    lp = np.full((5, 3), -10.0)
    for t, c in enumerate([0, 1, 1, 2, 0]):
        lp[t, c] = 0.0
    labels, segs = ox.ctc_forced_align(lp, [1, 2], blank=0)
    assert labels == [0, 1, 1, 2, 0]
    assert segs == [(1, 2), (3, 3)]
    with pytest.raises(ValueError):
        ox.ctc_forced_align(lp[:3], [1, 2])
    with pytest.raises(ValueError):
        ox.ctc_forced_align(lp, [0])


def test_normalize_text():
    assert ox.normalize_text("Pay $3.50 and €2.") == "pay three dollars and fifty cents and two euros"
    assert ox.normalize_text("Call 555-1234!") == "call five five five one two three four"
    assert ox.normalize_text("on 01/02/2023") == "on january second twenty twenty three"
    assert ox.normalize_text("3rd and 21st") == "third and twenty first"
    assert ox.normalize_text("Dr. Smith vs. Mr. Jones") == "doctor smith versus mister jones"
    assert ox.normalize_text("50% of 3.5") == "fifty percent of three point five"
    assert ox.normalize_text("at 3:45pm") == "at three forty five p m"
    assert ox.normalize_text("Room 101, e.g. lab") == "room one hundred one for example lab"
    with pytest.raises(ValueError):
        ox.normalize_text(123)
