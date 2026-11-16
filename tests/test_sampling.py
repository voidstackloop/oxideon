"""Z9: generation sampling (`SamplingConfig` + `Tensor.sample_token`) pinned from Python.

Covers greedy decoding (argmax, first-tie, NaN-skip), the HF-parity top-p edge
(lone dominant token collapses the nucleus), top-k support restriction,
repetition-penalty flips, seeded-stream reproducibility, and empty-logit errors.
"""
import numpy as np
import pytest

import oxideon as ox

T = ox.from_numpy


def test_greedy_argmax_tie_and_nan():
    g = ox.SamplingConfig.greedy()
    assert T(np.array([1.0, 5.0, 3.0])).sample_token(g) == 1
    assert T(np.array([2.0, 2.0, 2.0])).sample_token(g) == 0
    assert T(np.array([1.0, float("nan"), 3.0])).sample_token(g) == 2
    with pytest.raises(Exception, match="empty"):
        T(np.array([], dtype=np.float64)).sample_token(g)


def test_top_p_lone_dominant_collapses_nucleus():
    cfg = ox.SamplingConfig(temperature=1.0, top_p=0.4, seed=21)
    x = T(np.array([2.0, 2.0, -10.0, -10.0]))
    assert [x.sample_token(cfg) for _ in range(50)] == [0] * 50


def test_top_k_restricts_support_and_top1_is_argmax():
    cfg = ox.SamplingConfig(temperature=1.0, top_k=2, seed=7)
    x = T(np.array([1.0, 5.0, 3.0, 0.0]))
    assert {x.sample_token(cfg) for _ in range(50)} <= {1, 2}
    cfg1 = ox.SamplingConfig(temperature=1.0, top_k=1, seed=11)
    assert [x.sample_token(cfg1) for _ in range(10)] == [1] * 10


def test_repetition_penalty_flips_greedy_and_seed_reproduces():
    x = T(np.array([5.0, 4.0]))
    cfg = ox.SamplingConfig.greedy()
    cfg.repetition_penalty = 3.0
    assert x.sample_token(cfg, history=[0]) == 1
    assert x.sample_token(cfg) == 0
    a = ox.SamplingConfig(temperature=1.0, top_k=2, seed=99)
    b = ox.SamplingConfig(temperature=1.0, top_k=2, seed=99)
    xs = T(np.array([1.0, 5.0, 3.0, 0.0]))
    assert [xs.sample_token(a) for _ in range(5)] == [xs.sample_token(b) for _ in range(5)]
