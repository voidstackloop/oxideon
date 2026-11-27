"""Tests of oxideon.reasoning: sampling controls against NumPy references, think-budget decoding, voting / search, GRPO helpers."""
import numpy as np
import pytest

import oxideon as ox
from oxideon import reasoning as R

LOGITS = np.array([2.0, 1.0, 0.5, 0.1, -1.0, -2.0, 1.5, 0.0], dtype=np.float32)


def softmax(x):
    e = np.exp(x - np.max(x[np.isfinite(x)]))
    return e / e.sum()


def probs(**kw):
    return R.TokenSampler(R.SamplerConfig(**kw)).probs(LOGITS).to_numpy()


def test_temperature_scales_the_distribution_and_zero_is_greedy():
    for t in (0.3, 0.7, 2.0):
        np.testing.assert_allclose(probs(temperature=t), softmax(LOGITS / t), rtol=1e-5)
    p = probs(temperature=0.0)
    assert p.argmax() == 0 and p.max() == 1.0 and R.TokenSampler(R.SamplerConfig(temperature=0.0)).sample(LOGITS) == 0


def test_top_k_keeps_exactly_k_tokens():
    p = probs(top_k=3)
    assert (p > 0).sum() == 3 and set(np.flatnonzero(p)) == {0, 6, 1}
    kept = LOGITS.copy()
    kept[p == 0] = -np.inf
    np.testing.assert_allclose(p, softmax(kept), rtol=1e-5)


def test_top_p_keeps_the_smallest_prefix_reaching_the_mass():
    full = softmax(LOGITS)
    order = np.argsort(-full)
    for top_p in (0.5, 0.8, 0.95):
        cum = np.cumsum(full[order])
        n = int(np.searchsorted(cum, top_p) + 1)
        want = np.zeros_like(full)
        want[order[:n]] = full[order[:n]]
        want /= want.sum()
        np.testing.assert_allclose(probs(top_p=top_p), want, rtol=1e-5, atol=1e-7)
    assert (probs(top_p=1e-6) > 0).sum() == 1  # always at least the top token


def test_min_p_drops_tokens_below_a_fraction_of_the_top_probability():
    full = softmax(LOGITS)
    p = probs(min_p=0.3)
    np.testing.assert_array_equal(p > 0, full >= 0.3 * full.max())


def test_penalties_and_logit_bias():
    hist = [0, 0, 6]
    s = R.TokenSampler(R.SamplerConfig(repetition_penalty=2.0))
    got = s.process(LOGITS, hist).to_numpy()
    want = LOGITS.copy()
    for t in (0, 6):
        want[t] = want[t] / 2.0 if want[t] > 0 else want[t] * 2.0
    np.testing.assert_allclose(got, want, rtol=1e-6)
    neg = R.TokenSampler(R.SamplerConfig(repetition_penalty=2.0)).process(LOGITS, [5]).to_numpy()
    assert neg[5] == pytest.approx(-4.0)  # a negative logit gets more negative
    got = R.TokenSampler(R.SamplerConfig(presence_penalty=0.5, frequency_penalty=0.25)).process(LOGITS, hist).to_numpy()
    want = LOGITS.copy()
    want[0] -= 0.5 + 2 * 0.25
    want[6] -= 0.5 + 0.25
    np.testing.assert_allclose(got, want, rtol=1e-6)
    got = R.TokenSampler(R.SamplerConfig(logit_bias={3: 5.0, 0: -100.0})).process(LOGITS).to_numpy()
    assert got[3] == pytest.approx(5.1) and got[0] == pytest.approx(-98.0)


def test_seed_makes_sampling_reproducible_and_different_seeds_differ():
    a = [R.TokenSampler(R.SamplerConfig(temperature=1.5, seed=7)).sample(LOGITS) for _ in range(1)]
    s1, s2 = R.TokenSampler(R.SamplerConfig(temperature=1.5, seed=3)), R.TokenSampler(R.SamplerConfig(temperature=1.5, seed=3))
    d1 = [s1.sample(LOGITS) for _ in range(40)]
    assert d1 == [s2.sample(LOGITS) for _ in range(40)]
    s3 = R.TokenSampler(R.SamplerConfig(temperature=1.5, seed=4))
    assert d1 != [s3.sample(LOGITS) for _ in range(40)] and a


def test_sampled_frequencies_follow_the_processed_distribution():
    s = R.TokenSampler(R.SamplerConfig(temperature=0.8, top_k=4, seed=0))
    p = s.probs(LOGITS).to_numpy()
    draws = np.bincount([s.sample(LOGITS) for _ in range(4000)], minlength=len(LOGITS)) / 4000
    np.testing.assert_allclose(draws, p, atol=0.03)
    assert draws[p == 0].sum() == 0


def test_config_validation():
    for bad in (dict(temperature=-1), dict(top_p=0), dict(top_p=1.5), dict(top_k=-1), dict(min_p=2), dict(repetition_penalty=0)):
        with pytest.raises(ValueError):
            R.SamplerConfig(**bad)


# ----- think-budget decoding: a scripted "model" ----------------------------------------------------------------------
OPEN, CLOSE, EOS, WAIT, V = 10, 11, 12, 13, 16


def scripted(think_len, answer=(7, 8)):
    """Thinks `think_len` tokens (token 1), tries to close, then answers."""
    def step(ids):
        l = np.full(V, -10.0, dtype=np.float32)
        start = ids.index(OPEN) + 1
        end = ids.index(CLOSE) if CLOSE in ids else None
        if end is None:
            n_think = len(ids) - start
            l[CLOSE if n_think >= think_len else 1] = 10.0
        else:
            k = len(ids) - end - 1
            l[answer[k] if k < len(answer) else EOS] = 10.0
        return l
    return step


def test_thinking_then_answer():
    t = R.generate_with_reasoning(scripted(5), [3], think_open_id=OPEN, think_close_id=CLOSE, eos_id=EOS)
    assert t.thinking_ids == [1] * 5 and t.answer_ids == [7, 8] and t.finished and not t.forced_close and t.extensions == 0


def test_budget_forcing_closes_long_thinking_and_extends_short_thinking():
    t = R.generate_with_reasoning(scripted(50), [3], think_open_id=OPEN, think_close_id=CLOSE, eos_id=EOS, max_think_tokens=8)
    assert t.forced_close and len(t.thinking_ids) == 8
    t = R.generate_with_reasoning(scripted(2), [3], think_open_id=OPEN, think_close_id=CLOSE, eos_id=EOS, min_think_tokens=6,
                                  wait_ids=[WAIT], max_extensions=2)
    assert t.extensions == 2 and t.thinking_ids.count(WAIT) == 2 and t.answer_ids == [7, 8]


def test_separate_sampler_for_thinking_and_answer():
    calls = {"think": 0, "answer": 0}

    class Counting(R.TokenSampler):
        def __init__(self, key):
            super().__init__(R.SamplerConfig(temperature=0.0))
            self.key = key

        def sample(self, logits, history=()):
            calls[self.key] += 1
            return super().sample(logits, history)
    R.generate_with_reasoning(scripted(3), [3], think_open_id=OPEN, think_close_id=CLOSE, eos_id=EOS,
                              sampler=Counting("answer"), think_sampler=Counting("think"))
    assert calls["think"] == 4 and calls["answer"] == 3  # 3 thoughts + the close, 2 answer tokens + eos


# ----- voting / search / RL -------------------------------------------------------------------------------------------

def test_self_consistency_majority_and_weights():
    answers = ["4", "5", "4", "4", "5"]
    best, votes = R.self_consistency(lambda i: answers[i], 5)
    assert best == "4" and votes == {"4": 3.0, "5": 2.0}
    best, _ = R.self_consistency(lambda i: (answers[i], 10.0 if answers[i] == "5" else 1.0), 5, extract=lambda r: r[0], weights=lambda r: r[1])
    assert best == "5"
    with pytest.raises(ValueError):
        R.self_consistency(lambda i: None, 3)


def test_best_of_n_and_tree_of_thoughts():
    assert R.best_of_n([3, 9, 1, 7], lambda x: x, k=2) == [(9, 9.0), (7, 7.0)]
    # reach 24 from 1 by +1 / *2 steps, scoring by closeness
    best, score = R.tree_of_thoughts(1, lambda s: [s + 1, s * 2], lambda s: -abs(24 - s), beam_width=4, depth=8, is_solution=lambda s: s == 24)
    assert best == 24 and score == 0


def test_group_advantages_match_numpy_and_reasoning_reward():
    r = np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.5], dtype=np.float32)
    got = R.group_advantages(r, 3).to_numpy()
    g = r.reshape(2, 3)
    want = ((g - g.mean(1, keepdims=True)) / (g.std(1, keepdims=True) + 1e-6)).reshape(-1)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    assert np.all(got[3:] == 0)  # a group with identical rewards gets zero advantage
    np.testing.assert_allclose(R.group_advantages(r, 3, scale=False).to_numpy(), (g - g.mean(1, keepdims=True)).reshape(-1), atol=1e-6)
    assert R.reasoning_reward("<think>2+2</think>4", "4") == pytest.approx(1.2)
    assert R.reasoning_reward("4", "4") == pytest.approx(1.0)
    assert R.reasoning_reward("<think>x</think><think>y</think>4", "5") == 0.0


def test_degenerate_inputs_fail_loudly_instead_of_sampling_garbage():
    s = R.TokenSampler(R.SamplerConfig(seed=0))
    with pytest.raises(ValueError, match="NaN"):
        s.sample(np.full(8, np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="-inf"):
        s.sample(np.full(8, -np.inf, dtype=np.float32))
    with pytest.raises(ValueError, match="groups of 2"):
        R.group_advantages(np.zeros(5, dtype=np.float32), 2)
    assert R.answers_match(None, "4") is False and R.answers_match(4, "4.0") is True
    assert R.TokenSampler(R.SamplerConfig(top_k=1000, seed=0)).sample(np.zeros(8, dtype=np.float32)) in range(8)
