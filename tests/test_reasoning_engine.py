"""Generation engine, search strategies, answer utilities, GRPO and the ReasoningEngine in oxideon.reasoning."""
import itertools
import math

import numpy as np
import pytest

import oxideon as ox
from oxideon import reasoning as R

torch = pytest.importorskip("torch")


def table_model(V=6, seed=0, sharp=2.0):
    """A deterministic bigram 'model': next-token logits depend on the last token only."""
    tbl = np.random.default_rng(seed).standard_normal((V, V)).astype(np.float32) * sharp

    def step(ids):
        return tbl[ids[-1]]
    return step, tbl


def lsm(x):
    x = np.asarray(x, dtype=np.float64)
    return x - np.log(np.exp(x - x.max()).sum()) - x.max()


# ----------------------------------------------------------------------------------------------------------- generation engine

def test_greedy_generation_follows_the_model_and_reports_logprobs():
    step, tbl = table_model()
    r = R.generate_tokens(step, [0], R.GenerationConfig(max_new_tokens=6))
    ids, want = [0], []
    for _ in range(6):
        ids.append(int(np.argmax(tbl[ids[-1]])))
    assert r.token_ids == ids[1:] and r.finish_reason == "length"
    prev = [0] + r.token_ids[:-1]
    np.testing.assert_allclose(r.logprobs, [lsm(tbl[p])[t] for p, t in zip(prev, r.token_ids)], rtol=1e-4, atol=1e-5)
    ent = [-(np.exp(lsm(tbl[p])) * lsm(tbl[p])).sum() for p in prev]
    np.testing.assert_allclose(r.entropies, ent, rtol=1e-4, atol=1e-5)
    assert r.perplexity == pytest.approx(math.exp(-np.mean(r.logprobs)))


def test_stop_tokens_min_length_and_stop_strings():
    step, tbl = table_model()
    first = R.generate_tokens(step, [0], R.GenerationConfig(max_new_tokens=8)).token_ids
    stop = first[2]
    r = R.generate_tokens(step, [0], R.GenerationConfig(max_new_tokens=8, stop_token_ids=(stop,)))
    assert r.finish_reason == "stop" and r.token_ids[-1] == stop and len(r.token_ids) <= 3
    r = R.generate_tokens(step, [0], R.GenerationConfig(max_new_tokens=8, min_new_tokens=5, stop_token_ids=(stop,)))
    assert len(r.token_ids) >= 5 and stop not in r.token_ids[:4]
    decode = lambda ids: "".join(chr(97 + t) for t in ids)

    def counting(ids):  # emits 1, 2, 3, ... (mod 6): "bcdefabc..."
        l = np.zeros(6, dtype=np.float32)
        l[len(ids) % 6] = 9.0
        return l
    r = R.generate_tokens(counting, [0], R.GenerationConfig(max_new_tokens=8, stop_strings=("ef",)), decode_fn=decode)
    assert r.finish_reason == "stop_string" and r.text == "bcd" and r.token_ids == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        R.generate_tokens(step, [0], R.GenerationConfig(stop_strings=("a",)))


def test_streaming_matches_generate_and_returns_the_result():
    step, _ = table_model()
    cfg = R.GenerationConfig(max_new_tokens=5, sampler=R.SamplerConfig(temperature=1.0, seed=3))
    gen = R.stream_generate(step, [1], cfg)
    got = []
    while True:
        try:
            got.append(next(gen))
        except StopIteration as done:
            res = done.value
            break
    assert [t for t, _ in got] == res.token_ids == R.generate_tokens(step, [1], cfg).token_ids


def test_seeds_make_batches_reproducible_and_samples_independent():
    step, _ = table_model(sharp=0.3)
    cfg = R.GenerationConfig(max_new_tokens=12, sampler=R.SamplerConfig(temperature=1.0))
    a = R.generate_batch(step, [[0]] * 4, cfg, seed=10)
    b = R.generate_batch(step, [[0]] * 4, cfg, seed=10)
    assert [r.token_ids for r in a] == [r.token_ids for r in b]
    assert len({tuple(r.token_ids) for r in a}) > 1


def test_generation_input_validation():
    step, _ = table_model()
    with pytest.raises(ValueError):
        R.generate_tokens(step, [], R.GenerationConfig())
    with pytest.raises(ValueError):
        R.generate_tokens(step, [-1], R.GenerationConfig())
    with pytest.raises(ValueError):
        R.GenerationConfig(max_new_tokens=2, min_new_tokens=3)


def test_native_and_python_sampler_engines_agree_on_the_distribution():
    logits = np.random.default_rng(1).standard_normal(12).astype(np.float32) * 2
    cfg = R.SamplerConfig(temperature=0.9, top_k=6, top_p=0.9, min_p=0.02, seed=5)
    for engine in ("native", "python"):
        s = R.TokenSampler(cfg, engine=engine)
        p = s.probs(logits).to_numpy()
        freq = np.bincount([s.sample(logits) for _ in range(5000)], minlength=12) / 5000
        np.testing.assert_allclose(freq, p, atol=0.03)
        assert freq[p == 0].sum() == 0


def test_causal_lm_adapter_runs_the_real_model():
    model = ox.CausalLM(50, 16, 2, 2, 32, 16)
    step = R.causal_lm_step(model)
    logits = step([1, 2, 3])
    assert tuple(logits.shape) == (50,)
    r = R.generate_tokens(step, [1, 2, 3], R.GenerationConfig(max_new_tokens=4))
    ids = [1, 2, 3]
    for t in r.token_ids:
        assert int(step(ids).argmax()) == t
        ids.append(t)
    # sampling the real model with controls
    cfg = R.GenerationConfig(max_new_tokens=4, sampler=R.SamplerConfig(temperature=0.8, top_k=5, repetition_penalty=1.2, seed=1))
    assert R.generate_tokens(step, [1, 2, 3], cfg).token_ids == R.generate_tokens(step, [1, 2, 3], cfg).token_ids


# ------------------------------------------------------------------------------------------------------------------ search

def test_beam_search_matches_brute_force_when_the_beam_is_wide_enough():
    step, tbl = table_model(V=3, seed=4, sharp=1.5)
    L = 3
    best = max(itertools.product(range(3), repeat=L),
               key=lambda seq: sum(lsm(tbl[p])[t] for p, t in zip((0,) + seq[:-1], seq)))
    out = R.beam_search(step, [0], num_beams=9, max_new_tokens=L)
    assert tuple(out[0].token_ids) == best
    greedy = R.generate_tokens(step, [0], R.GenerationConfig(max_new_tokens=L)).token_ids
    assert out[0].logprob >= sum(lsm(tbl[p])[t] for p, t in zip([0] + greedy[:-1], greedy)) - 1e-6
    multi = R.beam_search(step, [0], num_beams=4, max_new_tokens=L, num_return=3)
    assert [b.score for b in multi] == sorted((b.score for b in multi), reverse=True)


def test_beam_search_eos_and_length_penalty():
    V, EOS = 4, 3
    def step(ids):  # eos is likely early, otherwise keep going with token 1
        l = np.full(V, -3.0, dtype=np.float32)
        l[1] = 0.0
        l[EOS] = 0.5 if len(ids) <= 2 else -1.0
        return l
    out = R.beam_search(step, [0], num_beams=3, max_new_tokens=6, eos_id=EOS, length_penalty=0.0)
    assert out[0].finished and out[0].token_ids[-1] == EOS
    longer = R.beam_search(step, [0], num_beams=3, max_new_tokens=6, eos_id=EOS, length_penalty=2.0)
    assert len(longer[0].token_ids) >= len(out[0].token_ids)


def test_mcts_finds_the_best_branch_and_is_deterministic():
    # state = tuple of digits chosen so far; reward = digit sum / max. Choosing 3 first is best.
    def expand(s):
        return [s + (d,) for d in (1, 2, 3)] if len(s) < 3 else []
    value = lambda s: sum(s) / 9.0
    a = R.mcts((), expand, value, simulations=200, is_terminal=lambda s: len(s) == 3)
    b = R.mcts((), expand, value, simulations=200, is_terminal=lambda s: len(s) == 3)
    assert a.best_child == (3,) and a.path[1] == (3,) and a.visits == b.visits
    assert a.visits[2] > a.visits[0]
    priors = R.mcts((), lambda s: [R.MCTSChild((1,), 0.01), R.MCTSChild((2,), 0.99)] if not s else [], lambda s: 0.5, simulations=50)
    assert priors.best_child == (2,)
    with pytest.raises(ValueError):
        R.mcts((), expand, value, simulations=0)


def test_adaptive_self_consistency_stops_early_when_settled():
    easy = R.adaptive_self_consistency(lambda i: "4", min_samples=3, max_samples=30, confidence=0.9)
    assert easy.answer == "4" and easy.n_samples < 10 and easy.stopped_early and easy.confidence >= 0.9
    seq = ["a", "b", "a", "b"] * 10
    hard = R.adaptive_self_consistency(lambda i: seq[i], min_samples=3, max_samples=20, confidence=0.95)
    assert hard.n_samples == 20 and not hard.stopped_early
    mostly = R.adaptive_self_consistency(lambda i: "a" if i % 4 else "b", min_samples=3, max_samples=40, confidence=0.9)
    assert mostly.answer == "a" and mostly.votes["a"] > mostly.votes["b"]
    again = R.adaptive_self_consistency(lambda i: "a" if i % 4 else "b", min_samples=3, max_samples=40, confidence=0.9)
    assert (again.n_samples, again.confidence) == (mostly.n_samples, mostly.confidence)
    with pytest.raises(ValueError):
        R.adaptive_self_consistency(lambda i: None, extract=lambda r: r, min_samples=2, max_samples=3)


def test_self_refine_stops_on_clean_critique_and_rejects_worse_revisions():
    final, hist = R.self_refine(1, lambda d: None if d >= 4 else "bigger", lambda d, f: d + 1, max_rounds=10)
    assert final == 4 and hist == [1, 2, 3, 4]
    final, hist = R.self_refine(5, lambda d: "x", lambda d, f: d - 1, max_rounds=3, score_fn=lambda d: d)
    assert final == 5 and hist == [5]
    final, _ = R.self_refine(0, lambda d: "x", lambda d, f: d + 1, max_rounds=10, accept_fn=lambda d: d >= 2)
    assert final == 2


# --------------------------------------------------------------------------------------------------------- answers and rewards

@pytest.mark.parametrize("text,want", [
    (r"so \boxed{42}.", "42"), (r"\boxed{\frac{1}{2}} and later \boxed{\frac{3}{4}}", r"\frac{3}{4}"), (r"\boxed{a{b}c}", "a{b}c"),
    ("x <answer> 7 </answer>", "7"), ("steps\n#### 1,234", "1,234"), ("Therefore the answer is 12.", "12"), ("It costs 3 then 15", "15"),
    ("no digits here", None),
])
def test_extract_final_answer(text, want):
    assert R.extract_final_answer(text) == want


@pytest.mark.parametrize("a,b,ok", [("1,000", "1000", True), ("0.5", "1/2", True), (r"\frac{1}{2}", "0.5", True), ("50%", "0.5", True),
                                    ("$12$", "12", True), (r"\text{Yes}", "yes", True), ("12.", "12", True), ("3", "4", False),
                                    ("1/0", "x", False), ("0.33", "1/3", False)])
def test_answers_match(a, b, ok):
    assert R.answers_match(a, b) is ok


def test_split_reasoning_and_rewards():
    assert R.split_reasoning("<think>a b</think> 4") == ("a b", "4", True)
    assert R.split_reasoning("<think>cut off") == ("cut off", "", False)
    assert R.split_reasoning("just 4") == (None, "just 4", False)
    assert R.split_reasoning("<think>a</think><think>b</think>4")[2] is False
    assert R.reasoning_reward("<think>x</think>1/2", "0.5") == pytest.approx(1.2)
    assert R.reasoning_reward("<think>x</think>\\boxed{3}", "3", extract=R.extract_final_answer) == pytest.approx(1.2)
    both = R.combine_rewards([(lambda t: 1.0, 2.0), (lambda t: 0.5, 1.0)])
    assert both("x") == pytest.approx(2.5)
    assert R.overlong_penalty(50, 100, 20) == 0.0 and R.overlong_penalty(90, 100, 20) == pytest.approx(-0.5)
    assert R.overlong_penalty(100, 100, 20) == -1.0 and R.overlong_penalty(120, 100) == -1.0


# --------------------------------------------------------------------------------------------------------------------- GRPO

def test_sequence_logprobs_match_torch():
    lg = np.random.default_rng(0).standard_normal((5, 7)).astype(np.float32)
    ids = [3, 0, 6, 2, 2]
    got = R.sequence_logprobs(ox.from_numpy(lg), ids).to_numpy()
    want = torch.log_softmax(torch.from_numpy(lg), -1)[torch.arange(5), torch.tensor(ids)].numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def torch_grpo(new, old, ref, adv, mask, clip_eps, beta, normalize):
    ratio = torch.exp(new - old)
    a = adv[:, None]
    per = -torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    d = ref - new
    per = per + beta * (torch.exp(d) - d - 1)
    masked = per * mask
    if normalize == "constant":
        return masked.sum() / new.numel()
    if normalize == "token":
        return masked.sum() / mask.sum()
    return (masked.sum(1) / mask.sum(1)).mean()


@pytest.mark.parametrize("normalize", ["sequence", "token", "constant"])
def test_grpo_loss_and_gradient_match_torch(normalize):
    rng = np.random.default_rng(2)
    B, T = 4, 6
    new = rng.normal(-1.0, 0.3, (B, T)).astype(np.float32)
    old = new + rng.normal(0, 0.25, (B, T)).astype(np.float32)
    ref = new + rng.normal(0, 0.1, (B, T)).astype(np.float32)
    adv = rng.standard_normal(B).astype(np.float32)
    mask = (rng.random((B, T)) > 0.25).astype(np.float32)
    mask[:, 0] = 1.0
    no = ox.from_numpy(new).requires_grad_(True)
    loss = R.grpo_policy_loss(no, ox.from_numpy(old), ox.from_numpy(adv), ox.from_numpy(ref), ox.from_numpy(mask), clip_eps=0.2, kl_beta=0.05,
                       normalize=normalize)
    nt = torch.from_numpy(new).requires_grad_(True)
    want = torch_grpo(nt, torch.from_numpy(old), torch.from_numpy(ref), torch.from_numpy(adv), torch.from_numpy(mask), 0.2, 0.05, normalize)
    loss.backward()
    want.backward()
    np.testing.assert_allclose(float(loss.item()), float(want), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(no.grad.to_numpy(), nt.grad.numpy(), rtol=1e-3, atol=1e-6)


def test_grpo_loss_without_reference_and_bad_args():
    z = ox.from_numpy(np.zeros((2, 3), dtype=np.float32))
    adv = ox.from_numpy(np.array([1.0, -1.0], dtype=np.float32))
    assert abs(float(R.grpo_policy_loss(z, z, adv).item())) < 1e-6  # ratio 1: the +A and -A terms cancel
    with pytest.raises(ValueError):
        R.grpo_policy_loss(z, z, adv, normalize="nope")


def test_collect_group_rewards_and_zero_mean_advantages():
    step, _ = table_model(sharp=0.4)
    cfg = R.GenerationConfig(max_new_tokens=6, sampler=R.SamplerConfig(temperature=1.0))
    g = R.collect_group(step, [0], lambda r: float(r.token_ids.count(2)), group_size=8, config=cfg, seed=0)
    assert len(g.results) == 8 and g.rewards.shape == (8,)
    assert abs(float(g.advantages.sum().item())) < 1e-4
    again = R.collect_group(step, [0], lambda r: float(r.token_ids.count(2)), group_size=8, config=cfg, seed=0)
    np.testing.assert_array_equal(g.rewards, again.rewards)
    with pytest.raises(ValueError):
        R.collect_group(step, [0], lambda r: 0.0, group_size=1)


# --------------------------------------------------------------------------------------------------------------- ReasoningEngine

OPEN, CLOSE, EOS, WAIT, THINK, ANS_IS, DOT = 10, 11, 12, 13, 1, 30, 31
V = 40
TEXT = {THINK: "x", ANS_IS: "the answer is ", DOT: "."}
TEXT.update({20 + d: str(d) for d in range(10)})
decode = lambda ids: "".join(TEXT.get(t, "") for t in ids)
encode = lambda s: [2, 3]


def answering_model(p_correct=0.7, think_len=4, right=4, wrong=5):
    """Thinks `think_len` tokens, then says 'the answer is <digit>.' with the right digit w.p. `p_correct` (under sampling)."""
    def step(ids):
        l = np.full(V, -20.0, dtype=np.float32)
        start = ids.index(OPEN) + 1
        if CLOSE not in ids:
            l[CLOSE if len(ids) - start >= think_len else THINK] = 10.0
            return l
        k = len(ids) - ids.index(CLOSE) - 1
        if k == 0:
            l[ANS_IS] = 10.0
        elif k == 1:
            l[20 + right], l[20 + wrong] = math.log(p_correct), math.log(1 - p_correct)
        elif k == 2:
            l[DOT] = 10.0
        else:
            l[EOS] = 10.0
        return l
    return step


def engine(step, **kw):
    base = dict(think_open_id=OPEN, think_close_id=CLOSE, eos_id=EOS, encode_fn=encode, decode_fn=decode,
                sampling=R.SamplerConfig(temperature=1.0), answer_sampling=R.SamplerConfig(temperature=1.0), max_think_tokens=16)
    base.update(kw)
    return R.ReasoningEngine(step, **base)


def test_engine_single_sample_and_text():
    out = engine(answering_model(p_correct=1.0 - 1e-9)).solve("what is 2+2?")
    assert out.answer == "4" and out.n_samples == 1 and "the answer is 4." in out.text and out.total_tokens > 0


def test_engine_voting_beats_a_single_noisy_sample():
    step = answering_model(p_correct=0.7)
    singles = [engine(step).solve("q", seed=s).answer == "4" for s in range(60)]
    voted = [engine(step, min_samples=3, max_samples=15, confidence=0.95).solve("q", seed=1000 * s).answer == "4" for s in range(30)]
    assert np.mean(voted) > np.mean(singles) and np.mean(voted) >= 0.9


def test_engine_stops_early_on_easy_questions_and_respects_the_token_budget():
    easy = engine(answering_model(p_correct=1.0 - 1e-9), min_samples=3, max_samples=20, confidence=0.9).solve("q")
    assert easy.n_samples < 10 and easy.stopped_early and easy.confidence >= 0.9
    capped = engine(answering_model(p_correct=0.5), min_samples=30, max_samples=30, max_total_tokens=40).solve("q")
    assert capped.n_samples < 30 and capped.total_tokens >= 40 and capped.answer in ("4", "5")


def test_engine_forces_budgeted_thinking_and_validates_inputs():
    step = answering_model(think_len=100)
    out = engine(step, max_think_tokens=6).solve("q")
    assert out.traces[0].forced_close and len(out.traces[0].thinking_ids) == 6
    with pytest.raises(ValueError):
        R.ReasoningEngine(step, think_open_id=OPEN, think_close_id=OPEN, eos_id=EOS)
    with pytest.raises(ValueError):
        engine(step, encode_fn=None).solve("text needs an encoder")
    assert engine(step).solve([2, 3]).traces  # token ids need no encoder


def test_engine_works_with_a_real_causal_lm():
    model = ox.CausalLM(60, 16, 2, 2, 32, 16)
    eng = R.ReasoningEngine(R.causal_lm_step(model), think_open_id=58, think_close_id=59, eos_id=57, decode_fn=lambda ids: " ".join(map(str, ids)),
                            sampling=R.SamplerConfig(temperature=0.8, top_k=10), max_think_tokens=4, max_answer_tokens=3)
    out = eng.solve([1, 2, 3], seed=0)
    assert len(out.traces) == 1 and len(out.traces[0].thinking_ids) <= 4 and len(out.traces[0].answer_ids) <= 3


def test_grpo_training_loop_increases_the_reward():
    """End to end: sample groups from a policy, score them, and take GRPO steps - the expected reward must go up."""
    V, T, TARGET = 8, 5, 7
    rng = np.random.default_rng(0)
    w = ox.nn.Parameter(ox.from_numpy(rng.standard_normal((V, V)).astype(np.float32) * 0.1))
    opt = ox.optim.Adam([w], lr=0.1)
    step = lambda ids: w.detach().to_numpy()[ids[-1]]
    cfg = R.GenerationConfig(max_new_tokens=T, sampler=R.SamplerConfig(temperature=1.0))
    reward = lambda r: float(r.token_ids.count(TARGET))

    def mean_reward(seed):
        res = R.generate_batch(step, [[0]] * 32, cfg, seed=seed)
        return float(np.mean([reward(r) for r in res]))

    before = mean_reward(10_000)
    for it in range(25):
        g = R.collect_group(step, [0], reward, group_size=16, config=cfg, seed=it * 16)
        ids = np.asarray([[0] + r.token_ids[:-1] for r in g.results])  # previous token at each step
        tgt = np.asarray([r.token_ids for r in g.results])
        new = R.sequence_logprobs(w[ox.from_numpy(ids.reshape(-1))], tgt.reshape(-1)).reshape([16, T])
        loss = R.grpo_policy_loss(new, new.detach(), g.advantages, normalize="token")
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert mean_reward(10_000) > before + 1.0
