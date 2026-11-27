"""Reasoning-model toolkit: sampling, a generation engine, think-budget decoding, test-time-compute search and RL helpers.

Everything is model-agnostic: a model is a function `step_fn(token_ids: list[int]) -> logits` for the next token (a tensor or array);
`causal_lm_step(model)` adapts an `ox.CausalLM`.

* **Sampling** - `SamplerConfig` / `TokenSampler`: temperature, top-k, top-p, min-p, repetition / presence / frequency penalties, logit bias,
  seed. `sample()` runs on the native Rust sampler (`Tensor.sample_token`); `process()` / `probs()` expose the filtered distribution.
* **Generation engine** - `generate_tokens`, `stream_generate`, `generate_batch`: stop tokens and stop strings, min / max length, per-token
  log-probabilities and entropies, finish reasons, deterministic per-sample seeds.
* **Reasoning decoding** - `generate_with_reasoning`: `<think> ... </think> answer` under a thinking budget (forced close, "wait"
  extension), separate samplers for thinking and answering.
* **Search / test-time compute** - `beam_search`, `mcts` (PUCT), `tree_of_thoughts`, `best_of_n`, `self_consistency`,
  `adaptive_self_consistency` (Bayesian early stopping), `self_refine`.
* **Answers and rewards** - `extract_boxed`, `extract_final_answer`, `normalize_final_answer`, `answers_match`, `split_reasoning`,
  `reasoning_reward`, `combine_rewards`, `overlong_penalty`.
* **RL** - `sequence_logprobs`, `group_advantages`, `grpo_policy_loss` (clipped surrogate + k3 KL, three normalisations), `collect_group`.
* **`ReasoningEngine`** - ties it together: encode, think under a budget, vote over samples with a total token budget, decode.

Thread-safety: a `TokenSampler` owns a random stream, so give each thread / sample its own (the engine seeds one per sample).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from .oxideon import Tensor as _Tensor, from_numpy as _from_numpy, SamplingConfig as _NativeSamplingConfig

__all__ = [
    "SamplerConfig", "TokenSampler", "GenerationConfig", "GenerationResult", "generate_tokens", "stream_generate", "generate_batch",
    "causal_lm_step", "ReasoningTrace", "generate_with_reasoning", "Beam", "beam_search", "MCTSChild", "MCTSResult", "mcts", "tree_of_thoughts",
    "best_of_n", "self_consistency", "VoteResult", "adaptive_self_consistency", "NoAnswerError", "self_refine", "extract_boxed", "extract_final_answer",
    "normalize_final_answer", "answers_match", "split_reasoning", "reasoning_reward", "combine_rewards", "overlong_penalty",
    "sequence_logprobs", "group_advantages", "grpo_policy_loss", "GroupRollout", "collect_group", "ReasoningResult", "ReasoningEngine",
]


# ======================================================================================================================= sampling

@dataclass
class SamplerConfig:
    """How the next token is chosen. `temperature=0` (or `greedy=True`) is argmax decoding."""
    temperature: float = 1.0
    top_k: int = 0  # 0 = off
    top_p: float = 1.0  # 1.0 = off
    min_p: float = 0.0  # keep tokens with prob >= min_p * max prob; 0 = off
    repetition_penalty: float = 1.0  # >1 discourages tokens already seen (CTRL / HF form)
    presence_penalty: float = 0.0  # subtracted once from every token already seen
    frequency_penalty: float = 0.0  # subtracted per occurrence
    logit_bias: Dict[int, float] = field(default_factory=dict)
    seed: Optional[int] = None
    greedy: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite number >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be a finite number > 0")

    def replace(self, **changes: Any) -> "SamplerConfig":
        d = dict(self.__dict__)
        d.update(changes)
        return SamplerConfig(**d)


def _as_logits(logits: Any) -> _Tensor:
    if isinstance(logits, _Tensor):
        x = logits.detach()
    else:
        x = _from_numpy(np.ascontiguousarray(np.asarray(logits, dtype=np.float32)))
    if x.dtype not in ("f32", "f64"):
        x = x.to_dtype("f32")
    x = x.reshape([-1])
    if int(x.shape[0]) == 0:
        raise ValueError("logits are empty")
    a = x.to_numpy()
    if np.isnan(a).any():
        raise ValueError("logits contain NaN (diverged model or bad step function)")
    if np.isneginf(a).all():
        raise ValueError("every logit is -inf: no token can be sampled")
    return x


class TokenSampler:
    """Turns next-token logits into a token. `probs()` returns the distribution after every control has been applied.

    `engine="native"` (default) draws with the Rust sampler; `engine="python"` draws from `probs()` with a NumPy generator.
    Both follow the same controls; they use different random streams, so seeds only reproduce within an engine."""

    def __init__(self, config: Optional[SamplerConfig] = None, engine: str = "native", **kwargs: Any) -> None:
        if engine not in ("native", "python"):
            raise ValueError("engine must be 'native' or 'python'")
        self.config = config if config is not None else SamplerConfig(**kwargs)
        self.engine = engine
        self.reseed(self.config.seed)

    def reseed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)
        self._native = self._build_native(seed)

    def _build_native(self, seed: Optional[int]):
        c = self.config
        if c.greedy or c.temperature == 0:
            cfg = _NativeSamplingConfig.greedy()
            cfg.repetition_penalty = c.repetition_penalty
        else:
            cfg = _NativeSamplingConfig(
                temperature=c.temperature, top_k=c.top_k or None, top_p=c.top_p if c.top_p < 1.0 else None,
                min_p=c.min_p or None, repetition_penalty=c.repetition_penalty, presence_penalty=c.presence_penalty,
                frequency_penalty=c.frequency_penalty, seed=seed)
        cfg.presence_penalty, cfg.frequency_penalty = c.presence_penalty, c.frequency_penalty
        for t, b in c.logit_bias.items():
            r = cfg.with_logit_bias(int(t), float(b))
            cfg = r if r is not None else cfg
        return cfg

    # -- processing ------------------------------------------------------------------------------------------------
    def process(self, logits: Any, history: Sequence[int] = ()) -> _Tensor:
        """The logits after bias, penalties, temperature and top-k / top-p / min-p filtering (filtered entries are -inf)."""
        c = self.config
        x = _as_logits(logits)
        v = int(x.shape[0])
        if c.logit_bias:
            bias = np.zeros(v, dtype=np.float32)
            for t, b in c.logit_bias.items():
                if not 0 <= int(t) < v:
                    raise ValueError(f"logit_bias token {t} is outside the vocabulary of {v}")
                bias[int(t)] = b
            x = x + _from_numpy(bias).to_dtype(x.dtype)
        if len(history) and (c.repetition_penalty != 1.0 or c.presence_penalty or c.frequency_penalty):
            h = np.asarray(history, dtype=np.int64)
            counts = np.bincount(h[(h >= 0) & (h < v)], minlength=v)[:v].astype(np.float32)
            seen = _from_numpy((counts > 0).astype(np.float32)).to_dtype(x.dtype)
            if c.repetition_penalty != 1.0:
                scaled = x.maximum(x * 0.0) * (1.0 / c.repetition_penalty) + x.minimum(x * 0.0) * c.repetition_penalty
                x = x + (scaled - x) * seen
            if c.presence_penalty or c.frequency_penalty:
                x = x - seen * c.presence_penalty - _from_numpy(counts).to_dtype(x.dtype) * c.frequency_penalty
        if c.temperature > 0 and c.temperature != 1.0:
            x = x * (1.0 / c.temperature)
        neg_inf = float("-inf")
        if c.top_k and c.top_k < v:
            kth = x.topk(c.top_k).values.min()
            x = x.masked_fill(x < kth, neg_inf)
        if c.min_p > 0.0:
            p = x.softmax(0)
            x = x.masked_fill(p < p.max() * c.min_p, neg_inf)
        if c.top_p < 1.0:
            s = x.sort(dim=0, descending=True)
            p_sorted = s.values.softmax(0)
            cum = p_sorted.cumsum(0)
            keep_sorted = (cum - p_sorted) < c.top_p  # the smallest prefix whose mass reaches top_p (at least one token)
            cutoff = s.values.masked_fill(~keep_sorted, float("inf")).min()
            x = x.masked_fill(x < cutoff, neg_inf)
        return x

    def probs(self, logits: Any, history: Sequence[int] = ()) -> _Tensor:
        c = self.config
        x = self.process(logits, history)
        if c.greedy or c.temperature == 0:
            one = np.zeros(int(x.shape[0]), dtype=np.float32)
            one[int(x.argmax())] = 1.0
            return _from_numpy(one)
        return x.softmax(0)

    def sample(self, logits: Any, history: Sequence[int] = ()) -> int:
        c = self.config
        if self.engine == "native":
            x = _as_logits(logits)
            if x.dtype != "f32":
                x = x.to_dtype("f32")
            return int(x.sample_token(self._native, list(history) if len(history) else None))
        if c.greedy or c.temperature == 0:
            return int(self.process(logits, history).argmax())
        p = self.probs(logits, history).to_numpy().astype(np.float64)
        return int(self.rng.choice(len(p), p=p / p.sum()))


def _log_softmax_np(logits: Any) -> np.ndarray:
    return _as_logits(logits).log_softmax(0).to_numpy().astype(np.float64)


# ================================================================================================================ generation engine

StepFn = Callable[[List[int]], Any]


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    stop_token_ids: Tuple[int, ...] = ()
    stop_strings: Tuple[str, ...] = ()  # needs `decode_fn`; the text is cut at the first match
    sampler: SamplerConfig = field(default_factory=lambda: SamplerConfig(temperature=0.0))
    seed: Optional[int] = None  # overrides sampler.seed
    track_logprobs: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0 or self.min_new_tokens < 0 or self.min_new_tokens > self.max_new_tokens:
            raise ValueError("need 0 <= min_new_tokens <= max_new_tokens")


@dataclass
class GenerationResult:
    token_ids: List[int]  # the generated tokens (a stop token, if one ended the generation, is included)
    logprobs: List[float]  # log-probability of each token under the model's own (unprocessed) distribution
    entropies: List[float]  # entropy (nats) of that distribution at each step
    finish_reason: str  # "stop" (a stop token), "stop_string", or "length"
    text: Optional[str] = None

    @property
    def mean_logprob(self) -> float:
        return float(np.mean(self.logprobs)) if self.logprobs else 0.0

    @property
    def perplexity(self) -> float:
        return math.exp(-self.mean_logprob) if self.logprobs else float("nan")


def _validate_ids(ids: Sequence[int], what: str) -> List[int]:
    out = [int(t) for t in ids]
    if any(t < 0 for t in out):
        raise ValueError(f"{what} contains a negative token id")
    return out


def stream_generate(step_fn: StepFn, prompt_ids: Sequence[int], config: Optional[GenerationConfig] = None, *,
                    decode_fn: Optional[Callable[[List[int]], str]] = None, sampler: Optional[TokenSampler] = None
                    ) -> Iterator[Tuple[int, float]]:
    """Yields `(token, logprob)` as tokens are produced. The generator's return value (`StopIteration.value`) is the
    `GenerationResult`; `generate_tokens` is the non-streaming wrapper."""
    cfg = config or GenerationConfig()
    ids = _validate_ids(prompt_ids, "prompt_ids")
    if not ids:
        raise ValueError("prompt_ids must not be empty")
    sampler = sampler or TokenSampler(cfg.sampler.replace(seed=cfg.seed) if cfg.seed is not None else cfg.sampler)
    stop = set(int(t) for t in cfg.stop_token_ids)
    if cfg.stop_strings and decode_fn is None:
        raise ValueError("stop_strings needs decode_fn")
    out: List[int] = []
    lps: List[float] = []
    ents: List[float] = []
    reason = "length"
    text: Optional[str] = None
    for step in range(cfg.max_new_tokens):
        logits = step_fn(ids + out)
        if step < cfg.min_new_tokens and stop:
            x = _as_logits(logits)
            logits = x.masked_fill(_from_numpy(np.isin(np.arange(int(x.shape[0])), list(stop)).astype(np.bool_)), float("-inf"))
        tok = sampler.sample(logits, ids + out)
        lp = ent = 0.0
        if cfg.track_logprobs:
            ls = _log_softmax_np(logits)
            lp = float(ls[tok])
            pr = np.exp(ls)
            ent = float(-(pr[pr > 0] * ls[pr > 0]).sum())
        out.append(tok)
        lps.append(lp)
        ents.append(ent)
        yield tok, lp
        if tok in stop:
            reason = "stop"
            break
        if cfg.stop_strings:
            tail = decode_fn(out)
            hit = [i for i in (tail.find(s) for s in cfg.stop_strings) if i != -1]
            if hit:
                reason, text = "stop_string", tail[:min(hit)]
                break
    if text is None and decode_fn is not None:
        text = decode_fn(out)
    return GenerationResult(out, lps if cfg.track_logprobs else [], ents if cfg.track_logprobs else [], reason, text)


def generate_tokens(step_fn: StepFn, prompt_ids: Sequence[int], config: Optional[GenerationConfig] = None, *,
             decode_fn: Optional[Callable[[List[int]], str]] = None, sampler: Optional[TokenSampler] = None) -> GenerationResult:
    """Generates up to `max_new_tokens` tokens with the sampler of `config` and returns a `GenerationResult`."""
    gen = stream_generate(step_fn, prompt_ids, config, decode_fn=decode_fn, sampler=sampler)
    while True:
        try:
            next(gen)
        except StopIteration as done:
            return done.value


def generate_batch(step_fn: StepFn, prompts: Sequence[Sequence[int]], config: Optional[GenerationConfig] = None, *,
                   decode_fn: Optional[Callable[[List[int]], str]] = None, seed: Optional[int] = None) -> List[GenerationResult]:
    """Generates for several prompts. Sample `i` gets seed `seed + i`, so the batch is reproducible and the samples independent."""
    cfg = config or GenerationConfig()
    base = seed if seed is not None else cfg.seed
    results = []
    for i, p in enumerate(prompts):
        c = GenerationConfig(**{**cfg.__dict__, "seed": None if base is None else base + i})
        results.append(generate_tokens(step_fn, p, c, decode_fn=decode_fn))
    return results


def causal_lm_step(model: Any, max_context: Optional[int] = None) -> StepFn:
    """Adapts an `ox.CausalLM` (`forward(tokens[1, T]) -> logits[1, T, V]`) to a `step_fn`. `max_context` keeps only the last
    tokens when the sequence grows past the model's window."""
    def step(ids: List[int]) -> _Tensor:
        ctx = ids if max_context is None else ids[-max_context:]
        out = model.forward(_Tensor.from_i64([int(t) for t in ctx], [1, len(ctx)]))
        return out[0, -1]
    return step


# ============================================================================================================ think-budget decoding

@dataclass
class ReasoningTrace:
    thinking_ids: List[int]
    answer_ids: List[int]
    forced_close: bool = False  # the thinking phase hit `max_think_tokens` and was closed for the model
    extensions: int = 0  # how many times "wait" tokens were inserted to lengthen thinking
    finished: bool = False  # the answer ended with `eos_id` (rather than running out of tokens)
    logprobs: List[float] = field(default_factory=list)  # of every model-chosen token (thinking, then answer)

    @property
    def token_ids(self) -> List[int]:
        return self.thinking_ids + self.answer_ids

    @property
    def mean_logprob(self) -> float:
        return float(np.mean(self.logprobs)) if self.logprobs else 0.0


def generate_with_reasoning(
    step_fn: StepFn,
    prompt_ids: Sequence[int],
    *,
    think_open_id: int,
    think_close_id: int,
    eos_id: int,
    sampler: Optional[TokenSampler] = None,
    think_sampler: Optional[TokenSampler] = None,
    max_think_tokens: int = 512,
    min_think_tokens: int = 0,
    max_answer_tokens: int = 256,
    wait_ids: Sequence[int] = (),
    max_extensions: int = 2,
    open_think: bool = True,
    track_logprobs: bool = True,
) -> ReasoningTrace:
    """Decodes `<think> thoughts </think> answer`. `think_sampler` (default: `sampler`) lets thinking run hotter than the answer,
    e.g. temperature 0.8 to explore and 0 to commit. `wait_ids` are inserted when the model tries to close thinking before
    `min_think_tokens` (budget forcing); at `max_think_tokens` the close token is inserted for it."""
    if max_think_tokens < 0 or min_think_tokens < 0 or max_answer_tokens < 0:
        raise ValueError("token budgets must be >= 0")
    if len({int(think_open_id), int(think_close_id), int(eos_id)}) != 3:
        raise ValueError("think_open_id, think_close_id and eos_id must be distinct")
    sampler = sampler or TokenSampler(SamplerConfig(temperature=0.0))
    think_sampler = think_sampler or sampler
    ids = _validate_ids(prompt_ids, "prompt_ids")
    if open_think:
        ids.append(int(think_open_id))
    thinking: List[int] = []
    answer: List[int] = []
    lps: List[float] = []
    forced = False
    extensions = 0

    def pick(s: TokenSampler) -> int:
        logits = step_fn(ids)
        tok = s.sample(logits, ids)
        if track_logprobs:
            lps.append(float(_log_softmax_np(logits)[tok]))
        return tok

    while True:
        if len(thinking) >= max_think_tokens:
            forced = True
            break
        tok = pick(think_sampler)
        if tok == think_close_id:
            if len(thinking) < min_think_tokens and wait_ids and extensions < max_extensions:
                for w in wait_ids:
                    thinking.append(int(w))
                    ids.append(int(w))
                extensions += 1
                continue
            break
        if tok == eos_id:  # the model ended without an answer
            return ReasoningTrace(thinking, [], forced_close=False, extensions=extensions, finished=True, logprobs=lps)
        thinking.append(tok)
        ids.append(tok)
    ids.append(int(think_close_id))
    finished = False
    for _ in range(max_answer_tokens):
        tok = pick(sampler)
        if tok == eos_id:
            finished = True
            break
        answer.append(tok)
        ids.append(tok)
    return ReasoningTrace(thinking, answer, forced_close=forced, extensions=extensions, finished=finished, logprobs=lps)


# =========================================================================================================== search / test-time compute

@dataclass
class Beam:
    token_ids: List[int]
    logprob: float  # summed log-probability
    finished: bool
    score: float  # length-normalised


def beam_search(step_fn: StepFn, prompt_ids: Sequence[int], *, num_beams: int = 4, max_new_tokens: int = 32,
                eos_id: Optional[int] = None, length_penalty: float = 1.0, num_return: int = 1, min_new_tokens: int = 0) -> List[Beam]:
    """Beam search on the model's log-probabilities. `score = logprob / len**length_penalty` (1.0 favours longer sequences than 0.0).
    Returns the `num_return` best beams, best first. Stops once `num_beams` beams have finished and no live beam can beat them."""
    if num_beams < 1 or num_return < 1 or num_return > num_beams:
        raise ValueError("need 1 <= num_return <= num_beams")
    base = _validate_ids(prompt_ids, "prompt_ids")
    live: List[Tuple[List[int], float]] = [([], 0.0)]
    done: List[Beam] = []
    norm = lambda lp, n: lp / (max(n, 1) ** length_penalty)
    for step in range(max_new_tokens):
        cands: List[Tuple[List[int], float, bool]] = []
        for seq, lp in live:
            ls = _log_softmax_np(step_fn(base + seq))
            if eos_id is not None and step < min_new_tokens:
                ls[eos_id] = -np.inf
            top = np.argpartition(-ls, min(2 * num_beams, len(ls) - 1))[:2 * num_beams]
            for t in top:
                if np.isfinite(ls[t]):
                    cands.append((seq + [int(t)], lp + float(ls[t]), eos_id is not None and int(t) == eos_id))
        cands.sort(key=lambda c: -c[1])
        live = []
        for rank, (seq, lp, fin) in enumerate(cands[:2 * num_beams]):
            if fin:
                if rank < num_beams:  # only a finished beam ranking among the best `num_beams` candidates counts
                    done.append(Beam(seq, lp, True, norm(lp, len(seq))))
            elif len(live) < num_beams:
                live.append((seq, lp))
        done.sort(key=lambda b: -b.score)
        done = done[:num_beams]
        if not live:
            break
        if len(done) >= num_beams and norm(live[0][1], len(live[0][0])) <= done[-1].score:
            break
    out = list(done)
    if len(out) < num_return:
        out += [Beam(s, lp, False, norm(lp, len(s))) for s, lp in live]
    out.sort(key=lambda b: -b.score)
    return out[:num_return]


class MCTSChild(NamedTuple):
    """A child state with a prior probability (use it in `expand_fn` results to guide the search)."""
    state: Any
    prior: float


@dataclass
class MCTSResult:
    best_child: Any  # the most-visited child of the root
    path: List[Any]  # root -> ... following most-visited children
    visits: Dict[int, int]  # root child index -> visit count
    values: Dict[int, float]  # root child index -> mean value
    simulations: int


class _Node:
    __slots__ = ("state", "parent", "children", "prior", "visits", "value_sum", "expanded")

    def __init__(self, state: Any, parent: Optional["_Node"], prior: float):
        self.state, self.parent, self.prior = state, parent, prior
        self.children: List["_Node"] = []
        self.visits, self.value_sum, self.expanded = 0, 0.0, False


def mcts(root: Any, expand_fn: Callable[[Any], Sequence[Any]], value_fn: Callable[[Any], float], *, simulations: int = 100,
         c_puct: float = 1.4, is_terminal: Optional[Callable[[Any], bool]] = None) -> MCTSResult:
    """Monte-Carlo tree search (PUCT) over reasoning states. `expand_fn(state)` returns child states, or `MCTSChild(state, prior)` entries;
    `value_fn(state)` scores a state (higher is better; e.g. a process-reward model or a verifier). Deterministic given the functions."""
    if simulations < 1:
        raise ValueError("simulations must be >= 1")
    top = _Node(root, None, 1.0)

    def expand(n: _Node) -> None:
        kids = list(expand_fn(n.state))
        pairs = [((k.state, k.prior) if isinstance(k, MCTSChild) else (k, 1.0)) for k in kids]
        total = sum(max(float(p), 0.0) for _, p in pairs) or 1.0
        n.children = [_Node(s, n, max(float(p), 0.0) / total) for s, p in pairs]
        n.expanded = True

    for _ in range(simulations):
        node = top
        while node.expanded and node.children:
            sq = math.sqrt(node.visits + 1)
            node = max(node.children, key=lambda c: (c.value_sum / c.visits if c.visits else 0.0) + c_puct * c.prior * sq / (1 + c.visits))
        if not node.expanded and not (is_terminal and is_terminal(node.state)):
            expand(node)
        v = float(value_fn(node.state))
        while node is not None:
            node.visits += 1
            node.value_sum += v
            node = node.parent
    if not top.children:
        return MCTSResult(root, [root], {}, {}, simulations)
    best = max(top.children, key=lambda c: (c.visits, c.value_sum / max(c.visits, 1)))
    path, cur = [root], top
    while cur.children:
        cur = max(cur.children, key=lambda c: (c.visits, c.value_sum / max(c.visits, 1)))
        if cur.visits == 0:
            break
        path.append(cur.state)
    return MCTSResult(best.state, path, {i: c.visits for i, c in enumerate(top.children)},
                      {i: c.value_sum / max(c.visits, 1) for i, c in enumerate(top.children)}, simulations)


def tree_of_thoughts(root: Any, expand_fn: Callable[[Any], Sequence[Any]], score_fn: Callable[[Any], float], *,
                     beam_width: int = 3, depth: int = 3, is_solution: Optional[Callable[[Any], bool]] = None) -> Tuple[Any, float]:
    """Beam search over partial reasoning states: expand every state in the beam with `expand_fn`, score the children with
    `score_fn`, keep the best `beam_width`, for `depth` levels. Stops early at a state where `is_solution` holds.
    Returns the best `(state, score)` seen."""
    beam = [(root, float(score_fn(root)))]
    best = beam[0]
    for _ in range(depth):
        children = [(c, float(score_fn(c))) for s, _ in beam for c in expand_fn(s)]
        if not children:
            break
        children.sort(key=lambda cs: -cs[1])
        beam = children[:beam_width]
        if beam[0][1] > best[1]:
            best = beam[0]
        if is_solution is not None:
            for s, sc in beam:
                if is_solution(s):
                    return s, sc
    return best


def best_of_n(candidates: Sequence[Any], scorer: Callable[[Any], float], k: int = 1) -> List[Tuple[Any, float]]:
    """The `k` best candidates under a verifier / reward `scorer`, best first, as `(candidate, score)` pairs."""
    scored = [(c, float(scorer(c))) for c in candidates]
    return sorted(scored, key=lambda cs: -cs[1])[:k]


def self_consistency(generate_fn: Callable[[int], Any], n: int, extract: Callable[[Any], Any] = lambda r: r,
                     weights: Optional[Callable[[Any], float]] = None) -> Tuple[Any, Dict[Any, float]]:
    """Samples `n` reasoning paths with `generate_fn(i)`, maps each to a final answer with `extract`, and returns
    `(winning_answer, votes)` by (optionally weighted) majority vote. Ties go to the answer seen first."""
    votes: Dict[Any, float] = {}
    for i in range(n):
        r = generate_fn(i)
        a = extract(r)
        if a is None:
            continue
        votes[a] = votes.get(a, 0.0) + (weights(r) if weights else 1.0)
    if not votes:
        raise NoAnswerError("no sample produced an answer")
    best = max(votes.items(), key=lambda kv: kv[1])[0]
    return best, votes


class NoAnswerError(ValueError):
    """No sample produced an extractable answer."""


@dataclass
class VoteResult:
    answer: Any
    votes: Dict[Any, float]
    n_samples: int
    confidence: float  # posterior probability that `answer` is the majority answer
    stopped_early: bool


def _leader_confidence(counts: Sequence[float], draws: int = 2000, seed: int = 0) -> float:
    """P(the current leader has the largest true share), from a Dirichlet(count + 1) posterior (one pseudo-answer for 'unseen')."""
    alpha = np.asarray(list(counts) + [0.0], dtype=np.float64) + 1.0
    if len(counts) == 1:
        alpha = np.asarray([counts[0] + 1.0, 1.0])
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(alpha, size=draws)
    lead = int(np.argmax(counts))
    return float(np.mean(np.argmax(p[:, :len(counts)], axis=1) == lead))


def adaptive_self_consistency(generate_fn: Callable[[int], Any], extract: Callable[[Any], Any] = lambda r: r, *, min_samples: int = 3,
                              max_samples: int = 40, confidence: float = 0.9, weights: Optional[Callable[[Any], float]] = None,
                              seed: int = 0) -> VoteResult:
    """Self-consistency that stops sampling once the leading answer is settled: after `min_samples`, as soon as its posterior
    probability of being the majority reaches `confidence` (or the lead can no longer be overturned within `max_samples`).
    Easy questions cost a few samples, hard ones the full budget."""
    if not 1 <= min_samples <= max_samples or not 0.0 < confidence < 1.0:
        raise ValueError("need 1 <= min_samples <= max_samples and 0 < confidence < 1")
    votes: Dict[Any, float] = {}
    counts: Dict[Any, int] = {}
    n = 0
    conf = 0.0
    stopped = False
    for i in range(max_samples):
        r = generate_fn(i)
        n += 1
        a = extract(r)
        if a is not None:
            votes[a] = votes.get(a, 0.0) + (weights(r) if weights else 1.0)
            counts[a] = counts.get(a, 0) + 1
        if not counts:
            continue
        ordered = sorted(counts.values(), reverse=True)
        lead_gap = ordered[0] - (ordered[1] if len(ordered) > 1 else 0)
        conf = _leader_confidence(list(counts.values()), seed=seed)
        if n >= min_samples and (conf >= confidence or lead_gap > max_samples - n):
            stopped = n < max_samples
            break
    if not votes:
        raise NoAnswerError("no sample produced an answer")
    return VoteResult(max(votes.items(), key=lambda kv: kv[1])[0], votes, n, conf, stopped)


def self_refine(initial: Any, critique_fn: Callable[[Any], Any], revise_fn: Callable[[Any, Any], Any], *, max_rounds: int = 3,
                accept_fn: Optional[Callable[[Any], bool]] = None, score_fn: Optional[Callable[[Any], float]] = None
                ) -> Tuple[Any, List[Any]]:
    """Draft -> critique -> revise loop. Stops when `critique_fn` returns None (nothing to fix) or `accept_fn(draft)` holds. With
    `score_fn`, a revision that scores worse than the current draft is discarded and the loop ends. Returns `(best, history)`."""
    draft = initial
    history = [draft]
    for _ in range(max_rounds):
        if accept_fn is not None and accept_fn(draft):
            break
        feedback = critique_fn(draft)
        if feedback is None:
            break
        new = revise_fn(draft, feedback)
        if score_fn is not None and score_fn(new) < score_fn(draft):
            break
        draft = new
        history.append(draft)
    return draft, history


# ========================================================================================================== answers and rewards

def split_reasoning(text: str, think_open: str = "<think>", think_close: str = "</think>") -> Tuple[Optional[str], str, bool]:
    """Splits generated text into `(thinking, answer, well_formed)`. Well-formed means exactly one open before one close. A missing
    close tag (cut-off thinking) yields `(text after open, "", False)`; no tags at all yields `(None, text, False)`."""
    i, j = text.find(think_open), text.find(think_close)
    if i == -1 and j == -1:
        return None, text.strip(), False
    if i != -1 and j == -1:
        return text[i + len(think_open):].strip(), "", False
    start = i + len(think_open) if i != -1 and i < j else 0
    well = i != -1 and i < j and text.count(think_open) == 1 and text.count(think_close) == 1
    return text[start:j].strip(), text[j + len(think_close):].strip(), well


def extract_boxed(text: str) -> Optional[str]:
    """The content of the last `\\boxed{...}` (nested braces handled), or None."""
    k = text.rfind("\\boxed")
    if k == -1:
        return None
    i = text.find("{", k)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += text[j] == "{"
        depth -= text[j] == "}"
        if depth == 0:
            return text[i + 1:j].strip()
    return None


_NUM = r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?"


def extract_final_answer(text: str) -> Optional[str]:
    """Best-effort final answer: `\\boxed{}`, then `<answer>..</answer>`, then `#### x`, then 'the answer is x', then the last number."""
    b = extract_boxed(text)
    if b is not None:
        return b
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if m:
        return m[-1].strip()
    m = re.findall(r"####\s*(.+)", text)
    if m:
        return m[-1].strip()
    m = re.findall(r"answer\s*(?:is|:)\s*[:=]?\s*\$?([^\n]+?)\s*\.?\s*(?:\n|$)", text, re.I)
    if m:
        return m[-1].strip().rstrip(".")
    m = re.findall(_NUM, text)
    return m[-1] if m else None


def normalize_final_answer(ans: str) -> str:
    """Canonical form for comparing answers: no `$`, `\\text{}`, `\\!`, thousands commas, trailing period; `\\frac{a}{b}` -> `a/b`."""
    s = ans.strip().lower()
    s = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    for junk in ("\\!", "\\,", "\\ ", "\\left", "\\right", "$", "\\%"):
        s = s.replace(junk, "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.rstrip(". ").strip()
    return re.sub(r"\s+", " ", s)


def _to_number(s: str) -> Optional[float]:
    s = s.strip()
    pct = s.endswith("%")
    s = s.rstrip("%")
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            v = float(a) / float(b)
        else:
            v = float(s)
    except (ValueError, ZeroDivisionError):
        return None
    return v / 100.0 if pct else v


def answers_match(a: str, b: str, rel_tol: float = 1e-6) -> bool:
    """Whether two answers agree after normalisation; numeric answers (integers, decimals, fractions, percents) compare by value."""
    if a is None or b is None:
        return False
    na, nb = normalize_final_answer(str(a)), normalize_final_answer(str(b))
    if na == nb:
        return True
    x, y = _to_number(na), _to_number(nb)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=rel_tol, abs_tol=1e-12)


def reasoning_reward(text: str, expected: str, *, think_open: str = "<think>", think_close: str = "</think>",
                     format_weight: float = 0.2, accuracy_weight: float = 1.0,
                     extract: Optional[Callable[[str], Optional[str]]] = None) -> float:
    """A rule-based reward: `format_weight` for a well-formed `<think>...</think>` followed by an answer, plus `accuracy_weight`
    when the answer (text after `</think>`, run through `extract`, default: stripped; compared with `answers_match`) is `expected`."""
    thinking, answer, well = split_reasoning(text, think_open, think_close)
    got = extract(answer) if extract else answer.strip()
    ok = got is not None and got != "" and answers_match(got, expected)
    return (format_weight if well and answer != "" else 0.0) + (accuracy_weight if ok else 0.0)


def combine_rewards(parts: Sequence[Tuple[Callable[..., float], float]]) -> Callable[..., float]:
    """A reward that is the weighted sum of reward functions, each called with the same arguments."""
    if not parts:
        raise ValueError("combine_rewards needs at least one (function, weight) pair")
    return lambda *a, **k: float(sum(w * f(*a, **k) for f, w in parts))


def overlong_penalty(n_tokens: int, max_len: int, buffer: int = 0) -> float:
    """DAPO-style soft length penalty: 0 up to `max_len - buffer`, falling linearly to -1 at `max_len`, -1 beyond."""
    if n_tokens <= max_len - buffer:
        return 0.0
    if n_tokens >= max_len or buffer <= 0:
        return -1.0
    return -(n_tokens - (max_len - buffer)) / buffer


# ===================================================================================================================== RL (GRPO)

def sequence_logprobs(logits: _Tensor, token_ids: Union[Sequence[int], _Tensor]) -> _Tensor:
    """`log p(token_t)` for every position of `logits[T, V]` (differentiable); `token_ids` has T entries."""
    ids = token_ids if isinstance(token_ids, _Tensor) else _Tensor.from_i64([int(t) for t in token_ids], [len(token_ids)])
    return logits.log_softmax(-1).gather(1, ids.reshape([-1, 1])).reshape([-1])


def group_advantages(rewards: Any, group_size: int, eps: float = 1e-6, scale: bool = True) -> _Tensor:
    """GRPO advantages: rewards of `group_size` samples per prompt, normalised within each group
    (`(r - mean) / (std + eps)`, or just `r - mean` with `scale=False`). `rewards` is a flat list / tensor of length B*group_size."""
    r = rewards if isinstance(rewards, _Tensor) else _from_numpy(np.ascontiguousarray(np.asarray(rewards, dtype=np.float32)))
    if group_size < 1 or int(r.numel()) % group_size:
        raise ValueError(f"{int(r.numel())} rewards cannot be split into groups of {group_size}")
    r = r.reshape([-1, group_size])
    centred = r - r.mean(1, keepdim=True)
    if not scale:
        return centred.reshape([-1])
    std = r.std(1, keepdim=True, unbiased=False) if group_size > 1 else r * 0.0
    return (centred / (std + eps)).reshape([-1])


def grpo_policy_loss(new_logprobs: _Tensor, old_logprobs: _Tensor, advantages: _Tensor, ref_logprobs: Optional[_Tensor] = None,
              mask: Optional[_Tensor] = None, *, clip_eps: float = 0.2, kl_beta: float = 0.04, normalize: str = "sequence") -> _Tensor:
    """The GRPO objective as a loss (to minimise). Inputs are `[B, T]` token log-probabilities (`advantages` is `[B]`, one per sequence).

    per token: `-min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + kl_beta * k3`, with `ratio = exp(new - old)` and the k3 KL estimator
    `exp(ref - new) - (ref - new) - 1` against the reference policy. `normalize`: "sequence" (mean over each sequence's tokens, then over
    sequences - the original GRPO), "token" (mean over all tokens), or "constant" (sum / (B * T), Dr. GRPO's length-unbiased form)."""
    if normalize not in ("sequence", "token", "constant"):
        raise ValueError("normalize must be 'sequence', 'token' or 'constant'")
    b, t = (int(v) for v in new_logprobs.shape)
    adv = advantages.reshape([b, 1])
    ratio = (new_logprobs - old_logprobs.detach()).exp()
    surrogate = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    per_tok = -surrogate.minimum(clipped)
    if ref_logprobs is not None and kl_beta:
        d = ref_logprobs.detach() - new_logprobs
        per_tok = per_tok + (d.exp() - d - 1.0) * kl_beta
    m = mask if mask is not None else _from_numpy(np.ones((b, t), dtype=np.float32))
    m = m.to_dtype(per_tok.dtype)
    masked = per_tok * m
    if normalize == "constant":
        return masked.sum() / float(b * t)
    if normalize == "token":
        return masked.sum() / m.sum().maximum(m.sum() * 0.0 + 1.0)
    per_seq = masked.sum(1) / m.sum(1).maximum(m.sum(1) * 0.0 + 1.0)
    return per_seq.mean()


@dataclass
class GroupRollout:
    prompt_ids: List[int]
    results: List[GenerationResult]
    rewards: np.ndarray
    advantages: _Tensor


def collect_group(step_fn: StepFn, prompt_ids: Sequence[int], reward_fn: Callable[[GenerationResult], float], *, group_size: int = 8,
                  config: Optional[GenerationConfig] = None, decode_fn: Optional[Callable[[List[int]], str]] = None, seed: int = 0,
                  scale_advantages: bool = True) -> GroupRollout:
    """Samples `group_size` completions of one prompt (seeds `seed .. seed+group_size-1`), scores each with `reward_fn(result)`,
    and returns the group with its GRPO advantages."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    cfg = config or GenerationConfig(sampler=SamplerConfig(temperature=1.0))
    results = generate_batch(step_fn, [prompt_ids] * group_size, cfg, decode_fn=decode_fn, seed=seed)
    rewards = np.asarray([float(reward_fn(r)) for r in results], dtype=np.float32)
    return GroupRollout(list(prompt_ids), results, rewards, group_advantages(rewards, group_size, scale=scale_advantages))


# ============================================================================================================== ReasoningEngine

@dataclass
class ReasoningResult:
    answer: Optional[str]
    text: str  # the generated text of the winning sample (thinking + answer)
    votes: Dict[Any, float]
    confidence: float
    n_samples: int
    total_tokens: int
    stopped_early: bool
    traces: List[ReasoningTrace]


class ReasoningEngine:
    """Think, answer and vote. `solve(prompt)` runs `generate_with_reasoning` up to `max_samples` times (each with its own seed),
    votes on the extracted answers with early stopping, and stops sampling when `max_total_tokens` is spent."""

    def __init__(self, step_fn: StepFn, *, think_open_id: int, think_close_id: int, eos_id: int,
                 encode_fn: Optional[Callable[[str], List[int]]] = None, decode_fn: Optional[Callable[[List[int]], str]] = None,
                 sampling: Optional[SamplerConfig] = None, answer_sampling: Optional[SamplerConfig] = None,
                 max_think_tokens: int = 512, min_think_tokens: int = 0, max_answer_tokens: int = 128,
                 wait_ids: Sequence[int] = (), max_extensions: int = 2,
                 extract_fn: Callable[[str], Optional[str]] = extract_final_answer,
                 min_samples: int = 1, max_samples: int = 1, confidence: float = 0.9, weight_by_logprob: bool = False,
                 max_total_tokens: Optional[int] = None) -> None:
        if len({int(think_open_id), int(think_close_id), int(eos_id)}) != 3:
            raise ValueError("think_open_id, think_close_id and eos_id must be distinct")
        if min(max_think_tokens, min_think_tokens, max_answer_tokens) < 0 or not 1 <= min_samples <= max_samples:
            raise ValueError("token budgets must be >= 0 and 1 <= min_samples <= max_samples")
        self.step_fn, self.encode_fn, self.decode_fn = step_fn, encode_fn, decode_fn
        self.ids = dict(think_open_id=int(think_open_id), think_close_id=int(think_close_id), eos_id=int(eos_id))
        self.sampling = sampling or SamplerConfig(temperature=0.7, top_p=0.95)
        self.answer_sampling = answer_sampling or self.sampling.replace(temperature=0.0)
        self.budget = dict(max_think_tokens=max_think_tokens, min_think_tokens=min_think_tokens, max_answer_tokens=max_answer_tokens,
                           wait_ids=tuple(wait_ids), max_extensions=max_extensions)
        self.extract_fn = extract_fn
        self.min_samples, self.max_samples, self.confidence = min_samples, max_samples, confidence
        self.weight_by_logprob = weight_by_logprob
        self.max_total_tokens = max_total_tokens

    def _text(self, ids: List[int]) -> str:
        if self.decode_fn is None:
            raise ValueError("decode_fn is required to turn tokens into text")
        return self.decode_fn(ids)

    def solve(self, prompt: Union[str, Sequence[int]], *, seed: int = 0) -> ReasoningResult:
        if isinstance(prompt, str):
            if self.encode_fn is None:
                raise ValueError("encode_fn is required to solve a text prompt")
            prompt_ids = self.encode_fn(prompt)
        else:
            prompt_ids = list(prompt)
        traces: List[ReasoningTrace] = []
        spent = [0]

        def one(i: int) -> ReasoningTrace:
            if self.max_total_tokens is not None and spent[0] >= self.max_total_tokens and traces:
                raise _OutOfBudget()
            think = TokenSampler(self.sampling.replace(seed=seed + i))
            ans = TokenSampler(self.answer_sampling.replace(seed=seed + i))
            tr = generate_with_reasoning(self.step_fn, prompt_ids, sampler=ans, think_sampler=think, **self.ids, **self.budget)
            spent[0] += len(tr.token_ids)
            traces.append(tr)
            return tr

        def answer_of(tr: ReasoningTrace) -> Optional[str]:
            a = self.extract_fn(self._text(tr.answer_ids)) if tr.answer_ids else None
            return None if a is None else normalize_final_answer(a)

        weights = (lambda tr: math.exp(tr.mean_logprob)) if self.weight_by_logprob else None
        try:
            vote = adaptive_self_consistency(one, answer_of, min_samples=self.min_samples, max_samples=self.max_samples,
                                             confidence=self.confidence, weights=weights, seed=seed)
        except _OutOfBudget:
            vote = None
        except NoAnswerError:
            vote = None
        if vote is None:
            votes: Dict[Any, float] = {}
            for tr in traces:
                a = answer_of(tr)
                if a is not None:
                    votes[a] = votes.get(a, 0.0) + (weights(tr) if weights else 1.0)
            answer = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
            conf = (votes[answer] / sum(votes.values())) if votes else 0.0
            vote = VoteResult(answer, votes, len(traces), conf, bool(votes) and len(traces) < self.max_samples)
        winner = next((tr for tr in traces if answer_of(tr) == vote.answer), traces[0] if traces else None)
        text = self._text(winner.token_ids) if (winner is not None and self.decode_fn is not None) else ""
        return ReasoningResult(vote.answer, text, vote.votes, vote.confidence, len(traces), spent[0], vote.stopped_early, traces)


class _OutOfBudget(Exception):
    pass
